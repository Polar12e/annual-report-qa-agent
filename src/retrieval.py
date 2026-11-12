"""Retrieval over annual-report pages: lexical, concept, dense and hybrid channels.

The corpus is ``corpus/pages.jsonl`` (one JSON object per PDF page, produced
from the fixed PDF).  Pages are split into overlapping chunks, then ranked by
one of four interchangeable channels:

``bm25``      BM25 over Chinese character bigrams plus a statement-page prior —
              precise, cheap, and the strong baseline for exact line items.
``expanded``  the same BM25, but questions whose wording barely exists in the
              corpus are first rewritten into accounting wording by
              ``src.lexicon``; the measured default.
``dense``     cosine over embeddings from any ``src.embedding.Embedder`` — the
              only channel that can reach wording the corpus never uses.
``hybrid``    weighted reciprocal-rank fusion of the channels above.  Ranks are
              the one thing BM25 scores and cosine similarities agree on, so no
              score calibration is invented; weights encode how much each
              channel is trusted.

All channels expose the same ``search(query, top_k) -> list[SearchHit]``
contract, which keeps ``src.tools`` and the agent loop agnostic about the
choice.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.embedding import (
    Embedder,
    EmbeddingCache,
    LsaEmbedder,
    embed_documents,
)
from src.lexicon import EXPANSION_WEIGHT, expand_terms

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pages.jsonl"
FALLBACK_REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
CHUNK_SIZE = 360
CHUNK_OVERLAP = 80
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
FUSION_DEPTH = 60
CHANNELS = ("bm25", "expanded", "dense", "hybrid")
# Off by default: the offline LSA provider measured worse than BM25 on every
# split.  Turn it on after pointing the embedder at a pretrained model and
# re-measuring (see docs/rag-notes.md).
DENSE_WEIGHT = 0.0

# "none" / "builtin" / "auto" are the shipping modes.  "lexicon" rewrites every
# query unconditionally; it is kept for experiments because it measured *worse*
# on the holdout split (see docs/rag-notes.md §2.2).
EXPANSION_MODES = ("none", "builtin", "lexicon", "auto")
# A query whose wording is mostly absent from the corpus is the case where
# synonyms pay off; when the corpus already speaks the query's language,
# rewriting only adds noise.  "auto" therefore expands on OOV pressure alone.
OOV_EXPANSION_THRESHOLD = 0.5

# Ratio names never appear literally in statements, so expand them with the
# line items the answer must be computed from.
QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "毛利率": ("营业收入", "营业成本"),
    "资产负债率": ("负债合计", "资产总计"),
    "营收同比": ("营业收入",),
    "同比增长": ("营业收入",),
    "净利率": ("净利润", "营业收入"),
    "重述": ("重分类", "会计政策变更", "可比期间"),
    "质保费": ("质量保证",),
}

# Line-item questions are usually answered from a primary statement, while
# notes pages mention the same terms far more often.  Prefer the pages whose
# heading is a primary statement; notes-style questions are exempt.
STATEMENT_TITLES = (
    "合并及公司利润表",
    "合并及公司资产负债表",
    "合并及公司现金流量表",
    "合并利润表",
    "合并资产负债表",
    "合并现金流量表",
)
STATEMENT_BONUS = 18.0
NOTES_MARKERS = ("为什么", "重述", "重分类", "影响", "附注", "会计政策", "调整")


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    page: int
    chunk: int
    text: str
    score: float
    channel: str = "bm25"


def normalize(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    # PDF extraction splits Chinese labels across lines ("归属于母公司股东的\n净利润").
    # Join those runs without gluing adjacent numeric columns together.
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return tokens


def chunk_page(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) > size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_length += len(previous)
            current = tail
            length = tail_length
        current.append(line)
        length += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


class ChunkedIndex:
    """Shared plumbing: page/chunk storage, query rewriting, page aggregation."""

    def __init__(self, chunks: list[Chunk], pages: dict[int, str]):
        self.chunks = chunks
        self.pages = pages

    @classmethod
    def from_corpus(cls, path: str | Path = DEFAULT_CORPUS, **options) -> "ChunkedIndex":
        chunks: list[Chunk] = []
        pages: dict[int, str] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                page = int(record["page"])
                text = record.get("text", "")
                pages[page] = text
                for index, piece in enumerate(chunk_page(text)):
                    if piece.strip():
                        chunks.append(Chunk(page=page, index=index, text=piece))
        return cls._build(chunks, pages, **options)

    @classmethod
    def from_report_dir(cls, directory: str | Path = FALLBACK_REPORT_DIR, **options) -> "ChunkedIndex":
        chunks: list[Chunk] = []
        pages: dict[int, str] = {}
        paths = sorted(Path(directory).glob("page-*.txt"), key=lambda item: int(item.stem.split("-")[1]))
        for path in paths:
            page = int(path.stem.split("-")[1])
            text = path.read_text(encoding="utf-8")
            pages[page] = text
            for index, piece in enumerate(chunk_page(text)):
                if piece.strip():
                    chunks.append(Chunk(page=page, index=index, text=piece))
        return cls._build(chunks, pages, **options)

    @classmethod
    def _build(cls, chunks: list[Chunk], pages: dict[int, str], **options) -> "ChunkedIndex":
        return cls(chunks, pages)

    def page_text(self, page: int) -> str:
        return self.pages.get(page, "")

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        raise NotImplementedError

    def _page_hits(self, page_scores: list[tuple[float, int]], top_k: int, channel: str) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for score, position in page_scores[:top_k]:
            chunk = self.chunks[position]
            hits.append(SearchHit(page=chunk.page, chunk=chunk.index, text=chunk.text,
                                  score=score, channel=channel))
        return hits


class PageIndex(ChunkedIndex):
    """BM25 channel (the baseline: strong on numbers and exact line items)."""

    def __init__(self, chunks: list[Chunk], pages: dict[int, str], expansion: str = "builtin"):
        super().__init__(chunks, pages)
        if expansion not in EXPANSION_MODES:
            raise ValueError(f"expansion must be one of {EXPANSION_MODES}")
        self.expansion = expansion
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(chunks)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def _build(cls, chunks: list[Chunk], pages: dict[int, str], **options) -> "PageIndex":
        return cls(chunks, pages, **options)

    def coverage(self, query: str) -> float:
        """Share of the query's tokens that exist somewhere in the corpus."""
        terms = tokenize(query)
        if not terms:
            return 0.0
        return sum(1 for term in set(terms) if term in self.idf) / len(set(terms))

    def weighted_terms(self, query: str) -> dict[str, float]:
        """Query tokens with weights; rewrites stay below the user's own words."""
        weights: dict[str, float] = {}
        for term in tokenize(query):
            weights[term] = 1.0
        if self.expansion == "none":
            return weights
        for key, terms in QUERY_EXPANSIONS.items():
            if key in query:
                for term in terms:
                    for token in tokenize(term):
                        weights.setdefault(token, 1.0)
        rewrite = self.expansion == "lexicon" or (
            self.expansion == "auto" and self.coverage(query) < OOV_EXPANSION_THRESHOLD
        )
        if rewrite:
            for term, weight in expand_terms(query):
                for token in tokenize(term):
                    weights[token] = max(weights.get(token, 0.0), min(weight, EXPANSION_WEIGHT))
        return weights

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        weights = self.weighted_terms(query)
        if not weights or not self.chunks:
            return []
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            for term, weight in weights.items():
                frequency = counts.get(term)
                if not frequency:
                    continue
                denominator = frequency + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[index] / self.avg_length)
                score += weight * self.idf.get(term, 0.0) * frequency * (BM25_K1 + 1) / denominator
            if score > 0:
                scored.append((score, index))

        # Rank pages, not chunks: one strong chunk per page, plus a structural
        # bonus when the page carries the statement the question is about.
        best_by_page: dict[int, tuple[float, int]] = {}
        for score, index in scored:
            page = self.chunks[index].page
            if page not in best_by_page or score > best_by_page[page][0]:
                best_by_page[page] = (score, index)
        ranked = [
            (score + self._statement_bonus(query, page), index)
            for page, (score, index) in best_by_page.items()
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return self._page_hits(ranked, top_k, "bm25")

    def _statement_bonus(self, query: str, page: int) -> float:
        if any(marker in query for marker in NOTES_MARKERS):
            return 0.0
        head = self.pages.get(page, "")[:240]
        return STATEMENT_BONUS if any(title in head for title in STATEMENT_TITLES) else 0.0


class DenseIndex(ChunkedIndex):
    """Cosine channel over chunk embeddings from any :class:`Embedder`."""

    def __init__(self, chunks: list[Chunk], pages: dict[int, str], vectors: np.ndarray,
                 embedder: Embedder):
        super().__init__(chunks, pages)
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors must have one row per chunk")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder = embedder

    @classmethod
    def _build(cls, chunks: list[Chunk], pages: dict[int, str], *, embedder: Embedder | None = None,
               cache: EmbeddingCache | None = None, chunk_tokens: list[list[str]] | None = None) -> "DenseIndex":
        provider = embedder or LsaEmbedder()
        texts = [chunk.text for chunk in chunks]
        tokens = chunk_tokens if chunk_tokens is not None else [tokenize(text) for text in texts]
        vectors = embed_documents(texts, provider, cache=cache, fit_on=tokens)
        return cls(chunks, pages, vectors, provider)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.chunks or not query.strip():
            return []
        query_vector = self.embedder.encode([query])[0]
        if not np.isfinite(query_vector).all() or not query_vector.any():
            return []
        similarities = self.vectors @ query_vector
        best_by_page: dict[int, tuple[float, int]] = {}
        for index, similarity in enumerate(similarities.tolist()):
            page = self.chunks[index].page
            if page not in best_by_page or similarity > best_by_page[page][0]:
                best_by_page[page] = (similarity, index)
        ranked = sorted(((score, index) for page, (score, index) in best_by_page.items()),
                        key=lambda item: (-item[0], item[1]))
        return self._page_hits(ranked, top_k, "dense")


class HybridIndex(ChunkedIndex):
    """Weighted reciprocal-rank fusion over heterogeneous channels.

    RRF uses ranks only, so a BM25 score of 25 and a cosine of 0.62 can be
    combined without inventing a calibration.  Weights encode how far each
    channel is trusted: with ``k=60`` a page ranked #1 by a weight-0.6 channel
    scores ``0.6/61``, which stays below the primary channel's winner
    (``1.0/61``) unless the primary channel does not retrieve it at all.
    """

    def __init__(self, channels: list[tuple[str, ChunkedIndex, float]], *, k: int = RRF_K,
                 depth: int = FUSION_DEPTH):
        if not channels:
            raise ValueError("a hybrid index needs at least one channel")
        primary = channels[0][1]
        super().__init__(primary.chunks, primary.pages)
        self.channels = tuple(channels)
        self.k = k
        self.depth = depth

    @classmethod
    def _build(cls, chunks: list[Chunk], pages: dict[int, str], *, channels: list[tuple[str, ChunkedIndex, float]]) -> "HybridIndex":
        return cls(channels)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            return []
        fused: dict[int, float] = {}
        best_rank: dict[int, tuple[float, int, str]] = {}
        for name, channel, weight in self.channels:
            if weight <= 0:
                continue
            hits = channel.search(query, top_k=max(self.depth, top_k))
            for rank, hit in enumerate(hits, start=1):
                fused[hit.page] = fused.get(hit.page, 0.0) + weight / (self.k + rank)
                contribution = weight / (self.k + rank)
                current = best_rank.get(hit.page)
                if current is None or contribution > current[0]:
                    best_rank[hit.page] = (contribution, hit.chunk, name)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        results: list[SearchHit] = []
        for page, score in ordered[:top_k]:
            _, chunk_index, name = best_rank[page]
            chunk = self._chunk_at(page, chunk_index)
            results.append(SearchHit(page=page, chunk=chunk_index, text=chunk.text,
                                     score=score, channel=name))
        return results

    def _chunk_at(self, page: int, chunk_index: int) -> Chunk:
        for chunk in self.chunks:
            if chunk.page == page and chunk.index == chunk_index:
                return chunk
        return Chunk(page=page, index=chunk_index, text=self.page_text(page))


def build_index(kind: str = "expanded", corpus: str | Path | None = None,
                report_dir: str | Path | None = None,
                embedder: Embedder | None = None, cache: EmbeddingCache | None = None,
                dense_weight: float = DENSE_WEIGHT) -> ChunkedIndex:
    """Build one retrieval channel by name.

    ``bm25``      lexical baseline (ratio-name expansion only).
    ``expanded``  BM25 plus OOV-gated concept rewriting; the measured default.
    ``dense``     embeddings only.
    ``hybrid``    reciprocal-rank fusion of ``expanded`` and ``dense``.  The
                  dense weight is 0 by default because the offline LSA provider
                  measured worse than BM25; fuse once a real model is wired in.

    ``corpus`` is a JSONL file; ``report_dir`` is the ``page-*.txt`` fallback.
    """
    if kind not in CHANNELS:
        raise ValueError(f"unknown retrieval kind: {kind} (choose from {', '.join(CHANNELS)})")
    if report_dir is not None:
        use_corpus, source = False, Path(report_dir)
    elif corpus is not None:
        use_corpus, source = True, Path(corpus)
    elif DEFAULT_CORPUS.exists():
        use_corpus, source = True, DEFAULT_CORPUS
    else:
        use_corpus, source = False, FALLBACK_REPORT_DIR

    def loader(base: type[ChunkedIndex]):
        return base.from_corpus if use_corpus else base.from_report_dir

    if kind == "bm25":
        return loader(PageIndex)(source, expansion="builtin")
    if kind == "expanded":
        return loader(PageIndex)(source, expansion="auto")
    if kind == "dense":
        return loader(DenseIndex)(source, embedder=embedder or LsaEmbedder(), cache=cache)

    channels: list[tuple[str, ChunkedIndex, float]] = [
        ("expanded", build_index("expanded", corpus, report_dir, cache=cache), 1.0),
    ]
    if dense_weight > 0:
        channels.append(("dense", build_index("dense", corpus, report_dir, embedder=embedder, cache=cache),
                         dense_weight))
    return HybridIndex(channels)


def load_index(corpus: str | Path | None = None, kind: str = "expanded", **options) -> ChunkedIndex:
    return build_index(kind, corpus, **options)


__all__ = [
    "CHANNELS",
    "Chunk",
    "ChunkedIndex",
    "DENSE_WEIGHT",
    "DenseIndex",
    "HybridIndex",
    "PageIndex",
    "SearchHit",
    "build_index",
    "chunk_page",
    "load_index",
    "normalize",
    "tokenize",
]
