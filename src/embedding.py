"""Embedding providers for the dense retrieval channel.

The dense channel needs to turn text into unit vectors.  Which model does that
is a deployment decision, not an architectural one, so everything downstream
only sees the small :class:`Embedder` protocol:

* :class:`LsaEmbedder` — offline default.  It fits a TF-IDF matrix over the
  corpus' own tokens and keeps the leading singular directions (LSA).  It is
  honest about what it is: a *corpus-fitted* space, not a pretrained semantic
  model.  It is deterministic, dependency-free (numpy only) and needs no
  network, which makes the whole pipeline runnable and testable offline.
* :class:`OpenAICompatibleEmbedder` — any ``POST /embeddings`` endpoint
  (OpenAI, DeepSeek-compatible gateways, Ollama, vLLM).  This is the provider
  you want in production; it is the only piece that changes when a real model
  becomes available.

Embeddings are cached on disk keyed by corpus content + embedding settings, so
rebuilding an index does not re-embed 1,450 chunks every run.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

EMBEDDING_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "embeddings"


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce vectors."""


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract the dense channel depends on."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of unit vectors."""


def unit_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def randomized_svd(matrix: np.ndarray, components: int, *, oversample: int = 20,
                   power_iterations: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Randomized SVD (Halko et al.) returning the top right singular vectors.

    Returns ``(singular_values, basis)`` where ``basis`` is ``(features, k)`` and
    ``matrix @ basis`` approximates the projected document matrix.  Numpy-only on
    purpose: the project otherwise has no numerical dependency besides numpy.
    """
    rows, features = matrix.shape
    k = min(components, min(rows, features))
    if k <= 0:
        raise ValueError("components must be positive")
    rng = np.random.default_rng(seed)
    sketch = rng.standard_normal((features, k + oversample)).astype(np.float32)
    for _ in range(power_iterations):
        sketch, _ = np.linalg.qr(matrix.T @ (matrix @ sketch))
    q, _ = np.linalg.qr(matrix @ sketch)          # (rows, k + oversample)
    small = q.T @ matrix                           # (k + oversample, features)
    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    basis = vt[:k].T.astype(np.float32)            # (features, k)
    return singular[:k], basis


@dataclass
class LsaEmbedder:
    """Corpus-fitted latent semantic space (offline fallback provider)."""

    dim: int = 192
    min_df: int = 2
    max_df_ratio: float = 0.25
    seed: int = 0
    name: str = field(default="lsa", init=False)
    vocab: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    idf: np.ndarray | None = field(default=None, init=False, repr=False)
    basis: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if not 0 < self.max_df_ratio <= 1:
            raise ValueError("max_df_ratio must be in (0, 1]")
        self.name = f"lsa-{self.dim}"

    @property
    def fitted(self) -> bool:
        return self.basis is not None

    def fit(self, token_lists: Sequence[Sequence[str]]) -> "LsaEmbedder":
        if not token_lists:
            raise EmbeddingError("cannot fit LSA on an empty corpus")
        documents = len(token_lists)
        document_frequency: Counter[str] = Counter()
        for tokens in token_lists:
            document_frequency.update(set(tokens))
        ceiling = max(self.min_df, int(self.max_df_ratio * documents))
        kept = {term: index for index, term in enumerate(
            sorted(term for term, count in document_frequency.items()
                   if self.min_df <= count <= ceiling)
        )}
        if len(kept) < self.dim:
            raise EmbeddingError(f"vocabulary too small for dim={self.dim}: {len(kept)} terms")
        self.vocab = kept
        self.idf = np.array(
            [math.log((documents + 1) / (document_frequency[term] + 0.5)) for term in kept],
            dtype=np.float32,
        )
        matrix = np.zeros((documents, len(kept)), dtype=np.float32)
        for row, tokens in enumerate(token_lists):
            for term, count in Counter(tokens).items():
                column = kept.get(term)
                if column is not None:
                    matrix[row, column] = (1.0 + math.log(count)) * self.idf[column]
        matrix = unit_normalize(matrix)
        _, self.basis = randomized_svd(matrix, self.dim, seed=self.seed)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            raise EmbeddingError("LsaEmbedder.encode() called before fit()")
        from src.retrieval import tokenize  # local import: retrieval imports this module

        assert self.basis is not None and self.idf is not None
        rows = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                column = self.vocab.get(term)
                if column is not None:
                    rows[row, column] = (1.0 + math.log(count)) * self.idf[column]
        projected = unit_normalize(rows) @ self.basis
        return unit_normalize(projected)

    def state(self) -> dict:
        return {"kind": "lsa", "dim": self.dim, "min_df": self.min_df,
                "max_df_ratio": self.max_df_ratio, "seed": self.seed}

    def dumps(self) -> bytes:
        if not self.fitted:
            raise EmbeddingError("cannot serialise an unfitted LsaEmbedder")
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        payload = {
            "state": self.state(),
            "terms": terms,
            "idf": self.idf.tolist() if self.idf is not None else [],
            "basis": self.basis.tolist() if self.basis is not None else [],
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    @classmethod
    def loads(cls, blob: bytes) -> "LsaEmbedder":
        payload = json.loads(blob.decode("utf-8"))
        state = payload["state"]
        embedder = cls(dim=state["dim"], min_df=state["min_df"],
                       max_df_ratio=state["max_df_ratio"], seed=state["seed"])
        embedder.vocab = {term: index for index, term in enumerate(payload["terms"])}
        embedder.idf = np.array(payload["idf"], dtype=np.float32)
        embedder.basis = np.array(payload["basis"], dtype=np.float32)
        return embedder


Transport = Callable[[urllib.request.Request, float], bytes]


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - endpoint is policy-checked
        return response.read()


@dataclass
class OpenAICompatibleEmbedder:
    """``POST {base_url}/embeddings`` provider (OpenAI, DeepSeek gateways, Ollama…)."""

    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    batch_size: int = 64
    timeout: float = 30.0
    retries: int = 2
    dim: int = 1536
    name: str = field(default="", init=False)
    transport: Transport = field(default=_urlopen, repr=False)

    def __post_init__(self) -> None:
        self.name = f"api:{self.model}"
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode_batch(list(texts[start:start + self.batch_size])))
        matrix = np.asarray([self._check_width(vector) for vector in vectors], dtype=np.float32)
        return unit_normalize(matrix)

    def _check_width(self, vector: Sequence[float]) -> Sequence[float]:
        if len(vector) != self.dim:
            raise EmbeddingError(
                f"endpoint returned {len(vector)} dimensions, expected {self.dim}; "
                "set OpenAICompatibleEmbedder.dim to match the model"
            )
        return vector

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/embeddings", data=body, headers=headers, method="POST"
            )
            try:
                payload = json.loads(self.transport(request, self.timeout).decode("utf-8"))
            except urllib.error.HTTPError as exc:  # 4xx should not be retried
                detail = exc.read().decode("utf-8", "replace")[:200]
                raise EmbeddingError(f"embedding endpoint returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            except json.JSONDecodeError as exc:
                raise EmbeddingError(f"embedding endpoint returned invalid JSON: {exc}") from exc
            else:
                return self._parse(payload, len(texts))
        raise EmbeddingError(f"embedding endpoint unreachable after {self.retries + 1} attempts: {last}")

    @staticmethod
    def _parse(payload: dict, expected: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(f"embedding response shape mismatch: expected {expected} vectors")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        try:
            return [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"embedding response is missing 'embedding': {exc}") from exc


def corpus_fingerprint(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class EmbeddingCache:
    """On-disk float32 cache for chunk vectors, keyed by content + provider."""

    def __init__(self, directory: str | Path = EMBEDDING_CACHE_DIR) -> None:
        self.directory = Path(directory)

    def key(self, provider: str, fingerprint: str, count: int, dim: int) -> str:
        raw = f"{provider}|{fingerprint}|{count}|{dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.npz"

    def load(self, key: str, count: int, dim: int) -> np.ndarray | None:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            with np.load(path) as archive:
                vectors = archive["vectors"]
        except (OSError, ValueError, KeyError):
            return None
        if vectors.shape != (count, dim) or vectors.dtype != np.float32:
            return None
        if not np.isfinite(vectors).all():
            return None
        return vectors

    def save(self, key: str, vectors: np.ndarray) -> None:
        # A cache is an optimisation: a read-only checkout must still retrieve.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path(key).with_suffix(".tmp.npz")
            np.savez(temporary, vectors=np.asarray(vectors, dtype=np.float32))
            temporary.replace(self.path(key))
        except OSError:
            return


def embed_documents(texts: Sequence[str], embedder: Embedder, *,
                    cache: EmbeddingCache | None = None,
                    fit_on: Sequence[Sequence[str]] | None = None) -> np.ndarray:
    """Embed ``texts`` with caching; fits the provider first when it needs it."""
    if isinstance(embedder, LsaEmbedder) and not embedder.fitted:
        if fit_on is None:
            raise EmbeddingError("LsaEmbedder needs corpus tokens: pass fit_on=[tokens, ...]")
        embedder.fit(fit_on)
    assert embedder.dim > 0
    store = cache or EmbeddingCache()
    key = store.key(embedder.name, corpus_fingerprint(texts), len(texts), embedder.dim)
    cached = store.load(key, len(texts), embedder.dim)
    if cached is not None:
        return cached
    vectors = embedder.encode(texts)
    if vectors.shape != (len(texts), embedder.dim):
        raise EmbeddingError(f"provider returned {vectors.shape}, expected {(len(texts), embedder.dim)}")
    store.save(key, vectors)
    return vectors


__all__ = [
    "EMBEDDING_CACHE_DIR",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingError",
    "LsaEmbedder",
    "OpenAICompatibleEmbedder",
    "corpus_fingerprint",
    "embed_documents",
    "randomized_svd",
    "unit_normalize",
]
