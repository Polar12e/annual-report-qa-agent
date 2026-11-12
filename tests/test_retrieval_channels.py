"""Channel behaviour: query rewriting, dense cosine, and RRF fusion."""
from __future__ import annotations

import unittest

import numpy as np

from src.embedding import unit_normalize
from src.retrieval import (
    Chunk,
    ChunkedIndex,
    DenseIndex,
    HybridIndex,
    PageIndex,
    SearchHit,
    build_index,
)

PAGES = {
    1: "合并及公司资产负债表\n存货 63,339,188\n",
    2: "合并及公司利润表\n营业收入 407,149,600\n",
    3: "九、公司员工情况\n在职员工的数量合计 198,432 人\n",
}
CHUNKS = [Chunk(page=page, index=0, text=text) for page, text in PAGES.items()]


class ConceptEmbedder:
    """Stand-in for a pretrained model: pools obvious synonyms onto one axis."""

    name = "fake-concepts"
    dim = 4

    def encode(self, texts):
        rows = []
        for text in texts:
            row = np.zeros(4, dtype=np.float32)
            if any(word in text for word in ("存货", "库存", "仓库", "货")):
                row[0] += 1
            if any(word in text for word in ("应收账款", "欠款", "客户")):
                row[1] += 1
            if any(word in text for word in ("营业收入", "收入")):
                row[2] += 1
            if any(word in text for word in ("员工", "薪酬", "上班")):
                row[3] += 1
            rows.append(row)
        return unit_normalize(np.array(rows, dtype=np.float32))


def dense_index(embedder=None) -> DenseIndex:
    provider = embedder or ConceptEmbedder()
    vectors = provider.encode([chunk.text for chunk in CHUNKS])
    return DenseIndex(CHUNKS, PAGES, vectors, provider)


class StubChannel(ChunkedIndex):
    """Channel returning a fixed page order, for fusion arithmetic."""

    def __init__(self, order: list[int]):
        super().__init__(CHUNKS, PAGES)
        self.order = order

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return [SearchHit(page=page, chunk=0, text=PAGES[page], score=1.0, channel="stub")
                for page in self.order[:top_k]]


class ExpansionTest(unittest.TestCase):
    def test_coverage_is_low_for_everyday_wording(self):
        index = PageIndex(CHUNKS, PAGES)
        self.assertLess(index.coverage("仓库里压了多少货"), 0.5)

    def test_coverage_is_high_for_line_items(self):
        index = PageIndex(CHUNKS, PAGES)
        self.assertGreaterEqual(index.coverage("存货 63,339,188"), 0.5)
        self.assertGreater(index.coverage("存货 63,339,188"), index.coverage("仓库里压了多少货"))

    def test_none_mode_only_uses_query_tokens(self):
        index = PageIndex(CHUNKS, PAGES, expansion="none")
        self.assertEqual(set(index.weighted_terms("存货是多少")), {"存货", "货是", "是多", "多少"})

    def test_auto_mode_rewrites_only_under_oov_pressure(self):
        index = PageIndex(CHUNKS, PAGES, expansion="auto")
        self.assertIn("库存", index.weighted_terms("仓库里压了多少货"))
        self.assertNotIn("库存", index.weighted_terms("存货 63,339,188"))

    def test_builtin_expansion_keeps_ratio_terms(self):
        index = PageIndex(CHUNKS, PAGES, expansion="builtin")
        weights = index.weighted_terms("毛利率是多少")
        self.assertIn("营业", weights)
        self.assertIn("成本", weights)

    def test_rewrite_terms_are_down_weighted(self):
        index = PageIndex(CHUNKS, PAGES, expansion="auto")
        weights = index.weighted_terms("仓库里压了多少货")
        self.assertLess(weights["库存"], weights["仓库"])

    def test_unknown_expansion_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            PageIndex(CHUNKS, PAGES, expansion="magic")

    def test_statement_bonus_needs_a_statement_page(self):
        index = PageIndex(CHUNKS, PAGES)
        self.assertGreater(index._statement_bonus("营业收入是多少", 2), 0)
        self.assertEqual(index._statement_bonus("营业收入为什么变化", 2), 0)
        self.assertEqual(index._statement_bonus("营业收入是多少", 3), 0)


class DenseIndexTest(unittest.TestCase):
    def test_similarity_finds_the_semantic_match(self):
        hits = dense_index().search("仓库里压了多少货", top_k=3)
        self.assertEqual(hits[0].page, 1)
        self.assertEqual(hits[0].channel, "dense")

    def test_bm25_misses_what_dense_finds(self):
        lexical = PageIndex(CHUNKS, PAGES, expansion="builtin")
        self.assertEqual([hit.page for hit in lexical.search("仓库里压了多少货", top_k=3)], [])
        self.assertEqual(dense_index().search("仓库里压了多少货", top_k=1)[0].page, 1)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(dense_index().search("   "), [])

    def test_blank_vectors_are_skipped(self):
        class Blank:
            name = "blank"
            dim = 2

            def encode(self, texts):
                return np.zeros((len(texts), 2), dtype=np.float32)

        index = DenseIndex(CHUNKS, PAGES, np.zeros((3, 2), dtype=np.float32), Blank())
        self.assertEqual(index.search("存货"), [])

    def test_vector_row_count_must_match_chunks(self):
        with self.assertRaises(ValueError):
            DenseIndex(CHUNKS, PAGES, np.zeros((2, 4), dtype=np.float32), ConceptEmbedder())


class HybridIndexTest(unittest.TestCase):
    def test_single_channel_preserves_order(self):
        index = HybridIndex([("only", StubChannel([3, 1, 2]), 1.0)])
        self.assertEqual([hit.page for hit in index.search("x")], [3, 1, 2])

    def test_zero_weight_channels_are_ignored(self):
        index = HybridIndex([("primary", StubChannel([2]), 1.0), ("silent", StubChannel([1, 3]), 0.0)])
        self.assertEqual([hit.page for hit in index.search("x")], [2])

    def test_pages_found_by_both_channels_win(self):
        index = HybridIndex([("a", StubChannel([1, 2, 3]), 1.0), ("b", StubChannel([3, 1]), 0.5)])
        self.assertEqual([hit.page for hit in index.search("x")], [1, 3, 2])

    def test_fusion_score_is_the_weighted_reciprocal_rank(self):
        index = HybridIndex([("a", StubChannel([2]), 1.0), ("b", StubChannel([1, 2]), 0.5)], k=60)
        scores = {hit.page: hit.score for hit in index.search("x")}
        self.assertAlmostEqual(scores[2], 1 / 61 + 0.5 / 62, places=9)
        self.assertAlmostEqual(scores[1], 0.5 / 61, places=9)

    def test_empty_channel_list_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridIndex([])

    def test_empty_query_returns_nothing(self):
        index = HybridIndex([("a", StubChannel([1]), 1.0)])
        self.assertEqual(index.search("  "), [])

    def test_fusion_beats_lexical_when_the_dense_channel_is_semantic(self):
        lexical = PageIndex(CHUNKS, PAGES, expansion="builtin")
        hybrid = HybridIndex([("bm25", lexical, 1.0), ("dense", dense_index(), 0.5)])
        self.assertEqual(hybrid.search("仓库里压了多少货", top_k=1)[0].page, 1)
        self.assertEqual(lexical.search("仓库里压了多少货", top_k=1), [])


class BuildIndexTest(unittest.TestCase):
    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            build_index("vectors")

    def test_hybrid_without_dense_weight_has_one_channel(self):
        index = build_index("hybrid")
        self.assertEqual(len(index.channels), 1)

    def test_hybrid_with_dense_weight_has_two_channels(self):
        index = build_index("hybrid", dense_weight=0.3)
        self.assertEqual([name for name, _, _ in index.channels], ["expanded", "dense"])


if __name__ == "__main__":
    unittest.main()
