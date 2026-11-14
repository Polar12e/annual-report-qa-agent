import unittest

from src.retrieval import (
    PageIndex,
    SearchHit,
    chunk_page,
    chunk_page_spans,
    load_index,
    tokenize,
    window_for_hit,
)


class TokenizeTests(unittest.TestCase):
    def test_commas_in_numbers_are_removed(self):
        self.assertIn("2928031", tokenize("2,928,031"))

    def test_cjk_line_break_is_joined(self):
        tokens = tokenize("归属于母公司股东的\n净利润")
        self.assertIn("的净", tokens)

    def test_bigrams(self):
        self.assertEqual(tokenize("营业收入"), ["营业", "业收", "收入"])


class ChunkTests(unittest.TestCase):
    def test_chunk_page_splits_long_pages(self):
        text = "\n".join(f"第{index}行" + "内容" * 50 for index in range(10))
        chunks = chunk_page(text, size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))


class ChunkSpanTests(unittest.TestCase):
    def test_chunk_spans_are_line_aware(self):
        text = "第一行\n\n第二行\n第三行\n第四行"
        spans = chunk_page_spans(text, size=4, overlap=0)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        self.assertGreater(len(spans), 1)
        for piece, start, end in spans:
            self.assertGreaterEqual(start, 1)
            self.assertLessEqual(start, end)
            self.assertEqual(piece, "\n".join(lines[start - 1:end]))
        self.assertEqual(spans[0][1], 1)

    def test_window_crosses_the_best_chunk_boundary(self):
        page = "标题\n表头\n销售费用 1\n营业成本 2\n三 税项"
        hit = SearchHit(
            page=1, chunk=0, text="标题\n表头\n销售费用 1",
            score=1.0, line_start=1, line_end=3,
        )
        window = window_for_hit(page, hit, radius=2, max_chars=900)
        self.assertIn("销售费用 1", window.text)
        self.assertIn("营业成本 2", window.text)
        self.assertEqual(window.line_start, 1)
        self.assertGreaterEqual(window.line_end, 4)

    def test_window_falls_back_to_chunk_text_without_line_metadata(self):
        page = "a\nb\nc"
        hit = SearchHit(page=1, chunk=0, text="b\nc", score=1.0)
        window = window_for_hit(page, hit, radius=1, max_chars=900)
        self.assertIn("b", window.text)
        self.assertIn("c", window.text)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = load_index()

    def test_income_statement_lookup(self):
        self.assertEqual(self.index.search("2024年营业收入是多少？", top_k=3)[0].page, 159)

    def test_balance_sheet_lookup(self):
        self.assertEqual(self.index.search("2024年末资产总计是多少？", top_k=3)[0].page, 157)

    def test_cash_flow_lookup(self):
        self.assertEqual(self.index.search("2024年经营活动产生的现金流量净额是多少？", top_k=3)[0].page, 161)

    def test_restatement_question_prefers_notes(self):
        pages = [hit.page for hit in self.index.search("质保费重分类对2023年营业成本的影响金额是多少？", top_k=3)]
        self.assertIn(199, pages)

    def test_unknown_query_returns_empty(self):
        self.assertEqual(self.index.search("xxyyzz", top_k=3), [])


class FallbackIndexTests(unittest.TestCase):
    def test_report_dir_fallback(self):
        index = PageIndex.from_report_dir()
        self.assertEqual(sorted(index.pages), [157, 158, 159, 161, 199])
        self.assertIn("营业收入", index.page_text(159))


if __name__ == "__main__":
    unittest.main()
