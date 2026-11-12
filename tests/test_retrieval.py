import unittest

from src.retrieval import PageIndex, chunk_page, load_index, tokenize


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
