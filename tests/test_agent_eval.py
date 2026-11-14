import unittest

from evals.agent_eval import (
    check_numeric,
    check_pages,
    cited_pages,
    normalize_numbers,
    select_items,
)


class AgentEvalHelperTests(unittest.TestCase):
    def test_fact_with_thousands_separators_is_recognised(self):
        item = {"check": {"kind": "fact", "expected": "407149600"}}
        self.assertTrue(check_numeric(item, "营业收入 407,149,600 千元")[0])

    def test_metric_rounded_percentage_is_recognised(self):
        item = {"check": {"kind": "metric", "expected": "0.2641895386855347518455132954"}}
        self.assertTrue(check_numeric(item, "毛利率 26.42%")[0])
        self.assertFalse(check_numeric(item, "毛利率 26.4%")[0])

    def test_missing_check_is_not_a_failure(self):
        self.assertIsNone(check_numeric({}, "任意回答")[0])
        self.assertIsNone(check_pages({}, "任意回答")[0])

    def test_page_citation_intersection(self):
        self.assertEqual(cited_pages("见 PDF 第 199 页、第 355 页"), [199, 355])
        self.assertTrue(check_pages({"expected_pages": [199, 126]}, "第 199 页")[0])
        self.assertFalse(check_pages({"expected_pages": [157]}, "没有页码")[0])

    def test_normalize_numbers_removes_both_comma_styles(self):
        self.assertEqual(normalize_numbers("407,149,600，26.42%"), "40714960026.42%")

    def test_select_items_rejects_unknown_ids(self):
        items = [{"id": "q01"}, {"id": "q03"}]
        self.assertEqual([item["id"] for item in select_items(items, "q01", None, None)], ["q01"])
        with self.assertRaises(ValueError):
            select_items(items, "q99", None, None)


if __name__ == "__main__":
    unittest.main()
