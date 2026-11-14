import unittest
from decimal import Decimal
from pathlib import Path

from src.policy import DataPolicy
from src.tools import TOOL_SPECS, FinancialToolBox

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"


class ToolSpecTests(unittest.TestCase):
    def test_specs_are_json_schema_shaped(self):
        self.assertEqual(len(TOOL_SPECS), 5)
        names = [spec["name"] for spec in TOOL_SPECS]
        self.assertEqual(len(names), len(set(names)))
        for spec in TOOL_SPECS:
            self.assertIsInstance(spec["name"], str)
            self.assertIsInstance(spec["description"], str)
            self.assertEqual(spec["parameters"]["type"], "object")


class ToolCallTests(unittest.TestCase):
    def setUp(self):
        self.box = FinancialToolBox()

    def test_list_available_data(self):
        result = self.box.call("list_available_data")
        self.assertTrue(result["ok"])
        self.assertIn("000333.SZ", result["result"]["companies"])
        self.assertEqual(result["result"]["fact_count"], 12)
        self.assertIn("revenue", result["result"]["metrics"])

    def test_query_financials(self):
        result = self.box.call("query_financials", {"company_code": "000333.SZ", "year": 2024, "metric_codes": ["revenue"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["count"], 1)
        self.assertEqual(result["result"]["facts"][0]["value"], Decimal(407149600))

    def test_query_financials_defaults_to_all_metrics(self):
        result = self.box.call("query_financials", {"company_code": "000333.SZ", "year": 2024})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["count"], 6)

    def test_calculate_metrics(self):
        result = self.box.call("calculate_metrics", {"company_code": "000333.SZ", "year": 2024})
        self.assertTrue(result["ok"])
        self.assertEqual([item["status"] for item in result["result"]["results"]], ["ok", "ok", "ok"])
        self.assertEqual(round(result["result"]["results"][1]["value"] * 100, 2), Decimal("26.42"))

    def test_calculate_metrics_for_unknown_company_is_not_an_error(self):
        result = self.box.call("calculate_metrics", {"company_code": "999999.SZ", "year": 2024})
        self.assertTrue(result["ok"])
        self.assertTrue(all(item["status"] == "missing" for item in result["result"]["results"]))

    def test_search_report(self):
        result = self.box.call("search_report", {"keyword": "营业成本", "max_results": 3})
        self.assertTrue(result["ok"])
        self.assertGreater(result["result"]["count"], 0)
        self.assertLessEqual(result["result"]["count"], 3)

    def test_search_report_returns_ranked_pages(self):
        result = self.box.call("search_report", {"keyword": "2024年营业收入是多少？", "max_results": 3})
        self.assertTrue(result["ok"])
        top = result["result"]["matches"][0]
        self.assertEqual(top["pdf_page"], 159)
        self.assertIn("score", top)

    def test_search_report_returns_hit_window(self):
        result = self.box.call(
            "search_report",
            {"keyword": "2023年营业成本为什么重述？", "max_results": 3},
        )
        self.assertTrue(result["ok"])
        top = result["result"]["matches"][0]
        self.assertEqual(top["pdf_page"], 199)
        self.assertIn("销售费用", top["text"])
        self.assertIn("营业成本", top["text"])
        self.assertIn("page_heading", top)
        self.assertGreaterEqual(top["line_start"], 1)
        self.assertGreaterEqual(top["line_end"], top["line_start"])
        self.assertIn("matched_terms", top)

    def test_search_snippets_follow_policy(self):
        box = FinancialToolBox(policy=DataPolicy(max_snippets=2, max_snippet_chars=30))
        result = box.call("search_report", {"keyword": "营业成本", "max_results": 5})
        self.assertTrue(result["ok"])
        self.assertLessEqual(result["result"]["count"], 2)
        self.assertTrue(all(len(match["text"]) <= 30 for match in result["result"]["matches"]))

    def test_search_audits_pages_only(self):
        events = []
        box = FinancialToolBox(policy=DataPolicy(audit_hook=events.append))
        box.call("search_report", {"keyword": "营业收入", "max_results": 1})
        self.assertEqual(events[0]["tool"], "search_report")
        self.assertIn(159, events[0]["pages"])
        self.assertNotIn("keyword", events[0])

    def test_read_page(self):
        result = self.box.call("read_page", {"page": 199})
        self.assertTrue(result["ok"])
        self.assertIn("质保费", result["result"]["text"])

    def test_read_page_unknown_page(self):
        result = self.box.call("read_page", {"page": 9999})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_search_report_missing_directory(self):
        box = FinancialToolBox(report_dir=REPORTS / "does-not-exist")
        result = box.call("search_report", {"keyword": "营业成本"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "DATA_ERROR")

    def test_unknown_tool(self):
        result = self.box.call("nope", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")

    def test_missing_arguments(self):
        result = self.box.call("query_financials", {"company_code": "000333.SZ"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_wrong_argument_types(self):
        result = self.box.call("query_financials", {"company_code": 1, "year": "x"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_unknown_metric_code(self):
        result = self.box.call("query_financials", {"company_code": "000333.SZ", "year": 2024, "metric_codes": ["roe"]})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_tool_boundary_never_raises(self):
        cases = [
            ("query_financials", {"company_code": "", "year": 2024}),
            ("calculate_metrics", {}),
            ("search_report", {"keyword": ""}),
            ("search_report", {"keyword": "营业成本", "max_results": 0}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                result = self.box.call(name, arguments)
                self.assertFalse(result["ok"])
                self.assertIn("code", result["error"])


if __name__ == "__main__":
    unittest.main()
