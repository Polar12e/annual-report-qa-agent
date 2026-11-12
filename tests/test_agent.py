import unittest

from src.agent import run
from src.tools import FinancialToolBox


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.toolbox = FinancialToolBox()

    def test_answers_gross_margin_question(self):
        result = run("美的集团 2024 年毛利率是多少？", toolbox=self.toolbox)
        self.assertEqual(result["stopped"], "final")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["tool"], "calculate_metrics")
        self.assertIn("gross_margin", result["answer"])
        self.assertIn("26.42%", result["answer"])

    def test_routes_data_inventory_question(self):
        result = run("现在有哪些数据？", toolbox=self.toolbox)
        self.assertEqual(result["steps"][0]["tool"], "list_available_data")
        self.assertIn("000333.SZ", result["answer"])

    def test_unknown_question_returns_final_without_tools(self):
        result = run("今天天气怎么样？", toolbox=self.toolbox)
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["stopped"], "final")

    def test_empty_query_is_reported(self):
        result = run("查询 999999.SZ 2024 年营收", toolbox=self.toolbox)
        self.assertEqual(result["steps"][0]["tool"], "query_financials")
        self.assertIn("没有查到", result["answer"])

    def test_routes_notes_question_to_retrieval(self):
        result = run("2023年营业成本为什么重述？", toolbox=self.toolbox)
        self.assertEqual([step["tool"] for step in result["steps"]], ["search_report", "read_page"])
        self.assertIn("199", result["answer"])
        self.assertIn("2,928,031", result["answer"])

    def test_max_steps_is_respected(self):
        class LoopingPolicy:
            def decide(self, question, history):
                return {"type": "tool", "tool": "list_available_data", "arguments": {}}

        result = run("x", policy=LoopingPolicy(), toolbox=self.toolbox, max_steps=2)
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["stopped"], "max_steps")

    def test_policy_failure_is_reported(self):
        class BrokenPolicy:
            def decide(self, question, history):
                raise RuntimeError("boom")

        result = run("x", policy=BrokenPolicy(), toolbox=self.toolbox)
        self.assertEqual(result["stopped"], "error")
        self.assertIn("boom", result["answer"])


if __name__ == "__main__":
    unittest.main()
