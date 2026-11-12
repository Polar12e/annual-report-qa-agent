import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import run
from src.llm_policy import OpenAIPolicy, to_openai_tools
from src.policy import PolicyViolation
from src.tools import FinancialToolBox


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def tool_response(name, arguments, call_id="call_1"):
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
            )
        ],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def final_response(text):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ToolSpecConversionTests(unittest.TestCase):
    def test_specs_convert_to_openai_shape(self):
        tools = to_openai_tools()
        self.assertEqual(len(tools), 5)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "list_available_data")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")


class PolicyDecisionTests(unittest.TestCase):
    def test_policy_returns_tool_action(self):
        client = FakeClient([tool_response("calculate_metrics", {"company_code": "000333.SZ", "year": 2024})])
        policy = OpenAIPolicy(client=client, model="test")
        action = policy.decide("毛利率？", [])
        self.assertEqual(action["type"], "tool")
        self.assertEqual(action["tool"], "calculate_metrics")
        self.assertEqual(action["arguments"]["year"], 2024)
        self.assertEqual(action["tool_call_id"], "call_1")

    def test_policy_returns_final_answer(self):
        policy = OpenAIPolicy(client=FakeClient([final_response("毛利率是 26.42%")]), model="test")
        self.assertEqual(policy.decide("毛利率？", []), {"type": "final", "answer": "毛利率是 26.42%"})

    def test_invalid_json_arguments_raise(self):
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="calculate_metrics", arguments="{not json"))],
        )
        policy = OpenAIPolicy(client=FakeClient([SimpleNamespace(choices=[SimpleNamespace(message=message)])]), model="test")
        with self.assertRaises(ValueError):
            policy.decide("x", [])

    def test_missing_credentials_raise(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                OpenAIPolicy()

    def test_external_endpoint_is_blocked_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PolicyViolation):
                OpenAIPolicy(api_key="k", base_url="https://api.openai.com/v1")


class LLMLoopTests(unittest.TestCase):
    def test_loop_executes_tool_and_replays_observation(self):
        client = FakeClient(
            [
                tool_response("calculate_metrics", {"company_code": "000333.SZ", "year": 2024}),
                final_response("毛利率是 26.42%，输入记录 midea24-revenue 与 midea24-cost。"),
            ]
        )
        policy = OpenAIPolicy(client=client, model="test")
        result = run("美的 2024 毛利率？", policy=policy, toolbox=FinancialToolBox())
        self.assertEqual(result["stopped"], "final")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["tool"], "calculate_metrics")
        self.assertEqual(result["steps"][0]["tool_call_id"], "call_1")
        self.assertIn("26.42", result["answer"])

        second_request = client.chat.completions.requests[1]["messages"]
        self.assertEqual(second_request[0]["role"], "system")
        self.assertEqual(second_request[1]["role"], "user")
        self.assertEqual(second_request[2]["role"], "assistant")
        self.assertEqual(second_request[3]["role"], "tool")
        self.assertEqual(second_request[3]["tool_call_id"], "call_1")
        self.assertIn("0.26418", second_request[3]["content"])


if __name__ == "__main__":
    unittest.main()
