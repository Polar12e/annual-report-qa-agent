"""A minimal tool-using agent loop.

The loop is model-agnostic: a *policy* decides the next action, the toolbox
executes it, and the observation is fed back into the policy.  ``ScriptedPolicy``
is a rule-based stand-in so the loop can be studied and tested without an API
key; ``OpenAIPolicy`` (src/llm_policy.py) implements the same interface with a
real function-calling model.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from src.llm_config import PROVIDERS, LLMSettings
from src.retrieval import CHANNELS
from src.tools import FinancialToolBox

COMPANY_PATTERN = re.compile(r"\b\d{6}\.(?:SZ|SH|BJ)\b")
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")


@dataclass
class Step:
    tool: str
    arguments: dict
    observation: dict
    tool_call_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "observation": self.observation,
            "tool_call_id": self.tool_call_id,
        }


class ScriptedPolicy:
    """Rule-based policy: enough to demonstrate the loop, not a real LLM."""

    def __init__(self, default_company: str = "000333.SZ", default_year: int = 2024):
        self.default_company = default_company
        self.default_year = default_year

    def decide(self, question: str, history: list[Step]) -> dict:
        if history:
            last = history[-1]
            if last.tool == "search_report" and len(history) == 1:
                observation = last.observation
                if observation.get("ok") and observation["result"]["count"]:
                    page = observation["result"]["matches"][0]["pdf_page"]
                    return {"type": "tool", "tool": "read_page", "arguments": {"page": page}}
            return {"type": "final", "answer": self._answer(last)}
        return self._first_action(question)

    def _first_action(self, question: str) -> dict:
        if any(word in question for word in ("有哪些", "哪些数据", "数据范围", "可用数据")):
            return {"type": "tool", "tool": "list_available_data", "arguments": {}}
        if any(word in question for word in ("为什么", "重述", "重分类", "质保费", "附注", "会计政策", "差异", "来源")):
            return {"type": "tool", "tool": "search_report", "arguments": {"keyword": question, "max_results": 3}}
        company = COMPANY_PATTERN.search(question)
        year = YEAR_PATTERN.search(question)
        arguments = {
            "company_code": company.group(0) if company else self.default_company,
            "year": int(year.group(0)) if year else self.default_year,
        }
        if any(word in question for word in ("毛利率", "资产负债率", "同比", "增长", "指标")):
            return {"type": "tool", "tool": "calculate_metrics", "arguments": arguments}
        if any(word in question for word in ("营收", "收入", "净利润", "现金流", "资产", "负债", "成本")):
            return {"type": "tool", "tool": "query_financials", "arguments": arguments}
        return {"type": "final", "answer": "我只能回答与已入库财务数据相关的问题。"}

    def _answer(self, step: Step) -> str:
        observation = step.observation
        if not observation.get("ok"):
            error = observation["error"]
            return f"工具调用失败（{error['code']}）：{error['message']}"
        result = observation["result"]
        if step.tool == "list_available_data":
            return (
                f"已核验数据：公司 {', '.join(result['companies'])}；"
                f"年份 {', '.join(map(str, result['years']))}；共 {result['fact_count']} 条事实。"
            )
        if step.tool == "query_financials":
            if result["count"] == 0:
                return "没有查到匹配的已核验事实。"
            lines = [
                f"{fact['metric_code']} = {fact['value']} {fact['original_unit']}（PDF 第 {fact['pdf_page']} 页）"
                for fact in result["facts"]
            ]
            return "查询到 {} 条事实：\n{}".format(result["count"], "\n".join(lines))
        if step.tool == "calculate_metrics":
            lines = []
            for item in result["results"]:
                if item["status"] == "ok":
                    lines.append(f"{item['metric_code']} = {item['value'] * 100:.2f}%")
                else:
                    lines.append(f"{item['metric_code']} = {item['status']}（{item['error']}）")
            return "\n".join(lines)
        if step.tool == "search_report":
            if result["count"] == 0:
                return f"没有找到包含“{result['keyword']}”的页面。"
            return "检索到 {} 段内容：\n{}".format(
                result["count"],
                "\n".join(
                    f"PDF 第 {match['pdf_page']} 页（相关度 {match['score']}）：{match['text'][:160]}"
                    for match in result["matches"]
                ),
            )
        if step.tool == "read_page":
            lines = [line.strip() for line in result["text"].splitlines()]
            numeric = [line for line in lines if re.search(r"\d{1,3}(?:,\d{3})+", line)]
            excerpt = "\n".join(numeric[:10]) or "\n".join(lines[:12])
            return f"PDF 第 {result['pdf_page']} 页（含金额的行）：\n{excerpt}"
        return "未知工具结果。"


def run(question: str, policy=None, toolbox: FinancialToolBox | None = None, max_steps: int = 6) -> dict:
    policy = policy or ScriptedPolicy()
    toolbox = toolbox or FinancialToolBox()
    history: list[Step] = []
    for _ in range(max_steps):
        try:
            action = policy.decide(question, history)
        except Exception as exc:  # policy/network failure must not crash the loop
            return {
                "answer": f"策略调用失败：{type(exc).__name__}: {exc}",
                "steps": [step.as_dict() for step in history],
                "stopped": "error",
            }
        if action.get("type") == "final":
            return {"answer": action["answer"], "steps": [step.as_dict() for step in history], "stopped": "final"}
        if action.get("type") != "tool":
            return {"answer": "策略返回了无法识别的动作。", "steps": [step.as_dict() for step in history], "stopped": "error"}
        arguments = action.get("arguments", {})
        observation = toolbox.call(action["tool"], arguments)
        history.append(
            Step(
                tool=action["tool"],
                arguments=arguments,
                observation=observation,
                tool_call_id=action.get("tool_call_id"),
            )
        )
    return {"answer": "达到最大步数仍未得出结论。", "steps": [step.as_dict() for step in history], "stopped": "max_steps"}


def build_policy(args):
    if args.policy == "llm":
        from src.llm_policy import OpenAIPolicy

        settings = LLMSettings.from_env(
            provider=getattr(args, "provider", None),
            model=args.model,
            base_url=args.base_url,
            timeout=getattr(args, "timeout", None),
            max_retries=getattr(args, "max_retries", None),
            thinking=getattr(args, "thinking", None),
            allow_external=getattr(args, "allow_external", None),
        )
        return OpenAIPolicy.from_settings(settings)
    return ScriptedPolicy()


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 Agent 循环（默认脚本策略，无需 API key）")
    parser.add_argument("question", nargs="?", default="美的集团 2024 年毛利率是多少？")
    parser.add_argument("--policy", choices=("scripted", "llm"), default="scripted")
    parser.add_argument("--provider", choices=PROVIDERS,
                        help="模型提供方，默认 auto（按环境变量推断）")
    parser.add_argument("--model", help="模型名，默认读 FIN_AGENT_MODEL；DeepSeek 默认 deepseek-flash")
    parser.add_argument("--base-url", help="OpenAI 兼容端点，默认读 FIN_AGENT_BASE_URL/OPENAI_BASE_URL")
    parser.add_argument("--thinking", action="store_true", default=None,
                        help="启用 DeepSeek 思考模式（默认关闭）")
    parser.add_argument("--timeout", type=float, default=None,
                        help="请求超时秒数，默认读 FIN_AGENT_TIMEOUT（30）")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="SDK 重试次数，默认读 FIN_AGENT_MAX_RETRIES（2）")
    parser.add_argument("--allow-external", action="store_true", default=None,
                        help="允许调用公网模型端点（默认只允许内网/本地）")
    parser.add_argument("--retriever", choices=CHANNELS, default="expanded", help="检索通道，默认 expanded")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    try:
        policy = build_policy(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    toolbox = FinancialToolBox(retriever=args.retriever)
    result = run(args.question, policy=policy, max_steps=args.max_steps, toolbox=toolbox)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
