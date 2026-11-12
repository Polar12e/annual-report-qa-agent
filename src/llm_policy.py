"""OpenAI-compatible function-calling policy for the agent loop.

The policy is deliberately thin: it turns the conversation into chat messages,
asks the model for one action, and returns the same action shape the scripted
policy uses.  A client can be injected (tests do this), so no network or API
key is required to exercise the loop.

Environment variables:

- ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``: credentials
- ``OPENAI_BASE_URL``: custom endpoint (DeepSeek defaults to https://api.deepseek.com)
- ``FIN_AGENT_MODEL``: model name (default ``gpt-4o-mini`` or ``deepseek-chat``)

Data boundary: public endpoints are refused unless ``allow_external=True``
(see ``src/policy.py``).  Internal deployments work out of the box.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import TYPE_CHECKING

from src.policy import DataPolicy
from src.tools import TOOL_SPECS

if TYPE_CHECKING:
    from src.agent import Step

SYSTEM_PROMPT = """你是一名严谨的财务分析助手。
规则：
1. 只能通过提供的工具获取数字，禁止凭记忆编造。
2. 每次只调用一个工具，拿到结果后再决定下一步。
3. 回答时引用工具返回的 PDF 页码或 record_id。
4. 如果工具返回 ok=false 或 status=invalid/missing，要说明原因，不要猜测。
"""


def to_openai_tools(specs: tuple[dict, ...] = TOOL_SPECS) -> list[dict]:
    """Convert the project's tool specs into the OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in specs
    ]


class OpenAIPolicy:
    """Policy backed by an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client=None,
        timeout: float = 30.0,
        policy: DataPolicy | None = None,
        allow_external: bool | None = None,
    ):
        self.policy = policy or DataPolicy()
        if allow_external is not None:
            self.policy = replace(self.policy, allow_external=allow_external)
        self.model = model or os.environ.get("FIN_AGENT_MODEL")
        self.base_url = base_url
        self.tools = to_openai_tools()
        self._client = client
        if client is not None:
            self.model = self.model or "gpt-4o-mini"
            return

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                base_url = base_url or "https://api.deepseek.com"
                self.model = self.model or "deepseek-chat"
        if not api_key:
            raise RuntimeError(
                "未配置模型凭据：请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY；"
                "也可以注入 client 以便离线测试。"
            )
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.base_url = base_url
        self.policy.ensure_endpoint(base_url)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai SDK 不可用，请先执行 pip install -U openai") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = self.model or "gpt-4o-mini"

    def decide(self, question: str, history: list[Step]) -> dict:
        messages = self._messages(question, history)
        self.policy.audit(
            {
                "provider": "openai-compatible",
                "model": self.model,
                "endpoint": self.base_url,
                "messages": len(messages),
                "history_steps": len(history),
                "pages": _pages_from_history(history),
            }
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if calls:
            call = calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"模型返回的工具参数不是合法 JSON：{exc}") from exc
            return {
                "type": "tool",
                "tool": call.function.name,
                "arguments": arguments,
                "tool_call_id": call.id,
            }
        return {"type": "final", "answer": message.content or ""}

    def _messages(self, question: str, history: list[Step]) -> list[dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for index, step in enumerate(history):
            call_id = step.tool_call_id or f"call_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": step.tool,
                                "arguments": json.dumps(step.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(step.observation, ensure_ascii=False, default=str),
                }
            )
        return messages


def _pages_from_history(history: list[Step]) -> list[int]:
    pages: set[int] = set()
    for step in history:
        result = step.observation.get("result", {})
        for key in ("matches", "facts"):
            for item in result.get(key, []) or []:
                page = item.get("pdf_page")
                if isinstance(page, int):
                    pages.add(page)
    return sorted(pages)
