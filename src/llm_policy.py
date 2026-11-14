"""OpenAI-compatible function-calling policy for the agent loop.

The policy is deliberately thin: it turns the conversation into chat messages,
asks the model for one action, and returns the same action shape the scripted
policy uses.  Connection details are resolved by :class:`src.llm_config.LLMSettings`;
the client can be injected (tests do this), so no network or API key is needed
to exercise the loop.

Environment variables are documented in ``src/llm_config.py``.  The most common
ones are ``DEEPSEEK_API_KEY``, ``OPENAI_API_KEY``, ``FIN_AGENT_PROVIDER`` and
``FIN_AGENT_MODEL``.

Data boundary: public endpoints are refused unless ``allow_external=True``
(see ``src/policy.py``).  Internal deployments work out of the box.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from src.llm_config import LLMSettings
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
5. list_available_data 只覆盖已核验的六个结构化指标；如果问题涉及存货、研发费用、销售费用、员工人数、分红等未入库科目，必须继续调用 search_report/read_page 查年报原文，不能仅凭 list_available_data 回答“没有数据”。
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
        timeout: float | None = None,
        policy: DataPolicy | None = None,
        allow_external: bool | None = None,
        thinking: bool | None = None,
        max_retries: int | None = None,
        settings: LLMSettings | None = None,
    ):
        self.tools = to_openai_tools()
        self._client = client

        # ``client`` is the test seam: no credentials, endpoint checks or SDK
        # import are needed when a client is injected.
        if client is not None:
            self.provider = settings.provider if settings is not None else "injected"
            self.model = model or (settings.model if settings is not None else "gpt-4o-mini")
            self.base_url = base_url or (settings.base_url if settings is not None else None)
            self.thinking = (
                thinking
                if thinking is not None
                else (settings.thinking if settings is not None else False)
            )
            self.max_retries = (
                max_retries
                if max_retries is not None
                else (settings.max_retries if settings is not None else 2)
            )
            self.policy = self._resolve_policy(policy, allow_external, settings)
            return

        if settings is None:
            settings = LLMSettings.from_env(
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                thinking=thinking,
                allow_external=allow_external,
            )

        self.provider = settings.provider
        self.model = settings.model
        self.base_url = settings.base_url
        self.thinking = settings.thinking
        self.max_retries = settings.max_retries
        self.policy = self._resolve_policy(policy, allow_external, settings)
        self.policy.ensure_endpoint(self.base_url)

        if not settings.api_key:
            raise RuntimeError("模型凭据缺失：LLMSettings.api_key 为空。")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai SDK 不可用，请先执行 pip install -U openai") from exc

        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=self.base_url,
            timeout=settings.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_settings(
        cls,
        settings: LLMSettings,
        *,
        client=None,
        policy: DataPolicy | None = None,
    ) -> OpenAIPolicy:
        """Build a policy from an already-resolved configuration."""
        return cls(settings=settings, client=client, policy=policy)

    @staticmethod
    def _resolve_policy(
        policy: DataPolicy | None,
        allow_external: bool | None,
        settings: LLMSettings | None,
    ) -> DataPolicy:
        # A caller-supplied policy is authoritative; settings only provide the
        # default when we construct one.  An explicit ``allow_external`` always
        # wins because it is the most local decision.
        if policy is None:
            resolved = DataPolicy(
                allow_external=settings.allow_external if settings is not None else False
            )
        else:
            resolved = policy
        if allow_external is not None:
            resolved = replace(resolved, allow_external=allow_external)
        return resolved

    @property
    def _uses_deepseek(self) -> bool:
        if self.provider == "deepseek":
            return True
        return bool(self.base_url and "deepseek" in self.base_url.lower())

    def decide(self, question: str, history: list[Step]) -> dict:
        messages = self._messages(question, history)
        self.policy.audit(
            {
                "provider": self.provider,
                "model": self.model,
                "endpoint": self.base_url,
                "messages": len(messages),
                "history_steps": len(history),
                "pages": _pages_from_history(history),
            }
        )
        request = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
        }
        if self._uses_deepseek:
            # DeepSeek enables thinking by default.  This loop needs a single
            # deterministic action per turn, so make the mode explicit instead
            # of depending on the provider default.
            request["extra_body"] = {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}
            }
        response = self._client.chat.completions.create(**request)
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


__all__ = ["SYSTEM_PROMPT", "OpenAIPolicy", "to_openai_tools"]
