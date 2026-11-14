"""Smoke-check the LLM configuration.

This script separates two questions:

1. *Is the configuration valid?* — no network, no key printed.
2. *Can the endpoint actually answer?* — only with ``--live``; this spends a
   small amount of tokens and never sends report text.

Usage::

    python scripts/check_llm.py
    python scripts/check_llm.py --live --allow-external

Environment variables are documented in ``src/llm_config.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_config import PROVIDERS, LLMSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 LLM 配置（默认不联网）")
    parser.add_argument("--provider", choices=PROVIDERS,
                        help="覆盖 FIN_AGENT_PROVIDER")
    parser.add_argument("--model", help="覆盖 FIN_AGENT_MODEL")
    parser.add_argument("--base-url", help="覆盖 FIN_AGENT_BASE_URL")
    parser.add_argument("--allow-external", action="store_true", default=None,
                        help="允许调用公网模型端点（默认只允许本机/内网）")
    parser.add_argument("--live", action="store_true",
                        help="实际发一个最小请求；会产生少量 token 费用")
    args = parser.parse_args(argv)

    try:
        settings = LLMSettings.from_env(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            allow_external=args.allow_external,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"config: FAILED\n{type(exc).__name__}: {exc}")
        return 2

    print("config: OK")
    print(json.dumps(settings.as_public_dict(), ensure_ascii=False, indent=2))
    if settings.base_url and not settings.is_local and not settings.allow_external:
        print("warning: 这是公网端点；live 调用会被 policy 拒绝，请加 --allow-external 或设置 FIN_AGENT_ALLOW_EXTERNAL=true。")

    if not args.live:
        print("live: skipped (pass --live to call the endpoint)")
        return 0

    try:
        from src.llm_policy import OpenAIPolicy

        policy = OpenAIPolicy.from_settings(settings)
        action = policy.decide("请只回复 ok，不要调用工具。", [])
    except Exception as exc:  # a smoke check should report, not traceback
        print(f"live: FAILED\n{type(exc).__name__}: {exc}")
        return 1

    print("live: OK")
    if action.get("type") == "final":
        print(f"reply: {str(action.get('answer', ''))[:200]}")
    else:
        print(f"action: {action.get('type')} / {action.get('tool')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
