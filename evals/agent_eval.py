"""End-to-end agent evaluation against ``evals/questions.json``.

This is deliberately separate from ``evals/run_eval.py``:

* ``run_eval`` measures the retrieval index directly (no LLM, no token cost);
* this script calls ``agent.run`` with ``OpenAIPolicy`` and checks the *final
  answer*, including tool traces and citations.

It is a dry run by default.  Pass ``--live`` only when you are willing to spend
tokens; the script never calls the model without that flag.

Examples::

    python -m evals.agent_eval
    python -m evals.agent_eval --ids q01,q03,q09,q13,h04,pa01 --live --allow-external
    python -m evals.agent_eval --split holdout --live --allow-external
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import run
from src.llm_config import LLMSettings
from src.llm_policy import OpenAIPolicy
from src.tools import FinancialToolBox

QUESTIONS = ROOT / "evals" / "questions.json"
DEFAULT_IDS = "q01,q03,q09,q13,h04,pa01"
PAGE_PATTERN = re.compile(r"第\s*(\d+)\s*页")


def select_items(items: list[dict], ids: str | None, split: str | None,
                 limit: int | None) -> list[dict]:
    selected = items
    if split:
        selected = [item for item in selected if item.get("split", "dev") == split]
    if ids:
        wanted = {part.strip() for part in ids.split(",") if part.strip()}
        selected = [item for item in selected if item["id"] in wanted]
        missing = wanted - {item["id"] for item in selected}
        if missing:
            raise ValueError(f"unknown question ids: {', '.join(sorted(missing))}")
    return selected[:limit] if limit else selected


def cited_pages(answer: str) -> list[int]:
    return sorted({int(match) for match in PAGE_PATTERN.findall(answer)})


def normalize_numbers(text: str) -> str:
    return text.replace(",", "").replace("，", "")


def check_numeric(item: dict, answer: str) -> tuple[bool | None, str]:
    """Check whether the answer repeats the verified value.

    ``None`` means the question has no numeric check.  The check is intentionally
    conservative: it looks for the number the tool returns (or its rounded
    percentage form), not for a value invented by the model.
    """
    check = item.get("check")
    if not check:
        return None, "no numeric check"
    try:
        expected = Decimal(check["expected"])
    except (InvalidOperation, KeyError, TypeError):
        return False, "invalid check definition"

    normalized = normalize_numbers(answer)
    if check["kind"] == "fact":
        candidates = {str(expected), f"{int(expected):,}"}
        if any(candidate in answer for candidate in candidates):
            return True, "expected fact present"
        # Accept an answer that repeats the verified value without commas.
        if str(expected) in normalized:
            return True, "expected fact present"
        return False, f"expected fact {expected} not found"

    if check["kind"] == "metric":
        percent = expected * 100
        candidates = {
            f"{percent:.2f}%",
            f"{percent:.2f}",
            f"{expected:.4f}",
            f"{expected:.6f}",
        }
        if any(candidate in answer for candidate in candidates):
            return True, "expected metric present"
        return False, f"expected metric {expected} not found"

    return False, f"unknown check kind: {check['kind']}"


def check_pages(item: dict, answer: str) -> tuple[bool | None, str]:
    expected = item.get("expected_pages")
    if not expected:
        return None, "no page check"
    got = cited_pages(answer)
    if not got:
        return False, "answer has no page citation"
    if set(got) & set(expected):
        return True, f"cites {sorted(set(got) & set(expected))}"
    return False, f"cites {got}, expected one of {expected}"


def tool_errors(steps: list[dict]) -> list[str]:
    errors = []
    for step in steps:
        observation = step.get("observation") or {}
        if not observation.get("ok", False):
            error = observation.get("error") or {}
            errors.append(f"{step.get('tool')}: {error.get('code', 'ERROR')}: {error.get('message', '')}")
    return errors


def evaluate_item(item: dict, policy, toolbox, max_steps: int) -> dict:
    result = run(item["question"], policy=policy, toolbox=toolbox, max_steps=max_steps)
    steps = result.get("steps", [])
    answer = str(result.get("answer", ""))
    numeric_ok, numeric_detail = check_numeric(item, answer)
    page_ok, page_detail = check_pages(item, answer)
    errors = tool_errors(steps)
    stopped = result.get("stopped", "?")
    passed = (
        stopped == "final"
        and not errors
        and numeric_ok is not False
        and page_ok is not False
    )
    return {
        "id": item["id"],
        "split": item.get("split", "dev"),
        "question": item["question"],
        "tools": [step.get("tool") for step in steps],
        "stopped": stopped,
        "answer": answer,
        "numeric_ok": numeric_ok,
        "numeric_detail": numeric_detail,
        "page_ok": page_ok,
        "page_detail": page_detail,
        "tool_errors": errors,
        "passed": passed,
    }


def print_result(result: dict, verbose: bool) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    tools = " → ".join(result["tools"]) or "(none)"
    print(
        f"{result['id']:4} {result['split']:12} {status:4} "
        f"stopped={result['stopped']:9} tools={tools}"
    )
    print(f"     numeric: {result['numeric_detail']}")
    print(f"     pages:   {result['page_detail']}")
    if result["tool_errors"]:
        for error in result["tool_errors"]:
            print(f"     tool error: {error}")
    if verbose:
        print("     answer:")
        for line in result["answer"].splitlines():
            print(f"       {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 端到端评测（默认 dry-run）")
    parser.add_argument("--ids", default=DEFAULT_IDS,
                        help=f"逗号分隔的题目 id；默认 {DEFAULT_IDS}")
    parser.add_argument("--all", action="store_true",
                        help="忽略 --ids，跑全部题目（可与 --split 组合）")
    parser.add_argument("--split", choices=("dev", "holdout", "para", "para_holdout"),
                        help="只选某个 split")
    parser.add_argument("--limit", type=int, help="最多跑几题")
    parser.add_argument("--live", action="store_true",
                        help="实际调用 LLM；不加时只打印将运行的题目")
    parser.add_argument("--allow-external", action="store_true", default=None,
                        help="允许公网模型端点")
    parser.add_argument("--provider", help="覆盖 FIN_AGENT_PROVIDER")
    parser.add_argument("--model", help="覆盖 FIN_AGENT_MODEL")
    parser.add_argument("--base-url", help="覆盖 FIN_AGENT_BASE_URL")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--json-out", type=Path, help="把结果写到 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="打印完整答案")
    args = parser.parse_args(argv)

    try:
        questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
        ids = None if args.all else args.ids
        items = select_items(questions, ids, args.split, args.limit)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"selection: FAILED\n{type(exc).__name__}: {exc}")
        return 2

    print(f"questions: {len(items)}")
    for item in items:
        print(f"  {item['id']:4} [{item.get('split', 'dev')}] {item['question']}")

    if not args.live:
        print("\ndry-run: no LLM call.  Add --live --allow-external to execute.")
        return 0

    try:
        settings = LLMSettings.from_env(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            allow_external=args.allow_external,
        )
        print("\nconfig:")
        print(json.dumps(settings.as_public_dict(), ensure_ascii=False, indent=2))
        policy = OpenAIPolicy.from_settings(settings)
        toolbox = FinancialToolBox()
    except (RuntimeError, ValueError) as exc:
        print(f"config: FAILED\n{type(exc).__name__}: {exc}")
        return 2

    results = []
    for position, item in enumerate(items, start=1):
        print(f"\n[{position}/{len(items)}] {item['id']} ...", flush=True)
        try:
            result = evaluate_item(item, policy, toolbox, args.max_steps)
        except Exception as exc:  # keep the run going so one bad item is visible
            result = {
                "id": item["id"],
                "split": item.get("split", "dev"),
                "question": item["question"],
                "tools": [],
                "stopped": "error",
                "answer": "",
                "numeric_ok": False,
                "numeric_detail": f"{type(exc).__name__}: {exc}",
                "page_ok": False,
                "page_detail": "not evaluated",
                "tool_errors": [f"{type(exc).__name__}: {exc}"],
                "passed": False,
            }
        results.append(result)
        print_result(result, args.verbose)

    passed = sum(1 for result in results if result["passed"])
    print(f"\nresult: {passed}/{len(results)} PASS")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"json: {args.json_out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
