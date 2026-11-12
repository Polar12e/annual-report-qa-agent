"""Offline evaluation for the annual-report agent.

Measures three things that do not need a language model:

1. retrieval quality: whether an expected PDF page appears in the top-k hits
2. channel comparison: the same questions answered by ``bm25``, ``dense`` and
   ``hybrid`` so a change can be attributed to a channel instead of "the system"
3. numeric correctness: whether tools return the verified values

Splits:

``dev``      questions written while tuning the lexical channel (optimistic)
``holdout``  questions written after tuning, never used for rules (honest, tiny)
``para``     vocabulary-gap questions: everyday wording, no line-item terms.
             Diagnostic only — it exists to show *where* a channel breaks, so it
             is reported but never gates the build.

Questions marked ``"split": "holdout"`` were written after the ranking rules
were tuned and are reported separately, so tuning cannot silently inflate the
headline number.

Run with ``python -m evals.run_eval``.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import CHANNELS, DENSE_WEIGHT, build_index  # noqa: E402
from src.tools import FinancialToolBox  # noqa: E402

QUESTIONS = ROOT / "evals" / "questions.json"
SPLITS = ("dev", "holdout", "para", "para_holdout")


def first_hit_rank(hits, expected_pages: list[int]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.page in expected_pages:
            return rank
    return None


def run_check(box: FinancialToolBox, check: dict) -> bool:
    company = check.get("company_code", "000333.SZ")
    if check["kind"] == "fact":
        result = box.call(
            "query_financials",
            {"company_code": company, "year": check["year"], "metric_codes": [check["metric"]]},
        )
        if not result["ok"] or result["result"]["count"] != 1:
            return False
        return Decimal(result["result"]["facts"][0]["value"]) == Decimal(check["expected"])
    if check["kind"] == "metric":
        result = box.call("calculate_metrics", {"company_code": company, "year": check["year"]})
        if not result["ok"]:
            return False
        tolerance = Decimal(check.get("tolerance", "1e-9"))
        for item in result["result"]["results"]:
            if item["metric_code"] == check["metric"]:
                if item["status"] != "ok":
                    return False
                return abs(Decimal(item["value"]) - Decimal(check["expected"])) <= tolerance
    return False


def evaluate(index, box: FinancialToolBox, items: list[dict], top_k: int) -> dict:
    hit1 = hit3 = hit5 = 0
    reciprocal = 0.0
    numeric_ok = numeric_total = 0
    rows = []
    for item in items:
        hits = index.search(item["question"], top_k=top_k)
        rank = first_hit_rank(hits, item["expected_pages"])
        if rank:
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit5 += rank <= 5
            reciprocal += 1 / rank
        if "check" in item:
            numeric_total += 1
            numeric_ok += run_check(box, item["check"])
        rows.append((item, [hit.page for hit in hits], rank))
    total = len(items) or 1
    return {
        "rows": rows,
        "hit1": hit1 / total,
        "hit3": hit3 / total,
        "hit5": hit5 / total,
        "mrr": reciprocal / total,
        "numeric": (numeric_ok, numeric_total),
    }


def print_rows(items: list[tuple], title: str) -> None:
    print(f"\n[{title}]")
    print(f"{'id':4} {'rank':>4}  {'top pages':<26} question")
    for item, pages, rank in items:
        print(f"{item['id']:4} {str(rank):>4}  {str(pages[:5]):<26} {item['question']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="年报问答 Agent 的离线评测")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--channels", default=",".join(CHANNELS),
                        help="要对比的检索通道，逗号分隔（bm25,expanded,dense,hybrid）")
    parser.add_argument("--gate-channel", default="expanded", help="用哪个通道的指标做 PASS/FAIL 判定")
    parser.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT,
                        help="混合检索里稠密通道的 RRF 权重（默认 0；离线 LSA 实测更差）")
    parser.add_argument("--min-dev-hit3", type=float, default=0.8)
    parser.add_argument("--min-holdout-hit3", type=float, default=0.5)
    args = parser.parse_args(argv)

    channels = [name.strip() for name in args.channels.split(",") if name.strip()]
    unknown = [name for name in channels if name not in CHANNELS]
    if unknown:
        parser.error(f"unknown channels: {', '.join(unknown)}")
    if args.gate_channel not in channels:
        parser.error("--gate-channel must be one of --channels")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    box = FinancialToolBox()

    indexes = {}
    for name in channels:
        indexes[name] = build_index(name, dense_weight=args.dense_weight)
    reference = indexes[channels[0]]
    print(f"corpus: {len(reference.pages)} pages, {len(reference.chunks)} chunks")

    results: dict[tuple[str, str], dict] = {}
    for channel, index in indexes.items():
        for split in SPLITS:
            items = [item for item in questions if item.get("split", "dev") == split]
            if not items:
                continue
            results[(channel, split)] = evaluate(index, box, items, args.top_k)

    print_rows(results[(args.gate_channel, "dev")]["rows"], f"dev · {args.gate_channel}")
    print_rows(results[(args.gate_channel, "holdout")]["rows"], f"holdout · {args.gate_channel}")

    for diagnostic in ("para", "para_holdout"):
        if (args.gate_channel, diagnostic) not in results:
            continue
        print(f"\n{diagnostic} 明细（rank / 通道；诊断集，不进判定）")
        print(f"{'id':4} " + "".join(f"{name:>9}" for name in channels) + "  question")
        for position, (item, _, _) in enumerate(results[(args.gate_channel, diagnostic)]["rows"]):
            cells = "".join(
                f"{str(results[(channel, diagnostic)]['rows'][position][2]):>9}" for channel in channels
            )
            print(f"{item['id']:4} {cells}  {item['question']}")

    print(f"\n{'split':14}" + "".join(f"{name:>27}" for name in channels))
    for split in SPLITS:
        if (args.gate_channel, split) not in results:
            continue
        cells = ""
        for channel in channels:
            metric = results[(channel, split)]
            cells += f"  h1={metric['hit1']:.2f} h3={metric['hit3']:.2f} mrr={metric['mrr']:.2f}"
        print(f"{split:14}{cells}")

    gate = results[(args.gate_channel, "dev")]
    gate_holdout = results[(args.gate_channel, "holdout")]
    numeric_ok = gate["numeric"][0] + gate_holdout["numeric"][0]
    numeric_total = gate["numeric"][1] + gate_holdout["numeric"][1]
    print(f"\nnumeric（{args.gate_channel}）  {numeric_ok}/{numeric_total}")
    for diagnostic in ("para", "para_holdout"):
        metric = results.get((args.gate_channel, diagnostic))
        if metric:
            solved = sum(1 for _, _, rank in metric["rows"] if rank)
            print(f"{diagnostic}（同义改写）  {args.gate_channel} 命中 {solved}/{len(metric['rows'])} 前 {args.top_k} 页")

    ok = (
        gate["hit3"] >= args.min_dev_hit3
        and gate_holdout["hit3"] >= args.min_holdout_hit3
        and numeric_ok == numeric_total
    )
    print("result:  PASS" if ok else "result:  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
