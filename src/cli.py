import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.capabilities.financial_analysis import calculate_metrics
from src.contracts.financial import METRIC_CODES, MetricResult, FinancialFact
from src.data import CsvFinancialStore
from src.data.csv_store import DataValidationError

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "financial_facts.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="上市公司财务分析 V0")
    parser.add_argument("--company", required=True, help="股票代码，例如 000333.SZ")
    parser.add_argument("--year", type=int, required=True, help="财年，例如 2024")
    parser.add_argument("--metric", choices=METRIC_CODES, help="只筛选输出的事实，不影响计算")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="财务事实 CSV 路径")
    return parser


def build_output(company: str, year: int, metric: str | None, store: CsvFinancialStore) -> dict:
    current = store.query(company, year)
    previous = store.query(company, year - 1)
    calculations = calculate_metrics(current, previous)
    facts = [asdict(fact) for fact in current if metric is None or fact.metric_code == metric]
    return {
        "facts": facts,
        "calculations": [asdict(result) for result in calculations],
        "limitations": build_limitations(current, calculations),
    }


def build_limitations(current: list[FinancialFact], calculations: list[MetricResult]) -> list[str]:
    notes = []
    if not current:
        notes.append("No verified financial facts are available for this company and year.")
    for result in calculations:
        if result.status != "ok":
            notes.append(f"{result.metric_code}: {result.status} - {result.error}")
    return notes


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    store = CsvFinancialStore(args.data)
    try:
        output = build_output(args.company, args.year, args.metric, store)
    except (DataValidationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
