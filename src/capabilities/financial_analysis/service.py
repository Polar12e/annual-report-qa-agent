"""Deterministic financial calculations on verified facts.

Bad input is reported as a ``MetricResult`` (status ``missing`` or ``invalid``)
instead of raising: the functions are meant to be exposed as agent tools, and a
tool must always return an observation the caller can reason about.
"""
from decimal import Decimal

from src.contracts.financial import FinancialFact, MetricResult

FORMULA_REVENUE_YOY = "current_revenue / previous_revenue - 1"
FORMULA_GROSS_MARGIN = "(revenue - cost_of_revenue) / revenue"
FORMULA_DEBT_TO_ASSETS = "total_liabilities / total_assets"


def query_financials(facts: list[FinancialFact], company_code: str, year: int, metric_codes: list[str]) -> list[FinancialFact]:
    """Filter facts by company, year and metric code.

    An empty ``metric_codes`` means "any metric", matching CsvFinancialStore.query.
    """
    wanted = set(metric_codes)
    return [
        fact
        for fact in facts
        if fact.company_code == company_code
        and fact.year == year
        and (not wanted or fact.metric_code in wanted)
    ]


def calculate_metrics(current: list[FinancialFact], previous: list[FinancialFact] | None = None) -> list[MetricResult]:
    """Calculate revenue growth, gross margin and debt ratio for one company-year."""
    duplicates_current = _duplicate_codes(current)
    duplicates_previous = _duplicate_codes(previous or [])
    by_code = _index(current, duplicates_current)
    results: list[MetricResult] = []

    if previous is not None:
        prior = _index(previous, duplicates_previous)
        results.append(
            _ratio(
                "revenue_yoy",
                by_code.get("revenue"),
                prior.get("revenue"),
                FORMULA_REVENUE_YOY,
                offset=Decimal(1),
                across_years=True,
                ambiguity=_ambiguity(("revenue",), current, previous),
            )
        )
    results.append(
        _difference_ratio(
            "gross_margin",
            by_code.get("revenue"),
            by_code.get("cost_of_revenue"),
            FORMULA_GROSS_MARGIN,
            ambiguity=_ambiguity(("revenue", "cost_of_revenue"), current),
        )
    )
    results.append(
        _ratio(
            "debt_to_assets",
            by_code.get("total_liabilities"),
            by_code.get("total_assets"),
            FORMULA_DEBT_TO_ASSETS,
            ambiguity=_ambiguity(("total_liabilities", "total_assets"), current),
        )
    )
    return results


def _pages(*facts: FinancialFact | None) -> tuple[int, ...]:
    """Unique source pages for present input facts."""
    return tuple(sorted({fact.pdf_page for fact in facts if fact is not None}))


def _ratio(
    code: str,
    numerator: FinancialFact | None,
    denominator: FinancialFact | None,
    formula: str,
    offset: Decimal = Decimal(0),
    across_years: bool = False,
    ambiguity: tuple[str, tuple[str, ...]] | None = None,
) -> MetricResult:
    if ambiguity is not None:
        message, record_ids = ambiguity
        return MetricResult(code, None, formula, record_ids, "ratio", "invalid", message)
    if numerator is None or denominator is None:
        present = tuple(fact.record_id for fact in (numerator, denominator) if fact)
        return MetricResult(code, None, formula, present, "ratio", "missing",
                            "required input is missing", _pages(numerator, denominator))
    error = _compatibility_error(numerator, denominator, across_years)
    if error:
        return MetricResult(code, None, formula, (numerator.record_id, denominator.record_id),
                            "ratio", "invalid", error, _pages(numerator, denominator))
    if denominator.value == 0:
        return MetricResult(code, None, formula, (numerator.record_id, denominator.record_id),
                            "ratio", "invalid", "denominator is zero", _pages(numerator, denominator))
    return MetricResult(
        code,
        numerator.value / denominator.value - offset,
        formula,
        (numerator.record_id, denominator.record_id),
        "ratio",
        "ok",
        input_pages=_pages(numerator, denominator),
    )


def _difference_ratio(
    code: str,
    revenue: FinancialFact | None,
    cost: FinancialFact | None,
    formula: str,
    ambiguity: tuple[str, tuple[str, ...]] | None = None,
) -> MetricResult:
    if ambiguity is not None:
        message, record_ids = ambiguity
        return MetricResult(code, None, formula, record_ids, "ratio", "invalid", message)
    if revenue is None or cost is None:
        present = tuple(fact.record_id for fact in (revenue, cost) if fact)
        return MetricResult(code, None, formula, present, "ratio", "missing",
                            "required input is missing", _pages(revenue, cost))
    error = _compatibility_error(revenue, cost, False)
    if error:
        return MetricResult(code, None, formula, (revenue.record_id, cost.record_id),
                            "ratio", "invalid", error, _pages(revenue, cost))
    if revenue.value == 0:
        return MetricResult(code, None, formula, (revenue.record_id, cost.record_id),
                            "ratio", "invalid", "denominator is zero", _pages(revenue, cost))
    return MetricResult(
        code,
        (revenue.value - cost.value) / revenue.value,
        formula,
        (revenue.record_id, cost.record_id),
        "ratio",
        "ok",
        input_pages=_pages(revenue, cost),
    )


def _compatibility_error(left: FinancialFact, right: FinancialFact, across_years: bool) -> str | None:
    fields = ("company_code", "currency", "original_unit", "statement_scope", "report_id")
    if any(getattr(left, field) != getattr(right, field) for field in fields):
        return "input basis differs; reconcile company, currency, unit, scope and report first"
    expected_year = right.year + (1 if across_years else 0)
    if left.year != expected_year:
        return "input periods are incompatible"
    return None


def _duplicate_codes(facts: list[FinancialFact]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fact in facts:
        if fact.metric_code in seen:
            duplicates.add(fact.metric_code)
        seen.add(fact.metric_code)
    return duplicates


def _index(facts: list[FinancialFact], duplicated: set[str]) -> dict[str, FinancialFact]:
    return {fact.metric_code: fact for fact in facts if fact.metric_code not in duplicated}


def _ambiguity(codes: tuple[str, ...], *fact_groups: list[FinancialFact]) -> tuple[str, tuple[str, ...]] | None:
    hits: list[str] = []
    record_ids: list[str] = []
    for code in codes:
        duplicated = False
        for group in fact_groups:
            matches = [fact for fact in group if fact.metric_code == code]
            if len(matches) > 1:
                duplicated = True
                record_ids.extend(fact.record_id for fact in matches)
        if duplicated:
            hits.append(code)
    if not hits:
        return None
    return f"ambiguous inputs: multiple facts for {', '.join(hits)}", tuple(record_ids)
