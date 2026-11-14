from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, get_args

MetricCode = Literal[
    "revenue", "cost_of_revenue", "net_income", "operating_cash_flow",
    "total_assets", "total_liabilities"
]

# Runtime counterpart of MetricCode: dataclasses and Literal do not validate
# values at runtime, so the data layer and the CLI need a real sequence.
METRIC_CODES: tuple[str, ...] = get_args(MetricCode)


@dataclass(frozen=True)
class FinancialFact:
    record_id: str
    company_code: str
    year: int
    metric_code: MetricCode
    value: Decimal
    currency: str
    original_unit: str
    statement_scope: str
    report_id: str
    pdf_page: int
    adjusted: bool


@dataclass(frozen=True)
class MetricResult:
    metric_code: str
    value: Decimal | None
    formula: str
    input_record_ids: tuple[str, ...]
    unit: str
    status: Literal["ok", "missing", "invalid"]
    error: str | None = None
    # Pages of the input facts, so a metric answer can cite its sources without
    # a second query_financials call.  Kept at the end and defaulted so existing
    # positional construction stays compatible.
    input_pages: tuple[int, ...] = ()
