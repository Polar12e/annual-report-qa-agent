import csv
from decimal import Decimal
from pathlib import Path

from src.contracts.financial import METRIC_CODES, FinancialFact

class DataValidationError(ValueError):
    """Raised when a financial facts file violates its contract."""

class CsvFinancialStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._facts: list[FinancialFact] | None = None

    def all(self) -> list[FinancialFact]:
        if self._facts is not None:
            return self._facts
        if not self.path.exists():
            raise DataValidationError(f"data file does not exist: {self.path}")
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        facts = [self._row(row, index + 2) for index, row in enumerate(rows)]
        ids = [fact.record_id for fact in facts]
        if len(ids) != len(set(ids)):
            raise DataValidationError("record_id must be unique")
        self._facts = facts
        return facts

    def query(self, company_code: str, year: int, metric_codes: list[str] | None = None) -> list[FinancialFact]:
        wanted = set(metric_codes or [])
        return [fact for fact in self.all() if fact.company_code == company_code and fact.year == year and (not wanted or fact.metric_code in wanted)]

    @staticmethod
    def _row(row: dict[str, str], line: int) -> FinancialFact:
        required = ("record_id", "company_code", "year", "metric_code", "value", "currency", "original_unit", "statement_scope", "report_id", "pdf_page", "adjusted")
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise DataValidationError(f"line {line}: missing fields: {', '.join(missing)}")
        try:
            metric_code = row["metric_code"]
            if metric_code not in METRIC_CODES:
                raise ValueError(f"unknown metric_code {metric_code}")
            year = int(row["year"])
            pdf_page = int(row["pdf_page"])
            if year < 1900 or pdf_page < 1:
                raise ValueError("year or pdf_page is out of range")
            value = Decimal(row["value"])
            if not value.is_finite():
                raise ValueError("value must be finite")
            adjusted = row["adjusted"].lower()
            if adjusted not in {"true", "false"}:
                raise ValueError("adjusted must be true or false")
        except (ValueError, ArithmeticError) as exc:
            raise DataValidationError(f"line {line}: {exc}") from exc
        return FinancialFact(record_id=row["record_id"], company_code=row["company_code"], year=year, metric_code=metric_code, value=value, currency=row["currency"], original_unit=row["original_unit"], statement_scope=row["statement_scope"], report_id=row["report_id"], pdf_page=pdf_page, adjusted=adjusted == "true")
