import shutil
import unittest
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from src.capabilities.financial_analysis import calculate_metrics, query_financials
from src.contracts.financial import FinancialFact
from src.data import CsvFinancialStore
from src.data.csv_store import DataValidationError

DATA = Path(__file__).resolve().parents[1] / "data" / "financial_facts.csv"
HEADER = (
    "record_id,company_code,year,metric_code,value,currency,original_unit,"
    "statement_scope,report_id,pdf_page,adjusted\n"
)


def fact(record_id, metric_code, value, year=2024, currency="CNY", unit="千元", scope="consolidated", report="MIDEA-2024-CAS"):
    return FinancialFact(
        record_id, "000333.SZ", year, metric_code, Decimal(str(value)),
        currency, unit, scope, report, 10, False,
    )


@contextmanager
def temp_csv(*rows):
    # A project-local temp directory, because tempfile.TemporaryDirectory
    # creates a 0o700 DACL that some sandboxed Windows processes cannot use.
    directory = Path(__file__).resolve().parents[1] / "_test_tmp"
    directory.mkdir(exist_ok=True)
    path = directory / "facts.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    try:
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class CalculationTests(unittest.TestCase):
    def test_three_metrics_and_traceability(self):
        current = [
            fact("r24", "revenue", 120),
            fact("c24", "cost_of_revenue", 90),
            fact("a24", "total_assets", 200),
            fact("l24", "total_liabilities", 100),
        ]
        results = calculate_metrics(current, [fact("r23", "revenue", 100, 2023)])
        self.assertEqual([result.status for result in results], ["ok", "ok", "ok"])
        self.assertEqual(results[0].value, Decimal("0.2"))
        self.assertEqual(results[1].value, Decimal("0.25"))
        self.assertEqual(results[2].input_record_ids, ("l24", "a24"))

    def test_previous_none_omits_revenue_yoy(self):
        results = calculate_metrics([fact("r", "revenue", 1)], None)
        self.assertEqual([result.metric_code for result in results], ["gross_margin", "debt_to_assets"])

    def test_zero_denominator_is_explicit_error(self):
        result = calculate_metrics([fact("r", "revenue", 0), fact("c", "cost_of_revenue", 1)])[0]
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.error, "denominator is zero")

    def test_incompatible_currency_is_rejected(self):
        result = calculate_metrics([fact("r", "revenue", 100), replace(fact("c", "cost_of_revenue", 50), currency="USD")])[0]
        self.assertEqual(result.status, "invalid")
        self.assertIn("input basis differs", result.error)

    def test_missing_is_not_zero(self):
        self.assertEqual(calculate_metrics([])[0].status, "missing")

    def test_duplicate_revenue_returns_invalid_instead_of_raising(self):
        results = calculate_metrics([fact("one", "revenue", 100), fact("two", "revenue", 200)])
        self.assertEqual(results[0].status, "invalid")
        self.assertIn("ambiguous", results[0].error)
        self.assertEqual(set(results[0].input_record_ids), {"one", "two"})

    def test_duplicate_total_assets_only_breaks_debt_ratio(self):
        current = [
            fact("r", "revenue", 100),
            fact("c", "cost_of_revenue", 50),
            fact("a1", "total_assets", 200),
            fact("a2", "total_assets", 300),
            fact("l", "total_liabilities", 100),
        ]
        by_code = {result.metric_code: result for result in calculate_metrics(current)}
        self.assertEqual(by_code["gross_margin"].status, "ok")
        self.assertEqual(by_code["debt_to_assets"].status, "invalid")
        self.assertIn("total_assets", by_code["debt_to_assets"].error)

    def test_metric_results_include_source_pages(self):
        current = [
            replace(fact("r24", "revenue", 120), pdf_page=159),
            replace(fact("c24", "cost_of_revenue", 90), pdf_page=159),
            replace(fact("a24", "total_assets", 200), pdf_page=157),
            replace(fact("l24", "total_liabilities", 100), pdf_page=158),
        ]
        previous = [replace(fact("r23", "revenue", 100, 2023), pdf_page=159)]
        by_code = {result.metric_code: result for result in calculate_metrics(current, previous)}
        self.assertEqual(by_code["revenue_yoy"].input_pages, (159,))
        self.assertEqual(by_code["gross_margin"].input_pages, (159,))
        self.assertEqual(by_code["debt_to_assets"].input_pages, (157, 158))

    def test_query_financials(self):
        facts = [fact("r", "revenue", 1), fact("c", "cost_of_revenue", 2)]
        self.assertEqual(len(query_financials(facts, "000333.SZ", 2024, [])), 2)
        self.assertEqual(len(query_financials(facts, "000333.SZ", 2024, ["revenue"])), 1)


class DataStoreTests(unittest.TestCase):
    def test_verified_dataset(self):
        store = CsvFinancialStore(DATA)
        self.assertEqual(len(store.all()), 12)
        results = calculate_metrics(store.query("000333.SZ", 2024), store.query("000333.SZ", 2023))
        self.assertTrue(all(result.status == "ok" for result in results))
        self.assertEqual([round(result.value * 100, 2) for result in results], [Decimal("9.44"), Decimal("26.42"), Decimal("62.33")])

    def test_rejects_duplicate_ids(self):
        rows = (
            "x,000333.SZ,2024,revenue,1,CNY,千元,consolidated,r,1,false",
            "x,000333.SZ,2023,revenue,1,CNY,千元,consolidated,r,1,false",
        )
        with temp_csv(*rows) as path:
            with self.assertRaisesRegex(DataValidationError, "unique"):
                CsvFinancialStore(path).all()

    def test_rejects_missing_field(self):
        with temp_csv("x,000333.SZ,2024,revenue,,CNY,千元,consolidated,r,1,false") as path:
            with self.assertRaisesRegex(DataValidationError, "missing fields: value"):
                CsvFinancialStore(path).all()

    def test_rejects_non_finite_value(self):
        with temp_csv("x,000333.SZ,2024,revenue,NaN,CNY,千元,consolidated,r,1,false") as path:
            with self.assertRaisesRegex(DataValidationError, "finite"):
                CsvFinancialStore(path).all()

    def test_rejects_unknown_metric(self):
        with temp_csv("x,000333.SZ,2024,roe,1,CNY,千元,consolidated,r,1,false") as path:
            with self.assertRaisesRegex(DataValidationError, "unknown metric_code"):
                CsvFinancialStore(path).all()

    def test_missing_file(self):
        with self.assertRaisesRegex(DataValidationError, "does not exist"):
            CsvFinancialStore(DATA.parent / "does-not-exist.csv").all()


if __name__ == "__main__":
    unittest.main()
