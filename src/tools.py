"""Agent-facing tool layer.

Every tool is described with a JSON Schema so that a model can decide when and
how to call it.  ``FinancialToolBox.call`` always returns a dictionary and never
raises: a model must always receive a structured observation it can reason
about, even when the arguments are wrong or the data file is broken.
"""
from dataclasses import asdict
from pathlib import Path

from src.capabilities.financial_analysis import calculate_metrics, query_financials
from src.contracts.financial import METRIC_CODES
from src.data import CsvFinancialStore
from src.data.csv_store import DataValidationError
from src.policy import DataPolicy
from src.retrieval import ChunkedIndex, load_index as load_retrieval_index

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "financial_facts.csv"

TOOL_SPECS: tuple[dict, ...] = (
    {
        "name": "list_available_data",
        "description": (
            "列出已核验的财务事实范围：公司、年份、六个结构化指标和事实条数。"
            "注意：它只代表 CSV 中已核验的指标，不代表年报全文；"
            "存货、研发费用、销售费用、员工人数、分红等未入库科目请继续用 search_report/read_page 查原文。"
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_financials",
        "description": "查询某公司某年的已核验财务事实，返回数值、原始单位、来源报告和 PDF 页码。",
        "parameters": {
            "type": "object",
            "properties": {
                "company_code": {"type": "string", "description": "股票代码，例如 000333.SZ"},
                "year": {"type": "integer", "description": "财年，例如 2024"},
                "metric_codes": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(METRIC_CODES)},
                    "description": "可选，省略表示全部指标",
                },
            },
            "required": ["company_code", "year"],
        },
    },
    {
        "name": "calculate_metrics",
        "description": "计算营收同比、毛利率和资产负债率，返回数值、公式、状态和输入记录 ID。",
        "parameters": {
            "type": "object",
            "properties": {
                "company_code": {"type": "string", "description": "股票代码，例如 000333.SZ"},
                "year": {"type": "integer", "description": "财年，例如 2024"},
            },
            "required": ["company_code", "year"],
        },
    },
    {
        "name": "search_report",
        "description": (
            "在 452 页年报全文中按关键词检索，返回页码、相关度和上下文。"
            "可查任意科目/附注/口径，包括未录入 CSV 的存货、研发费用、销售费用、员工人数、分红等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "要搜索的关键词或问题，例如 营业成本 重述"},
                "max_results": {"type": "integer", "description": "最多返回几条，默认 5"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "read_page",
        "description": "读取年报某一页的完整文本（用于核对检索到的上下文）。",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "PDF 页码，从 1 开始"},
            },
            "required": ["page"],
        },
    },
)


def _error(tool: str, code: str, message: str) -> dict:
    return {"ok": False, "tool": tool, "error": {"code": code, "message": message}}


class FinancialToolBox:
    """Executes tool calls against a fixed CSV data set."""

    def __init__(self, data_path: str | Path | None = None, report_dir: str | Path | None = None,
                 index: ChunkedIndex | None = None, policy: DataPolicy | None = None,
                 retriever: str = "expanded"):
        self.store = CsvFinancialStore(data_path or DEFAULT_DATA)
        self.report_dir = Path(report_dir) if report_dir is not None else None
        self._index = index
        self.policy = policy or DataPolicy()
        self.retriever = retriever

    @property
    def index(self) -> ChunkedIndex:
        if self._index is None:
            if self.report_dir is not None:
                if not self.report_dir.exists():
                    raise FileNotFoundError(f"report directory does not exist: {self.report_dir}")
                self._index = load_retrieval_index(report_dir=self.report_dir, kind=self.retriever)
            else:
                self._index = load_retrieval_index(kind=self.retriever)
        return self._index

    @staticmethod
    def specs() -> tuple[dict, ...]:
        return TOOL_SPECS

    def call(self, name: str, arguments: dict | None = None) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return _error(name, "UNKNOWN_TOOL", f"unknown tool: {name}")
        try:
            result = handler(**(arguments or {}))
        except DataValidationError as exc:
            return _error(name, "DATA_ERROR", str(exc))
        except (TypeError, ValueError) as exc:
            return _error(name, "INVALID_ARGUMENTS", str(exc))
        except OSError as exc:
            return _error(name, "DATA_ERROR", str(exc))
        except Exception as exc:  # tool boundary: never let a model see a traceback
            return _error(name, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
        return {"ok": True, "tool": name, "result": result}

    def _tool_list_available_data(self) -> dict:
        facts = self.store.all()
        return {
            "companies": sorted({fact.company_code for fact in facts}),
            "years": sorted({fact.year for fact in facts}),
            "metrics": sorted({fact.metric_code for fact in facts}),
            "fact_count": len(facts),
            "note": (
                "以上仅为已核验的 CSV 事实；存货、研发费用、销售费用、员工人数、分红等"
                "未入库科目请用 search_report 检索年报全文，不要仅凭本工具回答“没有数据”。"
            ),
        }

    def _tool_query_financials(self, company_code: str, year: int, metric_codes: list[str] | None = None) -> dict:
        company_code = self._string(company_code, "company_code")
        year = self._integer(year, "year")
        codes = list(metric_codes or METRIC_CODES)
        unknown = [code for code in codes if code not in METRIC_CODES]
        if unknown:
            raise ValueError(f"unknown metric_codes: {', '.join(map(str, unknown))}")
        facts = query_financials(self.store.all(), company_code, year, codes)
        return {"company_code": company_code, "year": year, "count": len(facts), "facts": [asdict(fact) for fact in facts]}

    def _tool_calculate_metrics(self, company_code: str, year: int) -> dict:
        company_code = self._string(company_code, "company_code")
        year = self._integer(year, "year")
        current = self.store.query(company_code, year)
        previous = self.store.query(company_code, year - 1)
        results = calculate_metrics(current, previous)
        return {"company_code": company_code, "year": year, "count": len(results), "results": [asdict(result) for result in results]}

    def _tool_search_report(self, keyword: str, max_results: int = 5) -> dict:
        keyword = self._string(keyword, "keyword")
        max_results = self._integer(max_results, "max_results")
        if max_results < 1:
            raise ValueError("max_results must be a positive integer")
        hits = self.index.search(keyword, top_k=min(max_results, self.policy.max_snippets))
        self.policy.audit({"tool": "search_report", "pages": [hit.page for hit in hits]})
        return {
            "keyword": keyword,
            "count": len(hits),
            "matches": [
                {
                    "pdf_page": hit.page,
                    "chunk": hit.chunk,
                    "score": round(hit.score, 4),
                    "text": self.policy.snippet(hit.text),
                }
                for hit in hits
            ],
        }

    def _tool_read_page(self, page: int) -> dict:
        page = self._integer(page, "page")
        text = self.index.page_text(page)
        if not text.strip():
            raise ValueError(f"page {page} is not available in the corpus")
        return {"pdf_page": page, "text": text[:6000]}

    @staticmethod
    def _string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value
