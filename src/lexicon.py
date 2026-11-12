"""Domain vocabulary used to rewrite questions into annual-report wording.

Shareholders ask about "仓库里压的货"; the report says "存货".  A lexical
retriever cannot bridge that gap no matter how the score is tuned, so the query
is rewritten *before* scoring.  This table is a domain prior (accounting
vocabulary plus the colloquialisms that show up in real questions), not a set of
per-question patches.

Matching happens on character bigrams, not substrings: "在技术上要投多少钱"
does not contain the string "技术投入", but it does share the bigram "技术",
which is enough to know the question is about R&D spending.

Terms are added with a lower weight than the user's own words, so a rewrite can
add recall but never outvote the original query.
"""
from __future__ import annotations

import re

CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("存货", "库存", "库存商品", "仓库", "备货", "积压", "压货", "卖不掉"),
    ("应收账款", "赊账", "赊销", "欠款", "回款", "坏账"),
    ("应付职工薪酬", "职工薪酬", "工资", "薪酬", "人工成本", "员工薪酬", "发工资"),
    ("现金分红", "分红", "派息", "股利", "利润分配", "送红股"),
    ("研发投入", "研发费用", "研发支出", "技术投入", "科研投入", "技术"),
    ("利息支出", "利息费用", "财务费用", "融资成本"),
    ("短期借款", "银行借款", "有息负债", "借款", "贷款"),
    ("质押", "抵押", "股份质押", "冻结"),
    ("商誉", "并购溢价", "收购溢价"),
    ("存货跌价准备", "跌价准备", "减值准备", "存货减值"),
    ("毛利率", "毛利", "赚钱能力", "盈利能力"),
    ("资产负债率", "偿债能力", "还债能力", "杠杆"),
    ("经营活动现金流量净额", "经营现金流", "造血能力"),
    ("汇率波动", "汇兑损益", "外汇风险", "升值", "贬值"),
    ("营业收入", "营收", "收入", "销售额"),
)

# 0.45 keeps a rewrite additive: two rewritten terms cannot outweigh the
# user's own keyword, which is what protects the exact-match queries.
EXPANSION_WEIGHT = 0.45
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _bigrams(text: str) -> set[str]:
    pairs: set[str] = set()
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            pairs.add(run)
        else:
            pairs.update(run[index:index + 2] for index in range(len(run) - 1))
    return pairs


def concepts_for(text: str) -> tuple[str, ...]:
    """Return sibling terms of every concept whose wording overlaps ``text``."""
    query_pairs = _bigrams(text)
    matched: list[str] = []
    for group in CONCEPT_GROUPS:
        if any(query_pairs & _bigrams(member) for member in group):
            matched.extend(member for member in group if member not in text)
    return tuple(dict.fromkeys(matched))


def expand_terms(query: str) -> list[tuple[str, float]]:
    """Return ``(term, weight)`` rewrites for concepts mentioned in ``query``."""
    return [(term, EXPANSION_WEIGHT) for term in concepts_for(query)]


__all__ = ["CONCEPT_GROUPS", "EXPANSION_WEIGHT", "concepts_for", "expand_terms"]
