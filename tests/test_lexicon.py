"""Concept-rewrite table: bigram matching, weights, and honest limits."""
from __future__ import annotations

import unittest

from src.lexicon import CONCEPT_GROUPS, EXPANSION_WEIGHT, concepts_for, expand_terms


class ConceptsForTest(unittest.TestCase):
    def test_colloquial_wording_maps_to_line_items(self):
        self.assertIn("存货", concepts_for("公司仓库里压了多少货"))
        self.assertIn("研发投入", concepts_for("公司一年在技术上要投多少钱"))
        self.assertIn("质押", concepts_for("大股东的股票有没有被抵押出去"))

    def test_known_limit_bare_verb_paraphrase_is_unreachable(self):
        # "客户欠的钱" shares no bigram with 应收账款 — no lexical table can
        # close that gap; it needs an embedding model, which is why the dense
        # channel exists (see docs/rag-notes.md).
        self.assertEqual(concepts_for("客户欠的钱收不回来"), ())

    def test_matching_is_on_bigrams_not_substrings(self):
        # "技术投入" never appears literally; "技术" is enough.
        self.assertNotIn("技术投入", "公司一年在技术上要投多少钱")
        self.assertIn("研发投入", concepts_for("公司一年在技术上要投多少钱"))

    def test_terms_already_present_are_not_repeated(self):
        self.assertNotIn("存货", concepts_for("存货有多少"))

    def test_unknown_vocabulary_maps_to_nothing(self):
        self.assertEqual(concepts_for("季度环比 环比增速"), ())

    def test_empty_query_maps_to_nothing(self):
        self.assertEqual(concepts_for(""), ())

    def test_no_duplicate_rewrites_across_groups(self):
        rewrites = concepts_for("存货 跌价 仓库 分红 利润")
        self.assertEqual(len(rewrites), len(set(rewrites)))

    def test_bigrams_of_single_characters_are_not_matched_blindly(self):
        # A one-character member would match almost any text; the table should
        # not contain such entries.
        for group in CONCEPT_GROUPS:
            for member in group:
                self.assertGreater(len(member), 1, f"concept member too short: {member}")


class ExpandTermsTest(unittest.TestCase):
    def test_every_rewrite_uses_the_documented_weight(self):
        for _, weight in expand_terms("仓库里压了多少货"):
            self.assertEqual(weight, EXPANSION_WEIGHT)

    def test_weight_stays_below_full_weight(self):
        self.assertLess(EXPANSION_WEIGHT, 1.0)

    def test_expansion_is_additive_not_unbounded(self):
        # A two-concept question must not return the whole table.
        rewrites = [term for term, _ in expand_terms("存货和分红")]
        table = {member for group in CONCEPT_GROUPS for member in group}
        self.assertLess(len(rewrites), len(table))


if __name__ == "__main__":
    unittest.main()
