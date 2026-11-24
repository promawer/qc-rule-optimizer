"""正则预算、局部修改和确定性简化的单元测试。"""
from __future__ import annotations

import unittest

from app.optimizer import normalize_proposal
from app.evaluator import EvaluationError
from app.regex_policy import (
    MAX_REGEX_LENGTH,
    RegexConstraintError,
    constrain_regex,
    simplify_regex,
)


class RegexPolicyTest(unittest.TestCase):
    def test_deduplicates_top_level_and_group_alternatives(self):
        self.assertEqual(simplify_regex("a|a|b"), "a|b")
        self.assertEqual(simplify_regex("(?:a|b|a)"), "a|b")
        self.assertEqual(simplify_regex("(?:工号|工号|代表)"), "工号|代表")

    def test_does_not_touch_char_class_or_escaped_pipe(self):
        self.assertEqual(simplify_regex("[a|b]"), "[a|b]")
        self.assertEqual(simplify_regex(r"a\|b|a\|b"), r"a\|b")

    def test_rejects_regex_over_300_chars(self):
        with self.assertRaises(RegexConstraintError) as ctx:
            constrain_regex("a" * (MAX_REGEX_LENGTH + 1))
        self.assertIn("300", str(ctx.exception))

    def test_rejects_bare_role_word_branch(self):
        with self.assertRaises(RegexConstraintError) as ctx:
            constrain_regex("坐席|客服")
        self.assertIn("角色词", str(ctx.exception))

    def test_rejects_complete_rewrite(self):
        with self.assertRaises(RegexConstraintError) as ctx:
            constrain_regex("xyz", "abc")
        self.assertIn("局部修改", str(ctx.exception))

    def test_allows_small_local_addition(self):
        self.assertEqual(constrain_regex("abc|abd", "abc"), "abc|abd")

    def test_rejects_too_many_new_branches(self):
        with self.assertRaises(RegexConstraintError) as ctx:
            constrain_regex("a|b|c|d|e", "a|b")
        self.assertIn("分支", str(ctx.exception))

    def test_proposal_rejects_long_candidate(self):
        with self.assertRaises(EvaluationError):
            normalize_proposal(
                {"regex": "a" * 301, "semantic": "语义"},
                {"regex": "a", "semantic": "语义"},
                "提示词",
            )


if __name__ == "__main__":
    unittest.main()
