"""结构化编辑操作与演示数据的基本测试。"""
from __future__ import annotations

import unittest

from app.demo_scenario import build_demo_calls
from app.edit_operations import OperationError, apply_operations


class EditOperationTest(unittest.TestCase):
    def test_add_anchor_and_semantic_exclusion(self):
        rule = {"regex": "验证码", "semantic": "坐席索要验证码", "logic": "OR"}
        result, applied = apply_operations(rule, [
            {"op": "ADD_REGEX_ANCHOR", "pattern": "动态密码", "targetCaseIds": ["D001"]},
            {"op": "ADD_SEMANTIC_EXCLUSION", "text": "提醒客户不要泄露验证码"},
        ])
        self.assertIn("动态密码", result["regex"])
        self.assertIn("排除：提醒客户不要泄露验证码", result["semantic"])
        self.assertEqual(len(applied), 2)

    def test_set_logic_and_remove_branch(self):
        rule = {"regex": "验证码|动态密码", "semantic": "索要验证码", "logic": "OR"}
        result, _ = apply_operations(rule, [
            {"op": "REMOVE_REGEX_BRANCH", "pattern": "动态密码"},
            {"op": "SET_LOGIC", "value": "AND"},
        ])
        self.assertEqual(result["regex"], "验证码")
        self.assertEqual(result["logic"], "AND")

    def test_rejects_unknown_operation_and_too_many(self):
        rule = {"regex": "验证码", "semantic": "条件"}
        with self.assertRaises(OperationError):
            apply_operations(rule, [{"op": "DELETE_RULE"}])
        with self.assertRaises(OperationError):
            apply_operations(rule, [{"op": "NO_CHANGE"}] * 4)


class DemoScenarioTest(unittest.TestCase):
    def test_demo_has_forty_calls_and_both_labels(self):
        calls = build_demo_calls()
        self.assertEqual(len(calls), 40)
        self.assertTrue(any(call["humanRules"] for call in calls))
        self.assertTrue(any(not call["humanRules"] for call in calls))
        self.assertTrue(all(call.get("annotations") for call in calls))


if __name__ == "__main__":
    unittest.main()
