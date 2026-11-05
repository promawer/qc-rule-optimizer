import asyncio
import json
import unittest
from pathlib import Path

from app.evaluator import EvaluationError, evaluate_rule
from app.importer import build_calls, calls_to_csv, cases_for_rule
from app.models import NORMAL, VIOLATION, normalize_label
from app.optimizer import MockProposer, compute_gate, normalize_proposal, optimize
from app.prompts import DEFAULT_SEMANTIC_PROMPT, render_prompt, validate_prompt
from app.seed import build_state
from app.splits import TEST, TRAIN, VALIDATION, build_splits, split_cases
from app.store import migrate

ROOT = Path(__file__).resolve().parent.parent


def make_cases():
    return [
        {"id": "1", "callId": "C1", "transcript": "坐席：请把验证码告诉我", "autoResult": VIOLATION, "humanResult": VIOLATION},
        {"id": "2", "callId": "C2", "transcript": "坐席：您好，请问办理什么业务", "autoResult": NORMAL, "humanResult": NORMAL},
    ]


class LabelTest(unittest.TestCase):
    def test_normalize_labels(self):
        self.assertEqual(normalize_label("违规"), VIOLATION)
        self.assertEqual(normalize_label("正常"), NORMAL)
        self.assertEqual(normalize_label("无法判断"), "UNKNOWN")


class ImporterTest(unittest.TestCase):
    def test_four_column_csv(self):
        content = (
            "通话ID,通话文本,自动违规规则,人工违规规则\n"
            "C1,坐席：请把验证码告诉我,R003,R003\n"
            "C2,坐席：请不要泄露验证码,R003,\n"
            "C3,坐席：去你单位找你领导,R004,R001;R004\n"
        ).encode("utf-8-sig")
        rules = [
            {"code": "R001", "name": "辱骂客户"},
            {"code": "R003", "name": "索要短信验证码"},
            {"code": "R004", "name": "威胁催收"},
        ]
        calls, report = build_calls("demo.csv", content, rules)
        self.assertEqual(len(calls), 3)
        self.assertEqual(report["imported"], 3)
        self.assertEqual(report["unknownHumanRules"], [])
        self.assertEqual(calls[0]["autoRules"], ["R003"])
        self.assertEqual(calls[1]["humanRules"], [])
        self.assertEqual(calls[2]["humanRules"], ["R001", "R004"])

    def test_rule_name_resolution_and_unknown_code(self):
        content = (
            "通话ID,通话文本,自动违规规则,人工违规规则\n"
            "C1,坐席：保本保息,承诺保本收益,承诺保本收益\n"
            "C2,坐席：你好,,R999\n"
        ).encode("utf-8")
        rules = [{"code": "R002", "name": "承诺保本收益"}]
        calls, report = build_calls("demo.csv", content, rules)
        self.assertEqual(calls[0]["autoRules"], ["R002"])
        self.assertEqual(calls[0]["humanRules"], ["R002"])
        self.assertEqual(report["unknownHumanRules"], ["R999"])

    def test_split_multiple_codes(self):
        content = (
            "通话ID,通话文本,自动违规规则,人工违规规则\n"
            "C1,坐席：测试,R001、R002;R003,R001；R003\n"
        ).encode("utf-8")
        rules = [{"code": f"R00{i}", "name": f"规则{i}"} for i in range(1, 4)]
        calls, _ = build_calls("demo.csv", content, rules)
        self.assertEqual(calls[0]["autoRules"], ["R001", "R002", "R003"])
        self.assertEqual(calls[0]["humanRules"], ["R001", "R003"])

    def test_missing_columns_raises(self):
        with self.assertRaises(ValueError):
            build_calls("bad.csv", "a,b\n1,2\n".encode(), [])

    def test_skip_empty_transcript(self):
        content = (
            "通话ID,通话文本,自动违规规则,人工违规规则\n"
            "C1,,R001,R001\n"
            "C2,坐席：正常通话,,\n"
        ).encode("utf-8")
        calls, report = build_calls("demo.csv", content, [{"code": "R001", "name": "规则"}])
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["skipped"], 1)


class DeriveTest(unittest.TestCase):
    def test_cases_for_rule(self):
        state = build_state()
        rule = next(item for item in state["rules"] if item["code"] == "R003")
        cases = cases_for_rule(state["calls"], rule, state["splits"])
        self.assertEqual(len(cases), 10)
        labels = {case["humanResult"] for case in cases}
        self.assertEqual(labels, {VIOLATION, NORMAL})
        positives = [case["callId"] for case in cases if case["humanResult"] == VIOLATION]
        self.assertEqual(positives, ["C002", "C006"])
        auto_positives = [case["callId"] for case in cases if case["autoResult"] == VIOLATION]
        self.assertEqual(auto_positives, ["C002", "C005"])

    def test_sample_file_roundtrip(self):
        state = build_state()
        sample = ROOT / "samples" / "import-calls-10.csv"
        self.assertTrue(sample.exists())
        calls, report = build_calls(sample.name, sample.read_bytes(), state["rules"])
        self.assertEqual(len(calls), 10)
        self.assertEqual(report["imported"], 10)
        self.assertEqual(report["unknownAutoRules"], [])
        self.assertEqual(report["unknownHumanRules"], [])
        self.assertEqual(calls[8]["humanRules"], ["R001", "R004"])

    def test_export_contains_four_columns(self):
        state = build_state()
        text = calls_to_csv(state["calls"])
        self.assertTrue(text.startswith("通话ID,通话文本,自动违规规则,人工违规规则"))
        self.assertIn("R001、R004".replace("、", ";"), text)


class MigrationTest(unittest.TestCase):
    def test_v1_state_migrates(self):
        old = {
            "version": 1,
            "rules": [
                {"id": "r1", "name": "规则一", "regex": "a", "semantic": "", "logic": "OR"},
            ],
            "datasets": {
                "r1": [
                    {"callId": "C1", "transcript": "t1", "autoResult": VIOLATION, "humanResult": VIOLATION},
                    {"callId": "C2", "transcript": "t2", "autoResult": NORMAL, "humanResult": NORMAL},
                ]
            },
            "runs": [],
            "settings": {},
        }
        migrated = migrate(old)
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["rules"][0]["code"], "R001")
        self.assertEqual(len(migrated["calls"]), 2)
        self.assertEqual(migrated["calls"][0]["autoRules"], ["R001"])
        self.assertNotIn("datasets", migrated)


class EvaluatorTest(unittest.TestCase):
    def test_or_logic_short_circuits_regex_hit(self):
        calls = {"count": 0}

        async def judge(case, rule):
            calls["count"] += 1
            return True, "语义命中"

        rule = {"regex": "验证码", "semantic": "是否索要验证码", "logic": "OR"}
        result = asyncio.run(evaluate_rule(rule, make_cases(), judge))
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(calls["count"], 1)

    def test_and_logic(self):
        calls = {"count": 0}

        async def judge(case, rule):
            calls["count"] += 1
            return True, "语义命中"

        rule = {"regex": "验证码", "semantic": "是否索要验证码", "logic": "AND"}
        result = asyncio.run(evaluate_rule(rule, make_cases(), judge))
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(calls["count"], 1)

    def test_invalid_regex(self):
        async def judge(case, rule):
            return False, ""

        with self.assertRaises(EvaluationError):
            asyncio.run(evaluate_rule({"regex": "(", "semantic": "", "logic": "OR"}, make_cases(), judge))


class OptimizerTest(unittest.TestCase):
    def test_mock_reaches_target_for_code_rule(self):
        state = build_state()
        rule = next(item for item in state["rules"] if item["code"] == "R003")
        cases = cases_for_rule(state["calls"], rule, state["splits"])

        async def judge(case, rule):
            return False, "mock"

        def factory(prompt, with_reason):
            return judge

        result = asyncio.run(
            optimize(
                rule, cases, 0.9, 3, factory, MockProposer(), DEFAULT_SEMANTIC_PROMPT,
                lambda event: None, lambda: False,
            )
        )
        self.assertIn(result["status"], ("reached", "exhausted"))
        self.assertGreaterEqual(
            result["best"]["validation"]["metrics"]["accuracy"],
            result["baseline"]["validation"]["metrics"]["accuracy"],
        )
        # R003 只有 2 条正样本，测试集缺少正样本，必须判定“样本不足”，不能给出验收通过
        self.assertEqual(result["gate"]["status"], "INSUFFICIENT")

    def test_optimization_passes_gate_on_clean_split(self):
        cases = []

        def add(split, count, positives):
            for position in range(count):
                index = len(cases) + 1
                is_positive = position < positives
                cases.append({
                    "id": f"case-{index}",
                    "callId": f"C{index}",
                    "split": split,
                    "transcript": "命中文本" if is_positive else "普通文本",
                    "humanResult": VIOLATION if is_positive else NORMAL,
                    "autoResult": NORMAL,
                })

        add(TRAIN, 6, 4)
        add(VALIDATION, 3, 1)
        add(TEST, 3, 1)

        async def judge(case, rule):
            return False, "mock"

        class Proposer:
            async def __call__(self, rule, result, target, history, semantic_prompt):
                return {
                    "regex": "命中文本", "semantic": "", "logic": "OR",
                    "rationale": "补充命中表达", "expectedEffect": "覆盖正样本",
                    "semanticPrompt": semantic_prompt + "\n补充：优先检查坐席发言。",
                    "promptChanged": True, "promptRationale": "强化坐席发言优先级",
                }

        def factory(prompt, with_reason):
            return judge

        rule = {"name": "测试规则", "regex": "不会命中", "semantic": "", "logic": "OR"}
        result = asyncio.run(
            optimize(
                rule, cases, 0.9, 2, factory, Proposer(), DEFAULT_SEMANTIC_PROMPT,
                lambda event: None, lambda: False,
            )
        )
        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["gate"]["status"], "PASS")
        self.assertEqual(result["best"]["test"]["metrics"]["accuracy"], 1.0)
        self.assertTrue(result["promptChanged"])
        self.assertIn("优先检查坐席发言", result["bestPrompt"])

    def test_candidate_with_invalid_regex_is_rejected(self):
        fallback = {"regex": "验证码", "semantic": "", "logic": "OR"}
        with self.assertRaises(EvaluationError):
            normalize_proposal({"regex": "(", "semantic": "", "logic": "OR"}, fallback, DEFAULT_SEMANTIC_PROMPT)

    def test_empty_candidate_is_rejected(self):
        fallback = {"regex": "验证码", "semantic": "", "logic": "OR"}
        with self.assertRaises(EvaluationError):
            normalize_proposal({"regex": "", "semantic": ""}, fallback, DEFAULT_SEMANTIC_PROMPT)


class SplitTest(unittest.TestCase):
    def test_per_rule_splits_cover_all_calls(self):
        state = build_state()
        call_ids = {call["callId"] for call in state["calls"]}
        for rule in state["rules"]:
            mapping = state["splits"][rule["code"]]
            self.assertEqual(set(mapping.keys()), call_ids)
            for split in (TRAIN, VALIDATION, TEST):
                self.assertTrue(any(value == split for value in mapping.values()))

    def test_split_of_rule_with_enough_positives_covers_all_sets(self):
        state = build_state()
        rule = next(item for item in state["rules"] if item["code"] == "R001")
        cases = cases_for_rule(state["calls"], rule, state["splits"])
        grouped = split_cases(cases)
        for split in (TRAIN, VALIDATION, TEST):
            self.assertGreaterEqual(sum(1 for c in grouped[split] if c["humanResult"] == VIOLATION), 1)


class GateTest(unittest.TestCase):
    def _metrics(self, accuracy):
        return {"accuracy": accuracy}

    def test_gate_passes_only_when_validation_and_test_both_pass(self):
        summary = {
            VALIDATION: {"positives": 2, "negatives": 4},
            TEST: {"positives": 2, "negatives": 4},
        }
        gate = compute_gate(
            {"metrics": self._metrics(0.95)},
            {"metrics": self._metrics(0.92)},
            {"metrics": self._metrics(0.70)},
            {"metrics": self._metrics(0.72)},
            0.9,
            summary,
            True,
        )
        self.assertEqual(gate["status"], "PASS")

    def test_gate_blocks_when_test_regresses(self):
        summary = {
            VALIDATION: {"positives": 2, "negatives": 4},
            TEST: {"positives": 2, "negatives": 4},
        }
        gate = compute_gate(
            {"metrics": self._metrics(0.95)},
            {"metrics": self._metrics(0.60)},
            {"metrics": self._metrics(0.70)},
            {"metrics": self._metrics(0.72)},
            0.9,
            summary,
            True,
        )
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(any("测试集" in reason for reason in gate["reasons"]))

    def test_gate_marks_insufficient_when_test_has_no_positives(self):
        summary = {
            VALIDATION: {"positives": 2, "negatives": 4},
            TEST: {"positives": 0, "negatives": 2},
        }
        gate = compute_gate(
            {"metrics": self._metrics(0.95)},
            {"metrics": self._metrics(1.0)},
            {"metrics": self._metrics(0.70)},
            {"metrics": self._metrics(0.80)},
            0.9,
            summary,
            True,
        )
        self.assertEqual(gate["status"], "INSUFFICIENT")


class ReasonTest(unittest.TestCase):
    def test_training_forces_both_reasons(self):
        cases = [
            {"id": "1", "callId": "C1", "transcript": "坐席：验证码", "humanResult": VIOLATION, "autoResult": VIOLATION},
            {"id": "2", "callId": "C2", "transcript": "坐席：你好", "humanResult": NORMAL, "autoResult": NORMAL},
        ]
        calls = {"count": 0}

        async def judge(case, rule):
            calls["count"] += 1
            return True, "语义判断理由"

        rule = {"regex": "验证码", "semantic": "是否索要验证码", "logic": "OR"}
        result = asyncio.run(
            evaluate_rule(rule, cases, judge, force_parts=True, with_reason=True)
        )
        self.assertEqual(calls["count"], 2)  # 正则命中/未命中都调用了语义
        self.assertIn("命中", result["rows"][0]["regexReason"])
        self.assertEqual(result["rows"][0]["semanticReason"], "语义判断理由")
        self.assertIn("未命中", result["rows"][1]["regexReason"])
        self.assertEqual(result["rows"][1]["semanticReason"], "语义判断理由")

    def test_validation_mode_does_not_keep_semantic_reason(self):
        cases = [
            {"id": "1", "callId": "C1", "transcript": "坐席：你好", "humanResult": VIOLATION, "autoResult": VIOLATION},
        ]

        async def judge(case, rule):
            return True, "不应该被保存"

        rule = {"regex": "", "semantic": "是否违规", "logic": "OR"}
        result = asyncio.run(evaluate_rule(rule, cases, judge, with_reason=False))
        self.assertEqual(result["rows"][0]["semanticReason"], "")
        self.assertEqual(result["rows"][0]["regexReason"], "未配置正则")


class PromptTest(unittest.TestCase):
    def test_prompt_renders_all_rules(self):
        state = build_state()
        text = render_prompt(DEFAULT_SEMANTIC_PROMPT, state["rules"])
        self.assertIn("R001", text)
        self.assertIn("R004", text)
        self.assertNotIn("{rules}", text)

    def test_prompt_requires_placeholder(self):
        with self.assertRaises(ValueError):
            validate_prompt("没有占位符的提示词")


if __name__ == "__main__":
    unittest.main()

