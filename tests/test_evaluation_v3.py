"""业务约束、未知值、固定分母与错误驱动搜索的回归测试。"""
import asyncio
import unittest

from app.evaluator import evaluate_rule
from app.models import NORMAL, UNKNOWN, VIOLATION
from app.optimizer import compute_gate, optimize
from app.prompt_regression import context_fingerprint, needs_regression
from app.prompts import DEFAULT_SEMANTIC_PROMPT
from app.splits import TEST, TRAIN, VALIDATION


def case(key, positive=False, split=TRAIN, text=None):
    return {"callId": key, "id": key, "split": split, "humanResult": VIOLATION if positive else NORMAL,
            "autoResult": NORMAL, "transcript": text or ("风险" if positive else "普通")}


def dataset(n=5):
    return [case(f"{split}{positive}{i}", positive, split) for split in (TRAIN, VALIDATION, TEST)
            for positive in (True, False) for i in range(n)]


async def false_judge(case, rule):
    return False, "本地替身"


class FixedDenominatorTests(unittest.TestCase):
    def test_failed_positive_does_not_inflate_accuracy(self):
        items = [case("failed", True)] + [case(str(i)) for i in range(9)]
        async def judge(item, rule):
            if item["callId"] == "failed":
                raise ValueError("timeout")
            return False, "正常"
        result = asyncio.run(evaluate_rule({"semantic": "风险"}, items, judge))
        self.assertEqual(result["metrics"]["total"], 10)
        self.assertEqual(result["metrics"]["evaluated"], 9)
        self.assertEqual(result["metrics"]["accuracy"], .9)
        self.assertEqual(result["metrics"]["validAccuracy"], 1)
        self.assertEqual(result["metrics"]["recall"], 0)
        self.assertEqual(result["metrics"]["unresolvedPositives"], 1)
        self.assertEqual(result["coverage"], .9)
        self.assertFalse(result["complete"])

    def test_unreviewed_labels_skip_judge_and_are_reported(self):
        calls = []
        async def judge(item, rule):
            calls.append(item["callId"])
            return False, "正常"
        items = [case("reviewed"), dict(case("unknown"), humanResult=UNKNOWN)]
        result = asyncio.run(evaluate_rule({"semantic": "风险"}, items, judge))
        self.assertEqual(calls, ["reviewed"])
        self.assertEqual(result["metrics"]["total"], 1)
        self.assertEqual(result["excludedUnknown"], 1)
        self.assertEqual(result["coverage"], 1)
        self.assertTrue(result["rows"][1]["excluded"])

    def test_structured_evidence_failure_is_unresolved(self):
        async def judge(item, rule):
            return {"violation": False, "reason": "证据不在原文", "evidenceValid": False,
                    "evidence": [{"quote": "虚构"}], "steps": {"complaintIntent": True}}
        result = asyncio.run(evaluate_rule({"semantic": "风险"}, [case("C")], judge))
        self.assertIsNone(result["rows"][0]["predicted"])
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["metrics"]["accuracy"], 0)
        self.assertTrue(result["rows"][0]["reviewRequired"])
        self.assertTrue(result["rows"][0]["steps"]["complaintIntent"])

    def test_structured_unknown_is_not_normal(self):
        async def judge(item, rule):
            return {"violation": None, "reason": "后续对话缺失", "reviewRequired": True}
        result = asyncio.run(evaluate_rule({"semantic": "风险"}, [case("C")], judge))
        self.assertEqual(result["metrics"]["tn"], 0)
        self.assertEqual(result["metrics"]["unresolved"], 1)

    def test_cache_tracks_approved_knowledge_and_role(self):
        async def run():
            seen, cache = [], {}
            async def judge(item, rule):
                seen.append(rule["_approvedKnowledge"])
                return True, "解释"
            rule = {"semantic": "风险", "actor": "AGENT", "_approvedKnowledge": [{"content": "v1"}]}
            items = [case("C", True, text="客户：您好\n坐席：风险")]
            await evaluate_rule(rule, items, judge, cache=cache)
            await evaluate_rule(dict(rule, _approvedKnowledge=[{"content": "v2"}]), items, judge, cache=cache)
            self.assertEqual(len(seen), 2)
        asyncio.run(run())

    def test_complaint_workflow_does_not_short_circuit_with_regex(self):
        called = []
        async def judge(item, rule):
            called.append(True)
            return {"violation": False, "reason": "坐席已回应", "evidenceValid": True,
                    "steps": {"complaintIntent": True, "adequateResponse": True, "timelyResponse": True}}
        rule = {"regex": "投诉", "semantic": "投诉回应", "logic": "OR", "actor": "BOTH", "workflow": "COMPLAINT_RESPONSE"}
        result = asyncio.run(evaluate_rule(rule, [case("C", text="客户：我要投诉\n坐席：我会马上处理")], judge))
        self.assertTrue(result["rows"][0]["regexHit"])
        self.assertFalse(result["rows"][0]["predicted"])
        self.assertEqual(len(called), 1)

    def test_actor_regex_cannot_match_other_speaker(self):
        result = asyncio.run(evaluate_rule({"regex": "验证码", "actor": "AGENT"},
            [case("C", text="客户：验证码不能告诉别人\n坐席：好的")], false_judge))
        self.assertFalse(result["rows"][0]["predicted"])
        result = asyncio.run(evaluate_rule({"regex": "验证码", "actor": "AGENT"},
            [case("U", text="没有角色的转写")], false_judge))
        self.assertIsNone(result["rows"][0]["predicted"])


class GatePolicyTests(unittest.TestCase):
    def perfect(self):
        return {"coverage": 1, "metrics": {"accuracy": 1, "recall": 1, "falsePositiveRate": 0,
                "total": 10, "fn": 0, "fp": 0, "f1": 1}}

    def test_unchanged_baseline_is_truthfully_labeled(self):
        result = self.perfect()
        summary = {s: {"positives": 5, "negatives": 5} for s in (VALIDATION, TEST)}
        gate = compute_gate(result, result, result, result, .9, summary, False)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["outcome"], "PASSED_BASELINE")
        self.assertFalse(gate["improved"])

    def test_one_per_class_is_insufficient_even_if_perfect(self):
        result = self.perfect()
        summary = {s: {"positives": 1, "negatives": 1} for s in (VALIDATION, TEST)}
        self.assertEqual(compute_gate(result, result, result, result, .9, summary, False)["status"], "INSUFFICIENT")

    def test_incomplete_coverage_cannot_pass(self):
        result = self.perfect()
        bad = dict(result, coverage=.9)
        summary = {s: {"positives": 5, "negatives": 5} for s in (VALIDATION, TEST)}
        self.assertEqual(compute_gate(bad, result, result, result, .9, summary, True)["status"], "FAIL")

    def test_shared_capability_changes_require_regression(self):
        rule = {"semantic": "原语义", "actor": "AGENT", "knowledgeIds": []}
        self.assertTrue(needs_regression(rule, dict(rule, semantic="新语义"), "p", "p"))
        self.assertTrue(needs_regression(rule, dict(rule, actor="CUSTOMER"), "p", "p"))
        self.assertTrue(needs_regression(rule, dict(rule, knowledgeIds=["k1"]), "p", "p"))
        self.assertFalse(needs_regression(rule, dict(rule, regex="变化"), "p", "p"))
        state = {"rules": [rule], "datasetVersion": "v1"}
        self.assertNotEqual(context_fingerprint(state), context_fingerprint(dict(state, datasetVersion="v2")))


class SearchPolicyTests(unittest.TestCase):
    def run_search(self, rule, proposer, cases=None, guard=None, policy=None, rounds=5, judge=false_judge):
        return asyncio.run(optimize(rule, cases or dataset(), 1, rounds, lambda prompt, reason: judge,
            proposer, DEFAULT_SEMANTIC_PROMPT, lambda event: None, lambda: False,
            prompt_guard=guard, policy=policy))

    def test_no_training_errors_stops_without_inventing_patch(self):
        cases = dataset()
        for item in cases:
            if item["split"] != TRAIN and item["humanResult"] == VIOLATION:
                item["transcript"] = "另一种风险表达"
        async def proposer(*args):
            self.fail("无训练错误不应调用生成器")
        result = self.run_search({"regex": "^风险$"}, proposer, cases)
        self.assertEqual(result["stopReason"], "NO_TRAIN_ERRORS")
        self.assertEqual(result["budget"]["used"], 0)

    def test_duplicate_candidates_use_patience_without_reevaluation(self):
        async def proposer(rule, result, target, history, prompt):
            return {"regex": "永不匹配", "semanticPrompt": prompt}
        result = self.run_search({"regex": "永不匹配"}, proposer)
        self.assertEqual(result["stopReason"], "PATIENCE_EXHAUSTED")
        self.assertEqual(result["budget"]["used"], 2)
        self.assertEqual(result["budget"]["candidateEvaluations"], 0)
        self.assertTrue(all(i["rejectionCode"] == "DUPLICATE_CANDIDATE" for i in result["iterations"]))

    def test_invalid_candidate_does_not_abort_later_round(self):
        attempts = []
        async def proposer(rule, result, target, history, prompt):
            attempts.append(True)
            return {"regex": "(" if len(attempts) == 1 else "永不匹配|风险", "semanticPrompt": prompt}
        result = self.run_search({"regex": "永不匹配"}, proposer)
        self.assertEqual(result["iterations"][0]["rejectionCode"], "INVALID_CANDIDATE")
        self.assertTrue(result["iterations"][1]["accepted"])
        self.assertEqual(result["gate"]["outcome"], "IMPROVED")

    def test_candidate_budget_and_policy_cannot_be_mutated_by_proposal(self):
        async def proposer(rule, result, target, history, prompt):
            return {"regex": "另一个不匹配", "evaluationPolicy": {"minRecall": 0}, "semanticPrompt": prompt}
        result = self.run_search({"regex": "永不匹配"}, proposer, policy={"maxCandidateCalls": 1})
        self.assertEqual(result["stopReason"], "CANDIDATE_BUDGET_EXHAUSTED")
        self.assertEqual(result["evaluationPolicy"]["minRecall"], .9)

    def test_semantic_change_calls_guard_when_template_unchanged(self):
        seen = []
        async def proposer(rule, result, target, history, prompt):
            return {"regex": "", "semantic": "能够识别", "logic": "OR", "semanticPrompt": prompt}
        async def judge(item, rule):
            return rule["semantic"] == "能够识别" and item["humanResult"] == VIOLATION, "替身判定"
        async def guard(candidate, prompt):
            seen.append(candidate["semantic"])
            return {"status": "PASS"}
        result = self.run_search({"semantic": "不能识别"}, proposer, guard=guard, judge=judge)
        self.assertEqual(seen, ["能够识别"])
        self.assertTrue(result["sharedInputChanged"])
        self.assertFalse(result["promptChanged"])

    def test_unknown_labels_never_reach_proposer(self):
        cases = dataset() + [dict(case("U", text="不能进入提示词"), humanResult=UNKNOWN)]
        async def proposer(rule, result, target, history, prompt):
            self.assertTrue(all(row["humanResult"] in (NORMAL, VIOLATION) for row in result["rows"]))
            self.assertNotIn("U", result["optimizationContext"]["summary"]["errorCallIds"])
            return {"regex": "永不匹配|风险"}
        result = self.run_search({"regex": "永不匹配"}, proposer, cases)
        self.assertEqual(result["excludedUnknown"], 1)
        self.assertEqual(result["gate"]["status"], "PASS")

    def test_recall_cannot_be_sacrificed_for_majority_accuracy(self):
        cases = []
        for split in (TRAIN, VALIDATION, TEST):
            cases.extend(case(f"{split}p{i}", True, split, "风险") for i in range(5))
            cases.extend(case(f"{split}n{i}", False, split, "普通误报" if i < 10 else "普通") for i in range(45))
        async def proposer(rule, result, target, history, prompt):
            return {"regex": "风险x", "semanticPrompt": prompt}
        result = self.run_search({"regex": "风险|误报"}, proposer, cases)
        self.assertFalse(result["iterations"][0]["accepted"])
        self.assertTrue("探针" in result["iterations"][0]["selectionReason"]
                        or "召回" in result["iterations"][0]["selectionReason"])
        self.assertEqual(result["best"]["validation"]["metrics"]["recall"], 1)


if __name__ == "__main__":
    unittest.main()
