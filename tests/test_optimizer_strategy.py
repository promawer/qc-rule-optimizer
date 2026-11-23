import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from app.evaluator import evaluate_rule
from app.llm import LLMClient, LLMError, build_judge_prompt, judge_semantic
from app.models import ApplyInput, NORMAL, VIOLATION
from app.optimization_context import build_optimization_context, extract_context
from app.optimizer import build_optimizer_prompt, optimize
from app.prompt_regression import context_fingerprint, make_prompt_guard
from app.prompts import DEFAULT_SEMANTIC_PROMPT
from app.splits import TEST, TRAIN, VALIDATION
from app.store import Store


def row(call_id, text, human=VIOLATION, predicted=False, **kwargs):
    return {"id": call_id, "callId": call_id, "transcript": text,
            "humanResult": human, "autoResult": NORMAL, "predicted": predicted,
            "regexHit": False, "semanticHit": False, "split": TRAIN, **kwargs}


class ContextTest(unittest.TestCase):
    def test_diverse_errors_and_similar_correct_controls(self):
        rows = [row(f"C{i:02}", "坐席：请把短信验证码告诉我") for i in range(10)]
        rows += [row("C20", "坐席：如果有人索要短信六位数字也要拒绝", NORMAL, True,
                     regexHit=True, semanticHit=False),
                 row("P", "坐席：请把短信验证码念给我", predicted=True),
                 row("N", "坐席：不要把短信验证码告诉别人", NORMAL, False),
                 row("V_SECRET", "验证集秘密", split=VALIDATION),
                 row("X_SECRET", "测试集秘密", split=TEST),
                 row("FAILED", "接口失败", predicted=None, error="timeout")]
        context = build_optimization_context({"regex": "验证码", "semantic": "索要验证码"}, {"rows": rows})
        self.assertEqual(context["summary"]["errorTotal"], 11)
        self.assertEqual(len(context["errors"]), 2)  # 十条重复错例只占一个名额
        self.assertEqual(set(context["summary"]["controlCallIds"]), {"P", "N"})
        text = json.dumps(context, ensure_ascii=False)
        self.assertNotIn("秘密", text)
        self.assertNotIn("FAILED", text)
        reverse = build_optimization_context({}, {"rows": list(reversed(rows))})
        self.assertEqual(context["summary"], reverse["summary"])

    def test_context_keeps_late_evidence_and_roles_within_budget(self):
        text = "坐席：欢迎咨询。\n" * 400 + "客户：短信到了。\n坐席：请把验证码念给我。\n" + "客户：我再考虑一下。\n" * 100
        start = text.index("验证码")
        excerpt = extract_context(text, {"semantic": "索要短信验证码"}, {
            "regexEvidence": {"start": start, "end": start + 3},
        })
        self.assertIn("坐席：请把验证码念给我", excerpt)
        self.assertIn("客户：短信到了", excerpt)
        self.assertLessEqual(len(excerpt), 1200)
        self.assertIn("省略", excerpt)
        # 没有标点/换行的长转写也必须能定位后半段语义线索。
        unbroken = "您好欢迎办理业务" * 1500 + "请把短信验证码告诉我" + "再见" * 300
        self.assertIn("验证码告诉我", extract_context(unbroken, {"semantic": "索要短信验证码"}))
        self.assertIn("验证码告诉我", build_judge_prompt({"semantic": "索要短信验证码"}, unbroken))

    def test_prompt_contains_reasons_controls_and_rejected_candidate(self):
        result = {"metrics": {"total": 2}, "rows": [
            row("E", "坐席：请不要泄露验证码", NORMAL, True,
                regexReason="正则命中「验证码」", semanticReason="此处是安全提醒", regexHit=True),
            row("P", "坐席：验证码念给我", predicted=True),
        ]}
        text = build_optimizer_prompt({"regex": "验证码"}, result, .9, [{
            "index": 1, "metrics": {"accuracy": .5}, "accepted": False,
            "rule": {"regex": "被丢弃的表达"}, "rationale": "范围过宽",
        }], DEFAULT_SEMANTIC_PROMPT)
        for value in ("安全提醒", "正确违规对照", "被丢弃的表达", "未采纳", "changePlan", "局部修改"):
            self.assertIn(value, text)


class EvaluationSafetyTest(unittest.TestCase):
    def test_semantic_only_and_does_not_skip_judge_and_cache_tracks_rule(self):
        calls = []
        async def judge(case, rule):
            calls.append(rule["description"])
            return rule["description"] == "索要", "判定理由"
        async def run():
            cache = {}
            rule = {"regex": "", "semantic": "检查验证码", "logic": "AND", "description": "索要"}
            first = await evaluate_rule(rule, [row("C", "验证码")], judge, cache=cache)
            second = await evaluate_rule(dict(rule, description="提醒"), [row("C", "验证码")], judge, cache=cache)
            self.assertTrue(first["rows"][0]["predicted"])
            self.assertFalse(second["rows"][0]["predicted"])
        asyncio.run(run())
        self.assertEqual(len(calls), 2)

    def test_malformed_verdict_cannot_be_scored_as_normal(self):
        class Client:
            async def chat_json(self, *args, **kwargs):
                return {"reason": "没有返回判定"}
        with self.assertRaises(LLMError):
            asyncio.run(judge_semantic(Client(), row("C", "测试"), {}, DEFAULT_SEMANTIC_PROMPT, []))


def regression_state():
    rules = [
        {"id": "one", "code": "R1", "name": "当前规则", "regex": "", "semantic": "索要验证码", "logic": "OR"},
        {"id": "two", "code": "R2", "name": "其他规则", "regex": "", "semantic": "威胁催收", "logic": "OR"},
    ]
    calls = [{"id": name, "callId": name, "transcript": name,
              "humanRules": ["R2"] if name == "VP" else [], "autoRules": []}
             for name in ("VP", "VN", "TRAIN_SECRET", "TEST_SECRET")]
    return {"rules": rules, "calls": calls, "semanticPrompt": DEFAULT_SEMANTIC_PROMPT,
            "settings": {}, "splits": {"R2": {"VP": VALIDATION, "VN": VALIDATION,
                                                "TRAIN_SECRET": TRAIN, "TEST_SECRET": TEST}}}


class PromptRegressionTest(unittest.TestCase):
    def test_regression_rejects_damage_and_uses_only_validation(self):
        state = regression_state()
        seen = []
        async def judge(case, rule, prompt, rules):
            seen.append(case["callId"])
            if "破坏" in prompt:
                self.assertEqual(rules[0]["semantic"], "候选语义")
                return False, ""
            return case["callId"] == "VP", ""
        guard = make_prompt_guard(state, state["rules"][0], judge, lambda event: None, lambda: False)
        result = asyncio.run(guard(dict(state["rules"][0], semantic="候选语义"), DEFAULT_SEMANTIC_PROMPT + "破坏"))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["checks"][0]["lostCorrectCount"], 1)
        self.assertEqual(set(seen), {"VP", "VN"})

    def test_regression_passes_safe_prompt_and_fails_closed_on_errors(self):
        state = regression_state()
        async def judge(case, rule, prompt, rules):
            if "错误" in prompt:
                raise LLMError("服务不可用")
            return case["callId"] == "VP", ""
        guard = make_prompt_guard(state, state["rules"][0], judge, lambda event: None, lambda: False)
        safe = asyncio.run(guard(state["rules"][0], DEFAULT_SEMANTIC_PROMPT + "安全"))
        failed = asyncio.run(guard(state["rules"][0], DEFAULT_SEMANTIC_PROMPT + "错误"))
        self.assertEqual(safe["status"], "PASS")
        self.assertEqual(safe["contextFingerprint"], context_fingerprint(state))
        self.assertEqual(failed["status"], "BLOCKED")

    def test_insufficient_other_rule_labels_blocks_without_model_calls(self):
        state = regression_state()
        state["calls"][0]["humanRules"] = []
        async def judge(*args):
            self.fail("样本不足不应该调用模型")
        guard = make_prompt_guard(state, state["rules"][0], judge, lambda event: None, lambda: False)
        result = asyncio.run(guard(state["rules"][0], DEFAULT_SEMANTIC_PROMPT + "新"))
        self.assertEqual(result["status"], "INSUFFICIENT")

    def test_fingerprint_ignores_runs_but_tracks_data_rules_and_model(self):
        state = regression_state()
        original = context_fingerprint(state)
        state["runs"] = [{"id": "new"}]
        self.assertEqual(original, context_fingerprint(state))
        for field, value in (("settings", {"model": "changed"}), ("calls", []), ("rules", []), ("splits", {})):
            updated = dict(state, **{field: value})
            self.assertNotEqual(original, context_fingerprint(updated))


class IterationStrategyTest(unittest.TestCase):
    def run_optimization(self, guard=None, change_prompt=True):
        cases = [row(split + str(i), "风险" if i == 0 else "普通", VIOLATION if i == 0 else NORMAL,
                     split=split) for split in (TRAIN, VALIDATION, TEST) for i in range(2)]
        test = self
        class Proposer:
            async def __call__(self, rule, result, target, history, prompt):
                context = result["optimizationContext"]
                test.assertTrue(all(s["callId"].startswith(TRAIN) for s in context["errors"] + context["controls"]))
                test.assertEqual(rule["code"], "R1")
                return {"regex": "风险", "semantic": "", "logic": "OR", "rationale": "补充表达",
                        "semanticPrompt": prompt + "\n新" if change_prompt else prompt}
        async def judge(case, rule):
            return False, ""
        return asyncio.run(optimize(
            {"id": "one", "code": "R1", "regex": "永不命中", "semantic": "", "logic": "OR"},
            cases, 1, 1, lambda prompt, reason: judge, Proposer(), DEFAULT_SEMANTIC_PROMPT,
            lambda event: None, lambda: False, prompt_guard=guard,
        ))

    def test_prompt_change_is_rejected_without_guard_even_if_accuracy_improves(self):
        result = self.run_optimization()
        self.assertFalse(result["iterations"][0]["accepted"])
        self.assertEqual(result["iterations"][0]["promptRegression"]["status"], "BLOCKED")
        self.assertFalse(result["promptChanged"])

    def test_guard_failure_blocks_candidate_but_rule_only_change_still_works(self):
        async def guard(candidate, prompt):
            return {"status": "BLOCKED", "reasons": ["R2 退步"]}
        blocked = self.run_optimization(guard)
        self.assertIn("R2 退步", blocked["iterations"][0]["selectionReason"])
        accepted = self.run_optimization(change_prompt=False)
        self.assertEqual(accepted["status"], "reached")
        self.assertEqual(accepted["iterations"][0]["changedFields"], ["regex"])


class ApplicationSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qco-test-")
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "store.json"
        with patch.dict(os.environ, {"QCO_STORE": str(path)}):
            self.main = importlib.import_module("app.main")
        self.store = Store(path)
        self.patcher = patch.object(self.main, "store", self.store)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def insert_run(self, regression=True):
        state = self.store.snapshot()
        rule = state["rules"][0]
        run = {"id": "run-test", "ruleId": rule["id"], "gate": {"status": "PASS"},
               "contextFingerprint": context_fingerprint(state), "codeFingerprint": self.main.code_fingerprint(),
               "bestRule": dict(rule, regex="改进表达"), "baselinePrompt": state["semanticPrompt"],
               "bestPrompt": state["semanticPrompt"] + "\n新提示词", "promptChanged": True}
        if regression:
            run["promptRegression"] = {"status": "PASS", "contextFingerprint": context_fingerprint(state)}
        self.store.mutate(lambda draft: draft["runs"].append(run))
        return run

    def test_apply_requires_regression_even_when_forced(self):
        self.insert_run(False)
        before = self.store.snapshot()
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.main.apply_run("run-test", ApplyInput(reason="应用", force=True)))
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(before, self.store.snapshot())

    def test_stale_regression_is_rejected_and_current_regression_applies(self):
        self.insert_run()
        self.store.mutate(lambda draft: draft["calls"][0].update(transcript="通话数据变化"))
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.main.apply_run("run-test", ApplyInput(reason="应用")))
        self.assertIn("过期", error.exception.detail)
        self.store.mutate(lambda draft: draft["runs"].clear())
        run = self.insert_run()
        asyncio.run(self.main.apply_run("run-test", ApplyInput(reason="应用")))
        self.assertEqual(self.store.snapshot()["semanticPrompt"], run["bestPrompt"])

    def test_rule_only_run_is_stale_after_manual_prompt_change(self):
        self.insert_run(False)
        self.store.mutate(lambda draft: draft["runs"][0].update(promptChanged=False))
        self.store.mutate(lambda draft: draft.update(semanticPrompt=DEFAULT_SEMANTIC_PROMPT + "\n人工已修改"))
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.main.apply_run("run-test", ApplyInput(reason="只应用规则")))
        self.assertEqual(error.exception.status_code, 409)
        self.assertTrue(self.store.snapshot()["semanticPrompt"].endswith("人工已修改"))

    def test_semantic_judge_receives_candidate_in_global_rule_list(self):
        state = self.store.snapshot()
        candidate = dict(state["rules"][0], semantic="本轮新语义")
        async def fake(client, case, rule, prompt, all_rules, with_reason):
            match = next(r for r in all_rules if r["id"] == candidate["id"])
            self.assertEqual(match["semantic"], "本轮新语义")
            return True, ""
        with patch.object(self.main, "judge_semantic", fake):
            judge = self.main._make_judge_factory(object(), state["rules"])(DEFAULT_SEMANTIC_PROMPT, True)
            asyncio.run(judge(row("C", "测试"), candidate))

    def test_real_mode_api_round_trip_with_local_model_and_prompt_regression(self):
        # 使用本地模型替身走完整真实模式流程，不访问外部模型或修改用户数据。
        state = regression_state()
        state.update(version=2, runs=[], settings={"mode": "deepseek", "requestIntervalSec": 0})
        state["rules"][0]["semantic"] = "原规则尚不能识别"
        state["calls"] = []
        state["splits"] = {"R1": {}, "R2": {}}
        for split in (TRAIN, VALIDATION, TEST):
            for positive in (True, False):
                call_id = split + str(positive)
                state["calls"].append({"id": call_id, "callId": call_id,
                                       "transcript": split + (" 风险" if positive else " 普通"),
                                       "humanRules": ["R1", "R2"] if positive else [], "autoRules": []})
                for code in ("R1", "R2"):
                    state["splits"][code][call_id] = split
        self.store.mutate(lambda draft: (draft.clear(), draft.update(state)))
        test = self
        async def model(client, system, user, **kwargs):
            if not kwargs.get("judge"):
                test.assertIn("正确正常对照", user)
                test.assertNotIn("VALIDATIONTrue", user)
                test.assertNotIn("TESTTrue", user)
                return {"regex": "风险", "semantic": "原规则尚不能识别", "logic": "OR",
                        "semanticPrompt": DEFAULT_SEMANTIC_PROMPT + "\n安全改进",
                        "rationale": "补充漏报", "changePlan": "补充正则并改进通用要求"}
            current_rule = user.split("通话文本：")[0]
            return {"violation": "威胁催收" in current_rule and "风险" in user.split("通话文本：")[1],
                    "reason": "本地判定"}
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.main.app), base_url="http://test") as client:
                response = await client.post("/api/optimize", json={"ruleId": "one", "targetAccuracy": 1, "maxRounds": 1,
                                                                   "evaluationPolicy": {"minPerClass": 1}})
                response.raise_for_status()
                job_id = response.json()["jobId"]
                await self.main.TASKS[job_id]
                job = (await client.get(f"/api/jobs/{job_id}")).json()
                self.assertEqual(job["status"], "reached", job.get("error"))
                self.assertEqual(job["promptRegression"]["status"], "PASS")
                self.assertTrue(job["iterations"][0]["contextSummary"])
                self.assertEqual(job["iterations"][0]["changedFields"], ["regex", "semanticPrompt"])
                cases = await client.get(f"/api/jobs/{job_id}/iterations/1/cases?split=TRAIN")
                self.assertEqual(cases.status_code, 200)
                applied = await client.post(f"/api/runs/{job_id}/apply", json={"reason": "本地回归通过"})
                self.assertEqual(applied.status_code, 200, applied.text)
        with patch.object(LLMClient, "chat_json", model), patch.dict(self.main.RUNTIME, {"apiKey": "local-test"}), \
                patch.dict(self.main.JOBS, {}, clear=True), patch.dict(self.main.TASKS, {}, clear=True):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
