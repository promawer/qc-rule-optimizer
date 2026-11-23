import asyncio
import copy
import hashlib
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException

from app.data_governance import refresh_dataset
from app.evaluator import evaluate_rule
from app.experiments import checkpoint, code_fingerprint, freeze_experiment, restore_jobs
from app.governance import model_manifest, register_governance
from app.importer import cases_for_rule
from app.models import RuleInput, ApplyInput
from app.prompt_regression import context_fingerprint
from app.store import Store


class MainIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qco-system-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.dict(os.environ, {"QCO_STORE": str(self.root / "store.json")}):
            self.main = importlib.import_module("app.main")
        self.store = Store(self.root / "store.json")
        for name, value in (("store", self.store), ("JOBS", {}), ("TASKS", {}),
                            ("CHECKPOINT_DIR", self.root / "jobs"), ("EXPERIMENT_DIR", self.root / "experiments")):
            p = patch.object(self.main, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_rename_preserves_stable_labels_and_unknown_new_rule(self):
        state = self.store.snapshot()
        rule = state["rules"][0]
        before = [c["humanResult"] for c in cases_for_rule(state["calls"], rule)]
        updated = asyncio.run(self.main.update_rule(rule["id"], RuleInput(**dict(rule, code="RENAMED", changeReason="业务编号调整"))))
        after = self.store.snapshot()
        self.assertEqual(before, [c["humanResult"] for c in cases_for_rule(after["calls"], updated)])
        self.assertGreater(after["datasetVersion"], state["datasetVersion"])
        fresh = asyncio.run(self.main.create_rule(RuleInput(code="NEW", name="新规则", regex="测试")))
        self.assertTrue(all(c["humanResult"] == "UNKNOWN" for c in cases_for_rule(self.store.snapshot()["calls"], fresh)))

    def test_active_task_blocks_rule_and_prompt_edits(self):
        self.main.JOBS["job-running"] = {"id": "job-running", "status": "running", "ruleName": "测试"}
        rule = self.store.snapshot()["rules"][0]
        with self.assertRaises(HTTPException):
            asyncio.run(self.main.update_rule(rule["id"], RuleInput(**dict(rule, regex="新表达", changeReason="修改"))))

    def test_rule_only_force_cannot_apply_stale_code_or_data(self):
        state = self.store.snapshot()
        rule = state["rules"][0]
        run = {"id": "run-frozen", "ruleId": rule["id"], "gate": {"status": "FAIL"},
               "bestRule": dict(rule, regex="改进表达"), "promptChanged": False,
               "contextFingerprint": context_fingerprint(state), "codeFingerprint": "old-code"}
        self.store.mutate(lambda s: s["runs"].append(run))
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.main.apply_run(run["id"], ApplyInput(reason="强制", force=True)))
        self.assertEqual(error.exception.status_code, 409)

    def test_snapshot_omits_credentials_and_restart_marks_interruption(self):
        state = self.store.snapshot()
        state["settings"].update(apiKey="SECRET", headers=[{"name":"X-Internal-Credential","value":"SECRET"}],
                                  requestTemplate='{"secret":"SECRET","prompt":"{{prompt}}"}')
        job = {"id":"job-snapshot", "status":"running", "createdAt":"now", "policy":{"minPerClass":5}}
        manifest = freeze_experiment(state, job, self.root / "experiments")
        text = json.dumps(manifest)
        self.assertNotIn("SECRET", text)
        self.assertEqual(manifest["calls"], state["calls"])
        checkpoint(job, self.root / "jobs")
        disk = (self.root / "jobs" / "job-snapshot.json").read_text()
        self.assertNotIn("SECRET", disk)
        self.assertNotIn('"_context"', disk)
        restored = restore_jobs(self.root / "jobs")["job-snapshot"]
        self.assertEqual(restored["status"], "cancelled")
        self.assertIn("重启", restored["error"])


class GovernanceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qco-governance-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "store.json")
        def seed(state):
            state["rules"] = [{"id":"r1", "code":"C1", "name":"测试能力", "regex":"需要处置", "semantic":"",
                               "logic":"OR", "actor":"AGENT", "workflow":"SIMPLE", "version":1}]
            state["settings"]["mode"] = "deepseek"
            state["calls"] = [{"id":f"d{i}", "callId":f"D{i}", "transcript":f"坐席：开发案例{i}，" + ("需要处置" if i % 2 else "您好"),
                               "annotations":{"r1":{"label":"VIOLATION" if i % 2 else "NORMAL","reviewStatus":"REVIEWED"}}}
                              for i in range(50)]
            state["datasetVersions"] = []
            refresh_dataset(state, "测试开发集")
        self.store.mutate(seed)
        self.evaluations = []
        self.runtime_calls = 0
        self.runtime_error_on = set()
        async def evaluator(bundle, calls):
            self.evaluations.append(bundle["rule"]["id"])
            async def judge(*args):
                raise AssertionError("纯正则验证不应调用外部模型")
            return await evaluate_rule(bundle["rule"], cases_for_rule(calls, bundle["rule"]), judge)
        async def runtime(bundle, case):
            self.runtime_calls += 1
            if self.runtime_calls in self.runtime_error_on:
                raise RuntimeError("test model unavailable")
            return {"predicted":False, "reason":"本地测试桩", "reviewRequired":False, "error":""}
        self.app = FastAPI()
        register_governance(self.app, self.store, evaluator, runtime, lambda:None)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def freeze(self):
        response = await self.client.post("/api/releases", json={"ruleId":"r1","author":"开发者","reason":"冻结候选"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    async def holdout(self, suffix="A"):
        lines = ["通话ID,通话文本,自动违规规则,人工违规规则"]
        for i in range(12):
            positive = i % 2
            lines.append(f"H{suffix}{i},坐席：独立案例{suffix}{i}，" + ("需要处置,,C1" if positive else "您好,,"))
        response = await self.client.post("/api/holdouts/import", files={"file":("hold.csv", "\n".join(lines).encode(), "text/csv")},
                                          data={"fullyReviewed":"true", "name":"封存A"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    async def evaluate(self, release, holdout):
        response = await self.client.post(f"/api/releases/{release}/evaluate", json={"holdoutId":holdout})
        self.assertEqual(response.status_code, 200, response.text)
        for _ in range(100):
            item = next(r for r in self.store.snapshot()["releases"] if r["id"]==release)
            if item["status"] != "EVALUATING":
                return item
            await asyncio.sleep(.001)
        self.fail("独立评测未完成")

    async def action(self, release, action, reviewer="审核者"):
        return await self.client.post(f"/api/releases/{release}/transition", json={"action":action,"reviewer":reviewer,"reason":"核对证据后操作","percent":50})

    async def test_full_local_release_flow_and_feedback(self):
        release = await self.freeze()
        self.assertEqual((await self.action(release,"APPROVE")).status_code,409)
        holdout = await self.holdout()
        item = await self.evaluate(release, holdout)
        self.assertEqual(item["status"],"EVALUATED",item.get("error"))
        self.assertEqual(item["evaluation"]["gate"]["outcome"],"PASSED_BASELINE")
        self.assertEqual(len(self.evaluations),2)
        self.assertEqual((await self.action(release,"APPROVE", "开发者")).status_code,409)
        self.assertEqual((await self.action(release,"APPROVE")).status_code,200)
        self.assertEqual((await self.action(release,"PUBLISH")).status_code,409)
        self.assertEqual((await self.action(release,"SHADOW")).status_code,200)
        self.assertEqual((await self.action(release,"CANARY")).status_code,409)
        response = await self.client.post("/api/runtime/judge",json={"ruleId":"r1","callId":"LIVE1","transcript":"坐席：您好"})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIsNotNone(response.json()["shadowResult"])
        self.assertEqual((await self.action(release,"CANARY")).status_code,200)
        cid = next(str(i) for i in range(100) if int(hashlib.sha256(str(i).encode()).hexdigest()[:8],16)%100<50)
        self.assertEqual((await self.client.post("/api/runtime/judge",json={"ruleId":"r1","callId":cid,"transcript":"坐席：您好"})).status_code,200)
        self.assertEqual((await self.action(release,"PUBLISH")).status_code,200)
        result = await self.client.post("/api/feedback",json={"releaseId":release,"callId":"F1","transcript":"坐席：新业务反馈，需要处置",
                          "label":"VIOLATION","reviewer":"反馈者","reason":"人工确认漏报","evidence":[{"quote":"需要处置"}]})
        self.assertEqual(result.status_code,200,result.text)
        fid=result.json()["id"]
        self.assertEqual((await self.client.post(f"/api/feedback/{fid}/accept",json={"reviewer":"反馈者","reason":"自审"})).status_code,409)
        before=self.store.snapshot()["datasetVersion"]
        accepted=await self.client.post(f"/api/feedback/{fid}/accept",json={"reviewer":"复核者","reason":"核对原文与口径"})
        self.assertEqual(accepted.status_code,200,accepted.text)
        self.assertGreater(accepted.json()["datasetVersion"],before)
        self.assertEqual((await self.action(release,"ROLLBACK")).status_code,200)
        self.assertEqual((await self.client.post("/api/runtime/judge",json={"ruleId":"r1","callId":"LIVE2","transcript":"坐席：您好"})).status_code,409)

    async def test_holdout_not_visible_or_reusable_for_another_candidate(self):
        h=await self.holdout()
        r=await self.freeze()
        await self.evaluate(r,h)
        other=await self.freeze()
        denied=await self.client.post(f"/api/releases/{other}/evaluate",json={"holdoutId":h})
        self.assertEqual(denied.status_code,409)
        public=(await self.client.get("/api/governance")).json()
        self.assertNotIn("calls",public["holdouts"][0])
        self.assertNotIn("evaluationRows",json.dumps(public))
        self.assertNotIn("独立案例",json.dumps(public,ensure_ascii=False))

    async def test_mock_cannot_gain_release_qualification(self):
        self.store.mutate(lambda s:s["settings"].update(mode="mock"))
        r=await self.freeze();h=await self.holdout()
        response=await self.client.post(f"/api/releases/{r}/evaluate",json={"holdoutId":h})
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.evaluations,[])

    async def test_holdout_rejects_historical_development_data(self):
        call=self.store.snapshot()["calls"][0]
        self.store.mutate(lambda s:(s.update(calls=[]),refresh_dataset(s,"清空开发集")))
        body="通话ID,通话文本,自动违规规则,人工违规规则\nCOPY,"+call["transcript"]+",,"
        response=await self.client.post("/api/holdouts/import",files={"file":("hold.csv",body.encode(),"text/csv")})
        self.assertEqual(response.status_code,400)

    async def test_knowledge_requires_training_sources_and_separate_counterexample(self):
        state=self.store.snapshot(); mapping=state["splits"]["C1"]
        train=[c["callId"] for c in state["calls"] if mapping[c["callId"]]=="TRAIN"]
        held=next(c["callId"] for c in state["calls"] if mapping[c["callId"]]!="TRAIN")
        payload={"ruleId":"r1","title":"候选经验","content":"按具体行为判断","scope":"业务A","exceptions":"普通咨询",
                 "sourceCallIds":[held],"counterExamples":[train[1]],"reason":"人工证据"}
        self.assertEqual((await self.client.post("/api/knowledge",json=payload)).status_code,400)
        payload["sourceCallIds"]=[train[0]]
        created=await self.client.post("/api/knowledge",json=payload)
        self.assertEqual(created.status_code,200,created.text)
        k=created.json()["id"]
        approved=await self.client.post(f"/api/knowledge/{k}/review",json={"action":"APPROVE","reviewer":"专家","reason":"核对反例"})
        self.assertEqual(approved.status_code,200,approved.text)
        self.assertEqual(approved.json()["status"],"APPROVED")
        self.assertNotIn(k,self.store.snapshot()["rules"][0].get("knowledgeIds",[]))

    async def test_tampered_bundle_cannot_be_approved(self):
        r=await self.freeze();h=await self.holdout()
        await self.evaluate(r,h)
        self.store.mutate(lambda s:s["releases"][0]["bundle"]["rule"].update(regex="损坏内容"))
        self.assertEqual((await self.action(r,"APPROVE")).status_code,409)

    async def ready_shadow(self):
        release = await self.freeze()
        await self.evaluate(release, await self.holdout())
        self.assertEqual((await self.action(release, "APPROVE")).status_code, 200)
        self.assertEqual((await self.action(release, "SHADOW")).status_code, 200)
        return release

    async def test_shadow_exception_preserves_primary_and_blocks_promotion(self):
        release = await self.ready_shadow()
        case = {"ruleId":"r1", "callId":"LIVE", "transcript":"坐席：您好"}
        self.assertEqual((await self.client.post("/api/runtime/judge", json=case)).status_code, 200)
        self.runtime_error_on.add(4)
        result = await self.client.post("/api/runtime/judge", json=case)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIs(result.json()["result"]["predicted"], False)
        self.assertIsNone(result.json()["shadowResult"]["predicted"])
        stats = self.store.snapshot()["releases"][0]["observations"]
        self.assertEqual(stats["shadow"], 2)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual((await self.action(release, "CANARY")).status_code, 409)

    async def test_primary_exception_is_recorded_before_returning_failure(self):
        release = await self.ready_shadow()
        self.runtime_error_on.add(1)
        case = {"ruleId":"r1", "callId":"LIVE", "transcript":"坐席：您好"}
        self.assertEqual((await self.client.post("/api/runtime/judge", json=case)).status_code, 502)
        self.assertEqual((await self.client.post("/api/runtime/judge", json=case)).status_code, 200)
        stats = self.store.snapshot()["releases"][0]["observations"]
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["shadow"], 1)
        self.assertEqual((await self.action(release, "CANARY")).status_code, 409)

    async def test_baseline_integrity_failure_still_allows_withdrawal(self):
        release = await self.ready_shadow()
        self.store.mutate(lambda s:s["releases"][0]["baselineBundle"]["rule"].update(regex="被改写的基线"))
        result = await self.client.post("/api/runtime/judge", json={"ruleId":"r1", "callId":"LIVE", "transcript":"坐席：您好"})
        self.assertEqual(result.status_code, 409)
        self.assertEqual((await self.action(release, "ROLLBACK")).status_code, 200)
        self.assertFalse(self.store.snapshot()["releaseChannels"]["r1"].get("shadow"))


if __name__ == "__main__":
    unittest.main()
