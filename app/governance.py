"""本地能力资产、封存验收及发布登记。发布通道供运行时接口使用，不连接外部生产系统。"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .data_governance import refresh_dataset
from .importer import build_calls, cases_for_rule
from .models import NORMAL, VIOLATION
from .prompt_regression import context_fingerprint, rules_with_candidate
from .prompts import DEFAULT_SEMANTIC_PROMPT
from .splits import normalized_transcript as split_normalized_transcript


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def model_manifest(settings):
    result = copy.deepcopy(settings)
    result.pop("apiKey", None)
    for header in result.get("headers") or []:
        if re.search(r"authorization|token|cookie|api.?key|secret", header.get("name", ""), re.I):
            header["value"] = "{{credential}}"
        elif header.get("name", "").lower() not in ("content-type", "accept", "user-agent"):
            header["valueHash"] = hashlib.sha256(str(header.pop("value", "")).encode()).hexdigest()
    # A literal credential in a custom body is not an exportable model configuration.
    template = result.get("requestTemplate") or ""
    if template:
        result["requestTemplateHash"] = hashlib.sha256(template.encode()).hexdigest()
        result.pop("requestTemplate", None)
    return result


def enrich_rule(state, rule):
    result = copy.deepcopy(rule)
    ids = set(rule.get("knowledgeIds") or [])
    result["_approvedKnowledge"] = [copy.deepcopy(item) for item in state.get("knowledge", [])
                                    if item.get("id") in ids and item.get("status") == "APPROVED"]
    return result


def enrich_rules(state):
    return [enrich_rule(state, rule) for rule in state.get("rules", [])]


def find(items, item_id, name):
    item = next((item for item in items if item.get("id") == item_id), None)
    if item is None:
        raise HTTPException(404, name + "不存在")
    return item


def audit(item, action, reviewer, reason):
    item.setdefault("history", []).append({"at": stamp(), "action": action,
                                           "reviewer": reviewer, "reason": reason})


def public_release(item):
    return {key: copy.deepcopy(value) for key, value in item.items()
            if key not in ("bundle", "baselineBundle", "evaluationRows")}


def verify_bundle(item):
    if (digest(item["bundle"]) != item["bundleHash"]
            or digest(item["baselineBundle"]) != item["bundle"].get("baselineHash")):
        raise HTTPException(409, "冻结能力包或基线完整性校验失败")


def normalized_transcript(call):
    # Remove formatting/time marks, retain role and content. Same source groups are also checked below.
    text = split_normalized_transcript(str(call.get("transcript") or ""))
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def assert_no_holdout_overlap(calls, others):
    texts = {normalized_transcript(c) for c in others}
    ids = {str(c.get("callId")) for c in others}
    groups = {str(c["groupId"]) for c in others if c.get("groupId")}
    if any(normalized_transcript(c) in texts or str(c.get("callId")) in ids
           or (c.get("groupId") and str(c["groupId"]) in groups) for c in calls):
        raise HTTPException(400, "封存数据与开发/其他封存数据存在相同通话、文本或同源分组，请提供独立新数据")


class NamedAction(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=1000)


class KnowledgeDraft(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=3000)
    scope: str = Field(min_length=1, max_length=1000)
    exceptions: str = Field(min_length=1, max_length=1500)
    sourceCallIds: List[str] = Field(default_factory=list)
    counterExamples: List[str] = Field(default_factory=list)
    ruleId: str
    reason: str = Field(min_length=1, max_length=1000)
    supersedes: Optional[str] = None


class AnnotationInput(NamedAction):
    callId: str
    ruleId: str
    label: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    steps: Dict[str, Optional[bool]] = Field(default_factory=dict)


class FreezeInput(BaseModel):
    ruleId: str
    runId: Optional[str] = None
    reason: str = Field(min_length=1, max_length=1000)
    author: str = Field(min_length=1, max_length=80)


class ReleaseEvaluation(BaseModel):
    holdoutId: str


class ReleaseAction(NamedAction):
    action: str
    percent: int = Field(default=10, ge=1, le=50)


class FeedbackInput(NamedAction):
    releaseId: str
    callId: str = Field(min_length=1, max_length=100)
    transcript: str = Field(min_length=1, max_length=20000)
    label: str
    groupId: str = ""
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    transcriptComplete: bool = False


class RuntimeInput(BaseModel):
    ruleId: str
    callId: str = Field(min_length=1, max_length=100)
    transcript: str = Field(min_length=1, max_length=20000)
    transcriptComplete: bool = False


def independent_gate(candidate, baseline, target, policy):
    m, b = candidate["metrics"], baseline["metrics"]
    minimum = max(5, int(policy.get("minPerClass", 5)))
    checks = []
    def add(key, passed, detail):
        checks.append({"key": key, "pass": bool(passed), "detail": detail})
    add("sample_size", m.get("violations", 0) >= minimum and m.get("normals", 0) >= minimum,
        "独立集每类至少 %s 条：违规 %s，正常 %s" % (minimum, m.get("violations", 0), m.get("normals", 0)))
    add("complete", candidate.get("coverage") == 1 and baseline.get("coverage") == 1
        and not candidate.get("errors") and not baseline.get("errors"), "基线与候选必须完成全部已复核样本")
    add("accuracy", m["accuracy"] >= target, "准确率 %.1f%% / 目标 %.1f%%" % (100*m["accuracy"], 100*target))
    add("recall", m["recall"] >= policy.get("minRecall", .9), "召回率满足冻结的业务底线")
    fpr = m["fp"] / max(m["fp"] + m["tn"], 1)
    add("fpr", fpr <= policy.get("maxFalsePositiveRate", .1), "误报率满足冻结的业务上限")
    add("regression", m["accuracy"] >= b["accuracy"] and m["recall"] >= b["recall"], "准确率与召回率不低于冻结基线")
    passed = all(c["pass"] for c in checks)
    return {"status": "PASS" if passed else "FAIL", "checks": checks,
            "outcome": "IMPROVED" if passed and (m["accuracy"] > b["accuracy"] or m["recall"] > b["recall"]) else "PASSED_BASELINE" if passed else "FAILED",
            "reasons": [c["detail"] for c in checks if not c["pass"]]}


def register_governance(app, store, evaluate_bundle, runtime_judge, ensure_idle):
    """Callbacks keep credentials and model transport outside the governance module."""
    workers = set()

    def check_named(payload):
        if not payload.reviewer.strip() or not payload.reason.strip():
            raise HTTPException(400, "复核人和理由不能为空")

    @app.get("/api/governance")
    async def overview():
        state = store.snapshot()
        return {"datasetVersion": state.get("datasetVersion", 1),
                "datasetVersions": [{k: v for k, v in x.items() if k not in ("calls", "splits", "rules")}
                                    for x in state.get("datasetVersions", [])],
                "migrationWarnings": state.get("dataWarnings", []),
                "knowledge": state.get("knowledge", []),
                "releases": [public_release(x) for x in state.get("releases", [])],
                "holdouts": [{k: v for k, v in x.items() if k != "calls"} for x in state.get("holdouts", [])],
                "channels": state.get("releaseChannels", {}), "feedback": state.get("feedback", []),
                "deploymentScope": "LOCAL_GATEWAY", "identityMode": "NAMED_LOCAL_REVIEW"}

    @app.put("/api/annotations")
    async def annotate(payload: AnnotationInput):
        ensure_idle()
        check_named(payload)
        if payload.label not in (NORMAL, VIOLATION, "UNKNOWN"):
            raise HTTPException(400, "label 必须为 NORMAL、VIOLATION 或 UNKNOWN")
        def change(state):
            rule = find(state["rules"], payload.ruleId, "规则")
            call = next((c for c in state.get("calls", []) if c["callId"] == payload.callId), None)
            if call is None:
                raise HTTPException(404, "通话不存在")
            for ev in payload.evidence:
                if not ev.get("quote") or ev["quote"] not in call["transcript"]:
                    raise HTTPException(400, "人工证据必须是通话中的原文")
            call.setdefault("annotations", {})[rule["id"]] = {
                "label": payload.label, "reviewStatus": "REVIEWED" if payload.label != "UNKNOWN" else "UNREVIEWED",
                "reviewer": payload.reviewer.strip(), "reason": payload.reason.strip(), "evidence": payload.evidence,
                "steps": payload.steps, "reviewedAt": stamp(), "source": "manual"}
            refresh_dataset(state, "人工复标：" + payload.reason.strip())
            return {"ok": True, "datasetVersion": state["datasetVersion"]}
        return store.mutate(change)

    def validate_knowledge_sources(state, payload):
        for key in ("title", "content", "scope", "exceptions", "reason"):
            if not getattr(payload, key).strip():
                raise HTTPException(400, "知识内容、范围、例外和依据不能为空")
        rule = find(state["rules"], payload.ruleId, "规则")
        rows = {c["callId"]: c for c in cases_for_rule(state.get("calls", []), rule, state.get("splits", {}))}
        if not payload.sourceCallIds or not payload.counterExamples:
            raise HTTPException(400, "经验需要来源案例和正确反例的通话ID")
        if set(payload.sourceCallIds) & set(payload.counterExamples):
            raise HTTPException(400, "来源案例与正确反例应为不同通话")
        for cid in payload.sourceCallIds + payload.counterExamples:
            case = rows.get(cid)
            if not case or case.get("split") != "TRAIN" or case.get("humanResult") not in (NORMAL, VIOLATION):
                raise HTTPException(400, "知识来源与反例只能引用已复核训练样本：" + cid)
        return rule

    @app.post("/api/knowledge")
    async def create_knowledge(payload: KnowledgeDraft):
        ensure_idle()
        def change(state):
            validate_knowledge_sources(state, payload)
            if payload.supersedes:
                find(state.get("knowledge", []), payload.supersedes, "被替代知识")
            item = dict(payload.dict(), id="knowledge-" + uuid.uuid4().hex[:10], status="DRAFT",
                        version=1, createdAt=stamp(), datasetVersion=state.get("datasetVersion"), history=[])
            audit(item, "DRAFT", "", payload.reason)
            state.setdefault("knowledge", []).append(item)
            return item
        return store.mutate(change)

    @app.post("/api/knowledge/{knowledge_id}/review")
    async def review_knowledge(knowledge_id: str, payload: ReleaseAction):
        ensure_idle()
        check_named(payload)
        if payload.action not in ("APPROVE", "REJECT", "RETIRE"):
            raise HTTPException(400, "知识操作为 APPROVE、REJECT 或 RETIRE")
        def change(state):
            item = find(state.get("knowledge", []), knowledge_id, "知识")
            if payload.action in ("APPROVE", "REJECT") and item["status"] != "DRAFT":
                raise HTTPException(409, "只有草稿知识可以审核；修改已批准知识请创建替代版本")
            if payload.action == "RETIRE" and item["status"] != "APPROVED":
                raise HTTPException(409, "只有已批准知识可以停用")
            if payload.action == "APPROVE":
                validate_knowledge_sources(state, KnowledgeDraft(**item))
                if item.get("datasetVersion") != state.get("datasetVersion"):
                    raise HTTPException(409, "知识依据的数据版本已变化，请基于新证据重新建立草稿")
            item["status"] = {"APPROVE": "APPROVED", "REJECT": "REJECTED", "RETIRE": "RETIRED"}[payload.action]
            item["reviewedAt"] = stamp()
            audit(item, payload.action, payload.reviewer.strip(), payload.reason.strip())
            # Knowledge is linked explicitly in rule editing; approval itself cannot silently publish a rule.
            return item
        return store.mutate(change)

    @app.post("/api/holdouts/import")
    async def import_holdout(file: UploadFile = File(...), name: str = Form("独立验收集"),
                             fullyReviewed: bool = Form(False)):
        ensure_idle()
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(400, "文件超过10MB")
        state = store.snapshot()
        try:
            calls, report = build_calls(file.filename or "holdout.csv", content, state["rules"], fully_reviewed=fullyReviewed)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        def change(draft):
            history = [c for version in draft.get("datasetVersions", []) for c in version.get("calls", [])]
            others = draft.get("calls", []) + history + [c for h in draft.get("holdouts", []) for c in h["calls"]]
            assert_no_holdout_overlap(calls, others)
            item = {"id": "holdout-" + uuid.uuid4().hex[:10], "name": name[:100], "createdAt": stamp(),
                    "calls": calls, "count": len(calls), "fingerprint": digest(calls), "usedBy": None,
                    "warnings": report.get("warnings", [])}
            draft.setdefault("holdouts", []).append(item)
            return {k: v for k, v in item.items() if k != "calls"}
        return store.mutate(change)

    @app.post("/api/releases")
    async def freeze(payload: FreezeInput):
        from .experiments import code_fingerprint
        ensure_idle()
        if not payload.author.strip() or not payload.reason.strip():
            raise HTTPException(400, "提交人和冻结理由不能为空")
        def change(state):
            current = find(state["rules"], payload.ruleId, "规则")
            rules = enrich_rules(state)
            candidate = enrich_rule(state, current)
            prompt = state.get("semanticPrompt") or DEFAULT_SEMANTIC_PROMPT
            baseline = {"rule": copy.deepcopy(candidate), "rules": copy.deepcopy(rules), "prompt": prompt}
            policy = {"minRecall": .9, "maxFalsePositiveRate": .1, "minPerClass": 5, "fnCost": 5, "fpCost": 1}
            target = .9
            if payload.runId:
                run = find(state.get("runs", []), payload.runId, "实验")
                if run.get("ruleId") != current["id"] or run.get("gate", {}).get("status") != "PASS":
                    raise HTTPException(409, "只能冻结当前规则开发验收通过的候选")
                if run.get("contextFingerprint") != context_fingerprint(state) or run.get("codeFingerprint") != code_fingerprint():
                    raise HTTPException(409, "实验依据已过期，请重新评测")
                candidate.update(run.get("bestRule") or {})
                candidate = enrich_rule(state, candidate)
                prompt = run.get("bestPrompt") or prompt
                policy.update(run.get("policy") or {})
                target = run.get("target") or .9
                rules = rules_with_candidate(rules, candidate)
            bundle = {"rule": candidate, "rules": rules, "prompt": prompt, "policy": policy, "target": target,
                      "model": model_manifest(state.get("settings", {})), "datasetVersion": state.get("datasetVersion"),
                      "codeVersion": "3.0.0", "sourceFingerprint": context_fingerprint(state)}
            bundle["codeFingerprint"] = code_fingerprint()
            bundle["baselineHash"] = digest(baseline)
            item = {"id": "release-" + uuid.uuid4().hex[:10], "ruleId": current["id"], "ruleName": current["name"],
                    "status": "FROZEN", "author": payload.author.strip(), "createdAt": stamp(), "runId": payload.runId,
                    "bundle": bundle, "baselineBundle": baseline, "bundleHash": digest(bundle),
                    "datasetVersion": state.get("datasetVersion"), "history": [], "deploymentScope": "LOCAL_GATEWAY"}
            item["observations"] = {"shadow": 0, "canary": 0, "published": 0, "unresolved": 0, "errors": 0}
            audit(item, "FREEZE", payload.author.strip(), payload.reason.strip())
            state.setdefault("releases", []).insert(0, item)
            return public_release(item)
        return store.mutate(change)

    async def evaluate_worker(release_id, holdout_id):
        try:
            state = store.snapshot()
            release = find(state["releases"], release_id, "版本")
            holdout = find(state["holdouts"], holdout_id, "独立集")
            bundle = release["bundle"]
            verify_bundle(release)
            if digest(holdout["calls"]) != holdout["fingerprint"]:
                raise ValueError("封存数据完整性校验失败")
            base_bundle = dict(bundle, **release["baselineBundle"])
            baseline = await evaluate_bundle(base_bundle, holdout["calls"])
            candidate = await evaluate_bundle(bundle, holdout["calls"])
            gate = independent_gate(candidate, baseline, bundle["target"], bundle["policy"])
            def finish(draft):
                item = find(draft["releases"], release_id, "版本")
                item.update(status="EVALUATED" if gate["status"] == "PASS" else "EVALUATION_FAILED",
                            evaluation={"baseline": baseline["metrics"], "candidate": candidate["metrics"],
                                        "gate": gate, "holdoutId": holdout_id, "at": stamp(),
                                        "bundleHash": release["bundleHash"], "holdoutHash": holdout["fingerprint"]})
                # Store for audit, do not expose raw sealed evidence through the development UI.
                item["evaluationRows"] = {"baseline": baseline["rows"], "candidate": candidate["rows"]}
                audit(item, item["status"], "独立评测", "冻结模型与能力包在封存数据上完成评测")
            store.mutate(finish)
        except Exception as exc:
            def fail(draft):
                item = find(draft["releases"], release_id, "版本")
                item.update(status="ERROR", error=str(exc)[:1000])
                audit(item, "ERROR", "独立评测", str(exc)[:500])
            store.mutate(fail)

    @app.post("/api/releases/{release_id}/evaluate")
    async def evaluate_release(release_id: str, payload: ReleaseEvaluation):
        from .experiments import code_fingerprint
        ensure_idle()
        def reserve(state):
            release = find(state.get("releases", []), release_id, "版本")
            holdout = find(state.get("holdouts", []), payload.holdoutId, "独立集")
            if release["status"] not in ("FROZEN", "INTERRUPTED", "ERROR"):
                raise HTTPException(409, "候选已评测或状态不允许；调整候选需重新冻结并使用新独立数据")
            if state.get("settings", {}).get("mode") != "deepseek":
                raise HTTPException(409, "正式独立验收需真实模型；模拟演示不能取得发布资格")
            if release["bundle"]["model"] != model_manifest(state.get("settings", {})):
                raise HTTPException(409, "模型配置与冻结时不同，请恢复配置或重新冻结")
            if release["bundle"].get("codeFingerprint") != code_fingerprint():
                raise HTTPException(409, "执行代码与冻结时不同，请重新冻结并验收")
            if holdout.get("usedBy") not in (None, release_id):
                raise HTTPException(409, "该独立集已用于其他候选；请使用新的封存数据")
            holdout["usedBy"] = release_id
            release.update(status="EVALUATING", holdoutId=payload.holdoutId, error=None)
            audit(release, "EVALUATING", "独立评测", "独立集锁定到此冻结版本")
            return public_release(release)
        result = store.mutate(reserve)
        task = asyncio.create_task(evaluate_worker(release_id, payload.holdoutId))
        workers.add(task)
        task.add_done_callback(workers.discard)
        return result

    @app.post("/api/releases/{release_id}/transition")
    async def transition(release_id: str, payload: ReleaseAction):
        from .experiments import code_fingerprint
        ensure_idle()
        check_named(payload)
        def change(state):
            item = find(state.get("releases", []), release_id, "版本")
            transitions = {"APPROVE": ({"EVALUATED"}, "APPROVED"), "REJECT": ({"FROZEN", "EVALUATED", "APPROVED"}, "REJECTED"),
                           "SHADOW": ({"APPROVED"}, "SHADOW"), "CANARY": ({"SHADOW"}, "CANARY"),
                           "PUBLISH": ({"CANARY"}, "PUBLISHED"), "ROLLBACK": ({"SHADOW", "CANARY", "PUBLISHED"}, "ROLLED_BACK")}
            if payload.action not in transitions or item["status"] not in transitions[payload.action][0]:
                raise HTTPException(409, "当前状态不允许该操作，请按冻结、验收、审核、影子、灰度、发布顺序操作")
            if payload.action not in ("REJECT", "ROLLBACK"):
                verify_bundle(item)
                evaluation = item.get("evaluation") or {}
                if evaluation.get("gate", {}).get("status") != "PASS" or evaluation.get("bundleHash") != item["bundleHash"]:
                    raise HTTPException(409, "缺少当前冻结版本的独立验收通过证据")
                if item["bundle"].get("codeFingerprint") != code_fingerprint() or item["bundle"]["model"] != model_manifest(state.get("settings", {})):
                    raise HTTPException(409, "执行代码或模型配置已变化，需重新冻结验收")
            if payload.action == "APPROVE" and payload.reviewer.strip() == item["author"]:
                raise HTTPException(409, "审核人应与提交人不同；本地登记不能代替组织身份认证")
            channel = state.setdefault("releaseChannels", {}).setdefault(item["ruleId"], {})
            observations = item.get("observations", {})
            stage = "shadow" if payload.action == "CANARY" else "canary" if payload.action == "PUBLISH" else None
            if stage and (not observations.get(stage) or observations.get("unresolved") or observations.get("errors")):
                raise HTTPException(409, "需先通过运行时接口完成当前阶段试判且无未解决错误/待复核，再记录人工结论；这些记录不等同于业务效果验证")
            if payload.action == "SHADOW":
                if channel.get("shadow") or channel.get("canary"):
                    raise HTTPException(409, "已有影子/灰度版本，请先完成或回滚")
                channel["shadow"] = release_id
            elif payload.action == "CANARY":
                channel.pop("shadow", None)
                channel["canary"] = {"releaseId": release_id, "percent": payload.percent}
            elif payload.action == "PUBLISH":
                item["previousPublished"] = channel.get("published")
                channel["published"] = release_id
                channel.pop("canary", None)
                channel.pop("shadow", None)
            elif payload.action == "ROLLBACK":
                if channel.get("published") == release_id:
                    channel["published"] = item.get("previousPublished")
                if channel.get("shadow") == release_id:
                    channel.pop("shadow", None)
                if channel.get("canary", {}).get("releaseId") == release_id:
                    channel.pop("canary", None)
            item["status"] = transitions[payload.action][1]
            audit(item, payload.action, payload.reviewer.strip(), payload.reason.strip())
            return public_release(item)
        return store.mutate(change)

    @app.get("/api/releases/{release_id}/manifest")
    async def manifest(release_id: str):
        item = find(store.snapshot().get("releases", []), release_id, "版本")
        return {**public_release(item), "bundle": item["bundle"],
                "integrationNote": "本地运行时通道；外部业务系统需接入 /api/runtime/judge。署名审核未绑定企业身份。"}

    @app.post("/api/runtime/judge")
    async def runtime(payload: RuntimeInput):
        from .experiments import code_fingerprint
        state = store.snapshot()
        channel = state.get("releaseChannels", {}).get(payload.ruleId, {})
        release_id = channel.get("published")
        canary = channel.get("canary") or {}
        bucket = int(hashlib.sha256(payload.callId.encode()).hexdigest()[:8], 16) % 100
        candidate_id = canary.get("releaseId") or channel.get("shadow")
        use_candidate = bool(canary and bucket < canary["percent"])
        chosen_id = candidate_id if use_candidate else release_id
        if chosen_id:
            chosen = find(state.get("releases", []), chosen_id, "运行版本")
            bundle = chosen["bundle"]
        elif candidate_id:
            chosen = find(state.get("releases", []), candidate_id, "运行版本")
            bundle = dict(chosen["bundle"], **chosen["baselineBundle"])
        else:
            raise HTTPException(409, "当前规则没有发布或影子/灰度版本")
        def record(rid, stage, verdict):
            def change(draft):
                item = find(draft["releases"], rid, "运行版本")
                stats = item.setdefault("observations", {})
                stats[stage] = stats.get(stage, 0) + 1
                stats["unresolved"] = stats.get("unresolved", 0) + int(verdict.get("predicted") is None)
                stats["errors"] = stats.get("errors", 0) + int(bool(verdict.get("error")))
                item["lastObservedAt"] = stamp()
            store.mutate(change)

        async def execute(item, execution_bundle, stage, shadow_only=False):
            try:
                verify_bundle(item)
                if item["bundle"].get("codeFingerprint") != code_fingerprint():
                    raise HTTPException(409, "运行版本的执行代码已变化，需重新验收")
                verdict = await runtime_judge(execution_bundle, payload.dict())
            except Exception as exc:
                # Failed attempts remain visible to promotion gates. Shadow failure must
                # not erase a successful primary verdict or masquerade as a normal call.
                verdict = {"predicted": None, "reviewRequired": True, "error": "运行判定失败",
                           "reason": "本次未完成有效判定，需排查并重新验收能力版本"}
                record(item["id"], stage, verdict)
                if not shadow_only:
                    if isinstance(exc, HTTPException):
                        raise
                    raise HTTPException(502, "运行判定失败，已记录错误并阻止版本晋级") from exc
                return verdict
            record(item["id"], stage, verdict)
            return verdict

        stage = "canary" if use_candidate else "published" if chosen_id else "baseline"
        result = await execute(chosen, bundle, stage)
        shadow_result = None
        if channel.get("shadow"):
            shadow = find(state.get("releases", []), channel["shadow"], "影子版本")
            shadow_result = await execute(shadow, shadow["bundle"], "shadow", shadow_only=True)
        return {"releaseId": chosen_id, "baselineOf": None if chosen_id else candidate_id,
                "callId": payload.callId, "result": result, "shadowResult": shadow_result,
                "shadowReleaseId": channel.get("shadow"), "bucket": bucket}

    @app.post("/api/feedback")
    async def feedback(payload: FeedbackInput):
        check_named(payload)
        if payload.label not in (NORMAL, VIOLATION, "UNKNOWN"):
            raise HTTPException(400, "反馈标签无效")
        def change(state):
            release = find(state.get("releases", []), payload.releaseId, "版本")
            if not any(h.get("action") in ("SHADOW", "CANARY", "PUBLISH") for h in release.get("history", [])):
                raise HTTPException(409, "反馈必须关联曾进入运行阶段的能力版本")
            for ev in payload.evidence:
                if not ev.get("quote") or ev["quote"] not in payload.transcript:
                    raise HTTPException(400, "反馈证据必须引用通话原文")
            item = dict(payload.dict(), id="feedback-" + uuid.uuid4().hex[:10], status="PENDING",
                        ruleId=release["ruleId"], bundleHash=release["bundleHash"], createdAt=stamp())
            state.setdefault("feedback", []).append(item)
            return item
        return store.mutate(change)

    @app.post("/api/feedback/{feedback_id}/accept")
    async def accept_feedback(feedback_id: str, payload: NamedAction):
        ensure_idle()
        check_named(payload)
        def change(state):
            item = find(state.get("feedback", []), feedback_id, "反馈")
            if item["status"] != "PENDING" or item["label"] == "UNKNOWN":
                raise HTTPException(409, "只能将已明确人工标签的待复核反馈纳入训练数据")
            if payload.reviewer.strip() == item["reviewer"].strip():
                raise HTTPException(409, "反馈复核人应与提交人不同")
            sealed = [c for h in state.get("holdouts", []) for c in h["calls"]]
            assert_no_holdout_overlap([item], sealed)
            if any(c["callId"] == item["callId"] for c in state.get("calls", [])):
                raise HTTPException(409, "同通话已存在，请通过标注复核入口修改")
            if len(state.get("calls", [])) >= 2000:
                raise HTTPException(409, "当前本地数据集达到2000通上限，请归档后再纳入反馈")
            rule = find(state["rules"], item["ruleId"], "规则")
            call = {"id": "call-" + uuid.uuid4().hex[:12], "callId": item["callId"], "transcript": item["transcript"],
                    "groupId": item.get("groupId", ""), "transcriptComplete": item.get("transcriptComplete", False),
                    "autoRules": [], "humanRules": [], "importedAt": stamp(),
                    "annotations": {rule["id"]: {"label": item["label"], "reviewStatus": "REVIEWED",
                        "reviewer": payload.reviewer.strip(), "reason": payload.reason.strip(), "source": "feedback",
                        "evidence": item.get("evidence", []), "releaseId": item["releaseId"]}}}
            state.setdefault("calls", []).append(call)
            item.update(status="ACCEPTED", acceptedBy=payload.reviewer.strip(), acceptedAt=stamp())
            refresh_dataset(state, "人工复核反馈入库：" + payload.reason)
            return {"ok": True, "datasetVersion": state["datasetVersion"]}
        return store.mutate(change)

