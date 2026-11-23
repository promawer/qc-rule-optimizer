"""共享执行输入变化的跨规则验证；测试集不参与候选选择。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .evaluator import EvaluationError, JobCancelled, evaluate_rule
from .importer import cases_for_rule
from .llm import LLMError
from .models import NORMAL, VIOLATION
from .prompts import DEFAULT_SEMANTIC_PROMPT
from .splits import VALIDATION

# 所有可渲染能力字段均进入快照，避免新增知识/流程字段绕过过期校验。
CAPABILITY_FIELDS = ("code", "name", "description", "semantic", "actor", "workflow",
                     "responseWindowTurns", "responseRequirement", "knowledgeIds", "_approvedKnowledge")


def needs_regression(original, candidate, original_prompt, candidate_prompt):
    """共享提示词或注入规则能力变化时，必须验证其他规则。"""
    if original_prompt != candidate_prompt:
        return True
    if not (str(original.get("semantic") or "").strip() or str(candidate.get("semantic") or "").strip()):
        return False
    return any(original.get(key) != candidate.get(key) for key in CAPABILITY_FIELDS)



def context_fingerprint(state: Dict[str, Any]) -> str:
    """不包含运行记录；规则、数据、切分、模型配置或提示词变化都会使验收过期。"""
    payload = {
        "rules": state.get("rules", []),
        "knowledge": state.get("knowledge", []), "datasetVersion": state.get("datasetVersion"),
        "calls": state.get("calls", []), "splits": state.get("splits", {}),
        "settings": state.get("settings", {}),
        "semanticPrompt": state.get("semanticPrompt") or DEFAULT_SEMANTIC_PROMPT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def rules_with_candidate(rules: List[Dict[str, Any]], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(rule, **candidate) if (
        (candidate.get("id") and rule.get("id") == candidate["id"])
        or (candidate.get("code") and rule.get("code") == candidate["code"])
    ) else dict(rule) for rule in rules]


def make_prompt_guard(state, optimized_rule, semantic_judge, emit, is_cancelled):
    """每个候选与任务开始时的全局基线比较，防止多轮累积退步。"""
    original_rules = state.get("rules") or []
    original_prompt = state.get("semanticPrompt") or DEFAULT_SEMANTIC_PROMPT
    fingerprint = context_fingerprint(state)
    cache = {}
    others = [r for r in original_rules if str(r.get("semantic") or "").strip()
              and r.get("id") != optimized_rule.get("id")]

    async def guard(candidate, prompt):
        report = {"status": "PASS", "contextFingerprint": fingerprint,
                  "checkedRules": 0, "checks": [], "reasons": []}
        datasets = []
        for other in others:
            cases = [c for c in cases_for_rule(state.get("calls") or [], other, state.get("splits") or {})
                     if c.get("split") == VALIDATION and c.get("humanResult") in (NORMAL, VIOLATION)]
            labels = {c.get("humanResult") for c in cases}
            if not {NORMAL, VIOLATION}.issubset(labels):
                report["checks"].append({"code": other.get("code"), "pass": False,
                                         "reason": "验证集缺少违规或正常样本", "total": len(cases)})
            datasets.append((other, cases))
        if report["checks"]:
            report["status"] = "INSUFFICIENT"
            report["reasons"] = [f"{c['code']}：{c['reason']}" for c in report["checks"]]
            return report
        candidate_rules = rules_with_candidate(original_rules, candidate)

        async def evaluate(other, cases, semantic_prompt, all_rules):
            async def judge(case, rule):
                return await semantic_judge(case, rule, semantic_prompt, all_rules)
            scope = json.dumps(all_rules, sort_keys=True, ensure_ascii=False) + "\x00" + semantic_prompt
            result = await evaluate_rule(
                other, cases, judge, force_parts=True, with_reason=False, cache=cache,
                cache_scope=scope, is_cancelled=is_cancelled,
                progress=lambda done, total: emit({"type": "progress", "done": done, "total": total}),
            )
            if result["errors"] or result["coverage"] < 1:
                raise EvaluationError("跨规则回归必须完成全部样本，不能忽略调用失败")
            return result

        for other, cases in datasets:
            if is_cancelled():
                raise JobCancelled()
            emit({"type": "phase", "phase": f"共享能力回归：{other.get('code', '')}（验证集）",
                  "done": 0, "total": len(cases)})
            check = {"code": other.get("code"), "total": len(cases)}
            try:
                base = await evaluate(other, cases, original_prompt, original_rules)
                updated = await evaluate(other, cases, prompt, candidate_rules)
                baseline_correct = {r["callId"] for r in base["rows"] if r["correct"]}
                lost = [r["callId"] for r in updated["rows"] if r["callId"] in baseline_correct and not r["correct"]]
                passed = (updated["metrics"]["accuracy"] >= base["metrics"]["accuracy"]
                          and updated["metrics"]["recall"] >= base["metrics"]["recall"] and not lost)
                check.update({"pass": passed, "baseline": base["metrics"], "candidate": updated["metrics"],
                              "lostCorrectCount": len(lost),
                              "reason": "准确率/召回率未下降，原正确样本未退步" if passed else "出现跨规则判定退步"})
            except (EvaluationError, LLMError) as exc:
                check.update({"pass": False, "reason": f"回归评测未完成：{exc}"[:500]})
            report["checkedRules"] += 1
            report["checks"].append(check)
            if not check["pass"]:
                report["status"] = "BLOCKED"
                report["reasons"].append(f"{check['code']}：{check['reason']}")
        if not others:
            report["reasons"] = ["没有其他启用语义的规则，无需跨规则评测"]
        return report

    return guard
