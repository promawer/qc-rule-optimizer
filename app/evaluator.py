"""规则评测：正则 + 语义按 OR/AND 组合，逐条对照人工标签计算指标。"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .models import NORMAL, VIOLATION, rule_flags
from .capabilities import regex_evidence, role_context_valid
from .regex_safety import RegexTimeout

Judge = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]]


class EvaluationError(Exception):
    def __init__(self, message: str, details: Optional[list] = None) -> None:
        super().__init__(message)
        self.details = details or []


class JobCancelled(Exception):
    pass


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """以已知人工标签为固定分母；待复核及失败判定不能虚增准确率。"""
    labeled = [r for r in rows if r.get("humanResult") in (VIOLATION, NORMAL)]
    evaluated = [r for r in labeled if r.get("predicted") is not None]
    tp = sum(1 for r in evaluated if r["predicted"] and r["humanResult"] == VIOLATION)
    fp = sum(1 for r in evaluated if r["predicted"] and r["humanResult"] == NORMAL)
    tn = sum(1 for r in evaluated if not r["predicted"] and r["humanResult"] == NORMAL)
    fn = sum(1 for r in evaluated if not r["predicted"] and r["humanResult"] == VIOLATION)
    positives = sum(r["humanResult"] == VIOLATION for r in labeled)
    negatives = len(labeled) - positives
    unresolved_positives = positives - tp - fn
    unresolved_normals = negatives - tn - fp
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auto_rows = [r for r in labeled if r.get("autoResult") in (VIOLATION, NORMAL)]
    auto_correct = sum(r["autoResult"] == r["humanResult"] for r in auto_rows)
    step_pairs = [(expected, (row.get("steps") or {}).get(key)) for row in labeled
                  for key, expected in (row.get("humanSteps") or {}).items() if isinstance(expected, bool)]
    gold_evidence = [(row, ev) for row in labeled for ev in (row.get("humanEvidence") or [])
                     if isinstance(ev, dict) and ev.get("quote")]
    matched_evidence = sum(any(
        expected["quote"] in observed.get("quote", "")
        and (not expected.get("turnId") or expected["turnId"] == observed.get("turnId"))
        and (not expected.get("role") or expected["role"] == observed.get("role"))
        for observed in row.get("evidence", [])) for row, expected in gold_evidence)
    return {
        "total": len(labeled), "inputTotal": len(rows), "evaluated": len(evaluated),
        "excludedUnknown": len(rows) - len(labeled),
        "unresolved": len(labeled) - len(evaluated),
        "unresolvedPositives": unresolved_positives, "unresolvedNormals": unresolved_normals,
        "failed": sum(bool(r.get("error")) for r in labeled),
        "correct": correct,
        "accuracy": round(correct / len(labeled), 4) if labeled else 0.0,
        "validAccuracy": round(correct / len(evaluated), 4) if evaluated else 0.0,
        "coverage": round(len(evaluated) / len(labeled), 4) if labeled else 0.0,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "falsePositiveRate": round(fp / negatives, 4) if negatives else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "autoTotal": len(auto_rows),
        "autoAccuracy": round(auto_correct / len(auto_rows), 4) if auto_rows else None,
        "violations": positives, "normals": negatives,
        "stepTotal": len(step_pairs),
        "stepAccuracy": round(sum(a is b for a, b in step_pairs) / len(step_pairs), 4) if step_pairs else None,
        "goldEvidenceTotal": len(gold_evidence),
        "goldEvidenceRecall": round(matched_evidence / len(gold_evidence), 4) if gold_evidence else None,
    }


def _judgment(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        result = copy.deepcopy(value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = {"violation": value[0], "reason": str(value[1] or "")}
    else:
        raise ValueError("判定结果必须为结构化对象或二元判定")
    if result.get("violation") is not None and not isinstance(result["violation"], bool):
        raise ValueError("violation 必须是布尔值或未知 null")
    result.setdefault("evidence", [])
    result.setdefault("steps", {})
    result.setdefault("evidenceValid", True)
    result.setdefault("reviewRequired", result.get("violation") is None)
    return result


async def evaluate_rule(
    rule: Dict[str, Any],
    cases: List[Dict[str, Any]],
    judge: Judge,
    *,
    cache: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    force_parts: bool = False,
    with_reason: bool = True,
    cache_scope: str = "",
    allow_unlabeled: bool = False,
) -> Dict[str, Any]:
    flags = rule_flags(rule)
    if not any(flags.values()):
        raise EvaluationError("规则内容为空：正则和语义至少需要一项")
    if not cases:
        raise EvaluationError("当前规则没有可评测的数据，请先导入通话数据")

    pattern: Optional[re.Pattern[str]] = None
    if flags["regex"]:
        try:
            pattern = re.compile(str(rule.get("regex") or ""))
        except re.error as exc:
            raise EvaluationError(f"正则表达式无法编译：{exc}") from exc

    logic = str(rule.get("logic") or "OR").upper()
    workflow = rule.get("workflow", "SIMPLE")
    cache = cache if cache is not None else {}
    total = len(cases)
    progress_state = {"done": 0, "semantic": 0, "errors": 0}
    error_samples: List[Dict[str, str]] = []
    started = time.time()

    async def one(case: Dict[str, Any]) -> Dict[str, Any]:
        if is_cancelled and is_cancelled():
            raise JobCancelled()
        transcript = str(case.get("transcript") or "")
        human = case.get("humanResult") if case.get("reviewStatus", "REVIEWED") == "REVIEWED" else "UNKNOWN"
        if human not in (VIOLATION, NORMAL) and not allow_unlabeled:
            progress_state["done"] += 1
            if progress:
                progress(progress_state["done"], total)
            return {**case, "humanResult": human, "transcript": transcript, "predicted": None, "correct": None,
                    "excluded": True, "reviewRequired": True, "error": "",
                    "regexHit": None, "semanticHit": None,
                    "reason": "人工标签未知或未经复核，不参与计分和候选生成"}
        role_valid = role_context_valid(rule, transcript)
        try:
            evidence = regex_evidence(rule, transcript) if pattern is not None else None
        except RegexTimeout as exc:
            raise EvaluationError(str(exc), [{"callId": case.get("callId"), "error": str(exc)}]) from exc
        regex_hit = bool(evidence) if pattern is not None else None
        if pattern is None:
            regex_reason = "未配置正则"
        elif evidence:
            regex_reason = f"正则命中「{evidence['text'][:60]}」"
        else:
            regex_reason = "正则未命中"
        if flags["semantic"]:
            need_semantic = workflow == "COMPLAINT_RESPONSE" or force_parts or not flags["regex"] or (
                (logic == "OR" and not regex_hit) or (logic == "AND" and bool(regex_hit))
            )
        else:
            need_semantic = False
        semantic_hit: Optional[bool] = None
        reason = ""
        error = ""
        judgment = {}
        if need_semantic and role_valid:
            reason_scope = "reason" if with_reason else "verdict"
            key = hashlib.sha1(
                (cache_scope + "\x00" + json.dumps(
                    rule,
                    ensure_ascii=False, sort_keys=True,
                ) + "\x00" + transcript + "\x00" + json.dumps(
                    {k: case.get(k) for k in ("transcriptComplete", "transcriptTruncated", "truncated")}, sort_keys=True
                ) + "\x00" + reason_scope).encode("utf-8")
            ).hexdigest()
            cached = cache.get(key)
            if cached is not None:
                judgment = _judgment(cached)
            else:
                try:
                    judgment = _judgment(await judge(case, rule))
                    cache[key] = copy.deepcopy(judgment)
                except JobCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001
                    progress_state["errors"] += 1
                    error = str(exc)[:500]
                    error_samples.append({"callId": str(case.get("callId") or ""), "error": error})
                    semantic_hit = None
            semantic_hit = judgment.get("violation")
            reason = str(judgment.get("reason") or "")
            progress_state["semantic"] += 1
        if not role_valid:
            reason = "无法确认规则要求的说话角色或话轮，转人工复核"
            judgment = {"reviewRequired": True, "evidenceValid": False, "evidenceErrors": [reason]}
        semantic_reason = reason if with_reason else ""

        if not role_valid or error or (need_semantic and (semantic_hit is None or judgment.get("reviewRequired")
                                       or judgment.get("evidenceValid") is False)):
            predicted: Optional[bool] = None
        elif workflow == "COMPLAINT_RESPONSE":
            predicted = semantic_hit
        elif flags["regex"] and flags["semantic"]:
            predicted = (bool(regex_hit) or bool(semantic_hit)) if logic == "OR" else (bool(regex_hit) and bool(semantic_hit))
        elif flags["regex"]:
            predicted = bool(regex_hit)
        else:
            predicted = bool(semantic_hit)

        correct: Optional[bool] = None
        if predicted is not None and human in (VIOLATION, NORMAL):
            correct = predicted == (human == VIOLATION)
        progress_state["done"] += 1
        if progress:
            progress(progress_state["done"], total)
        return {
            "id": case.get("id"),
            "callId": case.get("callId"),
            "split": case.get("split", "TRAIN"),
            "transcript": transcript,
            "autoResult": case.get("autoResult"),
            "humanResult": human,
            "regexHit": regex_hit,
            "regexReason": regex_reason,
            "regexEvidence": {**evidence, "text": evidence["text"][:120]} if evidence else None,
            "semanticHit": semantic_hit,
            "semanticReason": semantic_reason,
            "predicted": predicted,
            "correct": correct,
            "reason": semantic_reason,
            "error": error,
            "evidence": judgment.get("evidence", []), "steps": judgment.get("steps", {}),
            "evidenceValid": judgment.get("evidenceValid", True),
            "evidenceErrors": judgment.get("evidenceErrors", []),
            "responseWindow": judgment.get("responseWindow"),
            "humanEvidence": case.get("humanEvidence", []), "humanSteps": case.get("humanSteps", {}),
            "reviewRequired": predicted is None,
        }

    rows = []
    for case in cases:
        rows.append(await one(case))
    rows = list(rows)
    needed = progress_state["semantic"]
    errors = progress_state["errors"]
    if needed and errors / max(needed, 1) > 0.1:
        details = error_samples[:10]
        sample_text = "\n".join(
            f"- {item['callId']}：{item['error'].replace(chr(10), ' ')[:300]}"
            for item in details[:5]
        )
        hint = "请检查模型配置、网络、超时时间、重试次数和返回字段路径。"
        lowered = " ".join(item["error"].lower() for item in details)
        if "超时" in lowered:
            hint = "多数是调用超时：可以在「模型设置」中增大超时时间或重试次数。"
        elif "http 401" in lowered or "http 403" in lowered:
            hint = "鉴权失败：请检查 API Key 或自定义鉴权表头。"
        elif "不是 json" in lowered or "不是有效 json" in lowered:
            hint = "模型返回格式不对：请检查返回字段路径，以及模型是否按要求输出 JSON。"
        elif "找不到字段" in lowered:
            hint = "返回路径配置不对：请对照接口实际返回填写「模型文本内容」路径。"
        elif "connection" in lowered or "connect" in lowered:
            hint = "无法连接模型接口：请检查 Base URL、接口路径、端口和网络是否可达。"
        message = (
            f"语义判定失败 {errors}/{needed} 条，已停止评测。{hint}"
            + (f"\n错误示例：\n{sample_text}" if sample_text else "")
        )
        raise EvaluationError(message, details=details)
    metrics = _metrics(rows)
    return {
        "rule": copy.deepcopy(rule),
        "metrics": metrics,
        "rows": rows,
        "errors": errors,
        "coverage": metrics["coverage"],
        "excludedUnknown": metrics["excludedUnknown"],
        "complete": metrics["coverage"] == 1 and metrics["total"] > 0,
        "durationMs": int((time.time() - started) * 1000),
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
