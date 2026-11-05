"""规则评测：正则 + 语义按 OR/AND 组合，逐条对照人工标签计算指标。"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .models import NORMAL, VIOLATION, rule_flags

Judge = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Tuple[bool, str]]]


class EvaluationError(Exception):
    pass


class JobCancelled(Exception):
    pass


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [
        row for row in rows
        if row["predicted"] is not None and row["humanResult"] in (VIOLATION, NORMAL)
    ]
    tp = sum(1 for r in evaluated if r["predicted"] and r["humanResult"] == VIOLATION)
    fp = sum(1 for r in evaluated if r["predicted"] and r["humanResult"] == NORMAL)
    tn = sum(1 for r in evaluated if not r["predicted"] and r["humanResult"] == NORMAL)
    fn = sum(1 for r in evaluated if not r["predicted"] and r["humanResult"] == VIOLATION)
    total = len(evaluated)
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auto_rows = [r for r in rows if r["autoResult"] in (VIOLATION, NORMAL) and r["humanResult"] in (VIOLATION, NORMAL)]
    auto_correct = sum(1 for r in auto_rows if r["autoResult"] == r["humanResult"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "autoTotal": len(auto_rows),
        "autoAccuracy": round(auto_correct / len(auto_rows), 4) if auto_rows else None,
        "violations": sum(1 for r in rows if r["humanResult"] == VIOLATION),
        "normals": sum(1 for r in rows if r["humanResult"] == NORMAL),
    }


async def evaluate_rule(
    rule: Dict[str, Any],
    cases: List[Dict[str, Any]],
    judge: Judge,
    *,
    cache: Optional[Dict[str, Tuple[bool, str]]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    force_parts: bool = False,
    with_reason: bool = True,
    cache_scope: str = "",
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
    semantic_text = str(rule.get("semantic") or "").strip()
    cache = cache if cache is not None else {}
    total = len(cases)
    progress_state = {"done": 0, "semantic": 0, "errors": 0}
    started = time.time()

    async def one(case: Dict[str, Any]) -> Dict[str, Any]:
        if is_cancelled and is_cancelled():
            raise JobCancelled()
        transcript = str(case.get("transcript") or "")
        match = pattern.search(transcript) if pattern is not None else None
        regex_hit = bool(match) if pattern is not None else None
        if pattern is None:
            regex_reason = "未配置正则"
        elif match:
            regex_reason = f"正则命中「{match.group(0)[:60]}」"
        else:
            regex_reason = "正则未命中"
        if flags["semantic"]:
            need_semantic = force_parts or (
                (logic == "OR" and not regex_hit) or (logic == "AND" and bool(regex_hit))
            )
        else:
            need_semantic = False
        semantic_hit: Optional[bool] = None
        reason = ""
        error = ""
        if need_semantic:
            reason_scope = "reason" if with_reason else "verdict"
            key = hashlib.sha1(
                (cache_scope + "\x00" + semantic_text + "\x00" + transcript + "\x00" + reason_scope).encode("utf-8")
            ).hexdigest()
            cached = cache.get(key)
            if cached is not None:
                semantic_hit, reason = cached
            else:
                try:
                    semantic_hit, reason = await judge(case, rule)
                    cache[key] = (semantic_hit, reason)
                except JobCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001
                    progress_state["errors"] += 1
                    error = str(exc)[:300]
                    semantic_hit = None
            progress_state["semantic"] += 1
        semantic_reason = reason if with_reason else ""

        if error:
            predicted: Optional[bool] = None
        elif flags["regex"] and flags["semantic"]:
            predicted = (bool(regex_hit) or bool(semantic_hit)) if logic == "OR" else (bool(regex_hit) and bool(semantic_hit))
        elif flags["regex"]:
            predicted = bool(regex_hit)
        else:
            predicted = bool(semantic_hit)

        human = case.get("humanResult")
        correct: Optional[bool] = None
        if predicted is not None and human in (VIOLATION, NORMAL):
            correct = predicted == (human == VIOLATION)
        progress_state["done"] += 1
        if progress:
            progress(progress_state["done"], total)
        return {
            "id": case.get("id"),
            "callId": case.get("callId"),
            "transcript": transcript,
            "autoResult": case.get("autoResult"),
            "humanResult": human,
            "regexHit": regex_hit,
            "regexReason": regex_reason,
            "semanticHit": semantic_hit,
            "semanticReason": semantic_reason,
            "predicted": predicted,
            "correct": correct,
            "reason": semantic_reason,
            "error": error,
        }

    rows = await asyncio.gather(*(one(case) for case in cases))
    rows = list(rows)
    needed = progress_state["semantic"]
    errors = progress_state["errors"]
    if needed and errors / max(needed, 1) > 0.1:
        raise EvaluationError(
            f"语义判定失败 {errors}/{needed} 条，已停止评测。请检查模型配置或网络后重试。"
        )
    metrics = _metrics(rows)
    if metrics["total"] == 0:
        raise EvaluationError("没有可计分的样本：请确认人工复核结果包含违规和正常两类")
    return {
        "rule": {
            "regex": rule.get("regex") or "",
            "semantic": semantic_text,
            "logic": logic,
        },
        "metrics": metrics,
        "rows": rows,
        "errors": errors,
        "coverage": round(metrics["total"] / len(cases), 4),
        "durationMs": int((time.time() - started) * 1000),
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
