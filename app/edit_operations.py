"""结构化编辑操作：让模型提交局部变更，而不是整条规则重写。"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from .regex_policy import _split_alternatives

MAX_OPERATIONS = 3
OPERATION_NAMES = {
    "ADD_REGEX_ANCHOR",
    "REMOVE_REGEX_BRANCH",
    "REPLACE_REGEX_BRANCH",
    "ADD_SEMANTIC_CONDITION",
    "ADD_SEMANTIC_EXCLUSION",
    "ADD_NEGATIVE_CONTEXT",
    "SET_LOGIC",
    "REPLACE_SEMANTIC",
    "NO_CHANGE",
}


class OperationError(ValueError):
    pass


def _clean_text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _clean_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:10]


def _append_semantic(semantic: str, label: str, text: str) -> str:
    text = _clean_text(text, 400)
    if not text:
        raise OperationError(f"{label}不能为空")
    current = str(semantic or "").strip()
    if text in current:
        return current
    result = (current + "\n" + label + text).strip() if current else label + text
    return result[:1000]


def _operation(op: Dict[str, Any]) -> Dict[str, Any]:
    name = str(op.get("op") or op.get("operation") or "").strip().upper()
    if name not in OPERATION_NAMES:
        raise OperationError(f"不支持的编辑操作：{name or '空'}")
    return {
        "op": name,
        "pattern": _clean_text(op.get("pattern"), 300),
        "old": _clean_text(op.get("old"), 300),
        "new": _clean_text(op.get("new"), 300),
        "text": _clean_text(op.get("text") or op.get("condition"), 400),
        "value": _clean_text(op.get("value"), 20).upper(),
        "targetCaseIds": _clean_ids(op.get("targetCaseIds") or op.get("caseIds")),
        "reason": _clean_text(op.get("reason") or op.get("rationale"), 300),
    }


def apply_operations(rule: Dict[str, Any], operations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(operations, list):
        raise OperationError("operations 必须是数组")
    if len(operations) > MAX_OPERATIONS:
        raise OperationError(f"单轮最多 {MAX_OPERATIONS} 个编辑操作")
    result = copy.deepcopy(rule)
    applied: List[Dict[str, Any]] = []
    for raw in operations:
        if not isinstance(raw, dict):
            raise OperationError("每个 edit operation 必须是对象")
        op = _operation(raw)
        name = op["op"]
        before = {"regex": result.get("regex", ""), "semantic": result.get("semantic", ""),
                  "logic": result.get("logic", "OR")}
        if name == "NO_CHANGE":
            pass
        elif name == "ADD_REGEX_ANCHOR":
            pattern = op["pattern"]
            if not pattern:
                raise OperationError("ADD_REGEX_ANCHOR 缺少 pattern")
            current = str(result.get("regex") or "").strip()
            if not current:
                result["regex"] = pattern
            elif pattern not in _split_alternatives(current):
                result["regex"] = f"(?:{current}|(?:{pattern}))"
        elif name == "REMOVE_REGEX_BRANCH":
            branch = op["pattern"] or op["old"]
            if not branch:
                raise OperationError("REMOVE_REGEX_BRANCH 缺少 pattern")
            current = str(result.get("regex") or "")
            parts = _split_alternatives(current)
            if branch not in parts:
                raise OperationError(f"当前正则不存在分支：{branch}")
            parts = [part for part in parts if part != branch]
            result["regex"] = "|".join(parts)
            if not result["regex"] and not str(result.get("semantic") or "").strip():
                raise OperationError("删除后正则和语义都为空")
        elif name == "REPLACE_REGEX_BRANCH":
            old, new = op["old"], op["new"]
            if not old or not new:
                raise OperationError("REPLACE_REGEX_BRANCH 需要 old 和 new")
            parts = _split_alternatives(str(result.get("regex") or ""))
            if old not in parts:
                raise OperationError(f"当前正则不存在分支：{old}")
            result["regex"] = "|".join(new if part == old else part for part in parts)
        elif name == "ADD_SEMANTIC_CONDITION":
            result["semantic"] = _append_semantic(str(result.get("semantic") or ""), "判定条件：", op["text"])
        elif name in ("ADD_SEMANTIC_EXCLUSION", "ADD_NEGATIVE_CONTEXT"):
            result["semantic"] = _append_semantic(str(result.get("semantic") or ""), "排除：", op["text"])
        elif name == "REPLACE_SEMANTIC":
            result["semantic"] = _clean_text(op["text"], 1000)
        elif name == "SET_LOGIC":
            if op["value"] not in ("OR", "AND"):
                raise OperationError("SET_LOGIC 的 value 只能是 OR 或 AND")
            result["logic"] = op["value"]
        applied.append({**op, "before": before,
                        "after": {"regex": result.get("regex", ""), "semantic": result.get("semantic", ""),
                                  "logic": result.get("logic", "OR")}})
    return result, applied
