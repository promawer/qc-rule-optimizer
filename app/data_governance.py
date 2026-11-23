"""稳定规则标注、数据版本及旧数据兼容迁移的纯数据操作。"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any, Dict

from .splits import SPLIT_POLICY, build_splits, group_calls

LEGACY_WARNING = (
    "旧数据沿用原系统“未列出即正常”的全量复核假设，已标记 legacy-assumed-full；"
    "这不代表已重新人工复核。正式验收前请核对并补充复核范围和证据。新增规则不会继承该假设。"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rule_identity(rule: Dict[str, Any]) -> str:
    return str(rule.get("id") or rule.get("code") or "")


def normalize_annotation(annotation: Any, source: str = "explicit") -> Dict[str, Any]:
    raw = annotation if isinstance(annotation, dict) else {}
    label = str(raw.get("label") or "UNKNOWN").upper()
    status = str(raw.get("reviewStatus") or ("REVIEWED" if label in ("VIOLATION", "NORMAL") else "UNREVIEWED")).upper()
    if label not in ("VIOLATION", "NORMAL") or status != "REVIEWED":
        label, status = "UNKNOWN", "UNREVIEWED"
    result = {
        "label": label,
        "reviewStatus": status,
        "reviewer": str(raw.get("reviewer") or "")[:200],
        "reason": str(raw.get("reason") or "")[:2000],
        "source": str(raw.get("source") or source),
    }
    for key in ("evidence", "reviewedAt", "ruleCode", "stages", "steps"):
        if key in raw:
            result[key] = copy.deepcopy(raw[key])
    return result


def ensure_annotations(state: Dict[str, Any], legacy: bool = False) -> None:
    """只在显式旧版本迁移时接受历史全量假设，新规则和新数据默认未知。"""
    rules = state.get("rules") or []
    known_codes = {str(rule.get("code")) for rule in rules}
    legacy_count = 0
    for call in state.get("calls") or []:
        has_annotations = isinstance(call.get("annotations"), dict)
        annotations = call.get("annotations") if has_annotations else {}
        human_rules = list(call.get("humanRules") or [])
        for rule in rules:
            identity = rule_identity(rule)
            if identity in annotations:
                annotations[identity] = normalize_annotation(annotations[identity])
            elif legacy and not has_annotations:
                annotations[identity] = normalize_annotation({
                    "label": "VIOLATION" if rule.get("code") in human_rules else "NORMAL",
                    "reviewStatus": "REVIEWED", "source": "legacy-assumed-full",
                    "reason": "沿用旧系统全量复核假设，尚未重新人工核对",
                })
                legacy_count += 1
            else:
                annotations[identity] = normalize_annotation({}, "unreviewed")
            annotations[identity]["ruleCode"] = str(rule.get("code") or "")
        call["annotations"] = annotations
        call.setdefault("annotationSchema", 1)
        call.setdefault("unresolvedHumanRules", [code for code in human_rules if code not in known_codes])
        call.setdefault("autoRuleIds", [rule_identity(rule) for rule in rules
                                       if rule.get("code") in (call.get("autoRules") or [])])
        call.setdefault("unresolvedAutoRules", [code for code in call.get("autoRules") or [] if code not in known_codes])
    if legacy_count:
        warnings = state.setdefault("dataWarnings", [])
        if not any(item.get("code") == "LEGACY_FULL_REVIEW_ASSUMPTION" for item in warnings if isinstance(item, dict)):
            warnings.append({"code": "LEGACY_FULL_REVIEW_ASSUMPTION", "message": LEGACY_WARNING,
                             "createdAt": now_iso(), "annotationCount": legacy_count})
    sync_label_codes(state)


def sync_label_codes(state: Dict[str, Any]) -> None:
    """由稳定 ruleId 投影显示编号；保留未绑定编号，不自动绑定后来创建的规则。"""
    rules = state.get("rules") or []
    known_codes = {str(rule.get("code")) for rule in rules}
    for call in state.get("calls") or []:
        annotations = call.get("annotations")
        if not isinstance(annotations, dict):
            continue
        human, reviewed = [], []
        for rule in rules:
            annotation = annotations.get(rule_identity(rule)) or {}
            annotation["ruleCode"] = str(rule.get("code") or "")
            if annotation.get("reviewStatus") != "REVIEWED":
                continue
            if annotation.get("label") in ("NORMAL", "VIOLATION"):
                reviewed.append(rule["code"])
            if annotation.get("label") == "VIOLATION":
                human.append(rule["code"])
        call["humanRules"] = human + [code for code in call.get("unresolvedHumanRules") or [] if code not in known_codes]
        call["reviewedRules"] = reviewed
        if "autoRuleIds" in call:
            auto_ids = set(call.get("autoRuleIds") or [])
            call["autoRules"] = [rule["code"] for rule in rules if rule_identity(rule) in auto_ids]
            call["autoRules"] += [code for code in call.get("unresolvedAutoRules") or [] if code not in known_codes]


def refresh_dataset(state: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """创建新数据版本。历史快照是深拷贝，不包含 settings、请求头或密钥。"""
    ensure_annotations(state)
    state["splits"] = build_splits(state.get("rules") or [], state.get("calls") or [])
    state["datasetVersion"] = int(state.get("datasetVersion") or 0) + 1
    state["splitPolicy"] = SPLIT_POLICY
    identities = [{"id": rule_identity(rule), "code": rule.get("code"), "name": rule.get("name")}
                  for rule in state.get("rules") or []]
    snapshot = {
        "version": state["datasetVersion"], "createdAt": now_iso(), "reason": str(reason)[:1000],
        "splitPolicy": SPLIT_POLICY, "rules": identities,
        "calls": copy.deepcopy(state.get("calls") or []), "splits": copy.deepcopy(state["splits"]),
        "warnings": copy.deepcopy(state.get("dataWarnings") or []),
    }
    payload = {key: snapshot[key] for key in ("rules", "calls", "splits", "splitPolicy")}
    snapshot["fingerprint"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    snapshot["callCount"] = len(snapshot["calls"])
    snapshot["groupCount"] = len(group_calls(snapshot["calls"]))
    snapshot["reviewedCount"] = sum(1 for call in snapshot["calls"] for annotation in call["annotations"].values()
                                   if isinstance(annotation, dict) and annotation.get("reviewStatus") == "REVIEWED")
    state.setdefault("datasetVersions", []).append(snapshot)
    state["datasetFingerprint"] = snapshot["fingerprint"]
    return snapshot
