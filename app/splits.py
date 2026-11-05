"""按规则分层切分训练 / 验证 / 测试集。

每条规则独立切分：同一通电话对 R001 可能属于训练集，对 R003 可能属于测试集。
这样每条规则的三个集合都能尽量包含违规和正常样本，也避免某条规则的正样本
被其它规则的数据切分“挤”到同一个集合里。
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
TEST = "TEST"
SPLITS = (TRAIN, VALIDATION, TEST)

SPLIT_LABELS = {TRAIN: "训练集", VALIDATION: "验证集", TEST: "测试集"}


def _key(item: Dict[str, Any]) -> str:
    return hashlib.sha1(str(item.get("callId") or "").encode("utf-8")).hexdigest()


def _allocate(items: List[Dict[str, Any]], ratios=(0.6, 0.2, 0.2)) -> Dict[str, str]:
    total = len(items)
    if total == 0:
        return {}
    if total < 5:
        return {item["callId"]: TRAIN for item in items}

    test_size = max(1, int(round(total * ratios[2])))
    validation_size = max(1, int(round(total * ratios[1])))
    train_size = total - test_size - validation_size
    if train_size < 1:
        train_size = 1
        test_size = max(1, total - validation_size - train_size)
    capacity = {TRAIN: train_size, VALIDATION: validation_size, TEST: test_size}

    positives = sorted([i for i in items if i.get("humanResult") == "VIOLATION"], key=_key)
    negatives = sorted([i for i in items if i.get("humanResult") != "VIOLATION"], key=_key)
    assigned: Dict[str, List[Dict[str, Any]]] = {TRAIN: [], VALIDATION: [], TEST: []}

    def place(item: Dict[str, Any], split: str) -> bool:
        if len(assigned[split]) < capacity[split]:
            assigned[split].append(item)
            return True
        return False

    def place_any(item: Dict[str, Any], order=SPLITS) -> None:
        for split in order:
            if place(item, split):
                return
        assigned[TRAIN].append(item)

    for split in (TRAIN, VALIDATION, TEST):
        if positives:
            place_any(positives.pop(0), (split,))
    for split in (TEST, VALIDATION):
        if negatives:
            place_any(negatives.pop(0), (split,))
    for item in positives:
        place_any(item, (TRAIN, VALIDATION, TEST))
    for item in negatives:
        place_any(item, (TRAIN, VALIDATION, TEST))

    mapping: Dict[str, str] = {}
    for split, group in assigned.items():
        for item in group:
            mapping[item["callId"]] = split
    for item in items:
        mapping.setdefault(item["callId"], TRAIN)
    return mapping


def build_splits(rules: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for rule in rules:
        code = str(rule.get("code") or "")
        if not code:
            continue
        items = [
            {
                "callId": call.get("callId"),
                "humanResult": "VIOLATION" if code in (call.get("humanRules") or []) else "NORMAL",
            }
            for call in calls
        ]
        result[code] = _allocate(items)
    return result


def split_cases(cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result = {split: [] for split in SPLITS}
    for case in cases:
        split = case.get("split") if case.get("split") in SPLITS else TRAIN
        result[split].append(case)
    return result


def split_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = split_cases(cases)
    summary = {}
    for split in SPLITS:
        items = grouped[split]
        summary[split] = {
            "total": len(items),
            "positives": sum(1 for c in items if c.get("humanResult") == "VIOLATION"),
            "negatives": sum(1 for c in items if c.get("humanResult") == "NORMAL"),
        }
    return summary
