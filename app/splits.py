"""全局同源分组切分；一通电话在所有规则下只属于同一个集合。

分组依据显式 groupId 和去时间戳、空白后的相同转写文本，按组哈希进行
60/20/20 分配。普通追加不移动已有组；合并此前分离的同源组会重新确定
组的分配，必须由数据版本记录这次变更。小样本不人为补齐集合或复制样本。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List

TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
TEST = "TEST"
SPLITS = (TRAIN, VALIDATION, TEST)
SPLIT_LABELS = {TRAIN: "训练集", VALIDATION: "验证集", TEST: "测试集"}
SPLIT_POLICY = "global-group-sha256-v1"


def normalized_transcript(text: str) -> str:
    """保留角色和正文；仅清除不应导致同一转写被拆开的格式噪音。"""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", "", text)
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", "", text)
    return re.sub(r"[\s\[\](){}]+", "", text)


def group_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """并查集合并显式同源组及完全相同的规范化转写，包含传递关联。"""
    parent = list(range(len(calls)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(right)] = find(left)

    seen: Dict[str, int] = {}
    keys: List[List[str]] = []
    for index, call in enumerate(calls):
        tokens = []
        group_id = str(call.get("groupId") or "").strip()
        normalized = normalized_transcript(call.get("transcript") or "")
        if group_id:
            tokens.append("group:" + group_id)
        if normalized:
            tokens.append("text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        if not tokens:
            tokens.append("call:" + str(call.get("callId") or call.get("id") or index))
        keys.append(tokens)
        for token in tokens:
            if token in seen:
                union(index, seen[token])
            else:
                seen[token] = index

    components: Dict[int, List[int]] = {}
    for index in range(len(calls)):
        components.setdefault(find(index), []).append(index)
    result = []
    for indices in components.values():
        tokens = {token for index in indices for token in keys[index]}
        explicit = sorted(token for token in tokens if token.startswith("group:"))
        anchor = explicit[0] if explicit else min(tokens)
        result.append({"groupKey": anchor, "calls": [calls[index] for index in indices]})
    return result


def build_splits(rules: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    for group in group_calls(calls):
        digest = hashlib.sha256((SPLIT_POLICY + ":" + group["groupKey"]).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        split = TRAIN if bucket < 6000 else VALIDATION if bucket < 8000 else TEST
        for call in group["calls"]:
            mapping[str(call.get("callId") or "")] = split
    return {str(rule["code"]): dict(mapping) for rule in rules if rule.get("code")}


def split_cases(cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    for case in cases:
        split = case.get("split") if case.get("split") in SPLITS else TRAIN
        result[split].append(case)
    return result


def split_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = split_cases(cases)
    summary = {}
    for split in SPLITS:
        items = grouped[split]
        reviewed = [c for c in items if c.get("humanResult") in ("VIOLATION", "NORMAL")
                    and c.get("reviewStatus", "REVIEWED") == "REVIEWED"]
        summary[split] = {
            "total": len(items),
            "reviewed": len(reviewed),
            "unknown": len(items) - len(reviewed),
            "positives": sum(1 for c in reviewed if c.get("humanResult") == "VIOLATION"),
            "negatives": sum(1 for c in reviewed if c.get("humanResult") == "NORMAL"),
        }
    return summary
