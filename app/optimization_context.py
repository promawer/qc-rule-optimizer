"""从训练集评测结果构造代表错例、正确对照和有界的证据上下文。"""
from __future__ import annotations

import re
from .regex_safety import safe_search, RegexTimeout
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .models import NORMAL, VIOLATION


def _clean(text: str) -> str:
    text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


def _features(text: str) -> set:
    text = _clean(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _similarity(left: set, right: set) -> float:
    return len(left & right) / max(1, len(left | right))


def extract_context(
    transcript: str, rule: Dict[str, Any], row: Optional[Dict[str, Any]] = None, limit: int = 1200,
) -> str:
    """优先保留命中/理由附近的上下文，并留出首尾，避免长通话只看开头。"""
    text = str(transcript or "")
    if len(text) <= limit:
        return text
    row = row or {}
    anchors = []
    evidence = row.get("regexEvidence") or {}
    if evidence.get("end", 0) > evidence.get("start", 0):
        anchors.append((int(evidence["start"]), int(evidence["end"])))
    elif rule.get("regex"):
        try:
            match = safe_search(str(rule["regex"]), text)
            if match and match.end() > match.start():
                anchors.append(match.span())
        except (re.error, RegexTimeout):
            pass
    reason = str(row.get("semanticReason") or row.get("reason") or "")
    for quoted in re.findall(r"[「“\"]([^」”\"]{2,80})[」”\"]", reason):
        position = text.find(quoted)
        if position >= 0:
            anchors.append((position, position + len(quoted)))
    if not anchors:
        keywords = str(rule.get("semantic") or "") + str(rule.get("name") or "") + reason
        for word in ("坐席", "客户", "是否", "违规", "正常", "判断", "规则"):
            keywords = keywords.replace(word, "")
        features = _features(keywords)
        # 长度很大的单行转写也分窗评分，关键词在后半段时不能仍截取行首。
        windows = [(start, text[start:start + 400]) for start in range(0, len(text), 300)]
        best_start, best_text = max(windows, key=lambda item: len(_features(item[1]) & features))
        if _features(best_text) & features:
            anchors.append((best_start, best_start + len(best_text)))
    # 预留分隔符的空间；主证据窗口包含上下文与角色/时间，首尾补充对话走向。
    budget = max(0, limit - 60)
    edge = budget // 6
    windows = [(0, edge), (len(text) - edge, len(text))]
    if anchors:
        share = (budget - 2 * edge) // min(2, len(anchors))
        for start, end in anchors[:2]:
            start = max(0, min(start - share // 3, len(text) - share))
            line_start = text.rfind("\n", max(0, start - 100), start)
            if line_start >= 0:
                start = line_start + 1
            windows.append((start, min(len(text), start + share)))
    else:
        start = max(0, len(text) // 2 - (budget - 2 * edge) // 2)
        windows.append((start, start + budget - 2 * edge))
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return ("[通话节选，…表示省略]\n" + "\n…\n".join(text[a:b] for a, b in merged))[:limit]


def _group(row: Dict[str, Any]) -> str:
    regex, semantic = row.get("regexHit"), row.get("semanticHit")
    if regex is not None and semantic is not None and regex != semantic:
        part = "正则/语义判定冲突"
    elif regex:
        part = "正则命中"
    elif semantic:
        part = "语义命中"
    else:
        part = "已执行部分均未命中"
    text = str(row.get("transcript") or "")
    cues = []
    if re.search(r"不要|切勿|无需|不能|提醒|谨防|不必", text):
        cues.append("提醒/否定线索")
    if re.search(r"别人说|他说|她说|引用|假如|假设|如果", text):
        cues.append("引用/假设线索")
    return "；".join([part] + cues)


def _representatives(rows: List[Dict[str, Any]], limit: int, features: Dict[int, set]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r.get("callId") or r.get("id") or "")):
        groups[_group(row)].append(row)
    buckets = sorted(groups.values(), key=lambda items: (-len(items), _group(items[0])))
    selected = []
    while buckets and len(selected) < limit:
        remaining = []
        for bucket in buckets:
            if len(selected) >= limit:
                break
            # 跨组轮流取样，组内优先选择与已选文本差异大的样本。
            row = max(bucket, key=lambda r: min(
                (1 - _similarity(features[id(r)], features[id(s)]) for s in selected), default=1.0,
            ))
            selected.append(row)
            bucket.remove(row)
            bucket = [r for r in bucket if _similarity(features[id(r)], features[id(row)]) < 0.9]
            if bucket:
                remaining.append(bucket)
        buckets = remaining
    return selected


def build_optimization_context(rule: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    # 防御性限制来源：即使调用方传入混合集合，也不把验证/测试样本交给生成器。
    rows = [r for r in result.get("rows", []) if r.get("split", "TRAIN") == "TRAIN"
            and r.get("predicted") is not None and not r.get("error")
            and r.get("humanResult") in (NORMAL, VIOLATION)]
    features = {id(r): _features(str(r.get("transcript") or "")) for r in rows}
    errors = [r for r in rows if bool(r["predicted"]) != (r["humanResult"] == VIOLATION)]
    selected = []
    for label in (VIOLATION, NORMAL):
        selected.extend(_representatives([r for r in errors if r["humanResult"] == label], 6, features))
    correct = [r for r in rows if bool(r["predicted"]) == (r["humanResult"] == VIOLATION)]
    controls = []
    for label in (VIOLATION, NORMAL):
        candidates = sorted([r for r in correct if r["humanResult"] == label],
                            key=lambda r: str(r.get("callId") or r.get("id") or ""))
        for _ in range(2):
            if not candidates:
                break
            row = max(candidates, key=lambda r: (
                max((_similarity(features[id(r)], features[id(e)]) for e in selected), default=0)
                - max((_similarity(features[id(r)], features[id(c)]) for c in controls), default=0) * 0.25
            ))
            controls.append(row)
            candidates = [r for r in candidates if r is not row
                          and _similarity(features[id(r)], features[id(row)]) < 0.9]

    def sample(row: Dict[str, Any], control: bool) -> Dict[str, Any]:
        return {
            "callId": str(row.get("callId") or "")[:100],
            "kind": ("正确违规对照" if row["humanResult"] == VIOLATION else "正确正常对照") if control
                    else ("漏报 FN" if row["humanResult"] == VIOLATION else "误报 FP"),
            "group": _group(row),
            "humanResult": row["humanResult"], "predicted": row["predicted"],
            "regexHit": row.get("regexHit"), "semanticHit": row.get("semanticHit"),
            "regexReason": str(row.get("regexReason") or "未提供")[:200],
            "semanticReason": str(row.get("semanticReason") or row.get("reason") or "未生成/未执行")[:200],
            "transcript": extract_context(str(row.get("transcript") or ""), rule, row),
        }

    summary = {
        "errorTotal": len(errors), "selectedErrors": len(selected), "selectedControls": len(controls),
        "groups": dict(Counter(("FN / " if r["humanResult"] == VIOLATION else "FP / ") + _group(r) for r in errors)),
        "errorCallIds": [str(r.get("callId") or "")[:100] for r in selected],
        "controlCallIds": [str(r.get("callId") or "")[:100] for r in controls],
    }
    return {"errors": [sample(r, False) for r in selected],
            "controls": [sample(r, True) for r in controls], "summary": summary}
