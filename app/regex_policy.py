"""正则候选的确定性约束：300 字符上限、局部修改、去重简化、锚点检查。"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional

MAX_REGEX_LENGTH = 300
MAX_ADDED_ALTERNATIVES = 2
MIN_EDIT_SIMILARITY = 0.5
ROLE_WORDS = {"坐席", "客服", "客户", "员工", "代表"}


class RegexConstraintError(ValueError):
    pass


def _split_alternatives(pattern: str) -> List[str]:
    """按顶层 ``|`` 切分，跳过字符类和分组内部的竖线。"""
    parts: List[str] = []
    start = 0
    depth = 0
    in_class = False
    escaped = False
    for index, char in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "|" and depth == 0:
            parts.append(pattern[start:index])
            start = index + 1
    parts.append(pattern[start:])
    return parts


def _matching_paren(pattern: str, start: int) -> Optional[int]:
    depth = 0
    in_class = False
    escaped = False
    for index in range(start, len(pattern)):
        char = pattern[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _contains_capturing_group(pattern: str) -> bool:
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            end = pattern.find("]", index + 1)
            index = len(pattern) if end < 0 else end + 1
            continue
        if char == "(":
            if not pattern.startswith("(?", index):
                return True
            if pattern.startswith("(?P<", index) or pattern.startswith("(?'", index):
                return True
        index += 1
    return False


def _dedupe_alternatives(pattern: str) -> str:
    parts = _split_alternatives(pattern)
    if len(parts) < 2:
        return pattern
    if any(_contains_capturing_group(part) for part in parts):
        # 删除带捕获组的分支会改变组编号/反向引用，宁可不简化。
        return pattern
    seen = set()
    unique: List[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        unique.append(part)
    return "|".join(unique)


def _simplify_groups(pattern: str) -> str:
    out: List[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            out.append(pattern[index:index + 2])
            index += 2
            continue
        if char == "[":
            end = index + 1
            escaped = False
            while end < len(pattern):
                current = pattern[end]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == "]":
                    break
                end += 1
            out.append(pattern[index:end + 1])
            index = end + 1
            continue
        if pattern.startswith("(?:", index):
            end = _matching_paren(pattern, index)
            if end is None:
                out.append(pattern[index:])
                break
            inner = _simplify_groups(pattern[index + 3:end])
            inner = _dedupe_alternatives(inner)
            out.append("(?:" + inner + ")")
            index = end + 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def simplify_regex(pattern: str) -> str:
    """保守、确定性的等价简化：去重分支、递归处理非捕获组、去掉最外层冗余组。"""
    text = str(pattern or "").strip()
    if not text:
        return ""
    text = _simplify_groups(text)
    text = _dedupe_alternatives(text)
    if text.startswith("(?:") and _matching_paren(text, 0) == len(text) - 1:
        inner = text[3:-1]
        if not _contains_capturing_group(inner) or "|" not in inner:
            text = inner
    text = _dedupe_alternatives(text)
    return text


def _count_alternatives(pattern: str) -> int:
    count = max(0, len(_split_alternatives(pattern)) - 1)
    index = 0
    while index < len(pattern):
        if pattern.startswith("(?:", index):
            end = _matching_paren(pattern, index)
            if end is None:
                break
            count += _count_alternatives(pattern[index + 3:end])
            index = end + 1
            continue
        index += 1
    return count


def _bare_role_word(pattern: str) -> Optional[str]:
    for part in _split_alternatives(pattern):
        stripped = part.strip()
        if stripped in ROLE_WORDS:
            return stripped
        if stripped.startswith("(?:") and _matching_paren(stripped, 0) == len(stripped) - 1:
            found = _bare_role_word(stripped[3:-1])
            if found:
                return found
    return None


def regex_metrics(pattern: str) -> dict:
    text = str(pattern or "")
    return {
        "length": len(text),
        "alternatives": _count_alternatives(text),
        "groups": text.count("("),
    }


def constrain_regex(candidate: str, baseline: str = "") -> str:
    """返回简化后的候选；违反硬约束时抛 RegexConstraintError。"""
    raw = str(candidate or "").strip()
    if not raw:
        return ""
    simplified = simplify_regex(raw)
    if len(simplified) > MAX_REGEX_LENGTH:
        raise RegexConstraintError(
            f"候选正则 {len(simplified)} 字符，超过 {MAX_REGEX_LENGTH} 字符硬上限；"
            "请删除冗余分支，只保留高精度锚点"
        )
    try:
        re.compile(simplified)
    except re.error as exc:
        raise RegexConstraintError(f"候选正则无法编译：{exc}") from exc

    role = _bare_role_word(simplified)
    if role:
        raise RegexConstraintError(
            f"候选正则把角色词“{role}”单独作为分支，会命中每通电话的角色前缀；"
            "请改成“工号/金额等锚点 + 数字/明确业务上下文”，泛化交给语义"
        )

    base = str(baseline or "").strip()
    if base and len(base) <= MAX_REGEX_LENGTH:
        base_simplified = simplify_regex(base)
        if simplified != base_simplified:
            similarity = SequenceMatcher(None, base, simplified).ratio()
            if similarity < MIN_EDIT_SIMILARITY:
                raise RegexConstraintError(
                    "候选正则改动过大，要求每轮只做局部修改；"
                    f"与当前正则相似度仅 {similarity:.0%}，至少需保留 {MIN_EDIT_SIMILARITY:.0%} 的原有结构"
                )
            added = _count_alternatives(simplified) - _count_alternatives(base)
            if added > MAX_ADDED_ALTERNATIVES:
                raise RegexConstraintError(
                    f"候选正则新增了 {added} 个分支，单轮最多允许新增 {MAX_ADDED_ALTERNATIVES} 个"
                )
    return simplified
