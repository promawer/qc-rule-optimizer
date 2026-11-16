"""规则文件导入：规则编号、规则名词、关键词、规则语义、逻辑关系。"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .importer import parse_file

MAX_RULES = 2000

_CODE_KEYS = {
    "规则编号", "规则编码", "编号", "规则id", "rulecode", "rule_code", "code",
}
_NAME_KEYS = {
    "规则名词", "规则名称", "规则名", "名称", "rulename", "rule_name", "name",
}
_KEYWORD_KEYS = {
    "关键词", "关键字", "正则", "正则表达式", "keyword", "keywords", "regex", "pattern",
}
_SEMANTIC_KEYS = {
    "规则语义", "规则描述", "语义", "业务说明", "说明", "semantic", "description",
}
_LOGIC_KEYS = {
    "规则内容的逻辑关系", "逻辑关系", "逻辑", "logic", "operator", "logicalrelation",
}

_SEPARATORS = re.compile(r"[|,，;；、\n]+")


def _canon(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s_\-（）()【】\[\]:：/\\]", "", text)


def _match(headers: List[str], aliases: set) -> str:
    normalized = {_canon(alias) for alias in aliases}
    for header in headers:
        if _canon(header) in normalized:
            return header
    return ""


def _detect_columns(headers: List[str]) -> Dict[str, str]:
    return {
        "code": _match(headers, _CODE_KEYS),
        "name": _match(headers, _NAME_KEYS),
        "keywords": _match(headers, _KEYWORD_KEYS),
        "semantic": _match(headers, _SEMANTIC_KEYS),
        "logic": _match(headers, _LOGIC_KEYS),
    }


def keyword_to_regex(value: object) -> str:
    """普通关键词按字面量转正则；re: 前缀表示整段就是正则。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith(("re:", "regex:")):
        pattern = text.split(":", 1)[1].strip()
        if not pattern:
            raise ValueError("re: 后面没有正则内容")
        re.compile(pattern)
        return pattern
    tokens = [token.strip() for token in _SEPARATORS.split(text) if token.strip()]
    if not tokens:
        return ""
    pattern = "(?:" + "|".join(re.escape(token) for token in tokens) + ")"
    re.compile(pattern)
    return pattern


def normalize_logic(value: object) -> Tuple[str, str]:
    text = str(value or "").strip()
    upper = text.upper()
    if not text:
        return "OR", ""
    if upper in ("AND", "&&", "&") or text in ("与", "且", "并且", "和"):
        return "AND", ""
    if upper in ("OR", "||", "|") or text in ("或", "或者"):
        return "OR", ""
    return "OR", f"逻辑关系「{text}」无法识别，已按 OR 处理"


def parse_rule_file(filename: str, content: bytes) -> Dict[str, Any]:
    headers, rows = parse_file(filename, content)
    columns = _detect_columns(headers)
    missing = [key for key in ("code", "name") if not columns[key]]
    if not columns["keywords"] and not columns["semantic"]:
        missing.append("keywords/semantic")
    if missing:
        raise ValueError(
            "规则文件缺少列：" + "、".join(missing)
            + "。需要的表头是：规则编号、规则名词、关键词、规则语义、规则内容的逻辑关系"
        )
    if len(rows) > MAX_RULES:
        raise ValueError(f"规则文件最多 {MAX_RULES} 行")

    rules: List[Dict[str, str]] = []
    errors: List[str] = []
    warnings: List[str] = []
    seen = set()

    for offset, row in enumerate(rows, start=2):
        values = {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        code = str(values.get(columns["code"]) or "").strip().upper()
        name = str(values.get(columns["name"]) or "").strip()
        keyword_text = str(values.get(columns["keywords"]) or "").strip() if columns["keywords"] else ""
        semantic_raw = str(values.get(columns["semantic"]) or "").strip() if columns["semantic"] else ""
        logic_text = str(values.get(columns["logic"]) or "").strip() if columns["logic"] else ""

        if not code and not name and not keyword_text and not semantic:
            continue
        if not code:
            errors.append(f"第 {offset} 行：规则编号为空")
            continue
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.\-]{0,19}", code):
            errors.append(f"第 {offset} 行：规则编号「{code}」格式不合法")
            continue
        if code in seen:
            errors.append(f"第 {offset} 行：规则编号 {code} 在文件中重复")
            continue
        if not name:
            errors.append(f"第 {offset} 行：规则名称为空")
            continue
        if len(name) > 60:
            errors.append(f"第 {offset} 行：规则名称超过 60 字")
            continue
        if len(semantic_raw) > 1000:
            errors.append(f"第 {offset} 行：规则语义超过 1000 字")
            continue
        semantic = semantic_raw
        try:
            regex = keyword_to_regex(keyword_text)
        except re.error as exc:
            errors.append(f"第 {offset} 行：关键词转正则失败：{exc}")
            continue
        except ValueError as exc:
            errors.append(f"第 {offset} 行：{exc}")
            continue
        if len(regex) > 2000:
            errors.append(f"第 {offset} 行：关键词转换后的正则超过 2000 字符")
            continue
        if not regex and not semantic:
            errors.append(f"第 {offset} 行：关键词和规则语义至少填写一项")
            continue
        logic, warning = normalize_logic(logic_text)
        if warning:
            warnings.append(f"第 {offset} 行：{warning}")
        seen.add(code)
        rules.append({
            "code": code,
            "name": name,
            "regex": regex,
            "semantic": semantic,
            "logic": logic,
        })

    return {
        "headers": headers,
        "columns": columns,
        "rules": rules,
        "errors": errors,
        "warnings": warnings,
    }
