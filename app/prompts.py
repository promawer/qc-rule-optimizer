"""通用语义提示词模板与渲染。"""
from __future__ import annotations

from typing import Any, Dict, List

PLACEHOLDER = "{rules}"

DEFAULT_SEMANTIC_PROMPT = """你是外呼质检语义判定专家。请严格依据下面列出的质检规则，判断坐席在通话中是否违反了指定的规则。

需要检查的规则有：
{rules}

判定要求：
1. 只依据坐席的发言判断，客户的话不能单独作为违规依据；
2. 注意区分提醒、否定、引用、假设等表达，不要只看关键词；
3. 如果证据不足，按未违规处理；
4. 严格按人工质检口径判断，只输出 JSON。"""


def render_rules_block(rules: List[Dict[str, Any]]) -> str:
    lines = []
    for rule in rules:
        semantic = str(rule.get("semantic") or "").strip()
        if not semantic:
            continue
        lines.append(f"{rule.get('code', '')}（{rule.get('name', '')}）：{semantic}")
    return "\n".join(lines) if lines else "（当前没有配置语义规则）"


def render_prompt(template: str, rules: List[Dict[str, Any]]) -> str:
    text = str(template or "").strip() or DEFAULT_SEMANTIC_PROMPT
    block = render_rules_block(rules)
    if PLACEHOLDER not in text:
        text = text + "\n\n需要检查的规则有：\n" + PLACEHOLDER
    return text.replace(PLACEHOLDER, block)


def validate_prompt(template: str) -> str:
    text = str(template or "").strip()
    if not text:
        raise ValueError("通用语义提示词不能为空")
    if len(text) > 6000:
        raise ValueError("通用语义提示词不能超过 6000 字")
    if PLACEHOLDER not in text:
        raise ValueError("通用语义提示词必须包含 {rules} 占位符，用来注入全部语义规则")
    return text
