"""请求模型、规则校验与标签归一化。"""
from __future__ import annotations

import re
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

VIOLATION = "VIOLATION"
NORMAL = "NORMAL"
UNKNOWN = "UNKNOWN"

_VIOLATION_WORDS = {
    "违规", "命中", "违反", "是", "y", "yes", "true", "1", "violation",
    "fail", "不合格", "有问题", "异常", "触发",
}
_NORMAL_WORDS = {
    "正常", "未命中", "未违反", "否", "n", "no", "false", "0", "normal",
    "pass", "合格", "无问题", "通过", "未触发",
}
_UNKNOWN_WORDS = {"", "-", "—", "无", "unknown", "无法判断", "不确定", "待定", "n/a", "na"}


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in _VIOLATION_WORDS:
        return VIOLATION
    if text in _NORMAL_WORDS:
        return NORMAL
    if text in _UNKNOWN_WORDS:
        return UNKNOWN
    # 兼容「违规（确认）」这类带说明的写法
    if "违规" in text or "命中" in text or "违反" in text:
        return VIOLATION
    if "正常" in text or "未命中" in text or "未违反" in text:
        return NORMAL
    return UNKNOWN


class RuleInput(BaseModel):
    code: str = Field(min_length=1, max_length=20)
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=500)
    regex: str = Field(default="", max_length=2000)
    semantic: str = Field(default="", max_length=1000)
    logic: Literal["OR", "AND"] = "OR"
    changeReason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _validate_content(self) -> "RuleInput":
        self.code = self.code.strip().upper()
        self.name = self.name.strip()
        self.description = self.description.strip()
        self.regex = self.regex.strip()
        self.semantic = self.semantic.strip()
        self.changeReason = self.changeReason.strip()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.\-]{0,19}", self.code):
            raise ValueError("规则编号只能包含字母、数字、下划线、短横线或点，且以字母或数字开头")
        if not self.regex and not self.semantic:
            raise ValueError("正则表达式和语义描述至少填写一项")
        if self.regex:
            try:
                re.compile(self.regex)
            except re.error as exc:
                raise ValueError(f"正则表达式无法编译：{exc}") from exc
        return self


class SettingsInput(BaseModel):
    mode: Literal["mock", "deepseek"] = "mock"
    baseUrl: str = Field(default="https://api.deepseek.com", max_length=300)
    model: str = Field(default="deepseek-chat", max_length=100)
    apiKey: Optional[str] = Field(default=None, max_length=500)


class PromptInput(BaseModel):
    prompt: str = Field(default="", max_length=6000)
    reason: str = Field(default="", max_length=500)


class OptimizeInput(BaseModel):
    ruleId: str
    targetAccuracy: float = Field(default=0.9, ge=0.5, le=1.0)
    maxRounds: int = Field(default=5, ge=1, le=10)
    sampleLimit: int = Field(default=100, ge=5, le=500)


class ApplyInput(BaseModel):
    reason: str = Field(default="", max_length=500)
    force: bool = False


def rule_flags(rule: Dict[str, object]) -> Dict[str, bool]:
    """返回规则两部分是否启用。"""
    return {
        "regex": bool(str(rule.get("regex") or "").strip()),
        "semantic": bool(str(rule.get("semantic") or "").strip()),
    }
