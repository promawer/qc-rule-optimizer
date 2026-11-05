"""DeepSeek / OpenAI 兼容接口封装，以及语义判定 Agent。"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional, Tuple

import httpx


class LLMError(Exception):
    pass


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start:end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise LLMError("模型返回的不是有效 JSON：" + raw[:160].replace("\n", " "))


class LLMClient:
    """一个优化任务内复用的异步客户端，带并发限制与重试。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        concurrency: int = 6,
        timeout: float = 90.0,
    ) -> None:
        if not api_key:
            raise LLMError("未配置 API Key")
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model or "deepseek-chat"
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0))

    @property
    def endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self._semaphore:
            use_json_mode = True
            for attempt in range(3):
                try:
                    response = await self._client.post(
                        self.endpoint, json=payload, headers=headers
                    )
                    if response.status_code == 400 and use_json_mode:
                        # 部分兼容网关不支持 response_format，去掉后重试。
                        payload.pop("response_format", None)
                        use_json_mode = False
                        continue
                    if response.status_code in (429, 500, 502, 503, 504):
                        if attempt < 2:
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        raise LLMError(f"模型服务繁忙（HTTP {response.status_code}），请稍后重试")
                    if response.status_code >= 400:
                        body = response.text[:300].replace("\n", " ")
                        raise LLMError(f"模型接口返回 HTTP {response.status_code}：{body}")
                    data = response.json()
                    content = (
                        (data.get("choices") or [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return _extract_json(content)
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise LLMError("模型请求超时，请检查网络或 base URL")
                except httpx.HTTPError as exc:
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise LLMError(f"模型请求失败：{exc}") from exc
        raise LLMError("模型请求失败")


from .prompts import render_prompt


JUDGE_SYSTEM_SUFFIX = "\n\n你只判断指定的当前规则，不要评价业务建议，不要修改规则。严格按人工质检口径判断，只输出 JSON。"


def build_judge_prompt(rule: Dict[str, Any], transcript: str, with_reason: bool = True) -> str:
    schema = (
        '{"violation": true 或 false, "reason": "不超过40字的判断依据"}'
        if with_reason
        else '{"violation": true 或 false}'
    )
    extra = "" if with_reason else "\n只输出违规与否，不要输出理由，不要解释。"
    return f"""当前规则：{rule.get('code', '')} {rule.get('name', '')}
规则说明：{rule.get('description', '') or '无'}
违规语义条件：{rule.get('semantic', '')}

通话文本：
{transcript[:6000]}

请判断这通电话是否满足上述违规语义条件。注意：
1. 只有坐席的行为满足条件才算违规，客户说的话不能单独作为违规依据；
2. 提醒、否定、引用他人说法等场景，要按实际语义判断，不要只按关键词；
3. 只输出 JSON：{schema}。{extra}"""


async def judge_semantic(
    client: LLMClient,
    case: Dict[str, Any],
    rule: Dict[str, Any],
    semantic_prompt: str,
    all_rules: list,
    with_reason: bool = True,
) -> Tuple[bool, str]:
    system = render_prompt(semantic_prompt, all_rules) + JUDGE_SYSTEM_SUFFIX
    data = await client.chat_json(
        system,
        build_judge_prompt(rule, str(case.get("transcript") or ""), with_reason),
        max_tokens=300 if with_reason else 40,
        temperature=0.0,
    )
    value = data.get("violation")
    if isinstance(value, str):
        lowered = value.strip().lower()
        violation = lowered in ("true", "yes", "1", "是", "违规", "命中")
    else:
        violation = bool(value)
    reason = str(data.get("reason") or "")[:200] if with_reason else ""
    return violation, reason


async def test_connection(base_url: str, model: str, api_key: str) -> Dict[str, Any]:
    client = LLMClient(base_url, model, api_key, concurrency=1, timeout=30)
    try:
        data = await client.chat_json(
            "你是连通性测试助手，只输出 JSON。",
            '请输出 {"ok": true}',
            max_tokens=20,
            temperature=0.0,
        )
        return {"ok": True, "reply": data}
    finally:
        await client.aclose()
