"""模型接口封装：OpenAI 兼容模式 + 可自由配置表头/入参/出参的自定义模式。"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

DEFAULT_API_PATH = "/chat/completions"
DEFAULT_CONTENT_PATH = "choices.0.message.content"
DEFAULT_HEADERS: List[Dict[str, str]] = [
    {"name": "Content-Type", "value": "application/json"},
    {"name": "Authorization", "value": "Bearer {{apiKey}}"},
]
DEFAULT_REQUEST_TEMPLATE = """{
  "model": "{{model}}",
  "messages": "{{messages}}",
  "temperature": "{{temperature}}",
  "max_tokens": "{{max_tokens}}"
}"""

ALLOWED_VARIABLES = {
    "prompt": "完整提示词（系统提示词 + 业务提示词，单入参接口用）",
    "system": "系统提示词",
    "user": "用户提示词",
    "messages": "OpenAI 格式的消息数组",
    "model": "模型名",
    "apiKey": "API Key（内存中的值）",
    "requestId": "每次调用唯一的请求 ID（可用于 sessionId）",
    "max_tokens": "最大输出 token 数",
    "temperature": "温度",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


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


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_path(data: Any, path: str) -> Any:
    """读取 ``choices.0.message.content`` / ``data.items[0].answer`` 这类路径。"""
    text = (path or "").strip()
    if not text:
        return None
    if text.startswith("$"):
        text = text[1:]
    current = data
    for raw in re.split(r"[./]", text):
        token = raw.strip()
        if not token:
            continue
        if isinstance(current, str):
            stripped = current.strip()
            if stripped.startswith(("{", "[")):
                try:
                    current = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
        match = re.fullmatch(r"([^\[\]]*)(?:\[(\d+)\])?", token)
        if not match:
            return None
        key, index = match.group(1), match.group(2)
        if key:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
                current = current[int(key)]
            else:
                return None
        if index is not None:
            if isinstance(current, list) and int(index) < len(current):
                current = current[int(index)]
            else:
                return None
    return current


def _render_string(text: str, variables: Dict[str, Any]) -> Any:
    full = _PLACEHOLDER.fullmatch(text)
    if full:
        name = full.group(1)
        if name not in variables:
            raise LLMError(f"请求模板使用了未知变量 {{{{{name}}}}}")
        return variables[name]

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            raise LLMError(f"请求模板使用了未知变量 {{{{{name}}}}}")
        value = variables[name]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return _PLACEHOLDER.sub(replace, text)


def _render_value(value: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _render_value(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_value(item, variables) for item in value]
    if isinstance(value, str):
        return _render_string(value, variables)
    return value


def build_request_payload(template: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    text = (template or "").strip() or DEFAULT_REQUEST_TEMPLATE
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"请求体模板不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("请求体模板必须是 JSON 对象")
    payload = _render_value(parsed, variables)
    if not isinstance(payload, dict):
        raise LLMError("请求体模板必须是 JSON 对象")
    return payload


def _collect_placeholders(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_collect_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_placeholders(item))
    elif isinstance(value, str):
        found.extend(_PLACEHOLDER.findall(value))
    return found


def validate_request_template(template: str) -> None:
    text = (template or "").strip()
    if not text:
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"请求体模板不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("请求体模板必须是 JSON 对象")
    unknown = sorted(set(_collect_placeholders(parsed)) - set(ALLOWED_VARIABLES))
    if unknown:
        raise LLMError("请求体模板包含未知变量：" + "、".join(unknown))


def template_has_business_input(template: str) -> bool:
    """模板必须包含完整提示词、用户提示词或 messages，否则不会上送规则和通话文本。"""
    text = (template or "").strip()
    if not text:
        return True  # 空模板使用 OpenAI 默认模板
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    names = set(_collect_placeholders(parsed))
    return bool(names & {"prompt", "user", "messages"})


def normalize_headers(headers: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for item in headers or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "")
        if not name:
            continue
        if any(char in name or char in value for char in ("\r", "\n")):
            raise LLMError("请求表头的名称或值不能包含换行符")
        result.append({"name": name, "value": value})
    return result


def _needs_api_key(config: Dict[str, Any]) -> bool:
    blob = json.dumps(config.get("headers") or [], ensure_ascii=False)
    blob += "\n" + str(config.get("requestTemplate") or "")
    return "apiKey" in _PLACEHOLDER.findall(blob)


def config_needs_api_key(config: Dict[str, Any]) -> bool:
    return _needs_api_key(config)


class LLMClient:
    """一个优化任务内复用的异步客户端，带并发限制与重试。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        concurrency: int = 1,
        timeout: float = 90.0,
        trace: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = dict(config or {})
        self._trace = trace
        self.protocol = str(self.config.get("apiProtocol") or "openai").lower()
        if self.protocol not in ("openai", "custom"):
            self.protocol = "openai"
        if self.protocol == "custom":
            self.api_path = str(self.config.get("apiPath") or DEFAULT_API_PATH).strip()
            self.content_path = str(self.config.get("contentPath") or DEFAULT_CONTENT_PATH).strip()
        else:
            self.api_path = DEFAULT_API_PATH
            self.content_path = DEFAULT_CONTENT_PATH
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model or "deepseek-chat"
        self.api_key = api_key
        if (self.protocol != "custom" or _needs_api_key(self.config)) and not self.api_key:
            raise LLMError("未配置 API Key")
        if self.protocol == "custom" and not template_has_business_input(
            str(self.config.get("requestTemplate") or DEFAULT_REQUEST_TEMPLATE)
        ):
            raise LLMError(
                "自定义接口请求模板没有包含 {{prompt}}、{{user}} 或 {{messages}}，"
                "不会上送当前规则和通话文本；请改用 {{prompt}} 或单入参模板。"
            )
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        timeout_value = float(self.config.get("timeoutSec") or timeout)
        self.timeout_value = timeout_value
        self.max_retries = max(0, min(10, int(self.config.get("maxRetries", 3) or 0)))
        self.retry_backoff = max(0.2, min(30.0, float(self.config.get("retryBackoffSec", 1.5) or 1.5)))
        self.request_interval = max(0.0, min(30.0, float(self.config.get("requestIntervalSec", 0.5) or 0)))
        self._last_call_started = 0.0
        self.stats: Dict[str, Any] = {
            "calls": 0,
            "attempts": 0,
            "retries": 0,
            "errors": 0,
            "lastError": "",
        }
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_value, connect=min(15.0, timeout_value)))

    @property
    def endpoint(self) -> str:
        path = self.api_path
        if path.startswith(("http://", "https://")):
            return path
        if not path:
            return self.base_url
        if self.base_url.endswith(path):
            return self.base_url
        return self.base_url + "/" + path.lstrip("/")

    async def aclose(self) -> None:
        await self._client.aclose()

    def describe(self) -> str:
        return (
            f"协议={self.protocol}，地址={self.endpoint}，模型={self.model}，"
            f"超时={self.timeout_value:g}s，重试={self.max_retries}次，"
            f"并发=1，调用间隔={self.request_interval:g}s"
        )

    def _send_trace(self, event: Dict[str, Any]) -> None:
        if not self._trace:
            return
        try:
            self._trace(event)
        except Exception:
            pass

    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        sensitive = ("authorization", "token", "cookie", "api-key", "apikey", "secret")
        return {
            name: ("******" if any(word in name.lower() for word in sensitive) else value)
            for name, value in headers.items()
        }

    def _payload_text(self, payload: Dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False)
        if self.api_key:
            text = text.replace(self.api_key, "******")
        return text

    def _record_error(self, message: str) -> LLMError:
        self.stats["errors"] += 1
        self.stats["lastError"] = message[:400]
        return LLMError(message)

    async def _retry_sleep(self, attempt: int) -> None:
        self.stats["retries"] += 1
        await asyncio.sleep(self.retry_backoff * (attempt + 1))

    def _variables(self, system: str, user: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        return {
            "prompt": system.strip() + "\n\n" + user.strip(),
            "system": system,
            "user": user,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "model": self.model,
            "apiKey": self.api_key,
            "requestId": uuid.uuid4().hex,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _build_headers(self, variables: Dict[str, Any]) -> Dict[str, str]:
        if self.protocol != "custom":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        configured = normalize_headers(self.config.get("headers"))
        if not configured:
            configured = [{"name": "Content-Type", "value": "application/json"}]
        headers: Dict[str, str] = {}
        for item in configured:
            rendered = _render_string(item["value"], variables)
            headers[item["name"]] = str(rendered)
        if not any(name.lower() == "content-type" for name in headers):
            headers["Content-Type"] = "application/json"
        return headers

    def _extract(self, data: Any, judge: bool) -> Dict[str, Any]:
        if judge and self.protocol == "custom":
            violation_path = str(self.config.get("violationPath") or "").strip()
            reason_path = str(self.config.get("reasonPath") or "").strip()
            if violation_path:
                violation = extract_path(data, violation_path)
                if violation is not None:
                    reason = extract_path(data, reason_path) if reason_path else ""
                    return {"violation": violation, "reason": _stringify(reason)}
        content = extract_path(data, self.content_path)
        if content is None:
            top_keys = list(data.keys())[:12] if isinstance(data, dict) else []
            if isinstance(data, dict) and "result" in data:
                result_type = type(data.get("result")).__name__
            else:
                result_type = "缺失"
            raise LLMError(
                f"响应中找不到字段：{self.content_path}"
                f"（顶层字段：{top_keys}，result 类型：{result_type}）"
            )
        if isinstance(content, dict):
            return content
        if judge and isinstance(content, bool):
            return {"violation": content}
        return _extract_json(_stringify(content))

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        judge: bool = False,
    ) -> Dict[str, Any]:
        variables = self._variables(system, user, max_tokens, temperature)
        if self.protocol == "custom":
            payload = build_request_payload(
                str(self.config.get("requestTemplate") or DEFAULT_REQUEST_TEMPLATE),
                variables,
            )
            headers = self._build_headers(variables)
            use_json_mode = False
        else:
            payload = {
                "model": self.model,
                "messages": variables["messages"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            headers = self._build_headers(variables)
            use_json_mode = True

        self.stats["calls"] += 1
        attempts = self.max_retries + 1
        if use_json_mode:
            attempts = max(attempts, 2)
        async with self._semaphore:
            if self.request_interval > 0 and self._last_call_started:
                wait = self.request_interval - (time.monotonic() - self._last_call_started)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_call_started = time.monotonic()
            for attempt in range(attempts):
                self.stats["attempts"] += 1
                call_no = self.stats["calls"]
                attempt_started = time.monotonic()
                self._send_trace({
                    "event": "request",
                    "call": call_no,
                    "attempt": attempt + 1,
                    "endpoint": self.endpoint,
                    "judge": judge,
                    "template": str(self.config.get("requestTemplate") or ""),
                    "systemChars": len(system),
                    "userChars": len(user),
                    "promptChars": len(variables["prompt"]),
                    "containsTranscript": "通话文本：" in user,
                    "headers": self._redact_headers(headers),
                    "payloadText": self._payload_text(payload),
                })
                try:
                    response = await self._client.post(
                        self.endpoint,
                        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                    )
                    body_text = response.text or ""
                    self._send_trace({
                        "event": "response",
                        "call": call_no,
                        "attempt": attempt + 1,
                        "status": response.status_code,
                        "durationMs": int((time.monotonic() - attempt_started) * 1000),
                        "body": body_text,
                        "bodyTruncated": False,
                    })
                    if response.status_code == 400 and use_json_mode:
                        # 部分兼容网关不支持 response_format，去掉后重试。
                        payload.pop("response_format", None)
                        use_json_mode = False
                        continue
                    if response.status_code in (408, 425, 429, 500, 502, 503, 504):
                        if attempt < self.max_retries:
                            await self._retry_sleep(attempt)
                            continue
                        raise self._record_error(
                            f"模型服务繁忙或超时（HTTP {response.status_code}，"
                            f"已重试 {self.max_retries} 次）"
                        )
                    if response.status_code >= 400:
                        body = response.text[:500].replace("\n", " ")
                        raise self._record_error(f"模型接口返回 HTTP {response.status_code}：{body}")
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise self._record_error(
                            "模型接口返回的不是 JSON：" + response.text[:300].replace("\n", " ")
                        ) from exc
                    try:
                        return self._extract(data, judge)
                    except LLMError as exc:
                        self.stats["errors"] += 1
                        self.stats["lastError"] = str(exc)[:400]
                        raise
                except httpx.TimeoutException:
                    self._send_trace({
                        "event": "error",
                        "call": call_no,
                        "attempt": attempt + 1,
                        "durationMs": int((time.monotonic() - attempt_started) * 1000),
                        "error": f"timeout after {self.timeout_value:g}s",
                    })
                    if attempt < self.max_retries:
                        await self._retry_sleep(attempt)
                        continue
                    raise self._record_error(
                        f"模型请求超时（单次 {self.timeout_value:g} 秒，已重试 {self.max_retries} 次）"
                    )
                except httpx.HTTPError as exc:
                    self._send_trace({
                        "event": "error",
                        "call": call_no,
                        "attempt": attempt + 1,
                        "durationMs": int((time.monotonic() - attempt_started) * 1000),
                        "error": str(exc)[:1000],
                    })
                    if attempt < self.max_retries:
                        await self._retry_sleep(attempt)
                        continue
                    raise self._record_error(f"模型请求失败：{exc}") from exc
        raise self._record_error("模型请求失败")


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
        judge=True,
    )
    value = data.get("violation")
    if isinstance(value, str):
        lowered = value.strip().lower()
        violation = lowered in ("true", "yes", "1", "是", "违规", "命中")
    else:
        violation = bool(value)
    reason = str(data.get("reason") or "")[:200] if with_reason else ""
    return violation, reason


async def test_connection(
    base_url: str,
    model: str,
    api_key: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = LLMClient(base_url, model, api_key, config=config, concurrency=1, timeout=30)
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
