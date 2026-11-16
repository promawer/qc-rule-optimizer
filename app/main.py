"""质检规则优化台 —— FastAPI 后端。"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .evaluator import EvaluationError, JobCancelled, evaluate_rule
from .importer import build_calls, calls_to_csv, cases_for_rule
from .llm import (
    DEFAULT_API_PATH,
    DEFAULT_CONTENT_PATH,
    DEFAULT_HEADERS,
    DEFAULT_REQUEST_TEMPLATE,
    LLMClient,
    LLMError,
    config_needs_api_key,
    judge_semantic,
    normalize_headers,
    test_connection,
    validate_request_template,
    template_has_business_input,
)
from .models import (
    NORMAL, VIOLATION, ApplyInput, BatchOptimizeInput, OptimizeInput, PromptInput, RuleInput, SettingsInput, rule_flags,
)
from .optimizer import LLMProposer, MockProposer, optimize
from .prompts import DEFAULT_SEMANTIC_PROMPT, render_prompt, validate_prompt
from .rule_importer import parse_rule_file
from .splits import SPLITS, TEST, TRAIN, VALIDATION, build_splits, split_cases, split_summary
from .store import BASE_DIR, STORE_PATH, Store

STATIC_DIR = BASE_DIR / "static"
JOB_LOG_DIR = BASE_DIR / ".run" / "jobs"
JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="质检规则优化台", version="2.6.2")
store = Store(Path(os.environ.get("QCO_STORE", str(STORE_PATH))))
RUNTIME: Dict[str, str] = {"apiKey": os.environ.get("DEEPSEEK_API_KEY", "")}
JOBS: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, "asyncio.Task[Any]"] = {}
MAX_DATASET = 2000


@app.middleware("http")
async def _no_cache_frontend(request: Request, call_next):
    """本地单页应用禁用缓存，避免更新后浏览器继续用旧的 app.js。"""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _api_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    protocol = str(settings.get("apiProtocol") or "openai")
    headers = settings.get("headers")
    if headers is None:
        headers = DEFAULT_HEADERS
    return {
        "apiProtocol": protocol,
        "apiPath": str(settings.get("apiPath", DEFAULT_API_PATH) or ""),
        "headers": [dict(item) for item in headers if isinstance(item, dict)],
        "requestTemplate": str(settings.get("requestTemplate") or DEFAULT_REQUEST_TEMPLATE),
        "contentPath": str(settings.get("contentPath") or DEFAULT_CONTENT_PATH),
        "violationPath": str(settings.get("violationPath") or ""),
        "reasonPath": str(settings.get("reasonPath") or ""),
        "timeoutSec": int(settings.get("timeoutSec") or 90),
        "maxRetries": int(settings.get("maxRetries", 3) if settings.get("maxRetries") is not None else 3),
        "retryBackoffSec": float(settings.get("retryBackoffSec", 1.5) if settings.get("retryBackoffSec") is not None else 1.5),
        "requestIntervalSec": float(settings.get("requestIntervalSec", 0.5) if settings.get("requestIntervalSec") is not None else 0.5),
    }


def _settings_response(settings: Dict[str, Any]) -> Dict[str, Any]:
    config = _api_config(settings)
    return {
        "mode": settings.get("mode", "mock"),
        "baseUrl": settings.get("baseUrl", "https://api.deepseek.com"),
        "model": settings.get("model", "deepseek-chat"),
        "hasApiKey": bool(RUNTIME.get("apiKey")),
        **config,
        "defaultHeaders": [dict(item) for item in DEFAULT_HEADERS],
        "defaultRequestTemplate": DEFAULT_REQUEST_TEMPLATE,
        "requiresApiKey": config["apiProtocol"] != "custom" or config_needs_api_key(config),
    }


# ---------------------------------------------------------------- 基础
@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for error in exc.errors():
        message = str(error.get("msg") or "参数不合法")
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        messages.append(message)
    return JSONResponse(status_code=400, content={"detail": "；".join(messages)})


def _rule_or_404(state: Dict[str, Any], rule_id: str) -> Dict[str, Any]:
    for rule in state["rules"]:
        if rule["id"] == rule_id:
            return rule
    raise HTTPException(status_code=404, detail="规则不存在")


def _dataset_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {VIOLATION: 0, NORMAL: 0, "UNKNOWN": 0}
    for case in cases:
        labels[case.get("humanResult", "UNKNOWN")] = labels.get(case.get("humanResult", "UNKNOWN"), 0) + 1
    auto_rows = [c for c in cases if c.get("autoResult") in (VIOLATION, NORMAL)]
    auto_correct = sum(1 for c in auto_rows if c["autoResult"] == c["humanResult"])
    return {
        "total": len(cases),
        "labels": labels,
        "autoTotal": len(auto_rows),
        "autoAccuracy": round(auto_correct / len(auto_rows), 4) if auto_rows else None,
    }


def _semantic_prompt(state: Dict[str, Any]) -> str:
    return str(state.get("semanticPrompt") or DEFAULT_SEMANTIC_PROMPT)


def _rule_cases(state: Dict[str, Any], rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    return cases_for_rule(state.get("calls") or [], rule, state.get("splits") or {})


def _rebuild_splits(state: Dict[str, Any]) -> None:
    state["splits"] = build_splits(state.get("rules") or [], state.get("calls") or [])


def _ensure_code_unique(state: Dict[str, Any], code: str, exclude_id: Optional[str] = None) -> None:
    for rule in state["rules"]:
        if rule.get("id") == exclude_id:
            continue
        if str(rule.get("code") or "").upper() == code.upper():
            raise HTTPException(status_code=400, detail=f"规则编号 {code} 已被「{rule['name']}」使用")


CONTENT_FIELDS = ("code", "name", "description", "regex", "semantic", "logic")


def _rule_content(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {key: rule.get(key, "") for key in CONTENT_FIELDS}


def _append_change(
    rule: Dict[str, Any],
    reason: str,
    source: str,
    changed: List[str],
    before: Dict[str, Any],
    after: Dict[str, Any],
    run_id: Optional[str] = None,
) -> None:
    entry = {
        "version": int(rule.get("version", 1)),
        "at": now(),
        "source": source,
        "reason": reason,
        "changedFields": changed,
        "before": before,
        "after": after,
    }
    if run_id:
        entry["runId"] = run_id
    log = rule.setdefault("changeLog", [])
    log.append(entry)
    rule["changeLog"] = log[-50:]


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    state = store.snapshot()
    settings = store.settings()
    return {
        "status": "ok",
        "version": "2.6.2",
        "mode": settings.get("mode", "mock"),
        "model": settings.get("model", ""),
        "hasApiKey": bool(RUNTIME.get("apiKey")),
        "calls": len(state.get("calls") or []),
        "rules": len(state.get("rules") or []),
        "runningJobs": sum(1 for job in JOBS.values() if job["status"] == "running"),
    }


@app.get("/api/client-defaults")
async def get_client_defaults() -> Dict[str, Any]:
    path = BASE_DIR / "local-defaults.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@app.get("/api/settings")
async def get_settings() -> Dict[str, Any]:
    return _settings_response(store.settings())


def _config_from_input(payload: SettingsInput) -> Dict[str, Any]:
    protocol = payload.apiProtocol
    headers = normalize_headers(
        [{"name": item.name, "value": item.value} for item in payload.headers]
    )
    template = payload.requestTemplate.strip() or DEFAULT_REQUEST_TEMPLATE
    content_path = payload.contentPath.strip() or DEFAULT_CONTENT_PATH
    if protocol == "custom":
        validate_request_template(template)
        if not content_path:
            raise LLMError("自定义接口必须填写返回内容路径")
        if not template_has_business_input(template):
            raise LLMError(
                "请求体模板必须包含 {{prompt}}、{{user}} 或 {{messages}}，"
                "否则不会上送当前规则和通话文本；单入参接口请用 {{prompt}}"
            )
    return {
        "apiProtocol": protocol,
        "apiPath": payload.apiPath.strip(),
        "headers": headers,
        "requestTemplate": template,
        "contentPath": content_path,
        "violationPath": payload.violationPath.strip(),
        "reasonPath": payload.reasonPath.strip(),
        "timeoutSec": payload.timeoutSec,
        "maxRetries": payload.maxRetries,
        "retryBackoffSec": payload.retryBackoffSec,
        "requestIntervalSec": payload.requestIntervalSec,
    }


@app.put("/api/settings")
async def put_settings(payload: SettingsInput) -> Dict[str, Any]:
    try:
        config = _config_from_input(payload)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.apiKey is not None:
        RUNTIME["apiKey"] = payload.apiKey.strip()
    if payload.mode == "deepseek":
        key = RUNTIME.get("apiKey", "")
        if not key and (config["apiProtocol"] != "custom" or config_needs_api_key(config)):
            raise HTTPException(status_code=400, detail="当前接口配置需要 API Key，请先填写")

    def mutate(state: Dict[str, Any]) -> None:
        state["settings"] = {
            "mode": payload.mode,
            "baseUrl": payload.baseUrl.rstrip("/") or "https://api.deepseek.com",
            "model": payload.model.strip() or "deepseek-chat",
            **config,
        }

    store.mutate(mutate)
    return await get_settings()


@app.post("/api/settings/test")
async def settings_test(payload: SettingsInput) -> Dict[str, Any]:
    try:
        config = _config_from_input(payload)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    key = (payload.apiKey or "").strip() or RUNTIME.get("apiKey", "")
    if not key and (config["apiProtocol"] != "custom" or config_needs_api_key(config)):
        raise HTTPException(status_code=400, detail="未填写 API Key")
    try:
        result = await test_connection(
            payload.baseUrl, payload.model, key, config=config
        )
        return {"ok": True, "message": "连接成功", "result": result}
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/semantic-prompt")
async def get_semantic_prompt() -> Dict[str, Any]:
    state = store.snapshot()
    prompt = _semantic_prompt(state)
    return {
        "prompt": prompt,
        "version": int(state.get("semanticPromptVersion") or 1),
        "history": state.get("promptHistory") or [],
        "preview": render_prompt(prompt, state.get("rules") or []),
    }


@app.put("/api/semantic-prompt")
async def put_semantic_prompt(payload: PromptInput) -> Dict[str, Any]:
    state = store.snapshot()
    current = _semantic_prompt(state)
    try:
        prompt = validate_prompt(payload.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prompt == current:
        return await get_semantic_prompt()
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="修改通用语义提示词必须填写理由")

    def mutate(draft: Dict[str, Any]) -> None:
        version = int(draft.get("semanticPromptVersion") or 1) + 1
        draft["semanticPrompt"] = prompt
        draft["semanticPromptVersion"] = version
        history = draft.setdefault("promptHistory", [])
        history.append({
            "version": version,
            "at": now(),
            "source": "manual",
            "reason": payload.reason.strip(),
            "before": current,
            "after": prompt,
        })
        draft["promptHistory"] = history[-30:]

    store.mutate(mutate)
    return await get_semantic_prompt()


# ---------------------------------------------------------------- 规则
@app.get("/api/rules")
async def list_rules() -> List[Dict[str, Any]]:
    state = store.snapshot()
    result = []
    for rule in state["rules"]:
        summary = _dataset_summary(_rule_cases(state, rule))
        result.append({**rule, "dataset": summary})
    return result


@app.post("/api/rules")
async def create_rule(payload: RuleInput) -> Dict[str, Any]:
    def mutate(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_code_unique(state, payload.code)
        stamp = now()
        rule = {
            "id": "rule-" + uuid.uuid4().hex[:8],
            "code": payload.code,
            "name": payload.name,
            "description": payload.description,
            "regex": payload.regex,
            "semantic": payload.semantic,
            "logic": payload.logic,
            "version": 1,
            "createdAt": stamp,
            "updatedAt": stamp,
            "changeLog": [],
        }
        _append_change(
            rule,
            payload.changeReason or "初始创建",
            "manual",
            list(CONTENT_FIELDS),
            {},
            _rule_content(rule),
        )
        state["rules"].append(rule)
        _rebuild_splits(state)
        return rule

    return store.mutate(mutate)


@app.post("/api/rules/import")
async def import_rules(
    file: UploadFile = File(...),
    mode: str = Form("merge"),
    reason: str = Form("规则文件导入"),
) -> Dict[str, Any]:
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="规则文件不能超过 5 MB")
    try:
        parsed = parse_rule_file(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not parsed["rules"]:
        detail = "没有可导入的规则"
        if parsed["errors"]:
            detail += "：\n" + "\n".join(parsed["errors"][:20])
        raise HTTPException(status_code=400, detail=detail)
    import_reason = (reason or "").strip()[:500] or "规则文件导入"

    def mutate(state: Dict[str, Any]) -> Dict[str, Any]:
        existing = {str(rule.get("code") or "").upper(): rule for rule in state["rules"]}
        created = updated = skipped = 0
        handled: List[str] = []
        for item in parsed["rules"]:
            code = item["code"]
            rule = existing.get(code)
            if rule is None:
                stamp = now()
                rule = {
                    "id": "rule-" + uuid.uuid4().hex[:8],
                    "code": code,
                    "name": item["name"],
                    "description": "",
                    "regex": item["regex"],
                    "semantic": item["semantic"],
                    "logic": item["logic"],
                    "version": 1,
                    "createdAt": stamp,
                    "updatedAt": stamp,
                    "changeLog": [],
                }
                _append_change(rule, import_reason, "import", list(CONTENT_FIELDS), {}, _rule_content(rule))
                state["rules"].append(rule)
                existing[code] = rule
                created += 1
                handled.append(code)
                continue
            before = _rule_content(rule)
            after = {
                "code": code,
                "name": item["name"],
                "description": rule.get("description", ""),
                "regex": item["regex"],
                "semantic": item["semantic"],
                "logic": item["logic"],
            }
            changed = [key for key in CONTENT_FIELDS if before.get(key) != after.get(key)]
            if not changed:
                skipped += 1
                continue
            rule.update(after)
            rule["version"] = int(rule.get("version", 1)) + 1
            rule["updatedAt"] = now()
            _append_change(rule, import_reason, "import", changed, before, after)
            updated += 1
            handled.append(code)
        _rebuild_splits(state)
        return {
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "codes": handled,
            "errors": parsed["errors"],
            "warnings": parsed["warnings"],
            "mode": mode or "merge",
        }

    return store.mutate(mutate)


@app.put("/api/rules/{rule_id}")
async def update_rule(rule_id: str, payload: RuleInput) -> Dict[str, Any]:
    def mutate(state: Dict[str, Any]) -> Dict[str, Any]:
        rule = _rule_or_404(state, rule_id)
        _ensure_code_unique(state, payload.code, exclude_id=rule_id)
        before = _rule_content(rule)
        after = {
            "code": payload.code,
            "name": payload.name,
            "description": payload.description,
            "regex": payload.regex,
            "semantic": payload.semantic,
            "logic": payload.logic,
        }
        changed = [key for key in CONTENT_FIELDS if before.get(key) != after.get(key)]
        if not changed:
            return rule
        if not payload.changeReason:
            raise HTTPException(status_code=400, detail="修改规则内容时必须填写修改理由")
        rule.update(after)
        rule["version"] = int(rule.get("version", 1)) + 1
        rule["updatedAt"] = now()
        _append_change(rule, payload.changeReason, "manual", changed, before, after)
        _rebuild_splits(state)
        return rule

    return store.mutate(mutate)


@app.delete("/api/rules/{rule_id}")
async def delete_rule(rule_id: str) -> Dict[str, Any]:
    if any(job["ruleId"] == rule_id and job["status"] in ("queued", "running") for job in JOBS.values()):
        raise HTTPException(status_code=409, detail="该规则有正在执行的优化任务，请先取消")

    def mutate(state: Dict[str, Any]) -> None:
        _rule_or_404(state, rule_id)
        state["rules"] = [r for r in state["rules"] if r["id"] != rule_id]
        _rebuild_splits(state)

    store.mutate(mutate)
    return {"deleted": rule_id}


class RegexTestInput(BaseModel):
    transcript: str = ""


@app.post("/api/rules/{rule_id}/test-regex")
async def test_regex(rule_id: str, payload: RegexTestInput) -> Dict[str, Any]:
    state = store.snapshot()
    rule = _rule_or_404(state, rule_id)
    flags = rule_flags(rule)
    if not flags["regex"]:
        return {"regexHit": None, "error": "该规则没有配置正则表达式"}
    try:
        hit = bool(re.search(str(rule["regex"]), payload.transcript))
    except re.error as exc:
        return {"regexHit": None, "error": f"正则无法编译：{exc}"}
    return {"regexHit": hit, "error": ""}


# ---------------------------------------------------------------- 数据
@app.get("/api/dataset")
async def get_dataset_overview() -> Dict[str, Any]:
    state = store.snapshot()
    calls = state.get("calls") or []
    rules = []
    for rule in state["rules"]:
        code = rule["code"]
        rules.append({
            "code": code,
            "name": rule["name"],
            "positives": sum(1 for call in calls if code in (call.get("humanRules") or [])),
            "autoPositives": sum(1 for call in calls if code in (call.get("autoRules") or [])),
        })
    return {
        "total": len(calls),
        "callsWithHumanViolation": sum(1 for call in calls if call.get("humanRules")),
        "callsWithAutoViolation": sum(1 for call in calls if call.get("autoRules")),
        "rules": rules,
    }


@app.get("/api/rules/{rule_id}/dataset")
async def get_dataset(rule_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    state = store.snapshot()
    rule = _rule_or_404(state, rule_id)
    cases = _rule_cases(state, rule)
    limit = max(1, min(limit, 500))
    return {
        **_dataset_summary(cases),
        "ruleCode": rule["code"],
        "callTotal": len(state.get("calls") or []),
        "splitSummary": split_summary(cases),
        "rows": cases[offset:offset + limit],
        "offset": offset,
        "limit": limit,
    }


@app.post("/api/dataset/import")
async def import_dataset(file: UploadFile = File(...), mode: str = Form("replace")) -> Dict[str, Any]:
    if mode not in ("replace", "append"):
        raise HTTPException(status_code=400, detail="mode 只能是 replace 或 append")
    state = store.snapshot()
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件超过 10MB")
    try:
        calls, report = build_calls(file.filename or "upload.csv", content, state["rules"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def mutate(draft: Dict[str, Any]) -> int:
        existing = draft.get("calls") or []
        if mode == "replace":
            merged = calls
        else:
            seen = {call["callId"] for call in existing}
            merged = existing + [call for call in calls if call["callId"] not in seen]
        merged = merged[:MAX_DATASET]
        draft["calls"] = merged
        _rebuild_splits(draft)
        return len(merged)

    total = store.mutate(mutate)
    return {"report": report, "callTotal": total}


@app.delete("/api/dataset")
async def clear_dataset() -> Dict[str, Any]:
    def mutate(state: Dict[str, Any]) -> int:
        state["calls"] = []
        return 0

    store.mutate(mutate)
    return {"callTotal": 0}


@app.get("/api/dataset/export.csv")
async def export_dataset() -> Response:
    state = store.snapshot()
    csv_text = calls_to_csv(state.get("calls") or [])
    filename = quote("通话导入数据.csv")
    return Response(
        content="\ufeff" + csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


SAMPLE_PATH = BASE_DIR / "samples" / "import-calls-10.csv"


@app.get("/api/dataset/sample.csv")
async def sample_dataset() -> Response:
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail="示例文件不存在")
    text = SAMPLE_PATH.read_text(encoding="utf-8-sig")
    filename = quote("通话导入示例-10通.csv")
    return Response(
        content="\ufeff" + text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


# ---------------------------------------------------------------- 优化任务
def _job_public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in job.items() if not key.endswith("Rows") and key != "cancelRequested"}


def _append_job_file(job: Dict[str, Any], text: str) -> None:
    job_id = str(job.get("id") or "")
    if not job_id:
        return
    try:
        with (JOB_LOG_DIR / f"{job_id}.log").open("a", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass


def _make_llm_trace(job: Dict[str, Any]):
    def trace(event: Dict[str, Any]) -> None:
        kind = event.get("event")
        call = event.get("call")
        attempt = event.get("attempt")
        if kind == "request":
            _append_job_file(job, (
                f"{now()} [LLM REQUEST] call={call} attempt={attempt} "
                f"endpoint={event.get('endpoint')} judge={event.get('judge')} "
                f"systemChars={event.get('systemChars')} userChars={event.get('userChars')} "
                f"promptChars={event.get('promptChars')} "
                f"containsTranscript={event.get('containsTranscript')}\n"
                f"TEMPLATE: {event.get('template') or ''}\n"
                f"HEADERS: {json.dumps(event.get('headers') or {}, ensure_ascii=False)}\n"
                f"BODY: {event.get('payloadText') or ''}\n"
            ))
        elif kind == "response":
            status = int(event.get("status") or 0)
            body = str(event.get("body") or "")
            _append_job_file(job, (
                f"{now()} [LLM RESPONSE] call={call} attempt={attempt} "
                f"status={status} durationMs={event.get('durationMs')} "
                f"bodyChars={len(body)} truncated={event.get('bodyTruncated')}\n"
                f"BODY: {body}\n"
            ))
            _emit(job, {
                "type": "log",
                "level": "error" if status >= 400 else "info",
                "message": (
                    f"LLM #{call} 第{attempt}次 HTTP {status}，"
                    f"{event.get('durationMs')}ms，响应 {len(body)} 字符"
                ),
            })
        elif kind == "error":
            message = str(event.get("error") or "")
            _append_job_file(job, (
                f"{now()} [LLM ERROR] call={call} attempt={attempt} "
                f"durationMs={event.get('durationMs')}\nERROR: {message}\n"
            ))
            _emit(job, {
                "type": "log",
                "level": "error",
                "message": f"LLM #{call} 第{attempt}次调用失败：{message[:300]}",
            })
    return trace


def _emit(job: Dict[str, Any], event: Dict[str, Any]) -> None:
    kind = event.get("type")
    line = ""
    if kind == "log":
        level = str(event.get("level") or "info")
        entry = {"at": now(), "level": level, "message": str(event.get("message") or "")}
        job.setdefault("logs", []).append(entry)
        job["logs"] = job["logs"][-300:]
        line = f"[{level.upper()}] {entry['message']}"
    elif kind == "phase":
        job["phase"] = str(event.get("phase") or "")
        job["progress"] = {"done": int(event.get("done") or 0), "total": int(event.get("total") or 0)}
        line = f"[PHASE] {job['phase']}"
    elif kind == "progress":
        job["progress"] = {"done": int(event.get("done") or 0), "total": int(event.get("total") or 0)}
    elif kind == "baseline":
        job["baseline"] = event.get("metrics")
        job["baselineRule"] = event.get("rule")
    elif kind == "iteration":
        iteration = dict(event.get("iteration") or {})
        train_rows = iteration.pop("trainRows", None)
        validation_rows = iteration.pop("validationRows", None)
        job.setdefault("iterations", []).append(iteration)
        if train_rows is not None or validation_rows is not None:
            job.setdefault("iterationRows", {})[str(iteration.get("index"))] = {
                TRAIN: train_rows or [],
                VALIDATION: validation_rows or [],
            }
        line = (
            f"[ITER] 第{iteration.get('index')}轮 训练={iteration.get('metrics', {}).get('accuracy')} "
            f"验证={iteration.get('validationMetrics', {}).get('accuracy')} "
            f"采纳={iteration.get('accepted')} 说明={iteration.get('rationale', '')[:200]}"
        )
    elif kind == "best":
        job["best"] = event.get("metrics")
    job["updatedAt"] = now()
    if line:
        _append_job_file(job, f"{now()} {line}")


def _sample_cases(cases: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(cases) <= limit:
        return list(cases)
    ordered = sorted(cases, key=lambda c: str(c.get("callId") or ""))
    violations = [c for c in ordered if c.get("humanResult") == VIOLATION]
    normals = [c for c in ordered if c.get("humanResult") == NORMAL]
    picked: List[Dict[str, Any]] = []
    index_v = index_n = 0
    while len(picked) < limit and (index_v < len(violations) or index_n < len(normals)):
        if index_v < len(violations) and len(picked) < limit:
            picked.append(violations[index_v])
            index_v += 1
        if index_n < len(normals) and len(picked) < limit:
            picked.append(normals[index_n])
            index_n += 1
    return picked


def _save_run(job: Dict[str, Any]) -> None:
    run = {
        "id": job["id"],
        "ruleId": job["ruleId"],
        "ruleName": job["ruleName"],
        "kind": job.get("kind", "optimize"),
        "status": job["status"],
        "target": job.get("target"),
        "maxRounds": job.get("maxRounds"),
        "baseline": job.get("baseline"),
        "baselineValidation": job.get("baselineValidation"),
        "baselineTest": job.get("baselineTest"),
        "best": job.get("best"),
        "bestValidation": job.get("bestValidation"),
        "bestTest": job.get("bestTest"),
        "gate": job.get("gate"),
        "splitSummary": job.get("splitSummary"),
        "baselineRule": job.get("baselineRule"),
        "bestRule": job.get("bestRule"),
        "baselinePrompt": job.get("baselinePrompt"),
        "bestPrompt": job.get("bestPrompt"),
        "promptChanged": job.get("promptChanged"),
        "iterations": job.get("iterations", []),
        "createdAt": job["createdAt"],
        "finishedAt": job.get("finishedAt"),
        "applied": False,
        "applyReason": "",
    }

    def mutate(state: Dict[str, Any]) -> None:
        state["runs"].insert(0, run)
        del state["runs"][50:]

    store.mutate(mutate)


def _make_judge_factory(llm: Optional[LLMClient], all_rules: List[Dict[str, Any]]):
    def factory(prompt: str, with_reason: bool):
        if llm is None:
            async def mock_judge(case: Dict[str, Any], rule: Dict[str, Any]):
                return False, ("模拟模式：语义部分未调用模型" if with_reason else "")

            return mock_judge

        async def judge(case: Dict[str, Any], rule: Dict[str, Any]):
            return await judge_semantic(llm, case, rule, prompt, all_rules, with_reason)

        return judge

    return factory


async def _execute_optimize(
    job: Dict[str, Any],
    rule: Dict[str, Any],
    cases: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> None:
    llm: Optional[LLMClient] = None
    job["status"] = "running"
    job["startedAt"] = now()
    _emit(job, {"type": "log", "message": f"开始优化：{job['ruleName']}，样本 {len(cases)} 条，目标准确率 {job['target']:.0%}"})
    try:
        state = store.snapshot()
        all_rules = state.get("rules") or []
        prompt = _semantic_prompt(state)
        job["baselinePrompt"] = prompt
        job["bestPrompt"] = prompt
        if settings.get("mode") == "deepseek":
            key = RUNTIME.get("apiKey", "")
            config = _api_config(settings)
            if not key and (config["apiProtocol"] != "custom" or config_needs_api_key(config)):
                raise LLMError("未配置 API Key，无法调用真实模型")
            llm = LLMClient(
                settings.get("baseUrl", ""),
                settings.get("model", ""),
                key,
                config=config,
                trace=_make_llm_trace(job),
            )
            _emit(job, {"type": "log", "message": f"大模型调用参数：{llm.describe()}"})
            proposer = LLMProposer(llm)
        else:
            proposer = MockProposer()

        result = await optimize(
            rule, cases, job["target"], job["maxRounds"],
            _make_judge_factory(llm, all_rules), proposer, prompt,
            emit=lambda event: _emit(job, event),
            is_cancelled=lambda: bool(job.get("cancelRequested")),
        )
        job["status"] = result["status"]
        job["baseline"] = result["baseline"]["train"]["metrics"]
        job["baselineValidation"] = result["baseline"]["validation"]["metrics"]
        job["baselineTest"] = result["baseline"]["test"]["metrics"]
        job["best"] = result["best"]["train"]["metrics"]
        job["bestValidation"] = result["best"]["validation"]["metrics"]
        job["bestTest"] = result["best"]["test"]["metrics"]
        job["baselineRule"] = result["baseline"]["test"]["rule"]
        job["bestRule"] = result["best"]["test"]["rule"]
        for split, name in ((TRAIN, "Train"), (VALIDATION, "Validation"), (TEST, "Test")):
            job[f"baseline{name}Rows"] = result["baseline"][split.lower()]["rows"]
            job[f"best{name}Rows"] = result["best"][split.lower()]["rows"]
        job["baselineRows"] = result["baseline"]["test"]["rows"]
        job["bestRows"] = result["best"]["test"]["rows"]
        job["iterations"] = []
        job["iterationRows"] = {}
        for item in result["iterations"]:
            clean = dict(item)
            train_rows = clean.pop("trainRows", [])
            validation_rows = clean.pop("validationRows", [])
            job["iterations"].append(clean)
            job["iterationRows"][str(clean.get("index"))] = {
                TRAIN: train_rows or [],
                VALIDATION: validation_rows or [],
            }
        job["gate"] = result["gate"]
        job["splitSummary"] = result["splitSummary"]
        job["baselinePrompt"] = result["baselinePrompt"]
        job["bestPrompt"] = result["bestPrompt"]
        job["promptChanged"] = result["promptChanged"]
        _emit(job, {"type": "log", "message": (
            f"优化结束：{job['status']}；验证集 {job['bestValidation']['accuracy']:.1%}，"
            f"测试集 {job['bestTest']['accuracy']:.1%}；泛化验收 {job['gate']['status']}"
        )})
    except JobCancelled:
        job["status"] = "cancelled"
        _emit(job, {"type": "log", "message": "任务已取消"})
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        _emit(job, {"type": "log", "level": "info", "message": "任务已取消，正在中断的模型请求已停止"})
    except (LLMError, EvaluationError) as exc:
        job["status"] = "failed"
        job["error"] = str(exc)[:1500]
        job["errorDetails"] = list(getattr(exc, "details", []) or [])
        _emit(job, {"type": "log", "level": "error", "message": f"任务失败：{exc}"})
        for item in job["errorDetails"][:10]:
            _emit(job, {
                "type": "log",
                "level": "error",
                "message": f"失败样本 {item.get('callId') or '-'}：{item.get('error') or ''}",
            })
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = f"内部错误：{exc}"[:500]
        _emit(job, {"type": "log", "level": "error", "message": f"内部错误：{exc}"})
    finally:
        if llm is not None:
            job["llmStats"] = dict(llm.stats)
            await llm.aclose()
        job["finishedAt"] = now()
        job["progress"] = {"done": 0, "total": 0}
        job["phase"] = {
            "reached": "优化完成",
            "exhausted": "优化结束",
            "cancelled": "已取消",
            "failed": "任务失败",
        }.get(job["status"], "已结束")
        if job.get("llmStats"):
            stats = job["llmStats"]
            _emit(job, {
                "type": "log",
                "message": (
                    f"AI 调用统计：请求 {stats.get('calls', 0)} 次，"
                    f"尝试 {stats.get('attempts', 0)} 次，重试 {stats.get('retries', 0)} 次，"
                    f"失败 {stats.get('errors', 0)} 次"
                ),
            })
        if job.get("baseline") and job.get("kind") == "optimize":
            try:
                _save_run(job)
            except Exception as exc:  # noqa: BLE001
                _emit(job, {"type": "log", "level": "error", "message": f"运行记录保存失败：{exc}"})


async def _execute_evaluate(
    job: Dict[str, Any],
    rule: Dict[str, Any],
    cases: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> None:
    llm: Optional[LLMClient] = None
    job["status"] = "running"
    job["startedAt"] = now()
    try:
        state = store.snapshot()
        all_rules = state.get("rules") or []
        prompt = _semantic_prompt(state)
        job["baselinePrompt"] = prompt
        job["bestPrompt"] = prompt
        if settings.get("mode") == "deepseek":
            key = RUNTIME.get("apiKey", "")
            config = _api_config(settings)
            if not key and (config["apiProtocol"] != "custom" or config_needs_api_key(config)):
                raise LLMError("未配置 API Key，无法调用真实模型")
            llm = LLMClient(
                settings.get("baseUrl", ""),
                settings.get("model", ""),
                key,
                config=config,
                trace=_make_llm_trace(job),
            )
            _emit(job, {"type": "log", "message": f"大模型调用参数：{llm.describe()}"})

        grouped = split_cases(cases)
        if not grouped[TRAIN] or not grouped[VALIDATION] or not grouped[TEST]:
            raise EvaluationError("训练集、验证集、测试集不完整，无法评测；请补充通话数据")
        factory = _make_judge_factory(llm, all_rules)
        results = {}
        configs = (
            (TRAIN, "训练集（含语义理由）", True, True),
            (VALIDATION, "验证集（仅判定）", False, False),
            (TEST, "测试集（仅判定）", False, False),
        )
        for split, label, force_parts, with_reason in configs:
            _emit(job, {"type": "phase", "phase": f"评测 {label}", "done": 0, "total": len(grouped[split])})
            results[split] = await evaluate_rule(
                rule,
                grouped[split],
                factory(prompt, with_reason),
                progress=lambda done, total: _emit(job, {"type": "progress", "done": done, "total": total}),
                is_cancelled=lambda: bool(job.get("cancelRequested")),
                force_parts=force_parts,
                with_reason=with_reason,
                cache_scope=f"{prompt}\x00{'reason' if with_reason else 'verdict'}",
            )
        job["status"] = "evaluated"
        job["baseline"] = results[TRAIN]["metrics"]
        job["baselineValidation"] = results[VALIDATION]["metrics"]
        job["baselineTest"] = results[TEST]["metrics"]
        job["baselineRule"] = results[TRAIN]["rule"]
        job["baselineRows"] = results[TEST]["rows"]
        job["best"] = results[TRAIN]["metrics"]
        job["bestValidation"] = results[VALIDATION]["metrics"]
        job["bestTest"] = results[TEST]["metrics"]
        job["bestRule"] = results[TRAIN]["rule"]
        job["bestRows"] = results[TEST]["rows"]
        for split, name in ((TRAIN, "Train"), (VALIDATION, "Validation"), (TEST, "Test")):
            job[f"baseline{name}Rows"] = results[split]["rows"]
            job[f"best{name}Rows"] = results[split]["rows"]
        job["splitSummary"] = split_summary(cases)
        _emit(job, {"type": "log", "message": (
            f"评测完成：训练集 {results[TRAIN]['metrics']['accuracy']:.1%}，"
            f"验证集 {results[VALIDATION]['metrics']['accuracy']:.1%}，"
            f"测试集 {results[TEST]['metrics']['accuracy']:.1%}"
        )})
    except JobCancelled:
        job["status"] = "cancelled"
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        _emit(job, {"type": "log", "level": "info", "message": "任务已取消，正在中断的模型请求已停止"})
    except (LLMError, EvaluationError) as exc:
        job["status"] = "failed"
        job["error"] = str(exc)[:1500]
        job["errorDetails"] = list(getattr(exc, "details", []) or [])
        _emit(job, {"type": "log", "level": "error", "message": f"评测失败：{exc}"})
        for item in job["errorDetails"][:10]:
            _emit(job, {
                "type": "log",
                "level": "error",
                "message": f"失败样本 {item.get('callId') or '-'}：{item.get('error') or ''}",
            })
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = f"内部错误：{exc}"[:500]
        _emit(job, {"type": "log", "level": "error", "message": f"内部错误：{exc}"})
    finally:
        if llm is not None:
            job["llmStats"] = dict(llm.stats)
            await llm.aclose()
        job["finishedAt"] = now()
        job["progress"] = {"done": 0, "total": 0}
        job["phase"] = {
            "evaluated": "评测完成",
            "cancelled": "已取消",
            "failed": "评测失败",
        }.get(job["status"], "已结束")
        if job.get("llmStats"):
            stats = job["llmStats"]
            _emit(job, {
                "type": "log",
                "message": (
                    f"AI 调用统计：请求 {stats.get('calls', 0)} 次，"
                    f"尝试 {stats.get('attempts', 0)} 次，重试 {stats.get('retries', 0)} 次，"
                    f"失败 {stats.get('errors', 0)} 次"
                ),
            })


@app.post("/api/rules/{rule_id}/evaluate")
async def evaluate_current(rule_id: str) -> Dict[str, Any]:
    state = store.snapshot()
    rule = _rule_or_404(state, rule_id)
    cases = _rule_cases(state, rule)
    if len(cases) < 5:
        raise HTTPException(status_code=400, detail="至少需要 5 通已导入的通话才能评测")
    grouped = split_cases(cases)
    if not grouped[TRAIN] or not grouped[VALIDATION] or not grouped[TEST]:
        raise HTTPException(status_code=400, detail="数据太少，无法切分训练/验证/测试集；请再导入一些通话")
    settings = store.settings()
    _ensure_llm_ready(settings)
    _ensure_no_active_job()
    job_id = "job-" + uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "kind": "evaluate",
        "ruleId": rule_id,
        "ruleCode": rule.get("code", ""),
        "ruleName": rule["name"],
        "status": "queued",
        "phase": "排队中",
        "target": None,
        "maxRounds": 0,
        "sampleCount": len(cases),
        "createdAt": now(),
        "updatedAt": now(),
        "progress": {"done": 0, "total": len(cases)},
        "splitSummary": split_summary(cases),
        "logs": [],
        "iterations": [],
        "baseline": None,
        "best": None,
        "baselineRule": None,
        "bestRule": None,
        "error": None,
        "cancelRequested": False,
    }
    JOBS[job_id] = job
    _cleanup_jobs()
    TASKS[job_id] = asyncio.create_task(_execute_evaluate(job, dict(rule), cases, settings))
    return {"jobId": job_id}


def _ensure_llm_ready(settings: Dict[str, Any]) -> None:
    if settings.get("mode") != "deepseek":
        return
    config = _api_config(settings)
    key = RUNTIME.get("apiKey", "")
    if not key and (config["apiProtocol"] != "custom" or config_needs_api_key(config)):
        raise HTTPException(status_code=400, detail="当前是真实模型模式，但还没有配置 API Key")
    if config["apiProtocol"] == "custom":
        try:
            validate_request_template(config["requestTemplate"])
            if not template_has_business_input(config["requestTemplate"]):
                raise LLMError(
                    "请求体模板没有包含 {{prompt}}、{{user}} 或 {{messages}}，"
                    "不会上送规则和通话文本，请重新配置"
                )
        except LLMError as exc:
            raise HTTPException(status_code=400, detail=f"自定义接口配置无效：{exc}") from exc


def _new_optimize_job(
    rule: Dict[str, Any],
    sampled: List[Dict[str, Any]],
    target: float,
    max_rounds: int,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = "job-" + uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "kind": "optimize",
        "ruleId": rule["id"],
        "ruleCode": rule.get("code", ""),
        "ruleName": rule["name"],
        "status": "queued",
        "phase": "排队中",
        "target": target,
        "maxRounds": max_rounds,
        "sampleCount": len(sampled),
        "createdAt": now(),
        "updatedAt": now(),
        "progress": {"done": 0, "total": len(sampled)},
        "splitSummary": split_summary(sampled),
        "logs": [],
        "iterations": [],
        "baseline": None,
        "best": None,
        "baselineRule": None,
        "bestRule": None,
        "error": None,
        "cancelRequested": False,
    }
    if parent_id:
        job["parentId"] = parent_id
    return job


@app.post("/api/optimize")
async def start_optimize(payload: OptimizeInput) -> Dict[str, Any]:
    state = store.snapshot()
    rule = _rule_or_404(state, payload.ruleId)
    cases = _rule_cases(state, rule)
    if len(cases) < 5:
        raise HTTPException(status_code=400, detail="至少需要 5 通已导入的通话才能优化")
    labels = {case.get("humanResult") for case in cases}
    if VIOLATION not in labels or NORMAL not in labels:
        raise HTTPException(status_code=400, detail="数据需要同时包含人工判定的「违规」和「正常」样本")
    settings = store.settings()
    _ensure_llm_ready(settings)

    grouped = split_cases(cases)
    if not grouped[TRAIN] or not grouped[VALIDATION] or not grouped[TEST]:
        raise HTTPException(status_code=400, detail="数据太少，无法切分训练/验证/测试集；请再导入一些通话")
    sampled = (
        _sample_cases(grouped[TRAIN], payload.sampleLimit)
        + grouped[VALIDATION]
        + grouped[TEST]
    )
    _ensure_no_active_job()
    job = _new_optimize_job(rule, sampled, payload.targetAccuracy, payload.maxRounds)
    JOBS[job["id"]] = job
    _cleanup_jobs()
    TASKS[job["id"]] = asyncio.create_task(_execute_optimize(job, dict(rule), sampled, settings))
    return {"jobId": job["id"]}


async def _execute_batch(
    parent: Dict[str, Any],
    rules: List[Dict[str, Any]],
    target: float,
    max_rounds: int,
    sample_limit: int,
    settings: Dict[str, Any],
) -> None:
    parent["status"] = "running"
    parent["startedAt"] = now()
    total = len(rules)
    _emit(parent, {"type": "log", "message": f"开始批量优化：{total} 条规则，目标 {target:.0%}"})
    try:
        state = store.snapshot()
        for index, rule in enumerate(rules):
            if parent.get("cancelRequested"):
                raise JobCancelled()
            position = index + 1
            parent["currentRule"] = f"{rule.get('code', '')} {rule.get('name', '')}".strip()
            parent["phase"] = f"批量优化 {position}/{total}：{parent['currentRule']}"
            _emit(parent, {"type": "log", "message": parent["phase"]})
            cases = _rule_cases(state, rule)
            grouped = split_cases(cases)
            labels = {case.get("humanResult") for case in cases}
            if (
                len(cases) < 5
                or VIOLATION not in labels
                or NORMAL not in labels
                or not grouped[TRAIN]
                or not grouped[VALIDATION]
                or not grouped[TEST]
            ):
                result = {
                    "ruleId": rule["id"],
                    "ruleCode": rule.get("code", ""),
                    "ruleName": rule.get("name", ""),
                    "status": "skipped",
                    "error": "样本不足，或人工标签未同时包含违规/正常，或无法切分三集",
                }
                parent["ruleResults"].append(result)
                parent["progress"] = {"done": position, "total": total}
                _emit(parent, {"type": "log", "level": "error", "message": f"{parent['currentRule']} 已跳过：{result['error']}"})
                continue

            sampled = (
                _sample_cases(grouped[TRAIN], sample_limit)
                + grouped[VALIDATION]
                + grouped[TEST]
            )
            child = _new_optimize_job(rule, sampled, target, max_rounds, parent_id=parent["id"])
            JOBS[child["id"]] = child
            parent["currentJobId"] = child["id"]
            _emit(parent, {"type": "log", "message": f"{parent['currentRule']} 开始，任务 {child['id']}"})
            await _execute_optimize(child, dict(rule), sampled, settings)
            gate = child.get("gate") or {}
            result = {
                "jobId": child["id"],
                "ruleId": rule["id"],
                "ruleCode": rule.get("code", ""),
                "ruleName": rule.get("name", ""),
                "status": child.get("status"),
                "error": child.get("error"),
                "gate": gate.get("status") if isinstance(gate, dict) else None,
                "bestValidation": (child.get("bestValidation") or {}).get("accuracy"),
                "bestTest": (child.get("bestTest") or {}).get("accuracy"),
            }
            parent["ruleResults"].append(result)
            child_stats = child.get("llmStats") or {}
            if child_stats:
                totals = parent.setdefault("llmStats", {"calls": 0, "attempts": 0, "retries": 0, "errors": 0})
                for key in ("calls", "attempts", "retries", "errors"):
                    totals[key] = int(totals.get(key, 0)) + int(child_stats.get(key, 0) or 0)
            parent["progress"] = {"done": position, "total": total}
            _emit(parent, {
                "type": "log",
                "level": "error" if child.get("status") == "failed" else "info",
                "message": (
                    f"{parent['currentRule']} 结束：{child.get('status')}，"
                    f"验证 {result['bestValidation']}，测试 {result['bestTest']}"
                ),
            })
            if parent.get("cancelRequested"):
                break

        if parent.get("cancelRequested"):
            parent["status"] = "cancelled"
        else:
            parent["status"] = "batch_done"
    except JobCancelled:
        parent["status"] = "cancelled"
        _emit(parent, {"type": "log", "level": "info", "message": "批量优化已取消"})
    except asyncio.CancelledError:
        parent["status"] = "cancelled"
        _emit(parent, {"type": "log", "level": "info", "message": "批量优化已取消，当前模型请求已中断"})
    except Exception as exc:  # noqa: BLE001
        parent["status"] = "failed"
        parent["error"] = str(exc)[:1500]
        _emit(parent, {"type": "log", "level": "error", "message": f"批量优化失败：{exc}"})
    finally:
        parent["finishedAt"] = now()
        parent["progress"] = {"done": len(parent.get("ruleResults") or []), "total": total}
        parent["phase"] = {
            "batch_done": "批量优化完成",
            "cancelled": "批量优化已取消",
            "failed": "批量优化失败",
        }.get(parent["status"], "批量优化结束")
        _emit(parent, {"type": "log", "message": parent["phase"]})


@app.post("/api/optimize/batch")
async def start_batch_optimize(payload: BatchOptimizeInput) -> Dict[str, Any]:
    state = store.snapshot()
    if payload.ruleIds:
        by_id = {rule["id"]: rule for rule in state.get("rules") or []}
        missing = [rule_id for rule_id in payload.ruleIds if rule_id not in by_id]
        if missing:
            raise HTTPException(status_code=400, detail="规则不存在：" + "、".join(missing))
        rules = [by_id[rule_id] for rule_id in payload.ruleIds]
    else:
        rules = list(state.get("rules") or [])
    if not rules:
        raise HTTPException(status_code=400, detail="没有可优化的规则")
    settings = store.settings()
    _ensure_llm_ready(settings)
    _ensure_no_active_job()
    parent = {
        "id": "job-" + uuid.uuid4().hex[:8],
        "kind": "batch",
        "status": "queued",
        "phase": "排队中",
        "target": payload.targetAccuracy,
        "maxRounds": payload.maxRounds,
        "sampleLimit": payload.sampleLimit,
        "ruleIds": [rule["id"] for rule in rules],
        "ruleResults": [],
        "currentRule": "",
        "currentJobId": None,
        "progress": {"done": 0, "total": len(rules)},
        "createdAt": now(),
        "updatedAt": now(),
        "logs": [],
        "error": None,
        "cancelRequested": False,
    }
    JOBS[parent["id"]] = parent
    _cleanup_jobs()
    TASKS[parent["id"]] = asyncio.create_task(_execute_batch(
        parent,
        [dict(rule) for rule in rules],
        payload.targetAccuracy,
        payload.maxRounds,
        payload.sampleLimit,
        settings,
    ))
    return {"jobId": parent["id"]}


def _cleanup_jobs() -> None:
    finished = sorted(
        (job for job in JOBS.values() if job["status"] not in ("queued", "running")),
        key=lambda job: job.get("finishedAt") or "",
    )
    for job in finished[:max(0, len(finished) - 30)]:
        JOBS.pop(job["id"], None)
        TASKS.pop(job["id"], None)


def _active_job() -> Optional[Dict[str, Any]]:
    for job in JOBS.values():
        if job.get("status") in ("queued", "running"):
            return job
    return None


def _ensure_no_active_job() -> None:
    job = _active_job()
    if job:
        raise HTTPException(
            status_code=400,
            detail=(
                f"已有任务正在运行：{job.get('ruleName', '')}（{job.get('id')}）。"
                "请到「任务中心」查看，或等待/取消后再开始。"
            ),
        )


def _job_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": job.get("id"),
        "kind": job.get("kind"),
        "ruleId": job.get("ruleId"),
        "ruleCode": job.get("ruleCode", ""),
        "ruleName": job.get("ruleName", ""),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "progress": job.get("progress") or {"done": 0, "total": 0},
        "target": job.get("target"),
        "sampleCount": job.get("sampleCount"),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "error": job.get("error"),
        "llmStats": job.get("llmStats") or {},
        "hasCases": bool(job.get("baselineRows")),
        "ruleResults": job.get("ruleResults") if job.get("kind") == "batch" else None,
        "currentRule": job.get("currentRule"),
        "currentJobId": job.get("currentJobId"),
        "ruleIds": job.get("ruleIds") if job.get("kind") == "batch" else None,
    }


@app.get("/api/jobs")
async def list_jobs() -> Dict[str, Any]:
    roots = [job for job in JOBS.values() if not job.get("parentId")]
    active = [job for job in roots if job.get("status") in ("queued", "running")]
    finished = [job for job in roots if job.get("status") not in ("queued", "running")]
    active.sort(key=lambda job: job.get("createdAt") or "", reverse=True)
    finished.sort(key=lambda job: job.get("createdAt") or "", reverse=True)
    jobs = active + finished
    return {
        "jobs": [_job_summary(job) for job in jobs[:50]],
        "running": len(active),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    payload = _job_public(job)
    payload["hasCases"] = bool(job.get("baselineRows"))
    return payload


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(job_id: str) -> PlainTextResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    path = JOB_LOG_DIR / f"{job_id}.log"
    if path.exists():
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        content = "\n".join(f"{item.get('at')} {item.get('message')}" for item in job.get("logs") or [])
    return PlainTextResponse(
        content or "（暂无日志）",
        headers={"Content-Disposition": f"attachment; filename={job_id}.log"},
    )


@app.get("/api/jobs/{job_id}/iterations/{index}/cases")
async def get_iteration_cases(job_id: str, index: int, split: str = "TRAIN") -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    split = split.upper()
    if split not in (TRAIN, VALIDATION):
        raise HTTPException(status_code=400, detail="每轮逐条结果只支持 TRAIN / VALIDATION")
    rows = ((job.get("iterationRows") or {}).get(str(index)) or {}).get(split)
    if rows is None:
        raise HTTPException(status_code=404, detail="该轮没有逐条结果")
    return {
        "index": index,
        "split": split,
        "withReason": split == TRAIN,
        "rows": rows,
    }


@app.get("/api/jobs/{job_id}/cases")
async def get_job_cases(job_id: str, which: str = "best", split: str = "TEST") -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    split = split.upper()
    if split not in SPLITS:
        raise HTTPException(status_code=400, detail="split 只能是 TRAIN / VALIDATION / TEST")
    name = {"TRAIN": "Train", "VALIDATION": "Validation", "TEST": "Test"}[split]
    prefix = "baseline" if which == "baseline" else "best"
    rows = job.get(f"{prefix}{name}Rows") or []
    metrics_key = prefix if split == TRAIN else f"{prefix}{name}"
    return {
        "which": which,
        "split": split,
        "withReason": split == TRAIN,
        "metrics": job.get(metrics_key),
        "rule": job.get("baselineRule" if which == "baseline" else "bestRule"),
        "rows": rows,
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if job["status"] not in ("queued", "running"):
        return {"ok": False, "status": job["status"]}
    job["cancelRequested"] = True
    job["phase"] = "正在取消"
    _emit(job, {"type": "log", "level": "info", "message": "收到取消请求，正在中断当前模型请求"})
    task = TASKS.get(job_id)
    if task and not task.done():
        task.cancel()
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finishedAt"] = now()
            job["phase"] = "已取消"
        return {"ok": True, "status": "cancelling"}
    if job["status"] in ("queued", "running"):
        job["status"] = "cancelled"
        job["finishedAt"] = now()
        job["phase"] = "已取消"
        _emit(job, {"type": "log", "level": "info", "message": "任务已取消"})
    return {"ok": True, "status": job["status"]}


# ---------------------------------------------------------------- 运行记录
@app.get("/api/runs")
async def list_runs(ruleId: Optional[str] = None) -> List[Dict[str, Any]]:
    state = store.snapshot()
    runs = state.get("runs", [])
    if ruleId:
        runs = [run for run in runs if run.get("ruleId") == ruleId]
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    state = store.snapshot()
    for run in state.get("runs", []):
        if run["id"] == run_id:
            return run
    job = JOBS.get(run_id)
    if job:
        return _job_public(job)
    raise HTTPException(status_code=404, detail="记录不存在")


@app.post("/api/runs/{run_id}/apply")
async def apply_run(run_id: str, payload: Optional[ApplyInput] = None) -> Dict[str, Any]:
    reason = (payload.reason if payload else "").strip()
    force = bool(payload.force) if payload else False
    if not reason:
        raise HTTPException(status_code=400, detail="应用修改前必须填写修改理由")

    state = store.snapshot()
    run = next((item for item in state.get("runs", []) if item["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    gate = run.get("gate") or {}
    if gate.get("status") != "PASS" and not force:
        reasons = "；".join(gate.get("reasons") or ["泛化验收未通过"])
        raise HTTPException(status_code=409, detail=f"泛化验收未通过，不能直接应用：{reasons}")
    best_rule = run.get("bestRule") or {}
    if not best_rule.get("regex") and not best_rule.get("semantic"):
        raise HTTPException(status_code=400, detail="该记录没有可应用的规则内容")

    def mutate(draft: Dict[str, Any]) -> Dict[str, Any]:
        rule = _rule_or_404(draft, run["ruleId"])
        before = _rule_content(rule)
        after = {
            "code": rule.get("code"),
            "name": rule.get("name"),
            "description": rule.get("description"),
            "regex": best_rule.get("regex", ""),
            "semantic": best_rule.get("semantic", ""),
            "logic": best_rule.get("logic", "OR"),
        }
        changed = [key for key in CONTENT_FIELDS if before.get(key) != after.get(key)]
        prompt_before = _semantic_prompt(draft)
        prompt_after = str(run.get("bestPrompt") or prompt_before)
        prompt_changed = prompt_after != prompt_before
        if not changed and not prompt_changed:
            raise HTTPException(status_code=409, detail="候选规则与当前规则一致，无需应用")
        if changed:
            rule.update(after)
            rule["version"] = int(rule.get("version", 1)) + 1
            rule["updatedAt"] = now()
            _append_change(rule, reason, "optimize", changed, before, after, run_id=run_id)
        if prompt_changed:
            version = int(draft.get("semanticPromptVersion") or 1) + 1
            draft["semanticPrompt"] = prompt_after
            draft["semanticPromptVersion"] = version
            history = draft.setdefault("promptHistory", [])
            history.append({
                "version": version,
                "at": now(),
                "source": "optimize",
                "reason": reason,
                "before": prompt_before,
                "after": prompt_after,
                "runId": run_id,
            })
            draft["promptHistory"] = history[-30:]
        for item in draft.get("runs", []):
            if item["id"] == run_id:
                item["applied"] = True
                item["applyReason"] = reason
                item["forced"] = force and gate.get("status") != "PASS"
                item["promptApplied"] = prompt_changed
        return rule

    return store.mutate(mutate)


# ---------------------------------------------------------------- 静态页面
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
