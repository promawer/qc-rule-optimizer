"""单文件 JSON 持久化：原子写入 + 线程锁，适合单进程本地工具。"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from .prompts import DEFAULT_SEMANTIC_PROMPT
from . import prompts
from .data_governance import ensure_annotations, refresh_dataset
from .seed import build_state

T = TypeVar("T")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = DATA_DIR / "store.json"

_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_.\-]{0,19}$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _assign_codes(rules: List[Dict[str, Any]]) -> None:
    used = set()
    for rule in rules:
        code = str(rule.get("code") or "").strip().upper()
        if code and _CODE_PATTERN.match(code) and code not in used:
            rule["code"] = code
            used.add(code)
            continue
        index = 1
        while f"R{index:03d}" in used:
            index += 1
        code = f"R{index:03d}"
        rule["code"] = code
        used.add(code)


def migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """迁移到 v3：稳定标注、全局切分、可追溯的数据快照。v3 不重切分。"""
    previous_version = int(data.get("version") or 1)
    data.setdefault("rules", [])
    data.setdefault("runs", [])
    data.setdefault("settings", {})
    data.setdefault("semanticPrompt", DEFAULT_SEMANTIC_PROMPT)
    data.setdefault("semanticPromptVersion", 1)
    data.setdefault("promptHistory", [])
    data.setdefault("knowledge", [])
    data.setdefault("releases", [])
    data.setdefault("holdouts", [])
    data.setdefault("releaseChannels", {})
    data.setdefault("feedback", [])
    data.setdefault("dataWarnings", [])
    data.setdefault("datasetVersions", [])
    for release in data["releases"]:
        if release.get("status") == "EVALUATING":
            release["status"] = "INTERRUPTED"
            release["interruptedAt"] = _now()
            release["interruptionReason"] = "服务重启，独立评测中断；保留封存集绑定，可对同一能力版本重试"
    legacy_prompt = getattr(prompts, "LEGACY_SEMANTIC_PROMPT", None)
    if legacy_prompt and data.get("semanticPrompt") == legacy_prompt:
        data["promptHistory"].append({
            "version": int(data.get("semanticPromptVersion") or 1),
            "prompt": legacy_prompt, "createdAt": _now(),
            "reason": "v3 升级：保留原默认提示词，迁移到支持角色与分步骤判断的默认提示词",
        })
        data["semanticPrompt"] = DEFAULT_SEMANTIC_PROMPT
        data["semanticPromptVersion"] = int(data.get("semanticPromptVersion") or 1) + 1
    _assign_codes(data["rules"])
    for rule in data["rules"]:
        rule.setdefault("id", "rule-" + uuid.uuid4().hex[:12])
        rule.setdefault("changeLog", [])
        rule.setdefault("actor", "AGENT")
        rule.setdefault("workflow", "SIMPLE")
        rule.setdefault("responseWindowTurns", 3)
        rule.setdefault("responseRequirement", "回应客户具体诉求，并提供解释、处置或适当安抚")
        rule.setdefault("knowledgeIds", [])

    if previous_version >= 3:
        data.setdefault("calls", [])
        ensure_annotations(data)
        if not data.get("datasetVersion") or "splits" not in data:
            refresh_dataset(data, "初始化缺失的数据版本")
        return data

    if previous_version < 2:
        merged: Dict[str, Dict[str, Any]] = {}
        datasets = data.get("datasets") or {}
        for rule in data["rules"]:
            code = rule["code"]
            for case in datasets.get(rule["id"], []) or []:
                call_id = str(case.get("callId") or case.get("id") or "").strip()
                if not call_id:
                    continue
                call = merged.get(call_id)
                if call is None:
                    call = {
                        "id": "call-" + uuid.uuid4().hex[:8], "callId": call_id,
                        "transcript": str(case.get("transcript") or ""),
                        "autoRules": set(), "humanRules": set(),
                        "importedAt": case.get("importedAt") or _now(),
                    }
                    merged[call_id] = call
                if case.get("autoResult") == "VIOLATION":
                    call["autoRules"].add(code)
                if case.get("humanResult") == "VIOLATION":
                    call["humanRules"].add(code)
        data["calls"] = [
            {**call, "autoRules": sorted(call["autoRules"]), "humanRules": sorted(call["humanRules"])}
            for call in merged.values()
        ]
    data.setdefault("calls", [])
    data.pop("datasets", None)
    ensure_annotations(data, legacy=True)
    refresh_dataset(data, "迁移 v3：保存旧标签假设，改用全局同源分组切分")
    data["version"] = 3
    return data


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if int(raw.get("version") or 1) < 3:
                # 在修改内存/源文件之前先备份原始字节，失败则不继续迁移。
                backup_name = self.path.stem + ".v" + str(raw.get("version") or 1) + "-backup-" + uuid.uuid4().hex[:12] + ".json"
                backup_path = self.path.with_name(backup_name)
                descriptor = os.open(str(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as backup:
                    backup.write(self.path.read_bytes())
                    backup.flush()
                    os.fsync(backup.fileno())
            self._data = migrate(copy.deepcopy(raw))
            if self._data != raw:
                self._write(self._data)
        else:
            self._data = migrate(build_state())
            self._write(self._data)

    # ---------- 读取 ----------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def settings(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data.get("settings") or {})

    # ---------- 写入 ----------
    def mutate(self, fn: Callable[[Dict[str, Any]], T]) -> T:
        with self._lock:
            draft = copy.deepcopy(self._data)
            result = fn(draft)
            self._write(draft)
            self._data = draft
            return result

    def _write(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=".store-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
