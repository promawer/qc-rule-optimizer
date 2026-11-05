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
from .seed import build_state
from .splits import build_splits

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
    """把一期「按规则存案例」的数据转换为统一通话表（version 2）。"""
    data.setdefault("rules", [])
    data.setdefault("runs", [])
    data.setdefault("settings", {})
    data.setdefault("semanticPrompt", DEFAULT_SEMANTIC_PROMPT)
    data.setdefault("semanticPromptVersion", 1)
    data.setdefault("promptHistory", [])
    _assign_codes(data["rules"])
    for rule in data["rules"]:
        rule.setdefault("changeLog", [])

    if data.get("version") == 2:
        calls = data.setdefault("calls", [])
        data["splits"] = build_splits(data["rules"], calls)
        return data

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
                    "id": "call-" + uuid.uuid4().hex[:8],
                    "callId": call_id,
                    "transcript": str(case.get("transcript") or ""),
                    "autoRules": set(),
                    "humanRules": set(),
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
    data["splits"] = build_splits(data["rules"], data["calls"])
    data.pop("datasets", None)
    data["version"] = 2
    return data


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._data = migrate(copy.deepcopy(raw))
            if self._data != raw:
                self._write(self._data)
        else:
            self._data = build_state()
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
