"""不可变实验输入与任务检查点。快照包含脱敏配置，不包含模型密钥。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .governance import model_manifest, stamp
from .prompt_regression import context_fingerprint


def code_fingerprint():
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def atomic_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", dir=str(file.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, file)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def freeze_experiment(state, job, directory):
    manifest = {"id": job["id"], "createdAt": stamp(), "codeVersion": "3.0.0",
                "codeFingerprint": code_fingerprint(), "contextFingerprint": context_fingerprint(state),
                "datasetVersion": state.get("datasetVersion", 1), "rules": copy.deepcopy(state["rules"]),
                "calls": copy.deepcopy(state.get("calls", [])), "splits": copy.deepcopy(state.get("splits", {})),
                "knowledge": copy.deepcopy(state.get("knowledge", [])), "semanticPrompt": state.get("semanticPrompt"),
                "model": model_manifest(state.get("settings", {})), "policy": copy.deepcopy(job.get("policy", {})),
                "sampledCallIds": job.get("sampledCallIds", []), "target": job.get("target"),
                "maxRounds": job.get("maxRounds")}
    file = Path(directory) / (job["id"] + ".json")
    atomic_json(file, manifest)
    job["contextFingerprint"] = manifest["contextFingerprint"]
    job["codeFingerprint"] = manifest["codeFingerprint"]
    job["datasetVersion"] = manifest["datasetVersion"]
    job["manifestId"] = job["id"]
    job["_context"] = copy.deepcopy(state)
    return manifest


def checkpoint(job, directory):
    atomic_json(Path(directory) / (job["id"] + ".json"),
                {k: v for k, v in job.items() if not k.startswith("_")})


def restore_jobs(directory):
    jobs = {}
    for file in Path(directory).glob("job-*.json"):
        try:
            job = json.loads(file.read_text(encoding="utf-8"))
            if job.get("status") in ("queued", "running"):
                job.update(status="cancelled", phase="服务重启中断", error="服务重启；已保留输入快照及已完成结果，请重新启动实验", finishedAt=stamp())
            jobs[job["id"]] = job
        except (ValueError, KeyError, OSError):
            continue
    return jobs
