"""以分离进程启动 uvicorn，供 scripts/dev.ps1 调用。

不生成 .cmd、不隐藏窗口，避免安全软件把启动脚本误判为可疑行为。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / ".run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = RUN_DIR / "app.pid"
LOG_FILE = RUN_DIR / "app.log"
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> int:
    no_browser = "--no-browser" in sys.argv
    if health_ok():
        print(f"服务已在运行：{URL}")
        return 0

    log = open(LOG_FILE, "ab")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_CACHE_DIR"] = r"E:\qc-runtimes\pip-cache"

    flags = 0
    kwargs = {}
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        env=env,
        creationflags=flags,
        close_fds=True,
        **kwargs,
    )
    PID_FILE.write_text(str(process.pid), encoding="ascii")

    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            print("服务启动失败，日志末尾：")
            try:
                lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                print("\n".join(lines))
            except OSError:
                pass
            return 1
        if health_ok():
            print(f"质检规则优化台：{URL}")
            print(f"日志：{LOG_FILE}")
            if not no_browser:
                webbrowser.open(URL)
            return 0
        time.sleep(0.5)

    print("等待服务就绪超时，请查看日志：" + str(LOG_FILE))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
