"""在独立工作进程运行用户正则，超时后终止工作进程并在下一次调用重建。"""
from __future__ import annotations

import atexit
import multiprocessing
import re
import threading


class RegexTimeout(ValueError):
    pass


def _worker(connection):
    connection.send("ready")
    while True:
        try:
            pattern, text = connection.recv()
        except EOFError:
            return
        try:
            matched = re.search(pattern, text)
            connection.send({"match": (matched.start(), matched.end(), matched.group(0)) if matched else None})
        except Exception as exc:
            connection.send({"error": str(exc)})


class _Match:
    def __init__(self, data):
        self.data = data

    def start(self):
        return self.data[0]

    def end(self):
        return self.data[1]

    def span(self):
        return self.data[:2]

    def group(self, index=0):
        if index != 0:
            raise IndexError("只返回整体匹配")
        return self.data[2]


_process = None
_connection = None
_lock = threading.Lock()


def _close():
    global _process, _connection
    if _connection is not None:
        _connection.close()
    if _process is not None:
        if _process.is_alive():
            _process.terminate()
        _process.join(timeout=1)
    _process = _connection = None


def safe_search(pattern, text, timeout=.25):
    global _process, _connection
    with _lock:
        if _process is None or not _process.is_alive():
            _close()
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            _process = context.Process(target=_worker, args=(child,), daemon=True)
            _process.start()
            child.close()
            _connection = parent
            if not parent.poll(5):
                _close()
                raise RegexTimeout("正则隔离进程未能就绪")
            try:
                parent.recv()
            except EOFError as exc:
                _close()
                raise RegexTimeout("正则隔离进程启动失败") from exc
        try:
            _connection.send((str(pattern), str(text)))
            if not _connection.poll(timeout):
                _close()
                raise RegexTimeout("正则执行超过 %.0f 毫秒，已终止；请简化表达式" % (timeout * 1000))
            result = _connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            _close()
            raise RegexTimeout("正则隔离进程异常结束") from exc
        if result.get("error"):
            raise re.error(result["error"])
        return _Match(result["match"]) if result.get("match") else None


atexit.register(_close)
