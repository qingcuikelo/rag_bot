"""单元测试：并发控制（LLM 信号量 / 会话锁）。"""

from __future__ import annotations

import threading
import time

import pytest

from xingchi_rag.config import get_settings
from xingchi_rag.providers import llm as llm_module


@pytest.fixture(autouse=True)
def _reset_limiter():
    llm_module.reset_llm_limiter()
    get_settings.cache_clear()
    yield
    llm_module.reset_llm_limiter()
    get_settings.cache_clear()


def test_llm_slot_limits_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    get_settings.cache_clear()
    llm_module.reset_llm_limiter()

    current = 0
    peak = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal current, peak
        with llm_module.llm_slot(timeout=5):
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.1)
            with lock:
                current -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak <= 2


def test_llm_slot_times_out_when_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")
    get_settings.cache_clear()
    llm_module.reset_llm_limiter()

    release = threading.Event()

    def holder() -> None:
        with llm_module.llm_slot(timeout=2):
            release.wait(1)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    time.sleep(0.1)

    try:
        with pytest.raises(TimeoutError), llm_module.llm_slot(timeout=0.2):
            pass
    finally:
        release.set()
        holder_thread.join()


def test_session_lock_identity() -> None:
    from xingchi_rag.api.routes import _session_lock

    assert _session_lock("s1") is _session_lock("s1")
    assert _session_lock("s1") is not _session_lock("s2")
