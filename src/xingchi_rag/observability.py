"""可观测性（文档 §10.3）。

- LangSmith 全链路追踪（按配置注入环境变量）；
- 进程内轻量指标（请求耗时、拒答数、降级次数），供日志/健康检查查看。
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict

from loguru import logger

from xingchi_rag.config import get_settings

_METRICS: dict[str, float] = defaultdict(float)
_COUNTS: dict[str, int] = defaultdict(int)
_LOCK = threading.Lock()


def setup_tracing() -> bool:
    """按配置开启 LangSmith 追踪。返回是否启用。"""
    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info(f"LangSmith 追踪已启用，project={settings.langsmith_project}")
    return True


def record_latency(name: str, milliseconds: float) -> None:
    """累计耗时指标。"""
    with _LOCK:
        _METRICS[f"{name}_total_ms"] += milliseconds
        _COUNTS[f"{name}_count"] += 1


def increment(name: str, delta: int = 1) -> None:
    """累计计数指标。"""
    with _LOCK:
        _COUNTS[name] += delta


def metrics_snapshot() -> dict[str, dict[str, float]]:
    """返回指标快照（含平均耗时）。"""
    with _LOCK:
        averages: dict[str, float] = {}
        for key, total in _METRICS.items():
            base = key[: -len("_total_ms")]
            count = _COUNTS.get(f"{base}_count", 0)
            averages[f"{base}_avg_ms"] = total / count if count else 0.0
        return {
            "latency": averages,
            "counters": {k: float(v) for k, v in _COUNTS.items()},
        }


def reset_metrics() -> None:
    """清空指标（测试用）。"""
    with _LOCK:
        _METRICS.clear()
        _COUNTS.clear()
