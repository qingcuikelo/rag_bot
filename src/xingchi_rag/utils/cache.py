"""轻量 TTL 缓存（高频问答结果缓存，文档 §10.4）。

线程安全；惰性过期；容量上限采用 LRU 淘汰。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from xingchi_rag.config import get_settings


def cache_key(question: str, extra: str = "") -> str:
    """规范化缓存键（去空白、小写、附加上下文）。"""
    normalized = " ".join((question or "").lower().split())
    return f"{extra}|{normalized}" if extra else normalized


class TTLCache:
    """基于 TTL + LRU 的简单缓存。"""

    def __init__(self, max_size: int = 256, ttl_s: int = 300) -> None:
        self.max_size = max(1, max_size)
        self.ttl_s = max(1, ttl_s)
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._data.pop(key, None)
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "ttl_s": self.ttl_s,
                "hits": self._hits,
                "misses": self._misses,
            }


@lru_cache(maxsize=1)
def get_answer_cache() -> TTLCache:
    """按配置返回问答缓存单例。"""
    settings = get_settings()
    return TTLCache(max_size=settings.cache_max_size, ttl_s=settings.cache_ttl_s)
