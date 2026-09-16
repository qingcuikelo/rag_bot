"""单元测试：TTL 缓存。"""

from __future__ import annotations

import time

from xingchi_rag.utils.cache import TTLCache, cache_key


def test_cache_key_normalization() -> None:
    assert cache_key("  XC-L100   库存多少 ") == "xc-l100 库存多少"
    assert cache_key("q", "ctx") == "ctx|q"


def test_cache_set_get() -> None:
    cache = TTLCache(max_size=4, ttl_s=10)
    cache.set("a", {"answer": "1"})
    assert cache.get("a") == {"answer": "1"}
    assert cache.get("missing") is None
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_ttl_expiry() -> None:
    cache = TTLCache(max_size=4, ttl_s=1)
    cache.set("a", "value")
    time.sleep(1.1)
    assert cache.get("a") is None


def test_cache_lru_eviction() -> None:
    cache = TTLCache(max_size=2, ttl_s=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # a 变为最近使用
    cache.set("c", 3)  # 淘汰 b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_cache_clear() -> None:
    cache = TTLCache(max_size=2, ttl_s=10)
    cache.set("a", 1)
    cache.clear()
    assert cache.get("a") is None
