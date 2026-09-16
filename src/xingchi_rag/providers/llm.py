"""LLM 工厂（文档 §10.2）。

``LLM_PROVIDER`` 决定主模型，``LLM_FALLBACK`` 决定降级目标；
以 ``with_fallbacks`` 串联（API → Ollama）。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from xingchi_rag.config import LLMProvider, Settings, get_settings

_LLM_SEMAPHORE: threading.Semaphore | None = None
_SEMAPHORE_LOCK = threading.Lock()


def _semaphore() -> threading.Semaphore:
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _LLM_SEMAPHORE is None:
                limit = max(1, get_settings().llm_max_concurrency)
                _LLM_SEMAPHORE = threading.Semaphore(limit)
    return _LLM_SEMAPHORE


@contextmanager
def llm_slot(timeout: float | None = None) -> Iterator[None]:
    """LLM 并发信号量：限制同时进行的模型调用，避免线程池/供应商被打满。

    Raises:
        TimeoutError: 在超时时间内未获得槽位。
    """
    sem = _semaphore()
    budget = timeout if timeout is not None else float(get_settings().request_timeout_s)
    if not sem.acquire(timeout=budget):
        raise TimeoutError("LLM 并发已满，等待槽位超时")
    try:
        yield
    finally:
        sem.release()


def reset_llm_limiter() -> None:
    """重置信号量（配置变更/测试用）。"""
    global _LLM_SEMAPHORE
    with _SEMAPHORE_LOCK:
        _LLM_SEMAPHORE = None


def _primary(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == LLMProvider.OPENAI:
        api_key = SecretStr(settings.openai_api_key) if settings.openai_api_key else None
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openai_base_url or None,
            api_key=api_key,
            temperature=0,
            timeout=settings.llm_timeout_s,
        )
    if settings.llm_provider == LLMProvider.OLLAMA:
        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    raise ValueError(f"不支持的 LLM_PROVIDER: {settings.llm_provider}")


def _fallback(settings: Settings) -> BaseChatModel | None:
    if settings.llm_fallback == LLMProvider.OLLAMA:
        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    return None


def llm_factory() -> BaseChatModel:
    """构建主模型（含降级链）。"""
    settings = get_settings()
    primary = _primary(settings)
    fallback = _fallback(settings)
    if fallback is None:
        return primary
    return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """返回缓存的 LLM 单例。"""
    return llm_factory()
