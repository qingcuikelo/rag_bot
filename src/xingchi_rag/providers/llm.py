"""LLM 工厂（文档 §10.2）。

``LLM_PROVIDER`` 决定主模型，``LLM_FALLBACK`` 决定降级目标；
以 ``with_fallbacks`` 串联（API → Ollama）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from xingchi_rag.config import LLMProvider, Settings, get_settings


def _primary(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == LLMProvider.OPENAI:
        api_key = SecretStr(settings.openai_api_key) if settings.openai_api_key else None
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openai_base_url or None,
            api_key=api_key,
            temperature=0,
            timeout=settings.request_timeout_s,
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
