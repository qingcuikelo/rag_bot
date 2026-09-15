"""Embeddings 工厂（文档 §10.2）。

统一由 ``EMBED_PROVIDER`` 切换 OpenAI 兼容 API / Ollama。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from xingchi_rag.config import EmbedProvider, get_settings


def embeddings_factory() -> Embeddings:
    """按配置构建 Embeddings 客户端。"""
    settings = get_settings()
    if settings.embed_provider == EmbedProvider.OPENAI:
        api_key = SecretStr(settings.openai_api_key) if settings.openai_api_key else None
        return OpenAIEmbeddings(
            model=settings.embed_model,
            base_url=settings.openai_base_url or None,
            api_key=api_key,
            # 非 OpenAI 官方模型（如 Qwen）不支持 tiktoken 预分词
            check_embedding_ctx_length=False,
        )
    if settings.embed_provider == EmbedProvider.OLLAMA:
        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"不支持的 EMBED_PROVIDER: {settings.embed_provider}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """返回缓存的 Embeddings 单例。"""
    return embeddings_factory()
