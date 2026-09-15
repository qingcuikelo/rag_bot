"""检索器工厂（文档 §12.1）。

``mode`` 可选 ``vector`` / ``bm25`` / ``hybrid``（混合，Phase 3 再加重排）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever

from xingchi_rag.retrieval.bm25 import get_bm25_retriever
from xingchi_rag.retrieval.hybrid import build_ensemble
from xingchi_rag.retrieval.store import get_vector_retriever


def get_retriever(
    mode: str = "hybrid",
    *,
    k: int | None = None,
    filter: dict[str, Any] | None = None,
) -> BaseRetriever:
    """按模式返回检索器。

    Args:
        mode: ``vector`` | ``bm25`` | ``hybrid``。
        k: 各路召回数量。
        filter: metadata 硬过滤（仅对向量检索生效）。
    """
    vector_retriever = get_vector_retriever(k=k, filter=filter)
    if mode == "vector":
        return vector_retriever

    bm25_retriever = get_bm25_retriever(k=k)
    if mode == "bm25":
        return bm25_retriever

    if mode == "hybrid":
        return build_ensemble(vector_retriever, bm25_retriever)

    raise ValueError(f"不支持的检索模式: {mode}")
