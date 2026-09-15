"""检索器工厂（文档 §12.1）。

``mode`` 可选 ``vector`` / ``bm25`` / ``hybrid``（混合）；
``rerank=True`` 时用 ``ContextualCompressionRetriever`` 叠加 API 重排。
"""

from __future__ import annotations

from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever

from xingchi_rag.retrieval.bm25 import get_bm25_retriever
from xingchi_rag.retrieval.hybrid import build_ensemble
from xingchi_rag.retrieval.rerank import build_reranker
from xingchi_rag.retrieval.store import get_vector_retriever


def _base_retriever(
    mode: str,
    *,
    k: int | None = None,
    filter: dict[str, Any] | None = None,
) -> BaseRetriever:
    vector_retriever = get_vector_retriever(k=k, filter=filter)
    if mode == "vector":
        return vector_retriever

    bm25_retriever = get_bm25_retriever(k=k)
    if mode == "bm25":
        return bm25_retriever

    if mode == "hybrid":
        return build_ensemble(vector_retriever, bm25_retriever)

    raise ValueError(f"不支持的检索模式: {mode}")


def get_retriever(
    mode: str = "hybrid",
    *,
    k: int | None = None,
    filter: dict[str, Any] | None = None,
    rerank: bool = False,
    top_n: int | None = None,
) -> BaseRetriever:
    """按模式返回检索器。

    Args:
        mode: ``vector`` | ``bm25`` | ``hybrid``。
        k: 各路召回数量。
        filter: metadata 硬过滤（仅对向量检索生效）。
        rerank: 是否叠加 API 重排。
        top_n: 重排后保留数量。
    """
    base = _base_retriever(mode, k=k, filter=filter)
    if not rerank:
        return base
    return ContextualCompressionRetriever(
        base_compressor=build_reranker(top_n=top_n),
        base_retriever=base,
    )
