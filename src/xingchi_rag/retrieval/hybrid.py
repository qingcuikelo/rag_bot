"""混合检索：EnsembleRetriever（RRF 融合，文档 §7.1）。"""

from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from xingchi_rag.config import get_settings


def build_ensemble(
    vector_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever,
    *,
    weights: list[float] | None = None,
    c: int | None = None,
) -> EnsembleRetriever:
    """构建 RRF 融合检索器（默认向量/BM25 等权）。"""
    settings = get_settings()
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights or [0.5, 0.5],
        c=c or settings.rrf_c,
    )
