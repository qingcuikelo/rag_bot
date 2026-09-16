"""BM25 稀疏检索（jieba 分词 + 持久化）。

补足稠密检索对型号/数字/专有名词的弱点（文档 §3.2、§7.2）。
索引以 JSONL 持久化 chunk，加载时重建 ``BM25Retriever``。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from xingchi_rag.config import get_settings
from xingchi_rag.utils.docstore import load_documents, save_documents

BM25_FILENAME = "chunks.jsonl"


def jieba_tokenize(text: str) -> list[str]:
    """jieba 分词并去除空白 token。"""
    return [token for token in jieba.lcut(text) if token.strip()]


def _chunks_path() -> Path:
    settings = get_settings()
    return settings.resolve(settings.bm25_path) / BM25_FILENAME


def build_bm25_retriever(documents: list[Document], k: int | None = None) -> BM25Retriever:
    """从内存文档构建 BM25 检索器。"""
    settings = get_settings()
    return BM25Retriever.from_documents(
        documents,
        preprocess_func=jieba_tokenize,
        k=k or settings.bm25_k,
    )


def build_bm25_index(documents: list[Document]) -> BM25Retriever:
    """持久化 chunk 并返回 BM25 检索器。"""
    save_documents(documents, _chunks_path())
    get_bm25_retriever.cache_clear()
    return build_bm25_retriever(documents)


@lru_cache(maxsize=8)
def get_bm25_retriever(k: int | None = None) -> BM25Retriever:
    """从持久化 chunk 重建 BM25 检索器（按 k 缓存，避免每请求重建）。"""
    documents = load_documents(_chunks_path())
    if not documents:
        raise FileNotFoundError(f"BM25 索引缺失，请先构建: {_chunks_path()}")
    return build_bm25_retriever(documents, k=k)
