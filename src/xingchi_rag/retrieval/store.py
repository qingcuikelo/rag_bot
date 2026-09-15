"""Chroma 向量库构建与检索（文档 §7.1、§7.2）。

本地 ``PersistentClient`` 磁盘持久化，按 ``index_version`` 命名 collection。
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from xingchi_rag.config import get_settings
from xingchi_rag.providers.embeddings import get_embeddings


def collection_name() -> str:
    """当前索引版本的 collection 名（``xingchi_kb_v{N}``）。"""
    return f"xingchi_kb_v{get_settings().index_version}"


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma 仅接受 str/int/float/bool，过滤 ``None`` 并转换其余类型。"""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _store(reset: bool = False) -> Chroma:
    settings = get_settings()
    path = settings.resolve(settings.chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection_name(),
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )
    if reset:
        try:
            store.delete_collection()
        except Exception:  # 首次构建时 collection 不存在
            logger.info(f"collection {collection_name()} 不存在，直接创建")
        store = Chroma(
            collection_name=collection_name(),
            embedding_function=get_embeddings(),
            persist_directory=str(path),
        )
    return store


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), reraise=True)
def _add_batch(store: Chroma, docs: list[Document], ids: list[str]) -> None:
    store.add_documents(documents=docs, ids=ids)


def build_vector_store(documents: list[Document], *, reset: bool = True) -> Chroma:
    """写入 Chroma（按 ``embed_batch_size`` 批量，tenacity 重试）。"""
    settings = get_settings()
    batch_size = settings.embed_batch_size
    store = _store(reset=reset)
    total = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        ids = [
            str(doc.metadata.get("chunk_id", f"chunk-{start + i}")) for i, doc in enumerate(batch)
        ]
        for doc in batch:
            doc.metadata = sanitize_metadata(doc.metadata)
        _add_batch(store, batch, ids)
        total += len(batch)
        logger.info(f"Chroma 写入进度 {total}/{len(documents)}")
    return store


def get_vector_store() -> Chroma:
    """打开已持久化的 Chroma（不重置）。"""
    return _store(reset=False)


def get_vector_retriever(
    k: int | None = None,
    filter: dict[str, Any] | None = None,
) -> BaseRetriever:
    """向量检索器（相似度检索，支持 metadata 过滤）。"""
    settings = get_settings()
    search_kwargs: dict[str, Any] = {"k": k or settings.retrieve_k}
    if filter:
        search_kwargs["filter"] = filter
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)
