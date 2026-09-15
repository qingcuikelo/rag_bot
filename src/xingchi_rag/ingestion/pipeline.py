"""离线索引流水线（Phase 1：加载 → 治理 → 事实卡 → 切分 → SQLite → 报告）。

Chroma/BM25 写入在 Phase 2 接入；本阶段产出可向量化 chunks、结构化库与质量报告。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from xingchi_rag.config import get_settings
from xingchi_rag.ingestion.dispatch import load_directory
from xingchi_rag.ingestion.factcards import build_factcards
from xingchi_rag.ingestion.governance import govern, write_quality_report
from xingchi_rag.ingestion.splitters import split_documents
from xingchi_rag.retrieval.bm25 import build_bm25_index
from xingchi_rag.retrieval.store import build_vector_store, collection_name
from xingchi_rag.sql.load import load_all
from xingchi_rag.utils.docstore import save_documents


def _doc_id_for(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]


def _ensure_ids(documents: list[Document]) -> list[Document]:
    """补齐 doc_id 与 chunk_id，并保证 chunk_id 全局唯一（幂等）。"""
    counters: dict[str, int] = {}
    seen: set[str] = set()
    for doc in documents:
        source_file = str(doc.metadata.get("source_file", "doc"))
        doc_id = str(doc.metadata.get("doc_id") or _doc_id_for(source_file))
        doc.metadata["doc_id"] = doc_id

        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            idx = counters.get(doc_id, 0)
            chunk_id = f"{doc_id}-{idx:04d}"
            while chunk_id in seen:
                idx += 1
                chunk_id = f"{doc_id}-{idx:04d}"
            counters[doc_id] = idx + 1
            doc.metadata["chunk_id"] = chunk_id
        seen.add(str(chunk_id))
    return documents


def _corpus_hash(chunks: list[Document]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda d: str(d.metadata.get("chunk_id"))):
        digest.update(chunk.page_content.encode("utf-8"))
    return digest.hexdigest()


def build(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    report_path: str | Path | None = None,
    vectorize: bool = True,
) -> dict[str, Any]:
    """执行离线索引全链路，返回摘要。

    Args:
        data_dir: 原始数据目录（默认 ``data/``）。
        db_path: SQLite 路径。
        report_path: 质量报告路径。
        vectorize: 是否写入 Chroma 向量库（False 用于离线/无 API 的测试）。
    """
    settings = get_settings()
    settings.ensure_storage_dirs()

    warnings: list[str] = []
    target_dir = data_dir if data_dir else settings.resolve(settings.data_dir)
    raw_docs = load_directory(target_dir, warnings=warnings)
    governed = govern(raw_docs, warnings=warnings)
    fact_docs, conflicts = build_factcards(data_dir)

    # 文本类（非结构化、非 PII、非重复）先切分，事实卡本身即原子 chunk
    text_docs = [d for d in governed.documents if d.metadata.get("vectorize")]
    chunks = split_documents(text_docs)
    chunks = _ensure_ids(chunks + _ensure_ids(fact_docs))

    # BM25 与 parent_store（本地，无需模型）
    if chunks:
        build_bm25_index(chunks)
        parents = [c for c in chunks if c.metadata.get("parent_id")]
        save_documents(parents, settings.resolve(settings.parent_store_path) / "parents.jsonl")

    # Chroma 向量库（需要 Embedding API）
    if vectorize:
        build_vector_store(chunks)
        logger.info(f"Chroma 写入 {len(chunks)} 个 chunk")

    sql_counts = load_all(data_dir=data_dir, db_path=db_path)

    report = governed.report
    report["counts"]["chunks"] = len(chunks)
    report["counts"]["factcards"] = len(fact_docs)
    report["factcard_conflicts"] = conflicts
    report["sql_counts"] = sql_counts
    write_quality_report(report, report_path)

    manifest = {
        "index_version": f"v{settings.index_version}",
        "built_at": datetime.now(UTC).isoformat(),
        "corpus_hash": _corpus_hash(chunks),
        "counts": {
            "chunks": len(chunks),
            "vector_documents": len(chunks) if vectorize else 0,
            **sql_counts,
        },
        "collection_name": collection_name(),
        "embed_model": settings.embed_model,
        "embed_provider": settings.embed_provider.value,
        "rerank_model": settings.rerank_model,
    }
    manifest_path = settings.resolve(settings.manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"manifest 已写出: {manifest_path}")

    return {
        "manifest": manifest,
        "report_path": str(settings.resolve(settings.quality_report_path)),
        "chunks": len(chunks),
        "warnings": warnings,
    }
