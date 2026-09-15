"""Document 文档仓储：JSONL 持久化（用于 BM25 / parent_store）。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def save_documents(documents: list[Document], path: str | Path) -> Path:
    """把 ``Document`` 列表写为 JSONL（每行 ``{"page_content", "metadata"}``）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return target


def load_documents(path: str | Path) -> list[Document]:
    """读取 JSONL 为 ``Document`` 列表。"""
    source = Path(path)
    if not source.is_file():
        return []
    documents: list[Document] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(
                Document(page_content=record["page_content"], metadata=record.get("metadata", {}))
            )
    return documents
