"""切分策略（文档 §5.2）。

- 政策条款：每条一 chunk（保留条名于 ``section``）；
- FAQ：一问一答一 chunk；
- 手册/目录：递归切分；
- 统一注入 ``chunk_id``（``doc_id`` + 序号）。
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 目标 256~512 tokens；中文近 1 字≈1 token，取 400 字、重叠 60 字（约 15%）
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60

_CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百]+条)")
_FAQ_RE = re.compile(r"(Q\d+[：:].*?)(?=Q\d+[：:]|\Z)", re.S)


def _base_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "；", "。", " "],
        keep_separator=True,
    )


def _finalize(chunks: list[Document], doc_id: str, start_index: int = 0) -> list[Document]:
    for offset, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{doc_id}-{start_index + offset:04d}"
        # parent_id 用空串表示“无父块”（Chroma metadata 不支持 None）
        if not chunk.metadata.get("parent_id"):
            chunk.metadata["parent_id"] = ""
    return chunks


def split_policy(doc: Document) -> list[Document]:
    """政策条款：按「第X条」切分，每条一 chunk。"""
    parts = _CLAUSE_RE.split(doc.page_content)
    chunks: list[Document] = []
    # split 结果形如 ['前言', '第一条', '正文', '第二条', '正文', ...]
    if parts and parts[0].strip():
        chunks.append(Document(page_content=parts[0].strip(), metadata=dict(doc.metadata)))
    for i in range(1, len(parts), 2):
        clause = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        content = f"{clause} {body}".strip()
        if content:
            chunks.append(
                Document(
                    page_content=content,
                    metadata={**doc.metadata, "section": clause},
                )
            )
    return chunks or [Document(page_content=doc.page_content, metadata=dict(doc.metadata))]


def split_faq(doc: Document) -> list[Document]:
    """FAQ：一问一答一 chunk。"""
    matches = _FAQ_RE.findall(doc.page_content)
    chunks: list[Document] = []
    for block in matches:
        text = block.strip()
        if not text:
            continue
        question = text.splitlines()[0].split("：", 1)[0]
        chunks.append(Document(page_content=text, metadata={**doc.metadata, "section": question}))
    return chunks or [Document(page_content=doc.page_content, metadata=dict(doc.metadata))]


def split_recursive(doc: Document) -> list[Document]:
    """手册/目录/产品介绍：递归切分。"""
    return _base_splitter().split_documents([doc])


def split_documents(documents: list[Document]) -> list[Document]:
    """按 doc_type 路由切分，并注入 ``chunk_id``（按 doc_id 累计序号，避免多页冲突）。"""
    result: list[Document] = []
    offsets: dict[str, int] = {}
    for doc in documents:
        doc_type = str(doc.metadata.get("doc_type", "catalog"))
        if doc_type == "policy":
            chunks = split_policy(doc)
        elif doc_type == "faq":
            chunks = split_faq(doc)
        else:
            chunks = split_recursive(doc)
        doc_id = str(doc.metadata.get("doc_id", "doc"))
        start = offsets.get(doc_id, 0)
        result.extend(_finalize(chunks, doc_id, start_index=start))
        offsets[doc_id] = start + len(chunks)
    return result
