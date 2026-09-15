"""按后缀分发加载器（文档 §5.1）。

``DirectoryLoader`` 只支持单一 ``loader_cls``，无法自动分发多格式，
故自研 ``dispatch_loader``。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from xingchi_rag.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    load_csv,
    load_excel,
    load_pdf,
    load_text,
)


def dispatch_loader(path: str | Path, *, warnings: list[str] | None = None) -> list[Document]:
    """按文件后缀选择加载器，返回 ``Document`` 列表。

    Args:
        path: 文件路径。
        warnings: 可选的告警收集列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不支持的后缀。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        docs = load_pdf(p)
    elif suffix == ".txt":
        docs = load_text(p)
    elif suffix == ".csv":
        docs = load_csv(p, separator=",")
    elif suffix == ".tsv":
        docs = load_csv(p, separator="\t")
    elif suffix in {".xlsx", ".xls"}:
        docs = load_excel(p, warnings=warnings)
    else:
        raise ValueError(f"不支持的后缀: {suffix}（文件 {p.name}）")

    return [_stamp(doc, p) for doc in docs]


def load_directory(directory: str | Path, *, warnings: list[str] | None = None) -> list[Document]:
    """递归加载目录下所有受支持的文件（按文件名排序，保证确定性）。"""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.extend(dispatch_loader(path, warnings=warnings))
    logger.info(f"已加载 {root} 下 {len(documents)} 个 Document")
    return documents


def _stamp(doc: Document, path: Path) -> Document:
    """统一补充元数据（文件名、路径、后缀）。"""
    metadata = dict(doc.metadata or {})
    metadata["source_file"] = path.name
    metadata["source_path"] = str(path)
    metadata["file_type"] = path.suffix.lower().lstrip(".")
    doc.metadata = metadata
    return doc
