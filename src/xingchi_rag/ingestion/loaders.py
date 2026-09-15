"""文档加载器。

- PDF/TXT/CSV/TSV：复用 LangChain 社区加载器；
- XLSX/XLS：自研 ``ExcelDocumentLoader``（openpyxl），逐 sheet、逐行，
  跳过退化为表头的 sheet 并告警（文档 §5.1）。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from openpyxl import load_workbook

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".csv", ".tsv", ".xlsx", ".xls"}


class ExcelDocumentLoader(BaseLoader):
    """Excel 加载器：逐 sheet、逐行产出 ``Document``。

    每个数据行产出一个 ``Document``，``metadata`` 保留 sheet 名与行号，
    便于后续事实卡与溯源。整表退化为表头（数据行与表头完全相同）时跳过并告警。
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        sheet_names: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.sheet_names = sheet_names
        self.warnings = warnings if warnings is not None else []

    @staticmethod
    def _normalize_row(row: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple("" if c is None else str(c).strip() for c in row)

    @staticmethod
    def _is_empty(row: tuple[Any, ...]) -> bool:
        return all(c is None or str(c).strip() == "" for c in row)

    def lazy_load(self) -> Iterator[Document]:
        wb = load_workbook(self.file_path, read_only=True, data_only=True)
        try:
            for ws in wb.worksheets:
                if self.sheet_names and ws.title not in self.sheet_names:
                    continue
                rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
                if not rows:
                    self._warn(ws.title, "空 sheet")
                    continue

                header = self._normalize_row(rows[0])
                data_rows = rows[1:]
                distinct = [
                    r
                    for r in data_rows
                    if not self._is_empty(r) and self._normalize_row(r) != header
                ]
                if not distinct:
                    self._warn(ws.title, "数据行退化为表头，已跳过")
                    continue

                for line_no, row in enumerate(data_rows, start=2):
                    if self._is_empty(row) or self._normalize_row(row) == header:
                        continue
                    yield self._row_to_document(ws.title, line_no, header, row)
        finally:
            wb.close()

    def _row_to_document(
        self,
        sheet: str,
        line_no: int,
        header: tuple[str, ...],
        row: tuple[Any, ...],
    ) -> Document:
        pairs = []
        for head, value in zip(header, row, strict=False):
            if head == "" and (value is None or str(value).strip() == ""):
                continue
            pairs.append(f"{head}: {'' if value is None else value}")
        content = "；".join(pairs)
        return Document(
            page_content=content,
            metadata={
                "source_file": self.file_path.name,
                "source_path": str(self.file_path),
                "file_type": "xlsx",
                "sheet": sheet,
                "row": line_no,
            },
        )

    def _warn(self, sheet: str, reason: str) -> None:
        message = f"[{self.file_path.name}] sheet「{sheet}」{reason}"
        self.warnings.append(message)
        logger.warning(message)


def load_excel(
    file_path: str | Path,
    *,
    sheet_names: list[str] | None = None,
    warnings: list[str] | None = None,
) -> list[Document]:
    """加载 Excel 文件为 ``Document`` 列表。"""
    return list(
        ExcelDocumentLoader(file_path, sheet_names=sheet_names, warnings=warnings).lazy_load()
    )


def load_pdf(file_path: str | Path) -> list[Document]:
    """加载 PDF（逐页 ``Document``，metadata 含 ``page``）。"""
    return PyPDFLoader(str(file_path)).load()


def load_text(file_path: str | Path) -> list[Document]:
    """加载纯文本（UTF-8）。"""
    return TextLoader(str(file_path), encoding="utf-8").load()


def load_csv(file_path: str | Path, *, separator: str = ",") -> list[Document]:
    """加载 CSV/TSV（分隔符可指定），凭据来源到文件名。"""
    return CSVLoader(
        str(file_path),
        csv_args={"delimiter": separator},
        encoding="utf-8",
        source_column=None,
    ).load()
