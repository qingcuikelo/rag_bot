"""单元测试：生成结果校验（引用 / 数值）。"""

from __future__ import annotations

from langchain_core.documents import Document

from xingchi_rag.generation.answer import (
    CitationModel,
    format_evidence,
    validate_citations,
    validate_numbers,
)


def _evidence() -> list[Document]:
    return [
        Document(
            page_content="库存预警：XC-L100 深圳中心仓在库 180 台，安全库存 100 台。",
            metadata={"source_file": "库存清单.tsv", "section": "库存", "chunk_id": "a-0000"},
        )
    ]


def test_format_evidence() -> None:
    text = format_evidence(_evidence())
    assert "库存清单.tsv" in text
    assert "180" in text


def test_validate_numbers_ok_and_strips_list_markers() -> None:
    answer = "【依据】\n 1. 在库 180 台\n 2. 安全库存 100 台"
    ok, unknown = validate_numbers(answer, _evidence())
    assert ok is True
    assert unknown == []


def test_validate_numbers_flags_unknown() -> None:
    ok, unknown = validate_numbers("在库 999 台", _evidence())
    assert ok is False
    assert "999" in unknown


def test_validate_citations() -> None:
    answer = "在库 180 台（[来源: 库存清单.tsv · 库存]）"
    ok, unknown = validate_citations(answer, [], _evidence())
    assert ok is True
    assert unknown == []

    bad, bad_unknown = validate_citations("…（[来源: 不存在.txt]）", [], _evidence())
    assert bad is False
    assert "不存在.txt" in bad_unknown


def test_validate_citations_structured() -> None:
    citations = [CitationModel(source_file="库存清单.tsv", section="库存", chunk_id="a-0000")]
    ok, _ = validate_citations("…", citations, _evidence())
    assert ok is True
