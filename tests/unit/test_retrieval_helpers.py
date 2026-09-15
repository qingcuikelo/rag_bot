"""单元测试：检索辅助（metadata 清洗 / jieba / 相关性判定）。"""

from __future__ import annotations

from datetime import date

from langchain_core.documents import Document

from eval.run import is_relevant
from xingchi_rag.retrieval.bm25 import jieba_tokenize
from xingchi_rag.retrieval.store import sanitize_metadata


def test_sanitize_metadata() -> None:
    clean = sanitize_metadata(
        {
            "chunk_id": "abc-0001",
            "authority_rank": 10,
            "score": 0.9,
            "vectorize": True,
            "parent_id": None,
            "effective_date": date(2026, 3, 31),
        }
    )
    assert "parent_id" not in clean
    assert clean["authority_rank"] == 10
    assert clean["vectorize"] is True
    assert clean["effective_date"] == "2026-03-31"


def test_jieba_tokenize() -> None:
    tokens = jieba_tokenize("XC-L100 保修 3 年")
    assert "保修" in tokens
    assert all(t.strip() for t in tokens)


def test_is_relevant_by_file_and_section() -> None:
    doc = Document(
        page_content="库存预警：XC-L100 深圳中心仓在库 180 台",
        metadata={"source_file": "库存清单.tsv", "section": "库存"},
    )
    assert is_relevant(doc, ["库存清单.tsv#库存"]) is True
    assert is_relevant(doc, ["库存清单.tsv"]) is True
    assert is_relevant(doc, ["产品参数表.csv#产品参数"]) is False
    assert is_relevant(doc, ["库存清单.tsv#价格"]) is False
