"""单元测试：数据治理（去重 / 权威级 / PII / doc_type）。"""

from __future__ import annotations

from langchain_core.documents import Document

from xingchi_rag.ingestion.governance import (
    authority_map,
    govern,
    infer_doc_type,
    load_sources_config,
)


def _doc(text: str, source_file: str, file_type: str = "txt") -> Document:
    return Document(
        page_content=text,
        metadata={"source_file": source_file, "file_type": file_type},
    )


def test_authority_map_from_config() -> None:
    config = load_sources_config()
    mapping = authority_map(config)
    assert mapping["库存清单.tsv"] == 10
    assert mapping["产品价格与库存.xlsx"] == 10
    assert mapping["产品参数表.csv"] == 20
    assert mapping["产品参数表.tsv"] == 40
    assert mapping["售后工单统计.xlsx"] == 90


def test_infer_doc_type() -> None:
    assert infer_doc_type("保修政策与退换货说明.txt") == "policy"
    assert infer_doc_type("客服FAQ_售前售后.txt") == "faq"
    assert infer_doc_type("产品使用手册_星驰智能门锁.pdf") == "manual"
    assert infer_doc_type("产品参数表.csv") == "spec"
    assert infer_doc_type("库存清单.tsv") == "inventory"
    assert infer_doc_type("产品销售数据.xlsx") == "sales"


def test_governance_injects_metadata() -> None:
    docs = [_doc("第一条 保修三年。", "保修政策与退换货说明.txt")]
    result = govern(docs)
    meta = result.documents[0].metadata
    assert meta["authority_rank"] == 20
    assert meta["doc_type"] == "policy"
    assert meta["pii_flag"] is False
    assert meta["vectorize"] is True
    assert meta["structured"] is False
    assert meta["doc_id"]


def test_governance_pii_not_vectorized() -> None:
    docs = [_doc("客户 张伟 138****6621", "客户信息表.csv", file_type="csv")]
    result = govern(docs)
    meta = result.documents[0].metadata
    assert meta["pii_flag"] is True
    assert meta["vectorize"] is False
    assert "客户信息表.csv" in result.report["pii_excluded"]


def test_governance_structured_not_vectorized() -> None:
    docs = [_doc("型号: XC-L100", "产品参数表.csv", file_type="csv")]
    result = govern(docs)
    meta = result.documents[0].metadata
    assert meta["structured"] is True
    assert meta["vectorize"] is False


def test_governance_explicit_duplicate_dropped() -> None:
    docs = [
        _doc("型号,价格\nXC-L100,1999", "产品参数表.csv", file_type="csv"),
        _doc("型号\t价格\nXC-L100\t1999", "产品参数表.tsv", file_type="tsv"),
    ]
    result = govern(docs)
    dropped = {d["dropped"] for d in result.report["duplicates"]}
    assert "产品参数表.tsv" in dropped
    tsv_doc = next(d for d in result.documents if d.metadata["source_file"] == "产品参数表.tsv")
    assert tsv_doc.metadata["vectorize"] is False
