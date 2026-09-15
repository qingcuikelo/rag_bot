"""集成测试：向量/BM25/混合检索召回（需要 Embedding API，无密钥则跳过）。"""

from __future__ import annotations

import pytest

from eval.run import evaluate_retrieval, load_golden
from xingchi_rag.config import get_settings
from xingchi_rag.ingestion.pipeline import build
from xingchi_rag.retrieval.bm25 import get_bm25_retriever
from xingchi_rag.retrieval.factory import get_retriever
from xingchi_rag.retrieval.rerank import rerank_disabled
from xingchi_rag.utils.docstore import load_documents

pytestmark = pytest.mark.integration

requires_api = pytest.mark.skipif(
    not get_settings().openai_api_key,
    reason="需要 OPENAI_API_KEY 才能构建向量索引",
)


def _available_files() -> set[str]:
    settings = get_settings()
    chunks = load_documents(settings.resolve(settings.bm25_path) / "chunks.jsonl")
    return {str(c.metadata.get("source_file", "")) for c in chunks}


@requires_api
def test_retrieval_recall(isolated_storage) -> None:
    build(vectorize=True)

    # BM25 单元行为
    bm25_docs = get_bm25_retriever(k=20).invoke("XC-L100 库存多少")
    assert bm25_docs

    records = load_golden(split="dev")
    available = _available_files()

    hybrid = get_retriever("hybrid", k=20)
    base_report = evaluate_retrieval(
        hybrid.invoke, records, k=5, top_n=20, available_files=available
    )
    assert base_report["usable_questions"] > 0
    # P2 门禁：Recall@5 ≥ 0.80
    assert base_report["metrics"]["recall@k"] >= 0.80

    # P2/P3：叠加 API 重排，MRR 目标 ≥ 0.80（重排额度耗尽时会熔断降级）
    reranked = get_retriever("hybrid", k=20, rerank=True, top_n=5)
    rerank_report = evaluate_retrieval(
        reranked.invoke, records, k=5, top_n=20, available_files=available
    )
    assert rerank_report["metrics"]["recall@k"] >= 0.80
    if not rerank_disabled():
        assert rerank_report["metrics"]["mrr"] >= 0.80
