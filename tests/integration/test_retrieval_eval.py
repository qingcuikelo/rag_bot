"""集成测试：向量/BM25/混合检索召回（需要 Embedding API，无密钥则跳过）。"""

from __future__ import annotations

import pytest

from eval.run import evaluate_retrieval, load_golden
from xingchi_rag.config import get_settings
from xingchi_rag.ingestion.pipeline import build
from xingchi_rag.retrieval.bm25 import get_bm25_retriever
from xingchi_rag.retrieval.factory import get_retriever
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
    retriever = get_retriever("hybrid", k=20)
    report = evaluate_retrieval(
        retriever.invoke, records, k=5, top_n=20, available_files=_available_files()
    )
    assert report["usable_questions"] > 0
    # P2 门禁：Recall@5 ≥ 0.80
    assert report["metrics"]["recall@k"] >= 0.80
