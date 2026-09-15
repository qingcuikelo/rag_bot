"""单元测试：API 重排（无网络，桩 _score）。"""

from __future__ import annotations

from langchain_core.documents import Document

from xingchi_rag.retrieval.rerank import DashScopeReranker


class _StubReranker(DashScopeReranker):
    stub_scores: list[float] | None = None

    def _score(self, query: str, texts: list[str]) -> list[float] | None:
        return self.stub_scores


def _docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"chunk_id": "a"}),
        Document(page_content="B", metadata={"chunk_id": "b"}),
        Document(page_content="C", metadata={"chunk_id": "c"}),
    ]


def test_rerank_orders_and_truncates() -> None:
    reranker = _StubReranker(top_n=2, stub_scores=[0.1, 0.9, 0.5])
    result = reranker.compress_documents(_docs(), "q")
    assert [d.page_content for d in result] == ["B", "C"]
    assert result[0].metadata["relevance_score"] == 0.9
    assert result[0].metadata["score_norm"] == 0.9
    assert result[0].metadata["chunk_id"] == "b"


def test_rerank_degrades_on_failure() -> None:
    reranker = _StubReranker(top_n=2, stub_scores=None)
    result = reranker.compress_documents(_docs(), "q")
    # 降级：保持原序并截断到 top_n
    assert [d.page_content for d in result] == ["A", "B"]


def test_rerank_empty_documents() -> None:
    reranker = _StubReranker(top_n=2, stub_scores=[])
    assert reranker.compress_documents([], "q") == []


def test_rerank_respects_max_documents() -> None:
    reranker = _StubReranker(top_n=5, max_documents=2, stub_scores=[0.1, 0.9])
    result = reranker.compress_documents(_docs(), "q")
    assert len(result) == 2
    assert result[0].page_content == "B"
