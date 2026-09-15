"""集成测试：图的知识问答生成路径（需要 LLM+Embedding API）。"""

from __future__ import annotations

import pytest

from xingchi_rag.config import get_settings
from xingchi_rag.graph.build import build_graph
from xingchi_rag.ingestion.pipeline import build

pytestmark = pytest.mark.integration

requires_api = pytest.mark.skipif(
    not get_settings().openai_api_key,
    reason="需要 OPENAI_API_KEY 才能端到端生成",
)


@requires_api
def test_answer_policy_question(isolated_storage) -> None:
    build(vectorize=True)
    graph = build_graph(with_checkpointer=False)
    state = graph.invoke({"question": "智能门锁整机保修几年？"})
    assert state["route"] == "knowledge"
    assert state["refused"] is False
    assert state["grounded"] is True
    assert state["citations"]
    assert "3" in state["answer"]


@requires_api
def test_unanswerable_refused(isolated_storage) -> None:
    build(vectorize=True)
    graph = build_graph(with_checkpointer=False)
    state = graph.invoke({"question": "支持以旧换新吗？"})
    assert state["refused"] is True
