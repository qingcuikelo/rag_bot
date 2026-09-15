"""集成测试：图的拒答/消歧/闲聊路径（离线，无需 LLM）。"""

from __future__ import annotations

import pytest

from xingchi_rag.graph.build import build_graph
from xingchi_rag.ingestion.pipeline import build

pytestmark = pytest.mark.integration


@pytest.fixture
def graph(isolated_storage):
    # 仅构建本地索引（BM25 + SQLite），不调用 Embedding API
    build(vectorize=False)
    return build_graph(with_checkpointer=False)


def test_unknown_model(graph) -> None:
    state = graph.invoke({"question": "XC-L300 卖多少钱"})
    assert state["refused"] is True
    assert "没有型号" in state["answer"]


def test_pii_requires_auth(graph) -> None:
    state = graph.invoke({"question": "帮我查一下张伟的联系电话"})
    assert state["refused"] is True
    assert "身份校验" in state["answer"]


def test_chitchat(graph) -> None:
    state = graph.invoke({"question": "你好"})
    assert state["route"] == "chitchat"
    assert state["answer"]
