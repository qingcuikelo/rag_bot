"""单元测试：查询理解（含多轮指代消解）。"""

from __future__ import annotations

from xingchi_rag.graph.nodes import understand


def test_understand_basic() -> None:
    result = understand({"question": "XC-L100 库存多少"})
    assert result["product_model"] == "XC-L100"
    assert result["product_model_explicit"] is True
    assert result["standalone_query"] == "XC-L100 库存多少"


def test_understand_rewrites_pronoun_with_history() -> None:
    result = understand({"question": "它保修几年？", "history": "XC-L100 支持哪些解锁方式"})
    assert result["product_model"] == "XC-L100"
    assert "XC-L100" in result["standalone_query"]


def test_understand_no_rewrite_without_history() -> None:
    result = understand({"question": "它保修几年？"})
    assert result["product_model"] is None
    assert result["standalone_query"] == "它保修几年？"
