"""单元测试：型号归一化与消歧。"""

from __future__ import annotations

from xingchi_rag.utils.naming import (
    ambiguous_options,
    category_of,
    detect_unknown_model,
    is_known_model,
    normalize_model,
)


def test_normalize_model_explicit() -> None:
    assert normalize_model("XC-L100 多少钱") == "XC-L100"
    assert normalize_model("xcl200 库存") == "XC-L200"


def test_normalize_model_alias() -> None:
    assert normalize_model("L200 保修几年") == "XC-L200"
    assert normalize_model("门锁Pro 支持什么") == "XC-L100"


def test_normalize_model_none() -> None:
    assert normalize_model("你们支持以旧换新吗") is None


def test_detect_unknown_model() -> None:
    assert detect_unknown_model("XC-L300 卖多少钱") == "XC-L300"
    assert detect_unknown_model("XC-L100 多少钱") is None


def test_ambiguous_options() -> None:
    assert ambiguous_options("门锁多少钱") == ["XC-L50", "XC-L100", "XC-L200"]
    assert ambiguous_options("XC-L100 多少钱") is None


def test_is_known_model_and_category() -> None:
    assert is_known_model("XC-L100") is True
    assert is_known_model("XC-L300") is False
    assert category_of("XC-C50") == "智能摄像头"
