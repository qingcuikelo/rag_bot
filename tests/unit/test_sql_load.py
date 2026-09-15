"""单元测试：PII 处理与 SQLite 入库辅助函数。"""

from __future__ import annotations

from xingchi_rag.sql.load import hash_phone, mask_name, phone_tail


def test_mask_name() -> None:
    assert mask_name("张伟") == "张*"
    assert mask_name("欧阳锋") == "欧**"
    assert mask_name("李") == "李"
    assert mask_name("") == ""


def test_hash_phone_is_salted_and_stable() -> None:
    a = hash_phone("138****6621", "salt-a")
    b = hash_phone("138****6621", "salt-a")
    c = hash_phone("138****6621", "salt-b")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_phone_tail() -> None:
    assert phone_tail("138****6621") == "6621"
    assert phone_tail("13812345678") == "5678"
    assert phone_tail("") == ""
