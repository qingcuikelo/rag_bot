"""单元测试：SQL 只读校验与执行。"""

from __future__ import annotations

import sqlite3

import pytest

from xingchi_rag.sql.agent import execute_select, guard_sql


def test_guard_sql_adds_limit() -> None:
    sql = guard_sql("SELECT * FROM product")
    assert "LIMIT 100" in sql.upper()


def test_guard_sql_caps_limit() -> None:
    sql = guard_sql("SELECT * FROM product LIMIT 500")
    assert "LIMIT 100" in sql.upper()


def test_guard_sql_rejects_non_select() -> None:
    with pytest.raises(ValueError):
        guard_sql("DROP TABLE product")
    with pytest.raises(ValueError):
        guard_sql("SELECT 1; SELECT 2")
    with pytest.raises(ValueError):
        guard_sql("DELETE FROM product")


def test_execute_select_ro(tmp_path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE product(model TEXT, name TEXT)")
    conn.executemany(
        "INSERT INTO product VALUES(?,?)",
        [("XC-L100", "Pro"), ("XC-L200", "Max")],
    )
    conn.commit()
    conn.close()

    columns, rows = execute_select("SELECT model, name FROM product ORDER BY model", db_path=db)
    assert columns == ["model", "name"]
    assert rows == [("XC-L100", "Pro"), ("XC-L200", "Max")]


def test_execute_select_rejects_write(tmp_path) -> None:
    db = tmp_path / "t.db"
    sqlite3.connect(db).close()
    with pytest.raises(ValueError):
        execute_select("UPDATE product SET name='x'", db_path=db)
