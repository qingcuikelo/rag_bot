"""Text2SQL Agent（文档 §6.3、§12.9）。

- 基于 ``create_agent`` + 四个 SQL 工具；
- ``sqlglot`` 二次校验：仅允许单条 ``SELECT``，强制 ``LIMIT``；
- 连接只读、超时保护。
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlglot
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from loguru import logger
from sqlglot import exp

from xingchi_rag.config import get_settings
from xingchi_rag.providers.llm import get_llm

DEFAULT_MAX_ROWS = 100
SYSTEM_PROMPT = (
    "你是星驰科技结构化数据查询助手。仅使用提供的 SQL 工具查询 SQLite 数据库，"
    "只允许 SELECT 查询。步骤：先 list_tables 查看表名，再 schema 查看字段，"
    "然后编写 SELECT 语句，先经 query_checker 校验，再执行 query。"
    "用中文简要汇报查询结果（含关键数值），不要编造数据。"
)


def guard_sql(sql: str, *, dialect: str = "sqlite", max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """校验并改写 SQL：仅单条 SELECT，强制 LIMIT ≤ max_rows。

    Raises:
        ValueError: 非单条语句、非 SELECT、包含危险关键字。
    """
    if not sql or not sql.strip():
        raise ValueError("SQL 为空")

    statements = sqlglot.parse(sql, read=dialect)
    if len(statements) != 1:
        raise ValueError("仅允许单条 SQL 语句")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise ValueError("仅允许 SELECT 查询")

    lowered = sql.lower()
    for forbidden in ("pragma", "attach", "detach", "drop", "insert", "update", "delete"):
        if f" {forbidden} " in f" {lowered} ":
            raise ValueError(f"SQL 含被禁止的关键字: {forbidden}")

    limit = statement.args.get("limit")
    if limit is None:
        statement = statement.limit(max_rows)
    else:
        try:
            current = int(limit.expression.name)
            if current > max_rows:
                statement = statement.limit(max_rows)
        except Exception:  # 无法解析 LIMIT -> 覆盖为安全值
            statement = statement.limit(max_rows)

    return statement.sql(dialect=dialect)


def execute_select(
    sql: str,
    *,
    db_path: str | Path | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: int = 5,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """以只读方式执行 SELECT，返回 ``(列名, 行)``。"""
    settings = get_settings()
    target = settings.resolve(Path(db_path) if db_path else settings.sqlite_path)
    safe_sql = guard_sql(sql, max_rows=max_rows)

    uri = f"file:{target.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    try:
        cursor = conn.execute(safe_sql)
        columns = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchmany(max_rows)
        return columns, rows
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_sql_database() -> SQLDatabase:
    """连接结构化库（只读语义通过工具校验保证）。"""
    settings = get_settings()
    target = settings.resolve(settings.sqlite_path)
    return SQLDatabase.from_uri(f"sqlite:///{target.as_posix()}")


@tool
def sql_db_list_tables() -> str:
    """列出数据库中的表名，多个以逗号分隔。"""
    return ", ".join(get_sql_database().get_usable_table_names())


@tool
def sql_db_schema(table_names: str) -> str:
    """给定逗号分隔的表名，返回其建表语句与示例数据。"""
    names = [name.strip() for name in table_names.split(",") if name.strip()]
    return get_sql_database().get_table_info(names)


@tool
def sql_db_query_checker(query: str) -> str:
    """校验 SQL 是否合法（仅 SELECT、含 LIMIT）。返回改写后的安全 SQL 或错误信息。"""
    try:
        return f"校验通过，可执行的安全 SQL：{guard_sql(query)}"
    except ValueError as exc:
        return f"校验失败：{exc}"


@tool
def sql_db_query(query: str) -> str:
    """执行只读 SELECT 查询，返回结果行（最多 100 行）。"""
    try:
        columns, rows = execute_select(query)
    except ValueError as exc:
        return f"查询被拒绝：{exc}"
    except Exception as exc:
        return f"查询执行失败：{type(exc).__name__}: {exc}"
    if not rows:
        return "查询成功，但无匹配数据（0 行）。"
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    return f"列: {header}\n共 {len(rows)} 行:\n{body}"


SQL_TOOLS = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]


def build_sql_agent(llm: BaseChatModel) -> Runnable:
    """组装 SQL 查询 Agent。"""
    return create_agent(llm, SQL_TOOLS, system_prompt=SYSTEM_PROMPT)


def run_sql_agent(question: str, llm: BaseChatModel | None = None) -> tuple[str, bool]:
    """执行结构化查询，返回 ``(结果文本, 是否成功)``。"""
    if llm is None:
        llm = get_llm()
    agent = build_sql_agent(llm)
    try:
        result = agent.invoke({"messages": [("user", question)]})
    except Exception as exc:
        logger.warning(f"SQL Agent 失败: {type(exc).__name__} {exc}")
        return f"结构化查询失败：{type(exc).__name__}", False

    messages = result.get("messages", []) if isinstance(result, dict) else []
    text = str(messages[-1].content) if messages else ""
    success = bool(text) and "查询" not in text[:4]
    return text, success
