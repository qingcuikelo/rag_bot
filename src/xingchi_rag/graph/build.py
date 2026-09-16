"""LangGraph 图装配（文档 §12.6）。"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from xingchi_rag.config import get_settings
from xingchi_rag.graph.nodes import (
    check,
    generate,
    grade_merge,
    refuse_or_human,
    retrieve,
    retry_once,
    router,
    sql_agent,
    understand,
)
from xingchi_rag.graph.state import GraphState


def _route_selector(state: GraphState) -> list[str]:
    """路由分流（price_stock 为 fan-out 并行）。"""
    route = state.get("route")
    if route == "knowledge":
        return ["retrieve"]
    if route == "structured":
        return ["sql_agent"]
    if route == "price_stock":
        return ["retrieve", "sql_agent"]
    return ["refuse_or_human"]


def _after_grade(state: GraphState) -> Literal["generate", "refuse_or_human"]:
    return "generate" if state.get("sufficient") else "refuse_or_human"


def _after_check(state: GraphState) -> Literal["__end__", "retry_once", "refuse_or_human"]:
    if state.get("grounded"):
        return "__end__"
    if state.get("retry", 0) < get_settings().check_retry_max:
        return "retry_once"
    return "refuse_or_human"


def build_state_graph() -> StateGraph:
    """构建未编译的 StateGraph。"""
    builder = StateGraph(GraphState)
    builder.add_node("understand", understand)
    builder.add_node("router", router)
    builder.add_node("retrieve", retrieve)
    builder.add_node("sql_agent", sql_agent)
    builder.add_node("grade_merge", grade_merge)
    builder.add_node("generate", generate)
    builder.add_node("check", check)
    builder.add_node("retry_once", retry_once)
    builder.add_node("refuse_or_human", refuse_or_human)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "router")
    builder.add_conditional_edges("router", _route_selector)
    builder.add_edge("retrieve", "grade_merge")
    builder.add_edge("sql_agent", "grade_merge")
    builder.add_conditional_edges("grade_merge", _after_grade)
    builder.add_edge("generate", "check")
    builder.add_conditional_edges("check", _after_check)
    builder.add_edge("retry_once", "generate")
    builder.add_edge("refuse_or_human", END)
    return builder


def new_checkpointer() -> Any:
    """为当前线程创建独立的 SQLite 检查点连接（并发安全，§并发优化）。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = get_settings()
    path = settings.resolve(settings.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        timeout=settings.sqlite_busy_timeout_ms / 1000,
    )
    conn.execute(f"PRAGMA busy_timeout = {settings.sqlite_busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode = WAL")
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(checkpointer: Any | None = None, *, with_checkpointer: bool = True) -> Any:
    """编译图。测试可传入 ``InMemorySaver`` 或 ``with_checkpointer=False``。"""
    builder = build_state_graph()
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    if with_checkpointer:
        return builder.compile(checkpointer=new_checkpointer())
    return builder.compile()


_thread_local = threading.local()


def get_graph() -> Any:
    """返回当前线程的生产图（每线程独立检查点连接，避免跨线程共享连接）。

    以检查点路径为键缓存，配置/存储路径变化时自动重建。
    """
    settings = get_settings()
    key = str(settings.resolve(settings.checkpoint_path))
    cache: dict[str, Any] | None = getattr(_thread_local, "graphs", None)
    if cache is None:
        cache = {}
        _thread_local.graphs = cache
    graph = cache.get(key)
    if graph is None:
        graph = build_graph(checkpointer=new_checkpointer())
        cache[key] = graph
    return graph


def reset_graph() -> None:
    """释放当前线程的全部图/检查点连接（测试与配置变更用）。"""
    cache: dict[str, Any] = getattr(_thread_local, "graphs", None) or {}
    for graph in cache.values():
        saver = getattr(graph, "checkpointer", None)
        conn = getattr(saver, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    _thread_local.graphs = {}
