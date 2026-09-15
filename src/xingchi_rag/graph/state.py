"""LangGraph 状态定义（文档 §12.6）。"""

from __future__ import annotations

from typing import Any, TypedDict


class Evidence(TypedDict, total=False):
    """序列化后的证据块。"""

    page_content: str
    metadata: dict[str, Any]


class GraphState(TypedDict, total=False):
    """在线问答状态。"""

    # 输入
    question: str
    question_raw: str

    # 查询理解
    standalone_query: str
    intent: str
    route: str
    product_model: str | None
    product_model_explicit: bool
    unknown_model: str | None
    pii_request: bool
    options: list[str] | None

    # 认证
    customer_id: str | None
    authenticated: bool

    # 检索/SQL
    evidence: list[Evidence]
    retrieve_sufficient: bool
    sql_result: str
    sql_sufficient: bool
    sufficient: bool

    # 生成/判据
    answer: str
    citations: list[dict[str, Any]]
    refused: bool
    grounded: bool
    retry: int
    strict: bool
    prompt_version: str
    handoff: bool
    error: str | None
