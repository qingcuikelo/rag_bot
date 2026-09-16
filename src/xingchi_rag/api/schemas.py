"""API 请求/响应模型（文档 §12.4）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求。"""

    message: str
    session_id: str | None = None
    stream: bool = False
    customer_token: str | None = None


class Citation(BaseModel):
    """引用来源。"""

    source_file: str
    section: str | None = None
    chunk_id: str
    score: float | None = None


class ChatResponse(BaseModel):
    """问答响应。"""

    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    route: str = "unknown"
    prompt_version: str = ""
    latency_ms: int = 0
    trace_id: str | None = None
    handoff: bool = False
    handoff_reason: str | None = None


class ResumeRequest(BaseModel):
    """人工介入恢复请求（中断转人工）。"""

    session_id: str
    message: str


class FeedbackRequest(BaseModel):
    """用户反馈（用于回流金标）。"""

    session_id: str
    message_id: str | None = None
    rating: int = Field(ge=1, le=5)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    """反馈受理结果。"""

    accepted: bool = True


class HealthResponse(BaseModel):
    """健康检查。"""

    status: str = "ok"
    index_version: str
    collection_name: str
    embed_provider: str
    llm_provider: str
    auth_enabled: bool
