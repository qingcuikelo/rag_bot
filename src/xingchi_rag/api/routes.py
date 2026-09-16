"""FastAPI 路由与应用（文档 §12.4、§12.7、§8.4）。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from xingchi_rag.api.deps import resolve_customer_id, verify_service_key
from xingchi_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ResumeRequest,
)
from xingchi_rag.config import get_settings
from xingchi_rag.graph.build import get_graph
from xingchi_rag.observability import increment, metrics_snapshot, record_latency
from xingchi_rag.security import sanitize_output

router = APIRouter(prefix="/v1")


def _initial_state(message: str, customer_id: str | None, history: str = "") -> dict[str, Any]:
    return {
        "question": message,
        "question_raw": message,
        "history": history,
        "customer_id": customer_id,
        "authenticated": customer_id is not None,
        "enable_interrupt": True,
        "retry": 0,
        "strict": False,
    }


def _config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _history(session_id: str) -> str:
    """取会话上一次问题，用于多轮指代消解（§6.4）。"""
    try:
        snapshot = get_graph().get_state(_config(session_id))
    except Exception:
        return ""
    values = getattr(snapshot, "values", None) or {}
    if not isinstance(values, dict):
        return ""
    return str(values.get("standalone_query") or values.get("question") or "")


def _run_graph(message: str, session_id: str, customer_id: str | None) -> dict[str, Any]:
    graph = get_graph()
    history = _history(session_id)
    return graph.invoke(_initial_state(message, customer_id, history), _config(session_id))


def _citations(state: dict[str, Any]) -> list[Citation]:
    citations: list[Citation] = []
    for item in state.get("citations") or []:
        citations.append(
            Citation(
                source_file=str(item.get("source_file", "")),
                section=item.get("section"),
                chunk_id=str(item.get("chunk_id") or ""),
                score=item.get("score"),
            )
        )
    return citations


def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """提取中断信息（转人工）。"""
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", None)
    return value if isinstance(value, dict) else {"reason": "handoff", "suggested_reply": ""}


def _to_response(state: dict[str, Any], session_id: str, started: float) -> ChatResponse:
    latency = int((time.perf_counter() - started) * 1000)
    record_latency("chat", latency)

    payload = _interrupt_payload(state)
    if payload is not None:
        increment("handoff")
        return ChatResponse(
            session_id=session_id,
            answer=sanitize_output(str(payload.get("suggested_reply", ""))),
            refused=True,
            route=str(state.get("route", "unknown")),
            latency_ms=latency,
            handoff=True,
            handoff_reason=str(payload.get("reason", "handoff")),
        )

    refused = bool(state.get("refused"))
    increment("refused" if refused else "answered")
    return ChatResponse(
        session_id=session_id,
        answer=sanitize_output(str(state.get("answer", ""))),
        citations=_citations(state),
        refused=refused,
        route=str(state.get("route", "unknown")),
        prompt_version=str(state.get("prompt_version", "")),
        latency_ms=latency,
        handoff=bool(state.get("handoff")),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查（含索引版本与 Provider 状态）。"""
    from xingchi_rag.retrieval.store import collection_name

    settings = get_settings()
    return HealthResponse(
        status="ok",
        index_version=f"v{settings.index_version}",
        collection_name=collection_name(),
        embed_provider=settings.embed_provider.value,
        llm_provider=settings.llm_provider.value,
        auth_enabled=settings.auth_enabled,
    )


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    """轻量运行指标（耗时均值/计数）。"""
    return metrics_snapshot()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_service_key)])
def chat(request: ChatRequest) -> ChatResponse:
    """一次性问答。"""
    started = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    customer_id = resolve_customer_id(request.customer_token)
    try:
        state = _run_graph(request.message, session_id, customer_id)
    except Exception as exc:
        logger.exception(f"图执行失败: {exc}")
        increment("error")
        return ChatResponse(
            session_id=session_id,
            answer="抱歉，服务暂时不可用，请稍后重试或联系人工客服 400-820-6688。",
            refused=True,
            route="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return _to_response(state, session_id, started)


@router.post(
    "/chat/resume", response_model=ChatResponse, dependencies=[Depends(verify_service_key)]
)
def chat_resume(request: ResumeRequest) -> ChatResponse:
    """人工介入后恢复被中断的会话（§8.4）。"""
    from langgraph.types import Command

    started = time.perf_counter()
    try:
        state = get_graph().invoke(Command(resume=request.message), _config(request.session_id))
    except Exception as exc:
        logger.warning(f"恢复会话失败: {type(exc).__name__} {exc}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该会话不存在待恢复的中断",
        ) from exc
    return _to_response(state, request.session_id, started)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", dependencies=[Depends(verify_service_key)])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式问答：route → token → citations → done。"""
    session_id = request.session_id or str(uuid.uuid4())
    customer_id = resolve_customer_id(request.customer_token)

    def event_stream() -> Iterator[str]:
        started = time.perf_counter()
        try:
            state = _run_graph(request.message, session_id, customer_id)
        except Exception as exc:
            logger.exception(f"图执行失败: {exc}")
            increment("error")
            yield _sse("error", {"message": "服务暂时不可用"})
            yield _sse("done", {"session_id": session_id})
            return

        response = _to_response(state, session_id, started)
        yield _sse("route", {"route": response.route, "session_id": session_id})
        yield _sse("token", {"delta": response.answer})
        yield _sse(
            "citations",
            {
                "citations": [c.model_dump() for c in response.citations],
                "refused": response.refused,
            },
        )
        yield _sse(
            "done",
            {
                "session_id": session_id,
                "prompt_version": response.prompt_version,
                "latency_ms": response.latency_ms,
                "handoff": response.handoff,
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/feedback", response_model=FeedbackResponse, dependencies=[Depends(verify_service_key)]
)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """用户反馈落盘（用于回流金标）。"""
    settings = get_settings()
    path = settings.resolve(settings.quality_report_path).parent / "feedback.jsonl"
    record = {
        "received_at": datetime.now(UTC).isoformat(),
        **request.model_dump(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FeedbackResponse(accepted=True)


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="星驰科技 RAG 智能客服", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
