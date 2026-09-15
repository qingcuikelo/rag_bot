"""LangGraph 节点实现（文档 §12.6）。

节点：understand / router / retrieve / sql_agent / grade_merge /
generate / check / retry_once / refuse_or_human。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from loguru import logger

from xingchi_rag.config import get_settings
from xingchi_rag.generation.answer import (
    CitationModel,
    check_grounded,
    generate_answer,
)
from xingchi_rag.graph.intent import classify_intent, is_pii_request, route_for_intent
from xingchi_rag.graph.state import Evidence, GraphState
from xingchi_rag.providers.llm import get_llm
from xingchi_rag.retrieval.factory import get_retriever
from xingchi_rag.sql.agent import execute_select, run_sql_agent
from xingchi_rag.utils.naming import (
    ambiguous_options,
    detect_unknown_model,
    normalize_model,
)

REFUSAL_TEXT = (
    "抱歉，当前知识库中没有找到支持该问题的明确依据，为避免提供不准确的信息，"
    "建议您联系人工客服进一步确认。\n"
    "客服热线：400-820-6688；服务邮箱：service@xingchi-tech.example"
)
UNKNOWN_MODEL_TEXT = "抱歉，星驰目前没有型号 {model} 的产品，请确认型号是否正确。"
PII_TEXT = (
    "为保护您的个人信息，查询客户信息或工单需要先完成身份校验。"
    "请通过身份校验后重试，或联系人工客服 400-820-6688。"
)
CHITCHAT_TEXT = (
    "您好，我是星驰科技智能客服，可以为您解答智能门锁、摄像头、门铃等产品的"
    "参数、价格、库存与售后服务问题。请问有什么可以帮您？"
)
SQL_SOURCE = "结构化数据库"


def _to_doc(evidence: Evidence) -> Document:
    return Document(
        page_content=evidence["page_content"], metadata=dict(evidence.get("metadata", {}))
    )


def _to_evidence(doc: Document) -> Evidence:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}


# ----------------------------------------------------------------------
# 查询理解与路由
# ----------------------------------------------------------------------
def understand(state: GraphState) -> dict[str, Any]:
    """改写/意图分类/型号归一。"""
    question = state.get("question") or state.get("question_raw") or ""
    intent = classify_intent(question)
    model = normalize_model(question)
    unknown_model = detect_unknown_model(question)
    options = ambiguous_options(question) if intent in {"product_spec", "price_stock"} else None
    return {
        "standalone_query": question,
        "intent": intent,
        "product_model": model,
        "product_model_explicit": model is not None,
        "unknown_model": unknown_model,
        "pii_request": is_pii_request(question),
        "options": options,
        "retry": state.get("retry", 0),
        "strict": state.get("strict", False),
    }


def router(state: GraphState) -> dict[str, Any]:
    """决定路由（写入 state，条件边据此分流）。"""
    if state.get("unknown_model"):
        return {"route": "unknown"}
    if state.get("pii_request") and not state.get("authenticated"):
        return {"route": "unknown"}
    return {"route": route_for_intent(state.get("intent", "unknown"))}


# ----------------------------------------------------------------------
# 检索 / 结构化
# ----------------------------------------------------------------------
def retrieve(state: GraphState) -> dict[str, Any]:
    """混合检索 + 重排。"""
    settings = get_settings()
    model = state.get("product_model") if state.get("product_model_explicit") else None
    use_filter = model and state.get("intent") in {"product_spec", "price_stock"}
    metadata_filter = {"product_model": model} if use_filter else None

    retriever = get_retriever(
        "hybrid",
        k=settings.retrieve_k,
        filter=metadata_filter,
        rerank=True,
        top_n=settings.rerank_top_n,
    )
    query = state.get("standalone_query") or state.get("question", "")
    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.warning(f"检索失败: {type(exc).__name__} {exc}")
        return {"evidence": [], "retrieve_sufficient": False, "error": str(exc)}

    raw_scores = [doc.metadata.get("score_norm") for doc in docs]
    has_rerank = any(score is not None for score in raw_scores)
    if has_rerank:
        scores = [float(score or 0.0) for score in raw_scores]
        top1 = scores[0] if scores else 0.0
        count_above = sum(1 for score in scores if score >= 0.30)
        sufficient = top1 >= settings.grade_score_threshold and count_above >= 1
    else:
        # 重排不可用（§12.9 降级）：以 RRF 融合序为准，有候选即认为可作答
        sufficient = len(docs) > 0
    return {
        "evidence": [_to_evidence(doc) for doc in docs],
        "retrieve_sufficient": sufficient,
    }


def sql_agent(state: GraphState) -> dict[str, Any]:
    """结构化查询（Text2SQL）。"""
    text, success = run_sql_agent(state.get("question", ""))
    return {"sql_result": text, "sql_sufficient": bool(success and text)}


# ----------------------------------------------------------------------
# 判据汇合 / 生成 / 校验
# ----------------------------------------------------------------------
def grade_merge(state: GraphState) -> dict[str, Any]:
    """证据充分性判据（§12.5）。"""
    route = state.get("route")
    evidence: list[Evidence] = list(state.get("evidence") or [])

    sql_result = state.get("sql_result") or ""
    if sql_result:
        evidence.append(
            {
                "page_content": f"结构化查询结果：{sql_result}",
                "metadata": {
                    "source_file": SQL_SOURCE,
                    "section": "查询结果",
                    "chunk_id": "sql-result",
                },
            }
        )

    retrieve_sufficient = bool(state.get("retrieve_sufficient"))
    sql_sufficient = bool(state.get("sql_sufficient"))

    if route == "knowledge":
        sufficient = retrieve_sufficient
    elif route == "structured":
        sufficient = sql_sufficient
    elif route == "price_stock":
        sufficient = retrieve_sufficient or sql_sufficient
    else:
        sufficient = False

    return {"sufficient": sufficient, "evidence": evidence}


def generate(state: GraphState) -> dict[str, Any]:
    """基于证据生成结构化回复。"""
    docs = [_to_doc(item) for item in state.get("evidence") or []]
    result, version = generate_answer(
        get_llm(),
        state.get("question", ""),
        docs,
        strict=bool(state.get("strict")),
    )
    return {
        "answer": result.answer,
        "citations": [citation.model_dump() for citation in result.citations],
        "refused": result.refused,
        "prompt_version": version,
    }


def check(state: GraphState) -> dict[str, Any]:
    """引用/数值依据校验（§12.5）。"""
    docs = [_to_doc(item) for item in state.get("evidence") or []]
    citations = [CitationModel.model_validate(item) for item in state.get("citations") or []]
    result = check_grounded(state.get("answer", ""), citations, docs)
    return {"grounded": result.grounded}


def retry_once(state: GraphState) -> dict[str, Any]:
    """以更严格 Prompt 重试一次。"""
    return {"retry": state.get("retry", 0) + 1, "strict": True}


def _build_disambiguation(options: list[str]) -> str:
    """型号消歧：列举候选型号的价格与解锁方式（§8.5）。"""
    quoted = ",".join(f"'{model}'" for model in options)
    sql = (
        "SELECT p.model, p.name, p.unlock_methods, p.sale_status, pr.retail_price "
        "FROM product p LEFT JOIN price pr ON p.model = pr.model "
        f"WHERE p.model IN ({quoted}) ORDER BY pr.retail_price"
    )
    try:
        _, rows = execute_select(sql)
    except Exception as exc:
        logger.warning(f"消歧查询失败: {type(exc).__name__} {exc}")
        return "请问您想了解哪个型号？可提供的型号有：" + "、".join(options) + "。"

    lines = ["星驰目前在售/预售的相关型号有："]
    for model, name, unlock, status, price in rows:
        price_text = f"{price} 元" if price is not None else "价格待定"
        lines.append(f"· {model}（{name}）{price_text} · {unlock or '—'} · {status}")
    lines.append("请问您想了解哪一款？")
    return "\n".join(lines)


def refuse_or_human(state: GraphState) -> dict[str, Any]:
    """拒答 / 消歧 / 引导人工。"""
    unknown_model = state.get("unknown_model")
    options = state.get("options") or []
    route = state.get("route")

    if unknown_model:
        answer = UNKNOWN_MODEL_TEXT.format(model=unknown_model)
    elif state.get("pii_request") and not state.get("authenticated"):
        answer = PII_TEXT
    elif route == "chitchat":
        answer = CHITCHAT_TEXT
    elif options:
        answer = _build_disambiguation(options)
    else:
        answer = REFUSAL_TEXT

    return {
        "answer": answer,
        "citations": [],
        "refused": True,
        "grounded": True,
        "handoff": route not in {"chitchat"},
    }
