"""生成与依据校验（文档 §8.1、§12.5）。

LCEL 生成结构化回复，并对引用/数值做 groundedness 校验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from xingchi_rag.generation.prompts import get_prompt, render

GENERATE_PROMPT = "generate.v1"
MAX_EVIDENCE_CHARS = 700
GENERATION_FALLBACK = "抱歉，回答生成服务暂时不可用，请稍后重试或联系人工客服 400-820-6688。"

# 引用形如 [来源: 文件名 · 章节]
_CITATION_RE = re.compile(r"\[来源:\s*([^\]·]+?)\s*(?:·\s*([^\]]+?))?\s*\]")
# 客服热线等固定联系信息：不参与数值校验
_CONTACT_RE = re.compile(r"400[-\s]?820[-\s]?6688")
# 关键数值：数字 + 单位
_UNIT_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:年|元|台|个|小时|天|毫米|毫安|mAh|%)")
# 关键数值：3 位及以上整数（价格、数量）
_BIG_NUMBER_RE = re.compile(r"(?<![\d.])\d{3,}(?![\d.])")


class CitationModel(BaseModel):
    """结构化引用。"""

    source_file: str
    section: str | None = None
    chunk_id: str | None = None
    score: float | None = None


class GeneratedAnswer(BaseModel):
    """生成结果（结构化）。"""

    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    refused: bool = False


@dataclass
class GroundCheckResult:
    """依据校验结果。"""

    grounded: bool
    citation_ok: bool
    numbers_ok: bool
    unknown_citations: list[str]
    unknown_numbers: list[str]


def format_evidence(documents: list[Document]) -> str:
    """把证据装配为带来源标注的文本。"""
    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", "未知来源")
        section = doc.metadata.get("section") or ""
        header = f"[来源: {source}" + (f" · {section}" if section else "") + "]"
        content = doc.page_content
        if len(content) > MAX_EVIDENCE_CHARS:
            content = content[:MAX_EVIDENCE_CHARS].rstrip() + "…"
        blocks.append(f"{index}. {header}\n{content}")
    return "\n\n".join(blocks) if blocks else "（无证据）"


def evidence_sources(documents: list[Document]) -> set[str]:
    """证据涉及的文件名集合。"""
    return {str(doc.metadata.get("source_file", "")) for doc in documents}


def validate_citations(
    answer: str, citations: list[CitationModel], documents: list[Document]
) -> tuple[bool, list[str]]:
    """校验回答中的引用是否都能映射到本次证据。"""
    sources = evidence_sources(documents)
    unknown: list[str] = []

    for match in _CITATION_RE.finditer(answer or ""):
        source_file = match.group(1).strip()
        if source_file not in sources:
            unknown.append(source_file)

    for citation in citations:
        if citation.source_file not in sources:
            unknown.append(citation.source_file)

    return (not unknown), unknown


def _critical_numbers(text: str) -> set[str]:
    """提取需要校验的关键数值（数字+单位，或 ≥3 位整数）。"""
    cleaned = _CONTACT_RE.sub(" ", text or "")
    numbers = set(_UNIT_NUMBER_RE.findall(cleaned))
    numbers |= set(_BIG_NUMBER_RE.findall(cleaned))
    return numbers


def validate_numbers(answer: str, documents: list[Document]) -> tuple[bool, list[str]]:
    """校验回答中的关键数值（价格/年限/数量等）是否能在证据文本中匹配。"""
    evidence_text = "\n".join(doc.page_content for doc in documents)
    evidence_numbers = _critical_numbers(evidence_text)
    answer_numbers = _critical_numbers(answer)
    unknown = sorted(number for number in answer_numbers if number not in evidence_numbers)
    return (not unknown), unknown


def check_grounded(
    answer: str, citations: list[CitationModel], documents: list[Document]
) -> GroundCheckResult:
    """综合引用与数值校验。"""
    citation_ok, unknown_citations = validate_citations(answer, citations, documents)
    numbers_ok, unknown_numbers = validate_numbers(answer, documents)
    return GroundCheckResult(
        grounded=citation_ok and numbers_ok,
        citation_ok=citation_ok,
        numbers_ok=numbers_ok,
        unknown_citations=unknown_citations,
        unknown_numbers=unknown_numbers,
    )


def generate_answer(
    llm: BaseChatModel,
    question: str,
    documents: list[Document],
    *,
    profile: str = "",
    strict: bool = False,
) -> tuple[GeneratedAnswer, str]:
    """基于证据生成结构化回复，返回 ``(结果, prompt_version)``。"""
    template, version = get_prompt(GENERATE_PROMPT)
    system = render(
        template,
        question=question,
        evidence=format_evidence(documents),
        profile=profile or "（无）",
    )
    messages: list[tuple[str, str]] = [("system", system)]
    if strict:
        messages.append(
            (
                "human",
                "请务必严格只使用证据中的数字与来源；无法在证据中找到的数字不得出现。",
            )
        )

    try:
        structured = llm.with_structured_output(GeneratedAnswer)
        result = structured.invoke(messages)
        return GeneratedAnswer.model_validate(result), version
    except Exception as exc:  # 结构化输出不可用/超时 -> 纯文本兜底
        logger.warning(f"结构化生成失败({type(exc).__name__})，降级为文本模式")

    text: object | None = None
    last_exc: Exception | None = None
    for _ in range(2):  # 超时/限流抖动：重试一次
        try:
            text = llm.invoke(messages).content
            break
        except Exception as exc:
            last_exc = exc
    if text is None:
        logger.error(f"生成服务不可用: {type(last_exc).__name__} {last_exc}")
        return GeneratedAnswer(answer=GENERATION_FALLBACK, citations=[], refused=True), version

    try:
        import json

        payload = json.loads(str(text))
        return GeneratedAnswer.model_validate(payload), version
    except Exception:  # 非 JSON 文本 -> 原样返回
        return GeneratedAnswer(answer=str(text), citations=[], refused=False), version
