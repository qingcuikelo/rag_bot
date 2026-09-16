"""生成层评测（文档 §9.2、§9.3）。

默认使用内置 LLM-as-Judge（无需 RAGAS）；若安装了 ``requirements-eval.txt``
（独立环境）可通过 ``evaluate_with_ragas`` 切换到 RAGAS 指标。

指标：拒答准确率、引用准确率、忠实度、相关度、正确性。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from eval.metrics import mean
from xingchi_rag.generation.answer import (
    CitationModel,
    format_evidence,
    validate_citations,
)

# 期望拒答的题型
REFUSAL_TYPES = {"unanswerable", "pii"}
# 不参与拒答准确率统计的题型
SKIP_REFUSAL_TYPES = {"chitchat"}

JUDGE_PROMPT = """你是严格的问答质量评审员。请依据给定证据，对客服回答打分（0~1）。

评分维度：
- faithfulness：回答是否完全由证据支撑，无编造（1=完全有据，0=大量编造）；
- answer_relevance：回答是否切题、解决了用户问题；
- answer_correctness：回答与期望答案要点是否一致。

只输出 JSON：{{"faithfulness": 0-1, "answer_relevance": 0-1, "answer_correctness": 0-1}}

用户问题：{question}

期望答案要点：{expected}

证据：
{evidence}

客服回答：
{answer}
"""


class JudgeScores(BaseModel):
    """LLM 评审分数。"""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    answer_correctness: float = Field(ge=0.0, le=1.0)


def _docs(evidence: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(page_content=item.get("page_content", ""), metadata=dict(item.get("metadata", {})))
        for item in evidence
    ]


def judge_answer(
    llm: BaseChatModel,
    question: str,
    answer: str,
    evidence_text: str,
    expected: str,
) -> JudgeScores | None:
    """LLM-as-Judge 打分；失败返回 ``None``。"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected or "（无）",
        evidence=evidence_text or "（无）",
        answer=answer,
    )
    messages = [("system", prompt)]
    try:
        structured = llm.with_structured_output(JudgeScores)
        return JudgeScores.model_validate(structured.invoke(messages))
    except Exception:
        try:
            text = llm.invoke(messages).content
            payload = json.loads(str(text))
            return JudgeScores.model_validate(payload)
        except Exception:
            return None


def evaluate_generation(
    run_fn: Callable[[str], dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    judge_llm: BaseChatModel | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """运行生成评测。

    Args:
        run_fn: 问题 → 图状态（含 answer/refused/citations/evidence）。
        records: 金标记录。
        judge_llm: 用于 LLM-as-Judge 的模型；为 ``None`` 时跳过打分指标。
        limit: 仅评测前 N 条（便于快速回归）。
    """
    selected = records[:limit] if limit else records

    refusal_correct: list[float] = []
    citation_scores: list[float] = []
    faithfulness: list[float] = []
    relevance: list[float] = []
    correctness: list[float] = []
    per_type: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    refused_count = 0

    for record in selected:
        try:
            state = run_fn(record["question"])
        except Exception as exc:  # 单条异常不中断整体评测
            failures.append(
                {
                    "id": record["id"],
                    "type": record.get("type"),
                    "reason": f"执行异常: {type(exc).__name__}",
                }
            )
            continue
        answer = str(state.get("answer", ""))
        refused = bool(state.get("refused"))
        if refused:
            refused_count += 1
        evidence = state.get("evidence") or []
        citations = [CitationModel.model_validate(c) for c in (state.get("citations") or [])]
        docs = _docs(evidence)
        qtype = record.get("type", "unknown")

        expected_refused = qtype in REFUSAL_TYPES
        if qtype not in SKIP_REFUSAL_TYPES:
            ok = refused == expected_refused
            refusal_correct.append(1.0 if ok else 0.0)
            if not ok:
                failures.append(
                    {
                        "id": record["id"],
                        "type": qtype,
                        "reason": "拒答判定不符",
                        "answer": answer[:120],
                    }
                )

        if not refused:
            citation_ok, unknown = validate_citations(answer, citations, docs)
            citation_scores.append(1.0 if citation_ok else 0.0)
            if not citation_ok:
                failures.append(
                    {"id": record["id"], "type": qtype, "reason": "引用不实", "unknown": unknown}
                )
            per_type[qtype].append(1.0 if citation_ok else 0.0)

            if judge_llm is not None:
                scores = judge_answer(
                    judge_llm,
                    record["question"],
                    answer,
                    format_evidence(docs),
                    record.get("expected_answer", ""),
                )
                if scores is not None:
                    faithfulness.append(scores.faithfulness)
                    relevance.append(scores.answer_relevance)
                    correctness.append(scores.answer_correctness)

    metrics: dict[str, float] = {
        "refusal_accuracy": mean(refusal_correct),
    }
    # 仅在存在作答样本时统计引用/生成质量指标，避免空集被记为 0
    if citation_scores:
        metrics["citation_accuracy"] = mean(citation_scores)
    if faithfulness:
        metrics["faithfulness"] = mean(faithfulness)
        metrics["answer_relevance"] = mean(relevance)
        metrics["answer_correctness"] = mean(correctness)

    return {
        "records": len(selected),
        "answered": len(citation_scores),
        "refused": refused_count,
        "metrics": metrics,
        "by_type_citation": {t: mean(v) for t, v in sorted(per_type.items())},
        "failures": failures,
    }


def evaluate_with_ragas(
    samples: list[dict[str, Any]],
    llm: BaseChatModel,
    embeddings: Any,
) -> dict[str, float]:
    """RAGAS 指标（需在独立环境安装 requirements-eval.txt）。"""
    try:
        from ragas import (  # type: ignore[import-not-found]
            EvaluationDataset,
            SingleTurnSample,
            evaluate,
        )
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "未安装 RAGAS。请在独立环境安装 requirements-eval.txt（见该文件说明）。"
        ) from exc

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s.get("expected_answer", ""),
            )
            for s in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    return {key: float(value) for key, value in dict(result).items()}
