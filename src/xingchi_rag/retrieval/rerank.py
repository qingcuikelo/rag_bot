"""API 重排（DashScope text-rerank，文档 §7.1）。

按用户决定采用 API 重排模型（``RERANK_MODEL``，默认 qwen3.x-text-rerank），
保留文档的 ``ContextualCompressionRetriever`` 结构，仅替换压缩器。
接口失败时降级为「跳过重排，保持原序」（§12.9）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from loguru import logger

from xingchi_rag.config import get_settings


class DashScopeReranker(BaseDocumentCompressor):
    """基于 DashScope 原生 text-rerank 接口的精排压缩器。"""

    top_n: int = 5
    model: str = "qwen3.7-text-rerank"
    endpoint: str = ""
    api_key: str = ""
    timeout: int = 30
    max_documents: int = 20

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """对候选文档重排并返回 Top-N（附 ``relevance_score``）。"""
        if not documents:
            return []

        subset = list(documents[: self.max_documents])
        scores = self._score(query, [doc.page_content for doc in subset])
        if scores is None:  # 降级：跳过重排
            return subset[: self.top_n]

        ranked = sorted(zip(subset, scores, strict=False), key=lambda pair: pair[1], reverse=True)
        results: list[Document] = []
        for doc, score in ranked[: self.top_n]:
            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "relevance_score": score,
                        "score_norm": score,
                    },
                )
            )
        return results

    def _score(self, query: str, texts: list[str]) -> list[float] | None:
        """调用重排接口，返回与 ``texts`` 对齐的分数；失败返回 ``None``。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {"query": query, "documents": texts},
            "parameters": {"top_n": len(texts), "return_documents": False},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                self.endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            results = response.json()["output"]["results"]
        except Exception as exc:  # 网络/限流/解析失败 -> 降级
            logger.warning(f"Reranker 失败，降级为不重排: {type(exc).__name__} {exc}")
            return None

        scores = [0.0] * len(texts)
        for item in results:
            index = int(item["index"])
            if 0 <= index < len(scores):
                scores[index] = float(item["relevance_score"])
        return scores


def build_reranker(top_n: int | None = None) -> DashScopeReranker:
    """按配置构建 API 重排器。"""
    settings = get_settings()
    return DashScopeReranker(
        top_n=top_n or settings.rerank_top_n,
        model=settings.rerank_model,
        endpoint=settings.rerank_endpoint,
        api_key=settings.openai_api_key,
        timeout=settings.request_timeout_s,
        max_documents=settings.rerank_max_documents,
    )
