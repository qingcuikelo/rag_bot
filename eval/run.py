"""检索评测（文档 §9.3）。

以金标 ``gold_source``（``文件名#章节``）与检索结果的 metadata/内容比对，
计算 Recall@k / Precision@k / MRR / nDCG@k / Hit Rate，并按 type 分层统计。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from eval.metrics import (
    hit,
    mean,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)

GOLDEN_PATH = Path("eval/golden/golden_set.jsonl")

# 非检索路径（结构化/鉴权/拒答），不参与检索召回评测
NON_RETRIEVAL_TYPES = {"ticket", "pii", "unanswerable", "chitchat"}


def load_golden(path: str | Path = GOLDEN_PATH, split: str | None = None) -> list[dict[str, Any]]:
    """加载金标集（可按 dev/holdout 过滤）。"""
    target = Path(path)
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            records.append(record)
    return records


def _parse_source(source: str) -> tuple[str, str]:
    file, _, section = source.partition("#")
    return file, section


def is_relevant(doc: Document, gold_sources: list[str]) -> bool:
    """判断检索结果是否命中任一金标来源（文件 + 章节/内容）。"""
    source_file = str(doc.metadata.get("source_file", ""))
    section = str(doc.metadata.get("section", "") or "")
    content = doc.page_content
    for gold in gold_sources:
        file, want_section = _parse_source(gold)
        if source_file != file:
            continue
        if not want_section:
            return True
        if want_section in section or want_section in content:
            return True
    return False


def _gold_files(gold_sources: list[str]) -> set[str]:
    return {_parse_source(g)[0] for g in gold_sources}


def evaluate_retrieval(
    retrieve_fn: Callable[[str], list[Document]],
    records: list[dict[str, Any]],
    *,
    k: int = 5,
    top_n: int = 20,
    available_files: set[str] | None = None,
) -> dict[str, Any]:
    """执行检索评测。

    Args:
        retrieve_fn: 问题 → 排序后的 Document 列表。
        records: 金标记录。
        k: 主指标截断位次。
        top_n: 每次检索取回上限。
        available_files: 向量库可检索到的来源文件集合；用于跳过非检索题。
    """
    per_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    usable = 0
    skipped = 0

    for record in records:
        gold = record.get("gold_source") or []
        if record.get("type") in NON_RETRIEVAL_TYPES or not gold:
            skipped += 1
            continue
        if available_files is not None and not (_gold_files(gold) & available_files):
            skipped += 1
            continue

        docs = retrieve_fn(record["question"])[:top_n]
        flags = [is_relevant(doc, gold) for doc in docs]
        # 覆盖度：每条金标来源在 Top-k 内是否被任一检索结果命中
        covered = [any(is_relevant(doc, [source]) for doc in docs[:k]) for source in gold]
        per_type[record["type"]].append(
            {
                "recall": sum(covered) / len(gold),
                "precision": precision_at_k(flags, k),
                "mrr": reciprocal_rank(flags),
                "ndcg": ndcg_at_k(flags, len(gold), k),
                "hit": hit(flags, k),
            }
        )
        usable += 1

    def _aggregate(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            "recall@k": mean([i["recall"] for i in items]),
            "precision@k": mean([i["precision"] for i in items]),
            "mrr": mean([i["mrr"] for i in items]),
            "ndcg@k": mean([i["ndcg"] for i in items]),
            "hit_rate": mean([i["hit"] for i in items]),
        }

    all_items = [item for items in per_type.values() for item in items]
    overall = _aggregate(all_items)
    by_type = {t: _aggregate(items) for t, items in sorted(per_type.items())}

    return {
        "k": k,
        "top_n": top_n,
        "usable_questions": usable,
        "skipped_questions": skipped,
        "metrics": overall,
        "by_type": by_type,
    }
