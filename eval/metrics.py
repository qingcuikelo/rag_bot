"""检索层指标（文档 §9.2）。

纯函数，输入为按排名排序的「是否相关」布尔列表。
"""

from __future__ import annotations

import math


def recall_at_k(flags: list[bool], num_relevant: int, k: int) -> float:
    """Recall@k = Top-k 命中数 / 金标相关总数。"""
    if num_relevant <= 0:
        return 0.0
    return min(sum(flags[:k]), num_relevant) / num_relevant


def precision_at_k(flags: list[bool], k: int) -> float:
    """Precision@k = Top-k 命中数 / k。"""
    if k <= 0:
        return 0.0
    return sum(flags[:k]) / k


def reciprocal_rank(flags: list[bool]) -> float:
    """首个相关结果的倒数排名；无命中为 0。"""
    for idx, hit in enumerate(flags, start=1):
        if hit:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(flags: list[bool], num_relevant: int, k: int) -> float:
    """nDCG@k（二元相关性），结果截断到 [0, 1]。"""
    if num_relevant <= 0 or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(flags[:k]) if hit)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))
    if idcg <= 0:
        return 0.0
    return min(1.0, dcg / idcg)


def hit(flags: list[bool], k: int | None = None) -> float:
    """Hit Rate：Top-k 至少命中一个相关结果。"""
    window = flags if k is None else flags[:k]
    return 1.0 if any(window) else 0.0


def mean(values: list[float]) -> float:
    """算术平均；空列表返回 0。"""
    return sum(values) / len(values) if values else 0.0
