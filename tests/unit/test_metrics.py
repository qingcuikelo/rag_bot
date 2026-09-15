"""单元测试：检索指标。"""

from __future__ import annotations

from eval.metrics import (
    hit,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    assert recall_at_k([True, False, False], num_relevant=1, k=3) == 1.0
    assert recall_at_k([False, False, False], num_relevant=1, k=3) == 0.0
    assert recall_at_k([True, True], num_relevant=4, k=2) == 0.5


def test_precision_at_k() -> None:
    assert precision_at_k([True, False, True], k=3) == 2 / 3
    assert precision_at_k([True], k=2) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([True]) == 1.0


def test_ndcg_at_k() -> None:
    assert ndcg_at_k([True, False], num_relevant=1, k=2) == 1.0
    assert ndcg_at_k([False, False], num_relevant=1, k=2) == 0.0
    assert 0 < ndcg_at_k([False, True], num_relevant=1, k=2) < 1


def test_hit_rate() -> None:
    assert hit([False, True], k=1) == 0.0
    assert hit([False, True], k=2) == 1.0


def test_mean() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5
