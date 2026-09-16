"""单元测试：生成层评测（拒答/引用，无网络）。"""

from __future__ import annotations

from typing import Any

from eval.generation import evaluate_generation


def _records() -> list[dict[str, Any]]:
    return [
        {
            "id": "G1",
            "question": "q1",
            "type": "policy",
            "expected_answer": "3 年",
            "gold_source": [],
        },
        {
            "id": "G2",
            "question": "q2",
            "type": "unanswerable",
            "expected_answer": "",
            "gold_source": [],
        },
        {"id": "G3", "question": "q3", "type": "pii", "expected_answer": "", "gold_source": []},
    ]


def _run_fn(question: str) -> dict[str, Any]:
    mapping = {
        "q1": {
            "answer": "整机保修 3 年（[来源: 保修政策与退换货说明.txt · 第一条]）",
            "refused": False,
            "citations": [
                {"source_file": "保修政策与退换货说明.txt", "section": "第一条", "chunk_id": "x"}
            ],
            "evidence": [
                {
                    "page_content": "智能门锁整机保修 3 年。",
                    "metadata": {
                        "source_file": "保修政策与退换货说明.txt",
                        "section": "第一条",
                    },
                }
            ],
        },
        "q2": {"answer": "无依据，转人工。", "refused": True, "citations": [], "evidence": []},
        "q3": {"answer": "需身份校验。", "refused": True, "citations": [], "evidence": []},
    }
    return mapping[question]


def test_evaluate_generation_all_pass() -> None:
    report = evaluate_generation(_run_fn, _records())
    assert report["records"] == 3
    assert report["metrics"]["refusal_accuracy"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 1.0
    assert report["failures"] == []


def test_evaluate_generation_marks_failures() -> None:
    def bad_run(question: str) -> dict[str, Any]:
        if question == "q2":
            return {"answer": "编造答案", "refused": False, "citations": [], "evidence": []}
        return _run_fn(question)

    report = evaluate_generation(bad_run, _records())
    assert report["metrics"]["refusal_accuracy"] < 1.0
    assert any(item["reason"] == "拒答判定不符" for item in report["failures"])


def test_evaluate_generation_detects_bad_citation() -> None:
    def bad_citation(question: str) -> dict[str, Any]:
        if question == "q1":
            return {
                "answer": "保修 3 年（[来源: 不存在.txt]）",
                "refused": False,
                "citations": [],
                "evidence": _run_fn("q1")["evidence"],
            }
        return _run_fn(question)

    report = evaluate_generation(bad_citation, _records())
    assert report["metrics"]["citation_accuracy"] < 1.0
