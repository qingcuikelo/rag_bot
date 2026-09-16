"""评测门禁（文档 §9.4、§12.11）。

用法::

    # 检索评测
    python scripts/run_eval.py --mode hybrid --rerank --split dev --k 5 --gate 0.80
    # 生成评测（LLM-as-Judge）
    python scripts/run_eval.py --generation --split dev --limit 30
    # 生成评测（RAGAS，需独立环境）
    python scripts/run_eval.py --generation --use-ragas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval.generation import evaluate_generation  # noqa: E402
from eval.run import evaluate_retrieval, load_golden  # noqa: E402
from xingchi_rag.config import get_settings  # noqa: E402
from xingchi_rag.logging import setup_logging  # noqa: E402
from xingchi_rag.observability import setup_tracing  # noqa: E402
from xingchi_rag.retrieval.factory import get_retriever  # noqa: E402
from xingchi_rag.utils.docstore import load_documents  # noqa: E402

# 生成层门禁（§9.2）
GENERATION_GATES = {
    "refusal_accuracy": 0.95,
    "citation_accuracy": 0.90,
    "faithfulness": 0.85,
    "answer_correctness": 0.85,
}


def _available_files() -> set[str]:
    settings = get_settings()
    chunks = load_documents(settings.resolve(settings.bm25_path) / "chunks.jsonl")
    return {str(c.metadata.get("source_file", "")) for c in chunks}


def _run_retrieval_eval(args: argparse.Namespace) -> int:
    records = load_golden(split=None if args.split == "all" else args.split)
    available = _available_files()
    retriever = get_retriever(args.mode, k=args.candidates, rerank=args.rerank, top_n=args.top_n)
    report = evaluate_retrieval(
        retriever.invoke, records, k=args.k, top_n=args.candidates, available_files=available
    )

    mode_label = f"{args.mode}{'+rerank' if args.rerank else ''}"
    print(f"\n===== 检索评测（mode={mode_label}, split={args.split}, k={args.k}）=====")
    print(f"可用问题: {report['usable_questions']}  跳过: {report['skipped_questions']}")
    for name, value in report["metrics"].items():
        print(f"  {name:14s}: {value:.4f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写出: {args.out}")

    recall = report["metrics"]["recall@k"]
    passed = recall >= args.gate
    print(f"\n门禁 Recall@{args.k} >= {args.gate}: {recall:.4f} -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _run_generation_eval(args: argparse.Namespace) -> int:
    from xingchi_rag.graph.build import build_graph
    from xingchi_rag.providers.llm import get_llm

    records = load_golden(split=None if args.split == "all" else args.split)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        records = [r for r in records if r.get("type") in wanted]
    graph = build_graph(with_checkpointer=False)

    def run_fn(question: str) -> dict:
        return graph.invoke({"question": question})

    judge = None if args.no_judge else get_llm()
    report = evaluate_generation(run_fn, records, judge_llm=judge, limit=args.limit)

    print(f"\n===== 生成评测（split={args.split}, judge={'off' if judge is None else 'on'}）=====")
    print(f"题量: {report['records']}  作答: {report['answered']}")
    for name, value in report["metrics"].items():
        print(f"  {name:20s}: {value:.4f}")
    if report["failures"]:
        print(f"失败样例: {len(report['failures'])}")
        for item in report["failures"][:5]:
            print(f"  - {item}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写出: {args.out}")

    failed = [
        f"{name}={report['metrics'][name]:.3f}<{gate}"
        for name, gate in GENERATION_GATES.items()
        if name in report["metrics"] and report["metrics"][name] < gate
    ]
    if failed:
        print(f"\n生成门禁 FAIL: {', '.join(failed)}")
        return 1
    print("\n生成门禁 PASS")
    return 0


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    setup_tracing()

    parser = argparse.ArgumentParser(description="星驰 RAG 评测")
    parser.add_argument("--generation", action="store_true", help="运行生成评测（默认检索）")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--k", type=int, default=5, help="指标截断位次（Recall@k）")
    parser.add_argument("--candidates", type=int, default=20, help="基础检索召回数量")
    parser.add_argument("--top-n", type=int, default=5, help="重排后保留数量")
    parser.add_argument("--rerank", action="store_true", help="启用 API 重排")
    parser.add_argument("--gate", type=float, default=0.80, help="Recall@k 门禁阈值")
    parser.add_argument("--limit", type=int, default=None, help="生成评测仅取前 N 条")
    parser.add_argument("--types", default=None, help="生成评测按题型过滤，如 unanswerable,pii")
    parser.add_argument("--no-judge", action="store_true", help="生成评测关闭 LLM-as-Judge")
    parser.add_argument("--use-ragas", action="store_true", help="生成评测使用 RAGAS（需独立环境）")
    parser.add_argument("--out", default=None, help="指标报告输出路径(JSON)")
    args = parser.parse_args()

    if args.generation:
        if args.use_ragas:
            print("RAGAS 需在独立环境运行（见 requirements-eval.txt），当前使用内置 Judge。")
        return _run_generation_eval(args)
    return _run_retrieval_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
