"""评测门禁（文档 §9.4、§12.11）。

用法::

    python scripts/run_eval.py --mode hybrid --split dev --k 5 --gate 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval.run import evaluate_retrieval, load_golden  # noqa: E402
from xingchi_rag.config import get_settings  # noqa: E402
from xingchi_rag.logging import setup_logging  # noqa: E402
from xingchi_rag.retrieval.bm25 import get_bm25_retriever  # noqa: E402
from xingchi_rag.retrieval.factory import get_retriever  # noqa: E402
from xingchi_rag.utils.docstore import load_documents  # noqa: E402


def _available_files() -> set[str]:
    settings = get_settings()
    chunks = load_documents(settings.resolve(settings.bm25_path) / "chunks.jsonl")
    return {str(c.metadata.get("source_file", "")) for c in chunks}


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="星驰 RAG 检索评测")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--gate", type=float, default=0.80, help="Recall@k 门禁阈值")
    parser.add_argument("--out", default=None, help="指标报告输出路径(JSON)")
    args = parser.parse_args()

    records = load_golden(split=None if args.split == "all" else args.split)
    available = _available_files()

    if args.mode == "bm25":
        retriever = get_bm25_retriever(k=args.top_n)
    else:
        retriever = get_retriever(args.mode, k=args.top_n)
    retrieve_fn = retriever.invoke

    report = evaluate_retrieval(
        retrieve_fn, records, k=args.k, top_n=args.top_n, available_files=available
    )

    print(f"\n===== 检索评测（mode={args.mode}, split={args.split}, k={args.k}）=====")
    print(f"可用问题: {report['usable_questions']}  跳过: {report['skipped_questions']}")
    for name, value in report["metrics"].items():
        print(f"  {name:14s}: {value:.4f}")
    print("按类型:")
    for qtype, metrics in report["by_type"].items():
        print(
            f"  {qtype:12s} recall@k={metrics['recall@k']:.3f} "
            f"mrr={metrics['mrr']:.3f} hit={metrics['hit_rate']:.3f}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写出: {args.out}")

    recall = report["metrics"]["recall@k"]
    passed = recall >= args.gate
    print(f"\n门禁 Recall@{args.k} >= {args.gate}: {recall:.4f} -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
