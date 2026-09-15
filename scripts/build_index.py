"""构建/重建索引（文档 §12.11）。

用法::

    python scripts/build_index.py [--data-dir data] [--db-path storage/xingchi.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xingchi_rag.config import get_settings
from xingchi_rag.ingestion.pipeline import build
from xingchi_rag.logging import setup_logging


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="构建星驰 RAG 索引")
    parser.add_argument("--data-dir", default=None, help="原始数据目录（默认 data/）")
    parser.add_argument("--db-path", default=None, help="SQLite 路径（默认 storage/xingchi.db）")
    args = parser.parse_args()

    summary = build(data_dir=args.data_dir, db_path=args.db_path)
    manifest = summary["manifest"]

    print("\n===== 构建完成 =====")
    print(f"索引版本  : {manifest['index_version']}")
    print(f"语料哈希  : {manifest['corpus_hash'][:16]}")
    print(f"chunks    : {manifest['counts']['chunks']}")
    print(f"SQL 行数  : {manifest['counts']}")
    print(f"质量报告  : {summary['report_path']}")
    if summary["warnings"]:
        print("告警:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
