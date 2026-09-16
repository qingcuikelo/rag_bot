"""启动 API 服务（文档 §12.11）。

用法::

    python scripts/serve.py [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402

from xingchi_rag.config import get_settings  # noqa: E402
from xingchi_rag.logging import setup_logging  # noqa: E402
from xingchi_rag.observability import setup_tracing  # noqa: E402


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    setup_tracing()

    parser = argparse.ArgumentParser(description="启动星驰 RAG API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.api_workers,
        help="uvicorn 进程数（>1 时禁用 reload）",
    )
    args = parser.parse_args()

    reload = args.reload and args.workers <= 1
    if args.reload and args.workers > 1:
        print("workers>1 时不支持 --reload，已禁用热重载")

    uvicorn.run(
        "xingchi_rag.api.routes:app",
        host=args.host,
        port=args.port,
        reload=reload,
        workers=args.workers,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
