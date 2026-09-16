"""并发压测（文档 §12.8 单机 ≥20 QPS 目标）。

用法（需先启动服务）::

    python scripts/serve.py
    python scripts/load_test.py --url http://127.0.0.1:8000 --requests 100 --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from xingchi_rag.config import get_settings  # noqa: E402

QUESTIONS = [
    "智能门锁整机保修几年？",
    "XC-L100 支持哪些解锁方式？",
    "XC-L100 现在库存多少？",
    "XC-L100 和 XC-L200 有什么区别？",
    "门锁连不上 WiFi 怎么办？",
    "支持以旧换新吗？",
]


async def _one(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], question: str
) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/v1/chat", json={"message": question}, headers=headers, timeout=180
        )
        status = response.status_code
    except Exception:
        status = -1
    return status, (time.perf_counter() - started) * 1000


async def _run(url: str, requests: int, concurrency: int, service_key: str) -> int:
    headers = {"Authorization": f"Bearer {service_key}"} if service_key else {}
    semaphore = asyncio.Semaphore(concurrency)
    results: list[tuple[int, float]] = []

    async with httpx.AsyncClient(trust_env=False) as client:

        async def worker(index: int) -> None:
            async with semaphore:
                results.append(await _one(client, url, headers, QUESTIONS[index % len(QUESTIONS)]))

        tasks = [asyncio.create_task(worker(i)) for i in range(requests)]
        started = time.perf_counter()
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started

    latencies = sorted(r[1] for r in results)
    errors = sum(1 for r in results if r[0] < 200 or r[0] >= 300)

    def percentile(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(p / 100 * len(latencies)))
        return latencies[idx]

    print("\n===== 压测结果 =====")
    print(f"请求数      : {requests}  并发: {concurrency}")
    print(f"总耗时      : {elapsed:.2f}s")
    print(f"QPS         : {requests / elapsed:.2f}")
    print(f"平均延迟    : {statistics.mean(latencies):.0f} ms")
    print(f"P50/P95/P99 : {percentile(50):.0f} / {percentile(95):.0f} / {percentile(99):.0f} ms")
    print(f"错误率      : {errors}/{len(results)}")
    return 1 if errors else 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="星驰 RAG 并发压测")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--service-key", default=settings.service_api_key or "")
    args = parser.parse_args()
    return asyncio.run(_run(args.url, args.requests, args.concurrency, args.service_key))


if __name__ == "__main__":
    raise SystemExit(main())
