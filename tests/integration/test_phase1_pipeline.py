"""集成测试：Phase 1 全链路（加载 → 治理 → 事实卡 → SQLite → 报告）。"""

from __future__ import annotations

import json
import sqlite3

import pytest

from xingchi_rag.config import get_settings
from xingchi_rag.ingestion.pipeline import build

pytestmark = pytest.mark.integration


def test_build_phase1(isolated_storage) -> None:
    summary = build()

    settings = get_settings()
    report_path = settings.resolve(settings.quality_report_path)
    manifest_path = settings.resolve(settings.manifest_path)
    db_path = settings.resolve(settings.sqlite_path)

    assert report_path.is_file()
    assert manifest_path.is_file()
    assert db_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["chunks"] > 0
    # 客户信息表含 PII，应被排除出向量
    assert "客户信息表.csv" in report["pii_excluded"]
    # 退化 sheet 应被检测并告警
    assert any("售后工单统计.xlsx" in w for w in report["warnings"])
    # 库存冲突应被记录
    assert any(c["model"] == "XC-L100" for c in report["inventory_conflicts"])
    # 枚举校验无违规
    assert report["enum_violations"] == []

    conn = sqlite3.connect(str(db_path))
    try:
        product_count = conn.execute("SELECT COUNT(*) FROM product").fetchone()[0]
        inventory_count = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
        ticket_count = conn.execute("SELECT COUNT(*) FROM ticket").fetchone()[0]
        # 产品参数表库存列不得落库
        columns = {row[1] for row in conn.execute("PRAGMA table_info(product)")}
    finally:
        conn.close()

    assert product_count == 8
    assert inventory_count >= 7
    assert ticket_count == 6
    assert "库存" not in columns

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["chunks"] == summary["chunks"]
    assert manifest["corpus_hash"]
