"""SQLite 结构化库入库（文档 §12.3）。

严格遵循入库规则：库存列降级、来源限定、PII 脱敏、可疑来源不落库。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger

from xingchi_rag.config import get_settings
from xingchi_rag.utils.tables import read_csv_rows, read_sheet, read_tsv_rows

_INT_RE = re.compile(r"\d+")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开 SQLite 连接（自动建库目录）。"""
    settings = get_settings()
    target = settings.resolve(Path(db_path) if db_path else settings.sqlite_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """执行 DDL（幂等）。"""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    match = _INT_RE.search(str(value))
    return int(match.group()) if match else None


def mask_name(name: str) -> str:
    """姓名脱敏：保留姓氏，其余以 * 代替（如 张伟 → 张*）。"""
    name = (name or "").strip()
    if len(name) <= 1:
        return name
    return name[0] + "*" * (len(name) - 1)


def hash_phone(phone: str, salt: str) -> str:
    """手机号加盐 SHA-256。"""
    return hashlib.sha256(f"{salt}{phone}".encode()).hexdigest()


def phone_tail(phone: str) -> str:
    """取手机号尾 4 位数字。"""
    digits = re.sub(r"\D", "", phone or "")
    return digits[-4:] if len(digits) >= 4 else digits


def load_all(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """清空并重建结构化库，返回各表行数。"""
    settings = get_settings()
    base = settings.resolve(Path(data_dir) if data_dir else settings.data_dir)
    conn = get_connection(db_path)
    try:
        init_schema(conn)
        counts: dict[str, int] = {}
        for table in (
            "product",
            "price",
            "inventory",
            "ticket",
            "customer",
            "faq_category",
            "sales_monthly",
            "sales_region",
            "sales_summary",
        ):
            conn.execute(f"DELETE FROM {table}")
        counts["product"] = _load_product(conn, base)
        counts["price"] = _load_price(conn, base)
        counts["inventory"] = _load_inventory(conn, base)
        counts["ticket"] = _load_ticket(conn, base)
        counts["customer"] = _load_customer(conn, base, settings.pii_salt)
        counts["faq_category"] = _load_faq_category(conn, base)
        counts.update(_load_sales(conn, base))
        conn.commit()
        logger.info(f"SQLite 入库完成: {counts}")
        return counts
    finally:
        conn.close()


def _load_product(conn: sqlite3.Connection, base: Path) -> int:
    path = base / "csv" / "产品参数表.csv"
    if not path.is_file():
        return 0
    rows = [
        (
            r["型号"].strip(),
            r["名称"].strip(),
            r["类别"].strip(),
            (r.get("解锁方式") or "").strip(),
            (r.get("材质") or "").strip(),
            r["上市状态"].strip(),
            _to_int(r.get("保修期")),
        )
        for r in read_csv_rows(path)
        if (r.get("型号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO product"
        "(model,name,category,unlock_methods,material,sale_status,warranty_years)"
        " VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    # 注意：产品参数表库存列按 §2.3.1 降级，不落入 product 表
    return len(rows)


def _load_price(conn: sqlite3.Connection, base: Path) -> int:
    path = base / "xlsx" / "产品价格与库存.xlsx"
    if not path.is_file():
        return 0
    rows = [
        (
            str(r["型号"]).strip(),
            _to_int(r.get("出厂价(元)")),
            _to_int(r.get("建议零售价(元)")),
            str(r.get("毛利率") or "").strip(),
            "2026-03-31",
        )
        for r in read_sheet(path, "产品价格")
        if str(r.get("型号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO price"
        "(model,factory_price,retail_price,gross_margin,effective_date) VALUES(?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _load_inventory(conn: sqlite3.Connection, base: Path) -> int:
    inserted = 0
    tsv = base / "tsv" / "库存清单.tsv"
    if tsv.is_file():
        rows = [
            (
                r["SKU"].strip(),
                r["仓库"].strip(),
                _to_int(r.get("在库数量")) or 0,
                _to_int(r.get("安全库存")) or 0,
                (r.get("补货状态") or "").strip(),
                "2026-03-31",
            )
            for r in read_tsv_rows(tsv)
            if (r.get("SKU") or "").strip()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO inventory"
            "(sku,warehouse,qty,safety_qty,restock_status,updated_at) VALUES(?,?,?,?,?,?)",
            rows,
        )
        inserted += len(rows)

    xlsx = base / "xlsx" / "产品价格与库存.xlsx"
    if xlsx.is_file():
        extra = [
            (
                str(r["型号"]).strip(),
                str(r.get("仓库") or "").strip(),
                _to_int(r.get("数量")) or 0,
                0,
                None,
                str(r.get("更新时间") or "2026-03-31").strip(),
            )
            for r in read_sheet(xlsx, "库存分布")
            if str(r.get("型号") or "").strip()
        ]
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO inventory"
            "(sku,warehouse,qty,safety_qty,restock_status,updated_at) VALUES(?,?,?,?,?,?)",
            extra,
        )
        inserted += conn.total_changes - before
    return inserted


def _load_ticket(conn: sqlite3.Connection, base: Path) -> int:
    path = base / "csv" / "售后工单记录.csv"
    if not path.is_file():
        return 0
    rows = [
        (
            r["工单号"].strip(),
            r["客户ID"].strip(),
            (r.get("产品型号") or "").strip(),
            (r.get("问题类型") or "").strip(),
            r["状态"].strip(),
            (r.get("创建日期") or "").strip(),
            float(r["处理时长(小时)"]) if (r.get("处理时长(小时)") or "").strip() else None,
        )
        for r in read_csv_rows(path)
        if (r.get("工单号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO ticket"
        "(ticket_no,customer_id,product_model,issue_type,status,created_date,handle_hours)"
        " VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _load_customer(conn: sqlite3.Connection, base: Path, salt: str) -> int:
    path = base / "csv" / "客户信息表.csv"
    if not path.is_file():
        logger.warning("客户信息表缺失（可能未纳入版本库），跳过 customer 入库")
        return 0
    rows = []
    for r in read_csv_rows(path):
        cid = (r.get("客户ID") or "").strip()
        if not cid:
            continue
        phone = (r.get("联系电话") or "").strip()
        rows.append(
            (
                cid,
                mask_name(r.get("姓名") or ""),
                (r.get("城市") or "").strip(),
                hash_phone(phone, salt),
                phone_tail(phone),
                (r.get("注册日期") or "").strip(),
                (r.get("会员等级") or "").strip(),
            )
        )
    conn.executemany(
        "INSERT OR REPLACE INTO customer"
        "(customer_id,name_masked,city,phone_hash,phone_tail,register_date,tier)"
        " VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _load_faq_category(conn: sqlite3.Connection, base: Path) -> int:
    path = base / "tsv" / "常见问题分类.tsv"
    if not path.is_file():
        return 0
    rows = [
        (
            r["编号"].strip(),
            (r.get("问题类别") or "").strip(),
            (r.get("关键词") or "").strip(),
            (r.get("责任部门") or "").strip(),
            _to_int(r.get("首响时限(小时)")),
        )
        for r in read_tsv_rows(path)
        if (r.get("编号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO faq_category"
        "(code,category,keywords,dept,first_response_hours) VALUES(?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _load_sales(conn: sqlite3.Connection, base: Path) -> dict[str, int]:
    path = base / "xlsx" / "产品销售数据.xlsx"
    if not path.is_file():
        return {"sales_monthly": 0, "sales_region": 0, "sales_summary": 0}

    monthly = [
        (
            str(r.get("月份") or "").strip(),
            str(r.get("产品型号") or "").strip(),
            _to_int(r.get("销量")) or 0,
            _to_int(r.get("销售额(元)")) or 0,
            str(r.get("环比") or "").strip(),
        )
        for r in read_sheet(path, "月度销售")
        if str(r.get("产品型号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO sales_monthly(month,model,qty,amount,mom) VALUES(?,?,?,?,?)",
        monthly,
    )

    region = [
        (
            str(r.get("区域") or "").strip(),
            str(r.get("销售占比") or "").strip(),
            str(r.get("主要产品") or "").strip(),
            str(r.get("同比增长") or "").strip(),
        )
        for r in read_sheet(path, "区域分布")
        if str(r.get("区域") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO sales_region(region,share,top_model,yoy) VALUES(?,?,?,?)",
        region,
    )

    summary = [
        (
            str(r.get("型号") or "").strip(),
            _to_int(r.get("年度销量")) or 0,
            _to_int(r.get("年度销售额(元)")) or 0,
            str(r.get("退货率") or "").strip(),
        )
        for r in read_sheet(path, "产品汇总")
        if str(r.get("型号") or "").strip()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO sales_summary"
        "(model,year_qty,year_amount,return_rate) VALUES(?,?,?,?)",
        summary,
    )
    return {
        "sales_monthly": len(monthly),
        "sales_region": len(region),
        "sales_summary": len(summary),
    }
