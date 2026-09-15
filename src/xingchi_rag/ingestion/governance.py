"""数据治理：权威级、去重、PII、时效、枚举与交叉一致性校验。

对应文档 §2.3、§5.4、§12.3；产出 ``storage/data_quality_report.json``。
所有规则来自 ``configs/sources.yaml``。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from xingchi_rag.config import get_settings
from xingchi_rag.utils.io import load_yaml

# 后缀 → 是否为结构化表格（结构化表格走事实卡 + SQL，不进原始行向量）
STRUCTURED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}

# 文件名 → doc_type（§5.4 枚举）
DOC_TYPE_BY_FILE: dict[str, str] = {
    "2026产品目录.pdf": "catalog",
    "产品使用手册_星驰智能门锁.pdf": "manual",
    "售后服务指南.pdf": "policy",
    "客服FAQ_售前售后.txt": "faq",
    "保修政策与退换货说明.txt": "policy",
    "产品介绍_星驰智能门锁.txt": "catalog",
    "产品参数表.csv": "spec",
    "产品参数表.tsv": "spec",
    "售后工单记录.csv": "ticket",
    "客户信息表.csv": "customer",
    "库存清单.tsv": "inventory",
    "常见问题分类.tsv": "faq",
    "产品价格与库存.xlsx": "inventory",
    "产品销售数据.xlsx": "sales",
    "售后工单统计.xlsx": "ticket",
}


@dataclass
class GovernanceResult:
    """治理结果。"""

    documents: list[Document] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def load_sources_config(path: str | Path = "configs/sources.yaml") -> dict[str, Any]:
    """加载治理规则。"""
    return load_yaml(path)


def authority_map(config: dict[str, Any]) -> dict[str, int]:
    """构建 文件名 → 权威级别 映射。"""
    result: dict[str, int] = {}
    for rank, files in (config.get("authority_rank") or {}).items():
        for name in files or []:
            result[str(name)] = int(rank)
    return result


def infer_doc_type(source_file: str) -> str:
    """推断 doc_type（枚举：policy/manual/faq/catalog/inventory/sales/ticket/customer）。"""
    if source_file in DOC_TYPE_BY_FILE:
        return DOC_TYPE_BY_FILE[source_file]
    lower = source_file.lower()
    if "faq" in lower or "问题" in source_file:
        return "faq"
    if "库存" in source_file or "价格" in source_file:
        return "inventory"
    if "销售" in source_file:
        return "sales"
    if "工单" in source_file or "客户" in source_file:
        return "ticket"
    if "手册" in source_file:
        return "manual"
    if "政策" in source_file or "保修" in source_file or "退换" in source_file:
        return "policy"
    return "catalog"


def _file_content_hash(documents: list[Document]) -> str:
    """按原始顺序拼接文本并计算 SHA-256（用于跨文件去重）。"""
    joined = "\n".join(doc.page_content for doc in documents)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _logical_doc_id(source_file: str) -> str:
    """逻辑文档 ID（去重后），同文件所有行共享。"""
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]


def govern(
    documents: list[Document],
    *,
    config: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> GovernanceResult:
    """执行治理：权威级、时效、PII、去重、结构化标记。

    Returns:
        ``GovernanceResult``：保留的 ``Document``（已注入治理元数据）与质量报告。
    """
    config = config or load_sources_config()
    warnings = list(warnings or [])

    authority = authority_map(config)
    pii_cfg = config.get("pii") or {}
    pii_excluded = set(pii_cfg.get("excluded_from_vector") or [])
    explicit_dupes = set((config.get("dedup") or {}).get("ignore_duplicates") or [])
    effective_dates = (config.get("timeliness") or {}).get("known_effective_dates") or {}
    today = date.today().isoformat()

    # 按文件分组
    by_file: dict[str, list[Document]] = {}
    for doc in documents:
        by_file.setdefault(str(doc.metadata.get("source_file", "unknown")), []).append(doc)

    # 内容哈希去重（同哈希保留权威级别最高者）
    hash_to_files: dict[str, list[str]] = {}
    for source_file, docs in by_file.items():
        hash_to_files.setdefault(_file_content_hash(docs), []).append(source_file)

    duplicates: list[dict[str, str]] = []
    dropped_files: set[str] = set()

    for source_file in sorted(explicit_dupes):
        if source_file in by_file:
            dropped_files.add(source_file)
            duplicates.append(
                {"dropped": source_file, "kept": "", "reason": "显式重复源（sources.yaml）"}
            )

    for digest, files in hash_to_files.items():
        if len(files) < 2:
            continue
        ordered = sorted(files, key=lambda f: (authority.get(f, 50), f))
        winner = ordered[0]
        for loser in ordered[1:]:
            if loser in dropped_files:
                continue
            dropped_files.add(loser)
            duplicates.append(
                {
                    "dropped": loser,
                    "kept": winner,
                    "reason": f"内容 SHA-256 相同（{digest[:12]}）",
                }
            )

    if dropped_files:
        warnings.append(f"去重：忽略 {len(dropped_files)} 个重复源 → {sorted(dropped_files)}")

    kept: list[Document] = []
    excluded_pii: list[str] = []

    for source_file, docs in by_file.items():
        suffix = Path(source_file).suffix.lower()
        is_structured = suffix in STRUCTURED_SUFFIXES
        is_pii = source_file in pii_excluded

        if is_pii:
            excluded_pii.append(source_file)

        for doc in docs:
            metadata = dict(doc.metadata)
            metadata["doc_id"] = _logical_doc_id(source_file)
            metadata["doc_type"] = infer_doc_type(source_file)
            metadata["authority_rank"] = authority.get(source_file, 50)
            metadata["pii_flag"] = is_pii
            metadata["structured"] = is_structured
            sheet = str(metadata.get("sheet", ""))
            metadata["effective_date"] = effective_dates.get(sheet, today)
            if is_pii or source_file in dropped_files:
                metadata["vectorize"] = False
            else:
                # 结构化表格的行不进原始向量，改由事实卡承载（§5.2）
                metadata["vectorize"] = not is_structured
            doc.metadata = metadata
            kept.append(doc)

    report = _build_report(
        kept,
        warnings=warnings,
        duplicates=duplicates,
        excluded_pii=excluded_pii,
        config=config,
    )
    return GovernanceResult(documents=kept, report=report)


# ----------------------------------------------------------------------
# 质量校验
# ----------------------------------------------------------------------
def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_tsv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check_enums(data_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """枚举校验（§2.3.4）。"""
    checks: dict[str, list[str]] = (config.get("quality_gates") or {}).get("enum_checks") or {}
    violations: list[dict[str, Any]] = []

    spec = data_dir / "csv" / "产品参数表.csv"
    if spec.is_file() and "上市状态" in checks:
        allowed = set(checks["上市状态"])
        for row in _read_csv_records(spec):
            value = (row.get("上市状态") or "").strip()
            if value and value not in allowed:
                violations.append(
                    {
                        "file": spec.name,
                        "field": "上市状态",
                        "value": value,
                        "allowed": sorted(allowed),
                    }
                )

    ticket = data_dir / "csv" / "售后工单记录.csv"
    if ticket.is_file() and "状态" in checks:
        allowed = set(checks["状态"])
        for row in _read_csv_records(ticket):
            value = (row.get("状态") or "").strip()
            if value and value not in allowed:
                violations.append(
                    {
                        "file": ticket.name,
                        "field": "状态",
                        "value": value,
                        "allowed": sorted(allowed),
                    }
                )
    return violations


def check_inventory_conflicts(data_dir: Path) -> list[dict[str, Any]]:
    """交叉一致性：产品参数表库存列 vs 库存清单（§2.2 缺陷 1）。"""
    conflicts: list[dict[str, Any]] = []

    spec = data_dir / "csv" / "产品参数表.csv"
    inv = data_dir / "tsv" / "库存清单.tsv"
    if not (spec.is_file() and inv.is_file()):
        return conflicts

    spec_stock = {r["型号"].strip(): r["库存"].strip() for r in _read_csv_records(spec)}
    inv_stock = {r["SKU"].strip(): r["在库数量"].strip() for r in _read_tsv_records(inv)}

    for model, spec_value in spec_stock.items():
        inv_value = inv_stock.get(model)
        if inv_value is not None and spec_value != inv_value:
            conflicts.append(
                {
                    "model": model,
                    "产品参数表.csv": spec_value,
                    "库存清单.tsv": inv_value,
                    "authoritative": "库存清单.tsv",
                }
            )
    return conflicts


def _build_report(
    documents: list[Document],
    *,
    warnings: list[str],
    duplicates: list[dict[str, str]],
    excluded_pii: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = settings.resolve(settings.data_dir)

    vector_docs = sum(1 for d in documents if d.metadata.get("vectorize"))
    structured_docs = sum(1 for d in documents if d.metadata.get("structured"))

    return {
        "generated_at": date.today().isoformat(),
        "counts": {
            "documents": len(documents),
            "vector_documents": vector_docs,
            "structured_documents": structured_docs,
            "excluded_pii_files": len(excluded_pii),
            "duplicate_files": len(duplicates),
        },
        "warnings": warnings,
        "duplicates": duplicates,
        "pii_excluded": sorted(excluded_pii),
        "enum_violations": check_enums(data_dir, config),
        "inventory_conflicts": check_inventory_conflicts(data_dir),
        "missing_optional_sources": [
            name
            for name in (config.get("pii") or {}).get("excluded_from_vector", [])
            if not (data_dir / "csv" / name).is_file()
        ],
    }


def write_quality_report(report: dict[str, Any], path: str | Path | None = None) -> Path:
    """写出质量报告 JSON。"""
    import json

    settings = get_settings()
    target = Path(path) if path else settings.quality_report_path
    target = settings.resolve(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"质量报告已写出: {target}")
    return target
