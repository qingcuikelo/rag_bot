"""表格 → 事实卡（结构化语义化）。

对应文档 §5.3：把结构化表格行转成自然语言事实句，供向量检索；
并按 ``(product_model, field)`` 去重，保留权威来源值（§2.3.2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from xingchi_rag.config import get_settings
from xingchi_rag.utils.tables import read_csv_rows, read_sheet, read_tsv_rows

# 数据域 → 首选来源（同级权威时的优先者）
PRIMARY_SOURCE = {
    "库存": "库存清单.tsv",
    "价格": "产品价格与库存.xlsx",
}


@dataclass
class _Fact:
    """单个事实（用于跨源冲突消解）。"""

    model: str
    field_name: str
    value: str
    rank: int
    source: str


@dataclass
class _FactRegistry:
    """按 (model, field) 记录权威值。"""

    facts: dict[tuple[str, str], _Fact] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def consider(self, fact: _Fact, primary: str | None = None) -> None:
        key = (fact.model, fact.field_name)
        current = self.facts.get(key)
        if current is None:
            self.facts[key] = fact
            return
        if fact.rank < current.rank or (
            fact.rank == current.rank and fact.source == primary and current.source != primary
        ):
            self.conflicts.append(
                {
                    "model": fact.model,
                    "field": fact.field_name,
                    "kept": fact.value,
                    "kept_source": fact.source,
                    "dropped": current.value,
                    "dropped_source": current.source,
                }
            )
            self.facts[key] = fact
        elif fact.value != current.value:
            self.conflicts.append(
                {
                    "model": fact.model,
                    "field": fact.field_name,
                    "kept": current.value,
                    "kept_source": current.source,
                    "dropped": fact.value,
                    "dropped_source": fact.source,
                }
            )


def _doc(content: str, source_file: str, **metadata: Any) -> Document:
    base: dict[str, Any] = {
        "source_file": source_file,
        "doc_type": metadata.pop("doc_type", "inventory"),
        "vectorize": True,
        "structured": False,
        "pii_flag": False,
        "authority_rank": metadata.pop("authority_rank", 10),
    }
    base.update(metadata)
    return Document(page_content=content, metadata=base)


def build_factcards(
    data_dir: str | Path | None = None,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """生成事实卡。

    Returns:
        (factcard documents, conflicts)
    """
    settings = get_settings()
    base = Path(data_dir) if data_dir else settings.resolve(settings.data_dir)

    docs: list[Document] = []
    registry = _FactRegistry()

    docs.extend(_spec_cards(base, registry))
    docs.extend(_price_cards(base, registry))
    docs.extend(_inventory_cards(base, registry))
    docs.extend(_faq_category_cards(base))

    # 冲突消解：把被覆盖的库存值以事实卡形式保留为“降级不采用”说明（仅留痕）
    logger.info(f"事实卡生成完成：{len(docs)} 张，冲突 {len(registry.conflicts)} 条")
    return docs, registry.conflicts


# ----------------------------------------------------------------------
# 各来源事实卡
# ----------------------------------------------------------------------
def _spec_cards(base: Path, registry: _FactRegistry) -> list[Document]:
    path = base / "csv" / "产品参数表.csv"
    if not path.is_file():
        return []
    cards: list[Document] = []
    for row in read_csv_rows(path):
        model = (row.get("型号") or "").strip()
        if not model:
            continue
        # 库存列权威性降级，不参与回答（§2.3.1），故排除
        parts = [
            f"产品：{row.get('名称', '').strip()}（型号 {model}）",
            f"类别：{row.get('类别', '').strip()}",
            f"解锁方式：{row.get('解锁方式', '').strip()}",
            f"面板材质：{row.get('材质', '').strip()}",
            f"建议零售价：{row.get('价格(元)', '').strip()} 元",
            f"上市状态：{row.get('上市状态', '').strip()}",
            f"保修期：{row.get('保修期', '').strip()}",
        ]
        content = "；".join(p for p in parts if not p.endswith("：")) + "。"
        cards.append(
            _doc(
                content,
                path.name,
                doc_type="spec",
                authority_rank=20,
                product_model=model,
                product_category=(row.get("类别") or "").strip(),
                section="产品参数",
            )
        )
    return cards


def _price_cards(base: Path, registry: _FactRegistry) -> list[Document]:
    path = base / "xlsx" / "产品价格与库存.xlsx"
    if not path.is_file():
        return []
    cards: list[Document] = []
    for row in read_sheet(path, "产品价格"):
        model = str(row.get("型号") or "").strip()
        if not model:
            continue
        content = (
            f"价格：{row.get('名称', '')}（型号 {model}）"
            f"出厂价 {row.get('出厂价(元)', '')} 元，"
            f"建议零售价 {row.get('建议零售价(元)', '')} 元，"
            f"毛利率 {row.get('毛利率', '')}。"
        )
        registry.consider(
            _Fact(model, "价格", str(row.get("建议零售价(元)", "")), 10, path.name),
            primary=PRIMARY_SOURCE["价格"],
        )
        cards.append(
            _doc(
                content,
                path.name,
                doc_type="inventory",
                authority_rank=10,
                product_model=model,
                section="价格",
            )
        )
    return cards


def _inventory_cards(base: Path, registry: _FactRegistry) -> list[Document]:
    cards: list[Document] = []
    updated_at = "2026-03-31"

    # 库存清单.tsv（权威）
    tsv = base / "tsv" / "库存清单.tsv"
    if tsv.is_file():
        for row in read_tsv_rows(tsv):
            sku = (row.get("SKU") or "").strip()
            if not sku:
                continue
            qty = (row.get("在库数量") or "").strip()
            safety = (row.get("安全库存") or "").strip()
            status = (row.get("补货状态") or "").strip()
            registry.consider(_Fact(sku, "库存", qty, 10, tsv.name), primary=PRIMARY_SOURCE["库存"])
            content = (
                f"库存预警：{sku}（{row.get('产品名称', '')}）"
                f"{row.get('仓库', '')}在库 {qty} 台，"
                f"安全库存 {safety} 台，补货状态为「{status}」，"
                f"数据更新时间 {updated_at}。"
            )
            cards.append(
                _doc(
                    content,
                    tsv.name,
                    doc_type="inventory",
                    authority_rank=10,
                    product_model=sku,
                    section="库存",
                    effective_date=updated_at,
                )
            )

    # xlsx 库存分布（与库存清单同级，冲突时以 库存清单 为准）
    xlsx = base / "xlsx" / "产品价格与库存.xlsx"
    if xlsx.is_file():
        for row in read_sheet(xlsx, "库存分布"):
            model = str(row.get("型号") or "").strip()
            if not model:
                continue
            qty = str(row.get("数量", ""))
            registry.consider(
                _Fact(model, "库存", qty, 10, xlsx.name), primary=PRIMARY_SOURCE["库存"]
            )
    return cards


def _faq_category_cards(base: Path) -> list[Document]:
    path = base / "tsv" / "常见问题分类.tsv"
    if not path.is_file():
        return []
    cards: list[Document] = []
    for row in read_tsv_rows(path):
        code = (row.get("编号") or "").strip()
        if not code:
            continue
        content = (
            f"工单分类：{row.get('问题类别', '')}（编号 {code}），"
            f"关键词：{row.get('关键词', '')}，"
            f"责任部门：{row.get('责任部门', '')}，"
            f"首响时限：{row.get('首响时限(小时)', '')} 小时。"
        )
        cards.append(
            _doc(
                content,
                path.name,
                doc_type="faq",
                authority_rank=20,
                section="常见问题分类",
            )
        )
    return cards
