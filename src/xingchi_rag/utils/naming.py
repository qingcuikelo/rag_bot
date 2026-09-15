"""型号归一化与消歧（文档 §6.2）。

规则来自 ``configs/aliases.yaml``；用于查询理解阶段。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from xingchi_rag.utils.io import load_yaml

ALIASES_PATH = "configs/aliases.yaml"


@lru_cache(maxsize=1)
def load_aliases(path: str = ALIASES_PATH) -> dict[str, Any]:
    """加载型号别名配置（缓存）。"""
    return load_yaml(path)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """别名（小写）→ 规范型号。"""
    config = load_aliases()
    index: dict[str, str] = {}
    for model, meta in (config.get("models") or {}).items():
        index[model.lower()] = model
        index[model.replace("-", "").lower()] = model
        for alias in meta.get("aliases") or []:
            index[str(alias).lower()] = model
    return index


def known_models() -> list[str]:
    """全部规范型号。"""
    return list((load_aliases().get("models") or {}).keys())


def is_known_model(model: str) -> bool:
    """是否为已知规范型号。"""
    return model in known_models()


def normalize_model(text: str) -> str | None:
    """把文本中的型号/别称归一为规范型号；未命中返回 ``None``。"""
    if not text:
        return None
    lowered = text.lower()
    # 优先匹配显式型号（XC-L100 / XCL100）
    for pattern in (r"xc-[a-z]\d{1,3}", r"xc[a-z]\d{1,3}"):
        match = re.search(pattern, lowered)
        if match:
            key = match.group().replace("-", "").replace("xc", "")
            candidate = f"XC-{key.upper()}"
            if is_known_model(candidate):
                return candidate
    # 再按别名索引匹配（较长别名优先）
    for alias in sorted(_alias_index(), key=len, reverse=True):
        if alias and alias in lowered:
            return _alias_index()[alias]
    return None


def detect_unknown_model(text: str) -> str | None:
    """检测形如 ``XC-L300`` 但不存在的型号，返回该型号字符串（用于明确告知无此型号）。"""
    if not text:
        return None
    lowered = text.lower()
    for match in re.finditer(r"xc-[a-z]\d{1,3}", lowered):
        candidate = f"XC-{match.group().replace('xc-', '').upper()}"
        if not is_known_model(candidate):
            return candidate
    return None


def ambiguous_options(text: str) -> list[str] | None:
    """仅泛称（如"门锁"）而无具体型号时，返回候选型号列表。"""
    if normalize_model(text):
        return None
    config = load_aliases()
    terms: dict[str, list[str]] = (config.get("disambiguation") or {}).get("ambiguous_terms") or {}
    for term, options in terms.items():
        if term in text:
            return list(options)
    return None


def category_of(model: str) -> str | None:
    """型号对应产品类别。"""
    meta = (load_aliases().get("models") or {}).get(model) or {}
    return meta.get("category")
