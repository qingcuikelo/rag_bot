"""IO 工具：YAML 加载与路径解析。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from xingchi_rag.config import get_settings


def load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML 文件为字典。相对路径基于项目根目录解析。"""
    settings = get_settings()
    p = Path(path)
    if not p.is_absolute():
        p = settings.project_root / p
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须为映射: {p}")
    return data


def iter_files(directory: str | Path, suffixes: Iterable[str] | None = None) -> list[Path]:
    """递归列出目录下文件（按路径排序，保证确定性）。"""
    root = Path(directory)
    suffix_set = {s.lower() for s in suffixes} if suffixes else None
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and (suffix_set is None or p.suffix.lower() in suffix_set)
    ]
    return files
