"""版本化 Prompt 加载（文档 §8.2、§12.1）。

模板存放于 ``configs/prompts/{name}.md``，YAML frontmatter 提供 ``version`` 等元信息。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from xingchi_rag.config import get_settings

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _prompt_path(name: str) -> Path:
    settings = get_settings()
    return settings.resolve(settings.configs_dir) / "prompts" / f"{name}.md"


def load_prompt(name: str) -> tuple[str, str]:
    """加载 Prompt 模板，返回 ``(模板正文, 版本号)``。"""
    path = _prompt_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    raw = path.read_text(encoding="utf-8")
    version = name
    match = _FRONTMATTER_RE.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if line.strip().startswith("version:"):
                version = line.split(":", 1)[1].strip()
        raw = raw[match.end() :]
    return raw.strip(), version


@lru_cache(maxsize=16)
def _cached(name: str) -> tuple[str, str]:
    return load_prompt(name)


def get_prompt(name: str) -> tuple[str, str]:
    """带缓存的 Prompt 加载。"""
    return _cached(name)


def render(template: str, **values: str) -> str:
    """纯字符串替换渲染（避免与模板中的 JSON 花括号冲突）。"""
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", value)
    return result
