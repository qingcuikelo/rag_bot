"""测试公共夹具：隔离运行时产物目录。"""

from __future__ import annotations

import pytest

from xingchi_rag.config import get_settings


@pytest.fixture
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """把 storage 相关路径指向临时目录，避免测试污染工作区。"""
    overrides = {
        "CHROMA_PATH": "chroma",
        "SQLITE_PATH": "xingchi.db",
        "CHECKPOINT_PATH": "checkpoints.sqlite",
        "BM25_PATH": "bm25",
        "PARENT_STORE_PATH": "parent_store",
        "MANIFEST_PATH": "manifest.json",
        "QUALITY_REPORT_PATH": "data_quality_report.json",
    }
    for env_key, filename in overrides.items():
        monkeypatch.setenv(env_key, str(tmp_path / filename))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
