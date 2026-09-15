"""Phase 0 基线测试：配置加载与 PII 脱敏。"""

from __future__ import annotations

from xingchi_rag.config import Settings, get_settings
from xingchi_rag.logging import mask_pii


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.retrieve_k == 20
    assert settings.bm25_k == 20
    assert settings.rerank_top_n == 5
    assert settings.rrf_c == 60
    assert settings.grade_score_threshold == 0.50
    assert settings.check_retry_max == 1
    assert settings.llm_provider.value == "openai"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_resolve_relative_path() -> None:
    settings = Settings()
    resolved = settings.resolve(settings.sqlite_path)
    assert resolved.is_absolute()
    assert resolved.name == "xingchi.db"


def test_mask_pii_phone() -> None:
    masked = mask_pii("联系电话 13812345678 请回拨")
    assert "13812345678" not in masked
    assert "138****5678" in masked


def test_mask_pii_masked_phone_preserved() -> None:
    masked = mask_pii("客户 138****6621 已登记")
    assert "138****6621" in masked


def test_mask_pii_email() -> None:
    masked = mask_pii("邮箱 service@xingchi-tech.example")
    assert "service@xingchi-tech.example" not in masked
    assert "@xingchi-tech.example" in masked


def test_mask_pii_empty() -> None:
    assert mask_pii("") == ""


def test_doc_reference_example() -> None:
    """文档 §8.4 的拒答话术中的邮箱应被脱敏，号码不应被误伤。"""
    text = "客服热线：400-820-6688；服务邮箱：service@xingchi-tech.example"
    masked = mask_pii(text)
    assert "400-820-6688" in masked
    assert "service@xingchi-tech.example" not in masked
