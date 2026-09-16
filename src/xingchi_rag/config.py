"""应用配置（pydantic-settings）。

对应文档 §12.2 配置契约；字段与 `.env` / `.env.example` 一一对应。
所有值均可在 `.env` 中覆盖，切勿将真实密钥提交到仓库。
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(StrEnum):
    """LLM 提供方。"""

    OPENAI = "openai"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"


class EmbedProvider(StrEnum):
    """Embedding 提供方。"""

    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """全局配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # 空值视为未设置，回退到字段默认值（.env.example 中大量留空）
        env_ignore_empty=True,
        extra="ignore",
    )

    # ---------- LLM ----------
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_fallback: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llamacpp_model_path: str = ""

    # ---------- Embedding / Rerank ----------
    embed_provider: EmbedProvider = EmbedProvider.OPENAI
    embed_model: str = "text-embedding-3-small"
    # 部分供应商（如阿里云 Qwen）限制单批 embedding ≤20
    embed_batch_size: int = 20
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # API 重排端点（默认阿里云 DashScope 原生 text-rerank）
    rerank_endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    rerank_max_documents: int = 20

    # ---------- 检索参数 ----------
    retrieve_k: int = 20
    bm25_k: int = 20
    rerank_top_n: int = 5
    rrf_c: int = 60

    # ---------- 判据（§12.5）----------
    grade_score_threshold: float = 0.50
    check_retry_max: int = 1

    # ---------- 存储路径 ----------
    index_version: int = 1
    chroma_path: Path = Path("storage/chroma")
    sqlite_path: Path = Path("storage/xingchi.db")
    checkpoint_path: Path = Path("storage/checkpoints.sqlite")
    bm25_path: Path = Path("storage/bm25")
    parent_store_path: Path = Path("storage/parent_store")
    manifest_path: Path = Path("storage/manifest.json")
    quality_report_path: Path = Path("storage/data_quality_report.json")

    # ---------- 数据源 ----------
    data_dir: Path = Path("data")
    configs_dir: Path = Path("configs")

    # ---------- 可观测（LangSmith）----------
    langsmith_tracing: bool = False
    langsmith_project: str = "xingchi-rag-cs"
    langsmith_api_key: str = ""

    # ---------- 鉴权（§12.7）----------
    auth_enabled: bool = True
    jwt_secret: str = Field(default="change-me")
    jwt_ttl_minutes: int = 60
    service_api_key: str = ""
    pii_salt: str = "xingchi-pii-salt"

    # ---------- 运行时 ----------
    human_handoff_enabled: bool = True
    request_timeout_s: int = 30
    llm_timeout_s: int = 60
    log_level: str = "INFO"
    # LLM 并发上限（信号量）；SQLite 写锁等待；API 进程数
    llm_max_concurrency: int = 4
    sqlite_busy_timeout_ms: int = 5000
    api_workers: int = 1

    # ---------- 缓存（高频问答）----------
    cache_enabled: bool = True
    cache_ttl_s: int = 300
    cache_max_size: int = 256

    # ---------- 派生属性 ----------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, path: Path) -> Path:
        """将相对路径解析为相对项目根目录的绝对路径。"""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_storage_dirs(self) -> None:
        """创建运行时产物目录（幂等）。"""
        for p in (
            self.chroma_path,
            self.bm25_path,
            self.parent_store_path,
            self.sqlite_path.parent,
            self.checkpoint_path.parent,
        ):
            self.resolve(p).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的配置单例。"""
    return Settings()
