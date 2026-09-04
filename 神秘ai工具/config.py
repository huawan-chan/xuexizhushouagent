"""集中配置：环境变量 / .env 读取 + 统一路径管理。

所有密钥只从环境变量读取（.env 文件会被 python-dotenv 自动加载），
代码里不出现任何明文密钥，方便直接 push 到 GitHub。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:      # 未安装 python-dotenv 时不阻断(仅读取系统环境变量)
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# 项目根目录：以本文件所在位置为准，保证从任何 cwd 启动都能找到资源
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 数据目录
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"                      # 把要检索的 PDF 放进这里
CHROMA_DIR = DATA_DIR / "chroma_db"              # Chroma 持久化目录
MEMORY_DB = DATA_DIR / "memory.db"               # 长期用户画像(sqlite)
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"      # 短期会话(消息)持久化(sqlite)

for _d in (DATA_DIR, PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "")
    if v.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if v.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class Settings:
    # Groq
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.6"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_TOKENS", "4096"))
    )

    # Embedding / RAG
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "700")))
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120"))
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))

    # 网页搜索(可选)
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    web_search_enabled: bool = field(
        default_factory=lambda: _env_bool("WEB_SEARCH_ENABLED", True)
    )
    web_search_results: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_RESULTS", "5"))
    )

    # 记忆
    profile_llm_on: bool = field(
        default_factory=lambda: _env_bool("PROFILE_LLM_ON", True)
    )
    max_history: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY", "24")))

    # 运行方式
    app_ui: str = field(default_factory=lambda: os.getenv("APP_UI", "chainlit"))

    # 目录(便捷属性)
    @property
    def pdf_dir(self) -> Path:
        return PDF_DIR

    @property
    def chroma_dir(self) -> Path:
        return CHROMA_DIR

    @property
    def memory_db(self) -> Path:
        return MEMORY_DB

    @property
    def checkpoint_db(self) -> Path:
        return CHECKPOINT_DB

    def require_groq_key(self) -> str:
        """返回 Groq key，缺失时给出清晰报错。"""
        if not self.groq_api_key or self.groq_api_key.startswith("your_"):
            raise RuntimeError(
                "缺少 GROQ_API_KEY。请在项目根目录复制 .env.example 为 .env 并填入密钥：\n"
                "  copy .env.example .env   (Windows)\n"
                "  cp .env.example .env      (macOS / Linux)\n"
                "密钥申请: https://console.groq.com/keys (免费)"
            )
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
