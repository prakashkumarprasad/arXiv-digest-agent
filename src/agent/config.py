"""Configuration management for the arXiv Digest Agent.

Loads settings from environment variables with sensible defaults.
All external clients are lazily constructed via this module.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment."""

    # LLM Provider
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))

    # Groq
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))

    # Gemini
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

    # Ollama
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))

    # Embeddings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # Paths
    chroma_persist_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", ".data/chroma")))
    sqlite_db_path: Path = field(default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", ".data/sessions.sqlite")))
    pdf_cache_dir: Path = field(default_factory=lambda: Path(os.getenv("PDF_CACHE_DIR", ".data/pdfs")))

    # arXiv
    arxiv_max_results: int = field(default_factory=lambda: int(os.getenv("ARXIV_MAX_RESULTS", "15")))
    arxiv_sort_by: str = field(default_factory=lambda: os.getenv("ARXIV_SORT_BY", "Relevance"))

    # Chunking
    chunk_target_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_TARGET_TOKENS", "400")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "80")))

    # QA
    abstain_threshold: float = field(default_factory=lambda: float(os.getenv("ABSTAIN_THRESHOLD", "0.35")))
    qa_top_k: int = field(default_factory=lambda: int(os.getenv("QA_TOP_K", "6")))
    qa_mmr_lambda: float = field(default_factory=lambda: float(os.getenv("QA_MMR_LAMBDA", "0.6")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Create required directories."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (primarily for testing)."""
    global _settings
    _settings = None