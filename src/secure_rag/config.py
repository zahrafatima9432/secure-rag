"""Settings, read from the environment (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Two DSNs on purpose: ingestion/admin vs. the least-privilege app role.
    admin_dsn: str = field(default_factory=lambda: os.getenv(
        "ADMIN_DSN", "postgresql://admin:admin@localhost:5434/securerag"))
    app_dsn: str = field(default_factory=lambda: os.getenv(
        "APP_DSN", "postgresql://rag_app:rag_app_pw@localhost:5434/securerag"))
    app_password: str = field(default_factory=lambda: os.getenv("RAG_APP_PASSWORD", "rag_app_pw"))

    corpus_dir: Path = field(default_factory=lambda: Path(os.getenv("CORPUS_DIR", REPO_ROOT / "corpus")))

    # Models. "hash" / "none" are dependency-free stand-ins used by the test suite.
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "bge"))
    reranker: str = field(default_factory=lambda: os.getenv("RERANKER", "bge"))
    generator: str = field(default_factory=lambda: os.getenv("GENERATOR", "groq"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    # Retrieval.
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    candidates: int = field(default_factory=lambda: int(os.getenv("CANDIDATES", "30")))
    # off | relaxed_order | strict_order  (pgvector >= 0.8)
    iterative_scan: str = field(default_factory=lambda: os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order"))
    ef_search: int = field(default_factory=lambda: int(os.getenv("HNSW_EF_SEARCH", "40")))

    # Cache.
    cache_enabled: bool = field(default_factory=lambda: _bool("CACHE_ENABLED", True))
    cache_threshold: float = field(default_factory=lambda: float(os.getenv("CACHE_THRESHOLD", "0.95")))

    # Audit.
    audit_store_query: bool = field(default_factory=lambda: _bool("AUDIT_STORE_QUERY", False))

    # Auth.
    jwt_secret: str | None = field(default_factory=lambda: os.getenv("JWT_SECRET"))
    jwt_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_TTL_MINUTES", "60")))


def get_settings() -> Settings:
    return Settings()
