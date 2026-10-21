"""Test fixtures.

Tests run against a real Postgres (the security properties live in Postgres,
so mocking it would test nothing). They use a separate database,
`<your db>_test`, which is dropped and recreated per session. Models are the
dependency-free stand-ins, so no downloads and no API key are needed.
"""

from __future__ import annotations

import dataclasses

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from secure_rag.cache import SemanticCache
from secure_rag.config import get_settings
from secure_rag.db import admin_connect, init_db, make_app_pool, request_scope
from secure_rag.ingest import ingest, load_corpus
from secure_rag.models import HashEmbedder, MockGenerator
from secure_rag.principal import Principal
from secure_rag.service import RAGService


def _with_db(dsn: str, dbname: str) -> str:
    return make_conninfo(dsn, dbname=dbname)


@pytest.fixture(scope="session")
def settings():
    base = get_settings()
    test_db = conninfo_to_dict(base.admin_dsn).get("dbname", "securerag") + "_test"
    try:
        with psycopg.connect(_with_db(base.admin_dsn, "postgres"), autocommit=True, connect_timeout=3) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
            c.execute(f'CREATE DATABASE "{test_db}"')
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable at ADMIN_DSN ({e}). Run `docker compose up -d`.")
    return dataclasses.replace(
        base,
        admin_dsn=_with_db(base.admin_dsn, test_db),
        app_dsn=_with_db(base.app_dsn, test_db),
        embedder="hash", reranker="none", generator="mock",
        jwt_secret="test-secret-0123456789abcdef0123456789",
    )


@pytest.fixture(scope="session")
def corpus(settings):
    return load_corpus(settings.corpus_dir)


@pytest.fixture(scope="session")
def db(settings, corpus):
    init_db(settings)
    with admin_connect(settings) as conn:
        ingest(conn, corpus, HashEmbedder())
    return settings


@pytest.fixture(scope="session")
def pool(db):
    p = make_app_pool(db, max_size=4)
    yield p
    p.close()


@pytest.fixture()
def service(db, pool):
    return RAGService(db, pool, HashEmbedder(), MockGenerator(), None, SemanticCache(0.95))


@pytest.fixture()
def app_conn(db):
    with psycopg.connect(db.app_dsn) as conn:
        yield conn


def principal_for(corpus, user_id: str) -> Principal:
    u = corpus.users[user_id]
    return Principal(u.id, u.tenant, tuple(u.groups))


@pytest.fixture()
def as_user(app_conn, corpus):
    """`with as_user("bob") as conn:` -> an app-role connection scoped to bob."""
    def _scope(user_id: str, **kw):
        return request_scope(app_conn, principal_for(corpus, user_id), **kw)
    return _scope
