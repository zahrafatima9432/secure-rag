"""Database access: admin connections, the app pool, and the request scope.

The important function here is `request_scope`. It opens a transaction on an
app-role connection and sets the caller's identity with `set_config(..., true)`,
which is the parameterised equivalent of `SET LOCAL`. "Local" matters: the
settings vanish at COMMIT/ROLLBACK, so a pooled connection handed to the next
request carries no identity at all (and therefore sees no rows) until that
request sets its own. tests/test_rls.py checks exactly that.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool

from .config import REPO_ROOT, Settings
from .principal import Principal

SCHEMA_SQL = REPO_ROOT / "sql" / "schema.sql"
VALID_ITERATIVE = {"off", "relaxed_order", "strict_order"}


def admin_connect(settings: Settings, **kwargs) -> psycopg.Connection:
    return psycopg.connect(settings.admin_dsn, **kwargs)


def init_db(settings: Settings) -> None:
    """Create the extension, schema, policies and app role. Safe to re-run."""
    # Extension first, on its own connection, so register_vector works afterwards
    # (creating the extension and registering its type in one session is the
    # classic "type vector does not exist" footgun).
    with admin_connect(settings, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with admin_connect(settings, autocommit=True) as conn:
        conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute(sql.SQL("ALTER ROLE rag_app PASSWORD {}").format(sql.Literal(settings.app_password)))


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def make_app_pool(settings: Settings, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    return ConnectionPool(
        settings.app_dsn,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        open=True,
    )


@contextmanager
def request_scope(
    conn: psycopg.Connection,
    principal: Principal,
    *,
    iterative_scan: str = "relaxed_order",
    ef_search: int = 40,
) -> Iterator[psycopg.Connection]:
    """Run the body inside one transaction that carries the caller's identity."""
    if iterative_scan not in VALID_ITERATIVE:
        raise ValueError(f"iterative_scan must be one of {sorted(VALID_ITERATIVE)}")
    with conn.transaction():
        conn.execute(
            """
            SELECT set_config('app.tenant_id', %s, true),
                   set_config('app.groups',    %s, true),
                   set_config('app.user_id',   %s, true),
                   set_config('hnsw.iterative_scan', %s, true),
                   set_config('hnsw.ef_search', %s, true)
            """,
            (principal.tenant_id, principal.groups_setting, principal.user_id,
             iterative_scan, str(ef_search)),
        )
        yield conn
