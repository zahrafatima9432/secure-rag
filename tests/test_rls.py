"""Isolation is enforced by Postgres, so test it at the Postgres level."""

from __future__ import annotations

import psycopg
import pytest
from psycopg.errors import InsufficientPrivilege

from secure_rag.db import admin_connect


def test_app_role_cannot_bypass_rls(app_conn):
    row = app_conn.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    assert row == (False, False), "app must not connect as a superuser or BYPASSRLS role"
    owned = app_conn.execute(
        "SELECT count(*) FROM pg_tables WHERE schemaname = 'public' AND tableowner = current_user"
    ).fetchone()[0]
    assert owned == 0, "table owners bypass RLS; the app role must own nothing"


def test_no_context_means_no_rows(app_conn):
    for table in ("tenants", "documents", "chunks"):
        assert app_conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("user_id", ["alice", "bob", "carol", "dave", "erin",
                                     "hana", "ivan", "jin", "kai", "lena"])
def test_visible_documents_match_ground_truth(as_user, corpus, user_id):
    expected = {d.id for d in corpus.readable_by(corpus.users[user_id])}
    with as_user(user_id) as conn:
        docs = {r[0] for r in conn.execute("SELECT id FROM documents")}
        chunk_docs = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
    assert docs == expected
    assert chunk_docs == expected


def test_unfiltered_select_returns_no_foreign_canaries(as_user, corpus):
    """The query an engineer might write by accident: no WHERE clause at all."""
    for user in corpus.users.values():
        allowed = corpus.allowed_canaries(user)
        forbidden = corpus.all_canaries() - allowed
        with as_user(user.id) as conn:
            text = "\n".join(r[0] for r in conn.execute("SELECT content FROM chunks"))
        assert not any(c in text for c in forbidden), user.id


def test_identity_does_not_survive_the_transaction(as_user, app_conn):
    """Pooled connections must not carry the previous request's identity."""
    with as_user("carol") as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    assert app_conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert app_conn.execute("SELECT current_setting('app.tenant_id', true)").fetchone()[0] in (None, "")


def test_app_role_is_read_only(as_user):
    with pytest.raises(InsufficientPrivilege):
        with as_user("alice") as conn:
            conn.execute("UPDATE documents SET acl_groups = '{all}'")
    with pytest.raises(InsufficientPrivilege):
        with as_user("alice") as conn:
            conn.execute("DELETE FROM chunks")


def test_app_role_cannot_read_audit_log(as_user):
    with pytest.raises(InsufficientPrivilege):
        with as_user("carol") as conn:
            conn.execute("SELECT * FROM audit_log")


def test_revocation_is_immediate(db, as_user):
    doc = "acme/hr-salary-bands"
    with admin_connect(db) as a:
        a.execute("UPDATE documents SET acl_groups = array_remove(acl_groups, 'hr') WHERE id = %s", (doc,))
    try:
        with as_user("bob") as conn:
            assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc,)).fetchone()[0] == 0
    finally:
        with admin_connect(db) as a:
            a.execute("UPDATE documents SET acl_groups = '{hr,exec}' WHERE id = %s", (doc,))
    with as_user("bob") as conn:
        assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc,)).fetchone()[0] > 0


def test_chunk_cannot_claim_a_different_tenant_than_its_document(db):
    with admin_connect(db) as a, pytest.raises(psycopg.errors.ForeignKeyViolation):
        a.execute(
            "INSERT INTO chunks (doc_id, tenant_id, ord, content, embedding) "
            "VALUES ('acme/handbook', 'halcyon', 99, 'x', array_fill(0.1, ARRAY[384])::vector)"
        )
