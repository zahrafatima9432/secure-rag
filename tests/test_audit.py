from __future__ import annotations

import uuid

import pytest
from psycopg.errors import InsufficientPrivilege, RaiseException

from secure_rag.db import admin_connect
from tests.conftest import principal_for


def _row(db, audit_id):
    with admin_connect(db) as a:
        return a.execute(
            "SELECT user_id, tenant_id, groups, chunk_ids, doc_ids, cached, query_text "
            "FROM audit_log WHERE id = %s", (audit_id,)).fetchone()


def test_every_answer_is_audited_with_what_was_retrieved(db, service, corpus):
    q = f"deployment freeze windows {uuid.uuid4()}"
    ans = service.answer(principal_for(corpus, "alice"), q)
    user, tenant, groups, chunk_ids, doc_ids, cached, query_text = _row(db, ans.audit_id)
    assert (user, tenant, groups, cached) == ("alice", "acme", ["all", "engineering"], False)
    assert chunk_ids == [s["chunk_id"] for s in ans.sources]
    assert set(doc_ids) == {s["doc_id"] for s in ans.sources}
    assert query_text is None  # hash only by default


def test_cache_hits_are_audited_too(db, service, corpus):
    q = f"how many days of paid leave {uuid.uuid4()}"
    first = service.answer(principal_for(corpus, "dave"), q)
    second = service.answer(principal_for(corpus, "dave"), q)
    assert second.cached and not first.cached
    assert _row(db, second.audit_id)[3] == _row(db, first.audit_id)[3]
    assert _row(db, second.audit_id)[5] is True


def test_chain_verifies(db, service, corpus):
    service.answer(principal_for(corpus, "kai"), "annual leave")
    with admin_connect(db) as a:
        ok, n, bad = a.execute("SELECT * FROM verify_audit_chain()").fetchone()
    assert ok and n > 0 and bad is None


def test_audit_log_is_append_only_even_for_admin(db, service, corpus):
    service.answer(principal_for(corpus, "kai"), "annual leave")
    with admin_connect(db) as a, pytest.raises(RaiseException):
        a.execute("UPDATE audit_log SET user_id = 'mallory'")
    with admin_connect(db) as a, pytest.raises(RaiseException):
        a.execute("DELETE FROM audit_log")


def test_tampering_is_detected(db, service, corpus):
    ans = service.answer(principal_for(corpus, "kai"), "home working allowance")
    with admin_connect(db) as a:
        original = a.execute("SELECT doc_ids FROM audit_log WHERE id = %s", (ans.audit_id,)).fetchone()[0]
        a.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_no_update")
        a.execute("UPDATE audit_log SET doc_ids = '{}' WHERE id = %s", (ans.audit_id,))
        a.commit()
        try:
            ok, _, bad = a.execute("SELECT * FROM verify_audit_chain()").fetchone()
            assert not ok and bad == ans.audit_id
        finally:
            a.execute("UPDATE audit_log SET doc_ids = %s WHERE id = %s", (original, ans.audit_id))
            a.execute("ALTER TABLE audit_log ENABLE TRIGGER audit_no_update")
            a.commit()
        assert a.execute("SELECT ok FROM verify_audit_chain()").fetchone()[0]


def test_tripwire_rejects_unreadable_chunk_ids(db, as_user):
    """If the app ever hands the audit writer a chunk the caller can't read,
    the request aborts before generation."""
    with admin_connect(db) as a:
        secret_chunk = a.execute(
            "SELECT id FROM chunks WHERE doc_id = 'acme/exec-acquisition-memo' LIMIT 1").fetchone()[0]
        foreign_chunk = a.execute(
            "SELECT id FROM chunks WHERE tenant_id = 'halcyon' LIMIT 1").fetchone()[0]
    for chunk in (secret_chunk, foreign_chunk):
        with pytest.raises(InsufficientPrivilege):
            with as_user("dave") as conn:
                conn.execute("SELECT append_audit(%s, 'x', NULL, %s, false)", (str(uuid.uuid4()), [chunk]))


def test_audit_requires_request_context(app_conn):
    with pytest.raises(RaiseException):
        app_conn.execute("SELECT append_audit(%s, 'x', NULL, '{}', false)", (str(uuid.uuid4()),))
