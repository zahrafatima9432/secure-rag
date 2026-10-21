from __future__ import annotations

import uuid

from secure_rag.cache import SemanticCache
from secure_rag.db import admin_connect
from secure_rag.principal import Principal
from tests.conftest import principal_for


def test_scope_key_separates_tenants_groups_and_epochs():
    a = Principal("u1", "acme", ("all", "hr"))
    key = SemanticCache.scope_key
    assert key(a, 1) != key(Principal("u1", "halcyon", ("all", "hr")), 1)
    assert key(a, 1) != key(Principal("u1", "acme", ("all",)), 1)
    assert key(a, 1) != key(a, 2)
    # Same permission set -> same scope, regardless of user or group order.
    assert key(a, 1) == key(Principal("u2", "acme", ("hr", "all", "hr")), 1)


def test_privileged_answer_is_not_served_to_unprivileged_user(service, corpus):
    q = "What is the proposed price for Project Kestrel?"
    carol = service.answer(principal_for(corpus, "carol"), q)
    assert any(s["doc_id"] == "acme/exec-acquisition-memo" for s in carol.sources)
    dave = service.answer(principal_for(corpus, "dave"), q)
    assert not dave.cached
    assert all(s["doc_id"] != "acme/exec-acquisition-memo" for s in dave.sources)
    assert "26 million" not in dave.answer


def test_acl_change_invalidates_cache(db, service, corpus):
    bob = principal_for(corpus, "bob")
    q = f"senior engineer salary band {uuid.uuid4()}"
    service.answer(bob, q)
    assert service.answer(bob, q).cached
    with admin_connect(db) as a:  # any document change bumps the tenant's epoch
        a.execute("UPDATE documents SET updated_at = now() WHERE id = 'acme/hr-salary-bands'")
    assert not service.answer(bob, q).cached
