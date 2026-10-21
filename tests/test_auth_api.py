from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from secure_rag import api
from secure_rag.auth import AUDIENCE, AuthError, mint_token, principal_from_token

SECRET = "test-secret-0123456789abcdef0123456789"


def test_roundtrip():
    tok = mint_token(SECRET, "bob", "acme", ["hr", "all"])
    p = principal_from_token(tok, SECRET)
    assert (p.user_id, p.tenant_id, p.groups) == ("bob", "acme", ("all", "hr"))


@pytest.mark.parametrize("claims", [
    {"groups": ["all,exec"]},          # comma smuggling into the session setting
    {"groups": "all"},                 # not a list
    {"groups": []},                    # no groups
    {"tenant": "acme' OR '1'='1"},     # junk tenant
])
def test_malformed_claims_rejected(claims):
    base = {"sub": "bob", "tenant": "acme", "groups": ["all"], "aud": AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}
    tok = jwt.encode({**base, **claims}, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        principal_from_token(tok, SECRET)


def test_wrong_secret_and_expired_rejected():
    with pytest.raises(AuthError):
        principal_from_token(mint_token("other-secret-0123456789abcdef012345", "bob", "acme", ["all"]), SECRET)
    with pytest.raises(AuthError):
        principal_from_token(mint_token(SECRET, "bob", "acme", ["all"], ttl_minutes=-1), SECRET)


def test_api_end_to_end(db, monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: db)
    with TestClient(api.app) as client:
        assert client.post("/query", json={"question": "x"}).status_code in (401, 403)
        bad = client.post("/query", json={"question": "x"}, headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401

        tok = mint_token(db.jwt_secret, "dave", "acme", ["all"])
        r = client.post("/query", json={"question": "Acme salary bands 2026"},
                        headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert body["sources"], "dave can still read the handbook"
        assert all(s["doc_id"] == "acme/handbook" for s in body["sources"])
        assert "CANARY" not in body["answer"]
