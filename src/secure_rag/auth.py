"""JWT -> Principal.

In production the token comes from your identity provider and you'd verify it
against the IdP's JWKS (RS256). Here a shared HS256 secret plus `cli token`
stands in for the IdP so the demo is self-contained. Everything downstream only
sees a validated Principal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from .principal import InvalidPrincipal, Principal

ALGORITHM = "HS256"
AUDIENCE = "secure-rag"


class AuthError(Exception):
    pass


def mint_token(secret: str, user_id: str, tenant_id: str, groups: list[str], ttl_minutes: int = 60) -> str:
    Principal(user_id, tenant_id, tuple(groups))  # validate before signing
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "tenant": tenant_id,
        "groups": groups,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def principal_from_token(token: str, secret: str) -> Principal:
    try:
        claims = jwt.decode(
            token, secret, algorithms=[ALGORITHM], audience=AUDIENCE,
            options={"require": ["sub", "tenant", "groups", "exp", "aud"]},
        )
    except jwt.PyJWTError as e:
        raise AuthError(str(e)) from e
    groups = claims["groups"]
    if not isinstance(groups, list):
        raise AuthError("groups claim must be a list")
    try:
        return Principal(claims["sub"], claims["tenant"], tuple(groups))
    except InvalidPrincipal as e:
        raise AuthError(str(e)) from e
