"""The caller's identity, as far as retrieval is concerned."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Identifiers end up in a comma-joined session setting, so we constrain them
# tightly: a group named "hr,exec" must not be able to smuggle in a second group.
IDENT_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


class InvalidPrincipal(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    groups: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (("user_id", self.user_id), ("tenant_id", self.tenant_id)):
            if not isinstance(value, str) or not IDENT_RE.match(value):
                raise InvalidPrincipal(f"invalid {label}: {value!r}")
        if not self.groups:
            raise InvalidPrincipal("principal has no groups")
        for g in self.groups:
            if not isinstance(g, str) or not IDENT_RE.match(g):
                raise InvalidPrincipal(f"invalid group: {g!r}")
        # Normalise: sorted + deduplicated, so equal permission sets compare equal.
        object.__setattr__(self, "groups", tuple(sorted(set(self.groups))))

    @property
    def groups_setting(self) -> str:
        return ",".join(self.groups)
