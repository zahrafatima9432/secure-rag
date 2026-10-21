"""Permission-scoped semantic cache.

A semantic cache shared across users is a data-leak channel: if an exec asks
"what's the acquisition price?" and the answer is cached globally, the next
person to ask a similar question gets the exec's answer without ever touching
retrieval. So every entry lives under a scope key built from

    tenant_id + sorted(groups) + acl_epoch

* Different tenants or different group sets never share entries.
* Users with the *same* group set do share entries. That is correct here
  because ACLs are group-based: identical groups means identical visible data.
  (Per-user ACLs would need user_id in the key.)
* acl_epoch is bumped by a trigger on every document change, so a revocation
  makes every older entry for that tenant unreachable on the next request.

In-process and bounded: good enough to show the scoping. A production version
would sit in Redis or pgvector with the same key structure.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .principal import Principal


@dataclass
class CachedAnswer:
    question: str
    qvec: np.ndarray
    answer: str
    chunk_ids: list[int]
    sources: list[dict]


@dataclass
class SemanticCache:
    threshold: float = 0.95
    max_scopes: int = 1024
    max_entries_per_scope: int = 256
    _scopes: OrderedDict[str, list[CachedAnswer]] = field(default_factory=OrderedDict)

    @staticmethod
    def scope_key(principal: Principal, acl_epoch: int) -> str:
        raw = f"{principal.tenant_id}|{principal.groups_setting}|{acl_epoch}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, scope: str, qvec: np.ndarray) -> CachedAnswer | None:
        entries = self._scopes.get(scope)
        if not entries:
            return None
        self._scopes.move_to_end(scope)
        mat = np.stack([e.qvec for e in entries])
        sims = mat @ np.asarray(qvec, dtype=np.float32)
        best = int(np.argmax(sims))
        return entries[best] if sims[best] >= self.threshold else None

    def put(self, scope: str, entry: CachedAnswer) -> None:
        entries = self._scopes.setdefault(scope, [])
        self._scopes.move_to_end(scope)
        entries.append(entry)
        if len(entries) > self.max_entries_per_scope:
            del entries[0]
        while len(self._scopes) > self.max_scopes:
            self._scopes.popitem(last=False)

    def clear(self) -> None:
        self._scopes.clear()
