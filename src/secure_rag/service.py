"""The request pipeline: scope -> cache/retrieve -> audit -> commit -> generate.

Ordering is the design:

1. Everything that touches the corpus happens inside ONE transaction that
   carries the caller's identity (db.request_scope).
2. The audit row is written in that same transaction, *before* generation.
   append_audit() re-checks every chunk id against the current ACL and raises
   if any is out of bounds, which aborts the request before the LLM sees it.
3. The transaction commits, so the record of what was retrieved for whom is
   durable even if the LLM call later fails, and the DB connection isn't held
   open during a slow model call.
4. Only then is the context sent to the generator.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from .cache import CachedAnswer, SemanticCache
from .config import Settings
from .db import request_scope
from .models import ContextChunk, Embedder, Generator, Reranker
from .principal import Principal
from .retrieval import Hit, retrieve


class UnknownTenant(PermissionError):
    pass


@dataclass
class Answer:
    request_id: str
    audit_id: int
    answer: str
    sources: list[dict]
    cached: bool
    context: list[Hit] | None = None  # populated for evals; never returned by the API


class RAGService:
    def __init__(
        self,
        settings: Settings,
        pool: ConnectionPool,
        embedder: Embedder,
        generator: Generator,
        reranker: Reranker | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        self.cache = cache

    def answer(self, principal: Principal, question: str, k: int | None = None,
               *, iterative_scan: str | None = None) -> Answer:
        s = self.settings
        k = k or s.top_k
        request_id = str(uuid.uuid4())
        qvec = self.embedder.embed_query(question)
        hits: list[Hit] = []
        cached_entry: CachedAnswer | None = None

        with self.pool.connection() as conn, request_scope(
            conn, principal,
            iterative_scan=iterative_scan or s.iterative_scan,
            ef_search=s.ef_search,
        ):
            row = conn.execute("SELECT acl_epoch FROM tenants").fetchone()  # RLS: own tenant only
            if row is None:
                raise UnknownTenant(principal.tenant_id)
            scope = SemanticCache.scope_key(principal, row[0])

            if self.cache is not None:
                cached_entry = self.cache.get(scope, qvec)
            if cached_entry is None:
                hits = retrieve(conn, question, qvec, k=k, candidates=s.candidates, reranker=self.reranker)
                chunk_ids = [h.chunk_id for h in hits]
            else:
                chunk_ids = cached_entry.chunk_ids

            audit_id = conn.execute(
                "SELECT append_audit(%s, %s, %s, %s, %s)",
                (
                    request_id,
                    hashlib.sha256(question.encode()).hexdigest(),
                    question if s.audit_store_query else None,
                    chunk_ids,
                    cached_entry is not None,
                ),
            ).fetchone()[0]
        # ---- committed: audit is durable, connection returned to the pool ----

        if cached_entry is not None:
            return Answer(request_id, audit_id, cached_entry.answer, cached_entry.sources, True)

        context = [ContextChunk(i + 1, h.doc_id, h.title, h.content) for i, h in enumerate(hits)]
        text = self.generator.generate(question, context)
        sources = [{"n": c.n, "doc_id": c.doc_id, "title": c.title, "chunk_id": h.chunk_id}
                   for c, h in zip(context, hits)]
        if self.cache is not None:
            self.cache.put(scope, CachedAnswer(question, qvec, text, chunk_ids, sources))
        return Answer(request_id, audit_id, text, sources, False, context=hits)
