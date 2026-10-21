"""Hybrid retrieval (dense + lexical, RRF, optional cross-encoder rerank).

Read this file for what is NOT here: there is no `WHERE tenant_id = ...` and no
ACL check anywhere. These queries must run inside `db.request_scope`, and the
row-level-security policies in sql/schema.sql decide which rows exist. If a
future change adds a new query and forgets a filter, it still cannot see rows
the caller isn't entitled to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

from .models import Reranker

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    content: str
    score: float


# Materialised CTE + outer ORDER BY is the pattern pgvector recommends with
# iterative scans in relaxed_order mode, which can return slightly out of order.
DENSE_SQL = """
WITH nn AS MATERIALIZED (
  SELECT id, doc_id, content, embedding <=> %(q)s AS distance
  FROM chunks
  ORDER BY distance
  LIMIT %(n)s
)
SELECT nn.id, nn.doc_id, d.title, nn.content, nn.distance
FROM nn JOIN documents d ON d.id = nn.doc_id
ORDER BY nn.distance
"""

LEXICAL_SQL = """
SELECT c.id, c.doc_id, d.title, c.content, ts_rank_cd(c.tsv, q) AS rank
FROM chunks c
JOIN documents d ON d.id = c.doc_id,
     websearch_to_tsquery('english', %(text)s) q
WHERE c.tsv @@ q
ORDER BY rank DESC
LIMIT %(n)s
"""


def dense_search(conn: psycopg.Connection, qvec: np.ndarray, n: int) -> list[Hit]:
    rows = conn.execute(DENSE_SQL, {"q": np.asarray(qvec, dtype=np.float32), "n": n}).fetchall()
    return [Hit(r[0], r[1], r[2], r[3], 1.0 - float(r[4])) for r in rows]


def lexical_search(conn: psycopg.Connection, text: str, n: int) -> list[Hit]:
    rows = conn.execute(LEXICAL_SQL, {"text": text, "n": n}).fetchall()
    return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]


def rrf(*ranked_lists: list[Hit]) -> list[Hit]:
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranked in ranked_lists:
        for rank, h in enumerate(ranked, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            hits.setdefault(h.chunk_id, h)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [Hit(**{**hits[c].__dict__, "score": scores[c]}) for c in ordered]


def retrieve(
    conn: psycopg.Connection,
    question: str,
    qvec: np.ndarray,
    *,
    k: int = 5,
    candidates: int = 30,
    reranker: Reranker | None = None,
) -> list[Hit]:
    fused = rrf(dense_search(conn, qvec, candidates), lexical_search(conn, question, candidates))
    if reranker is None or not fused:
        return fused[:k]
    pool = fused[: max(k * 4, 20)]
    scores = reranker.score(question, [f"{h.title}\n{h.content}" for h in pool])
    reranked = sorted(zip(pool, scores), key=lambda t: t[1], reverse=True)
    return [Hit(**{**h.__dict__, "score": s}) for h, s in reranked[:k]]
