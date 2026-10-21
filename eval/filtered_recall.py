"""The cost of filtering: HNSW + row-level security at different selectivities.

    python eval/filtered_recall.py [--chunks 20000] [--queries 50]

HNSW finds its nearest candidates first and applies filters afterwards. With
the default ef_search=40 the index hands back ~40 candidates; if the caller can
only read 2% of rows, about one of those survives the RLS filter and the user
gets 1 result instead of 10. pgvector 0.8's iterative index scans keep walking
the graph until enough rows pass the filter.

This script builds a throwaway tenant (`scale`) with synthetic clustered
vectors, gives three users different visibility (2% / 20% / 100%), and runs
the same dense query through the real request scope (app role, RLS on) under
four settings. Ground truth is an exact scan under the *same* RLS policy, so
recall measures only what the index loses. Vectors are synthetic on purpose:
this isolates index behaviour from embedding quality. The tenant is deleted at
the end.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from common import md_table, save  # noqa: E402  (sets sys.path)

from secure_rag.config import get_settings
from secure_rag.db import admin_connect, make_app_pool, request_scope
from secure_rag.principal import Principal

TENANT = "scale"
DIM = 384
K = 10
DOCS = 200
NARROW_DOCS, MEDIUM_DOCS = 4, 40  # 2% and 20% of docs (and of chunks)

USERS = {
    "narrow (2%)": Principal("scale-narrow", TENANT, ("narrow",)),
    "medium (20%)": Principal("scale-medium", TENANT, ("medium",)),
    "broad (100%)": Principal("scale-broad", TENANT, ("narrow", "medium", "bulk")),
}
MODES = [  # (label, iterative_scan, ef_search)
    ("post-filter, ef_search=40", "off", 40),
    ("post-filter, ef_search=200", "off", 200),
    ("iterative relaxed_order", "relaxed_order", 40),
    ("iterative strict_order", "strict_order", 40),
]

# Exact search: the "+ 0" makes the ORDER BY expression unmatchable by the
# index, forcing a sequential scan under the same RLS policy. (Toggling
# enable_indexscan is not enough: psycopg prepares repeated statements and a
# cached generic plan keeps using the index.)
EXACT = """
SELECT id FROM chunks ORDER BY (embedding <=> %(q)s) + 0 LIMIT %(k)s
"""

DENSE = """
WITH nn AS MATERIALIZED (
  SELECT id, embedding <=> %(q)s AS distance FROM chunks ORDER BY distance LIMIT %(k)s
) SELECT id FROM nn ORDER BY distance
"""


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in v) + "]"


def build(settings, n_chunks: int, rng: np.random.Generator, centers: np.ndarray) -> None:
    per_doc = n_chunks // DOCS
    with admin_connect(settings) as conn:
        conn.execute("DELETE FROM documents WHERE tenant_id = %s", (TENANT,))
        conn.execute("INSERT INTO tenants (id, name) VALUES (%s, 'Synthetic scale test') "
                     "ON CONFLICT (id) DO NOTHING", (TENANT,))
        for d in range(DOCS):
            group = "narrow" if d < NARROW_DOCS else "medium" if d < NARROW_DOCS + MEDIUM_DOCS else "bulk"
            conn.execute("INSERT INTO documents (id, tenant_id, title, source, acl_groups) "
                         "VALUES (%s, %s, %s, 'synthetic', %s)",
                         (f"{TENANT}/doc-{d:03d}", TENANT, f"Synthetic {d}", [group]))
        # Each chunk sits near a random cluster, independent of its ACL group,
        # so visible rows are scattered uniformly through the graph (the hard case).
        assign = rng.integers(0, len(centers), size=DOCS * per_doc)
        vecs = unit(centers[assign] + rng.normal(0, 0.05, size=(DOCS * per_doc, DIM)))
        with conn.cursor().copy(
            "COPY chunks (doc_id, tenant_id, ord, content, embedding) FROM STDIN"
        ) as cp:
            for i, v in enumerate(vecs):
                d, o = divmod(i, per_doc)
                cp.write_row((f"{TENANT}/doc-{d:03d}", TENANT, o, "synthetic chunk", vec_literal(v)))
        conn.commit()
        conn.execute("ANALYZE chunks")


def run_queries(pool, principal, mode, ef, queries, exact: bool) -> tuple[list[list[int]], list[float]]:
    results, lat = [], []
    with pool.connection() as conn:
        for q in queries:
            with request_scope(conn, principal, iterative_scan=mode, ef_search=ef):
                t = time.perf_counter()
                ids = [r[0] for r in conn.execute(EXACT if exact else DENSE, {"q": q, "k": K})]
                lat.append((time.perf_counter() - t) * 1000)
            results.append(ids)
    return results, lat


def plan_of(pool, principal, sql, q) -> str:
    with pool.connection() as conn, request_scope(conn, principal):
        return "\n".join(r[0] for r in conn.execute("EXPLAIN " + sql, {"q": q, "k": K}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--keep", action="store_true", help="don't delete the synthetic tenant")
    args = ap.parse_args()

    s = get_settings()
    rng = np.random.default_rng(7)
    centers = unit(rng.normal(size=(100, DIM)))
    print(f"building {args.chunks} synthetic chunks in tenant '{TENANT}'...")
    t0 = time.perf_counter()
    build(s, args.chunks, rng, centers)
    print(f"  built + indexed in {time.perf_counter() - t0:.1f}s")

    queries = [q.astype(np.float32) for q in
               unit(centers[rng.integers(0, len(centers), args.queries)]
                    + rng.normal(0, 0.05, size=(args.queries, DIM)))]
    pool = make_app_pool(s, max_size=2)
    rows, payload = [], {"chunks": args.chunks, "queries": args.queries, "k": K, "results": []}
    try:
        narrow = USERS["narrow (2%)"]
        assert "chunks_embedding_hnsw" in plan_of(pool, narrow, DENSE, queries[0]), \
            "planner is not using the HNSW index under RLS; numbers would be meaningless"
        assert "chunks_embedding_hnsw" not in plan_of(pool, narrow, EXACT, queries[0]), \
            "ground-truth query must not use the index"
        for ulabel, principal in USERS.items():
            truth, _ = run_queries(pool, principal, "off", 40, queries, exact=True)
            for mlabel, mode, ef in MODES:
                got, lat = run_queries(pool, principal, mode, ef, queries, exact=False)
                returned = statistics.mean(len(g) for g in got)
                recall = statistics.mean(len(set(g) & set(t)) / len(t) for g, t in zip(got, truth) if t)
                p50 = statistics.median(lat)
                rows.append([ulabel, mlabel, f"{returned:.1f}/{K}", f"{recall:.3f}", f"{p50:.1f}"])
                payload["results"].append({"user": ulabel, "mode": mlabel, "avg_returned": returned,
                                           "recall_at_k": recall, "p50_ms": p50})
    finally:
        pool.close()
        if not args.keep:
            with admin_connect(s) as conn:
                conn.execute("DELETE FROM documents WHERE tenant_id = %s", (TENANT,))
                conn.execute("DELETE FROM tenants WHERE id = %s", (TENANT,))

    print()
    print(md_table(["user visibility", "mode", "results returned", f"recall@{K}", "p50 ms"], rows))
    print(f"\nsaved {save('filtered_recall', payload)}")


if __name__ == "__main__":
    main()
