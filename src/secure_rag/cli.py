"""Command line entry point:  python -m secure_rag.cli <command> ...

  init-db                     create extension, schema, RLS policies, app role
  ingest                      load corpus/ into Postgres (admin role)
  token USER                  mint a dev JWT for a user from corpus/tenants.yaml
  ask USER "question"         run the full pipeline as USER (prints sources)
  revoke DOC_ID GROUP         remove GROUP from a document's ACL (instant)
  grant DOC_ID GROUP          add GROUP to a document's ACL
  audit-tail [-n N]           show the latest audit rows
  audit-verify                walk the hash chain
  audit-head                  print the chain head hash (anchor it externally)
"""

from __future__ import annotations

import argparse
import json
import sys

from .cache import SemanticCache
from .config import get_settings
from .db import admin_connect, init_db, make_app_pool
from .ingest import ingest, load_corpus
from .models import make_embedder, make_generator, make_reranker
from .principal import Principal
from .service import RAGService


def _principal(settings, user_id: str) -> Principal:
    corpus = load_corpus(settings.corpus_dir)
    if user_id not in corpus.users:
        sys.exit(f"unknown user {user_id!r}; see corpus/tenants.yaml")
    u = corpus.users[user_id]
    return Principal(u.id, u.tenant, tuple(u.groups))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="secure_rag.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("ingest")
    t = sub.add_parser("token"); t.add_argument("user")
    a = sub.add_parser("ask"); a.add_argument("user"); a.add_argument("question")
    for name in ("revoke", "grant"):
        x = sub.add_parser(name); x.add_argument("doc_id"); x.add_argument("group")
    tail = sub.add_parser("audit-tail"); tail.add_argument("-n", type=int, default=10)
    sub.add_parser("audit-verify")
    sub.add_parser("audit-head")
    args = p.parse_args(argv)
    s = get_settings()

    if args.cmd == "init-db":
        init_db(s)
        print("schema, policies and rag_app role are in place")

    elif args.cmd == "ingest":
        corpus = load_corpus(s.corpus_dir)
        with admin_connect(s) as conn:
            stats = ingest(conn, corpus, make_embedder(s))
        print(f"ingested {stats['documents']} documents / {stats['chunks']} chunks "
              f"for {len(corpus.tenants)} tenants")

    elif args.cmd == "token":
        from .auth import mint_token
        if not s.jwt_secret:
            sys.exit("set JWT_SECRET first")
        pr = _principal(s, args.user)
        print(mint_token(s.jwt_secret, pr.user_id, pr.tenant_id, list(pr.groups), s.jwt_ttl_minutes))

    elif args.cmd == "ask":
        pr = _principal(s, args.user)
        pool = make_app_pool(s, max_size=2)
        try:
            svc = RAGService(s, pool, make_embedder(s), make_generator(s), make_reranker(s),
                             SemanticCache(s.cache_threshold) if s.cache_enabled else None)
            ans = svc.answer(pr, args.question)
        finally:
            pool.close()
        print(f"[{pr.user_id} @ {pr.tenant_id} groups={','.join(pr.groups)}]  audit #{ans.audit_id}\n")
        print(ans.answer, "\n")
        for src in ans.sources:
            print(f"  [{src['n']}] {src['title']}  ({src['doc_id']}, chunk {src['chunk_id']})")
        if not ans.sources:
            print("  (no readable sources)")

    elif args.cmd in {"revoke", "grant"}:
        op = "array_remove(acl_groups, %s)" if args.cmd == "revoke" else \
             "(SELECT array_agg(DISTINCT g) FROM unnest(acl_groups || %s::text) g)"
        with admin_connect(s) as conn:
            row = conn.execute(
                f"UPDATE documents SET acl_groups = {op}, updated_at = now() "
                "WHERE id = %s RETURNING acl_groups", (args.group, args.doc_id)).fetchone()
        if row is None:
            sys.exit(f"no such document {args.doc_id!r}")
        print(f"{args.doc_id} acl is now {row[0]}")

    elif args.cmd == "audit-tail":
        with admin_connect(s) as conn:
            rows = conn.execute(
                "SELECT id, ts, user_id, tenant_id, groups, doc_ids, cached, left(hash, 12) "
                "FROM audit_log ORDER BY id DESC LIMIT %s", (args.n,)).fetchall()
        for r in reversed(rows):
            print(json.dumps({"id": r[0], "ts": r[1].isoformat(), "user": r[2], "tenant": r[3],
                              "groups": r[4], "docs": r[5], "cached": r[6], "hash": r[7]}))

    elif args.cmd == "audit-verify":
        with admin_connect(s) as conn:
            ok, n, bad = conn.execute("SELECT * FROM verify_audit_chain()").fetchone()
        print(f"chain OK ({n} rows)" if ok else f"CHAIN BROKEN at row {bad} (checked {n})")
        sys.exit(0 if ok else 1)

    elif args.cmd == "audit-head":
        with admin_connect(s) as conn:
            row = conn.execute("SELECT id, hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        print(f"{row[0]} {row[1]}" if row else "empty")


if __name__ == "__main__":
    main()
