"""Quality eval: do users still get good answers from what they CAN see?

    python eval/quality.py            # retrieval metrics only (mock generator)
    python eval/quality.py --groq     # also generate answers for the denied set

* allowed questions: the user can read the relevant document. Reports doc-level
  hit@k and MRR, per user, so restricted roles aren't hidden inside an average.
* denied questions: the answer lives in a document the user can't read. Checks
  that no forbidden fact appears in the answer. With --groq this tests the LLM
  says "I don't have access" instead of guessing; without it, the check is on
  the extractive mock, which can only echo retrieved text.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from collections import defaultdict

import yaml

from common import ROOT, md_table, save  # noqa: E402  (sets sys.path)

from secure_rag.config import get_settings
from secure_rag.db import make_app_pool
from secure_rag.ingest import load_corpus
from secure_rag.models import make_embedder, make_generator, make_reranker
from secure_rag.principal import Principal
from secure_rag.service import RAGService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groq", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    s = dataclasses.replace(s, generator="groq" if args.groq else "mock", cache_enabled=False)
    gold = yaml.safe_load((ROOT / "eval" / "gold.yaml").read_text(encoding="utf-8"))
    corpus = load_corpus(s.corpus_dir)
    pool = make_app_pool(s, max_size=2)
    svc = RAGService(s, pool, make_embedder(s), make_generator(s), make_reranker(s), None)
    P = lambda uid: Principal(uid, corpus.users[uid].tenant, tuple(corpus.users[uid].groups))  # noqa: E731

    per_user: dict[str, list[tuple[int, float]]] = defaultdict(list)
    details = []
    for item in gold["allowed"]:
        ans = svc.answer(P(item["user"]), item["q"])
        docs = [src["doc_id"] for src in ans.sources]
        rank = docs.index(item["relevant"]) + 1 if item["relevant"] in docs else 0
        per_user[item["user"]].append((int(rank > 0), 1.0 / rank if rank else 0.0))
        details.append({**item, "rank": rank, "retrieved": docs})

    denied = []
    for item in gold["denied"]:
        ans = svc.answer(P(item["user"]), item["q"])
        leaked = [f for f in item["forbidden_facts"] if f in ans.answer]
        denied.append({**item, "answer": ans.answer, "leaked_facts": leaked})
    pool.close()

    rows = []
    for user, vals in sorted(per_user.items()):
        u = corpus.users[user]
        rows.append([f"{user} ({u.tenant}: {','.join(g for g in u.groups if g != 'all') or 'all only'})",
                     len(vals), f"{statistics.mean(v[0] for v in vals):.2f}",
                     f"{statistics.mean(v[1] for v in vals):.2f}"])
    allv = [v for vals in per_user.values() for v in vals]
    rows.append(["**overall**", len(allv), f"{statistics.mean(v[0] for v in allv):.2f}",
                 f"{statistics.mean(v[1] for v in allv):.2f}"])
    print(md_table(["user", "questions", f"hit@{s.top_k}", "MRR"], rows))

    n_leak = sum(bool(d["leaked_facts"]) for d in denied)
    print(f"\ndenied questions: {len(denied)}, answers containing a forbidden fact: {n_leak}")
    if args.groq:
        for d in denied:
            print(f"\n[{d['user']}] {d['q']}\n  -> {d['answer'][:200]}")
    misses = [d for d in details if d["rank"] == 0]
    for m in misses:
        print(f"\nMISS [{m['user']}] {m['q']}\n  wanted {m['relevant']}, got {m['retrieved']}")
    print(f"\nembedder={s.embedder} reranker={s.reranker} generator={s.generator}")
    print(f"saved {save('quality', {'embedder': s.embedder, 'reranker': s.reranker, 'generator': s.generator, 'allowed': details, 'denied': denied})}")


if __name__ == "__main__":
    main()
