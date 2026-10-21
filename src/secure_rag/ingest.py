"""Corpus loading, chunking and ingestion (runs as the ADMIN role)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import psycopg
import yaml
from pgvector.psycopg import register_vector

from .models import Embedder
from .principal import IDENT_RE

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
CANARY_RE = re.compile(r"CANARY-[A-Z0-9-]+")
MAX_CHUNK_CHARS = 700


@dataclass
class Document:
    id: str
    tenant_id: str
    title: str
    acl: list[str]
    body: str
    source: str

    @property
    def canaries(self) -> set[str]:
        return set(CANARY_RE.findall(self.body))


@dataclass
class User:
    id: str
    tenant: str
    groups: list[str]


@dataclass
class Corpus:
    tenants: dict[str, str]
    users: dict[str, User]
    documents: list[Document]

    def readable_by(self, user: User) -> list[Document]:
        """Ground-truth ACL, computed from the files, independent of Postgres."""
        g = set(user.groups)
        return [d for d in self.documents if d.tenant_id == user.tenant and g & set(d.acl)]

    def allowed_canaries(self, user: User) -> set[str]:
        return set().union(*(d.canaries for d in self.readable_by(user)))

    def all_canaries(self) -> set[str]:
        return set().union(*(d.canaries for d in self.documents))


def load_corpus(corpus_dir: Path) -> Corpus:
    meta = yaml.safe_load((corpus_dir / "tenants.yaml").read_text(encoding="utf-8"))
    tenants = {t["id"]: t["name"] for t in meta["tenants"]}
    users = {u["id"]: User(u["id"], u["tenant"], list(u["groups"])) for u in meta["users"]}
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*/*.md")):
        tenant_id = path.parent.name
        if tenant_id not in tenants:
            raise ValueError(f"{path}: folder {tenant_id!r} is not a tenant in tenants.yaml")
        m = FRONTMATTER_RE.match(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
        if not m:
            raise ValueError(f"{path}: missing frontmatter")
        fm, body = yaml.safe_load(m.group(1)), m.group(2).strip()
        if not str(fm["id"]).startswith(f"{tenant_id}/"):
            raise ValueError(f"{path}: id must start with '{tenant_id}/'")
        acl = list(fm["acl"])
        if not acl or not all(IDENT_RE.match(g) for g in acl):
            raise ValueError(f"{path}: invalid acl {acl!r}")
        docs.append(Document(fm["id"], tenant_id, fm["title"], acl, body, str(path.relative_to(corpus_dir))))
    return Corpus(tenants, users, docs)


def chunk_text(body: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Greedy paragraph packing. Paragraphs are never split mid-sentence."""
    chunks, buf = [], ""
    for para in (p.strip() for p in body.split("\n\n") if p.strip()):
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def ingest(conn: psycopg.Connection, corpus: Corpus, embedder: Embedder) -> dict[str, int]:
    register_vector(conn)
    n_docs = n_chunks = 0
    with conn.transaction():
        for tid, name in corpus.tenants.items():
            conn.execute(
                "INSERT INTO tenants (id, name) VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                (tid, name),
            )
        for doc in corpus.documents:
            conn.execute(
                """
                INSERT INTO documents (id, tenant_id, title, source, acl_groups)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                  SET title = EXCLUDED.title, source = EXCLUDED.source,
                      acl_groups = EXCLUDED.acl_groups, updated_at = now()
                """,
                (doc.id, doc.tenant_id, doc.title, doc.source, doc.acl),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc.id,))
            pieces = chunk_text(doc.body)
            # Embed with the title for context; store the raw text.
            vecs = embedder.embed_documents([f"{doc.title}\n{p}" for p in pieces])
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, tenant_id, ord, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(doc.id, doc.tenant_id, i, p, v) for i, (p, v) in enumerate(zip(pieces, vecs))],
                )
            n_docs += 1
            n_chunks += len(pieces)
        # Files deleted from the corpus are deleted from the index (chunks cascade).
        conn.execute(
            "DELETE FROM documents WHERE tenant_id = ANY(%s) AND NOT (id = ANY(%s))",
            (list(corpus.tenants), [d.id for d in corpus.documents]),
        )
    return {"documents": n_docs, "chunks": n_chunks}
