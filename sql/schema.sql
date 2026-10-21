-- secure-rag schema
-- Run by the ADMIN role (see `python -m secure_rag.cli init-db`). Idempotent.
--
-- Security model in one paragraph:
--   The application connects as `rag_app`, a role that is NOT a superuser, does
--   NOT own any table, and does NOT have BYPASSRLS. Every query it runs passes
--   through the row-level-security policies below, which read the caller's
--   identity from transaction-local settings (app.tenant_id / app.groups /
--   app.user_id). If those settings are missing, the helper functions return
--   NULL / an empty array and every policy evaluates to false: the system fails
--   closed. Retrieval code in Python deliberately contains NO tenant or ACL
--   filters; isolation lives here, in one place.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Application role (cluster-wide). Password is set by init-db from the env.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
    CREATE ROLE rag_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  END IF;
END $$;
ALTER ROLE rag_app NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;

-- ---------------------------------------------------------------------------
-- Request context helpers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION app_tenant() RETURNS text
LANGUAGE sql STABLE AS
$$ SELECT nullif(current_setting('app.tenant_id', true), '') $$;

CREATE OR REPLACE FUNCTION app_groups() RETURNS text[]
LANGUAGE sql STABLE AS
$$ SELECT coalesce(string_to_array(nullif(current_setting('app.groups', true), ''), ','), '{}'::text[]) $$;

CREATE OR REPLACE FUNCTION app_user() RETURNS text
LANGUAGE sql STABLE AS
$$ SELECT nullif(current_setting('app.user_id', true), '') $$;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
  id         text PRIMARY KEY CHECK (id ~ '^[a-z0-9_-]+$'),
  name       text NOT NULL,
  -- Bumped on every document insert/update/delete. Part of the cache key, so a
  -- revocation invalidates every cached answer for the tenant immediately.
  acl_epoch  bigint NOT NULL DEFAULT 0
);

-- ACLs live on the DOCUMENT, not on each chunk: revoking a group is a
-- single-row UPDATE that takes effect for the next query, with no re-indexing.
CREATE TABLE IF NOT EXISTS documents (
  id          text PRIMARY KEY,
  tenant_id   text NOT NULL REFERENCES tenants(id),
  title       text NOT NULL,
  source      text NOT NULL,
  acl_groups  text[] NOT NULL CHECK (cardinality(acl_groups) > 0),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (id, tenant_id)
);
CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS documents_acl_idx ON documents USING gin (acl_groups);

CREATE TABLE IF NOT EXISTS chunks (
  id         bigserial PRIMARY KEY,
  doc_id     text NOT NULL,
  tenant_id  text NOT NULL,
  ord        int  NOT NULL,
  content    text NOT NULL,
  embedding  vector(384) NOT NULL,
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  -- Composite FK: a chunk can never claim a different tenant than its document.
  FOREIGN KEY (doc_id, tenant_id) REFERENCES documents (id, tenant_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
  USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- ACL epoch maintenance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bump_acl_epoch() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    UPDATE tenants SET acl_epoch = acl_epoch + 1 WHERE id = OLD.tenant_id;
  ELSE
    UPDATE tenants SET acl_epoch = acl_epoch + 1 WHERE id = NEW.tenant_id;
  END IF;
  RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS documents_acl_epoch ON documents;
CREATE TRIGGER documents_acl_epoch
  AFTER INSERT OR UPDATE OR DELETE ON documents
  FOR EACH ROW EXECUTE FUNCTION bump_acl_epoch();

-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------
-- Superusers and table owners bypass RLS by design, which is why the app must
-- never connect as the admin role. tests/test_rls.py asserts the app role's
-- attributes so a misconfigured deployment fails loudly.
ALTER TABLE tenants   ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_self ON tenants;
CREATE POLICY tenant_self ON tenants
  FOR SELECT TO rag_app
  USING (id = app_tenant());

DROP POLICY IF EXISTS document_read ON documents;
CREATE POLICY document_read ON documents
  FOR SELECT TO rag_app
  USING (tenant_id = app_tenant() AND acl_groups && app_groups());

DROP POLICY IF EXISTS chunk_read ON chunks;
CREATE POLICY chunk_read ON chunks
  FOR SELECT TO rag_app
  USING (
    tenant_id = app_tenant()
    AND EXISTS (
      SELECT 1 FROM documents d
      WHERE d.id = chunks.doc_id
        AND d.acl_groups && app_groups()
    )
  );

-- ---------------------------------------------------------------------------
-- Audit log: append-only, hash-chained, unreadable by the app
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
  id            bigserial PRIMARY KEY,
  ts            timestamptz NOT NULL,
  request_id    uuid   NOT NULL,
  user_id       text   NOT NULL,
  tenant_id     text   NOT NULL,
  groups        text[] NOT NULL,
  query_sha256  text   NOT NULL,
  query_text    text,              -- NULL unless AUDIT_STORE_QUERY=true
  chunk_ids     bigint[] NOT NULL,
  doc_ids       text[] NOT NULL,
  acl_epoch     bigint NOT NULL,
  cached        boolean NOT NULL,
  prev_hash     text NOT NULL,
  hash          text NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_tenant_ts_idx ON audit_log (tenant_id, ts);
CREATE INDEX IF NOT EXISTS audit_user_ts_idx ON audit_log (user_id, ts);

-- Canonical payload used for hashing. Timestamps go in as integer microseconds
-- so verification does not depend on the session's TimeZone setting.
CREATE OR REPLACE FUNCTION audit_payload(r audit_log) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
  SELECT jsonb_build_object(
    'id', r.id,
    'ts_us', (extract(epoch FROM r.ts) * 1000000)::bigint,
    'request_id', r.request_id,
    'user_id', r.user_id,
    'tenant_id', r.tenant_id,
    'groups', r.groups,
    'query_sha256', r.query_sha256,
    'query_text', r.query_text,
    'chunk_ids', r.chunk_ids,
    'doc_ids', r.doc_ids,
    'acl_epoch', r.acl_epoch,
    'cached', r.cached
  )::text
$$;

-- The ONLY way the app can write to the audit log. It runs with the owner's
-- privileges, takes identity from the same session settings RLS used (so the
-- log can't disagree with the filter), and doubles as a tripwire: if any chunk
-- id it is handed belongs to another tenant or to a document the caller cannot
-- read *right now*, it raises and the whole request transaction aborts before
-- anything reaches the LLM.
CREATE OR REPLACE FUNCTION append_audit(
  p_request_id   uuid,
  p_query_sha256 text,
  p_query_text   text,
  p_chunk_ids    bigint[],
  p_cached       boolean
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
  v_row     audit_log;
  v_tenant  text   := app_tenant();
  v_groups  text[] := app_groups();
  v_user    text   := app_user();
  v_missing int;
BEGIN
  IF v_tenant IS NULL OR v_user IS NULL THEN
    RAISE EXCEPTION 'append_audit: request context is not set';
  END IF;

  SELECT count(*) INTO v_missing
  FROM unnest(coalesce(p_chunk_ids, '{}'::bigint[])) AS want(id)
  LEFT JOIN chunks c ON c.id = want.id
  LEFT JOIN documents d ON d.id = c.doc_id
  WHERE c.id IS NULL
     OR c.tenant_id <> v_tenant
     OR NOT (d.acl_groups && v_groups);
  IF v_missing > 0 THEN
    RAISE EXCEPTION 'append_audit: % chunk(s) not readable by % in tenant %',
      v_missing, v_user, v_tenant
      USING ERRCODE = 'insufficient_privilege';
  END IF;

  -- Serialise appends so the chain has a single head.
  PERFORM pg_advisory_xact_lock(hashtext('secure_rag.audit_chain'));

  v_row.id           := nextval(pg_get_serial_sequence('audit_log', 'id'));
  v_row.ts           := date_trunc('microseconds', clock_timestamp());
  v_row.request_id   := p_request_id;
  v_row.user_id      := v_user;
  v_row.tenant_id    := v_tenant;
  v_row.groups       := v_groups;
  v_row.query_sha256 := p_query_sha256;
  v_row.query_text   := p_query_text;
  v_row.chunk_ids    := coalesce(p_chunk_ids, '{}'::bigint[]);
  SELECT coalesce(array_agg(DISTINCT c.doc_id ORDER BY c.doc_id), '{}'::text[])
    INTO v_row.doc_ids
    FROM chunks c WHERE c.id = ANY (v_row.chunk_ids);
  SELECT acl_epoch INTO v_row.acl_epoch FROM tenants WHERE id = v_tenant;
  v_row.cached       := p_cached;
  SELECT coalesce((SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1), repeat('0', 64))
    INTO v_row.prev_hash;
  v_row.hash := encode(sha256(convert_to(v_row.prev_hash || audit_payload(v_row), 'UTF8')), 'hex');

  INSERT INTO audit_log SELECT v_row.*;
  RETURN v_row.id;
END $$;

-- Walks the chain and reports the first row whose hash doesn't match.
CREATE OR REPLACE FUNCTION verify_audit_chain()
RETURNS TABLE (ok boolean, rows_checked bigint, first_bad_id bigint)
LANGUAGE plpgsql STABLE AS $$
DECLARE
  r        audit_log;
  v_prev   text := repeat('0', 64);
  v_n      bigint := 0;
BEGIN
  FOR r IN SELECT * FROM audit_log ORDER BY id LOOP
    v_n := v_n + 1;
    IF r.prev_hash <> v_prev
       OR r.hash <> encode(sha256(convert_to(r.prev_hash || audit_payload(r), 'UTF8')), 'hex') THEN
      RETURN QUERY SELECT false, v_n, r.id;
      RETURN;
    END IF;
    v_prev := r.hash;
  END LOOP;
  RETURN QUERY SELECT true, v_n, NULL::bigint;
END $$;

CREATE OR REPLACE FUNCTION audit_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'audit_log is append-only';
END $$;

DROP TRIGGER IF EXISTS audit_no_update ON audit_log;
CREATE TRIGGER audit_no_update BEFORE UPDATE OR DELETE ON audit_log
  FOR EACH ROW EXECUTE FUNCTION audit_immutable();
DROP TRIGGER IF EXISTS audit_no_truncate ON audit_log;
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log
  FOR EACH STATEMENT EXECUTE FUNCTION audit_immutable();

-- ---------------------------------------------------------------------------
-- Grants: read-only on the corpus, execute-only on the audit writer
-- ---------------------------------------------------------------------------
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM rag_app;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM rag_app;
GRANT USAGE ON SCHEMA public TO rag_app;
GRANT SELECT ON tenants, documents, chunks TO rag_app;

REVOKE ALL ON FUNCTION append_audit(uuid, text, text, bigint[], boolean) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION append_audit(uuid, text, text, bigint[], boolean) TO rag_app;
REVOKE ALL ON FUNCTION verify_audit_chain() FROM PUBLIC;
