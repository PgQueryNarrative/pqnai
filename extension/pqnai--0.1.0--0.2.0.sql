\echo Use "ALTER EXTENSION pqnai UPDATE TO '0.2.0'" to load this file. \quit

ALTER TABLE jobs DROP CONSTRAINT jobs_job_type_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_job_type_check CHECK (job_type IN ('forecast', 'embed', 'ask'));

-- version() now looks up pg_extension at call time instead of returning a
-- hardcoded string (see pqnai.c), so it's no longer IMMUTABLE. CREATE OR
-- REPLACE (not DROP+CREATE): an extension member function can't be
-- dropped directly -- Postgres ties it to the extension and refuses.
CREATE OR REPLACE FUNCTION version()
RETURNS text
AS 'MODULE_PATHNAME', 'pqnai_version'
LANGUAGE C STABLE;

-- 384 dimensions matches the all-minilm embedding model pqnaid uses by
-- default (see internal/rag). A different embedding model with a
-- different output size would need its own migration.
CREATE TABLE documents (
    id         bigserial PRIMARY KEY,
    content    text NOT NULL,
    embedding  vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);

CREATE PROCEDURE embed(IN p_text text, IN p_timeout_seconds numeric DEFAULT 30, INOUT document_id bigint DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id bigint;
    v_result jsonb;
BEGIN
    v_job_id := pqnai.enqueue_job('embed', jsonb_build_object('text', p_text));
    COMMIT;

    v_result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
    document_id := (v_result ->> 'document_id')::bigint;
END;
$$;

CREATE PROCEDURE ask(IN p_question text, IN p_top_k int DEFAULT 3, IN p_timeout_seconds numeric DEFAULT 60, INOUT result jsonb DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id bigint;
BEGIN
    IF p_top_k IS NULL OR p_top_k <= 0 THEN
        RAISE EXCEPTION 'pqnai: top_k must be a positive integer';
    END IF;

    v_job_id := pqnai.enqueue_job('ask', jsonb_build_object('question', p_question, 'top_k', p_top_k));
    COMMIT;

    result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
END;
$$;
