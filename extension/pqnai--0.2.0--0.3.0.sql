\echo Use "ALTER EXTENSION pqnai UPDATE TO '0.3.0'" to load this file. \quit

-- Phase A of docs/rag-advanced-plan.md: split the flat `documents` table
-- into `sources` (one row per embedded document) and `chunks` (one row
-- per chunk of that document), so long documents can be split before
-- embedding instead of embedding the whole thing as a single vector.
ALTER EXTENSION pqnai DROP TABLE documents;
DROP TABLE documents;

CREATE TABLE sources (
    id         bigserial PRIMARY KEY,
    title      text,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- tsv/GIN exist from Phase A onward (schema redesign is a one-time
-- change) but aren't queried until hybrid search ships in Phase B.
CREATE TABLE chunks (
    id          bigserial PRIMARY KEY,
    source_id   bigint NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    content     text NOT NULL,
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_source_id_idx ON chunks (source_id);

-- embed() gains title/metadata parameters and now returns a source_id
-- (one source may produce many chunks), which changes its signature --
-- CREATE OR REPLACE can't do that, so drop and recreate.
ALTER EXTENSION pqnai DROP PROCEDURE embed(text, numeric, bigint);
DROP PROCEDURE embed(text, numeric, bigint);

-- See forecast()'s comment in pqnai--0.2.0.sql for why this is a
-- PROCEDURE with no SET clause.
CREATE PROCEDURE embed(IN p_text text, IN p_title text DEFAULT NULL, IN p_metadata jsonb DEFAULT '{}'::jsonb, IN p_timeout_seconds numeric DEFAULT 30, INOUT source_id bigint DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id bigint;
    v_result jsonb;
BEGIN
    v_job_id := pqnai.enqueue_job('embed', jsonb_build_object('text', p_text, 'title', p_title, 'metadata', p_metadata));
    COMMIT;

    v_result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
    source_id := (v_result ->> 'source_id')::bigint;
END;
$$;
