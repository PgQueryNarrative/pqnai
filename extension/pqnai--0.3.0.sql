\echo Use "CREATE EXTENSION pqnai" to load this file. \quit

CREATE TABLE jobs (
    id          bigserial PRIMARY KEY,
    job_type    text NOT NULL CHECK (job_type IN ('forecast', 'embed', 'ask')),
    status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    payload     jsonb NOT NULL,
    result      jsonb,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX jobs_queued_idx ON jobs (id) WHERE status = 'queued';

-- One row per embedded document. Long documents are split into chunks
-- (below) before embedding rather than embedded as a single vector --
-- see docs/rag-advanced-plan.md Phase A.
CREATE TABLE sources (
    id         bigserial PRIMARY KEY,
    title      text,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 384 dimensions matches the all-minilm embedding model pqnaid uses by
-- default (see internal/rag). A different embedding model with a
-- different output size would need its own migration.
--
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

CREATE FUNCTION version()
RETURNS text
AS 'MODULE_PATHNAME', 'pqnai_version'
LANGUAGE C STABLE;

CREATE FUNCTION enqueue_job(p_job_type text, p_payload jsonb)
RETURNS bigint
LANGUAGE plpgsql
SET search_path = pqnai, pg_catalog
AS $$
DECLARE
    v_id bigint;
BEGIN
    INSERT INTO jobs (job_type, payload)
    VALUES (p_job_type, p_payload)
    RETURNING id INTO v_id;

    PERFORM pg_notify('pqnai_jobs', v_id::text);

    RETURN v_id;
END;
$$;

CREATE FUNCTION wait_for_job(p_job_id bigint, p_timeout_seconds numeric DEFAULT 30)
RETURNS jsonb
LANGUAGE plpgsql
SET search_path = pqnai, pg_catalog
AS $$
DECLARE
    v_status  text;
    v_result  jsonb;
    v_error   text;
    v_elapsed numeric := 0;
    v_step    numeric := 0.1;
BEGIN
    LOOP
        SELECT status, result, error INTO v_status, v_result, v_error
        FROM jobs WHERE id = p_job_id;

        IF v_status IS NULL THEN
            RAISE EXCEPTION 'pqnai: job % not found', p_job_id;
        ELSIF v_status = 'done' THEN
            RETURN v_result;
        ELSIF v_status = 'failed' THEN
            RAISE EXCEPTION 'pqnai: job % failed: %', p_job_id, v_error;
        END IF;

        IF v_elapsed >= p_timeout_seconds THEN
            RAISE EXCEPTION 'pqnai: job % timed out after % seconds', p_job_id, p_timeout_seconds;
        END IF;

        PERFORM pg_sleep(v_step);
        v_elapsed := v_elapsed + v_step;
    END LOOP;
END;
$$;

-- forecast/embed/ask are PROCEDUREs, not functions: each must COMMIT after
-- enqueuing so the job row becomes visible to the pqnaid worker's own
-- connection before this session starts polling for the result. Plain
-- functions run inside the caller's transaction and cannot COMMIT, which
-- would make the enqueued row invisible to the worker until the function
-- itself returned -- a deadlock. Call with CALL, e.g.:
--   CALL pqnai.forecast(series, horizon[, timeout]);
--
-- No SET search_path on any of them: Postgres disallows internal
-- COMMIT/ROLLBACK in a procedure that carries a SET clause, so calls
-- below are schema-qualified instead.
CREATE PROCEDURE forecast(IN p_series double precision[], IN p_horizon int, IN p_timeout_seconds numeric DEFAULT 30, INOUT result jsonb DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id bigint;
BEGIN
    IF p_horizon IS NULL OR p_horizon <= 0 THEN
        RAISE EXCEPTION 'pqnai: horizon must be a positive integer';
    END IF;

    v_job_id := pqnai.enqueue_job('forecast', jsonb_build_object('series', p_series, 'horizon', p_horizon));
    COMMIT;

    result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
END;
$$;

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
