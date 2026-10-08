\echo Use "ALTER EXTENSION pqnai UPDATE TO '0.4.0'" to load this file. \quit

-- Hybrid retrieval: ask() now fuses vector search with full-text search
-- (chunks.tsv, added in 0.3.0) and can be scoped by source metadata.
-- jsonb_path_ops is smaller and faster than the default jsonb_ops for
-- containment-only (@>) lookups, which is the only way filters query it.
CREATE INDEX sources_metadata_idx ON sources USING gin (metadata jsonb_path_ops);

-- ask() gains a filters parameter, which changes its signature, so it
-- must be dropped and recreated rather than CREATE OR REPLACEd. filters is
-- added after p_timeout_seconds so existing positional calls such as
-- ask(question, top_k) and ask(question, top_k, timeout) are unaffected.
ALTER EXTENSION pqnai DROP PROCEDURE ask(text, integer, numeric, jsonb);
DROP PROCEDURE ask(text, integer, numeric, jsonb);

-- See forecast()'s comment in pqnai--0.4.0.sql for why this is a
-- PROCEDURE with no SET clause.
CREATE PROCEDURE ask(IN p_question text, IN p_top_k int DEFAULT 3, IN p_timeout_seconds numeric DEFAULT 60, IN p_filters jsonb DEFAULT '{}'::jsonb, INOUT result jsonb DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id bigint;
BEGIN
    IF p_top_k IS NULL OR p_top_k <= 0 THEN
        RAISE EXCEPTION 'pqnai: top_k must be a positive integer';
    END IF;
    -- An array or scalar is valid jsonb but would silently match no
    -- source via @>; reject it here rather than after a worker round-trip.
    IF p_filters IS NULL OR jsonb_typeof(p_filters) <> 'object' THEN
        RAISE EXCEPTION 'pqnai: filters must be a JSON object, e.g. ''{"tenant_id": "acme"}''';
    END IF;

    v_job_id := pqnai.enqueue_job('ask', jsonb_build_object('question', p_question, 'top_k', p_top_k, 'filters', p_filters));
    COMMIT;

    result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
END;
$$;
