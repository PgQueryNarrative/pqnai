\echo Use "ALTER EXTENSION pqnai UPDATE TO '0.5.0'" to load this file. \quit

-- ask() gains p_rerank, which changes its signature, so it must be
-- dropped and recreated rather than CREATE OR REPLACEd. p_rerank is added
-- last so every existing positional call is unaffected.
ALTER EXTENSION pqnai DROP PROCEDURE ask(text, integer, numeric, jsonb, jsonb);
DROP PROCEDURE ask(text, integer, numeric, jsonb, jsonb);

-- See forecast()'s comment in pqnai--0.5.0.sql for why this is a
-- PROCEDURE with no SET clause.
CREATE PROCEDURE ask(IN p_question text, IN p_top_k int DEFAULT 3, IN p_timeout_seconds numeric DEFAULT 60, IN p_filters jsonb DEFAULT '{}'::jsonb, IN p_rerank boolean DEFAULT false, INOUT result jsonb DEFAULT NULL)
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

    v_job_id := pqnai.enqueue_job('ask', jsonb_build_object(
        'question', p_question,
        'top_k', p_top_k,
        'filters', p_filters,
        'rerank', coalesce(p_rerank, false)
    ));
    COMMIT;

    result := pqnai.wait_for_job(v_job_id, p_timeout_seconds);
END;
$$;
