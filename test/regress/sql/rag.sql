-- No worker is running during pg_regress, so embed/ask jobs can never
-- complete; this exercises validation and the enqueue/timeout path the
-- same way basic.sql does for forecast.
CALL pqnai.ask('what is in the documents?', 0);

SELECT pqnai.enqueue_job('embed', '{"text": "hello world"}'::jsonb) > 0 AS job_enqueued;

SELECT job_type, status FROM pqnai.jobs WHERE job_type = 'embed' ORDER BY id;

CALL pqnai.embed('hello world', p_timeout_seconds => 1);

CALL pqnai.ask('what is in the documents?', 3, 1);

-- filters must be a JSON object: an array or scalar is valid jsonb but
-- would silently match no source, so it's rejected before enqueuing.
CALL pqnai.ask('what is in the documents?', 3, 1, '["not", "an", "object"]'::jsonb);
CALL pqnai.ask('what is in the documents?', 3, 1, '"acme"'::jsonb);
CALL pqnai.ask('what is in the documents?', 3, 1, NULL);

-- A valid filter reaches the worker via the job payload; re-ranking is
-- off unless asked for.
CALL pqnai.ask('what is in the documents?', 3, 1, '{"tenant_id": "acme"}'::jsonb);
SELECT payload -> 'filters' AS filters, payload -> 'rerank' AS rerank
FROM pqnai.jobs WHERE job_type = 'ask' ORDER BY id DESC LIMIT 1;

-- p_rerank reaches the worker; an explicit NULL means off.
CALL pqnai.ask('what is in the documents?', 3, 1, p_rerank => true);
SELECT payload -> 'rerank' AS rerank FROM pqnai.jobs WHERE job_type = 'ask' ORDER BY id DESC LIMIT 1;
CALL pqnai.ask('what is in the documents?', 3, 1, p_rerank => NULL);
SELECT payload -> 'rerank' AS rerank FROM pqnai.jobs WHERE job_type = 'ask' ORDER BY id DESC LIMIT 1;

-- Only the four valid ask() calls enqueued jobs; the rejected ones did not.
SELECT count(*) AS ask_jobs FROM pqnai.jobs WHERE job_type = 'ask';
