-- No worker is running during pg_regress, so embed/ask jobs can never
-- complete; this exercises validation and the enqueue/timeout path the
-- same way basic.sql does for forecast.
CALL pqnai.ask('what is in the documents?', 0);

SELECT pqnai.enqueue_job('embed', '{"text": "hello world"}'::jsonb) > 0 AS job_enqueued;

SELECT job_type, status FROM pqnai.jobs WHERE job_type = 'embed' ORDER BY id;

CALL pqnai.embed('hello world', 1);

CALL pqnai.ask('what is in the documents?', 3, 1);
