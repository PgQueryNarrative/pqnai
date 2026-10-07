CREATE EXTENSION pqnai;

SELECT pqnai.version();

SELECT pqnai.enqueue_job('forecast', '{"series": [1, 2, 3], "horizon": 2}'::jsonb) > 0 AS job_enqueued;

SELECT job_type, status FROM pqnai.jobs ORDER BY id;

SELECT pqnai.wait_for_job(1, 1);

CALL pqnai.forecast(ARRAY[1, 2, 3]::double precision[], 2, 1);
