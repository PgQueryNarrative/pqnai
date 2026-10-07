# Architecture

pqnai has two independently-releasable components:

- **`extension/`** (C) — a thin, in-process Postgres extension. It owns the `pqnai.jobs`
  table and a small set of SQL-callable functions/procedures (`pqnai.enqueue_job`,
  `pqnai.wait_for_job`, `pqnai.forecast`). It never performs network I/O and never links
  Go code into its shared library: Go's runtime installs its own signal handlers via cgo,
  which conflicts with Postgres's signal handling and is a documented source of backend
  crashes (see golang.org/issue/18328, golang.org/issue/21905). Keeping Go out-of-process
  is a hard requirement, not a style choice.

- **`cmd/pqnaid/`** (Go) — an out-of-process worker. It connects to Postgres over `pgx`,
  `LISTEN`s on the `pqnai_jobs` channel, and on each notification (plus a periodic poll as
  a safety net against missed notifications) claims queued jobs with
  `UPDATE ... FOR UPDATE SKIP LOCKED` and executes them: `internal/forecast` for
  statistical forecasting, with `internal/rag` planned for embeddings/RAG. Results are
  written back to `pqnai.jobs`.

## Why a job queue instead of a direct call

A Postgres backend process cannot safely block on an external network call or model
inference without risking connection pool exhaustion and holding locks for unpredictable
periods. Instead, SQL functions enqueue a row and return; the worker (a separate process,
separate connection) picks it up asynchronously. This is the same "outbox" pattern used by
job-queue libraries like River and Que, applied to Postgres-as-a-platform rather than
Postgres-as-a-queued-client.

`pqnai.forecast()` is a convenience **procedure**, not a function, specifically so it can
synchronously enqueue-and-wait within a single `CALL` for demos and simple scripts: plain
PL/pgSQL functions run inside the caller's transaction and can't `COMMIT`, so the enqueued
row would stay invisible to the worker's connection until the function itself returned —
a deadlock. Procedures can `COMMIT` mid-execution (Postgres 11+), which is what makes the
single-call convenience possible. Note that a procedure with a `SET` clause (e.g.
`SET search_path`) cannot `COMMIT`/`ROLLBACK` internally, which is why `forecast()`
schema-qualifies its calls instead of relying on `search_path`.

For production workloads, call `pqnai.enqueue_job()` and `pqnai.wait_for_job()` (or just
poll `pqnai.jobs`) as separate statements from the application side instead.

## Versioning

SQL changes ship as proper `ALTER EXTENSION` upgrade paths
(`pqnai--X.Y.Z--X.Y.Z+1.sql`), not flat re-dumps of the whole schema.
