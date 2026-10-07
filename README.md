# pqnai

AI forecasting and retrieval for Postgres. A thin C extension exposes SQL functions; the
actual work (model inference, embeddings) runs in an external Go worker, `pqnaid`, so the
Postgres backend process never blocks on network calls or runs non-C code in-process.

Part of the [PgQueryNarrative](https://github.com/PgQueryNarrative) organization.

## Status

Early stage (v0.1). The job-queue plumbing and a simple moving-average forecaster work
end-to-end. ARIMA/ETS forecasting and embeddings/RAG are in progress — see
[docs/architecture.md](docs/architecture.md) and the roadmap below.

## Quickstart

```sh
docker compose up --build
```

This builds and starts Postgres (with the `pqnai` extension installed) and the `pqnaid`
worker. Then, from another terminal:

```sh
docker compose exec postgres psql -U postgres -d pqnai \
  -c "CALL pqnai.forecast(ARRAY[10,20,30,40,50]::double precision[], 3);"
```

```
                          result
----------------------------------------------------------
 {"method": "moving_average", "forecast": [30, 34, 36.8]}
(1 row)
```

## How it works

```sql
CALL pqnai.forecast(series, horizon);
```

enqueues a row in `pqnai.jobs`, notifies the worker via `pg_notify`, and waits for the
result. The worker (`pqnaid`) is a separate process that `LISTEN`s for jobs, runs the
forecasting model, and writes the result back. See [docs/architecture.md](docs/architecture.md)
for why this is a job queue rather than a direct in-process call.

## Building from source

Requires a Postgres installation with `pg_config` on `PATH`, and a C compiler.

```sh
cd extension
make
make install
make installcheck   # runs the regression suite against a local Postgres
```

For the worker:

```sh
go build ./...
go test ./...
```

## Roadmap

- [x] Job-queue plumbing: `pqnai.enqueue_job`, `pqnai.wait_for_job`, `LISTEN`/`NOTIFY`
- [x] Forecasting: moving-average baseline
- [ ] Forecasting: ARIMA / Holt-Winters (ETS)
- [ ] RAG: `pgvector`-backed embeddings, chunking, `pqnai.embed()` / `pqnai.ask()`
- [ ] Packaging: PGXN, Docker image releases, prebuilt binaries
- [ ] Deep-learning forecasting via ONNX Runtime (models trained offline; no Python at
      runtime)

## License

[PostgreSQL License](LICENSE).

## Origin

Rebuilt from [PGAI](https://github.com/Postgres-artificialintelligence/PGAI) — see
[docs/ORIGIN.md](docs/ORIGIN.md) for attribution.
