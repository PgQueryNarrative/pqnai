# pqnai

AI forecasting and retrieval for Postgres. A thin C extension exposes SQL functions; the
actual work (model inference, embeddings) runs in an external Go worker, `pqnaid`, so the
Postgres backend process never blocks on network calls or runs non-C code in-process.

Part of the [PgQueryNarrative](https://github.com/PgQueryNarrative) organization.

## Status

Early stage (v0.2). The job-queue plumbing, a moving-average forecaster, and a full
embed/retrieve/answer RAG pipeline (backed by `pgvector` and a self-hosted Ollama model,
not a third-party cloud AI API) work end-to-end. ARIMA/ETS forecasting is still in
progress — see [docs/architecture.md](docs/architecture.md) and the roadmap below.

## Quickstart

```sh
docker compose up --build
```

This builds and starts Postgres (with the `pqnai` and `vector` extensions installed), a
self-hosted Ollama instance (pulling a small embedding model and a small chat model on
first run), and the `pqnaid` worker. Then, from another terminal:

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

```sh
docker compose exec postgres psql -U postgres -d pqnai \
  -c "CALL pqnai.embed('The pqnaid worker listens for jobs using LISTEN and NOTIFY.');"
docker compose exec postgres psql -U postgres -d pqnai \
  -c "CALL pqnai.ask('What does the pqnaid worker use to listen for jobs?', 1);"
```

```
                                       result
-------------------------------------------------------------------------------------
 {"answer": "The pqnaid worker listens for jobs using LISTEN and NOTIFY.", "sources":
  ["The pqnaid worker listens for jobs using LISTEN and NOTIFY."]}
(1 row)
```

## How it works

```sql
CALL pqnai.forecast(series, horizon);
CALL pqnai.embed(text);
CALL pqnai.ask(question, top_k);
```

Each enqueues a row in `pqnai.jobs`, notifies the worker via `pg_notify`, and waits for
the result. The worker (`pqnaid`) is a separate process that `LISTEN`s for jobs and does
the actual work: `forecast` runs the forecasting model; `embed` calls Ollama for an
embedding and stores it in `pqnai.documents` (a `pgvector` column); `ask` embeds the
question, retrieves the closest stored documents by cosine distance, and asks Ollama to
answer using only that retrieved context. Results are written back for the SQL call to
read. See [docs/architecture.md](docs/architecture.md) for why this is a job queue
rather than a direct in-process call.

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
- [x] RAG: `pgvector`-backed embeddings, `pqnai.embed()` / `pqnai.ask()`, self-hosted Ollama
- [ ] Forecasting: ARIMA / Holt-Winters (ETS)
- [ ] RAG: chunking for long documents, hybrid/reranked retrieval
- [ ] Packaging: PGXN, Docker image releases, prebuilt binaries
- [ ] Deep-learning forecasting via ONNX Runtime (models trained offline; no Python at
      runtime)
- [ ] Pluggable embedding/chat providers beyond Ollama (OpenAI, Anthropic, etc.), opt-in

## License

[PostgreSQL License](LICENSE).

## Origin

Rebuilt from [PGAI](https://github.com/Postgres-artificialintelligence/PGAI) — see
[docs/ORIGIN.md](docs/ORIGIN.md) for attribution.
