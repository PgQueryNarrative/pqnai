# pqnai

AI forecasting and retrieval for Postgres. A thin C extension exposes SQL functions; the
actual work (model inference, embeddings) runs in an external Go worker, `pqnaid`, so the
Postgres backend process never blocks on network calls or runs non-C code in-process.

Part of the [PgQueryNarrative](https://github.com/PgQueryNarrative) organization.

## Status

Early stage (v0.5). The job-queue plumbing, a moving-average forecaster, and a full
embed/retrieve/answer RAG pipeline work end-to-end: chunked storage, hybrid retrieval
(vector + full-text, fused with Reciprocal Rank Fusion), metadata filtering, and optional
model-based re-ranking, backed by `pgvector` and a self-hosted Ollama model rather than a
third-party cloud AI API. ARIMA/ETS forecasting and answer groundedness checks are still
in progress — see [docs/architecture.md](docs/architecture.md) and the roadmap below.

## Quickstart

```sh
docker compose up --build
```

This builds and starts Postgres (with the `pqnai` and `vector` extensions installed), a
self-hosted Ollama instance (pulling an embedding model, a small chat model, and a 3B
re-ranking model — about 2.4 GB in total — on first run), and the `pqnaid` worker. Then, from another terminal:

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
CALL pqnai.embed(text, p_metadata => '{"tenant_id": "acme"}');
CALL pqnai.ask(question, top_k, p_filters => '{"tenant_id": "acme"}', p_rerank => true);
```

`p_metadata` / `p_filters` are optional; filters scope retrieval to sources whose
metadata contains them. **Filters are not a security boundary** — the caller chooses
them, so they can't isolate tenants from each other; see
[docs/architecture.md](docs/architecture.md). `p_rerank` (default off) re-scores the top
candidates with a model before picking `top_k`, at the cost of one extra model call; if
it fails, `ask()` still answers from the un-reranked order and reports
`"reranked": false` with the reason.

The worker is configured with environment variables: `DATABASE_URL`, `OLLAMA_URL`,
`OLLAMA_EMBED_MODEL` (default `all-minilm`), `OLLAMA_CHAT_MODEL` (default
`qwen2.5:0.5b`), `OLLAMA_RERANK_MODEL` (default: the chat model; Docker Compose sets
`qwen2.5:3b` — re-ranking needs a model of roughly that size, see
[docs/architecture.md](docs/architecture.md#evaluation)), and `OLLAMA_NUM_CTX` (default
16384 — prompts longer than this fail with an error rather than being silently
truncated).

Each enqueues a row in `pqnai.jobs`, notifies the worker via `pg_notify`, and waits for
the result. The worker (`pqnaid`) is a separate process that `LISTEN`s for jobs and does
the actual work: `forecast` runs the forecasting model; `embed` splits long text into
overlapping chunks, calls Ollama for an embedding per chunk, and stores them in
`pqnai.chunks` (a `pgvector` column); `ask` retrieves chunks by both vector similarity
and full-text match, fuses the two rankings, and asks Ollama to answer using only that
retrieved context. Results are written back for the SQL call to read. See
[docs/architecture.md](docs/architecture.md) for why this is a job queue
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
go test ./...                                             # unit tests (CI)
go test -tags integration ./test/integration/...          # real Postgres + pgvector via Docker
OLLAMA_URL=http://localhost:11434 go test -tags eval -v ./test/eval/...   # real-model quality gate
```

## Roadmap

- [x] Job-queue plumbing: `pqnai.enqueue_job`, `pqnai.wait_for_job`, `LISTEN`/`NOTIFY`
- [x] Forecasting: moving-average baseline
- [x] RAG: `pgvector`-backed embeddings, `pqnai.embed()` / `pqnai.ask()`, self-hosted Ollama
- [ ] Forecasting: ARIMA / Holt-Winters (ETS)
- [x] RAG: chunking (Phase A) — splits long documents into overlapping chunks before embedding
- [x] RAG: hybrid retrieval (vector + full-text via Reciprocal Rank Fusion) and metadata filters
- [x] RAG: optional model-based re-ranking
- [ ] RAG: groundedness guardrails, access control
- [ ] Packaging: PGXN, Docker image releases, prebuilt binaries
- [ ] Deep-learning forecasting via ONNX Runtime (models trained offline; no Python at
      runtime)
- [ ] Pluggable embedding/chat providers beyond Ollama (OpenAI, Anthropic, etc.), opt-in

## License

[PostgreSQL License](LICENSE).

## Origin

Rebuilt from [PGAI](https://github.com/Postgres-artificialintelligence/PGAI) — see
[docs/ORIGIN.md](docs/ORIGIN.md) for attribution.
