# Architecture

pqnai has two independently-releasable components:

- **`extension/`** (C) — a thin, in-process Postgres extension. It owns the `pqnai.jobs`,
  `pqnai.sources`, and `pqnai.chunks` tables (the latter a `pgvector` column) and a small
  set of SQL-callable functions/procedures (`pqnai.enqueue_job`, `pqnai.wait_for_job`,
  `pqnai.forecast`, `pqnai.embed`, `pqnai.ask`). It never performs network I/O and never
  links Go code into its shared library: Go's runtime installs its own signal handlers via
  cgo, which conflicts with Postgres's signal handling and is a documented source of
  backend crashes (see golang.org/issue/18328, golang.org/issue/21905). Keeping Go
  out-of-process is a hard requirement, not a style choice.

- **`cmd/pqnaid/`** (Go) — an out-of-process worker. It connects to Postgres over `pgx`,
  `LISTEN`s on the `pqnai_jobs` channel, and on each notification (plus a periodic poll as
  a safety net against missed notifications) claims queued jobs with
  `UPDATE ... FOR UPDATE SKIP LOCKED` and executes them: `internal/forecast` for
  statistical forecasting, `internal/rag` for embeddings and retrieval-augmented answers.
  Results are written back to `pqnai.jobs`.

## RAG: embed and ask

`internal/rag` talks to a self-hosted [Ollama](https://ollama.com) instance over HTTP
rather than a third-party cloud AI API — no API key required, and the whole pipeline is
runnable and verifiable offline/self-hosted. Two small models are pulled on first run
(see `docker-compose.yml`'s `ollama-init` service): `all-minilm` (384-dim embeddings) and
`qwen2.5:0.5b` (chat/generation). Swapping in a different embedding model requires
updating the `vector(384)` column width in a new migration to match its output size.

- `pqnai.embed(text, title?, metadata?)` → splits `text` into overlapping chunks
  (`internal/rag.Chunk`, preferring paragraph/sentence/word boundaries over mid-word
  cuts), embeds each chunk via Ollama, and stores one `pqnai.sources` row plus one
  `pqnai.chunks` row per chunk, all in a single transaction (so a failure partway through
  never leaves a source with only some of its chunks). Returns the new source's id.
  Embedding itself happens before the transaction opens, since it's a slow network call
  per chunk and a Postgres transaction shouldn't sit open across that.
- `pqnai.ask(question, top_k, timeout?, filters?)` → worker retrieves candidates two ways
  and fuses them (see *Hybrid retrieval* below), keeps the top `top_k`, and asks Ollama
  to answer using only that retrieved context. Returns `{answer, sources}`. `filters` is
  an optional JSON object matched against `pqnai.sources.metadata` with jsonb containment
  (`@>`), e.g. `'{"tenant_id": "acme"}'`, so retrieval can be scoped per tenant/tag.

Chunking exists because embedding a whole long document as a single vector makes
retrieval imprecise (the vector is an average of everything the document talks about,
diluting any specific passage a question might be about). `pqnai.sources` / `pqnai.chunks`
is a one-time schema split done in version 0.3.0 for this reason; `chunks.tsv` (a
generated `tsvector`) was added at the same time to back hybrid retrieval in 0.4.0.

### Hybrid retrieval

Pure vector search blurs exact terms — names, codes, acronyms — together with
semantically similar words; pure keyword search misses paraphrases. `ask()` runs both
(`internal/rag/hybrid.go`):

1. **Vector**: top 20 chunks by cosine distance (`chunks.embedding`, HNSW index).
2. **Keyword**: top 20 chunks matching `plainto_tsquery(question)` against `chunks.tsv`,
   ranked by `ts_rank` (GIN index).
3. **Reciprocal Rank Fusion**: `score = Σ 1/(60 + rank)` over each list a chunk appears
   in. RRF uses only rank position, so cosine distance and `ts_rank` never need to be
   normalized against each other; a chunk ranked well by *both* beats one ranked first by
   only one. Ties break by first-seen order, so results are deterministic.

A question made only of stopwords yields an empty `tsquery`; keyword search then
contributes nothing and retrieval falls back to the vector half.

Two accuracy safeguards worth knowing about:

- **Filtered HNSW searches use `hnsw.iterative_scan = strict_order`** (requires pgvector
  ≥ 0.8.0). By default an HNSW index scan yields only `ef_search` (40) nearest chunks
  *overall* and the metadata filter runs afterwards — so a tenant whose chunks all fall
  outside the global top 40 would silently get **zero** results despite having matching
  chunks. Iterative scan keeps walking the index until enough rows pass the filter.
  `strict_order` rather than `relaxed_order` because rank position feeds RRF directly.
- **No retrieval, no model call.** If nothing is retrieved (e.g. a filter matching no
  sources), `ask()` returns `"No relevant context was found for this question."` without
  calling the model at all — given an empty context, a model tends to answer from its own
  training data anyway, producing an ungrounded answer that looks like a retrieval result.

### Testing

`internal/rag`'s `Embed`/`Ask` functions take an `Embedder`/`Generator` interface rather
than a concrete Ollama type, so tests can substitute fakes:

- **Unit** (`internal/rag/*_test.go`, every commit, CI): chunk boundaries, RRF ordering
  (including the exact formula and determinism), filter validation, error propagation.
- **Integration** (`test/integration/rag_test.go`, `testcontainers-go`, real
  Postgres+pgvector, fake model): storage, chunking, hybrid keyword rescue, filter
  isolation, stopword fallback, and the filtered-HNSW iterative-scan fix. Each hybrid and
  HNSW test first runs a *control* proving the failure it guards against actually
  reproduces in its fixture (e.g. that vector-only search misses the target, or that
  `iterative_scan = off` under-returns), and the HNSW test asserts via `EXPLAIN` that the
  index is really used — so none of them can pass vacuously.
- **Manual** (`docker compose up`, real Ollama models, before each release): real
  embedding/generation quality, which is too slow and non-deterministic for CI.

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
