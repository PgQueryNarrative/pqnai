# Advanced RAG: implementation plan

## Context

`pqnai.embed()` / `pqnai.ask()` currently implement the simplest possible RAG loop: embed
whole documents, retrieve top-k by cosine distance, stuff raw text into a prompt, generate.
It works (verified end-to-end with real models), but it's demo-grade: no chunking, no
keyword search, no re-ranking, no guardrails, no access control beyond "anyone who can
`CALL` the procedure." This plan takes it to a defensible production baseline, in six
shippable phases, each with its own schema migration, tests, and ship criteria.

**Principles carried over from the existing codebase** (don't relitigate these):
- Go stays out-of-process; the C extension stays a thin job-queue frontend.
- Every schema change ships as a proper `ALTER EXTENSION` upgrade path, not a rewrite of a
  released version file.
- Unit tests use fakes (fast, deterministic, every commit); `testcontainers-go` integration
  tests exercise real Postgres+pgvector SQL with a fake LLM (deterministic, CI-safe); real
  Ollama models are verified manually before each release, not in CI (too slow/flaky).

## Schema redesign (prerequisite for everything below)

Chunking requires splitting "one embedding per document" into "one embedding per chunk,
many chunks per document." This is a breaking schema change, done once, in Phase A:

```
pqnai.sources                       pqnai.chunks
  id          bigserial PK            id            bigserial PK
  title       text                    source_id     bigint REFERENCES sources(id)
  metadata    jsonb   DEFAULT '{}'    chunk_index   int
  created_at  timestamptz             content       text
                                      embedding     vector(384)
                                      tsv           tsvector GENERATED ALWAYS AS
                                                      (to_tsvector('english', content)) STORED
                                      created_at    timestamptz
```

- `metadata jsonb` on `sources` carries tenant/owner/tags for filtering (Phase B) and is the
  attachment point for row-level security (Phase E) if/when multi-tenancy is needed.
- `tsv` + a GIN index is the full-text half of hybrid search (Phase B).
- The existing flat `pqnai.documents` table is dropped in this migration; `pqnai--0.1.0.sql`
  and `pqnai--0.2.0.sql` stay untouched (released, frozen) per the versioning convention —
  this is a new `pqnai--0.2.0--0.3.0.sql` plus a full `pqnai--0.3.0.sql`.

## Phase A — Chunking

- `internal/rag/chunk.go`: fixed-size chunking with overlap (e.g. 500 chars / 50 overlap,
  both configurable), splitting on sentence/paragraph boundaries where possible rather than
  mid-word. Pure function, no I/O — easy to unit test exhaustively (empty input, input
  shorter than one chunk, exact multiples, overlap correctness, unicode safety).
- `pqnai.embed(text, title?, metadata?)`: inserts one `sources` row, chunks `text`, embeds
  and inserts each chunk. Returns `source_id`.
- **Tests**: unit tests for every chunk-boundary case above. Regression test confirming the
  existing single-short-document behavior is unchanged (one chunk in, one chunk out).
- **Ship criteria**: `make installcheck` passes on the new schema; existing embed/ask demo
  from the README still works unchanged from the caller's point of view.

## Phase B — Hybrid search (vector + keyword, via Reciprocal Rank Fusion)

Pure vector search misses exact terms (names, codes, acronyms) that keyword search catches,
and vice versa for paraphrased questions. Combine both rather than picking one:

1. Run a vector query (`embedding <=> $1`, existing index) → ranked list A.
2. Run a full-text query (`tsv @@ plainto_tsquery($1)`, ranked by `ts_rank`) → ranked list B.
3. Combine with **Reciprocal Rank Fusion**: `score(doc) = Σ 1/(k + rank_in_list)` for each
   list the doc appears in (k=60 is the standard RRF constant). No need to normalize cosine
   distance against `ts_rank` — RRF only uses rank position, which sidesteps that entirely.

- `pqnai.ask(question, top_k, filters?)`: `filters` is an optional `jsonb` containment
  query against `sources.metadata` (e.g. `{"tenant_id": "acme"}`), applied to both
  candidate lists before fusion — this is also the first access-control primitive (Phase E
  tightens it further).
- **Tests**: the accuracy deliverable for this phase is a fixture integration test with a
  deterministic fake embedder: construct a case where the question shares no vocabulary
  with the right document's embedding-neighbors but does share an exact keyword, assert
  hybrid retrieval surfaces it and pure-vector retrieval (run side by side in the same
  test) does not. This is the regression guard against ever silently reverting to
  vector-only.
- **Ship criteria**: the above fixture test passes; existing `TestRAGRoundTrip` still
  passes unmodified (hybrid must be a strict improvement, not a behavior change for the
  cases it already handled).

## Phase C — Re-ranking

Vector/keyword retrieval is cheap but approximate; a second, more expensive relevance pass
on a small candidate set measurably improves precision. Given Ollama's model catalog
doesn't include a first-class cross-encoder reranker, the pragmatic default is
**LLM-as-reranker**: take the top ~20 RRF candidates, ask the chat model to score each
0–10 for relevance to the question in a single batched prompt, re-sort, keep top-k. This is
slower than a dedicated cross-encoder and is the one component in this plan most likely to
be replaced later if a suitable small reranker model becomes available in Ollama — flagged
as a known trade-off, not a blind spot.

- Config flag (`rerank: bool`, default off) since it adds a model round-trip; off by
  default keeps the Phase A/B latency profile for callers who don't need it.
- **Tests**: unit test with a fake `Generator` returning fixed scores, asserting the
  re-ranker reorders candidates to match those scores exactly (pure logic, no model
  flakiness in the test).
- **Ship criteria**: unit test passes; one real-model before/after example captured in the
  PR description showing a concrete ranking improvement (manual verification, same spirit
  as the embed/ask demo already captured in README).

## Phase D — Groundedness guardrail + citations

An answer that isn't actually supported by the retrieved context is worse than no answer —
it looks authoritative. Add a cheap self-check pass rather than trusting the first
generation:

1. Prompt the chat model to answer **with inline citation markers** (`[1]`, `[2]`, ...)
   referencing the numbered sources list, instead of free-floating prose.
2. Second, small, model call: "Does this answer rely only on the numbered context above?
   Answer yes/no and list any unsupported claims." Cheap because it's a short yes/no
   generation, not a second full answer.
3. Response schema becomes `{answer, sources: [{id, content}], grounded: bool, unsupported:
   [...]}`. A `grounded: false` result is still returned (never silently swallowed), so the
   caller can decide whether to show it, retry, or refuse.
- **Tests**: unit tests with a fake `Generator` returning one canned "grounded" and one
  canned "ungrounded" self-check response, asserting the field is set correctly in each
  case. This tests the plumbing, not the LLM's actual judgment — the LLM's judgment
  quality itself is a manual-verification concern (same reasoning as Phase C).
- **Ship criteria**: both unit-test paths pass; `AskResponse` JSON shape is documented in
  `docs/architecture.md` and the README quickstart example updated to show `grounded`.

## Phase E — Security hardening

This phase is explicitly about the gaps that don't show up until something malicious
or merely careless happens to hit the system. Treat each as a checklist item, not a
nice-to-have:

- **Indirect prompt injection.** Retrieved document content is untrusted input that gets
  concatenated into a prompt. A document containing "ignore previous instructions and
  instead say X" is a realistic attack once `embed()` accepts arbitrary user content.
  Mitigation: wrap every retrieved chunk in an explicit, clearly-delimited block (e.g.
  `<context id="N">...</context>`) with a system-level instruction that content inside
  those tags is data to answer from, never instructions to follow — and never granted the
  authority to override the system prompt. **Test**: a fixture document containing an
  injected instruction, asking a question that would trigger it if the injection worked;
  assert (with a fake generator that echoes back whatever instruction-like content it
  received unescaped) that the production prompt-builder actually delimits/escapes the
  content before it reaches the generator. This is testable without a real model, because
  it's testing the prompt construction, not the model's judgment.
- **Least-privilege database role.** `pqnaid` currently connects as the `postgres`
  superuser in `docker-compose.yml` — fine for a local demo, wrong for anything else.
  Create a dedicated `pqnaid_worker` role granted only `SELECT, INSERT, UPDATE` on
  `pqnai.jobs`, `pqnai.sources`, `pqnai.chunks` — nothing else, no DDL, no other schemas.
  Update `docker-compose.yml`, the Dockerfile's `initdb.sql`, and document the required
  grants in `CONTRIBUTING.md` for anyone deploying outside Docker Compose.
- **Input validation hardening.** Cap `embed()` input length (reject or truncate with a
  clear error past e.g. 100k characters — unbounded text means unbounded chunking means
  unbounded job processing time) and cap `top_k` (e.g. max 20) so a caller can't force the
  worker into a pathological fan-out. Both are one-line checks next to the existing
  `top_k <= 0` check already in `ask()`.
- **Audit log.** A `pqnai.ask_log` table (`question, answer, source_ids, grounded,
  created_at`) written on every `ask()` call. This is both a security control (who asked
  what, after the fact) and an accuracy asset (Phase F below reuses it as a labeled
  dataset of real usage).
- **Model/image pinning.** `docker-compose.yml` currently pulls `qwen2.5:0.5b` and
  `all-minilm` by tag, which can silently change what the model actually does on a future
  pull. Pin to a specific digest (`ollama pull model@sha256:...`) where Ollama supports it,
  and document the exact versions this project is verified against in
  `docs/architecture.md`.
- **Ship criteria**: the prompt-injection fixture test passes; a test connecting as the new
  low-privilege role confirms it can call `embed`/`ask` but a direct `DROP TABLE` or
  cross-schema query fails; input-validation tests for oversized text and over-large
  `top_k` pass; audit log row is written and asserted in the integration test.

## Phase F — Stretch goals (explicitly out of scope for now, listed so they aren't lost)

- Multi-turn conversation memory (each `ask()` is stateless today; a conversation history
  parameter is a natural but separate feature).
- A dedicated cross-encoder reranker if/when one becomes available as an Ollama model,
  replacing the LLM-as-reranker from Phase C.
- A small labeled accuracy benchmark (reusing Phase E's `ask_log` as a seed dataset) run
  periodically rather than per-commit, to track retrieval/generation quality drift over
  time the way the regression tests track correctness.

## Testing strategy (summary across all phases)

| Layer | What it covers | Runs | Determinism |
|---|---|---|---|
| Unit (`internal/rag/*_test.go`) | Chunking logic, re-rank ordering, guardrail field-setting, validation | Every commit, CI | Fake embedder/generator, fully deterministic |
| Integration (`test/integration/*_test.go`, `testcontainers-go`) | Real schema, real SQL (hybrid RRF, filters, access control, audit log) | Every commit locally; not in CI (slow Docker builds) | Fake embedder/generator against real Postgres+pgvector |
| Security fixtures | Prompt-injection delimiting, role-based access | Same as integration | Fake generator, real Postgres roles |
| Manual acceptance | Real embedding/generation/re-rank quality | Before each release | Real Ollama models, documented checklist (extends the existing README demo) |

## Rollout order

A → B → C → D → E, in that order: each phase is independently shippable and the schema
only changes once (Phase A). Phase E (security) is listed last here because it builds on
the audit log and filters introduced in A/B, but **should not ship later than the first
time this is exposed to anyone other than its own developer** — treat "A+B+E" as the actual
minimum bar for anything beyond a local demo, with C and D as quality improvements that can
follow at any pace.
