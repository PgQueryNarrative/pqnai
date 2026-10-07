# Origin

pqnai began as a rewrite of [PGAI](https://github.com/Postgres-artificialintelligence/PGAI),
a predictive-analytics-for-Postgres project by Abdul Moiez Ibrar and Damil Shahzad. The
original idea — run forecasting models and read predictions back through SQL — is preserved
here. The implementation is new: the Python forecasting scripts have been replaced by an
external Go worker, the C extension has been rebuilt as a thin, correct job-queue interface
instead of a placeholder, and the project now also covers embeddings/RAG, under a new name and
a new org (`PgQueryNarrative`).

No code was carried over line-for-line; this document exists for attribution and history, not
as a changelog.
