# Contributing

## Development setup

- `extension/` needs a Postgres install with `pg_config` on `PATH` (or use
  `docker compose build postgres`, which builds it inside a container).
- `go build ./...` / `go test ./...` for the worker.
- `docker compose up --build` runs the full stack locally.

## Making changes

- **Extension SQL changes**: never edit a released `pqnai--X.Y.Z.sql` file. Add a new
  `pqnai--X.Y.Z--X.Y.Z+1.sql` upgrade script and bump `default_version` in
  `pqnai.control`.
- **Go changes**: run `go vet ./...` and `golangci-lint run` before opening a PR; CI
  enforces both.
- Add a regression test (`test/regress/sql/*.sql` + `test/regress/expected/*.out`) for
  any new SQL-visible behavior, and a Go test for any new worker logic.

## Pull requests

Keep PRs scoped to one change. Describe what changed and why in the description, not in
code comments.
