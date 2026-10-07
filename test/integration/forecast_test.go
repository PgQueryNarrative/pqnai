//go:build integration

// Package integration exercises the real job-queue contract against a live
// Postgres + pqnai extension in a container, simulating the pqnaid worker
// side in-process instead of spawning the compiled binary.
package integration

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/PgQueryNarrative/pqnai/internal/forecast"
	"github.com/PgQueryNarrative/pqnai/internal/pgjobs"
)

func TestForecastRoundTrip(t *testing.T) {
	ctx := context.Background()

	req := testcontainers.ContainerRequest{
		FromDockerfile: testcontainers.FromDockerfile{
			Context:    "../..",
			Dockerfile: "docker/Dockerfile.postgres",
		},
		ExposedPorts: []string{"5432/tcp"},
		Env: map[string]string{
			"POSTGRES_PASSWORD": "pqnai",
			"POSTGRES_DB":       "pqnai",
		},
		WaitingFor: wait.ForLog("database system is ready to accept connections").WithOccurrence(2),
	}

	pg, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		t.Fatalf("start container: %v", err)
	}
	defer func() { _ = pg.Terminate(ctx) }()

	host, err := pg.Host(ctx)
	if err != nil {
		t.Fatalf("container host: %v", err)
	}
	port, err := pg.MappedPort(ctx, "5432")
	if err != nil {
		t.Fatalf("container port: %v", err)
	}

	dbURL := "postgres://postgres:pqnai@" + host + ":" + port.Port() + "/pqnai"

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	jobID, err := enqueueForecastJob(ctx, pool, []float64{1, 2, 3, 4, 5}, 2)
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	// Simulate the pqnaid worker: claim the job and process it, the same
	// way cmd/pqnaid's drainQueue does.
	job, ok, err := pgjobs.ClaimNext(ctx, pool)
	if err != nil || !ok {
		t.Fatalf("claim: ok=%v err=%v", ok, err)
	}
	if job.ID != jobID {
		t.Fatalf("claimed job %d, expected %d", job.ID, jobID)
	}

	var fReq forecast.Request
	if err := json.Unmarshal(job.Payload, &fReq); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	resp, err := forecast.Run(fReq)
	if err != nil {
		t.Fatalf("forecast.Run: %v", err)
	}
	result, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	if err := pgjobs.Complete(ctx, pool, job.ID, result); err != nil {
		t.Fatalf("complete: %v", err)
	}

	var status string
	var stored []byte
	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.QueryRow(waitCtx, "SELECT status, result FROM pqnai.jobs WHERE id = $1", jobID).Scan(&status, &stored); err != nil {
		t.Fatalf("read back job: %v", err)
	}
	if status != "done" {
		t.Fatalf("expected status done, got %q", status)
	}

	var got forecast.Response
	if err := json.Unmarshal(stored, &got); err != nil {
		t.Fatalf("unmarshal stored result: %v", err)
	}
	if len(got.Forecast) != 2 {
		t.Fatalf("expected 2 forecast points, got %d", len(got.Forecast))
	}
}

func enqueueForecastJob(ctx context.Context, pool *pgxpool.Pool, series []float64, horizon int) (int64, error) {
	payload, err := json.Marshal(forecast.Request{Series: series, Horizon: horizon})
	if err != nil {
		return 0, err
	}
	var id int64
	err = pool.QueryRow(ctx, "SELECT pqnai.enqueue_job('forecast', $1::jsonb)", payload).Scan(&id)
	return id, err
}
