// pqnaid is the out-of-process worker for pqnai: it listens for jobs
// enqueued by the pqnai Postgres extension and executes them, since
// Postgres backend processes must never run Go code or block on
// network I/O in-process.
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/PgQueryNarrative/pqnai/internal/forecast"
	"github.com/PgQueryNarrative/pqnai/internal/pgjobs"
)

const pollInterval = 5 * time.Second

func main() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("pqnaid: DATABASE_URL must be set")
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("pqnaid: connect: %v", err)
	}
	defer pool.Close()

	log.Println("pqnaid: connected, draining any backlog")
	drainQueue(ctx, pool)

	go listenForNotifications(ctx, dbURL, pool)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("pqnaid: shutting down")
			return
		case <-ticker.C:
			drainQueue(ctx, pool)
		}
	}
}

// listenForNotifications holds a dedicated LISTEN connection and drains
// the queue on every notification. The periodic poll in main is the
// safety net for any notification missed while this connection is down.
func listenForNotifications(ctx context.Context, dbURL string, pool *pgxpool.Pool) {
	for ctx.Err() == nil {
		conn, err := pgx.Connect(ctx, dbURL)
		if err != nil {
			log.Printf("pqnaid: listen connect failed, retrying: %v", err)
			time.Sleep(pollInterval)
			continue
		}

		if _, err := conn.Exec(ctx, "LISTEN pqnai_jobs"); err != nil {
			log.Printf("pqnaid: LISTEN failed, retrying: %v", err)
			conn.Close(ctx)
			time.Sleep(pollInterval)
			continue
		}

		for ctx.Err() == nil {
			if _, err := conn.WaitForNotification(ctx); err != nil {
				log.Printf("pqnaid: notification wait failed, reconnecting: %v", err)
				break
			}
			drainQueue(ctx, pool)
		}
		conn.Close(ctx)
	}
}

func drainQueue(ctx context.Context, pool *pgxpool.Pool) {
	for {
		job, ok, err := pgjobs.ClaimNext(ctx, pool)
		if err != nil {
			log.Printf("pqnaid: claim failed: %v", err)
			return
		}
		if !ok {
			return
		}
		handle(ctx, pool, job)
	}
}

func handle(ctx context.Context, pool *pgxpool.Pool, job *pgjobs.Job) {
	switch job.Type {
	case "forecast":
		handleForecast(ctx, pool, job)
	default:
		failJob(ctx, pool, job.ID, "unknown job type: "+job.Type)
	}
}

func handleForecast(ctx context.Context, pool *pgxpool.Pool, job *pgjobs.Job) {
	var req forecast.Request
	if err := json.Unmarshal(job.Payload, &req); err != nil {
		failJob(ctx, pool, job.ID, "invalid payload: "+err.Error())
		return
	}

	resp, err := forecast.Run(req)
	if err != nil {
		failJob(ctx, pool, job.ID, err.Error())
		return
	}

	result, err := json.Marshal(resp)
	if err != nil {
		failJob(ctx, pool, job.ID, "marshal result: "+err.Error())
		return
	}

	if err := pgjobs.Complete(ctx, pool, job.ID, result); err != nil {
		log.Printf("pqnaid: complete job %d failed: %v", job.ID, err)
	}
}

func failJob(ctx context.Context, pool *pgxpool.Pool, id int64, msg string) {
	if err := pgjobs.Fail(ctx, pool, id, msg); err != nil {
		log.Printf("pqnaid: fail job %d failed: %v", id, err)
	}
}
