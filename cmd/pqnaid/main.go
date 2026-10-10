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
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/PgQueryNarrative/pqnai/internal/forecast"
	"github.com/PgQueryNarrative/pqnai/internal/pgjobs"
	"github.com/PgQueryNarrative/pqnai/internal/rag"
)

const pollInterval = 5 * time.Second

// worker bundles the dependencies every job handler needs.
type worker struct {
	pool          *pgxpool.Pool
	ollama        *rag.OllamaClient
	reranker      rag.Reranker
	groundChecker rag.JSONGenerator
}

func main() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("pqnaid: DATABASE_URL must be set")
	}

	ollamaURL := envOr("OLLAMA_URL", "http://localhost:11434")
	embedModel := envOr("OLLAMA_EMBED_MODEL", "all-minilm")
	chatModel := envOr("OLLAMA_CHAT_MODEL", "qwen2.5:0.5b")
	// Grading relevance is a different task from answering; a stronger
	// model can be used for it without slowing every answer down.
	rerankModel := envOr("OLLAMA_RERANK_MODEL", chatModel)
	// Fact-checking an answer is also a grading task, not an answering
	// one, so it defaults to the rerank model rather than the chat model.
	groundednessModel := envOr("OLLAMA_GROUNDEDNESS_MODEL", rerankModel)
	numCtx, err := strconv.Atoi(envOr("OLLAMA_NUM_CTX", strconv.Itoa(rag.DefaultNumCtx)))
	if err != nil || numCtx <= 0 {
		log.Fatalf("pqnaid: OLLAMA_NUM_CTX must be a positive integer, got %q", os.Getenv("OLLAMA_NUM_CTX"))
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("pqnaid: connect: %v", err)
	}
	defer pool.Close()

	ollama := rag.NewOllamaClient(ollamaURL, embedModel, chatModel)
	ollama.NumCtx = numCtx
	rerankClient := rag.NewOllamaClient(ollamaURL, embedModel, rerankModel)
	rerankClient.NumCtx = numCtx
	groundednessClient := rag.NewOllamaClient(ollamaURL, embedModel, groundednessModel)
	groundednessClient.NumCtx = numCtx

	w := &worker{
		pool:          pool,
		ollama:        ollama,
		reranker:      rag.NewLLMReranker(rerankClient),
		groundChecker: groundednessClient,
	}

	log.Println("pqnaid: connected, draining any backlog")
	w.drainQueue(ctx)

	go w.listenForNotifications(ctx, dbURL)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("pqnaid: shutting down")
			return
		case <-ticker.C:
			w.drainQueue(ctx)
		}
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// listenForNotifications holds a dedicated LISTEN connection and drains
// the queue on every notification. The periodic poll in main is the
// safety net for any notification missed while this connection is down.
func (w *worker) listenForNotifications(ctx context.Context, dbURL string) {
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
			w.drainQueue(ctx)
		}
		conn.Close(ctx)
	}
}

func (w *worker) drainQueue(ctx context.Context) {
	for {
		job, ok, err := pgjobs.ClaimNext(ctx, w.pool)
		if err != nil {
			log.Printf("pqnaid: claim failed: %v", err)
			return
		}
		if !ok {
			return
		}
		w.handle(ctx, job)
	}
}

func (w *worker) handle(ctx context.Context, job *pgjobs.Job) {
	switch job.Type {
	case "forecast":
		w.handleForecast(ctx, job)
	case "embed":
		w.handleEmbed(ctx, job)
	case "ask":
		w.handleAsk(ctx, job)
	default:
		w.failJob(ctx, job.ID, "unknown job type: "+job.Type)
	}
}

func (w *worker) handleForecast(ctx context.Context, job *pgjobs.Job) {
	var req forecast.Request
	if err := json.Unmarshal(job.Payload, &req); err != nil {
		w.failJob(ctx, job.ID, "invalid payload: "+err.Error())
		return
	}

	resp, err := forecast.Run(req)
	if err != nil {
		w.failJob(ctx, job.ID, err.Error())
		return
	}

	w.completeWithJSON(ctx, job.ID, resp)
}

func (w *worker) handleEmbed(ctx context.Context, job *pgjobs.Job) {
	var req rag.EmbedRequest
	if err := json.Unmarshal(job.Payload, &req); err != nil {
		w.failJob(ctx, job.ID, "invalid payload: "+err.Error())
		return
	}

	resp, err := rag.Embed(ctx, w.pool, w.ollama, req)
	if err != nil {
		w.failJob(ctx, job.ID, err.Error())
		return
	}

	w.completeWithJSON(ctx, job.ID, resp)
}

func (w *worker) handleAsk(ctx context.Context, job *pgjobs.Job) {
	var req rag.AskRequest
	if err := json.Unmarshal(job.Payload, &req); err != nil {
		w.failJob(ctx, job.ID, "invalid payload: "+err.Error())
		return
	}

	resp, err := rag.Ask(ctx, w.pool, w.ollama, w.ollama, w.reranker, w.groundChecker, req)
	if err != nil {
		w.failJob(ctx, job.ID, err.Error())
		return
	}

	w.completeWithJSON(ctx, job.ID, resp)
}

func (w *worker) completeWithJSON(ctx context.Context, jobID int64, v any) {
	result, err := json.Marshal(v)
	if err != nil {
		w.failJob(ctx, jobID, "marshal result: "+err.Error())
		return
	}
	if err := pgjobs.Complete(ctx, w.pool, jobID, result); err != nil {
		log.Printf("pqnaid: complete job %d failed: %v", jobID, err)
	}
}

func (w *worker) failJob(ctx context.Context, id int64, msg string) {
	if err := pgjobs.Fail(ctx, w.pool, id, msg); err != nil {
		log.Printf("pqnaid: fail job %d failed: %v", id, err)
	}
}
