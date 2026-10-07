//go:build integration

package integration

import (
	"context"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/PgQueryNarrative/pqnai/internal/rag"
)

// fakeEmbedder maps known strings to hand-picked vectors so cosine
// distance has a deterministic, checkable winner, without depending on
// a real embedding model being available in CI.
type fakeEmbedder struct {
	vectors map[string][]float32
}

func (f fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if v, ok := f.vectors[text]; ok {
		return v, nil
	}
	return make([]float32, 384), nil
}

type fakeGenerator struct{}

func (fakeGenerator) Generate(ctx context.Context, prompt string) (string, error) {
	return "fake answer", nil
}

func vec384(nonZeroIdx int) []float32 {
	v := make([]float32, 384)
	v[nonZeroIdx] = 1
	return v
}

// TestRAGRoundTrip proves the real pgvector-backed storage and
// cosine-distance retrieval in internal/rag against a live Postgres +
// pgvector container -- the unit tests in internal/rag pass a nil pool
// and never touch the database, so they can't catch a bug in the SQL
// itself (wrong operator, wrong ORDER BY direction, etc).
func TestRAGRoundTrip(t *testing.T) {
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

	embedder := fakeEmbedder{vectors: map[string][]float32{
		"about dogs": vec384(0),
		"about cars": vec384(1),
		"query":      vec384(0), // closest to "about dogs"
	}}

	if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: "about dogs"}); err != nil {
		t.Fatalf("embed dogs: %v", err)
	}
	if _, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{Text: "about cars"}); err != nil {
		t.Fatalf("embed cars: %v", err)
	}

	resp, err := rag.Ask(ctx, pool, embedder, fakeGenerator{}, rag.AskRequest{Question: "query", TopK: 1})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if len(resp.Sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(resp.Sources))
	}
	if resp.Sources[0] != "about dogs" {
		t.Fatalf("expected closest match 'about dogs', got %q", resp.Sources[0])
	}

	testChunkedEmbed(t, ctx, pool, embedder)
}

// testChunkedEmbed proves a long document is actually split into
// multiple rows in pqnai.chunks, each independently retrievable, rather
// than silently stored as one oversized vector -- the behavior added in
// docs/rag-advanced-plan.md Phase A.
func testChunkedEmbed(t *testing.T, ctx context.Context, pool *pgxpool.Pool, embedder fakeEmbedder) {
	t.Helper()

	longText := strings.Repeat("sentence about elephants. ", 20) // ~540 runes
	resp, err := rag.Embed(ctx, pool, embedder, rag.EmbedRequest{
		Text:         longText,
		ChunkSize:    50,
		ChunkOverlap: 5,
	})
	if err != nil {
		t.Fatalf("embed long text: %v", err)
	}
	if resp.ChunkCount <= 1 {
		t.Fatalf("expected multiple chunks for a long document, got %d", resp.ChunkCount)
	}

	var storedCount int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM pqnai.chunks WHERE source_id = $1", resp.SourceID,
	).Scan(&storedCount); err != nil {
		t.Fatalf("count chunks: %v", err)
	}
	if storedCount != resp.ChunkCount {
		t.Fatalf("expected %d stored chunk rows, got %d", resp.ChunkCount, storedCount)
	}

	var maxIndex int
	if err := pool.QueryRow(ctx,
		"SELECT max(chunk_index) FROM pqnai.chunks WHERE source_id = $1", resp.SourceID,
	).Scan(&maxIndex); err != nil {
		t.Fatalf("max chunk_index: %v", err)
	}
	if maxIndex != resp.ChunkCount-1 {
		t.Fatalf("expected chunk_index to run 0..%d, max was %d", resp.ChunkCount-1, maxIndex)
	}
}
