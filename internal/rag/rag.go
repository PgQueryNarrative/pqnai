package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// Embedder and Generator are satisfied by *OllamaClient; tests substitute
// fakes so they don't depend on a running model.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

type EmbedRequest struct {
	Text         string          `json:"text"`
	Title        *string         `json:"title,omitempty"`
	Metadata     json.RawMessage `json:"metadata,omitempty"`
	ChunkSize    int             `json:"chunk_size,omitempty"`
	ChunkOverlap int             `json:"chunk_overlap,omitempty"`
}

type EmbedResponse struct {
	SourceID   int64 `json:"source_id"`
	ChunkCount int   `json:"chunk_count"`
}

// Embed splits req.Text into chunks, embeds each one, and stores them as
// a new source + its chunks in a single transaction: either the whole
// document lands, or none of it does, never a source with partial chunks.
func Embed(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, req EmbedRequest) (EmbedResponse, error) {
	if strings.TrimSpace(req.Text) == "" {
		return EmbedResponse{}, fmt.Errorf("rag: text must not be empty")
	}

	size := req.ChunkSize
	if size <= 0 {
		size = DefaultChunkSize
	}
	overlap := req.ChunkOverlap
	if overlap <= 0 {
		overlap = DefaultChunkOverlap
	}

	chunks, err := Chunk(req.Text, ChunkOptions{Size: size, Overlap: overlap})
	if err != nil {
		return EmbedResponse{}, err
	}
	if len(chunks) == 0 {
		return EmbedResponse{}, fmt.Errorf("rag: text produced no chunks")
	}

	// Embed every chunk before touching the database: these are slow
	// network calls to the model, and a Postgres transaction shouldn't
	// sit open (holding locks) for their duration.
	vectors := make([]string, len(chunks))
	for i, chunk := range chunks {
		vec, err := embedder.Embed(ctx, chunk)
		if err != nil {
			return EmbedResponse{}, fmt.Errorf("rag: embed chunk %d: %w", i, err)
		}
		vectors[i] = pgvector.NewVector(vec).String()
	}

	metadata := req.Metadata
	if len(metadata) == 0 {
		metadata = json.RawMessage("{}")
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return EmbedResponse{}, fmt.Errorf("rag: begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }() // no-op once committed

	var sourceID int64
	err = tx.QueryRow(ctx,
		"INSERT INTO pqnai.sources (title, metadata) VALUES ($1, $2::jsonb) RETURNING id",
		req.Title, []byte(metadata),
	).Scan(&sourceID)
	if err != nil {
		return EmbedResponse{}, fmt.Errorf("rag: store source: %w", err)
	}

	for i, chunk := range chunks {
		_, err = tx.Exec(ctx,
			"INSERT INTO pqnai.chunks (source_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4::vector)",
			sourceID, i, chunk, vectors[i],
		)
		if err != nil {
			return EmbedResponse{}, fmt.Errorf("rag: store chunk %d: %w", i, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return EmbedResponse{}, fmt.Errorf("rag: commit: %w", err)
	}

	return EmbedResponse{SourceID: sourceID, ChunkCount: len(chunks)}, nil
}

type AskRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k"`
	// Filters restricts retrieval to chunks whose source metadata
	// contains it (jsonb @>), e.g. {"tenant_id": "acme"}. Empty means no
	// restriction.
	Filters json.RawMessage `json:"filters,omitempty"`
	// Rerank re-scores the top fused candidates with the Reranker before
	// truncating to TopK. Costs an extra model call.
	Rerank bool `json:"rerank,omitempty"`
}

// NoContextAnswer is returned, without calling the model, when retrieval
// finds no chunks for a question (e.g. its filters match no sources).
const NoContextAnswer = "No relevant context was found for this question."

type AskResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
	// Reranked reports whether the sources' order came from the
	// reranker. False when re-ranking wasn't requested, or when it was
	// but failed and fell back to the fused order (see RerankError).
	Reranked    bool   `json:"reranked"`
	RerankError string `json:"rerank_error,omitempty"`
}

// normalizeFilters defaults an absent filter to {} (matches every source,
// since every jsonb object contains the empty object) and rejects anything
// that isn't a JSON object: an array or scalar would be a valid jsonb
// value but would silently match nothing via @>, hiding a caller mistake.
func normalizeFilters(raw json.RawMessage) (json.RawMessage, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return json.RawMessage("{}"), nil
	}

	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("rag: filters must be a JSON object: %w", err)
	}
	return raw, nil
}

const askPromptTemplate = `Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
%s

Question: %s

Answer:`

// Ask retrieves context for req.Question and generates an answer from it.
// reranker may be nil unless req.Rerank is set.
func Ask(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, generator Generator, reranker Reranker, req AskRequest) (AskResponse, error) {
	if strings.TrimSpace(req.Question) == "" {
		return AskResponse{}, fmt.Errorf("rag: question must not be empty")
	}
	if req.TopK <= 0 {
		return AskResponse{}, fmt.Errorf("rag: top_k must be positive, got %d", req.TopK)
	}
	if req.Rerank && reranker == nil {
		return AskResponse{}, fmt.Errorf("rag: rerank requested but no reranker is configured")
	}

	filters, err := normalizeFilters(req.Filters)
	if err != nil {
		return AskResponse{}, err
	}

	qvec, err := embedder.Embed(ctx, req.Question)
	if err != nil {
		return AskResponse{}, err
	}

	vectorHits, err := vectorSearch(ctx, pool, qvec, candidatePoolSize, filters)
	if err != nil {
		return AskResponse{}, err
	}
	keywordHits, err := fullTextSearch(ctx, pool, req.Question, candidatePoolSize, filters)
	if err != nil {
		return AskResponse{}, err
	}

	fused := reciprocalRankFusion(vectorHits, keywordHits)

	// With nothing retrieved, don't call any model: given an empty
	// context the generator tends to answer from its own training data
	// despite the prompt's instruction, producing an ungrounded answer
	// that looks like a retrieval result.
	if len(fused) == 0 {
		return AskResponse{Answer: NoContextAnswer, Sources: []string{}}, nil
	}

	var resp AskResponse
	if req.Rerank {
		reordered, err := rerankCandidates(ctx, reranker, req.Question, fused)
		if err != nil {
			resp.RerankError = err.Error()
		} else {
			resp.Reranked = true
		}
		fused = reordered
	}

	if len(fused) > req.TopK {
		fused = fused[:req.TopK]
	}

	sources := make([]string, len(fused))
	for i, c := range fused {
		sources[i] = c.Content
	}
	resp.Sources = sources

	prompt := fmt.Sprintf(askPromptTemplate, strings.Join(sources, "\n---\n"), req.Question)
	answer, err := generator.Generate(ctx, prompt)
	if err != nil {
		return AskResponse{}, err
	}

	resp.Answer = strings.TrimSpace(answer)
	return resp, nil
}
