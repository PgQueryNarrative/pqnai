package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// rrfK is the standard Reciprocal Rank Fusion smoothing constant. It
// dampens how much a single top-ranked hit dominates the fused score,
// so a chunk ranked moderately well by both retrievers can outrank one
// ranked first by only one of them.
const rrfK = 60

// candidatePoolSize is how many results each retriever contributes
// before fusion. Larger than any realistic top_k so fusion has room to
// promote a chunk one retriever ranked low but the other ranked high.
const candidatePoolSize = 20

type candidate struct {
	ChunkID int64
	Content string
}

// vectorSearch returns up to limit chunks ordered by cosine distance to
// vec, restricted to chunks whose source metadata contains filters.
func vectorSearch(ctx context.Context, pool *pgxpool.Pool, vec []float32, limit int, filters json.RawMessage) ([]candidate, error) {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return nil, fmt.Errorf("rag: vector search: begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }() // read-only; nothing to commit

	// By default pgvector's HNSW index yields at most ef_search (40)
	// nearest chunks overall, and the metadata filter is applied after
	// that -- so a filter whose matches all fall outside the global top 40
	// silently returns nothing, even though matching chunks exist.
	// Iterative scan keeps walking the index until enough rows pass the
	// filter. strict_order (not relaxed_order) because rank position
	// feeds reciprocalRankFusion directly.
	if _, err := tx.Exec(ctx, "SET LOCAL hnsw.iterative_scan = strict_order"); err != nil {
		return nil, fmt.Errorf("rag: vector search: enable hnsw iterative scan (requires pgvector >= 0.8.0): %w", err)
	}

	rows, err := tx.Query(ctx, `
		SELECT c.id, c.content
		FROM pqnai.chunks c
		JOIN pqnai.sources s ON s.id = c.source_id
		WHERE s.metadata @> $1::jsonb
		ORDER BY c.embedding <=> $2::vector
		LIMIT $3
	`, []byte(filters), pgvector.NewVector(vec).String(), limit)
	if err != nil {
		return nil, fmt.Errorf("rag: vector search: %w", err)
	}
	return collectCandidates(rows, "vector search")
}

// fullTextSearch returns up to limit chunks matching query as a
// Postgres full-text search, ordered by ts_rank, restricted the same
// way as vectorSearch. Catches exact terms (names, codes, acronyms) that
// an embedding can blur together with semantically similar words.
func fullTextSearch(ctx context.Context, pool *pgxpool.Pool, query string, limit int, filters json.RawMessage) ([]candidate, error) {
	rows, err := pool.Query(ctx, `
		SELECT c.id, c.content
		FROM pqnai.chunks c
		JOIN pqnai.sources s ON s.id = c.source_id
		WHERE s.metadata @> $1::jsonb
		  AND c.tsv @@ plainto_tsquery('english', $2)
		ORDER BY ts_rank(c.tsv, plainto_tsquery('english', $2)) DESC
		LIMIT $3
	`, []byte(filters), query, limit)
	if err != nil {
		return nil, fmt.Errorf("rag: full-text search: %w", err)
	}
	return collectCandidates(rows, "full-text search")
}

func collectCandidates(rows pgx.Rows, label string) ([]candidate, error) {
	defer rows.Close()

	var out []candidate
	for rows.Next() {
		var c candidate
		if err := rows.Scan(&c.ChunkID, &c.Content); err != nil {
			return nil, fmt.Errorf("rag: scan %s result: %w", label, err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rag: iterate %s results: %w", label, err)
	}
	return out, nil
}

// reciprocalRankFusion merges independently-ranked candidate lists into
// one ranking: score(chunk) = Σ 1/(rrfK + rank) across every list the
// chunk appears in (rank is 1-based). Using only rank position, not raw
// scores, means cosine distance and ts_rank never need to be normalized
// against each other. Chunks are deduplicated by ChunkID; ties keep the
// order in which each chunk was first seen, so results are deterministic.
func reciprocalRankFusion(lists ...[]candidate) []candidate {
	type scored struct {
		candidate
		score float64
		first int
	}

	byID := map[int64]*scored{}
	seen := 0
	for _, list := range lists {
		for rank, c := range list {
			s, ok := byID[c.ChunkID]
			if !ok {
				s = &scored{candidate: c, first: seen}
				byID[c.ChunkID] = s
				seen++
			}
			s.score += 1.0 / float64(rrfK+rank+1)
		}
	}

	merged := make([]*scored, 0, len(byID))
	for _, s := range byID {
		merged = append(merged, s)
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].score != merged[j].score {
			return merged[i].score > merged[j].score
		}
		return merged[i].first < merged[j].first
	})

	out := make([]candidate, len(merged))
	for i, s := range merged {
		out[i] = s.candidate
	}
	return out
}
