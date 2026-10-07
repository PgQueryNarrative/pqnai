package rag

import (
	"context"
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
	Text string `json:"text"`
}

type EmbedResponse struct {
	DocumentID int64 `json:"document_id"`
}

func Embed(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, req EmbedRequest) (EmbedResponse, error) {
	if strings.TrimSpace(req.Text) == "" {
		return EmbedResponse{}, fmt.Errorf("rag: text must not be empty")
	}

	vec, err := embedder.Embed(ctx, req.Text)
	if err != nil {
		return EmbedResponse{}, err
	}

	var id int64
	err = pool.QueryRow(ctx,
		"INSERT INTO pqnai.documents (content, embedding) VALUES ($1, $2::vector) RETURNING id",
		req.Text, pgvector.NewVector(vec).String(),
	).Scan(&id)
	if err != nil {
		return EmbedResponse{}, fmt.Errorf("rag: store document: %w", err)
	}

	return EmbedResponse{DocumentID: id}, nil
}

type AskRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k"`
}

type AskResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
}

const askPromptTemplate = `Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
%s

Question: %s

Answer:`

func Ask(ctx context.Context, pool *pgxpool.Pool, embedder Embedder, generator Generator, req AskRequest) (AskResponse, error) {
	if strings.TrimSpace(req.Question) == "" {
		return AskResponse{}, fmt.Errorf("rag: question must not be empty")
	}
	if req.TopK <= 0 {
		return AskResponse{}, fmt.Errorf("rag: top_k must be positive, got %d", req.TopK)
	}

	qvec, err := embedder.Embed(ctx, req.Question)
	if err != nil {
		return AskResponse{}, err
	}

	rows, err := pool.Query(ctx,
		"SELECT content FROM pqnai.documents ORDER BY embedding <=> $1::vector LIMIT $2",
		pgvector.NewVector(qvec).String(), req.TopK,
	)
	if err != nil {
		return AskResponse{}, fmt.Errorf("rag: retrieve documents: %w", err)
	}
	defer rows.Close()

	var sources []string
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return AskResponse{}, fmt.Errorf("rag: scan document: %w", err)
		}
		sources = append(sources, content)
	}
	if err := rows.Err(); err != nil {
		return AskResponse{}, fmt.Errorf("rag: iterate documents: %w", err)
	}

	prompt := fmt.Sprintf(askPromptTemplate, strings.Join(sources, "\n---\n"), req.Question)
	answer, err := generator.Generate(ctx, prompt)
	if err != nil {
		return AskResponse{}, err
	}

	return AskResponse{Answer: strings.TrimSpace(answer), Sources: sources}, nil
}
