package rag

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOllamaClientEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embeddings" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		var body embedRequestBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body.Model != "all-minilm" {
			t.Fatalf("expected model all-minilm, got %q", body.Model)
		}
		_ = json.NewEncoder(w).Encode(embedResponseBody{Embedding: []float32{0.1, 0.2, 0.3}})
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	vec, err := c.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3-dim vector, got %d", len(vec))
	}
}

func TestOllamaClientEmbedEmptyResult(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponseBody{})
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	if _, err := c.Embed(context.Background(), "hello"); err == nil {
		t.Fatal("expected error for empty embedding")
	}
}

func TestOllamaClientGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(generateResponseBody{Response: "the answer"})
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	answer, err := c.Generate(context.Background(), "some prompt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if answer != "the answer" {
		t.Fatalf("expected %q, got %q", "the answer", answer)
	}
}

func TestOllamaClientErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("model not found"))
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	if _, err := c.Embed(context.Background(), "hello"); err == nil {
		t.Fatal("expected error for 500 status")
	}
}
