package rag

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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

// captureGenerate returns a server that records the decoded
// /api/generate request body and replies with a fixed response.
func captureGenerate(t *testing.T, got *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(generateResponseBody{Response: `{"scores": []}`})
	}))
}

// Without an explicit context window and truncate=false, Ollama silently
// drops the start of an over-long prompt -- where pqnai puts its grounding
// instructions -- so every generation call must send both.
func TestOllamaClientGenerateSendsContextGuards(t *testing.T) {
	var got map[string]any
	srv := captureGenerate(t, &got)
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	c.NumCtx = 12345
	if _, err := c.Generate(context.Background(), "p"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got["truncate"] != false {
		t.Fatalf("expected truncate=false, got %v", got["truncate"])
	}
	opts, _ := got["options"].(map[string]any)
	if opts["num_ctx"] != float64(12345) {
		t.Fatalf("expected num_ctx 12345, got %v", opts["num_ctx"])
	}
	if _, set := opts["temperature"]; set {
		t.Fatalf("expected plain Generate to leave temperature at the model default, got %v", opts["temperature"])
	}
	if _, set := got["format"]; set {
		t.Fatalf("expected no format for plain Generate, got %v", got["format"])
	}
}

func TestOllamaClientGenerateJSONSendsSchemaAndZeroTemperature(t *testing.T) {
	var got map[string]any
	srv := captureGenerate(t, &got)
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	if _, err := c.GenerateJSON(context.Background(), "p", rerankSchema(2)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got["truncate"] != false {
		t.Fatalf("expected truncate=false, got %v", got["truncate"])
	}
	format, ok := got["format"].(map[string]any)
	if !ok || format["type"] != "object" {
		t.Fatalf("expected the JSON schema as format, got %v", got["format"])
	}
	opts, _ := got["options"].(map[string]any)
	if opts["temperature"] != float64(0) {
		t.Fatalf("expected temperature 0 for deterministic grading, got %v", opts["temperature"])
	}
	if opts["num_ctx"] != float64(DefaultNumCtx) {
		t.Fatalf("expected default num_ctx %d, got %v", DefaultNumCtx, opts["num_ctx"])
	}
}

// An over-long prompt must surface as an error, not a truncated answer.
func TestOllamaClientSurfacesContextOverflow(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"request (11637 tokens) exceeds the available context size (4096 tokens)"}`))
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL, "all-minilm", "qwen2.5:0.5b")
	_, err := c.Generate(context.Background(), "long prompt")
	if err == nil || !strings.Contains(err.Error(), "exceeds the available context size") {
		t.Fatalf("expected the overflow error to surface, got %v", err)
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
