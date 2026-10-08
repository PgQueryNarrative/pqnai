package rag

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

type fakeEmbedder struct {
	vec []float32
	err error
}

func (f fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return f.vec, f.err
}

type fakeGenerator struct {
	answer    string
	err       error
	gotPrompt string
}

func (f *fakeGenerator) Generate(ctx context.Context, prompt string) (string, error) {
	f.gotPrompt = prompt
	return f.answer, f.err
}

func TestAskValidation(t *testing.T) {
	embedder := fakeEmbedder{vec: []float32{0.1, 0.2}}
	generator := &fakeGenerator{answer: "yes"}

	if _, err := Ask(context.Background(), nil, embedder, generator, AskRequest{Question: "", TopK: 3}); err == nil {
		t.Fatal("expected error for empty question")
	}
	if _, err := Ask(context.Background(), nil, embedder, generator, AskRequest{Question: "hi", TopK: 0}); err == nil {
		t.Fatal("expected error for non-positive top_k")
	}
}

func TestNormalizeFilters(t *testing.T) {
	valid := []struct {
		in, want string
	}{
		{"", "{}"},
		{"null", "{}"},
		{"  ", "{}"},
		{"{}", "{}"},
		{`{"tenant_id": "acme"}`, `{"tenant_id": "acme"}`},
		{`{"tags": ["a", "b"], "nested": {"x": 1}}`, `{"tags": ["a", "b"], "nested": {"x": 1}}`},
	}
	for _, v := range valid {
		got, err := normalizeFilters(json.RawMessage(v.in))
		if err != nil {
			t.Fatalf("unexpected error for %q: %v", v.in, err)
		}
		if string(got) != v.want {
			t.Fatalf("for %q expected %q, got %q", v.in, v.want, string(got))
		}
	}

	invalid := []string{
		`["tenant_id", "acme"]`, // array: valid jsonb, but would silently match nothing
		`"acme"`,                // scalar
		`42`,
		`{not json`,
	}
	for _, in := range invalid {
		if _, err := normalizeFilters(json.RawMessage(in)); err == nil {
			t.Fatalf("expected error for non-object filters %q", in)
		}
	}
}

// A malformed filter must fail before the embedder is called, so a caller
// mistake never costs a model round-trip.
func TestAskRejectsBadFiltersBeforeEmbedding(t *testing.T) {
	embedder := &countingEmbedder{}
	generator := &fakeGenerator{answer: "yes"}

	_, err := Ask(context.Background(), nil, embedder, generator, AskRequest{
		Question: "hi",
		TopK:     3,
		Filters:  json.RawMessage(`["not", "an", "object"]`),
	})
	if err == nil {
		t.Fatal("expected error for non-object filters")
	}
	if embedder.calls != 0 {
		t.Fatalf("expected embedder not to be called, got %d calls", embedder.calls)
	}
}

type countingEmbedder struct{ calls int }

func (c *countingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	c.calls++
	return []float32{0.1}, nil
}

func TestAskEmbedderError(t *testing.T) {
	embedder := fakeEmbedder{err: errors.New("boom")}
	generator := &fakeGenerator{}

	if _, err := Ask(context.Background(), nil, embedder, generator, AskRequest{Question: "hi", TopK: 3}); err == nil {
		t.Fatal("expected error to propagate from embedder")
	}
}

func TestEmbedValidation(t *testing.T) {
	embedder := fakeEmbedder{vec: []float32{0.1}}
	if _, err := Embed(context.Background(), nil, embedder, EmbedRequest{Text: "  "}); err == nil {
		t.Fatal("expected error for blank text")
	}
}

func TestEmbedEmbedderError(t *testing.T) {
	embedder := fakeEmbedder{err: errors.New("boom")}
	if _, err := Embed(context.Background(), nil, embedder, EmbedRequest{Text: "hello"}); err == nil {
		t.Fatal("expected error to propagate from embedder")
	}
}
