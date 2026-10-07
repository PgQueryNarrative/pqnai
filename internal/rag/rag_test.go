package rag

import (
	"context"
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
