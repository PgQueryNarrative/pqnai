// Package rag implements the embed/ask job handlers: turning text into
// vectors (stored via pgvector) and answering questions grounded in those
// stored vectors, using a self-hosted Ollama instance rather than a
// third-party cloud AI API.
package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// DefaultNumCtx is the context window (in tokens) requested for every
// generation call. The largest prompt pqnai builds is roughly 40 retrieved
// chunks of up to 500 characters (~5k tokens of English), so this leaves
// ample headroom; anything that still doesn't fit is rejected rather than
// truncated (see generate).
const DefaultNumCtx = 16384

type OllamaClient struct {
	BaseURL    string
	EmbedModel string
	ChatModel  string
	NumCtx     int
	HTTPClient *http.Client
}

func NewOllamaClient(baseURL, embedModel, chatModel string) *OllamaClient {
	return &OllamaClient{
		BaseURL:    baseURL,
		EmbedModel: embedModel,
		ChatModel:  chatModel,
		NumCtx:     DefaultNumCtx,
		HTTPClient: &http.Client{Timeout: 2 * time.Minute},
	}
}

type embedRequestBody struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type embedResponseBody struct {
	Embedding []float32 `json:"embedding"`
}

func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float32, error) {
	var resp embedResponseBody
	if err := c.postJSON(ctx, "/api/embeddings", embedRequestBody{Model: c.EmbedModel, Prompt: text}, &resp); err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	if len(resp.Embedding) == 0 {
		return nil, fmt.Errorf("ollama embed: empty embedding returned")
	}
	return resp.Embedding, nil
}

type generateOptions struct {
	NumCtx      int      `json:"num_ctx,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type generateRequestBody struct {
	Model    string          `json:"model"`
	Prompt   string          `json:"prompt"`
	Stream   bool            `json:"stream"`
	Truncate bool            `json:"truncate"`
	Format   json.RawMessage `json:"format,omitempty"`
	Options  generateOptions `json:"options"`
}

type generateResponseBody struct {
	Response string `json:"response"`
}

func (c *OllamaClient) Generate(ctx context.Context, prompt string) (string, error) {
	return c.generate(ctx, prompt, nil, nil)
}

// GenerateJSON constrains the output to schema (a JSON Schema) using
// Ollama's structured outputs, at temperature 0 so the same input is
// graded the same way every time.
func (c *OllamaClient) GenerateJSON(ctx context.Context, prompt string, schema json.RawMessage) (string, error) {
	zero := 0.0
	return c.generate(ctx, prompt, schema, &zero)
}

// generate sets an explicit context window and truncate=false on every
// call. Without them, Ollama silently drops tokens from the *start* of a
// prompt longer than its default window -- verified against Ollama 0.40:
// an 11.6k-token prompt was cut to ~2k tokens with no error, losing the
// instruction at its top. pqnai's prompts lead with their grounding
// instructions and highest-ranked context, the parts that must survive,
// so an over-long prompt now fails loudly instead.
func (c *OllamaClient) generate(ctx context.Context, prompt string, format json.RawMessage, temperature *float64) (string, error) {
	body := generateRequestBody{
		Model:    c.ChatModel,
		Prompt:   prompt,
		Stream:   false,
		Truncate: false,
		Format:   format,
		Options:  generateOptions{NumCtx: c.NumCtx, Temperature: temperature},
	}
	var resp generateResponseBody
	if err := c.postJSON(ctx, "/api/generate", body, &resp); err != nil {
		return "", fmt.Errorf("ollama generate: %w", err)
	}
	return resp.Response, nil
}

func (c *OllamaClient) postJSON(ctx context.Context, path string, body, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	return json.Unmarshal(respBody, out)
}
