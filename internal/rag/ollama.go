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

type OllamaClient struct {
	BaseURL    string
	EmbedModel string
	ChatModel  string
	HTTPClient *http.Client
}

func NewOllamaClient(baseURL, embedModel, chatModel string) *OllamaClient {
	return &OllamaClient{
		BaseURL:    baseURL,
		EmbedModel: embedModel,
		ChatModel:  chatModel,
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

type generateRequestBody struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type generateResponseBody struct {
	Response string `json:"response"`
}

func (c *OllamaClient) Generate(ctx context.Context, prompt string) (string, error) {
	var resp generateResponseBody
	if err := c.postJSON(ctx, "/api/generate", generateRequestBody{Model: c.ChatModel, Prompt: prompt, Stream: false}, &resp); err != nil {
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
