package embeddings

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
)

type EmbedRequest struct {
	Model      string
	Text       string
	Dimensions int
}

type EmbedResult struct {
	Model      string
	Dimensions int
	Vector     []float32
}

type Client interface {
	Embed(ctx context.Context, req EmbedRequest) (EmbedResult, error)
}

func NewClient(config Config) Client {
	if !config.Normalized().EmbeddingsEnabled {
		return DisabledClient{Reason: ErrDisabled.Error()}
	}
	return NewOpenAICompatibleClient(config)
}

type DisabledClient struct {
	Reason string
}

func (c DisabledClient) Embed(_ context.Context, _ EmbedRequest) (EmbedResult, error) {
	reason := strings.TrimSpace(c.Reason)
	if reason == "" {
		reason = ErrDisabled.Error()
	}
	return EmbedResult{}, errorsText(reason)
}

type OpenAICompatibleClient struct {
	config     Config
	httpClient *http.Client
}

func NewOpenAICompatibleClient(config Config) *OpenAICompatibleClient {
	config = config.Normalized()
	return &OpenAICompatibleClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

func (c *OpenAICompatibleClient) Embed(ctx context.Context, req EmbedRequest) (EmbedResult, error) {
	if c == nil {
		return EmbedResult{}, fmt.Errorf("memory embedding client is not configured")
	}
	config := c.config.Normalized()
	if strings.TrimSpace(req.Model) != "" {
		config.Model = strings.TrimSpace(req.Model)
	}
	if req.Dimensions > 0 {
		config.Dimensions = req.Dimensions
	}
	if err := config.ReadyForIndexing(); err != nil {
		return EmbedResult{}, err
	}
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return EmbedResult{}, fmt.Errorf("memory embedding text is required")
	}

	body := embeddingRequest{
		Model:      config.Model,
		Input:      text,
		Dimensions: config.Dimensions,
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return EmbedResult{}, fmt.Errorf("marshal memory embedding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, config.BaseURL+"/embeddings", bytes.NewReader(encoded))
	if err != nil {
		return EmbedResult{}, fmt.Errorf("build memory embedding request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return EmbedResult{}, fmt.Errorf("call memory embedding provider: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return EmbedResult{}, fmt.Errorf("read memory embedding response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return EmbedResult{}, fmt.Errorf("memory embedding request failed: %s: %s", resp.Status, string(responseBody))
	}

	var decoded embeddingResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return EmbedResult{}, fmt.Errorf("decode memory embedding response: %w", err)
	}
	if decoded.Error != nil && strings.TrimSpace(decoded.Error.Message) != "" {
		return EmbedResult{}, fmt.Errorf("memory embedding provider error: %s", decoded.Error.Message)
	}
	if len(decoded.Data) == 0 || len(decoded.Data[0].Embedding) == 0 {
		return EmbedResult{}, fmt.Errorf("memory embedding response had no embedding data")
	}
	vector := append([]float32(nil), decoded.Data[0].Embedding...)
	if len(vector) != config.Dimensions {
		return EmbedResult{}, fmt.Errorf("memory embedding dimensions = %d, want %d", len(vector), config.Dimensions)
	}
	model := strings.TrimSpace(decoded.Model)
	if model == "" {
		model = config.Model
	}
	return EmbedResult{
		Model:      model,
		Dimensions: len(vector),
		Vector:     vector,
	}, nil
}

type FakeClient struct {
	Model      string
	Dimensions int
}

func NewFakeClient(dimensions int) FakeClient {
	return FakeClient{Model: "fake-memory-embedding", Dimensions: dimensions}
}

func (c FakeClient) Embed(ctx context.Context, req EmbedRequest) (EmbedResult, error) {
	select {
	case <-ctx.Done():
		return EmbedResult{}, ctx.Err()
	default:
	}
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return EmbedResult{}, fmt.Errorf("memory embedding text is required")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = strings.TrimSpace(c.Model)
	}
	if model == "" {
		model = "fake-memory-embedding"
	}
	dimensions := req.Dimensions
	if dimensions <= 0 {
		dimensions = c.Dimensions
	}
	if dimensions <= 0 {
		return EmbedResult{}, fmt.Errorf("memory embedding dimensions must be positive")
	}
	vector := deterministicVector(model, text, dimensions)
	return EmbedResult{
		Model:      model,
		Dimensions: dimensions,
		Vector:     vector,
	}, nil
}

type embeddingRequest struct {
	Model      string `json:"model"`
	Input      string `json:"input"`
	Dimensions int    `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type,omitempty"`
	} `json:"error,omitempty"`
}

func deterministicVector(model string, text string, dimensions int) []float32 {
	vector := make([]float32, dimensions)
	sumSquares := 0.0
	for i := range vector {
		seed := fmt.Sprintf("%s\x00%s\x00%d", model, text, i)
		hash := sha256.Sum256([]byte(seed))
		raw := binary.BigEndian.Uint32(hash[:4])
		scaled := (float64(raw)/float64(1<<32-1))*2 - 1
		vector[i] = float32(scaled)
		sumSquares += scaled * scaled
	}
	if sumSquares == 0 {
		vector[0] = 1
		return vector
	}
	norm := math.Sqrt(sumSquares)
	for i := range vector {
		vector[i] = float32(float64(vector[i]) / norm)
	}
	return vector
}

type errorsText string

func (e errorsText) Error() string {
	return string(e)
}
