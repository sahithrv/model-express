package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type OpenAICompatibleClient struct {
	config     Config
	httpClient *http.Client
}

func NewOpenAICompatibleClient(config Config) OpenAICompatibleClient {
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = ConfigFromEnv(config.Enabled, config.Provider, config.Model).Timeout
	}
	return OpenAICompatibleClient{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c OpenAICompatibleClient) GenerateJSON(ctx context.Context, req JSONRequest) ([]byte, error) {
	if !c.config.Enabled {
		return nil, fmt.Errorf("llm is disabled")
	}
	if c.config.BaseURL == "" {
		return nil, fmt.Errorf("MODEL_EXPRESS_LLM_BASE_URL is required for provider %s", c.config.Provider)
	}
	if c.config.Model == "" && req.Model == "" {
		return nil, fmt.Errorf("MODEL_EXPRESS_LLM_MODEL is required")
	}
	if c.config.APIKey == "" && c.config.Provider != ProviderLocal {
		return nil, fmt.Errorf("MODEL_EXPRESS_LLM_API_KEY is required for provider %s", c.config.Provider)
	}

	model := req.Model
	if model == "" {
		model = c.config.Model
	}

	body := chatCompletionRequest{
		Model:       model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal llm request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.config.BaseURL+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("build llm request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call llm: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read llm response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("llm request failed: %s: %s", resp.Status, string(responseBody))
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return nil, fmt.Errorf("decode llm response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("llm response had no choices")
	}

	content := strings.TrimSpace(decoded.Choices[0].Message.Content)
	if content == "" {
		return nil, fmt.Errorf("llm response content was empty")
	}

	return []byte(content), nil
}

type chatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}
