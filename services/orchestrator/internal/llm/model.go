package llm

import (
	"context"
	"os"
	"strings"
	"time"
)

const (
	ProviderOpenAI           = "openai"
	ProviderOpenAICompatible = "openai_compatible"
	ProviderLocal            = "local"

	AgentModePropose    = "propose"
	AgentModeAutonomous = "autonomous"
)

type Config struct {
	Enabled  bool
	Provider string
	BaseURL  string
	APIKey   string
	Model    string
	Timeout  time.Duration
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type JSONRequest struct {
	Model       string
	Messages    []Message
	Temperature float64
}

type JSONGenerator interface {
	GenerateJSON(ctx context.Context, req JSONRequest) ([]byte, error)
}

func ConfigFromEnv(enabled bool, provider string, model string) Config {
	normalizedProvider := strings.ToLower(strings.TrimSpace(provider))
	if normalizedProvider == "" {
		normalizedProvider = strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_PROVIDER")))
	}
	if normalizedProvider == "" {
		normalizedProvider = ProviderOpenAI
	}

	selectedModel := strings.TrimSpace(model)
	if selectedModel == "" {
		selectedModel = strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_MODEL"))
	}
	if selectedModel == "" {
		selectedModel = defaultModelForProvider(normalizedProvider)
	}

	baseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_BASE_URL")), "/")
	if baseURL == "" && normalizedProvider == ProviderOpenAI {
		baseURL = "https://api.openai.com/v1"
	}

	return Config{
		Enabled:  enabled,
		Provider: normalizedProvider,
		BaseURL:  baseURL,
		APIKey:   strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_API_KEY")),
		Model:    selectedModel,
		Timeout:  45 * time.Second,
	}
}

func NormalizeAgentMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case AgentModeAutonomous:
		return AgentModeAutonomous
	default:
		return AgentModePropose
	}
}

func defaultModelForProvider(_ string) string {
	return ""
}
