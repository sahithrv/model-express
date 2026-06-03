package llm

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	ProviderOpenAI           = "openai"
	ProviderOpenAICompatible = "openai_compatible"
	ProviderLocal            = "local"

	APIStyleChatCompletions = "chat_completions"
	APIStyleResponses       = "responses"

	ReasoningEffortNone    = "none"
	ReasoningEffortMinimal = "minimal"
	ReasoningEffortLow     = "low"
	ReasoningEffortMedium  = "medium"
	ReasoningEffortHigh    = "high"
	ReasoningEffortXHigh   = "xhigh"

	DefaultMaxToolRounds  = 4
	DefaultTimeoutSeconds = 180
	DefaultModel          = "gpt-5.4-mini"

	AgentModePropose    = "propose"
	AgentModeAutonomous = "autonomous"
)

type Config struct {
	Enabled                bool
	Provider               string
	BaseURL                string
	APIKey                 string
	Model                  string
	Timeout                time.Duration
	APIStyle               string
	ReasoningEffort        string
	PlateauReasoningEffort string
	StoredResponses        bool
	MaxToolRounds          int
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type JSONRequest struct {
	Model              string
	Messages           []Message
	Temperature        float64
	ReasoningEffort    string
	PreviousResponseID string
}

type Usage struct {
	InputTokens       int    `json:"input_tokens"`
	OutputTokens      int    `json:"output_tokens"`
	TotalTokens       int    `json:"total_tokens"`
	CachedInputTokens int    `json:"cached_input_tokens"`
	ReasoningTokens   int    `json:"reasoning_tokens"`
	RequestModel      string `json:"request_model,omitempty"`
	APIStyle          string `json:"api_style,omitempty"`
	ToolRounds        int    `json:"tool_rounds,omitempty"`
}

type JSONResult struct {
	RawJSON []byte
	Usage   *Usage
}

type JSONGenerator interface {
	GenerateJSON(ctx context.Context, req JSONRequest) ([]byte, error)
}

type JSONUsageGenerator interface {
	GenerateJSONWithUsage(ctx context.Context, req JSONRequest) (JSONResult, error)
}

func ConfigFromEnv(enabled bool, provider string, model string) Config {
	normalizedProvider := strings.ToLower(strings.TrimSpace(provider))
	if normalizedProvider == "" {
		normalizedProvider = strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_PROVIDER")))
	}
	if normalizedProvider == "" {
		normalizedProvider = strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER")))
	}
	if normalizedProvider == "" {
		normalizedProvider = ProviderOpenAI
	}

	selectedModel := strings.TrimSpace(model)
	if selectedModel == "" {
		selectedModel = strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_MODEL"))
	}
	if selectedModel == "" {
		selectedModel = strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL"))
	}
	if selectedModel == "" {
		selectedModel = defaultModelForProvider(normalizedProvider)
	}

	baseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_BASE_URL")), "/")
	if baseURL == "" {
		baseURL = strings.TrimRight(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_BASE_URL")), "/")
	}
	if baseURL == "" && normalizedProvider == ProviderOpenAI {
		baseURL = "https://api.openai.com/v1"
	}

	apiKey := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_API_KEY"))
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY"))
	}

	apiStyle := NormalizeAPIStyle(firstEnv("MODEL_EXPRESS_LLM_API_STYLE", "MODEL_EXPRESS_VISUAL_LLM_API_STYLE"))
	reasoningEffort := NormalizeReasoningEffort(firstEnv("MODEL_EXPRESS_LLM_REASONING_EFFORT", "MODEL_EXPRESS_VISUAL_LLM_REASONING_EFFORT"))
	plateauReasoningEffort := NormalizeReasoningEffort(firstEnv("MODEL_EXPRESS_LLM_PLATEAU_REASONING_EFFORT", "MODEL_EXPRESS_VISUAL_LLM_PLATEAU_REASONING_EFFORT"))
	timeoutSeconds := envIntDefault(DefaultTimeoutSeconds, "MODEL_EXPRESS_LLM_TIMEOUT_SECONDS", "MODEL_EXPRESS_VISUAL_LLM_TIMEOUT_SECONDS")

	return Config{
		Enabled:                enabled,
		Provider:               normalizedProvider,
		BaseURL:                baseURL,
		APIKey:                 apiKey,
		Model:                  selectedModel,
		Timeout:                time.Duration(timeoutSeconds) * time.Second,
		APIStyle:               apiStyle,
		ReasoningEffort:        reasoningEffort,
		PlateauReasoningEffort: plateauReasoningEffort,
		StoredResponses:        envBoolDefault(true, "MODEL_EXPRESS_LLM_STORED_RESPONSES", "MODEL_EXPRESS_VISUAL_LLM_STORED_RESPONSES"),
		MaxToolRounds:          envIntDefault(DefaultMaxToolRounds, "MODEL_EXPRESS_LLM_MAX_TOOL_ROUNDS", "MODEL_EXPRESS_VISUAL_LLM_MAX_TOOL_ROUNDS"),
	}
}

func NormalizeAPIStyle(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case APIStyleResponses:
		return APIStyleResponses
	case "chat", "chat-completions", APIStyleChatCompletions:
		return APIStyleChatCompletions
	default:
		return APIStyleChatCompletions
	}
}

func NormalizeReasoningEffort(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ReasoningEffortNone:
		return ReasoningEffortNone
	case ReasoningEffortMinimal:
		return ReasoningEffortMinimal
	case ReasoningEffortLow:
		return ReasoningEffortLow
	case ReasoningEffortMedium:
		return ReasoningEffortMedium
	case ReasoningEffortHigh:
		return ReasoningEffortHigh
	case ReasoningEffortXHigh:
		return ReasoningEffortXHigh
	default:
		return ""
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

func firstEnv(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

func envBoolDefault(defaultValue bool, names ...string) bool {
	for _, name := range names {
		value := strings.TrimSpace(os.Getenv(name))
		if value == "" {
			continue
		}
		parsed, err := strconv.ParseBool(value)
		if err == nil {
			return parsed
		}
		switch strings.ToLower(value) {
		case "on", "yes", "y":
			return true
		case "off", "no", "n":
			return false
		}
	}
	return defaultValue
}

func envIntDefault(defaultValue int, names ...string) int {
	for _, name := range names {
		value := strings.TrimSpace(os.Getenv(name))
		if value == "" {
			continue
		}
		parsed, err := strconv.Atoi(value)
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultValue
}

func defaultModelForProvider(_ string) string {
	return DefaultModel
}
