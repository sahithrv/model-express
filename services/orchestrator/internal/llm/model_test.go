package llm

import (
	"testing"
	"time"
)

func TestConfigFromEnvUsesVisualModelFallback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", "")
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_BASE_URL", "http://visual-llm")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "visual-key")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "visual-model")
	t.Setenv("OPENAI_API_KEY", "openai-fallback-key")

	config := ConfigFromEnv(true, "", "")
	if config.Provider != ProviderLocal {
		t.Fatalf("expected visual provider fallback, got %q", config.Provider)
	}
	if config.BaseURL != "http://visual-llm" {
		t.Fatalf("expected visual base URL fallback, got %q", config.BaseURL)
	}
	if config.APIKey != "visual-key" {
		t.Fatalf("expected visual API key fallback, got %q", config.APIKey)
	}
	if config.APIKeySource != "MODEL_EXPRESS_VISUAL_LLM_API_KEY" {
		t.Fatalf("expected visual API key source, got %q", config.APIKeySource)
	}
	if config.Model != "visual-model" {
		t.Fatalf("expected visual model fallback, got %q", config.Model)
	}
}

func TestConfigFromEnvPrefersGeneralModelOverVisualFallback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "planner-model")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "visual-model")

	config := ConfigFromEnv(true, ProviderLocal, "")
	if config.Model != "planner-model" {
		t.Fatalf("expected general LLM model, got %q", config.Model)
	}
}

func TestConfigFromEnvUsesOpenAIAPIKeyFallbackForOpenAI(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "openai-fallback-key")

	config := ConfigFromEnv(true, ProviderOpenAI, "")
	if config.APIKey != "openai-fallback-key" {
		t.Fatalf("expected OPENAI_API_KEY fallback, got %q", config.APIKey)
	}
	if config.APIKeySource != "OPENAI_API_KEY" {
		t.Fatalf("expected OPENAI_API_KEY source, got %q", config.APIKeySource)
	}
}

func TestConfigFromEnvPrefersProductKeyOverOpenAIAPIKeyFallback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "product-key")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "visual-key")
	t.Setenv("OPENAI_API_KEY", "openai-fallback-key")

	config := ConfigFromEnv(true, ProviderOpenAI, "")
	if config.APIKey != "product-key" {
		t.Fatalf("expected product API key, got %q", config.APIKey)
	}
	if config.APIKeySource != "MODEL_EXPRESS_LLM_API_KEY" {
		t.Fatalf("expected product API key source, got %q", config.APIKeySource)
	}
}

func TestConfigFromEnvDoesNotUseOpenAIAPIKeyFallbackForNonOpenAI(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "openai-fallback-key")

	config := ConfigFromEnv(true, ProviderOpenAICompatible, "")
	if config.APIKey != "" {
		t.Fatalf("expected no OPENAI_API_KEY fallback for provider %q, got %q", config.Provider, config.APIKey)
	}
}

func TestConfigFromEnvDefaultsToMiniModel(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "")

	config := ConfigFromEnv(true, ProviderOpenAI, "")
	if config.Model != DefaultModel {
		t.Fatalf("expected default mini model %q, got %q", DefaultModel, config.Model)
	}
}

func TestConfigFromEnvReadsResponsesSettings(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_API_STYLE", "responses")
	t.Setenv("MODEL_EXPRESS_LLM_REASONING_EFFORT", "high")
	t.Setenv("MODEL_EXPRESS_LLM_PLATEAU_REASONING_EFFORT", "xhigh")
	t.Setenv("MODEL_EXPRESS_LLM_STORED_RESPONSES", "false")
	t.Setenv("MODEL_EXPRESS_LLM_MAX_TOOL_ROUNDS", "7")
	t.Setenv("MODEL_EXPRESS_LLM_MAX_RETRIES", "4")
	t.Setenv("MODEL_EXPRESS_LLM_TIMEOUT_SECONDS", "240")

	config := ConfigFromEnv(true, ProviderOpenAI, "test-model")
	if config.APIStyle != APIStyleResponses {
		t.Fatalf("expected responses API style, got %q", config.APIStyle)
	}
	if config.ReasoningEffort != ReasoningEffortHigh {
		t.Fatalf("expected high reasoning effort, got %q", config.ReasoningEffort)
	}
	if config.PlateauReasoningEffort != ReasoningEffortXHigh {
		t.Fatalf("expected xhigh plateau reasoning effort, got %q", config.PlateauReasoningEffort)
	}
	if config.StoredResponses {
		t.Fatalf("expected stored responses to be disabled")
	}
	if config.MaxToolRounds != 7 {
		t.Fatalf("expected max tool rounds 7, got %d", config.MaxToolRounds)
	}
	if config.MaxRetries != 4 {
		t.Fatalf("expected max retries 4, got %d", config.MaxRetries)
	}
	if config.Timeout != 240*time.Second {
		t.Fatalf("expected timeout 240s, got %s", config.Timeout)
	}
}

func TestConfigFromEnvDefaultsToLongerLLMTimeout(t *testing.T) {
	config := ConfigFromEnv(true, ProviderOpenAI, "test-model")

	if config.Timeout != DefaultTimeoutSeconds*time.Second {
		t.Fatalf("expected default timeout %ds, got %s", DefaultTimeoutSeconds, config.Timeout)
	}
}
