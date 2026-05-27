package llm

import "testing"

func TestConfigFromEnvUsesVisualModelFallback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", "")
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_BASE_URL", "http://visual-llm")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "visual-key")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "visual-model")

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
