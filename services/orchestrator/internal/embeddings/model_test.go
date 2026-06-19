package embeddings

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestConfigFromEnvReadsMemoryEmbeddingSettings(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "openai_compatible")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "text-embedding-test")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL", "http://embedding.local/v1/")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY", "test-key")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS", "8")
	t.Setenv("MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT", "42")

	config := ConfigFromEnv()

	if !config.RetrievalEnabled || !config.EmbeddingsEnabled {
		t.Fatalf("expected retrieval and embeddings flags to be enabled")
	}
	if config.Provider != ProviderOpenAICompatible {
		t.Fatalf("Provider = %q, want %q", config.Provider, ProviderOpenAICompatible)
	}
	if config.Model != "text-embedding-test" {
		t.Fatalf("Model = %q", config.Model)
	}
	if config.BaseURL != "http://embedding.local/v1" {
		t.Fatalf("BaseURL = %q", config.BaseURL)
	}
	if config.APIKey != "test-key" {
		t.Fatalf("APIKey = %q", config.APIKey)
	}
	if config.APIKeySource != "MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY" {
		t.Fatalf("APIKeySource = %q", config.APIKeySource)
	}
	if config.Dimensions != 8 {
		t.Fatalf("Dimensions = %d, want 8", config.Dimensions)
	}
	if config.BackfillLimit != 42 {
		t.Fatalf("BackfillLimit = %d, want 42", config.BackfillLimit)
	}
	if config.Timeout != DefaultTimeout {
		t.Fatalf("Timeout = %s, want %s", config.Timeout, DefaultTimeout)
	}
}

func TestConfigFromEnvDefaultsDisabledAndSafe(t *testing.T) {
	clearMemoryEmbeddingEnv(t)

	config := ConfigFromEnv()

	if config.RetrievalEnabled || config.EmbeddingsEnabled {
		t.Fatalf("memory retrieval and embeddings should default disabled")
	}
	if config.Provider != ProviderOpenAI {
		t.Fatalf("Provider = %q, want %q", config.Provider, ProviderOpenAI)
	}
	if config.BaseURL != DefaultOpenAIBaseURL {
		t.Fatalf("BaseURL = %q, want %q", config.BaseURL, DefaultOpenAIBaseURL)
	}
	if config.Dimensions != DefaultEmbeddingDimensions {
		t.Fatalf("Dimensions = %d, want %d", config.Dimensions, DefaultEmbeddingDimensions)
	}
	if config.BackfillLimit != DefaultBackfillLimit {
		t.Fatalf("BackfillLimit = %d, want %d", config.BackfillLimit, DefaultBackfillLimit)
	}
	if !errors.Is(config.ReadyForIndexing(), ErrDisabled) {
		t.Fatalf("ReadyForIndexing() should return ErrDisabled when embeddings are off")
	}
}

func TestConfigFromEnvUsesOpenAIAPIKeyFallbackForOpenAIProvider(t *testing.T) {
	clearMemoryEmbeddingEnv(t)
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "openai")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
	t.Setenv("OPENAI_API_KEY", "sk-memory-fallback")

	config := ConfigFromEnv()

	if config.APIKey != "sk-memory-fallback" {
		t.Fatalf("APIKey = %q", config.APIKey)
	}
	if config.APIKeySource != "OPENAI_API_KEY" {
		t.Fatalf("APIKeySource = %q", config.APIKeySource)
	}
	if err := config.ReadyForIndexing(); err != nil {
		t.Fatalf("ReadyForIndexing() = %v", err)
	}
}

func TestConfigFromEnvDoesNotUseOpenAIAPIKeyFallbackForCompatibleProvider(t *testing.T) {
	clearMemoryEmbeddingEnv(t)
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "openai_compatible")
	t.Setenv("OPENAI_API_KEY", "sk-memory-fallback")

	config := ConfigFromEnv()

	if config.APIKey != "" {
		t.Fatalf("APIKey = %q, want empty for openai_compatible fallback", config.APIKey)
	}
	if config.APIKeySource != "" {
		t.Fatalf("APIKeySource = %q, want empty", config.APIKeySource)
	}
}

func clearMemoryEmbeddingEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED",
		"MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED",
		"MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER",
		"MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL",
		"MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL",
		"MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY",
		"MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS",
		"MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT",
		"OPENAI_API_KEY",
	} {
		t.Setenv(name, "")
	}
}

func TestReadyForIndexingRequiresConfiguredOpenAIModelAndKey(t *testing.T) {
	missingModel := Config{
		EmbeddingsEnabled: true,
		Provider:          ProviderOpenAI,
		BaseURL:           DefaultOpenAIBaseURL,
		APIKey:            "test-key",
		Dimensions:        4,
	}
	if err := missingModel.ReadyForIndexing(); err == nil || !strings.Contains(err.Error(), "MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL") {
		t.Fatalf("ReadyForIndexing() missing model error = %v", err)
	}

	missingKey := Config{
		EmbeddingsEnabled: true,
		Provider:          ProviderOpenAI,
		Model:             "text-embedding-test",
		BaseURL:           DefaultOpenAIBaseURL,
		Dimensions:        4,
	}
	if err := missingKey.ReadyForIndexing(); err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("ReadyForIndexing() missing key error = %v", err)
	}
}

func TestNormalizeProvider(t *testing.T) {
	cases := map[string]string{
		"":                  ProviderOpenAI,
		"OPENAI":            ProviderOpenAI,
		"openai-compatible": ProviderOpenAICompatible,
		"openai compatible": ProviderOpenAICompatible,
		"LOCAL":             ProviderLocal,
	}
	for input, want := range cases {
		if got := NormalizeProvider(input); got != want {
			t.Fatalf("NormalizeProvider(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestConfigNormalizedAppliesPositiveDefaults(t *testing.T) {
	config := Config{
		Provider:      ProviderOpenAI,
		Dimensions:    -1,
		BackfillLimit: -1,
		Timeout:       -time.Second,
	}.Normalized()

	if config.Dimensions != DefaultEmbeddingDimensions {
		t.Fatalf("Dimensions = %d, want default", config.Dimensions)
	}
	if config.BackfillLimit != DefaultBackfillLimit {
		t.Fatalf("BackfillLimit = %d, want default", config.BackfillLimit)
	}
	if config.Timeout != DefaultTimeout {
		t.Fatalf("Timeout = %s, want default", config.Timeout)
	}
}
