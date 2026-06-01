package embeddings

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	ProviderOpenAI           = "openai"
	ProviderOpenAICompatible = "openai_compatible"
	ProviderLocal            = "local"

	DefaultEmbeddingDimensions = 1536
	DefaultBackfillLimit       = 500
	DefaultTimeout             = 30 * time.Second
	DefaultOpenAIBaseURL       = "https://api.openai.com/v1"
)

var ErrDisabled = errors.New("memory embeddings disabled")

type Config struct {
	RetrievalEnabled  bool
	EmbeddingsEnabled bool
	Provider          string
	Model             string
	BaseURL           string
	APIKey            string
	Dimensions        int
	BackfillLimit     int
	Timeout           time.Duration
}

func ConfigFromEnv() Config {
	config := Config{
		RetrievalEnabled:  envBoolDefault(false, "MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED"),
		EmbeddingsEnabled: envBoolDefault(false, "MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED"),
		Provider:          envStringDefault(ProviderOpenAI, "MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER"),
		Model:             envStringDefault("", "MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL"),
		BaseURL:           envStringDefault("", "MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL"),
		APIKey:            envStringDefault("", "MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY"),
		Dimensions:        envIntDefault(DefaultEmbeddingDimensions, "MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS"),
		BackfillLimit:     envIntDefault(DefaultBackfillLimit, "MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT"),
		Timeout:           DefaultTimeout,
	}
	return config.Normalized()
}

func (c Config) Normalized() Config {
	c.Provider = NormalizeProvider(c.Provider)
	c.Model = strings.TrimSpace(c.Model)
	c.BaseURL = strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	c.APIKey = strings.TrimSpace(c.APIKey)
	if c.BaseURL == "" && c.Provider == ProviderOpenAI {
		c.BaseURL = DefaultOpenAIBaseURL
	}
	if c.Dimensions <= 0 {
		c.Dimensions = DefaultEmbeddingDimensions
	}
	if c.BackfillLimit <= 0 {
		c.BackfillLimit = DefaultBackfillLimit
	}
	if c.Timeout <= 0 {
		c.Timeout = DefaultTimeout
	}
	return c
}

func (c Config) ReadyForIndexing() error {
	c = c.Normalized()
	if !c.EmbeddingsEnabled {
		return ErrDisabled
	}
	if c.Model == "" {
		return errors.New("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL is required when memory embeddings are enabled")
	}
	if c.Dimensions <= 0 {
		return errors.New("MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS must be positive")
	}
	switch c.Provider {
	case ProviderOpenAI:
		if c.BaseURL == "" {
			return fmt.Errorf("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL is required for provider %s", c.Provider)
		}
		if c.APIKey == "" {
			return fmt.Errorf("MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY is required for provider %s", c.Provider)
		}
	case ProviderOpenAICompatible, ProviderLocal:
		if c.BaseURL == "" {
			return fmt.Errorf("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL is required for provider %s", c.Provider)
		}
	default:
		return fmt.Errorf("unsupported memory embedding provider %q", c.Provider)
	}
	return nil
}

func NormalizeProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", ProviderOpenAI:
		return ProviderOpenAI
	case "openai-compatible", "openai compatible", ProviderOpenAICompatible:
		return ProviderOpenAICompatible
	case ProviderLocal:
		return ProviderLocal
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func envStringDefault(defaultValue string, name string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	return value
}

func envBoolDefault(defaultValue bool, name string) bool {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
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
	default:
		return defaultValue
	}
}

func envIntDefault(defaultValue int, name string) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return defaultValue
	}
	return parsed
}
