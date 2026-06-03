package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"time"
)

const (
	EmbeddingUsagePurposeSourceIndex    = "source_index"
	EmbeddingUsagePurposeRetrievalQuery = "retrieval_query"
)

type MemoryEmbeddingSourceState struct {
	SourceTable         string    `json:"source_table"`
	SourceID            string    `json:"source_id"`
	ProjectID           string    `json:"project_id"`
	EmbeddingModel      string    `json:"embedding_model"`
	EmbeddingDimensions int       `json:"embedding_dimensions"`
	EmbeddingTextHash   string    `json:"embedding_text_hash"`
	UpdatedAt           time.Time `json:"updated_at"`
}

type MemoryEmbeddingUsageEvent struct {
	ID                  string         `json:"id"`
	ProjectID           string         `json:"project_id"`
	DatasetID           string         `json:"dataset_id,omitempty"`
	PlanID              string         `json:"plan_id,omitempty"`
	JobID               string         `json:"job_id,omitempty"`
	InvocationID        string         `json:"invocation_id,omitempty"`
	Purpose             string         `json:"purpose"`
	RetrievalPurpose    string         `json:"retrieval_purpose,omitempty"`
	SourceTable         string         `json:"source_table,omitempty"`
	SourceID            string         `json:"source_id,omitempty"`
	EmbeddingModel      string         `json:"embedding_model,omitempty"`
	EmbeddingDimensions int            `json:"embedding_dimensions,omitempty"`
	InputBytes          int            `json:"input_bytes,omitempty"`
	ProviderCallCount   int            `json:"provider_call_count,omitempty"`
	RetrievedCount      int            `json:"retrieved_count,omitempty"`
	Injected            bool           `json:"injected,omitempty"`
	LogOnly             bool           `json:"log_only,omitempty"`
	Cached              bool           `json:"cached,omitempty"`
	Skipped             bool           `json:"skipped,omitempty"`
	SkipReason          string         `json:"skip_reason,omitempty"`
	SourceTextHash      string         `json:"source_text_hash,omitempty"`
	QueryHash           string         `json:"query_hash,omitempty"`
	ProviderUsage       map[string]any `json:"provider_usage,omitempty"`
	Metadata            map[string]any `json:"metadata,omitempty"`
	CreatedAt           time.Time      `json:"created_at"`
}

type MemoryRetrievalQueryCacheRecord struct {
	ID                  string    `json:"id"`
	ProjectID           string    `json:"project_id"`
	DatasetID           string    `json:"dataset_id,omitempty"`
	Purpose             string    `json:"purpose"`
	EmbeddingModel      string    `json:"embedding_model"`
	EmbeddingDimensions int       `json:"embedding_dimensions"`
	NormalizedQueryHash string    `json:"normalized_query_hash"`
	QueryText           string    `json:"query_text"`
	Embedding           []float32 `json:"embedding"`
	CreatedAt           time.Time `json:"created_at"`
	UpdatedAt           time.Time `json:"updated_at"`
	ExpiresAt           time.Time `json:"expires_at"`
}

func NormalizeEmbeddingText(text string) string {
	text = strings.TrimSpace(strings.ToLower(text))
	if text == "" {
		return ""
	}
	return strings.Join(strings.Fields(text), " ")
}

func HashEmbeddingText(text string) string {
	normalized := NormalizeEmbeddingText(text)
	if normalized == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:])
}

func NormalizeRetrievalQueryText(text string) string {
	return NormalizeEmbeddingText(text)
}

func HashRetrievalQueryText(text string) string {
	normalized := NormalizeRetrievalQueryText(text)
	if normalized == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:])
}
