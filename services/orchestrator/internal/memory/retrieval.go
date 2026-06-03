package memory

import "time"

const MemoryCardVersion = "memory_card_v1"

const (
	SourceAgentMemoryRecord = "agent_memory_records"
	SourceStrategyScorecard = "strategy_scorecards"
)

const (
	ScopeProject = "project"
	ScopeDataset = "dataset"
	ScopePlan    = "plan"
	ScopeJob     = "job"
)

type EmbeddableMemoryCard struct {
	CardVersion  string         `json:"card_version"`
	SourceTable  string         `json:"source_table"`
	SourceID     string         `json:"source_id"`
	ProjectID    string         `json:"project_id"`
	DatasetID    string         `json:"dataset_id,omitempty"`
	PlanID       string         `json:"plan_id,omitempty"`
	JobID        string         `json:"job_id,omitempty"`
	InvocationID string         `json:"invocation_id,omitempty"`
	Kind         string         `json:"kind"`
	Scope        string         `json:"scope"`
	Text         string         `json:"text"`
	SummaryCard  map[string]any `json:"summary_card"`
	Metadata     map[string]any `json:"metadata"`
	QualityScore float64        `json:"quality_score"`
	OutcomeScore float64        `json:"outcome_score"`
}

type MemoryRetrievalQuery struct {
	ProjectID           string    `json:"project_id"`
	DatasetID           string    `json:"dataset_id,omitempty"`
	AgentName           string    `json:"agent_name,omitempty"`
	Purpose             string    `json:"purpose,omitempty"`
	Text                string    `json:"text"`
	EmbeddingModel      string    `json:"embedding_model,omitempty"`
	EmbeddingDimensions int       `json:"embedding_dimensions,omitempty"`
	Embedding           []float32 `json:"embedding,omitempty"`
	Kinds               []string  `json:"kinds,omitempty"`
	Mechanisms          []string  `json:"mechanisms,omitempty"`
	DatasetTraits       []string  `json:"dataset_traits,omitempty"`
	Objective           string    `json:"objective,omitempty"`
	Limit               int       `json:"limit"`
	CrossProjectOK      bool      `json:"cross_project_ok,omitempty"`
}

type MemoryRetrievalResult struct {
	SourceTable     string         `json:"source_table"`
	SourceID        string         `json:"source_id"`
	ProjectID       string         `json:"project_id"`
	DatasetID       string         `json:"dataset_id,omitempty"`
	Kind            string         `json:"kind"`
	Score           float64        `json:"score"`
	SemanticScore   float64        `json:"semantic_score"`
	StructuredScore float64        `json:"structured_score"`
	RetrievalReason string         `json:"retrieval_reason"`
	SummaryCard     map[string]any `json:"summary_card"`
	Metadata        map[string]any `json:"metadata"`
}

type MemoryEmbeddingRecord struct {
	ID                  string         `json:"id"`
	SourceTable         string         `json:"source_table"`
	SourceID            string         `json:"source_id"`
	ProjectID           string         `json:"project_id"`
	DatasetID           string         `json:"dataset_id,omitempty"`
	PlanID              string         `json:"plan_id,omitempty"`
	JobID               string         `json:"job_id,omitempty"`
	InvocationID        string         `json:"invocation_id,omitempty"`
	Kind                string         `json:"kind"`
	Scope               string         `json:"scope"`
	EmbeddingModel      string         `json:"embedding_model"`
	EmbeddingDimensions int            `json:"embedding_dimensions"`
	Embedding           []float32      `json:"embedding"`
	EmbeddingText       string         `json:"embedding_text"`
	EmbeddingTextHash   string         `json:"embedding_text_hash,omitempty"`
	SummaryCard         map[string]any `json:"summary_card"`
	Metadata            map[string]any `json:"metadata"`
	QualityScore        float64        `json:"quality_score"`
	OutcomeScore        float64        `json:"outcome_score"`
	CreatedAt           time.Time      `json:"created_at"`
	UpdatedAt           time.Time      `json:"updated_at"`
}
