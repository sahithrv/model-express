package memory

import "time"

const (
	KindDatasetAnalysis             = "dataset_analysis"
	KindPreprocessingRecommendation = "preprocessing_recommendation"
	KindTrainingEvaluation          = "training_evaluation"
	KindPlanningFeedback            = "planning_feedback"
	KindModelRanking                = "model_ranking"
)

type AgentMemoryRecord struct {
	ID           string         `json:"id"`
	InvocationID string         `json:"invocation_id,omitempty"`
	ProjectID    string         `json:"project_id"`
	DatasetID    string         `json:"dataset_id,omitempty"`
	PlanID       string         `json:"plan_id,omitempty"`
	JobID        string         `json:"job_id,omitempty"`
	AgentName    string         `json:"agent_name"`
	Kind         string         `json:"kind"`
	Summary      string         `json:"summary"`
	Payload      map[string]any `json:"payload"`
	Tags         []string       `json:"tags"`
	CreatedAt    time.Time      `json:"created_at"`
}

type AgentMemoryFilter struct {
	DatasetID string
	PlanID    string
	JobID     string
	Kind      string
	Limit     int
}

const (
	InvocationValidationValid   = "valid"
	InvocationValidationInvalid = "invalid"
	InvocationValidationFailed  = "failed"
)

type AgentInvocation struct {
	ID                string              `json:"id"`
	ProjectID         string              `json:"project_id"`
	DatasetID         string              `json:"dataset_id,omitempty"`
	PlanID            string              `json:"plan_id,omitempty"`
	JobID             string              `json:"job_id,omitempty"`
	AgentName         string              `json:"agent_name"`
	AgentVersion      string              `json:"agent_version,omitempty"`
	PromptVersion     string              `json:"prompt_version,omitempty"`
	Provider          string              `json:"provider,omitempty"`
	Model             string              `json:"model,omitempty"`
	InputMessages     []map[string]string `json:"input_messages"`
	InputContext      map[string]any      `json:"input_context"`
	RawOutput         string              `json:"raw_output"`
	ParsedOutput      map[string]any      `json:"parsed_output"`
	ValidationStatus  string              `json:"validation_status"`
	ValidationError   string              `json:"validation_error,omitempty"`
	AcceptedForMemory bool                `json:"accepted_for_memory"`
	HumanFeedback     map[string]any      `json:"human_feedback"`
	DownstreamOutcome map[string]any      `json:"downstream_outcome"`
	CreatedAt         time.Time           `json:"created_at"`
}

type AgentInvocationFilter struct {
	DatasetID string
	PlanID    string
	JobID     string
	AgentName string
	Limit     int
}
