package memory

import (
	"time"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/plannervalidation"
)

const (
	KindDatasetAnalysis             = "dataset_analysis"
	KindPreprocessingRecommendation = "preprocessing_recommendation"
	KindTrainingEvaluation          = "training_evaluation"
	KindPlanningFeedback            = "planning_feedback"
	KindPlanningOutcome             = "planning_outcome"
	KindModelRanking                = "model_ranking"
	KindChampionFeedback            = "champion_feedback"
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
	ID                      string                                `json:"id"`
	ProjectID               string                                `json:"project_id"`
	DatasetID               string                                `json:"dataset_id,omitempty"`
	PlanID                  string                                `json:"plan_id,omitempty"`
	JobID                   string                                `json:"job_id,omitempty"`
	AgentName               string                                `json:"agent_name"`
	AgentVersion            string                                `json:"agent_version,omitempty"`
	PromptVersion           string                                `json:"prompt_version,omitempty"`
	PlannerVariantID        string                                `json:"planner_variant_id"`
	PlannerVariant          *PlannerVariant                       `json:"planner_variant,omitempty"`
	RolloutCohortID         string                                `json:"rollout_cohort_id"`
	RolloutPolicyID         string                                `json:"rollout_policy_id"`
	RolloutAssignment       *calibration.PlannerRolloutAssignment `json:"rollout_assignment,omitempty"`
	ValidationMode          string                                `json:"validation_mode,omitempty"`
	AttemptGroupID          string                                `json:"attempt_group_id,omitempty"`
	AttemptIndex            int                                   `json:"attempt_index"`
	RetryReason             string                                `json:"retry_reason,omitempty"`
	WallLatencyMS           float64                               `json:"wall_latency_ms"`
	ProviderUsage           map[string]any                        `json:"provider_usage,omitempty"`
	DerivedCost             *PlannerInvocationCost                `json:"derived_cost,omitempty"`
	Provider                string                                `json:"provider,omitempty"`
	Model                   string                                `json:"model,omitempty"`
	InputMessages           []map[string]string                   `json:"input_messages"`
	InputContext            map[string]any                        `json:"input_context"`
	RawOutput               string                                `json:"raw_output"`
	ParsedOutput            map[string]any                        `json:"parsed_output"`
	ValidationStatus        string                                `json:"validation_status"`
	ValidationError         string                                `json:"validation_error,omitempty"`
	StrictValidationVerdict *plannervalidation.Verdict            `json:"strict_validation_verdict,omitempty"`
	ValidationOutcome       *plannervalidation.Outcome            `json:"validation_outcome,omitempty"`
	AcceptedForMemory       bool                                  `json:"accepted_for_memory"`
	HumanFeedback           map[string]any                        `json:"human_feedback"`
	DownstreamOutcome       map[string]any                        `json:"downstream_outcome"`
	CreatedAt               time.Time                             `json:"created_at"`
}

type AgentInvocationFilter struct {
	DatasetID        string
	PlanID           string
	JobID            string
	AgentName        string
	PlannerVariantID string
	Limit            int
}

// AgentInvocationActivity is the bounded read model used by the live activity
// feed. It deliberately excludes prompts, context, and model output fields.
type AgentInvocationActivity struct {
	ID                string         `json:"id"`
	ProjectID         string         `json:"project_id"`
	PlanID            string         `json:"plan_id,omitempty"`
	JobID             string         `json:"job_id,omitempty"`
	AgentName         string         `json:"agent_name"`
	ValidationStatus  string         `json:"validation_status"`
	ValidationError   string         `json:"validation_error,omitempty"`
	DownstreamOutcome map[string]any `json:"downstream_outcome"`
	CreatedAt         time.Time      `json:"created_at"`
}
