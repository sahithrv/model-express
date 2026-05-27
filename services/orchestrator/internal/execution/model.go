package execution

import "time"

const (
	WorkerRequirementPending   = "PENDING"
	WorkerRequirementStarting  = "STARTING"
	WorkerRequirementActive    = "ACTIVE"
	WorkerRequirementFailed    = "FAILED"
	WorkerRequirementCancelled = "CANCELLED"
)

const (
	EventJobsQueued                  = "JOBS_QUEUED"
	EventWorkersRequired             = "WORKERS_REQUIRED"
	EventWorkerScalingUpdated        = "WORKER_SCALING_UPDATED"
	EventWorkersStarting             = "WORKERS_STARTING"
	EventWorkersActive               = "WORKERS_ACTIVE"
	EventChampionSelected            = "CHAMPION_SELECTED"
	EventChampionExportRequested     = "CHAMPION_EXPORT_REQUESTED"
	EventChampionDemoPrediction      = "CHAMPION_DEMO_PREDICTION"
	EventDatasetVisualAnalysisQueued = "DATASET_VISUAL_ANALYSIS_QUEUED"
	EventDatasetVisualAnalysisResult = "DATASET_VISUAL_ANALYSIS_RESULT"
	EventExperimentationReopened     = "EXPERIMENTATION_REOPENED"
	EventExecutionFailed             = "EXECUTION_FAILED"
	EventAgentRecommendationRecorded = "AGENT_RECOMMENDATION_RECORDED"
	EventAgentOutcomeRecorded        = "AGENT_OUTCOME_RECORDED"
	EventAgentFailed                 = "AGENT_FAILED"
)

type WorkerRequirement struct {
	ID          string    `json:"id"`
	ProjectID   string    `json:"project_id"`
	PlanID      string    `json:"plan_id"`
	Provider    string    `json:"provider"`
	GPUType     string    `json:"gpu_type"`
	TargetCount int       `json:"target_count"`
	Status      string    `json:"status"`
	Source      string    `json:"source"`
	LastError   string    `json:"last_error,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type WorkerRequirementUpdate struct {
	Status    *string `json:"status"`
	LastError *string `json:"last_error"`
}

type ExecutionEvent struct {
	ID        string         `json:"id"`
	ProjectID string         `json:"project_id"`
	PlanID    string         `json:"plan_id,omitempty"`
	EventType string         `json:"event_type"`
	Message   string         `json:"message"`
	Payload   map[string]any `json:"payload"`
	CreatedAt time.Time      `json:"created_at"`
}
