package execution

import "time"

const (
	WorkerRequirementPending   = "PENDING"
	WorkerRequirementStarting  = "STARTING"
	WorkerRequirementActive    = "ACTIVE"
	WorkerRequirementSatisfied = "SATISFIED"
	WorkerRequirementFailed    = "FAILED"
	WorkerRequirementCancelled = "CANCELLED"
)

const (
	DatasetMaterializationUnknown       = "UNKNOWN"
	DatasetMaterializationCold          = "COLD"
	DatasetMaterializationMaterializing = "MATERIALIZING"
	DatasetMaterializationWarm          = "WARM"
	DatasetMaterializationStagingOnly   = "STAGING_ONLY"

	ColdCachePolicySingleMaterialization = "single_materialization_per_checksum"
)

const (
	EventJobsQueued                  = "JOBS_QUEUED"
	EventWorkersRequired             = "WORKERS_REQUIRED"
	EventWorkerScalingUpdated        = "WORKER_SCALING_UPDATED"
	EventWorkersStarting             = "WORKERS_STARTING"
	EventWorkersActive               = "WORKERS_ACTIVE"
	EventDispatcherStatus            = "DISPATCHER_STATUS"
	EventDispatcherIdleExit          = "DISPATCHER_IDLE_EXIT"
	EventChampionSelected            = "CHAMPION_SELECTED"
	EventChampionExportRequested     = "CHAMPION_EXPORT_REQUESTED"
	EventChampionDemoPrediction      = "CHAMPION_DEMO_PREDICTION"
	EventChampionFeedbackRecorded    = "CHAMPION_FEEDBACK_RECORDED"
	EventJobRetryQueued              = "JOB_RETRY_QUEUED"
	EventCostBudgetBlocked           = "COST_BUDGET_BLOCKED"
	EventDatasetVisualAnalysisQueued = "DATASET_VISUAL_ANALYSIS_QUEUED"
	EventDatasetVisualAnalysisResult = "DATASET_VISUAL_ANALYSIS_RESULT"
	EventExperimentationReopened     = "EXPERIMENTATION_REOPENED"
	EventExecutionFailed             = "EXECUTION_FAILED"
	EventMemoryRetrievalLogged       = "MEMORY_RETRIEVAL_LOGGED"
	EventAgentStarted                = "AGENT_STARTED"
	EventAgentRecommendationRecorded = "AGENT_RECOMMENDATION_RECORDED"
	EventAgentOutcomeRecorded        = "AGENT_OUTCOME_RECORDED"
	EventAgentFailed                 = "AGENT_FAILED"
)

type WorkerRequirement struct {
	ID                             string    `json:"id"`
	ProjectID                      string    `json:"project_id"`
	PlanID                         string    `json:"plan_id"`
	Provider                       string    `json:"provider"`
	GPUType                        string    `json:"gpu_type"`
	TargetCount                    int       `json:"target_count"`
	Status                         string    `json:"status"`
	Source                         string    `json:"source"`
	DatasetID                      string    `json:"dataset_id,omitempty"`
	DatasetChecksum                string    `json:"dataset_checksum,omitempty"`
	DatasetCacheKey                string    `json:"dataset_cache_key,omitempty"`
	DatasetMaterializationStatus   string    `json:"dataset_materialization_status,omitempty"`
	ColdCachePolicy                string    `json:"cold_cache_policy,omitempty"`
	MaxConcurrentJobs              int       `json:"max_concurrent_jobs,omitempty"`
	MaxColdDatasetMaterializations int       `json:"max_cold_dataset_materializations,omitempty"`
	LastError                      string    `json:"last_error,omitempty"`
	CreatedAt                      time.Time `json:"created_at"`
	UpdatedAt                      time.Time `json:"updated_at"`
}

type WorkerRequirementUpdate struct {
	Status                       *string `json:"status"`
	LastError                    *string `json:"last_error"`
	DatasetMaterializationStatus *string `json:"dataset_materialization_status"`
}

type WorkerRequirementPolicy struct {
	DatasetID                      string
	DatasetChecksum                string
	DatasetCacheKey                string
	DatasetMaterializationStatus   string
	ColdCachePolicy                string
	MaxConcurrentJobs              int
	MaxColdDatasetMaterializations int
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
