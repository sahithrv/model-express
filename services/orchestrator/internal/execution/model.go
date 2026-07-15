package execution

import "time"

const (
	ExecutionLifecyclePending     = "PENDING"
	ExecutionLifecycleInitialized = "INITIALIZED"
	ExecutionLifecycleFinalized   = "FINALIZED"
	ExecutionLifecycleNotRealized = "NOT_REALIZED"

	ExecutionVerdictMatched            = "MATCHED"
	ExecutionVerdictApprovedAdjustment = "APPROVED_ADJUSTMENT"
	ExecutionVerdictMismatch           = "MISMATCH"
	ExecutionVerdictSimulated          = "SIMULATED"
	ExecutionVerdictUnverified         = "UNVERIFIED"

	ExecutionAdjustmentReasonBatchSizeReduced = "batch_size_reduced_by_resource_recovery"

	ExecutionObservationInitialized = "INITIALIZED"
	ExecutionObservationFinalized   = "FINALIZED"
	ExecutionObservationSchemaV1    = "execution_realization_v1"
)

// JobExecutionSpec is the immutable, job-scoped snapshot accepted before a job
// can be assigned. AcceptedSpec is intentionally independent from mutable job
// configuration and attempt/resource metadata.
type JobExecutionSpec struct {
	JobID               string         `json:"job_id"`
	ProjectID           string         `json:"project_id"`
	SchemaVersion       string         `json:"schema_version"`
	CapabilityVersion   string         `json:"capability_version"`
	Task                string         `json:"task"`
	Runner              string         `json:"runner"`
	RequestedConfigHash string         `json:"requested_config_hash"`
	AcceptedSpecHash    string         `json:"accepted_spec_hash"`
	AcceptedSpec        map[string]any `json:"accepted_spec"`
	PolicyEvaluationID  string         `json:"policy_evaluation_id,omitempty"`
	EffectivePolicyHash string         `json:"effective_policy_hash,omitempty"`
	CreatedAt           time.Time      `json:"created_at"`
}

type AttemptExecutionRecord struct {
	ID                         string                   `json:"id"`
	JobID                      string                   `json:"job_id"`
	ProjectID                  string                   `json:"project_id"`
	AttemptID                  string                   `json:"attempt_id"`
	AttemptNumber              int                      `json:"attempt_number"`
	LifecycleStatus            string                   `json:"lifecycle_status"`
	FidelityVerdict            *string                  `json:"fidelity_verdict"`
	RealizedEffectiveHash      string                   `json:"realized_effective_hash,omitempty"`
	AdjustmentReasonCodes      []string                 `json:"adjustment_reason_codes,omitempty"`
	LatestRealizedConfig       map[string]any           `json:"latest_realized_config,omitempty"`
	DispatchPolicyEvaluationID string                   `json:"dispatch_policy_evaluation_id,omitempty"`
	EffectivePolicyHash        string                   `json:"effective_policy_hash,omitempty"`
	CreatedAt                  time.Time                `json:"created_at"`
	UpdatedAt                  time.Time                `json:"updated_at"`
	Observations               []RealizationObservation `json:"observations,omitempty"`
}

type RealizationObservation struct {
	ID                    string         `json:"id"`
	AttemptRecordID       string         `json:"attempt_record_id"`
	AttemptID             string         `json:"attempt_id"`
	SchemaVersion         string         `json:"schema_version"`
	Stage                 string         `json:"stage"`
	IdempotencyKey        string         `json:"idempotency_key"`
	RealizedConfig        map[string]any `json:"realized_config"`
	FrameworkArguments    map[string]any `json:"framework_arguments,omitempty"`
	Evidence              map[string]any `json:"evidence,omitempty"`
	AdjustmentPolicy      string         `json:"adjustment_policy,omitempty"`
	AdjustmentReasonCodes []string       `json:"adjustment_reason_codes,omitempty"`
	Simulated             bool           `json:"simulated,omitempty"`
	RealizedEffectiveHash string         `json:"realized_effective_hash"`
	FidelityVerdict       string         `json:"fidelity_verdict"`
	CreatedAt             time.Time      `json:"created_at"`
}

type RealizationObservationCreate struct {
	AttemptID          string
	SchemaVersion      string
	Stage              string
	IdempotencyKey     string
	RealizedConfig     map[string]any
	FrameworkArguments map[string]any
	Evidence           map[string]any
	AdjustmentPolicy   string
	Simulated          bool
}

type ExecutionRecord struct {
	AcceptedSpec JobExecutionSpec         `json:"accepted_spec"`
	Attempts     []AttemptExecutionRecord `json:"attempts"`
}

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
	EventJobsQueued                     = "JOBS_QUEUED"
	EventWorkersRequired                = "WORKERS_REQUIRED"
	EventWorkerScalingUpdated           = "WORKER_SCALING_UPDATED"
	EventWorkersStarting                = "WORKERS_STARTING"
	EventWorkersActive                  = "WORKERS_ACTIVE"
	EventDispatcherStatus               = "DISPATCHER_STATUS"
	EventDispatcherIdleExit             = "DISPATCHER_IDLE_EXIT"
	EventChampionSelected               = "CHAMPION_SELECTED"
	EventChampionExportRequested        = "CHAMPION_EXPORT_REQUESTED"
	EventChampionDemoPrediction         = "CHAMPION_DEMO_PREDICTION"
	EventChampionFeedbackRecorded       = "CHAMPION_FEEDBACK_RECORDED"
	EventJobRetryQueued                 = "JOB_RETRY_QUEUED"
	EventJobStaleCallbackIgnored        = "JOB_STALE_CALLBACK_IGNORED"
	EventJobPolicyBlocked               = "JOB_POLICY_BLOCKED"
	EventJobPolicyReconciled            = "JOB_POLICY_RECONCILED"
	EventExecutionCancellationRequested = "EXECUTION_CANCELLATION_REQUESTED"
	EventExecutionCancelled             = "EXECUTION_CANCELLED"
	EventRemoteWorkCancelRequested      = "REMOTE_WORK_CANCEL_REQUESTED"
	EventRemoteWorkCancelFailed         = "REMOTE_WORK_CANCEL_FAILED"
	EventCostBudgetBlocked              = "COST_BUDGET_BLOCKED"
	EventDatasetVisualAnalysisQueued    = "DATASET_VISUAL_ANALYSIS_QUEUED"
	EventDatasetVisualAnalysisResult    = "DATASET_VISUAL_ANALYSIS_RESULT"
	EventExperimentationReopened        = "EXPERIMENTATION_REOPENED"
	EventExecutionFailed                = "EXECUTION_FAILED"
	EventExecutionValidationReported    = "EXECUTION_VALIDATION_REPORTED"
	EventMemoryRetrievalLogged          = "MEMORY_RETRIEVAL_LOGGED"
	EventAgentStarted                   = "AGENT_STARTED"
	EventAgentRecommendationRecorded    = "AGENT_RECOMMENDATION_RECORDED"
	EventAgentOutcomeRecorded           = "AGENT_OUTCOME_RECORDED"
	EventAgentFailed                    = "AGENT_FAILED"
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
	ID             string         `json:"id"`
	ProjectID      string         `json:"project_id"`
	PlanID         string         `json:"plan_id,omitempty"`
	EventType      string         `json:"event_type"`
	Message        string         `json:"message"`
	Payload        map[string]any `json:"payload"`
	CreatedAt      time.Time      `json:"created_at"`
	Sequence       int64          `json:"-"`
	IdempotencyKey string         `json:"-"`
}

type ExecutionEventCursorState struct {
	LastSequence          int64
	RetainedSequenceFloor int64
}

// SafeExecutionEventMetadataKeys is the storage fetch allowlist for the
// bounded v2 execution-event projection. The API remaps error source keys to
// stable public aliases.
func SafeExecutionEventMetadataKeys() []string {
	return []string{
		"category",
		"phase",
		"status",
		"severity",
		"agent_name",
		"invocation_id",
		"decision_id",
		"source_decision_id",
		"decision_type",
		"job_id",
		"attempt_id",
		"job_ids",
		"worker_requirement_id",
		"open_job_count",
		"active_worker_count",
		"target_count",
		"previous_slot_count",
		"slot_count",
		"desired_slot_count",
		"registered_slot_count",
		"active_slot_count",
		"idle_seconds",
		"idle_exit_seconds",
		"dispatcher",
		"provider",
		"gpu_type",
		"requirement_status",
		"template",
		"attempt",
		"taxonomy_version",
		"stage",
		"detail_code",
		"revision",
		"current",
		"total",
		"unit",
		"max_attempts",
		"requeued",
		"backend_validation_status",
		"backend_stop_guard",
		"reason",
		"reason_code",
		"reason_codes",
		"policy_evaluation_id",
		"effective_policy_hash",
		"policy_eligibility_status",
		"model",
		"selection_source",
		"materialization_status",
		"max_concurrent_jobs",
		"max_cold_dataset_materializations",
		"retry_attempt",
		"will_retry",
		"completed_run_count",
		"memory_count",
		"evaluation_count",
		"purpose",
		"retrieved_count",
		"log_only",
		"cross_project_ok",
		"backend_validation_error",
		"error",
		"last_error",
	}
}
