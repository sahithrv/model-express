package jobs

import (
	"time"

	"model-express/services/orchestrator/internal/execution"
)

const (
	StatusQueued    = "QUEUED"
	StatusAssigned  = "ASSIGNED"
	StatusRunning   = "RUNNING"
	StatusSucceeded = "SUCCEEDED"
	StatusFailed    = "FAILED"
)

const (
	TemplateProfileDataset          = "profile_dataset"
	TemplateTrainExperiment         = "train_experiment"
	TemplateLabelQualityAudit       = "label_quality_audit"
	TemplateExportChampion          = "export_champion"
	TemplateChampionDemoPrediction  = "champion_demo_prediction"
	TemplateGenerateVisualExemplars = "generate_visual_exemplars"
	TemplateAnalyzeDatasetVisuals   = "analyze_dataset_visuals"
)

type ExperimentJob struct {
	ID                   string         `json:"id"`
	ProjectID            string         `json:"project_id"`
	WorkerID             string         `json:"worker_id,omitempty"`
	Template             string         `json:"template"`
	Status               string         `json:"status"`
	ExecutionSpecStatus  string         `json:"execution_spec_status,omitempty"`
	Config               map[string]any `json:"config"`
	MLflowRunID          string         `json:"mlflow_run_id,omitempty"`
	Error                string         `json:"error,omitempty"`
	Attempt              int            `json:"attempt"`
	MaxAttempts          int            `json:"max_attempts"`
	LeaseOwnerWorkerID   string         `json:"lease_owner_worker_id,omitempty"`
	LeaseExpiresAt       *time.Time     `json:"lease_expires_at,omitempty"`
	LeaseLastHeartbeatAt *time.Time     `json:"lease_last_heartbeat_at,omitempty"`
	CreatedAt            time.Time      `json:"created_at"`
	StartedAt            *time.Time     `json:"started_at,omitempty"`
	CompletedAt          *time.Time     `json:"completed_at,omitempty"`
}

func WithExecutionSpecStatus(job ExperimentJob) ExperimentJob {
	if job.Template != TemplateTrainExperiment {
		job.ExecutionSpecStatus = ""
		return job
	}
	job.ExecutionSpecStatus = execution.ExecutionSpecStatusLegacyUnversioned
	payload, ok := job.Config[execution.ExecutionSpecConfigKey].(map[string]any)
	if ok && payload["schema_version"] == execution.ExecutionSpecSchemaVersionV1 {
		job.ExecutionSpecStatus = execution.ExecutionSpecStatusVersioned
	}
	return job
}

type EpochMetric struct {
	JobID     string             `json:"job_id"`
	Epoch     int                `json:"epoch"`
	Metrics   map[string]float64 `json:"metrics"`
	CreatedAt time.Time          `json:"created_at"`
}
