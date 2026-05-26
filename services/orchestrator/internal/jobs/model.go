package jobs

import (
	"time"
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
	TemplateExportChampion          = "export_champion"
	TemplateChampionDemoPrediction  = "champion_demo_prediction"
	TemplateGenerateVisualExemplars = "generate_visual_exemplars"
)

type ExperimentJob struct {
	ID                   string         `json:"id"`
	ProjectID            string         `json:"project_id"`
	WorkerID             string         `json:"worker_id,omitempty"`
	Template             string         `json:"template"`
	Status               string         `json:"status"`
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

type EpochMetric struct {
	JobID     string             `json:"job_id"`
	Epoch     int                `json:"epoch"`
	Metrics   map[string]float64 `json:"metrics"`
	CreatedAt time.Time          `json:"created_at"`
}
