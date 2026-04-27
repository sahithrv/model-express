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
	TemplateProfileDataset  = "profile_dataset"
	TemplateTrainExperiment = "train_experiment"
)

type ExperimentJob struct {
	ID          string         `json:"id"`
	ProjectID   string         `json:"project_id"`
	WorkerID    string         `json:"worker_id,omitempty"`
	Template    string         `json:"template"`
	Status      string         `json:"status"`
	Config      map[string]any `json:"config"`
	MLflowRunID string         `json:"mlflow_run_id,omitempty"`
	Error       string         `json:"error,omitempty"`
	CreatedAt   time.Time      `json:"created_at"`
	StartedAt   *time.Time     `json:"started_at,omitempty"`
	CompletedAt *time.Time     `json:"completed_at,omitempty"`
}

type EpochMetric struct {
	JobID     string             `json:"job_id"`
	Epoch     int                `json:"epoch"`
	Metrics   map[string]float64 `json:"metrics"`
	CreatedAt time.Time          `json:"created_at"`
}
