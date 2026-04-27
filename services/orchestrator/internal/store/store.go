package store

import (
	"errors"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/workers"
)

var (
	ErrNotFound       = errors.New("not found")
	ErrNoJob          = errors.New("no job available")
	ErrInvalidRequest = errors.New("invalid request")
)

type Store interface {
	CreateProject(name string, goal string) (projects.Project, error)
	GetProject(id string) (projects.Project, error)
	ListProjects() ([]projects.Project, error)

	CreateDataset(projectID string, name string, storageURI string, checksumSHA256 string, sizeBytes int64) (datasets.Dataset, error)
	GetDataset(id string) (datasets.Dataset, error)
	ListProjectDatasets(projectID string) ([]datasets.Dataset, error)
	UpdateDatasetProfile(id string, profile map[string]any) (datasets.Dataset, error)

	RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error)
	ListWorkers() ([]workers.Worker, error)
	ListProjectWorkers(projectID string) ([]workers.Worker, error)
	GetWorker(workerID string) (workers.Worker, error)
	HeartbeatWorker(id string) (workers.Worker, error)
	PollJob(workerID string) (*jobs.ExperimentJob, error)

	CreateJob(projectID string, template string, config map[string]any) (jobs.ExperimentJob, error)
	GetJob(id string) (jobs.ExperimentJob, error)
	ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error)
	ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error)
	ListJobMetrics(jobID string) ([]jobs.EpochMetric, error)
	CompleteJob(jobID string, mlflowRunID string) (jobs.ExperimentJob, error)
	FailJob(jobID string, message string) (jobs.ExperimentJob, error)

	CreateExperimentPlan(projectID string, datasetID string, targetMetric string, recommendedWorkers int, estimatedMinutes int, experiments []plans.PlannedExperiment, warnings []string) (plans.ExperimentPlan, error)
	GetExperimentPlan(id string) (plans.ExperimentPlan, error)
	ListProjectExperimentPlans(projectID string) ([]plans.ExperimentPlan, error)
}
