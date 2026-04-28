package plans

import "time"

const (
	StatusProposed = "PROPOSED"
	StatusApproved = "APPROVED"
	StatusRejected = "REJECTED"
)

type ExperimentPlan struct {
	ID                 string              `json:"id"`
	ProjectID          string              `json:"project_id"`
	DatasetID          string              `json:"dataset_id"`
	Status             string              `json:"status"`
	SourceDecisionID   string              `json:"source_decision_id,omitempty"`
	TargetMetric       string              `json:"target_metric"`
	RecommendedWorkers int                 `json:"recommended_workers"`
	EstimatedMinutes   int                 `json:"estimated_minutes"`
	Experiments        []PlannedExperiment `json:"experiments"`
	Warnings           []string            `json:"warnings"`
	CreatedAt          time.Time           `json:"created_at"`
}

type PlannedExperiment struct {
	Template     string  `json:"template"`
	Model        string  `json:"model"`
	Epochs       int     `json:"epochs"`
	BatchSize    int     `json:"batch_size"`
	LearningRate float64 `json:"learning_rate"`
	Reason       string  `json:"reason"`
}
