package runs

import "time"

const (
	ChampionExportStatusPendingArtifact = "PENDING_ARTIFACT"
	ChampionExportStatusPending         = "PENDING"
	ChampionExportStatusReady           = "READY"
	ChampionExportStatusFailed          = "FAILED"
)

const (
	ChampionDemoPredictionStatusPending            = "PENDING"
	ChampionDemoPredictionStatusRuntimeUnavailable = "RUNTIME_UNAVAILABLE"
	ChampionDemoPredictionStatusSucceeded          = "SUCCEEDED"
	ChampionDemoPredictionStatusFailed             = "FAILED"
)

const (
	ChampionFeedbackRatingGood     = "good"
	ChampionFeedbackRatingMediocre = "mediocre"
	ChampionFeedbackRatingBad      = "bad"
)

type TrainingRunSummary struct {
	JobID               string    `json:"job_id"`
	ProjectID           string    `json:"project_id"`
	PlanID              string    `json:"plan_id,omitempty"`
	DatasetID           string    `json:"dataset_id,omitempty"`
	Model               string    `json:"model"`
	Provider            string    `json:"provider"`
	GPUType             string    `json:"gpu_type"`
	Status              string    `json:"status"`
	RuntimeSeconds      float64   `json:"runtime_seconds"`
	EstimatedCostUSD    float64   `json:"estimated_cost_usd"`
	BestMacroF1         float64   `json:"best_macro_f1"`
	BestAccuracy        float64   `json:"best_accuracy"`
	FinalTrainLoss      float64   `json:"final_train_loss"`
	FinalValLoss        float64   `json:"final_val_loss"`
	EpochsCompleted     int       `json:"epochs_completed"`
	ModalFunctionCallID string    `json:"modal_function_call_id,omitempty"`
	ModalInputID        string    `json:"modal_input_id,omitempty"`
	CreatedAt           time.Time `json:"created_at"`
	UpdatedAt           time.Time `json:"updated_at"`
}

type TrainingRunSummaryUpdate struct {
	Model               string   `json:"model"`
	Provider            string   `json:"provider"`
	GPUType             string   `json:"gpu_type"`
	Status              string   `json:"status"`
	RuntimeSeconds      *float64 `json:"runtime_seconds"`
	EstimatedCostUSD    *float64 `json:"estimated_cost_usd"`
	BestMacroF1         *float64 `json:"best_macro_f1"`
	BestAccuracy        *float64 `json:"best_accuracy"`
	FinalTrainLoss      *float64 `json:"final_train_loss"`
	FinalValLoss        *float64 `json:"final_val_loss"`
	EpochsCompleted     *int     `json:"epochs_completed"`
	ModalFunctionCallID string   `json:"modal_function_call_id"`
	ModalInputID        string   `json:"modal_input_id"`
}

type TrainingRunEvaluation struct {
	JobID                 string         `json:"job_id"`
	ProjectID             string         `json:"project_id"`
	PlanID                string         `json:"plan_id,omitempty"`
	DatasetID             string         `json:"dataset_id,omitempty"`
	ObjectiveProfile      map[string]any `json:"objective_profile"`
	PerClassMetrics       map[string]any `json:"per_class_metrics"`
	ConfusionMatrix       [][]int        `json:"confusion_matrix"`
	ModelProfile          map[string]any `json:"model_profile"`
	HolisticScores        map[string]any `json:"holistic_scores"`
	RecommendationSummary string         `json:"recommendation_summary"`
	CreatedAt             time.Time      `json:"created_at"`
	UpdatedAt             time.Time      `json:"updated_at"`
}

type TrainingRunEvaluationUpdate struct {
	ObjectiveProfile      map[string]any `json:"objective_profile"`
	PerClassMetrics       map[string]any `json:"per_class_metrics"`
	ConfusionMatrix       [][]int        `json:"confusion_matrix"`
	ModelProfile          map[string]any `json:"model_profile"`
	HolisticScores        map[string]any `json:"holistic_scores"`
	RecommendationSummary string         `json:"recommendation_summary"`
}

type ProjectChampion struct {
	ID                string         `json:"id"`
	ProjectID         string         `json:"project_id"`
	DatasetID         string         `json:"dataset_id"`
	PlanID            string         `json:"plan_id"`
	JobID             string         `json:"job_id"`
	SourceDecisionID  string         `json:"source_decision_id"`
	SelectionReason   string         `json:"selection_reason"`
	Metrics           map[string]any `json:"metrics"`
	Evaluation        map[string]any `json:"evaluation"`
	DeploymentProfile map[string]any `json:"deployment_profile"`
	CreatedAt         time.Time      `json:"created_at"`
	UpdatedAt         time.Time      `json:"updated_at"`
}

type ProjectChampionUpsert struct {
	ProjectID         string
	DatasetID         string
	PlanID            string
	JobID             string
	SourceDecisionID  string
	SelectionReason   string
	Metrics           map[string]any
	Evaluation        map[string]any
	DeploymentProfile map[string]any
}

type ChampionExport struct {
	ID               string         `json:"id"`
	ProjectID        string         `json:"project_id"`
	ChampionID       string         `json:"champion_id"`
	JobID            string         `json:"job_id"`
	Status           string         `json:"status"`
	Format           string         `json:"format"`
	ArtifactURI      string         `json:"artifact_uri,omitempty"`
	Metadata         map[string]any `json:"metadata"`
	ValidationErrors []string       `json:"validation_errors"`
	CreatedAt        time.Time      `json:"created_at"`
	UpdatedAt        time.Time      `json:"updated_at"`
}

type ChampionExportCreate struct {
	ProjectID        string
	ChampionID       string
	JobID            string
	Status           string
	Format           string
	ArtifactURI      string
	Metadata         map[string]any
	ValidationErrors []string
}

type ChampionExportUpdate struct {
	Status           string
	ArtifactURI      string
	Metadata         map[string]any
	ValidationErrors []string
	Error            string
}

type DemoPredictionTopK struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

type ChampionDemoPrediction struct {
	ID             string               `json:"id"`
	ProjectID      string               `json:"project_id"`
	ChampionID     string               `json:"champion_id"`
	JobID          string               `json:"job_id"`
	DatasetID      string               `json:"dataset_id"`
	ImageURI       string               `json:"image_uri"`
	ImageID        string               `json:"image_id,omitempty"`
	ImageMetadata  map[string]any       `json:"image_metadata"`
	Status         string               `json:"status"`
	PredictedLabel string               `json:"predicted_label,omitempty"`
	TrueLabel      string               `json:"true_label,omitempty"`
	Confidence     *float64             `json:"confidence,omitempty"`
	TopK           []DemoPredictionTopK `json:"top_k"`
	LatencyMS      *float64             `json:"latency_ms,omitempty"`
	Correct        *bool                `json:"correct,omitempty"`
	Error          string               `json:"error,omitempty"`
	CreatedAt      time.Time            `json:"created_at"`
}

type ChampionDemoPredictionCreate struct {
	ProjectID      string
	ChampionID     string
	JobID          string
	DatasetID      string
	ImageURI       string
	ImageID        string
	ImageMetadata  map[string]any
	Status         string
	PredictedLabel string
	TrueLabel      string
	Confidence     *float64
	TopK           []DemoPredictionTopK
	LatencyMS      *float64
	Correct        *bool
	Error          string
}

type ChampionDemoPredictionUpdate struct {
	Status         string
	PredictedLabel string
	TrueLabel      string
	Confidence     *float64
	TopK           []DemoPredictionTopK
	LatencyMS      *float64
	Correct        *bool
	Error          string
	ImageMetadata  map[string]any
}

type ChampionFeedback struct {
	ID                 string         `json:"id"`
	ProjectID          string         `json:"project_id"`
	ChampionID         string         `json:"champion_id"`
	PredictionID       string         `json:"prediction_id,omitempty"`
	JobID              string         `json:"job_id"`
	DatasetID          string         `json:"dataset_id"`
	ImageURI           string         `json:"image_uri,omitempty"`
	ImageID            string         `json:"image_id,omitempty"`
	Rating             string         `json:"rating"`
	Message            string         `json:"message,omitempty"`
	PredictionSnapshot map[string]any `json:"prediction_snapshot"`
	MetricsSnapshot    map[string]any `json:"metrics_snapshot"`
	Metadata           map[string]any `json:"metadata"`
	CreatedAt          time.Time      `json:"created_at"`
}

type ChampionFeedbackCreate struct {
	ProjectID          string
	ChampionID         string
	PredictionID       string
	JobID              string
	DatasetID          string
	ImageURI           string
	ImageID            string
	Rating             string
	Message            string
	PredictionSnapshot map[string]any
	MetricsSnapshot    map[string]any
	Metadata           map[string]any
}
