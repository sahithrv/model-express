package plans

import (
	"time"

	"model-express/services/orchestrator/internal/automl"
)

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
	Template                 string                    `json:"template"`
	Model                    string                    `json:"model"`
	Mechanism                string                    `json:"mechanism,omitempty"`
	Intervention             string                    `json:"intervention,omitempty"`
	EvidenceUsed             []string                  `json:"evidence_used,omitempty"`
	ExpectedEffect           string                    `json:"expected_effect,omitempty"`
	Epochs                   int                       `json:"epochs"`
	BatchSize                int                       `json:"batch_size"`
	LearningRate             float64                   `json:"learning_rate"`
	Reason                   string                    `json:"reason"`
	ImageSize                int                       `json:"image_size,omitempty"`
	ResolutionStrategy       string                    `json:"resolution_strategy,omitempty"`
	Preprocessing            *Preprocessing            `json:"preprocessing,omitempty"`
	Optimizer                string                    `json:"optimizer,omitempty"`
	Scheduler                string                    `json:"scheduler,omitempty"`
	WeightDecay              float64                   `json:"weight_decay,omitempty"`
	Dropout                  float64                   `json:"dropout,omitempty"`
	OptimizerMomentum        float64                   `json:"optimizer_momentum,omitempty"`
	SchedulerStepSize        int                       `json:"scheduler_step_size,omitempty"`
	SchedulerGamma           float64                   `json:"scheduler_gamma,omitempty"`
	LabelSmoothing           float64                   `json:"label_smoothing,omitempty"`
	GradientClipNorm         float64                   `json:"gradient_clip_norm,omitempty"`
	Augmentation             map[string]any            `json:"augmentation,omitempty"`
	AugmentationPolicy       string                    `json:"augmentation_policy,omitempty"`
	AugmentationPolicyConfig *AugmentationPolicyConfig `json:"augmentation_policy_config,omitempty"`
	ClassBalancing           string                    `json:"class_balancing,omitempty"`
	ClassBalancingConfig     map[string]any            `json:"class_balancing_config,omitempty"`
	SamplingStrategy         string                    `json:"sampling_strategy,omitempty"`
	EarlyStoppingPatience    int                       `json:"early_stopping_patience,omitempty"`
	Strategy                 string                    `json:"strategy,omitempty"`
	Pretrained               bool                      `json:"pretrained,omitempty"`
	FreezeBackbone           bool                      `json:"freeze_backbone,omitempty"`
	FineTuneStrategy         string                    `json:"fine_tune_strategy,omitempty"`
	AutoML                   *automl.ExperimentAutoML  `json:"automl,omitempty"`
}

type AugmentationPolicyConfig struct {
	PolicyType       string  `json:"policy_type,omitempty"`
	Magnitude        int     `json:"magnitude,omitempty"`
	NumOps           int     `json:"num_ops,omitempty"`
	NumMagnitudeBins int     `json:"num_magnitude_bins,omitempty"`
	Probability      float64 `json:"probability,omitempty"`
	Alpha            float64 `json:"alpha,omitempty"`
}

type Preprocessing struct {
	ResizeStrategy          string `json:"resize_strategy,omitempty"`
	Normalization           string `json:"normalization,omitempty"`
	CropStrategy            string `json:"crop_strategy,omitempty"`
	BBoxMode                string `json:"bbox_mode,omitempty"`
	UseDatasetNormalization bool   `json:"use_dataset_normalization,omitempty"`
}
