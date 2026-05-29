package automl

import "time"

type HyperparameterProvenance string

const (
	ProvenanceLLM               HyperparameterProvenance = "llm"
	ProvenanceBackendDefault    HyperparameterProvenance = "backend_default"
	ProvenanceUserManual        HyperparameterProvenance = "user_manual"
	ProvenanceRandomSearch      HyperparameterProvenance = "random_search"
	ProvenanceGridSearch        HyperparameterProvenance = "grid_search"
	ProvenanceBayesianOptimizer HyperparameterProvenance = "bayesian_optimizer"
	ProvenanceOtherSampler      HyperparameterProvenance = "other_sampler"
)

type ParameterType string

const (
	ParameterFloat       ParameterType = "float"
	ParameterInteger     ParameterType = "integer"
	ParameterCategorical ParameterType = "categorical"
)

type SearchScale string

const (
	SearchScaleLinear SearchScale = "linear"
	SearchScaleLog    SearchScale = "log"
)

type ExperimentIntent struct {
	Summary             string   `json:"summary,omitempty"`
	PlanningMode        string   `json:"planning_mode,omitempty"`
	ExplorationIntent   string   `json:"exploration_intent,omitempty"`
	Goals               []string `json:"goals,omitempty"`
	AllowedParameters   []string `json:"allowed_parameters,omitempty"`
	StrategyDescription string   `json:"strategy_description,omitempty"`
}

type ParameterCondition struct {
	Field  string `json:"field"`
	Equals string `json:"equals"`
}

type HyperparameterParameterSpec struct {
	Name       string                   `json:"name"`
	Type       ParameterType            `json:"type"`
	Min        *float64                 `json:"min,omitempty"`
	Max        *float64                 `json:"max,omitempty"`
	Step       *float64                 `json:"step,omitempty"`
	Scale      SearchScale              `json:"scale,omitempty"`
	Choices    []string                 `json:"choices,omitempty"`
	IntChoices []int                    `json:"int_choices,omitempty"`
	Default    any                      `json:"default,omitempty"`
	Source     HyperparameterProvenance `json:"source,omitempty"`
	Condition  *ParameterCondition      `json:"condition,omitempty"`
	Notes      string                   `json:"notes,omitempty"`
}

type HyperparameterSearchSpace struct {
	Parameters []HyperparameterParameterSpec `json:"parameters"`
}

type HyperparameterSuggestion struct {
	ID               string                              `json:"id,omitempty"`
	StudyID          string                              `json:"study_id,omitempty"`
	Sampler          string                              `json:"sampler"`
	Seed             int64                               `json:"seed"`
	Values           map[string]any                      `json:"values"`
	FinalValues      map[string]any                      `json:"final_values,omitempty"`
	Provenance       map[string]HyperparameterProvenance `json:"provenance"`
	ValidationStatus string                              `json:"validation_status,omitempty"`
	ValidationErrors []string                            `json:"validation_errors,omitempty"`
	CreatedAt        time.Time                           `json:"created_at,omitempty"`
}

type ExperimentAutoML struct {
	Enabled          bool                                `json:"enabled"`
	Intent           ExperimentIntent                    `json:"intent,omitempty"`
	Sampler          string                              `json:"sampler,omitempty"`
	Seed             int64                               `json:"seed,omitempty"`
	SearchSpace      *HyperparameterSearchSpace          `json:"search_space,omitempty"`
	Suggestion       *HyperparameterSuggestion           `json:"suggestion,omitempty"`
	FinalValues      map[string]any                      `json:"final_values,omitempty"`
	ValueProvenance  map[string]HyperparameterProvenance `json:"value_provenance,omitempty"`
	StrategySnapshot map[string]any                      `json:"strategy_snapshot,omitempty"`
	ValidationStatus string                              `json:"validation_status,omitempty"`
	ValidationErrors []string                            `json:"validation_errors,omitempty"`
}

type OptimizerStudy struct {
	ID               string                    `json:"id"`
	ProjectID        string                    `json:"project_id"`
	PlanID           string                    `json:"plan_id,omitempty"`
	DatasetID        string                    `json:"dataset_id"`
	SourceDecisionID string                    `json:"source_decision_id,omitempty"`
	ExperimentIndex  int                       `json:"experiment_index"`
	Model            string                    `json:"model"`
	Intent           ExperimentIntent          `json:"intent"`
	Sampler          string                    `json:"sampler"`
	Seed             int64                     `json:"seed"`
	SearchSpace      HyperparameterSearchSpace `json:"search_space"`
	StrategySnapshot map[string]any            `json:"strategy_snapshot"`
	CreatedAt        time.Time                 `json:"created_at"`
}

type OptimizerSuggestion struct {
	ID               string                              `json:"id"`
	StudyID          string                              `json:"study_id"`
	ProjectID        string                              `json:"project_id"`
	PlanID           string                              `json:"plan_id,omitempty"`
	DatasetID        string                              `json:"dataset_id"`
	JobID            string                              `json:"job_id,omitempty"`
	ExperimentIndex  int                                 `json:"experiment_index"`
	Model            string                              `json:"model"`
	Sampler          string                              `json:"sampler"`
	Seed             int64                               `json:"seed"`
	Values           map[string]any                      `json:"values"`
	FinalValues      map[string]any                      `json:"final_values"`
	Provenance       map[string]HyperparameterProvenance `json:"provenance"`
	ValidationStatus string                              `json:"validation_status"`
	ValidationErrors []string                            `json:"validation_errors"`
	CreatedAt        time.Time                           `json:"created_at"`
}

type OptimizerTrial struct {
	ID           string         `json:"id"`
	StudyID      string         `json:"study_id"`
	SuggestionID string         `json:"suggestion_id"`
	ProjectID    string         `json:"project_id"`
	PlanID       string         `json:"plan_id,omitempty"`
	DatasetID    string         `json:"dataset_id"`
	JobID        string         `json:"job_id"`
	Status       string         `json:"status"`
	TargetMetric string         `json:"target_metric"`
	Score        float64        `json:"score"`
	Metrics      map[string]any `json:"metrics"`
	Error        string         `json:"error,omitempty"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
}

type OptimizerFeedbackSummary struct {
	StudyID                 string         `json:"study_id,omitempty"`
	TrialCount              int            `json:"trial_count"`
	SucceededTrialCount     int            `json:"succeeded_trial_count"`
	FailedTrialCount        int            `json:"failed_trial_count"`
	TargetMetric            string         `json:"target_metric"`
	BestScore               float64        `json:"best_score,omitempty"`
	BestJobID               string         `json:"best_job_id,omitempty"`
	BestHyperparameters     map[string]any `json:"best_hyperparameters,omitempty"`
	TrainValidationGap      float64        `json:"train_validation_gap,omitempty"`
	Trend                   string         `json:"trend,omitempty"`
	FailedParameterPatterns []string       `json:"failed_parameter_patterns,omitempty"`
	RecommendedNarrowing    []string       `json:"recommended_narrowing,omitempty"`
}

type StrategyContext struct {
	Model                  string
	Template               string
	Optimizer              string
	Scheduler              string
	AugmentationPolicy     string
	AugmentationPolicyType string
	ClassBalancing         string
}
