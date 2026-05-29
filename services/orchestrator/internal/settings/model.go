package settings

import "time"

type AutomationSettings struct {
	AutoReviewExperiments   bool      `json:"auto_review_experiments"`
	AutoScheduleFollowUps   bool      `json:"auto_schedule_followups"`
	AutoExecutePlans        bool      `json:"auto_execute_plans"`
	MaxFollowUpRounds       int       `json:"max_followup_rounds"`
	DefaultTrainingProvider string    `json:"default_training_provider"`
	DefaultGPUType          string    `json:"default_gpu_type"`
	LLMEnabled              bool      `json:"llm_enabled"`
	AgentMode               string    `json:"agent_mode"`
	LLMProvider             string    `json:"llm_provider"`
	LLMModel                string    `json:"llm_model"`
	AutoMLEnabled           bool      `json:"automl_enabled"`
	AutoMLSampler           string    `json:"automl_sampler"`
	UpdatedAt               time.Time `json:"updated_at"`
}

type AutomationSettingsUpdate struct {
	AutoReviewExperiments   *bool   `json:"auto_review_experiments"`
	AutoScheduleFollowUps   *bool   `json:"auto_schedule_followups"`
	AutoExecutePlans        *bool   `json:"auto_execute_plans"`
	MaxFollowUpRounds       *int    `json:"max_followup_rounds"`
	DefaultTrainingProvider *string `json:"default_training_provider"`
	DefaultGPUType          *string `json:"default_gpu_type"`
	LLMEnabled              *bool   `json:"llm_enabled"`
	AgentMode               *string `json:"agent_mode"`
	LLMProvider             *string `json:"llm_provider"`
	LLMModel                *string `json:"llm_model"`
	AutoMLEnabled           *bool   `json:"automl_enabled"`
	AutoMLSampler           *string `json:"automl_sampler"`
}
