package runs

import "time"

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
