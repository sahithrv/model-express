package calibration

import (
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	CandidateProvenanceSchemaVersionV1 = "candidate_provenance_v1"
	CandidateForecastScoreVersionV1    = "planner_candidate_score_v1"
	CandidatePredictionSource          = "candidate.expected_metric_impact"
	CandidateForecastUnits             = "fractional_score"

	MetricDirectionHigherIsBetter = "higher_is_better"
	MetricDirectionLowerIsBetter  = "lower_is_better"

	CandidateSelectionSelected   = "selected"
	CandidateSelectionRejected   = "rejected"
	CandidateSelectionUnselected = "unselected"

	CandidateOutcomeUnknown = "unknown"
)

type CandidateForecastRange struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

type CandidateForecastContract struct {
	ForecastTarget   string                 `json:"forecast_target"`
	MetricDirection  string                 `json:"metric_direction"`
	ScoreBasis       string                 `json:"score_basis"`
	ScoreVersion     string                 `json:"score_version"`
	BaselineJobID    string                 `json:"baseline_job_id,omitempty"`
	BaselineScore    float64                `json:"baseline_score"`
	PredictedDelta   float64                `json:"predicted_delta"`
	PredictionSource string                 `json:"prediction_source"`
	Units            string                 `json:"units"`
	ValidRange       CandidateForecastRange `json:"valid_range"`
}

type CandidateProvenanceCreate struct {
	InvocationID            string                    `json:"invocation_id"`
	PlannerVariantID        string                    `json:"planner_variant_id"`
	CandidateIndex          int                       `json:"candidate_index"`
	RequestedConfigHash     string                    `json:"requested_config_hash"`
	AcceptedSpecHash        string                    `json:"accepted_spec_hash"`
	Task                    string                    `json:"task"`
	Mechanism               string                    `json:"mechanism"`
	Forecast                CandidateForecastContract `json:"forecast"`
	BaseScore               float64                   `json:"base_score"`
	SelectionTraceReference string                    `json:"selection_trace_reference"`
	Selected                bool                      `json:"selected"`
	Rejected                bool                      `json:"rejected"`
	SelectionState          string                    `json:"selection_state"`
	SelectedExperimentIndex *int                      `json:"selected_experiment_index,omitempty"`
	OutcomeStatus           string                    `json:"outcome_status"`
	Reasons                 []string                  `json:"reasons"`
}

type CandidateProvenance struct {
	ID         string `json:"id"`
	ProjectID  string `json:"project_id"`
	DecisionID string `json:"decision_id"`
	CandidateProvenanceCreate
	FollowUpPlanID        *string   `json:"follow_up_plan_id,omitempty"`
	ExperimentID          *string   `json:"experiment_id,omitempty"`
	JobID                 *string   `json:"job_id,omitempty"`
	RealizedEffectiveHash *string   `json:"realized_effective_hash,omitempty"`
	CreatedAt             time.Time `json:"created_at"`
}

func ValidateForecastContract(contract CandidateForecastContract) error {
	if strings.TrimSpace(contract.ForecastTarget) == "" {
		return fmt.Errorf("candidate forecast_target is required")
	}
	switch contract.MetricDirection {
	case MetricDirectionHigherIsBetter:
		if contract.PredictedDelta < 0 {
			return fmt.Errorf("candidate predicted_delta must be non-negative for higher_is_better")
		}
	case MetricDirectionLowerIsBetter:
		if contract.PredictedDelta > 0 {
			return fmt.Errorf("candidate predicted_delta must be non-positive for lower_is_better")
		}
	default:
		return fmt.Errorf("candidate metric_direction %q is invalid", contract.MetricDirection)
	}
	if strings.TrimSpace(contract.ScoreBasis) == "" {
		return fmt.Errorf("candidate score_basis is required")
	}
	if contract.ScoreVersion != CandidateForecastScoreVersionV1 {
		return fmt.Errorf("candidate score_version must be %q", CandidateForecastScoreVersionV1)
	}
	if contract.PredictionSource != CandidatePredictionSource {
		return fmt.Errorf("candidate prediction_source must be %q", CandidatePredictionSource)
	}
	if contract.Units != CandidateForecastUnits {
		return fmt.Errorf("candidate forecast units must be %q", CandidateForecastUnits)
	}
	values := []float64{contract.BaselineScore, contract.PredictedDelta, contract.ValidRange.Min, contract.ValidRange.Max}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("candidate forecast values must be finite")
		}
	}
	if contract.ValidRange.Min >= contract.ValidRange.Max {
		return fmt.Errorf("candidate forecast valid_range must have min < max")
	}
	if contract.BaselineScore < 0 || contract.BaselineScore > 1 {
		return fmt.Errorf("candidate baseline_score %.6f is outside fractional score range [0, 1]", contract.BaselineScore)
	}
	if contract.PredictedDelta < contract.ValidRange.Min || contract.PredictedDelta > contract.ValidRange.Max {
		return fmt.Errorf("candidate predicted_delta %.6f is outside valid range [%.6f, %.6f]", contract.PredictedDelta, contract.ValidRange.Min, contract.ValidRange.Max)
	}
	return nil
}

func ValidateCandidateProvenanceCreate(candidate CandidateProvenanceCreate) error {
	if strings.TrimSpace(candidate.InvocationID) == "" || strings.TrimSpace(candidate.PlannerVariantID) == "" {
		return fmt.Errorf("candidate invocation_id and planner_variant_id are required")
	}
	if candidate.CandidateIndex < 0 {
		return fmt.Errorf("candidate_index must be zero-based")
	}
	if strings.TrimSpace(candidate.RequestedConfigHash) == "" || strings.TrimSpace(candidate.AcceptedSpecHash) == "" {
		return fmt.Errorf("candidate requested and accepted hashes are required")
	}
	if strings.TrimSpace(candidate.Task) == "" || strings.TrimSpace(candidate.Mechanism) == "" {
		return fmt.Errorf("candidate task and mechanism are required")
	}
	if math.IsNaN(candidate.BaseScore) || math.IsInf(candidate.BaseScore, 0) || candidate.BaseScore < 0 || candidate.BaseScore > 1 {
		return fmt.Errorf("candidate base_score must be finite and between 0 and 1")
	}
	if strings.TrimSpace(candidate.SelectionTraceReference) == "" {
		return fmt.Errorf("candidate selection_trace_reference is required")
	}
	if candidate.Selected && candidate.Rejected {
		return fmt.Errorf("candidate cannot be both selected and rejected")
	}
	expectedState := CandidateSelectionUnselected
	if candidate.Selected {
		expectedState = CandidateSelectionSelected
	} else if candidate.Rejected {
		expectedState = CandidateSelectionRejected
	}
	if candidate.SelectionState != expectedState {
		return fmt.Errorf("candidate selection_state %q does not match selected/rejected flags", candidate.SelectionState)
	}
	if candidate.Selected && (candidate.SelectedExperimentIndex == nil || *candidate.SelectedExperimentIndex < 0) {
		return fmt.Errorf("selected candidate requires a zero-based selected_experiment_index")
	}
	if !candidate.Selected && candidate.SelectedExperimentIndex != nil {
		return fmt.Errorf("unselected candidate cannot have selected_experiment_index")
	}
	if candidate.OutcomeStatus != CandidateOutcomeUnknown {
		return fmt.Errorf("new candidate outcome_status must be %q", CandidateOutcomeUnknown)
	}
	if err := ValidateForecastContract(candidate.Forecast); err != nil {
		return err
	}
	return nil
}

func CandidateProvenanceMatchesCreate(row CandidateProvenance, create CandidateProvenanceCreate) bool {
	if row.CandidateProvenanceCreate.InvocationID != create.InvocationID ||
		row.PlannerVariantID != create.PlannerVariantID ||
		row.CandidateIndex != create.CandidateIndex ||
		row.RequestedConfigHash != create.RequestedConfigHash ||
		row.AcceptedSpecHash != create.AcceptedSpecHash ||
		row.Task != create.Task ||
		row.Mechanism != create.Mechanism ||
		row.Forecast != create.Forecast ||
		row.BaseScore != create.BaseScore ||
		row.SelectionTraceReference != create.SelectionTraceReference ||
		row.Selected != create.Selected ||
		row.Rejected != create.Rejected ||
		row.SelectionState != create.SelectionState ||
		row.OutcomeStatus != create.OutcomeStatus ||
		!equalOptionalInt(row.SelectedExperimentIndex, create.SelectedExperimentIndex) ||
		len(row.Reasons) != len(create.Reasons) {
		return false
	}
	for index := range row.Reasons {
		if row.Reasons[index] != create.Reasons[index] {
			return false
		}
	}
	return true
}

func equalOptionalInt(left, right *int) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}
