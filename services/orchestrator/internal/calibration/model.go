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

	CandidateOutcomeUnknown    = "unknown"
	CandidateOutcomeObserved   = "observed"
	CandidateOutcomeUnobserved = "unobserved"

	CandidateTerminalSucceeded = "SUCCEEDED"
	CandidateTerminalFailed    = "FAILED"
	CandidateTerminalCancelled = "CANCELLED"
	CandidateTerminalSkipped   = "SKIPPED"
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
	FollowUpPlanID        *string    `json:"follow_up_plan_id,omitempty"`
	ExperimentID          *string    `json:"experiment_id,omitempty"`
	JobID                 *string    `json:"job_id,omitempty"`
	AttemptID             *string    `json:"attempt_id,omitempty"`
	RealizedEffectiveHash *string    `json:"realized_effective_hash,omitempty"`
	ActualScore           *float64   `json:"actual_score,omitempty"`
	ActualDelta           *float64   `json:"actual_delta,omitempty"`
	TerminalState         *string    `json:"terminal_state,omitempty"`
	CostUSD               *float64   `json:"cost_usd,omitempty"`
	RuntimeSeconds        *float64   `json:"runtime_seconds,omitempty"`
	CalibrationEligible   *bool      `json:"calibration_eligible,omitempty"`
	EligibilityReason     *string    `json:"eligibility_reason,omitempty"`
	FinalizedAt           *time.Time `json:"finalized_at,omitempty"`
	CreatedAt             time.Time  `json:"created_at"`
}

// CandidateOutcomeUpdate attaches only execution-time lineage and outcomes.
// Proposal-time forecast and selection fields are deliberately absent.
type CandidateOutcomeUpdate struct {
	CandidateIndex        int
	FollowUpPlanID        string
	ExperimentID          string
	JobID                 *string
	AttemptID             *string
	RealizedEffectiveHash *string
	OutcomeStatus         string
	ActualScore           *float64
	ActualDelta           *float64
	TerminalState         *string
	CostUSD               *float64
	RuntimeSeconds        *float64
	CalibrationEligible   *bool
	EligibilityReason     *string
}

func ExperimentLineageID(planID string, experimentIndex int) string {
	return fmt.Sprintf("%s:experiment-%d", strings.TrimSpace(planID), experimentIndex)
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

func ApplyCandidateOutcomeUpdate(row CandidateProvenance, update CandidateOutcomeUpdate, now time.Time) (CandidateProvenance, error) {
	if !row.Selected || row.SelectedExperimentIndex == nil {
		return CandidateProvenance{}, fmt.Errorf("candidate %d is not selected", row.CandidateIndex)
	}
	if update.CandidateIndex != row.CandidateIndex {
		return CandidateProvenance{}, fmt.Errorf("candidate outcome index %d does not match row %d", update.CandidateIndex, row.CandidateIndex)
	}
	if strings.TrimSpace(update.FollowUpPlanID) == "" || strings.TrimSpace(update.ExperimentID) == "" {
		return CandidateProvenance{}, fmt.Errorf("candidate follow-up plan and experiment lineage are required")
	}
	var err error
	if row.FollowUpPlanID, err = mergeImmutableString(row.FollowUpPlanID, &update.FollowUpPlanID, "follow_up_plan_id"); err != nil {
		return CandidateProvenance{}, err
	}
	if row.ExperimentID, err = mergeImmutableString(row.ExperimentID, &update.ExperimentID, "experiment_id"); err != nil {
		return CandidateProvenance{}, err
	}
	if row.JobID, err = mergeImmutableString(row.JobID, update.JobID, "job_id"); err != nil {
		return CandidateProvenance{}, err
	}
	if row.AttemptID, err = mergeImmutableString(row.AttemptID, update.AttemptID, "attempt_id"); err != nil {
		return CandidateProvenance{}, err
	}
	if row.RealizedEffectiveHash, err = mergeImmutableString(row.RealizedEffectiveHash, update.RealizedEffectiveHash, "realized_effective_hash"); err != nil {
		return CandidateProvenance{}, err
	}

	if update.OutcomeStatus == "" || update.OutcomeStatus == CandidateOutcomeUnknown {
		if update.ActualScore != nil || update.ActualDelta != nil || update.TerminalState != nil || update.CostUSD != nil ||
			update.RuntimeSeconds != nil || update.CalibrationEligible != nil || update.EligibilityReason != nil {
			return CandidateProvenance{}, fmt.Errorf("non-terminal candidate lineage cannot include outcome fields")
		}
		return row, nil
	}
	if err := validateCandidateFinalOutcome(row, update); err != nil {
		return CandidateProvenance{}, err
	}
	if row.OutcomeStatus != CandidateOutcomeUnknown {
		if row.CostUSD, err = mergeImmutableFloat(row.CostUSD, update.CostUSD, "cost_usd"); err != nil {
			return CandidateProvenance{}, err
		}
		if row.RuntimeSeconds, err = mergeImmutableFloat(row.RuntimeSeconds, update.RuntimeSeconds, "runtime_seconds"); err != nil {
			return CandidateProvenance{}, err
		}
		if !candidateFinalOutcomeMatches(row, update) {
			return CandidateProvenance{}, fmt.Errorf("candidate %d outcome conflicts with the immutable finalized record", row.CandidateIndex)
		}
		return row, nil
	}

	row.OutcomeStatus = update.OutcomeStatus
	row.ActualScore = cloneFloat(update.ActualScore)
	row.ActualDelta = cloneFloat(update.ActualDelta)
	row.TerminalState = cloneString(update.TerminalState)
	row.CostUSD = cloneFloat(update.CostUSD)
	row.RuntimeSeconds = cloneFloat(update.RuntimeSeconds)
	row.CalibrationEligible = cloneBool(update.CalibrationEligible)
	row.EligibilityReason = cloneString(update.EligibilityReason)
	finalizedAt := now.UTC()
	row.FinalizedAt = &finalizedAt
	return row, nil
}

func validateCandidateFinalOutcome(row CandidateProvenance, update CandidateOutcomeUpdate) error {
	if update.OutcomeStatus != CandidateOutcomeObserved && update.OutcomeStatus != CandidateOutcomeUnobserved {
		return fmt.Errorf("candidate outcome_status %q is invalid", update.OutcomeStatus)
	}
	if update.TerminalState == nil {
		return fmt.Errorf("final candidate outcome requires terminal_state")
	}
	switch *update.TerminalState {
	case CandidateTerminalSucceeded, CandidateTerminalFailed, CandidateTerminalCancelled, CandidateTerminalSkipped:
	default:
		return fmt.Errorf("candidate terminal_state %q is invalid", *update.TerminalState)
	}
	if update.CalibrationEligible == nil || update.EligibilityReason == nil || strings.TrimSpace(*update.EligibilityReason) == "" {
		return fmt.Errorf("final candidate outcome requires calibration eligibility and reason")
	}
	if update.OutcomeStatus == CandidateOutcomeObserved {
		if !*update.CalibrationEligible || *update.TerminalState != CandidateTerminalSucceeded || update.ActualScore == nil || update.ActualDelta == nil {
			return fmt.Errorf("observed candidate outcome requires eligible succeeded score and delta")
		}
		if !finiteInRange(*update.ActualScore, 0, 1) {
			return fmt.Errorf("candidate actual_score must be finite and between 0 and 1")
		}
		if math.Abs(*update.ActualDelta-(*update.ActualScore-row.Forecast.BaselineScore)) > 1e-9 {
			return fmt.Errorf("candidate actual_delta must equal actual_score minus the frozen baseline")
		}
	} else if *update.CalibrationEligible || update.ActualScore != nil || update.ActualDelta != nil {
		return fmt.Errorf("unobserved candidate outcome cannot be calibration eligible or include score/delta")
	}
	for name, value := range map[string]*float64{"cost_usd": update.CostUSD, "runtime_seconds": update.RuntimeSeconds} {
		if value != nil && !finiteInRange(*value, 0, math.MaxFloat64) {
			return fmt.Errorf("candidate %s must be finite and non-negative", name)
		}
	}
	return nil
}

func candidateFinalOutcomeMatches(row CandidateProvenance, update CandidateOutcomeUpdate) bool {
	return row.OutcomeStatus == update.OutcomeStatus &&
		equalOptionalString(row.TerminalState, update.TerminalState) &&
		equalOptionalFloat(row.ActualScore, update.ActualScore) &&
		equalOptionalFloat(row.ActualDelta, update.ActualDelta) &&
		equalOptionalFloat(row.CostUSD, update.CostUSD) &&
		equalOptionalFloat(row.RuntimeSeconds, update.RuntimeSeconds) &&
		equalOptionalBool(row.CalibrationEligible, update.CalibrationEligible) &&
		equalOptionalString(row.EligibilityReason, update.EligibilityReason)
}

func mergeImmutableString(existing *string, update *string, field string) (*string, error) {
	if update == nil {
		return existing, nil
	}
	value := strings.TrimSpace(*update)
	if value == "" {
		return nil, fmt.Errorf("candidate %s cannot be empty", field)
	}
	if existing != nil && *existing != value {
		return nil, fmt.Errorf("candidate %s conflicts with existing lineage", field)
	}
	return &value, nil
}

func mergeImmutableFloat(existing *float64, update *float64, field string) (*float64, error) {
	if update == nil {
		return existing, nil
	}
	if existing != nil && math.Abs(*existing-*update) > 1e-9 {
		return nil, fmt.Errorf("candidate %s conflicts with existing outcome", field)
	}
	return cloneFloat(update), nil
}

func finiteInRange(value, min, max float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= min && value <= max
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneFloat(value *float64) *float64 {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func equalOptionalString(left, right *string) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func equalOptionalFloat(left, right *float64) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func equalOptionalBool(left, right *bool) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func equalOptionalInt(left, right *int) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}
