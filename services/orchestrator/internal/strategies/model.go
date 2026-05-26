package strategies

import "time"

const (
	OutcomeImprovedChampion = "improved_champion"
	OutcomeMinorImprovement = "minor_improvement"
	OutcomeNoImprovement    = "no_improvement"
	OutcomeFailed           = "failed"
	OutcomeInvalidated      = "invalidated"
	OutcomePending          = "pending"
)

type StrategyScorecard struct {
	ID                string         `json:"id"`
	ProjectID         string         `json:"project_id"`
	DatasetID         string         `json:"dataset_id"`
	SourceDecisionID  string         `json:"source_decision_id"`
	SourcePlanID      string         `json:"source_plan_id"`
	FollowUpPlanID    string         `json:"followup_plan_id"`
	StrategyType      string         `json:"strategy_type"`
	PlanningMode      string         `json:"planning_mode"`
	Mechanism         string         `json:"mechanism"`
	Intervention      string         `json:"intervention"`
	DiagnosisTriggers []string       `json:"diagnosis_triggers"`
	EvidenceUsed      []string       `json:"evidence_used"`
	ExpectedEffect    string         `json:"expected_effect"`
	DatasetTraits     map[string]any `json:"dataset_traits"`
	ObjectiveProfile  map[string]any `json:"objective_profile"`
	ProposedChanges   map[string]any `json:"proposed_changes"`
	ExpectedDelta     float64        `json:"expected_delta"`
	ActualDelta       float64        `json:"actual_delta"`
	ConfidenceBefore  float64        `json:"confidence_before"`
	ConfidenceAfter   float64        `json:"confidence_after"`
	CostUSD           float64        `json:"cost_usd"`
	RuntimeSeconds    float64        `json:"runtime_seconds"`
	Outcome           string         `json:"outcome"`
	Lesson            string         `json:"lesson"`
	Tags              []string       `json:"tags"`
	CreatedAt         time.Time      `json:"created_at"`
}

type StrategyScorecardCreate struct {
	ProjectID         string
	DatasetID         string
	SourceDecisionID  string
	SourcePlanID      string
	FollowUpPlanID    string
	StrategyType      string
	PlanningMode      string
	Mechanism         string
	Intervention      string
	DiagnosisTriggers []string
	EvidenceUsed      []string
	ExpectedEffect    string
	DatasetTraits     map[string]any
	ObjectiveProfile  map[string]any
	ProposedChanges   map[string]any
	ExpectedDelta     float64
	ConfidenceBefore  float64
	Outcome           string
	Lesson            string
	Tags              []string
}

type StrategyScorecardOutcomeUpdate struct {
	ActualDelta     float64
	ConfidenceAfter float64
	CostUSD         float64
	RuntimeSeconds  float64
	Outcome         string
	Lesson          string
	Tags            []string
}
