package decisions

import "time"

const (
	TypeWait           = "WAIT"
	TypeSelectChampion = "SELECT_CHAMPION"
	TypeAddExperiments = "ADD_EXPERIMENTS"
	TypeStopProject    = "STOP_PROJECT"
)

type AgentDecision struct {
	ID           string         `json:"id"`
	ProjectID    string         `json:"project_id"`
	PlanID       string         `json:"plan_id,omitempty"`
	DecisionType string         `json:"decision_type"`
	Rationale    string         `json:"rationale"`
	Payload      map[string]any `json:"payload"`
	CreatedAt    time.Time      `json:"created_at"`
}

type AgentDecisionRecommendation struct {
	PlanID       string         `json:"plan_id,omitempty"`
	DecisionType string         `json:"decision_type"`
	Rationale    string         `json:"rationale"`
	Payload      map[string]any `json:"payload"`
}
