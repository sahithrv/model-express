package store

import (
	"context"
	"encoding/json"
	"fmt"

	"model-express/services/orchestrator/internal/strategies"
)

func (s *PostgresStore) CreateStrategyScorecard(scorecard strategies.StrategyScorecardCreate) (strategies.StrategyScorecard, error) {
	if err := s.requireProject(scorecard.ProjectID); err != nil {
		return strategies.StrategyScorecard{}, err
	}
	if scorecard.Outcome == "" {
		scorecard.Outcome = strategies.OutcomePending
	}
	datasetTraitsJSON, err := json.Marshal(emptyMapIfNil(scorecard.DatasetTraits))
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard dataset_traits: %w", err)
	}
	objectiveProfileJSON, err := json.Marshal(emptyMapIfNil(scorecard.ObjectiveProfile))
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard objective_profile: %w", err)
	}
	proposedChangesJSON, err := json.Marshal(emptyMapIfNil(scorecard.ProposedChanges))
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard proposed_changes: %w", err)
	}
	mechanism, intervention, diagnosisTriggers, evidenceUsed, expectedEffect := hydrateStrategyScorecardMechanismFields(scorecard)
	diagnosisTriggersJSON, err := json.Marshal(diagnosisTriggers)
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard diagnosis_triggers: %w", err)
	}
	evidenceUsedJSON, err := json.Marshal(evidenceUsed)
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard evidence_used: %w", err)
	}
	tagsJSON, err := json.Marshal(scorecard.Tags)
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard tags: %w", err)
	}

	query := `
		INSERT INTO strategy_scorecards (
			project_id, dataset_id, source_decision_id, source_plan_id, followup_plan_id,
			strategy_type, planning_mode, mechanism, intervention, diagnosis_triggers, evidence_used,
			expected_effect, dataset_traits, objective_profile, proposed_changes, expected_delta,
			confidence_before, outcome, lesson, tags
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
		RETURNING ` + strategyScorecardSelectColumns() + `
	`
	return scanStrategyScorecard(s.db.QueryRowContext(
		context.Background(),
		query,
		scorecard.ProjectID,
		scorecard.DatasetID,
		scorecard.SourceDecisionID,
		scorecard.SourcePlanID,
		scorecard.FollowUpPlanID,
		scorecard.StrategyType,
		scorecard.PlanningMode,
		mechanism,
		intervention,
		diagnosisTriggersJSON,
		evidenceUsedJSON,
		expectedEffect,
		datasetTraitsJSON,
		objectiveProfileJSON,
		proposedChangesJSON,
		scorecard.ExpectedDelta,
		scorecard.ConfidenceBefore,
		scorecard.Outcome,
		scorecard.Lesson,
		tagsJSON,
	))
}

func (s *PostgresStore) UpdateStrategyScorecardOutcomeByFollowUpPlan(followUpPlanID string, update strategies.StrategyScorecardOutcomeUpdate) (strategies.StrategyScorecard, error) {
	tagsJSON, err := json.Marshal(update.Tags)
	if err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("marshal strategy scorecard tags: %w", err)
	}
	query := `
		UPDATE strategy_scorecards
		SET actual_delta = $1,
			confidence_after = $2,
			cost_usd = $3,
			runtime_seconds = $4,
			outcome = $5,
			lesson = $6,
			tags = $7
		WHERE followup_plan_id = $8
		RETURNING ` + strategyScorecardSelectColumns() + `
	`
	return scanStrategyScorecard(s.db.QueryRowContext(
		context.Background(),
		query,
		update.ActualDelta,
		update.ConfidenceAfter,
		update.CostUSD,
		update.RuntimeSeconds,
		update.Outcome,
		update.Lesson,
		tagsJSON,
		followUpPlanID,
	))
}

func (s *PostgresStore) ListProjectStrategyScorecards(projectID string, limit int) ([]strategies.StrategyScorecard, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 25
	}
	query := `
		SELECT ` + strategyScorecardSelectColumns() + `
		FROM strategy_scorecards
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []strategies.StrategyScorecard{}
	for rows.Next() {
		scorecard, err := scanStrategyScorecard(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, scorecard)
	}
	return out, rows.Err()
}
