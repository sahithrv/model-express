package store

import (
	"context"
	"encoding/json"
	"fmt"

	"model-express/services/orchestrator/internal/plans"
)

func (s *PostgresStore) CreateExperimentPlan(projectID string, datasetID string, targetMetric string, recommendedWorkers int, estimatedMinutes int, experiments []plans.PlannedExperiment, warnings []string, sourceDecisionID string) (plans.ExperimentPlan, error) {
	if err := s.requireProject(projectID); err != nil {
		return plans.ExperimentPlan{}, err
	}
	if err := s.requireDatasetBelongsToProject(projectID, datasetID); err != nil {
		return plans.ExperimentPlan{}, err
	}
	if targetMetric == "" {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: target_metric is required", ErrInvalidRequest)
	}
	if recommendedWorkers < 1 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: recommended_workers must be at least 1", ErrInvalidRequest)
	}
	if estimatedMinutes < 1 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: estimated_minutes must be at least 1", ErrInvalidRequest)
	}
	if len(experiments) == 0 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: at least one planned experiment is required", ErrInvalidRequest)
	}
	if warnings == nil {
		warnings = []string{}
	}

	experimentsJSON, err := json.Marshal(experiments)
	if err != nil {
		return plans.ExperimentPlan{}, fmt.Errorf("marshal planned experiments: %w", err)
	}

	warningsJSON, err := json.Marshal(warnings)
	if err != nil {
		return plans.ExperimentPlan{}, fmt.Errorf("marshal experiment plan warnings: %w", err)
	}

	const query = `
		INSERT INTO experiment_plans (
			project_id,
			dataset_id,
			status,
			source_decision_id,
			target_metric,
			recommended_workers,
			estimated_minutes,
			experiments,
			warnings
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING id, project_id, dataset_id, status, source_decision_id, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
	`

	return scanExperimentPlan(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		datasetID,
		plans.StatusProposed,
		sourceDecisionID,
		targetMetric,
		recommendedWorkers,
		estimatedMinutes,
		experimentsJSON,
		warningsJSON,
	))
}

func (s *PostgresStore) GetExperimentPlan(id string) (plans.ExperimentPlan, error) {
	const query = `
		SELECT id, project_id, dataset_id, status, source_decision_id, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
		FROM experiment_plans
		WHERE id = $1
	`

	return scanExperimentPlan(s.db.QueryRowContext(context.Background(), query, id))
}

func (s *PostgresStore) ListProjectExperimentPlans(projectID string) ([]plans.ExperimentPlan, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, dataset_id, status, source_decision_id, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
		FROM experiment_plans
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []plans.ExperimentPlan{}
	for rows.Next() {
		plan, err := scanExperimentPlan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, plan)
	}

	return out, rows.Err()
}
