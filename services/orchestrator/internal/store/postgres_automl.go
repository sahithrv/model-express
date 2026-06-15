package store

import (
	"context"
	"encoding/json"
	"fmt"

	"model-express/services/orchestrator/internal/automl"
)

func (s *PostgresStore) CreateOptimizerStudy(study automl.OptimizerStudy) (automl.OptimizerStudy, error) {
	if err := s.requireProject(study.ProjectID); err != nil {
		return automl.OptimizerStudy{}, err
	}
	if err := s.requireDatasetBelongsToProject(study.ProjectID, study.DatasetID); err != nil {
		return automl.OptimizerStudy{}, err
	}
	intentJSON, err := json.Marshal(study.Intent)
	if err != nil {
		return automl.OptimizerStudy{}, fmt.Errorf("marshal automl intent: %w", err)
	}
	searchSpaceJSON, err := json.Marshal(study.SearchSpace)
	if err != nil {
		return automl.OptimizerStudy{}, fmt.Errorf("marshal automl search space: %w", err)
	}
	strategyJSON, err := json.Marshal(emptyMapIfNil(study.StrategySnapshot))
	if err != nil {
		return automl.OptimizerStudy{}, fmt.Errorf("marshal automl strategy snapshot: %w", err)
	}
	query := `
		INSERT INTO automl_studies (project_id, plan_id, dataset_id, source_decision_id, experiment_index, model, intent, sampler, seed, search_space, strategy_snapshot)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		RETURNING ` + automlStudySelectColumns() + `
	`
	return scanOptimizerStudy(s.db.QueryRowContext(
		context.Background(),
		query,
		study.ProjectID,
		study.PlanID,
		study.DatasetID,
		study.SourceDecisionID,
		study.ExperimentIndex,
		study.Model,
		intentJSON,
		study.Sampler,
		study.Seed,
		searchSpaceJSON,
		strategyJSON,
	))
}

func (s *PostgresStore) ListProjectOptimizerStudies(projectID string, limit int) ([]automl.OptimizerStudy, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 25
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+automlStudySelectColumns()+`
		FROM automl_studies
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []automl.OptimizerStudy{}
	for rows.Next() {
		study, err := scanOptimizerStudy(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, study)
	}
	return out, rows.Err()
}

func (s *PostgresStore) CreateOptimizerSuggestion(suggestion automl.OptimizerSuggestion) (automl.OptimizerSuggestion, error) {
	if err := s.requireProject(suggestion.ProjectID); err != nil {
		return automl.OptimizerSuggestion{}, err
	}
	valuesJSON, err := json.Marshal(emptyMapIfNil(suggestion.Values))
	if err != nil {
		return automl.OptimizerSuggestion{}, fmt.Errorf("marshal automl values: %w", err)
	}
	finalValuesJSON, err := json.Marshal(emptyMapIfNil(suggestion.FinalValues))
	if err != nil {
		return automl.OptimizerSuggestion{}, fmt.Errorf("marshal automl final values: %w", err)
	}
	provenanceJSON, err := json.Marshal(suggestion.Provenance)
	if err != nil {
		return automl.OptimizerSuggestion{}, fmt.Errorf("marshal automl provenance: %w", err)
	}
	validationErrorsJSON, err := json.Marshal(suggestion.ValidationErrors)
	if err != nil {
		return automl.OptimizerSuggestion{}, fmt.Errorf("marshal automl validation errors: %w", err)
	}
	query := `
		INSERT INTO automl_suggestions (study_id, project_id, plan_id, dataset_id, job_id, experiment_index, model, sampler, seed, values, final_values, provenance, validation_status, validation_errors)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
		RETURNING ` + automlSuggestionSelectColumns() + `
	`
	return scanOptimizerSuggestion(s.db.QueryRowContext(
		context.Background(),
		query,
		suggestion.StudyID,
		suggestion.ProjectID,
		suggestion.PlanID,
		suggestion.DatasetID,
		suggestion.JobID,
		suggestion.ExperimentIndex,
		suggestion.Model,
		suggestion.Sampler,
		suggestion.Seed,
		valuesJSON,
		finalValuesJSON,
		provenanceJSON,
		suggestion.ValidationStatus,
		validationErrorsJSON,
	))
}

func (s *PostgresStore) UpdateOptimizerSuggestionJob(suggestionID string, jobID string) (automl.OptimizerSuggestion, error) {
	if _, err := s.GetJob(jobID); err != nil {
		return automl.OptimizerSuggestion{}, err
	}
	return scanOptimizerSuggestion(s.db.QueryRowContext(context.Background(), `
		UPDATE automl_suggestions
		SET job_id = $1
		WHERE id = $2
		RETURNING `+automlSuggestionSelectColumns()+`
	`, jobID, suggestionID))
}

func (s *PostgresStore) ListPlanOptimizerSuggestions(planID string) ([]automl.OptimizerSuggestion, error) {
	if _, err := s.GetExperimentPlan(planID); err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+automlSuggestionSelectColumns()+`
		FROM automl_suggestions
		WHERE plan_id = $1
		ORDER BY created_at DESC
	`, planID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []automl.OptimizerSuggestion{}
	for rows.Next() {
		suggestion, err := scanOptimizerSuggestion(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, suggestion)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpsertOptimizerTrial(trial automl.OptimizerTrial) (automl.OptimizerTrial, error) {
	if err := s.requireProject(trial.ProjectID); err != nil {
		return automl.OptimizerTrial{}, err
	}
	metricsJSON, err := json.Marshal(emptyMapIfNil(trial.Metrics))
	if err != nil {
		return automl.OptimizerTrial{}, fmt.Errorf("marshal automl trial metrics: %w", err)
	}
	query := `
		INSERT INTO automl_trials (study_id, suggestion_id, project_id, plan_id, dataset_id, job_id, status, target_metric, score, metrics, error)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		ON CONFLICT (job_id) WHERE job_id <> '' DO UPDATE SET
			study_id = EXCLUDED.study_id,
			suggestion_id = EXCLUDED.suggestion_id,
			project_id = EXCLUDED.project_id,
			plan_id = EXCLUDED.plan_id,
			dataset_id = EXCLUDED.dataset_id,
			status = EXCLUDED.status,
			target_metric = EXCLUDED.target_metric,
			score = EXCLUDED.score,
			metrics = EXCLUDED.metrics,
			error = EXCLUDED.error,
			updated_at = now()
		RETURNING ` + automlTrialSelectColumns() + `
	`
	return scanOptimizerTrial(s.db.QueryRowContext(
		context.Background(),
		query,
		trial.StudyID,
		trial.SuggestionID,
		trial.ProjectID,
		trial.PlanID,
		trial.DatasetID,
		trial.JobID,
		trial.Status,
		trial.TargetMetric,
		trial.Score,
		metricsJSON,
		trial.Error,
	))
}

func (s *PostgresStore) ListProjectOptimizerTrials(projectID string, limit int) ([]automl.OptimizerTrial, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+automlTrialSelectColumns()+`
		FROM automl_trials
		WHERE project_id = $1
		ORDER BY updated_at DESC
		LIMIT $2
	`, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []automl.OptimizerTrial{}
	for rows.Next() {
		trial, err := scanOptimizerTrial(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, trial)
	}
	return out, rows.Err()
}

func (s *PostgresStore) ListStudyOptimizerTrials(studyID string) ([]automl.OptimizerTrial, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+automlTrialSelectColumns()+`
		FROM automl_trials
		WHERE study_id = $1
		ORDER BY created_at
	`, studyID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []automl.OptimizerTrial{}
	for rows.Next() {
		trial, err := scanOptimizerTrial(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, trial)
	}
	return out, rows.Err()
}
