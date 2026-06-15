package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"model-express/services/orchestrator/internal/runs"
)

func (s *PostgresStore) UpsertTrainingRunSummary(jobID string, update runs.TrainingRunSummaryUpdate) (runs.TrainingRunSummary, error) {
	job, err := s.GetJob(jobID)
	if err != nil {
		return runs.TrainingRunSummary{}, err
	}

	now := time.Now().UTC()
	summary, err := s.GetTrainingRunSummary(jobID)
	if errors.Is(err, ErrNotFound) {
		summary = newPostgresTrainingRunSummaryFromJob(job, now)
	} else if err != nil {
		return runs.TrainingRunSummary{}, err
	}

	applyPostgresTrainingRunSummaryUpdate(&summary, update, now)

	const query = `
		INSERT INTO training_run_summaries (
			job_id,
			project_id,
			plan_id,
			dataset_id,
			model,
			provider,
			gpu_type,
			status,
			runtime_seconds,
			estimated_cost_usd,
			best_macro_f1,
			best_accuracy,
			final_train_loss,
			final_val_loss,
			epochs_completed,
			modal_function_call_id,
			modal_input_id,
			dataset_materialization,
			stage_telemetry,
			created_at,
			updated_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
		ON CONFLICT (job_id) DO UPDATE SET
			project_id = EXCLUDED.project_id,
			plan_id = EXCLUDED.plan_id,
			dataset_id = EXCLUDED.dataset_id,
			model = EXCLUDED.model,
			provider = EXCLUDED.provider,
			gpu_type = EXCLUDED.gpu_type,
			status = EXCLUDED.status,
			runtime_seconds = EXCLUDED.runtime_seconds,
			estimated_cost_usd = EXCLUDED.estimated_cost_usd,
			best_macro_f1 = EXCLUDED.best_macro_f1,
			best_accuracy = EXCLUDED.best_accuracy,
			final_train_loss = EXCLUDED.final_train_loss,
			final_val_loss = EXCLUDED.final_val_loss,
			epochs_completed = EXCLUDED.epochs_completed,
			modal_function_call_id = EXCLUDED.modal_function_call_id,
			modal_input_id = EXCLUDED.modal_input_id,
			dataset_materialization = EXCLUDED.dataset_materialization,
			stage_telemetry = EXCLUDED.stage_telemetry,
			updated_at = EXCLUDED.updated_at
		RETURNING job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, dataset_materialization, stage_telemetry, created_at, updated_at
	`
	datasetMaterializationJSON, err := json.Marshal(emptyMapIfNil(summary.DatasetMaterialization))
	if err != nil {
		return runs.TrainingRunSummary{}, fmt.Errorf("marshal dataset materialization: %w", err)
	}
	stageTelemetryJSON, err := json.Marshal(emptyMapIfNil(summary.StageTelemetry))
	if err != nil {
		return runs.TrainingRunSummary{}, fmt.Errorf("marshal stage telemetry: %w", err)
	}

	return scanTrainingRunSummary(s.db.QueryRowContext(
		context.Background(),
		query,
		summary.JobID,
		summary.ProjectID,
		summary.PlanID,
		summary.DatasetID,
		summary.Model,
		summary.Provider,
		summary.GPUType,
		summary.Status,
		summary.RuntimeSeconds,
		summary.EstimatedCostUSD,
		summary.BestMacroF1,
		summary.BestAccuracy,
		summary.FinalTrainLoss,
		summary.FinalValLoss,
		summary.EpochsCompleted,
		summary.ModalFunctionCallID,
		summary.ModalInputID,
		datasetMaterializationJSON,
		stageTelemetryJSON,
		summary.CreatedAt,
		summary.UpdatedAt,
	))
}

func (s *PostgresStore) GetTrainingRunSummary(jobID string) (runs.TrainingRunSummary, error) {
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, dataset_materialization, stage_telemetry, created_at, updated_at
		FROM training_run_summaries
		WHERE job_id = $1
	`

	return scanTrainingRunSummary(s.db.QueryRowContext(context.Background(), query, jobID))
}

func (s *PostgresStore) ListProjectTrainingRunSummaries(projectID string) ([]runs.TrainingRunSummary, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, dataset_materialization, stage_telemetry, created_at, updated_at
		FROM training_run_summaries
		WHERE project_id = $1
		ORDER BY updated_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.TrainingRunSummary{}
	for rows.Next() {
		summary, err := scanTrainingRunSummary(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, summary)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ListProjectTrainingRunSummariesPage(projectID string, options PageOptions) ([]runs.TrainingRunSummary, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit, offset := postgresPageLimitOffset(options)
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, dataset_materialization, stage_telemetry, created_at, updated_at
		FROM training_run_summaries
		WHERE project_id = $1
		ORDER BY updated_at DESC
		LIMIT $2 OFFSET $3
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.TrainingRunSummary{}
	for rows.Next() {
		summary, err := scanTrainingRunSummary(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, summary)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpsertTrainingRunEvaluation(jobID string, update runs.TrainingRunEvaluationUpdate) (runs.TrainingRunEvaluation, error) {
	job, err := s.GetJob(jobID)
	if err != nil {
		return runs.TrainingRunEvaluation{}, err
	}
	objectiveProfileJSON, err := json.Marshal(emptyMapIfNil(update.ObjectiveProfile))
	if err != nil {
		return runs.TrainingRunEvaluation{}, fmt.Errorf("marshal objective profile: %w", err)
	}
	perClassMetricsJSON, err := json.Marshal(emptyMapIfNil(update.PerClassMetrics))
	if err != nil {
		return runs.TrainingRunEvaluation{}, fmt.Errorf("marshal per-class metrics: %w", err)
	}
	confusionMatrixJSON, err := json.Marshal(update.ConfusionMatrix)
	if err != nil {
		return runs.TrainingRunEvaluation{}, fmt.Errorf("marshal confusion matrix: %w", err)
	}
	modelProfileJSON, err := json.Marshal(emptyMapIfNil(update.ModelProfile))
	if err != nil {
		return runs.TrainingRunEvaluation{}, fmt.Errorf("marshal model profile: %w", err)
	}
	holisticScoresJSON, err := json.Marshal(emptyMapIfNil(update.HolisticScores))
	if err != nil {
		return runs.TrainingRunEvaluation{}, fmt.Errorf("marshal holistic scores: %w", err)
	}

	const query = `
		INSERT INTO training_run_evaluations (
			job_id, project_id, plan_id, dataset_id, objective_profile, per_class_metrics, confusion_matrix, model_profile, holistic_scores, recommendation_summary
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (job_id) DO UPDATE SET
			objective_profile = EXCLUDED.objective_profile,
			per_class_metrics = EXCLUDED.per_class_metrics,
			confusion_matrix = EXCLUDED.confusion_matrix,
			model_profile = EXCLUDED.model_profile,
			holistic_scores = EXCLUDED.holistic_scores,
			recommendation_summary = EXCLUDED.recommendation_summary,
			updated_at = now()
		RETURNING job_id, project_id, plan_id, dataset_id, objective_profile, per_class_metrics, confusion_matrix, model_profile, holistic_scores, recommendation_summary, created_at, updated_at
	`
	return scanTrainingRunEvaluation(s.db.QueryRowContext(
		context.Background(),
		query,
		job.ID,
		job.ProjectID,
		postgresConfigString(job.Config, "plan_id"),
		postgresConfigString(job.Config, "dataset_id"),
		objectiveProfileJSON,
		perClassMetricsJSON,
		confusionMatrixJSON,
		modelProfileJSON,
		holisticScoresJSON,
		update.RecommendationSummary,
	))
}

func (s *PostgresStore) GetTrainingRunEvaluation(jobID string) (runs.TrainingRunEvaluation, error) {
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, objective_profile, per_class_metrics, confusion_matrix, model_profile, holistic_scores, recommendation_summary, created_at, updated_at
		FROM training_run_evaluations
		WHERE job_id = $1
	`
	return scanTrainingRunEvaluation(s.db.QueryRowContext(context.Background(), query, jobID))
}

func (s *PostgresStore) ListProjectTrainingRunEvaluations(projectID string) ([]runs.TrainingRunEvaluation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, objective_profile, per_class_metrics, confusion_matrix, model_profile, holistic_scores, recommendation_summary, created_at, updated_at
		FROM training_run_evaluations
		WHERE project_id = $1
		ORDER BY updated_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.TrainingRunEvaluation{}
	for rows.Next() {
		evaluation, err := scanTrainingRunEvaluation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, evaluation)
	}
	return out, rows.Err()
}

func (s *PostgresStore) ListProjectTrainingRunEvaluationsPage(projectID string, options PageOptions) ([]runs.TrainingRunEvaluation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit, offset := postgresPageLimitOffset(options)
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, objective_profile, per_class_metrics, confusion_matrix, model_profile, holistic_scores, recommendation_summary, created_at, updated_at
		FROM training_run_evaluations
		WHERE project_id = $1
		ORDER BY updated_at DESC
		LIMIT $2 OFFSET $3
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.TrainingRunEvaluation{}
	for rows.Next() {
		evaluation, err := scanTrainingRunEvaluation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, evaluation)
	}
	return out, rows.Err()
}
