package store

import (
	"context"
	"encoding/json"
	"fmt"

	"model-express/services/orchestrator/internal/runs"
)

func (s *PostgresStore) UpsertProjectChampion(champion runs.ProjectChampionUpsert) (runs.ProjectChampion, error) {
	if err := s.requireProject(champion.ProjectID); err != nil {
		return runs.ProjectChampion{}, err
	}
	metricsJSON, err := json.Marshal(emptyMapIfNil(champion.Metrics))
	if err != nil {
		return runs.ProjectChampion{}, fmt.Errorf("marshal champion metrics: %w", err)
	}
	evaluationJSON, err := json.Marshal(emptyMapIfNil(champion.Evaluation))
	if err != nil {
		return runs.ProjectChampion{}, fmt.Errorf("marshal champion evaluation: %w", err)
	}
	deploymentJSON, err := json.Marshal(emptyMapIfNil(champion.DeploymentProfile))
	if err != nil {
		return runs.ProjectChampion{}, fmt.Errorf("marshal champion deployment profile: %w", err)
	}
	const query = `
		INSERT INTO project_champions (project_id, dataset_id, plan_id, job_id, source_decision_id, selection_reason, metrics, evaluation, deployment_profile)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (project_id) DO UPDATE SET
			dataset_id = EXCLUDED.dataset_id,
			plan_id = EXCLUDED.plan_id,
			job_id = EXCLUDED.job_id,
			source_decision_id = EXCLUDED.source_decision_id,
			selection_reason = EXCLUDED.selection_reason,
			metrics = EXCLUDED.metrics,
			evaluation = EXCLUDED.evaluation,
			deployment_profile = EXCLUDED.deployment_profile,
			updated_at = now()
		RETURNING id, project_id, dataset_id, plan_id, job_id, source_decision_id, selection_reason, metrics, evaluation, deployment_profile, created_at, updated_at
	`
	return scanProjectChampion(s.db.QueryRowContext(
		context.Background(),
		query,
		champion.ProjectID,
		champion.DatasetID,
		champion.PlanID,
		champion.JobID,
		champion.SourceDecisionID,
		champion.SelectionReason,
		metricsJSON,
		evaluationJSON,
		deploymentJSON,
	))
}

func (s *PostgresStore) GetProjectChampion(projectID string) (runs.ProjectChampion, error) {
	const query = `
		SELECT id, project_id, dataset_id, plan_id, job_id, source_decision_id, selection_reason, metrics, evaluation, deployment_profile, created_at, updated_at
		FROM project_champions
		WHERE project_id = $1
	`
	return scanProjectChampion(s.db.QueryRowContext(context.Background(), query, projectID))
}

func (s *PostgresStore) CreateChampionExport(export runs.ChampionExportCreate) (runs.ChampionExport, error) {
	if err := s.requireProject(export.ProjectID); err != nil {
		return runs.ChampionExport{}, err
	}
	metadataJSON, err := json.Marshal(emptyMapIfNil(export.Metadata))
	if err != nil {
		return runs.ChampionExport{}, fmt.Errorf("marshal champion export metadata: %w", err)
	}
	validationErrorsJSON, err := json.Marshal(export.ValidationErrors)
	if err != nil {
		return runs.ChampionExport{}, fmt.Errorf("marshal champion export validation errors: %w", err)
	}
	const query = `
		WITH existing AS (
			SELECT id
			FROM champion_exports
			WHERE project_id = $1 AND champion_id = $2 AND format = $5
			ORDER BY created_at ASC
			LIMIT 1
		), updated AS (
			UPDATE champion_exports
			SET job_id = $3,
				status = $4,
				artifact_uri = $6,
				metadata = $7,
				validation_errors = $8,
				updated_at = now()
			WHERE id = (SELECT id FROM existing)
			RETURNING id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at
		), inserted AS (
			INSERT INTO champion_exports (project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors)
			SELECT $1, $2, $3, $4, $5, $6, $7, $8
			WHERE NOT EXISTS (SELECT 1 FROM updated)
			RETURNING id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at
		)
		SELECT id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at FROM updated
		UNION ALL
		SELECT id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at FROM inserted
		LIMIT 1
	`
	return scanChampionExport(s.db.QueryRowContext(
		context.Background(),
		query,
		export.ProjectID,
		export.ChampionID,
		export.JobID,
		export.Status,
		export.Format,
		export.ArtifactURI,
		metadataJSON,
		validationErrorsJSON,
	))
}

func (s *PostgresStore) ListProjectChampionExports(projectID string) ([]runs.ChampionExport, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	const query = `
		SELECT id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at
		FROM champion_exports
		WHERE project_id = $1
		ORDER BY created_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.ChampionExport{}
	for rows.Next() {
		export, err := scanChampionExport(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, export)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpdateChampionExport(id string, update runs.ChampionExportUpdate) (runs.ChampionExport, error) {
	metadataJSON, err := json.Marshal(emptyMapIfNil(update.Metadata))
	if err != nil {
		return runs.ChampionExport{}, fmt.Errorf("marshal champion export metadata: %w", err)
	}
	validationErrors := append([]string(nil), update.ValidationErrors...)
	if update.Error != "" {
		validationErrors = append(validationErrors, update.Error)
	}
	validationErrorsJSON, err := json.Marshal(validationErrors)
	if err != nil {
		return runs.ChampionExport{}, fmt.Errorf("marshal champion export validation errors: %w", err)
	}
	const query = `
		UPDATE champion_exports
		SET status = COALESCE(NULLIF($2, ''), status),
			artifact_uri = COALESCE(NULLIF($3, ''), artifact_uri),
			metadata = CASE WHEN $4 THEN $5 ELSE metadata END,
			validation_errors = CASE WHEN $6 THEN $7 ELSE validation_errors END,
			updated_at = now()
		WHERE id = $1
		RETURNING id, project_id, champion_id, job_id, status, format, artifact_uri, metadata, validation_errors, created_at, updated_at
	`
	return scanChampionExport(s.db.QueryRowContext(
		context.Background(),
		query,
		id,
		update.Status,
		update.ArtifactURI,
		update.Metadata != nil,
		metadataJSON,
		update.ValidationErrors != nil || update.Error != "",
		validationErrorsJSON,
	))
}

func (s *PostgresStore) CreateChampionDemoPrediction(prediction runs.ChampionDemoPredictionCreate) (runs.ChampionDemoPrediction, error) {
	if err := s.requireProject(prediction.ProjectID); err != nil {
		return runs.ChampionDemoPrediction{}, err
	}
	imageMetadataJSON, err := json.Marshal(emptyMapIfNil(prediction.ImageMetadata))
	if err != nil {
		return runs.ChampionDemoPrediction{}, fmt.Errorf("marshal champion demo prediction image metadata: %w", err)
	}
	topKJSON, err := json.Marshal(prediction.TopK)
	if err != nil {
		return runs.ChampionDemoPrediction{}, fmt.Errorf("marshal champion demo prediction top-k: %w", err)
	}
	const query = `
		INSERT INTO champion_demo_predictions (
			project_id, champion_id, job_id, dataset_id, image_uri, image_id, image_metadata,
			status, predicted_label, true_label, confidence, top_k, latency_ms, correct, error
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		RETURNING id, project_id, champion_id, job_id, dataset_id, image_uri, image_id, image_metadata,
			status, predicted_label, true_label, confidence, top_k, latency_ms, correct, error, created_at
	`
	return scanChampionDemoPrediction(s.db.QueryRowContext(
		context.Background(),
		query,
		prediction.ProjectID,
		prediction.ChampionID,
		prediction.JobID,
		prediction.DatasetID,
		prediction.ImageURI,
		prediction.ImageID,
		imageMetadataJSON,
		prediction.Status,
		prediction.PredictedLabel,
		prediction.TrueLabel,
		prediction.Confidence,
		topKJSON,
		prediction.LatencyMS,
		prediction.Correct,
		prediction.Error,
	))
}

func (s *PostgresStore) ListProjectChampionDemoPredictions(projectID string) ([]runs.ChampionDemoPrediction, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	const query = `
		SELECT id, project_id, champion_id, job_id, dataset_id, image_uri, image_id, image_metadata,
			status, predicted_label, true_label, confidence, top_k, latency_ms, correct, error, created_at
		FROM champion_demo_predictions
		WHERE project_id = $1
		ORDER BY created_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.ChampionDemoPrediction{}
	for rows.Next() {
		prediction, err := scanChampionDemoPrediction(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, prediction)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpdateChampionDemoPrediction(id string, update runs.ChampionDemoPredictionUpdate) (runs.ChampionDemoPrediction, error) {
	topKJSON, err := json.Marshal(update.TopK)
	if err != nil {
		return runs.ChampionDemoPrediction{}, fmt.Errorf("marshal champion demo prediction top-k: %w", err)
	}
	imageMetadataJSON, err := json.Marshal(emptyMapIfNil(update.ImageMetadata))
	if err != nil {
		return runs.ChampionDemoPrediction{}, fmt.Errorf("marshal champion demo prediction image metadata: %w", err)
	}
	const query = `
		UPDATE champion_demo_predictions
		SET status = COALESCE(NULLIF($2, ''), status),
			predicted_label = COALESCE(NULLIF($3, ''), predicted_label),
			true_label = COALESCE(NULLIF($4, ''), true_label),
			confidence = COALESCE($5, confidence),
			top_k = CASE WHEN $6 THEN $7 ELSE top_k END,
			latency_ms = COALESCE($8, latency_ms),
			correct = COALESCE($9, correct),
			error = COALESCE(NULLIF($10, ''), error),
			image_metadata = CASE WHEN $11 THEN $12 ELSE image_metadata END
		WHERE id = $1
		RETURNING id, project_id, champion_id, job_id, dataset_id, image_uri, image_id, image_metadata,
			status, predicted_label, true_label, confidence, top_k, latency_ms, correct, error, created_at
	`
	return scanChampionDemoPrediction(s.db.QueryRowContext(
		context.Background(),
		query,
		id,
		update.Status,
		update.PredictedLabel,
		update.TrueLabel,
		update.Confidence,
		update.TopK != nil,
		topKJSON,
		update.LatencyMS,
		update.Correct,
		update.Error,
		update.ImageMetadata != nil,
		imageMetadataJSON,
	))
}

func (s *PostgresStore) CreateChampionFeedback(feedback runs.ChampionFeedbackCreate) (runs.ChampionFeedback, error) {
	if err := s.requireProject(feedback.ProjectID); err != nil {
		return runs.ChampionFeedback{}, err
	}
	predictionSnapshotJSON, err := json.Marshal(emptyMapIfNil(feedback.PredictionSnapshot))
	if err != nil {
		return runs.ChampionFeedback{}, fmt.Errorf("marshal champion feedback prediction snapshot: %w", err)
	}
	metricsSnapshotJSON, err := json.Marshal(emptyMapIfNil(feedback.MetricsSnapshot))
	if err != nil {
		return runs.ChampionFeedback{}, fmt.Errorf("marshal champion feedback metrics snapshot: %w", err)
	}
	metadataJSON, err := json.Marshal(emptyMapIfNil(feedback.Metadata))
	if err != nil {
		return runs.ChampionFeedback{}, fmt.Errorf("marshal champion feedback metadata: %w", err)
	}
	const query = `
		INSERT INTO champion_feedback (
			project_id, champion_id, prediction_id, job_id, dataset_id, image_uri, image_id,
			rating, message, prediction_snapshot, metrics_snapshot, metadata
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		RETURNING id, project_id, champion_id, prediction_id, job_id, dataset_id, image_uri, image_id,
			rating, message, prediction_snapshot, metrics_snapshot, metadata, created_at
	`
	return scanChampionFeedback(s.db.QueryRowContext(
		context.Background(),
		query,
		feedback.ProjectID,
		feedback.ChampionID,
		feedback.PredictionID,
		feedback.JobID,
		feedback.DatasetID,
		feedback.ImageURI,
		feedback.ImageID,
		feedback.Rating,
		feedback.Message,
		predictionSnapshotJSON,
		metricsSnapshotJSON,
		metadataJSON,
	))
}

func (s *PostgresStore) ListProjectChampionFeedback(projectID string) ([]runs.ChampionFeedback, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	const query = `
		SELECT id, project_id, champion_id, prediction_id, job_id, dataset_id, image_uri, image_id,
			rating, message, prediction_snapshot, metrics_snapshot, metadata, created_at
		FROM champion_feedback
		WHERE project_id = $1
		ORDER BY created_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []runs.ChampionFeedback{}
	for rows.Next() {
		feedback, err := scanChampionFeedback(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, feedback)
	}
	return out, rows.Err()
}
