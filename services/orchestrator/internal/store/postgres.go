package store

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/workers"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

type PostgresStore struct {
	db *sql.DB
}

type rowScanner interface {
	Scan(dest ...any) error
}

func NewPostgresStore(ctx context.Context, databaseURL string) (*PostgresStore, error) {
	db, err := sql.Open("pgx", databaseURL)
	if err != nil {
		return nil, fmt.Errorf("open postgres: %w", err)
	}

	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(30 * time.Minute)

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	store := &PostgresStore{db: db}
	if err := store.migrate(ctx); err != nil {
		return nil, err
	}

	return store, nil
}

func (s *PostgresStore) Close() error {
	return s.db.Close()
}

func (s *PostgresStore) migrate(ctx context.Context) error {
	entries, err := migrationFiles.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		sqlText, err := migrationFiles.ReadFile("migrations/" + entry.Name())
		if err != nil {
			return fmt.Errorf("read migration %s: %w", entry.Name(), err)
		}

		if _, err := s.db.ExecContext(ctx, string(sqlText)); err != nil {
			return fmt.Errorf("apply migration %s: %w", entry.Name(), err)
		}
	}

	return nil
}

func (s *PostgresStore) CreateProject(name string, goal string) (projects.Project, error) {
	const query = `
		INSERT INTO projects (name, goal, status)
		VALUES ($1, $2, $3)
		RETURNING id, name, goal, status, created_at, updated_at
	`

	return scanProject(s.db.QueryRowContext(context.Background(), query, name, goal, projects.StatusCreated))
}

func (s *PostgresStore) GetProject(id string) (projects.Project, error) {
	const query = `
		SELECT id, name, goal, status, created_at, updated_at
		FROM projects
		WHERE id = $1
	`

	return scanProject(s.db.QueryRowContext(context.Background(), query, id))
}

func (s *PostgresStore) ListProjects() ([]projects.Project, error) {
	const query = `
		SELECT id, name, goal, status, created_at, updated_at
		FROM projects
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []projects.Project{}
	for rows.Next() {
		project, err := scanProject(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, project)
	}

	return out, rows.Err()
}

func (s *PostgresStore) CreateDataset(projectID string, name string, storageURI string, checksumSHA256 string, sizeBytes int64) (datasets.Dataset, error) {
	if err := s.requireProject(projectID); err != nil {
		return datasets.Dataset{}, err
	}
	if name == "" {
		return datasets.Dataset{}, fmt.Errorf("%w: name is required", ErrInvalidRequest)
	}
	if storageURI == "" {
		return datasets.Dataset{}, fmt.Errorf("%w: storage_uri is required", ErrInvalidRequest)
	}

	const query = `
		INSERT INTO datasets (project_id, name, storage_uri, checksum_sha256, size_bytes, status)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
	`

	return scanDataset(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		name,
		storageURI,
		checksumSHA256,
		sizeBytes,
		datasets.StatusRegistered,
	))
}

func (s *PostgresStore) GetDataset(id string) (datasets.Dataset, error) {
	const query = `
		SELECT id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
		FROM datasets
		WHERE id = $1
	`

	return scanDataset(s.db.QueryRowContext(context.Background(), query, id))
}

func (s *PostgresStore) ListProjectDatasets(projectID string) ([]datasets.Dataset, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
		FROM datasets
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []datasets.Dataset{}
	for rows.Next() {
		dataset, err := scanDataset(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, dataset)
	}

	return out, rows.Err()
}

func (s *PostgresStore) UpdateDatasetProfile(id string, profile map[string]any) (datasets.Dataset, error) {
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		return datasets.Dataset{}, fmt.Errorf("marshal dataset profile: %w", err)
	}

	const query = `
		UPDATE datasets
		SET profile = $1, status = $2, profiled_at = now()
		WHERE id = $3
		RETURNING id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
	`

	return scanDataset(s.db.QueryRowContext(context.Background(), query, profileJSON, datasets.StatusProfiled, id))
}

func (s *PostgresStore) RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error) {
	if projectID == "" {
		return workers.Worker{}, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(projectID); err != nil {
		return workers.Worker{}, err
	}
	if err := s.requireProjectDataset(projectID); err != nil {
		return workers.Worker{}, err
	}

	const query = `
		INSERT INTO workers (project_id, name, status, gpu_type)
		VALUES ($1, $2, $3, $4)
		RETURNING id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, projectID, name, workers.StatusIdle, gpuType))
}

func (s *PostgresStore) ListWorkers() ([]workers.Worker, error) {
	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		ORDER BY last_heartbeat DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []workers.Worker{}
	for rows.Next() {
		worker, err := scanWorker(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, worker)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ListProjectWorkers(projectID string) ([]workers.Worker, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE project_id = $1
		ORDER BY last_heartbeat DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []workers.Worker{}
	for rows.Next() {
		worker, err := scanWorker(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, worker)
	}

	return out, rows.Err()
}

func (s *PostgresStore) GetWorker(workerID string) (workers.Worker, error) {
	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id=$1
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, workerID))
}

func (s *PostgresStore) HeartbeatWorker(id string) (workers.Worker, error) {
	const query = `
		UPDATE workers
		SET last_heartbeat = now()
		WHERE id = $1
		RETURNING id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, id))
}

func (s *PostgresStore) PollJob(workerID string) (*jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	worker, err := scanWorker(tx.QueryRowContext(ctx, `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id = $1
		FOR UPDATE
	`, workerID))
	if err != nil {
		return nil, err
	}

	if worker.CurrentJobID != "" {
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); err != nil {
			return nil, err
		}

		job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id"), worker.CurrentJobID))
		if err != nil {
			return nil, err
		}

		if err := tx.Commit(); err != nil {
			return nil, err
		}

		return &job, nil
	}

	if worker.ProjectID == "" {
		if _, updateErr := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); updateErr != nil {
			return nil, updateErr
		}
		if commitErr := tx.Commit(); commitErr != nil {
			return nil, commitErr
		}
		return nil, ErrNoJob
	}

	job, err := scanJob(tx.QueryRowContext(ctx, `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE status = $1 AND project_id = $2
		ORDER BY created_at
		LIMIT 1
		FOR UPDATE SKIP LOCKED
	`, jobs.StatusQueued, worker.ProjectID))
	if errors.Is(err, ErrNotFound) {
		if _, updateErr := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); updateErr != nil {
			return nil, updateErr
		}

		if commitErr := tx.Commit(); commitErr != nil {
			return nil, commitErr
		}

		return nil, ErrNoJob
	}
	if err != nil {
		return nil, err
	}

	assignedJob, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET worker_id = $1, status = $2, started_at = now()
		WHERE id = $3
		RETURNING id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
	`, workerID, jobs.StatusAssigned, job.ID))
	if err != nil {
		return nil, err
	}

	if _, err := tx.ExecContext(ctx, `
		UPDATE workers
		SET status = $1, current_job_id = $2, last_heartbeat = now()
		WHERE id = $3
	`, workers.StatusRunning, assignedJob.ID, workerID); err != nil {
		return nil, err
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}

	return &assignedJob, nil
}

func (s *PostgresStore) CreateJob(projectID string, template string, config map[string]any) (jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return jobs.ExperimentJob{}, err
	}
	if err := s.requireDatasetConfig(projectID, config); err != nil {
		return jobs.ExperimentJob{}, err
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return jobs.ExperimentJob{}, fmt.Errorf("marshal job config: %w", err)
	}

	const query = `
		INSERT INTO experiment_jobs (project_id, template, status, config)
		VALUES ($1, $2, $3, $4)
		RETURNING id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
	`

	return scanJob(s.db.QueryRowContext(context.Background(), query, projectID, template, jobs.StatusQueued, configJSON))
}

func (s *PostgresStore) GetJob(id string) (jobs.ExperimentJob, error) {
	return scanJob(s.db.QueryRowContext(context.Background(), selectJobSQL("id"), id))
}

func (s *PostgresStore) ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.ExperimentJob{}
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, job)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.EpochMetric{}, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.EpochMetric{}, err
	}

	if job.Status == jobs.StatusAssigned {
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET status = $1
			WHERE id = $2
		`, jobs.StatusRunning, jobID); err != nil {
			return jobs.EpochMetric{}, err
		}
	}

	if job.WorkerID != "" {
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = $1, last_heartbeat = now()
			WHERE id = $2
		`, workers.StatusRunning, job.WorkerID); err != nil {
			return jobs.EpochMetric{}, err
		}
	}

	metricsJSON, err := json.Marshal(values)
	if err != nil {
		return jobs.EpochMetric{}, fmt.Errorf("marshal metrics: %w", err)
	}

	metric, err := scanMetric(tx.QueryRowContext(ctx, `
		INSERT INTO epoch_metrics (job_id, epoch, metrics)
		VALUES ($1, $2, $3)
		ON CONFLICT (job_id, epoch)
		DO UPDATE SET metrics = EXCLUDED.metrics, created_at = now()
		RETURNING job_id, epoch, metrics, created_at
	`, jobID, epoch, metricsJSON))
	if err != nil {
		return jobs.EpochMetric{}, err
	}

	if err := tx.Commit(); err != nil {
		return jobs.EpochMetric{}, err
	}

	return metric, nil
}

func (s *PostgresStore) ListJobMetrics(jobID string) ([]jobs.EpochMetric, error) {
	if _, err := s.GetJob(jobID); err != nil {
		return nil, err
	}

	const query = `
		SELECT job_id, epoch, metrics, created_at
		FROM epoch_metrics
		WHERE job_id = $1
		ORDER BY epoch
	`

	rows, err := s.db.QueryContext(context.Background(), query, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.EpochMetric{}
	for rows.Next() {
		metric, err := scanMetric(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, metric)
	}

	return out, rows.Err()
}

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
			created_at,
			updated_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
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
			updated_at = EXCLUDED.updated_at
		RETURNING job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, created_at, updated_at
	`

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
		summary.CreatedAt,
		summary.UpdatedAt,
	))
}

func (s *PostgresStore) GetTrainingRunSummary(jobID string) (runs.TrainingRunSummary, error) {
	const query = `
		SELECT job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, created_at, updated_at
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
		SELECT job_id, project_id, plan_id, dataset_id, model, provider, gpu_type, status, runtime_seconds, estimated_cost_usd, best_macro_f1, best_accuracy, final_train_loss, final_val_loss, epochs_completed, modal_function_call_id, modal_input_id, created_at, updated_at
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

func (s *PostgresStore) CreateExperimentPlan(projectID string, datasetID string, targetMetric string, recommendedWorkers int, estimatedMinutes int, experiments []plans.PlannedExperiment, warnings []string) (plans.ExperimentPlan, error) {
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
			target_metric,
			recommended_workers,
			estimated_minutes,
			experiments,
			warnings
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id, project_id, dataset_id, status, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
	`

	return scanExperimentPlan(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		datasetID,
		plans.StatusProposed,
		targetMetric,
		recommendedWorkers,
		estimatedMinutes,
		experimentsJSON,
		warningsJSON,
	))
}

func (s *PostgresStore) GetExperimentPlan(id string) (plans.ExperimentPlan, error) {
	const query = `
		SELECT id, project_id, dataset_id, status, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
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
		SELECT id, project_id, dataset_id, status, target_metric, recommended_workers, estimated_minutes, experiments, warnings, created_at
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

func (s *PostgresStore) CompleteJob(jobID string, mlflowRunID string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusSucceeded, mlflowRunID, "")
}

func (s *PostgresStore) FailJob(jobID string, message string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusFailed, "", message)
}

func (s *PostgresStore) finishJob(jobID string, status string, mlflowRunID string, message string) (jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET status = $1, mlflow_run_id = $2, error = $3, completed_at = now()
		WHERE id = $4
		RETURNING id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
	`, status, mlflowRunID, message, jobID))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}

	if job.WorkerID != "" {
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = $1, current_job_id = '', last_heartbeat = now()
			WHERE id = $2
		`, workers.StatusIdle, job.WorkerID); err != nil {
			return jobs.ExperimentJob{}, err
		}
	}

	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, err
	}

	return job, nil
}

func (s *PostgresStore) requireProject(projectID string) error {
	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1)
	`, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return ErrNotFound
	}

	return nil
}

func (s *PostgresStore) requireProjectDataset(projectID string) error {
	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM datasets WHERE project_id = $1)
	`, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("%w: project must have a dataset before workers or jobs can be created", ErrInvalidRequest)
	}

	return nil
}

func (s *PostgresStore) requireDatasetBelongsToProject(projectID string, datasetID string) error {
	if datasetID == "" {
		return fmt.Errorf("%w: dataset_id is required", ErrInvalidRequest)
	}

	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM datasets WHERE id = $1 AND project_id = $2)
	`, datasetID, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}

	return nil
}

func (s *PostgresStore) requireDatasetConfig(projectID string, config map[string]any) error {
	value, ok := config["dataset_id"]
	if !ok {
		return fmt.Errorf("%w: job config must include dataset_id", ErrInvalidRequest)
	}

	datasetID, ok := value.(string)
	if !ok || datasetID == "" {
		return fmt.Errorf("%w: dataset_id must be a non-empty string", ErrInvalidRequest)
	}

	return s.requireDatasetBelongsToProject(projectID, datasetID)
}

func scanProject(row rowScanner) (projects.Project, error) {
	var project projects.Project
	if err := row.Scan(
		&project.ID,
		&project.Name,
		&project.Goal,
		&project.Status,
		&project.CreatedAt,
		&project.UpdatedAt,
	); err != nil {
		return projects.Project{}, normalizeSQLError(err)
	}

	return project, nil
}

func scanDataset(row rowScanner) (datasets.Dataset, error) {
	var dataset datasets.Dataset
	var profileJSON []byte
	var profiledAt sql.NullTime

	if err := row.Scan(
		&dataset.ID,
		&dataset.ProjectID,
		&dataset.Name,
		&dataset.StorageURI,
		&dataset.ChecksumSHA256,
		&dataset.SizeBytes,
		&profileJSON,
		&dataset.Status,
		&dataset.CreatedAt,
		&profiledAt,
	); err != nil {
		return datasets.Dataset{}, normalizeSQLError(err)
	}

	dataset.Profile = map[string]any{}
	if len(profileJSON) > 0 {
		if err := json.Unmarshal(profileJSON, &dataset.Profile); err != nil {
			return datasets.Dataset{}, fmt.Errorf("unmarshal dataset profile: %w", err)
		}
	}

	if profiledAt.Valid {
		dataset.ProfiledAt = &profiledAt.Time
	}

	return dataset, nil
}

func scanWorker(row rowScanner) (workers.Worker, error) {
	var worker workers.Worker
	if err := row.Scan(
		&worker.ID,
		&worker.ProjectID,
		&worker.Name,
		&worker.Status,
		&worker.GPUType,
		&worker.LastHeartbeat,
		&worker.CurrentJobID,
	); err != nil {
		return workers.Worker{}, normalizeSQLError(err)
	}

	if time.Since(worker.LastHeartbeat) > workers.HeartbeatLimit {
		worker.Status = workers.StatusOffline
	}

	return worker, nil
}

func scanJob(row rowScanner) (jobs.ExperimentJob, error) {
	var job jobs.ExperimentJob
	var configJSON []byte
	var startedAt sql.NullTime
	var completedAt sql.NullTime

	if err := row.Scan(
		&job.ID,
		&job.ProjectID,
		&job.WorkerID,
		&job.Template,
		&job.Status,
		&configJSON,
		&job.MLflowRunID,
		&job.Error,
		&job.CreatedAt,
		&startedAt,
		&completedAt,
	); err != nil {
		return jobs.ExperimentJob{}, normalizeSQLError(err)
	}

	if len(configJSON) > 0 {
		if err := json.Unmarshal(configJSON, &job.Config); err != nil {
			return jobs.ExperimentJob{}, fmt.Errorf("unmarshal job config: %w", err)
		}
	}

	if startedAt.Valid {
		job.StartedAt = &startedAt.Time
	}
	if completedAt.Valid {
		job.CompletedAt = &completedAt.Time
	}

	return job, nil
}

func scanMetric(row rowScanner) (jobs.EpochMetric, error) {
	var metric jobs.EpochMetric
	var metricsJSON []byte

	if err := row.Scan(&metric.JobID, &metric.Epoch, &metricsJSON, &metric.CreatedAt); err != nil {
		return jobs.EpochMetric{}, normalizeSQLError(err)
	}

	if err := json.Unmarshal(metricsJSON, &metric.Metrics); err != nil {
		return jobs.EpochMetric{}, fmt.Errorf("unmarshal metrics: %w", err)
	}

	return metric, nil
}

func scanTrainingRunSummary(row rowScanner) (runs.TrainingRunSummary, error) {
	var summary runs.TrainingRunSummary
	if err := row.Scan(
		&summary.JobID,
		&summary.ProjectID,
		&summary.PlanID,
		&summary.DatasetID,
		&summary.Model,
		&summary.Provider,
		&summary.GPUType,
		&summary.Status,
		&summary.RuntimeSeconds,
		&summary.EstimatedCostUSD,
		&summary.BestMacroF1,
		&summary.BestAccuracy,
		&summary.FinalTrainLoss,
		&summary.FinalValLoss,
		&summary.EpochsCompleted,
		&summary.ModalFunctionCallID,
		&summary.ModalInputID,
		&summary.CreatedAt,
		&summary.UpdatedAt,
	); err != nil {
		return runs.TrainingRunSummary{}, normalizeSQLError(err)
	}

	return summary, nil
}

func scanExperimentPlan(row rowScanner) (plans.ExperimentPlan, error) {
	var plan plans.ExperimentPlan
	var experimentsJSON []byte
	var warningsJSON []byte

	if err := row.Scan(
		&plan.ID,
		&plan.ProjectID,
		&plan.DatasetID,
		&plan.Status,
		&plan.TargetMetric,
		&plan.RecommendedWorkers,
		&plan.EstimatedMinutes,
		&experimentsJSON,
		&warningsJSON,
		&plan.CreatedAt,
	); err != nil {
		return plans.ExperimentPlan{}, normalizeSQLError(err)
	}

	plan.Experiments = []plans.PlannedExperiment{}
	if len(experimentsJSON) > 0 {
		if err := json.Unmarshal(experimentsJSON, &plan.Experiments); err != nil {
			return plans.ExperimentPlan{}, fmt.Errorf("unmarshal planned experiments: %w", err)
		}
	}
	if plan.Experiments == nil {
		plan.Experiments = []plans.PlannedExperiment{}
	}

	plan.Warnings = []string{}
	if len(warningsJSON) > 0 {
		if err := json.Unmarshal(warningsJSON, &plan.Warnings); err != nil {
			return plans.ExperimentPlan{}, fmt.Errorf("unmarshal experiment plan warnings: %w", err)
		}
	}
	if plan.Warnings == nil {
		plan.Warnings = []string{}
	}

	return plan, nil
}

func normalizeSQLError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return ErrNotFound
	}

	return err
}

func selectJobSQL(column string) string {
	return fmt.Sprintf(`
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE %s = $1
	`, column)
}

func newPostgresTrainingRunSummaryFromJob(job jobs.ExperimentJob, now time.Time) runs.TrainingRunSummary {
	provider := postgresConfigString(job.Config, "provider")
	if provider == "" {
		provider = "local"
	}

	return runs.TrainingRunSummary{
		JobID:     job.ID,
		ProjectID: job.ProjectID,
		PlanID:    postgresConfigString(job.Config, "plan_id"),
		DatasetID: postgresConfigString(job.Config, "dataset_id"),
		Model:     postgresConfigString(job.Config, "model"),
		Provider:  provider,
		GPUType:   postgresConfigString(job.Config, "gpu_type"),
		Status:    job.Status,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func applyPostgresTrainingRunSummaryUpdate(summary *runs.TrainingRunSummary, update runs.TrainingRunSummaryUpdate, now time.Time) {
	if update.Model != "" {
		summary.Model = update.Model
	}
	if update.Provider != "" {
		summary.Provider = update.Provider
	}
	if update.GPUType != "" {
		summary.GPUType = update.GPUType
	}
	if update.Status != "" {
		summary.Status = update.Status
	}
	if update.RuntimeSeconds != nil {
		summary.RuntimeSeconds = *update.RuntimeSeconds
	}
	if update.EstimatedCostUSD != nil {
		summary.EstimatedCostUSD = *update.EstimatedCostUSD
	}
	if update.BestMacroF1 != nil {
		summary.BestMacroF1 = *update.BestMacroF1
	}
	if update.BestAccuracy != nil {
		summary.BestAccuracy = *update.BestAccuracy
	}
	if update.FinalTrainLoss != nil {
		summary.FinalTrainLoss = *update.FinalTrainLoss
	}
	if update.FinalValLoss != nil {
		summary.FinalValLoss = *update.FinalValLoss
	}
	if update.EpochsCompleted != nil {
		summary.EpochsCompleted = *update.EpochsCompleted
	}
	if update.ModalFunctionCallID != "" {
		summary.ModalFunctionCallID = update.ModalFunctionCallID
	}
	if update.ModalInputID != "" {
		summary.ModalInputID = update.ModalInputID
	}

	summary.UpdatedAt = now
}

func postgresConfigString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}
