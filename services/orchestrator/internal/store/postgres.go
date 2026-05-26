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
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/strategies"
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

func (s *PostgresStore) CreateDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error) {
	return s.insertDatasetVisualAnalysis(analysis, datasets.VisualValidationStatusAccepted)
}

func (s *PostgresStore) RejectDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error) {
	return s.insertDatasetVisualAnalysis(analysis, datasets.VisualValidationStatusRejected)
}

func (s *PostgresStore) GetLatestAcceptedDatasetVisualAnalysis(datasetID string) (datasets.DatasetVisualAnalysis, error) {
	query := `
		SELECT ` + datasetVisualAnalysisSelectColumns() + `
		FROM dataset_visual_analyses
		WHERE dataset_id = $1 AND validation_status = $2
		ORDER BY analysis_version DESC, created_at DESC
		LIMIT 1
	`
	return scanDatasetVisualAnalysis(s.db.QueryRowContext(context.Background(), query, datasetID, datasets.VisualValidationStatusAccepted))
}

func (s *PostgresStore) ListDatasetVisualAnalyses(datasetID string) ([]datasets.DatasetVisualAnalysis, error) {
	if _, err := s.GetDataset(datasetID); err != nil {
		return nil, err
	}

	query := `
		SELECT ` + datasetVisualAnalysisSelectColumns() + `
		FROM dataset_visual_analyses
		WHERE dataset_id = $1
		ORDER BY analysis_version DESC, created_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []datasets.DatasetVisualAnalysis{}
	for rows.Next() {
		analysis, err := scanDatasetVisualAnalysis(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, analysis)
	}
	return out, rows.Err()
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

	now := time.Now().UTC()
	if _, err := s.recoverExpiredJobLeasesTx(ctx, tx, now); err != nil {
		return nil, err
	}
	worker, err = scanWorker(tx.QueryRowContext(ctx, `
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
		if !isTerminalJobStatus(job.Status) {
			leaseExpiresAt := now.Add(defaultJobLeaseDuration)
			job, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET lease_owner_worker_id = $1, lease_last_heartbeat_at = now(), lease_expires_at = $2
				WHERE id = $3
				RETURNING `+jobSelectColumns()+`
			`, workerID, leaseExpiresAt, job.ID))
			if err != nil {
				return nil, err
			}
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
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
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
		SET worker_id = $1,
			status = $2,
			started_at = now(),
			attempt = attempt + 1,
			lease_owner_worker_id = $1,
			lease_last_heartbeat_at = now(),
			lease_expires_at = $4
		WHERE id = $3
		RETURNING `+jobSelectColumns()+`
	`, workerID, jobs.StatusAssigned, job.ID, now.Add(defaultJobLeaseDuration)))
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
		INSERT INTO experiment_jobs (project_id, template, status, config, max_attempts)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
	`

	return scanJob(s.db.QueryRowContext(context.Background(), query, projectID, template, jobs.StatusQueued, configJSON, defaultJobMaxAttempts))
}

func (s *PostgresStore) GetJob(id string) (jobs.ExperimentJob, error) {
	return scanJob(s.db.QueryRowContext(context.Background(), selectJobSQL("id"), id))
}

func (s *PostgresStore) ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
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

func (s *PostgresStore) RecoverExpiredJobLeases(now time.Time) ([]jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	recovered, err := s.recoverExpiredJobLeasesTx(ctx, tx, now.UTC())
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return recovered, nil
}

func (s *PostgresStore) recoverExpiredJobLeasesTx(ctx context.Context, tx *sql.Tx, now time.Time) ([]jobs.ExperimentJob, error) {
	rows, err := tx.QueryContext(ctx, `
		SELECT `+jobSelectColumns()+`
		FROM experiment_jobs
		WHERE status IN ($1, $2)
			AND lease_expires_at IS NOT NULL
			AND lease_expires_at <= $3
		ORDER BY lease_expires_at ASC
		FOR UPDATE
	`, jobs.StatusAssigned, jobs.StatusRunning, now)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	expired := []jobs.ExperimentJob{}
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		expired = append(expired, job)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	recovered := []jobs.ExperimentJob{}
	for _, job := range expired {
		if job.MaxAttempts < 1 {
			job.MaxAttempts = defaultJobMaxAttempts
		}
		var updated jobs.ExperimentJob
		if job.Attempt >= job.MaxAttempts {
			updated, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET status = $1,
					error = $2,
					worker_id = '',
					completed_at = $3,
					lease_owner_worker_id = '',
					lease_expires_at = NULL,
					lease_last_heartbeat_at = NULL
				WHERE id = $4
				RETURNING `+jobSelectColumns()+`
			`, jobs.StatusFailed, "job lease expired after maximum attempts", now, job.ID))
		} else {
			updated, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET status = $1,
					error = '',
					worker_id = '',
					started_at = NULL,
					lease_owner_worker_id = '',
					lease_expires_at = NULL,
					lease_last_heartbeat_at = NULL
				WHERE id = $2
				RETURNING `+jobSelectColumns()+`
			`, jobs.StatusQueued, job.ID))
		}
		if err != nil {
			return nil, err
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = CASE WHEN status = $3 THEN status ELSE $1 END,
				current_job_id = ''
			WHERE current_job_id = $2
		`, workers.StatusIdle, job.ID, workers.StatusOffline); err != nil {
			return nil, err
		}
		recovered = append(recovered, updated)
	}

	return recovered, nil
}

func (s *PostgresStore) ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error) {
	if epoch < 1 {
		return jobs.EpochMetric{}, fmt.Errorf("%w: epoch must be positive", ErrInvalidRequest)
	}
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
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET lease_owner_worker_id = $1, lease_last_heartbeat_at = now(), lease_expires_at = $2
			WHERE id = $3
		`, job.WorkerID, time.Now().UTC().Add(defaultJobLeaseDuration), jobID); err != nil {
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

func (s *PostgresStore) CreateAgentDecision(projectID string, planID string, decisionType string, rationale string, payload map[string]any) (decisions.AgentDecision, error) {
	if err := s.requireProject(projectID); err != nil {
		return decisions.AgentDecision{}, err
	}
	if payload == nil {
		payload = map[string]any{}
	}

	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return decisions.AgentDecision{}, fmt.Errorf("marshal agent decision payload: %w", err)
	}

	const query = `
		INSERT INTO agent_decisions (project_id, plan_id, decision_type, rationale, payload)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, plan_id, decision_type, rationale, payload, created_at
	`

	return scanAgentDecision(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		planID,
		decisionType,
		rationale,
		payloadJSON,
	))
}

func (s *PostgresStore) ListProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, plan_id, decision_type, rationale, payload, created_at
		FROM agent_decisions
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []decisions.AgentDecision{}
	for rows.Next() {
		decision, err := scanAgentDecision(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, decision)
	}

	return out, rows.Err()
}

func (s *PostgresStore) GetAutomationSettings() (settings.AutomationSettings, error) {
	const query = `
		SELECT auto_review_experiments, auto_schedule_followups, auto_execute_plans, max_followup_rounds, default_training_provider, default_gpu_type, llm_enabled, agent_mode, llm_provider, llm_model, updated_at
		FROM automation_settings
		WHERE singleton = true
	`

	return scanAutomationSettings(s.db.QueryRowContext(context.Background(), query))
}

func (s *PostgresStore) SaveAutomationSettings(automationSettings settings.AutomationSettings) (settings.AutomationSettings, error) {
	const query = `
		INSERT INTO automation_settings (
			singleton,
			auto_review_experiments,
			auto_schedule_followups,
			auto_execute_plans,
			max_followup_rounds,
			default_training_provider,
			default_gpu_type,
			llm_enabled,
			agent_mode,
			llm_provider,
			llm_model
		)
		VALUES (true, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (singleton) DO UPDATE SET
			auto_review_experiments = EXCLUDED.auto_review_experiments,
			auto_schedule_followups = EXCLUDED.auto_schedule_followups,
			auto_execute_plans = EXCLUDED.auto_execute_plans,
			max_followup_rounds = EXCLUDED.max_followup_rounds,
			default_training_provider = EXCLUDED.default_training_provider,
			default_gpu_type = EXCLUDED.default_gpu_type,
			llm_enabled = EXCLUDED.llm_enabled,
			agent_mode = EXCLUDED.agent_mode,
			llm_provider = EXCLUDED.llm_provider,
			llm_model = EXCLUDED.llm_model,
			updated_at = now()
		RETURNING auto_review_experiments, auto_schedule_followups, auto_execute_plans, max_followup_rounds, default_training_provider, default_gpu_type, llm_enabled, agent_mode, llm_provider, llm_model, updated_at
	`

	return scanAutomationSettings(s.db.QueryRowContext(
		context.Background(),
		query,
		automationSettings.AutoReviewExperiments,
		automationSettings.AutoScheduleFollowUps,
		automationSettings.AutoExecutePlans,
		automationSettings.MaxFollowUpRounds,
		automationSettings.DefaultTrainingProvider,
		automationSettings.DefaultGPUType,
		automationSettings.LLMEnabled,
		automationSettings.AgentMode,
		automationSettings.LLMProvider,
		automationSettings.LLMModel,
	))
}

func (s *PostgresStore) UpsertWorkerRequirement(projectID string, planID string, provider string, gpuType string, targetCount int, source string) (execution.WorkerRequirement, bool, error) {
	if err := s.requireProject(projectID); err != nil {
		return execution.WorkerRequirement{}, false, err
	}
	if targetCount < 1 {
		return execution.WorkerRequirement{}, false, fmt.Errorf("%w: target_count must be at least 1", ErrInvalidRequest)
	}

	existing, err := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), `
		SELECT id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
		FROM worker_requirements
		WHERE project_id = $1 AND plan_id = $2
	`, projectID, planID))
	if err == nil {
		status := existing.Status
		lastError := existing.LastError
		if existing.TargetCount != targetCount || status == execution.WorkerRequirementFailed || status == execution.WorkerRequirementCancelled {
			status = execution.WorkerRequirementPending
			lastError = ""
		}
		const updateQuery = `
			UPDATE worker_requirements
			SET provider = $1, gpu_type = $2, target_count = $3, source = $4, status = $5, last_error = $6, updated_at = now()
			WHERE id = $7
			RETURNING id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
		`
		requirement, updateErr := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), updateQuery, provider, gpuType, targetCount, source, status, lastError, existing.ID))
		return requirement, false, updateErr
	}
	if !errors.Is(err, ErrNotFound) {
		return execution.WorkerRequirement{}, false, err
	}

	const insertQuery = `
		INSERT INTO worker_requirements (project_id, plan_id, provider, gpu_type, target_count, status, source)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
	`
	requirement, err := scanWorkerRequirement(s.db.QueryRowContext(
		context.Background(),
		insertQuery,
		projectID,
		planID,
		provider,
		gpuType,
		targetCount,
		execution.WorkerRequirementPending,
		source,
	))
	return requirement, true, err
}

func (s *PostgresStore) ListProjectWorkerRequirements(projectID string) ([]execution.WorkerRequirement, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
		FROM worker_requirements
		WHERE project_id = $1
		ORDER BY updated_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []execution.WorkerRequirement{}
	for rows.Next() {
		requirement, err := scanWorkerRequirement(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, requirement)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpdateWorkerRequirement(id string, update execution.WorkerRequirementUpdate) (execution.WorkerRequirement, error) {
	requirement, err := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), `
		SELECT id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
		FROM worker_requirements
		WHERE id = $1
	`, id))
	if err != nil {
		return execution.WorkerRequirement{}, err
	}
	if update.Status != nil {
		requirement.Status = *update.Status
	}
	if update.LastError != nil {
		requirement.LastError = *update.LastError
	}

	const query = `
		UPDATE worker_requirements
		SET status = $1, last_error = $2, updated_at = now()
		WHERE id = $3
		RETURNING id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
	`
	return scanWorkerRequirement(s.db.QueryRowContext(context.Background(), query, requirement.Status, requirement.LastError, id))
}

func (s *PostgresStore) CreateExecutionEvent(projectID string, planID string, eventType string, message string, payload map[string]any) (execution.ExecutionEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return execution.ExecutionEvent{}, err
	}
	if payload == nil {
		payload = map[string]any{}
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return execution.ExecutionEvent{}, fmt.Errorf("marshal execution event payload: %w", err)
	}

	const query = `
		INSERT INTO execution_events (project_id, plan_id, event_type, message, payload)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, plan_id, event_type, message, payload, created_at
	`
	return scanExecutionEvent(s.db.QueryRowContext(context.Background(), query, projectID, planID, eventType, message, payloadJSON))
}

func (s *PostgresStore) ListProjectExecutionEvents(projectID string, limit int) ([]execution.ExecutionEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	const query = `
		SELECT id, project_id, plan_id, event_type, message, payload, created_at
		FROM execution_events
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []execution.ExecutionEvent{}
	for rows.Next() {
		event, err := scanExecutionEvent(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, rows.Err()
}

func (s *PostgresStore) CreateAgentInvocation(invocation memory.AgentInvocation) (memory.AgentInvocation, error) {
	if err := s.requireProject(invocation.ProjectID); err != nil {
		return memory.AgentInvocation{}, err
	}
	if invocation.InputMessages == nil {
		invocation.InputMessages = []map[string]string{}
	}
	if invocation.InputContext == nil {
		invocation.InputContext = map[string]any{}
	}
	if invocation.ParsedOutput == nil {
		invocation.ParsedOutput = map[string]any{}
	}
	if invocation.HumanFeedback == nil {
		invocation.HumanFeedback = map[string]any{}
	}
	if invocation.DownstreamOutcome == nil {
		invocation.DownstreamOutcome = map[string]any{}
	}

	inputMessagesJSON, err := json.Marshal(invocation.InputMessages)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation input messages: %w", err)
	}
	inputContextJSON, err := json.Marshal(invocation.InputContext)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation input context: %w", err)
	}
	parsedOutputJSON, err := json.Marshal(invocation.ParsedOutput)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation parsed output: %w", err)
	}
	humanFeedbackJSON, err := json.Marshal(invocation.HumanFeedback)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation human feedback: %w", err)
	}
	downstreamOutcomeJSON, err := json.Marshal(invocation.DownstreamOutcome)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation downstream outcome: %w", err)
	}

	const query = `
		INSERT INTO agent_invocations (
			project_id,
			dataset_id,
			plan_id,
			job_id,
			agent_name,
			agent_version,
			prompt_version,
			provider,
			model,
			input_messages,
			input_context,
			raw_output,
			parsed_output,
			validation_status,
			validation_error,
			accepted_for_memory,
			human_feedback,
			downstream_outcome
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
		RETURNING id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
	`
	return scanAgentInvocation(s.db.QueryRowContext(
		context.Background(),
		query,
		invocation.ProjectID,
		invocation.DatasetID,
		invocation.PlanID,
		invocation.JobID,
		invocation.AgentName,
		invocation.AgentVersion,
		invocation.PromptVersion,
		invocation.Provider,
		invocation.Model,
		inputMessagesJSON,
		inputContextJSON,
		invocation.RawOutput,
		parsedOutputJSON,
		invocation.ValidationStatus,
		invocation.ValidationError,
		invocation.AcceptedForMemory,
		humanFeedbackJSON,
		downstreamOutcomeJSON,
	))
}

func (s *PostgresStore) GetAgentInvocation(invocationID string) (memory.AgentInvocation, error) {
	const query = `
		SELECT id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
		FROM agent_invocations
		WHERE id = $1
	`
	return scanAgentInvocation(s.db.QueryRowContext(context.Background(), query, invocationID))
}

func (s *PostgresStore) UpdateAgentInvocationDownstreamOutcome(invocationID string, outcome map[string]any) (memory.AgentInvocation, error) {
	if outcome == nil {
		outcome = map[string]any{}
	}
	outcomeJSON, err := json.Marshal(outcome)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation downstream outcome: %w", err)
	}

	const query = `
		UPDATE agent_invocations
		SET downstream_outcome = $2
		WHERE id = $1
		RETURNING id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
	`
	return scanAgentInvocation(s.db.QueryRowContext(context.Background(), query, invocationID, outcomeJSON))
}

func (s *PostgresStore) ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	const query = `
		SELECT id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
		FROM agent_invocations
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR plan_id = $3)
			AND ($4 = '' OR job_id = $4)
			AND ($5 = '' OR agent_name = $5)
		ORDER BY created_at DESC
		LIMIT $6
	`
	rows, err := s.db.QueryContext(
		context.Background(),
		query,
		projectID,
		filter.DatasetID,
		filter.PlanID,
		filter.JobID,
		filter.AgentName,
		filter.Limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []memory.AgentInvocation{}
	for rows.Next() {
		invocation, err := scanAgentInvocation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, invocation)
	}
	return out, rows.Err()
}

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

func (s *PostgresStore) CreateAgentMemoryRecord(record memory.AgentMemoryRecord) (memory.AgentMemoryRecord, error) {
	if err := s.requireProject(record.ProjectID); err != nil {
		return memory.AgentMemoryRecord{}, err
	}
	if record.Payload == nil {
		record.Payload = map[string]any{}
	}
	if record.Tags == nil {
		record.Tags = []string{}
	}

	payloadJSON, err := json.Marshal(record.Payload)
	if err != nil {
		return memory.AgentMemoryRecord{}, fmt.Errorf("marshal agent memory payload: %w", err)
	}
	tagsJSON, err := json.Marshal(record.Tags)
	if err != nil {
		return memory.AgentMemoryRecord{}, fmt.Errorf("marshal agent memory tags: %w", err)
	}

	const query = `
		INSERT INTO agent_memory_records (invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id, invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags, created_at
	`
	return scanAgentMemoryRecord(s.db.QueryRowContext(
		context.Background(),
		query,
		record.InvocationID,
		record.ProjectID,
		record.DatasetID,
		record.PlanID,
		record.JobID,
		record.AgentName,
		record.Kind,
		record.Summary,
		payloadJSON,
		tagsJSON,
	))
}

func (s *PostgresStore) ListProjectAgentMemoryRecords(projectID string, filter memory.AgentMemoryFilter) ([]memory.AgentMemoryRecord, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	const query = `
		SELECT id, invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags, created_at
		FROM agent_memory_records
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR plan_id = $3)
			AND ($4 = '' OR job_id = $4)
			AND ($5 = '' OR kind = $5)
		ORDER BY created_at DESC
		LIMIT $6
	`
	rows, err := s.db.QueryContext(
		context.Background(),
		query,
		projectID,
		filter.DatasetID,
		filter.PlanID,
		filter.JobID,
		filter.Kind,
		filter.Limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []memory.AgentMemoryRecord{}
	for rows.Next() {
		record, err := scanAgentMemoryRecord(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

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
		SET status = $1,
			mlflow_run_id = $2,
			error = $3,
			completed_at = now(),
			lease_owner_worker_id = '',
			lease_expires_at = NULL,
			lease_last_heartbeat_at = NULL
		WHERE id = $4
		RETURNING `+jobSelectColumns()+`
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

func (s *PostgresStore) insertDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis, validationStatus string) (datasets.DatasetVisualAnalysis, error) {
	dataset, err := s.GetDataset(analysis.DatasetID)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, err
	}
	if analysis.ProjectID == "" {
		analysis.ProjectID = dataset.ProjectID
	}
	if analysis.ProjectID != dataset.ProjectID {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}
	if analysis.DatasetName == "" {
		analysis.DatasetName = dataset.Name
	}
	if analysis.SchemaVersion == "" {
		analysis.SchemaVersion = datasets.VisualAnalysisSchemaVersion
	}
	if analysis.AgentName == "" {
		analysis.AgentName = datasets.VisualAnalysisAgentName
	}
	if analysis.TriggerReason == "" {
		analysis.TriggerReason = datasets.VisualTriggerManual
	}
	if analysis.TriggerDetails == nil {
		analysis.TriggerDetails = map[string]any{}
	}
	if analysis.CoverageReport.PerClassCounts == nil {
		analysis.CoverageReport.PerClassCounts = map[string]int{}
	}
	if analysis.ClassesToWatch == nil {
		analysis.ClassesToWatch = []datasets.ClassWatchItem{}
	}
	if analysis.VisualTraits == nil {
		analysis.VisualTraits = []datasets.VisualTrait{}
	}
	if analysis.PreprocessingHypotheses == nil {
		analysis.PreprocessingHypotheses = []datasets.PreprocessingHypothesis{}
	}
	if analysis.Cautions == nil {
		analysis.Cautions = []datasets.VisualCaution{}
	}
	if analysis.Limitations == nil {
		analysis.Limitations = []string{}
	}
	if analysis.ValidationErrors == nil {
		analysis.ValidationErrors = []string{}
	}

	triggerDetailsJSON, err := json.Marshal(analysis.TriggerDetails)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis trigger details: %w", err)
	}
	coverageReportJSON, err := json.Marshal(analysis.CoverageReport)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis coverage report: %w", err)
	}
	classesToWatchJSON, err := json.Marshal(analysis.ClassesToWatch)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis classes to watch: %w", err)
	}
	visualTraitsJSON, err := json.Marshal(analysis.VisualTraits)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis traits: %w", err)
	}
	preprocessingHypothesesJSON, err := json.Marshal(analysis.PreprocessingHypotheses)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis preprocessing hypotheses: %w", err)
	}
	cautionsJSON, err := json.Marshal(analysis.Cautions)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis cautions: %w", err)
	}
	limitationsJSON, err := json.Marshal(analysis.Limitations)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis limitations: %w", err)
	}
	validationErrorsJSON, err := json.Marshal(analysis.ValidationErrors)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis validation errors: %w", err)
	}

	query := `
		INSERT INTO dataset_visual_analyses (
			project_id,
			dataset_id,
			dataset_name,
			schema_version,
			analysis_version,
			prompt_version,
			agent_name,
			agent_version,
			provider,
			model,
			trigger_reason,
			trigger_details,
			source_job_id,
			source_invocation_id,
			profile_schema_version,
			profile_fingerprint,
			total_images,
			images_analyzed,
			coverage_report,
			classes_to_watch,
			confidence,
			visual_traits,
			preprocessing_hypotheses,
			cautions,
			limitations,
			validation_status,
			validation_errors
		)
		VALUES (
			$1,
			$2,
			$3,
			$4,
			COALESCE((SELECT MAX(analysis_version) + 1 FROM dataset_visual_analyses WHERE dataset_id = $2), 1),
			$5,
			$6,
			$7,
			$8,
			$9,
			$10,
			$11,
			$12,
			$13,
			$14,
			$15,
			$16,
			$17,
			$18,
			$19,
			$20,
			$21,
			$22,
			$23,
			$24,
			$25,
			$26
		)
		RETURNING ` + datasetVisualAnalysisSelectColumns() + `
	`
	return scanDatasetVisualAnalysis(s.db.QueryRowContext(
		context.Background(),
		query,
		analysis.ProjectID,
		analysis.DatasetID,
		analysis.DatasetName,
		analysis.SchemaVersion,
		analysis.PromptVersion,
		analysis.AgentName,
		analysis.AgentVersion,
		analysis.Provider,
		analysis.Model,
		string(analysis.TriggerReason),
		triggerDetailsJSON,
		analysis.SourceJobID,
		analysis.SourceInvocationID,
		analysis.ProfileSchemaVersion,
		analysis.ProfileFingerprint,
		analysis.TotalImages,
		analysis.ImagesAnalyzed,
		coverageReportJSON,
		classesToWatchJSON,
		analysis.Confidence,
		visualTraitsJSON,
		preprocessingHypothesesJSON,
		cautionsJSON,
		limitationsJSON,
		validationStatus,
		validationErrorsJSON,
	))
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

func scanDatasetVisualAnalysis(row rowScanner) (datasets.DatasetVisualAnalysis, error) {
	var analysis datasets.DatasetVisualAnalysis
	var triggerDetailsJSON []byte
	var coverageReportJSON []byte
	var classesToWatchJSON []byte
	var visualTraitsJSON []byte
	var preprocessingHypothesesJSON []byte
	var cautionsJSON []byte
	var limitationsJSON []byte
	var validationErrorsJSON []byte
	var triggerReason string

	if err := row.Scan(
		&analysis.ID,
		&analysis.ProjectID,
		&analysis.DatasetID,
		&analysis.DatasetName,
		&analysis.SchemaVersion,
		&analysis.AnalysisVersion,
		&analysis.PromptVersion,
		&analysis.AgentName,
		&analysis.AgentVersion,
		&analysis.Provider,
		&analysis.Model,
		&triggerReason,
		&triggerDetailsJSON,
		&analysis.SourceJobID,
		&analysis.SourceInvocationID,
		&analysis.ProfileSchemaVersion,
		&analysis.ProfileFingerprint,
		&analysis.TotalImages,
		&analysis.ImagesAnalyzed,
		&coverageReportJSON,
		&classesToWatchJSON,
		&analysis.Confidence,
		&visualTraitsJSON,
		&preprocessingHypothesesJSON,
		&cautionsJSON,
		&limitationsJSON,
		&analysis.ValidationStatus,
		&validationErrorsJSON,
		&analysis.CreatedAt,
		&analysis.UpdatedAt,
	); err != nil {
		return datasets.DatasetVisualAnalysis{}, normalizeSQLError(err)
	}

	analysis.TriggerReason = datasets.VisualReanalysisTrigger(triggerReason)
	analysis.TriggerDetails = map[string]any{}
	if len(triggerDetailsJSON) > 0 {
		if err := json.Unmarshal(triggerDetailsJSON, &analysis.TriggerDetails); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis trigger details: %w", err)
		}
	}
	if len(coverageReportJSON) > 0 {
		if err := json.Unmarshal(coverageReportJSON, &analysis.CoverageReport); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis coverage report: %w", err)
		}
	}
	analysis.ClassesToWatch = []datasets.ClassWatchItem{}
	if len(classesToWatchJSON) > 0 {
		if err := json.Unmarshal(classesToWatchJSON, &analysis.ClassesToWatch); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis classes to watch: %w", err)
		}
	}
	analysis.VisualTraits = []datasets.VisualTrait{}
	if len(visualTraitsJSON) > 0 {
		if err := json.Unmarshal(visualTraitsJSON, &analysis.VisualTraits); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis traits: %w", err)
		}
	}
	analysis.PreprocessingHypotheses = []datasets.PreprocessingHypothesis{}
	if len(preprocessingHypothesesJSON) > 0 {
		if err := json.Unmarshal(preprocessingHypothesesJSON, &analysis.PreprocessingHypotheses); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis preprocessing hypotheses: %w", err)
		}
	}
	analysis.Cautions = []datasets.VisualCaution{}
	if len(cautionsJSON) > 0 {
		if err := json.Unmarshal(cautionsJSON, &analysis.Cautions); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis cautions: %w", err)
		}
	}
	analysis.Limitations = []string{}
	if len(limitationsJSON) > 0 {
		if err := json.Unmarshal(limitationsJSON, &analysis.Limitations); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis limitations: %w", err)
		}
	}
	analysis.ValidationErrors = []string{}
	if len(validationErrorsJSON) > 0 {
		if err := json.Unmarshal(validationErrorsJSON, &analysis.ValidationErrors); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis validation errors: %w", err)
		}
	}

	return analysis, nil
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
	var leaseExpiresAt sql.NullTime
	var leaseLastHeartbeatAt sql.NullTime

	if err := row.Scan(
		&job.ID,
		&job.ProjectID,
		&job.WorkerID,
		&job.Template,
		&job.Status,
		&configJSON,
		&job.MLflowRunID,
		&job.Error,
		&job.Attempt,
		&job.MaxAttempts,
		&job.LeaseOwnerWorkerID,
		&leaseExpiresAt,
		&leaseLastHeartbeatAt,
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
	if leaseExpiresAt.Valid {
		job.LeaseExpiresAt = &leaseExpiresAt.Time
	}
	if leaseLastHeartbeatAt.Valid {
		job.LeaseLastHeartbeatAt = &leaseLastHeartbeatAt.Time
	}
	if job.MaxAttempts < 1 {
		job.MaxAttempts = defaultJobMaxAttempts
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

func scanTrainingRunEvaluation(row rowScanner) (runs.TrainingRunEvaluation, error) {
	var evaluation runs.TrainingRunEvaluation
	var objectiveProfileJSON []byte
	var perClassMetricsJSON []byte
	var confusionMatrixJSON []byte
	var modelProfileJSON []byte
	var holisticScoresJSON []byte
	if err := row.Scan(
		&evaluation.JobID,
		&evaluation.ProjectID,
		&evaluation.PlanID,
		&evaluation.DatasetID,
		&objectiveProfileJSON,
		&perClassMetricsJSON,
		&confusionMatrixJSON,
		&modelProfileJSON,
		&holisticScoresJSON,
		&evaluation.RecommendationSummary,
		&evaluation.CreatedAt,
		&evaluation.UpdatedAt,
	); err != nil {
		return runs.TrainingRunEvaluation{}, normalizeSQLError(err)
	}

	evaluation.ObjectiveProfile = map[string]any{}
	if len(objectiveProfileJSON) > 0 {
		if err := json.Unmarshal(objectiveProfileJSON, &evaluation.ObjectiveProfile); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal objective profile: %w", err)
		}
	}
	evaluation.PerClassMetrics = map[string]any{}
	if len(perClassMetricsJSON) > 0 {
		if err := json.Unmarshal(perClassMetricsJSON, &evaluation.PerClassMetrics); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal per-class metrics: %w", err)
		}
	}
	evaluation.ConfusionMatrix = [][]int{}
	if len(confusionMatrixJSON) > 0 {
		if err := json.Unmarshal(confusionMatrixJSON, &evaluation.ConfusionMatrix); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal confusion matrix: %w", err)
		}
	}
	evaluation.ModelProfile = map[string]any{}
	if len(modelProfileJSON) > 0 {
		if err := json.Unmarshal(modelProfileJSON, &evaluation.ModelProfile); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal model profile: %w", err)
		}
	}
	evaluation.HolisticScores = map[string]any{}
	if len(holisticScoresJSON) > 0 {
		if err := json.Unmarshal(holisticScoresJSON, &evaluation.HolisticScores); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal holistic scores: %w", err)
		}
	}
	return evaluation, nil
}

func scanProjectChampion(row rowScanner) (runs.ProjectChampion, error) {
	var champion runs.ProjectChampion
	var metricsJSON []byte
	var evaluationJSON []byte
	var deploymentProfileJSON []byte
	if err := row.Scan(
		&champion.ID,
		&champion.ProjectID,
		&champion.DatasetID,
		&champion.PlanID,
		&champion.JobID,
		&champion.SourceDecisionID,
		&champion.SelectionReason,
		&metricsJSON,
		&evaluationJSON,
		&deploymentProfileJSON,
		&champion.CreatedAt,
		&champion.UpdatedAt,
	); err != nil {
		return runs.ProjectChampion{}, normalizeSQLError(err)
	}

	champion.Metrics = map[string]any{}
	if len(metricsJSON) > 0 {
		if err := json.Unmarshal(metricsJSON, &champion.Metrics); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion metrics: %w", err)
		}
	}
	champion.Evaluation = map[string]any{}
	if len(evaluationJSON) > 0 {
		if err := json.Unmarshal(evaluationJSON, &champion.Evaluation); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion evaluation: %w", err)
		}
	}
	champion.DeploymentProfile = map[string]any{}
	if len(deploymentProfileJSON) > 0 {
		if err := json.Unmarshal(deploymentProfileJSON, &champion.DeploymentProfile); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion deployment profile: %w", err)
		}
	}
	return champion, nil
}

func scanChampionExport(row rowScanner) (runs.ChampionExport, error) {
	var export runs.ChampionExport
	var metadataJSON []byte
	var validationErrorsJSON []byte
	if err := row.Scan(
		&export.ID,
		&export.ProjectID,
		&export.ChampionID,
		&export.JobID,
		&export.Status,
		&export.Format,
		&export.ArtifactURI,
		&metadataJSON,
		&validationErrorsJSON,
		&export.CreatedAt,
		&export.UpdatedAt,
	); err != nil {
		return runs.ChampionExport{}, normalizeSQLError(err)
	}

	export.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &export.Metadata); err != nil {
			return runs.ChampionExport{}, fmt.Errorf("unmarshal champion export metadata: %w", err)
		}
	}
	export.ValidationErrors = []string{}
	if len(validationErrorsJSON) > 0 {
		if err := json.Unmarshal(validationErrorsJSON, &export.ValidationErrors); err != nil {
			return runs.ChampionExport{}, fmt.Errorf("unmarshal champion export validation errors: %w", err)
		}
	}
	return export, nil
}

func scanChampionDemoPrediction(row rowScanner) (runs.ChampionDemoPrediction, error) {
	var prediction runs.ChampionDemoPrediction
	var imageMetadataJSON []byte
	var topKJSON []byte
	var confidence sql.NullFloat64
	var latencyMS sql.NullFloat64
	var correct sql.NullBool
	if err := row.Scan(
		&prediction.ID,
		&prediction.ProjectID,
		&prediction.ChampionID,
		&prediction.JobID,
		&prediction.DatasetID,
		&prediction.ImageURI,
		&prediction.ImageID,
		&imageMetadataJSON,
		&prediction.Status,
		&prediction.PredictedLabel,
		&prediction.TrueLabel,
		&confidence,
		&topKJSON,
		&latencyMS,
		&correct,
		&prediction.Error,
		&prediction.CreatedAt,
	); err != nil {
		return runs.ChampionDemoPrediction{}, normalizeSQLError(err)
	}

	prediction.ImageMetadata = map[string]any{}
	if len(imageMetadataJSON) > 0 {
		if err := json.Unmarshal(imageMetadataJSON, &prediction.ImageMetadata); err != nil {
			return runs.ChampionDemoPrediction{}, fmt.Errorf("unmarshal champion demo prediction image metadata: %w", err)
		}
	}
	prediction.TopK = []runs.DemoPredictionTopK{}
	if len(topKJSON) > 0 {
		if err := json.Unmarshal(topKJSON, &prediction.TopK); err != nil {
			return runs.ChampionDemoPrediction{}, fmt.Errorf("unmarshal champion demo prediction top-k: %w", err)
		}
	}
	if confidence.Valid {
		prediction.Confidence = &confidence.Float64
	}
	if latencyMS.Valid {
		prediction.LatencyMS = &latencyMS.Float64
	}
	if correct.Valid {
		prediction.Correct = &correct.Bool
	}
	return prediction, nil
}

func scanAgentDecision(row rowScanner) (decisions.AgentDecision, error) {
	var decision decisions.AgentDecision
	var payloadJSON []byte

	if err := row.Scan(
		&decision.ID,
		&decision.ProjectID,
		&decision.PlanID,
		&decision.DecisionType,
		&decision.Rationale,
		&payloadJSON,
		&decision.CreatedAt,
	); err != nil {
		return decisions.AgentDecision{}, normalizeSQLError(err)
	}

	decision.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &decision.Payload); err != nil {
			return decisions.AgentDecision{}, fmt.Errorf("unmarshal agent decision payload: %w", err)
		}
	}

	return decision, nil
}

func scanWorkerRequirement(row rowScanner) (execution.WorkerRequirement, error) {
	var requirement execution.WorkerRequirement
	if err := row.Scan(
		&requirement.ID,
		&requirement.ProjectID,
		&requirement.PlanID,
		&requirement.Provider,
		&requirement.GPUType,
		&requirement.TargetCount,
		&requirement.Status,
		&requirement.Source,
		&requirement.LastError,
		&requirement.CreatedAt,
		&requirement.UpdatedAt,
	); err != nil {
		return execution.WorkerRequirement{}, normalizeSQLError(err)
	}
	return requirement, nil
}

func scanExecutionEvent(row rowScanner) (execution.ExecutionEvent, error) {
	var event execution.ExecutionEvent
	var payloadJSON []byte
	if err := row.Scan(
		&event.ID,
		&event.ProjectID,
		&event.PlanID,
		&event.EventType,
		&event.Message,
		&payloadJSON,
		&event.CreatedAt,
	); err != nil {
		return execution.ExecutionEvent{}, normalizeSQLError(err)
	}
	event.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &event.Payload); err != nil {
			return execution.ExecutionEvent{}, fmt.Errorf("unmarshal execution event payload: %w", err)
		}
	}
	return event, nil
}

func scanAgentMemoryRecord(row rowScanner) (memory.AgentMemoryRecord, error) {
	var record memory.AgentMemoryRecord
	var payloadJSON []byte
	var tagsJSON []byte
	if err := row.Scan(
		&record.ID,
		&record.InvocationID,
		&record.ProjectID,
		&record.DatasetID,
		&record.PlanID,
		&record.JobID,
		&record.AgentName,
		&record.Kind,
		&record.Summary,
		&payloadJSON,
		&tagsJSON,
		&record.CreatedAt,
	); err != nil {
		return memory.AgentMemoryRecord{}, normalizeSQLError(err)
	}
	record.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &record.Payload); err != nil {
			return memory.AgentMemoryRecord{}, fmt.Errorf("unmarshal agent memory payload: %w", err)
		}
	}
	record.Tags = []string{}
	if len(tagsJSON) > 0 {
		if err := json.Unmarshal(tagsJSON, &record.Tags); err != nil {
			return memory.AgentMemoryRecord{}, fmt.Errorf("unmarshal agent memory tags: %w", err)
		}
	}
	return record, nil
}

func scanAgentInvocation(row rowScanner) (memory.AgentInvocation, error) {
	var invocation memory.AgentInvocation
	var inputMessagesJSON []byte
	var inputContextJSON []byte
	var parsedOutputJSON []byte
	var humanFeedbackJSON []byte
	var downstreamOutcomeJSON []byte

	if err := row.Scan(
		&invocation.ID,
		&invocation.ProjectID,
		&invocation.DatasetID,
		&invocation.PlanID,
		&invocation.JobID,
		&invocation.AgentName,
		&invocation.AgentVersion,
		&invocation.PromptVersion,
		&invocation.Provider,
		&invocation.Model,
		&inputMessagesJSON,
		&inputContextJSON,
		&invocation.RawOutput,
		&parsedOutputJSON,
		&invocation.ValidationStatus,
		&invocation.ValidationError,
		&invocation.AcceptedForMemory,
		&humanFeedbackJSON,
		&downstreamOutcomeJSON,
		&invocation.CreatedAt,
	); err != nil {
		return memory.AgentInvocation{}, normalizeSQLError(err)
	}

	invocation.InputMessages = []map[string]string{}
	if len(inputMessagesJSON) > 0 {
		if err := json.Unmarshal(inputMessagesJSON, &invocation.InputMessages); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation input messages: %w", err)
		}
	}
	invocation.InputContext = map[string]any{}
	if len(inputContextJSON) > 0 {
		if err := json.Unmarshal(inputContextJSON, &invocation.InputContext); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation input context: %w", err)
		}
	}
	invocation.ParsedOutput = map[string]any{}
	if len(parsedOutputJSON) > 0 {
		if err := json.Unmarshal(parsedOutputJSON, &invocation.ParsedOutput); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation parsed output: %w", err)
		}
	}
	invocation.HumanFeedback = map[string]any{}
	if len(humanFeedbackJSON) > 0 {
		if err := json.Unmarshal(humanFeedbackJSON, &invocation.HumanFeedback); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation human feedback: %w", err)
		}
	}
	invocation.DownstreamOutcome = map[string]any{}
	if len(downstreamOutcomeJSON) > 0 {
		if err := json.Unmarshal(downstreamOutcomeJSON, &invocation.DownstreamOutcome); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation downstream outcome: %w", err)
		}
	}

	return invocation, nil
}

func scanStrategyScorecard(row rowScanner) (strategies.StrategyScorecard, error) {
	var scorecard strategies.StrategyScorecard
	var diagnosisTriggersJSON []byte
	var evidenceUsedJSON []byte
	var datasetTraitsJSON []byte
	var objectiveProfileJSON []byte
	var proposedChangesJSON []byte
	var tagsJSON []byte
	if err := row.Scan(
		&scorecard.ID,
		&scorecard.ProjectID,
		&scorecard.DatasetID,
		&scorecard.SourceDecisionID,
		&scorecard.SourcePlanID,
		&scorecard.FollowUpPlanID,
		&scorecard.StrategyType,
		&scorecard.PlanningMode,
		&scorecard.Mechanism,
		&scorecard.Intervention,
		&diagnosisTriggersJSON,
		&evidenceUsedJSON,
		&scorecard.ExpectedEffect,
		&datasetTraitsJSON,
		&objectiveProfileJSON,
		&proposedChangesJSON,
		&scorecard.ExpectedDelta,
		&scorecard.ActualDelta,
		&scorecard.ConfidenceBefore,
		&scorecard.ConfidenceAfter,
		&scorecard.CostUSD,
		&scorecard.RuntimeSeconds,
		&scorecard.Outcome,
		&scorecard.Lesson,
		&tagsJSON,
		&scorecard.CreatedAt,
	); err != nil {
		return strategies.StrategyScorecard{}, normalizeSQLError(err)
	}
	scorecard.DiagnosisTriggers = []string{}
	if len(diagnosisTriggersJSON) > 0 {
		if err := json.Unmarshal(diagnosisTriggersJSON, &scorecard.DiagnosisTriggers); err != nil {
			return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard diagnosis_triggers: %w", err)
		}
	}
	scorecard.EvidenceUsed = []string{}
	if len(evidenceUsedJSON) > 0 {
		if err := json.Unmarshal(evidenceUsedJSON, &scorecard.EvidenceUsed); err != nil {
			return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard evidence_used: %w", err)
		}
	}
	if err := json.Unmarshal(datasetTraitsJSON, &scorecard.DatasetTraits); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard dataset_traits: %w", err)
	}
	if err := json.Unmarshal(objectiveProfileJSON, &scorecard.ObjectiveProfile); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard objective_profile: %w", err)
	}
	if err := json.Unmarshal(proposedChangesJSON, &scorecard.ProposedChanges); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard proposed_changes: %w", err)
	}
	if err := json.Unmarshal(tagsJSON, &scorecard.Tags); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard tags: %w", err)
	}
	return scorecard, nil
}

func scanAutomationSettings(row rowScanner) (settings.AutomationSettings, error) {
	var automationSettings settings.AutomationSettings
	if err := row.Scan(
		&automationSettings.AutoReviewExperiments,
		&automationSettings.AutoScheduleFollowUps,
		&automationSettings.AutoExecutePlans,
		&automationSettings.MaxFollowUpRounds,
		&automationSettings.DefaultTrainingProvider,
		&automationSettings.DefaultGPUType,
		&automationSettings.LLMEnabled,
		&automationSettings.AgentMode,
		&automationSettings.LLMProvider,
		&automationSettings.LLMModel,
		&automationSettings.UpdatedAt,
	); err != nil {
		return settings.AutomationSettings{}, normalizeSQLError(err)
	}

	return automationSettings, nil
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
		&plan.SourceDecisionID,
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
		SELECT `+jobSelectColumns()+`
		FROM experiment_jobs
		WHERE %s = $1
	`, column)
}

func jobSelectColumns() string {
	return "id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at"
}

func datasetVisualAnalysisSelectColumns() string {
	return "id, project_id, dataset_id, dataset_name, schema_version, analysis_version, prompt_version, agent_name, agent_version, provider, model, trigger_reason, trigger_details, source_job_id, source_invocation_id, profile_schema_version, profile_fingerprint, total_images, images_analyzed, coverage_report, classes_to_watch, confidence, visual_traits, preprocessing_hypotheses, cautions, limitations, validation_status, validation_errors, created_at, updated_at"
}

func strategyScorecardSelectColumns() string {
	return "id, project_id, dataset_id, source_decision_id, source_plan_id, followup_plan_id, strategy_type, planning_mode, mechanism, intervention, diagnosis_triggers, evidence_used, expected_effect, dataset_traits, objective_profile, proposed_changes, expected_delta, actual_delta, confidence_before, confidence_after, cost_usd, runtime_seconds, outcome, lesson, tags, created_at"
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
