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
		const updateQuery = `
			UPDATE worker_requirements
			SET provider = $1, gpu_type = $2, target_count = $3, source = $4, updated_at = now()
			WHERE id = $5
			RETURNING id, project_id, plan_id, provider, gpu_type, target_count, status, source, last_error, created_at, updated_at
		`
		requirement, updateErr := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), updateQuery, provider, gpuType, targetCount, source, existing.ID))
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
