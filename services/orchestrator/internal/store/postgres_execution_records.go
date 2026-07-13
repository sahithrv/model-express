package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/execution"
)

func insertJobExecutionSpecTx(ctx context.Context, tx *sql.Tx, spec execution.JobExecutionSpec) error {
	acceptedJSON, err := json.Marshal(spec.AcceptedSpec)
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO job_execution_specs (job_id, project_id, schema_version, capability_version, task, runner, requested_config_hash, accepted_spec_hash, accepted_spec, created_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`, spec.JobID, spec.ProjectID, spec.SchemaVersion, spec.CapabilityVersion, spec.Task, spec.Runner, spec.RequestedConfigHash, spec.AcceptedSpecHash, acceptedJSON, spec.CreatedAt)
	return err
}

func createAttemptExecutionRecordTx(ctx context.Context, tx *sql.Tx, jobID, attemptID string, attemptNumber int) (execution.AttemptExecutionRecord, error) {
	row := tx.QueryRowContext(ctx, `
		INSERT INTO attempt_execution_records (job_id, project_id, attempt_id, attempt_number)
		SELECT $1, project_id, $2, $3 FROM job_execution_specs WHERE job_id = $1
		ON CONFLICT (job_id, attempt_id) DO UPDATE SET attempt_id = EXCLUDED.attempt_id
		RETURNING id, job_id, project_id, attempt_id, attempt_number, lifecycle_status, fidelity_verdict, realized_effective_hash, adjustment_reason_codes, latest_realized_config, created_at, updated_at
	`, jobID, attemptID, attemptNumber)
	return scanAttemptExecutionRecord(row)
}

func (s *PostgresStore) CreateAttemptExecutionRecord(jobID, attemptID string, attemptNumber int) (execution.AttemptExecutionRecord, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return execution.AttemptExecutionRecord{}, err
	}
	defer tx.Rollback()
	record, err := createAttemptExecutionRecordTx(ctx, tx, jobID, attemptID, attemptNumber)
	if err != nil {
		return execution.AttemptExecutionRecord{}, err
	}
	if err := tx.Commit(); err != nil {
		return execution.AttemptExecutionRecord{}, err
	}
	return record, nil
}

func (s *PostgresStore) GetJobExecutionRecord(jobID string) (execution.ExecutionRecord, error) {
	spec, err := scanJobExecutionSpec(s.db.QueryRowContext(context.Background(), `SELECT job_id, project_id, schema_version, capability_version, task, runner, requested_config_hash, accepted_spec_hash, accepted_spec, created_at FROM job_execution_specs WHERE job_id=$1`, jobID))
	if err != nil {
		return execution.ExecutionRecord{}, err
	}
	attempts, err := s.listAttemptExecutionRecords(`WHERE job_id=$1`, jobID)
	if err != nil {
		return execution.ExecutionRecord{}, err
	}
	return execution.ExecutionRecord{AcceptedSpec: spec, Attempts: attempts}, nil
}

func (s *PostgresStore) ListProjectExecutionRecords(projectID string, options PageOptions) ([]execution.ExecutionRecord, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit, offset := postgresPageLimitOffset(options)
	rows, err := s.db.QueryContext(context.Background(), `SELECT job_id, project_id, schema_version, capability_version, task, runner, requested_config_hash, accepted_spec_hash, accepted_spec, created_at FROM job_execution_specs WHERE project_id=$1 ORDER BY created_at DESC LIMIT $2 OFFSET $3`, projectID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []execution.ExecutionRecord{}
	for rows.Next() {
		spec, err := scanJobExecutionSpec(rows)
		if err != nil {
			return nil, err
		}
		attempts, err := s.listAttemptExecutionRecords(`WHERE job_id=$1`, spec.JobID)
		if err != nil {
			return nil, err
		}
		out = append(out, execution.ExecutionRecord{AcceptedSpec: spec, Attempts: attempts})
	}
	return out, rows.Err()
}

func (s *PostgresStore) listAttemptExecutionRecords(where string, arg any) ([]execution.AttemptExecutionRecord, error) {
	rows, err := s.db.QueryContext(context.Background(), `SELECT id, job_id, project_id, attempt_id, attempt_number, lifecycle_status, fidelity_verdict, realized_effective_hash, adjustment_reason_codes, latest_realized_config, created_at, updated_at FROM attempt_execution_records `+where+` ORDER BY attempt_number ASC`, arg)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []execution.AttemptExecutionRecord{}
	for rows.Next() {
		record, err := scanAttemptExecutionRecord(rows)
		if err != nil {
			return nil, err
		}
		observations, err := s.listRealizationObservations(record.ID, 20)
		if err != nil {
			return nil, err
		}
		record.Observations = observations
		out = append(out, record)
	}
	return out, rows.Err()
}

func (s *PostgresStore) listRealizationObservations(attemptRecordID string, limit int) ([]execution.RealizationObservation, error) {
	rows, err := s.db.QueryContext(context.Background(), observationSelectSQL()+` WHERE attempt_record_id=$1 ORDER BY created_at ASC LIMIT $2`, attemptRecordID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []execution.RealizationObservation{}
	for rows.Next() {
		item, err := scanRealizationObservation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *PostgresStore) AppendRealizationObservation(jobID string, create execution.RealizationObservationCreate) (execution.RealizationObservation, bool, error) {
	if err := execution.ValidateObservation(create); err != nil {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	create.FrameworkArguments = execution.RedactSensitiveMap(create.FrameworkArguments)
	create.Evidence = execution.RedactSensitiveMap(create.Evidence)
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return execution.RealizationObservation{}, false, err
	}
	defer tx.Rollback()
	spec, err := scanJobExecutionSpec(tx.QueryRowContext(ctx, `SELECT job_id, project_id, schema_version, capability_version, task, runner, requested_config_hash, accepted_spec_hash, accepted_spec, created_at FROM job_execution_specs WHERE job_id=$1`, jobID))
	if err != nil {
		return execution.RealizationObservation{}, false, err
	}
	record, err := scanAttemptExecutionRecord(tx.QueryRowContext(ctx, `SELECT id, job_id, project_id, attempt_id, attempt_number, lifecycle_status, fidelity_verdict, realized_effective_hash, adjustment_reason_codes, latest_realized_config, created_at, updated_at FROM attempt_execution_records WHERE job_id=$1 AND attempt_id=$2 FOR UPDATE`, jobID, create.AttemptID))
	if err != nil {
		return execution.RealizationObservation{}, false, err
	}
	existing, err := scanRealizationObservation(tx.QueryRowContext(ctx, observationSelectSQL()+` WHERE attempt_record_id=$1 AND idempotency_key=$2`, record.ID, create.IdempotencyKey))
	if err == nil {
		if err := tx.Commit(); err != nil {
			return execution.RealizationObservation{}, false, err
		}
		return existing, false, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return execution.RealizationObservation{}, false, err
	}
	if record.LifecycleStatus == execution.ExecutionLifecycleFinalized {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: attempt realization is already finalized", ErrInvalidRequest)
	}
	realized, hash, verdict, adjustmentReasonCodes, err := execution.DeriveRealization(spec, create)
	if err != nil {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	realizedJSON, _ := jsonMap(realized)
	frameworkJSON, _ := jsonMap(create.FrameworkArguments)
	evidenceJSON, _ := jsonMap(create.Evidence)
	adjustmentReasonsJSON, _ := json.Marshal(adjustmentReasonCodes)
	stage := strings.ToUpper(strings.TrimSpace(create.Stage))
	observation, err := scanRealizationObservation(tx.QueryRowContext(ctx, `INSERT INTO execution_realization_observations (attempt_record_id, attempt_id, schema_version, stage, idempotency_key, realized_config, framework_arguments, evidence, adjustment_policy, adjustment_reason_codes, simulated, realized_effective_hash, fidelity_verdict) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING id, attempt_record_id, attempt_id, schema_version, stage, idempotency_key, realized_config, framework_arguments, evidence, adjustment_policy, adjustment_reason_codes, simulated, realized_effective_hash, fidelity_verdict, created_at`, record.ID, create.AttemptID, execution.ExecutionObservationSchemaV1, stage, create.IdempotencyKey, realizedJSON, frameworkJSON, evidenceJSON, create.AdjustmentPolicy, adjustmentReasonsJSON, create.Simulated, hash, verdict))
	if err != nil {
		return execution.RealizationObservation{}, false, err
	}
	lifecycle := execution.ExecutionLifecycleInitialized
	if stage == execution.ExecutionObservationFinalized {
		lifecycle = execution.ExecutionLifecycleFinalized
	}
	if _, err := tx.ExecContext(ctx, `UPDATE attempt_execution_records SET lifecycle_status=$1, fidelity_verdict=$2, realized_effective_hash=$3, adjustment_reason_codes=$4, latest_realized_config=$5, updated_at=now() WHERE id=$6`, lifecycle, verdict, hash, adjustmentReasonsJSON, realizedJSON, record.ID); err != nil {
		return execution.RealizationObservation{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return execution.RealizationObservation{}, false, err
	}
	return observation, true, nil
}

func (s *PostgresStore) MarkAttemptNotRealized(jobID, attemptID string) (execution.AttemptExecutionRecord, error) {
	return scanAttemptExecutionRecord(s.db.QueryRowContext(context.Background(), `UPDATE attempt_execution_records SET lifecycle_status=CASE WHEN lifecycle_status=$1 THEN $2 ELSE lifecycle_status END, updated_at=CASE WHEN lifecycle_status=$1 THEN now() ELSE updated_at END WHERE job_id=$3 AND attempt_id=$4 RETURNING id, job_id, project_id, attempt_id, attempt_number, lifecycle_status, fidelity_verdict, realized_effective_hash, adjustment_reason_codes, latest_realized_config, created_at, updated_at`, execution.ExecutionLifecyclePending, execution.ExecutionLifecycleNotRealized, jobID, attemptID))
}

func observationSelectSQL() string {
	return `SELECT id, attempt_record_id, attempt_id, schema_version, stage, idempotency_key, realized_config, framework_arguments, evidence, adjustment_policy, adjustment_reason_codes, simulated, realized_effective_hash, fidelity_verdict, created_at FROM execution_realization_observations`
}

func scanJobExecutionSpec(scanner rowScanner) (execution.JobExecutionSpec, error) {
	var out execution.JobExecutionSpec
	var raw []byte
	if err := scanner.Scan(&out.JobID, &out.ProjectID, &out.SchemaVersion, &out.CapabilityVersion, &out.Task, &out.Runner, &out.RequestedConfigHash, &out.AcceptedSpecHash, &raw, &out.CreatedAt); err != nil {
		return out, normalizeSQLError(err)
	}
	if err := json.Unmarshal(raw, &out.AcceptedSpec); err != nil {
		return out, err
	}
	return out, nil
}
func scanAttemptExecutionRecord(scanner rowScanner) (execution.AttemptExecutionRecord, error) {
	var out execution.AttemptExecutionRecord
	var verdict sql.NullString
	var realizedHash sql.NullString
	var reasons, raw []byte
	if err := scanner.Scan(&out.ID, &out.JobID, &out.ProjectID, &out.AttemptID, &out.AttemptNumber, &out.LifecycleStatus, &verdict, &realizedHash, &reasons, &raw, &out.CreatedAt, &out.UpdatedAt); err != nil {
		return out, normalizeSQLError(err)
	}
	if verdict.Valid {
		out.FidelityVerdict = &verdict.String
	}
	if realizedHash.Valid {
		out.RealizedEffectiveHash = realizedHash.String
	}
	if err := json.Unmarshal(reasons, &out.AdjustmentReasonCodes); err != nil {
		return out, err
	}
	if err := json.Unmarshal(raw, &out.LatestRealizedConfig); err != nil {
		return out, err
	}
	return out, nil
}
func scanRealizationObservation(scanner rowScanner) (execution.RealizationObservation, error) {
	var out execution.RealizationObservation
	var realized, framework, evidence, reasons []byte
	if err := scanner.Scan(&out.ID, &out.AttemptRecordID, &out.AttemptID, &out.SchemaVersion, &out.Stage, &out.IdempotencyKey, &realized, &framework, &evidence, &out.AdjustmentPolicy, &reasons, &out.Simulated, &out.RealizedEffectiveHash, &out.FidelityVerdict, &out.CreatedAt); err != nil {
		return out, normalizeSQLError(err)
	}
	if err := json.Unmarshal(realized, &out.RealizedConfig); err != nil {
		return out, err
	}
	if err := json.Unmarshal(framework, &out.FrameworkArguments); err != nil {
		return out, err
	}
	if err := json.Unmarshal(evidence, &out.Evidence); err != nil {
		return out, err
	}
	if err := json.Unmarshal(reasons, &out.AdjustmentReasonCodes); err != nil {
		return out, err
	}
	return out, nil
}
