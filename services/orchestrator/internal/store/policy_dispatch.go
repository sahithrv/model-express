package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/workers"
)

const defaultPolicyDispatchCandidateLimit = 64

func (s *MemoryStore) ListQueuedJobsForWorker(workerID string, filter JobPollFilter, limit int) ([]jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	worker, ok := s.workers[workerID]
	if !ok {
		return nil, ErrNotFound
	}
	if worker.ProjectID == "" {
		return []jobs.ExperimentJob{}, nil
	}
	if limit <= 0 {
		limit = defaultPolicyDispatchCandidateLimit
	}
	out := []jobs.ExperimentJob{}
	for _, job := range s.jobs {
		if job.Status != jobs.StatusQueued || job.ProjectID != worker.ProjectID || job.PolicyEligibilityStatus == jobs.PolicyEligibilityBlocked || !filter.Matches(job) {
			continue
		}
		out = append(out, jobs.WithExecutionSpecStatus(job))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID < out[j].ID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func (s *MemoryStore) ApplyQueuedJobPolicyEvaluation(jobID string, evaluation policies.Evaluation) (jobs.ExperimentJob, policies.Evaluation, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, ErrNotFound
	}
	if job.Status != jobs.StatusQueued {
		return job, policies.Evaluation{}, false, nil
	}
	if !s.policySourcesCurrentLocked(evaluation) {
		return job, policies.Evaluation{}, false, ErrPolicyChanged
	}
	created, err := s.createExperimentPolicyEvaluationLocked(evaluation)
	if err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	job.PolicyEligibilityStatus = policyEligibilityForDecision(created.Decision)
	s.jobs[job.ID] = job
	return jobs.WithExecutionSpecStatus(job), created, true, nil
}

func (s *MemoryStore) ClaimJobIfQueuedAndPolicyCurrent(workerID string, jobID string, filter JobPollFilter, evaluation policies.Evaluation) (*jobs.ExperimentJob, policies.Evaluation, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	worker, ok := s.workers[workerID]
	if !ok {
		return nil, policies.Evaluation{}, false, ErrNotFound
	}
	job, ok := s.jobs[jobID]
	if !ok {
		return nil, policies.Evaluation{}, false, ErrNotFound
	}
	if worker.CurrentJobID != "" || worker.ProjectID != job.ProjectID || job.Status != jobs.StatusQueued || job.PolicyEligibilityStatus == jobs.PolicyEligibilityBlocked || !filter.Matches(job) {
		return nil, policies.Evaluation{}, false, nil
	}
	if evaluation.Decision != policies.DecisionAllowed {
		return nil, policies.Evaluation{}, false, fmt.Errorf("%w: dispatch evaluation must be allowed before claim", ErrInvalidRequest)
	}
	if !s.policySourcesCurrentLocked(evaluation) {
		return nil, policies.Evaluation{}, false, ErrPolicyChanged
	}
	created, err := s.createExperimentPolicyEvaluationLocked(evaluation)
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}

	now := time.Now().UTC()
	job.WorkerID = workerID
	job.Status = jobs.StatusAssigned
	job.Error = ""
	job.Attempt++
	job.Config = jobConfigWithActiveAttempt(job.Config, job.ID, job.Attempt)
	if job.MaxAttempts < 1 {
		job.MaxAttempts = defaultJobMaxAttempts
	}
	job.StartedAt = &now
	job.LeaseOwnerWorkerID = workerID
	job.LeaseLastHeartbeatAt = &now
	leaseExpiresAt := now.Add(defaultJobLeaseDuration)
	job.LeaseExpiresAt = &leaseExpiresAt
	job.PolicyEligibilityStatus = jobs.PolicyEligibilityAllowed
	if _, ok := s.jobExecutionSpecs[job.ID]; ok {
		record, createErr := s.createAttemptExecutionRecordLocked(job.ID, jobAttemptID(job.ID, job.Attempt), job.Attempt)
		if createErr != nil {
			return nil, policies.Evaluation{}, false, createErr
		}
		record.DispatchPolicyEvaluationID = created.ID
		record.EffectivePolicyHash = created.EffectivePolicyHash
		record.UpdatedAt = now
		s.attemptExecutions[record.ID] = record
	}
	transition, progress := newJobLifecycleTransition(job, execution.TransitionJobAssigned, job.Attempt, jobs.ProgressStageWorkerStarting, jobs.ProgressStatusRunning, progressRevisionAssigned, "worker_assigned", "worker_assignment")
	if _, _, _, err := s.commitJobLifecycleLocked(job, progress, transition, now); err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	worker.Status = workers.StatusRunning
	worker.CurrentJobID = job.ID
	worker.LastHeartbeat = now
	s.workers[workerID] = worker
	claimed := jobs.WithExecutionSpecStatus(job)
	return &claimed, created, true, nil
}

func (s *MemoryStore) policySourcesCurrentLocked(evaluation policies.Evaluation) bool {
	current := map[string]policies.Binding{}
	wanted := policyEvaluationSubjects(evaluation)
	for _, binding := range s.policyBindings {
		if binding.Active && wanted[binding.Scope] == binding.SubjectID {
			current[policySourceKey(binding.Scope, binding.SubjectID)] = binding
		}
	}
	if len(current) != len(evaluation.PolicySources) {
		return false
	}
	for _, source := range evaluation.PolicySources {
		binding, ok := current[policySourceKey(source.Scope, source.SubjectID)]
		if !ok || binding.ID != source.BindingID || binding.Revision != source.BindingRevision || binding.PolicyVersionID != source.PolicyVersionID {
			return false
		}
	}
	return true
}

func (s *PostgresStore) ListQueuedJobsForWorker(workerID string, filter JobPollFilter, limit int) ([]jobs.ExperimentJob, error) {
	worker, err := s.GetWorker(workerID)
	if err != nil {
		return nil, err
	}
	if worker.ProjectID == "" {
		return []jobs.ExperimentJob{}, nil
	}
	if limit <= 0 {
		limit = defaultPolicyDispatchCandidateLimit
	}
	query, args := queuedPolicyCandidateQuery(worker.ProjectID, filter, limit)
	rows, err := s.db.QueryContext(context.Background(), query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []jobs.ExperimentJob{}
	for rows.Next() {
		job, scanErr := scanJob(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		out = append(out, job)
	}
	return out, rows.Err()
}

func queuedPolicyCandidateQuery(projectID string, filter JobPollFilter, limit int) (string, []any) {
	args := []any{jobs.StatusQueued, projectID, jobs.PolicyEligibilityBlocked}
	clauses := []string{"status = $1", "project_id = $2", "policy_eligibility_status <> $3"}
	if templates := normalizedPollValues(filter.Templates); len(templates) > 0 {
		placeholders := []string{}
		for _, template := range templates {
			args = append(args, template)
			placeholders = append(placeholders, fmt.Sprintf("$%d", len(args)))
		}
		clauses = append(clauses, "lower(template) IN ("+strings.Join(placeholders, ",")+")")
	}
	if provider := strings.ToLower(strings.TrimSpace(filter.Provider)); provider != "" {
		args = append(args, provider)
		providerPlaceholder := fmt.Sprintf("$%d", len(args))
		providerClause := "lower(coalesce(config->>'provider', '')) = " + providerPlaceholder
		if fallbacks := normalizedPollValues(filter.IncludeUnspecifiedProviderTemplates); len(fallbacks) > 0 {
			placeholders := []string{}
			for _, template := range fallbacks {
				args = append(args, template)
				placeholders = append(placeholders, fmt.Sprintf("$%d", len(args)))
			}
			providerClause = "(" + providerClause + " OR (coalesce(config->>'provider', '') = '' AND lower(template) IN (" + strings.Join(placeholders, ",") + ")))"
		}
		clauses = append(clauses, providerClause)
	}
	args = append(args, limit)
	return `SELECT ` + jobSelectColumns() + ` FROM experiment_jobs WHERE ` + strings.Join(clauses, " AND ") + ` ORDER BY created_at, id LIMIT $` + fmt.Sprintf("%d", len(args)), args
}

func (s *PostgresStore) ApplyQueuedJobPolicyEvaluation(jobID string, evaluation policies.Evaluation) (jobs.ExperimentJob, policies.Evaluation, bool, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	defer tx.Rollback()
	if err := lockAndValidatePolicySourcesTx(ctx, tx, evaluation); err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	if job.Status != jobs.StatusQueued {
		return job, policies.Evaluation{}, false, tx.Commit()
	}
	created, err := insertExperimentPolicyEvaluation(ctx, tx, evaluation)
	if err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	job, err = scanJob(tx.QueryRowContext(ctx, `UPDATE experiment_jobs SET policy_eligibility_status=$1 WHERE id=$2 AND status=$3 RETURNING `+jobSelectColumns(), policyEligibilityForDecision(created.Decision), job.ID, jobs.StatusQueued))
	if err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, policies.Evaluation{}, false, err
	}
	return job, created, true, nil
}

func (s *PostgresStore) ClaimJobIfQueuedAndPolicyCurrent(workerID string, jobID string, filter JobPollFilter, evaluation policies.Evaluation) (*jobs.ExperimentJob, policies.Evaluation, bool, error) {
	if evaluation.Decision != policies.DecisionAllowed {
		return nil, policies.Evaluation{}, false, fmt.Errorf("%w: dispatch evaluation must be allowed before claim", ErrInvalidRequest)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	defer tx.Rollback()
	if err := lockAndValidatePolicySourcesTx(ctx, tx, evaluation); err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	worker, err := scanWorker(tx.QueryRowContext(ctx, `SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id FROM workers WHERE id=$1 FOR UPDATE`, workerID))
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	if worker.CurrentJobID != "" || worker.ProjectID != job.ProjectID || job.Status != jobs.StatusQueued || job.PolicyEligibilityStatus == jobs.PolicyEligibilityBlocked || !filter.Matches(job) {
		return nil, policies.Evaluation{}, false, tx.Commit()
	}
	created, err := insertExperimentPolicyEvaluation(ctx, tx, evaluation)
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	now := time.Now().UTC()
	assignedConfig := jobConfigWithActiveAttempt(job.Config, job.ID, job.Attempt+1)
	assignedConfigJSON, err := json.Marshal(assignedConfig)
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	assigned, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs SET worker_id=$1, status=$2, error='', started_at=$3, attempt=attempt+1,
			config=$4, lease_owner_worker_id=$1, lease_last_heartbeat_at=$3, lease_expires_at=$5,
			policy_eligibility_status=$6
		WHERE id=$7 AND status=$8 RETURNING `+jobSelectColumns(), workerID, jobs.StatusAssigned, now, assignedConfigJSON, now.Add(defaultJobLeaseDuration), jobs.PolicyEligibilityAllowed, job.ID, jobs.StatusQueued))
	if err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	record, recordErr := createAttemptExecutionRecordTx(ctx, tx, assigned.ID, jobAttemptID(assigned.ID, assigned.Attempt), assigned.Attempt)
	if recordErr == nil {
		if _, err := tx.ExecContext(ctx, `UPDATE attempt_execution_records SET dispatch_policy_evaluation_id=$1, effective_policy_hash=$2, updated_at=$3 WHERE id=$4`, created.ID, created.EffectivePolicyHash, now, record.ID); err != nil {
			return nil, policies.Evaluation{}, false, err
		}
	} else if !errors.Is(normalizeSQLError(recordErr), ErrNotFound) {
		return nil, policies.Evaluation{}, false, recordErr
	}
	transition, progress := newJobLifecycleTransition(assigned, execution.TransitionJobAssigned, assigned.Attempt, jobs.ProgressStageWorkerStarting, jobs.ProgressStatusRunning, progressRevisionAssigned, "worker_assigned", "worker_assignment")
	if _, _, _, err := commitJobLifecycleTx(ctx, tx, assigned, progress, transition, now); err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE workers SET status=$1, current_job_id=$2, last_heartbeat=$3 WHERE id=$4`, workers.StatusRunning, assigned.ID, now, workerID); err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return nil, policies.Evaluation{}, false, err
	}
	return &assigned, created, true, nil
}

func lockAndValidatePolicySourcesTx(ctx context.Context, tx *sql.Tx, evaluation policies.Evaluation) error {
	subjects := policyEvaluationSubjects(evaluation)
	keys := []string{}
	for scope, subjectID := range subjects {
		if subjectID != "" {
			keys = append(keys, policySourceKey(scope, subjectID))
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, key); err != nil {
			return err
		}
	}
	rows, err := tx.QueryContext(ctx, `
		SELECT `+policyBindingSelectColumns()+` FROM experiment_policy_bindings
		WHERE active AND ((scope='account' AND account_id=NULLIF($1,'')) OR (scope='project' AND project_id=NULLIF($2,'')) OR (scope='dataset' AND dataset_id=NULLIF($3,'')) OR (scope='run' AND experiment_job_id=NULLIF($4,'')))
	`, evaluation.AccountID, evaluation.ProjectID, evaluation.DatasetID, evaluation.JobID)
	if err != nil {
		return err
	}
	defer rows.Close()
	current := map[string]policies.Binding{}
	for rows.Next() {
		binding, scanErr := scanPolicyBinding(rows)
		if scanErr != nil {
			return scanErr
		}
		current[policySourceKey(binding.Scope, binding.SubjectID)] = binding
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if len(current) != len(evaluation.PolicySources) {
		return ErrPolicyChanged
	}
	for _, source := range evaluation.PolicySources {
		binding, ok := current[policySourceKey(source.Scope, source.SubjectID)]
		if !ok || binding.ID != source.BindingID || binding.Revision != source.BindingRevision || binding.PolicyVersionID != source.PolicyVersionID {
			return ErrPolicyChanged
		}
	}
	return nil
}

func policyEvaluationSubjects(evaluation policies.Evaluation) map[policies.Scope]string {
	return map[policies.Scope]string{
		policies.ScopeAccount: evaluation.AccountID,
		policies.ScopeProject: evaluation.ProjectID,
		policies.ScopeDataset: evaluation.DatasetID,
		policies.ScopeRun:     evaluation.JobID,
	}
}

func policySourceKey(scope policies.Scope, subjectID string) string {
	return string(scope) + ":" + strings.TrimSpace(subjectID)
}

func policyEligibilityForDecision(decision string) string {
	if decision == policies.DecisionAllowed {
		return jobs.PolicyEligibilityAllowed
	}
	return jobs.PolicyEligibilityBlocked
}

func (s *MemoryStore) markQueuedJobsPolicyPendingLocked(scope policies.Scope, subjectID string) {
	for id, job := range s.jobs {
		if job.Status != jobs.StatusQueued || !s.jobMatchesPolicySubjectLocked(job, scope, subjectID) {
			continue
		}
		job.PolicyEligibilityStatus = jobs.PolicyEligibilityPending
		s.jobs[id] = job
	}
}

func (s *MemoryStore) jobMatchesPolicySubjectLocked(job jobs.ExperimentJob, scope policies.Scope, subjectID string) bool {
	switch scope {
	case policies.ScopeAccount:
		project, ok := s.projects[job.ProjectID]
		return ok && project.AccountID == subjectID
	case policies.ScopeProject:
		return job.ProjectID == subjectID
	case policies.ScopeDataset:
		return job.DatasetID == subjectID || strings.TrimSpace(configString(job.Config, "dataset_id")) == subjectID
	case policies.ScopeRun:
		return job.ID == subjectID
	default:
		return false
	}
}

func markQueuedJobsPolicyPendingTx(ctx context.Context, tx *sql.Tx, scope policies.Scope, subjectID string) error {
	var query string
	switch scope {
	case policies.ScopeAccount:
		query = `UPDATE experiment_jobs j SET policy_eligibility_status=$1 FROM projects p WHERE j.project_id=p.id AND p.account_id=$2 AND j.status=$3`
	case policies.ScopeProject:
		query = `UPDATE experiment_jobs SET policy_eligibility_status=$1 WHERE project_id=$2 AND status=$3`
	case policies.ScopeDataset:
		query = `UPDATE experiment_jobs SET policy_eligibility_status=$1 WHERE (dataset_id=$2 OR config->>'dataset_id'=$2) AND status=$3`
	case policies.ScopeRun:
		query = `UPDATE experiment_jobs SET policy_eligibility_status=$1 WHERE id=$2 AND status=$3`
	default:
		return fmt.Errorf("%w: unsupported policy binding scope %q", ErrInvalidRequest, scope)
	}
	_, err := tx.ExecContext(ctx, query, jobs.PolicyEligibilityPending, subjectID, jobs.StatusQueued)
	return err
}
