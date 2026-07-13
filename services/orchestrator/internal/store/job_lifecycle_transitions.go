package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

const (
	progressRevisionQueued   int64 = 1
	progressRevisionAssigned int64 = 2
	progressRevisionRunning  int64 = 3
	progressRevisionTerminal int64 = 4
)

func newJobLifecycleTransition(
	job jobs.ExperimentJob,
	transition execution.ExecutionTransition,
	attempt int,
	stage string,
	status string,
	revision int64,
	detailCode string,
	reasonCode string,
) (execution.ExecutionTransitionEventInput, jobs.JobProgressUpsert) {
	if attempt < 1 {
		attempt = 1
	}
	return execution.ExecutionTransitionEventInput{
			Transition: transition,
			ProjectID:  job.ProjectID,
			PlanID:     configString(job.Config, "plan_id"),
			JobID:      job.ID,
			AttemptID:  jobAttemptID(job.ID, attempt),
			Attempt:    attempt,
			ReasonCode: reasonCode,
		}, jobs.JobProgressUpsert{
			Attempt:         attempt,
			TaxonomyVersion: jobs.ProgressTaxonomyVersion,
			Stage:           stage,
			DetailCode:      detailCode,
			Status:          status,
			Revision:        revision,
			Message:         jobProgressMessage(stage),
			Metadata:        map[string]any{},
		}
}

func jobProgressMessage(stage string) string {
	switch stage {
	case jobs.ProgressStageQueued:
		return "Waiting for a worker."
	case jobs.ProgressStageWorkerStarting:
		return "Worker is starting the job."
	case jobs.ProgressStageTraining:
		return "Training is running."
	case jobs.ProgressStageCompleted:
		return "Job completed."
	case jobs.ProgressStageFailed:
		return "Job failed."
	case jobs.ProgressStageCancelled:
		return "Job cancelled."
	default:
		return "Job progress updated."
	}
}

func progressRevisionForEpoch(epoch int) int64 {
	if epoch < 1 {
		return progressRevisionRunning
	}
	revision := progressRevisionAssigned + int64(epoch)
	if revision < progressRevisionRunning || revision > jobs.JobProgressMaxRevision {
		return jobs.JobProgressMaxRevision
	}
	return revision
}

func pendingJobProgressAttempt(job jobs.ExperimentJob) int {
	attempt := job.Attempt + 1
	if attempt < 1 {
		return 1
	}
	return attempt
}

func activeJobProgressAttempt(job jobs.ExperimentJob) int {
	if attempt := jobConfigPositiveInt(job.Config, "active_attempt_number"); attempt > 0 {
		return attempt
	}
	if job.Status == jobs.StatusQueued {
		return pendingJobProgressAttempt(job)
	}
	if job.Attempt < 1 {
		return 1
	}
	return job.Attempt
}

func jobConfigPositiveInt(config map[string]any, key string) int {
	switch value := config[key].(type) {
	case int:
		return value
	case int32:
		return int(value)
	case int64:
		return int(value)
	case float64:
		if value > 0 && value == float64(int(value)) {
			return int(value)
		}
	}
	return 0
}

// commitJobLifecycleLocked is the in-memory composition point for the three
// authoritative records. It validates both derived records before changing
// any map, so validation failure leaves job, progress, and cursor untouched.
func (s *MemoryStore) commitJobLifecycleLocked(
	job jobs.ExperimentJob,
	progressUpdate jobs.JobProgressUpsert,
	transitionInput execution.ExecutionTransitionEventInput,
	now time.Time,
) (jobs.JobProgress, execution.ExecutionEvent, bool, error) {
	if _, ok := s.projects[job.ProjectID]; !ok {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, ErrNotFound
	}
	if _, err := jobs.NormalizeJobProgressUpsert(progressUpdate); err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, fmt.Errorf("%w: invalid job progress: %v", ErrInvalidRequest, err)
	}
	create, err := execution.NewExecutionTransitionEvent(transitionInput)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, fmt.Errorf("%w: invalid execution transition: %v", ErrInvalidRequest, err)
	}

	// From this point onward every helper has already been validated and cannot
	// fail for caller-controlled content.
	s.jobs[job.ID] = job
	progress, err := s.upsertJobProgressLocked(job, progressUpdate, now)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, err
	}
	event, created, err := s.appendExecutionTransitionLocked(create, now)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, err
	}
	return progress, event, created, nil
}

func commitJobLifecycleTx(
	ctx context.Context,
	tx *sql.Tx,
	job jobs.ExperimentJob,
	progressUpdate jobs.JobProgressUpsert,
	transitionInput execution.ExecutionTransitionEventInput,
	now time.Time,
) (jobs.JobProgress, execution.ExecutionEvent, bool, error) {
	create, err := execution.NewExecutionTransitionEvent(transitionInput)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, fmt.Errorf("%w: invalid execution transition: %v", ErrInvalidRequest, err)
	}
	progress, err := upsertJobProgressTx(ctx, tx, job, progressUpdate, now)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, err
	}
	event, created, err := appendExecutionTransitionTx(ctx, tx, create)
	if err != nil {
		return jobs.JobProgress{}, execution.ExecutionEvent{}, false, err
	}
	return progress, event, created, nil
}
