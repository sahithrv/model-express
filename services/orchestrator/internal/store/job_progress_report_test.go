package store

import (
	"errors"
	"reflect"
	"sort"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

func TestMemoryReportJobProgressIsRevisionedAndBoundsEvents(t *testing.T) {
	s, job, attemptID := newAssignedProgressFixture(t)

	first, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageRemoteScheduled, Status: jobs.ProgressStatusRunning,
		Revision: 3, DetailCode: "provider.submitted", Metadata: map[string]any{"provider": "modal"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !first.Updated || !first.EventCreated || first.Progress.Attempt != 1 || first.Progress.Stage != jobs.ProgressStageRemoteScheduled {
		t.Fatalf("first report=%#v", first)
	}
	if got := memoryProgressBoundaryEvents(s, job.ID); len(got) != 1 {
		t.Fatalf("first boundary events=%#v", got)
	}

	heartbeat, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageRemoteScheduled, Status: jobs.ProgressStatusRunning,
		Revision: 4, DetailCode: "provider.waiting", Message: "Remote worker is still queued.",
		Metadata: map[string]any{"provider": "modal", "execution_mode": "remote"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !heartbeat.Updated || heartbeat.EventCreated || heartbeat.Progress.Revision != 4 {
		t.Fatalf("same-stage heartbeat=%#v", heartbeat)
	}
	if heartbeat.Progress.HeartbeatAt.Before(first.Progress.HeartbeatAt) {
		t.Fatalf("server receipt time regressed: first=%s heartbeat=%s", first.Progress.HeartbeatAt, heartbeat.Progress.HeartbeatAt)
	}
	if got := memoryProgressBoundaryEvents(s, job.ID); len(got) != 1 {
		t.Fatalf("same-stage heartbeat appended an event: %#v", got)
	}

	duplicate, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning, Revision: 4,
	})
	if err != nil || duplicate.Updated || duplicate.EventCreated || duplicate.Progress.Stage != jobs.ProgressStageRemoteScheduled {
		t.Fatalf("duplicate revision=%#v err=%v", duplicate, err)
	}
	older, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning, Revision: 3,
	})
	if err != nil || older.Updated || older.Progress.Revision != 4 {
		t.Fatalf("older revision=%#v err=%v", older, err)
	}

	current, total := int64(1), int64(10)
	boundary, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning,
		Revision: 5, Current: &current, Total: &total, Unit: "epoch",
	})
	if err != nil || !boundary.Updated || !boundary.EventCreated {
		t.Fatalf("range boundary=%#v err=%v", boundary, err)
	}
	events := memoryProgressBoundaryEvents(s, job.ID)
	if len(events) != 2 || events[1].Payload["current"] != float64(1) || events[1].Payload["total"] != float64(10) {
		t.Fatalf("range boundary events=%#v", events)
	}
}

func TestMemoryReportJobProgressRejectsStaleAndCannotRegressTerminal(t *testing.T) {
	s, job, attemptID := newAssignedProgressFixture(t)
	if _, _, err := s.RetryJob(job.ID, "retry", RetryJobOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning, Revision: 3,
	}); !errors.Is(err, ErrStaleAttempt) {
		t.Fatalf("stale attempt err=%v", err)
	}

	// Exercise the defensive snapshot guard independently of job terminal
	// config: a terminal progress row can never produce a worker event or
	// regress even if a nonterminal job row remains active.
	s, job, attemptID = newAssignedProgressFixture(t)
	terminal, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageCompleted, Status: jobs.ProgressStatusCompleted, Revision: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	beforeEvents := len(memoryProgressBoundaryEvents(s, job.ID))
	result, err := s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning, Revision: 11,
	})
	if err != nil || result.Updated || result.EventCreated || !reflect.DeepEqual(result.Progress, terminal) {
		t.Fatalf("post-terminal report=%#v err=%v terminal=%#v", result, err, terminal)
	}
	if len(memoryProgressBoundaryEvents(s, job.ID)) != beforeEvents {
		t.Fatal("post-terminal worker report appended a boundary event")
	}
}

func TestMemoryReportJobProgressValidationRollbackLeavesSnapshotAndEventTogether(t *testing.T) {
	s, job, _ := newAssignedProgressFixture(t)
	before, err := s.GetJobProgress(job.ID, 1)
	if err != nil {
		t.Fatal(err)
	}
	beforeSequence := s.nextExecutionEventSequence
	beforeEventCount := len(s.executionEvents)

	const unsafeAttemptID = "unsafe/attempt"
	s.mu.Lock()
	stored := s.jobs[job.ID]
	stored.Config["active_attempt_id"] = unsafeAttemptID
	stored.Config["active_attempt_number"] = 1
	s.jobs[job.ID] = stored
	s.mu.Unlock()

	_, err = s.ReportJobProgress(job.ID, unsafeAttemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning, Revision: 3,
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected derived event validation failure, got %v", err)
	}
	after, getErr := s.GetJobProgress(job.ID, 1)
	if getErr != nil {
		t.Fatal(getErr)
	}
	if !reflect.DeepEqual(before, after) || beforeSequence != s.nextExecutionEventSequence || beforeEventCount != len(s.executionEvents) {
		t.Fatalf("atomic rollback failed: progress %#v -> %#v, sequence %d -> %d, events %d -> %d", before, after, beforeSequence, s.nextExecutionEventSequence, beforeEventCount, len(s.executionEvents))
	}
}

func newAssignedProgressFixture(t *testing.T) (*MemoryStore, jobs.ExperimentJob, string) {
	t.Helper()
	s, projectID, datasetID := newJobLifecycleFixture(t)
	worker, err := s.RegisterWorker(projectID, "progress-worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	_, err = s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "plan_id": "plan_1"})
	if err != nil {
		t.Fatal(err)
	}
	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	return s, *assigned, configString(assigned.Config, "active_attempt_id")
}

func memoryProgressBoundaryEvents(s *MemoryStore, jobID string) []execution.ExecutionEvent {
	events := []execution.ExecutionEvent{}
	for _, event := range s.executionEvents {
		if event.EventType == execution.EventJobProgressBoundary && event.Payload["job_id"] == jobID {
			events = append(events, event)
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].Sequence < events[j].Sequence })
	return events
}
