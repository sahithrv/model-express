package store

import (
	"context"
	"fmt"
	"os"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

func TestPostgresLifecycleAtomicIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_DATABASE_URL is not set")
	}
	store, err := NewPostgresStore(context.Background(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	project, err := store.CreateProject("postgres lifecycle", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := store.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := store.RegisterWorker(project.ID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	job, err := store.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	assertPostgresProgressStage(t, store, job.ID, 1, jobs.ProgressStageQueued)
	assigned, err := store.PollJob(worker.ID, JobPollFilter{})
	if err != nil || assigned.Attempt != 1 {
		t.Fatalf("assignment=%#v err=%v", assigned, err)
	}
	assertPostgresProgressStage(t, store, job.ID, 1, jobs.ProgressStageWorkerStarting)
	attemptID := configString(assigned.Config, "active_attempt_id")
	reported, err := store.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageRemoteScheduled, Status: jobs.ProgressStatusRunning, Revision: 3,
	})
	if err != nil || !reported.Updated || !reported.EventCreated {
		t.Fatalf("postgres progress boundary=%#v err=%v", reported, err)
	}
	heartbeat, err := store.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageRemoteScheduled, Status: jobs.ProgressStatusRunning,
		Revision: 4, Message: "Remote scheduling is active.",
	})
	if err != nil || !heartbeat.Updated || heartbeat.EventCreated {
		t.Fatalf("postgres progress heartbeat=%#v err=%v", heartbeat, err)
	}
	duplicate, err := store.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning, Revision: 4,
	})
	if err != nil || duplicate.Updated || duplicate.Progress.Stage != jobs.ProgressStageRemoteScheduled {
		t.Fatalf("postgres duplicate progress=%#v err=%v", duplicate, err)
	}
	if _, err := store.ReportMetric(job.ID, 3, map[string]float64{"loss": .5}); err != nil {
		t.Fatal(err)
	}
	training := assertPostgresProgressStage(t, store, job.ID, 1, jobs.ProgressStageTraining)
	if training.Current == nil || *training.Current != 3 {
		t.Fatalf("training progress=%#v", training)
	}
	if _, err := store.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning,
		Revision: 100,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CompleteJob(job.ID, "run_1"); err != nil {
		t.Fatal(err)
	}
	terminal := assertPostgresProgressStage(t, store, job.ID, 1, jobs.ProgressStageCompleted)
	if terminal.Revision <= 100 {
		t.Fatalf("terminal revision did not supersede worker: %#v", terminal)
	}
	events, err := store.ListProjectExecutionEvents(project.ID, 20)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		execution.EventJobQueued: false, execution.EventJobAssigned: false,
		execution.EventJobRunning: false, execution.EventJobCompleted: false,
		execution.EventJobProgressBoundary: false,
	}
	for _, event := range events {
		if _, ok := want[event.EventType]; ok {
			want[event.EventType] = true
		}
	}
	for eventType, found := range want {
		if !found {
			t.Errorf("missing lifecycle event %s: %#v", eventType, events)
		}
	}

	// Force the boundary event insert to fail after a second job is assigned.
	// PostgreSQL must roll back its snapshot update in the same transaction.
	rollbackProgressJob, err := store.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	rollbackAssigned, err := store.PollJob(worker.ID, JobPollFilter{})
	if err != nil || rollbackAssigned.ID != rollbackProgressJob.ID {
		t.Fatalf("assign rollback job=%#v err=%v", rollbackAssigned, err)
	}
	beforeRollbackProgress := assertPostgresProgressStage(t, store, rollbackAssigned.ID, rollbackAssigned.Attempt, jobs.ProgressStageWorkerStarting)
	const progressConstraint = "test_reject_atomic_progress_event"
	if _, err := store.db.Exec(`ALTER TABLE execution_events DROP CONSTRAINT IF EXISTS ` + progressConstraint); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.Exec(`ALTER TABLE execution_events ADD CONSTRAINT ` + progressConstraint + ` CHECK (event_type <> '` + execution.EventJobProgressBoundary + `')`); err != nil {
		t.Fatal(err)
	}
	_, reportErr := store.ReportJobProgress(rollbackAssigned.ID, configString(rollbackAssigned.Config, "active_attempt_id"), jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning, Revision: 3,
	})
	if _, err := store.db.Exec(`ALTER TABLE execution_events DROP CONSTRAINT IF EXISTS ` + progressConstraint); err != nil {
		t.Fatal(err)
	}
	if reportErr == nil {
		t.Fatal("expected forced progress boundary failure")
	}
	afterRollbackProgress := assertPostgresProgressStage(t, store, rollbackAssigned.ID, rollbackAssigned.Attempt, jobs.ProgressStageWorkerStarting)
	if afterRollbackProgress.Revision != beforeRollbackProgress.Revision || !afterRollbackProgress.UpdatedAt.Equal(beforeRollbackProgress.UpdatedAt) {
		t.Fatalf("progress/event rollback diverged: before=%#v after=%#v", beforeRollbackProgress, afterRollbackProgress)
	}

	rollbackProject, err := store.CreateProject("postgres rollback", "")
	if err != nil {
		t.Fatal(err)
	}
	rollbackDataset, err := store.CreateDataset(rollbackProject.ID, "dataset", "memory://rollback", "checksum-rollback", 1)
	if err != nil {
		t.Fatal(err)
	}
	constraint := "test_reject_atomic_lifecycle_event"
	if _, err := store.db.Exec(`ALTER TABLE execution_events DROP CONSTRAINT IF EXISTS ` + constraint); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.Exec(fmt.Sprintf(
		`ALTER TABLE execution_events ADD CONSTRAINT %s CHECK (project_id <> '%s')`,
		constraint,
		rollbackProject.ID,
	)); err != nil {
		t.Fatal(err)
	}
	defer store.db.Exec(`ALTER TABLE execution_events DROP CONSTRAINT IF EXISTS ` + constraint)
	if _, err := store.CreateJob(rollbackProject.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": rollbackDataset.ID}); err == nil {
		t.Fatal("expected forced event failure")
	}
	for table, column := range map[string]string{
		"experiment_jobs":  "project_id",
		"job_progress":     "project_id",
		"execution_events": "project_id",
	} {
		var count int
		if err := store.db.QueryRow(`SELECT count(*) FROM `+table+` WHERE `+column+` = $1`, rollbackProject.ID).Scan(&count); err != nil {
			t.Fatal(err)
		}
		if count != 0 {
			t.Errorf("forced rollback left %d row(s) in %s", count, table)
		}
	}
}

func assertPostgresProgressStage(t *testing.T, store *PostgresStore, jobID string, attempt int, stage string) jobs.JobProgress {
	t.Helper()
	progress, err := store.GetJobProgress(jobID, attempt)
	if err != nil {
		t.Fatal(err)
	}
	if progress.Stage != stage {
		t.Fatalf("progress stage=%q want=%q: %#v", progress.Stage, stage, progress)
	}
	return progress
}
