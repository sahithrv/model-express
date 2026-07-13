package store

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

func TestMemoryProjectLiveStateTracksCurrentAttemptsAndAggregates(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	worker, err := s.RegisterWorker(projectID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
	if err != nil {
		t.Fatal(err)
	}

	queued := mustProjectLiveState(t, s, projectID)
	if queued.Jobs.Total != 1 || queued.Jobs.Queued != 1 || queued.Jobs.Retrying != 0 {
		t.Fatalf("queued counts = %#v", queued.Jobs)
	}
	if len(queued.ActiveProgress) != 1 || queued.ActiveProgress[0].Progress.Stage != jobs.ProgressStageQueued {
		t.Fatalf("queued progress = %#v", queued.ActiveProgress)
	}
	if !queued.ActiveProgress[0].ElapsedStartedAt.Equal(job.CreatedAt) {
		t.Fatalf("queued elapsed start = %s, want creation %s", queued.ActiveProgress[0].ElapsedStartedAt, job.CreatedAt)
	}

	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	active := mustProjectLiveState(t, s, projectID)
	if active.Jobs.Assigned != 1 || active.Jobs.Queued != 0 || len(active.ActiveProgress) != 1 {
		t.Fatalf("active snapshot = %#v", active)
	}
	if active.ActiveProgress[0].Progress.Attempt != assigned.Attempt || active.ActiveProgress[0].Progress.Stage != jobs.ProgressStageWorkerStarting {
		t.Fatalf("active attempt progress = %#v", active.ActiveProgress[0])
	}
	if assigned.StartedAt == nil || !active.ActiveProgress[0].ElapsedStartedAt.Equal(*assigned.StartedAt) {
		t.Fatalf("assigned elapsed start = %s, want started_at %v", active.ActiveProgress[0].ElapsedStartedAt, assigned.StartedAt)
	}

	retried, requeued, err := s.RetryJob(job.ID, "retry", RetryJobOptions{})
	if err != nil || !requeued || retried.Status != jobs.StatusQueued {
		t.Fatalf("retry = %#v requeued=%t err=%v", retried, requeued, err)
	}
	retrying := mustProjectLiveState(t, s, projectID)
	if retrying.Jobs.Retrying != 1 || retrying.Jobs.Queued != 0 {
		t.Fatalf("retrying counts = %#v", retrying.Jobs)
	}
	if retrying.ActiveProgress[0].Progress.Attempt != 2 || retrying.ActiveProgress[0].Progress.Stage != jobs.ProgressStageQueued {
		t.Fatalf("retry progress = %#v", retrying.ActiveProgress[0])
	}

	second, err := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
	if err != nil {
		t.Fatal(err)
	}
	assignedSecond, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.CompleteJob(assignedSecond.ID, "run"); err != nil {
		t.Fatal(err)
	}
	mixed := mustProjectLiveState(t, s, projectID)
	if mixed.Jobs.Total != 2 || mixed.Jobs.Succeeded != 1 || mixed.Jobs.Queued+mixed.Jobs.Retrying != 1 {
		t.Fatalf("mixed counts = %#v (second=%s)", mixed.Jobs, second.ID)
	}
	if mixed.ActiveProgressTotal != 1 || len(mixed.ActiveProgress) != 1 {
		t.Fatalf("mixed active progress = total %d rows %#v", mixed.ActiveProgressTotal, mixed.ActiveProgress)
	}
}

func TestMemoryProjectLiveStateCancellationAndBlockerCounts(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	worker, _ := s.RegisterWorker(projectID, "worker", "gpu")
	job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
	_, _ = s.PollJob(worker.ID, JobPollFilter{})
	if _, err := s.CancelJob(job.ID, "cancelled", nil); err != nil {
		t.Fatal(err)
	}

	requirement, _, err := s.UpsertWorkerRequirement(
		projectID,
		"",
		"modal",
		"T4",
		1,
		"test",
		execution.WorkerRequirementPolicy{},
	)
	if err != nil {
		t.Fatal(err)
	}
	failed := execution.WorkerRequirementFailed
	if _, err := s.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &failed}); err != nil {
		t.Fatal(err)
	}

	snapshot := mustProjectLiveState(t, s, projectID)
	if snapshot.Jobs.Cancelled != 1 || snapshot.Jobs.Failed != 0 {
		t.Fatalf("cancelled job counts = %#v", snapshot.Jobs)
	}
	if snapshot.Requirements.Failed != 1 {
		t.Fatalf("requirement counts = %#v", snapshot.Requirements)
	}
}

func TestMemoryProjectLiveStateSnapshotCursorCannotMissRacingTransition(t *testing.T) {
	assertProjectLiveStateRace(t, NewMemoryStore())
}

func TestPostgresProjectLiveStateReadPlanIsRepeatableAndFixed(t *testing.T) {
	source, err := os.ReadFile("postgres_live_state.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	if !strings.Contains(text, "Isolation: sql.LevelRepeatableRead") || !strings.Contains(text, "ReadOnly: true") {
		t.Fatal("PostgreSQL live state must use a read-only repeatable-read transaction")
	}
	if strings.Index(text, "SELECT last_sequence") > strings.Index(text, "scanProjectLiveStateJobCounts") {
		t.Fatal("snapshot cursor must be read before snapshot rows")
	}
	queryCount := strings.Count(text, ".QueryRowContext(") + strings.Count(text, ".QueryContext(")
	if queryCount != 6 {
		t.Fatalf("PostgreSQL live-state query count = %d, want fixed budget 6", queryCount)
	}
	if !strings.Contains(text, "LIMIT 8") || !strings.Contains(text, "executionEventV2SelectColumns()") {
		t.Fatal("PostgreSQL live state must cap progress and reuse the safe event projection")
	}
	if !strings.Contains(text, "COALESCE(job.started_at, job.created_at)") {
		t.Fatal("PostgreSQL live state must derive elapsed start from started_at with a created_at fallback")
	}
}

func TestScanProjectLiveStateProgressKeepsElapsedWorkerAndLeaseTimesAligned(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Microsecond)
	elapsedStartedAt := now.Add(-10 * time.Minute)
	workerHeartbeat := now.Add(-20 * time.Second)
	leaseHeartbeat := now.Add(-10 * time.Second)
	row := fixedLiveStateProgressRow{values: []any{
		3,
		"project_1", "job_1", 2,
		jobs.ProgressTaxonomyVersion, jobs.ProgressStageTraining, "epoch.complete",
		jobs.ProgressStatusRunning, sql.NullInt64{Int64: 2, Valid: true}, sql.NullInt64{Int64: 5, Valid: true}, "epoch",
		"Training is running.", int64(9), now, now, []byte(`{"provider":"local"}`),
		jobs.StatusRunning, elapsedStartedAt,
		sql.NullTime{Time: workerHeartbeat, Valid: true}, sql.NullTime{Time: leaseHeartbeat, Valid: true},
	}}

	item, total, err := scanProjectLiveStateProgress(row)
	if err != nil {
		t.Fatal(err)
	}
	if total != 3 || !item.ElapsedStartedAt.Equal(elapsedStartedAt) {
		t.Fatalf("total/elapsed = %d/%s, want 3/%s", total, item.ElapsedStartedAt, elapsedStartedAt)
	}
	if item.WorkerHeartbeatAt == nil || !item.WorkerHeartbeatAt.Equal(workerHeartbeat) {
		t.Fatalf("worker heartbeat misaligned: %v", item.WorkerHeartbeatAt)
	}
	if item.LeaseHeartbeatAt == nil || !item.LeaseHeartbeatAt.Equal(leaseHeartbeat) {
		t.Fatalf("lease heartbeat misaligned: %v", item.LeaseHeartbeatAt)
	}
}

type fixedLiveStateProgressRow struct {
	values []any
}

func (r fixedLiveStateProgressRow) Scan(dest ...any) error {
	if len(dest) != len(r.values) {
		return fmt.Errorf("scan destination count = %d, want %d", len(dest), len(r.values))
	}
	for index, value := range r.values {
		target := reflect.ValueOf(dest[index])
		if target.Kind() != reflect.Pointer || target.IsNil() {
			return fmt.Errorf("scan destination %d is not a pointer", index)
		}
		source := reflect.ValueOf(value)
		if !source.Type().AssignableTo(target.Elem().Type()) {
			return fmt.Errorf("scan value %d type %s is not assignable to %s", index, source.Type(), target.Elem().Type())
		}
		target.Elem().Set(source)
	}
	return nil
}

func TestPostgresProjectLiveStateParityIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_DATABASE_URL is not set")
	}
	postgres, err := NewPostgresStore(context.Background(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer postgres.Close()

	memory := NewMemoryStore()
	want := buildLiveStateParityFixture(t, memory, "memory parity")
	got := buildLiveStateParityFixture(t, postgres, "postgres parity")
	if !reflect.DeepEqual(got.Jobs, want.Jobs) || !reflect.DeepEqual(got.Workers, want.Workers) || !reflect.DeepEqual(got.Requirements, want.Requirements) {
		t.Fatalf("memory/postgres aggregates differ:\nmemory=%#v/%#v/%#v\npostgres=%#v/%#v/%#v", want.Jobs, want.Workers, want.Requirements, got.Jobs, got.Workers, got.Requirements)
	}
	if len(got.ActiveProgress) != 1 || len(want.ActiveProgress) != 1 {
		t.Fatalf("memory/postgres progress rows = %d/%d", len(want.ActiveProgress), len(got.ActiveProgress))
	}
	wantProgress := want.ActiveProgress[0].Progress
	gotProgress := got.ActiveProgress[0].Progress
	if gotProgress.Attempt != wantProgress.Attempt || gotProgress.TaxonomyVersion != wantProgress.TaxonomyVersion || gotProgress.Stage != wantProgress.Stage || gotProgress.DetailCode != wantProgress.DetailCode || gotProgress.Revision != wantProgress.Revision || !reflect.DeepEqual(gotProgress.Metadata, wantProgress.Metadata) {
		t.Fatalf("memory/postgres progress differs:\nmemory=%#v\npostgres=%#v", wantProgress, gotProgress)
	}
	if got.ActiveProgress[0].ElapsedStartedAt.IsZero() || want.ActiveProgress[0].ElapsedStartedAt.IsZero() {
		t.Fatalf("memory/postgres elapsed starts missing: memory=%s postgres=%s", want.ActiveProgress[0].ElapsedStartedAt, got.ActiveProgress[0].ElapsedStartedAt)
	}
	if got.LatestEvent == nil || want.LatestEvent == nil || got.LatestEvent.EventType != want.LatestEvent.EventType {
		t.Fatalf("memory/postgres latest event differs: memory=%#v postgres=%#v", want.LatestEvent, got.LatestEvent)
	}
}

func TestPostgresProjectLiveStateSnapshotCursorRaceIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_DATABASE_URL is not set")
	}
	postgres, err := NewPostgresStore(context.Background(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer postgres.Close()
	assertProjectLiveStateRace(t, postgres)
}

func TestMemoryProjectLiveStateProgressIsBounded(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	for index := 0; index < ProjectLiveStateProgressLimit+20; index++ {
		if _, err := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID}); err != nil {
			t.Fatal(err)
		}
	}
	snapshot := mustProjectLiveState(t, s, projectID)
	if snapshot.ActiveProgressTotal != ProjectLiveStateProgressLimit+20 || len(snapshot.ActiveProgress) != ProjectLiveStateProgressLimit {
		t.Fatalf("bounded progress = total %d rows %d", snapshot.ActiveProgressTotal, len(snapshot.ActiveProgress))
	}
}

func mustProjectLiveState(t *testing.T, s Store, projectID string) ProjectLiveStateSnapshot {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	snapshot, err := s.GetProjectLiveState(ctx, projectID)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func buildLiveStateParityFixture(t *testing.T, s Store, name string) ProjectLiveStateSnapshot {
	t.Helper()
	project, err := s.CreateProject(name, "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := s.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := s.RegisterWorker(project.ID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportJobProgress(job.ID, configString(assigned.Config, "active_attempt_id"), jobs.JobProgressUpsert{
		TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage:           jobs.ProgressStageTraining,
		Status:          jobs.ProgressStatusRunning,
		Revision:        3,
		DetailCode:      "simulator_training",
		Metadata: map[string]any{
			"provider":       "local",
			"execution_mode": "local_simulator",
		},
	}); err != nil {
		t.Fatal(err)
	}
	return mustProjectLiveState(t, s, project.ID)
}

func assertProjectLiveStateRace(t *testing.T, s Store) {
	t.Helper()
	project, err := s.CreateProject("live-state race", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := s.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := s.RegisterWorker(project.ID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	attemptID := configString(assigned.Config, "active_attempt_id")

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		_, _ = s.ReportJobProgress(job.ID, attemptID, jobs.JobProgressUpsert{
			TaxonomyVersion: jobs.ProgressTaxonomyVersion,
			Stage:           jobs.ProgressStageTraining,
			Status:          jobs.ProgressStatusRunning,
			Revision:        3,
		})
	}()
	close(start)
	snapshot := mustProjectLiveState(t, s, project.ID)
	wg.Wait()

	represented := len(snapshot.ActiveProgress) == 1 && snapshot.ActiveProgress[0].Progress.Stage == jobs.ProgressStageTraining
	events, err := s.ListProjectExecutionEventsAfter(context.Background(), project.ID, snapshot.SnapshotCursor, 10)
	if err != nil {
		t.Fatal(err)
	}
	streamed := false
	for _, event := range events {
		if event.EventType == execution.EventJobProgressBoundary && event.Payload["stage"] == jobs.ProgressStageTraining {
			streamed = true
		}
	}
	if !represented && !streamed {
		t.Fatalf("racing transition was neither in snapshot nor after cursor: snapshot=%#v events=%#v", snapshot, events)
	}
}
