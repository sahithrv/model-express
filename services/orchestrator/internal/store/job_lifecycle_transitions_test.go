package store

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

func TestMemoryJobLifecycleWritesJobProgressAndEventAtomically(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	worker, err := s.RegisterWorker(projectID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	job, err := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": datasetID,
		"plan_id":    "plan_1",
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageQueued)
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued)

	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageWorkerStarting)
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued, execution.EventJobAssigned)

	if _, err := s.ReportMetric(job.ID, 1, map[string]float64{"loss": 1}); err != nil {
		t.Fatal(err)
	}
	running := assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageTraining)
	if running.Current == nil || *running.Current != 1 || running.Unit != "epoch" {
		t.Fatalf("first metric progress = %#v", running)
	}
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued, execution.EventJobAssigned, execution.EventJobRunning)

	if _, err := s.ReportMetric(job.ID, 2, map[string]float64{"loss": .5}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportMetric(job.ID, 10, map[string]float64{"loss": .1}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportMetric(job.ID, 3, map[string]float64{"loss": .4}); err != nil {
		t.Fatal(err)
	}
	monotonic := assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageTraining)
	if monotonic.Current == nil || *monotonic.Current != 10 {
		t.Fatalf("out-of-order epoch regressed progress: %#v", monotonic)
	}
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued, execution.EventJobAssigned, execution.EventJobRunning)
	if _, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageFinalizing, Status: jobs.ProgressStatusRunning,
		Revision: 100,
	}); err != nil {
		t.Fatal(err)
	}

	completed, err := s.CompleteJob(job.ID, "run_1")
	if err != nil || completed.Status != jobs.StatusSucceeded {
		t.Fatalf("complete job = %#v err=%v", completed, err)
	}
	terminal := assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageCompleted)
	if terminal.Revision <= 100 {
		t.Fatalf("terminal revision %d did not advance beyond worker revision 100", terminal.Revision)
	}
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued, execution.EventJobAssigned, execution.EventJobRunning, execution.EventJobCompleted)

	// Repeated and opposite terminal callbacks are idempotent, and later metric
	// traffic cannot regress the server-owned terminal snapshot.
	sequence := s.nextExecutionEventSequence
	if repeated, err := s.CompleteJob(job.ID, "run_2"); err != nil || repeated.Status != jobs.StatusSucceeded {
		t.Fatalf("repeat complete = %#v err=%v", repeated, err)
	}
	if opposite, err := s.FailJob(job.ID, "late failure"); err != nil || opposite.Status != jobs.StatusSucceeded {
		t.Fatalf("opposite terminal = %#v err=%v", opposite, err)
	}
	if _, _, err := s.RetryJob(job.ID, "late retry", RetryJobOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportMetric(job.ID, 99, map[string]float64{"late": 1}); err != nil {
		t.Fatal(err)
	}
	if s.nextExecutionEventSequence != sequence {
		t.Fatalf("terminal callbacks appended events: before=%d after=%d", sequence, s.nextExecutionEventSequence)
	}
	after := assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageCompleted)
	if after.Revision != terminal.Revision {
		t.Fatalf("terminal progress changed: before=%#v after=%#v", terminal, after)
	}
	storedJob, _ := s.GetJob(job.ID)
	storedWorker, _ := s.GetWorker(worker.ID)
	if storedJob.LeaseExpiresAt != nil || storedJob.LeaseOwnerWorkerID != "" || storedWorker.Status != "IDLE" {
		t.Fatalf("late metric recreated terminal lease/worker state: job=%#v worker=%#v", storedJob, storedWorker)
	}
	if assigned.Attempt != 1 {
		t.Fatalf("assignment attempt = %d", assigned.Attempt)
	}
}

func TestMemoryRetryAndLeaseRecoveryUseFreshAttemptSnapshots(t *testing.T) {
	t.Run("retry", func(t *testing.T) {
		s, projectID, datasetID := newJobLifecycleFixture(t)
		worker, _ := s.RegisterWorker(projectID, "worker", "gpu")
		job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
		_, _ = s.PollJob(worker.ID, JobPollFilter{})
		_, _ = s.ReportMetric(job.ID, 1, map[string]float64{"loss": 1})

		retried, requeued, err := s.RetryJob(job.ID, "s3://secret/path", RetryJobOptions{})
		if err != nil || !requeued || retried.Status != jobs.StatusQueued {
			t.Fatalf("retry = %#v requeued=%t err=%v", retried, requeued, err)
		}
		assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageTraining)
		queued := assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageQueued)
		if activeJobProgressAttempt(retried) != queued.Attempt {
			t.Fatalf("queued apparent attempt = %d, snapshot=%d", activeJobProgressAttempt(retried), queued.Attempt)
		}
		before := s.nextExecutionEventSequence
		if _, requeued, err := s.RetryJob(job.ID, "duplicate callback", RetryJobOptions{}); err != nil || !requeued {
			t.Fatalf("duplicate retry requeued=%t err=%v", requeued, err)
		}
		if s.nextExecutionEventSequence != before {
			t.Fatal("duplicate retry allocated another event cursor")
		}

		assigned, err := s.PollJob(worker.ID, JobPollFilter{})
		if err != nil || assigned.Attempt != 2 {
			t.Fatalf("second assignment=%#v err=%v", assigned, err)
		}
		assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageWorkerStarting)
		failed, requeued, err := s.RetryJob(job.ID, "secret final failure", RetryJobOptions{ForceFail: true})
		if err != nil || requeued || failed.Status != jobs.StatusFailed {
			t.Fatalf("exhausted retry=%#v requeued=%t err=%v", failed, requeued, err)
		}
		assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageFailed)
		assertJobEventsContainNoText(t, s, job.ID, "secret")
	})

	t.Run("lease recovery", func(t *testing.T) {
		s, projectID, datasetID := newJobLifecycleFixture(t)
		worker, _ := s.RegisterWorker(projectID, "worker", "gpu")
		job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
		assigned, _ := s.PollJob(worker.ID, JobPollFilter{})
		expireMemoryJobLease(t, s, assigned.ID, 2)

		recovered, err := s.RecoverExpiredJobLeases(time.Now().UTC())
		if err != nil || len(recovered) != 1 || recovered[0].Status != jobs.StatusQueued {
			t.Fatalf("lease requeue=%#v err=%v", recovered, err)
		}
		assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageQueued)
		before := s.nextExecutionEventSequence
		if repeated, err := s.RecoverExpiredJobLeases(time.Now().UTC()); err != nil || len(repeated) != 0 {
			t.Fatalf("repeat recovery=%#v err=%v", repeated, err)
		}
		if s.nextExecutionEventSequence != before {
			t.Fatal("repeat recovery allocated another event cursor")
		}

		assigned, err = s.PollJob(worker.ID, JobPollFilter{})
		if err != nil || assigned.Attempt != 2 {
			t.Fatalf("recovered assignment=%#v err=%v", assigned, err)
		}
		expireMemoryJobLease(t, s, assigned.ID, 2)
		recovered, err = s.RecoverExpiredJobLeases(time.Now().UTC())
		if err != nil || len(recovered) != 1 || recovered[0].Status != jobs.StatusFailed {
			t.Fatalf("terminal recovery=%#v err=%v", recovered, err)
		}
		assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageFailed)
	})

	t.Run("pending retry terminalization keeps pending attempt identity", func(t *testing.T) {
		for _, cancel := range []bool{false, true} {
			s, projectID, datasetID := newJobLifecycleFixture(t)
			worker, _ := s.RegisterWorker(projectID, "worker", "gpu")
			job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
			_, _ = s.PollJob(worker.ID, JobPollFilter{})
			queued, requeued, err := s.RetryJob(job.ID, "retry", RetryJobOptions{})
			if err != nil || !requeued || activeJobProgressAttempt(queued) != 2 {
				t.Fatalf("queue pending attempt: job=%#v requeued=%t err=%v", queued, requeued, err)
			}
			if cancel {
				queued, err = s.CancelJob(job.ID, "cancel", map[string]any{"failure_class": "cancelled"})
			} else {
				queued, _, err = s.RetryJob(job.ID, "force fail", RetryJobOptions{ForceFail: true})
			}
			if err != nil || activeJobProgressAttempt(queued) != 2 {
				t.Fatalf("terminal pending attempt: cancel=%t job=%#v err=%v", cancel, queued, err)
			}
			wantStage := jobs.ProgressStageFailed
			if cancel {
				wantStage = jobs.ProgressStageCancelled
			}
			assertJobProgressStage(t, s, job.ID, 2, wantStage)
		}
	})

	t.Run("poll performs recovery before reassignment", func(t *testing.T) {
		s, projectID, datasetID := newJobLifecycleFixture(t)
		worker, _ := s.RegisterWorker(projectID, "worker", "gpu")
		job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
		assigned, _ := s.PollJob(worker.ID, JobPollFilter{})
		expireMemoryJobLease(t, s, assigned.ID, 2)

		reassigned, err := s.PollJob(worker.ID, JobPollFilter{})
		if err != nil || reassigned.Attempt != 2 || reassigned.Status != jobs.StatusAssigned {
			t.Fatalf("poll recovery reassignment=%#v err=%v", reassigned, err)
		}
		assertJobProgressStage(t, s, job.ID, 2, jobs.ProgressStageWorkerStarting)
		types := []string{}
		for _, event := range memoryJobTransitionEvents(s, job.ID) {
			types = append(types, event.EventType)
		}
		if !containsString(types, execution.EventJobLeaseRecovered) {
			t.Fatalf("poll recovery did not emit lease event: %#v", types)
		}
	})
}

func TestMemoryCancellationIsExplicitAndTerminal(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
	cancelled, err := s.CancelJob(job.ID, "cancelled because s3://private/path", map[string]any{
		"failure_class": "cancelled",
		"cancel_reason": "user_requested",
	})
	if err != nil || cancelled.Status != jobs.StatusFailed {
		t.Fatalf("cancel = %#v err=%v", cancelled, err)
	}
	progress := assertJobProgressStage(t, s, job.ID, 1, jobs.ProgressStageCancelled)
	if progress.Status != jobs.ProgressStatusCancelled || cancelled.Config["failure_class"] != "cancelled" {
		t.Fatalf("cancellation authority mismatch: job=%#v progress=%#v", cancelled, progress)
	}
	assertJobTransitionTypes(t, s, job.ID, execution.EventJobQueued, execution.EventJobCancelled)
	assertJobEventsContainNoText(t, s, job.ID, "private")
	before := s.nextExecutionEventSequence
	if _, err := s.CancelJob(job.ID, "duplicate", nil); err != nil {
		t.Fatal(err)
	}
	if s.nextExecutionEventSequence != before {
		t.Fatal("duplicate cancellation appended an event")
	}
}

func TestMemoryLifecycleValidationRollbackLeavesAllThreeRecordsUnchanged(t *testing.T) {
	s, projectID, datasetID := newJobLifecycleFixture(t)
	job, _ := s.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID})
	beforeJob, _ := s.GetJob(job.ID)
	beforeProgress, _ := s.GetJobProgress(job.ID, 1)
	beforeSequence := s.nextExecutionEventSequence
	beforeEvents := len(s.executionEvents)

	next := beforeJob
	next.Status = jobs.StatusAssigned
	transition, progress := newJobLifecycleTransition(
		next,
		execution.TransitionJobAssigned,
		1,
		jobs.ProgressStageWorkerStarting,
		jobs.ProgressStatusRunning,
		progressRevisionAssigned,
		"worker_assigned",
		"worker_assignment",
	)
	progress.Message = "unsafe\nmessage"
	s.mu.Lock()
	_, _, _, err := s.commitJobLifecycleLocked(next, progress, transition, time.Now().UTC())
	s.mu.Unlock()
	if err == nil {
		t.Fatal("expected lifecycle validation failure")
	}

	afterJob, _ := s.GetJob(job.ID)
	afterProgress, _ := s.GetJobProgress(job.ID, 1)
	if !reflect.DeepEqual(afterJob, beforeJob) || !reflect.DeepEqual(afterProgress, beforeProgress) {
		t.Fatalf("rollback changed job/progress: before=%#v %#v after=%#v %#v", beforeJob, beforeProgress, afterJob, afterProgress)
	}
	if s.nextExecutionEventSequence != beforeSequence || len(s.executionEvents) != beforeEvents {
		t.Fatalf("rollback changed events: sequence %d->%d count %d->%d", beforeSequence, s.nextExecutionEventSequence, beforeEvents, len(s.executionEvents))
	}
}

func TestLifecycleMutationSitesUseAtomicCompositionHelpers(t *testing.T) {
	postgresSites := map[string][]string{
		"postgres_job_records.go": {"CreateJobWithOptions", "PollJob", "ReportMetric", "recoverExpiredJobLeasesTx"},
		"postgres_jobs.go":        {"RetryJob", "finishJobWithTransition"},
	}
	for filename, functions := range postgresSites {
		for _, function := range functions {
			if !sourceFunctionCalls(t, filename, function, "commitJobLifecycleTx") {
				t.Errorf("%s.%s does not compose job, progress, and event through commitJobLifecycleTx", filename, function)
			}
		}
	}
	memorySites := []string{"CreateJobWithOptions", "PollJob", "ReportMetric", "recoverExpiredJobLeasesLocked", "RetryJob", "finishJobWithTransition"}
	for _, function := range memorySites {
		if !sourceFunctionCalls(t, "memory.go", function, "commitJobLifecycleLocked") {
			t.Errorf("memory.go.%s does not compose job, progress, and event through commitJobLifecycleLocked", function)
		}
	}
}

func newJobLifecycleFixture(t *testing.T) (*MemoryStore, string, string) {
	t.Helper()
	s := NewMemoryStore()
	project, err := s.CreateProject("lifecycle", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := s.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	if err != nil {
		t.Fatal(err)
	}
	return s, project.ID, dataset.ID
}

func assertJobProgressStage(t *testing.T, s *MemoryStore, jobID string, attempt int, stage string) jobs.JobProgress {
	t.Helper()
	progress, err := s.GetJobProgress(jobID, attempt)
	if err != nil {
		t.Fatal(err)
	}
	if progress.Stage != stage {
		t.Fatalf("job %s attempt %d stage=%q want=%q: %#v", jobID, attempt, progress.Stage, stage, progress)
	}
	return progress
}

func assertJobTransitionTypes(t *testing.T, s *MemoryStore, jobID string, want ...string) {
	t.Helper()
	events := memoryJobTransitionEvents(s, jobID)
	got := make([]string, len(events))
	for index, event := range events {
		got[index] = event.EventType
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("job transition types=%#v want=%#v", got, want)
	}
}

func memoryJobTransitionEvents(s *MemoryStore, jobID string) []execution.ExecutionEvent {
	events := []execution.ExecutionEvent{}
	for _, event := range s.executionEvents {
		if execution.IsDurableTransitionEventType(event.EventType) && event.Payload["job_id"] == jobID {
			events = append(events, event)
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].Sequence < events[j].Sequence })
	return events
}

func assertJobEventsContainNoText(t *testing.T, s *MemoryStore, jobID string, forbidden string) {
	t.Helper()
	for _, event := range memoryJobTransitionEvents(s, jobID) {
		if strings.Contains(event.Message, forbidden) {
			t.Fatalf("event message leaked %q: %#v", forbidden, event)
		}
		for _, value := range event.Payload {
			if text, ok := value.(string); ok && strings.Contains(text, forbidden) {
				t.Fatalf("event payload leaked %q: %#v", forbidden, event)
			}
		}
	}
}

func expireMemoryJobLease(t *testing.T, s *MemoryStore, jobID string, maxAttempts int) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	job := s.jobs[jobID]
	expired := time.Now().UTC().Add(-time.Minute)
	job.LeaseExpiresAt = &expired
	job.MaxAttempts = maxAttempts
	s.jobs[jobID] = job
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func sourceFunctionCalls(t *testing.T, filename string, functionName string, calledName string) bool {
	t.Helper()
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve lifecycle test source")
	}
	parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(currentFile), filename), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, declaration := range parsed.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != functionName {
			continue
		}
		found := false
		ast.Inspect(function.Body, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch target := call.Fun.(type) {
			case *ast.Ident:
				found = found || target.Name == calledName
			case *ast.SelectorExpr:
				found = found || target.Sel.Name == calledName
			}
			return !found
		})
		return found
	}
	t.Fatalf("function %s not found in %s", functionName, filename)
	return false
}
