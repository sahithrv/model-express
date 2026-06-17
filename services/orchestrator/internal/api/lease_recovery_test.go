package api

import (
	"testing"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/workers"
)

func TestRecoverExpiredLeasesOnceRequeuesExpiredAssignedJob(t *testing.T) {
	memoryStore, server, _, worker, assigned := newLeaseRecoveryTestFixture(t)

	recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
	if err != nil {
		t.Fatalf("recover expired lease: %v", err)
	}
	if len(recovered) != 1 {
		t.Fatalf("expected one recovered job, got %#v", recovered)
	}
	if recovered[0].Status != jobs.StatusQueued {
		t.Fatalf("expected job to be requeued, got %#v", recovered[0])
	}
	if recovered[0].LeaseExpiresAt != nil || recovered[0].LeaseOwnerWorkerID != "" {
		t.Fatalf("expected recovered job lease to be cleared, got %#v", recovered[0])
	}

	stored, err := memoryStore.GetJob(assigned.ID)
	if err != nil {
		t.Fatalf("get recovered job: %v", err)
	}
	if stored.Status != jobs.StatusQueued {
		t.Fatalf("expected stored job to be queued, got %#v", stored)
	}
	updatedWorker, err := memoryStore.GetWorker(worker.ID)
	if err != nil {
		t.Fatalf("get worker: %v", err)
	}
	if updatedWorker.CurrentJobID != "" {
		t.Fatalf("expected worker current job to be cleared, got %#v", updatedWorker)
	}
}

func TestRecoverExpiredLeasesOnceFailsExpiredMaxAttemptJob(t *testing.T) {
	memoryStore, server, project, worker, assigned := newLeaseRecoveryTestFixture(t)

	for attempt := 1; attempt < assigned.MaxAttempts; attempt++ {
		recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
		if err != nil {
			t.Fatalf("recover attempt %d: %v", attempt, err)
		}
		if len(recovered) != 1 || recovered[0].Status != jobs.StatusQueued {
			t.Fatalf("expected queued recovery at attempt %d, got %#v", attempt, recovered)
		}
		nextAssigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
		if err != nil {
			t.Fatalf("poll attempt %d: %v", attempt+1, err)
		}
		assigned = *nextAssigned
		if assigned.Attempt != attempt+1 {
			t.Fatalf("expected attempt %d, got %#v", attempt+1, assigned)
		}
	}

	recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
	if err != nil {
		t.Fatalf("recover max attempt: %v", err)
	}
	if len(recovered) != 1 {
		t.Fatalf("expected one recovered job, got %#v", recovered)
	}
	if recovered[0].Status != jobs.StatusFailed {
		t.Fatalf("expected job to fail at max attempts, got %#v", recovered[0])
	}
	if recovered[0].CompletedAt == nil {
		t.Fatalf("expected failed recovered job to have completed_at, got %#v", recovered[0])
	}

	summary, err := memoryStore.GetTrainingRunSummary(assigned.ID)
	if err != nil {
		t.Fatalf("get training summary: %v", err)
	}
	if summary.Status != jobs.StatusFailed {
		t.Fatalf("expected failed training summary, got %#v", summary)
	}

	events, err := memoryStore.ListProjectExecutionEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !leaseRecoveryHasExecutionEvent(events, execution.EventExecutionFailed) {
		t.Fatalf("expected execution failure event, got %#v", events)
	}
}

func TestRecoverExpiredLeasesOnceTerminalFailureRunsTrainingHooks(t *testing.T) {
	t.Setenv(leaseRecoveryIntervalEnv, "0")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "false")
	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "false")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("lease recovery", "select prior model")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 4, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 2),
		testExperiment("efficientnet_b0", 2),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}
	priorJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"plan_id":    plan.ID,
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create prior job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(priorJob.ID, "prior-run"); err != nil {
		t.Fatalf("complete prior job: %v", err)
	}
	score := 0.87
	if _, err := memoryStore.UpsertTrainingRunSummary(priorJob.ID, runs.TrainingRunSummaryUpdate{
		Status:       jobs.StatusSucceeded,
		BestMacroF1:  &score,
		BestAccuracy: &score,
	}); err != nil {
		t.Fatalf("upsert prior summary: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunEvaluation(priorJob.ID, runs.TrainingRunEvaluationUpdate{
		ModelProfile: map[string]any{
			"artifact_uri": "s3://model-express/model-express/artifacts/" + priorJob.ID + "/model.onnx",
		},
	}); err != nil {
		t.Fatalf("upsert prior evaluation: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	failingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"plan_id":    plan.ID,
		"model":      "efficientnet_b0",
		"provider":   "modal",
	})
	if err != nil {
		t.Fatalf("create failing job: %v", err)
	}
	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{Provider: "modal"})
	if err != nil {
		t.Fatalf("poll failing job: %v", err)
	}
	if assigned.ID != failingJob.ID {
		t.Fatalf("expected failing job assignment, got %#v", assigned)
	}
	for attempt := 1; attempt < assigned.MaxAttempts; attempt++ {
		recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
		if err != nil {
			t.Fatalf("recover attempt %d: %v", attempt, err)
		}
		if len(recovered) != 1 || recovered[0].Status != jobs.StatusQueued {
			t.Fatalf("expected queued recovery at attempt %d, got %#v", attempt, recovered)
		}
		assigned, err = memoryStore.PollJob(worker.ID, store.JobPollFilter{Provider: "modal"})
		if err != nil {
			t.Fatalf("poll retry attempt %d: %v", attempt+1, err)
		}
	}

	recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
	if err != nil {
		t.Fatalf("recover max attempt: %v", err)
	}
	if len(recovered) != 1 || recovered[0].Status != jobs.StatusFailed {
		t.Fatalf("expected terminal failed recovery, got %#v", recovered)
	}
	champion := waitForLeaseRecoveryChampion(t, memoryStore, project.ID)
	if champion.JobID != priorJob.ID {
		t.Fatalf("expected prior successful job %s as champion, got %#v", priorJob.ID, champion)
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list champion exports: %v", err)
	}
	if len(exports) != 1 || exports[0].JobID != priorJob.ID {
		t.Fatalf("expected one export for prior champion, got %#v", exports)
	}

	server.runTrainingTerminalHooks(failingJob.ID)
	exports, err = memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list champion exports after rerun: %v", err)
	}
	if len(exports) != 1 {
		t.Fatalf("expected terminal hooks to be idempotent, got %#v", exports)
	}
}

func TestRecoverExpiredLeasesOnceLeavesNonExpiredJobUntouched(t *testing.T) {
	memoryStore, server, _, worker, assigned := newLeaseRecoveryTestFixture(t)

	recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(-time.Second))
	if err != nil {
		t.Fatalf("recover non-expired lease: %v", err)
	}
	if len(recovered) != 0 {
		t.Fatalf("expected no recovered jobs, got %#v", recovered)
	}

	stored, err := memoryStore.GetJob(assigned.ID)
	if err != nil {
		t.Fatalf("get assigned job: %v", err)
	}
	if stored.Status != jobs.StatusAssigned {
		t.Fatalf("expected assigned job to remain assigned, got %#v", stored)
	}
	if stored.LeaseExpiresAt == nil || !stored.LeaseExpiresAt.Equal(*assigned.LeaseExpiresAt) {
		t.Fatalf("expected lease expiry to remain unchanged, got %#v", stored)
	}
	updatedWorker, err := memoryStore.GetWorker(worker.ID)
	if err != nil {
		t.Fatalf("get worker: %v", err)
	}
	if updatedWorker.CurrentJobID != assigned.ID {
		t.Fatalf("expected worker to keep assigned job, got %#v", updatedWorker)
	}
}

func newLeaseRecoveryTestFixture(t *testing.T) (*store.MemoryStore, *Server, projects.Project, workers.Worker, jobs.ExperimentJob) {
	t.Helper()
	t.Setenv(leaseRecoveryIntervalEnv, "0")
	t.Setenv("MODEL_EXPRESS_LOG_DIR", t.TempDir())

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("lease recovery", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"plan_id":    "plan_lease_recovery",
	}); err != nil {
		t.Fatalf("create job: %v", err)
	}
	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
	if err != nil {
		t.Fatalf("poll job: %v", err)
	}
	if assigned.LeaseExpiresAt == nil {
		t.Fatalf("expected assigned job to have lease: %#v", assigned)
	}

	return memoryStore, newServer(memoryStore), project, worker, *assigned
}

func leaseRecoveryHasExecutionEvent(events []execution.ExecutionEvent, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}

func waitForLeaseRecoveryChampion(t *testing.T, memoryStore *store.MemoryStore, projectID string) runs.ProjectChampion {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		champion, err := memoryStore.GetProjectChampion(projectID)
		if err == nil {
			return champion
		}
		time.Sleep(10 * time.Millisecond)
	}
	champion, err := memoryStore.GetProjectChampion(projectID)
	if err != nil {
		t.Fatalf("expected champion after terminal hooks: %v", err)
	}
	return champion
}
