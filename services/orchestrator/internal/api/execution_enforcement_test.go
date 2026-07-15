package api

import (
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

func TestTrainingCompletionEnforcementUsesFinalEvidenceEligibility(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "enforce")
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	server := newServer(memoryStore)

	pendingJob := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "pending")
	if err := server.validateTrainingCompletionFidelity(pendingJob); err == nil || !strings.Contains(err.Error(), "final realization") {
		t.Fatalf("pending real training completion was not blocked: %v", err)
	}

	mismatchJob := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "mismatch")
	finalizeCompletionTestJob(t, memoryStore, mismatchJob.ID, true)
	if err := server.validateTrainingCompletionFidelity(mismatchJob); err == nil || !strings.Contains(err.Error(), execution.ExecutionVerdictMismatch) {
		t.Fatalf("mismatch completion was not blocked by evidence verdict: %v", err)
	}

	matchedJob := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "matched")
	finalizeCompletionTestJob(t, memoryStore, matchedJob.ID, false)
	if err := server.validateTrainingCompletionFidelity(matchedJob); err != nil {
		t.Fatalf("matched finalized completion was blocked: %v", err)
	}
}

func TestTrainingCompletionShadowRollbackDoesNotRewriteExecutionRecord(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "shadow")
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	job := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "pending")

	if err := newServer(memoryStore).validateTrainingCompletionFidelity(job); err != nil {
		t.Fatalf("shadow rollback did not permit compatibility completion: %v", err)
	}
	record, _ := memoryStore.GetJobExecutionRecord(job.ID)
	if len(record.Attempts) != 1 || record.Attempts[0].LifecycleStatus != execution.ExecutionLifecyclePending || record.Attempts[0].FidelityVerdict != nil {
		t.Fatalf("shadow rollback rewrote stored execution evidence: %#v", record)
	}
}

func TestTrainingCompletionEnforcementKeepsLegacyJobsCompatible(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "enforce")
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	legacy, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID, "model": "legacy-worker-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	server := newServer(memoryStore)
	if err := server.validateTrainingCompletionFidelity(legacy); err != nil {
		t.Fatalf("legacy completion lost compatibility: %v", err)
	}
	references := server.executionReferencesForJob(legacy.ID, nil)
	if references == nil || references.FidelityVerdict != execution.ExecutionVerdictUnverified {
		t.Fatalf("legacy read was not visibly unverified: %#v", references)
	}
}

func createVersionedCompletionTestJob(t *testing.T, memoryStore *store.MemoryStore, projectID, datasetID, label string) jobs.ExperimentJob {
	t.Helper()
	requested := map[string]any{"model": "resnet18", "epochs": 3, "batch_size": 16}
	spec, err := execution.BuildExecutionSpecV1("image_classification", "modal_torchvision", requested, requested)
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := spec.Payload()
	job, err := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{
		execution.ExecutionSpecConfigKey: payload,
		"dataset_id":                     datasetID,
		"model":                          "resnet18",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.CreateAttemptExecutionRecord(job.ID, job.ID+"-"+label, 1); err != nil {
		t.Fatal(err)
	}
	return job
}

func finalizeCompletionTestJob(t *testing.T, memoryStore *store.MemoryStore, jobID string, mismatch bool) {
	t.Helper()
	record, _ := memoryStore.GetJobExecutionRecord(jobID)
	realized := clonePayload(record.AcceptedSpec.AcceptedSpec)
	if mismatch {
		realized["batch_size"] = 8
	}
	if _, _, err := memoryStore.AppendRealizationObservation(jobID, execution.RealizationObservationCreate{
		AttemptID: record.Attempts[0].AttemptID, Stage: execution.ExecutionObservationFinalized,
		IdempotencyKey: "finalized", RealizedConfig: realized,
	}); err != nil {
		t.Fatal(err)
	}
}
