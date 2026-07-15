package api

import (
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/store"
)

func TestRunExecutionReferencesExposeNarrowAdjustmentSummary(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	job := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "adjusted-reference")
	record, _ := memoryStore.GetJobExecutionRecord(job.ID)
	realized := clonePayload(record.AcceptedSpec.AcceptedSpec)
	realized["batch_size"] = 8
	if _, _, err := memoryStore.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{
		AttemptID: record.Attempts[0].AttemptID, Stage: execution.ExecutionObservationFinalized,
		IdempotencyKey: "adjusted-reference-final", RealizedConfig: realized, AdjustmentPolicy: "batch_size_recovery",
	}); err != nil {
		t.Fatal(err)
	}

	references := newServer(memoryStore).executionReferencesForJob(job.ID, nil)
	if references == nil || references.FidelityVerdict != execution.ExecutionVerdictApprovedAdjustment {
		t.Fatalf("adjusted execution references are missing: %#v", references)
	}
	if len(references.AdjustmentReasonCodes) != 1 || references.AdjustmentReasonCodes[0] != execution.ExecutionAdjustmentReasonBatchSizeReduced {
		t.Fatalf("adjustment summary is missing or duplicated receipt data: %#v", references)
	}
}
