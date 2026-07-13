package api

import (
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

func TestExecutionOperationalMetricsCoverRolloutGatesAndUnverifiedReads(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "enforce")
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	server := newServer(memoryStore)

	legacy, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID, "model": "legacy"})
	if err != nil {
		t.Fatal(err)
	}
	server.compactJobPayload(legacy)

	matched := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "matched")
	finalizeCompletionTestJob(t, memoryStore, matched.ID, false)
	mismatch := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "mismatch")
	finalizeCompletionTestJob(t, memoryStore, mismatch.ID, true)
	if _, err := memoryStore.CompleteJob(mismatch.ID, "bad-mismatch-success"); err != nil {
		t.Fatal(err)
	}
	pending := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "pending")
	if _, err := memoryStore.CompleteJob(pending.ID, "bad-success"); err != nil {
		t.Fatal(err)
	}
	notRealized := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "not-realized")
	if _, err := memoryStore.MarkAttemptNotRealized(notRealized.ID, notRealized.ID+"-not-realized"); err != nil {
		t.Fatal(err)
	}
	adjusted := createVersionedCompletionTestJob(t, memoryStore, project.ID, dataset.ID, "adjusted")
	adjustedRecord, _ := memoryStore.GetJobExecutionRecord(adjusted.ID)
	adjustedConfig := clonePayload(adjustedRecord.AcceptedSpec.AcceptedSpec)
	adjustedConfig["batch_size"] = 8
	if _, _, err := memoryStore.AppendRealizationObservation(adjusted.ID, execution.RealizationObservationCreate{
		AttemptID: adjustedRecord.Attempts[0].AttemptID, Stage: execution.ExecutionObservationFinalized,
		IdempotencyKey: "adjusted-final", RealizedConfig: adjustedConfig, AdjustmentPolicy: "batch_size_recovery",
	}); err != nil {
		t.Fatal(err)
	}

	report := execution.ExecutionValidationReport{
		SchemaVersion: execution.ExecutionValidationSchemaVersionV1, Mode: execution.ValidationModeEnforce,
		Task: "image_classification", Runner: "modal_torchvision", AcceptedSpecHash: "blocked",
		WouldBlock: true, Findings: []execution.ExecutionValidationFinding{{
			Field: "unsupported", Classification: "unsupported", WouldBlock: true,
		}},
	}
	if _, err := memoryStore.CreateExecutionEvent(project.ID, "", execution.EventExecutionValidationReported, "blocked", map[string]any{"report": report}); err != nil {
		t.Fatal(err)
	}

	metrics, err := server.executionOperationalMetrics(project.ID, 100)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.ValidationMode != execution.ValidationModeEnforce || metrics.UnsupportedProposals != 1 || metrics.UnsupportedFindings != 1 {
		t.Fatalf("unsupported proposal metrics are wrong: %#v", metrics)
	}
	if metrics.PendingAttempts != 1 || metrics.NotRealizedAttempts != 1 || metrics.SuccessfulWithoutFinalRealization != 1 {
		t.Fatalf("pending/not-realized rollout metrics are wrong: %#v", metrics)
	}
	if metrics.MatchedVerdicts != 1 || metrics.MismatchVerdicts != 1 || metrics.ApprovedAdjustments != 1 {
		t.Fatalf("fidelity verdict metrics are wrong: %#v", metrics)
	}
	if metrics.SuccessfulMismatchVerdicts != 1 {
		t.Fatalf("successful mismatch rollout gate is wrong: %#v", metrics)
	}
	if metrics.LegacyUnverifiedJobs != 1 || metrics.UnverifiedReads != 1 {
		t.Fatalf("legacy compatibility metrics are wrong: %#v", metrics)
	}
	if metrics.MatchedByTask["image_classification"] != 1 {
		t.Fatalf("classification rollout gate is not measurable: %#v", metrics.MatchedByTask)
	}
}
