package store

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

func TestExecutionIdentityMigrationMakesPendingRealizedHashNullable(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/014_execution_identity_semantics.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"SET realized_effective_hash = NULL",
		"ALTER COLUMN realized_effective_hash DROP DEFAULT",
		"ALTER COLUMN realized_effective_hash DROP NOT NULL",
		"adjustment_reason_codes jsonb",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("identity migration omitted %q", required)
		}
	}
}

func TestLearningChampionFidelityMigrationPersistsScorecardEligibility(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/015_learning_champion_fidelity.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"fidelity_verdicts jsonb",
		"evidence_eligible boolean",
		"requested_mechanism text",
		"realized_mechanism_identity text",
		"accepted_spec_hash text",
		"realized_effective_hash text",
		"adjustment_reason_codes jsonb",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("learning/champion fidelity migration omitted %q", required)
		}
	}
}

func TestExportExecutionReferencesMigrationKeepsRunEvidenceNarrow(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/016_export_execution_references.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	if strings.Count(sqlText, "execution_references jsonb") != 2 || strings.Contains(sqlText, "framework_arguments") {
		t.Fatalf("export reference migration must add two compact reference columns without receipt payloads: %s", sqlText)
	}
}

func versionedTestConfig(t *testing.T) map[string]any {
	t.Helper()
	spec, err := execution.BuildExecutionSpecV1("image_classification", "local_simulator", map[string]any{"model": "resnet18", "epochs": 3}, map[string]any{"model": "resnet18", "epochs": 3})
	if err != nil {
		t.Fatal(err)
	}
	payload, err := spec.Payload()
	if err != nil {
		t.Fatal(err)
	}
	return map[string]any{execution.ExecutionSpecConfigKey: payload, "provider": "local"}
}

func TestAttemptExecutionRecordsAreImmutableIdempotentAndRetryScoped(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("p", "g")
	dataset, _ := s.CreateDataset(project.ID, "d", "s3://bucket/data", "sha", 1)
	config := versionedTestConfig(t)
	config["dataset_id"] = dataset.ID
	job, err := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, config)
	if err != nil {
		t.Fatal(err)
	}
	config[execution.ExecutionSpecConfigKey].(map[string]any)["accepted_config"].(map[string]any)["model"] = "mutated-after-queue"
	immutableRecord, err := s.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if immutableRecord.AcceptedSpec.AcceptedSpec["model"] == "mutated-after-queue" {
		t.Fatal("accepted spec snapshot changed after scheduling")
	}
	if _, err := s.UpdateJobConfig(job.ID, map[string]any{execution.ExecutionSpecConfigKey: map[string]any{}}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected immutable spec rejection, got %v", err)
	}
	worker, _ := s.RegisterWorker(project.ID, "w", "local")
	assigned, err := s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	record, err := s.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Attempts) != 1 || record.Attempts[0].LifecycleStatus != execution.ExecutionLifecyclePending || record.Attempts[0].FidelityVerdict != nil {
		t.Fatalf("unexpected pending attempt: %#v", record.Attempts)
	}
	if record.Attempts[0].RealizedEffectiveHash != "" || len(record.Attempts[0].AdjustmentReasonCodes) != 0 {
		t.Fatalf("pending attempt fabricated realized identity: %#v", record.Attempts[0])
	}
	pendingJSON, err := json.Marshal(record.Attempts[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(pendingJSON), "realized_effective_hash") {
		t.Fatalf("pending attempt API fabricated realized identity: %s", pendingJSON)
	}

	create := execution.RealizationObservationCreate{AttemptID: jobAttemptID(job.ID, 1), SchemaVersion: execution.ExecutionObservationSchemaV1, Stage: execution.ExecutionObservationInitialized, IdempotencyKey: "init-1", RealizedConfig: record.AcceptedSpec.AcceptedSpec, FrameworkArguments: map[string]any{"api_token": "must-not-persist", "optimizer": "adamw"}}
	first, created, err := s.AppendRealizationObservation(job.ID, create)
	if err != nil || !created {
		t.Fatalf("first callback: created=%v err=%v", created, err)
	}
	if first.FrameworkArguments["api_token"] != "[REDACTED]" || first.FrameworkArguments["optimizer"] != "adamw" {
		t.Fatalf("framework argument redaction failed: %#v", first.FrameworkArguments)
	}
	second, created, err := s.AppendRealizationObservation(job.ID, create)
	if err != nil || created || second.ID != first.ID {
		t.Fatalf("duplicate callback was not idempotent: created=%v err=%v", created, err)
	}
	if _, requeued, err := s.RetryJob(job.ID, "retry", RetryJobOptions{}); err != nil || !requeued {
		t.Fatalf("retry: requeued=%v err=%v", requeued, err)
	}
	assigned, err = s.PollJob(worker.ID, JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if assigned.Attempt != 2 {
		t.Fatalf("attempt=%d", assigned.Attempt)
	}
	if _, err := s.FailJob(job.ID, "pre-init failure"); err != nil {
		t.Fatal(err)
	}
	record, err = s.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Attempts) != 2 {
		t.Fatalf("attempt count=%d", len(record.Attempts))
	}
	if record.Attempts[0].LifecycleStatus != execution.ExecutionLifecycleInitialized {
		t.Fatalf("first lifecycle=%s", record.Attempts[0].LifecycleStatus)
	}
	if record.Attempts[1].LifecycleStatus != execution.ExecutionLifecycleNotRealized || record.Attempts[1].FidelityVerdict != nil {
		t.Fatalf("second attempt=%#v", record.Attempts[1])
	}
}

func TestRealizedIdentityRetainsAcceptedLineageAndRecordsAdjustmentReasons(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("p", "g")
	dataset, _ := s.CreateDataset(project.ID, "d", "s3://bucket/data", "sha", 1)
	config := versionedTestConfig(t)
	config["dataset_id"] = dataset.ID
	job, err := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, config)
	if err != nil {
		t.Fatal(err)
	}
	worker, _ := s.RegisterWorker(project.ID, "w", "local")
	if _, err := s.PollJob(worker.ID, JobPollFilter{}); err != nil {
		t.Fatal(err)
	}
	record, err := s.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	attemptID := jobAttemptID(job.ID, 1)
	matched, created, err := s.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{
		AttemptID: attemptID, Stage: execution.ExecutionObservationInitialized,
		IdempotencyKey: "identity-initialized", RealizedConfig: record.AcceptedSpec.AcceptedSpec,
	})
	if err != nil || !created || matched.FidelityVerdict != execution.ExecutionVerdictMatched {
		t.Fatalf("matched observation=%#v created=%v err=%v", matched, created, err)
	}
	adjustedConfig := cloneJSONMap(record.AcceptedSpec.AcceptedSpec)
	acceptedBatch, ok := adjustedConfig["batch_size"].(float64)
	if !ok || acceptedBatch <= 1 {
		t.Fatalf("unexpected accepted batch: %#v", adjustedConfig["batch_size"])
	}
	adjustedConfig["batch_size"] = acceptedBatch / 2
	adjusted, created, err := s.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{
		AttemptID: attemptID, Stage: execution.ExecutionObservationFinalized,
		IdempotencyKey: "identity-finalized", RealizedConfig: adjustedConfig,
		AdjustmentPolicy: "batch_size_recovery",
	})
	if err != nil || !created || adjusted.FidelityVerdict != execution.ExecutionVerdictApprovedAdjustment {
		t.Fatalf("adjusted observation=%#v created=%v err=%v", adjusted, created, err)
	}
	if adjusted.RealizedEffectiveHash == matched.RealizedEffectiveHash {
		t.Fatal("training-semantic batch adjustment did not change realized identity")
	}
	if len(adjusted.AdjustmentReasonCodes) != 1 || adjusted.AdjustmentReasonCodes[0] != execution.ExecutionAdjustmentReasonBatchSizeReduced {
		t.Fatalf("adjustment reasons=%#v", adjusted.AdjustmentReasonCodes)
	}
	updated, err := s.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.AcceptedSpec.AcceptedSpecHash != record.AcceptedSpec.AcceptedSpecHash {
		t.Fatal("realization changed accepted-spec lineage")
	}
	latest := updated.Attempts[0]
	if latest.RealizedEffectiveHash != adjusted.RealizedEffectiveHash || len(latest.AdjustmentReasonCodes) != 1 {
		t.Fatalf("attempt did not retain latest realized identity and reasons: %#v", latest)
	}
}

func TestSimulatedFinalizationIsExplicit(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("p", "g")
	dataset, _ := s.CreateDataset(project.ID, "d", "s3://bucket/data", "sha", 1)
	config := versionedTestConfig(t)
	config["dataset_id"] = dataset.ID
	job, _ := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, config)
	worker, _ := s.RegisterWorker(project.ID, "w", "local")
	_, _ = s.PollJob(worker.ID, JobPollFilter{})
	record, _ := s.GetJobExecutionRecord(job.ID)
	observation, created, err := s.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{AttemptID: jobAttemptID(job.ID, 1), Stage: "FINALIZED", IdempotencyKey: "sim-final", RealizedConfig: record.AcceptedSpec.AcceptedSpec, Simulated: true})
	if err != nil || !created || observation.FidelityVerdict != execution.ExecutionVerdictSimulated {
		t.Fatalf("observation=%#v created=%v err=%v", observation, created, err)
	}
	record, _ = s.GetJobExecutionRecord(job.ID)
	if record.Attempts[0].LifecycleStatus != execution.ExecutionLifecycleFinalized || record.Attempts[0].FidelityVerdict == nil || *record.Attempts[0].FidelityVerdict != execution.ExecutionVerdictSimulated {
		t.Fatalf("attempt=%#v", record.Attempts[0])
	}
}
