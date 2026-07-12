package store

import (
	"errors"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
)

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
