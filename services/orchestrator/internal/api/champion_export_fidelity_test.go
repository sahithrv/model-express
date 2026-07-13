package api

import (
	"encoding/json"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestValidateChampionExportExecutionContractRequiresMatchingHashesAndPreprocessing(t *testing.T) {
	contract := map[string]any{
		"schema_version":              "export_execution_contract_v1",
		"execution_record_ref":        "/jobs/job_1/execution-record",
		"attempt_id":                  "attempt-1",
		"task":                        "image_classification",
		"capability_version":          "cap-v1",
		"accepted_spec_hash":          "accepted-hash",
		"realized_effective_hash":     "realized-hash",
		"fidelity_verdict":            "MATCHED",
		"image_size":                  256,
		"preprocessing_contract_hash": "preprocessing-hash",
		"preprocessing": map[string]any{
			"resize_strategy": "preserve_aspect_pad",
			"crop_strategy":   "none",
			"normalization":   "imagenet",
			"bbox_mode":       "ignore",
		},
	}
	manifestContract := copyPayloadMap(contract)
	manifest := map[string]any{
		"metadata": map[string]any{
			"execution_contract": manifestContract,
			"input_shape":        []int{1, 3, 256, 256},
			"preprocessing_contract": map[string]any{
				"config": map[string]any{
					"resize_strategy": "preserve_aspect_pad",
					"crop_strategy":   "none",
					"normalization":   "imagenet",
					"bbox_mode":       "ignore",
				},
			},
		},
	}
	if err := validateChampionExportExecutionContract(contract, manifest); err != nil {
		t.Fatalf("matching execution contract was rejected: %v", err)
	}

	manifestContract["realized_effective_hash"] = "wrong-realized-hash"
	if err := validateChampionExportExecutionContract(contract, manifest); err == nil {
		t.Fatal("expected mismatched realized hash to block export")
	}
	manifestContract["realized_effective_hash"] = "realized-hash"
	preprocessing := payloadMap(payloadMap(payloadMap(manifest, "metadata"), "preprocessing_contract"), "config")
	preprocessing["normalization"] = "none"
	if err := validateChampionExportExecutionContract(contract, manifest); err == nil {
		t.Fatal("expected contradictory normalization to block export")
	}
}

func TestCompactJobPayloadExposesHashesAndReferencesWithoutReceipts(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	spec, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"local_simulator",
		map[string]any{"model": "resnet18", "epochs": 3},
		map[string]any{"model": "resnet18", "epochs": 3},
	)
	if err != nil {
		t.Fatal(err)
	}
	specPayload, _ := spec.Payload()
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		execution.ExecutionSpecConfigKey: specPayload,
		"dataset_id":                     dataset.ID,
		"large_raw_config":               strings.Repeat("x", 10_000),
	})
	if err != nil {
		t.Fatal(err)
	}
	payload := newServer(memoryStore).compactJobPayload(job)
	encoded, _ := json.Marshal(payload)
	body := string(encoded)
	if strings.Contains(body, "large_raw_config") || strings.Contains(body, "framework_arguments") || strings.Contains(body, "accepted_spec\"") {
		t.Fatalf("compact job leaked execution receipt/config: %s", body)
	}
	references, ok := payload["execution_references"].(*runs.ExecutionArtifactReferences)
	if !ok || references.AcceptedSpecHash == "" || references.ExecutionRecordRef != "/jobs/"+job.ID+"/execution-record" {
		t.Fatalf("compact job omitted execution hashes/references: %#v", payload)
	}
}

func TestChampionExportJobUsesFinalizedRealizationContract(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "checksum", 1)
	spec, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		map[string]any{"model": "resnet18", "epochs": 3, "image_size": 256},
		map[string]any{"model": "resnet18", "epochs": 3, "image_size": 256},
	)
	if err != nil {
		t.Fatal(err)
	}
	specPayload, _ := spec.Payload()
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		execution.ExecutionSpecConfigKey: specPayload,
		"dataset_id":                     dataset.ID,
		"model":                          "resnet18",
	})
	if err != nil {
		t.Fatal(err)
	}
	worker, _ := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	attemptID := jobConfigString(assigned.Config, "active_attempt_id")
	record, _ := memoryStore.GetJobExecutionRecord(job.ID)
	realized := record.AcceptedSpec.AcceptedSpec
	preprocessing := payloadMap(realized, "preprocessing")
	frameworkArguments := map[string]any{"preprocessing": map[string]any{"config": preprocessing}}
	for _, observation := range []execution.RealizationObservationCreate{
		{AttemptID: attemptID, Stage: execution.ExecutionObservationInitialized, IdempotencyKey: "initialized", RealizedConfig: realized, FrameworkArguments: frameworkArguments},
		{AttemptID: attemptID, Stage: execution.ExecutionObservationFinalized, IdempotencyKey: "finalized", RealizedConfig: realized, FrameworkArguments: frameworkArguments},
	} {
		if _, _, err := memoryStore.AppendRealizationObservation(job.ID, observation); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := memoryStore.CompleteJob(job.ID, "run"); err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{Model: "resnet18", Status: jobs.StatusSucceeded}); err != nil {
		t.Fatal(err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID: project.ID, DatasetID: dataset.ID, JobID: job.ID,
		Metrics: map[string]any{"model": "resnet18"}, DeploymentProfile: map[string]any{"artifact_uri": "file:///checkpoint.pt"},
	})
	if err != nil {
		t.Fatal(err)
	}
	server := newServer(memoryStore)
	if _, err := server.ensureChampionExport(project.ID, champion, job, "onnx", "", nil); err != nil {
		t.Fatalf("ensure champion export: %v", err)
	}
	projectJobs, _ := memoryStore.ListProjectJobs(project.ID)
	var exportJob jobs.ExperimentJob
	for _, candidate := range projectJobs {
		if candidate.Template == jobs.TemplateExportChampion {
			exportJob = candidate
		}
	}
	contract := payloadMap(exportJob.Config, "execution_contract")
	updatedRecord, _ := memoryStore.GetJobExecutionRecord(job.ID)
	latest := updatedRecord.Attempts[0]
	if payloadString(contract, "accepted_spec_hash") != updatedRecord.AcceptedSpec.AcceptedSpecHash ||
		payloadString(contract, "realized_effective_hash") != latest.RealizedEffectiveHash ||
		payloadString(contract, "execution_record_ref") != "/jobs/"+job.ID+"/execution-record" {
		t.Fatalf("export job does not reference finalized execution record: %#v", contract)
	}
	if !canonicalValuesEqual(payloadMap(contract, "preprocessing"), preprocessing) {
		t.Fatalf("export preprocessing differs from training realization: contract=%#v realized=%#v", payloadMap(contract, "preprocessing"), preprocessing)
	}
	if _, leaked := exportJob.Config["framework_arguments"]; leaked {
		t.Fatal("export job duplicated full framework arguments")
	}
}

func TestCompactRunEvaluationKeepsOnlyNarrowExecutionReferences(t *testing.T) {
	references := &runs.ExecutionArtifactReferences{
		SchemaVersion: "run_execution_references_v1", AcceptedSpecHash: "accepted", RealizedEffectiveHash: "realized",
		ExecutionRecordRef: "/jobs/job_1/execution-record", TrainingExportManifestURI: "file:///manifest.json",
	}
	compact := compactTrainingRunEvaluations([]runs.TrainingRunEvaluation{{
		JobID: "job_1", ModelProfile: map[string]any{"export_manifest": map[string]any{"large": strings.Repeat("x", 10_000)}},
		ObjectiveProfile: map[string]any{"large": true}, PerClassMetrics: map[string]any{"cat": map[string]any{"f1": 1}},
		ConfusionMatrix: [][]int{{1}}, HolisticScores: map[string]any{"score": 1}, ExecutionReferences: references,
	}})[0]
	if compact["execution_references"] != references || len(compact) != 3 {
		t.Fatalf("compact evaluation retained a large receipt or dropped references: %#v", compact)
	}
}
