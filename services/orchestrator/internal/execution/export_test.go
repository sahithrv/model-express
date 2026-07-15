package execution

import (
	"strings"
	"testing"
	"time"
)

func TestBuildExportExecutionContractUsesFinalRealizedPreprocessing(t *testing.T) {
	record := finalizedExportRecord(t)
	contract, err := BuildExportExecutionContract(record, "/jobs/job_1/execution-record")
	if err != nil {
		t.Fatalf("build export contract: %v", err)
	}
	if contract.CapabilityVersion != "cap-v1" || contract.AcceptedSpecHash != "accepted-hash" || contract.RealizedEffectiveHash != "realized-hash" {
		t.Fatalf("contract does not reference the finalized execution: %#v", contract)
	}
	if contract.ImageSize != 256 || contract.Preprocessing["resize_strategy"] != "preserve_aspect_pad" || contract.Preprocessing["normalization"] != "dataset" {
		t.Fatalf("contract did not preserve realized preprocessing: %#v", contract.Preprocessing)
	}
	metadata, ok := contract.Preprocessing["normalization_metadata"].(map[string]any)
	if !ok || len(metadata) != 2 {
		t.Fatalf("contract did not preserve realized dataset normalization statistics: %#v", contract.Preprocessing)
	}
	if contract.PreprocessingContractHash == "" {
		t.Fatal("expected preprocessing contract hash")
	}
	if _, copiedRuntime := contract.Preprocessing["runtime"]; copiedRuntime {
		t.Fatal("full framework arguments leaked into compact export contract")
	}
}

func TestBuildExportExecutionContractRejectsContradictoryPreprocessing(t *testing.T) {
	for _, key := range []string{"resize_strategy", "normalization"} {
		t.Run(key, func(t *testing.T) {
			record := finalizedExportRecord(t)
			framework := record.Attempts[0].Observations[0].FrameworkArguments["preprocessing"].(map[string]any)
			config := framework["config"].(map[string]any)
			if key == "resize_strategy" {
				config[key] = "squash"
			} else {
				config[key] = "imagenet"
			}
			_, err := BuildExportExecutionContract(record, "/jobs/job_1/execution-record")
			if err == nil || !strings.Contains(err.Error(), key) {
				t.Fatalf("expected %s contradiction, got %v", key, err)
			}
		})
	}
}

func finalizedExportRecord(t *testing.T) ExecutionRecord {
	t.Helper()
	verdict := ExecutionVerdictMatched
	createdAt := time.Now().UTC()
	preprocessing := map[string]any{
		"resize_strategy":           "preserve_aspect_pad",
		"crop_strategy":             "none",
		"normalization":             "dataset",
		"bbox_mode":                 "ignore",
		"use_dataset_normalization": true,
	}
	frameworkPreprocessing := cloneMap(preprocessing)
	return ExecutionRecord{
		AcceptedSpec: JobExecutionSpec{
			SchemaVersion: "job_execution_spec_v1", CapabilityVersion: "cap-v1", Task: "classification", AcceptedSpecHash: "accepted-hash",
		},
		Attempts: []AttemptExecutionRecord{{
			AttemptID: "attempt-1", AttemptNumber: 1, LifecycleStatus: ExecutionLifecycleFinalized,
			FidelityVerdict: &verdict, RealizedEffectiveHash: "realized-hash", UpdatedAt: createdAt,
			Observations: []RealizationObservation{{
				Stage: ExecutionObservationFinalized, RealizedEffectiveHash: "realized-hash", CreatedAt: createdAt,
				RealizedConfig: map[string]any{"image_size": 256, "preprocessing": preprocessing},
				FrameworkArguments: map[string]any{
					"runtime": map[string]any{"torch": "2.x"},
					"preprocessing": map[string]any{
						"config":             frameworkPreprocessing,
						"normalization_mean": []any{0.1, 0.2, 0.3},
						"normalization_std":  []any{0.9, 0.8, 0.7},
					},
				},
			}},
		}},
	}
}
