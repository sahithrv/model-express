package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func (s *Server) championExportExecutionContract(jobID string) (map[string]any, error) {
	record, err := s.store.GetJobExecutionRecord(jobID)
	if errors.Is(err, store.ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	contract, err := execution.BuildExportExecutionContract(record, "/jobs/"+strings.TrimSpace(jobID)+"/execution-record")
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(contract)
	if err != nil {
		return nil, fmt.Errorf("marshal export execution contract: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(payload, &result); err != nil {
		return nil, fmt.Errorf("decode export execution contract: %w", err)
	}
	return result, nil
}

func validateChampionExportExecutionContract(jobContract map[string]any, manifest map[string]any) error {
	if len(jobContract) == 0 {
		return nil
	}
	metadata := payloadMap(manifest, "metadata")
	manifestContract := payloadMap(metadata, "execution_contract")
	if len(manifestContract) == 0 {
		return fmt.Errorf("worker export manifest is missing the finalized execution contract")
	}
	for _, key := range []string{
		"schema_version",
		"execution_record_ref",
		"attempt_id",
		"task",
		"capability_version",
		"accepted_spec_hash",
		"realized_effective_hash",
		"fidelity_verdict",
		"preprocessing_contract_hash",
	} {
		if payloadString(manifestContract, key) == "" || payloadString(manifestContract, key) != payloadString(jobContract, key) {
			return fmt.Errorf("worker export manifest execution contract has mismatched %s", key)
		}
	}
	expectedImageSize := int(payloadFloat(jobContract, "image_size"))
	if expectedImageSize <= 0 || manifestExportImageSize(metadata) != expectedImageSize {
		return fmt.Errorf("worker export manifest image size contradicts the finalized execution")
	}
	expectedPreprocessing := payloadMap(jobContract, "preprocessing")
	manifestPreprocessing := payloadMap(payloadMap(metadata, "preprocessing_contract"), "config")
	if len(manifestPreprocessing) == 0 {
		return fmt.Errorf("worker export manifest is missing its preprocessing contract")
	}
	for _, key := range []string{"resize_strategy", "crop_strategy", "normalization", "bbox_mode", "use_dataset_normalization", "normalization_metadata"} {
		expected, expectedOK := expectedPreprocessing[key]
		if !expectedOK {
			continue
		}
		actual, actualOK := manifestPreprocessing[key]
		if !actualOK || !canonicalValuesEqual(expected, actual) {
			return fmt.Errorf("worker export preprocessing contradicts finalized %s", key)
		}
	}
	return nil
}

func manifestExportImageSize(metadata map[string]any) int {
	shapes := []any{metadata["input_shape"]}
	inferenceInput := payloadMap(payloadMap(metadata, "inference_contract"), "input")
	shapes = append(shapes, inferenceInput["model_tensor_shape"])
	for _, shape := range shapes {
		switch values := shape.(type) {
		case []any:
			if len(values) == 4 {
				if value, ok := values[3].(float64); ok {
					return int(value)
				}
			}
		case []int:
			if len(values) == 4 {
				return values[3]
			}
		}
	}
	return 0
}

func canonicalValuesEqual(left, right any) bool {
	leftHash, leftErr := execution.CanonicalJSONHash(left)
	rightHash, rightErr := execution.CanonicalJSONHash(right)
	return leftErr == nil && rightErr == nil && leftHash == rightHash
}

func compactChampionExportMetadata(champion runs.ProjectChampion, format string, requestMetadata map[string]any) map[string]any {
	metadata := map[string]any{
		"format":           format,
		"source_job_id":    champion.JobID,
		"selection_reason": champion.SelectionReason,
	}
	for key, value := range requestMetadata {
		metadata[key] = value
	}
	return metadata
}
