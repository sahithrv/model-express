package execution

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

const ExportExecutionContractSchemaV1 = "export_execution_contract_v1"

// ExportExecutionContract is the compact, immutable bridge between a finalized
// training execution and every exported inference artifact produced from it.
// Full framework arguments remain solely in the execution record.
type ExportExecutionContract struct {
	SchemaVersion             string         `json:"schema_version"`
	ExecutionRecordRef        string         `json:"execution_record_ref"`
	AttemptID                 string         `json:"attempt_id"`
	Task                      string         `json:"task"`
	CapabilityVersion         string         `json:"capability_version"`
	AcceptedSpecHash          string         `json:"accepted_spec_hash"`
	RealizedEffectiveHash     string         `json:"realized_effective_hash"`
	FidelityVerdict           string         `json:"fidelity_verdict"`
	ImageSize                 int            `json:"image_size"`
	Preprocessing             map[string]any `json:"preprocessing"`
	PreprocessingContractHash string         `json:"preprocessing_contract_hash"`
}

// BuildExportExecutionContract derives inference preprocessing only from the
// final realization receipt. Framework arguments may add effective values such
// as dataset normalization statistics, but may not contradict realized
// semantics.
func BuildExportExecutionContract(record ExecutionRecord, executionRecordRef string) (ExportExecutionContract, error) {
	attempt, ok := latestAttempt(record.Attempts)
	if !ok || attempt.LifecycleStatus != ExecutionLifecycleFinalized {
		return ExportExecutionContract{}, fmt.Errorf("execution has no finalized realization")
	}
	if attempt.FidelityVerdict == nil {
		return ExportExecutionContract{}, fmt.Errorf("finalized execution has no fidelity verdict")
	}
	verdict := strings.ToUpper(strings.TrimSpace(*attempt.FidelityVerdict))
	if verdict != ExecutionVerdictMatched && verdict != ExecutionVerdictApprovedAdjustment {
		return ExportExecutionContract{}, fmt.Errorf("execution verdict %s is not exportable", verdict)
	}
	observation, ok := finalObservation(attempt.Observations)
	if !ok {
		return ExportExecutionContract{}, fmt.Errorf("execution attempt has no final observation")
	}
	if observation.RealizedEffectiveHash == "" || observation.RealizedEffectiveHash != attempt.RealizedEffectiveHash {
		return ExportExecutionContract{}, fmt.Errorf("final observation hash does not match the execution attempt")
	}

	realizedPreprocessing := objectMap(observation.RealizedConfig["preprocessing"])
	if len(realizedPreprocessing) == 0 {
		return ExportExecutionContract{}, fmt.Errorf("final observation has no realized preprocessing")
	}
	frameworkPreprocessing := objectMap(observation.FrameworkArguments["preprocessing"])
	if nested := objectMap(frameworkPreprocessing["config"]); len(nested) > 0 {
		frameworkPreprocessing = nested
	}
	if err := ensurePreprocessingAgreement(realizedPreprocessing, frameworkPreprocessing); err != nil {
		return ExportExecutionContract{}, err
	}

	preprocessing := cloneMap(realizedPreprocessing)
	for key, value := range frameworkPreprocessing {
		preprocessing[key] = value
	}
	frameworkContainer := objectMap(observation.FrameworkArguments["preprocessing"])
	if strings.EqualFold(stringValue(preprocessing["normalization"]), "dataset") {
		metadata := objectMap(preprocessing["normalization_metadata"])
		if len(metadata) == 0 {
			mean, meanOK := numericTriple(frameworkContainer["normalization_mean"])
			std, stdOK := numericTriple(frameworkContainer["normalization_std"])
			if !meanOK || !stdOK || !allPositive(std) {
				return ExportExecutionContract{}, fmt.Errorf("dataset normalization is missing realized mean/std metadata")
			}
			preprocessing["normalization_metadata"] = map[string]any{"mean": mean, "std": std}
		} else {
			mean, meanOK := numericTriple(metadata["mean"])
			std, stdOK := numericTriple(metadata["std"])
			if !meanOK || !stdOK || !allPositive(std) {
				return ExportExecutionContract{}, fmt.Errorf("dataset normalization has invalid realized mean/std metadata")
			}
			preprocessing["normalization_metadata"] = map[string]any{"mean": mean, "std": std}
		}
	}
	if strings.Contains(strings.ToLower(record.AcceptedSpec.Task), "detect") {
		setDefault(preprocessing, "normalization", "none")
		setDefault(preprocessing, "crop_strategy", "none")
		setDefault(preprocessing, "bbox_mode", "ignore")
	}

	imageSize, ok := positiveInt(observation.RealizedConfig["image_size"])
	if !ok {
		return ExportExecutionContract{}, fmt.Errorf("final observation has no valid realized image_size")
	}
	contract := ExportExecutionContract{
		SchemaVersion:         ExportExecutionContractSchemaV1,
		ExecutionRecordRef:    strings.TrimSpace(executionRecordRef),
		AttemptID:             attempt.AttemptID,
		Task:                  record.AcceptedSpec.Task,
		CapabilityVersion:     record.AcceptedSpec.CapabilityVersion,
		AcceptedSpecHash:      record.AcceptedSpec.AcceptedSpecHash,
		RealizedEffectiveHash: attempt.RealizedEffectiveHash,
		FidelityVerdict:       verdict,
		ImageSize:             imageSize,
		Preprocessing:         preprocessing,
	}
	hash, err := CanonicalJSONHash(map[string]any{
		"schema_version":          contract.SchemaVersion,
		"execution_record_ref":    contract.ExecutionRecordRef,
		"attempt_id":              contract.AttemptID,
		"task":                    contract.Task,
		"capability_version":      contract.CapabilityVersion,
		"accepted_spec_hash":      contract.AcceptedSpecHash,
		"realized_effective_hash": contract.RealizedEffectiveHash,
		"image_size":              contract.ImageSize,
		"preprocessing":           contract.Preprocessing,
	})
	if err != nil {
		return ExportExecutionContract{}, fmt.Errorf("hash export preprocessing contract: %w", err)
	}
	contract.PreprocessingContractHash = hash
	return contract, nil
}

func finalObservation(observations []RealizationObservation) (RealizationObservation, bool) {
	finalized := make([]RealizationObservation, 0, len(observations))
	for _, observation := range observations {
		if strings.EqualFold(observation.Stage, ExecutionObservationFinalized) {
			finalized = append(finalized, observation)
		}
	}
	if len(finalized) == 0 {
		return RealizationObservation{}, false
	}
	sort.SliceStable(finalized, func(i, j int) bool { return finalized[i].CreatedAt.After(finalized[j].CreatedAt) })
	return finalized[0], true
}

func ensurePreprocessingAgreement(realized, framework map[string]any) error {
	for _, key := range []string{"resize_strategy", "crop_strategy", "normalization", "bbox_mode", "use_dataset_normalization"} {
		realizedValue, realizedOK := realized[key]
		frameworkValue, frameworkOK := framework[key]
		if realizedOK && frameworkOK && !reflect.DeepEqual(realizedValue, frameworkValue) {
			return fmt.Errorf("realized preprocessing %s contradicts framework arguments", key)
		}
	}
	return nil
}

func objectMap(value any) map[string]any {
	if value == nil {
		return nil
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return nil
}

func cloneMap(value map[string]any) map[string]any {
	out := make(map[string]any, len(value))
	for key, item := range value {
		out[key] = item
	}
	return out
}

func setDefault(values map[string]any, key string, value any) {
	if _, ok := values[key]; !ok {
		values[key] = value
	}
}

func stringValue(value any) string {
	valueString, _ := value.(string)
	return strings.TrimSpace(valueString)
}

func positiveInt(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, typed > 0
	case int32:
		return int(typed), typed > 0
	case int64:
		return int(typed), typed > 0
	case float64:
		converted := int(typed)
		return converted, typed == float64(converted) && converted > 0
	default:
		return 0, false
	}
}

func numericTriple(value any) ([]float64, bool) {
	items, ok := value.([]any)
	if !ok {
		if floats, floatOK := value.([]float64); floatOK && len(floats) == 3 {
			return append([]float64(nil), floats...), true
		}
		return nil, false
	}
	if len(items) != 3 {
		return nil, false
	}
	out := make([]float64, 3)
	for index, item := range items {
		switch typed := item.(type) {
		case float64:
			out[index] = typed
		case int:
			out[index] = float64(typed)
		default:
			return nil, false
		}
	}
	return out, true
}

func allPositive(values []float64) bool {
	for _, value := range values {
		if value <= 0 {
			return false
		}
	}
	return true
}
