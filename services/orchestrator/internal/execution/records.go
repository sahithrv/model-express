package execution

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

const MaxObservationJSONBytes = 64 * 1024

// DeriveRealization compares only the accepted semantic object with the worker's
// allowlisted realization. Framework/evidence payloads are audit context and do
// not participate in semantic identity.
func DeriveRealization(spec JobExecutionSpec, observation RealizationObservationCreate) (map[string]any, string, string, error) {
	realized, err := boundedSemanticConfig(spec.AcceptedSpec, observation.RealizedConfig)
	if err != nil {
		return nil, "", "", err
	}
	hash, err := CanonicalJSONHash(map[string]any{
		"schema_version":     ExecutionObservationSchemaV1,
		"capability_version": spec.CapabilityVersion,
		"task":               spec.Task,
		"runner":             spec.Runner,
		"realized_config":    realized,
	})
	if err != nil {
		return nil, "", "", err
	}
	if observation.Simulated {
		return realized, hash, ExecutionVerdictSimulated, nil
	}
	if reflect.DeepEqual(spec.AcceptedSpec, realized) {
		return realized, hash, ExecutionVerdictMatched, nil
	}
	if approvedBatchAdjustment(spec.AcceptedSpec, realized, observation.AdjustmentPolicy) {
		return realized, hash, ExecutionVerdictApprovedAdjustment, nil
	}
	return realized, hash, ExecutionVerdictMismatch, nil
}

func ValidateObservation(create RealizationObservationCreate) error {
	create.Stage = strings.ToUpper(strings.TrimSpace(create.Stage))
	if create.Stage != ExecutionObservationInitialized && create.Stage != ExecutionObservationFinalized {
		return fmt.Errorf("stage must be INITIALIZED or FINALIZED")
	}
	if strings.TrimSpace(create.IdempotencyKey) == "" || len(create.IdempotencyKey) > 200 {
		return fmt.Errorf("idempotency_key is required and must be at most 200 characters")
	}
	if create.SchemaVersion != "" && create.SchemaVersion != ExecutionObservationSchemaV1 {
		return fmt.Errorf("unsupported observation schema_version %q", create.SchemaVersion)
	}
	for name, value := range map[string]any{
		"realized_config":     create.RealizedConfig,
		"framework_arguments": create.FrameworkArguments,
		"evidence":            valueOrEmpty(create.Evidence),
	} {
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("marshal %s: %w", name, err)
		}
		if len(data) > MaxObservationJSONBytes {
			return fmt.Errorf("%s exceeds %d bytes", name, MaxObservationJSONBytes)
		}
	}
	return nil
}

// RedactSensitiveMap preserves evidence shape while ensuring credentials can
// never enter an execution record. Semantic realized_config is not redacted;
// it is separately restricted to the accepted field allowlist.
func RedactSensitiveMap(value map[string]any) map[string]any {
	redacted, _ := redactSensitiveValue(value).(map[string]any)
	if redacted == nil {
		return map[string]any{}
	}
	return redacted
}

func boundedSemanticConfig(accepted, realized map[string]any) (map[string]any, error) {
	if realized == nil {
		return nil, fmt.Errorf("realized_config is required")
	}
	allowed := leafPaths(accepted, "")
	provided := leafPaths(realized, "")
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, path := range allowed {
		allowedSet[path] = struct{}{}
	}
	for _, path := range provided {
		if _, ok := allowedSet[path]; !ok {
			return nil, fmt.Errorf("realized_config field %q is not in the accepted semantic allowlist", path)
		}
	}
	return cloneCapabilityMap(realized), nil
}

func leafPaths(value map[string]any, prefix string) []string {
	out := []string{}
	for key, item := range value {
		path := key
		if prefix != "" {
			path = prefix + "." + key
		}
		if nested, ok := item.(map[string]any); ok {
			out = append(out, leafPaths(nested, path)...)
		} else {
			out = append(out, path)
		}
	}
	sort.Strings(out)
	return out
}

func approvedBatchAdjustment(accepted, realized map[string]any, policy string) bool {
	if strings.TrimSpace(policy) != "batch_size_recovery" {
		return false
	}
	acceptedBatch, aok := numericValue(accepted["batch_size"])
	realizedBatch, rok := numericValue(realized["batch_size"])
	if !aok || !rok || realizedBatch <= 0 || realizedBatch >= acceptedBatch {
		return false
	}
	copyAccepted := cloneCapabilityMap(accepted)
	copyRealized := cloneCapabilityMap(realized)
	delete(copyAccepted, "batch_size")
	delete(copyRealized, "batch_size")
	return reflect.DeepEqual(copyAccepted, copyRealized)
}

func numericValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func redactSensitiveValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			if sensitiveKey(key) {
				out[key] = "[REDACTED]"
			} else {
				out[key] = redactSensitiveValue(item)
			}
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = redactSensitiveValue(item)
		}
		return out
	default:
		return typed
	}
}

func sensitiveKey(key string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(key, "-", "_"), " ", "_"))
	for _, marker := range []string{"token", "secret", "password", "credential", "api_key", "authorization", "cookie"} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}

func valueOrEmpty(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}
