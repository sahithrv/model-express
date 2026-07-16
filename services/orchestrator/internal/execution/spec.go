package execution

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	ExecutionSpecSchemaVersionV1         = "execution_spec_v1"
	ExecutionSpecConfigKey               = "execution_spec_v1"
	ExecutionSpecStatusVersioned         = "versioned"
	ExecutionSpecStatusLegacyUnversioned = "legacy_unversioned"
)

type ExecutionSpecV1 struct {
	SchemaVersion       string         `json:"schema_version"`
	CapabilityVersion   string         `json:"capability_version"`
	Task                string         `json:"task"`
	Runner              string         `json:"runner"`
	RequestedConfig     map[string]any `json:"requested_config"`
	AcceptedConfig      map[string]any `json:"accepted_config"`
	ArtifactPlan        ArtifactPlanV1 `json:"artifact_plan"`
	RequestedConfigHash string         `json:"requested_config_hash"`
	AcceptedSpecHash    string         `json:"accepted_spec_hash"`
}

func BuildExecutionSpecV1(
	task string,
	runner string,
	requestedConfig map[string]any,
	resolutionInput map[string]any,
) (ExecutionSpecV1, error) {
	plan, err := BuildAutomaticArtifactPlanV1(task, runner, ArtifactPolicySelection{})
	if err != nil {
		return ExecutionSpecV1{}, err
	}
	return BuildExecutionSpecV1WithArtifactPlan(task, runner, requestedConfig, resolutionInput, plan)
}

func BuildExecutionSpecV1WithArtifactPlan(
	task string,
	runner string,
	requestedConfig map[string]any,
	resolutionInput map[string]any,
	artifactPlan ArtifactPlanV1,
) (ExecutionSpecV1, error) {
	if requestedConfig == nil {
		requestedConfig = map[string]any{}
	}
	if resolutionInput == nil {
		resolutionInput = map[string]any{}
	}
	document, err := parseCapabilitiesV1()
	if err != nil {
		return ExecutionSpecV1{}, err
	}
	acceptedConfig, err := ResolveAcceptedConfig(task, runner, resolutionInput)
	if err != nil {
		return ExecutionSpecV1{}, err
	}
	if err := ValidateArtifactPlanV1(artifactPlan, task, runner); err != nil {
		return ExecutionSpecV1{}, fmt.Errorf("validate artifact plan: %w", err)
	}
	requestedHash, err := CanonicalJSONHash(requestedConfig)
	if err != nil {
		return ExecutionSpecV1{}, fmt.Errorf("hash requested execution config: %w", err)
	}
	acceptedHash, err := CanonicalJSONHash(map[string]any{
		"schema_version":     ExecutionSpecSchemaVersionV1,
		"capability_version": document.CapabilityVersion,
		"task":               task,
		"runner":             runner,
		"accepted_config":    acceptedConfig,
	})
	if err != nil {
		return ExecutionSpecV1{}, fmt.Errorf("hash accepted execution spec: %w", err)
	}
	return ExecutionSpecV1{
		SchemaVersion:       ExecutionSpecSchemaVersionV1,
		CapabilityVersion:   document.CapabilityVersion,
		Task:                task,
		Runner:              runner,
		RequestedConfig:     cloneCapabilityMap(requestedConfig),
		AcceptedConfig:      acceptedConfig,
		ArtifactPlan:        artifactPlan,
		RequestedConfigHash: requestedHash,
		AcceptedSpecHash:    acceptedHash,
	}, nil
}

func (spec ExecutionSpecV1) Payload() (map[string]any, error) {
	data, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("marshal execution spec: %w", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("decode execution spec payload: %w", err)
	}
	return payload, nil
}

func ResolveAcceptedConfig(task, runner string, input map[string]any) (map[string]any, error) {
	document, err := parseCapabilitiesV1()
	if err != nil {
		return nil, err
	}
	profile, err := profileFromDocument(document, task, runner)
	if err != nil {
		return nil, err
	}
	normalized, err := NormalizeExecutionConfig(task, runner, input)
	if err != nil {
		return nil, err
	}
	applyCanonicalEmptyDefaults(normalized, profile.Defaults)
	accepted := map[string]any{}
	for path, capability := range profile.Fields {
		if capability.Classification != "executed" && capability.Classification != "conditional" {
			continue
		}
		definition := document.FieldCatalog[path]
		if definition.Type == "object" && catalogHasChildFields(document.FieldCatalog, path) {
			continue
		}
		value, ok := capabilityValueAtPath(normalized, path)
		if !ok || !conditionalCapabilityActive(path, capability, normalized) {
			continue
		}
		setCapabilityValueAtPath(accepted, path, cloneCapabilityValue(value))
	}
	for path, value := range profile.FixedSemantics {
		setCapabilityValueAtPath(accepted, path, cloneCapabilityValue(value))
	}
	canonicalizeAcceptedConfig(task, runner, accepted)
	return accepted, nil
}

func canonicalizeAcceptedConfig(task, runner string, accepted map[string]any) {
	if task != "image_classification" || runner != "modal_torchvision" || accepted == nil {
		return
	}
	freezeBackbone, freezeOK := capabilityValueAtPath(accepted, "freeze_backbone")
	fineTuneStrategy, strategyOK := capabilityValueAtPath(accepted, "fine_tune_strategy")
	if (freezeOK && freezeBackbone == false) || (strategyOK && capabilityText(fineTuneStrategy) == "full") {
		setCapabilityValueAtPath(accepted, "freeze_backbone", false)
		setCapabilityValueAtPath(accepted, "fine_tune_strategy", "full")
	}
	useDatasetNormalization, useDatasetOK := capabilityValueAtPath(accepted, "preprocessing.use_dataset_normalization")
	if useDatasetOK && useDatasetNormalization == true {
		setCapabilityValueAtPath(accepted, "preprocessing.normalization", "dataset")
	}
}

func capabilityText(value any) string {
	text, _ := value.(string)
	return strings.ToLower(strings.TrimSpace(text))
}

func applyCanonicalEmptyDefaults(config map[string]any, defaults map[string]any) {
	for path, defaultValue := range defaults {
		textDefault, isStringDefault := defaultValue.(string)
		if !isStringDefault {
			continue
		}
		value, ok := capabilityValueAtPath(config, path)
		if !ok {
			continue
		}
		text, isString := value.(string)
		if isString && strings.TrimSpace(text) == "" {
			setCapabilityValueAtPath(config, path, textDefault)
		}
	}
}

func CanonicalJSONHash(value any) (string, error) {
	canonical, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return "sha256:" + base64.RawURLEncoding.EncodeToString(sum[:]), nil
}

func catalogHasChildFields(catalog map[string]FieldDefinition, path string) bool {
	prefix := path + "."
	for candidate := range catalog {
		if strings.HasPrefix(candidate, prefix) {
			return true
		}
	}
	return false
}

func conditionalCapabilityActive(
	path string,
	capability FieldCapability,
	normalized map[string]any,
) bool {
	if capability.Classification != "conditional" {
		return true
	}
	switch capability.ReasonCode {
	case "conditional_optimizer_sgd":
		return capabilityString(normalized, "optimizer") == "sgd"
	case "conditional_scheduler_step":
		return capabilityString(normalized, "scheduler") == "step"
	case "conditional_class_balancing":
		strategy := capabilityString(normalized, "class_balancing")
		switch path {
		case "class_balancing", "sampling_strategy":
			return true
		case "class_balancing_config.effective_number_beta":
			return strategy == "effective_number_loss"
		case "class_balancing_config.focal_loss_gamma":
			return strategy == "focal_loss"
		default:
			return true
		}
	case "conditional_augmentation_policy":
		policy := capabilityString(normalized, "augmentation_policy_config.policy_type")
		if policy == "" {
			policy = capabilityString(normalized, "augmentation_policy")
		}
		switch path {
		case "augmentation_policy":
			return true
		case "augmentation_policy_config.alpha":
			return policy == "mixup" || policy == "cutmix"
		case "augmentation_policy_config.magnitude", "augmentation_policy_config.num_ops":
			return policy == "randaugment"
		case "augmentation_policy_config.num_magnitude_bins":
			return policy == "trivialaugment"
		case "augmentation_policy_config.probability":
			return policy != "" && policy != "none" && policy != "custom"
		default:
			return true
		}
	default:
		return true
	}
}

func capabilityString(config map[string]any, path string) string {
	value, ok := capabilityValueAtPath(config, path)
	if !ok {
		return ""
	}
	text, _ := value.(string)
	return text
}

func cloneCapabilityMap(values map[string]any) map[string]any {
	if values == nil {
		return map[string]any{}
	}
	cloned, _ := cloneCapabilityValue(values).(map[string]any)
	return cloned
}
