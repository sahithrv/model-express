package execution

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

const (
	ExecutionValidationSchemaVersionV1 = "execution_validation_v1"
	ExecutionValidationConfigKey       = "execution_validation_v1"

	ValidationModeShadow  = "shadow"
	ValidationModeEnforce = "enforce"

	FindingSeverityInfo     = "info"
	FindingSeverityWarning  = "warning"
	FindingSeverityBlocking = "blocking"
)

type ExecutionValidationFinding struct {
	Field                string `json:"field"`
	Classification       string `json:"classification"`
	ReasonCode           string `json:"reason_code"`
	Severity             string `json:"severity"`
	WouldBlock           bool   `json:"would_block"`
	Message              string `json:"message"`
	SuggestedAlternative string `json:"suggested_alternative,omitempty"`
	RequestedValue       any    `json:"requested_value,omitempty"`
	AcceptedValue        any    `json:"accepted_value,omitempty"`
}

type AcceptedDuplicateDecision struct {
	Skip           bool     `json:"skip"`
	MatchingJobIDs []string `json:"matching_job_ids,omitempty"`
	DecisionBasis  string   `json:"decision_basis"`
}

type ExecutionValidationReport struct {
	SchemaVersion     string                       `json:"schema_version"`
	Mode              string                       `json:"mode"`
	CapabilityVersion string                       `json:"capability_version"`
	Task              string                       `json:"task"`
	Runner            string                       `json:"runner"`
	ModelFamily       string                       `json:"model_family,omitempty"`
	AcceptedSpecHash  string                       `json:"accepted_spec_hash"`
	WouldBlock        bool                         `json:"would_block"`
	Findings          []ExecutionValidationFinding `json:"findings"`
	AcceptedDuplicate AcceptedDuplicateDecision    `json:"accepted_duplicate"`
}

type PlannerConditionalCapability struct {
	Field      string `json:"field"`
	ReasonCode string `json:"reason_code"`
}

type PlannerCapabilityRule struct {
	Field  string   `json:"field"`
	Values []string `json:"values,omitempty"`
	Range  string   `json:"range,omitempty"`
}

type PlannerCapabilityCard struct {
	SchemaVersion     string                         `json:"schema_version"`
	CapabilityVersion string                         `json:"capability_version"`
	Mode              string                         `json:"validation_mode"`
	Task              string                         `json:"task"`
	Runner            string                         `json:"runner"`
	ModelFamilies     []string                       `json:"model_families,omitempty"`
	ExecutedFields    []string                       `json:"executed_fields"`
	ConditionalFields []PlannerConditionalCapability `json:"conditional_fields"`
	UnsupportedFields []string                       `json:"unsupported_fields"`
	Rules             []PlannerCapabilityRule        `json:"rules,omitempty"`
	FixedSemantics    map[string]any                 `json:"fixed_semantics,omitempty"`
}

type EnforcementFeedback struct {
	Task                 string `json:"task"`
	Runner               string `json:"runner"`
	ModelFamily          string `json:"model_family,omitempty"`
	Field                string `json:"field"`
	Classification       string `json:"classification"`
	ReasonCode           string `json:"reason_code"`
	Count                int    `json:"count"`
	SuggestedAlternative string `json:"suggested_alternative,omitempty"`
}

func NormalizeValidationMode(mode string) string {
	if strings.EqualFold(strings.TrimSpace(mode), ValidationModeShadow) {
		return ValidationModeShadow
	}
	return ValidationModeEnforce
}

// ValidateExecutionSpecV1 reports settings that the selected task/runner cannot
// faithfully execute. The scheduler attaches and applies the accepted-spec-hash
// duplicate decision after validation.
func ValidateExecutionSpecV1(spec ExecutionSpecV1, modelFamily, mode string) (ExecutionValidationReport, error) {
	document, err := parseCapabilitiesV1()
	if err != nil {
		return ExecutionValidationReport{}, err
	}
	profile, err := profileFromDocument(document, spec.Task, spec.Runner)
	if err != nil {
		return ExecutionValidationReport{}, err
	}
	report := ExecutionValidationReport{
		SchemaVersion:     ExecutionValidationSchemaVersionV1,
		Mode:              NormalizeValidationMode(mode),
		CapabilityVersion: document.CapabilityVersion,
		Task:              spec.Task,
		Runner:            spec.Runner,
		ModelFamily:       strings.ToLower(strings.TrimSpace(modelFamily)),
		AcceptedSpecHash:  spec.AcceptedSpecHash,
		Findings:          []ExecutionValidationFinding{},
		AcceptedDuplicate: AcceptedDuplicateDecision{
			DecisionBasis: "accepted_spec_hash_v1",
		},
	}

	paths := leafRequestedCapabilityPaths(document.FieldCatalog, spec.RequestedConfig)
	for _, path := range paths {
		capability, ok := profile.Fields[path]
		if !ok || capability.Classification == "metadata_only" {
			continue
		}
		requested, _ := capabilityValueAtPath(spec.RequestedConfig, path)
		accepted, acceptedOK := capabilityValueAtPath(spec.AcceptedConfig, path)
		fixed, fixedOK := profile.FixedSemantics[path]

		switch capability.Classification {
		case "unsupported":
			if fixedOK && capabilityValuesEqual(requested, fixed) {
				report.Findings = append(report.Findings, ExecutionValidationFinding{
					Field: path, Classification: "normalized", ReasonCode: capability.ReasonCode,
					Severity: FindingSeverityInfo, Message: fmt.Sprintf("%s is fixed by %s and already matches the runner semantic", path, spec.Runner),
					RequestedValue: cloneCapabilityValue(requested), AcceptedValue: cloneCapabilityValue(fixed),
				})
				continue
			}
			report.Findings = append(report.Findings, blockedFinding(
				path, "unsupported", capability.ReasonCode, requested, accepted,
				fmt.Sprintf("omit %s and choose an executed field from the capability card", path),
			))
		case "conditional":
			if !acceptedOK {
				report.Findings = append(report.Findings, blockedFinding(
					path, "normalized_away", capability.ReasonCode, requested, nil,
					conditionalAlternative(path, capability.ReasonCode),
				))
				continue
			}
			report.Findings = append(report.Findings, ExecutionValidationFinding{
				Field: path, Classification: "conditional", ReasonCode: capability.ReasonCode,
				Severity: FindingSeverityWarning, Message: fmt.Sprintf("%s is executable only while its capability condition remains active", path),
				SuggestedAlternative: conditionalAlternative(path, capability.ReasonCode),
				RequestedValue:       cloneCapabilityValue(requested), AcceptedValue: cloneCapabilityValue(accepted),
			})
			if !capabilityValuesEqual(requested, accepted) {
				report.Findings = append(report.Findings, normalizedFinding(spec, path, requested, accepted))
			}
		case "executed":
			if acceptedOK && !capabilityValuesEqual(requested, accepted) {
				report.Findings = append(report.Findings, normalizedFinding(spec, path, requested, accepted))
			}
		}
	}
	appendImplicitClassificationCanonicalFindings(&report, spec, profile)

	for _, finding := range report.Findings {
		if finding.WouldBlock {
			report.WouldBlock = true
			break
		}
	}
	return report, nil
}

func normalizedFinding(spec ExecutionSpecV1, path string, requested, accepted any) ExecutionValidationFinding {
	reasonCode, suggestedAlternative := normalizedFindingMetadata(spec, path)
	return ExecutionValidationFinding{
		Field: path, Classification: "normalized", ReasonCode: reasonCode,
		Severity: FindingSeverityInfo, Message: fmt.Sprintf("%s was normalized to the canonical runner value", path),
		SuggestedAlternative: suggestedAlternative,
		RequestedValue:       cloneCapabilityValue(requested), AcceptedValue: cloneCapabilityValue(accepted),
	}
}

func normalizedFindingMetadata(spec ExecutionSpecV1, path string) (string, string) {
	if spec.Task == "image_classification" && spec.Runner == "modal_torchvision" {
		switch path {
		case "freeze_backbone", "fine_tune_strategy":
			return "classification_transfer_semantics_canonicalized", "use freeze_backbone=false with fine_tune_strategy=full when requesting full fine-tuning or an unfrozen backbone"
		case "preprocessing.normalization":
			if capabilityBool(spec.AcceptedConfig, "preprocessing.use_dataset_normalization") {
				return "classification_dataset_normalization_canonicalized", "set preprocessing.normalization=dataset when preprocessing.use_dataset_normalization=true"
			}
		}
	}
	return "value_normalized", fmt.Sprintf("use the canonical value reported for %s in future proposals", path)
}

func appendImplicitClassificationCanonicalFindings(report *ExecutionValidationReport, spec ExecutionSpecV1, profile CapabilityProfile) {
	if report == nil || spec.Task != "image_classification" || spec.Runner != "modal_torchvision" {
		return
	}
	normalized, err := NormalizeExecutionConfig(spec.Task, spec.Runner, spec.RequestedConfig)
	if err != nil {
		return
	}
	applyCanonicalEmptyDefaults(normalized, profile.Defaults)
	freezeBackbone, freezeOK := capabilityValueAtPath(normalized, "freeze_backbone")
	fineTuneStrategy := strings.ToLower(strings.TrimSpace(capabilityString(normalized, "fine_tune_strategy")))
	if (freezeOK && freezeBackbone == false) || fineTuneStrategy == "full" {
		appendImplicitCanonicalFinding(report, spec, normalized, "freeze_backbone")
		appendImplicitCanonicalFinding(report, spec, normalized, "fine_tune_strategy")
	}
	if capabilityBool(normalized, "preprocessing.use_dataset_normalization") {
		appendImplicitCanonicalFinding(report, spec, normalized, "preprocessing.normalization")
	}
}

func appendImplicitCanonicalFinding(report *ExecutionValidationReport, spec ExecutionSpecV1, normalized map[string]any, path string) {
	if reportHasFinding(*report, path) {
		return
	}
	requested, requestedOK := capabilityValueAtPath(normalized, path)
	accepted, acceptedOK := capabilityValueAtPath(spec.AcceptedConfig, path)
	if !requestedOK || !acceptedOK || capabilityValuesEqual(requested, accepted) {
		return
	}
	report.Findings = append(report.Findings, normalizedFinding(spec, path, requested, accepted))
}

func reportHasFinding(report ExecutionValidationReport, path string) bool {
	for _, finding := range report.Findings {
		if finding.Field == path {
			return true
		}
	}
	return false
}

func capabilityBool(config map[string]any, path string) bool {
	value, ok := capabilityValueAtPath(config, path)
	if !ok {
		return false
	}
	boolValue, _ := value.(bool)
	return boolValue
}
func blockedFinding(path, classification, reasonCode string, requested, accepted any, alternative string) ExecutionValidationFinding {
	return ExecutionValidationFinding{
		Field: path, Classification: classification, ReasonCode: reasonCode,
		Severity: FindingSeverityBlocking, WouldBlock: true,
		Message:              fmt.Sprintf("%s would be ignored or have no executable effect for the selected runner", path),
		SuggestedAlternative: alternative,
		RequestedValue:       cloneCapabilityValue(requested), AcceptedValue: cloneCapabilityValue(accepted),
	}
}

func conditionalAlternative(path, reasonCode string) string {
	switch reasonCode {
	case "conditional_optimizer_sgd":
		return fmt.Sprintf("set optimizer=sgd before using %s, or omit it", path)
	case "conditional_scheduler_step":
		return fmt.Sprintf("set scheduler=step before using %s, or omit it", path)
	case "conditional_class_balancing":
		return fmt.Sprintf("select the matching class-balancing strategy before using %s, or omit it", path)
	case "conditional_augmentation_policy":
		return fmt.Sprintf("select the matching active augmentation policy before using %s, or omit it", path)
	case "conditional_transfer_learning":
		return fmt.Sprintf("select a compatible transfer-learning configuration before using %s, or omit it", path)
	case "conditional_preprocessing":
		return fmt.Sprintf("use a supported preprocessing path with the required dataset metadata before using %s, or omit it", path)
	default:
		return fmt.Sprintf("satisfy the capability prerequisite for %s, or omit it", path)
	}
}

func leafRequestedCapabilityPaths(catalog map[string]FieldDefinition, requested map[string]any) []string {
	paths := []string{}
	for path := range catalog {
		if catalogHasChildFields(catalog, path) {
			continue
		}
		if _, ok := capabilityValueAtPath(requested, path); ok {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	return paths
}

func capabilityValuesEqual(left, right any) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	if leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON) {
		return true
	}
	return reflect.DeepEqual(left, right)
}

func (report *ExecutionValidationReport) SetAcceptedDuplicate(matchingJobIDs []string) {
	if report == nil {
		return
	}
	report.AcceptedDuplicate.MatchingJobIDs = append([]string(nil), matchingJobIDs...)
	sort.Strings(report.AcceptedDuplicate.MatchingJobIDs)
	report.AcceptedDuplicate.Skip = len(report.AcceptedDuplicate.MatchingJobIDs) > 0
}

func BuildPlannerCapabilityCard(task, runner, mode string, modelFamilies []string) (PlannerCapabilityCard, error) {
	document, err := parseCapabilitiesV1()
	if err != nil {
		return PlannerCapabilityCard{}, err
	}
	profile, err := profileFromDocument(document, task, runner)
	if err != nil {
		return PlannerCapabilityCard{}, err
	}
	card := PlannerCapabilityCard{
		SchemaVersion: ExecutionValidationSchemaVersionV1, CapabilityVersion: document.CapabilityVersion,
		Mode: NormalizeValidationMode(mode), Task: task, Runner: runner,
		ModelFamilies: uniqueSortedStrings(modelFamilies), FixedSemantics: cloneCapabilityMap(profile.FixedSemantics),
		ExecutedFields: []string{}, ConditionalFields: []PlannerConditionalCapability{}, UnsupportedFields: []string{},
	}
	paths := make([]string, 0, len(profile.Fields))
	for path := range profile.Fields {
		if !catalogHasChildFields(document.FieldCatalog, path) {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	for _, path := range paths {
		capability := profile.Fields[path]
		switch capability.Classification {
		case "executed":
			card.ExecutedFields = append(card.ExecutedFields, path)
		case "conditional":
			card.ConditionalFields = append(card.ConditionalFields, PlannerConditionalCapability{Field: path, ReasonCode: capability.ReasonCode})
		case "unsupported":
			card.UnsupportedFields = append(card.UnsupportedFields, path)
		}
		if capability.Classification != "executed" && capability.Classification != "conditional" {
			continue
		}
		definition := document.FieldCatalog[path]
		constraint := profile.Constraints[path]
		values := definition.Values
		if len(constraint.Values) > 0 {
			values = constraint.Values
		}
		rule := PlannerCapabilityRule{Field: path, Values: append([]string(nil), values...)}
		numericRange := definition.Range
		if constraint.Range != nil {
			numericRange = constraint.Range
		}
		if numericRange != nil {
			rule.Range = formatCapabilityRange(numericRange)
		}
		if len(rule.Values) > 0 || rule.Range != "" {
			card.Rules = append(card.Rules, rule)
		}
	}
	return card, nil
}

func formatCapabilityRange(value *NumericRange) string {
	if value == nil {
		return ""
	}
	parts := []string{}
	if value.Min != nil {
		parts = append(parts, fmt.Sprintf("min=%g", *value.Min))
	}
	if value.ExclusiveMin != nil {
		parts = append(parts, fmt.Sprintf("exclusive_min=%g", *value.ExclusiveMin))
	}
	if value.Max != nil {
		parts = append(parts, fmt.Sprintf("max=%g", *value.Max))
	}
	return strings.Join(parts, ",")
}

func SummarizeEnforcementFeedback(reports []ExecutionValidationReport, task, runner string, limit int) []EnforcementFeedback {
	if limit <= 0 {
		limit = 12
	}
	byKey := map[string]EnforcementFeedback{}
	for _, report := range reports {
		if report.Task != task || report.Runner != runner {
			continue
		}
		for _, finding := range report.Findings {
			if !finding.WouldBlock {
				continue
			}
			key := strings.Join([]string{report.Task, report.Runner, report.ModelFamily, finding.Field, finding.ReasonCode}, "|")
			item := byKey[key]
			item.Task, item.Runner, item.ModelFamily = report.Task, report.Runner, report.ModelFamily
			item.Field, item.Classification, item.ReasonCode = finding.Field, finding.Classification, finding.ReasonCode
			item.SuggestedAlternative = finding.SuggestedAlternative
			item.Count++
			byKey[key] = item
		}
	}
	out := make([]EnforcementFeedback, 0, len(byKey))
	for _, item := range byKey {
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Field < out[j].Field
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func uniqueSortedStrings(values []string) []string {
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value != "" {
			seen[value] = true
		}
	}
	out := make([]string, 0, len(seen))
	for value := range seen {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
