package execution

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

const CapabilitySchemaVersionV1 = "experiment_execution_capabilities.v1"

type CapabilityDocument struct {
	SchemaVersion     string                       `json:"schema_version"`
	CapabilityVersion string                       `json:"capability_version"`
	Classifications   []string                     `json:"classifications"`
	ReasonCodes       map[string]string            `json:"reason_codes"`
	Runners           map[string]RunnerDefinition  `json:"runners"`
	FieldCatalog      map[string]FieldDefinition   `json:"field_catalog"`
	Profiles          map[string]CapabilityProfile `json:"profiles"`
	Tasks             map[string]TaskCapability    `json:"tasks"`
}

type RunnerDefinition struct {
	Providers []string `json:"providers"`
	ModelKind string   `json:"model_kind,omitempty"`
	Simulated bool     `json:"simulated"`
}

type FieldDefinition struct {
	Type            string            `json:"type"`
	Normalization   string            `json:"normalization,omitempty"`
	CatalogCategory string            `json:"catalog_category,omitempty"`
	CatalogID       string            `json:"catalog_id,omitempty"`
	Values          []string          `json:"values,omitempty"`
	Aliases         map[string]string `json:"aliases,omitempty"`
	Range           *NumericRange     `json:"range,omitempty"`
	Required        bool              `json:"required,omitempty"`
}

type NumericRange struct {
	Min          *float64 `json:"min,omitempty"`
	ExclusiveMin *float64 `json:"exclusive_min,omitempty"`
	Max          *float64 `json:"max,omitempty"`
}

type CapabilityProfile struct {
	Task               string                     `json:"task"`
	Runner             string                     `json:"runner"`
	Defaults           map[string]any             `json:"defaults,omitempty"`
	DefaultExpressions map[string]string          `json:"default_expressions,omitempty"`
	Constraints        map[string]FieldConstraint `json:"constraints,omitempty"`
	FixedSemantics     map[string]any             `json:"fixed_semantics,omitempty"`
	Fields             map[string]FieldCapability `json:"fields"`
}

type FieldConstraint struct {
	Values  []string          `json:"values,omitempty"`
	Aliases map[string]string `json:"aliases,omitempty"`
	Range   *NumericRange     `json:"range,omitempty"`
}

type FieldCapability struct {
	Classification string `json:"classification"`
	ReasonCode     string `json:"reason_code"`
}

type TaskCapability struct {
	Runners map[string]RunnerSelection `json:"runners"`
}

type RunnerSelection struct {
	Profile string `json:"profile"`
}

func CapabilitiesV1() CapabilityDocument {
	document, err := parseCapabilitiesV1()
	if err != nil {
		panic(err)
	}
	return document
}

func CapabilityProfileFor(task, runner string) (CapabilityProfile, error) {
	document, err := parseCapabilitiesV1()
	if err != nil {
		return CapabilityProfile{}, err
	}
	taskCapability, ok := document.Tasks[task]
	if !ok {
		return CapabilityProfile{}, fmt.Errorf("unknown execution capability task %q", task)
	}
	selection, ok := taskCapability.Runners[runner]
	if !ok {
		return CapabilityProfile{}, fmt.Errorf(
			"unknown execution capability runner %q for task %q",
			runner,
			task,
		)
	}
	profile, ok := document.Profiles[selection.Profile]
	if !ok {
		return CapabilityProfile{}, fmt.Errorf(
			"execution capability profile %q is not embedded",
			selection.Profile,
		)
	}
	return profile, nil
}

// NormalizeExecutionConfig applies only contract defaults and aliases. It is not
// wired into scheduling in Fidelity PR 1, so introducing the contract cannot
// change existing job or worker behavior.
func NormalizeExecutionConfig(task, runner string, input map[string]any) (map[string]any, error) {
	document, err := parseCapabilitiesV1()
	if err != nil {
		return nil, err
	}
	profile, err := profileFromDocument(document, task, runner)
	if err != nil {
		return nil, err
	}

	rootFields := map[string]bool{}
	for path := range document.FieldCatalog {
		rootFields[strings.Split(path, ".")[0]] = true
	}
	out := map[string]any{}
	for key, value := range input {
		if rootFields[key] {
			out[key] = cloneCapabilityValue(value)
		}
	}

	for path, value := range profile.Defaults {
		if _, ok := capabilityValueAtPath(out, path); !ok {
			setCapabilityValueAtPath(out, path, cloneCapabilityValue(value))
		}
	}
	for path, expression := range profile.DefaultExpressions {
		if _, ok := capabilityValueAtPath(out, path); ok {
			continue
		}
		value, err := evaluateCapabilityDefault(expression, out)
		if err != nil {
			return nil, fmt.Errorf("default %s: %w", path, err)
		}
		setCapabilityValueAtPath(out, path, value)
	}

	paths := make([]string, 0, len(document.FieldCatalog))
	for path := range document.FieldCatalog {
		paths = append(paths, path)
	}
	sort.Slice(paths, func(i, j int) bool {
		return strings.Count(paths[i], ".") < strings.Count(paths[j], ".") ||
			(strings.Count(paths[i], ".") == strings.Count(paths[j], ".") && paths[i] < paths[j])
	})
	for _, path := range paths {
		value, ok := capabilityValueAtPath(out, path)
		if !ok {
			continue
		}
		definition := document.FieldCatalog[path]
		constraint := profile.Constraints[path]
		normalized, err := normalizeCapabilityValue(path, value, definition, constraint)
		if err != nil {
			return nil, err
		}
		setCapabilityValueAtPath(out, path, normalized)
	}
	return out, nil
}

func GeneratedCapabilitiesJSONV1() string {
	return generatedExecutionCapabilitiesJSON
}

func parseCapabilitiesV1() (CapabilityDocument, error) {
	var document CapabilityDocument
	if err := json.Unmarshal([]byte(generatedExecutionCapabilitiesJSON), &document); err != nil {
		return CapabilityDocument{}, fmt.Errorf("decode embedded execution capabilities: %w", err)
	}
	if err := validateCapabilityDocument(document); err != nil {
		return CapabilityDocument{}, fmt.Errorf("validate embedded execution capabilities: %w", err)
	}
	return document, nil
}

func validateCapabilityDocument(document CapabilityDocument) error {
	if document.SchemaVersion != CapabilitySchemaVersionV1 {
		return fmt.Errorf("unsupported schema version %q", document.SchemaVersion)
	}
	if strings.TrimSpace(document.CapabilityVersion) == "" {
		return fmt.Errorf("capability_version is required")
	}
	classifications := stringSet(document.Classifications)
	expectedClassifications := stringSet([]string{"executed", "conditional", "metadata_only", "unsupported"})
	if !sameStringSet(classifications, expectedClassifications) {
		return fmt.Errorf("classifications must be executed, conditional, metadata_only, and unsupported")
	}
	if len(document.FieldCatalog) == 0 || len(document.Profiles) == 0 || len(document.Tasks) == 0 {
		return fmt.Errorf("field catalog, profiles, and tasks are required")
	}

	catalogFields := map[string]bool{}
	for path, definition := range document.FieldCatalog {
		if strings.TrimSpace(path) == "" || strings.TrimSpace(definition.Type) == "" {
			return fmt.Errorf("field catalog entry %q must declare a type", path)
		}
		catalogFields[path] = true
		values := stringSet(definition.Values)
		for alias, target := range definition.Aliases {
			if len(values) > 0 && !values[target] {
				return fmt.Errorf("field %q alias %q targets unknown value %q", path, alias, target)
			}
		}
	}

	for name, profile := range document.Profiles {
		if _, ok := document.Tasks[profile.Task]; !ok {
			return fmt.Errorf("profile %q references unknown task %q", name, profile.Task)
		}
		if _, ok := document.Runners[profile.Runner]; !ok {
			return fmt.Errorf("profile %q references unknown runner %q", name, profile.Runner)
		}
		if len(profile.Fields) != len(catalogFields) {
			return fmt.Errorf("profile %q does not classify every catalog field", name)
		}
		for path := range catalogFields {
			capability, ok := profile.Fields[path]
			if !ok {
				return fmt.Errorf("profile %q does not classify field %q", name, path)
			}
			if !classifications[capability.Classification] {
				return fmt.Errorf(
					"profile %q field %q uses unknown classification %q",
					name,
					path,
					capability.Classification,
				)
			}
			if _, ok := document.ReasonCodes[capability.ReasonCode]; !ok {
				return fmt.Errorf(
					"profile %q field %q uses unknown reason code %q",
					name,
					path,
					capability.ReasonCode,
				)
			}
		}
		for _, fields := range []map[string]any{profile.Defaults, profile.FixedSemantics} {
			for path := range fields {
				if !catalogFields[path] {
					return fmt.Errorf("profile %q references unknown field %q", name, path)
				}
			}
		}
		for path, constraint := range profile.Constraints {
			definition, ok := document.FieldCatalog[path]
			if !ok {
				return fmt.Errorf("profile %q constrains unknown field %q", name, path)
			}
			values := definition.Values
			if len(constraint.Values) > 0 {
				values = constraint.Values
			}
			allowedValues := stringSet(values)
			for alias, target := range constraint.Aliases {
				if len(allowedValues) > 0 && !allowedValues[target] {
					return fmt.Errorf(
						"profile %q field %q alias %q targets unknown value %q",
						name,
						path,
						alias,
						target,
					)
				}
			}
		}
		for path, expression := range profile.DefaultExpressions {
			if !catalogFields[path] {
				return fmt.Errorf("profile %q references unknown default field %q", name, path)
			}
			if expression != "max(1, floor(epochs / 3))" {
				return fmt.Errorf("profile %q uses unsupported default expression %q", name, expression)
			}
		}
	}

	referencedProfiles := map[string]bool{}
	for taskName, task := range document.Tasks {
		for runnerName, selection := range task.Runners {
			if _, ok := document.Runners[runnerName]; !ok {
				return fmt.Errorf("task %q references unknown runner %q", taskName, runnerName)
			}
			profile, ok := document.Profiles[selection.Profile]
			if !ok {
				return fmt.Errorf("task %q references unknown profile %q", taskName, selection.Profile)
			}
			if profile.Task != taskName || profile.Runner != runnerName {
				return fmt.Errorf("task %q runner %q selects mismatched profile", taskName, runnerName)
			}
			referencedProfiles[selection.Profile] = true
		}
	}
	if len(referencedProfiles) != len(document.Profiles) {
		return fmt.Errorf("every capability profile must be selected by a task")
	}
	return nil
}

func profileFromDocument(
	document CapabilityDocument,
	task string,
	runner string,
) (CapabilityProfile, error) {
	taskCapability, ok := document.Tasks[task]
	if !ok {
		return CapabilityProfile{}, fmt.Errorf("unknown execution capability task %q", task)
	}
	selection, ok := taskCapability.Runners[runner]
	if !ok {
		return CapabilityProfile{}, fmt.Errorf(
			"unknown execution capability runner %q for task %q",
			runner,
			task,
		)
	}
	return document.Profiles[selection.Profile], nil
}

func normalizeCapabilityValue(
	path string,
	value any,
	definition FieldDefinition,
	constraint FieldConstraint,
) (any, error) {
	switch definition.Type {
	case "string":
		text, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("execution capability field %q must be a string", path)
		}
		switch definition.Normalization {
		case "trim":
			text = strings.TrimSpace(text)
		case "lowercase_trim":
			text = strings.ToLower(strings.TrimSpace(text))
		}
		if alias, ok := definition.Aliases[text]; ok {
			text = alias
		}
		if alias, ok := constraint.Aliases[text]; ok {
			text = alias
		}
		return text, nil
	case "string_array":
		values, ok := value.([]any)
		if !ok {
			if stringsValue, stringsOK := value.([]string); stringsOK {
				out := make([]string, 0, len(stringsValue))
				for _, item := range stringsValue {
					if item = strings.TrimSpace(item); item != "" {
						out = append(out, item)
					}
				}
				return out, nil
			}
			return nil, fmt.Errorf("execution capability field %q must be a string array", path)
		}
		out := make([]string, 0, len(values))
		for _, value := range values {
			item, ok := value.(string)
			if !ok {
				return nil, fmt.Errorf("execution capability field %q must be a string array", path)
			}
			if item = strings.TrimSpace(item); item != "" {
				out = append(out, item)
			}
		}
		return out, nil
	default:
		return cloneCapabilityValue(value), nil
	}
}

func evaluateCapabilityDefault(expression string, config map[string]any) (any, error) {
	if expression != "max(1, floor(epochs / 3))" {
		return nil, fmt.Errorf("unsupported expression %q", expression)
	}
	epochs, ok := capabilityValueAtPath(config, "epochs")
	if !ok {
		return nil, fmt.Errorf("epochs is required")
	}
	numericEpochs, ok := capabilityInteger(epochs)
	if !ok {
		return nil, fmt.Errorf("epochs must be an integer")
	}
	return max(1, int(math.Floor(float64(numericEpochs)/3))), nil
}

func capabilityInteger(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		if math.Trunc(typed) == typed {
			return int(typed), true
		}
	}
	return 0, false
}

func capabilityValueAtPath(config map[string]any, path string) (any, bool) {
	parts := strings.Split(path, ".")
	current := config
	for index, part := range parts {
		value, ok := current[part]
		if !ok {
			return nil, false
		}
		if index == len(parts)-1 {
			return value, true
		}
		nested, ok := value.(map[string]any)
		if !ok {
			return nil, false
		}
		current = nested
	}
	return nil, false
}

func setCapabilityValueAtPath(config map[string]any, path string, value any) {
	parts := strings.Split(path, ".")
	current := config
	for _, part := range parts[:len(parts)-1] {
		nested, ok := current[part].(map[string]any)
		if !ok {
			nested = map[string]any{}
			current[part] = nested
		}
		current = nested
	}
	current[parts[len(parts)-1]] = value
}

func cloneCapabilityValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = cloneCapabilityValue(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = cloneCapabilityValue(item)
		}
		return out
	case []string:
		return append([]string(nil), typed...)
	default:
		return value
	}
}

func stringSet(values []string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		out[value] = true
	}
	return out
}

func sameStringSet(left, right map[string]bool) bool {
	if len(left) != len(right) {
		return false
	}
	for value := range left {
		if !right[value] {
			return false
		}
	}
	return true
}
