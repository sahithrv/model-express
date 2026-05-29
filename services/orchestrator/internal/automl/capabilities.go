package automl

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type HyperparameterCapability struct {
	Name               string              `json:"name"`
	Type               ParameterType       `json:"type"`
	Min                float64             `json:"min,omitempty"`
	Max                float64             `json:"max,omitempty"`
	Scale              SearchScale         `json:"scale,omitempty"`
	Choices            []string            `json:"choices,omitempty"`
	IntChoices         []int               `json:"int_choices,omitempty"`
	AllowedPolicyTypes []string            `json:"allowed_policy_types,omitempty"`
	RequiresEffective  bool                `json:"requires_effective_number_class_balancing,omitempty"`
	RequiresFocalLoss  bool                `json:"requires_focal_loss,omitempty"`
	Condition          *ParameterCondition `json:"condition,omitempty"`
}

type HyperparameterCapabilityRegistry struct {
	capabilities map[string]HyperparameterCapability
}

func DefaultCapabilityRegistry() HyperparameterCapabilityRegistry {
	capabilities := []HyperparameterCapability{
		{Name: "learning_rate", Type: ParameterFloat, Min: 1e-6, Max: 1e-1, Scale: SearchScaleLog},
		{Name: "weight_decay", Type: ParameterFloat, Min: 0, Max: 1, Scale: SearchScaleLinear},
		{Name: "batch_size", Type: ParameterInteger, IntChoices: []int{4, 8, 16, 32, 64, 128}},
		{Name: "epochs", Type: ParameterInteger, Min: 1, Max: 100},
		{Name: "early_stopping_patience", Type: ParameterInteger, Min: 0, Max: 50},
		{Name: "optimizer", Type: ParameterCategorical, Choices: []string{"adamw", "adam", "sgd"}},
		{Name: "scheduler", Type: ParameterCategorical, Choices: []string{"none", "cosine", "step"}},
		{Name: "dropout", Type: ParameterFloat, Min: 0, Max: 0.7, Scale: SearchScaleLinear},
		{Name: "optimizer_momentum", Type: ParameterFloat, Min: 0, Max: 0.99, Scale: SearchScaleLinear, Condition: &ParameterCondition{Field: "optimizer", Equals: "sgd"}},
		{Name: "scheduler_step_size", Type: ParameterInteger, Min: 1, Max: 100, Condition: &ParameterCondition{Field: "scheduler", Equals: "step"}},
		{Name: "scheduler_gamma", Type: ParameterFloat, Min: 0.05, Max: 0.95, Scale: SearchScaleLinear, Condition: &ParameterCondition{Field: "scheduler", Equals: "step"}},
		{Name: "label_smoothing", Type: ParameterFloat, Min: 0, Max: 0.3, Scale: SearchScaleLinear},
		{Name: "gradient_clip_norm", Type: ParameterFloat, Min: 0, Max: 10, Scale: SearchScaleLinear},
		{Name: "augmentation_policy_config.magnitude", Type: ParameterInteger, Min: 0, Max: 15, AllowedPolicyTypes: []string{"randaugment"}},
		{Name: "augmentation_policy_config.num_ops", Type: ParameterInteger, Min: 1, Max: 3, AllowedPolicyTypes: []string{"randaugment"}},
		{Name: "augmentation_policy_config.num_magnitude_bins", Type: ParameterInteger, Min: 2, Max: 31, AllowedPolicyTypes: []string{"trivialaugment", "trivialaugmentwide"}},
		{Name: "augmentation_policy_config.probability", Type: ParameterFloat, Min: 0, Max: 1, AllowedPolicyTypes: []string{"randaugment", "trivialaugment", "trivialaugmentwide", "autoaugment", "mixup", "cutmix"}},
		{Name: "augmentation_policy_config.alpha", Type: ParameterFloat, Min: 0, Max: 1, AllowedPolicyTypes: []string{"mixup", "cutmix"}},
		{Name: "class_balancing_config.effective_number_beta", Type: ParameterFloat, Min: 0.9, Max: 0.99999, RequiresEffective: true},
		{Name: "class_balancing_config.focal_loss_gamma", Type: ParameterFloat, Min: 0.5, Max: 5, RequiresFocalLoss: true},
	}
	out := HyperparameterCapabilityRegistry{capabilities: map[string]HyperparameterCapability{}}
	for _, capability := range capabilities {
		out.capabilities[normalizeParamName(capability.Name)] = capability
	}
	return out
}

func (r HyperparameterCapabilityRegistry) Capabilities() []HyperparameterCapability {
	out := make([]HyperparameterCapability, 0, len(r.capabilities))
	for _, capability := range r.capabilities {
		out = append(out, capability)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

func (r HyperparameterCapabilityRegistry) Capability(name string) (HyperparameterCapability, bool) {
	capability, ok := r.capabilities[normalizeParamName(name)]
	return capability, ok
}

func DefaultSearchSpace(parameterNames []string, strategy StrategyContext) (HyperparameterSearchSpace, error) {
	registry := DefaultCapabilityRegistry()
	parameters := []HyperparameterParameterSpec{}
	for _, name := range parameterNames {
		capability, ok := registry.Capability(name)
		if !ok {
			return HyperparameterSearchSpace{}, fmt.Errorf("unsupported AutoML hyperparameter %q", name)
		}
		spec := HyperparameterParameterSpec{
			Name:      capability.Name,
			Type:      capability.Type,
			Scale:     capability.Scale,
			Choices:   append([]string(nil), capability.Choices...),
			Condition: capability.Condition,
		}
		if len(capability.IntChoices) > 0 {
			spec.IntChoices = append([]int(nil), capability.IntChoices...)
		} else if capability.Type == ParameterFloat || capability.Type == ParameterInteger {
			minValue := capability.Min
			maxValue := capability.Max
			spec.Min = &minValue
			spec.Max = &maxValue
		}
		parameters = append(parameters, spec)
	}
	space := HyperparameterSearchSpace{Parameters: parameters}
	return space, ValidateSearchSpace(space, strategy)
}

func ValidateSearchSpace(space HyperparameterSearchSpace, strategy StrategyContext) error {
	registry := DefaultCapabilityRegistry()
	if len(space.Parameters) == 0 {
		return fmt.Errorf("automl search space must include at least one hyperparameter")
	}
	seen := map[string]bool{}
	for index, spec := range space.Parameters {
		name := normalizeParamName(spec.Name)
		if name == "" {
			return fmt.Errorf("automl search space parameter %d is missing name", index)
		}
		if seen[name] {
			return fmt.Errorf("automl search space duplicates parameter %q", spec.Name)
		}
		seen[name] = true
		capability, ok := registry.Capability(name)
		if !ok {
			return fmt.Errorf("automl cannot tune strategy or unsupported field %q", spec.Name)
		}
		if err := validateCapabilityCondition(capability, strategy, space); err != nil {
			return err
		}
		if spec.Type == "" {
			spec.Type = capability.Type
		}
		if spec.Type != capability.Type {
			return fmt.Errorf("automl parameter %q must have type %q", spec.Name, capability.Type)
		}
		switch capability.Type {
		case ParameterFloat:
			if err := validateNumericSpec(spec, capability); err != nil {
				return err
			}
		case ParameterInteger:
			if err := validateIntegerSpec(spec, capability); err != nil {
				return err
			}
		case ParameterCategorical:
			if err := validateCategoricalSpec(spec, capability); err != nil {
				return err
			}
		default:
			return fmt.Errorf("automl parameter %q has unsupported type %q", spec.Name, spec.Type)
		}
	}
	return nil
}

func ValidateSuggestion(suggestion HyperparameterSuggestion, space HyperparameterSearchSpace, strategy StrategyContext) error {
	if suggestion.Values == nil {
		return fmt.Errorf("automl suggestion values are required")
	}
	specs := map[string]HyperparameterParameterSpec{}
	for _, spec := range space.Parameters {
		specs[normalizeParamName(spec.Name)] = spec
	}
	for rawName, value := range suggestion.Values {
		name := normalizeParamName(rawName)
		spec, ok := specs[name]
		if !ok {
			return fmt.Errorf("automl suggestion includes parameter %q outside validated search space", rawName)
		}
		capability, ok := DefaultCapabilityRegistry().Capability(name)
		if !ok {
			return fmt.Errorf("automl suggestion includes unsupported parameter %q", rawName)
		}
		if err := validateCapabilityCondition(capability, strategy, space); err != nil {
			return err
		}
		if !conditionActive(parameterCondition(spec, capability), strategy, suggestion.Values) {
			return fmt.Errorf("automl suggestion includes inactive conditional parameter %q", rawName)
		}
		switch capability.Type {
		case ParameterFloat:
			number, ok := NumberValue(value)
			if !ok {
				return fmt.Errorf("automl suggestion %q must be numeric", rawName)
			}
			minValue, maxValue := numericBounds(spec, capability)
			if number < minValue || number > maxValue {
				return fmt.Errorf("automl suggestion %q=%g is outside [%g, %g]", rawName, number, minValue, maxValue)
			}
		case ParameterInteger:
			integer, ok := IntValue(value)
			if !ok {
				return fmt.Errorf("automl suggestion %q must be an integer", rawName)
			}
			if len(spec.IntChoices) > 0 {
				if !intIn(integer, spec.IntChoices) {
					return fmt.Errorf("automl suggestion %q=%d is not an allowed choice", rawName, integer)
				}
			} else {
				minValue, maxValue := numericBounds(spec, capability)
				if float64(integer) < minValue || float64(integer) > maxValue {
					return fmt.Errorf("automl suggestion %q=%d is outside [%g, %g]", rawName, integer, minValue, maxValue)
				}
			}
		case ParameterCategorical:
			text := strings.ToLower(strings.TrimSpace(fmt.Sprint(value)))
			if !stringIn(text, normalizedChoices(spec.Choices)) {
				return fmt.Errorf("automl suggestion %q=%q is not an allowed choice", rawName, text)
			}
		}
	}
	for _, spec := range space.Parameters {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if !conditionActive(parameterCondition(spec, capability), strategy, suggestion.Values) {
			continue
		}
		if _, ok := suggestion.Values[spec.Name]; ok {
			continue
		}
		if _, ok := suggestion.Values[normalizeParamName(spec.Name)]; ok {
			continue
		}
		return fmt.Errorf("automl suggestion is missing value for %q", spec.Name)
	}
	return nil
}

func CoversParameter(space *HyperparameterSearchSpace, name string) bool {
	if space == nil {
		return false
	}
	needle := normalizeParamName(name)
	for _, spec := range space.Parameters {
		if normalizeParamName(spec.Name) == needle {
			return true
		}
	}
	return false
}

func NumberValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, !math.IsNaN(typed) && !math.IsInf(typed, 0)
	case float32:
		out := float64(typed)
		return out, !math.IsNaN(out) && !math.IsInf(out, 0)
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	default:
		return 0, false
	}
}

func IntValue(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), float64(int(typed)) == float64(typed)
	case int32:
		return int(typed), true
	case float64:
		if math.Trunc(typed) != typed {
			return 0, false
		}
		return int(typed), true
	case float32:
		if math.Trunc(float64(typed)) != float64(typed) {
			return 0, false
		}
		return int(typed), true
	default:
		return 0, false
	}
}

func NormalizeParameterName(name string) string {
	return normalizeParamName(name)
}

func validateNumericSpec(spec HyperparameterParameterSpec, capability HyperparameterCapability) error {
	minValue, maxValue := numericBounds(spec, capability)
	if minValue < capability.Min || maxValue > capability.Max || minValue > maxValue {
		return fmt.Errorf("automl parameter %q range [%g, %g] must stay inside backend bounds [%g, %g]", spec.Name, minValue, maxValue, capability.Min, capability.Max)
	}
	if spec.Scale != "" && spec.Scale != SearchScaleLinear && spec.Scale != SearchScaleLog {
		return fmt.Errorf("automl parameter %q has unsupported scale %q", spec.Name, spec.Scale)
	}
	if spec.Scale == SearchScaleLog && minValue <= 0 {
		return fmt.Errorf("automl parameter %q uses log scale and must have min > 0", spec.Name)
	}
	return nil
}

func validateIntegerSpec(spec HyperparameterParameterSpec, capability HyperparameterCapability) error {
	if len(spec.IntChoices) > 0 {
		if len(capability.IntChoices) > 0 {
			for _, choice := range spec.IntChoices {
				if !intIn(choice, capability.IntChoices) {
					return fmt.Errorf("automl parameter %q choice %d is outside backend choices", spec.Name, choice)
				}
			}
			return nil
		}
		for _, choice := range spec.IntChoices {
			if float64(choice) < capability.Min || float64(choice) > capability.Max {
				return fmt.Errorf("automl parameter %q choice %d is outside backend bounds [%g, %g]", spec.Name, choice, capability.Min, capability.Max)
			}
		}
		return nil
	}
	minValue, maxValue := numericBounds(spec, capability)
	if math.Trunc(minValue) != minValue || math.Trunc(maxValue) != maxValue {
		return fmt.Errorf("automl integer parameter %q bounds must be integers", spec.Name)
	}
	if minValue < capability.Min || maxValue > capability.Max || minValue > maxValue {
		return fmt.Errorf("automl parameter %q range [%g, %g] must stay inside backend bounds [%g, %g]", spec.Name, minValue, maxValue, capability.Min, capability.Max)
	}
	return nil
}

func validateCategoricalSpec(spec HyperparameterParameterSpec, capability HyperparameterCapability) error {
	if len(spec.Choices) == 0 {
		return fmt.Errorf("automl categorical parameter %q must include choices", spec.Name)
	}
	allowed := normalizedChoices(capability.Choices)
	for _, choice := range spec.Choices {
		normalized := strings.ToLower(strings.TrimSpace(choice))
		if !stringIn(normalized, allowed) {
			return fmt.Errorf("automl parameter %q choice %q is outside backend choices", spec.Name, choice)
		}
	}
	return nil
}

func validateCapabilityCondition(capability HyperparameterCapability, strategy StrategyContext, space HyperparameterSearchSpace) error {
	if len(capability.AllowedPolicyTypes) > 0 {
		policyType := normalizePolicyType(strategy.AugmentationPolicyType)
		if policyType == "" {
			policyType = normalizePolicyType(strategy.AugmentationPolicy)
		}
		if !stringIn(policyType, normalizedChoices(capability.AllowedPolicyTypes)) {
			return fmt.Errorf("automl parameter %q requires augmentation policy type in [%s], got %q", capability.Name, strings.Join(capability.AllowedPolicyTypes, ", "), policyType)
		}
	}
	if capability.RequiresEffective && !isEffectiveNumber(strategy.ClassBalancing) {
		return fmt.Errorf("automl parameter %q requires effective-number class balancing", capability.Name)
	}
	if capability.RequiresFocalLoss && normalizeParamName(strategy.ClassBalancing) != "focal_loss" {
		return fmt.Errorf("automl parameter %q requires focal_loss class balancing", capability.Name)
	}
	if capability.Condition != nil && !conditionCanBecomeActive(*capability.Condition, strategy, space) {
		return fmt.Errorf("automl parameter %q requires %s=%q from the LLM strategy or AutoML search space", capability.Name, capability.Condition.Field, capability.Condition.Equals)
	}
	return nil
}

func numericBounds(spec HyperparameterParameterSpec, capability HyperparameterCapability) (float64, float64) {
	minValue := capability.Min
	maxValue := capability.Max
	if spec.Min != nil {
		minValue = *spec.Min
	}
	if spec.Max != nil {
		maxValue = *spec.Max
	}
	return minValue, maxValue
}

func parameterCondition(spec HyperparameterParameterSpec, capability HyperparameterCapability) *ParameterCondition {
	if spec.Condition != nil {
		return spec.Condition
	}
	return capability.Condition
}

func conditionCanBecomeActive(condition ParameterCondition, strategy StrategyContext, space HyperparameterSearchSpace) bool {
	if conditionMatchesStrategy(condition, strategy) {
		return true
	}
	dependency := normalizeParamName(condition.Field)
	required := normalizeParamName(condition.Equals)
	for _, spec := range space.Parameters {
		if normalizeParamName(spec.Name) != dependency || spec.Type != ParameterCategorical {
			continue
		}
		for _, choice := range spec.Choices {
			if normalizeParamName(choice) == required {
				return true
			}
		}
	}
	return false
}

func conditionActive(condition *ParameterCondition, strategy StrategyContext, values map[string]any) bool {
	if condition == nil {
		return true
	}
	field := normalizeParamName(condition.Field)
	required := normalizeParamName(condition.Equals)
	if values != nil {
		if value, ok := values[field]; ok {
			return normalizeParamName(fmt.Sprint(value)) == required
		}
		for key, value := range values {
			if normalizeParamName(key) == field {
				return normalizeParamName(fmt.Sprint(value)) == required
			}
		}
	}
	return conditionMatchesStrategy(*condition, strategy)
}

func conditionMatchesStrategy(condition ParameterCondition, strategy StrategyContext) bool {
	required := normalizeParamName(condition.Equals)
	switch normalizeParamName(condition.Field) {
	case "optimizer":
		return normalizeParamName(strategy.Optimizer) == required
	case "scheduler":
		return normalizeParamName(strategy.Scheduler) == required
	default:
		return false
	}
}

func normalizeParamName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func normalizePolicyType(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	switch normalized {
	case "rand_augment":
		normalized = "randaugment"
	case "trivial_augment", "trivial_augment_wide":
		normalized = "trivialaugment"
	case "auto_augment":
		normalized = "autoaugment"
	}
	return strings.ReplaceAll(normalized, "_", "")
}

func normalizedChoices(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, normalizePolicyType(value))
	}
	return out
}

func stringIn(value string, choices []string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}

func intIn(value int, choices []int) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}

func isEffectiveNumber(strategy string) bool {
	switch normalizeParamName(strategy) {
	case "effective_number", "effective_number_loss", "effective_number_class_balanced_loss", "class_balanced_effective_number":
		return true
	default:
		return false
	}
}
