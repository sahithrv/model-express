package automl

import (
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/execution"
)

type ExecutionScope struct {
	CapabilityVersion string `json:"capability_version"`
	Task              string `json:"task"`
	Runner            string `json:"runner"`
}

func CurrentExecutionScope(task, runner string) (ExecutionScope, error) {
	document := execution.CapabilitiesV1()
	if _, err := execution.CapabilityProfileFor(task, runner); err != nil {
		return ExecutionScope{}, err
	}
	return ExecutionScope{
		CapabilityVersion: document.CapabilityVersion,
		Task:              strings.TrimSpace(task),
		Runner:            strings.TrimSpace(runner),
	}, nil
}

func CapabilityRegistryForExecution(scope ExecutionScope) (HyperparameterCapabilityRegistry, error) {
	document := execution.CapabilitiesV1()
	profile, err := execution.CapabilityProfileFor(scope.Task, scope.Runner)
	if err != nil {
		return HyperparameterCapabilityRegistry{}, err
	}
	if scope.CapabilityVersion != "" && scope.CapabilityVersion != document.CapabilityVersion {
		return HyperparameterCapabilityRegistry{}, fmt.Errorf(
			"automl capability version %q does not match embedded execution capability version %q",
			scope.CapabilityVersion,
			document.CapabilityVersion,
		)
	}
	out := HyperparameterCapabilityRegistry{capabilities: map[string]HyperparameterCapability{}}
	for _, capability := range DefaultCapabilityRegistry().Capabilities() {
		field, ok := profile.Fields[capability.Name]
		if !ok || (field.Classification != "executed" && field.Classification != "conditional") {
			continue
		}
		definition := document.FieldCatalog[capability.Name]
		constraint := profile.Constraints[capability.Name]
		scoped := capability
		if len(scoped.IntChoices) > 0 && scoped.Min == 0 && scoped.Max == 0 {
			scoped.Min = float64(scoped.IntChoices[0])
			scoped.Max = float64(scoped.IntChoices[0])
			for _, choice := range scoped.IntChoices[1:] {
				if float64(choice) < scoped.Min {
					scoped.Min = float64(choice)
				}
				if float64(choice) > scoped.Max {
					scoped.Max = float64(choice)
				}
			}
		}
		narrowCapabilityRange(&scoped, definition.Range)
		narrowCapabilityRange(&scoped, constraint.Range)
		if len(constraint.Values) > 0 {
			scoped.Choices = intersectCapabilityChoices(scoped.Choices, constraint.Values)
		} else if len(definition.Values) > 0 {
			scoped.Choices = intersectCapabilityChoices(scoped.Choices, definition.Values)
		}
		if len(scoped.IntChoices) > 0 {
			choices := scoped.IntChoices[:0]
			for _, choice := range scoped.IntChoices {
				if float64(choice) >= scoped.Min && float64(choice) <= scoped.Max {
					choices = append(choices, choice)
				}
			}
			scoped.IntChoices = append([]int(nil), choices...)
		}
		out.capabilities[normalizeParamName(scoped.Name)] = scoped
	}
	return out, nil
}

func DefaultSearchSpaceForExecution(parameterNames []string, strategy StrategyContext, scope ExecutionScope) (HyperparameterSearchSpace, error) {
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		return HyperparameterSearchSpace{}, err
	}
	space, err := defaultSearchSpaceWithRegistry(parameterNames, strategy, registry)
	if err != nil {
		return HyperparameterSearchSpace{}, err
	}
	stampSearchSpace(&space, scope)
	return space, nil
}

func FilterSearchSpaceForExecution(space HyperparameterSearchSpace, strategy StrategyContext, scope ExecutionScope) (HyperparameterSearchSpace, error) {
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		return HyperparameterSearchSpace{}, err
	}
	out := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{}}
	stampSearchSpace(&out, scope)
	for _, spec := range space.Parameters {
		capability, ok := registry.Capability(spec.Name)
		if !ok {
			continue
		}
		if err := validateCapabilityCondition(capability, strategy, space); err != nil {
			continue
		}
		out.Parameters = append(out.Parameters, spec)
	}
	if len(out.Parameters) == 0 {
		return HyperparameterSearchSpace{}, fmt.Errorf(
			"automl search space has no executable parameters for task %q and runner %q",
			scope.Task,
			scope.Runner,
		)
	}
	if err := validateSearchSpaceWithRegistry(out, strategy, registry); err != nil {
		return HyperparameterSearchSpace{}, err
	}
	return out, nil
}

func ValidateSearchSpaceForExecution(space HyperparameterSearchSpace, strategy StrategyContext, scope ExecutionScope) error {
	if err := validateSearchSpaceScope(space, scope); err != nil {
		return err
	}
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		return err
	}
	return validateSearchSpaceWithRegistry(space, strategy, registry)
}

func ValidateSuggestionForExecution(suggestion HyperparameterSuggestion, space HyperparameterSearchSpace, strategy StrategyContext, scope ExecutionScope) error {
	if err := validateSearchSpaceScope(space, scope); err != nil {
		return err
	}
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		return err
	}
	return validateSuggestionWithRegistry(suggestion, space, strategy, registry)
}

func EligibleParametersForExecution(strategy StrategyContext, scope ExecutionScope) ([]HyperparameterCapability, error) {
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		return nil, err
	}
	out := []HyperparameterCapability{}
	for _, capability := range registry.Capabilities() {
		space := HyperparameterSearchSpace{}
		if err := validateCapabilityCondition(capability, strategy, space); err != nil {
			continue
		}
		out = append(out, capability)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func stampSearchSpace(space *HyperparameterSearchSpace, scope ExecutionScope) {
	if space == nil {
		return
	}
	space.CapabilityVersion = scope.CapabilityVersion
	space.Task = scope.Task
	space.Runner = scope.Runner
}

func validateSearchSpaceScope(space HyperparameterSearchSpace, scope ExecutionScope) error {
	for label, pair := range map[string][2]string{
		"capability_version": {space.CapabilityVersion, scope.CapabilityVersion},
		"task":               {space.Task, scope.Task},
		"runner":             {space.Runner, scope.Runner},
	} {
		if pair[0] != "" && pair[0] != pair[1] {
			return fmt.Errorf("automl search space %s %q does not match execution scope %q", label, pair[0], pair[1])
		}
	}
	return nil
}

func narrowCapabilityRange(capability *HyperparameterCapability, value *execution.NumericRange) {
	if capability == nil || value == nil {
		return
	}
	if value.Min != nil && *value.Min > capability.Min {
		capability.Min = *value.Min
	}
	if value.ExclusiveMin != nil && *value.ExclusiveMin >= capability.Min {
		capability.Min = *value.ExclusiveMin
	}
	if value.Max != nil && *value.Max < capability.Max {
		capability.Max = *value.Max
	}
}

func intersectCapabilityChoices(left, right []string) []string {
	if len(left) == 0 {
		return append([]string(nil), right...)
	}
	allowed := map[string]bool{}
	for _, value := range right {
		allowed[normalizeParamName(value)] = true
	}
	out := []string{}
	for _, value := range left {
		if allowed[normalizeParamName(value)] {
			out = append(out, value)
		}
	}
	return out
}
