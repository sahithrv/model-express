package automl

import (
	"reflect"
	"testing"
)

func TestDetectionExecutionScopeExcludesClassificationOnlyTunables(t *testing.T) {
	scope, err := CurrentExecutionScope("object_detection", "modal_ultralytics")
	if err != nil {
		t.Fatalf("build detection scope: %v", err)
	}
	registry, err := CapabilityRegistryForExecution(scope)
	if err != nil {
		t.Fatalf("build detection registry: %v", err)
	}
	for _, field := range []string{"dropout", "label_smoothing", "weight_decay", "class_balancing_config.focal_loss_gamma"} {
		if _, ok := registry.Capability(field); ok {
			t.Fatalf("detection registry exposed classification-only field %s", field)
		}
	}
	for _, field := range []string{"learning_rate", "batch_size", "epochs"} {
		if _, ok := registry.Capability(field); !ok {
			t.Fatalf("detection registry omitted executable field %s", field)
		}
	}
}

func TestScopedSearchSpaceRejectsInactiveConditionalParameter(t *testing.T) {
	scope, err := CurrentExecutionScope("image_classification", "modal_torchvision")
	if err != nil {
		t.Fatalf("build classification scope: %v", err)
	}
	_, err = DefaultSearchSpaceForExecution(
		[]string{"learning_rate", "optimizer_momentum"},
		StrategyContext{Optimizer: "adamw"},
		scope,
	)
	if err == nil {
		t.Fatal("inactive SGD momentum was accepted without optimizer=sgd")
	}

	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "learning_rate", Type: ParameterFloat},
		{Name: "optimizer_momentum", Type: ParameterFloat},
	}}
	filtered, err := FilterSearchSpaceForExecution(space, StrategyContext{Optimizer: "adamw"}, scope)
	if err != nil {
		t.Fatalf("filter default search space: %v", err)
	}
	if len(filtered.Parameters) != 1 || filtered.Parameters[0].Name != "learning_rate" {
		t.Fatalf("inactive conditional parameter was not excluded: %#v", filtered.Parameters)
	}
}

func TestScopedSearchSpaceIsDeterministicForCapabilityVersion(t *testing.T) {
	scope, err := CurrentExecutionScope("image_classification", "modal_torchvision")
	if err != nil {
		t.Fatalf("build classification scope: %v", err)
	}
	names := []string{"learning_rate", "batch_size", "dropout"}
	first, err := DefaultSearchSpaceForExecution(names, StrategyContext{}, scope)
	if err != nil {
		t.Fatalf("build first search space: %v", err)
	}
	second, err := DefaultSearchSpaceForExecution(names, StrategyContext{}, scope)
	if err != nil {
		t.Fatalf("build second search space: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("fixed capability version produced different spaces: first=%#v second=%#v", first, second)
	}
	if first.CapabilityVersion != scope.CapabilityVersion || first.Task != scope.Task || first.Runner != scope.Runner {
		t.Fatalf("search space omitted execution scope: %#v", first)
	}
}

func TestDetectionSearchSpaceValidationRejectsDropout(t *testing.T) {
	scope, err := CurrentExecutionScope("object_detection", "modal_ultralytics")
	if err != nil {
		t.Fatalf("build detection scope: %v", err)
	}
	minValue, maxValue := 0.0, 0.5
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{{
		Name: "dropout", Type: ParameterFloat, Min: &minValue, Max: &maxValue,
	}}}
	if err := ValidateSearchSpaceForExecution(space, StrategyContext{}, scope); err == nil {
		t.Fatal("detection search space accepted classifier-only dropout")
	}
}
