package automl

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func TestSeededRandomSamplerDeterministicAndBounded(t *testing.T) {
	minLR := 1e-5
	maxLR := 3e-4
	minWD := 0.01
	maxWD := 0.08
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "learning_rate", Type: ParameterFloat, Min: &minLR, Max: &maxLR, Scale: SearchScaleLog},
		{Name: "weight_decay", Type: ParameterFloat, Min: &minWD, Max: &maxWD},
		{Name: "batch_size", Type: ParameterInteger, IntChoices: []int{16, 32, 64}},
	}}
	request := SuggestRequest{SearchSpace: space, Seed: 42}
	first, err := SeededRandomSampler{}.Suggest(context.Background(), request)
	if err != nil {
		t.Fatalf("suggest first: %v", err)
	}
	second, err := SeededRandomSampler{}.Suggest(context.Background(), request)
	if err != nil {
		t.Fatalf("suggest second: %v", err)
	}
	if first.Values["learning_rate"] != second.Values["learning_rate"] ||
		first.Values["weight_decay"] != second.Values["weight_decay"] ||
		first.Values["batch_size"] != second.Values["batch_size"] {
		t.Fatalf("expected deterministic values, got %#v and %#v", first.Values, second.Values)
	}
	if err := ValidateSuggestion(first, space, StrategyContext{}); err != nil {
		t.Fatalf("suggestion out of bounds: %v", err)
	}
}

func TestValidateSearchSpaceRejectsStrategyFields(t *testing.T) {
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "model", Type: ParameterCategorical, Choices: []string{"resnet18"}},
	}}
	if err := ValidateSearchSpace(space, StrategyContext{}); err == nil {
		t.Fatal("expected strategy field rejection")
	}
}

func TestValidateSearchSpaceRequiresConditionalStrategy(t *testing.T) {
	minAlpha := 0.1
	maxAlpha := 0.8
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "augmentation_policy_config.alpha", Type: ParameterFloat, Min: &minAlpha, Max: &maxAlpha},
	}}
	if err := ValidateSearchSpace(space, StrategyContext{AugmentationPolicyType: "randaugment"}); err == nil {
		t.Fatal("expected alpha rejection without mixup/cutmix")
	}
	if err := ValidateSearchSpace(space, StrategyContext{AugmentationPolicyType: "mixup"}); err != nil {
		t.Fatalf("expected valid mixup alpha search: %v", err)
	}
}

func TestConditionalOptimizerAndSchedulerParameters(t *testing.T) {
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "optimizer", Type: ParameterCategorical, Choices: []string{"adamw", "sgd"}},
		{Name: "optimizer_momentum", Type: ParameterFloat},
		{Name: "scheduler", Type: ParameterCategorical, Choices: []string{"none", "step"}},
		{Name: "scheduler_gamma", Type: ParameterFloat},
	}}
	suggestion, err := SeededRandomSampler{}.Suggest(context.Background(), SuggestRequest{
		SearchSpace: space,
		Seed:        9,
	})
	if err != nil {
		t.Fatalf("conditional suggestion should validate: %v", err)
	}
	if suggestion.Values["optimizer"] != "sgd" {
		if _, ok := suggestion.Values["optimizer_momentum"]; ok {
			t.Fatalf("momentum should be omitted unless optimizer is sgd: %#v", suggestion.Values)
		}
	}
	if suggestion.Values["scheduler"] != "step" {
		if _, ok := suggestion.Values["scheduler_gamma"]; ok {
			t.Fatalf("scheduler gamma should be omitted unless scheduler is step: %#v", suggestion.Values)
		}
	}
}

func TestAdaptiveBayesianSamplerUsesBestHistory(t *testing.T) {
	minLR := 1e-5
	maxLR := 3e-4
	space := HyperparameterSearchSpace{Parameters: []HyperparameterParameterSpec{
		{Name: "learning_rate", Type: ParameterFloat, Min: &minLR, Max: &maxLR, Scale: SearchScaleLog},
		{Name: "optimizer", Type: ParameterCategorical, Choices: []string{"adamw", "sgd"}},
	}}
	history := []OptimizerTrial{
		{
			Status: "SUCCEEDED",
			Score:  0.5,
			Metrics: map[string]any{"hyperparameters": map[string]any{
				"learning_rate": 2.8e-4,
				"optimizer":     "adamw",
			}},
		},
		{
			Status: "SUCCEEDED",
			Score:  0.8,
			Metrics: map[string]any{"hyperparameters": map[string]any{
				"learning_rate": 8e-5,
				"optimizer":     "sgd",
			}},
		},
	}
	suggestion, err := AdaptiveBayesianSampler{}.Suggest(context.Background(), SuggestRequest{
		SearchSpace: space,
		History:     history,
		Seed:        3,
	})
	if err != nil {
		t.Fatalf("adaptive suggest: %v", err)
	}
	if suggestion.Provenance["learning_rate"] != ProvenanceBayesianOptimizer {
		t.Fatalf("expected bayesian provenance, got %#v", suggestion.Provenance)
	}
	if err := ValidateSuggestion(suggestion, space, StrategyContext{}); err != nil {
		t.Fatalf("adaptive suggestion should stay valid: %v", err)
	}
}

func TestAdaptiveBayesianSamplerClampsFloatBoundary(t *testing.T) {
	minLR := 1.6000000000000003e-05
	maxLR := 0.0004
	spec := HyperparameterParameterSpec{
		Name:  "learning_rate",
		Type:  ParameterFloat,
		Min:   &minLR,
		Max:   &maxLR,
		Scale: SearchScaleLog,
	}
	value, err := adaptiveValue(rand.New(rand.NewSource(1)), spec, 1.0)
	if err != nil {
		t.Fatalf("adaptive value should clamp boundary float: %v", err)
	}
	got, ok := NumberValue(value)
	if !ok {
		t.Fatalf("learning rate should be numeric: %#v", value)
	}
	if got < minLR || got > maxLR {
		t.Fatalf("learning rate should stay inside [%g, %g], got %.18g", minLR, maxLR, got)
	}
	if math.Abs(got-maxLR) > 1e-18 {
		t.Fatalf("expected boundary value to clamp to max %g, got %.18g", maxLR, got)
	}
}
