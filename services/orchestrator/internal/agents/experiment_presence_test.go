package agents

import (
	"encoding/json"
	"testing"
)

func TestExperimentPlannerJSONDecodePreservesExplicitFalseAndZero(t *testing.T) {
	raw := []byte(`{
		"proposed_experiments":[{
			"template":"resnet_transfer",
			"model":"resnet18",
			"epochs":8,
			"batch_size":16,
			"learning_rate":0.001,
			"reason":"LLM presence test",
			"optimizer":"sgd",
			"optimizer_momentum":0,
			"pretrained":false,
			"freeze_backbone":false,
			"augmentation_policy_config":{
				"policy_type":"mixup",
				"probability":0,
				"alpha":0
			}
		}]
	}`)
	var recommendation ExperimentPlanningRecommendation
	if err := json.Unmarshal(raw, &recommendation); err != nil {
		t.Fatalf("decode planner recommendation: %v", err)
	}
	if len(recommendation.ProposedExperiments) != 1 {
		t.Fatalf("expected one proposed experiment, got %#v", recommendation)
	}
	experiment := recommendation.ProposedExperiments[0]
	for _, field := range []string{
		"optimizer_momentum",
		"pretrained",
		"freeze_backbone",
		"augmentation_policy_config.probability",
		"augmentation_policy_config.alpha",
	} {
		if !experiment.IsFieldPresent(field) {
			t.Fatalf("LLM decode lost presence for %s", field)
		}
	}
}
