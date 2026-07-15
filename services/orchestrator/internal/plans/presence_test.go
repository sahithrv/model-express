package plans

import (
	"encoding/json"
	"testing"
)

func TestPlannedExperimentPresenceSurvivesJSONRoundTrip(t *testing.T) {
	raw := []byte(`{
		"template":"train_experiment",
		"model":"resnet18",
		"epochs":4,
		"batch_size":8,
		"learning_rate":0.001,
		"reason":"presence test",
		"pretrained":false,
		"freeze_backbone":false,
		"optimizer":"sgd",
		"optimizer_momentum":0,
		"augmentation":{},
		"augmentation_policy":"mixup",
		"augmentation_policy_config":{
			"policy_type":"mixup",
			"probability":0,
			"alpha":0
		},
		"preprocessing":{"use_dataset_normalization":false}
	}`)

	var experiment PlannedExperiment
	if err := json.Unmarshal(raw, &experiment); err != nil {
		t.Fatalf("decode experiment: %v", err)
	}
	for _, field := range []string{
		"pretrained",
		"freeze_backbone",
		"optimizer_momentum",
		"augmentation",
		"augmentation_policy_config.probability",
		"augmentation_policy_config.alpha",
		"preprocessing.use_dataset_normalization",
	} {
		if !experiment.IsFieldPresent(field) {
			t.Fatalf("expected %s to be marked present", field)
		}
	}

	roundTrip, err := json.Marshal(experiment)
	if err != nil {
		t.Fatalf("encode experiment: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(roundTrip, &payload); err != nil {
		t.Fatalf("decode experiment payload: %v", err)
	}
	assertPayloadValue(t, payload, "pretrained", false)
	assertPayloadValue(t, payload, "freeze_backbone", false)
	assertPayloadValue(t, payload, "optimizer_momentum", float64(0))
	policy := payload["augmentation_policy_config"].(map[string]any)
	assertPayloadValue(t, policy, "probability", float64(0))
	assertPayloadValue(t, policy, "alpha", float64(0))
	preprocessing := payload["preprocessing"].(map[string]any)
	assertPayloadValue(t, preprocessing, "use_dataset_normalization", false)

	var decodedAgain PlannedExperiment
	if err := json.Unmarshal(roundTrip, &decodedAgain); err != nil {
		t.Fatalf("decode experiment again: %v", err)
	}
	if !decodedAgain.IsFieldPresent("pretrained") || !decodedAgain.IsFieldPresent("augmentation_policy_config.alpha") {
		t.Fatalf("presence was lost after second decode: %#v", decodedAgain)
	}
}

func TestPlannedExperimentOmittedValuesRemainOmitted(t *testing.T) {
	var experiment PlannedExperiment
	if err := json.Unmarshal([]byte(`{
		"template":"train_experiment",
		"model":"resnet18",
		"epochs":4,
		"batch_size":8,
		"learning_rate":0.001,
		"reason":"omitted test"
	}`), &experiment); err != nil {
		t.Fatalf("decode experiment: %v", err)
	}
	if experiment.IsFieldPresent("pretrained") || experiment.IsFieldPresent("optimizer_momentum") {
		t.Fatal("omitted values were marked present")
	}
	payload, err := experiment.RequestedConfig()
	if err != nil {
		t.Fatalf("requested config: %v", err)
	}
	if _, ok := payload["pretrained"]; ok {
		t.Fatalf("omitted pretrained appeared in requested config: %#v", payload)
	}
	if _, ok := payload["optimizer_momentum"]; ok {
		t.Fatalf("omitted optimizer_momentum appeared in requested config: %#v", payload)
	}
}

func TestPresenceMutationIsCopyOnWrite(t *testing.T) {
	var original PlannedExperiment
	if err := json.Unmarshal([]byte(`{
		"template":"train_experiment",
		"model":"resnet18",
		"epochs":4,
		"batch_size":8,
		"learning_rate":0.001,
		"reason":"copy test",
		"pretrained":false
	}`), &original); err != nil {
		t.Fatalf("decode experiment: %v", err)
	}
	copyOfExperiment := original
	copyOfExperiment.MarkFieldPresent("optimizer_momentum")
	if original.IsFieldPresent("optimizer_momentum") {
		t.Fatal("marking a copied experiment mutated the original presence set")
	}
}

func assertPayloadValue(t *testing.T, payload map[string]any, key string, expected any) {
	t.Helper()
	actual, ok := payload[key]
	if !ok {
		t.Fatalf("payload is missing %s: %#v", key, payload)
	}
	if actual != expected {
		t.Fatalf("payload %s = %#v, want %#v", key, actual, expected)
	}
}
