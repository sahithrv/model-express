package plans

import (
	"encoding/json"
	"testing"
)

func TestStoredExperimentsEnvelopeDistinguishesVersionedAndLegacyPlans(t *testing.T) {
	var experiment PlannedExperiment
	if err := json.Unmarshal([]byte(`{
		"template":"train_experiment",
		"model":"resnet18",
		"epochs":4,
		"batch_size":8,
		"learning_rate":0.001,
		"reason":"storage test",
		"pretrained":false
	}`), &experiment); err != nil {
		t.Fatalf("decode experiment: %v", err)
	}

	versioned, err := MarshalVersionedExperiments([]PlannedExperiment{experiment}, "1.0.0")
	if err != nil {
		t.Fatalf("marshal versioned experiments: %v", err)
	}
	experiments, capabilityVersion, legacy, err := UnmarshalStoredExperiments(versioned)
	if err != nil {
		t.Fatalf("unmarshal versioned experiments: %v", err)
	}
	if legacy || capabilityVersion != "1.0.0" || len(experiments) != 1 {
		t.Fatalf("unexpected versioned result: version=%q legacy=%v experiments=%#v", capabilityVersion, legacy, experiments)
	}
	if !experiments[0].IsFieldPresent("pretrained") || experiments[0].Pretrained {
		t.Fatalf("versioned plan lost explicit pretrained=false: %#v", experiments[0])
	}

	legacyPayload, err := json.Marshal([]PlannedExperiment{experiment})
	if err != nil {
		t.Fatalf("marshal legacy experiments: %v", err)
	}
	experiments, capabilityVersion, legacy, err = UnmarshalStoredExperiments(legacyPayload)
	if err != nil {
		t.Fatalf("unmarshal legacy experiments: %v", err)
	}
	if !legacy || capabilityVersion != "" || len(experiments) != 1 {
		t.Fatalf("unexpected legacy result: version=%q legacy=%v experiments=%#v", capabilityVersion, legacy, experiments)
	}
}
