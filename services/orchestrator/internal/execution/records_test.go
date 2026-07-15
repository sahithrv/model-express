package execution

import "testing"

func TestDeriveRealizationVerdicts(t *testing.T) {
	spec := JobExecutionSpec{CapabilityVersion: "v1", Task: "image_classification", Runner: "local_simulator", AcceptedSpec: map[string]any{"batch_size": float64(16), "epochs": float64(3)}}
	tests := []struct {
		name   string
		create RealizationObservationCreate
		want   string
	}{
		{"matched", RealizationObservationCreate{RealizedConfig: map[string]any{"batch_size": float64(16), "epochs": float64(3)}}, ExecutionVerdictMatched},
		{"approved batch recovery", RealizationObservationCreate{RealizedConfig: map[string]any{"batch_size": float64(8), "epochs": float64(3)}, AdjustmentPolicy: "batch_size_recovery"}, ExecutionVerdictApprovedAdjustment},
		{"mismatch", RealizationObservationCreate{RealizedConfig: map[string]any{"batch_size": float64(8), "epochs": float64(3)}}, ExecutionVerdictMismatch},
		{"simulated", RealizationObservationCreate{RealizedConfig: map[string]any{"batch_size": float64(16), "epochs": float64(3)}, Simulated: true}, ExecutionVerdictSimulated},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, got, _, err := DeriveRealization(spec, test.create)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("got %s, want %s", got, test.want)
			}
		})
	}
}

func TestRealizedEffectiveHashIsDeterministic(t *testing.T) {
	spec := JobExecutionSpec{CapabilityVersion: "2026-07-09", Task: "image_classification", Runner: "modal_torchvision", AcceptedSpec: map[string]any{"model": "resnet18", "batch_size": float64(16), "pretrained": false}}
	observation := RealizationObservationCreate{RealizedConfig: map[string]any{"pretrained": false, "batch_size": float64(16), "model": "resnet18"}}
	_, firstHash, firstVerdict, _, err := DeriveRealization(spec, observation)
	if err != nil {
		t.Fatal(err)
	}
	_, secondHash, secondVerdict, _, err := DeriveRealization(spec, observation)
	if err != nil {
		t.Fatal(err)
	}
	if firstHash != secondHash || firstVerdict != ExecutionVerdictMatched || secondVerdict != ExecutionVerdictMatched {
		t.Fatalf("nondeterministic realization: first=%s/%s second=%s/%s", firstHash, firstVerdict, secondHash, secondVerdict)
	}
}

func TestObservationRejectsUnacceptedAndRedactsSensitiveFields(t *testing.T) {
	spec := JobExecutionSpec{AcceptedSpec: map[string]any{"epochs": float64(3)}}
	if _, _, _, _, err := DeriveRealization(spec, RealizationObservationCreate{RealizedConfig: map[string]any{"epochs": float64(3), "token": "secret"}}); err == nil {
		t.Fatal("expected unaccepted semantic field to be rejected")
	}
	redacted := RedactSensitiveMap(map[string]any{"api_token": "secret", "nested": map[string]any{"password": "p", "safe": "value"}})
	if redacted["api_token"] != "[REDACTED]" || redacted["nested"].(map[string]any)["password"] != "[REDACTED]" || redacted["nested"].(map[string]any)["safe"] != "value" {
		t.Fatalf("unexpected redaction: %#v", redacted)
	}
}
