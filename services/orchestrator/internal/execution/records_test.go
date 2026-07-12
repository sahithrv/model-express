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
			_, _, got, err := DeriveRealization(spec, test.create)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("got %s, want %s", got, test.want)
			}
		})
	}
}

func TestObservationRejectsUnacceptedAndRedactsSensitiveFields(t *testing.T) {
	spec := JobExecutionSpec{AcceptedSpec: map[string]any{"epochs": float64(3)}}
	if _, _, _, err := DeriveRealization(spec, RealizationObservationCreate{RealizedConfig: map[string]any{"epochs": float64(3), "token": "secret"}}); err == nil {
		t.Fatal("expected unaccepted semantic field to be rejected")
	}
	redacted := RedactSensitiveMap(map[string]any{"api_token": "secret", "nested": map[string]any{"password": "p", "safe": "value"}})
	if redacted["api_token"] != "[REDACTED]" || redacted["nested"].(map[string]any)["password"] != "[REDACTED]" || redacted["nested"].(map[string]any)["safe"] != "value" {
		t.Fatalf("unexpected redaction: %#v", redacted)
	}
}
