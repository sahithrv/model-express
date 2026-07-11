package execution_test

import (
	"testing"

	"model-express/services/orchestrator/internal/execution"
)

func TestExecutionValidationReportsUnsupportedDetectionFieldWithAlternative(t *testing.T) {
	requested := map[string]any{
		"template": "yolo11_detection", "model": "yolo11n.pt", "epochs": 8,
		"batch_size": 8, "learning_rate": 0.001, "image_size": 640,
		"optimizer": "sgd",
	}
	spec, err := execution.BuildExecutionSpecV1("object_detection", "modal_ultralytics", requested, requested)
	if err != nil {
		t.Fatalf("build spec: %v", err)
	}
	report, err := execution.ValidateExecutionSpecV1(spec, "yolo11", execution.ValidationModeShadow)
	if err != nil {
		t.Fatalf("validate spec: %v", err)
	}
	if !report.WouldBlock || report.Mode != execution.ValidationModeShadow {
		t.Fatalf("unexpected report: %#v", report)
	}
	finding, ok := validationFinding(report, "optimizer")
	if !ok || finding.Classification != "unsupported" || finding.ReasonCode != "runner_does_not_consume" {
		t.Fatalf("missing typed optimizer finding: %#v", report.Findings)
	}
	if finding.SuggestedAlternative == "" {
		t.Fatal("unsupported finding did not offer an alternative")
	}
}

func TestExecutionValidationReportsInactiveConditionalField(t *testing.T) {
	requested := map[string]any{
		"template": "resnet_transfer", "model": "resnet18", "epochs": 8,
		"batch_size": 16, "learning_rate": 0.001, "optimizer": "adamw",
		"optimizer_momentum": 0.8,
	}
	spec, err := execution.BuildExecutionSpecV1("image_classification", "modal_torchvision", requested, requested)
	if err != nil {
		t.Fatalf("build spec: %v", err)
	}
	report, err := execution.ValidateExecutionSpecV1(spec, "resnet", execution.ValidationModeEnforce)
	if err != nil {
		t.Fatalf("validate spec: %v", err)
	}
	finding, ok := validationFinding(report, "optimizer_momentum")
	if !ok || !finding.WouldBlock || finding.Classification != "normalized_away" {
		t.Fatalf("missing inactive conditional finding: %#v", report.Findings)
	}
	if finding.SuggestedAlternative != "set optimizer=sgd before using optimizer_momentum, or omit it" {
		t.Fatalf("unexpected alternative: %q", finding.SuggestedAlternative)
	}
}

func TestExecutionValidationNormalizationIsNonBlocking(t *testing.T) {
	requested := map[string]any{
		"template": "resnet_transfer", "model": "resnet18", "epochs": 8,
		"batch_size": 16, "learning_rate": 0.001, "augmentation_policy": "rand_augment",
	}
	spec, err := execution.BuildExecutionSpecV1("image_classification", "modal_torchvision", requested, requested)
	if err != nil {
		t.Fatalf("build spec: %v", err)
	}
	report, err := execution.ValidateExecutionSpecV1(spec, "resnet", execution.ValidationModeEnforce)
	if err != nil {
		t.Fatalf("validate spec: %v", err)
	}
	finding, ok := validationFinding(report, "augmentation_policy")
	if !ok || finding.Classification != "conditional" || finding.WouldBlock {
		t.Fatalf("unexpected normalized conditional finding: %#v", report.Findings)
	}
}

func TestPlannerCapabilityCardIsTaskAndRunnerScoped(t *testing.T) {
	card, err := execution.BuildPlannerCapabilityCard(
		"object_detection", "modal_ultralytics", execution.ValidationModeShadow, []string{"yolo11"},
	)
	if err != nil {
		t.Fatalf("build capability card: %v", err)
	}
	if card.Task != "object_detection" || card.Runner != "modal_ultralytics" || len(card.ModelFamilies) != 1 {
		t.Fatalf("unexpected card target: %#v", card)
	}
	if !containsCapabilityField(card.UnsupportedFields, "class_balancing") {
		t.Fatalf("detection card omitted unsupported class_balancing: %#v", card.UnsupportedFields)
	}
	if containsCapabilityField(card.ExecutedFields, "dropout") {
		t.Fatalf("detection card exposed classification-only dropout as executed: %#v", card.ExecutedFields)
	}
}

func validationFinding(report execution.ExecutionValidationReport, field string) (execution.ExecutionValidationFinding, bool) {
	for _, finding := range report.Findings {
		if finding.Field == field {
			return finding, true
		}
	}
	return execution.ExecutionValidationFinding{}, false
}

func containsCapabilityField(fields []string, expected string) bool {
	for _, field := range fields {
		if field == expected {
			return true
		}
	}
	return false
}
