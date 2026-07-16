package execution_test

import (
	"testing"

	"model-express/services/orchestrator/internal/execution"
)

func TestExecutionValidationModeDefaultsToEnforceWithExplicitShadowRollback(t *testing.T) {
	if got := execution.NormalizeValidationMode(""); got != execution.ValidationModeEnforce {
		t.Fatalf("default validation mode = %s, want enforce", got)
	}
	if got := execution.NormalizeValidationMode("shadow"); got != execution.ValidationModeShadow {
		t.Fatalf("shadow rollback mode = %s, want shadow", got)
	}
	if got := execution.NormalizeValidationMode("invalid"); got != execution.ValidationModeEnforce {
		t.Fatalf("invalid validation mode should fail closed to enforce, got %s", got)
	}
}

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

func TestExecutionValidationTypesEveryUnsupportedDetectionSemantic(t *testing.T) {
	requested := map[string]any{
		"template": "yolo11_detection", "model": "yolo11n.pt", "epochs": 8,
		"batch_size": 8, "learning_rate": 0.001, "image_size": 640,
		"augmentation":    map[string]any{"horizontal_flip": true},
		"class_balancing": "focal_loss", "sampling_strategy": "weighted_random_sampler",
		"optimizer": "adamw", "scheduler": "cosine", "weight_decay": 0.1,
		"freeze_backbone": true, "pretrained": false,
	}
	spec, err := execution.BuildExecutionSpecV1("object_detection", "modal_ultralytics", requested, requested)
	if err != nil {
		t.Fatalf("build spec: %v", err)
	}
	report, err := execution.ValidateExecutionSpecV1(spec, "yolo11", execution.ValidationModeShadow)
	if err != nil {
		t.Fatalf("validate spec: %v", err)
	}
	for _, field := range []string{
		"augmentation.horizontal_flip",
		"class_balancing",
		"sampling_strategy",
		"optimizer",
		"scheduler",
		"weight_decay",
		"freeze_backbone",
		"pretrained",
	} {
		finding, ok := validationFinding(report, field)
		if !ok || !finding.WouldBlock || finding.Classification != "unsupported" {
			t.Fatalf("field %s missing typed blocking finding: %#v", field, report.Findings)
		}
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

func TestExecutionValidationReportsClassificationSemanticCanonicalization(t *testing.T) {
	cases := []struct {
		name           string
		requested      map[string]any
		field          string
		reason         string
		requestedValue any
		acceptedValue  any
	}{
		{
			name: "full fine-tuning freezes false",
			requested: map[string]any{
				"template": "resnet_transfer", "model": "resnet18", "epochs": 8,
				"batch_size": 16, "learning_rate": 0.001, "image_size": 224,
				"freeze_backbone": true, "fine_tune_strategy": "full",
			},
			field: "freeze_backbone", reason: "classification_transfer_semantics_canonicalized",
			requestedValue: true, acceptedValue: false,
		},
		{
			name: "unfrozen backbone implies full strategy",
			requested: map[string]any{
				"template": "resnet_transfer", "model": "resnet18", "epochs": 8,
				"batch_size": 16, "learning_rate": 0.001, "image_size": 224,
				"freeze_backbone": false, "fine_tune_strategy": "head_only",
			},
			field: "fine_tune_strategy", reason: "classification_transfer_semantics_canonicalized",
			requestedValue: "head_only", acceptedValue: "full",
		},
		{
			name: "dataset normalization owns normalization value",
			requested: map[string]any{
				"template": "resnet_transfer", "model": "resnet18", "epochs": 8,
				"batch_size": 16, "learning_rate": 0.001, "image_size": 224,
				"preprocessing": map[string]any{"normalization": "imagenet", "use_dataset_normalization": true},
			},
			field: "preprocessing.normalization", reason: "classification_dataset_normalization_canonicalized",
			requestedValue: "imagenet", acceptedValue: "dataset",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec, err := execution.BuildExecutionSpecV1("image_classification", "modal_torchvision", tc.requested, tc.requested)
			if err != nil {
				t.Fatalf("build spec: %v", err)
			}
			report, err := execution.ValidateExecutionSpecV1(spec, "resnet", execution.ValidationModeEnforce)
			if err != nil {
				t.Fatalf("validate spec: %v", err)
			}
			if report.WouldBlock {
				t.Fatalf("safe semantic canonicalization should not block: %#v", report)
			}
			finding, ok := validationFindingByReason(report, tc.field, tc.reason)
			if !ok {
				t.Fatalf("missing semantic canonicalization finding for %s: %#v", tc.field, report.Findings)
			}
			if finding.ReasonCode != tc.reason || finding.RequestedValue != tc.requestedValue || finding.AcceptedValue != tc.acceptedValue || finding.SuggestedAlternative == "" {
				t.Fatalf("unexpected semantic finding: %#v", finding)
			}
		})
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

func validationFindingByReason(report execution.ExecutionValidationReport, field string, reasonCode string) (execution.ExecutionValidationFinding, bool) {
	for _, finding := range report.Findings {
		if finding.Field == field && finding.ReasonCode == reasonCode {
			return finding, true
		}
	}
	return execution.ExecutionValidationFinding{}, false
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
