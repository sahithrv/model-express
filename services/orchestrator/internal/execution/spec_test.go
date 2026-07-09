package execution_test

import (
	"testing"

	"model-express/services/orchestrator/internal/execution"
)

func TestCanonicalJSONHashIsStableAcrossMapOrdering(t *testing.T) {
	left := map[string]any{
		"model": "resnet18",
		"augmentation": map[string]any{
			"horizontal_flip": true,
			"probability":     0.0,
		},
	}
	right := map[string]any{
		"augmentation": map[string]any{
			"probability":     0,
			"horizontal_flip": true,
		},
		"model": "resnet18",
	}
	leftHash, err := execution.CanonicalJSONHash(left)
	if err != nil {
		t.Fatalf("hash left config: %v", err)
	}
	rightHash, err := execution.CanonicalJSONHash(right)
	if err != nil {
		t.Fatalf("hash right config: %v", err)
	}
	if leftHash != rightHash {
		t.Fatalf("map ordering changed canonical hash: %s != %s", leftHash, rightHash)
	}
}

func TestAcceptedSpecHashUsesAliasesDefaultsAndExcludesInfrastructure(t *testing.T) {
	requested := map[string]any{
		"template":      "train_experiment",
		"model":         "resnet18",
		"epochs":        8,
		"batch_size":    16,
		"learning_rate": 0.001,
		"reason":        "hash test",
	}
	firstInput := map[string]any{
		"template":            "train_experiment",
		"model":               "resnet18",
		"epochs":              8,
		"batch_size":          16,
		"learning_rate":       0.001,
		"reason":              "hash test",
		"augmentation_policy": "rand_augment",
		"gpu_type":            "T4",
		"attempt":             1,
		"host":                "worker-a",
	}
	secondInput := map[string]any{
		"learning_rate":       0.001,
		"batch_size":          16,
		"epochs":              8,
		"model":               "resnet18",
		"template":            "train_experiment",
		"reason":              "hash test",
		"augmentation_policy": "randaugment",
		"gpu_type":            "L4",
		"attempt":             99,
		"host":                "worker-b",
	}
	first, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		requested,
		firstInput,
	)
	if err != nil {
		t.Fatalf("build first spec: %v", err)
	}
	second, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		requested,
		secondInput,
	)
	if err != nil {
		t.Fatalf("build second spec: %v", err)
	}
	if first.AcceptedSpecHash != second.AcceptedSpecHash {
		t.Fatalf("aliases or infrastructure changed accepted hash: %s != %s", first.AcceptedSpecHash, second.AcceptedSpecHash)
	}
	for _, infrastructureField := range []string{"gpu_type", "attempt", "host"} {
		if _, ok := first.AcceptedConfig[infrastructureField]; ok {
			t.Fatalf("accepted config contains infrastructure field %s: %#v", infrastructureField, first.AcceptedConfig)
		}
	}
	if first.AcceptedConfig["pretrained"] != true || first.AcceptedConfig["freeze_backbone"] != true {
		t.Fatalf("legacy defaults were not resolved: %#v", first.AcceptedConfig)
	}
	if first.AcceptedConfig["augmentation_policy"] != "randaugment" {
		t.Fatalf("augmentation alias was not resolved: %#v", first.AcceptedConfig)
	}
}

func TestDetectionAcceptedSpecDropsUnsupportedRequestsAndIncludesFixedSemantics(t *testing.T) {
	base := map[string]any{
		"template":      "yolo11_detection",
		"model":         "yolo11n.pt",
		"epochs":        8,
		"batch_size":    8,
		"learning_rate": 0.001,
		"image_size":    640,
		"reason":        "detection hash test",
	}
	withUnsupported := make(map[string]any, len(base)+2)
	for key, value := range base {
		withUnsupported[key] = value
	}
	withUnsupported["optimizer"] = "sgd"
	withUnsupported["pretrained"] = false

	first, err := execution.BuildExecutionSpecV1(
		"object_detection",
		"modal_ultralytics",
		base,
		base,
	)
	if err != nil {
		t.Fatalf("build base detection spec: %v", err)
	}
	second, err := execution.BuildExecutionSpecV1(
		"object_detection",
		"modal_ultralytics",
		withUnsupported,
		withUnsupported,
	)
	if err != nil {
		t.Fatalf("build unsupported detection spec: %v", err)
	}
	if first.AcceptedSpecHash != second.AcceptedSpecHash {
		t.Fatalf("unsupported detection settings changed accepted hash: %s != %s", first.AcceptedSpecHash, second.AcceptedSpecHash)
	}
	if second.AcceptedConfig["pretrained"] != true {
		t.Fatalf("fixed pretrained semantic missing: %#v", second.AcceptedConfig)
	}
	preprocessing := second.AcceptedConfig["preprocessing"].(map[string]any)
	if preprocessing["resize_strategy"] != "yolo_letterbox" {
		t.Fatalf("fixed YOLO preprocessing semantic missing: %#v", second.AcceptedConfig)
	}
}

func TestAcceptedSpecResolvesExplicitEmptyPolicyToCanonicalDefault(t *testing.T) {
	base := map[string]any{
		"template":      "train_experiment",
		"model":         "resnet18",
		"epochs":        8,
		"batch_size":    16,
		"learning_rate": 0.001,
		"reason":        "empty policy test",
	}
	explicitEmpty := make(map[string]any, len(base)+1)
	for key, value := range base {
		explicitEmpty[key] = value
	}
	explicitEmpty["augmentation_policy"] = ""

	omitted, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		base,
		base,
	)
	if err != nil {
		t.Fatalf("build omitted-policy spec: %v", err)
	}
	empty, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		explicitEmpty,
		explicitEmpty,
	)
	if err != nil {
		t.Fatalf("build empty-policy spec: %v", err)
	}
	if omitted.AcceptedSpecHash != empty.AcceptedSpecHash {
		t.Fatalf("explicit empty policy did not resolve to the default: %s != %s", omitted.AcceptedSpecHash, empty.AcceptedSpecHash)
	}
	if empty.AcceptedConfig["augmentation_policy"] != "none" {
		t.Fatalf("empty policy default = %#v", empty.AcceptedConfig["augmentation_policy"])
	}
	if omitted.RequestedConfigHash == empty.RequestedConfigHash {
		t.Fatal("requested hash did not preserve explicit empty policy intent")
	}
}

func TestRequestedAndAcceptedHashesDistinguishExplicitFalseFromOmitted(t *testing.T) {
	base := map[string]any{
		"template":      "train_experiment",
		"model":         "resnet18",
		"epochs":        8,
		"batch_size":    16,
		"learning_rate": 0.001,
		"reason":        "false presence test",
	}
	explicitFalse := make(map[string]any, len(base)+1)
	for key, value := range base {
		explicitFalse[key] = value
	}
	explicitFalse["pretrained"] = false

	omitted, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		base,
		base,
	)
	if err != nil {
		t.Fatalf("build omitted-pretrained spec: %v", err)
	}
	disabled, err := execution.BuildExecutionSpecV1(
		"image_classification",
		"modal_torchvision",
		explicitFalse,
		explicitFalse,
	)
	if err != nil {
		t.Fatalf("build pretrained=false spec: %v", err)
	}
	if omitted.RequestedConfigHash == disabled.RequestedConfigHash {
		t.Fatal("requested hash collapsed omitted and explicit pretrained=false")
	}
	if omitted.AcceptedSpecHash == disabled.AcceptedSpecHash {
		t.Fatal("accepted hash collapsed pretrained=true default and explicit pretrained=false")
	}
}
