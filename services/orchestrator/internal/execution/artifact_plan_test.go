package execution

import (
	"reflect"
	"testing"
)

func TestArtifactPlanV1PreservesLegacyAutomaticArtifactsWithoutPolicy(t *testing.T) {
	plan, err := BuildAutomaticArtifactPlanV1("image_classification", "modal_torchvision", ArtifactPolicySelection{})
	if err != nil {
		t.Fatal(err)
	}
	formats := []string{}
	for _, artifact := range plan.Artifacts {
		formats = append(formats, artifact.Format)
		if artifact.Precision != "fp32" || artifact.ExecutionProvider != "cpu_execution_provider" || !reflect.DeepEqual(artifact.ExecutionRequirements, []string{"cpu_compatible"}) {
			t.Fatalf("unsupported automatic artifact directive: %#v", artifact)
		}
	}
	if !reflect.DeepEqual(formats, []string{"onnx", "torchscript", "pytorch"}) || plan.PolicyRestricted || len(plan.RequiredWorkerCapabilities.PolicyContractVersions) != 0 {
		t.Fatalf("legacy-compatible artifact plan = %#v", plan)
	}
	if err := ValidateArtifactPlanV1(plan, "image_classification", "modal_torchvision"); err != nil {
		t.Fatal(err)
	}
}

func TestRestrictedArtifactPlanNegotiatesExactKnownWorkerVersions(t *testing.T) {
	selection := ArtifactPolicySelection{
		PolicyRestricted: true, EffectivePolicyHash: "sha256:policy",
		PermittedCatalog: map[string][]string{
			"export_formats": {"onnx"}, "precisions": {"fp32"}, "runtimes": {"onnxruntime"},
			"execution_providers": {"cpu_execution_provider"}, "execution_requirements": {"cpu_compatible"},
		},
	}
	plan, err := BuildAutomaticArtifactPlanV1("image_classification", "modal_torchvision", selection)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Artifacts) != 1 || plan.Artifacts[0].Format != "onnx" || len(plan.FallbackFormats) != 0 {
		t.Fatalf("restricted plan = %#v", plan)
	}
	if plan.ArtifactPlanHash != "sha256:Psw64Nasw8unmzo7Tu37kJuWD7SkEoTu8BOpGkEpiD4" {
		t.Fatalf("artifact-plan cross-runtime hash = %s", plan.ArtifactPlanHash)
	}
	if WorkerCapabilitiesSatisfyForSpec(plan, "image_classification", "modal_torchvision", nil, nil) ||
		WorkerCapabilitiesSatisfyForSpec(plan, "image_classification", "modal_torchvision", []string{"policy_contract_v999"}, []string{WorkerArtifactPlanV1}) {
		t.Fatal("restricted plan accepted missing or unknown worker capability versions")
	}
	if !WorkerCapabilitiesSatisfyForSpec(plan, "image_classification", "modal_torchvision", []string{WorkerPolicyContractV1}, []string{WorkerArtifactPlanV1}) {
		t.Fatal("restricted plan rejected the exact supported worker capability versions")
	}
}

func TestUnavailablePrecisionCannotProduceArtifactPlan(t *testing.T) {
	_, err := BuildAutomaticArtifactPlanV1("image_classification", "modal_torchvision", ArtifactPolicySelection{
		PermittedCatalog: map[string][]string{
			"export_formats": {"onnx"}, "precisions": {"fp16", "int8"}, "runtimes": {"onnxruntime"},
			"execution_providers": {"cpu_execution_provider"}, "execution_requirements": {"cpu_compatible"},
		},
	})
	if err == nil {
		t.Fatal("FP16/INT8 unexpectedly produced an artifact plan")
	}
}

func TestArtifactPlanFailsClosedOnUnknownRequiredCapability(t *testing.T) {
	plan, err := BuildAutomaticArtifactPlanV1("image_classification", "modal_torchvision", ArtifactPolicySelection{})
	if err != nil {
		t.Fatal(err)
	}
	plan.RequiredWorkerCapabilities.ArtifactPlanVersions = []string{"artifact_plan_v999"}
	plan.ArtifactPlanHash, err = artifactPlanHash(plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateArtifactPlanV1(plan, "image_classification", "modal_torchvision"); err == nil {
		t.Fatal("unknown required worker capability was accepted")
	}
}

func TestNoPolicyManualExportPreservesLocalSimulatorChampionHandoff(t *testing.T) {
	plan, err := BuildManualArtifactPlanV1("image_classification", "local_simulator", "onnx", ArtifactPolicySelection{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Automatic || len(plan.Artifacts) != 1 || plan.Artifacts[0].Format != "onnx" || plan.PolicyRestricted {
		t.Fatalf("legacy manual export plan = %#v", plan)
	}
	if err := ValidateArtifactPlanV1(plan, "image_classification", "local_simulator"); err != nil {
		t.Fatal(err)
	}
}
