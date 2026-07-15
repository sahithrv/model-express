package execution

import (
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/catalog"
)

const (
	ArtifactPlanSchemaVersionV1 = "artifact_plan_v1"
	ArtifactPlanConfigKey       = "artifact_plan_v1"
	WorkerPolicyContractV1      = "policy_contract_v1"
	WorkerArtifactPlanV1        = "artifact_plan_v1"
)

// ArtifactDirective is one immutable output/runtime combination authorized by
// the server. Workers must never infer an additional output from another
// directive (for example, a PyTorch fallback from an ONNX directive).
type ArtifactDirective struct {
	Format                string   `json:"format"`
	Precision             string   `json:"precision"`
	Runtime               string   `json:"runtime"`
	ExecutionProvider     string   `json:"execution_provider"`
	ExecutionRequirements []string `json:"execution_requirements"`
}

type WorkerCapabilityRequirements struct {
	PolicyContractVersions []string `json:"policy_contract_versions"`
	ArtifactPlanVersions   []string `json:"artifact_plan_versions"`
}

// ArtifactPlanV1 is a server-issued, hash-bound artifact realization plan.
// PolicyRestricted distinguishes legacy-compatible plans from plans that need
// an artifact-aware worker to avoid producing capabilities removed by policy.
type ArtifactPlanV1 struct {
	SchemaVersion              string                       `json:"schema_version"`
	Artifacts                  []ArtifactDirective          `json:"artifacts"`
	FallbackFormats            []string                     `json:"fallback_formats"`
	Automatic                  bool                         `json:"automatic"`
	PolicyRestricted           bool                         `json:"policy_restricted"`
	EffectivePolicyHash        string                       `json:"effective_policy_hash,omitempty"`
	RequiredWorkerCapabilities WorkerCapabilityRequirements `json:"required_worker_capabilities"`
	ArtifactPlanHash           string                       `json:"artifact_plan_hash"`
}

type ArtifactPolicySelection struct {
	PermittedCatalog    map[string][]string
	PolicyRestricted    bool
	EffectivePolicyHash string
}

func BuildAutomaticArtifactPlanV1(task, runner string, selection ArtifactPolicySelection) (ArtifactPlanV1, error) {
	formats := automaticArtifactFormats(task, runner)
	return buildArtifactPlanV1(task, runner, formats, automaticFallbackFormats(task, runner), true, selection)
}

func BuildManualArtifactPlanV1(task, runner, requestedFormat string, selection ArtifactPolicySelection) (ArtifactPlanV1, error) {
	requestedFormat = strings.TrimSpace(requestedFormat)
	if requestedFormat == "" {
		requestedFormat = "onnx"
	}
	return buildArtifactPlanV1(task, runner, []string{requestedFormat}, nil, false, selection)
}

func buildArtifactPlanV1(
	task, runner string,
	formats, fallbackFormats []string,
	automatic bool,
	selection ArtifactPolicySelection,
) (ArtifactPlanV1, error) {
	capabilityRunner := artifactCapabilityRunner(task, runner, automatic)
	permitted := selection.PermittedCatalog
	if permitted == nil {
		permitted = availableArtifactCatalog(task, capabilityRunner)
	}
	artifacts := make([]ArtifactDirective, 0, len(formats))
	for _, requested := range formats {
		format, ok := permittedCatalogID(permitted, "export_formats", requested)
		if !ok {
			continue
		}
		directive, err := artifactDirectiveForFormat(task, capabilityRunner, format, permitted)
		if err != nil {
			return ArtifactPlanV1{}, err
		}
		artifacts = append(artifacts, directive)
	}
	if len(formats) > 0 && len(artifacts) == 0 && runner != "local_simulator" {
		return ArtifactPlanV1{}, fmt.Errorf("no policy-permitted artifact/runtime combination for %s/%s", task, runner)
	}

	allowedFormats := map[string]bool{}
	for _, artifact := range artifacts {
		allowedFormats[artifact.Format] = true
	}
	fallbacks := []string{}
	for _, format := range fallbackFormats {
		if allowedFormats[format] {
			fallbacks = append(fallbacks, format)
		}
	}
	sort.Strings(fallbacks)
	requirements := WorkerCapabilityRequirements{PolicyContractVersions: []string{}, ArtifactPlanVersions: []string{}}
	if selection.PolicyRestricted {
		requirements.PolicyContractVersions = []string{WorkerPolicyContractV1}
		requirements.ArtifactPlanVersions = []string{WorkerArtifactPlanV1}
	}
	plan := ArtifactPlanV1{
		SchemaVersion: ArtifactPlanSchemaVersionV1, Artifacts: artifacts,
		FallbackFormats: fallbacks, Automatic: automatic,
		PolicyRestricted: selection.PolicyRestricted, EffectivePolicyHash: strings.TrimSpace(selection.EffectivePolicyHash),
		RequiredWorkerCapabilities: requirements,
	}
	hash, err := artifactPlanHash(plan)
	if err != nil {
		return ArtifactPlanV1{}, err
	}
	plan.ArtifactPlanHash = hash
	return plan, nil
}

func ValidateArtifactPlanV1(plan ArtifactPlanV1, task, runner string) error {
	if plan.SchemaVersion != ArtifactPlanSchemaVersionV1 {
		return fmt.Errorf("unknown artifact plan schema %q", plan.SchemaVersion)
	}
	wantHash, err := artifactPlanHash(plan)
	if err != nil {
		return err
	}
	if plan.ArtifactPlanHash == "" || plan.ArtifactPlanHash != wantHash {
		return fmt.Errorf("artifact plan hash mismatch")
	}
	capabilityRunner := artifactCapabilityRunner(task, runner, plan.Automatic)
	permitted := availableArtifactCatalog(task, capabilityRunner)
	seen := map[string]bool{}
	for _, directive := range plan.Artifacts {
		if seen[directive.Format] {
			return fmt.Errorf("artifact plan duplicates format %q", directive.Format)
		}
		seen[directive.Format] = true
		expected, err := artifactDirectiveForFormat(task, capabilityRunner, directive.Format, permitted)
		if err != nil {
			return err
		}
		if expected.Precision != directive.Precision || expected.Runtime != directive.Runtime || expected.ExecutionProvider != directive.ExecutionProvider || strings.Join(expected.ExecutionRequirements, "\x00") != strings.Join(directive.ExecutionRequirements, "\x00") {
			return fmt.Errorf("unsupported artifact capability combination for format %q", directive.Format)
		}
	}
	for _, fallback := range plan.FallbackFormats {
		if !seen[fallback] {
			return fmt.Errorf("artifact fallback %q is not an authorized output", fallback)
		}
	}
	for _, required := range plan.RequiredWorkerCapabilities.PolicyContractVersions {
		if required != WorkerPolicyContractV1 {
			return fmt.Errorf("unknown required worker policy capability %q", required)
		}
	}
	for _, required := range plan.RequiredWorkerCapabilities.ArtifactPlanVersions {
		if required != WorkerArtifactPlanV1 {
			return fmt.Errorf("unknown required worker artifact capability %q", required)
		}
	}
	if plan.PolicyRestricted {
		if !contains(plan.RequiredWorkerCapabilities.PolicyContractVersions, WorkerPolicyContractV1) || !contains(plan.RequiredWorkerCapabilities.ArtifactPlanVersions, WorkerArtifactPlanV1) {
			return fmt.Errorf("restricted artifact plan is missing required worker capability versions")
		}
	}
	return nil
}

// WorkerCapabilitiesSatisfyForSpec validates both the plan and the exact
// worker advertisements used by dispatch negotiation.
func WorkerCapabilitiesSatisfyForSpec(plan ArtifactPlanV1, task, runner string, policyVersions, artifactVersions []string) bool {
	if err := ValidateArtifactPlanV1(plan, task, runner); err != nil {
		return false
	}
	return containsEvery(policyVersions, plan.RequiredWorkerCapabilities.PolicyContractVersions) &&
		containsEvery(artifactVersions, plan.RequiredWorkerCapabilities.ArtifactPlanVersions)
}

func WorkerCapabilitiesSatisfyRequirements(plan ArtifactPlanV1, policyVersions, artifactVersions []string) bool {
	if plan.SchemaVersion != ArtifactPlanSchemaVersionV1 {
		return false
	}
	for _, required := range plan.RequiredWorkerCapabilities.PolicyContractVersions {
		if required != WorkerPolicyContractV1 {
			return false
		}
	}
	for _, required := range plan.RequiredWorkerCapabilities.ArtifactPlanVersions {
		if required != WorkerArtifactPlanV1 {
			return false
		}
	}
	return containsEvery(policyVersions, plan.RequiredWorkerCapabilities.PolicyContractVersions) &&
		containsEvery(artifactVersions, plan.RequiredWorkerCapabilities.ArtifactPlanVersions)
}

func NegotiatedWorkerCapabilityVersions(plan ArtifactPlanV1, policyVersions, artifactVersions []string) (string, string) {
	policy := negotiatedCapabilityVersion(plan.RequiredWorkerCapabilities.PolicyContractVersions, policyVersions, WorkerPolicyContractV1)
	artifact := negotiatedCapabilityVersion(plan.RequiredWorkerCapabilities.ArtifactPlanVersions, artifactVersions, WorkerArtifactPlanV1)
	return policy, artifact
}

func negotiatedCapabilityVersion(required, advertised []string, current string) string {
	if contains(required, current) && contains(advertised, current) {
		return current
	}
	if len(required) == 0 && contains(advertised, current) {
		return current
	}
	return ""
}

func artifactPlanHash(plan ArtifactPlanV1) (string, error) {
	plan.ArtifactPlanHash = ""
	return CanonicalJSONHash(plan)
}

func automaticArtifactFormats(task, runner string) []string {
	switch task + "/" + runner {
	case "image_classification/modal_torchvision":
		return []string{"onnx", "torchscript", "pytorch"}
	case "object_detection/modal_ultralytics":
		return []string{"onnx", "pytorch"}
	default:
		return []string{}
	}
}

func automaticFallbackFormats(task, runner string) []string {
	if task == "object_detection" && runner == "modal_ultralytics" {
		return []string{"pytorch"}
	}
	return []string{}
}

// A legacy local-simulator champion can still be sent to the real model export
// worker. This applies only to manual export realization; automatic training
// outputs remain tied to their actual training runner.
func artifactCapabilityRunner(task, runner string, automatic bool) string {
	if automatic || runner != "local_simulator" {
		return runner
	}
	switch task {
	case "image_classification":
		return "modal_torchvision"
	case "object_detection":
		return "modal_ultralytics"
	default:
		return runner
	}
}

func artifactDirectiveForFormat(task, runner, format string, permitted map[string][]string) (ArtifactDirective, error) {
	formatEntry, ok := catalog.Resolve("export_formats", format)
	if !ok || !formatEntry.Available || !catalogEntryMatches(formatEntry.Tasks, formatEntry.Runners, task, runner) {
		return ArtifactDirective{}, fmt.Errorf("unknown or unavailable artifact format %q for %s/%s", format, task, runner)
	}
	runtime := map[string]string{
		"onnx": "onnxruntime", "torchscript": "torchscript", "pytorch": "pytorch", "safetensors": "pytorch",
	}[formatEntry.ID]
	if runtime == "" {
		return ArtifactDirective{}, fmt.Errorf("artifact format %q has no supported runtime", formatEntry.ID)
	}
	for category, id := range map[string]string{
		"precisions": "fp32", "runtimes": runtime,
		"execution_providers": "cpu_execution_provider", "execution_requirements": "cpu_compatible",
	} {
		if _, ok := permittedCatalogID(permitted, category, id); !ok {
			return ArtifactDirective{}, fmt.Errorf("artifact format %q requires policy-permitted %s/%s", formatEntry.ID, category, id)
		}
	}
	return ArtifactDirective{
		Format: formatEntry.ID, Precision: "fp32", Runtime: runtime,
		ExecutionProvider: "cpu_execution_provider", ExecutionRequirements: []string{"cpu_compatible"},
	}, nil
}

func availableArtifactCatalog(task, runner string) map[string][]string {
	out := map[string][]string{}
	for _, category := range []string{"export_formats", "precisions", "runtimes", "execution_providers", "execution_requirements"} {
		for _, entry := range catalog.CatalogV1().Categories[category] {
			if entry.Available && catalogEntryMatches(entry.Tasks, entry.Runners, task, runner) {
				out[category] = append(out[category], entry.ID)
			}
		}
	}
	return out
}

func catalogEntryMatches(tasks, runners []string, task, runner string) bool {
	return (len(tasks) == 0 || contains(tasks, task)) && (len(runners) == 0 || contains(runners, runner))
}

func permittedCatalogID(permitted map[string][]string, category, requested string) (string, bool) {
	entry, ok := catalog.Resolve(category, requested)
	if !ok || !entry.Available {
		return "", false
	}
	for _, id := range permitted[category] {
		if id == entry.ID {
			return entry.ID, true
		}
	}
	return "", false
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func containsEvery(advertised, required []string) bool {
	for _, value := range required {
		if !contains(advertised, value) {
			return false
		}
	}
	return true
}
