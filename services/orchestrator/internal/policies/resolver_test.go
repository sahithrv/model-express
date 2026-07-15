package policies_test

import (
	"errors"
	"reflect"
	"sort"
	"testing"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

type policyFixture struct {
	store     *store.MemoryStore
	projectID string
	datasetID string
	jobID     string
}

func newPolicyFixture(t *testing.T) policyFixture {
	t.Helper()
	memory := store.NewMemoryStore()
	project, err := memory.CreateProject("policy fixture", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memory.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	job, err := memory.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	return policyFixture{store: memory, projectID: project.ID, datasetID: dataset.ID, jobID: job.ID}
}

func (f policyFixture) context() policies.ScopeContext {
	return policies.ScopeContext{
		AccountID: policies.LocalDefaultAccountID, ProjectID: f.projectID,
		DatasetID: f.datasetID, ExperimentJobID: f.jobID,
	}
}

func createPolicyVersion(t *testing.T, f policyFixture, refs []policies.ProfileRef, rules ...policies.Rule) policies.PolicyVersion {
	t.Helper()
	version, err := f.store.CreateExperimentPolicyVersion(policies.PolicyVersion{
		OwnerAccountID: policies.LocalDefaultAccountID,
		Document: policies.PolicyDocument{
			SchemaVersion: policies.PolicySchemaVersionV1,
			ProfileRefs:   refs,
			Rules:         rules,
		},
		CreatedBy: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	return version
}

func bindPolicy(t *testing.T, f policyFixture, scope policies.Scope, subjectID string, version policies.PolicyVersion, expected int64) policies.Binding {
	t.Helper()
	binding, err := f.store.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: scope, SubjectID: subjectID, PolicyVersionID: version.ID,
		ExpectedRevision: &expected, CreatedBy: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	return binding
}

func denyIDs(ruleID string, category string, ids ...string) policies.Rule {
	return policies.Rule{
		ID: ruleID, Effect: policies.EffectDeny,
		Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: category, IDs: ids},
	}
}

func createProfile(t *testing.T, f policyFixture, key string, modelIDs []string) policies.CompatibilityProfile {
	t.Helper()
	document := allowAllProfileDocument()
	for index := range document.Rules {
		if document.Rules[index].Selector.Catalog == "models" && modelIDs != nil {
			document.Rules[index].Selector.IDs = append([]string(nil), modelIDs...)
		}
	}
	profile, err := f.store.CreateCompatibilityProfile(policies.CompatibilityProfile{
		ProfileKey: key, SemanticVersion: "1.0.0", Document: document, CreatedBy: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	return profile
}

func allowAllProfileDocument() policies.CompatibilityProfileDocument {
	categories := make([]string, 0, len(catalog.CatalogV1().Categories))
	for category := range catalog.CatalogV1().Categories {
		categories = append(categories, category)
	}
	sort.Strings(categories)
	rules := make([]policies.Rule, 0, len(categories))
	for _, category := range categories {
		rules = append(rules, policies.Rule{
			ID: "allow_" + category, Effect: policies.EffectAllow,
			Selector: policies.Selector{
				Kind: policies.SelectorCatalogIDs, Catalog: category,
				IDs: catalog.CanonicalIDs(category, false),
			},
		})
	}
	return policies.CompatibilityProfileDocument{
		SchemaVersion:  policies.CompatibilityProfileSchemaVersionV1,
		CatalogVersion: catalog.CatalogV1().CatalogVersion,
		Rules:          rules,
	}
}

func permitted(result policies.EffectivePolicy, category string, id string) bool {
	for _, entry := range result.PermittedCatalog[category] {
		if entry.ID == id {
			return true
		}
	}
	return false
}

func TestNoBindingsResolveToImplicitAllowAllV0(t *testing.T) {
	fixture := newPolicyFixture(t)
	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != policies.DecisionAllowed || result.Snapshot.ImplicitProfile == nil {
		t.Fatalf("unexpected no-policy result: %#v", result)
	}
	if result.Snapshot.ImplicitProfile.ID != policies.ImplicitAllowAllProfileKey {
		t.Fatalf("implicit profile = %#v", result.Snapshot.ImplicitProfile)
	}
	for category, entries := range catalog.CatalogV1().Categories {
		available := 0
		for _, entry := range entries {
			if entry.Available {
				available++
			}
		}
		if result.PermittedCounts[category] != available {
			t.Fatalf("%s permitted count = %d, want %d", category, result.PermittedCounts[category], available)
		}
	}
	if permitted(result, "precisions", "fp16") || !permitted(result, "precisions", "fp32") {
		t.Fatalf("platform availability was not an upper bound: %#v", result.Snapshot.PermittedCatalog["precisions"])
	}
}

func TestImplicitAllowAllV0PreservesHistoricallyAvailableTaskCapabilities(t *testing.T) {
	fixture := newPolicyFixture(t)
	context := fixture.context()
	context.Task = "image_classification"
	context.Runner = "local_simulator"
	result, err := policies.NewResolver(fixture.store).Resolve(context)
	if err != nil {
		t.Fatal(err)
	}
	for category, id := range map[string]string{
		"augmentation_policies": "mixup",
		"fine_tuning_modes":     "full",
		"crop_strategies":       "bbox_crop_ablation",
	} {
		if !permitted(result, category, id) {
			t.Fatalf("implicit allow_all_v0 removed historically accepted %s/%s", category, id)
		}
	}
	if permitted(result, "precisions", "fp16") {
		t.Fatal("implicit allow_all_v0 admitted a catalog-unavailable capability")
	}
}

func TestScopeInheritancePrecedenceAndDenyWins(t *testing.T) {
	fixture := newPolicyFixture(t)
	profile := createProfile(t, fixture, "allow_all_test", nil)

	account := createPolicyVersion(t, fixture, nil, denyIDs("deny_convnext", "models", "convnext_tiny"))
	project := createPolicyVersion(t, fixture, nil, denyIDs("deny_swin", "models", "swin_t"))
	dataset := createPolicyVersion(t, fixture, []policies.ProfileRef{{ID: profile.ProfileKey, Version: profile.SemanticVersion}}, denyIDs("deny_vit", "models", "vit_b_16"))
	run := createPolicyVersion(t, fixture, nil, denyIDs("deny_resnet", "models", "resnet34"))
	bindPolicy(t, fixture, policies.ScopeAccount, policies.LocalDefaultAccountID, account, 0)
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, project, 0)
	bindPolicy(t, fixture, policies.ScopeDataset, fixture.datasetID, dataset, 0)
	bindPolicy(t, fixture, policies.ScopeRun, fixture.jobID, run, 0)

	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"convnext_tiny", "swin_t", "vit_b_16", "resnet34"} {
		if permitted(result, "models", id) {
			t.Fatalf("inherited denial did not remove model %s", id)
		}
	}
	if !permitted(result, "models", "resnet18") {
		t.Fatal("unrelated model was removed")
	}
	wantScopes := []policies.Scope{policies.ScopeAccount, policies.ScopeProject, policies.ScopeDataset, policies.ScopeRun}
	gotScopes := make([]policies.Scope, 0, len(result.Snapshot.PolicySources))
	for _, source := range result.Snapshot.PolicySources {
		gotScopes = append(gotScopes, source.Scope)
	}
	if !reflect.DeepEqual(gotScopes, wantScopes) {
		t.Fatalf("policy source precedence = %v, want %v", gotScopes, wantScopes)
	}
	if result.Snapshot.ImplicitProfile != nil {
		t.Fatal("explicit profile should replace the implicit profile")
	}
}

func TestExplicitDenyOnlyPolicyDoesNotAdvertiseImplicitProfile(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, denyIDs("deny_resnet", "models", "resnet18"))
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Snapshot.ImplicitProfile != nil || len(result.Snapshot.PolicySources) != 1 {
		t.Fatalf("explicit deny-only policy was represented as implicit: %#v", result.Snapshot)
	}
}

func TestCompatibilityProfilesAreClosedWorldAndIntersect(t *testing.T) {
	fixture := newPolicyFixture(t)
	first := createProfile(t, fixture, "profile_first", []string{"mobilenet_v3_small", "resnet18"})
	second := createProfile(t, fixture, "profile_second", []string{"resnet18", "convnext_tiny"})
	version := createPolicyVersion(t, fixture, []policies.ProfileRef{
		{ID: second.ProfileKey, Version: second.SemanticVersion},
		{ID: first.ProfileKey, Version: first.SemanticVersion},
	})
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)

	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Snapshot.PermittedCatalog["models"]; !reflect.DeepEqual(got, []string{"resnet18"}) {
		t.Fatalf("profile intersection models = %#v", got)
	}
	if permitted(result, "models", "mobilenet_v3_large") {
		t.Fatal("closed-world profile admitted an unlisted supported model")
	}
}

func TestSelectorsAliasesImplicationsAndEquivalentFieldCannotBypass(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil,
		policies.Rule{
			ID: "deny_convnext_family", Effect: policies.EffectDeny,
			Selector: policies.Selector{
				Kind: policies.SelectorCatalogAttributeValues, Catalog: "models",
				Attribute: "family", Values: []any{"convnext"},
			},
		},
		denyIDs("deny_erasing", "augmentation_operations", "random_erasing"),
		policies.Rule{
			ID: "deny_dataset_normalization_alias", Effect: policies.EffectDeny,
			Selector: policies.Selector{
				Kind:  policies.SelectorFieldValues,
				Field: "preprocessing.use_dataset_normalization", Values: []any{true},
			},
		},
	)
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if permitted(result, "models", "convnext_tiny") {
		t.Fatal("family attribute selector did not remove convnext_tiny")
	}
	if permitted(result, "augmentation_policies", "strong") {
		t.Fatal("strong remained available despite implying denied random_erasing")
	}
	if permitted(result, "normalization_strategies", "dataset") {
		t.Fatal("dataset normalization semantic alias bypassed field denial")
	}
	if !permitted(result, "normalization_strategies", "imagenet") {
		t.Fatal("unrelated normalization strategy was removed")
	}
}

func TestCatalogBackedFieldDenialIsRemovedFromSelectablePromptCatalog(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, policies.Rule{
		ID: "deny_model_field", Effect: policies.EffectDeny,
		Selector: policies.Selector{Kind: policies.SelectorFieldValues, Field: "model", Values: []any{"convnext_tiny"}},
	})
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	context := fixture.context()
	context.Task = "image_classification"
	context.Runner = "local_simulator"
	result, err := policies.NewResolver(fixture.store).Resolve(context)
	if err != nil {
		t.Fatal(err)
	}
	if permitted(result, "models", "convnext_tiny") {
		t.Fatal("catalog-backed field denial remained selectable")
	}
	card := policies.PromptCardFromEffectivePolicy(result)
	for _, denial := range card.FieldDenials {
		if denial.Field == "model" {
			t.Fatalf("prompt exposed a denied catalog ID instead of representing it through the permitted list: %#v", denial)
		}
	}
}

func TestTierSelectorExpansion(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, policies.Rule{
		ID: "deny_quality_tier", Effect: policies.EffectDeny,
		Selector: policies.Selector{
			Kind: policies.SelectorCatalogAttributeValues, Catalog: "models",
			Attribute: "deployment_tier", Values: []any{"quality_challenger"},
		},
	})
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"convnext_tiny", "swin_t", "vit_b_16"} {
		if permitted(result, "models", id) {
			t.Fatalf("tier selector did not remove %s", id)
		}
	}
}

func TestDeniedExecutionDefaultUsesDeterministicPermittedFallback(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, denyIDs("deny_default_augmentation", "augmentation_policies", "none"))
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	context := fixture.context()
	context.Task = "image_classification"
	context.Runner = "modal_torchvision"
	result, err := policies.NewResolver(fixture.store).Resolve(context)
	if err != nil {
		t.Fatal(err)
	}
	if permitted(result, "augmentation_policies", "none") {
		t.Fatal("denied execution default was restored to the effective catalog")
	}
	var found *policies.ResolvedDefault
	for index := range result.Snapshot.ResolvedDefaults {
		candidate := &result.Snapshot.ResolvedDefaults[index]
		if candidate.Field == "augmentation_policy" {
			found = candidate
			break
		}
	}
	if found == nil || found.Origin != "policy_fallback" || found.OriginalID != "none" || found.ID == "none" {
		t.Fatalf("resolved augmentation default = %#v", found)
	}
	if !permitted(result, found.Catalog, found.ID) {
		t.Fatalf("fallback default is not permitted: %#v", found)
	}
}

func TestCanonicalizationHashAndTypedFieldValidation(t *testing.T) {
	docA := policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		ProfileRefs:   []policies.ProfileRef{{ID: "profile_b", Version: "1.0.0"}, {ID: "profile_a", Version: "1.0.0"}},
		Rules: []policies.Rule{
			denyIDs("deny_resize", "resize_strategies", "letterbox", "squash"),
			denyIDs("deny_provider", "execution_providers", "CPUExecutionProvider"),
		},
	}
	docB := policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		ProfileRefs:   []policies.ProfileRef{{ID: "profile_a", Version: "1.0.0"}, {ID: "profile_b", Version: "1.0.0"}},
		Rules: []policies.Rule{
			denyIDs("deny_provider", "execution_providers", "cpu_execution_provider"),
			denyIDs("deny_resize", "resize_strategies", "squash", "yolo_letterbox"),
		},
	}
	normalized, _, hashA, err := policies.NormalizePolicyDocument(docA)
	if err != nil {
		t.Fatal(err)
	}
	_, _, hashB, err := policies.NormalizePolicyDocument(docB)
	if err != nil {
		t.Fatal(err)
	}
	if hashA != hashB {
		t.Fatalf("canonical document hashes differ: %s != %s", hashA, hashB)
	}
	if got := normalized.Rules[1].Selector.IDs; !reflect.DeepEqual(got, []string{"squash", "yolo_letterbox"}) {
		t.Fatalf("resize aliases were not canonicalized: %#v", got)
	}

	fixture := newPolicyFixture(t)
	_, err = fixture.store.CreateExperimentPolicyVersion(policies.PolicyVersion{
		Document: policies.PolicyDocument{
			SchemaVersion: policies.PolicySchemaVersionV1,
			ProfileRefs:   []policies.ProfileRef{},
			Rules: []policies.Rule{{
				ID: "bad_boolean", Effect: policies.EffectDeny,
				Selector: policies.Selector{Kind: policies.SelectorFieldValues, Field: "pretrained", Values: []any{"false"}},
			}},
		},
	})
	if !errors.Is(err, store.ErrInvalidRequest) {
		t.Fatalf("string-valued boolean error = %v", err)
	}
}

func TestUnknownIdentifierAndUnavailableProfileReturnStructuredReasons(t *testing.T) {
	_, _, _, err := policies.NormalizePolicyDocument(policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		Rules:         []policies.Rule{denyIDs("unknown", "models", "not_a_model")},
	})
	var policyErr *policies.PolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonUnknownCatalogIdentifier {
		t.Fatalf("unknown identifier error = %#v", err)
	}

	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, []policies.ProfileRef{{ID: "missing_profile", Version: "9.9.9"}})
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	_, err = policies.NewResolver(fixture.store).Resolve(fixture.context())
	if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonProfileVersionUnavailable {
		t.Fatalf("missing profile error = %#v", err)
	}
}

func TestEmptySearchSpaceReturnsNoValidConfiguration(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, denyIDs("deny_all_models", "models", catalog.CanonicalIDs("models", true)...))
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	result, err := policies.NewResolver(fixture.store).Resolve(fixture.context())
	var policyErr *policies.PolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonNoValidConfiguration {
		t.Fatalf("empty space error = %#v", err)
	}
	if result.Decision != policies.DecisionDenied || result.EffectivePolicyHash == "" || !reflect.DeepEqual(result.BlockedDimensions, []string{"models"}) {
		t.Fatalf("empty-space result = %#v", result)
	}
}

func TestEffectiveHashIsStableAndEvaluationIsAppendOnly(t *testing.T) {
	fixture := newPolicyFixture(t)
	version := createPolicyVersion(t, fixture, nil, denyIDs("deny_vit", "models", "vit_b_16"))
	bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, version, 0)
	resolver := policies.NewResolver(fixture.store)
	first, err := resolver.Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	second, err := resolver.Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if first.EffectivePolicyHash != second.EffectivePolicyHash {
		t.Fatalf("effective hashes differ: %s != %s", first.EffectivePolicyHash, second.EffectivePolicyHash)
	}
	result, evaluation, err := resolver.ResolveAndRecord(fixture.context(), "preview_test", "actor", "request")
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.ID == "" || evaluation.EffectivePolicyHash != result.EffectivePolicyHash || len(evaluation.PolicySources) != 1 {
		t.Fatalf("incomplete audit evaluation: %#v", evaluation)
	}
	stored, err := fixture.store.GetExperimentPolicyEvaluation(evaluation.ID)
	if err != nil {
		t.Fatal(err)
	}
	stored.Findings = append(stored.Findings, policies.Finding{Code: policies.ReasonChangedAfterQueue})
	reloaded, err := fixture.store.GetExperimentPolicyEvaluation(evaluation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(stored.Findings) == len(reloaded.Findings) {
		t.Fatal("audit record was mutable through a returned slice")
	}
	if _, err := fixture.store.CreateExperimentPolicyEvaluation(evaluation); !errors.Is(err, policies.ErrImmutableConflict) {
		t.Fatalf("duplicate immutable evaluation error = %v", err)
	}
}

func TestBindingOptimisticConcurrency(t *testing.T) {
	fixture := newPolicyFixture(t)
	firstVersion := createPolicyVersion(t, fixture, nil, denyIDs("first", "models", "vit_b_16"))
	secondVersion := createPolicyVersion(t, fixture, nil, denyIDs("second", "models", "swin_t"))
	first := bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, firstVersion, 0)
	stale := int64(0)
	if _, err := fixture.store.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: fixture.projectID,
		PolicyVersionID: secondVersion.ID, ExpectedRevision: &stale,
	}); !errors.Is(err, policies.ErrRevisionConflict) {
		t.Fatalf("stale binding write error = %v", err)
	}
	second := bindPolicy(t, fixture, policies.ScopeProject, fixture.projectID, secondVersion, first.Revision)
	if second.Revision != 2 || second.SupersedesID != first.ID {
		t.Fatalf("superseding binding = %#v", second)
	}
	active, err := fixture.store.ListActiveExperimentPolicyBindings(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if len(active) != 1 || active[0].PolicyVersionID != secondVersion.ID {
		t.Fatalf("active bindings = %#v", active)
	}
}

func TestRemovingChildBindingRestoresOnlyChildCapabilities(t *testing.T) {
	fixture := newPolicyFixture(t)
	account := createPolicyVersion(t, fixture, nil, denyIDs("deny_vit", "models", "vit_b_16"))
	dataset := createPolicyVersion(t, fixture, nil, denyIDs("deny_swin", "models", "swin_t"))
	bindPolicy(t, fixture, policies.ScopeAccount, policies.LocalDefaultAccountID, account, 0)
	child := bindPolicy(t, fixture, policies.ScopeDataset, fixture.datasetID, dataset, 0)
	resolver := policies.NewResolver(fixture.store)
	before, err := resolver.Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if permitted(before, "models", "vit_b_16") || permitted(before, "models", "swin_t") {
		t.Fatal("expected parent and child denials before clearing child binding")
	}
	cleared, err := fixture.store.ClearExperimentPolicyBinding(policies.ScopeDataset, fixture.datasetID, child.Revision)
	if err != nil {
		t.Fatal(err)
	}
	if cleared.Active || cleared.SupersededAt == nil {
		t.Fatalf("cleared binding = %#v", cleared)
	}
	after, err := resolver.Resolve(fixture.context())
	if err != nil {
		t.Fatal(err)
	}
	if permitted(after, "models", "vit_b_16") || !permitted(after, "models", "swin_t") {
		t.Fatalf("clearing child widened the wrong scope: %#v", after.Snapshot.PermittedCatalog["models"])
	}
}
