package policies

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/plans"
)

const PromptPolicyCardSchemaVersionV1 = "model_express_prompt_policy_card.v1"

func PromptCardFromEffectivePolicy(effective EffectivePolicy) PromptPolicyCard {
	return PromptPolicyCard{
		SchemaVersion:         PromptPolicyCardSchemaVersionV1,
		CatalogVersion:        effective.CatalogVersion,
		EffectivePolicyHash:   effective.EffectivePolicyHash,
		ImplicitProfile:       cloneJSON(effective.Snapshot.ImplicitProfile),
		CompatibilityProfiles: cloneJSON(effective.Snapshot.CompatibilityProfiles),
		PolicySources:         cloneJSON(effective.Snapshot.PolicySources),
		PermittedCatalog:      cloneJSON(effective.PermittedCatalog),
		FieldDenials:          promptFieldDenials(effective.Snapshot.FieldDenials),
		ResolvedDefaults:      cloneJSON(effective.Snapshot.ResolvedDefaults),
		RequiredCatalogs:      append([]string(nil), effective.Snapshot.RequiredCatalogs...),
		Instruction:           "Select every proposal capability exclusively from permitted_catalog and exclude exact values in field_denials; historical capabilities are non-actionable unless present here.",
	}
}

func promptFieldDenials(denials []FieldDenial) []FieldDenial {
	out := make([]FieldDenial, 0, len(denials))
	for _, denial := range denials {
		var value any
		decoder := json.NewDecoder(strings.NewReader(denial.CanonicalValue))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err == nil && len(fieldValueCapabilities(denial.Field, value)) > 0 {
			continue
		}
		out = append(out, denial)
	}
	return cloneJSON(out)
}

// ImplicitPromptPolicyCard preserves the no-binding allow_all_v0 proposal
// behavior for direct agent callers that do not have a policy repository.
func ImplicitPromptPolicyCard(task, runner string) PromptPolicyCard {
	return PromptCardFromEffectivePolicy(ImplicitEffectivePolicy(task, runner))
}

func ImplicitEffectivePolicy(task, runner string) EffectivePolicy {
	document := catalog.CatalogV1()
	context := ScopeContext{AccountID: LocalDefaultAccountID, Task: strings.TrimSpace(task), Runner: strings.TrimSpace(runner)}
	sets := map[string]map[string]catalog.Entry{}
	for category, entries := range document.Categories {
		sets[category] = map[string]catalog.Entry{}
		for _, entry := range entries {
			if entry.Available && entryMatchesImplicitContext(category, entry, context) {
				sets[category][entry.ID] = entry
			}
		}
	}
	resolvedDefaults, _ := resolveDefaults(context, sets, sets)
	snapshot := EffectiveSnapshot{
		SchemaVersion: EffectiveSnapshotSchemaVersionV1, CatalogVersion: document.CatalogVersion, Context: context,
		ImplicitProfile:       &ProfileRef{ID: ImplicitAllowAllProfileKey, Version: ImplicitAllowAllProfileVersion},
		CompatibilityProfiles: []ProfileSource{}, PolicySources: []PolicySource{},
		PermittedCatalog: entrySetIDs(sets), DeniedCatalog: map[string][]string{}, FieldDenials: []FieldDenial{},
		ResolvedDefaults: resolvedDefaults, RequiredCatalogs: requiredCatalogs(sets),
	}
	_, hash, _ := canonicalJSONAndHash(snapshot)
	return EffectivePolicy{
		Decision: DecisionAllowed, CatalogVersion: document.CatalogVersion, EffectivePolicyHash: hash,
		Snapshot: snapshot, PermittedCatalog: entrySetsAsSlices(sets), PermittedCounts: entrySetCounts(sets),
	}
}

func IsPermitted(effective EffectivePolicy, category, requestedID string) bool {
	entry, ok := catalog.Resolve(category, requestedID)
	if !ok {
		return false
	}
	for _, permitted := range effective.PermittedCatalog[category] {
		if permitted.ID == entry.ID {
			return true
		}
	}
	return false
}

// IsFieldValuePermitted applies the canonical exact-value field denials in an
// effective policy. It is used to prune deterministic prompt hints and AutoML
// choice lists before a candidate is materialized.
func IsFieldValuePermitted(effective EffectivePolicy, field string, value any) bool {
	canonical, err := canonicalValueString(value)
	if err != nil {
		return false
	}
	for _, denial := range effective.Snapshot.FieldDenials {
		if denial.Field == field && denial.CanonicalValue == canonical {
			return false
		}
	}
	return true
}

// EvaluateProposal validates canonical requested and realized proposal
// capabilities against one already-resolved effective policy. It does not
// write audit state; callers can enrich and persist the returned Evaluation.
func EvaluateProposal(effective EffectivePolicy, operation string, experiments []plans.PlannedExperiment) (Evaluation, error) {
	evaluation, err := EvaluationFromEffectivePolicy(effective, operation, "", "")
	if err != nil {
		return Evaluation{}, err
	}
	configJSON, configHash, err := canonicalJSONAndHash(experiments)
	if err != nil {
		return Evaluation{}, fmt.Errorf("hash proposal configuration: %w", err)
	}
	_ = configJSON
	evaluation.CandidateConfigHash = configHash
	evaluation.Decision = DecisionAllowed
	evaluation.Findings = []Finding{}
	evaluation.ReasonCodes = []ReasonCode{}

	requested := []CapabilityUse{}
	realized := []CapabilityUse{}
	findings := []Finding{}
	for index, experiment := range experiments {
		uses, fieldValues := proposalExperimentUses(experiment, index)
		for _, use := range uses {
			canonicalUse, entry, useErr := canonicalizeProposalUse(use)
			if useErr != nil {
				findings = append(findings, proposalUnknownFinding(use))
				continue
			}
			requested = append(requested, canonicalUse)
			if !IsPermitted(effective, canonicalUse.Catalog, canonicalUse.ID) {
				findings = append(findings, proposalDeniedFinding(effective, canonicalUse))
				continue
			}
			realized = append(realized, canonicalUse)
			realized = append(realized, impliedCapabilityUses(entry, canonicalUse.FieldPath)...)
		}
		findings = append(findings, deniedFieldValueFindings(effective.Snapshot.FieldDenials, fieldValues, index)...)
	}
	for _, resolved := range effective.Snapshot.ResolvedDefaults {
		use := CapabilityUse{Catalog: resolved.Catalog, ID: resolved.ID, FieldPath: resolved.Field, Origin: resolved.Origin}
		if IsPermitted(effective, use.Catalog, use.ID) {
			realized = append(realized, use)
		}
	}
	if task := effective.Snapshot.Context.Task; task != "" {
		realized = append(realized, CapabilityUse{Catalog: "tasks", ID: task, FieldPath: "task", Origin: "fixed"})
	}
	if runner := effective.Snapshot.Context.Runner; runner != "" {
		realized = append(realized, CapabilityUse{Catalog: "runners", ID: runner, FieldPath: "runner", Origin: "fixed"})
	}
	evaluation.RequestedCapabilityUses = uniqueSortedCapabilityUses(requested)
	evaluation.EffectiveCapabilityUses = uniqueSortedCapabilityUses(realized)
	evaluation.Findings = uniqueSortedFindings(findings)
	evaluation.ReasonCodes = findingReasonCodes(evaluation.Findings)
	if len(evaluation.Findings) == 0 {
		return evaluation, nil
	}
	evaluation.Decision = DecisionDenied
	code := evaluation.Findings[0].Code
	return evaluation, &PolicyError{
		Code: code, Message: "proposal contains capabilities excluded by the effective experiment policy",
		EffectivePolicyHash: effective.EffectivePolicyHash, Findings: append([]Finding(nil), evaluation.Findings...),
		ContributingScopes: scopeContributions(evaluation.Findings),
	}
}

type proposalFieldValue struct {
	Value   any
	Present bool
}

func proposalExperimentUses(experiment plans.PlannedExperiment, index int) ([]CapabilityUse, map[string]proposalFieldValue) {
	prefix := fmt.Sprintf("experiments[%d].", index)
	uses := []CapabilityUse{}
	add := func(category, value, path string) {
		if strings.TrimSpace(value) != "" {
			uses = append(uses, CapabilityUse{Catalog: category, ID: value, FieldPath: prefix + path, Origin: "explicit"})
		}
	}
	add("models", experiment.Model, "model")
	add("optimizers", experiment.Optimizer, "optimizer")
	add("schedulers", experiment.Scheduler, "scheduler")
	add("resolution_strategies", experiment.ResolutionStrategy, "resolution_strategy")
	add("augmentation_policies", experiment.AugmentationPolicy, "augmentation_policy")
	add("class_balancing_strategies", experiment.ClassBalancing, "class_balancing")
	add("sampling_strategies", experiment.SamplingStrategy, "sampling_strategy")
	add("fine_tuning_modes", experiment.FineTuneStrategy, "fine_tune_strategy")
	if experiment.AugmentationPolicyConfig != nil {
		add("augmentation_policies", experiment.AugmentationPolicyConfig.PolicyType, "augmentation_policy_config.policy_type")
	}
	if experiment.Preprocessing != nil {
		add("resize_strategies", experiment.Preprocessing.ResizeStrategy, "preprocessing.resize_strategy")
		add("crop_strategies", experiment.Preprocessing.CropStrategy, "preprocessing.crop_strategy")
		add("bounding_box_modes", experiment.Preprocessing.BBoxMode, "preprocessing.bbox_mode")
		add("normalization_strategies", experiment.Preprocessing.Normalization, "preprocessing.normalization")
		if experiment.Preprocessing.UseDatasetNormalization {
			uses = append(uses, CapabilityUse{Catalog: "normalization_strategies", ID: "dataset", FieldPath: prefix + "preprocessing.use_dataset_normalization", Origin: "alias"})
		}
	}
	for operation, raw := range experiment.Augmentation {
		enabled, ok := raw.(bool)
		if ok && enabled {
			add("augmentation_operations", operation, "augmentation."+operation)
		}
	}
	fields := proposalScalarFields(experiment)
	if value, ok := fields["preprocessing.use_dataset_normalization"]; ok {
		fields["use_dataset_normalization"] = value
	}
	return uses, fields
}

func proposalScalarFields(experiment plans.PlannedExperiment) map[string]proposalFieldValue {
	out := map[string]proposalFieldValue{}
	blob, err := json.Marshal(experiment)
	if err != nil {
		return out
	}
	var root map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(blob)))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return out
	}
	var visit func(string, any)
	visit = func(path string, value any) {
		switch typed := value.(type) {
		case map[string]any:
			keys := make([]string, 0, len(typed))
			for key := range typed {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				next := key
				if path != "" {
					next = path + "." + key
				}
				visit(next, typed[key])
			}
		case string, bool, json.Number, float64:
			out[path] = proposalFieldValue{Value: typed, Present: true}
		}
	}
	visit("", root)
	return out
}

func canonicalizeProposalUse(use CapabilityUse) (CapabilityUse, catalog.Entry, error) {
	entry, ok := catalog.Resolve(use.Catalog, use.ID)
	if !ok {
		return CapabilityUse{}, catalog.Entry{}, fmt.Errorf("unknown %s/%s", use.Catalog, use.ID)
	}
	canonical := use
	canonical.ID = entry.ID
	if !strings.EqualFold(strings.TrimSpace(use.ID), entry.ID) {
		canonical.Origin = "alias"
	}
	return canonical, entry, nil
}

func proposalUnknownFinding(use CapabilityUse) Finding {
	return Finding{Code: ReasonUnknownCatalogIdentifier, Catalog: use.Catalog, ID: strings.TrimSpace(use.ID), FieldPath: use.FieldPath,
		Origin: use.Origin, Remediation: "Choose an identifier present in the effective permitted catalog."}
}

func proposalDeniedFinding(effective EffectivePolicy, use CapabilityUse) Finding {
	for _, finding := range effective.Findings {
		if finding.Catalog == use.Catalog && finding.ID == use.ID {
			copy := finding
			copy.FieldPath = use.FieldPath
			if copy.Origin == "" {
				copy.Origin = use.Origin
			}
			return copy
		}
	}
	return Finding{Code: ReasonCatalogIDDenied, Catalog: use.Catalog, ID: use.ID, FieldPath: use.FieldPath, Origin: use.Origin,
		Remediation: "Choose a capability present in the effective permitted catalog."}
}

func impliedCapabilityUses(entry catalog.Entry, fieldPath string) []CapabilityUse {
	out := []CapabilityUse{}
	seen := map[string]bool{}
	var visit func(string)
	visit = func(raw string) {
		parts := strings.SplitN(raw, "/", 2)
		if len(parts) != 2 || seen[raw] {
			return
		}
		seen[raw] = true
		implied, ok := catalog.Resolve(parts[0], parts[1])
		if !ok {
			return
		}
		out = append(out, CapabilityUse{Catalog: parts[0], ID: implied.ID, FieldPath: fieldPath, Origin: "implied"})
		for _, nested := range append(append([]string{}, implied.Implies...), implied.EquivalentTo...) {
			visit(nested)
		}
	}
	for _, raw := range append(append([]string{}, entry.Implies...), entry.EquivalentTo...) {
		visit(raw)
	}
	return out
}

func deniedFieldValueFindings(denials []FieldDenial, fields map[string]proposalFieldValue, experimentIndex int) []Finding {
	out := []Finding{}
	for _, denial := range denials {
		field, ok := fields[denial.Field]
		if !ok || !field.Present {
			continue
		}
		canonical, err := canonicalValueString(field.Value)
		if err != nil || canonical != denial.CanonicalValue {
			continue
		}
		out = append(out, Finding{
			Code: ReasonFieldValueDenied, ID: canonical, FieldPath: fmt.Sprintf("experiments[%d].%s", experimentIndex, denial.Field), Origin: "explicit",
			Scope: denial.Scope, SubjectID: denial.SubjectID, PolicyVersionID: denial.PolicyVersionID, RuleID: denial.RuleID,
			Remediation: "Choose a field value permitted by every inherited policy scope.",
		})
	}
	return out
}

func uniqueSortedCapabilityUses(values []CapabilityUse) []CapabilityUse {
	seen := map[string]CapabilityUse{}
	for _, value := range values {
		key := value.Catalog + "/" + value.ID + "/" + value.FieldPath + "/" + value.Origin
		seen[key] = value
	}
	out := make([]CapabilityUse, 0, len(seen))
	for _, value := range seen {
		out = append(out, value)
	}
	sort.Slice(out, func(i, j int) bool {
		left, _ := json.Marshal(out[i])
		right, _ := json.Marshal(out[j])
		return string(left) < string(right)
	})
	return out
}

func uniqueSortedFindings(values []Finding) []Finding {
	seen := map[string]Finding{}
	for _, value := range values {
		blob, _ := json.Marshal(value)
		seen[string(blob)] = value
	}
	out := make([]Finding, 0, len(seen))
	for _, value := range seen {
		out = append(out, value)
	}
	sortFindings(out)
	return out
}

func findingReasonCodes(findings []Finding) []ReasonCode {
	seen := map[ReasonCode]bool{}
	for _, finding := range findings {
		seen[finding.Code] = true
	}
	out := make([]ReasonCode, 0, len(seen))
	for code := range seen {
		out = append(out, code)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
