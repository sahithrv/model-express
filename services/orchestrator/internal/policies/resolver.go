package policies

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/execution"
)

type Resolver struct {
	repository Repository
	document   catalog.Document
}

func NewResolver(repository Repository) *Resolver {
	return &Resolver{repository: repository, document: catalog.CatalogV1()}
}

type capabilityRef struct {
	Catalog string
	ID      string
}

func (ref capabilityRef) key() string {
	return ref.Catalog + "/" + ref.ID
}

type sourcedRule struct {
	Rule            Rule
	Scope           Scope
	SubjectID       string
	PolicyVersionID string
}

func (r *Resolver) Resolve(input ScopeContext) (EffectivePolicy, error) {
	if r == nil || r.repository == nil {
		return EffectivePolicy{}, fmt.Errorf("policy resolver repository is required")
	}
	context, err := r.normalizeContext(input)
	if err != nil {
		return EffectivePolicy{}, err
	}
	bindings, err := r.repository.ListActiveExperimentPolicyBindings(context)
	if err != nil {
		return EffectivePolicy{}, err
	}
	sort.Slice(bindings, func(i, j int) bool {
		if ScopeRank(bindings[i].Scope) != ScopeRank(bindings[j].Scope) {
			return ScopeRank(bindings[i].Scope) < ScopeRank(bindings[j].Scope)
		}
		if bindings[i].SubjectID != bindings[j].SubjectID {
			return bindings[i].SubjectID < bindings[j].SubjectID
		}
		return bindings[i].Revision < bindings[j].Revision
	})

	policySources := make([]PolicySource, 0, len(bindings))
	denyRules := []sourcedRule{}
	profileRefs := map[string]ProfileRef{}
	for _, binding := range bindings {
		if !binding.Active {
			continue
		}
		version, getErr := r.repository.GetExperimentPolicyVersion(binding.PolicyVersionID)
		if getErr != nil {
			return EffectivePolicy{}, fmt.Errorf("load policy version %s: %w", binding.PolicyVersionID, getErr)
		}
		policySources = append(policySources, PolicySource{
			BindingID:       binding.ID,
			BindingRevision: binding.Revision,
			Scope:           binding.Scope,
			SubjectID:       binding.SubjectID,
			PolicyVersionID: version.ID,
			PolicyRevision:  version.Revision,
			PolicyHash:      version.DocumentHash,
		})
		for _, ref := range version.Document.ProfileRefs {
			profileRefs[ref.ID+"@"+ref.Version] = ref
		}
		for _, rule := range version.Document.Rules {
			denyRules = append(denyRules, sourcedRule{
				Rule:            rule,
				Scope:           binding.Scope,
				SubjectID:       binding.SubjectID,
				PolicyVersionID: version.ID,
			})
		}
	}

	platform := r.platformCatalog(context)
	if len(policySources) == 0 {
		// The built-in allow_all_v0 profile is a compatibility shim: proposal
		// generation must retain all historically available task capabilities
		// until a user explicitly opts into policy enforcement. Execution
		// fidelity still owns runner-specific validation.
		platform = r.implicitPlatformCatalog(context)
	}
	effective := cloneEntrySets(platform)
	forbidden := map[string]Finding{}
	allFindings := []Finding{}
	profileSources := make([]ProfileSource, 0, len(profileRefs))
	refs := make([]ProfileRef, 0, len(profileRefs))
	for _, ref := range profileRefs {
		refs = append(refs, ref)
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].ID != refs[j].ID {
			return refs[i].ID < refs[j].ID
		}
		return refs[i].Version < refs[j].Version
	})
	for _, ref := range refs {
		if ref.ID == ImplicitAllowAllProfileKey && ref.Version == ImplicitAllowAllProfileVersion {
			profileSources = append(profileSources, ProfileSource{ID: ref.ID, Version: ref.Version, DocumentHash: "builtin:allow_all_v0"})
			continue
		}
		profile, getErr := r.repository.GetCompatibilityProfile(ref.ID, ref.Version)
		if getErr != nil {
			finding := Finding{
				Code:           ReasonProfileVersionUnavailable,
				ProfileKey:     ref.ID,
				ProfileVersion: ref.Version,
				Origin:         "profile_ref",
				Remediation:    "Bind an available immutable compatibility-profile version.",
			}
			return EffectivePolicy{}, &PolicyError{
				Code:     ReasonProfileVersionUnavailable,
				Message:  fmt.Sprintf("compatibility profile %s@%s is unavailable", ref.ID, ref.Version),
				Findings: []Finding{finding},
			}
		}
		if profile.CatalogVersion != r.document.CatalogVersion {
			finding := Finding{
				Code:           ReasonProfileVersionUnavailable,
				ProfileKey:     ref.ID,
				ProfileVersion: ref.Version,
				Origin:         "catalog_version",
				Remediation:    "Use a compatibility profile pinned to the active catalog version.",
			}
			return EffectivePolicy{}, &PolicyError{
				Code:     ReasonProfileVersionUnavailable,
				Message:  fmt.Sprintf("compatibility profile %s@%s targets unavailable catalog %s", ref.ID, ref.Version, profile.CatalogVersion),
				Findings: []Finding{finding},
			}
		}
		profileSources = append(profileSources, ProfileSource{ID: ref.ID, Version: ref.Version, DocumentHash: profile.DocumentHash})
		allowed := map[string]map[string]struct{}{}
		for category := range r.document.Categories {
			allowed[category] = map[string]struct{}{}
		}
		for _, rule := range profile.Document.Rules {
			for _, capability := range r.expandCatalogSelector(rule.Selector) {
				allowed[capability.Catalog][capability.ID] = struct{}{}
			}
		}
		for category, entries := range platform {
			for id := range entries {
				if _, ok := allowed[category][id]; ok {
					continue
				}
				capability := capabilityRef{Catalog: category, ID: id}
				finding := Finding{
					Code:           ReasonNotInCompatibilityProfile,
					Catalog:        category,
					ID:             id,
					Origin:         "profile",
					ProfileKey:     ref.ID,
					ProfileVersion: ref.Version,
					Remediation:    "Choose a capability allowed by every inherited compatibility profile.",
				}
				if _, exists := forbidden[capability.key()]; !exists {
					forbidden[capability.key()] = finding
				}
				allFindings = append(allFindings, finding)
				delete(effective[category], id)
			}
		}
	}

	fieldDenials := []FieldDenial{}
	for _, sourced := range denyRules {
		switch sourced.Rule.Selector.Kind {
		case SelectorCatalogIDs, SelectorCatalogAttributeValues:
			for _, capability := range r.expandCatalogSelector(sourced.Rule.Selector) {
				finding := denialFinding(sourced, capability)
				if sourced.Rule.Selector.Kind == SelectorCatalogAttributeValues {
					finding.Code = ReasonCatalogAttributeDenied
					finding.Attribute = sourced.Rule.Selector.Attribute
					for _, value := range sourced.Rule.Selector.Values {
						if text, ok := value.(string); ok && attributeContains(r.entry(capability).Attributes[finding.Attribute], text) {
							finding.AttributeValue = text
							break
						}
					}
				}
				forbidden[capability.key()] = finding
				allFindings = append(allFindings, finding)
				delete(effective[capability.Catalog], capability.ID)
			}
		case SelectorFieldValues:
			for _, value := range sourced.Rule.Selector.Values {
				valueText, _ := canonicalValueString(value)
				fieldDenials = append(fieldDenials, FieldDenial{
					Field: sourced.Rule.Selector.Field, CanonicalValue: valueText,
					Scope: sourced.Scope, SubjectID: sourced.SubjectID,
					PolicyVersionID: sourced.PolicyVersionID, RuleID: sourced.Rule.ID,
				})
				fieldFinding := Finding{
					Code: ReasonFieldValueDenied, FieldPath: sourced.Rule.Selector.Field,
					ID: valueText, Origin: "field", Scope: sourced.Scope,
					SubjectID: sourced.SubjectID, PolicyVersionID: sourced.PolicyVersionID,
					RuleID:      sourced.Rule.ID,
					Remediation: "Choose a field value permitted by every inherited policy scope.",
				}
				allFindings = append(allFindings, fieldFinding)
				for _, capability := range fieldValueCapabilities(sourced.Rule.Selector.Field, value) {
					mapped := fieldFinding
					mapped.Catalog = capability.Catalog
					mapped.ID = capability.ID
					mapped.Origin = "alias"
					forbidden[capability.key()] = mapped
					allFindings = append(allFindings, mapped)
					delete(effective[capability.Catalog], capability.ID)
				}
			}
		}
	}

	r.applyImplicationClosure(effective, forbidden, &allFindings)
	resolvedDefaults, fixedBlocked := resolveDefaults(context, platform, effective)
	required := requiredCatalogs(platform)
	blocked := []string{}
	for _, category := range required {
		if len(effective[category]) == 0 {
			blocked = append(blocked, category)
		}
	}
	for _, category := range fixedBlocked {
		if !containsString(blocked, category) {
			blocked = append(blocked, category)
		}
	}
	sort.Strings(blocked)

	permittedIDs := entrySetIDs(effective)
	deniedIDs := forbiddenIDs(forbidden)
	sort.Slice(policySources, func(i, j int) bool {
		if ScopeRank(policySources[i].Scope) != ScopeRank(policySources[j].Scope) {
			return ScopeRank(policySources[i].Scope) < ScopeRank(policySources[j].Scope)
		}
		if policySources[i].SubjectID != policySources[j].SubjectID {
			return policySources[i].SubjectID < policySources[j].SubjectID
		}
		return policySources[i].BindingRevision < policySources[j].BindingRevision
	})
	sort.Slice(profileSources, func(i, j int) bool {
		if profileSources[i].ID != profileSources[j].ID {
			return profileSources[i].ID < profileSources[j].ID
		}
		return profileSources[i].Version < profileSources[j].Version
	})
	sort.Slice(fieldDenials, func(i, j int) bool {
		left := string(fieldDenials[i].Scope) + "/" + fieldDenials[i].SubjectID + "/" + fieldDenials[i].Field + "/" + fieldDenials[i].CanonicalValue + "/" + fieldDenials[i].RuleID
		right := string(fieldDenials[j].Scope) + "/" + fieldDenials[j].SubjectID + "/" + fieldDenials[j].Field + "/" + fieldDenials[j].CanonicalValue + "/" + fieldDenials[j].RuleID
		return left < right
	})
	sortFindings(allFindings)

	snapshot := EffectiveSnapshot{
		SchemaVersion:         EffectiveSnapshotSchemaVersionV1,
		CatalogVersion:        r.document.CatalogVersion,
		Context:               context,
		CompatibilityProfiles: profileSources,
		PolicySources:         policySources,
		PermittedCatalog:      permittedIDs,
		DeniedCatalog:         deniedIDs,
		FieldDenials:          fieldDenials,
		ResolvedDefaults:      resolvedDefaults,
		RequiredCatalogs:      required,
	}
	if len(policySources) == 0 {
		snapshot.ImplicitProfile = &ProfileRef{ID: ImplicitAllowAllProfileKey, Version: ImplicitAllowAllProfileVersion}
	}
	_, effectiveHash, hashErr := canonicalJSONAndHash(snapshot)
	if hashErr != nil {
		return EffectivePolicy{}, hashErr
	}
	result := EffectivePolicy{
		Decision:            DecisionAllowed,
		CatalogVersion:      r.document.CatalogVersion,
		EffectivePolicyHash: effectiveHash,
		Snapshot:            snapshot,
		PermittedCatalog:    entrySetsAsSlices(effective),
		PermittedCounts:     entrySetCounts(effective),
		Findings:            allFindings,
		BlockedDimensions:   blocked,
	}
	if len(blocked) > 0 {
		result.Decision = DecisionDenied
		findings := make([]Finding, 0, len(blocked))
		for _, category := range blocked {
			findings = append(findings, Finding{
				Code:        ReasonNoValidConfiguration,
				Catalog:     category,
				Origin:      "satisfiability",
				Remediation: "Remove a contributing denial or select a compatible profile version.",
			})
		}
		result.Findings = append(result.Findings, findings...)
		sortFindings(result.Findings)
		result.ContributingScopes = scopeContributions(result.Findings)
		return result, &PolicyError{
			Code:                ReasonNoValidConfiguration,
			Message:             fmt.Sprintf("experiment policy leaves no valid configuration for: %s", strings.Join(blocked, ", ")),
			EffectivePolicyHash: effectiveHash,
			Findings:            append([]Finding(nil), result.Findings...),
			BlockedDimensions:   append([]string(nil), blocked...),
			ContributingScopes:  append([]ScopeContribution(nil), result.ContributingScopes...),
		}
	}
	return result, nil
}

func scopeContributions(findings []Finding) []ScopeContribution {
	seen := map[string]ScopeContribution{}
	for _, finding := range findings {
		if finding.Scope == "" || finding.SubjectID == "" || finding.PolicyVersionID == "" {
			continue
		}
		item := ScopeContribution{Scope: finding.Scope, SubjectID: finding.SubjectID, PolicyVersionID: finding.PolicyVersionID}
		seen[string(item.Scope)+"/"+item.SubjectID+"/"+item.PolicyVersionID] = item
	}
	out := make([]ScopeContribution, 0, len(seen))
	for _, item := range seen {
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool {
		if ScopeRank(out[i].Scope) != ScopeRank(out[j].Scope) {
			return ScopeRank(out[i].Scope) < ScopeRank(out[j].Scope)
		}
		if out[i].SubjectID != out[j].SubjectID {
			return out[i].SubjectID < out[j].SubjectID
		}
		return out[i].PolicyVersionID < out[j].PolicyVersionID
	})
	return out
}

// ContributingScopesFromFindings returns the stable, de-duplicated policy
// sources responsible for a set of proposal findings.
func ContributingScopesFromFindings(findings []Finding) []ScopeContribution {
	return scopeContributions(findings)
}

// BlockedDimensionsFromFindings names the catalog or field dimensions that
// prevent candidate construction without exposing denied catalog IDs.
func BlockedDimensionsFromFindings(findings []Finding) []string {
	seen := map[string]bool{}
	for _, finding := range findings {
		dimension := strings.TrimSpace(finding.Catalog)
		if dimension == "" {
			dimension = strings.TrimSpace(finding.FieldPath)
			if index := strings.LastIndex(dimension, "."); index >= 0 {
				dimension = dimension[index+1:]
			}
		}
		if dimension != "" {
			seen[dimension] = true
		}
	}
	out := make([]string, 0, len(seen))
	for dimension := range seen {
		out = append(out, dimension)
	}
	sort.Strings(out)
	return out
}

func (r *Resolver) ResolveAndRecord(input ScopeContext, operation string, actorID string, requestID string) (EffectivePolicy, Evaluation, error) {
	result, resolveErr := r.Resolve(input)
	if result.EffectivePolicyHash == "" {
		return result, Evaluation{}, resolveErr
	}
	evaluation, err := EvaluationFromEffectivePolicy(result, operation, actorID, requestID)
	if err != nil {
		return result, Evaluation{}, err
	}
	created, err := r.repository.CreateExperimentPolicyEvaluation(evaluation)
	if err != nil {
		return result, Evaluation{}, err
	}
	return result, created, resolveErr
}

func EvaluationFromEffectivePolicy(result EffectivePolicy, operation string, actorID string, requestID string) (Evaluation, error) {
	snapshot, _, err := canonicalJSONAndHash(result.Snapshot)
	if err != nil {
		return Evaluation{}, err
	}
	reasonSet := map[ReasonCode]struct{}{}
	for _, finding := range result.Findings {
		reasonSet[finding.Code] = struct{}{}
	}
	reasons := make([]ReasonCode, 0, len(reasonSet))
	for reason := range reasonSet {
		reasons = append(reasons, reason)
	}
	sort.Slice(reasons, func(i, j int) bool { return reasons[i] < reasons[j] })
	return Evaluation{
		Operation:               strings.TrimSpace(operation),
		Decision:                result.Decision,
		AccountID:               result.Snapshot.Context.AccountID,
		ProjectID:               result.Snapshot.Context.ProjectID,
		DatasetID:               result.Snapshot.Context.DatasetID,
		JobID:                   result.Snapshot.Context.ExperimentJobID,
		CatalogVersion:          result.CatalogVersion,
		CompatibilityProfiles:   append([]ProfileSource{}, result.Snapshot.CompatibilityProfiles...),
		PolicySources:           append([]PolicySource{}, result.Snapshot.PolicySources...),
		EffectiveSnapshot:       snapshot,
		EffectivePolicyHash:     result.EffectivePolicyHash,
		RequestedCapabilityUses: []CapabilityUse{},
		EffectiveCapabilityUses: []CapabilityUse{},
		ReasonCodes:             reasons,
		Findings:                append([]Finding{}, result.Findings...),
		ActorID:                 strings.TrimSpace(actorID),
		RequestID:               strings.TrimSpace(requestID),
	}, nil
}

func ValidateEvaluation(input Evaluation) error {
	if strings.TrimSpace(input.Operation) == "" {
		return fmt.Errorf("policy evaluation operation is required")
	}
	if input.Decision != DecisionAllowed && input.Decision != DecisionDenied {
		return fmt.Errorf("policy evaluation decision must be %q or %q", DecisionAllowed, DecisionDenied)
	}
	if input.CatalogVersion == "" || input.EffectivePolicyHash == "" || len(input.EffectiveSnapshot) == 0 {
		return fmt.Errorf("policy evaluation catalog version, snapshot, and hash are required")
	}
	computed, err := HashCanonicalJSON(input.EffectiveSnapshot)
	if err != nil {
		return fmt.Errorf("invalid effective policy snapshot: %w", err)
	}
	if computed != input.EffectivePolicyHash {
		return fmt.Errorf("effective policy snapshot hash mismatch: got %s want %s", computed, input.EffectivePolicyHash)
	}
	return nil
}

func (r *Resolver) normalizeContext(input ScopeContext) (ScopeContext, error) {
	input.AccountID = strings.TrimSpace(input.AccountID)
	input.ProjectID = strings.TrimSpace(input.ProjectID)
	input.DatasetID = strings.TrimSpace(input.DatasetID)
	input.ExperimentJobID = strings.TrimSpace(input.ExperimentJobID)
	if input.AccountID == "" {
		return ScopeContext{}, fmt.Errorf("policy scope account_id is required")
	}
	if input.DatasetID != "" && input.ProjectID == "" {
		return ScopeContext{}, fmt.Errorf("policy scope project_id is required with dataset_id")
	}
	if input.ExperimentJobID != "" && input.ProjectID == "" {
		return ScopeContext{}, fmt.Errorf("policy scope project_id is required with experiment_job_id")
	}
	if strings.TrimSpace(input.Task) != "" {
		entry, ok := catalog.Resolve("tasks", input.Task)
		if !ok {
			return ScopeContext{}, unknownCatalogError("tasks", strings.TrimSpace(input.Task))
		}
		input.Task = entry.ID
	}
	if strings.TrimSpace(input.Runner) != "" {
		entry, ok := catalog.Resolve("runners", input.Runner)
		if !ok {
			return ScopeContext{}, unknownCatalogError("runners", strings.TrimSpace(input.Runner))
		}
		input.Runner = entry.ID
	}
	return input, nil
}

func (r *Resolver) platformCatalog(context ScopeContext) map[string]map[string]catalog.Entry {
	out := map[string]map[string]catalog.Entry{}
	for category, entries := range r.document.Categories {
		out[category] = map[string]catalog.Entry{}
		for _, entry := range entries {
			if !entry.Available || !entryMatchesContext(category, entry, context) {
				continue
			}
			out[category][entry.ID] = entry
		}
	}
	return out
}

func entryMatchesContext(category string, entry catalog.Entry, context ScopeContext) bool {
	if context.Task != "" {
		if category == "tasks" && entry.ID != context.Task {
			return false
		}
		if len(entry.Tasks) > 0 && !containsString(entry.Tasks, context.Task) {
			return false
		}
	}
	if context.Runner != "" {
		if category == "runners" && entry.ID != context.Runner {
			return false
		}
		if len(entry.Runners) > 0 && !containsString(entry.Runners, context.Runner) {
			return false
		}
	}
	return true
}

func (r *Resolver) implicitPlatformCatalog(context ScopeContext) map[string]map[string]catalog.Entry {
	out := map[string]map[string]catalog.Entry{}
	for category, entries := range r.document.Categories {
		out[category] = map[string]catalog.Entry{}
		for _, entry := range entries {
			if !entry.Available || !entryMatchesImplicitContext(category, entry, context) {
				continue
			}
			out[category][entry.ID] = entry
		}
	}
	return out
}

func entryMatchesImplicitContext(category string, entry catalog.Entry, context ScopeContext) bool {
	if context.Task != "" {
		if category == "tasks" && entry.ID != context.Task {
			return false
		}
		if len(entry.Tasks) > 0 && !containsString(entry.Tasks, context.Task) {
			return false
		}
	}
	if context.Runner != "" && category == "runners" && entry.ID != context.Runner {
		return false
	}
	return true
}

func (r *Resolver) expandCatalogSelector(selector Selector) []capabilityRef {
	out := []capabilityRef{}
	switch selector.Kind {
	case SelectorCatalogIDs:
		for _, id := range selector.IDs {
			out = append(out, capabilityRef{Catalog: selector.Catalog, ID: id})
		}
	case SelectorCatalogAttributeValues:
		for _, entry := range r.document.Categories[selector.Catalog] {
			for _, value := range selector.Values {
				if text, ok := value.(string); ok && attributeContains(entry.Attributes[selector.Attribute], text) {
					out = append(out, capabilityRef{Catalog: selector.Catalog, ID: entry.ID})
					break
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].key() < out[j].key() })
	return out
}

func (r *Resolver) applyImplicationClosure(effective map[string]map[string]catalog.Entry, forbidden map[string]Finding, findings *[]Finding) {
	changed := true
	for changed {
		changed = false
		// Equivalence is symmetric even if the catalog records only one direction.
		for category, entries := range r.document.Categories {
			for _, entry := range entries {
				self := capabilityRef{Catalog: category, ID: entry.ID}
				for _, raw := range entry.EquivalentTo {
					other, ok := parseCapabilityRef(raw)
					if !ok {
						continue
					}
					pairs := [][2]capabilityRef{{self, other}, {other, self}}
					for _, pair := range pairs {
						cause, blocked := forbidden[pair[0].key()]
						if !blocked {
							continue
						}
						if _, exists := forbidden[pair[1].key()]; exists {
							continue
						}
						cause.Catalog, cause.ID = pair[1].Catalog, pair[1].ID
						cause.Origin = "equivalent"
						cause.CauseCatalog, cause.CauseID = pair[0].Catalog, pair[0].ID
						forbidden[pair[1].key()] = cause
						*findings = append(*findings, cause)
						changed = true
					}
				}
			}
		}
		for category, entries := range effective {
			for id, entry := range entries {
				capability := capabilityRef{Catalog: category, ID: id}
				if _, ok := forbidden[capability.key()]; ok {
					delete(entries, id)
					changed = true
					continue
				}
				dependencies := append(append([]string(nil), entry.Implies...), entry.EquivalentTo...)
				for _, raw := range dependencies {
					dependency, ok := parseCapabilityRef(raw)
					if !ok {
						continue
					}
					cause, blocked := forbidden[dependency.key()]
					if !blocked {
						continue
					}
					finding := cause
					finding.Catalog = category
					finding.ID = id
					finding.Origin = "implied"
					finding.CauseCatalog = dependency.Catalog
					finding.CauseID = dependency.ID
					finding.Remediation = "Choose a capability that does not imply an inherited denial."
					forbidden[capability.key()] = finding
					*findings = append(*findings, finding)
					delete(entries, id)
					changed = true
					break
				}
			}
		}
	}
}

func denialFinding(sourced sourcedRule, capability capabilityRef) Finding {
	return Finding{
		Code:            ReasonCatalogIDDenied,
		Catalog:         capability.Catalog,
		ID:              capability.ID,
		Origin:          "explicit",
		Scope:           sourced.Scope,
		SubjectID:       sourced.SubjectID,
		PolicyVersionID: sourced.PolicyVersionID,
		RuleID:          sourced.Rule.ID,
		Remediation:     "Choose a permitted capability or update the contributing policy scope.",
	}
}

func fieldValueCapabilities(field string, value any) []capabilityRef {
	if text, ok := value.(string); ok {
		definition, exists := execution.CapabilitiesV1().FieldCatalog[field]
		if exists && definition.CatalogCategory != "" {
			if entry, resolved := catalog.Resolve(definition.CatalogCategory, text); resolved {
				return []capabilityRef{{Catalog: definition.CatalogCategory, ID: entry.ID}}
			}
		}
	}
	boolean, ok := value.(bool)
	if !ok || !boolean {
		return nil
	}
	switch field {
	case "preprocessing.use_dataset_normalization", "use_dataset_normalization":
		return []capabilityRef{{Catalog: "normalization_strategies", ID: "dataset"}}
	default:
		return nil
	}
}

func requiredCatalogs(platform map[string]map[string]catalog.Entry) []string {
	requiredCandidates := []string{
		"tasks", "runners", "models", "fine_tuning_modes", "resize_strategies",
		"crop_strategies", "bounding_box_modes", "normalization_strategies",
		"augmentation_policies", "optimizers", "schedulers", "resolution_strategies",
		"class_balancing_strategies", "sampling_strategies", "export_formats",
		"precisions", "runtimes", "execution_providers", "execution_requirements",
	}
	out := []string{}
	for _, category := range requiredCandidates {
		if len(platform[category]) > 0 {
			out = append(out, category)
		}
	}
	sort.Strings(out)
	return out
}

func resolveDefaults(context ScopeContext, platform map[string]map[string]catalog.Entry, effective map[string]map[string]catalog.Entry) ([]ResolvedDefault, []string) {
	if context.Task == "" || context.Runner == "" {
		return []ResolvedDefault{}, nil
	}
	profile, err := execution.CapabilityProfileFor(context.Task, context.Runner)
	if err != nil {
		return []ResolvedDefault{}, nil
	}
	document := execution.CapabilitiesV1()
	resolved := []ResolvedDefault{}
	blocked := []string{}
	resolveValues := func(values map[string]any, fixed bool) {
		fields := make([]string, 0, len(values))
		for field := range values {
			fields = append(fields, field)
		}
		sort.Strings(fields)
		for _, field := range fields {
			definition, ok := document.FieldCatalog[field]
			if !ok || definition.CatalogCategory == "" || len(platform[definition.CatalogCategory]) == 0 {
				continue
			}
			requested, ok := values[field].(string)
			if !ok {
				continue
			}
			entry, ok := catalog.Resolve(definition.CatalogCategory, requested)
			if !ok {
				continue
			}
			if _, permitted := effective[definition.CatalogCategory][entry.ID]; permitted {
				origin := "default"
				if fixed {
					origin = "fixed"
				}
				resolved = append(resolved, ResolvedDefault{
					Field: field, Catalog: definition.CatalogCategory, ID: entry.ID, Origin: origin,
				})
				continue
			}
			if fixed {
				blocked = append(blocked, definition.CatalogCategory)
				continue
			}
			ids := make([]string, 0, len(effective[definition.CatalogCategory]))
			for id := range effective[definition.CatalogCategory] {
				ids = append(ids, id)
			}
			sort.Strings(ids)
			if len(ids) > 0 {
				resolved = append(resolved, ResolvedDefault{
					Field: field, Catalog: definition.CatalogCategory, ID: ids[0],
					Origin: "policy_fallback", OriginalID: entry.ID,
				})
			}
		}
	}
	resolveValues(profile.Defaults, false)
	resolveValues(profile.FixedSemantics, true)
	sort.Slice(resolved, func(i, j int) bool {
		if resolved[i].Field != resolved[j].Field {
			return resolved[i].Field < resolved[j].Field
		}
		return resolved[i].Catalog < resolved[j].Catalog
	})
	return resolved, uniqueSortedStrings(blocked)
}

func uniqueSortedStrings(values []string) []string {
	set := map[string]struct{}{}
	for _, value := range values {
		if value != "" {
			set[value] = struct{}{}
		}
	}
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func (r *Resolver) entry(ref capabilityRef) catalog.Entry {
	entry, _ := catalog.Resolve(ref.Catalog, ref.ID)
	return entry
}

func parseCapabilityRef(value string) (capabilityRef, bool) {
	parts := strings.SplitN(strings.TrimSpace(value), "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return capabilityRef{}, false
	}
	return capabilityRef{Catalog: parts[0], ID: parts[1]}, true
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func cloneEntrySets(input map[string]map[string]catalog.Entry) map[string]map[string]catalog.Entry {
	out := map[string]map[string]catalog.Entry{}
	for category, entries := range input {
		out[category] = map[string]catalog.Entry{}
		for id, entry := range entries {
			out[category][id] = entry
		}
	}
	return out
}

func entrySetIDs(input map[string]map[string]catalog.Entry) map[string][]string {
	out := map[string][]string{}
	for category, entries := range input {
		ids := make([]string, 0, len(entries))
		for id := range entries {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		out[category] = ids
	}
	return out
}

func entrySetsAsSlices(input map[string]map[string]catalog.Entry) map[string][]catalog.Entry {
	out := map[string][]catalog.Entry{}
	for category, entries := range input {
		ids := make([]string, 0, len(entries))
		for id := range entries {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		out[category] = make([]catalog.Entry, 0, len(ids))
		for _, id := range ids {
			out[category] = append(out[category], entries[id])
		}
	}
	return out
}

func entrySetCounts(input map[string]map[string]catalog.Entry) map[string]int {
	out := map[string]int{}
	for category, entries := range input {
		out[category] = len(entries)
	}
	return out
}

func forbiddenIDs(forbidden map[string]Finding) map[string][]string {
	sets := map[string]map[string]struct{}{}
	for _, finding := range forbidden {
		if finding.Catalog == "" || finding.ID == "" {
			continue
		}
		if sets[finding.Catalog] == nil {
			sets[finding.Catalog] = map[string]struct{}{}
		}
		sets[finding.Catalog][finding.ID] = struct{}{}
	}
	out := map[string][]string{}
	for category, ids := range sets {
		for id := range ids {
			out[category] = append(out[category], id)
		}
		sort.Strings(out[category])
	}
	return out
}

func sortFindings(findings []Finding) {
	sort.Slice(findings, func(i, j int) bool {
		left, _ := json.Marshal(findings[i])
		right, _ := json.Marshal(findings[j])
		return string(left) < string(right)
	})
}
