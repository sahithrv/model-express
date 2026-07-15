package policies

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/plans"
)

func proposalTestExperiment(model string) plans.PlannedExperiment {
	return plans.PlannedExperiment{
		Template: "image_classifier_transfer", Model: model, Epochs: 7,
		BatchSize: 16, LearningRate: 0.0003, Reason: "policy test",
	}
}

func restrictProposalTestCategory(effective *EffectivePolicy, category string, ids ...string) {
	allowed := map[string]bool{}
	for _, id := range ids {
		entry, ok := catalog.Resolve(category, id)
		if ok {
			allowed[entry.ID] = true
		}
	}
	entries := effective.PermittedCatalog[category]
	filtered := entries[:0]
	for _, entry := range entries {
		if allowed[entry.ID] {
			filtered = append(filtered, entry)
		}
	}
	effective.PermittedCatalog[category] = filtered
	effective.PermittedCounts[category] = len(filtered)
	ids = ids[:0]
	for _, entry := range filtered {
		ids = append(ids, entry.ID)
	}
	effective.Snapshot.PermittedCatalog[category] = ids
}

func TestEvaluateProposalRejectsHallucinatedOrDeniedCapabilities(t *testing.T) {
	effective := ImplicitEffectivePolicy("image_classification", "local_simulator")
	restrictProposalTestCategory(&effective, "models", "resnet18")
	effective.Findings = append(effective.Findings, Finding{
		Code: ReasonCatalogIDDenied, Catalog: "models", ID: "convnext_tiny",
		Scope: ScopeProject, SubjectID: "project_1", PolicyVersionID: "policy_1", RuleID: "deny_convnext",
	})

	for _, test := range []struct {
		name  string
		model string
		code  ReasonCode
	}{
		{name: "denied", model: "convnext_tiny", code: ReasonCatalogIDDenied},
		{name: "hallucinated", model: "imaginary_net", code: ReasonUnknownCatalogIdentifier},
	} {
		t.Run(test.name, func(t *testing.T) {
			evaluation, err := EvaluateProposal(effective, "propose", []plans.PlannedExperiment{proposalTestExperiment(test.model)})
			if err == nil || evaluation.Decision != DecisionDenied || evaluation.CandidateConfigHash == "" {
				t.Fatalf("evaluation = %#v, err = %v", evaluation, err)
			}
			var policyErr *PolicyError
			if !errors.As(err, &policyErr) || policyErr.Code != test.code {
				t.Fatalf("policy error = %#v", policyErr)
			}
			if len(evaluation.Findings) != 1 || evaluation.Findings[0].FieldPath != "experiments[0].model" {
				t.Fatalf("findings = %#v", evaluation.Findings)
			}
		})
	}
}

func TestEvaluateProposalRecordsAllowedUsesAndFieldDenials(t *testing.T) {
	effective := ImplicitEffectivePolicy("image_classification", "local_simulator")
	effective.Snapshot.FieldDenials = append(effective.Snapshot.FieldDenials, FieldDenial{
		Field: "epochs", CanonicalValue: "7", Scope: ScopeDataset, SubjectID: "dataset_1",
		PolicyVersionID: "policy_1", RuleID: "deny_seven_epochs",
	})
	experiment := proposalTestExperiment("resnet18")

	evaluation, err := EvaluateProposal(effective, "persist_plan", []plans.PlannedExperiment{experiment})
	if err == nil || evaluation.Decision != DecisionDenied {
		t.Fatalf("evaluation = %#v, err = %v", evaluation, err)
	}
	if len(evaluation.Findings) != 1 || evaluation.Findings[0].Code != ReasonFieldValueDenied {
		t.Fatalf("findings = %#v", evaluation.Findings)
	}
	var policyErr *PolicyError
	if !errors.As(err, &policyErr) || !reflect.DeepEqual(policyErr.ContributingScopes, []ScopeContribution{{
		Scope: ScopeDataset, SubjectID: "dataset_1", PolicyVersionID: "policy_1",
	}}) {
		t.Fatalf("policy error scopes = %#v", policyErr)
	}

	effective.Snapshot.FieldDenials = nil
	allowed, err := EvaluateProposal(effective, "persist_plan", []plans.PlannedExperiment{experiment})
	if err != nil || allowed.Decision != DecisionAllowed || allowed.EffectivePolicyHash == "" || allowed.CandidateConfigHash == "" {
		t.Fatalf("allowed evaluation = %#v, err = %v", allowed, err)
	}
	if len(allowed.RequestedCapabilityUses) == 0 || len(allowed.EffectiveCapabilityUses) == 0 {
		t.Fatalf("capability uses were not recorded: %#v", allowed)
	}
}

func TestEvaluateProposalCannotBypassDeniedFalseWithExplicitJSON(t *testing.T) {
	effective := ImplicitEffectivePolicy("image_classification", "local_simulator")
	effective.Snapshot.FieldDenials = []FieldDenial{{
		Field: "pretrained", CanonicalValue: "false", Scope: ScopeProject,
		SubjectID: "project_1", PolicyVersionID: "policy_1", RuleID: "require_pretraining",
	}}
	var experiment plans.PlannedExperiment
	if err := json.Unmarshal([]byte(`{
		"template":"image_classifier_transfer","model":"resnet18","epochs":7,
		"batch_size":16,"learning_rate":0.0003,"reason":"presence test","pretrained":false
	}`), &experiment); err != nil {
		t.Fatal(err)
	}
	evaluation, err := EvaluateProposal(effective, "persist_plan", []plans.PlannedExperiment{experiment})
	if err == nil || evaluation.Decision != DecisionDenied || len(evaluation.Findings) != 1 {
		t.Fatalf("explicit false evaluation = %#v, err = %v", evaluation, err)
	}
	if evaluation.Findings[0].FieldPath != "experiments[0].pretrained" || evaluation.Findings[0].Code != ReasonFieldValueDenied {
		t.Fatalf("explicit false finding = %#v", evaluation.Findings)
	}
}
