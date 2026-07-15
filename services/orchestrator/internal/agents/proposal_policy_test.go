package agents

import (
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
)

func restrictAgentPolicyModels(effective *policies.EffectivePolicy, models ...string) {
	allowed := map[string]bool{}
	for _, model := range models {
		allowed[model] = true
	}
	filtered := effective.PermittedCatalog["models"][:0]
	permittedIDs := []string{}
	for _, entry := range effective.PermittedCatalog["models"] {
		if allowed[entry.ID] {
			filtered = append(filtered, entry)
			permittedIDs = append(permittedIDs, entry.ID)
		}
	}
	effective.PermittedCatalog["models"] = filtered
	effective.PermittedCounts["models"] = len(filtered)
	effective.Snapshot.PermittedCatalog["models"] = permittedIDs
}

func TestPlannerPromptUsesOnlyEffectiveCapabilitiesAndCompactionKeepsCompletePolicyCard(t *testing.T) {
	effective := policies.ImplicitEffectivePolicy("image_classification", "local_simulator")
	restrictAgentPolicyModels(&effective, "resnet18")
	effective.Snapshot.FieldDenials = append(effective.Snapshot.FieldDenials, policies.FieldDenial{
		Field: "epochs", CanonicalValue: "7", Scope: policies.ScopeProject, SubjectID: "project_1",
		PolicyVersionID: "policy_1", RuleID: "deny_seven_epochs",
	})
	input := testExperimentPlannerInput()
	input.EffectivePolicy = &effective
	input.EffectivePolicyCard = policies.PromptCardFromEffectivePolicy(effective)
	input.ModelCatalog = []SupportedModelSpec{
		{Name: "resnet18", TaskType: "image_classification", TrainingEnabled: true},
		{Name: "convnext_tiny", TaskType: "image_classification", TrainingEnabled: true},
	}
	input.ExecutionCapabilityCard = execution.PlannerCapabilityCard{
		Task: "image_classification", Runner: "local_simulator",
		Rules: []execution.PlannerCapabilityRule{{Field: "model", Values: []string{"resnet18", "convnext_tiny"}}},
	}

	v1 := buildPlannerContextSnapshot(input, "v1", ExperimentPlannerPromptVersion)
	v2 := buildPlannerContextSnapshot(input, "v2", ExperimentPlannerPromptVersion)
	if !reflect.DeepEqual(v1.EffectivePolicyCard, input.EffectivePolicyCard) || !reflect.DeepEqual(v2.EffectivePolicyCard, input.EffectivePolicyCard) {
		t.Fatalf("policy card changed during context construction or compaction\nv1=%#v\nv2=%#v", v1.EffectivePolicyCard, v2.EffectivePolicyCard)
	}
	if got := v2.EffectivePolicyCard.PermittedCatalog["models"]; len(got) != 1 || got[0].ID != "resnet18" {
		t.Fatalf("effective prompt models = %#v", got)
	}
	if !reflect.DeepEqual(v2.EffectivePolicyCard.FieldDenials, effective.Snapshot.FieldDenials) {
		t.Fatalf("field constraints were dropped from compact prompt: %#v", v2.EffectivePolicyCard.FieldDenials)
	}
	for _, snapshot := range []PlannerContextSnapshot{v1, v2} {
		if len(snapshot.ModelCatalog) != 1 || snapshot.ModelCatalog[0].ID != "resnet18" {
			t.Fatalf("selectable model catalog leaked a denied model: %#v", snapshot.ModelCatalog)
		}
		if len(snapshot.ExecutionCapabilities.Rules) != 1 || !reflect.DeepEqual(snapshot.ExecutionCapabilities.Rules[0].Values, []string{"resnet18"}) {
			t.Fatalf("execution capability card leaked a denied model: %#v", snapshot.ExecutionCapabilities.Rules)
		}
	}
}

func TestCandidateRankingRejectsPolicyDeniedHallucinationBeforeSelection(t *testing.T) {
	effective := policies.ImplicitEffectivePolicy("image_classification", "local_simulator")
	restrictAgentPolicyModels(&effective, "resnet18")
	input := testExperimentPlannerInput()
	input.EffectivePolicy = &effective
	candidate := rankingCandidate("convnext_tiny", 0.03)

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})
	if !ranking.Rejected || ranking.Score != 0 || len(ranking.PolicyFindings) != 1 {
		t.Fatalf("ranking = %#v", ranking)
	}
	if ranking.PolicyFindings[0].Code != policies.ReasonCatalogIDDenied || ranking.PolicyFindings[0].FieldPath != "experiments[0].model" {
		t.Fatalf("policy findings = %#v", ranking.PolicyFindings)
	}
}

func TestDeterministicPlannerAndReviewerFallbackStayWithinEffectivePolicy(t *testing.T) {
	effective := policies.ImplicitEffectivePolicy("image_classification", "local_simulator")
	restrictAgentPolicyModels(&effective, "resnet18")
	project := projects.Project{ID: "project_1", Name: "policy project"}
	dataset := datasets.Dataset{
		ID: "dataset_1", ProjectID: project.ID, Status: datasets.StatusProfiled,
		Profile: map[string]any{"total_images": 120, "class_count": 3, "imbalance_ratio": 1.2},
	}
	recommendation, err := NewDatasetPlanner().BuildExperimentPlan(project, dataset, PlanPreferences{EffectivePolicy: &effective})
	if err != nil || len(recommendation.Experiments) == 0 {
		t.Fatalf("deterministic recommendation = %#v, err = %v", recommendation, err)
	}
	for _, experiment := range recommendation.Experiments {
		if experiment.Model != "resnet18" {
			t.Fatalf("deterministic planner selected denied model: %#v", recommendation.Experiments)
		}
	}

	plan := plans.ExperimentPlan{ID: "plan_1", ProjectID: project.ID, DatasetID: dataset.ID, TargetMetric: "macro_f1", Experiments: []plans.PlannedExperiment{{Model: "resnet18"}}}
	review, err := NewExperimentReviewer().ReviewWithPolicy(project, plan, []runs.TrainingRunSummary{{
		JobID: "job_1", PlanID: plan.ID, Model: "resnet18", Status: jobs.StatusFailed, EstimatedCostUSD: 0.1,
	}}, &effective)
	if err != nil || review.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("review = %#v, err = %v", review, err)
	}
	experiments, ok := review.Payload["proposed_experiments"].([]plans.PlannedExperiment)
	if !ok || len(experiments) == 0 {
		t.Fatalf("reviewer experiments = %#v", review.Payload["proposed_experiments"])
	}
	for _, experiment := range experiments {
		if experiment.Model != "resnet18" {
			t.Fatalf("reviewer selected denied model: %#v", experiments)
		}
	}
}
