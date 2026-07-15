package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

func bindAPIProjectPolicy(t *testing.T, memoryStore *store.MemoryStore, projectID string, rules ...policies.Rule) policies.PolicyVersion {
	t.Helper()
	version, err := memoryStore.CreateExperimentPolicyVersion(policies.PolicyVersion{
		Document: policies.PolicyDocument{SchemaVersion: policies.PolicySchemaVersionV1, ProfileRefs: []policies.ProfileRef{}, Rules: rules},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := int64(0)
	if _, err := memoryStore.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: projectID, PolicyVersionID: version.ID, ExpectedRevision: &expected,
	}); err != nil {
		t.Fatal(err)
	}
	return version
}

func createPolicyTestProjectDataset(t *testing.T, memoryStore *store.MemoryStore) (string, string) {
	t.Helper()
	project, err := memoryStore.CreateProject("proposal policy", "classify images")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type": "image_classification", "total_images": 120, "class_count": 3, "imbalance_ratio": 1.2,
	}); err != nil {
		t.Fatal(err)
	}
	return project.ID, dataset.ID
}

func TestCreatePlanPersistsImplicitPolicyEvaluationAndHash(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	body, err := json.Marshal(createExperimentPlanRequest{
		DatasetID: datasetID, TargetMetric: "macro_f1", RecommendedWorkers: 1, EstimatedMinutes: 10,
		Experiments: []plans.PlannedExperiment{testExperiment("resnet18", 6)},
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/projects/"+projectID+"/plans", bytes.NewReader(body)))
	if response.Code != http.StatusCreated {
		t.Fatalf("create plan status = %d: %s", response.Code, response.Body.String())
	}
	var plan plans.ExperimentPlan
	if err := json.Unmarshal(response.Body.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.ProposalPolicyEvaluationID == "" || plan.EffectivePolicyHash == "" || plan.PolicyStatus != policyStatusAllowed {
		t.Fatalf("plan policy reference = %#v", plan)
	}
	evaluation, err := memoryStore.GetExperimentPolicyEvaluation(plan.ProposalPolicyEvaluationID)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Decision != policies.DecisionAllowed || evaluation.Operation != policyOperationPersistPlan || evaluation.CandidateConfigHash == "" || evaluation.EffectivePolicyHash != plan.EffectivePolicyHash {
		t.Fatalf("persisted evaluation = %#v", evaluation)
	}
	if len(evaluation.EffectiveCapabilityUses) == 0 || !bytes.Contains(evaluation.EffectiveSnapshot, []byte(policies.ImplicitAllowAllProfileKey)) {
		t.Fatalf("implicit allow_all_v0 evaluation = %#v", evaluation)
	}
}

func TestNoValidConfigurationReturnsBlockedDimensionsScopesAndAuditID(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	version := bindAPIProjectPolicy(t, memoryStore, projectID, policies.Rule{
		ID: "deny_all_models", Effect: policies.EffectDeny,
		Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: catalog.CanonicalIDs("models", true)},
	})
	body, _ := json.Marshal(createExperimentPlanRequest{DatasetID: datasetID})
	response := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/projects/"+projectID+"/plans", bytes.NewReader(body)))
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("create plan status = %d: %s", response.Code, response.Body.String())
	}
	var payload struct {
		Code                policies.ReasonCode          `json:"code"`
		PolicyEvaluationID  string                       `json:"policy_evaluation_id"`
		EffectivePolicyHash string                       `json:"effective_policy_hash"`
		BlockedDimensions   []string                     `json:"blocked_dimensions"`
		ContributingScopes  []policies.ScopeContribution `json:"contributing_scopes"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Code != policies.ReasonNoValidConfiguration || payload.PolicyEvaluationID == "" || payload.EffectivePolicyHash == "" {
		t.Fatalf("policy error = %#v", payload)
	}
	if !reflect.DeepEqual(payload.BlockedDimensions, []string{"models"}) || !reflect.DeepEqual(payload.ContributingScopes, []policies.ScopeContribution{{
		Scope: policies.ScopeProject, SubjectID: projectID, PolicyVersionID: version.ID,
	}}) {
		t.Fatalf("structured block = %#v", payload)
	}
	evaluation, err := memoryStore.GetExperimentPolicyEvaluation(payload.PolicyEvaluationID)
	if err != nil || evaluation.Decision != policies.DecisionDenied || evaluation.Operation != policyOperationPropose {
		t.Fatalf("denied evaluation = %#v, err = %v", evaluation, err)
	}
}

func TestPostAutoMLMaterializationIsPolicyValidated(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, _ := createPolicyTestProjectDataset(t, memoryStore)
	server := newServer(memoryStore)
	settings := server.currentAutomationSettings()
	settings.AutoMLEnabled = true
	settings.DefaultTrainingProvider = "local"
	server.setAutomationSettings(settings)

	t.Run("choice lists are pruned before sampling", func(t *testing.T) {
		effective := policies.ImplicitEffectivePolicy("image_classification", "local_simulator")
		effective.Snapshot.FieldDenials = []policies.FieldDenial{{
			Field: "epochs", CanonicalValue: "7", Scope: policies.ScopeProject, SubjectID: projectID,
			PolicyVersionID: "policy_1", RuleID: "deny_seven_epochs",
		}}
		experiment := testExperiment("resnet18", 6)
		experiment.AutoML = &automl.ExperimentAutoML{
			Enabled: true, Seed: 3, Sampler: automl.SamplerSeededRandom,
			SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{{
				Name: "epochs", Type: automl.ParameterInteger, IntChoices: []int{7, 8},
			}}},
		}
		prepared, _, err := server.prepareAutoMLExperimentsForProjectWithPolicy(projectID, []plans.PlannedExperiment{experiment}, &effective)
		if err != nil || len(prepared) != 1 || prepared[0].Epochs != 8 {
			t.Fatalf("policy-filtered AutoML result = %#v, err = %v", prepared, err)
		}
	})

	t.Run("continuous sampled values are revalidated", func(t *testing.T) {
		effective := policies.ImplicitEffectivePolicy("image_classification", "local_simulator")
		effective.Snapshot.FieldDenials = []policies.FieldDenial{{
			Field: "weight_decay", CanonicalValue: "0.05", Scope: policies.ScopeProject, SubjectID: projectID,
			PolicyVersionID: "policy_1", RuleID: "deny_weight_decay",
		}}
		experiment := testExperiment("resnet18", 8)
		value := 0.05
		experiment.AutoML = &automl.ExperimentAutoML{
			Enabled: true, Seed: 3, Sampler: automl.SamplerSeededRandom,
			SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{{
				Name: "weight_decay", Type: automl.ParameterFloat, Min: &value, Max: &value,
			}}},
		}
		_, _, err := server.prepareAutoMLExperimentsForProjectWithPolicy(projectID, []plans.PlannedExperiment{experiment}, &effective)
		var policyErr *policies.PolicyError
		if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonFieldValueDenied {
			t.Fatalf("post-AutoML policy error = %#v, err = %v", policyErr, err)
		}
	})
}

func TestReusedFollowUpPlanIsRevalidatedAgainstCurrentPolicy(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	sourcePlan, err := memoryStore.CreateExperimentPlan(projectID, datasetID, "macro_f1", 1, 10, []plans.PlannedExperiment{testExperiment("resnet18", 6)}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	decision, err := memoryStore.CreateAgentDecision(projectID, sourcePlan.ID, decisions.TypeAddExperiments, "try another model", map[string]any{
		"proposed_experiments": []plans.PlannedExperiment{testExperiment("convnext_tiny", 8)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.CreateExperimentPlan(projectID, datasetID, "macro_f1", 1, 10, []plans.PlannedExperiment{testExperiment("convnext_tiny", 8)}, nil, decision.ID); err != nil {
		t.Fatal(err)
	}
	bindAPIProjectPolicy(t, memoryStore, projectID, policies.Rule{
		ID: "deny_convnext", Effect: policies.EffectDeny,
		Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: []string{"convnext_tiny"}},
	})

	_, _, err = newServer(memoryStore).ensureFollowUpPlan(projectID, sourcePlan, decision)
	var policyErr *policies.PolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonCatalogIDDenied || policyErr.PolicyEvaluationID == "" {
		t.Fatalf("reuse policy error = %#v, err = %v", policyErr, err)
	}
	evaluation, err := memoryStore.GetExperimentPolicyEvaluation(policyErr.PolicyEvaluationID)
	if err != nil || evaluation.Operation != policyOperationReusePlan || evaluation.Decision != policies.DecisionDenied {
		t.Fatalf("reuse evaluation = %#v, err = %v", evaluation, err)
	}
}

func TestReviewerProposalPersistsPolicyAuditAndDecisionHash(t *testing.T) {
	server, projectID, sourcePlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("resnet18", 6)})
	recordTrainingSummary(t, server, sourcePlan, sourcePlan.Experiments[0], jobs.StatusFailed, 0, 0.1)
	_, decision, err := server.createReviewerDecision(projectID)
	if err != nil {
		t.Fatal(err)
	}
	if decision.DecisionType != decisions.TypeAddExperiments || decision.ProposalPolicyEvaluationID == "" || decision.EffectivePolicyHash == "" {
		t.Fatalf("reviewer decision policy reference = %#v", decision)
	}
	evaluation, err := server.store.GetExperimentPolicyEvaluation(decision.ProposalPolicyEvaluationID)
	if err != nil || evaluation.Decision != policies.DecisionAllowed || evaluation.Operation != policyOperationPersistProposal || evaluation.EffectivePolicyHash != decision.EffectivePolicyHash {
		t.Fatalf("reviewer evaluation = %#v, err = %v", evaluation, err)
	}
}
