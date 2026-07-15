package store

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
)

func TestMemoryClaimRejectsEvaluationAfterPolicyHeadChanges(t *testing.T) {
	persistence := NewMemoryStore()
	project, _ := persistence.CreateProject("claim race", "")
	dataset, _ := persistence.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	job, _ := persistence.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID, "model": "resnet18", "provider": "local"})
	worker, _ := persistence.RegisterWorker(project.ID, "worker", "local")
	effective, err := policies.NewResolver(persistence).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, DatasetID: dataset.ID,
		ExperimentJobID: job.ID, Task: "image_classification", Runner: "local_simulator",
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := policies.EvaluateProposal(effective, "dispatch_run", []plans.PlannedExperiment{{Model: "resnet18"}})
	if err != nil {
		t.Fatal(err)
	}
	evaluation.ProjectID = project.ID
	evaluation.DatasetID = dataset.ID
	evaluation.JobID = job.ID
	version, err := persistence.CreateExperimentPolicyVersion(policies.PolicyVersion{Document: policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		Rules:         []policies.Rule{{ID: "deny_resnet", Effect: policies.EffectDeny, Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: []string{"resnet18"}}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	expected := int64(0)
	if _, err := persistence.SetExperimentPolicyBinding(policies.BindingWrite{Scope: policies.ScopeProject, SubjectID: project.ID, PolicyVersionID: version.ID, ExpectedRevision: &expected}); err != nil {
		t.Fatal(err)
	}
	claimed, _, assigned, err := persistence.ClaimJobIfQueuedAndPolicyCurrent(worker.ID, job.ID, JobPollFilter{}, evaluation)
	if claimed != nil || assigned || !errors.Is(err, ErrPolicyChanged) {
		t.Fatalf("stale policy evaluation claim = %#v, assigned=%v, err=%v", claimed, assigned, err)
	}
	stored, _ := persistence.GetJob(job.ID)
	if stored.Status != jobs.StatusQueued || stored.WorkerID != "" || stored.Attempt != 0 {
		t.Fatalf("stale evaluation mutated job: %#v", stored)
	}
}

func TestMemoryAtomicClaimRejectsIncompatibleWorkerArtifactCapabilities(t *testing.T) {
	persistence := NewMemoryStore()
	project, _ := persistence.CreateProject("artifact capability claim", "")
	dataset, _ := persistence.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	effective, err := policies.NewResolver(persistence).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, DatasetID: dataset.ID,
		Task: "image_classification", Runner: "modal_torchvision",
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := execution.BuildAutomaticArtifactPlanV1("image_classification", "modal_torchvision", execution.ArtifactPolicySelection{
		PolicyRestricted: true, EffectivePolicyHash: effective.EffectivePolicyHash,
		PermittedCatalog: map[string][]string{
			"export_formats": {"onnx"}, "precisions": {"fp32"}, "runtimes": {"onnxruntime"},
			"execution_providers": {"cpu_execution_provider"}, "execution_requirements": {"cpu_compatible"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	config := map[string]any{"dataset_id": dataset.ID, "model": "resnet18", "provider": "modal"}
	spec, err := execution.BuildExecutionSpecV1WithArtifactPlan("image_classification", "modal_torchvision", config, config, plan)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := spec.Payload()
	if err != nil {
		t.Fatal(err)
	}
	config[execution.ExecutionSpecConfigKey] = payload
	job, err := persistence.CreateJob(project.ID, jobs.TemplateTrainExperiment, config)
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := policies.EvaluateProposal(effective, "dispatch_run", []plans.PlannedExperiment{{Model: "resnet18"}})
	if err != nil {
		t.Fatal(err)
	}
	evaluation.ProjectID, evaluation.DatasetID, evaluation.JobID = project.ID, dataset.ID, job.ID
	legacy, _ := persistence.RegisterWorkerWithCapabilities(project.ID, "legacy", "local", nil, nil)
	claimed, _, assigned, err := persistence.ClaimJobIfQueuedAndPolicyCurrent(legacy.ID, job.ID, JobPollFilter{Provider: "modal"}, evaluation)
	if err != nil || claimed != nil || assigned {
		t.Fatalf("incompatible atomic claim = %#v, assigned=%v, err=%v", claimed, assigned, err)
	}
	stored, _ := persistence.GetJob(job.ID)
	if stored.Status != jobs.StatusQueued || stored.WorkerID != "" || stored.Attempt != 0 {
		t.Fatalf("incompatible claim mutated job: %#v", stored)
	}
}

func TestPostgresClaimAndBindingWritesSharePolicyHeadLocks(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve policy dispatch test path")
	}
	root := filepath.Dir(currentFile)
	dispatchSource, err := os.ReadFile(filepath.Join(root, "policy_dispatch.go"))
	if err != nil {
		t.Fatal(err)
	}
	policySource, err := os.ReadFile(filepath.Join(root, "postgres_policies.go"))
	if err != nil {
		t.Fatal(err)
	}
	for name, source := range map[string]string{"claim": string(dispatchSource), "binding write": string(policySource)} {
		if !strings.Contains(source, "pg_advisory_xact_lock(hashtextextended($1, 0))") {
			t.Fatalf("%s path does not lock the shared policy head", name)
		}
	}
	if !strings.Contains(string(dispatchSource), "binding.PolicyVersionID != source.PolicyVersionID") || !strings.Contains(string(dispatchSource), "binding.Revision != source.BindingRevision") {
		t.Fatal("atomic claim does not compare the resolved policy source revision and version")
	}
}
