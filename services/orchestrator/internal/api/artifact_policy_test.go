package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

func TestRestrictedArtifactJobSkipsLegacyAndUnknownWorkers(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	bindAPIProjectPolicy(t, memoryStore, projectID,
		denyPolicyIDs("deny_torchscript", "export_formats", "torchscript"),
		denyPolicyIDs("deny_pytorch", "export_formats", "pytorch"),
		denyPolicyIDs("deny_safetensors", "export_formats", "safetensors"),
	)
	router := NewRouter(memoryStore)
	created := performJSONRequest(t, router, http.MethodPost, "/projects/"+projectID+"/jobs", map[string]any{
		"template": jobs.TemplateTrainExperiment,
		"config": map[string]any{
			"dataset_id": datasetID, "model": "resnet18", "provider": "modal",
			execution.ArtifactPlanConfigKey: map[string]any{"artifact_plan_hash": "client-spoof"},
		},
	})
	if created.Code != http.StatusCreated {
		t.Fatalf("create status = %d: %s", created.Code, created.Body.String())
	}
	var job jobs.ExperimentJob
	if err := json.Unmarshal(created.Body.Bytes(), &job); err != nil {
		t.Fatal(err)
	}
	plan, ok, err := artifactPlanFromJobConfig(job.Config)
	if err != nil || !ok || len(plan.Artifacts) != 1 || plan.Artifacts[0].Format != "onnx" || !plan.PolicyRestricted {
		t.Fatalf("restricted artifact plan = %#v, ok=%v err=%v", plan, ok, err)
	}
	if _, retained := job.Config[execution.ArtifactPlanConfigKey]; retained || plan.ArtifactPlanHash == "client-spoof" {
		t.Fatalf("client-supplied artifact plan was retained: %#v", job.Config)
	}

	legacy, _ := memoryStore.RegisterWorkerWithCapabilities(projectID, "legacy", "modal", nil, nil)
	unknown, _ := memoryStore.RegisterWorkerWithCapabilities(projectID, "unknown", "modal", []string{"policy_contract_v999"}, []string{"artifact_plan_v999"})
	server := newServer(memoryStore)
	for _, workerID := range []string{legacy.ID, unknown.ID} {
		if assigned, err := server.pollNextPolicyPermittedJob(workerID, store.JobPollFilter{Provider: "modal"}); assigned != nil || !errors.Is(err, store.ErrNoJob) {
			t.Fatalf("incompatible worker %s received restricted job %#v, err=%v", workerID, assigned, err)
		}
	}
	compatible, _ := memoryStore.RegisterWorkerWithCapabilities(projectID, "compatible", "modal", []string{execution.WorkerPolicyContractV1}, []string{execution.WorkerArtifactPlanV1})
	assigned, err := server.pollNextPolicyPermittedJob(compatible.ID, store.JobPollFilter{Provider: "modal"})
	if err != nil || assigned == nil || assigned.ID != job.ID {
		t.Fatalf("compatible assignment = %#v, err=%v", assigned, err)
	}
	record, err := memoryStore.GetJobExecutionRecord(job.ID)
	if err != nil || len(record.Attempts) != 1 {
		t.Fatalf("execution record = %#v, err=%v", record, err)
	}
	attempt := record.Attempts[0]
	if attempt.WorkerPolicyCapabilityVersion != execution.WorkerPolicyContractV1 || attempt.WorkerArtifactCapabilityVersion != execution.WorkerArtifactPlanV1 || attempt.ArtifactPlanHash != plan.ArtifactPlanHash {
		t.Fatalf("negotiated attempt = %#v", attempt)
	}
}

func TestNoPolicyJobKeepsLegacyAutomaticArtifactsAndDispatchesLegacyWorker(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	router := NewRouter(memoryStore)
	created := performJSONRequest(t, router, http.MethodPost, "/projects/"+projectID+"/jobs", map[string]any{
		"template": jobs.TemplateTrainExperiment,
		"config":   map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "modal"},
	})
	if created.Code != http.StatusCreated {
		t.Fatalf("create status = %d: %s", created.Code, created.Body.String())
	}
	var job jobs.ExperimentJob
	_ = json.Unmarshal(created.Body.Bytes(), &job)
	plan, ok, err := artifactPlanFromJobConfig(job.Config)
	formats := []string{}
	for _, artifact := range plan.Artifacts {
		formats = append(formats, artifact.Format)
	}
	if err != nil || !ok || plan.PolicyRestricted || len(plan.RequiredWorkerCapabilities.ArtifactPlanVersions) != 0 || len(formats) != 3 {
		t.Fatalf("legacy-compatible plan = %#v, formats=%#v err=%v", plan, formats, err)
	}
	legacy, _ := memoryStore.RegisterWorkerWithCapabilities(projectID, "legacy", "modal", nil, nil)
	assigned, err := newServer(memoryStore).pollNextPolicyPermittedJob(legacy.ID, store.JobPollFilter{Provider: "modal"})
	if err != nil || assigned == nil || assigned.ID != job.ID {
		t.Fatalf("legacy worker could not claim no-policy job: %#v, %v", assigned, err)
	}
}

func TestUnavailablePrecisionRequestFailsClosed(t *testing.T) {
	for _, test := range []struct {
		precision string
		status    int
		code      policies.ReasonCode
	}{
		{precision: "fp16", status: http.StatusUnprocessableEntity, code: policies.ReasonCatalogIDDenied},
		{precision: "int8", status: http.StatusUnprocessableEntity, code: policies.ReasonCatalogIDDenied},
		{precision: "future_precision", status: http.StatusBadRequest, code: policies.ReasonUnknownCatalogIdentifier},
	} {
		t.Run(test.precision, func(t *testing.T) {
			memoryStore := store.NewMemoryStore()
			projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
			response := performJSONRequest(t, NewRouter(memoryStore), http.MethodPost, "/projects/"+projectID+"/jobs", map[string]any{
				"template": jobs.TemplateTrainExperiment,
				"config":   map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "modal", "precision": test.precision},
			})
			if response.Code != test.status {
				t.Fatalf("precision %s status = %d: %s", test.precision, response.Code, response.Body.String())
			}
			var payload struct {
				Code policies.ReasonCode `json:"code"`
			}
			_ = json.Unmarshal(response.Body.Bytes(), &payload)
			if payload.Code != test.code {
				t.Fatalf("precision %s code = %s", test.precision, payload.Code)
			}
		})
	}
}

func TestPolicyAdminScopesQueueImpactAuditAndOptimisticRevision(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	job, _ := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"})
	router := NewRouter(memoryStore)
	document := map[string]any{
		"schema_version": policies.PolicySchemaVersionV1,
		"profile_refs":   []any{map[string]any{"id": "roasty_v1", "version": "1.0.0"}},
		"rules":          []any{}, "metadata": map[string]any{"display_name": "Roasty"},
	}
	for _, target := range []string{
		"/settings/experiment-policy", "/projects/" + projectID + "/experiment-policy",
		"/datasets/" + datasetID + "/experiment-policy", "/jobs/" + job.ID + "/experiment-policy",
	} {
		response := performJSONRequest(t, router, http.MethodPut, target, map[string]any{"document": document, "expected_revision": 0})
		if response.Code != http.StatusOK {
			t.Fatalf("PUT %s = %d: %s", target, response.Code, response.Body.String())
		}
	}
	projectResponse := performJSONRequest(t, router, http.MethodGet, "/projects/"+projectID+"/experiment-policy", nil)
	if projectResponse.Code != http.StatusOK {
		t.Fatalf("project policy GET = %d: %s", projectResponse.Code, projectResponse.Body.String())
	}
	var admin experimentPolicyAdminResponse
	if err := json.Unmarshal(projectResponse.Body.Bytes(), &admin); err != nil {
		t.Fatal(err)
	}
	if len(admin.EffectivePolicy.Snapshot.PolicySources) != 2 || admin.QueueImpact.Queued != 1 || admin.QueueImpact.Pending != 1 || admin.QueueImpact.Blocked != 1 || admin.QueueImpact.Allowed != 0 {
		t.Fatalf("project policy preview = %#v", admin)
	}
	conflict := performJSONRequest(t, router, http.MethodPut, "/projects/"+projectID+"/experiment-policy", map[string]any{"document": document, "expected_revision": 0})
	if conflict.Code != http.StatusConflict {
		t.Fatalf("stale revision = %d: %s", conflict.Code, conflict.Body.String())
	}
	audit := performJSONRequest(t, router, http.MethodGet, "/projects/"+projectID+"/experiment-policy/audit", nil)
	if audit.Code != http.StatusOK {
		t.Fatalf("audit GET = %d: %s", audit.Code, audit.Body.String())
	}
}
