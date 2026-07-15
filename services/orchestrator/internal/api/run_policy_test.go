package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

func TestDirectTrainingAndExportJobsCannotBypassCurrentPolicy(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	bindAPIProjectPolicy(t, memoryStore, projectID,
		denyPolicyIDs("deny_resnet", "models", "resnet18"),
		denyPolicyIDs("deny_onnx", "export_formats", "onnx"),
	)
	router := NewRouter(memoryStore)

	for _, test := range []struct {
		name string
		body map[string]any
		code policies.ReasonCode
	}{
		{name: "train_experiment", body: map[string]any{"template": jobs.TemplateTrainExperiment, "config": map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"}}, code: policies.ReasonCatalogIDDenied},
		{name: "export_champion", body: map[string]any{"template": jobs.TemplateExportChampion, "config": map[string]any{"dataset_id": datasetID, "format": "onnx"}}, code: policies.ReasonCatalogIDDenied},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := performJSONRequest(t, router, http.MethodPost, "/projects/"+projectID+"/jobs", test.body)
			if response.Code != http.StatusUnprocessableEntity {
				t.Fatalf("status = %d: %s", response.Code, response.Body.String())
			}
			var payload struct {
				Code                policies.ReasonCode `json:"code"`
				PolicyEvaluationID  string              `json:"policy_evaluation_id"`
				EffectivePolicyHash string              `json:"effective_policy_hash"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
				t.Fatal(err)
			}
			if payload.Code != test.code || payload.PolicyEvaluationID == "" || payload.EffectivePolicyHash == "" {
				t.Fatalf("structured policy error = %#v", payload)
			}
		})
	}
	projectJobs, err := memoryStore.ListProjectJobs(projectID)
	if err != nil {
		t.Fatal(err)
	}
	if len(projectJobs) != 0 {
		t.Fatalf("policy-forbidden direct jobs persisted: %#v", projectJobs)
	}
	evaluations, err := memoryStore.ListExperimentPolicyEvaluations(projectID)
	if err != nil || len(evaluations) != 2 {
		t.Fatalf("direct-job policy audit = %#v, err = %v", evaluations, err)
	}
	events, err := memoryStore.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatal(err)
	}
	blockedActivityCount := 0
	for _, event := range events {
		if event.EventType != "JOB_POLICY_BLOCKED" {
			continue
		}
		if configString(event.Payload, "policy_evaluation_id") == "" || configString(event.Payload, "effective_policy_hash") == "" {
			t.Fatalf("policy activity omitted audit references: %#v", event)
		}
		if activity := activityFromExecutionEvent(event); activity.Type != "job.policy_blocked" || activity.Status != "blocked" {
			t.Fatalf("policy activity projection = %#v", activity)
		}
		blockedActivityCount++
	}
	if blockedActivityCount != 2 {
		t.Fatalf("blocked activity count = %d, events = %#v", blockedActivityCount, events)
	}
}

func TestStoredPlanExecutionUsesCurrentPolicyAndRecordsScheduleAudit(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	plan, err := memoryStore.CreateExperimentPlan(projectID, datasetID, "macro_f1", 1, 10, []plans.PlannedExperiment{testExperiment("resnet18", 6)}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	_, err = newServer(memoryStore).executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "local"})
	var policyErr *policies.PolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != policies.ReasonCatalogIDDenied || policyErr.PolicyEvaluationID == "" {
		t.Fatalf("plan schedule policy error = %#v, err = %v", policyErr, err)
	}
	evaluation, err := memoryStore.GetExperimentPolicyEvaluation(policyErr.PolicyEvaluationID)
	if err != nil || evaluation.Operation != policyOperationScheduleRun || evaluation.PlanID != plan.ID || evaluation.Decision != policies.DecisionDenied {
		t.Fatalf("schedule evaluation = %#v, err = %v", evaluation, err)
	}
	projectJobs, _ := memoryStore.ListProjectJobs(projectID)
	if len(projectJobs) != 0 {
		t.Fatalf("blocked stored plan created jobs: %#v", projectJobs)
	}
}

func TestScheduleAndDispatchPersistImplicitPolicyHashes(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	plan, err := memoryStore.CreateExperimentPlan(projectID, datasetID, "macro_f1", 1, 10, []plans.PlannedExperiment{testExperiment("resnet18", 6)}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	server := newServer(memoryStore)
	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil || len(result.Jobs) != 1 {
		t.Fatalf("execute plan jobs = %#v, err = %v", result.Jobs, err)
	}
	queued := result.Jobs[0]
	if queued.SchedulePolicyEvaluationID == "" || queued.EffectivePolicyHash == "" || queued.PolicyEligibilityStatus != jobs.PolicyEligibilityAllowed {
		t.Fatalf("scheduled policy reference = %#v", queued)
	}
	schedule, err := memoryStore.GetExperimentPolicyEvaluation(queued.SchedulePolicyEvaluationID)
	if err != nil || schedule.Operation != policyOperationScheduleRun || !bytes.Contains(schedule.EffectiveSnapshot, []byte(policies.ImplicitAllowAllProfileKey)) {
		t.Fatalf("implicit schedule evaluation = %#v, err = %v", schedule, err)
	}
	worker, err := memoryStore.RegisterWorker(projectID, "policy worker", "local")
	if err != nil {
		t.Fatal(err)
	}
	assigned, err := server.pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{})
	if err != nil || assigned == nil || assigned.ID != queued.ID {
		t.Fatalf("assigned = %#v, err = %v", assigned, err)
	}
	record, err := memoryStore.GetJobExecutionRecord(queued.ID)
	if err != nil || len(record.Attempts) != 1 {
		t.Fatalf("execution record = %#v, err = %v", record, err)
	}
	attempt := record.Attempts[0]
	if attempt.DispatchPolicyEvaluationID == "" || attempt.EffectivePolicyHash == "" {
		t.Fatalf("dispatch policy reference = %#v", attempt)
	}
	dispatch, err := memoryStore.GetExperimentPolicyEvaluation(attempt.DispatchPolicyEvaluationID)
	if err != nil || dispatch.Operation != policyOperationDispatchRun || dispatch.JobID != queued.ID || dispatch.EffectivePolicyHash != attempt.EffectivePolicyHash {
		t.Fatalf("dispatch evaluation = %#v, err = %v", dispatch, err)
	}
}

func TestRetryAndOOMMutationUseCurrentPolicy(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	job, err := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": datasetID, "model": "resnet18", "provider": "modal", "gpu_type": "T4", "batch_size": 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	worker, _ := memoryStore.RegisterWorker(projectID, "modal worker", "modal")
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	failBody := map[string]any{
		"error": "CUDA out of memory", "retryable": true,
		"training_attempt_id": configString(assigned.Config, "active_attempt_id"),
		"oom":                 true, "failure_class": "oom", "effective_gpu_type": "T4", "effective_batch_size": 8,
	}
	blob, _ := json.Marshal(failBody)
	request := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", bytes.NewReader(blob))
	request.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, request, assigned)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("retry response = %d: %s", response.Code, response.Body.String())
	}
	blocked, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if blocked.Status != jobs.StatusFailed || blocked.PolicyEligibilityStatus != jobs.PolicyEligibilityBlocked || configString(blocked.Config, "gpu_type") != "L4" {
		t.Fatalf("policy-blocked OOM mutation = %#v", blocked)
	}
	evaluation, err := memoryStore.GetExperimentPolicyEvaluation(blocked.SchedulePolicyEvaluationID)
	if err != nil || evaluation.Operation != policyOperationRetryRun || evaluation.Decision != policies.DecisionDenied {
		t.Fatalf("retry evaluation = %#v, err = %v", evaluation, err)
	}
}

func TestLeaseRecoveryRequiresFreshPolicyEvaluation(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	job, _ := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"})
	worker, _ := memoryStore.RegisterWorker(projectID, "worker", "local")
	server := newServer(memoryStore)
	assigned, err := server.pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{})
	if err != nil || assigned.LeaseExpiresAt == nil {
		t.Fatalf("assigned lease = %#v, err = %v", assigned, err)
	}
	bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	recovered, err := server.recoverExpiredLeasesOnce(assigned.LeaseExpiresAt.Add(time.Second))
	if err != nil || len(recovered) != 1 {
		t.Fatalf("recovered = %#v, err = %v", recovered, err)
	}
	blocked, _ := memoryStore.GetJob(job.ID)
	if blocked.Status != jobs.StatusQueued || blocked.PolicyEligibilityStatus != jobs.PolicyEligibilityBlocked {
		t.Fatalf("lease-recovered job policy state = %#v", blocked)
	}
	if _, err := server.pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{}); !errors.Is(err, store.ErrNoJob) {
		t.Fatalf("blocked recovered job dispatched: %v", err)
	}
	evaluations, _ := memoryStore.ListExperimentPolicyEvaluations(projectID)
	if !hasPolicyEvaluation(evaluations, policyOperationRequeueRun, policies.DecisionDenied) {
		t.Fatalf("missing denied requeue evaluation: %#v", evaluations)
	}
}

func TestDispatchSkipsBlockedCandidateAndClaimsNextPermittedJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	blockedCandidates := make([]jobs.ExperimentJob, 0, 65)
	for index := 0; index < 65; index++ {
		blockedCandidate, err := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"})
		if err != nil {
			t.Fatal(err)
		}
		blockedCandidates = append(blockedCandidates, blockedCandidate)
	}
	permittedCandidate, _ := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "mobilenet_v3_small", "provider": "local"})
	bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	worker, _ := memoryStore.RegisterWorker(projectID, "worker", "local")
	assigned, err := newServer(memoryStore).pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{})
	if err != nil || assigned.ID != permittedCandidate.ID {
		t.Fatalf("next permitted assignment = %#v, err = %v", assigned, err)
	}
	for _, candidate := range blockedCandidates {
		blocked, _ := memoryStore.GetJob(candidate.ID)
		if blocked.Status != jobs.StatusQueued || blocked.PolicyEligibilityStatus != jobs.PolicyEligibilityBlocked {
			t.Fatalf("blocked candidate state = %#v", blocked)
		}
	}
	evaluations, _ := memoryStore.ListExperimentPolicyEvaluations(projectID)
	if !hasPolicyEvaluation(evaluations, policyOperationDispatchRun, policies.DecisionDenied) || !hasPolicyEvaluation(evaluations, policyOperationDispatchRun, policies.DecisionAllowed) {
		t.Fatalf("dispatch evaluations = %#v", evaluations)
	}
}

func TestPolicyBindingUpdateReconcilesBlockedQueuedJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	job, _ := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"})
	bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	worker, _ := memoryStore.RegisterWorker(projectID, "worker", "local")
	server := newServer(memoryStore)
	if assigned, err := server.pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{}); assigned != nil || !errors.Is(err, store.ErrNoJob) {
		t.Fatalf("denied job unexpectedly assigned: %#v, %v", assigned, err)
	}
	blocked, _ := memoryStore.GetJob(job.ID)
	if blocked.PolicyEligibilityStatus != jobs.PolicyEligibilityBlocked {
		t.Fatalf("job was not blocked before reconciliation: %#v", blocked)
	}
	bindings, err := memoryStore.ListActiveExperimentPolicyBindings(policies.ScopeContext{AccountID: policies.LocalDefaultAccountID, ProjectID: projectID})
	if err != nil || len(bindings) != 1 {
		t.Fatalf("active binding = %#v, err = %v", bindings, err)
	}
	if _, err := memoryStore.ClearExperimentPolicyBinding(policies.ScopeProject, projectID, bindings[0].Revision); err != nil {
		t.Fatal(err)
	}
	pending, _ := memoryStore.GetJob(job.ID)
	if pending.PolicyEligibilityStatus != jobs.PolicyEligibilityPending {
		t.Fatalf("policy update did not mark queued job pending: %#v", pending)
	}
	assigned, err := server.pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{})
	if err != nil || assigned == nil || assigned.ID != job.ID {
		t.Fatalf("reconciled job assignment = %#v, err = %v", assigned, err)
	}
}

type policyRaceStore struct {
	store.Store
	once        sync.Once
	beforeClaim func()
}

func (s *policyRaceStore) ClaimJobIfQueuedAndPolicyCurrent(workerID string, jobID string, filter store.JobPollFilter, evaluation policies.Evaluation) (*jobs.ExperimentJob, policies.Evaluation, bool, error) {
	s.once.Do(s.beforeClaim)
	return s.Store.ClaimJobIfQueuedAndPolicyCurrent(workerID, jobID, filter, evaluation)
}

func TestPolicyUpdateRacingWorkerClaimCannotAssignNewlyForbiddenJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	projectID, datasetID := createPolicyTestProjectDataset(t, memoryStore)
	job, _ := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": datasetID, "model": "resnet18", "provider": "local"})
	worker, _ := memoryStore.RegisterWorker(projectID, "worker", "local")
	racing := &policyRaceStore{Store: memoryStore}
	racing.beforeClaim = func() {
		bindAPIProjectPolicy(t, memoryStore, projectID, denyPolicyIDs("deny_resnet", "models", "resnet18"))
	}
	assigned, err := newServer(racing).pollNextPolicyPermittedJob(worker.ID, store.JobPollFilter{})
	if assigned != nil || !errors.Is(err, store.ErrNoJob) {
		t.Fatalf("racing claim assigned forbidden job %#v, err = %v", assigned, err)
	}
	blocked, _ := memoryStore.GetJob(job.ID)
	if blocked.Status != jobs.StatusQueued || blocked.PolicyEligibilityStatus != jobs.PolicyEligibilityBlocked || blocked.WorkerID != "" {
		t.Fatalf("racing job state = %#v", blocked)
	}
	evaluations, _ := memoryStore.ListExperimentPolicyEvaluations(projectID)
	if !hasReasonCode(evaluations, policies.ReasonChangedAfterQueue) {
		t.Fatalf("race audit omitted %s: %#v", policies.ReasonChangedAfterQueue, evaluations)
	}
}

func denyPolicyIDs(ruleID, category string, ids ...string) policies.Rule {
	return policies.Rule{ID: ruleID, Effect: policies.EffectDeny, Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: category, IDs: ids}}
}

func performJSONRequest(t *testing.T, handler http.Handler, method, path string, payload any) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func hasPolicyEvaluation(evaluations []policies.Evaluation, operation, decision string) bool {
	for _, evaluation := range evaluations {
		if evaluation.Operation == operation && evaluation.Decision == decision {
			return true
		}
	}
	return false
}

func hasReasonCode(evaluations []policies.Evaluation, code policies.ReasonCode) bool {
	for _, evaluation := range evaluations {
		for _, candidate := range evaluation.ReasonCodes {
			if candidate == code {
				return true
			}
		}
	}
	return false
}
