package api

import (
	"errors"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestMismatchedRunIsNotCreditedAsPlannerEvidence(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "quality")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "sha", 1)
	plan, _ := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 1, []plans.PlannedExperiment{{Model: "resnet18", Mechanism: "class_imbalance"}}, nil, "")
	job, summary := createFidelityTestRun(t, memoryStore, project.ID, dataset.ID, plan.ID, "class_imbalance", execution.ExecutionVerdictMismatch, 0.91)
	server := newServer(memoryStore)
	evidenceByJob, err := server.executionEvidenceForJobs([]jobs.ExperimentJob{job})
	if err != nil {
		t.Fatal(err)
	}
	outcome, err := experimentPlanningOutcomeForPlan(
		decisions.AgentDecision{ID: "decision", Payload: map[string]any{"expected_delta_vs_champion": 0.01}},
		plan,
		[]plans.ExperimentPlan{plan},
		[]runs.TrainingRunSummary{summary},
		nil,
		agents.ProjectObjectiveContext{},
		evidenceByJob,
	)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.OutcomeStatus != agents.ExperimentPlanningOutcomeExecutionIneligible || outcome.ActualBestRun != nil || outcome.EvidenceEligibleRunCount != 0 {
		t.Fatalf("mismatched outcome was credited: %#v", outcome)
	}
	if len(outcome.ExecutionEvidence) != 1 || outcome.ExecutionEvidence[0].RequestedMechanism != "class_imbalance" || outcome.ExecutionEvidence[0].RealizedMechanismIdentity == "" {
		t.Fatalf("requested/realized identity was not preserved separately: %#v", outcome.ExecutionEvidence)
	}
}

func TestAutomaticChampionSelectionSkipsMismatchAndSelectsMatchedRun(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "quality")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "sha", 1)
	plan, _ := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 1, nil, nil, "")
	mismatchJob, _ := createFidelityTestRun(t, memoryStore, project.ID, dataset.ID, plan.ID, "resolution_crop", execution.ExecutionVerdictMismatch, 0.99)
	matchedJob, _ := createFidelityTestRun(t, memoryStore, project.ID, dataset.ID, plan.ID, "optimizer_change", execution.ExecutionVerdictMatched, 0.72)
	decision, _ := memoryStore.CreateAgentDecision(project.ID, plan.ID, decisions.TypeSelectChampion, "select best", map[string]any{
		"champion_job_id": mismatchJob.ID,
		"target_metric":   "macro_f1",
		"decision_source": "automatic_test",
	})
	server := newServer(memoryStore)
	if err := server.persistProjectChampionFromDecision(project.ID, decision); err != nil {
		t.Fatal(err)
	}
	champion, err := memoryStore.GetProjectChampion(project.ID)
	if err != nil {
		t.Fatal(err)
	}
	if champion.JobID != matchedJob.ID {
		t.Fatalf("champion = %s, want fidelity-eligible %s", champion.JobID, matchedJob.ID)
	}
	evidence := payloadMap(champion.Metrics, "execution_evidence")
	if payloadString(evidence, "fidelity_verdict") != execution.ExecutionVerdictMatched {
		t.Fatalf("champion fidelity evidence = %#v", evidence)
	}
}

func TestSimulatedRunCannotBecomeAutomaticChampion(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "quality")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "sha", 1)
	plan, _ := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 1, nil, nil, "")
	job, _ := createFidelityTestRun(t, memoryStore, project.ID, dataset.ID, plan.ID, "baseline_control", execution.ExecutionVerdictSimulated, 0.95)
	decision, _ := memoryStore.CreateAgentDecision(project.ID, plan.ID, decisions.TypeSelectChampion, "select simulator", map[string]any{"champion_job_id": job.ID})
	err := newServer(memoryStore).persistProjectChampionFromDecision(project.ID, decision)
	if err == nil || !errors.Is(err, store.ErrInvalidRequest) {
		t.Fatalf("expected simulated champion rejection, got %v", err)
	}
	if _, err := memoryStore.GetProjectChampion(project.ID); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("simulated run became champion: %v", err)
	}
}

func TestLegacyExecutionEvidencePolicyKeepsOldRunsReadable(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("project", "quality")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/data", "sha", 1)
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"mechanism": "legacy_mechanism", "dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	server := newServer(memoryStore)

	t.Setenv("MODEL_EXPRESS_LEGACY_EXECUTION_EVIDENCE_POLICY", execution.LegacyEvidencePolicyVisibleOnly)
	visibleOnly, err := server.executionEvidenceForJobs([]jobs.ExperimentJob{job})
	if err != nil {
		t.Fatal(err)
	}
	legacy := visibleOnly[job.ID]
	if legacy.FidelityVerdict != execution.ExecutionVerdictUnverified || legacy.LearningEligible || legacy.AutomaticChampionEligible {
		t.Fatalf("visible-only legacy policy = %#v", legacy)
	}
	if legacy.RequestedMechanism != "legacy_mechanism" {
		t.Fatalf("legacy requested mechanism was lost: %#v", legacy)
	}

	t.Setenv("MODEL_EXPRESS_LEGACY_EXECUTION_EVIDENCE_POLICY", execution.LegacyEvidencePolicyAllow)
	compatible, _ := server.executionEvidenceForJobs([]jobs.ExperimentJob{job})
	if !compatible[job.ID].LearningEligible || !compatible[job.ID].AutomaticChampionEligible {
		t.Fatalf("legacy compatibility policy did not retain readability/eligibility: %#v", compatible[job.ID])
	}
}

func createFidelityTestRun(
	t *testing.T,
	memoryStore *store.MemoryStore,
	projectID string,
	datasetID string,
	planID string,
	mechanism string,
	verdict string,
	score float64,
) (jobs.ExperimentJob, runs.TrainingRunSummary) {
	t.Helper()
	requested := map[string]any{"model": "resnet18", "epochs": 3, "batch_size": 16, "mechanism": mechanism}
	spec, err := execution.BuildExecutionSpecV1("image_classification", "local_simulator", requested, requested)
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := spec.Payload()
	job, err := memoryStore.CreateJob(projectID, jobs.TemplateTrainExperiment, map[string]any{
		execution.ExecutionSpecConfigKey: payload,
		"dataset_id":                     datasetID,
		"plan_id":                        planID,
		"model":                          "resnet18",
		"mechanism":                      mechanism,
	})
	if err != nil {
		t.Fatal(err)
	}
	attemptID := job.ID + "-attempt-1"
	if _, err := memoryStore.CreateAttemptExecutionRecord(job.ID, attemptID, 1); err != nil {
		t.Fatal(err)
	}
	record, _ := memoryStore.GetJobExecutionRecord(job.ID)
	realized := clonePayload(record.AcceptedSpec.AcceptedSpec)
	simulated := verdict == execution.ExecutionVerdictSimulated
	if verdict == execution.ExecutionVerdictMismatch {
		realized["batch_size"] = 8
	}
	if _, _, err := memoryStore.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{
		AttemptID: attemptID, SchemaVersion: execution.ExecutionObservationSchemaV1,
		Stage: execution.ExecutionObservationFinalized, IdempotencyKey: "final",
		RealizedConfig: realized, Simulated: simulated,
	}); err != nil {
		t.Fatal(err)
	}
	runtime := 12.0
	cost := 0.2
	accuracy := score
	epochs := 3
	summary, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model: "resnet18", Status: jobs.StatusSucceeded, BestMacroF1: &score,
		BestAccuracy: &accuracy, RuntimeSeconds: &runtime, EstimatedCostUSD: &cost, EpochsCompleted: &epochs,
	})
	if err != nil {
		t.Fatal(err)
	}
	return job, summary
}

func finalizeMatchedExecutionForTest(t *testing.T, memoryStore *store.MemoryStore, jobID string) {
	t.Helper()
	attemptID := jobID + "-matched-attempt"
	if _, err := memoryStore.CreateAttemptExecutionRecord(jobID, attemptID, 1); err != nil {
		t.Fatal(err)
	}
	record, err := memoryStore.GetJobExecutionRecord(jobID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := memoryStore.AppendRealizationObservation(jobID, execution.RealizationObservationCreate{
		AttemptID: attemptID, SchemaVersion: execution.ExecutionObservationSchemaV1,
		Stage: execution.ExecutionObservationFinalized, IdempotencyKey: "matched-final",
		RealizedConfig: record.AcceptedSpec.AcceptedSpec,
	}); err != nil {
		t.Fatal(err)
	}
}
