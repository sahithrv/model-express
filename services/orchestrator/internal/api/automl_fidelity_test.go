package api

import (
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
)

func TestDetectionPlanValidationRejectsClassificationOnlyAutoMLField(t *testing.T) {
	experiment := plans.PlannedExperiment{
		Template: "yolo11_detection", Model: "yolo11n.pt", Epochs: 8,
		BatchSize: 8, LearningRate: 0.001, ImageSize: 640, Reason: "invalid detector AutoML",
	}
	minValue, maxValue := 0.0, 0.5
	experiment.AutoML = &automl.ExperimentAutoML{
		Enabled: true,
		SearchSpace: &automl.HyperparameterSearchSpace{
			Parameters: []automl.HyperparameterParameterSpec{{
				Name: "dropout", Type: automl.ParameterFloat, Min: &minValue, Max: &maxValue,
			}},
		},
	}
	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(err.Error(), "dropout") {
		t.Fatalf("detection plan accepted classifier-only AutoML dropout: %v", err)
	}
}

func TestDetectionAutoMLTrialUsesOnlyExecutableCapabilityScopedFields(t *testing.T) {
	experiment := plans.PlannedExperiment{
		Template: "yolo11_detection", Model: "yolo11n.pt", Epochs: 8,
		BatchSize: 8, LearningRate: 0.001, ImageSize: 640, Reason: "detector AutoML fidelity",
	}
	scope, err := automl.CurrentExecutionScope("object_detection", "modal_ultralytics")
	if err != nil {
		t.Fatalf("build detection scope: %v", err)
	}
	experiment.AutoML, err = defaultBackendAutoMLForExperiment(experiment, automl.SamplerSeededRandom, scope)
	if err != nil {
		t.Fatalf("build detection AutoML defaults: %v", err)
	}
	prepared, err := prepareAutoMLExperimentWithHistoryForExecution(experiment, 0, automl.SamplerSeededRandom, nil, scope)
	if err != nil {
		t.Fatalf("prepare detection AutoML trial: %v", err)
	}
	if prepared.AutoML.CapabilityVersion != scope.CapabilityVersion || prepared.AutoML.Task != scope.Task || prepared.AutoML.Runner != scope.Runner {
		t.Fatalf("prepared trial omitted capability scope: %#v", prepared.AutoML)
	}
	for _, forbidden := range []string{"dropout", "label_smoothing", "weight_decay", "early_stopping_patience", "gradient_clip_norm"} {
		if automl.CoversParameter(prepared.AutoML.SearchSpace, forbidden) {
			t.Fatalf("detection search space included classifier/no-op field %s: %#v", forbidden, prepared.AutoML.SearchSpace)
		}
		if _, ok := prepared.AutoML.Suggestion.Values[forbidden]; ok {
			t.Fatalf("detection suggestion sampled classifier/no-op field %s: %#v", forbidden, prepared.AutoML.Suggestion.Values)
		}
	}
	for _, required := range []string{"learning_rate", "batch_size", "epochs"} {
		if !automl.CoversParameter(prepared.AutoML.SearchSpace, required) {
			t.Fatalf("detection search space omitted executable field %s: %#v", required, prepared.AutoML.SearchSpace)
		}
	}
	spec, err := buildExecutionSpecV1(prepared, "modal")
	if err != nil {
		t.Fatalf("build prepared execution spec: %v", err)
	}
	report, err := execution.ValidateExecutionSpecV1(spec, "yolo11", execution.ValidationModeEnforce)
	if err != nil {
		t.Fatalf("validate prepared execution spec: %v", err)
	}
	if report.WouldBlock {
		t.Fatalf("generated detection trial failed task-aware execution validation: %#v", report.Findings)
	}
}

func TestLocalClassificationAutoMLSpaceValidatesWithRecordedRunnerScope(t *testing.T) {
	experiment := testExperiment("resnet18", 8)
	scope, err := automl.CurrentExecutionScope("image_classification", "local_simulator")
	if err != nil {
		t.Fatalf("build local classification scope: %v", err)
	}
	experiment.AutoML, err = defaultBackendAutoMLForExperiment(experiment, automl.SamplerSeededRandom, scope)
	if err != nil {
		t.Fatalf("build local AutoML defaults: %v", err)
	}
	prepared, err := prepareAutoMLExperimentWithHistoryForExecution(experiment, 0, automl.SamplerSeededRandom, nil, scope)
	if err != nil {
		t.Fatalf("prepare local AutoML: %v", err)
	}
	if automl.CoversParameter(prepared.AutoML.SearchSpace, "early_stopping_patience") {
		t.Fatalf("local simulator space retained an unsupported conditional: %#v", prepared.AutoML.SearchSpace)
	}
	if err := validatePlannedExperiment(prepared, 0); err != nil {
		t.Fatalf("recorded local runner scope did not survive plan validation: %v", err)
	}
}

func TestPersistedAutoMLStudyAndSuggestionRecordCapabilityScope(t *testing.T) {
	server, _, sourcePlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	experiment := testExperiment("resnet18", 8)
	scope, err := automl.CurrentExecutionScope("image_classification", "modal_torchvision")
	if err != nil {
		t.Fatalf("build classification scope: %v", err)
	}
	experiment.AutoML, err = defaultBackendAutoMLForExperiment(experiment, automl.SamplerSeededRandom, scope)
	if err != nil {
		t.Fatalf("build AutoML defaults: %v", err)
	}
	experiment, err = prepareAutoMLExperimentWithHistoryForExecution(experiment, 0, automl.SamplerSeededRandom, nil, scope)
	if err != nil {
		t.Fatalf("prepare AutoML experiment: %v", err)
	}
	plan, err := server.store.CreateExperimentPlan(
		sourcePlan.ProjectID, sourcePlan.DatasetID, "macro_f1", 1, 10,
		[]plans.PlannedExperiment{experiment}, nil, "",
	)
	if err != nil {
		t.Fatalf("create scoped AutoML plan: %v", err)
	}
	if err := server.persistAutoMLForPlan(plan); err != nil {
		t.Fatalf("persist scoped AutoML records: %v", err)
	}
	studies, err := server.store.ListProjectOptimizerStudies(plan.ProjectID, 10)
	if err != nil || len(studies) == 0 {
		t.Fatalf("list persisted studies: studies=%#v err=%v", studies, err)
	}
	if studies[0].CapabilityVersion != scope.CapabilityVersion || studies[0].Task != scope.Task || studies[0].Runner != scope.Runner {
		t.Fatalf("study omitted capability scope: %#v", studies[0])
	}
	suggestions, err := server.store.ListPlanOptimizerSuggestions(plan.ID)
	if err != nil || len(suggestions) == 0 {
		t.Fatalf("list persisted suggestions: suggestions=%#v err=%v", suggestions, err)
	}
	if suggestions[0].CapabilityVersion != scope.CapabilityVersion || suggestions[0].Task != scope.Task || suggestions[0].Runner != scope.Runner {
		t.Fatalf("suggestion omitted capability scope: %#v", suggestions[0])
	}
	job, err := server.store.CreateJob(plan.ProjectID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id": plan.ID, "dataset_id": plan.DatasetID,
	})
	if err != nil {
		t.Fatalf("create AutoML trial job: %v", err)
	}
	trial, err := server.store.UpsertOptimizerTrial(automl.OptimizerTrial{
		StudyID: studies[0].ID, SuggestionID: suggestions[0].ID,
		ProjectID: plan.ProjectID, PlanID: plan.ID, DatasetID: plan.DatasetID, JobID: job.ID,
		Status: jobs.StatusSucceeded, TargetMetric: "macro_f1", CapabilityVersion: scope.CapabilityVersion,
		Task: scope.Task, Runner: scope.Runner, Metrics: map[string]any{"capability_version": scope.CapabilityVersion},
	})
	if err != nil {
		t.Fatalf("persist scoped AutoML trial: %v", err)
	}
	if trial.CapabilityVersion != scope.CapabilityVersion || trial.Task != scope.Task || trial.Runner != scope.Runner {
		t.Fatalf("trial omitted capability scope: %#v", trial)
	}
}
