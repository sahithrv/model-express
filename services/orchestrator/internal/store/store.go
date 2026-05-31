package store

import (
	"errors"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/strategies"
	"model-express/services/orchestrator/internal/workers"
)

var (
	ErrNotFound       = errors.New("not found")
	ErrNoJob          = errors.New("no job available")
	ErrInvalidRequest = errors.New("invalid request")
)

type JobPollFilter struct {
	Provider                            string
	Templates                           []string
	IncludeUnspecifiedProviderTemplates []string
}

func (filter JobPollFilter) Matches(job jobs.ExperimentJob) bool {
	templates := normalizedSet(filter.Templates)
	if len(templates) > 0 {
		if _, ok := templates[strings.ToLower(strings.TrimSpace(job.Template))]; !ok {
			return false
		}
	}

	provider := strings.ToLower(strings.TrimSpace(filter.Provider))
	if provider == "" {
		return true
	}
	jobProvider := strings.ToLower(strings.TrimSpace(configString(job.Config, "provider")))
	if jobProvider == provider {
		return true
	}
	if jobProvider != "" {
		return false
	}
	includeUnspecified := normalizedSet(filter.IncludeUnspecifiedProviderTemplates)
	_, ok := includeUnspecified[strings.ToLower(strings.TrimSpace(job.Template))]
	return ok
}

type Store interface {
	CreateProject(name string, goal string) (projects.Project, error)
	GetProject(id string) (projects.Project, error)
	ListProjects() ([]projects.Project, error)

	CreateDataset(projectID string, name string, storageURI string, checksumSHA256 string, sizeBytes int64) (datasets.Dataset, error)
	GetDataset(id string) (datasets.Dataset, error)
	ListProjectDatasets(projectID string) ([]datasets.Dataset, error)
	UpdateDatasetProfile(id string, profile map[string]any) (datasets.Dataset, error)
	CreateDatasetMetadataImport(importRecord datasets.DatasetMetadataImport) (datasets.DatasetMetadataImport, error)
	GetDatasetMetadataImport(importID string) (datasets.DatasetMetadataImport, error)
	GetActiveDatasetMetadataImport(datasetID string) (datasets.DatasetMetadataImport, error)
	ListDatasetMetadataImports(datasetID string) ([]datasets.DatasetMetadataImport, error)
	GetDatasetMetadataBundle(datasetID string, importID string, includeAnnotations bool, limit int, offset int) (datasets.DatasetMetadataBundle, error)
	CreateDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error)
	RejectDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error)
	GetLatestAcceptedDatasetVisualAnalysis(datasetID string) (datasets.DatasetVisualAnalysis, error)
	ListDatasetVisualAnalyses(datasetID string) ([]datasets.DatasetVisualAnalysis, error)

	RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error)
	ListWorkers() ([]workers.Worker, error)
	ListProjectWorkers(projectID string) ([]workers.Worker, error)
	GetWorker(workerID string) (workers.Worker, error)
	HeartbeatWorker(id string) (workers.Worker, error)
	PollJob(workerID string, filter JobPollFilter) (*jobs.ExperimentJob, error)

	CreateJob(projectID string, template string, config map[string]any) (jobs.ExperimentJob, error)
	GetJob(id string) (jobs.ExperimentJob, error)
	ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error)
	RecoverExpiredJobLeases(now time.Time) ([]jobs.ExperimentJob, error)
	ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error)
	ListJobMetrics(jobID string) ([]jobs.EpochMetric, error)
	CompleteJob(jobID string, mlflowRunID string) (jobs.ExperimentJob, error)
	RetryJob(jobID string, message string) (jobs.ExperimentJob, bool, error)
	FailJob(jobID string, message string) (jobs.ExperimentJob, error)

	UpsertTrainingRunSummary(jobID string, update runs.TrainingRunSummaryUpdate) (runs.TrainingRunSummary, error)
	GetTrainingRunSummary(jobID string) (runs.TrainingRunSummary, error)
	ListProjectTrainingRunSummaries(projectID string) ([]runs.TrainingRunSummary, error)
	UpsertTrainingRunEvaluation(jobID string, update runs.TrainingRunEvaluationUpdate) (runs.TrainingRunEvaluation, error)
	GetTrainingRunEvaluation(jobID string) (runs.TrainingRunEvaluation, error)
	ListProjectTrainingRunEvaluations(projectID string) ([]runs.TrainingRunEvaluation, error)
	UpsertProjectChampion(champion runs.ProjectChampionUpsert) (runs.ProjectChampion, error)
	GetProjectChampion(projectID string) (runs.ProjectChampion, error)
	CreateChampionExport(export runs.ChampionExportCreate) (runs.ChampionExport, error)
	UpdateChampionExport(id string, update runs.ChampionExportUpdate) (runs.ChampionExport, error)
	ListProjectChampionExports(projectID string) ([]runs.ChampionExport, error)
	CreateChampionDemoPrediction(prediction runs.ChampionDemoPredictionCreate) (runs.ChampionDemoPrediction, error)
	UpdateChampionDemoPrediction(id string, update runs.ChampionDemoPredictionUpdate) (runs.ChampionDemoPrediction, error)
	ListProjectChampionDemoPredictions(projectID string) ([]runs.ChampionDemoPrediction, error)

	CreateAgentDecision(projectID string, planID string, decisionType string, rationale string, payload map[string]any) (decisions.AgentDecision, error)
	ListProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error)

	GetAutomationSettings() (settings.AutomationSettings, error)
	SaveAutomationSettings(automationSettings settings.AutomationSettings) (settings.AutomationSettings, error)

	UpsertWorkerRequirement(projectID string, planID string, provider string, gpuType string, targetCount int, source string, policy execution.WorkerRequirementPolicy) (execution.WorkerRequirement, bool, error)
	ListProjectWorkerRequirements(projectID string) ([]execution.WorkerRequirement, error)
	UpdateWorkerRequirement(id string, update execution.WorkerRequirementUpdate) (execution.WorkerRequirement, error)
	CreateExecutionEvent(projectID string, planID string, eventType string, message string, payload map[string]any) (execution.ExecutionEvent, error)
	ListProjectExecutionEvents(projectID string, limit int) ([]execution.ExecutionEvent, error)

	CreateAgentMemoryRecord(record memory.AgentMemoryRecord) (memory.AgentMemoryRecord, error)
	ListProjectAgentMemoryRecords(projectID string, filter memory.AgentMemoryFilter) ([]memory.AgentMemoryRecord, error)
	CreateAgentInvocation(invocation memory.AgentInvocation) (memory.AgentInvocation, error)
	GetAgentInvocation(invocationID string) (memory.AgentInvocation, error)
	UpdateAgentInvocationDownstreamOutcome(invocationID string, outcome map[string]any) (memory.AgentInvocation, error)
	ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error)
	CreateStrategyScorecard(scorecard strategies.StrategyScorecardCreate) (strategies.StrategyScorecard, error)
	UpdateStrategyScorecardOutcomeByFollowUpPlan(followUpPlanID string, update strategies.StrategyScorecardOutcomeUpdate) (strategies.StrategyScorecard, error)
	ListProjectStrategyScorecards(projectID string, limit int) ([]strategies.StrategyScorecard, error)
	CreateOptimizerStudy(study automl.OptimizerStudy) (automl.OptimizerStudy, error)
	ListProjectOptimizerStudies(projectID string, limit int) ([]automl.OptimizerStudy, error)
	CreateOptimizerSuggestion(suggestion automl.OptimizerSuggestion) (automl.OptimizerSuggestion, error)
	UpdateOptimizerSuggestionJob(suggestionID string, jobID string) (automl.OptimizerSuggestion, error)
	ListPlanOptimizerSuggestions(planID string) ([]automl.OptimizerSuggestion, error)
	UpsertOptimizerTrial(trial automl.OptimizerTrial) (automl.OptimizerTrial, error)
	ListProjectOptimizerTrials(projectID string, limit int) ([]automl.OptimizerTrial, error)
	ListStudyOptimizerTrials(studyID string) ([]automl.OptimizerTrial, error)

	CreateExperimentPlan(projectID string, datasetID string, targetMetric string, recommendedWorkers int, estimatedMinutes int, experiments []plans.PlannedExperiment, warnings []string, sourceDecisionID string) (plans.ExperimentPlan, error)
	GetExperimentPlan(id string) (plans.ExperimentPlan, error)
	ListProjectExperimentPlans(projectID string) ([]plans.ExperimentPlan, error)
}

func normalizedSet(values []string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, value := range values {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if normalized == "" {
			continue
		}
		out[normalized] = struct{}{}
	}
	return out
}

func configString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}
