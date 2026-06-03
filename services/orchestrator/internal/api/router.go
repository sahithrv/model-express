package api

import (
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

type Server struct {
	store                       store.Store
	autoReviewMu                sync.Mutex
	trainingTerminalHooksMu     sync.Mutex
	trainingTerminalHooksQueued map[string]bool
	automationSettings          settings.AutomationSettings
	settingsMu                  sync.RWMutex
}

func NewRouter(store store.Store) *gin.Engine {
	server := newServer(store)

	router := gin.Default()

	router.GET("/healthz", server.health)
	router.GET("/automl/capabilities", server.getAutoMLCapabilities)
	router.GET("/settings/automation", server.getAutomationSettings)
	router.PATCH("/settings/automation", server.updateAutomationSettings)

	router.POST("/projects", server.createProject)
	router.GET("/projects", server.listProjects)
	router.GET("/projects/:id", server.getProject)
	router.POST("/projects/:id/datasets", server.createDataset)
	router.GET("/projects/:id/datasets", server.listProjectDatasets)
	router.POST("/projects/:id/jobs", server.createJob)
	router.GET("/projects/:id/jobs", server.listProjectJobs)
	router.GET("/projects/:id/training-run-summaries", server.listProjectTrainingRunSummaries)
	router.GET("/projects/:id/training-run-evaluations", server.listProjectTrainingRunEvaluations)
	router.GET("/projects/:id/champion", server.getProjectChampion)
	router.GET("/projects/:id/champion/exports", server.listProjectChampionExports)
	router.POST("/projects/:id/champion/exports", server.createProjectChampionExport)
	router.GET("/projects/:id/champion/demo-images", server.listProjectChampionDemoImages)
	router.GET("/projects/:id/champion/demo-predictions", server.listProjectChampionDemoPredictions)
	router.POST("/projects/:id/champion/demo-predictions", server.createProjectChampionDemoPrediction)
	router.GET("/projects/:id/champion/feedback", server.listProjectChampionFeedback)
	router.POST("/projects/:id/champion/feedback", server.createProjectChampionFeedback)
	router.POST("/projects/:id/experimentation/reopen", server.reopenProjectExperimentation)
	router.POST("/projects/:id/review-experiments", server.reviewProjectExperiments)
	router.POST("/projects/:id/schedule-follow-up-experiments", server.scheduleFollowUpExperiments)
	router.GET("/projects/:id/agent-decisions", server.listProjectAgentDecisions)
	router.GET("/projects/:id/agent-memory", server.listProjectAgentMemoryRecords)
	router.POST("/projects/:id/memory-embeddings/backfill", server.backfillProjectMemoryEmbeddings)
	router.GET("/projects/:id/agent-invocations", server.listProjectAgentInvocations)
	router.GET("/projects/:id/telemetry-summary", server.getProjectTelemetrySummary)
	router.GET("/projects/:id/strategy-scorecards", server.listProjectStrategyScorecards)
	router.GET("/projects/:id/worker-requirements", server.listProjectWorkerRequirements)
	router.GET("/projects/:id/execution-events", server.listProjectExecutionEvents)
	router.GET("/projects/:id/events/stream", server.streamProjectExecutionEvents)
	router.GET("/projects/:id/activity-stream", server.streamProjectActivityEvents)
	router.GET("/projects/:id/workers", server.listProjectWorkers)
	router.POST("/projects/:id/plans", server.createExperimentPlan)
	router.GET("/projects/:id/plans", server.listProjectPlans)
	router.GET("/plans/:id", server.listExperimentPlans)
	router.POST("/plans/:id/execute", server.executeExperimentPlan)

	router.GET("/datasets/:id", server.getDataset)
	router.POST("/datasets/:id/profile", server.updateDatasetProfile)
	router.POST("/datasets/:id/metadata/imports", server.importDatasetMetadata)
	router.GET("/datasets/:id/metadata/imports", server.listDatasetMetadataImports)
	router.GET("/datasets/:id/metadata/imports/:import_id", server.getDatasetMetadataImport)
	router.GET("/datasets/:id/metadata/summary", server.getDatasetMetadataSummary)
	router.GET("/datasets/:id/metadata/bundle", server.getDatasetMetadataBundle)
	router.GET("/datasets/:id/visual-exemplars", server.listDatasetVisualExemplars)
	router.POST("/datasets/:id/visual-exemplars", server.mergeDatasetVisualExemplars)
	router.GET("/datasets/:id/visual-analyses", server.listDatasetVisualAnalyses)
	router.GET("/datasets/:id/visual-analyses/latest", server.getLatestDatasetVisualAnalysis)
	router.POST("/datasets/:id/visual-analyses/run", server.runDatasetVisualAnalysis)
	router.POST("/datasets/:id/visual-analysis-result", server.reportDatasetVisualAnalysisResult)

	router.GET("/jobs/:id", server.getJob)
	router.POST("/jobs/:id/metrics", server.reportMetric)
	router.GET("/jobs/:id/metrics", server.listJobMetrics)
	router.POST("/jobs/:id/training-run-summary", server.upsertTrainingRunSummary)
	router.GET("/jobs/:id/training-run-summary", server.getTrainingRunSummary)
	router.POST("/jobs/:id/training-run-evaluation", server.upsertTrainingRunEvaluation)
	router.GET("/jobs/:id/training-run-evaluation", server.getTrainingRunEvaluation)
	router.POST("/jobs/:id/champion-export-result", server.reportChampionExportResult)
	router.POST("/jobs/:id/champion-demo-prediction-result", server.reportChampionDemoPredictionResult)
	router.POST("/jobs/:id/complete", server.completeJob)
	router.POST("/jobs/:id/fail", server.failJob)

	router.PATCH("/worker-requirements/:id", server.updateWorkerRequirement)

	router.GET("/workers", server.listWorkers)
	router.POST("/workers/register", server.registerWorker)
	router.GET("/workers/:id", server.getWorker)
	router.POST("/workers/:id/heartbeat", server.heartbeatWorker)
	router.POST("/workers/:id/poll", server.pollJob)

	return router
}

func newServer(store store.Store) *Server {
	server := &Server{
		store:                       store,
		trainingTerminalHooksQueued: make(map[string]bool),
		automationSettings:          automationSettingsFromEnv(),
	}

	if automationSettings, err := store.GetAutomationSettings(); err == nil {
		server.automationSettings = automationSettings
	}
	server.automationSettings.AgentMode = llm.NormalizeAgentMode(server.automationSettings.AgentMode)
	if server.automationSettings.LLMProvider == "" {
		server.automationSettings.LLMProvider = llm.ProviderOpenAI
	}
	server.automationSettings.AutoMLSampler = normalizeAutoMLSampler(server.automationSettings.AutoMLSampler)

	return server
}

type healthResponse struct {
	Status    string    `json:"status"`
	Service   string    `json:"service"`
	Timestamp time.Time `json:"timestamp"`
}
