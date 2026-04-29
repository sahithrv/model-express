package api

import (
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

type Server struct {
	store              store.Store
	autoReviewMu       sync.Mutex
	automationSettings settings.AutomationSettings
	settingsMu         sync.RWMutex
}

func NewRouter(store store.Store) *gin.Engine {
	server := newServer(store)

	router := gin.Default()

	router.GET("/healthz", server.health)
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
	router.POST("/projects/:id/review-experiments", server.reviewProjectExperiments)
	router.POST("/projects/:id/schedule-follow-up-experiments", server.scheduleFollowUpExperiments)
	router.GET("/projects/:id/agent-decisions", server.listProjectAgentDecisions)
	router.GET("/projects/:id/workers", server.listProjectWorkers)
	router.POST("/projects/:id/plans", server.createExperimentPlan)
	router.GET("/projects/:id/plans", server.listProjectPlans)
	router.GET("/plans/:id", server.listExperimentPlans)
	router.POST("/plans/:id/execute", server.executeExperimentPlan)

	router.GET("/datasets/:id", server.getDataset)
	router.POST("/datasets/:id/profile", server.updateDatasetProfile)

	router.GET("/jobs/:id", server.getJob)
	router.POST("/jobs/:id/metrics", server.reportMetric)
	router.GET("/jobs/:id/metrics", server.listJobMetrics)
	router.POST("/jobs/:id/training-run-summary", server.upsertTrainingRunSummary)
	router.GET("/jobs/:id/training-run-summary", server.getTrainingRunSummary)
	router.POST("/jobs/:id/complete", server.completeJob)
	router.POST("/jobs/:id/fail", server.failJob)

	router.GET("/workers", server.listWorkers)
	router.POST("/workers/register", server.registerWorker)
	router.GET("/workers/:id", server.getWorker)
	router.POST("/workers/:id/heartbeat", server.heartbeatWorker)
	router.POST("/workers/:id/poll", server.pollJob)

	return router
}

func newServer(store store.Store) *Server {
	server := &Server{
		store:              store,
		automationSettings: automationSettingsFromEnv(),
	}

	if automationSettings, err := store.GetAutomationSettings(); err == nil {
		server.automationSettings = automationSettings
	}

	return server
}

type healthResponse struct {
	Status    string    `json:"status"`
	Service   string    `json:"service"`
	Timestamp time.Time `json:"timestamp"`
}
