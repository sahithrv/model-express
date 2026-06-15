package api

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

type Server struct {
	store                       store.Store
	callbackSecret              []byte
	autoReviewMu                sync.Mutex
	trainingTerminalHooksMu     sync.Mutex
	trainingTerminalHooksQueued map[string]bool
	automationSettings          settings.AutomationSettings
	settingsMu                  sync.RWMutex
}

const defaultOrchestratorAddr = "127.0.0.1:8080"
const defaultJSONRequestBodyMaxBytes int64 = 2 * 1024 * 1024

func ResolveOrchestratorListenAddr() (string, error) {
	addr := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_ORCHESTRATOR_ADDR"))
	if addr == "" {
		addr = defaultOrchestratorAddr
	}
	apiToken := os.Getenv("MODEL_EXPRESS_API_TOKEN")
	if err := validateOrchestratorListenAddr(
		addr,
		envFlag("MODEL_EXPRESS_ALLOW_LAN", false),
		apiToken,
	); err != nil {
		return "", err
	}
	if err := validateOrchestratorExposureConfig(apiToken); err != nil {
		return "", err
	}
	return addr, nil
}

func validateOrchestratorListenAddr(addr string, allowLAN bool, apiToken string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("MODEL_EXPRESS_ORCHESTRATOR_ADDR must be host:port: %w", err)
	}
	if listenHostIsLoopback(host) {
		return nil
	}
	if !allowLAN {
		return fmt.Errorf("refusing non-loopback orchestrator bind %q without MODEL_EXPRESS_ALLOW_LAN=true", addr)
	}
	if strings.TrimSpace(apiToken) == "" {
		return fmt.Errorf("MODEL_EXPRESS_API_TOKEN is required when binding orchestrator to %q", addr)
	}
	return nil
}

func validateOrchestratorExposureConfig(apiToken string) error {
	if !orchestratorExposedMode() {
		return nil
	}
	if strings.TrimSpace(apiToken) == "" {
		return fmt.Errorf("MODEL_EXPRESS_API_TOKEN is required when LAN, tunnel, or public Modal orchestrator URL exposure is enabled")
	}
	return nil
}

func listenHostIsLoopback(host string) bool {
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func NewRouter(store store.Store) *gin.Engine {
	server := newServer(store)

	router := gin.Default()
	router.Use(jsonRequestBodyLimitMiddleware(jsonRequestBodyMaxBytes()))

	router.GET("/healthz", server.health)

	if apiToken, required := apiTokenForExposedMode(); required {
		router.Use(apiTokenMiddleware(apiToken))
	}

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
	router.POST("/projects/:id/cancel-active-executions", server.cancelProjectActiveExecutions)
	router.GET("/projects/:id/execution-events", server.listProjectExecutionEvents)
	router.POST("/projects/:id/dispatcher-events", server.reportProjectDispatcherEvent)
	router.GET("/projects/:id/events/stream", server.streamProjectExecutionEvents)
	router.GET("/projects/:id/activity-stream", server.streamProjectActivityEvents)
	router.GET("/projects/:id/workers", server.listProjectWorkers)
	router.POST("/projects/:id/plans", server.createExperimentPlan)
	router.GET("/projects/:id/plans", server.listProjectPlans)
	router.GET("/plans/:id", server.listExperimentPlans)
	router.POST("/plans/:id/execute", server.executeExperimentPlan)
	router.POST("/plans/:id/cancel-active-execution", server.cancelPlanActiveExecution)

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
	router.POST("/jobs/:id/modal-call", server.recordJobModalCall)
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
		callbackSecret:              callbackSecretFromEnv(),
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

func jsonRequestBodyLimitMiddleware(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Body != nil && requestHasJSONBody(c.Request) {
			if c.Request.ContentLength > maxBytes {
				c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "JSON request body too large"})
				return
			}
			c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		}
		c.Next()
	}
}

func requestHasJSONBody(req *http.Request) bool {
	if req == nil || req.Body == nil {
		return false
	}
	contentType := strings.ToLower(strings.TrimSpace(req.Header.Get("Content-Type")))
	return contentType == "" || strings.Contains(contentType, "application/json")
}

func jsonRequestBodyMaxBytes() int64 {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_JSON_BODY_MAX_BYTES"))
	if value == "" {
		return defaultJSONRequestBodyMaxBytes
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed <= 0 {
		return defaultJSONRequestBodyMaxBytes
	}
	if parsed < 64*1024 {
		return 64 * 1024
	}
	if parsed > 16*1024*1024 {
		return 16 * 1024 * 1024
	}
	return parsed
}

func apiTokenForExposedMode() (string, bool) {
	if !orchestratorExposedMode() {
		return "", false
	}
	return strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")), true
}

func orchestratorExposedMode() bool {
	if envFlag("MODEL_EXPRESS_ALLOW_LAN", false) || envFlag("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", false) {
		return true
	}
	for _, name := range []string{"MODAL_ORCHESTRATOR_URL", "MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL"} {
		if publicOrTunnelURL(os.Getenv(name)) {
			return true
		}
	}
	return false
}

func publicOrTunnelURL(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Hostname() == "" {
		return false
	}
	return !listenHostIsLoopback(parsed.Hostname())
}

func apiTokenMiddleware(apiToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if callbackEndpointUsesAttemptToken(c.Request.Method, c.Request.URL.Path) {
			c.Next()
			return
		}
		if !secureTokenEqual(apiTokenFromRequest(c), apiToken) {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid API token"})
			return
		}
		c.Next()
	}
}

func callbackEndpointUsesAttemptToken(method string, path string) bool {
	if method != http.MethodPost {
		return false
	}
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) != 3 || parts[0] != "jobs" || strings.TrimSpace(parts[1]) == "" {
		return false
	}
	switch parts[2] {
	case "metrics",
		"training-run-summary",
		"training-run-evaluation",
		"modal-call",
		"champion-export-result",
		"champion-demo-prediction-result",
		"complete",
		"fail":
		return true
	default:
		return false
	}
}

func apiTokenFromRequest(c *gin.Context) string {
	if token := strings.TrimSpace(c.GetHeader("X-Model-Express-Api-Token")); token != "" {
		return token
	}
	return bearerToken(c.GetHeader("Authorization"))
}

func bearerToken(header string) string {
	header = strings.TrimSpace(header)
	const prefix = "Bearer "
	if len(header) < len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return ""
	}
	return strings.TrimSpace(header[len(prefix):])
}

type healthResponse struct {
	Status    string    `json:"status"`
	Service   string    `json:"service"`
	Timestamp time.Time `json:"timestamp"`
}
