package api

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	datasetmetadata "model-express/services/orchestrator/internal/datasets/metadata"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/embeddings"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
	"model-express/services/orchestrator/internal/workers"
)

type createProjectRequest struct {
	Name string `json:"name" binding:"required"`
	Goal string `json:"goal"`
}

type registerWorkerRequest struct {
	ProjectID string `json:"project_id" binding:"required"`
	Name      string `json:"name" binding:"required"`
	GPUType   string `json:"gpu_type"`
}

type createJobRequest struct {
	Template string         `json:"template" binding:"required"`
	Config   map[string]any `json:"config"`
}

type createDatasetRequest struct {
	Name           string `json:"name" binding:"required"`
	StorageURI     string `json:"storage_uri" binding:"required"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

type createExperimentPlanRequest struct {
	DatasetID          string                    `json:"dataset_id" binding:"required"`
	TargetMetric       string                    `json:"target_metric"`
	Priority           string                    `json:"priority"`
	MaxWorkers         int                       `json:"max_workers"`
	TimeBudgetMinutes  int                       `json:"time_budget_minutes"`
	RecommendedWorkers int                       `json:"recommended_workers"`
	EstimatedMinutes   int                       `json:"estimated_minutes"`
	Experiments        []plans.PlannedExperiment `json:"experiments"`
	Warnings           []string                  `json:"warnings"`
}

type executeExperimentPlanRequest struct {
	Provider string `json:"provider"`
	GPUType  string `json:"gpu_type"`
}

type executeExperimentPlanResponse struct {
	Plan plans.ExperimentPlan `json:"plan"`
	Jobs []jobs.ExperimentJob `json:"jobs"`
}

type scheduleFollowUpExperimentsResponse struct {
	Decision     decisions.AgentDecision `json:"decision"`
	FollowUpPlan *plans.ExperimentPlan   `json:"follow_up_plan,omitempty"`
}

type reopenExperimentationRequest struct {
	Reason       string `json:"reason"`
	SourcePlanID string `json:"source_plan_id"`
}

type reopenExperimentationResponse struct {
	Decision decisions.AgentDecision  `json:"decision"`
	Event    execution.ExecutionEvent `json:"event"`
}

type automaticExperimentReviewResult struct {
	Decision     *decisions.AgentDecision
	FollowUpPlan *plans.ExperimentPlan
	Jobs         []jobs.ExperimentJob
}

const (
	llmExperimentPlannerDecisionSource     = "llm_experiment_planner"
	minLLMDecisionConfidence               = 0.50
	maxLLMPlannerExperiments               = 5
	plannerMinimumMeaningfulImprovement    = 0.005
	plannerAutonomousMeaningfulImprovement = 0.010
	plannerNoImprovementRoundsToSelect     = 2
	plannerDefaultMaxFollowUpRounds        = 10
	plannerAutonomousMaxFollowUpRounds     = 3
	plannerBackendValidationRetryLimit     = 1

	visualAnalysisDefaultCooldownMinutes      = 360
	visualAnalysisDefaultMaxRunsPerProfile    = 3
	visualAnalysisDefaultLowMacroF1Threshold  = 0.55
	visualAnalysisDefaultWorstRecallThreshold = 0.40
	visualAnalysisDefaultConfusionThreshold   = 0.20

	datasetMetadataMaxSourceBytes      = 2_000_000
	datasetMetadataMaxTotalSourceBytes = 10_000_000

	memoryRetrievalDefaultMaxCards = 10
	memoryRetrievalDefaultMinScore = 0.55
)

var (
	errNoNovelFollowUpExperiments      = fmt.Errorf("%w: no novel follow-up experiments", store.ErrInvalidRequest)
	errChampionSelectedFollowUpBlocked = fmt.Errorf("%w: champion selected guard", errNoNovelFollowUpExperiments)
)

type updateDatasetProfileRequest struct {
	Profile map[string]any `json:"profile" binding:"required"`
}

type activeDatasetMetadataImportGetter interface {
	GetActiveDatasetMetadataImport(datasetID string) (datasets.DatasetMetadataImport, error)
}

type activeDatasetMetadataImportGetterWithContext interface {
	GetActiveDatasetMetadataImport(ctx context.Context, datasetID string) (datasets.DatasetMetadataImport, error)
}

type importDatasetMetadataRequest struct {
	StrictMode      bool                                 `json:"strict_mode"`
	SourceKind      string                               `json:"source_kind"`
	Sources         []datasetMetadataSourceRequest       `json:"sources"`
	Inventory       datasetmetadata.DatasetFileInventory `json:"inventory"`
	WorkerDiscovery map[string]any                       `json:"worker_discovery"`
	Warnings        []datasets.MetadataIssue             `json:"warnings"`
}

type datasetMetadataSourceRequest struct {
	RelativePath   string   `json:"relative_path"`
	StorageURI     string   `json:"storage_uri"`
	ChecksumSHA256 string   `json:"checksum_sha256"`
	SizeBytes      int64    `json:"size_bytes"`
	DeclaredFormat string   `json:"declared_format"`
	ContentBase64  string   `json:"content_base64"`
	Warnings       []string `json:"warnings"`
}

type reportMetricRequest struct {
	Epoch   int                `json:"epoch" binding:"required"`
	Metrics map[string]float64 `json:"metrics" binding:"required"`
}

type createChampionExportRequest struct {
	Format      string         `json:"format"`
	ArtifactURI string         `json:"artifact_uri"`
	Metadata    map[string]any `json:"metadata"`
}

type createChampionDemoPredictionRequest struct {
	ImageURI      string         `json:"image_uri" binding:"required"`
	ImageID       string         `json:"image_id"`
	TrueLabel     string         `json:"true_label"`
	ImageMetadata map[string]any `json:"image_metadata"`
	TopK          int            `json:"top_k"`
}

type championExportResultRequest struct {
	Status           string         `json:"status"`
	ArtifactURI      string         `json:"artifact_uri"`
	Metadata         map[string]any `json:"metadata"`
	ValidationErrors []string       `json:"validation_errors"`
	Error            string         `json:"error"`
}

type championDemoPredictionResultRequest struct {
	Status         string                    `json:"status"`
	PredictedLabel string                    `json:"predicted_label"`
	TrueLabel      string                    `json:"true_label"`
	Confidence     *float64                  `json:"confidence"`
	TopK           []runs.DemoPredictionTopK `json:"top_k"`
	LatencyMS      *float64                  `json:"latency_ms"`
	Correct        *bool                     `json:"correct"`
	Error          string                    `json:"error"`
	ImageMetadata  map[string]any            `json:"image_metadata"`
}

type mergeDatasetVisualExemplarsRequest struct {
	VisualExemplars []datasets.VisualExemplar `json:"visual_exemplars"`
	DemoImages      []datasets.VisualExemplar `json:"demo_images"`
	Exemplars       []datasets.VisualExemplar `json:"exemplars"`
}

type runDatasetVisualAnalysisRequest struct {
	TriggerReason  string         `json:"trigger_reason"`
	TriggerDetails map[string]any `json:"trigger_details"`
	Provider       string         `json:"provider"`
	MaxImages      int            `json:"max_images"`
	HighDetailCap  int            `json:"high_detail_cap"`
}

type datasetVisualAnalysisResultRequest struct {
	datasets.DatasetVisualAnalysis
	SampleManifest []datasets.VisualSampleManifestItem `json:"sample_manifest"`
	RawOutput      string                              `json:"raw_output"`
	InputContext   map[string]any                      `json:"input_context"`
	InputMessages  []map[string]string                 `json:"input_messages"`
}

type datasetVisualAnalysisRerunPolicy struct {
	Enabled                  bool       `json:"enabled"`
	AutomationEnabled        bool       `json:"automation_enabled"`
	ManualRunAllowed         bool       `json:"manual_run_allowed"`
	InitialRunAllowed        bool       `json:"initial_run_allowed"`
	DeficiencyRunAllowed     bool       `json:"deficiency_run_allowed"`
	RunAllowed               bool       `json:"run_allowed"`
	DisabledReason           string     `json:"disabled_reason,omitempty"`
	Reason                   string     `json:"reason,omitempty"`
	NextAllowedAt            *time.Time `json:"next_allowed_at,omitempty"`
	CooldownSeconds          int        `json:"cooldown_seconds"`
	MaxRunsPerProfile        int        `json:"max_runs_per_profile"`
	RunsForProfile           int        `json:"runs_for_profile"`
	AcceptedRunsForProfile   int        `json:"accepted_runs_for_profile"`
	ActiveJobID              string     `json:"active_job_id,omitempty"`
	ActiveJobStatus          string     `json:"active_job_status,omitempty"`
	ProfileFingerprint       string     `json:"profile_fingerprint,omitempty"`
	DeficiencyTriggers       []string   `json:"deficiency_triggers,omitempty"`
	DeficiencySeverity       float64    `json:"deficiency_severity,omitempty"`
	LatestAnalysisID         string     `json:"latest_analysis_id,omitempty"`
	LatestAnalysisCreatedAt  *time.Time `json:"latest_analysis_created_at,omitempty"`
	LatestAnalysisValidation string     `json:"latest_analysis_validation_status,omitempty"`
}

type visualAnalysisDeficiencyAssessment struct {
	Eligible bool
	Severity float64
	Triggers []string
	Details  map[string]any
}

type reportTrainingRunSummaryRequest = runs.TrainingRunSummaryUpdate
type reportTrainingRunEvaluationRequest = runs.TrainingRunEvaluationUpdate

type completeJobRequest struct {
	MLflowRunID string `json:"mlflow_run_id"`
}

type failJobRequest struct {
	Error     string `json:"error" binding:"required"`
	Retryable bool   `json:"retryable"`
}

type pollJobRequest struct {
	Provider                            string   `json:"provider"`
	Templates                           []string `json:"templates"`
	IncludeUnspecifiedProviderTemplates []string `json:"include_unspecified_provider_templates"`
}

type pollJobResponse struct {
	Job *jobs.ExperimentJob `json:"job"`
}

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, healthResponse{
		Status:    "ok",
		Service:   "orchestrator",
		Timestamp: time.Now().UTC(),
	})
}

func (s *Server) getAutoMLCapabilities(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"enabled":                   s.currentAutomationSettings().AutoMLEnabled,
		"default_sampler":           s.currentAutomationSettings().AutoMLSampler,
		"capabilities":              automl.DefaultCapabilityRegistry().Capabilities(),
		"strategy_fields_forbidden": []string{"model", "template", "preprocessing", "resolution_strategy", "image_size", "augmentation_policy", "augmentation_policy_config.policy_type", "class_balancing", "sampling_strategy", "pretrained", "freeze_backbone", "fine_tune_strategy"},
		"scheduling_authority":      false,
	})
}

func (s *Server) getAutomationSettings(c *gin.Context) {
	c.JSON(http.StatusOK, s.currentAutomationSettings())
}

func (s *Server) updateAutomationSettings(c *gin.Context) {
	var req settings.AutomationSettingsUpdate
	if !bindJSON(c, &req) {
		return
	}

	current := s.currentAutomationSettings()
	updated, err := applyAutomationSettingsUpdate(current, req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	saved, err := s.store.SaveAutomationSettings(updated)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	s.setAutomationSettings(saved)
	c.JSON(http.StatusOK, saved)
}

func (s *Server) createProject(c *gin.Context) {
	var req createProjectRequest
	if !bindJSON(c, &req) {
		return
	}

	project, err := s.store.CreateProject(req.Name, req.Goal)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, project)
}

func (s *Server) listProjects(c *gin.Context) {
	projects, err := s.store.ListProjects()
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"projects": projects})
}

func (s *Server) getProject(c *gin.Context) {
	project, err := s.store.GetProject(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, project)
}

func (s *Server) createDataset(c *gin.Context) {
	var req createDatasetRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.CreateDataset(
		c.Param("id"),
		req.Name,
		req.StorageURI,
		req.ChecksumSHA256,
		req.SizeBytes,
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if _, err := s.store.CreateJob(dataset.ProjectID, jobs.TemplateProfileDataset, map[string]any{
		"dataset_id": dataset.ID,
	}); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, dataset)
}

func (s *Server) listProjectDatasets(c *gin.Context) {
	datasets, err := s.store.ListProjectDatasets(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"datasets": datasets})
}

func (s *Server) getDataset(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}

func (s *Server) updateDatasetProfile(c *gin.Context) {
	var req updateDatasetProfileRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.UpdateDatasetProfile(c.Param("id"), req.Profile)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	s.indexMemoryCard(context.Background(), memory.NewDatasetProfileMemoryCard(dataset))

	visualQueued, err := s.maybeQueueInitialDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if visualQueued {
		c.JSON(http.StatusOK, gin.H{
			"dataset":                         dataset,
			"initial_experiment_plan_status":  "waiting_for_visual_analysis",
			"visual_analysis_required_before": "initial_experiment_plan",
		})
		return
	}

	if err := s.createInitialPlanForDataset(dataset.ID); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}

func (s *Server) importDatasetMetadata(c *gin.Context) {
	var req importDatasetMetadataRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	parseRequest, warnings, err := datasetMetadataParseRequest(req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if parseRequest.SourceKind == "" && len(req.WorkerDiscovery) > 0 {
		parseRequest.SourceKind = datasets.MetadataSourceKindWorkerDiscovery
	}
	importRecord, err := datasetmetadata.NewService().Parse(dataset, parseRequest)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord.Warnings = append(importRecord.Warnings, normalizeMetadataIssues(req.Warnings)...)
	importRecord.Warnings = append(importRecord.Warnings, warnings...)
	importRecord.Summary = datasetmetadata.BuildSummary(importRecord)
	importRecord.AgentSafeSummary = datasetmetadata.BuildAgentSafeSummary(importRecord.Summary)

	stored, err := s.store.CreateDatasetMetadataImport(importRecord)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"metadata_import":    stored,
		"summary":            stored.Summary,
		"agent_safe_summary": stored.AgentSafeSummary,
	})
}

func (s *Server) listDatasetMetadataImports(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	imports, err := s.store.ListDatasetMetadataImports(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"dataset_id":        dataset.ID,
		"metadata_imports":  imports,
		"source_of_truth":   "dataset_metadata_imports",
		"safe_summary_only": false,
	})
}

func (s *Server) getDatasetMetadataImport(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord, err := s.store.GetDatasetMetadataImport(c.Param("import_id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if importRecord.DatasetID != dataset.ID {
		writeStoreError(c, store.ErrNotFound)
		return
	}
	c.JSON(http.StatusOK, importRecord)
}

func (s *Server) getDatasetMetadataSummary(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord, err := s.store.GetActiveDatasetMetadataImport(dataset.ID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			c.JSON(http.StatusOK, gin.H{
				"dataset_id":         dataset.ID,
				"metadata_import_id": "",
				"summary":            nil,
				"agent_safe_summary": nil,
				"source_of_truth":    "dataset_metadata_imports",
				"safe_summary_only":  true,
				"message":            "No active dataset metadata import has been recorded.",
			})
			return
		}
		writeStoreError(c, err)
		return
	}
	if strings.EqualFold(c.Query("safe"), "false") || strings.EqualFold(c.Query("agent_safe"), "false") {
		c.JSON(http.StatusOK, gin.H{
			"dataset_id":           dataset.ID,
			"metadata_import_id":   importRecord.ID,
			"summary":              importRecord.Summary,
			"agent_safe_summary":   importRecord.AgentSafeSummary,
			"source_of_truth":      "dataset_metadata_imports",
			"raw_sources_included": false,
		})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"dataset_id":         dataset.ID,
		"metadata_import_id": importRecord.ID,
		"summary":            importRecord.AgentSafeSummary,
		"agent_safe_summary": importRecord.AgentSafeSummary,
		"source_of_truth":    "dataset_metadata_imports",
		"safe_summary_only":  true,
	})
}

func (s *Server) getDatasetMetadataBundle(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	limit := queryInt(c, "limit", 1000, 1, 1000)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	importID := strings.TrimSpace(c.Query("metadata_import_id"))
	if importID == "" {
		importID = strings.TrimSpace(c.Query("import_id"))
	}
	includeAnnotations := false
	for _, include := range strings.Split(c.Query("include"), ",") {
		if strings.EqualFold(strings.TrimSpace(include), "bbox") || strings.EqualFold(strings.TrimSpace(include), "annotations") {
			includeAnnotations = true
		}
	}
	bundle, err := s.store.GetDatasetMetadataBundle(dataset.ID, importID, includeAnnotations, limit, offset)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	purpose := strings.TrimSpace(c.Query("purpose"))
	if purpose != "" {
		bundle.Purpose = purpose
	}
	c.JSON(http.StatusOK, gin.H{"bundle": bundle})
}

func datasetMetadataParseRequest(req importDatasetMetadataRequest) (datasetmetadata.ImportRequest, []datasets.MetadataIssue, error) {
	out := datasetmetadata.ImportRequest{
		StrictMode: req.StrictMode,
		SourceKind: strings.TrimSpace(req.SourceKind),
		Inventory:  req.Inventory,
		Sources:    make([]datasetmetadata.SourceInput, 0, len(req.Sources)),
	}
	warnings := []datasets.MetadataIssue{}
	var totalBytes int64
	for _, source := range req.Sources {
		input := datasetmetadata.SourceInput{
			RelativePath:   source.RelativePath,
			StorageURI:     source.StorageURI,
			ChecksumSHA256: source.ChecksumSHA256,
			SizeBytes:      source.SizeBytes,
			DeclaredFormat: source.DeclaredFormat,
		}
		for _, warning := range source.Warnings {
			warnings = append(warnings, datasets.MetadataIssue{
				Severity: "warning",
				Code:     strings.TrimSpace(warning),
				Message:  metadataWarningMessage(warning),
			})
		}
		if strings.TrimSpace(source.ContentBase64) != "" {
			decoded, err := base64.StdEncoding.DecodeString(source.ContentBase64)
			if err != nil {
				return datasetmetadata.ImportRequest{}, nil, fmt.Errorf("%w: metadata source content_base64 is invalid", store.ErrInvalidRequest)
			}
			if len(decoded) > datasetMetadataMaxSourceBytes {
				warnings = append(warnings, datasets.MetadataIssue{Severity: "warning", Code: "backend_source_size_cap", Message: "metadata source content exceeded backend source byte cap and was skipped"})
			} else if totalBytes+int64(len(decoded)) > datasetMetadataMaxTotalSourceBytes {
				warnings = append(warnings, datasets.MetadataIssue{Severity: "warning", Code: "backend_total_source_size_cap", Message: "metadata source content exceeded backend total byte cap and was skipped"})
			} else {
				input.Content = decoded
				totalBytes += int64(len(decoded))
			}
		}
		out.Sources = append(out.Sources, input)
	}
	return out, warnings, nil
}

func normalizeMetadataIssues(issues []datasets.MetadataIssue) []datasets.MetadataIssue {
	out := make([]datasets.MetadataIssue, 0, len(issues))
	for _, issue := range issues {
		issue.Code = strings.TrimSpace(issue.Code)
		issue.Message = strings.TrimSpace(issue.Message)
		if issue.Code == "" && issue.Message == "" {
			continue
		}
		if issue.Severity == "" {
			issue.Severity = "warning"
		}
		out = append(out, issue)
	}
	return out
}

func metadataWarningMessage(code string) string {
	code = strings.TrimSpace(code)
	switch code {
	case "unsupported_metadata_format":
		return "metadata candidate was marked unsupported by worker discovery"
	case "content_skipped_source_size_cap":
		return "metadata candidate content was skipped by the worker source byte cap"
	case "content_skipped_total_size_cap":
		return "metadata candidate content was skipped by the worker total byte cap"
	default:
		if code == "" {
			return "metadata candidate warning"
		}
		return strings.ReplaceAll(code, "_", " ")
	}
}

func (s *Server) listDatasetVisualExemplars(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	caps := visualExemplarCapsFromQuery(c, 24, 4, 1_500_000)
	exemplars := cappedVisualExemplars(dataset.Profile, caps, "visual_exemplars", "exemplars")

	c.JSON(http.StatusOK, gin.H{
		"dataset_id":       dataset.ID,
		"source_of_truth":  "datasets.profile",
		"caps":             caps,
		"visual_exemplars": exemplars,
	})
}

func (s *Server) mergeDatasetVisualExemplars(c *gin.Context) {
	var req mergeDatasetVisualExemplarsRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if dataset.Status != datasets.StatusProfiled {
		writeStoreError(c, fmt.Errorf("%w: dataset must be profiled before visual exemplars can be merged", store.ErrInvalidRequest))
		return
	}

	visualExemplars := req.VisualExemplars
	if len(visualExemplars) == 0 && len(req.Exemplars) > 0 {
		visualExemplars = req.Exemplars
	}
	if err := validateVisualExemplarPack(visualExemplars, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}); err != nil {
		writeStoreError(c, err)
		return
	}
	if err := validateVisualExemplarPack(req.DemoImages, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}); err != nil {
		writeStoreError(c, err)
		return
	}

	profile := copyMap(dataset.Profile)
	if len(visualExemplars) > 0 {
		profile["visual_exemplars"] = visualExemplarsToProfileValues(mergeVisualExemplars(cappedVisualExemplars(profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "visual_exemplars"), visualExemplars, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}))
	}
	if len(req.DemoImages) > 0 {
		profile["demo_images"] = visualExemplarsToProfileValues(mergeVisualExemplars(cappedVisualExemplars(profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "demo_images"), req.DemoImages, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}))
	}
	profile["visual_exemplar_audit"] = map[string]any{
		"updated_at":             time.Now().UTC().Format(time.RFC3339),
		"visual_exemplar_count":  len(cappedVisualExemplars(profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "visual_exemplars")),
		"demo_image_count":       len(cappedVisualExemplars(profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "demo_images")),
		"source_of_truth":        "datasets.profile",
		"max_total_images":       48,
		"max_images_per_class":   6,
		"max_total_size_bytes":   3_000_000,
		"backend_validated_pack": true,
	}

	updated, err := s.store.UpdateDatasetProfile(dataset.ID, profile)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	s.indexMemoryCard(context.Background(), memory.NewDatasetProfileMemoryCard(updated))

	c.JSON(http.StatusOK, gin.H{
		"dataset":          updated,
		"visual_exemplars": cappedVisualExemplars(updated.Profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "visual_exemplars"),
		"demo_images":      cappedVisualExemplars(updated.Profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "demo_images"),
	})
}

func (s *Server) listDatasetVisualAnalyses(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	analyses, err := s.store.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	assessment, assessmentErr := s.assessDatasetVisualDeficiency(dataset, runs.TrainingRunEvaluation{})
	if assessmentErr != nil {
		log.Printf("visual analysis deficiency assessment failed for dataset %s: %v", dataset.ID, assessmentErr)
	}
	policy, err := s.datasetVisualAnalysisRunPolicy(dataset, datasets.VisualTriggerManual, assessment)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	latest := latestVisualAnalysisFromList(analyses)
	c.JSON(http.StatusOK, gin.H{
		"dataset_id":             dataset.ID,
		"visual_analyses":        analyses,
		"latest":                 latest,
		"rerun_policy":           policy,
		"manual_run_supported":   true,
		"source_of_truth":        "dataset_visual_analyses",
		"evidence_only":          true,
		"raw_images_shown":       false,
		"raw_images_for_planner": false,
	})
}

func (s *Server) getLatestDatasetVisualAnalysis(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	analysis, err := s.store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, analysis)
}

func (s *Server) runDatasetVisualAnalysis(c *gin.Context) {
	var req runDatasetVisualAnalysisRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if dataset.Status != datasets.StatusProfiled {
		writeStoreError(c, fmt.Errorf("%w: dataset must be profiled before visual analysis can run", store.ErrInvalidRequest))
		return
	}

	trigger := datasets.VisualReanalysisTrigger(strings.ToLower(strings.TrimSpace(req.TriggerReason)))
	if trigger == "" {
		trigger = datasets.VisualTriggerManual
	}
	if !allowedVisualTrigger(trigger) {
		writeStoreError(c, fmt.Errorf("%w: unsupported visual analysis trigger_reason %q", store.ErrInvalidRequest, req.TriggerReason))
		return
	}
	assessment := visualAnalysisDeficiencyAssessment{}
	if trigger == datasets.VisualTriggerDeficiencyReanalysis {
		assessment, err = s.assessDatasetVisualDeficiency(dataset, runs.TrainingRunEvaluation{})
		if err != nil {
			writeStoreError(c, err)
			return
		}
	}
	policy, err := s.datasetVisualAnalysisRunPolicy(dataset, trigger, assessment)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if !policy.RunAllowed {
		c.JSON(http.StatusConflict, gin.H{
			"error":        policy.DisabledReason,
			"rerun_policy": policy,
		})
		return
	}
	job, event, err := s.queueDatasetVisualAnalysis(dataset, trigger, req.TriggerDetails, req.Provider, req.MaxImages, req.HighDetailCap)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusCreated, gin.H{
		"job":          job,
		"event":        event,
		"rerun_policy": policy,
	})
}

func (s *Server) reportDatasetVisualAnalysisResult(c *gin.Context) {
	var req datasetVisualAnalysisResultRequest
	if !bindJSON(c, &req) {
		return
	}
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	analysis := req.DatasetVisualAnalysis
	analysis.DatasetID = strings.TrimSpace(analysis.DatasetID)
	if analysis.DatasetID == "" {
		analysis.DatasetID = dataset.ID
	}
	if analysis.ProjectID == "" {
		analysis.ProjectID = dataset.ProjectID
	}
	if analysis.DatasetName == "" {
		analysis.DatasetName = dataset.Name
	}
	if analysis.SourceJobID == "" {
		analysis.SourceJobID = c.Query("job_id")
	}
	if analysis.AgentName == "" {
		analysis.AgentName = datasets.VisualAnalysisAgentName
	}
	if analysis.AgentVersion == "" {
		analysis.AgentVersion = "v1"
	}
	if analysis.PromptVersion == "" {
		analysis.PromptVersion = datasets.VisualAnalysisSchemaVersion
	}
	if analysis.ProfileFingerprint == "" {
		analysis.ProfileFingerprint = datasetProfileFingerprint(dataset.Profile)
	}
	if analysis.ProfileSchemaVersion == "" {
		analysis.ProfileSchemaVersion = datasetProfileSchemaVersion(dataset.Profile)
	}

	validationDataset := dataset
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if len(metadataSummary) > 0 {
		validationDataset.Profile = profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary)
	}
	inputContext := visualAnalysisInvocationContext(req.InputContext, req.SampleManifest)
	validationErrors := validateDatasetVisualAnalysisResult(validationDataset, &analysis, req.SampleManifest, req.RawOutput, inputContext)
	validationStatus := memory.InvocationValidationValid
	if len(validationErrors) > 0 {
		validationStatus = memory.InvocationValidationInvalid
		analysis.ValidationErrors = validationErrors
	}
	parsedOutput, _ := mapFromStruct(analysis)
	invocation, invocationErr := s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         dataset.ProjectID,
		DatasetID:         dataset.ID,
		JobID:             analysis.SourceJobID,
		AgentName:         datasets.VisualAnalysisAgentName,
		AgentVersion:      analysis.AgentVersion,
		PromptVersion:     analysis.PromptVersion,
		Provider:          analysis.Provider,
		Model:             analysis.Model,
		InputMessages:     req.InputMessages,
		InputContext:      inputContext,
		RawOutput:         req.RawOutput,
		ParsedOutput:      parsedOutput,
		ValidationStatus:  validationStatus,
		ValidationError:   strings.Join(validationErrors, "; "),
		AcceptedForMemory: len(validationErrors) == 0,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
	if invocationErr != nil {
		log.Printf("visual dataset analysis invocation write failed for dataset %s: %v", dataset.ID, invocationErr)
	} else {
		analysis.SourceInvocationID = invocation.ID
	}

	var stored datasets.DatasetVisualAnalysis
	if len(validationErrors) > 0 {
		stored, err = s.store.RejectDatasetVisualAnalysis(analysis)
	} else {
		stored, err = s.store.CreateDatasetVisualAnalysis(analysis)
	}
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if stored.ValidationStatus == datasets.VisualValidationStatusAccepted {
		s.indexMemoryCard(context.Background(), memory.NewDatasetVisualAnalysisMemoryCard(stored))
		for _, card := range memory.BuildDatasetPreprocessingHypothesisCards(stored) {
			s.indexMemoryCard(context.Background(), card)
		}
	}

	if _, eventErr := s.store.CreateExecutionEvent(dataset.ProjectID, "", execution.EventDatasetVisualAnalysisResult, "Dataset visual analysis result recorded.", map[string]any{
		"dataset_id":        dataset.ID,
		"analysis_id":       stored.ID,
		"analysis_version":  stored.AnalysisVersion,
		"validation_status": stored.ValidationStatus,
		"validation_errors": stored.ValidationErrors,
		"source_job_id":     stored.SourceJobID,
	}); eventErr != nil {
		log.Printf("record visual analysis event failed: %v", eventErr)
	}

	if stored.TriggerReason == datasets.VisualTriggerInitialProfile {
		if err := s.createInitialPlanForDataset(dataset.ID); err != nil {
			writeStoreError(c, err)
			return
		}
	}

	if len(validationErrors) > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"analysis":          stored,
			"validation_errors": validationErrors,
		})
		return
	}
	c.JSON(http.StatusCreated, stored)
}

func (s *Server) createJob(c *gin.Context) {
	var req createJobRequest
	if !bindJSON(c, &req) {
		return
	}

	if req.Config == nil {
		req.Config = map[string]any{}
	}

	job, err := s.store.CreateJob(c.Param("id"), req.Template, req.Config)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, job)
}

func (s *Server) listProjectJobs(c *gin.Context) {
	jobs, err := s.store.ListProjectJobs(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"jobs": jobs})
}

func (s *Server) getJob(c *gin.Context) {
	job, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) reportMetric(c *gin.Context) {
	var req reportMetricRequest
	if !bindJSON(c, &req) {
		return
	}
	if req.Epoch < 1 {
		writeStoreError(c, fmt.Errorf("%w: epoch must be positive", store.ErrInvalidRequest))
		return
	}

	metric, err := s.store.ReportMetric(c.Param("id"), req.Epoch, req.Metrics)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, metric)
}

func (s *Server) listJobMetrics(c *gin.Context) {
	metrics, err := s.store.ListJobMetrics(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"metrics": metrics})
}

func (s *Server) upsertTrainingRunSummary(c *gin.Context) {
	var req reportTrainingRunSummaryRequest
	if !bindJSON(c, &req) {
		return
	}

	summary, err := s.store.UpsertTrainingRunSummary(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if summary.Status == jobs.StatusSucceeded || summary.Status == jobs.StatusFailed {
		if job, err := s.store.GetJob(c.Param("id")); err == nil {
			if err := s.observeAutoMLTrialForJob(job); err != nil {
				log.Printf("AutoML trial observation failed for job %s: %v", job.ID, err)
			}
		}
	}

	c.JSON(http.StatusOK, summary)
}

func (s *Server) getTrainingRunSummary(c *gin.Context) {
	summary, err := s.store.GetTrainingRunSummary(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, summary)
}

func (s *Server) upsertTrainingRunEvaluation(c *gin.Context) {
	var req reportTrainingRunEvaluationRequest
	if !bindJSON(c, &req) {
		return
	}

	req = s.enrichTrainingRunEvaluationUpdate(c.Param("id"), req)
	evaluation, err := s.store.UpsertTrainingRunEvaluation(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if err := s.maybeQueueDeficiencyDatasetVisualAnalysis(evaluation); err != nil {
		log.Printf("visual dataset deficiency reanalysis check failed for job %s: %v", evaluation.JobID, err)
	}

	c.JSON(http.StatusOK, evaluation)
}

func (s *Server) getTrainingRunEvaluation(c *gin.Context) {
	evaluation, err := s.store.GetTrainingRunEvaluation(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, evaluation)
}

func (s *Server) listProjectTrainingRunSummaries(c *gin.Context) {
	summaries, err := s.store.ListProjectTrainingRunSummaries(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"summaries": summaries})
}

func (s *Server) listProjectTrainingRunEvaluations(c *gin.Context) {
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"evaluations": evaluations})
}

func (s *Server) enrichTrainingRunEvaluationUpdate(jobID string, update runs.TrainingRunEvaluationUpdate) runs.TrainingRunEvaluationUpdate {
	summary, err := s.store.GetTrainingRunSummary(jobID)
	if err != nil {
		return update
	}
	metrics, err := s.store.ListJobMetrics(jobID)
	if err != nil {
		metrics = nil
	}
	diagnostics := trainingRunDiagnostics(summary, metrics)
	holisticScores := copyPayloadMap(update.HolisticScores)
	if len(diagnostics) > 0 {
		holisticScores["training_diagnostics"] = diagnostics
		holisticScores["train_validation_gap"] = diagnostics["train_validation_gap"]
		holisticScores["divergence_status"] = diagnostics["status"]
		holisticScores["divergence_detected"] = diagnostics["divergence_detected"]
	}
	if job, err := s.store.GetJob(jobID); err == nil {
		if feedback := s.optimizerFeedbackSummary(jobConfigString(job.Config, "automl_study_id"), trainingMonitorTargetMetricFromJob(job, summary)); feedback != nil {
			holisticScores["optimizer_feedback_summary"] = feedback
		}
	}
	update.HolisticScores = holisticScores
	return update
}

func trainingMonitorTargetMetricFromJob(job jobs.ExperimentJob, summary runs.TrainingRunSummary) string {
	if targetMetric := jobConfigString(job.Config, "target_metric"); targetMetric != "" {
		return targetMetric
	}
	return "macro_f1"
}

func trainingRunDiagnostics(summary runs.TrainingRunSummary, metrics []jobs.EpochMetric) map[string]any {
	trainLoss, valLoss, hasLosses := finalTrainValidationLosses(summary, metrics)
	if !hasLosses {
		return nil
	}

	firstTrainLoss, lastTrainLoss, hasTrainTrend := metricFirstLast(metrics, "train_loss", "training_loss", "loss")
	firstValLoss, lastValLoss, hasValTrend := metricFirstLast(metrics, "val_loss", "validation_loss")
	if !hasTrainTrend {
		firstTrainLoss = trainLoss
		lastTrainLoss = trainLoss
	}
	if !hasValTrend {
		firstValLoss = valLoss
		lastValLoss = valLoss
	}

	gap := valLoss - trainLoss
	ratio := 0.0
	if trainLoss > 0 {
		ratio = valLoss / trainLoss
	}
	trainDelta := lastTrainLoss - firstTrainLoss
	valDelta := lastValLoss - firstValLoss
	diverging := hasTrainTrend && hasValTrend && trainDelta < -0.01 && valDelta > 0.01

	status := "stable"
	interpretation := "Training and validation losses are moving together closely enough for the current run."
	if diverging {
		status = "diverging"
		interpretation = "Training loss is improving while validation loss is worsening; treat this as an overfitting or data-shift signal."
	} else if gap > 0.20 && ratio > 1.25 {
		status = "overfitting_risk"
		interpretation = "Validation loss is materially higher than training loss, so the run may not generalize well."
	} else if gap < -0.10 && ratio > 0 && ratio < 0.90 {
		status = "validation_easier_than_train"
		interpretation = "Validation loss is lower than training loss; check split difficulty before comparing this run to others."
	}

	severity := 0.0
	if diverging {
		severity = 0.75
	}
	if gap > 0 {
		severity = maxFloat(severity, minFloat(1, gap/0.75))
	}
	if ratio > 1 {
		severity = maxFloat(severity, minFloat(1, (ratio-1)/1.5))
	}

	return map[string]any{
		"computed_by":           "backend_training_diagnostics_v1",
		"status":                status,
		"interpretation":        interpretation,
		"divergence_detected":   diverging || status == "overfitting_risk",
		"train_loss":            roundDiagnosticFloat(trainLoss),
		"val_loss":              roundDiagnosticFloat(valLoss),
		"train_validation_gap":  roundDiagnosticFloat(gap),
		"val_train_loss_ratio":  roundDiagnosticFloat(ratio),
		"train_loss_delta":      roundDiagnosticFloat(trainDelta),
		"val_loss_delta":        roundDiagnosticFloat(valDelta),
		"severity":              roundDiagnosticFloat(severity),
		"epochs_observed":       maxInt(summary.EpochsCompleted, len(metrics)),
		"trend_epochs_observed": len(metrics),
	}
}

func finalTrainValidationLosses(summary runs.TrainingRunSummary, metrics []jobs.EpochMetric) (float64, float64, bool) {
	trainLoss := 0.0
	valLoss := 0.0
	hasTrain := false
	hasVal := false
	for _, metric := range metrics {
		if value, ok := metricFloat(metric.Metrics, "train_loss", "training_loss", "loss"); ok {
			trainLoss = value
			hasTrain = true
		}
		if value, ok := metricFloat(metric.Metrics, "val_loss", "validation_loss"); ok {
			valLoss = value
			hasVal = true
		}
	}
	if hasTrain && hasVal {
		return trainLoss, valLoss, true
	}
	if summary.FinalTrainLoss > 0 && summary.FinalValLoss > 0 {
		return summary.FinalTrainLoss, summary.FinalValLoss, true
	}
	return 0, 0, false
}

func metricFirstLast(metrics []jobs.EpochMetric, keys ...string) (float64, float64, bool) {
	first := 0.0
	last := 0.0
	found := false
	for _, metric := range metrics {
		value, ok := metricFloat(metric.Metrics, keys...)
		if !ok {
			continue
		}
		if !found {
			first = value
		}
		last = value
		found = true
	}
	return first, last, found
}

func metricFloat(metrics map[string]float64, keys ...string) (float64, bool) {
	for _, key := range keys {
		value, ok := metrics[key]
		if ok && isFiniteFloat(value) {
			return value, true
		}
	}
	return 0, false
}

func copyPayloadMap(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func roundDiagnosticFloat(value float64) float64 {
	if !isFiniteFloat(value) {
		return 0
	}
	return math.Round(value*1_000_000) / 1_000_000
}

func isFiniteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func (s *Server) getProjectChampion(c *gin.Context) {
	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			c.JSON(http.StatusOK, gin.H{"champion": nil})
			return
		}
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"champion": champion})
}

func (s *Server) listProjectChampionExports(c *gin.Context) {
	exports, err := s.store.ListProjectChampionExports(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"exports": exports})
}

func (s *Server) createProjectChampionExport(c *gin.Context) {
	projectID := c.Param("id")
	var req createChampionExportRequest
	if !bindJSON(c, &req) {
		return
	}

	champion, err := s.store.GetProjectChampion(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	job, err := s.store.GetJob(champion.JobID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if job.Status != jobs.StatusSucceeded {
		writeStoreError(c, fmt.Errorf("%w: champion job must be succeeded before export", store.ErrInvalidRequest))
		return
	}
	if _, err := s.store.GetTrainingRunSummary(champion.JobID); err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			writeStoreError(c, err)
			return
		}
		writeStoreError(c, fmt.Errorf("%w: champion job must have a training run summary before export", store.ErrInvalidRequest))
		return
	}

	format := normalizeChampionExportFormat(req.Format)
	if format == "" {
		writeStoreError(c, fmt.Errorf("%w: champion export format must be one of onnx, torchscript, pytorch, safetensors", store.ErrInvalidRequest))
		return
	}

	requestArtifactURI := strings.TrimSpace(req.ArtifactURI)
	artifactURI := requestArtifactURI
	if artifactURI == "" {
		artifactURI = championArtifactURI(champion.DeploymentProfile)
	}
	status := runs.ChampionExportStatusPending
	validationErrors := []string{}
	if artifactURI == "" {
		status = runs.ChampionExportStatusPendingArtifact
		validationErrors = append(validationErrors, "selected champion has no exportable artifact URI yet")
	} else if artifactMatchesChampionExportFormat(artifactURI, format) {
		status = runs.ChampionExportStatusReady
	} else if requestArtifactURI != "" {
		status = runs.ChampionExportStatusReady
	}

	metadata := championExportMetadata(champion, format, req.Metadata)
	export, err := s.store.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:        projectID,
		ChampionID:       champion.ID,
		JobID:            champion.JobID,
		Status:           status,
		Format:           format,
		ArtifactURI:      artifactURI,
		Metadata:         metadata,
		ValidationErrors: validationErrors,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	datasetID := championDatasetID(champion, job)
	if datasetID != "" && export.Status != runs.ChampionExportStatusReady {
		if _, err := s.ensureOpenJob(champion.ProjectID, jobs.TemplateExportChampion, map[string]any{
			"dataset_id":      datasetID,
			"champion_id":     champion.ID,
			"champion_job_id": champion.JobID,
			"export_id":       export.ID,
			"format":          export.Format,
			"artifact_uri":    artifactURI,
			"metadata":        metadata,
		}, func(existing jobs.ExperimentJob) bool {
			return jobConfigString(existing.Config, "export_id") == export.ID
		}); err != nil {
			writeStoreError(c, err)
			return
		}
	}
	if _, err := s.store.CreateExecutionEvent(projectID, champion.PlanID, execution.EventChampionExportRequested, fmt.Sprintf("Champion export requested for job %s.", champion.JobID), map[string]any{
		"champion_id": champion.ID,
		"export_id":   export.ID,
		"job_id":      champion.JobID,
		"status":      export.Status,
		"format":      export.Format,
	}); err != nil {
		log.Printf("record champion export event failed: %v", err)
	}

	c.JSON(http.StatusCreated, export)
}

func (s *Server) listProjectChampionDemoImages(c *gin.Context) {
	projectID := c.Param("id")
	champion, err := s.store.GetProjectChampion(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	dataset, err := s.store.GetDataset(champion.DatasetID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	caps := visualExemplarCapsFromQuery(c, 24, 4, 1_500_000)
	exemplars := cappedVisualExemplars(championHeldoutDemoImageProfile(champion), caps, "heldout_demo_images", "demo_images", "test_images")
	if len(exemplars) == 0 {
		exemplars = testOnlyVisualExemplars(cappedVisualExemplars(dataset.Profile, caps, "demo_images", "visual_exemplars", "test_images"))
	}

	c.JSON(http.StatusOK, gin.H{
		"project_id":      projectID,
		"dataset_id":      dataset.ID,
		"champion_job_id": champion.JobID,
		"source_of_truth": "datasets.profile",
		"caps":            caps,
		"images":          exemplars,
	})
}

func (s *Server) listProjectChampionDemoPredictions(c *gin.Context) {
	predictions, err := s.store.ListProjectChampionDemoPredictions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"predictions": predictions})
}

func (s *Server) createProjectChampionDemoPrediction(c *gin.Context) {
	var req createChampionDemoPredictionRequest
	if !bindJSON(c, &req) {
		return
	}
	if req.TopK < 1 {
		req.TopK = 5
	}
	if req.TopK > 10 {
		writeStoreError(c, fmt.Errorf("%w: top_k must be at most 10", store.ErrInvalidRequest))
		return
	}
	imageURI := strings.TrimSpace(req.ImageURI)
	if imageURI == "" {
		writeStoreError(c, fmt.Errorf("%w: image_uri is required", store.ErrInvalidRequest))
		return
	}

	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	dataset, err := s.store.GetDataset(champion.DatasetID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		writeStoreError(c, err)
		return
	}
	datasetID := champion.DatasetID
	if err == nil {
		datasetID = dataset.ID
	}
	imageID := strings.TrimSpace(req.ImageID)
	trueLabel := strings.TrimSpace(req.TrueLabel)
	imageMetadata := map[string]any{}
	for key, value := range req.ImageMetadata {
		imageMetadata[key] = value
	}
	if err == nil {
		if matchedImageID, matchedTrueLabel, matchedMetadata, ok := championDemoImageMetadata(dataset.Profile, imageURI); ok {
			if imageID == "" {
				imageID = matchedImageID
			}
			if trueLabel == "" {
				trueLabel = matchedTrueLabel
			}
			for key, value := range matchedMetadata {
				if _, exists := imageMetadata[key]; !exists {
					imageMetadata[key] = value
				}
			}
		}
	}

	prediction, err := s.store.CreateChampionDemoPrediction(runs.ChampionDemoPredictionCreate{
		ProjectID:     champion.ProjectID,
		ChampionID:    champion.ID,
		JobID:         champion.JobID,
		DatasetID:     champion.DatasetID,
		ImageURI:      imageURI,
		ImageID:       imageID,
		ImageMetadata: imageMetadata,
		Status:        runs.ChampionDemoPredictionStatusPending,
		TrueLabel:     trueLabel,
		TopK:          []runs.DemoPredictionTopK{},
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	readyExport, hasReadyExport := usableChampionExport(s.store, champion.ProjectID, champion.ID)
	runtimeAvailable := hasReadyExport
	if hasReadyExport {
		if _, err := s.ensureOpenJob(champion.ProjectID, jobs.TemplateChampionDemoPrediction, map[string]any{
			"dataset_id":             datasetID,
			"champion_id":            champion.ID,
			"champion_job_id":        champion.JobID,
			"prediction_id":          prediction.ID,
			"export_id":              readyExport.ID,
			"export_format":          readyExport.Format,
			"export_artifact_uri":    readyExport.ArtifactURI,
			"manifest_path":          championExportManifestPath(readyExport.Metadata),
			"export_metadata":        readyExport.Metadata,
			"image_uri":              imageURI,
			"image_id":               imageID,
			"true_label":             trueLabel,
			"top_k":                  req.TopK,
			"requested_at":           time.Now().UTC().Format(time.RFC3339),
			"prediction_contract":    "worker reports via /jobs/:id/champion-demo-prediction-result",
			"backend_runs_inference": false,
		}, func(existing jobs.ExperimentJob) bool {
			return jobConfigString(existing.Config, "prediction_id") == prediction.ID
		}); err != nil {
			writeStoreError(c, err)
			return
		}
	} else {
		prediction, err = s.store.UpdateChampionDemoPrediction(prediction.ID, runs.ChampionDemoPredictionUpdate{
			Status: runs.ChampionDemoPredictionStatusRuntimeUnavailable,
			Error:  "no READY champion export is available for worker-backed demo prediction",
		})
		if err != nil {
			writeStoreError(c, err)
			return
		}
	}
	if _, err := s.store.CreateExecutionEvent(champion.ProjectID, champion.PlanID, execution.EventChampionDemoPrediction, fmt.Sprintf("Champion demo prediction requested for job %s.", champion.JobID), map[string]any{
		"champion_id":   champion.ID,
		"prediction_id": prediction.ID,
		"job_id":        champion.JobID,
		"status":        prediction.Status,
		"image_uri":     prediction.ImageURI,
	}); err != nil {
		log.Printf("record champion demo prediction event failed: %v", err)
	}

	c.JSON(http.StatusAccepted, gin.H{
		"prediction":        prediction,
		"runtime_available": runtimeAvailable,
		"contract": gin.H{
			"champion_job_id": champion.JobID,
			"image_uri":       imageURI,
			"top_k":           req.TopK,
			"returns":         []string{"predicted_label", "true_label", "confidence", "top_k", "latency_ms", "correct"},
		},
	})
}

func (s *Server) reviewProjectExperiments(c *gin.Context) {
	_, decision, err := s.createReviewerDecision(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, decision)
}

func (s *Server) scheduleFollowUpExperiments(c *gin.Context) {
	projectID := c.Param("id")

	sourcePlan, decision, err := s.followUpSourceDecision(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if decision.DecisionType != decisions.TypeAddExperiments {
		c.JSON(http.StatusOK, scheduleFollowUpExperimentsResponse{
			Decision: decision,
		})
		return
	}

	plan, created, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			c.JSON(http.StatusOK, scheduleFollowUpExperimentsResponse{
				Decision: decision,
			})
			return
		}
		writeStoreError(c, err)
		return
	}

	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	c.JSON(status, scheduleFollowUpExperimentsResponse{
		Decision:     decision,
		FollowUpPlan: &plan,
	})
}

func (s *Server) reopenProjectExperimentation(c *gin.Context) {
	projectID := c.Param("id")
	var req reopenExperimentationRequest
	if !bindJSON(c, &req) {
		return
	}

	project, err := s.store.GetProject(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	state, err := s.projectChampionSelectionState(project.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if !state.Terminal {
		writeStoreError(c, fmt.Errorf("%w: experimentation can only be reopened after a selected champion or SELECT_CHAMPION decision exists", store.ErrInvalidRequest))
		return
	}

	planID := strings.TrimSpace(req.SourcePlanID)
	if planID != "" {
		plan, err := s.store.GetExperimentPlan(planID)
		if err != nil {
			writeStoreError(c, err)
			return
		}
		if plan.ProjectID != project.ID {
			writeStoreError(c, fmt.Errorf("%w: source plan does not belong to project", store.ErrInvalidRequest))
			return
		}
	} else if state.PlanID != "" {
		planID = state.PlanID
	} else if projectPlans, err := s.store.ListProjectExperimentPlans(project.ID); err == nil {
		if latestPlan, ok := latestExperimentPlan(projectPlans); ok {
			planID = latestPlan.ID
		}
	} else {
		writeStoreError(c, err)
		return
	}

	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		reason = "User explicitly reopened experimentation after champion selection."
	}

	payload := map[string]any{
		"explicit_user_action":     true,
		"new_exploration_round":    true,
		"reason":                   reason,
		"reopened_at":              time.Now().UTC().Format(time.RFC3339),
		"terminal_reason":          state.Reason,
		"terminal_at":              state.TerminalAt.Format(time.RFC3339),
		"previous_source_plan_id":  state.PlanID,
		"previous_champion_job_id": state.ChampionJobID,
	}
	if state.DecisionID != "" {
		payload["previous_decision_id"] = state.DecisionID
	}
	if state.ChampionID != "" {
		payload["previous_champion_id"] = state.ChampionID
	}

	decision, err := s.store.CreateAgentDecision(project.ID, planID, decisions.TypeReopenExperimentation, reason, payload)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	event, err := s.store.CreateExecutionEvent(project.ID, planID, execution.EventExperimentationReopened, "Experimentation reopened by explicit user action after champion selection.", map[string]any{
		"decision_id":              decision.ID,
		"explicit_user_action":     true,
		"new_exploration_round":    true,
		"reason":                   reason,
		"previous_source_plan_id":  state.PlanID,
		"previous_champion_job_id": state.ChampionJobID,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, reopenExperimentationResponse{
		Decision: decision,
		Event:    event,
	})
}

func (s *Server) createReviewerDecision(projectID string) (plans.ExperimentPlan, decisions.AgentDecision, error) {
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(project.ID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(project.ID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	latestPlan, ok := latestExperimentPlan(projectPlans)
	if !ok {
		recommendation := agents.NewExperimentReviewer().Review(project, plans.ExperimentPlan{}, summaries)
		decision, err := s.store.CreateAgentDecision(
			project.ID,
			recommendation.PlanID,
			recommendation.DecisionType,
			recommendation.Rationale,
			recommendation.Payload,
		)
		if err != nil {
			return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
		}
		if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
			log.Printf("persist reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
		}
		return plans.ExperimentPlan{}, decision, nil
	}

	recommendation := agents.NewExperimentReviewer().Review(project, latestPlan, summaries)
	decision, err := s.store.CreateAgentDecision(
		project.ID,
		recommendation.PlanID,
		recommendation.DecisionType,
		recommendation.Rationale,
		recommendation.Payload,
	)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}
	if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
		log.Printf("persist reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
	}

	return latestPlan, decision, nil
}

func (s *Server) followUpSourceDecision(projectID string) (plans.ExperimentPlan, decisions.AgentDecision, error) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	if len(agentDecisions) > 0 && agentDecisions[0].DecisionType == decisions.TypeAddExperiments && agentDecisions[0].PlanID != "" {
		plan, err := s.store.GetExperimentPlan(agentDecisions[0].PlanID)
		if err == nil {
			return plan, agentDecisions[0], nil
		}
		if !errors.Is(err, store.ErrNotFound) {
			return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
		}
	}

	return s.createReviewerDecision(projectID)
}

func (s *Server) ensureFollowUpPlan(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision) (plans.ExperimentPlan, bool, error) {
	if decision.DecisionType != decisions.TypeAddExperiments {
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: reviewer decision is not ADD_EXPERIMENTS", store.ErrInvalidRequest)
	}
	if sourcePlan.ID == "" {
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: follow-up experiments require a source plan", store.ErrInvalidRequest)
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		if stopReason, stopDetails, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
			return plans.ExperimentPlan{}, false, err
		} else if ok {
			message := "Follow-up scheduling blocked because the project already has a selected champion."
			s.recordChampionSelectedFollowUpBlocked(projectID, sourcePlan.ID, decision.ID, "", message, stopReason, stopDetails)
			return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errChampionSelectedFollowUpBlocked, stopReason)
		}
	}
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		if err := s.validateExistingFollowUpPlanStillNovel(projectID, decision.ID, existingPlan, projectPlans); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
		return existingPlan, false, nil
	}

	experiments, err := plannedExperimentsFromPayload(decision.Payload)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	experiments, err = plannedExperimentsWithStoredProposalMechanisms(decision.Payload, experiments)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	var automlWarnings []string
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProject(projectID, experiments)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if err := validateLLMPlannerStoredMechanismContract(decision, experiments); err != nil {
		message := "Follow-up scheduling blocked because the stored planner decision lacks a valid mechanism contract."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, []string{err.Error()})
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	}
	experiments, err = s.validateFollowUpExperimentMechanismsAgainstDataset(projectID, sourcePlan.DatasetID, sourcePlan.ID, decision.ID, "", experiments, payloadStringSlice(decision.Payload, "evidence_used"))
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	experiments, skippedExperiments := filterNovelPlannedExperiments(experiments, projectPlans)
	if len(experiments) == 0 {
		message := "Follow-up scheduling blocked because every proposed experiment duplicated an existing experiment or only changed minor tuning knobs."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, skippedExperiments)
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: follow-up decision has no novel experiments after filtering duplicate or minor-only repeats", errNoNovelFollowUpExperiments)
	}

	warnings := []string{
		fmt.Sprintf("Follow-up plan generated from reviewer decision %s.", decision.ID),
		fmt.Sprintf("Previous plan: %s.", sourcePlan.ID),
	}
	warnings = append(warnings, skippedExperiments...)
	warnings = append(warnings, automlWarnings...)

	plan, err := s.store.CreateExperimentPlan(
		projectID,
		sourcePlan.DatasetID,
		sourcePlan.TargetMetric,
		recommendedWorkersForExperiments(experiments),
		estimateFollowUpMinutes(experiments),
		experiments,
		warnings,
		decision.ID,
	)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if err := s.persistAutoMLForPlan(plan); err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if _, err := s.createPendingStrategyScorecard(projectID, sourcePlan, decision, plan); err != nil {
		log.Printf("create pending strategy scorecard failed for decision %s: %v", decision.ID, err)
	}

	return plan, true, nil
}

func (s *Server) validateExistingFollowUpPlanStillNovel(projectID string, decisionID string, followUpPlan plans.ExperimentPlan, projectPlans []plans.ExperimentPlan) error {
	priorPlans := make([]plans.ExperimentPlan, 0, len(projectPlans))
	for _, plan := range projectPlans {
		if plan.ID == followUpPlan.ID {
			continue
		}
		priorPlans = append(priorPlans, plan)
	}
	for index, experiment := range followUpPlan.Experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			message := fmt.Sprintf("Existing follow-up plan %s is blocked because experiment %d is no longer valid.", followUpPlan.ID, index)
			s.recordFollowUpValidationBlocked(projectID, followUpPlan.ID, decisionID, followUpPlan.ID, message, []string{err.Error()})
			return fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
		}
	}
	if _, err := s.validateFollowUpExperimentMechanismsAgainstDataset(projectID, followUpPlan.DatasetID, followUpPlan.ID, decisionID, followUpPlan.ID, followUpPlan.Experiments, nil); err != nil {
		return err
	}
	filtered, skippedExperiments := filterNovelPlannedExperiments(followUpPlan.Experiments, priorPlans)
	if len(skippedExperiments) == 0 && len(filtered) == len(followUpPlan.Experiments) {
		return nil
	}
	message := fmt.Sprintf("Existing follow-up plan %s is blocked because it no longer passes backend novelty validation.", followUpPlan.ID)
	s.recordFollowUpValidationBlocked(projectID, followUpPlan.ID, decisionID, followUpPlan.ID, message, skippedExperiments)
	return fmt.Errorf("%w: existing follow-up plan %s is no longer schedulable", errNoNovelFollowUpExperiments, followUpPlan.ID)
}

func (s *Server) validateFollowUpExperimentMechanismsAgainstDataset(
	projectID string,
	datasetID string,
	planID string,
	decisionID string,
	followUpPlanID string,
	experiments []plans.PlannedExperiment,
	planEvidence []string,
) ([]plans.PlannedExperiment, error) {
	if len(experiments) == 0 {
		return experiments, nil
	}
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return experiments, err
	}
	enrichedExperiments, err := s.experimentsWithAcceptedVisualEvidence(dataset, experiments)
	if err != nil {
		return experiments, err
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return experiments, err
	}
	if err := validateMechanismDatasetEvidence(profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary), enrichedExperiments, planEvidence); err != nil {
		message := "Follow-up scheduling blocked because one or more proposed mechanisms lack backend-verifiable diagnosis or dataset support."
		if followUpPlanID != "" {
			message = fmt.Sprintf("Existing follow-up plan %s is blocked because one or more mechanisms lack backend-verifiable diagnosis or dataset support.", followUpPlanID)
		}
		s.recordFollowUpValidationBlocked(projectID, planID, decisionID, followUpPlanID, message, []string{err.Error()})
		return experiments, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	}
	return enrichedExperiments, nil
}

func (s *Server) recordFollowUpValidationBlocked(projectID string, planID string, decisionID string, followUpPlanID string, message string, skippedExperiments []string) {
	payload := map[string]any{
		"decision_id":               decisionID,
		"backend_validation_status": "blocked",
		"backend_validation_error":  "no novel follow-up experiments after filtering duplicate or minor-only repeats",
		"skipped_experiments":       skippedExperiments,
	}
	if followUpPlanID != "" {
		payload["follow_up_plan_id"] = followUpPlanID
	}
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record follow-up validation block failed for project %s decision %s: %v", projectID, decisionID, err)
	}
}

func (s *Server) experimentsWithAcceptedVisualEvidence(dataset datasets.Dataset, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, error) {
	if len(experiments) == 0 {
		return experiments, nil
	}
	analysis, err := s.store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return experiments, nil
		}
		return experiments, err
	}
	out := append([]plans.PlannedExperiment(nil), experiments...)
	for index := range out {
		visualEvidence := acceptedVisualEvidenceForExperiment(analysis, out[index])
		if len(visualEvidence) == 0 {
			continue
		}
		out[index].EvidenceUsed = uniqueStrings(append(out[index].EvidenceUsed, visualEvidence...))
	}
	return out, nil
}

func acceptedVisualEvidenceForExperiment(analysis datasets.DatasetVisualAnalysis, experiment plans.PlannedExperiment) []string {
	mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
	if mechanism == "" {
		return nil
	}
	experimentText := experimentMechanismEvidenceText(experiment, nil)
	evidence := []string{}
	for _, hypothesis := range analysis.PreprocessingHypotheses {
		if strings.EqualFold(strings.TrimSpace(hypothesis.SupportStatus), "unsupported") {
			continue
		}
		hypothesisMechanism := strings.ToLower(strings.TrimSpace(hypothesis.Mechanism))
		if hypothesisMechanism != mechanism && !visualHypothesisCited(experimentText, hypothesis.ID) {
			continue
		}
		evidence = append(evidence, visualHypothesisEvidenceText(analysis.ID, hypothesis))
	}
	evidence = append(evidence, acceptedVisualTraitEvidenceForMechanism(analysis, mechanism)...)
	return uniqueStrings(evidence)
}

func visualHypothesisCited(experimentText string, hypothesisID string) bool {
	hypothesisID = strings.ToLower(strings.TrimSpace(hypothesisID))
	return hypothesisID != "" && strings.Contains(strings.ToLower(experimentText), hypothesisID)
}

func visualHypothesisEvidenceText(analysisID string, hypothesis datasets.PreprocessingHypothesis) string {
	parts := []string{
		"accepted visual analysis " + strings.TrimSpace(analysisID),
		"visual hypothesis " + strings.TrimSpace(hypothesis.ID),
		"mechanism " + strings.TrimSpace(hypothesis.Mechanism),
		"support_status " + strings.TrimSpace(hypothesis.SupportStatus),
		"confidence " + strings.TrimSpace(hypothesis.Confidence),
		strings.TrimSpace(hypothesis.Summary),
		strings.TrimSpace(hypothesis.ExpectedEffect),
	}
	parts = append(parts, hypothesis.Evidence...)
	return strings.Join(nonEmptyStringValues(parts), " ")
}

func acceptedVisualTraitEvidenceForMechanism(analysis datasets.DatasetVisualAnalysis, mechanism string) []string {
	wanted := map[string]bool{}
	switch mechanism {
	case "resolution_crop":
		wanted = map[string]bool{
			"small_objects":           true,
			"large_objects":           true,
			"background_dominance":    true,
			"fine_grained_similarity": true,
			"crop_bbox_useful":        true,
			"orientation_sensitive":   true,
			"domain_shift_possible":   true,
			"color_texture_signal":    true,
		}
	case "augmentation_basic", "augmentation_auto":
		wanted = map[string]bool{
			"lighting_variation":    true,
			"blur":                  true,
			"orientation_sensitive": true,
			"domain_shift_possible": true,
			"color_texture_signal":  true,
			"background_dominance":  true,
		}
	case "augmentation_mixed_sample", "regularization":
		wanted = map[string]bool{
			"fine_grained_similarity": true,
			"visual_ambiguity":        true,
			"color_texture_signal":    true,
			"background_dominance":    true,
		}
	default:
		return nil
	}
	evidence := []string{}
	for _, trait := range analysis.VisualTraits {
		traitName := strings.ToLower(strings.TrimSpace(trait.Trait))
		if !wanted[traitName] {
			continue
		}
		parts := []string{
			"accepted visual analysis " + strings.TrimSpace(analysis.ID),
			"visual trait " + traitName,
			"level " + strings.TrimSpace(trait.Level),
			"confidence " + strings.TrimSpace(trait.Confidence),
		}
		parts = append(parts, trait.Evidence...)
		evidence = append(evidence, strings.Join(nonEmptyStringValues(parts), " "))
	}
	return evidence
}

func (s *Server) recordChampionSelectedFollowUpBlocked(projectID string, planID string, decisionID string, followUpPlanID string, message string, reason string, details map[string]any) {
	payload := map[string]any{
		"decision_id":               decisionID,
		"backend_validation_status": "blocked",
		"backend_validation_error":  "champion selected guard",
		"backend_stop_guard":        "champion_selected_guard",
		"reason":                    reason,
	}
	for key, value := range details {
		payload[key] = value
	}
	if followUpPlanID != "" {
		payload["follow_up_plan_id"] = followUpPlanID
	}
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record champion-selected follow-up block failed for project %s decision %s: %v", projectID, decisionID, err)
	}
}

type projectChampionSelectionState struct {
	Terminal      bool
	TerminalAt    time.Time
	Reason        string
	PlanID        string
	DecisionID    string
	ChampionID    string
	ChampionJobID string
	Reopened      bool
	ReopenID      string
	ReopenAt      time.Time
}

func (s *Server) projectChampionSelectedFollowUpStopReason(projectID string) (string, map[string]any, bool, error) {
	state, err := s.projectChampionSelectionState(projectID)
	if err != nil {
		return "", nil, false, err
	}
	if !state.Terminal || state.Reopened {
		return "", nil, false, nil
	}
	details := map[string]any{
		"source_decision_id": state.DecisionID,
		"champion_id":        state.ChampionID,
		"champion_job_id":    state.ChampionJobID,
		"source_plan_id":     state.PlanID,
		"terminal_at":        state.TerminalAt.Format(time.RFC3339),
		"reopen_required":    true,
	}
	if state.ReopenID != "" {
		details["latest_reopen_decision_id"] = state.ReopenID
		details["latest_reopen_at"] = state.ReopenAt.Format(time.RFC3339)
	}
	return state.Reason, details, true, nil
}

func (s *Server) projectChampionSelectionState(projectID string) (projectChampionSelectionState, error) {
	state := projectChampionSelectionState{}

	champion, err := s.store.GetProjectChampion(projectID)
	if err == nil {
		state.Terminal = true
		state.TerminalAt = champion.UpdatedAt
		if state.TerminalAt.IsZero() {
			state.TerminalAt = champion.CreatedAt
		}
		state.Reason = fmt.Sprintf(
			"Project already has selected champion %s; autonomous follow-up scheduling requires an explicit reopen or new exploration round.",
			champion.JobID,
		)
		state.PlanID = champion.PlanID
		state.DecisionID = champion.SourceDecisionID
		state.ChampionID = champion.ID
		state.ChampionJobID = champion.JobID
	} else if !errors.Is(err, store.ErrNotFound) {
		return state, err
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		return state, err
	}
	for _, decision := range agentDecisions {
		decisionType := strings.ToUpper(strings.TrimSpace(decision.DecisionType))
		switch decisionType {
		case decisions.TypeSelectChampion:
			if !state.Terminal || decision.CreatedAt.After(state.TerminalAt) {
				state.Terminal = true
				state.TerminalAt = decision.CreatedAt
				state.Reason = fmt.Sprintf(
					"Project already has SELECT_CHAMPION decision %s; autonomous follow-up scheduling requires an explicit reopen or new exploration round.",
					decision.ID,
				)
				state.PlanID = decision.PlanID
				state.DecisionID = decision.ID
				state.ChampionID = ""
				state.ChampionJobID = payloadString(decision.Payload, "champion_job_id")
			}
		case decisions.TypeReopenExperimentation:
			if state.ReopenID == "" || decision.CreatedAt.After(state.ReopenAt) {
				state.ReopenID = decision.ID
				state.ReopenAt = decision.CreatedAt
			}
		}
	}
	if state.Terminal && state.ReopenID != "" && !state.ReopenAt.Before(state.TerminalAt) {
		state.Reopened = true
	}

	return state, nil
}

func (s *Server) createPendingStrategyScorecard(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, followUpPlan plans.ExperimentPlan) (strategies.StrategyScorecard, error) {
	datasetProfile := map[string]any{}
	computedTraits := []string{}
	if dataset, err := s.store.GetDataset(sourcePlan.DatasetID); err == nil {
		datasetProfile = dataset.Profile
		computedTraits = datasetProfileTraits(dataset.Profile)
	}
	datasetTraits := map[string]any{
		"dataset_id":                sourcePlan.DatasetID,
		"profile":                   datasetProfile,
		"computed_traits":           computedTraits,
		"dataset_planning_insights": decision.Payload["dataset_planning_insights"],
		"deterministic_diagnosis":   decision.Payload["deterministic_diagnosis"],
	}
	objectiveProfile := payloadMap(decision.Payload, "objective_context")
	proposedChanges := map[string]any{
		"hypothesis":            decision.Payload["hypothesis"],
		"changed_variables":     decision.Payload["changed_variables"],
		"proposed_experiments":  decision.Payload["proposed_experiments"],
		"proposal_mechanisms":   decision.Payload["proposal_mechanisms"],
		"candidate_hypotheses":  decision.Payload["candidate_hypotheses"],
		"candidate_rankings":    decision.Payload["candidate_rankings"],
		"rejected_options":      decision.Payload["rejected_options"],
		"success_criteria":      decision.Payload["success_criteria"],
		"deployment_tradeoff":   decision.Payload["deployment_tradeoff"],
		"why_can_beat_champion": decision.Payload["why_can_beat_champion"],
	}
	planningMode := payloadString(decision.Payload, "planning_mode")
	strategyType := planningMode
	if strategyType == "" {
		strategyType = "planner_followup"
	}
	mechanism, intervention, diagnosisTriggers, evidenceUsed, expectedEffect := strategyScorecardMechanismFields(decision.Payload)
	return s.store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:         projectID,
		DatasetID:         sourcePlan.DatasetID,
		SourceDecisionID:  decision.ID,
		SourcePlanID:      sourcePlan.ID,
		FollowUpPlanID:    followUpPlan.ID,
		StrategyType:      strategyType,
		PlanningMode:      planningMode,
		Mechanism:         mechanism,
		Intervention:      intervention,
		DiagnosisTriggers: diagnosisTriggers,
		EvidenceUsed:      evidenceUsed,
		ExpectedEffect:    expectedEffect,
		DatasetTraits:     datasetTraits,
		ObjectiveProfile:  objectiveProfile,
		ProposedChanges:   proposedChanges,
		ExpectedDelta:     payloadFloat(decision.Payload, "expected_delta_vs_champion"),
		ConfidenceBefore:  payloadFloat(decision.Payload, "confidence"),
		Outcome:           strategies.OutcomePending,
		Lesson:            "Pending follow-up outcome.",
		Tags:              uniqueStrings([]string{"strategy_scorecard", planningMode, mechanism, strategies.OutcomePending}),
	})
}

func strategyScorecardMechanismFields(payload map[string]any) (string, string, []string, []string, string) {
	diagnosisTriggers := payloadStringSlice(payload, "deterministic_diagnosis_used")
	if len(diagnosisTriggers) == 0 {
		diagnosisTriggers = payloadStringSlice(payload, "diagnosis_triggers")
	}
	evidenceUsed := payloadStringSlice(payload, "evidence_used")
	mechanism := payloadString(payload, "mechanism")
	intervention := payloadString(payload, "intervention")
	expectedEffect := payloadString(payload, "expected_effect")

	if proposals, ok, err := plannerProposalMechanismsFromPayload(payload); err == nil && ok {
		for _, proposal := range proposals {
			if mechanism == "" {
				mechanism = proposal.Mechanism
			}
			if intervention == "" {
				intervention = proposal.Intervention
			}
			if len(evidenceUsed) == 0 {
				evidenceUsed = proposal.EvidenceUsed
			}
			if expectedEffect == "" {
				expectedEffect = proposal.ExpectedEffect
			}
			if mechanism != "" && intervention != "" && len(evidenceUsed) > 0 && expectedEffect != "" {
				break
			}
		}
	}
	if mechanism == "" || intervention == "" || len(evidenceUsed) == 0 || expectedEffect == "" {
		if experiments, err := plannedExperimentsFromPayloadLenient(payload); err == nil {
			for _, experiment := range experiments {
				if mechanism == "" {
					mechanism = experiment.Mechanism
				}
				if intervention == "" {
					intervention = experiment.Intervention
				}
				if len(evidenceUsed) == 0 {
					evidenceUsed = experiment.EvidenceUsed
				}
				if expectedEffect == "" {
					expectedEffect = experiment.ExpectedEffect
				}
				if mechanism != "" && intervention != "" && len(evidenceUsed) > 0 && expectedEffect != "" {
					break
				}
			}
		}
	}
	return strings.TrimSpace(mechanism), strings.TrimSpace(intervention), uniqueStrings(diagnosisTriggers), uniqueStrings(evidenceUsed), strings.TrimSpace(expectedEffect)
}

func (s *Server) runAutomaticExperimentReview(projectID string) (automaticExperimentReviewResult, error) {
	if !s.shouldAutoReviewExperimentJobs() {
		return automaticExperimentReviewResult{}, nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	project, err := s.store.GetProject(projectID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	latestPlan, ok := latestExperimentPlan(projectPlans)
	if !ok {
		return automaticExperimentReviewResult{}, nil
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	recommendation := agents.NewExperimentReviewer().Review(project, latestPlan, summaries)
	if recommendation.DecisionType == decisions.TypeWait {
		return automaticExperimentReviewResult{}, nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	decision, ok := actionDecisionForPlan(agentDecisions, latestPlan.ID)
	if !ok {
		if terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
			if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(project.ID); err != nil {
				return automaticExperimentReviewResult{}, err
			} else if selected {
				message := fmt.Sprintf("Automatic experiment review skipped for plan %s because the project already has a selected champion.", latestPlan.ID)
				s.recordChampionSelectedFollowUpBlocked(project.ID, latestPlan.ID, "", "", message, stopReason, stopDetails)
				return automaticExperimentReviewResult{}, nil
			}
		}
		decision, err = s.store.CreateAgentDecision(
			project.ID,
			recommendation.PlanID,
			recommendation.DecisionType,
			recommendation.Rationale,
			recommendation.Payload,
		)
		if err != nil {
			return automaticExperimentReviewResult{}, err
		}
	}
	if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
		log.Printf("persist automatic reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
	}

	result := automaticExperimentReviewResult{
		Decision: &decision,
	}

	if decision.DecisionType != decisions.TypeAddExperiments {
		return result, nil
	}
	if !s.shouldAutoScheduleFollowUps() {
		return result, nil
	}

	if _, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		followUpPlan, _, err := s.ensureFollowUpPlan(project.ID, latestPlan, decision)
		if err != nil {
			if errors.Is(err, errNoNovelFollowUpExperiments) {
				return result, nil
			}
			return automaticExperimentReviewResult{}, err
		}
		result.FollowUpPlan = &followUpPlan
		return s.executeAutomaticFollowUpPlan(result)
	}

	maxRounds := s.maxAutoFollowUpRounds()
	if followUpRoundCount(projectPlans) >= maxRounds {
		log.Printf(
			"automatic follow-up scheduling skipped for project %s plan %s: max follow-up rounds reached (%d)",
			project.ID,
			latestPlan.ID,
			maxRounds,
		)
		return result, nil
	}

	followUpPlan, _, err := s.ensureFollowUpPlan(project.ID, latestPlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return result, nil
		}
		return automaticExperimentReviewResult{}, err
	}

	result.FollowUpPlan = &followUpPlan
	return s.executeAutomaticFollowUpPlan(result)
}

func (s *Server) executeAutomaticFollowUpPlan(result automaticExperimentReviewResult) (automaticExperimentReviewResult, error) {
	if result.FollowUpPlan == nil || !s.shouldAutoExecuteExperimentPlans() {
		return result, nil
	}

	req := s.defaultExecuteExperimentPlanRequest()
	planExecution, err := s.executeStoredExperimentPlan(result.FollowUpPlan.ID, req)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return result, nil
		}
		if _, eventErr := s.store.CreateExecutionEvent(
			result.FollowUpPlan.ProjectID,
			result.FollowUpPlan.ID,
			execution.EventExecutionFailed,
			fmt.Sprintf("Automatic execution failed for plan %s.", result.FollowUpPlan.ID),
			map[string]any{"error": err.Error()},
		); eventErr != nil {
			log.Printf("record automatic execution failure event failed: %v", eventErr)
		}
		return automaticExperimentReviewResult{}, err
	}

	result.Jobs = planExecution.Jobs
	if err := s.recordAutomaticExecutionQueued(*result.FollowUpPlan, req, planExecution.Jobs); err != nil {
		return automaticExperimentReviewResult{}, err
	}
	return result, nil
}

func (s *Server) recordAutomaticExecutionQueued(plan plans.ExperimentPlan, req executeExperimentPlanRequest, queuedJobs []jobs.ExperimentJob) error {
	openJobCount := openTrainingJobCount(queuedJobs)
	if openJobCount == 0 {
		_, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Plan %s has no open automatic experiment jobs requiring workers.", plan.ID), map[string]any{
			"job_ids":        experimentJobIDs(queuedJobs),
			"open_job_count": 0,
		})
		return err
	}
	provider := req.Provider
	if provider == "" {
		provider = "local"
	}
	targetCount := s.targetWorkerCountForPlan(plan, openJobCount)
	activeWorkerCount := s.activeOrStartingWorkersForProject(plan.ProjectID, provider, req.GPUType)
	requirementPolicy, err := s.workerRequirementPolicyForPlan(plan, provider, targetCount)
	if err != nil {
		return err
	}

	requirement, created, err := s.store.UpsertWorkerRequirement(
		plan.ProjectID,
		plan.ID,
		provider,
		req.GPUType,
		targetCount,
		"auto_followup",
		requirementPolicy,
	)
	if err != nil {
		return err
	}

	requirementStatus := requirement.Status
	if activeWorkerCount >= requirement.TargetCount {
		active := execution.WorkerRequirementActive
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active})
		if updateErr != nil {
			return updateErr
		}
		requirement = updated
		requirementStatus = updated.Status
	} else if requirement.Status == execution.WorkerRequirementActive {
		pending := execution.WorkerRequirementPending
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &pending})
		if updateErr != nil {
			return updateErr
		}
		requirement = updated
		requirementStatus = updated.Status
	}

	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Queued %d automatic experiment job(s) for plan %s.", len(queuedJobs), plan.ID), map[string]any{
		"job_ids":                           experimentJobIDs(queuedJobs),
		"open_job_count":                    openJobCount,
		"active_worker_count":               activeWorkerCount,
		"worker_requirement_id":             requirement.ID,
		"target_count":                      requirement.TargetCount,
		"requirement_status":                requirementStatus,
		"provider":                          provider,
		"gpu_type":                          req.GPUType,
		"dataset_id":                        requirement.DatasetID,
		"dataset_checksum":                  requirement.DatasetChecksum,
		"dataset_cache_key":                 requirement.DatasetCacheKey,
		"materialization_status":            requirement.DatasetMaterializationStatus,
		"cold_cache_policy":                 requirement.ColdCachePolicy,
		"max_concurrent_jobs":               requirement.MaxConcurrentJobs,
		"max_cold_dataset_materializations": requirement.MaxColdDatasetMaterializations,
	}); err != nil {
		return err
	}

	eventType := execution.EventWorkersRequired
	if !created {
		eventType = execution.EventWorkerScalingUpdated
	}
	message := fmt.Sprintf(
		"Automatic execution targets %d worker(s) for %d open job(s); %d worker(s) are already active or starting.",
		requirement.TargetCount,
		openJobCount,
		activeWorkerCount,
	)
	_, err = s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, eventType, message, map[string]any{
		"worker_requirement_id":             requirement.ID,
		"target_count":                      requirement.TargetCount,
		"open_job_count":                    openJobCount,
		"active_worker_count":               activeWorkerCount,
		"provider":                          requirement.Provider,
		"gpu_type":                          requirement.GPUType,
		"dataset_id":                        requirement.DatasetID,
		"dataset_checksum":                  requirement.DatasetChecksum,
		"dataset_cache_key":                 requirement.DatasetCacheKey,
		"materialization_status":            requirement.DatasetMaterializationStatus,
		"cold_cache_policy":                 requirement.ColdCachePolicy,
		"max_concurrent_jobs":               requirement.MaxConcurrentJobs,
		"max_cold_dataset_materializations": requirement.MaxColdDatasetMaterializations,
	})
	return err
}

func experimentJobIDs(experimentJobs []jobs.ExperimentJob) []string {
	out := make([]string, 0, len(experimentJobs))
	for _, job := range experimentJobs {
		out = append(out, job.ID)
	}
	return out
}

func openTrainingJobCount(experimentJobs []jobs.ExperimentJob) int {
	count := 0
	for _, job := range experimentJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(job.Status)) {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
			count++
		}
	}
	return count
}

func (s *Server) targetWorkerCountForPlan(plan plans.ExperimentPlan, openJobCount int) int {
	if openJobCount < 1 {
		return 1
	}
	targetCount := plan.RecommendedWorkers
	if targetCount < 1 {
		targetCount = 1
	}
	if targetCount > openJobCount {
		targetCount = openJobCount
	}
	if maxWorkers := maxAutoWorkersFromEnv(); maxWorkers > 0 && targetCount > maxWorkers {
		targetCount = maxWorkers
	}
	return targetCount
}

func (s *Server) activeOrStartingWorkersForProject(projectID string, provider string, gpuType string) int {
	projectWorkers, err := s.store.ListProjectWorkers(projectID)
	if err != nil {
		return 0
	}
	count := 0
	for _, worker := range projectWorkers {
		if time.Since(worker.LastHeartbeat) > workers.HeartbeatLimit {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(worker.Status)) {
		case workers.StatusIdle, workers.StatusRunning:
			if !workerMatchesProviderCapacity(worker, provider, gpuType) {
				continue
			}
			count++
		}
	}
	return count
}

func workerMatchesProviderCapacity(worker workers.Worker, provider string, gpuType string) bool {
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" {
		provider = "local"
	}
	workerGPU := strings.ToLower(strings.TrimSpace(worker.GPUType))
	requiredGPU := strings.ToLower(strings.TrimSpace(gpuType))
	if provider == "local" {
		if requiredGPU == "" || requiredGPU == "local" {
			return workerGPU == "" || workerGPU == "local"
		}
		return workerGPU == requiredGPU
	}
	if provider == "modal" {
		if workerGPU == "modal" {
			return true
		}
		if strings.HasPrefix(workerGPU, "modal:") {
			modalGPU := strings.TrimPrefix(workerGPU, "modal:")
			return requiredGPU == "" || modalGPU == requiredGPU
		}
		return false
	}
	if requiredGPU != "" {
		return workerGPU == provider+":"+requiredGPU
	}
	return workerGPU == provider
}

func (s *Server) workerRequirementPolicyForPlan(plan plans.ExperimentPlan, provider string, targetCount int) (execution.WorkerRequirementPolicy, error) {
	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return execution.WorkerRequirementPolicy{}, err
	}
	return datasetMaterializationPolicy(dataset, provider, targetCount), nil
}

func datasetMaterializationPolicy(dataset datasets.Dataset, provider string, maxConcurrentJobs int) execution.WorkerRequirementPolicy {
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" {
		provider = "local"
	}
	checksum := normalizedDatasetChecksum(dataset.ChecksumSHA256)
	policy := execution.WorkerRequirementPolicy{
		DatasetID:                    dataset.ID,
		DatasetChecksum:              checksum,
		DatasetCacheKey:              datasetCacheKey(dataset),
		DatasetMaterializationStatus: execution.DatasetMaterializationUnknown,
		MaxConcurrentJobs:            maxConcurrentJobs,
	}
	if provider == "modal" {
		policy.ColdCachePolicy = execution.ColdCachePolicySingleMaterialization
		policy.MaxColdDatasetMaterializations = 1
		if checksum != "" {
			policy.DatasetMaterializationStatus = execution.DatasetMaterializationCold
		}
	}
	return policy
}

func datasetCacheKey(dataset datasets.Dataset) string {
	checksum := normalizedDatasetChecksum(dataset.ChecksumSHA256)
	if checksum != "" {
		return "sha256-" + checksum
	}
	fingerprint := storageURIFingerprint(dataset.StorageURI)
	if fingerprint == "" {
		return ""
	}
	return "uri-" + fingerprint
}

func storageURIFingerprint(storageURI string) string {
	normalized := normalizedStorageURI(storageURI)
	if normalized == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(normalized))
	return fmt.Sprintf("%x", sum[:])
}

func normalizedDatasetChecksum(checksumSHA256 string) string {
	checksum := strings.ToLower(strings.TrimSpace(checksumSHA256))
	if len(checksum) != 64 {
		return ""
	}
	for _, ch := range checksum {
		if (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') {
			continue
		}
		return ""
	}
	return checksum
}

func normalizedStorageURI(storageURI string) string {
	trimmed := strings.TrimSpace(storageURI)
	if trimmed == "" {
		return ""
	}
	parsed, err := url.Parse(trimmed)
	if err == nil && parsed.Scheme == "s3" {
		return "s3://" + strings.ToLower(parsed.Host) + "/" + strings.TrimLeft(parsed.EscapedPath(), "/")
	}
	return trimmed
}

func mapFromStruct(value any) (map[string]any, error) {
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func actionDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID == planID && decision.DecisionType != decisions.TypeWait && decisionAutoExecutable(decision) {
			return decision, true
		}
	}

	return decisions.AgentDecision{}, false
}

func decisionAutoExecutable(decision decisions.AgentDecision) bool {
	value, ok := decision.Payload["auto_executable"].(bool)
	if ok && !value {
		return false
	}
	return true
}

func followUpRoundCount(projectPlans []plans.ExperimentPlan) int {
	count := 0
	for _, plan := range projectPlans {
		if plan.SourceDecisionID != "" {
			count++
		}
	}

	return count
}

func (s *Server) recordWorkerRequirementStatusEvent(requirement execution.WorkerRequirement) {
	eventType := ""
	message := ""
	switch requirement.Status {
	case execution.WorkerRequirementStarting:
		eventType = execution.EventWorkersStarting
		message = fmt.Sprintf("Starting %d worker(s) for plan %s.", requirement.TargetCount, requirement.PlanID)
	case execution.WorkerRequirementActive:
		eventType = execution.EventWorkersActive
		message = fmt.Sprintf("%d worker(s) are active for plan %s.", requirement.TargetCount, requirement.PlanID)
	case execution.WorkerRequirementFailed:
		eventType = execution.EventExecutionFailed
		message = fmt.Sprintf("Worker startup failed for plan %s.", requirement.PlanID)
	default:
		return
	}

	if _, err := s.store.CreateExecutionEvent(requirement.ProjectID, requirement.PlanID, eventType, message, map[string]any{
		"worker_requirement_id": requirement.ID,
		"status":                requirement.Status,
		"target_count":          requirement.TargetCount,
		"provider":              requirement.Provider,
		"gpu_type":              requirement.GPUType,
		"last_error":            requirement.LastError,
	}); err != nil {
		log.Printf("record worker requirement event failed: %v", err)
	}
}

func validWorkerRequirementStatus(status string) bool {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case execution.WorkerRequirementPending,
		execution.WorkerRequirementStarting,
		execution.WorkerRequirementActive,
		execution.WorkerRequirementFailed,
		execution.WorkerRequirementCancelled:
		return true
	default:
		return false
	}
}

func (s *Server) listProjectAgentDecisions(c *gin.Context) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"decisions": agentDecisions})
}

func (s *Server) listProjectAgentMemoryRecords(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	records, err := s.store.ListProjectAgentMemoryRecords(c.Param("id"), memory.AgentMemoryFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		Kind:      c.Query("kind"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"records": records})
}

func (s *Server) backfillProjectMemoryEmbeddings(c *gin.Context) {
	projectID := c.Param("id")
	if _, err := s.store.GetProject(projectID); err != nil {
		writeStoreError(c, err)
		return
	}
	result, err := memory.NewIndexer(s.store, nil, embeddings.ConfigFromEnv()).BackfillProject(c.Request.Context(), projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, result)
}

func (s *Server) listProjectAgentInvocations(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	invocations, err := s.store.ListProjectAgentInvocations(c.Param("id"), memory.AgentInvocationFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		AgentName: c.Query("agent_name"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"invocations": invocations})
}

func (s *Server) listProjectStrategyScorecards(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	scorecards, err := s.store.ListProjectStrategyScorecards(c.Param("id"), limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"scorecards": scorecards})
}

func (s *Server) listProjectWorkerRequirements(c *gin.Context) {
	requirements, err := s.store.ListProjectWorkerRequirements(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"requirements": requirements})
}

func (s *Server) updateWorkerRequirement(c *gin.Context) {
	var req execution.WorkerRequirementUpdate
	if !bindJSON(c, &req) {
		return
	}
	if req.Status != nil {
		normalizedStatus := strings.ToUpper(strings.TrimSpace(*req.Status))
		if !validWorkerRequirementStatus(normalizedStatus) {
			writeStoreError(c, fmt.Errorf("%w: invalid worker requirement status", store.ErrInvalidRequest))
			return
		}
		req.Status = &normalizedStatus
	}

	requirement, err := s.store.UpdateWorkerRequirement(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if req.Status != nil {
		s.recordWorkerRequirementStatusEvent(requirement)
	}

	c.JSON(http.StatusOK, requirement)
}

func (s *Server) listProjectExecutionEvents(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))
	events, err := s.store.ListProjectExecutionEvents(c.Param("id"), limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"events": events})
}

func (s *Server) streamProjectExecutionEvents(c *gin.Context) {
	projectID := c.Param("id")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))
	if limit < 1 || limit > 200 {
		limit = 50
	}
	interval, _ := strconv.Atoi(c.DefaultQuery("interval_ms", "2000"))
	if interval < 500 {
		interval = 500
	}
	if interval > 10000 {
		interval = 10000
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	lastID := c.GetHeader("Last-Event-ID")
	delivered := map[string]bool{}
	ticker := time.NewTicker(time.Duration(interval) * time.Millisecond)
	defer ticker.Stop()

	send := func() bool {
		events, err := s.store.ListProjectExecutionEvents(projectID, limit)
		if err != nil {
			c.SSEvent("error", gin.H{"error": err.Error()})
			c.Writer.Flush()
			return false
		}
		sort.Slice(events, func(i, j int) bool {
			return events[i].CreatedAt.Before(events[j].CreatedAt)
		})
		seenLastID := lastID == ""
		for _, event := range events {
			if delivered[event.ID] {
				continue
			}
			if !seenLastID {
				if event.ID == lastID {
					seenLastID = true
				}
				continue
			}
			c.Writer.WriteString("id: " + event.ID + "\n")
			c.SSEvent("execution_event", event)
			delivered[event.ID] = true
			lastID = event.ID
		}
		if !seenLastID {
			lastID = ""
		}
		c.Writer.Flush()
		return true
	}

	send()
	for {
		select {
		case <-c.Request.Context().Done():
			return
		case <-ticker.C:
			if !send() {
				return
			}
		}
	}
}

func (s *Server) reportChampionExportResult(c *gin.Context) {
	var req championExportResultRequest
	if !bindJSON(c, &req) {
		return
	}
	job, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if job.Template != jobs.TemplateExportChampion {
		writeStoreError(c, fmt.Errorf("%w: job is not a champion export job", store.ErrInvalidRequest))
		return
	}
	exportID := jobConfigString(job.Config, "export_id")
	if exportID == "" {
		writeStoreError(c, fmt.Errorf("%w: export job is missing export_id", store.ErrInvalidRequest))
		return
	}
	status := normalizeChampionExportResultStatus(req.Status)
	if status == "" {
		writeStoreError(c, fmt.Errorf("%w: export status must be READY, FAILED, or PENDING_ARTIFACT", store.ErrInvalidRequest))
		return
	}
	if status == runs.ChampionExportStatusReady && strings.TrimSpace(req.ArtifactURI) == "" {
		writeStoreError(c, fmt.Errorf("%w: artifact_uri is required for READY export result", store.ErrInvalidRequest))
		return
	}

	export, err := s.store.UpdateChampionExport(exportID, runs.ChampionExportUpdate{
		Status:           status,
		ArtifactURI:      strings.TrimSpace(req.ArtifactURI),
		Metadata:         req.Metadata,
		ValidationErrors: req.ValidationErrors,
		Error:            strings.TrimSpace(req.Error),
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if status == runs.ChampionExportStatusReady {
		job, err = s.store.CompleteJob(job.ID, "")
	} else if status == runs.ChampionExportStatusFailed {
		message := strings.TrimSpace(req.Error)
		if message == "" {
			message = "champion export failed"
		}
		job, err = s.store.FailJob(job.ID, message)
	}
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"export": export, "job": job})
}

func (s *Server) reportChampionDemoPredictionResult(c *gin.Context) {
	var req championDemoPredictionResultRequest
	if !bindJSON(c, &req) {
		return
	}
	job, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if job.Template != jobs.TemplateChampionDemoPrediction {
		writeStoreError(c, fmt.Errorf("%w: job is not a champion demo prediction job", store.ErrInvalidRequest))
		return
	}
	predictionID := jobConfigString(job.Config, "prediction_id")
	if predictionID == "" {
		writeStoreError(c, fmt.Errorf("%w: prediction job is missing prediction_id", store.ErrInvalidRequest))
		return
	}
	status := normalizeChampionDemoPredictionResultStatus(req.Status)
	if status == "" {
		writeStoreError(c, fmt.Errorf("%w: prediction status must be SUCCEEDED, FAILED, or RUNTIME_UNAVAILABLE", store.ErrInvalidRequest))
		return
	}
	if status == runs.ChampionDemoPredictionStatusSucceeded && strings.TrimSpace(req.PredictedLabel) == "" {
		writeStoreError(c, fmt.Errorf("%w: predicted_label is required for successful prediction", store.ErrInvalidRequest))
		return
	}

	prediction, err := s.store.UpdateChampionDemoPrediction(predictionID, runs.ChampionDemoPredictionUpdate{
		Status:         status,
		PredictedLabel: strings.TrimSpace(req.PredictedLabel),
		TrueLabel:      strings.TrimSpace(req.TrueLabel),
		Confidence:     req.Confidence,
		TopK:           req.TopK,
		LatencyMS:      req.LatencyMS,
		Correct:        req.Correct,
		Error:          strings.TrimSpace(req.Error),
		ImageMetadata:  req.ImageMetadata,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if status == runs.ChampionDemoPredictionStatusSucceeded {
		job, err = s.store.CompleteJob(job.ID, "")
	} else if status == runs.ChampionDemoPredictionStatusFailed {
		message := strings.TrimSpace(req.Error)
		if message == "" {
			message = "champion demo prediction failed"
		}
		job, err = s.store.FailJob(job.ID, message)
	} else {
		job, err = s.store.CompleteJob(job.ID, "")
	}
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"prediction": prediction, "job": job})
}

func (s *Server) completeJob(c *gin.Context) {
	var req completeJobRequest
	if !bindJSON(c, &req) {
		return
	}

	job, err := s.store.CompleteJob(c.Param("id"), req.MLflowRunID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusSucceeded,
		}); err != nil {
			writeStoreError(c, err)
			return
		}
		s.enqueueTrainingTerminalHooks(job)
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) failJob(c *gin.Context) {
	var req failJobRequest
	if !bindJSON(c, &req) {
		return
	}

	if req.Retryable {
		job, requeued, err := s.store.RetryJob(c.Param("id"), req.Error)
		if err != nil {
			writeStoreError(c, err)
			return
		}
		diagnostics.Event("warn", "job_retryable_failure", map[string]any{
			"job_id":       job.ID,
			"project_id":   job.ProjectID,
			"worker_id":    job.WorkerID,
			"template":     job.Template,
			"attempt":      job.Attempt,
			"max_attempts": job.MaxAttempts,
			"requeued":     requeued,
			"error":        req.Error,
		})
		s.recordRetryableJobFailureEvent(job, requeued, req.Error)
		if job.Template == jobs.TemplateTrainExperiment {
			status := jobs.StatusQueued
			if !requeued {
				status = jobs.StatusFailed
			}
			if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
				Status: status,
			}); err != nil {
				writeStoreError(c, err)
				return
			}
			if !requeued {
				s.enqueueTrainingTerminalHooks(job)
			}
		}
		if !requeued && job.Template == jobs.TemplateAnalyzeDatasetVisuals && jobConfigString(job.Config, "trigger_reason") == string(datasets.VisualTriggerInitialProfile) {
			if err := s.createInitialPlanForDataset(jobConfigString(job.Config, "dataset_id")); err != nil {
				writeStoreError(c, err)
				return
			}
		}
		c.JSON(http.StatusOK, job)
		return
	}

	job, err := s.store.FailJob(c.Param("id"), req.Error)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	diagnostics.Event("error", "job_failed", map[string]any{
		"job_id":     job.ID,
		"project_id": job.ProjectID,
		"worker_id":  job.WorkerID,
		"template":   job.Template,
		"attempt":    job.Attempt,
		"error":      req.Error,
	})

	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusFailed,
		}); err != nil {
			writeStoreError(c, err)
			return
		}
		s.enqueueTrainingTerminalHooks(job)
	}
	if job.Template == jobs.TemplateAnalyzeDatasetVisuals && jobConfigString(job.Config, "trigger_reason") == string(datasets.VisualTriggerInitialProfile) {
		if err := s.createInitialPlanForDataset(jobConfigString(job.Config, "dataset_id")); err != nil {
			writeStoreError(c, err)
			return
		}
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) recordRetryableJobFailureEvent(job jobs.ExperimentJob, requeued bool, message string) {
	planID := jobConfigString(job.Config, "plan_id")
	nextAttempt := job.Attempt + 1
	if nextAttempt > job.MaxAttempts {
		nextAttempt = job.MaxAttempts
	}
	eventType := execution.EventJobRetryQueued
	eventMessage := fmt.Sprintf("Job %s reported a retryable failure and was requeued for attempt %d of %d.", job.ID, nextAttempt, job.MaxAttempts)
	if !requeued {
		eventType = execution.EventExecutionFailed
		eventMessage = fmt.Sprintf("Job %s reported a retryable failure and exhausted %d attempts.", job.ID, job.MaxAttempts)
	}
	if _, err := s.store.CreateExecutionEvent(job.ProjectID, planID, eventType, eventMessage, map[string]any{
		"job_id":       job.ID,
		"worker_id":    job.WorkerID,
		"template":     job.Template,
		"attempt":      job.Attempt,
		"max_attempts": job.MaxAttempts,
		"requeued":     requeued,
		"error":        message,
	}); err != nil {
		log.Printf("record retryable job failure event failed: %v", err)
	}
}

func (s *Server) runAutomaticExperimentReviewAfterTrainingJob(job jobs.ExperimentJob) {
	if _, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
	}
}

func (s *Server) enqueueTrainingTerminalHooks(job jobs.ExperimentJob) {
	if job.ID == "" || job.Template != jobs.TemplateTrainExperiment {
		return
	}
	if !s.markTrainingTerminalHooksQueued(job.ID) {
		log.Printf("post-training hooks already queued for job %s", job.ID)
		return
	}
	go s.runTrainingTerminalHooks(job.ID)
}

func (s *Server) markTrainingTerminalHooksQueued(jobID string) bool {
	s.trainingTerminalHooksMu.Lock()
	defer s.trainingTerminalHooksMu.Unlock()

	if s.trainingTerminalHooksQueued == nil {
		s.trainingTerminalHooksQueued = make(map[string]bool)
	}
	if s.trainingTerminalHooksQueued[jobID] {
		return false
	}
	s.trainingTerminalHooksQueued[jobID] = true
	return true
}

func (s *Server) runTrainingTerminalHooks(jobID string) {
	job, err := s.store.GetJob(jobID)
	if err != nil {
		log.Printf("post-training hooks skipped for job %s: %v", jobID, err)
		return
	}
	if job.Template != jobs.TemplateTrainExperiment {
		return
	}
	if job.Status != jobs.StatusSucceeded && job.Status != jobs.StatusFailed {
		log.Printf("post-training hooks skipped for job %s: non-terminal status %s", job.ID, job.Status)
		return
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			message := fmt.Sprintf("post-training hooks panic for job %s: %v", job.ID, recovered)
			log.Print(message)
			s.recordTrainingTerminalHookEvent(job, "failed", message, fmt.Sprint(recovered))
		}
	}()

	s.recordTrainingTerminalHookEvent(job, "started", fmt.Sprintf("Post-training agent hooks started for job %s.", job.ID), "")
	if err := s.observeAutoMLTrialForJob(job); err != nil {
		log.Printf("AutoML trial observation failed for job %s: %v", job.ID, err)
	}
	s.runTrainingMonitorAfterTrainingJob(job)
	s.runPlanningLoopAfterTrainingJob(job)
	s.recordTrainingTerminalHookEvent(job, "finished", fmt.Sprintf("Post-training agent hooks finished for job %s.", job.ID), "")
}

func (s *Server) recordTrainingTerminalHookEvent(job jobs.ExperimentJob, status string, message string, errorText string) {
	planID := jobConfigString(job.Config, "plan_id")
	payload := map[string]any{
		"job_id":                       job.ID,
		"job_status":                   job.Status,
		"post_training_hooks_status":   status,
		"post_training_hooks_async":    true,
		"training_monitor_after_job":   true,
		"experiment_planner_after_job": true,
	}
	if errorText != "" {
		payload["error"] = errorText
	}
	if _, err := s.store.CreateExecutionEvent(job.ProjectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record post-training hook event failed for job %s: %v", job.ID, err)
	}
}

func (s *Server) runPlanningLoopAfterTrainingJob(job jobs.ExperimentJob) {
	if err := s.recordExperimentPlannerOutcomeAfterTrainingJob(job); err != nil {
		log.Printf("record experiment planner outcome failed after training job %s: %v", job.ID, err)
	}

	handled, err := s.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		log.Printf("llm experiment planner failed after training job %s: %v", job.ID, err)
		return
	}
	if handled {
		return
	}
	s.runAutomaticExperimentReviewAfterTrainingJob(job)
}

func (s *Server) recordExperimentPlannerOutcomeAfterTrainingJob(job jobs.ExperimentJob) error {
	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil || summary.PlanID == "" {
		return nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	plan, err := s.store.GetExperimentPlan(summary.PlanID)
	if err != nil {
		return err
	}
	if plan.SourceDecisionID == "" {
		return nil
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(job.ProjectID)
	if err != nil {
		return err
	}
	planSummaries := summariesForPlanID(summaries, plan.ID)
	if !planTrainingRunsComplete(plan, planSummaries) {
		return nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(job.ProjectID)
	if err != nil {
		return err
	}
	sourceDecision, ok := agentDecisionByID(agentDecisions, plan.SourceDecisionID)
	if !ok || sourceDecision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
		return nil
	}

	invocationID := payloadString(sourceDecision.Payload, "invocation_id")
	if invocationID == "" {
		return nil
	}
	invocation, err := s.store.GetAgentInvocation(invocationID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		return err
	}
	if payloadString(invocation.DownstreamOutcome, "follow_up_plan_id") == plan.ID {
		return nil
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(job.ProjectID)
	if err != nil {
		return err
	}
	outcome, err := experimentPlanningOutcomeForPlan(sourceDecision, plan, projectPlans, summaries)
	if err != nil {
		return err
	}
	outcomePayload, err := mapFromStruct(outcome)
	if err != nil {
		return err
	}

	updatedInvocation, err := s.store.UpdateAgentInvocationDownstreamOutcome(invocationID, outcomePayload)
	if err != nil {
		return err
	}

	tags := plannerOutcomeTags(outcome)
	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: updatedInvocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    plan.DatasetID,
		PlanID:       plan.ID,
		AgentName:    agents.ExperimentPlannerAgentName,
		Kind:         memory.KindPlanningOutcome,
		Summary:      outcome.Lesson,
		Payload:      outcomePayload,
		Tags:         tags,
	})
	if err != nil {
		return err
	}
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, plan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Experiment Planner outcome recorded for follow-up plan %s.", plan.ID), map[string]any{
		"invocation_id":      updatedInvocation.ID,
		"memory_record_id":   record.ID,
		"source_decision_id": sourceDecision.ID,
		"outcome_status":     outcome.OutcomeStatus,
	}); err != nil {
		log.Printf("record experiment planner outcome event failed: %v", err)
	}
	if updatedScorecard, err := s.store.UpdateStrategyScorecardOutcomeByFollowUpPlan(plan.ID, strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:     outcome.ActualDeltaVsChampion,
		ConfidenceAfter: plannerOutcomeConfidence(outcome),
		CostUSD:         outcome.TotalCostUSD,
		RuntimeSeconds:  outcome.TotalRuntimeSeconds,
		Outcome:         outcome.OutcomeStatus,
		Lesson:          outcome.Lesson,
		Tags:            tags,
	}); err == nil {
		s.indexMemoryCard(context.Background(), memory.NewStrategyScorecardMemoryCard(updatedScorecard))
	} else if !errors.Is(err, store.ErrNotFound) {
		log.Printf("update strategy scorecard failed for follow-up plan %s: %v", plan.ID, err)
	}
	return nil
}

func (s *Server) runTrainingMonitorAfterTrainingJob(job jobs.ExperimentJob) {
	if !trainingMonitorLLMEnabled() {
		return
	}
	if !s.shouldRunLLMAgents() {
		return
	}

	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		log.Printf("training monitor skipped for job %s: summary unavailable: %v", job.ID, err)
		return
	}

	metrics, err := s.store.ListJobMetrics(job.ID)
	if err != nil {
		log.Printf("training monitor skipped for job %s: metrics unavailable: %v", job.ID, err)
		return
	}

	plan := plans.ExperimentPlan{}
	if summary.PlanID != "" {
		plan, err = s.store.GetExperimentPlan(summary.PlanID)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			log.Printf("training monitor skipped for job %s: plan unavailable: %v", job.ID, err)
			return
		}
	}
	goalText := ""
	project, err := s.store.GetProject(job.ProjectID)
	if err != nil {
		log.Printf("training monitor skipped project objective context for job %s: project unavailable: %v", job.ID, err)
	} else {
		goalText = project.Goal
	}
	var evaluation *runs.TrainingRunEvaluation
	if storedEvaluation, err := s.store.GetTrainingRunEvaluation(job.ID); err == nil {
		evaluation = &storedEvaluation
	} else if !errors.Is(err, store.ErrNotFound) {
		log.Printf("training monitor evaluation lookup failed for job %s: %v", job.ID, err)
	}

	priorMemory, err := s.store.ListProjectAgentMemoryRecords(job.ProjectID, memory.AgentMemoryFilter{
		DatasetID: summary.DatasetID,
		Limit:     12,
	})
	if err != nil {
		log.Printf("training monitor memory lookup failed for job %s: %v", job.ID, err)
		priorMemory = []memory.AgentMemoryRecord{}
	}

	automationSettings := s.currentAutomationSettings()
	config := llm.ConfigFromEnv(automationSettings.LLMEnabled, automationSettings.LLMProvider, automationSettings.LLMModel)
	client := llm.NewClient(config)
	agent := agents.NewTrainingMonitorAgentWithRuntime(client, config.Model, config)

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	retrievedRunMemory := s.retrieveTrainingMonitorMemory(ctx, plan, job, summary, projectObjectiveContext(goalText))
	trace, err := agent.EvaluateWithTrace(ctx, agents.TrainingMonitorInput{
		Plan:               plan,
		Job:                job,
		Summary:            summary,
		Evaluation:         evaluation,
		Metrics:            metrics,
		ObjectiveContext:   projectObjectiveContext(goalText),
		MemoryRecords:      priorMemory,
		RetrievedRunMemory: retrievedRunMemory,
		OptimizerFeedback:  s.optimizerFeedbackSummary(jobConfigString(job.Config, "automl_study_id"), trainingMonitorTargetMetricFromJob(job, summary)),
	})
	acceptedForMemory := err == nil
	invocation, invocationErr := s.recordTrainingMonitorInvocation(job, summary, config, trace, acceptedForMemory)
	if invocationErr != nil {
		log.Printf("training monitor invocation write failed for job %s: %v", job.ID, invocationErr)
	}
	if err != nil {
		log.Printf("training monitor failed for job %s: %v", job.ID, err)
		diagnostics.Event("error", "training_monitor_failed", map[string]any{
			"job_id":        job.ID,
			"project_id":    job.ProjectID,
			"plan_id":       summary.PlanID,
			"invocation_id": invocation.ID,
			"provider":      config.Provider,
			"model":         config.Model,
			"api_style":     config.APIStyle,
			"error":         err.Error(),
		})
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentFailed, fmt.Sprintf("Training Monitor agent failed for job %s.", job.ID), map[string]any{
			"job_id":        job.ID,
			"invocation_id": invocation.ID,
			"error":         err.Error(),
			"agent_name":    agents.TrainingMonitorAgentName,
		}); eventErr != nil {
			log.Printf("record training monitor failure event failed: %v", eventErr)
		}
		return
	}

	recommendation := trace.Recommendation
	payload, err := mapFromStruct(recommendation)
	if err != nil {
		log.Printf("training monitor payload conversion failed for job %s: %v", job.ID, err)
		return
	}

	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    summary.DatasetID,
		PlanID:       summary.PlanID,
		JobID:        job.ID,
		AgentName:    agents.TrainingMonitorAgentName,
		Kind:         memory.KindTrainingEvaluation,
		Summary:      recommendation.Summary,
		Payload:      payload,
		Tags:         recommendation.Tags,
	})
	if err != nil {
		log.Printf("training monitor memory write failed for job %s: %v", job.ID, err)
		return
	}
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentRecommendationRecorded, fmt.Sprintf("Training Monitor recorded an evaluation for job %s.", job.ID), map[string]any{
		"job_id":           job.ID,
		"invocation_id":    invocation.ID,
		"memory_record_id": record.ID,
		"agent_name":       record.AgentName,
		"kind":             record.Kind,
	}); err != nil {
		log.Printf("record training monitor event failed: %v", err)
	}
}

func trainingMonitorLLMEnabled() bool {
	return envFlag("MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED", false)
}

func (s *Server) recordTrainingMonitorInvocation(
	job jobs.ExperimentJob,
	summary runs.TrainingRunSummary,
	config llm.Config,
	trace agents.TrainingMonitorEvaluationTrace,
	acceptedForMemory bool,
) (memory.AgentInvocation, error) {
	validationStatus := trace.ValidationStatus
	if validationStatus == "" {
		validationStatus = memory.InvocationValidationFailed
	}
	inputContext := agentInvocationInputContext(trace.PromptContext, config, trace.ToolRounds, trace.ToolCalls, trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults)

	return s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         job.ProjectID,
		DatasetID:         summary.DatasetID,
		PlanID:            summary.PlanID,
		JobID:             job.ID,
		AgentName:         agents.TrainingMonitorAgentName,
		AgentVersion:      trace.AgentVersion,
		PromptVersion:     trace.PromptVersion,
		Provider:          config.Provider,
		Model:             config.Model,
		InputMessages:     llmMessagesForMemory(trace.Request.Messages),
		InputContext:      inputContext,
		RawOutput:         string(trace.RawOutput),
		ParsedOutput:      trace.ParsedOutput,
		ValidationStatus:  validationStatus,
		ValidationError:   trace.ValidationError,
		AcceptedForMemory: acceptedForMemory,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
}

func llmMessagesForMemory(messages []llm.Message) []map[string]string {
	out := make([]map[string]string, 0, len(messages))
	for _, message := range messages {
		out = append(out, map[string]string{
			"role":    message.Role,
			"content": message.Content,
		})
	}
	return out
}

func agentInvocationInputContext(
	promptContext map[string]any,
	config llm.Config,
	toolRounds int,
	toolCalls []agents.AgentToolCallTrace,
	toolResults []agents.AgentToolResultTrace,
	rejectedToolCalls []agents.AgentToolResultTrace,
	dryRunValidationResults []map[string]any,
) map[string]any {
	out := map[string]any{}
	for key, value := range promptContext {
		out[key] = value
	}
	out["invocation_runtime"] = map[string]any{
		"api_style":                  config.APIStyle,
		"provider":                   config.Provider,
		"model":                      config.Model,
		"reasoning_effort":           config.ReasoningEffort,
		"plateau_reasoning_effort":   config.PlateauReasoningEffort,
		"stored_responses":           config.StoredResponses,
		"max_tool_rounds":            config.MaxToolRounds,
		"tool_rounds":                toolRounds,
		"tool_calls":                 toolCalls,
		"tool_results":               toolResults,
		"tool_names":                 agentToolNames(toolCalls),
		"rejected_tool_calls":        rejectedToolCalls,
		"dry_run_validation_results": dryRunValidationResults,
		"tool_calls_are_questions":   true,
		"mutation_authority":         false,
	}
	return out
}

func agentToolNames(calls []agents.AgentToolCallTrace) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, call := range calls {
		name := strings.TrimSpace(call.Name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

func (s *Server) runExperimentPlannerAfterTrainingJob(job jobs.ExperimentJob) (bool, error) {
	if !s.shouldRunLLMAgents() || !s.shouldAutoReviewExperimentJobs() {
		return false, nil
	}

	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		return false, nil
	}
	if summary.PlanID == "" {
		return false, nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	input, ready, err := s.buildExperimentPlannerInput(job.ProjectID, summary.PlanID)
	if err != nil || !ready {
		return false, err
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(job.ProjectID)
	if err != nil {
		return false, err
	}
	if decision, ok := experimentPlannerDecisionForPlan(agentDecisions, input.SourcePlan.ID); ok {
		if err := s.persistProjectChampionFromDecision(job.ProjectID, decision); err != nil {
			log.Printf("persist planner champion failed for project %s decision %s: %v", job.ProjectID, decision.ID, err)
		}
		result := automaticExperimentReviewResult{Decision: &decision}
		if decision.DecisionType == decisions.TypeAddExperiments &&
			s.shouldAutoScheduleFollowUps() &&
			s.currentAutomationSettings().AgentMode == llm.AgentModeAutonomous {
			return true, s.schedulePlannerDecision(job.ProjectID, input.SourcePlan, decision, result)
		}
		return true, nil
	}
	automationSettings := s.currentAutomationSettings()
	if terminalPlannerGuardsEnabledForMode(automationSettings.AgentMode) {
		if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(job.ProjectID); err != nil {
			return false, err
		} else if selected {
			message := fmt.Sprintf("Experiment Planner skipped for plan %s because the project already has a selected champion.", input.SourcePlan.ID)
			s.recordChampionSelectedFollowUpBlocked(job.ProjectID, input.SourcePlan.ID, "", "", message, stopReason, stopDetails)
			return true, nil
		}
	}

	config := llm.ConfigFromEnv(automationSettings.LLMEnabled, automationSettings.LLMProvider, automationSettings.LLMModel)
	client := llm.NewClient(config)
	agent := agents.NewExperimentPlannerAgentWithRuntime(client, config.Model, config, agents.PlannerInformationToolOptions{
		ValidateCandidateExperiments: plannerCandidateDryRunValidator(input),
	})

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	plannerAttempt, err := s.runExperimentPlannerWithBackendValidationRetry(ctx, agent, input, config, automationSettings.AgentMode)
	invocation := plannerAttempt.Invocation
	if err != nil {
		diagnostics.Event("error", "experiment_planner_failed", map[string]any{
			"job_id":        job.ID,
			"project_id":    job.ProjectID,
			"plan_id":       summary.PlanID,
			"invocation_id": invocation.ID,
			"provider":      config.Provider,
			"model":         config.Model,
			"api_style":     config.APIStyle,
			"error":         err.Error(),
		})
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentFailed, fmt.Sprintf("Experiment Planner agent failed for plan %s.", summary.PlanID), map[string]any{
			"invocation_id": invocation.ID,
			"error":         err.Error(),
			"agent_name":    agents.ExperimentPlannerAgentName,
		}); eventErr != nil {
			log.Printf("record experiment planner failure event failed: %v", eventErr)
		}
		return false, err
	}

	input = plannerAttempt.Input
	recommendation := plannerAttempt.Recommendation
	payload := plannerAttempt.Payload

	memoryPayload, err := mapFromStruct(recommendation)
	if err != nil {
		return false, err
	}
	memoryPayload["current_champion"] = input.CurrentChampion
	memoryPayload["source_plan_baseline_champion"] = input.SourcePlanBaselineChampion
	memoryPayload["source_plan_run_deltas"] = input.SourcePlanDeltas
	memoryPayload["dataset_planning_insights"] = input.DatasetInsights
	memoryPayload["objective_context"] = input.ObjectiveContext
	memoryPayload["deterministic_diagnosis"] = input.DeterministicDiagnosis
	memoryPayload["model_catalog"] = input.ModelCatalog
	memoryPayload["plan_evaluations"] = input.PlanEvaluations
	memoryPayload["successful_strategy_memory"] = input.SuccessfulStrategyMemory
	memoryPayload["failed_strategy_memory"] = input.FailedStrategyMemory
	memoryPayload["rejected_strategy_memory"] = input.RejectedStrategyMemory
	memoryPayload["strategy_scorecards"] = input.StrategyScorecards
	memoryPayload["optimizer_feedback_summary"] = input.OptimizerFeedback
	memoryPayload["no_improvement_rounds"] = input.NoImprovementRounds
	memoryPayload["stop_signals"] = input.StopSignals
	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    input.SourcePlan.DatasetID,
		PlanID:       input.SourcePlan.ID,
		AgentName:    agents.ExperimentPlannerAgentName,
		Kind:         memory.KindPlanningFeedback,
		Summary:      recommendation.Summary,
		Payload:      memoryPayload,
		Tags:         recommendation.Tags,
	})
	if err != nil {
		return false, err
	}
	payload["memory_record_id"] = record.ID
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	decisionType := strings.ToUpper(strings.TrimSpace(recommendation.DecisionType))
	if decisionType == decisions.TypeWait {
		return true, nil
	}

	decision, err := s.store.CreateAgentDecision(
		job.ProjectID,
		input.SourcePlan.ID,
		decisionType,
		recommendation.Rationale,
		payload,
	)
	if err != nil {
		return false, err
	}
	if err := s.persistProjectChampionFromDecision(job.ProjectID, decision); err != nil {
		log.Printf("persist planner champion failed for project %s decision %s: %v", job.ProjectID, decision.ID, err)
	}

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, input.SourcePlan.ID, execution.EventAgentRecommendationRecorded, fmt.Sprintf("Experiment Planner recorded a plan-level decision for plan %s.", input.SourcePlan.ID), map[string]any{
		"invocation_id":    invocation.ID,
		"memory_record_id": record.ID,
		"decision_id":      decision.ID,
		"decision_type":    decision.DecisionType,
		"agent_name":       agents.ExperimentPlannerAgentName,
	}); err != nil {
		log.Printf("record experiment planner event failed: %v", err)
	}

	result := automaticExperimentReviewResult{Decision: &decision}
	if decision.DecisionType != decisions.TypeAddExperiments ||
		!s.shouldAutoScheduleFollowUps() ||
		automationSettings.AgentMode != llm.AgentModeAutonomous {
		return true, nil
	}

	return true, s.schedulePlannerDecision(job.ProjectID, input.SourcePlan, decision, result)
}

func (s *Server) buildExperimentPlannerInput(projectID string, planID string) (agents.ExperimentPlannerInput, bool, error) {
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	if plan.ProjectID != projectID {
		return agents.ExperimentPlannerInput{}, false, fmt.Errorf("%w: plan does not belong to project", store.ErrInvalidRequest)
	}

	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return agents.ExperimentPlannerInput{}, false, err
	}

	planJobs := jobsForPlan(projectJobs, plan.ID)
	planSummaries := summariesForPlanID(summaries, plan.ID)
	planEvaluations := evaluationsForPlanID(evaluations, plan.ID)
	if !planTrainingRunsComplete(plan, planSummaries) {
		return agents.ExperimentPlannerInput{}, false, nil
	}
	automationSettings := s.currentAutomationSettings()
	minimumMeaningfulImprovement := plannerMinimumMeaningfulImprovementFromEnv(automationSettings.AgentMode)
	objectiveContext := projectObjectiveContext(project.Goal)
	currentChampion, baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals := experimentPlannerPerformanceContext(
		plan.TargetMetric,
		projectPlans,
		summaries,
		evaluations,
		objectiveContext,
		plan.ID,
	)

	planMetrics := map[string][]jobs.EpochMetric{}
	for _, planJob := range planJobs {
		metrics, err := s.store.ListJobMetrics(planJob.ID)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			return agents.ExperimentPlannerInput{}, false, err
		}
		planMetrics[planJob.ID] = metrics
	}
	visualContext, err := s.plannerVisualEvidenceContext(dataset)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}

	partialInput := agents.ExperimentPlannerInput{
		Project:                      project,
		Dataset:                      dataset,
		SourcePlan:                   plan,
		PlanJobs:                     planJobs,
		PlanSummaries:                planSummaries,
		PlanEvaluations:              planEvaluations,
		PlanMetrics:                  planMetrics,
		DatasetInsights:              datasetPlanningInsights(dataset, metadataSummary),
		VisualExemplarContext:        visualContext,
		ObjectiveContext:             objectiveContext,
		CurrentChampion:              currentChampion,
		SourcePlanBaselineChampion:   baselineChampion,
		SourcePlanDeltas:             sourcePlanDeltas,
		NoImprovementRounds:          noImprovementRounds,
		MinimumMeaningfulImprovement: minimumMeaningfulImprovement,
	}
	deterministicDiagnosis := agents.ComputePlannerDiagnosis(partialInput)

	priorMemory, err := s.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		DatasetID: plan.DatasetID,
		Limit:     25,
	})
	if err != nil {
		priorMemory = []memory.AgentMemoryRecord{}
	}
	successfulStrategyMemory, failedStrategyMemory := plannerStrategyMemory(priorMemory)
	rejectedStrategyMemory := plannerRejectedOptions(priorMemory, deterministicDiagnosis)
	scorecards, err := s.store.ListProjectStrategyScorecards(projectID, 12)
	if err != nil {
		scorecards = []strategies.StrategyScorecard{}
	}
	strategyScorecards := plannerStrategyScorecards(scorecards, plan.DatasetID)

	input := agents.ExperimentPlannerInput{
		Project:                      project,
		Dataset:                      dataset,
		SourcePlan:                   plan,
		PlanJobs:                     planJobs,
		PlanSummaries:                planSummaries,
		PlanEvaluations:              planEvaluations,
		PlanMetrics:                  planMetrics,
		DatasetInsights:              partialInput.DatasetInsights,
		VisualExemplarContext:        visualContext,
		ObjectiveContext:             objectiveContext,
		DeterministicDiagnosis:       deterministicDiagnosis,
		ModelCatalog:                 supportedModelCatalog(),
		CurrentChampion:              currentChampion,
		SourcePlanBaselineChampion:   baselineChampion,
		SourcePlanDeltas:             sourcePlanDeltas,
		NoImprovementRounds:          noImprovementRounds,
		StopSignals:                  stopSignals,
		MinimumMeaningfulImprovement: minimumMeaningfulImprovement,
		SuccessfulStrategyMemory:     successfulStrategyMemory,
		FailedStrategyMemory:         failedStrategyMemory,
		RejectedStrategyMemory:       rejectedStrategyMemory,
		StrategyScorecards:           strategyScorecards,
		OptimizerFeedback:            s.optimizerFeedbackSummariesForProject(projectID, plan.TargetMetric),
		PriorPlans:                   projectPlans,
		PriorJobs:                    projectJobs,
		PriorSummaries:               summaries,
		PriorEvaluations:             evaluations,
		PriorMemory:                  priorMemory,
		ExistingExperimentSignatures: experimentSignaturesForPlans(projectPlans),
		AgentMode:                    automationSettings.AgentMode,
		MaxExperiments:               maxLLMPlannerExperiments,
		MaxFollowUpRounds:            s.maxAutoFollowUpRounds(),
		FollowUpRound:                followUpRoundCount(projectPlans),
	}
	input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
	input.RetrievedMemory = s.retrievePlannerMemory(context.Background(), input)
	return input, true, nil
}

func (s *Server) recordExperimentPlannerInvocation(
	input agents.ExperimentPlannerInput,
	config llm.Config,
	trace agents.ExperimentPlanningTrace,
	acceptedForMemory bool,
) (memory.AgentInvocation, error) {
	validationStatus := trace.ValidationStatus
	if validationStatus == "" {
		validationStatus = memory.InvocationValidationFailed
	}
	inputContext := agentInvocationInputContext(trace.PromptContext, config, trace.ToolRounds, trace.ToolCalls, trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults)

	return s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         input.Project.ID,
		DatasetID:         input.SourcePlan.DatasetID,
		PlanID:            input.SourcePlan.ID,
		AgentName:         agents.ExperimentPlannerAgentName,
		AgentVersion:      trace.AgentVersion,
		PromptVersion:     trace.PromptVersion,
		Provider:          config.Provider,
		Model:             config.Model,
		InputMessages:     llmMessagesForMemory(trace.Request.Messages),
		InputContext:      inputContext,
		RawOutput:         string(trace.RawOutput),
		ParsedOutput:      trace.ParsedOutput,
		ValidationStatus:  validationStatus,
		ValidationError:   trace.ValidationError,
		AcceptedForMemory: acceptedForMemory,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
}

type experimentPlannerAttemptResult struct {
	Input          agents.ExperimentPlannerInput
	Trace          agents.ExperimentPlanningTrace
	Invocation     memory.AgentInvocation
	Recommendation agents.ExperimentPlanningRecommendation
	Payload        map[string]any
}

func (s *Server) runExperimentPlannerWithBackendValidationRetry(
	ctx context.Context,
	agent agents.ExperimentPlannerAgent,
	input agents.ExperimentPlannerInput,
	config llm.Config,
	agentMode string,
) (experimentPlannerAttemptResult, error) {
	attemptInput := input
	var result experimentPlannerAttemptResult
	var lastErr error
	for attempt := 0; attempt <= plannerBackendValidationRetryLimit; attempt++ {
		trace, err := agent.PlanWithTrace(ctx, attemptInput)
		acceptedForMemory := err == nil
		invocation, invocationErr := s.recordExperimentPlannerInvocation(attemptInput, config, trace, acceptedForMemory)
		if invocationErr != nil {
			log.Printf("experiment planner invocation write failed for plan %s: %v", input.SourcePlan.ID, invocationErr)
		}
		result = experimentPlannerAttemptResult{
			Input:      attemptInput,
			Trace:      trace,
			Invocation: invocation,
		}
		if err != nil {
			return result, err
		}

		recommendation := applyExperimentPlannerStopCriteria(trace.Recommendation, attemptInput)
		if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
			experiments, automlWarnings, prepareErr := s.prepareAutoMLExperimentsForProject(input.Project.ID, recommendation.ProposedExperiments)
			if prepareErr != nil {
				lastErr = prepareErr
				s.recordPlannerValidationRejection(invocation, prepareErr, attempt, attempt < plannerBackendValidationRetryLimit && shouldRetryExperimentPlannerValidation(recommendation))
				if attempt >= plannerBackendValidationRetryLimit || !shouldRetryExperimentPlannerValidation(recommendation) {
					result.Recommendation = recommendation
					return result, prepareErr
				}
				attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, prepareErr, attempt+1))
				continue
			}
			recommendation.ProposedExperiments = experiments
			recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, automlWarnings...)
		}
		payload, err := experimentPlannerDecisionPayload(recommendation, invocation, agentMode, attemptInput)
		if err == nil {
			if attempt > 0 {
				payload["validation_retry_count"] = attempt
				payload["validation_feedback_applied"] = attemptInput.ValidationFeedback
			}
			result.Recommendation = recommendation
			result.Payload = payload
			return result, nil
		}

		lastErr = err
		s.recordPlannerValidationRejection(invocation, err, attempt, attempt < plannerBackendValidationRetryLimit && shouldRetryExperimentPlannerValidation(recommendation))
		if attempt >= plannerBackendValidationRetryLimit || !shouldRetryExperimentPlannerValidation(recommendation) {
			result.Recommendation = recommendation
			return result, err
		}
		attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, err, attempt+1))
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("%w: experiment planner validation retry failed", store.ErrInvalidRequest)
	}
	return result, lastErr
}

func (s *Server) recordPlannerValidationRejection(invocation memory.AgentInvocation, validationErr error, attempt int, willRetry bool) {
	if invocation.ID == "" || validationErr == nil {
		return
	}
	outcome := map[string]any{
		"backend_validation_status": "rejected",
		"backend_validation_error":  validationErr.Error(),
		"retry_attempt":             attempt,
		"will_retry":                willRetry,
	}
	if runtime, ok := invocation.InputContext["invocation_runtime"].(map[string]any); ok {
		if dryRuns, ok := runtime["dry_run_validation_results"]; ok {
			outcome["dry_run_validation_results"] = dryRuns
		}
		if rejected, ok := runtime["rejected_tool_calls"]; ok {
			outcome["rejected_tool_calls"] = rejected
		}
	}
	if _, err := s.store.UpdateAgentInvocationDownstreamOutcome(invocation.ID, outcome); err != nil {
		log.Printf("update planner invocation validation outcome failed for invocation %s: %v", invocation.ID, err)
	}
}

func shouldRetryExperimentPlannerValidation(recommendation agents.ExperimentPlanningRecommendation) bool {
	return strings.EqualFold(strings.TrimSpace(recommendation.DecisionType), decisions.TypeAddExperiments)
}

func plannerCandidateDryRunValidator(input agents.ExperimentPlannerInput) agents.PlannerCandidateDryRunValidator {
	return func(recommendation agents.ExperimentPlanningRecommendation) agents.PlannerCandidateDryRunResult {
		result := agents.PlannerCandidateDryRunResult{
			Valid:                   true,
			ValidationStatus:        "valid",
			ProposedExperimentCount: len(recommendation.ProposedExperiments),
			CandidateCount:          len(recommendation.CandidateHypotheses),
			SelectedCandidateCount:  len(recommendation.ProposedExperiments),
			WouldWriteRows:          false,
			WouldScheduleJobs:       false,
			Details: map[string]any{
				"dry_run_only": true,
			},
		}
		if strings.EqualFold(strings.TrimSpace(recommendation.DecisionType), decisions.TypeAddExperiments) {
			experiments, err := plannerExperimentsWithProposalMechanisms(recommendation)
			if err != nil {
				return invalidPlannerDryRunResult(result, err)
			}
			for index := range experiments {
				if experiments[index].AutoML == nil || !experiments[index].AutoML.Enabled {
					continue
				}
				prepared, err := prepareAutoMLExperiment(experiments[index], index, automl.SamplerSeededRandom)
				if err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
				experiments[index] = prepared
			}
			if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
				return invalidPlannerDryRunResult(result, err)
			}
			if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
				return invalidPlannerDryRunResult(result, err)
			}
			for index, experiment := range experiments {
				if err := validatePlannedExperiment(experiment, index); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
			}
			if err := validateMechanismDatasetEvidence(profileWithAgentSafeMetadataSummary(input.Dataset.Profile, input.DatasetInsights.AgentSafeMetadataSummary), experiments, recommendation.EvidenceUsed); err != nil {
				return invalidPlannerDryRunResult(result, err)
			}
			result.Details["validated_experiment_count"] = len(experiments)
		}
		return result
	}
}

func invalidPlannerDryRunResult(result agents.PlannerCandidateDryRunResult, err error) agents.PlannerCandidateDryRunResult {
	result.Valid = false
	result.ValidationStatus = "invalid"
	if err != nil {
		result.ValidationError = err.Error()
	}
	result.WouldWriteRows = false
	result.WouldScheduleJobs = false
	return result
}

func plannerValidationFeedback(recommendation agents.ExperimentPlanningRecommendation, validationErr error, attempt int) agents.PlannerValidationFeedback {
	rejectedExperiments := make([]string, 0, len(recommendation.ProposedExperiments))
	rejectedModels := []string{}
	seenModels := map[string]bool{}
	for _, experiment := range recommendation.ProposedExperiments {
		rejectedExperiments = append(rejectedExperiments, experimentFeedbackSummary(experiment))
		model := strings.ToLower(strings.TrimSpace(experiment.Model))
		if model != "" && !seenModels[model] {
			seenModels[model] = true
			rejectedModels = append(rejectedModels, experiment.Model)
		}
	}
	return agents.PlannerValidationFeedback{
		Attempt:             attempt,
		ValidationError:     validationErr.Error(),
		RejectedDecision:    recommendation.DecisionType,
		RejectedModels:      rejectedModels,
		RejectedExperiments: rejectedExperiments,
		Instructions: []string{
			"Return corrected JSON only.",
			"Do not repeat rejected experiment mechanisms.",
			"Change a meaningful mechanism such as model family, preprocessing, augmentation policy, sampling/class balancing, scheduler, optimizer, regularization, or resolution strategy.",
			"Only propose experiments that backend validation can schedule.",
		},
	}
}

func experimentFeedbackSummary(experiment plans.PlannedExperiment) string {
	return strings.TrimSpace(fmt.Sprintf(
		"template=%s model=%s epochs=%d batch_size=%d learning_rate=%g optimizer=%s scheduler=%s weight_decay=%g dropout=%g label_smoothing=%g gradient_clip_norm=%g image_size=%d resolution_strategy=%s augmentation_policy=%s class_balancing=%s sampling_strategy=%s reason=%s",
		experiment.Template,
		experiment.Model,
		experiment.Epochs,
		experiment.BatchSize,
		experiment.LearningRate,
		experiment.Optimizer,
		experiment.Scheduler,
		experiment.WeightDecay,
		experiment.Dropout,
		experiment.LabelSmoothing,
		experiment.GradientClipNorm,
		experiment.ImageSize,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
		experiment.ClassBalancing,
		experiment.SamplingStrategy,
		experiment.Reason,
	))
}

func applyExperimentPlannerStopCriteria(
	recommendation agents.ExperimentPlanningRecommendation,
	input agents.ExperimentPlannerInput,
) agents.ExperimentPlanningRecommendation {
	decisionType := strings.ToUpper(strings.TrimSpace(recommendation.DecisionType))
	recommendation.DecisionType = decisionType
	if decisionType != decisions.TypeAddExperiments {
		return recommendation
	}
	stopReason, guardTag, ok := experimentPlannerBackendStopReason(input)
	if !ok {
		return recommendation
	}
	if !terminalPlannerGuardsEnabledForMode(input.AgentMode) {
		recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend stop advisory only; continuing is allowed because terminal planner guards are disabled: "+stopReason)
		recommendation.Tags = append(recommendation.Tags, "backend_stop_advisory", guardTag)
		return recommendation
	}

	recommendation.DecisionType = decisions.TypeSelectChampion
	recommendation.ChampionJobID = input.CurrentChampion.JobID
	recommendation.ProposedExperiments = nil
	recommendation.StopReason = stopReason
	recommendation.Summary = fmt.Sprintf("Select champion %s; backend stop criteria found no meaningful follow-up upside.", input.CurrentChampion.JobID)
	recommendation.Rationale = strings.TrimSpace(recommendation.Rationale + " Backend stop criteria applied: " + stopReason)
	recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend guard converted ADD_EXPERIMENTS to SELECT_CHAMPION because additional training had insufficient meaningful upside.")
	recommendation.Tags = append(recommendation.Tags, "select_champion", guardTag)
	return recommendation
}

func experimentPlannerBackendStopReason(input agents.ExperimentPlannerInput) (string, string, bool) {
	if reason, ok := nearMetricCeilingChampionStopReason(input); ok {
		return reason, "near_metric_ceiling_guard", true
	}
	if input.CurrentChampion == nil || input.NoImprovementRounds < plannerNoImprovementRoundsToSelect {
		return "", "", false
	}
	minimumMeaningfulImprovement := plannerMeaningfulImprovementThreshold(input.MinimumMeaningfulImprovement)
	return fmt.Sprintf(
		"Current champion %s remains unbeaten after %d consecutive follow-up plan(s) with less than %.3f target-metric improvement.",
		input.CurrentChampion.JobID,
		input.NoImprovementRounds,
		minimumMeaningfulImprovement,
	), "no_improvement_guard", true
}

func nearMetricCeilingChampionStopReason(input agents.ExperimentPlannerInput) (string, bool) {
	if input.CurrentChampion == nil {
		return "", false
	}
	ceiling, ok := boundedHigherIsBetterMetricCeiling(input.CurrentChampion.TargetMetric)
	if !ok {
		return "", false
	}
	minimumMeaningfulImprovement := plannerMeaningfulImprovementThreshold(input.MinimumMeaningfulImprovement)
	headroom := ceiling - input.CurrentChampion.Score
	if headroom < 0 {
		headroom = 0
	}
	if headroom > minimumMeaningfulImprovement {
		return "", false
	}
	return fmt.Sprintf(
		"Current champion %s already has %s %.3f, leaving %.3f possible headroom before the %.1f metric ceiling, which is below the minimum meaningful improvement %.3f.",
		input.CurrentChampion.JobID,
		input.CurrentChampion.TargetMetric,
		input.CurrentChampion.Score,
		headroom,
		ceiling,
		minimumMeaningfulImprovement,
	), true
}

func plannerMeaningfulImprovementThreshold(value float64) float64 {
	if value > 0 {
		return value
	}
	return plannerMinimumMeaningfulImprovement
}

func boundedHigherIsBetterMetricCeiling(metric string) (float64, bool) {
	switch normalizedPlannerTargetMetric(metric) {
	case "accuracy", "macro_f1":
		return 1.0, true
	default:
		return 0, false
	}
}

func experimentPlannerDecisionPayload(
	recommendation agents.ExperimentPlanningRecommendation,
	invocation memory.AgentInvocation,
	agentMode string,
	input agents.ExperimentPlannerInput,
) (map[string]any, error) {
	payload := map[string]any{
		"decision_source":                 llmExperimentPlannerDecisionSource,
		"agent_name":                      agents.ExperimentPlannerAgentName,
		"invocation_id":                   invocation.ID,
		"confidence":                      recommendation.Confidence,
		"auto_executable":                 agentMode == llm.AgentModeAutonomous,
		"planning_mode":                   recommendation.PlanningMode,
		"hypothesis":                      recommendation.Hypothesis,
		"dataset_preprocessing_rationale": recommendation.DatasetPreprocessingRationale,
		"changed_variables":               recommendation.ChangedVariables,
		"success_criteria":                recommendation.SuccessCriteria,
		"deployment_tradeoff":             recommendation.DeploymentTradeoff,
		"candidate_hypotheses":            recommendation.CandidateHypotheses,
		"candidate_rankings":              recommendation.CandidateRankings,
		"proposal_mechanisms":             recommendation.ProposalMechanisms,
		"risks":                           recommendation.Risks,
		"expected_tradeoffs":              recommendation.ExpectedTradeoffs,
		"novelty_notes":                   recommendation.NoveltyNotes,
		"champion_job_id":                 recommendation.ChampionJobID,
		"why_can_beat_champion":           recommendation.WhyCanBeatChampion,
		"expected_delta_vs_champion":      recommendation.ExpectedDeltaVsChampion,
		"stop_reason":                     recommendation.StopReason,
		"current_champion":                input.CurrentChampion,
		"source_plan_baseline_champion":   input.SourcePlanBaselineChampion,
		"source_plan_run_deltas":          input.SourcePlanDeltas,
		"dataset_planning_insights":       input.DatasetInsights,
		"objective_context":               input.ObjectiveContext,
		"deterministic_diagnosis":         input.DeterministicDiagnosis,
		"deterministic_diagnosis_used":    recommendation.DeterministicDiagnosisUsed,
		"project_trajectory_card":         input.ProjectTrajectory,
		"evidence_used":                   recommendation.EvidenceUsed,
		"expected_failure_modes":          recommendation.ExpectedFailureModes,
		"stop_condition":                  recommendation.StopCondition,
		"rejected_options":                recommendation.RejectedOptions,
		"model_catalog":                   input.ModelCatalog,
		"plan_evaluations":                input.PlanEvaluations,
		"successful_strategy_memory":      input.SuccessfulStrategyMemory,
		"failed_strategy_memory":          input.FailedStrategyMemory,
		"rejected_strategy_memory":        input.RejectedStrategyMemory,
		"strategy_scorecards":             input.StrategyScorecards,
		"optimizer_feedback_summary":      input.OptimizerFeedback,
		"no_improvement_rounds":           input.NoImprovementRounds,
		"minimum_meaningful_improvement":  input.MinimumMeaningfulImprovement,
		"stop_signals":                    input.StopSignals,
	}

	if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
		experiments, err := plannerExperimentsWithProposalMechanisms(recommendation)
		if err != nil {
			return nil, err
		}
		if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
			return nil, err
		}
		if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
			return nil, err
		}
		payload["proposed_experiments"] = experiments
	}

	return payload, nil
}

func (s *Server) persistProjectChampionFromDecision(projectID string, decision decisions.AgentDecision) error {
	decisionType := strings.ToUpper(strings.TrimSpace(decision.DecisionType))
	if decisionType != decisions.TypeSelectChampion && decisionType != decisions.TypeStopProject {
		return nil
	}

	championJobID := payloadString(decision.Payload, "champion_job_id")
	if championJobID == "" {
		if champion, ok := experimentChampionFromPayload(decision.Payload["current_champion"]); ok {
			championJobID = champion.JobID
		}
	}
	fallbackSelection := false
	if championJobID == "" && decisionType == decisions.TypeStopProject {
		fallbackJobID, ok, err := s.bestAvailableChampionJobForStoppedProject(projectID, decision)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		championJobID = fallbackJobID
		fallbackSelection = true
	}
	if championJobID == "" {
		return fmt.Errorf("%w: SELECT_CHAMPION decision is missing champion_job_id", store.ErrInvalidRequest)
	}

	summary, err := s.store.GetTrainingRunSummary(championJobID)
	if err != nil {
		return err
	}
	job, err := s.store.GetJob(championJobID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return err
	}
	goalText := ""
	if project, err := s.store.GetProject(projectID); err == nil {
		goalText = project.Goal
	}
	targetMetric := "macro_f1"
	if summary.PlanID != "" {
		if plan, err := s.store.GetExperimentPlan(summary.PlanID); err == nil && plan.TargetMetric != "" {
			targetMetric = plan.TargetMetric
		}
	}

	evaluationPayload := map[string]any{}
	deploymentProfile := map[string]any{
		"objective_context": projectObjectiveContext(goalText),
		"target_metric":     normalizedPlannerTargetMetric(targetMetric),
		"diagnostics":       "pending",
		"model_card": map[string]any{
			"intended_use":              goalText,
			"known_limitations":         []string{"Final export and production inference validation are still pending."},
			"recommended_preprocessing": []string{"Use the same image size, normalization, and augmentation assumptions from the winning experiment config."},
			"export_status":             "pending",
		},
	}
	if evaluation, err := s.store.GetTrainingRunEvaluation(championJobID); err == nil {
		if payload, payloadErr := mapFromStruct(evaluation); payloadErr == nil {
			evaluationPayload = payload
		}
		deploymentProfile["model_profile"] = evaluation.ModelProfile
		deploymentProfile["holistic_scores"] = evaluation.HolisticScores
		deploymentProfile["diagnostics"] = "available"
		if artifactURI := championArtifactURIFromEvaluation(evaluation.ModelProfile); artifactURI != "" {
			deploymentProfile["artifact_uri"] = artifactURI
			deploymentProfile["onnx_artifact_uri"] = artifactURI
			deploymentProfile["export_status"] = firstString(evaluation.ModelProfile, "export_status")
		}
		if manifestURI := firstString(evaluation.ModelProfile, "export_manifest_uri", "manifest_uri"); manifestURI != "" {
			deploymentProfile["export_manifest_uri"] = manifestURI
		}
		if manifest := payloadMap(evaluation.ModelProfile, "export_manifest"); len(manifest) > 0 {
			deploymentProfile["export_manifest"] = manifest
		}
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}

	selectionReason := strings.TrimSpace(decision.Rationale)
	if stopReason := payloadString(decision.Payload, "stop_reason"); stopReason != "" {
		selectionReason = strings.TrimSpace(selectionReason + " " + stopReason)
	}
	if fallbackSelection {
		fallbackReason := "Backend selected the best successful run so far because the planner stopped the project without naming a champion."
		if selectionReason == "" {
			selectionReason = fallbackReason
		} else {
			selectionReason = strings.TrimSpace(selectionReason + " " + fallbackReason)
		}
	}
	metrics := map[string]any{
		"model":                  summary.Model,
		"status":                 summary.Status,
		"best_macro_f1":          summary.BestMacroF1,
		"best_accuracy":          summary.BestAccuracy,
		"estimated_cost_usd":     summary.EstimatedCostUSD,
		"runtime_seconds":        summary.RuntimeSeconds,
		"epochs_completed":       summary.EpochsCompleted,
		"final_train_loss":       summary.FinalTrainLoss,
		"final_val_loss":         summary.FinalValLoss,
		"modal_function_call_id": summary.ModalFunctionCallID,
		"modal_input_id":         summary.ModalInputID,
	}
	if fallbackSelection {
		metrics["selection_source"] = "terminal_stop_best_available"
	}
	if job.ID != "" {
		metrics["job_config"] = job.Config
		if model := configString(job.Config, "model"); model != "" && summary.Model == "" {
			metrics["model"] = model
		}
	}

	champion, err := s.store.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:         projectID,
		DatasetID:         summary.DatasetID,
		PlanID:            summary.PlanID,
		JobID:             championJobID,
		SourceDecisionID:  decision.ID,
		SelectionReason:   selectionReason,
		Metrics:           metrics,
		Evaluation:        evaluationPayload,
		DeploymentProfile: deploymentProfile,
	})
	if err != nil {
		return err
	}

	_, err = s.store.CreateExecutionEvent(projectID, summary.PlanID, execution.EventChampionSelected, fmt.Sprintf("Champion selected: %s for project %s.", championJobID, projectID), map[string]any{
		"champion_id":        champion.ID,
		"champion_job_id":    champion.JobID,
		"source_decision_id": decision.ID,
		"selection_source":   metrics["selection_source"],
		"model":              metrics["model"],
	})
	return err
}

func (s *Server) bestAvailableChampionJobForStoppedProject(projectID string, decision decisions.AgentDecision) (string, bool, error) {
	targetMetric := payloadString(decision.Payload, "target_metric")
	if targetMetric == "" && decision.PlanID != "" {
		if plan, err := s.store.GetExperimentPlan(decision.PlanID); err == nil {
			targetMetric = plan.TargetMetric
		} else if !errors.Is(err, store.ErrNotFound) {
			return "", false, err
		}
	}
	if targetMetric == "" {
		targetMetric = "macro_f1"
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return "", false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return "", false, err
	}
	goalText := ""
	if project, err := s.store.GetProject(projectID); err == nil {
		goalText = project.Goal
	} else if !errors.Is(err, store.ErrNotFound) {
		return "", false, err
	}
	best, ok := bestSuccessfulTrainingSummaryForObjective(targetMetric, summaries, evaluations, projectObjectiveContext(goalText))
	if !ok {
		return "", false, nil
	}
	return best.JobID, true, nil
}

func (s *Server) schedulePlannerDecision(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, result automaticExperimentReviewResult) error {
	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return err
	}
	if stopReason, guardTag, ok, err := s.plannerFollowUpStopReason(projectID, sourcePlan, projectPlans); err != nil {
		return err
	} else if ok {
		if _, eventErr := s.store.CreateExecutionEvent(projectID, sourcePlan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Planner follow-up scheduling blocked for plan %s.", sourcePlan.ID), map[string]any{
			"source_decision_id":        decision.ID,
			"backend_validation_status": "blocked",
			"backend_stop_guard":        guardTag,
			"reason":                    stopReason,
		}); eventErr != nil {
			log.Printf("record planner follow-up stop event failed: %v", eventErr)
		}
		return nil
	}
	if _, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		followUpPlan, _, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
		if err != nil {
			if errors.Is(err, errNoNovelFollowUpExperiments) {
				return nil
			}
			return err
		}
		result.FollowUpPlan = &followUpPlan
		_, err = s.executeAutomaticFollowUpPlan(result)
		return err
	}

	maxRounds := s.maxAutoFollowUpRounds()
	if followUpRoundCount(projectPlans) >= maxRounds {
		log.Printf(
			"llm planner follow-up scheduling skipped for project %s plan %s: max follow-up rounds reached (%d)",
			projectID,
			sourcePlan.ID,
			maxRounds,
		)
		return nil
	}

	followUpPlan, _, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return nil
		}
		return err
	}

	result.FollowUpPlan = &followUpPlan
	_, err = s.executeAutomaticFollowUpPlan(result)
	return err
}

func (s *Server) plannerFollowUpStopReason(projectID string, sourcePlan plans.ExperimentPlan, projectPlans []plans.ExperimentPlan) (string, string, bool, error) {
	if !terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		return "", "", false, nil
	}
	if stopReason, _, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
		return "", "", false, err
	} else if ok {
		return stopReason, "champion_selected_guard", true, nil
	}
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return "", "", false, err
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return "", "", false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return "", "", false, err
	}
	currentChampion, _, _, noImprovementRounds, _ := experimentPlannerPerformanceContext(
		sourcePlan.TargetMetric,
		projectPlans,
		summaries,
		evaluations,
		projectObjectiveContext(project.Goal),
		sourcePlan.ID,
	)
	stopReason, guardTag, ok := experimentPlannerBackendStopReason(agents.ExperimentPlannerInput{
		CurrentChampion:              currentChampion,
		NoImprovementRounds:          noImprovementRounds,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	})
	return stopReason, guardTag, ok, nil
}

func planTrainingRunsComplete(plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) bool {
	if plan.ID == "" || len(plan.Experiments) == 0 {
		return false
	}

	if len(summaries) < len(plan.Experiments) {
		return false
	}

	for _, summary := range summaries {
		if !isTerminalTrainingSummary(summary) {
			return false
		}
	}
	return true
}

func isTerminalTrainingSummary(summary runs.TrainingRunSummary) bool {
	switch strings.ToUpper(strings.TrimSpace(summary.Status)) {
	case jobs.StatusSucceeded, jobs.StatusFailed:
		return true
	default:
		return false
	}
}

func experimentPlannerDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID != planID {
			continue
		}
		if decision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
			continue
		}
		return decision, true
	}
	return decisions.AgentDecision{}, false
}

func datasetPlanningInsights(dataset datasets.Dataset, agentSafeMetadataSummary map[string]any) agents.DatasetPlanningInsights {
	profile := dataset.Profile
	taskType := profileString(profile, "task_type")
	if taskType == "" {
		taskType = "image_classification"
	}
	classCount := profileInt(profile, "class_count")
	totalImages := profileInt(profile, "total_images")
	imageCount := profileInt(profile, "image_count")
	if metadataClassCount := int(payloadNumber(agentSafeMetadataSummary["class_count"])); metadataClassCount > classCount {
		classCount = metadataClassCount
	}
	if metadataSampleCount := int(payloadNumber(agentSafeMetadataSummary["sample_count"])); metadataSampleCount > 0 {
		if totalImages == 0 {
			totalImages = metadataSampleCount
		}
		if imageCount == 0 {
			imageCount = metadataSampleCount
		}
	}
	if totalImages == 0 {
		totalImages = imageCount
	}
	if imageCount == 0 {
		imageCount = totalImages
	}
	imbalanceRatio := profileFloat(profile, "imbalance_ratio")
	corruptImageCount := profileInt(profile, "corrupt_image_count")
	corruptFileCount := profileInt(profile, "corrupt_file_count")
	if corruptImageCount == 0 {
		corruptImageCount = corruptFileCount
	}
	if corruptFileCount == 0 {
		corruptFileCount = corruptImageCount
	}
	widthMin := profileInt(profile, "width_min")
	widthMax := profileInt(profile, "width_max")
	heightMin := profileInt(profile, "height_min")
	heightMax := profileInt(profile, "height_max")
	classDistribution := profileMap(profile, "class_distribution")
	if len(classDistribution) == 0 {
		classDistribution = profileMap(profile, "images_per_class")
	}
	imageDimensionStats := profileMap(profile, "image_dimension_stats")
	splitSummary := profileMap(profile, "split_summary")
	metadataSummary := safeLegacyMetadataSummary(profileMap(profile, "metadata_summary"))
	if len(agentSafeMetadataSummary) > 0 {
		metadataSummary = mergeMetadataEvidenceSummary(metadataSummary, agentSafeMetadataSummary)
	}
	leakageWarnings := profileStringSlice(profile, "leakage_warnings")
	datasetTraits := profileStringSlice(profile, "dataset_traits")
	artifacts := profileMapSlice(profile, "artifacts")

	constraints := []string{}
	recommendedPreprocessing := []string{"normalize with ImageNet statistics for transfer learning"}
	recommendedAugmentations := []string{}
	recommendedMetrics := []string{"accuracy", "macro_f1"}
	liveInferencePriorities := []string{
		"Prefer compact architectures when quality is close so the final model can classify live images with low latency.",
		"Only increase image_size when prior results show a meaningful quality gain over the deployment cost.",
	}

	if totalImages == 0 {
		constraints = append(constraints, "Dataset has not been profiled yet; use conservative transfer-learning defaults and prioritize profiling before aggressive search.")
	} else if totalImages < 500 {
		constraints = append(constraints, "Small dataset; avoid overfitting and prefer stronger augmentation, early stopping, and regularization.")
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter", "random_crop")
	} else if totalImages < 2000 {
		constraints = append(constraints, "Medium-small dataset; compare efficient transfer models with moderate augmentation.")
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter")
	}
	if imbalanceRatio >= 1.5 {
		constraints = append(constraints, fmt.Sprintf("Class imbalance detected (ratio %.2f); optimize macro-F1 and test class balancing.", imbalanceRatio))
		recommendedPreprocessing = append(recommendedPreprocessing, "class balancing with weighted_loss")
		recommendedMetrics = append(recommendedMetrics, "per_class_f1")
	}
	if corruptImageCount > 0 {
		constraints = append(constraints, fmt.Sprintf("%d corrupt image(s) were detected; clean or skip them before trusting final metrics.", corruptImageCount))
	}
	if len(leakageWarnings) > 0 {
		constraints = append(constraints, leakageWarnings...)
	}
	if metadataSummaryHasBBoxEvidence(metadataSummary) || containsString(datasetTraits, "bbox_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "bbox_crop_if_available as an ablation against full-image training")
	}
	if metadataBool(metadataSummary, "metadata_available") || containsString(datasetTraits, "metadata_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "preserve metadata artifacts for controlled preprocessing ablations")
	}
	if widthMax > 0 && heightMax > 0 {
		maxDimension := widthMax
		if heightMax > maxDimension {
			maxDimension = heightMax
		}
		minDimension := widthMin
		if heightMin > 0 && (minDimension == 0 || heightMin < minDimension) {
			minDimension = heightMin
		}
		if maxDimension >= 512 {
			recommendedPreprocessing = append(recommendedPreprocessing, "compare 224 and 256 image_size before trying larger inputs")
		} else if maxDimension <= 160 {
			recommendedPreprocessing = append(recommendedPreprocessing, "avoid unnecessary upscaling beyond 224 unless validation gains justify latency")
		}
		if minDimension > 0 && maxDimension > minDimension*2 {
			constraints = append(constraints, "Large variation in image dimensions; prefer resize plus random crop to improve robustness.")
			recommendedAugmentations = append(recommendedAugmentations, "random_crop")
		}
	}
	if len(recommendedAugmentations) == 0 {
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip if class semantics allow it")
	}

	summary := fmt.Sprintf(
		"%s dataset with %d classes, %d images, imbalance ratio %.2f, and %d corrupt image(s).",
		taskType,
		classCount,
		totalImages,
		imbalanceRatio,
		corruptImageCount,
	)

	return agents.DatasetPlanningInsights{
		Summary:                  summary,
		TaskType:                 taskType,
		ClassCount:               classCount,
		TotalImages:              totalImages,
		ImageCount:               imageCount,
		ClassDistribution:        classDistribution,
		ImbalanceRatio:           imbalanceRatio,
		CorruptImageCount:        corruptImageCount,
		CorruptFileCount:         corruptFileCount,
		WidthMin:                 widthMin,
		WidthMax:                 widthMax,
		HeightMin:                heightMin,
		HeightMax:                heightMax,
		ImageDimensionStats:      imageDimensionStats,
		SplitSummary:             splitSummary,
		MetadataSummary:          metadataSummary,
		AgentSafeMetadataSummary: agentSafeMetadataSummary,
		LeakageWarnings:          leakageWarnings,
		DatasetTraits:            datasetTraits,
		Artifacts:                artifacts,
		Constraints:              uniqueStrings(constraints),
		RecommendedPreprocessing: uniqueStrings(recommendedPreprocessing),
		RecommendedAugmentations: uniqueStrings(recommendedAugmentations),
		RecommendedMetrics:       uniqueStrings(recommendedMetrics),
		LiveInferencePriorities:  liveInferencePriorities,
	}
}

func (s *Server) activeAgentSafeDatasetMetadataSummary(dataset datasets.Dataset) (map[string]any, error) {
	var metadataImport datasets.DatasetMetadataImport
	var err error
	switch getter := any(s.store).(type) {
	case activeDatasetMetadataImportGetterWithContext:
		metadataImport, err = getter.GetActiveDatasetMetadataImport(context.Background(), dataset.ID)
	case activeDatasetMetadataImportGetter:
		metadataImport, err = getter.GetActiveDatasetMetadataImport(dataset.ID)
	default:
		return nil, nil
	}
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}
	if metadataImport.DatasetID != "" && metadataImport.DatasetID != dataset.ID {
		return nil, fmt.Errorf("%w: metadata import dataset_id does not match planner dataset", store.ErrInvalidRequest)
	}
	if metadataImport.ProjectID != "" && metadataImport.ProjectID != dataset.ProjectID {
		return nil, fmt.Errorf("%w: metadata import project_id does not match planner project", store.ErrInvalidRequest)
	}
	summary := metadataImport.AgentSafeSummary
	if summary.DatasetID == "" {
		summary.DatasetID = metadataImport.DatasetID
	}
	if summary.ImportID == "" {
		summary.ImportID = metadataImport.ID
	}
	if summary.Status == "" {
		summary.Status = metadataImport.Status
	}
	if summary.ImportVersion == 0 {
		summary.ImportVersion = metadataImport.ImportVersion
	}
	if summary.SourceKind == "" {
		summary.SourceKind = metadataImport.SourceKind
	}
	return agentSafeDatasetMetadataSummaryMap(summary), nil
}

func agentSafeDatasetMetadataSummaryMap(summary datasets.AgentSafeDatasetMetadataSummary) map[string]any {
	out := map[string]any{}
	addString := func(key string, value string) {
		if strings.TrimSpace(value) != "" {
			out[key] = strings.TrimSpace(value)
		}
	}
	addInt := func(key string, value int) {
		if value > 0 {
			out[key] = value
		}
	}
	addFloat := func(key string, value float64) {
		if value > 0 && isFiniteFloat(value) {
			out[key] = roundDiagnosticFloat(value)
		}
	}
	addString("dataset_id", summary.DatasetID)
	addString("import_id", summary.ImportID)
	addString("status", summary.Status)
	addInt("import_version", summary.ImportVersion)
	addString("source_kind", summary.SourceKind)
	addInt("source_count", summary.SourceCount)
	addInt("parsed_source_count", summary.ParsedSourceCount)
	addInt("unsupported_source_count", summary.UnsupportedSourceCount)
	addInt("error_source_count", summary.ErrorSourceCount)
	if len(summary.SourceFormats) > 0 {
		out["source_formats"] = uniqueStrings(summary.SourceFormats)
	}
	addInt("class_count", summary.ClassCount)
	addInt("sample_count", summary.SampleCount)
	addInt("labeled_sample_count", summary.LabeledSampleCount)
	addInt("missing_label_count", summary.MissingLabelCount)
	if len(summary.SplitCounts) > 0 {
		out["split_counts"] = copyStringIntMap(summary.SplitCounts)
	}
	if summary.OfficialSplitAvailable {
		out["official_split_available"] = true
	}
	if len(summary.AnnotationCounts) > 0 {
		out["annotation_counts"] = copyStringIntMap(summary.AnnotationCounts)
	}
	addInt("bbox_annotation_count", summary.BBoxAnnotationCount)
	addInt("bbox_sample_count", summary.BBoxSampleCount)
	addFloat("bbox_coverage_ratio", summary.BBoxCoverageRatio)
	if len(summary.Warnings) > 0 {
		out["warnings"] = metadataIssueSummaries(summary.Warnings, 8)
	}
	if len(summary.Errors) > 0 {
		out["errors"] = metadataIssueSummaries(summary.Errors, 8)
	}
	if len(summary.Capabilities) > 0 {
		out["capabilities"] = copyStringBoolMap(summary.Capabilities)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func metadataIssueSummaries(issues []datasets.MetadataIssue, limit int) []map[string]any {
	out := []map[string]any{}
	for _, issue := range issues {
		item := map[string]any{}
		if strings.TrimSpace(issue.Severity) != "" {
			item["severity"] = strings.TrimSpace(issue.Severity)
		}
		if strings.TrimSpace(issue.Code) != "" {
			item["code"] = strings.TrimSpace(issue.Code)
		}
		if strings.TrimSpace(issue.Message) != "" && !metadataSummaryTextLooksRaw(issue.Message) {
			item["message"] = strings.TrimSpace(issue.Message)
		}
		if issue.Count > 0 {
			item["count"] = issue.Count
		}
		if len(item) == 0 {
			continue
		}
		out = append(out, item)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func safeLegacyMetadataSummary(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	allowed := map[string]bool{
		"annotation_counts":        true,
		"annotations_available":    true,
		"artifact_counts":          true,
		"bbox_annotation_count":    true,
		"bbox_annotations_count":   true,
		"bbox_available":           true,
		"bbox_count":               true,
		"bbox_coverage_ratio":      true,
		"bbox_sample_count":        true,
		"capabilities":             true,
		"class_count":              true,
		"detected_formats":         true,
		"error_source_count":       true,
		"errors":                   true,
		"formats":                  true,
		"import_id":                true,
		"import_version":           true,
		"labeled_sample_count":     true,
		"metadata_available":       true,
		"missing_label_count":      true,
		"official_split_available": true,
		"parsed_source_count":      true,
		"sample_count":             true,
		"source_count":             true,
		"source_formats":           true,
		"source_kind":              true,
		"split_counts":             true,
		"status":                   true,
		"unsupported_source_count": true,
		"warnings":                 true,
	}
	out := map[string]any{}
	for key, value := range input {
		normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(key)), "-", "_")
		if !allowed[normalized] || metadataSummaryKeyLooksRaw(normalized) {
			continue
		}
		if text, ok := value.(string); ok && metadataSummaryTextLooksRaw(text) {
			continue
		}
		out[normalized] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func mergeMetadataEvidenceSummary(base map[string]any, overlay map[string]any) map[string]any {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}
	out := copyPayloadMap(base)
	for key, value := range overlay {
		out[key] = value
	}
	return out
}

func profileWithAgentSafeMetadataSummary(profile map[string]any, summary map[string]any) map[string]any {
	if len(summary) == 0 {
		return profile
	}
	out := copyPayloadMap(profile)
	out["agent_safe_metadata_summary"] = summary
	out["metadata_summary"] = mergeMetadataEvidenceSummary(safeLegacyMetadataSummary(profileMap(profile, "metadata_summary")), summary)
	if classCount := int(payloadNumber(summary["class_count"])); classCount > profileInt(out, "class_count") {
		out["class_count"] = classCount
	}
	if sampleCount := int(payloadNumber(summary["sample_count"])); sampleCount > 0 {
		if profileInt(out, "total_images") == 0 {
			out["total_images"] = sampleCount
		}
		if profileInt(out, "image_count") == 0 {
			out["image_count"] = sampleCount
		}
	}
	if metadataSummaryHasBBoxEvidence(summary) {
		out["bbox_available"] = true
	}
	out["metadata_available"] = true
	return out
}

func metadataSummaryKeyLooksRaw(key string) bool {
	normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(key)), "-", "_")
	if strings.HasSuffix(normalized, "_count") || strings.HasSuffix(normalized, "_counts") || strings.HasSuffix(normalized, "_ratio") {
		return false
	}
	return containsAnyText(normalized, "path", "uri", "url", "raw", "preview", "content", "sidecar", "storage", "checksum", "filename", "file_name", "manifest_record", "source_row")
}

func metadataSummaryTextLooksRaw(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, "://") || strings.Contains(lower, `:\`) || strings.HasPrefix(lower, "/") {
		return true
	}
	if strings.Contains(lower, "/") && containsAnyText(lower, ".csv", ".json", ".xml", ".txt", ".tsv", ".jpg", ".jpeg", ".png", ".parquet", ".yaml", ".yml") {
		return true
	}
	return containsAnyText(lower, "raw_preview", "raw row", "source row", "sidecar content", "storage uri", "local path")
}

func copyStringIntMap(input map[string]int) map[string]int {
	out := make(map[string]int, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func copyStringBoolMap(input map[string]bool) map[string]bool {
	out := make(map[string]bool, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func (s *Server) plannerVisualEvidenceContext(dataset datasets.Dataset) (*agents.PlannerVisualExemplarContext, error) {
	analysis, err := s.store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return plannerVisualExemplarContext(dataset), nil
		}
		return nil, err
	}
	if analysis.DatasetID != "" && analysis.DatasetID != dataset.ID {
		return nil, fmt.Errorf("%w: visual analysis dataset_id does not match planner dataset", store.ErrInvalidRequest)
	}
	if analysis.ProjectID != "" && analysis.ProjectID != dataset.ProjectID {
		return nil, fmt.Errorf("%w: visual analysis project_id does not match planner project", store.ErrInvalidRequest)
	}
	return agents.NewPlannerVisualExemplarContextFromAnalysis(analysis), nil
}

func plannerVisualExemplarContext(dataset datasets.Dataset) *agents.PlannerVisualExemplarContext {
	caps := visualExemplarCaps{MaxTotalImages: 24, MaxPerClass: 2, MaxBytes: 512 * 1024}
	exemplars := cappedVisualExemplars(dataset.Profile, caps, "visual_exemplars")
	if len(exemplars) == 0 {
		return &agents.PlannerVisualExemplarContext{
			Enabled:                 false,
			EvidenceOnly:            true,
			Source:                  "none",
			ByteBudget:              int(caps.MaxBytes),
			PromptBudget:            4000,
			Summary:                 "No backend-curated visual evidence is available for this dataset.",
			RawImagesIncluded:       false,
			RawVisualOutputIncluded: false,
		}
	}

	byClass := map[string][]datasets.VisualExemplar{}
	var totalBytes int64
	for _, exemplar := range exemplars {
		className := exemplar.ClassName
		if className == "" {
			className = exemplar.Label
		}
		if className == "" {
			className = "unknown"
		}
		byClass[className] = append(byClass[className], exemplar)
		totalBytes += maxInt64(exemplar.SizeBytes, 0)
	}

	classNames := make([]string, 0, len(byClass))
	for className := range byClass {
		classNames = append(classNames, className)
	}
	sort.Strings(classNames)
	classEvidence := make([]agents.PlannerClassExemplar, 0, len(classNames))
	for _, className := range classNames {
		classExemplars := byClass[className]
		metadata := map[string]any{}
		first := classExemplars[0]
		if first.Split != "" {
			metadata["split"] = first.Split
		}
		if first.Width > 0 {
			metadata["width"] = first.Width
		}
		if first.Height > 0 {
			metadata["height"] = first.Height
		}
		classEvidence = append(classEvidence, agents.PlannerClassExemplar{
			ClassName:     className,
			ExemplarCount: len(classExemplars),
			Metadata:      metadata,
		})
	}

	warnings := []string{}
	if len(exemplars) >= caps.MaxTotalImages {
		warnings = append(warnings, "visual exemplar cap reached; planner saw a bounded class-balanced subset")
	}
	if totalBytes >= caps.MaxBytes {
		warnings = append(warnings, "visual exemplar byte budget reached")
	}

	return &agents.PlannerVisualExemplarContext{
		Enabled:                 true,
		EvidenceOnly:            true,
		Source:                  "datasets.profile.visual_exemplars",
		ExemplarCount:           len(exemplars),
		ClassCount:              len(classEvidence),
		ByteBudget:              int(caps.MaxBytes),
		PromptBudget:            4000,
		Summary:                 fmt.Sprintf("%d backend-curated visual exemplar(s) across %d class(es) are available as evidence only.", len(exemplars), len(classEvidence)),
		ObservedTraits:          datasetProfileTraits(dataset.Profile),
		ClassEvidence:           classEvidence,
		Warnings:                warnings,
		RawImagesIncluded:       false,
		RawVisualOutputIncluded: false,
	}
}

func datasetProfileTraits(profile map[string]any) []string {
	traits := []string{}
	classCount := profileInt(profile, "class_count")
	totalImages := profileInt(profile, "total_images")
	imbalanceRatio := profileFloat(profile, "imbalance_ratio")
	widthMax := profileInt(profile, "width_max")
	heightMax := profileInt(profile, "height_max")
	maxDimension := widthMax
	if heightMax > maxDimension {
		maxDimension = heightMax
	}

	if totalImages > 0 && totalImages < 500 {
		traits = append(traits, "small_dataset")
	} else if totalImages >= 500 && totalImages < 5000 {
		traits = append(traits, "medium_dataset")
	}
	if classCount >= 20 {
		traits = append(traits, "many_classes")
	}
	if classCount >= 10 && totalImages > 0 && totalImages/classCount < 120 {
		traits = append(traits, "fine_grained_possible")
	}
	if imbalanceRatio >= 1.5 {
		traits = append(traits, "imbalanced")
	}
	if maxDimension > 0 && maxDimension <= 160 {
		traits = append(traits, "low_resolution")
	} else if maxDimension >= 512 {
		traits = append(traits, "high_resolution")
	}
	if profileBool(profile, "metadata_available") || profile["metadata_summary"] != nil {
		traits = append(traits, "metadata_available")
	}
	if profileBool(profile, "bbox_available") || profileBool(profile, "bounding_boxes_available") {
		traits = append(traits, "bbox_available")
	}
	return uniqueStrings(traits)
}

func projectObjectiveContext(goal string) agents.ProjectObjectiveContext {
	goalText := strings.TrimSpace(goal)
	normalized := strings.ToLower(goalText)
	context := agents.ProjectObjectiveContext{
		GoalText:             goalText,
		PrimaryObjective:     "balanced_quality",
		MetricPreferences:    []string{"macro_f1", "accuracy"},
		DeploymentPriorities: []string{"explain quality, cost, and runtime tradeoffs"},
		Constraints:          []string{},
		RankingWeights: map[string]float64{
			"macro_f1":           0.35,
			"accuracy":           0.25,
			"per_class_behavior": 0.15,
			"latency":            0.10,
			"training_cost":      0.08,
			"runtime":            0.07,
		},
	}

	if containsAny(normalized, "live", "real-time", "realtime", "instant", "fast", "quick", "low latency") {
		context.PrimaryObjective = "low_latency_live_service"
		context.DeploymentPriorities = append(context.DeploymentPriorities, "treat inference latency under roughly 25ms as acceptable for live use", "use latency as a tiebreaker when quality is close")
		context.Constraints = append(context.Constraints, "allow stronger quality challengers when expected or observed latency remains within the live budget")
		context.RankingWeights["latency"] = 0.08
		context.RankingWeights["model_size"] = 0.04
		context.RankingWeights["macro_f1"] = 0.40
		context.RankingWeights["accuracy"] = 0.26
	}
	if containsAny(normalized, "cheap", "budget", "cost", "low cost", "inexpensive") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "budget_sensitive"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer lower training and inference cost when quality is close")
		context.RankingWeights["training_cost"] = 0.18
		context.RankingWeights["runtime"] = 0.10
	}
	if containsAny(normalized, "accurate", "accuracy", "best", "quality", "high quality") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "quality_first"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "allow stronger models when they produce meaningful quality gains")
		context.RankingWeights["macro_f1"] = 0.45
		context.RankingWeights["accuracy"] = 0.30
		context.RankingWeights["per_class_behavior"] = 0.18
	}
	if containsAny(normalized, "imbalanced", "minority", "rare class", "rare classes", "fair", "per-class", "per class") {
		context.MetricPreferences = append(context.MetricPreferences, "per_class_f1", "recall_by_class")
		context.DeploymentPriorities = append(context.DeploymentPriorities, "avoid selecting a model that hides weak minority-class behavior behind average accuracy")
		context.RankingWeights["per_class_behavior"] = 0.24
		context.RankingWeights["macro_f1"] = 0.40
	}
	if containsAny(normalized, "mobile", "edge", "browser", "desktop", "cpu") {
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer compact CPU-friendly models only when quality is close or latency exceeds the live budget")
		context.Constraints = append(context.Constraints, "do not reject quality challengers solely for being larger when latency remains acceptable")
		context.RankingWeights["model_size"] = maxFloat(context.RankingWeights["model_size"], 0.08)
		context.RankingWeights["latency"] = maxFloat(context.RankingWeights["latency"], 0.10)
	}

	context.MetricPreferences = uniqueStrings(context.MetricPreferences)
	context.DeploymentPriorities = uniqueStrings(context.DeploymentPriorities)
	context.Constraints = uniqueStrings(context.Constraints)
	return context
}

func (s *Server) retrievePlannerMemory(ctx context.Context, input agents.ExperimentPlannerInput) []memory.MemoryRetrievalResult {
	if !memoryRetrievalEnabled() {
		return nil
	}
	query := memory.MemoryRetrievalQuery{
		ProjectID:      input.Project.ID,
		DatasetID:      input.Dataset.ID,
		AgentName:      agents.ExperimentPlannerAgentName,
		Purpose:        "experiment_planner",
		Text:           plannerMemoryRetrievalText(input),
		Kinds:          plannerMemoryRetrievalKinds(),
		Mechanisms:     plannerMemoryRetrievalMechanisms(input),
		DatasetTraits:  uniqueStrings(append(input.DatasetInsights.DatasetTraits, datasetProfileTraits(input.Dataset.Profile)...)),
		Objective:      strings.Join([]string{input.ObjectiveContext.PrimaryObjective, input.ObjectiveContext.GoalText, input.SourcePlan.TargetMetric}, " "),
		Limit:          memoryRetrievalMaxCards(),
		CrossProjectOK: memoryRetrievalCrossProjectOK(),
	}
	results := s.searchRetrievedMemory(ctx, query)
	s.logMemoryRetrieval("planner", input.Project.ID, input.Dataset.ID, results)
	if memoryRetrievalLogOnly() {
		return nil
	}
	return results
}

func (s *Server) retrieveTrainingMonitorMemory(ctx context.Context, plan plans.ExperimentPlan, job jobs.ExperimentJob, summary runs.TrainingRunSummary, objective agents.ProjectObjectiveContext) []memory.MemoryRetrievalResult {
	if !memoryRetrievalEnabled() {
		return nil
	}
	mechanism := strings.TrimSpace(jobConfigString(job.Config, "mechanism"))
	if mechanism == "" {
		mechanism = strings.TrimSpace(jobConfigString(job.Config, "intervention"))
	}
	query := memory.MemoryRetrievalQuery{
		ProjectID:      job.ProjectID,
		DatasetID:      summary.DatasetID,
		AgentName:      agents.TrainingMonitorAgentName,
		Purpose:        "training_monitor",
		Text:           trainingMonitorMemoryRetrievalText(plan, job, summary, objective),
		Kinds:          []string{memory.KindTrainingEvaluation, memory.KindPlanningOutcome, "strategy_scorecard"},
		Mechanisms:     uniqueStrings([]string{mechanism}),
		Objective:      strings.Join([]string{objective.PrimaryObjective, objective.GoalText, plan.TargetMetric}, " "),
		Limit:          minInt(memoryRetrievalMaxCards(), trainingMonitorMemoryRetrievalMaxCards()),
		CrossProjectOK: memoryRetrievalCrossProjectOK(),
	}
	results := s.searchRetrievedMemory(ctx, query)
	s.logMemoryRetrieval("training_monitor", job.ProjectID, summary.DatasetID, results)
	if memoryRetrievalLogOnly() {
		return nil
	}
	return results
}

func (s *Server) searchRetrievedMemory(ctx context.Context, query memory.MemoryRetrievalQuery) []memory.MemoryRetrievalResult {
	query.Text = strings.TrimSpace(query.Text)
	if query.ProjectID == "" || query.Text == "" {
		return nil
	}
	if query.Limit <= 0 {
		query.Limit = memoryRetrievalMaxCards()
	}
	query = s.withMemoryQueryEmbedding(ctx, query)
	results, err := s.store.SearchMemoryEmbeddings(query)
	if err != nil {
		log.Printf("memory retrieval failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, err)
		return nil
	}
	return filterMemoryRetrievalResults(results, memoryRetrievalMinScore(), query.Limit)
}

func (s *Server) withMemoryQueryEmbedding(ctx context.Context, query memory.MemoryRetrievalQuery) memory.MemoryRetrievalQuery {
	config := embeddings.ConfigFromEnv()
	if !config.EmbeddingsEnabled || config.ReadyForIndexing() != nil {
		return query
	}
	result, err := embeddings.NewClient(config).Embed(ctx, embeddings.EmbedRequest{
		Model:      config.Model,
		Text:       query.Text,
		Dimensions: config.Dimensions,
	})
	if err != nil {
		log.Printf("memory retrieval embedding failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, err)
		return query
	}
	query.EmbeddingModel = result.Model
	query.EmbeddingDimensions = result.Dimensions
	query.Embedding = result.Vector
	return query
}

func (s *Server) logMemoryRetrieval(purpose string, projectID string, datasetID string, results []memory.MemoryRetrievalResult) {
	if len(results) == 0 {
		return
	}
	diagnostics.Event("info", "memory_retrieval", map[string]any{
		"purpose":          purpose,
		"project_id":       projectID,
		"dataset_id":       datasetID,
		"retrieved_count":  len(results),
		"log_only":         memoryRetrievalLogOnly(),
		"cross_project_ok": memoryRetrievalCrossProjectOK(),
	})
}

func (s *Server) indexMemoryCard(ctx context.Context, card memory.EmbeddableMemoryCard) {
	if !embeddings.ConfigFromEnv().EmbeddingsEnabled {
		return
	}
	result, err := memory.NewIndexer(s.store, nil, embeddings.ConfigFromEnv()).IndexCards(ctx, []memory.EmbeddableMemoryCard{card})
	if err != nil {
		log.Printf("memory card indexing failed for %s/%s: %v", card.SourceTable, card.SourceID, err)
		return
	}
	if result.Disabled && result.NoopReason != "" {
		log.Printf("memory card indexing skipped for %s/%s: %s", card.SourceTable, card.SourceID, result.NoopReason)
	}
}

func plannerMemoryRetrievalText(input agents.ExperimentPlannerInput) string {
	return strings.Join(uniqueStrings([]string{
		input.Project.Goal,
		input.Dataset.Name,
		input.DatasetInsights.Summary,
		input.DatasetInsights.TaskType,
		strings.Join(input.DatasetInsights.DatasetTraits, " "),
		input.SourcePlan.TargetMetric,
		input.ObjectiveContext.PrimaryObjective,
		input.ObjectiveContext.GoalText,
		strings.Join(input.DeterministicDiagnosis.RecommendedFailureModes, " "),
		strings.Join(input.DeterministicDiagnosis.Evidence, " "),
		plannerChampionMemoryText(input.CurrentChampion),
	}), "\n")
}

func trainingMonitorMemoryRetrievalText(plan plans.ExperimentPlan, job jobs.ExperimentJob, summary runs.TrainingRunSummary, objective agents.ProjectObjectiveContext) string {
	return strings.Join(uniqueStrings([]string{
		summary.Model,
		memoryModelFamily(summary.Model),
		summary.Status,
		job.Template,
		jobConfigString(job.Config, "mechanism"),
		jobConfigString(job.Config, "intervention"),
		plan.TargetMetric,
		objective.PrimaryObjective,
		objective.GoalText,
		fmt.Sprintf("best_macro_f1 %.4f best_accuracy %.4f final_train_loss %.4f final_val_loss %.4f epochs %d", summary.BestMacroF1, summary.BestAccuracy, summary.FinalTrainLoss, summary.FinalValLoss, summary.EpochsCompleted),
	}), "\n")
}

func plannerMemoryRetrievalKinds() []string {
	return []string{
		"strategy_scorecard",
		memory.KindPlanningOutcome,
		memory.KindPlanningFeedback,
		memory.KindTrainingEvaluation,
		memory.KindDatasetProfile,
		memory.KindDatasetVisualAnalysis,
		memory.KindDatasetPreprocessingHypothesis,
	}
}

func plannerMemoryRetrievalMechanisms(input agents.ExperimentPlannerInput) []string {
	values := append([]string{}, input.DeterministicDiagnosis.RecommendedFailureModes...)
	for _, scorecard := range input.StrategyScorecards {
		values = append(values, scorecard.Mechanism)
	}
	return uniqueStrings(values)
}

func plannerChampionMemoryText(champion *agents.ExperimentChampion) string {
	if champion == nil {
		return ""
	}
	return strings.Join(uniqueStrings([]string{champion.Model, champion.TargetMetric, fmt.Sprintf("score %.4f", champion.Score)}), " ")
}

func filterMemoryRetrievalResults(results []memory.MemoryRetrievalResult, minScore float64, limit int) []memory.MemoryRetrievalResult {
	out := []memory.MemoryRetrievalResult{}
	for _, result := range results {
		if result.Score < minScore {
			continue
		}
		out = append(out, result)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func memoryRetrievalEnabled() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", false)
}

func memoryRetrievalCrossProjectOK() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_CROSS_PROJECT_ENABLED", false)
}

func memoryRetrievalLogOnly() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", true)
}

func memoryRetrievalMaxCards() int {
	return envInt("MODEL_EXPRESS_MEMORY_RETRIEVAL_MAX_CARDS", memoryRetrievalDefaultMaxCards, 1, 50)
}

func trainingMonitorMemoryRetrievalMaxCards() int {
	return minInt(memoryRetrievalMaxCards(), 8)
}

func memoryRetrievalMinScore() float64 {
	return envFloat("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", memoryRetrievalDefaultMinScore, 0, 1)
}

func memoryModelFamily(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.Contains(normalized, "mobilenet"):
		return "mobilenet"
	case strings.Contains(normalized, "efficientnet"):
		return "efficientnet"
	case strings.Contains(normalized, "resnet"):
		return "resnet"
	case strings.Contains(normalized, "regnet"):
		return "regnet"
	case strings.Contains(normalized, "convnext"):
		return "convnext"
	case strings.Contains(normalized, "swin"):
		return "swin"
	case strings.Contains(normalized, "vit"):
		return "vit"
	default:
		return normalized
	}
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func supportedModelCatalog() []agents.SupportedModelSpec {
	return []agents.SupportedModelSpec{
		{Name: "mobilenet_v3_small", Family: "mobilenet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 50, SupportsTransfer: true, ExpectedLatencyClass: "very_fast", RecommendedUse: "fast live baseline and compact champion refinement", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "mobilenet_v3_large", Family: "mobilenet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 80, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "higher-capacity MobileNet challenger for live use", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b0", Family: "efficientnet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 80, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "strong quality/latency baseline", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "regnet_y_400mf", Family: "regnet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 100, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "compact architecture challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b1", Family: "efficientnet", DeploymentTier: "balanced", DefaultImageSize: 240, MinRecommendedImages: 150, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "balanced quality challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b2", Family: "efficientnet", DeploymentTier: "balanced", DefaultImageSize: 260, MinRecommendedImages: 250, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "stronger quality challenger when budget allows", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "resnet18", Family: "resnet", DeploymentTier: "balanced", DefaultImageSize: 224, MinRecommendedImages: 100, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "stable control architecture", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "resnet34", Family: "resnet", DeploymentTier: "balanced", DefaultImageSize: 224, MinRecommendedImages: 150, SupportsTransfer: true, ExpectedLatencyClass: "medium_slow", RecommendedUse: "larger ResNet comparison", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "convnext_tiny", Family: "convnext", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 300, SupportsTransfer: true, ExpectedLatencyClass: "slow", RecommendedUse: "quality-first challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "swin_t", Family: "swin", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 500, SupportsTransfer: true, ExpectedLatencyClass: "slow", RecommendedUse: "transformer challenger for larger datasets", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "vit_b_16", Family: "vit", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 800, SupportsTransfer: true, ExpectedLatencyClass: "slowest", RecommendedUse: "explicit quality-first experiments on larger datasets", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
	}
}

func supportedModelNames() map[string]bool {
	out := map[string]bool{}
	for _, spec := range supportedModelCatalog() {
		out[strings.ToLower(spec.Name)] = true
	}
	return out
}

func plannerStrategyMemory(records []memory.AgentMemoryRecord) ([]agents.PlannerStrategyMemory, []agents.PlannerStrategyMemory) {
	successful := []agents.PlannerStrategyMemory{}
	failed := []agents.PlannerStrategyMemory{}
	for _, record := range records {
		if record.Kind != memory.KindPlanningOutcome {
			continue
		}
		entry := plannerStrategyMemoryFromRecord(record)
		switch entry.OutcomeStatus {
		case agents.ExperimentPlanningOutcomeImprovedChampion, agents.ExperimentPlanningOutcomeMinorImprovement:
			if len(successful) < 6 {
				successful = append(successful, entry)
			}
		case agents.ExperimentPlanningOutcomeNoImprovement, agents.ExperimentPlanningOutcomeFailed:
			if len(failed) < 6 {
				failed = append(failed, entry)
			}
		}
	}
	return successful, failed
}

func plannerRejectedOptions(records []memory.AgentMemoryRecord, diagnosis agents.PlannerDiagnosis) []agents.RejectedPlannerOption {
	out := []agents.RejectedPlannerOption{}
	failureModes := map[string]bool{}
	for _, mode := range diagnosis.RecommendedFailureModes {
		failureModes[strings.ToLower(strings.TrimSpace(mode))] = true
	}
	for _, record := range records {
		if record.Kind != memory.KindPlanningFeedback && record.Kind != memory.KindPlanningOutcome {
			continue
		}
		value, ok := record.Payload["rejected_options"]
		if !ok {
			continue
		}
		blob, err := json.Marshal(value)
		if err != nil {
			continue
		}
		var options []agents.RejectedPlannerOption
		if err := json.Unmarshal(blob, &options); err != nil {
			continue
		}
		for _, option := range options {
			if !rejectedOptionApplies(option, failureModes) {
				continue
			}
			out = append(out, option)
			if len(out) >= 8 {
				return out
			}
		}
	}
	return out
}

func rejectedOptionApplies(option agents.RejectedPlannerOption, failureModes map[string]bool) bool {
	if len(option.AppliesWhen) == 0 || len(failureModes) == 0 {
		return true
	}
	for _, condition := range option.AppliesWhen {
		normalized := strings.ToLower(strings.TrimSpace(condition))
		if failureModes[normalized] {
			return true
		}
	}
	return false
}

func plannerStrategyScorecards(scorecards []strategies.StrategyScorecard, datasetID string) []agents.PlannerStrategyScorecard {
	out := []agents.PlannerStrategyScorecard{}
	for _, scorecard := range scorecards {
		if scorecard.DatasetID != datasetID && len(out) >= 6 {
			continue
		}
		out = append(out, agents.PlannerStrategyScorecard{
			ID:                scorecard.ID,
			DatasetID:         scorecard.DatasetID,
			SourceDecisionID:  scorecard.SourceDecisionID,
			SourcePlanID:      scorecard.SourcePlanID,
			FollowUpPlanID:    scorecard.FollowUpPlanID,
			StrategyType:      scorecard.StrategyType,
			PlanningMode:      scorecard.PlanningMode,
			Mechanism:         scorecard.Mechanism,
			Intervention:      scorecard.Intervention,
			DiagnosisTriggers: scorecard.DiagnosisTriggers,
			EvidenceUsed:      scorecard.EvidenceUsed,
			ExpectedEffect:    scorecard.ExpectedEffect,
			DatasetTraits:     scorecard.DatasetTraits,
			ObjectiveProfile:  scorecard.ObjectiveProfile,
			ProposedChanges:   scorecard.ProposedChanges,
			ExpectedDelta:     scorecard.ExpectedDelta,
			ActualDelta:       scorecard.ActualDelta,
			ConfidenceBefore:  scorecard.ConfidenceBefore,
			ConfidenceAfter:   scorecard.ConfidenceAfter,
			CostUSD:           scorecard.CostUSD,
			RuntimeSeconds:    scorecard.RuntimeSeconds,
			Outcome:           scorecard.Outcome,
			Lesson:            scorecard.Lesson,
			Tags:              scorecard.Tags,
		})
		if len(out) >= 10 {
			break
		}
	}
	return out
}

func plannerStrategyMemoryFromRecord(record memory.AgentMemoryRecord) agents.PlannerStrategyMemory {
	bestModel := ""
	if champion, ok := experimentChampionFromPayload(record.Payload["actual_best_run"]); ok {
		bestModel = champion.Model
	}
	return agents.PlannerStrategyMemory{
		MemoryID:                record.ID,
		OutcomeStatus:           payloadString(record.Payload, "outcome_status"),
		Lesson:                  payloadString(record.Payload, "lesson"),
		BestModel:               bestModel,
		ActualDeltaVsChampion:   payloadFloat(record.Payload, "actual_delta_vs_champion"),
		ExpectedDeltaVsChampion: payloadFloat(record.Payload, "expected_delta_vs_champion"),
		TotalCostUSD:            payloadFloat(record.Payload, "total_cost_usd"),
		TotalRuntimeSeconds:     payloadFloat(record.Payload, "total_runtime_seconds"),
		ProposedModels:          proposedModelsFromPayload(record.Payload),
		Tags:                    record.Tags,
	}
}

func proposedModelsFromPayload(payload map[string]any) []string {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return []string{}
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return []string{}
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return []string{}
	}
	models := []string{}
	for _, experiment := range experiments {
		if strings.TrimSpace(experiment.Model) != "" {
			models = append(models, experiment.Model)
		}
	}
	return uniqueStrings(models)
}

func agentDecisionByID(agentDecisions []decisions.AgentDecision, decisionID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.ID == decisionID {
			return decision, true
		}
	}
	return decisions.AgentDecision{}, false
}

func experimentPlanningOutcomeForPlan(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) (agents.ExperimentPlanningOutcome, error) {
	planSummaries := summariesForPlanID(summaries, followUpPlan.ID)
	proposedExperiments, err := plannedExperimentsFromPayload(sourceDecision.Payload)
	if err != nil {
		proposedExperiments = []plans.PlannedExperiment{}
	}

	baselineChampion := baselineChampionForPlannerOutcome(sourceDecision, followUpPlan, projectPlans, summaries)
	bestSummary, hasBest := bestSuccessfulTrainingSummary(followUpPlan.TargetMetric, planSummaries)

	var actualBest *agents.ExperimentChampion
	actualDelta := 0.0
	if hasBest {
		best := experimentChampionFromSummary(followUpPlan.TargetMetric, bestSummary)
		actualBest = &best
		if baselineChampion != nil {
			actualDelta = best.Score - baselineChampion.Score
		} else {
			actualDelta = best.Score
		}
	}

	expectedDelta := payloadFloat(sourceDecision.Payload, "expected_delta_vs_champion")
	metExpectedDelta := hasBest && actualDelta > plannerMinimumMeaningfulImprovement
	if expectedDelta > 0 {
		metExpectedDelta = hasBest && actualDelta >= expectedDelta
	}
	outcomeStatus := plannerOutcomeStatus(actualDelta, hasBest)
	outcome := agents.ExperimentPlanningOutcome{
		OutcomeType:             "planner_followup_result",
		OutcomeStatus:           outcomeStatus,
		SourceDecisionID:        sourceDecision.ID,
		SourcePlanID:            sourceDecision.PlanID,
		FollowUpPlanID:          followUpPlan.ID,
		BaselineChampion:        baselineChampion,
		ActualBestRun:           actualBest,
		ExpectedDeltaVsChampion: expectedDelta,
		ActualDeltaVsChampion:   actualDelta,
		MetExpectedDelta:        metExpectedDelta,
		TotalCostUSD:            totalSummaryCost(planSummaries),
		TotalRuntimeSeconds:     totalSummaryRuntime(planSummaries),
		TerminalRunCount:        len(planSummaries),
		SuccessfulRunCount:      successfulSummaryCount(planSummaries),
		FailedRunCount:          failedSummaryCount(planSummaries),
		ProposedExperiments:     proposedExperiments,
		CompletedAt:             time.Now().UTC(),
	}
	outcome.Lesson = plannerOutcomeLesson(followUpPlan.TargetMetric, outcome)
	return outcome, nil
}

func baselineChampionForPlannerOutcome(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) *agents.ExperimentChampion {
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["current_champion"]); ok {
		return champion
	}
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["source_plan_baseline_champion"]); ok {
		return champion
	}
	if summary, ok := bestSuccessfulTrainingSummaryBeforePlan(followUpPlan.TargetMetric, projectPlans, summaries, followUpPlan.ID); ok {
		champion := experimentChampionFromSummary(followUpPlan.TargetMetric, summary)
		return &champion
	}
	return nil
}

func experimentChampionFromPayload(value any) (*agents.ExperimentChampion, bool) {
	if value == nil {
		return nil, false
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var champion agents.ExperimentChampion
	if err := json.Unmarshal(blob, &champion); err != nil {
		return nil, false
	}
	if champion.JobID == "" {
		return nil, false
	}
	return &champion, true
}

func plannerOutcomeStatus(actualDelta float64, hasBest bool) string {
	if !hasBest {
		return agents.ExperimentPlanningOutcomeFailed
	}
	if actualDelta > plannerMinimumMeaningfulImprovement {
		return agents.ExperimentPlanningOutcomeImprovedChampion
	}
	if actualDelta > 0 {
		return agents.ExperimentPlanningOutcomeMinorImprovement
	}
	return agents.ExperimentPlanningOutcomeNoImprovement
}

func plannerOutcomeConfidence(outcome agents.ExperimentPlanningOutcome) float64 {
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return minFloat(0.95, 0.70+maxFloat(0, outcome.ActualDeltaVsChampion)*4)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return 0.55
	case agents.ExperimentPlanningOutcomeFailed:
		return 0.20
	default:
		return 0.35
	}
}

func plannerOutcomeLesson(targetMetric string, outcome agents.ExperimentPlanningOutcome) string {
	metric := normalizedPlannerTargetMetric(targetMetric)
	if outcome.OutcomeStatus == agents.ExperimentPlanningOutcomeFailed {
		return fmt.Sprintf("Planner follow-up plan %s produced no successful runs after %.3f total cost; avoid repeating this failed strategy without changing the setup.", outcome.FollowUpPlanID, outcome.TotalCostUSD)
	}
	bestModel := ""
	if outcome.ActualBestRun != nil {
		bestModel = outcome.ActualBestRun.Model
	}
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return fmt.Sprintf("Planner follow-up plan %s improved the champion with %s by %.3f %s; similar strategy changes are worth reusing.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return fmt.Sprintf("Planner follow-up plan %s only slightly improved the champion with %s by %.3f %s, below the meaningful threshold %.3f; treat this as weak evidence.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, plannerMinimumMeaningfulImprovement)
	default:
		return fmt.Sprintf("Planner follow-up plan %s failed to beat the prior champion; best run %s trailed by %.3f %s after %.3f total cost.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, outcome.TotalCostUSD)
	}
}

func plannerOutcomeTags(outcome agents.ExperimentPlanningOutcome) []string {
	tags := []string{"planner_outcome", outcome.OutcomeStatus}
	if outcome.MetExpectedDelta {
		tags = append(tags, "met_expected_delta")
	} else {
		tags = append(tags, "missed_expected_delta")
	}
	if outcome.ActualBestRun != nil && outcome.ActualBestRun.Model != "" {
		tags = append(tags, strings.ToLower(strings.TrimSpace(outcome.ActualBestRun.Model)))
	}
	return tags
}

func totalSummaryCost(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.EstimatedCostUSD
	}
	return total
}

func totalSummaryRuntime(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.RuntimeSeconds
	}
	return total
}

func successfulSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusSucceeded {
			count++
		}
	}
	return count
}

func failedSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusFailed {
			count++
		}
	}
	return count
}

func payloadString(payload map[string]any, key string) string {
	value, _ := payload[key].(string)
	return value
}

func payloadStringSlice(payload map[string]any, key string) []string {
	value, ok := payload[key]
	if !ok || value == nil {
		return nil
	}
	if typed, ok := value.([]string); ok {
		return append([]string(nil), typed...)
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	out := []string{}
	if err := json.Unmarshal(blob, &out); err == nil {
		return out
	}
	values, ok := value.([]any)
	if !ok {
		return nil
	}
	for _, item := range values {
		if text, ok := item.(string); ok {
			out = append(out, text)
		}
	}
	return out
}

func nonEmptyStringValues(values []string) []string {
	out := []string{}
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, value)
		}
	}
	return out
}

func payloadMap(payload map[string]any, key string) map[string]any {
	value, ok := payload[key]
	if !ok || value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func mapFromAny(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func payloadFloat(payload map[string]any, key string) float64 {
	switch value := payload[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func payloadBool(payload map[string]any, key string) bool {
	switch value := payload[key].(type) {
	case bool:
		return value
	case string:
		parsed, ok := envFlagValueFromString(value)
		return ok && parsed
	default:
		return false
	}
}

func maxFloat(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func minFloat(left float64, right float64) float64 {
	if left < right {
		return left
	}
	return right
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func profileString(profile map[string]any, key string) string {
	value, _ := profile[key].(string)
	return value
}

func profileInt(profile map[string]any, key string) int {
	switch value := profile[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		out, _ := value.Int64()
		return int(out)
	default:
		return 0
	}
}

func profileFloat(profile map[string]any, key string) float64 {
	switch value := profile[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func profileBool(profile map[string]any, key string) bool {
	switch value := profile[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func profileMap(profile map[string]any, key string) map[string]any {
	value, ok := profile[key].(map[string]any)
	if ok {
		return value
	}
	return map[string]any{}
}

func profileStringSlice(profile map[string]any, key string) []string {
	values, ok := profile[key].([]string)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []string{}
	}
	out := []string{}
	for _, raw := range rawValues {
		if value, ok := raw.(string); ok {
			out = append(out, value)
		}
	}
	return out
}

func profileMapSlice(profile map[string]any, key string) []map[string]any {
	values, ok := profile[key].([]map[string]any)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []map[string]any{}
	}
	out := []map[string]any{}
	for _, raw := range rawValues {
		if value, ok := raw.(map[string]any); ok {
			out = append(out, value)
		}
	}
	return out
}

func metadataBool(metadata map[string]any, key string) bool {
	switch value := metadata[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func containsString(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func jobsForPlan(projectJobs []jobs.ExperimentJob, planID string) []jobs.ExperimentJob {
	out := []jobs.ExperimentJob{}
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == planID {
			out = append(out, job)
		}
	}
	return out
}

func summariesForPlanID(summaries []runs.TrainingRunSummary, planID string) []runs.TrainingRunSummary {
	out := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == planID {
			out = append(out, summary)
		}
	}
	return out
}

func evaluationsForPlanID(evaluations []runs.TrainingRunEvaluation, planID string) []runs.TrainingRunEvaluation {
	out := []runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		if evaluation.PlanID == planID {
			out = append(out, evaluation)
		}
	}
	return out
}

func experimentPlannerPerformanceContext(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	sourcePlanID string,
) (*agents.ExperimentChampion, *agents.ExperimentChampion, []agents.ExperimentRunDelta, int, []string) {
	championSummary, hasChampion := bestSuccessfulTrainingSummaryForObjective(targetMetric, summaries, evaluations, objectiveContext)
	if !hasChampion {
		return nil, nil, []agents.ExperimentRunDelta{}, 0, []string{"No successful champion run is available yet."}
	}

	champion := experimentChampionFromSummary(targetMetric, championSummary)
	baselineChampion := champion
	if baselineSummary, ok := bestSuccessfulTrainingSummaryBeforePlan(targetMetric, projectPlans, summaries, sourcePlanID); ok {
		baselineChampion = experimentChampionFromSummary(targetMetric, baselineSummary)
	}
	sourcePlanDeltas := experimentRunDeltasForPlan(targetMetric, summariesForPlanID(summaries, sourcePlanID), baselineChampion)
	noImprovementRounds := consecutiveNoImprovementFollowUpRounds(targetMetric, projectPlans, summaries)
	stopSignals := experimentPlannerStopSignals(champion, noImprovementRounds)
	return &champion, &baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals
}

func experimentChampionFromSummary(targetMetric string, summary runs.TrainingRunSummary) agents.ExperimentChampion {
	return agents.ExperimentChampion{
		JobID:            summary.JobID,
		PlanID:           summary.PlanID,
		Model:            summary.Model,
		TargetMetric:     normalizedPlannerTargetMetric(targetMetric),
		Score:            plannerTargetMetricValue(targetMetric, summary),
		BestMacroF1:      summary.BestMacroF1,
		BestAccuracy:     summary.BestAccuracy,
		EstimatedCostUSD: summary.EstimatedCostUSD,
		RuntimeSeconds:   summary.RuntimeSeconds,
		EpochsCompleted:  summary.EpochsCompleted,
	}
}

func experimentRunDeltasForPlan(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	champion agents.ExperimentChampion,
) []agents.ExperimentRunDelta {
	out := make([]agents.ExperimentRunDelta, 0, len(summaries))
	for _, summary := range summaries {
		score := plannerTargetMetricValue(targetMetric, summary)
		out = append(out, agents.ExperimentRunDelta{
			JobID:                    summary.JobID,
			PlanID:                   summary.PlanID,
			Model:                    summary.Model,
			Status:                   summary.Status,
			TargetMetric:             normalizedPlannerTargetMetric(targetMetric),
			Score:                    score,
			BestMacroF1:              summary.BestMacroF1,
			BestAccuracy:             summary.BestAccuracy,
			EstimatedCostUSD:         summary.EstimatedCostUSD,
			RuntimeSeconds:           summary.RuntimeSeconds,
			EpochsCompleted:          summary.EpochsCompleted,
			ChampionJobID:            champion.JobID,
			DeltaScoreVsChampion:     score - champion.Score,
			DeltaCostVsChampion:      summary.EstimatedCostUSD - champion.EstimatedCostUSD,
			DeltaRuntimeVsChampion:   summary.RuntimeSeconds - champion.RuntimeSeconds,
			MeaningfullyImprovedOver: score > champion.Score+plannerMinimumMeaningfulImprovement,
		})
	}
	return out
}

func consecutiveNoImprovementFollowUpRounds(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) int {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	hasChampion := false
	championScore := 0.0
	noImprovementRounds := 0
	for _, plan := range orderedPlans {
		planSummaries := summariesForPlanID(summaries, plan.ID)
		if !planTrainingRunsComplete(plan, planSummaries) {
			continue
		}

		best, ok := bestSuccessfulTrainingSummary(targetMetric, planSummaries)
		if !ok {
			if plan.SourceDecisionID != "" && hasChampion {
				noImprovementRounds++
			}
			continue
		}

		score := plannerTargetMetricValue(targetMetric, best)
		if !hasChampion {
			hasChampion = true
			championScore = score
			continue
		}

		if plan.SourceDecisionID != "" {
			if score > championScore+plannerMinimumMeaningfulImprovement {
				noImprovementRounds = 0
			} else {
				noImprovementRounds++
			}
		}
		if score > championScore {
			championScore = score
		}
	}
	return noImprovementRounds
}

func experimentPlannerStopSignals(champion agents.ExperimentChampion, noImprovementRounds int) []string {
	signals := []string{
		fmt.Sprintf("Current champion is %s (%s) with %s %.3f.", champion.JobID, champion.Model, champion.TargetMetric, champion.Score),
	}
	if reason, ok := nearMetricCeilingChampionStopReason(agents.ExperimentPlannerInput{
		CurrentChampion:              &champion,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	}); ok {
		signals = append(signals, reason)
	}
	if noImprovementRounds > 0 {
		signals = append(signals, fmt.Sprintf("%d consecutive completed follow-up plan(s) did not improve the champion by at least %.3f.", noImprovementRounds, plannerMinimumMeaningfulImprovement))
	}
	if noImprovementRounds >= plannerNoImprovementRoundsToSelect {
		if terminalPlannerGuardsEnabled() {
			signals = append(signals, "Backend policy will select the current champion instead of scheduling another follow-up unless a future run meaningfully improves it.")
		} else {
			signals = append(signals, "No-improvement rounds are advisory only; continue by pivoting to a substantive backend-valid mechanism rather than stopping solely because the current champion remains ahead.")
		}
	}
	return signals
}

func bestSuccessfulTrainingSummary(targetMetric string, summaries []runs.TrainingRunSummary) (runs.TrainingRunSummary, bool) {
	var best runs.TrainingRunSummary
	hasBest := false
	bestScore := 0.0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := plannerTargetMetricValue(targetMetric, summary)
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func bestSuccessfulTrainingSummaryForObjective(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
) (runs.TrainingRunSummary, bool) {
	if len(evaluations) == 0 {
		return bestSuccessfulTrainingSummary(targetMetric, summaries)
	}
	evaluationsByJob := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		evaluationsByJob[evaluation.JobID] = evaluation
	}

	var best runs.TrainingRunSummary
	bestScore := 0.0
	hasBest := false
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := holisticRunScore(targetMetric, summary, evaluationsByJob[summary.JobID], objectiveContext)
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func holisticRunScore(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation, objectiveContext agents.ProjectObjectiveContext) float64 {
	quality := plannerTargetMetricValue(targetMetric, summary)
	if overall := payloadFloat(evaluation.HolisticScores, "overall_score"); overall > 0 {
		quality = maxFloat(quality, overall)
	}
	latencyScore := 0.5
	if latencyMS := payloadFloat(evaluation.ModelProfile, "estimated_latency_ms"); latencyMS > 0 {
		latencyScore = maxFloat(0, minFloat(1, 1-latencyMS/160))
	}
	costScore := maxFloat(0, minFloat(1, 1-summary.EstimatedCostUSD/10))
	runtimeScore := maxFloat(0, minFloat(1, 1-summary.RuntimeSeconds/1800))

	if objectiveContext.PrimaryObjective == "low_latency_live_service" {
		return quality*0.76 + latencyScore*0.10 + costScore*0.08 + runtimeScore*0.06
	}
	if objectiveContext.PrimaryObjective == "budget_sensitive" {
		return quality*0.68 + costScore*0.18 + latencyScore*0.08 + runtimeScore*0.06
	}
	if objectiveContext.PrimaryObjective == "quality_first" {
		return quality*0.82 + latencyScore*0.07 + costScore*0.06 + runtimeScore*0.05
	}
	return quality*0.74 + latencyScore*0.12 + costScore*0.08 + runtimeScore*0.06
}

func bestSuccessfulTrainingSummaryBeforePlan(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	sourcePlanID string,
) (runs.TrainingRunSummary, bool) {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	priorPlanIDs := map[string]bool{}
	for _, plan := range orderedPlans {
		if plan.ID == sourcePlanID {
			break
		}
		priorPlanIDs[plan.ID] = true
	}

	priorSummaries := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if priorPlanIDs[summary.PlanID] {
			priorSummaries = append(priorSummaries, summary)
		}
	}
	return bestSuccessfulTrainingSummary(targetMetric, priorSummaries)
}

func plannerTargetMetricValue(targetMetric string, summary runs.TrainingRunSummary) float64 {
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return summary.BestAccuracy
	default:
		return summary.BestMacroF1
	}
}

func normalizedPlannerTargetMetric(targetMetric string) string {
	normalized := strings.ToLower(strings.TrimSpace(targetMetric))
	if normalized == "" {
		return "macro_f1"
	}
	return normalized
}

func experimentSignaturesForPlans(projectPlans []plans.ExperimentPlan) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			signature := experimentSignature(experiment)
			if seen[signature] {
				continue
			}
			seen[signature] = true
			out = append(out, signature)
		}
	}
	sort.Strings(out)
	return out
}

func validateNovelProposedExperiments(experiments []plans.PlannedExperiment, projectPlans []plans.ExperimentPlan) error {
	existing := map[string]bool{}
	existingMechanisms := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			existing[experimentSignature(experiment)] = true
			existingMechanisms[experimentMechanismSignature(experiment)] = true
		}
	}

	proposed := map[string]bool{}
	proposedMechanisms := map[string]bool{}
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return err
		}
		signature := experimentSignature(experiment)
		if existing[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates an existing experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		if proposed[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates another proposed experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		mechanismSignature := experimentMechanismSignature(experiment)
		if existingMechanisms[mechanismSignature] {
			return fmt.Errorf("%w: proposed experiment %d only changes minor tuning knobs for an already tested experiment mechanism", store.ErrInvalidRequest, index)
		}
		if proposedMechanisms[mechanismSignature] {
			return fmt.Errorf("%w: proposed experiment %d only changes minor tuning knobs relative to another proposed experiment", store.ErrInvalidRequest, index)
		}
		proposed[signature] = true
		proposedMechanisms[mechanismSignature] = true
	}
	return nil
}

func filterNovelPlannedExperiments(experiments []plans.PlannedExperiment, projectPlans []plans.ExperimentPlan) ([]plans.PlannedExperiment, []string) {
	existing := map[string]bool{}
	existingMechanisms := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			existing[experimentSignature(experiment)] = true
			existingMechanisms[experimentMechanismSignature(experiment)] = true
		}
	}

	out := []plans.PlannedExperiment{}
	warnings := []string{}
	proposed := map[string]bool{}
	proposedMechanisms := map[string]bool{}
	for index, experiment := range experiments {
		signature := experimentSignature(experiment)
		mechanismSignature := experimentMechanismSignature(experiment)
		switch {
		case existing[signature] || proposed[signature]:
			warnings = append(warnings, fmt.Sprintf("Skipped follow-up experiment %d because it duplicated an existing experiment signature.", index))
		case existingMechanisms[mechanismSignature] || proposedMechanisms[mechanismSignature]:
			warnings = append(warnings, fmt.Sprintf("Skipped follow-up experiment %d because it only changed minor tuning knobs for an already tested mechanism.", index))
		default:
			out = append(out, experiment)
			proposed[signature] = true
			proposedMechanisms[mechanismSignature] = true
		}
	}
	return out, warnings
}

func experimentSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	augmentationPolicyConfigBlob, _ := json.Marshal(experiment.AugmentationPolicyConfig)
	classBalancingConfigBlob, _ := json.Marshal(experiment.ClassBalancingConfig)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	automlBlob, _ := json.Marshal(experiment.AutoML)
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.Epochs),
		strconv.Itoa(experiment.BatchSize),
		strconv.FormatFloat(experiment.LearningRate, 'g', -1, 64),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		strconv.FormatFloat(experiment.Dropout, 'g', -1, 64),
		strconv.FormatFloat(experiment.OptimizerMomentum, 'g', -1, 64),
		strconv.Itoa(experiment.SchedulerStepSize),
		strconv.FormatFloat(experiment.SchedulerGamma, 'g', -1, 64),
		strconv.FormatFloat(experiment.LabelSmoothing, 'g', -1, 64),
		strconv.FormatFloat(experiment.GradientClipNorm, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		string(augmentationPolicyConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		string(classBalancingConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.Itoa(experiment.EarlyStoppingPatience),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
		string(automlBlob),
	}, ":")
}

func experimentMechanismSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	augmentationPolicyConfigBlob, _ := json.Marshal(experiment.AugmentationPolicyConfig)
	classBalancingConfigBlob, _ := json.Marshal(experiment.ClassBalancingConfig)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	automlSearchBlob, _ := json.Marshal(nil)
	if experiment.AutoML != nil {
		automlSearchBlob, _ = json.Marshal(experiment.AutoML.SearchSpace)
	}
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		strconv.FormatFloat(experiment.Dropout, 'g', -1, 64),
		strconv.FormatFloat(experiment.OptimizerMomentum, 'g', -1, 64),
		strconv.Itoa(experiment.SchedulerStepSize),
		strconv.FormatFloat(experiment.SchedulerGamma, 'g', -1, 64),
		strconv.FormatFloat(experiment.LabelSmoothing, 'g', -1, 64),
		strconv.FormatFloat(experiment.GradientClipNorm, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		string(augmentationPolicyConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		string(classBalancingConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
		string(automlSearchBlob),
	}, ":")
}

func (s *Server) listWorkers(c *gin.Context) {
	workers, err := s.store.ListWorkers()
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"workers": workers})
}

func (s *Server) listProjectWorkers(c *gin.Context) {
	workers, err := s.store.ListProjectWorkers(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"workers": workers})
}

func (s *Server) getWorker(c *gin.Context) {
	worker, err := s.store.GetWorker(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, worker)
}

func (s *Server) registerWorker(c *gin.Context) {
	var req registerWorkerRequest
	if !bindJSON(c, &req) {
		return
	}

	worker, err := s.store.RegisterWorker(req.ProjectID, req.Name, req.GPUType)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, worker)
}

func (s *Server) heartbeatWorker(c *gin.Context) {
	worker, err := s.store.HeartbeatWorker(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, worker)
}

func (s *Server) pollJob(c *gin.Context) {
	var req pollJobRequest
	if c.Request.ContentLength > 0 {
		if !bindJSON(c, &req) {
			return
		}
	}
	job, err := s.store.PollJob(c.Param("id"), store.JobPollFilter{
		Provider:                            req.Provider,
		Templates:                           req.Templates,
		IncludeUnspecifiedProviderTemplates: req.IncludeUnspecifiedProviderTemplates,
	})
	if err == nil {
		c.JSON(http.StatusOK, pollJobResponse{Job: job})
		return
	}

	if errors.Is(err, store.ErrNoJob) {
		c.JSON(http.StatusOK, pollJobResponse{Job: nil})
		return
	}

	writeStoreError(c, err)
}

func (s *Server) createExperimentPlan(c *gin.Context) {
	var req createExperimentPlanRequest
	if !bindJSON(c, &req) {
		return
	}

	targetMetric := req.TargetMetric
	recommendedWorkers := req.RecommendedWorkers
	estimatedMinutes := req.EstimatedMinutes
	experiments := req.Experiments
	warnings := req.Warnings

	if len(experiments) == 0 {
		project, err := s.store.GetProject(c.Param("id"))
		if err != nil {
			writeStoreError(c, err)
			return
		}

		dataset, err := s.store.GetDataset(req.DatasetID)
		if err != nil {
			writeStoreError(c, err)
			return
		}

		recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
			Priority:          req.Priority,
			MaxWorkers:        req.MaxWorkers,
			TimeBudgetMinutes: req.TimeBudgetMinutes,
			TargetMetric:      req.TargetMetric,
		})
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		targetMetric = recommendation.TargetMetric
		recommendedWorkers = recommendation.RecommendedWorkers
		estimatedMinutes = recommendation.EstimatedMinutes
		experiments = recommendation.Experiments
		warnings = append(warnings, recommendation.Warnings...)
	}
	var automlWarnings []string
	var err error
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProject(c.Param("id"), experiments)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	warnings = append(warnings, automlWarnings...)
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			writeStoreError(c, err)
			return
		}
	}

	plan, err := s.store.CreateExperimentPlan(
		c.Param("id"),
		req.DatasetID,
		targetMetric,
		recommendedWorkers,
		estimatedMinutes,
		experiments,
		warnings,
		"",
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if err := s.persistAutoMLForPlan(plan); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, plan)
}

func (s *Server) listProjectPlans(c *gin.Context) {
	plans, err := s.store.ListProjectExperimentPlans(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"plans": plans})
}

func (s *Server) listExperimentPlans(c *gin.Context) {
	plans, err := s.store.GetExperimentPlan(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, plans)
}

func (s *Server) executeExperimentPlan(c *gin.Context) {
	var req executeExperimentPlanRequest
	if !bindJSON(c, &req) {
		return
	}

	response, err := s.executeStoredExperimentPlan(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, response)
}

func (s *Server) createInitialPlanForDataset(datasetID string) error {
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return err
	}

	existingPlans, err := s.store.ListProjectExperimentPlans(dataset.ProjectID)
	if err != nil {
		return err
	}
	for _, plan := range existingPlans {
		if plan.DatasetID == dataset.ID {
			return nil
		}
	}

	project, err := s.store.GetProject(dataset.ProjectID)
	if err != nil {
		return err
	}

	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return err
	}
	if len(metadataSummary) > 0 {
		dataset.Profile = profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary)
	}
	recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
		Priority: agents.PriorityBalanced,
	})
	if err != nil {
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}
	experiments, automlWarnings, err := s.prepareAutoMLExperimentsForProject(project.ID, recommendation.Experiments)
	if err != nil {
		return err
	}
	warnings := append([]string(nil), recommendation.Warnings...)
	warnings = append(warnings, automlWarnings...)

	plan, err := s.store.CreateExperimentPlan(
		project.ID,
		dataset.ID,
		recommendation.TargetMetric,
		recommendation.RecommendedWorkers,
		recommendation.EstimatedMinutes,
		experiments,
		warnings,
		"",
	)
	if err != nil {
		return err
	}
	if err := s.persistAutoMLForPlan(plan); err != nil {
		return err
	}

	if s.shouldAutoExecuteExperimentPlans() {
		req := s.defaultExecuteExperimentPlanRequest()
		executionResult, err := s.executeStoredExperimentPlan(plan.ID, req)
		if err != nil {
			return err
		}
		return s.recordAutomaticExecutionQueued(plan, req, executionResult.Jobs)
	}

	return nil
}

func (s *Server) executeStoredExperimentPlan(planID string, req executeExperimentPlanRequest) (executeExperimentPlanResponse, error) {
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}

	if len(plan.Experiments) == 0 {
		return executeExperimentPlanResponse{}, fmt.Errorf("%w: plan has no experiments to execute", store.ErrInvalidRequest)
	}
	if err := s.validateFollowUpPlanCanExecute(plan); err != nil {
		return executeExperimentPlanResponse{}, err
	}

	provider := req.Provider
	if provider == "" {
		provider = "local"
	}
	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	materializationPolicy := datasetMaterializationPolicy(dataset, provider, s.targetWorkerCountForPlan(plan, len(plan.Experiments)))

	existingJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	automlSuggestions := s.automlSuggestionsByExperiment(plan.ID)

	jobsByExperiment := map[int]jobs.ExperimentJob{}
	for _, job := range existingJobs {
		if job.Template != jobs.TemplateTrainExperiment && job.Template != jobs.TemplateLabelQualityAudit {
			continue
		}
		if job.Status == jobs.StatusFailed {
			continue
		}
		if configString(job.Config, "plan_id") != plan.ID {
			continue
		}
		jobProvider := configString(job.Config, "provider")
		if jobProvider == "" {
			jobProvider = "local"
		}
		if jobProvider != provider {
			continue
		}

		index, ok := configInt(job.Config, "experiment_index")
		if !ok {
			continue
		}
		jobsByExperiment[index] = job
	}

	out := make([]jobs.ExperimentJob, 0, len(plan.Experiments))
	for index, experiment := range plan.Experiments {
		if job, ok := jobsByExperiment[index]; ok {
			out = append(out, job)
			continue
		}

		jobTemplate := experimentExecutionTemplate(experiment)
		config := map[string]any{
			"plan_id":             plan.ID,
			"dataset_id":          plan.DatasetID,
			"experiment_index":    index,
			"experiment_template": experiment.Template,
			"target_metric":       plan.TargetMetric,
			"provider":            provider,
			"gpu_type":            req.GPUType,
		}
		addDatasetMaterializationConfig(config, materializationPolicy)
		if jobTemplate == jobs.TemplateTrainExperiment {
			config["model"] = experiment.Model
			config["epochs"] = experiment.Epochs
			config["batch_size"] = experiment.BatchSize
			config["learning_rate"] = experiment.LearningRate
		} else if jobTemplate == jobs.TemplateLabelQualityAudit {
			config["audit_type"] = strings.ToLower(strings.TrimSpace(experiment.Mechanism))
			config["report_only"] = true
		}
		addOptionalExperimentConfig(config, experiment)
		if metadataImport, err := s.store.GetActiveDatasetMetadataImport(plan.DatasetID); err == nil {
			config["metadata_import_id"] = metadataImport.ID
			config["metadata_summary"] = metadataImport.AgentSafeSummary
		} else if err != nil && !errors.Is(err, store.ErrNotFound) {
			return executeExperimentPlanResponse{}, err
		}
		if suggestion, ok := automlSuggestions[index]; ok {
			config["automl_study_id"] = suggestion.StudyID
			config["automl_suggestion_id"] = suggestion.ID
			config["automl_summary"] = automlJobSummary(experiment, suggestion)
		}

		job, err := s.store.CreateJob(plan.ProjectID, jobTemplate, config)
		if err != nil {
			return executeExperimentPlanResponse{}, err
		}
		if suggestion, ok := automlSuggestions[index]; ok && suggestion.ID != "" {
			if _, err := s.store.UpdateOptimizerSuggestionJob(suggestion.ID, job.ID); err != nil {
				log.Printf("link AutoML suggestion %s to job %s failed: %v", suggestion.ID, job.ID, err)
			}
		}

		out = append(out, job)
	}

	return executeExperimentPlanResponse{
		Plan: plan,
		Jobs: out,
	}, nil
}

func (s *Server) validateFollowUpPlanCanExecute(plan plans.ExperimentPlan) error {
	if plan.SourceDecisionID == "" {
		return nil
	}
	if terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		if stopReason, stopDetails, ok, err := s.projectChampionSelectedFollowUpStopReason(plan.ProjectID); err != nil {
			return err
		} else if ok {
			message := fmt.Sprintf("Follow-up execution blocked for plan %s because the project already has a selected champion.", plan.ID)
			s.recordChampionSelectedFollowUpBlocked(plan.ProjectID, plan.ID, plan.SourceDecisionID, plan.ID, message, stopReason, stopDetails)
			return fmt.Errorf("%w: %s", errChampionSelectedFollowUpBlocked, stopReason)
		}
	}
	projectPlans, err := s.store.ListProjectExperimentPlans(plan.ProjectID)
	if err != nil {
		return err
	}
	return s.validateExistingFollowUpPlanStillNovel(plan.ProjectID, plan.SourceDecisionID, plan, projectPlans)
}

func experimentExecutionTemplate(experiment plans.PlannedExperiment) string {
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return jobs.TemplateLabelQualityAudit
	}
	return jobs.TemplateTrainExperiment
}

func addOptionalExperimentConfig(config map[string]any, experiment plans.PlannedExperiment) {
	if experiment.Mechanism != "" {
		config["mechanism"] = experiment.Mechanism
	}
	if experiment.Intervention != "" {
		config["intervention"] = experiment.Intervention
	}
	if len(experiment.EvidenceUsed) > 0 {
		config["evidence_used"] = experiment.EvidenceUsed
	}
	if experiment.ExpectedEffect != "" {
		config["expected_effect"] = experiment.ExpectedEffect
	}
	if experiment.ImageSize > 0 {
		config["image_size"] = experiment.ImageSize
	}
	if experiment.ResolutionStrategy != "" {
		config["resolution_strategy"] = experiment.ResolutionStrategy
	}
	if experiment.Preprocessing != nil {
		config["preprocessing"] = experiment.Preprocessing
	}
	if experiment.Optimizer != "" {
		config["optimizer"] = experiment.Optimizer
	}
	if experiment.Scheduler != "" {
		config["scheduler"] = experiment.Scheduler
	}
	if experiment.WeightDecay > 0 {
		config["weight_decay"] = experiment.WeightDecay
	}
	if experiment.Dropout > 0 {
		config["dropout"] = experiment.Dropout
	}
	if experiment.OptimizerMomentum > 0 {
		config["optimizer_momentum"] = experiment.OptimizerMomentum
	}
	if experiment.SchedulerStepSize > 0 {
		config["scheduler_step_size"] = experiment.SchedulerStepSize
	}
	if experiment.SchedulerGamma > 0 {
		config["scheduler_gamma"] = experiment.SchedulerGamma
	}
	if experiment.LabelSmoothing > 0 {
		config["label_smoothing"] = experiment.LabelSmoothing
	}
	if experiment.GradientClipNorm > 0 {
		config["gradient_clip_norm"] = experiment.GradientClipNorm
	}
	if len(experiment.Augmentation) > 0 {
		config["augmentation"] = experiment.Augmentation
	}
	if experiment.AugmentationPolicy != "" {
		config["augmentation_policy"] = experiment.AugmentationPolicy
	}
	if experiment.AugmentationPolicyConfig != nil {
		config["augmentation_policy_config"] = experiment.AugmentationPolicyConfig
	}
	if experiment.ClassBalancing != "" {
		config["class_balancing"] = experiment.ClassBalancing
	}
	if len(experiment.ClassBalancingConfig) > 0 {
		config["class_balancing_config"] = experiment.ClassBalancingConfig
	}
	if experiment.SamplingStrategy != "" {
		config["sampling_strategy"] = experiment.SamplingStrategy
	}
	if experiment.EarlyStoppingPatience > 0 {
		config["early_stopping_patience"] = experiment.EarlyStoppingPatience
	}
	if experiment.Strategy != "" {
		config["strategy"] = experiment.Strategy
	}
	if experiment.Pretrained {
		config["pretrained"] = experiment.Pretrained
	}
	if experiment.FreezeBackbone {
		config["freeze_backbone"] = experiment.FreezeBackbone
	}
	if experiment.FineTuneStrategy != "" {
		config["fine_tune_strategy"] = experiment.FineTuneStrategy
	}
}

func (s *Server) prepareAutoMLExperiments(experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string, error) {
	return s.prepareAutoMLExperimentsForProject("", experiments)
}

func (s *Server) prepareAutoMLExperimentsForProject(projectID string, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string, error) {
	out := append([]plans.PlannedExperiment(nil), experiments...)
	warnings := []string{}
	automationSettings := s.currentAutomationSettings()
	history := []automl.OptimizerTrial{}
	if strings.TrimSpace(projectID) != "" && automationSettings.AutoMLEnabled {
		var err error
		history, err = s.store.ListProjectOptimizerTrials(projectID, 200)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML history failed for project %s: %v", projectID, err)
			history = nil
		}
	}
	for index := range out {
		if out[index].AutoML == nil && automationSettings.AutoMLEnabled && !strings.EqualFold(strings.TrimSpace(out[index].Template), jobs.TemplateLabelQualityAudit) {
			out[index].AutoML = defaultBackendAutoMLForExperiment(out[index], automationSettings.AutoMLSampler)
			warnings = append(warnings, fmt.Sprintf("AutoML auto-enabled for experiment %d; backend sampled concrete hyperparameters inside the LLM-selected strategy.", index))
		}
		if out[index].AutoML == nil || !out[index].AutoML.Enabled {
			continue
		}
		if !automationSettings.AutoMLEnabled {
			out[index].AutoML.ValidationStatus = "disabled"
			out[index].AutoML.Enabled = false
			warnings = append(warnings, fmt.Sprintf("AutoML disabled for experiment %d; using LLM-provided concrete hyperparameters.", index))
			continue
		}
		prepared, err := prepareAutoMLExperimentWithHistory(out[index], index, automationSettings.AutoMLSampler, history)
		if err != nil {
			return nil, warnings, err
		}
		out[index] = prepared
		warnings = append(warnings, fmt.Sprintf("AutoML sampled %d hyperparameter(s) for experiment %d using %s.", len(prepared.AutoML.Suggestion.Values), index, prepared.AutoML.Sampler))
	}
	return out, warnings, nil
}

func defaultBackendAutoMLForExperiment(experiment plans.PlannedExperiment, defaultSampler string) *automl.ExperimentAutoML {
	searchSpace := defaultBackendAutoMLSearchSpace(experiment)
	return &automl.ExperimentAutoML{
		Enabled:     true,
		Sampler:     normalizeAutoMLSampler(defaultSampler),
		SearchSpace: &searchSpace,
		Intent: automl.ExperimentIntent{
			Summary:           "Backend default AutoML samples concrete hyperparameters only; LLM-owned strategy fields are frozen.",
			PlanningMode:      strings.TrimSpace(experiment.Strategy),
			ExplorationIntent: "backend_default_hyperparameter_sampling",
			Goals: []string{
				"sample executable hyperparameters inside the validated experiment strategy",
				"preserve LLM-selected model, preprocessing, augmentation policy, class balancing, and fine-tuning choices",
			},
			AllowedParameters:   autoMLSearchSpaceParameterNames(searchSpace),
			StrategyDescription: strings.TrimSpace(experiment.Reason),
		},
	}
}

func defaultBackendAutoMLSearchSpace(experiment plans.PlannedExperiment) automl.HyperparameterSearchSpace {
	params := []automl.HyperparameterParameterSpec{}

	baseLR := experiment.LearningRate
	if baseLR <= 0 {
		baseLR = 3e-4
	}
	lrMin := maxFloat(1e-6, baseLR/5)
	lrMax := minFloat(3e-3, baseLR*5)
	if lrMax <= lrMin {
		lrMin = 1e-5
		lrMax = 3e-4
	}
	params = append(params, autoMLFloatSpec("learning_rate", lrMin, lrMax, automl.SearchScaleLog, "centered around the LLM-provided learning rate"))

	wdMin := 0.0
	wdMax := 0.08
	if experiment.WeightDecay > 0 {
		wdMin = maxFloat(0, experiment.WeightDecay/5)
		wdMax = minFloat(0.2, maxFloat(experiment.WeightDecay*4, experiment.WeightDecay+0.02))
	}
	params = append(params, autoMLFloatSpec("weight_decay", wdMin, wdMax, automl.SearchScaleLinear, "regularization strength"))

	params = append(params, autoMLIntChoicesSpec("batch_size", defaultAutoMLBatchSizeChoices(experiment.BatchSize), "worker-supported batch sizes near the LLM budget"))

	epochBase := experiment.Epochs
	if epochBase < 3 {
		epochBase = 10
	}
	epochMin := maxInt(3, epochBase-4)
	epochMax := minInt(40, epochBase+8)
	if epochMax < epochMin {
		epochMax = epochMin
	}
	params = append(params, autoMLIntRangeSpec("epochs", epochMin, epochMax, "bounded training budget around the LLM proposal"))
	params = append(params, autoMLIntChoicesSpec("early_stopping_patience", defaultAutoMLPatienceChoices(epochMax), "bounded early-stopping patience"))

	dropoutMin := 0.0
	dropoutMax := 0.35
	if experiment.Dropout > 0 {
		dropoutMin = maxFloat(0, experiment.Dropout/3)
		dropoutMax = minFloat(0.7, maxFloat(0.35, experiment.Dropout*2.5))
	}
	params = append(params, autoMLFloatSpec("dropout", dropoutMin, dropoutMax, automl.SearchScaleLinear, "head regularization"))
	params = append(params, autoMLFloatSpec("label_smoothing", 0, 0.15, automl.SearchScaleLinear, "classification regularization"))
	params = append(params, autoMLFloatSpec("gradient_clip_norm", 0, 3, automl.SearchScaleLinear, "gradient stability"))

	if strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		params = append(params, autoMLFloatSpec("optimizer_momentum", 0.7, 0.95, automl.SearchScaleLinear, "SGD momentum"))
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		stepMax := minInt(10, maxInt(2, epochMax/2))
		params = append(params, autoMLIntRangeSpec("scheduler_step_size", 1, stepMax, "step scheduler cadence"))
		params = append(params, autoMLFloatSpec("scheduler_gamma", 0.2, 0.8, automl.SearchScaleLinear, "step scheduler decay"))
	}

	if experiment.AugmentationPolicyConfig != nil {
		switch strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType)) {
		case "randaugment":
			params = append(params,
				autoMLIntRangeSpec("augmentation_policy_config.magnitude", 4, 12, "numeric strength for the LLM-selected randaugment policy"),
				autoMLIntRangeSpec("augmentation_policy_config.num_ops", 1, 3, "operation count for the LLM-selected randaugment policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0.5, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"),
			)
		case "trivialaugment", "trivialaugmentwide":
			params = append(params,
				autoMLIntRangeSpec("augmentation_policy_config.num_magnitude_bins", 8, 31, "magnitude bins for the LLM-selected augmentation policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0.5, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"),
			)
		case "autoaugment":
			params = append(params, autoMLFloatSpec("augmentation_policy_config.probability", 0.5, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"))
		case "mixup", "cutmix":
			params = append(params,
				autoMLFloatSpec("augmentation_policy_config.alpha", 0.05, 0.6, automl.SearchScaleLinear, "mixing strength for the LLM-selected mixed-sample policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0.3, 0.9, automl.SearchScaleLinear, "application probability for the LLM-selected mixed-sample policy"),
			)
		}
	}

	if effectiveNumberClassBalancing(experiment.ClassBalancing) {
		params = append(params, autoMLFloatSpec("class_balancing_config.effective_number_beta", 0.95, 0.9999, automl.SearchScaleLinear, "effective-number loss beta"))
	}
	if strings.EqualFold(strings.TrimSpace(experiment.ClassBalancing), "focal_loss") {
		params = append(params, autoMLFloatSpec("class_balancing_config.focal_loss_gamma", 1, 4, automl.SearchScaleLinear, "focal loss gamma"))
	}

	return automl.HyperparameterSearchSpace{Parameters: params}
}

func autoMLSearchSpaceParameterNames(space automl.HyperparameterSearchSpace) []string {
	names := make([]string, 0, len(space.Parameters))
	for _, spec := range space.Parameters {
		names = append(names, spec.Name)
	}
	return names
}

func autoMLFloatSpec(name string, minValue float64, maxValue float64, scale automl.SearchScale, notes string) automl.HyperparameterParameterSpec {
	return automl.HyperparameterParameterSpec{
		Name:   name,
		Type:   automl.ParameterFloat,
		Min:    &minValue,
		Max:    &maxValue,
		Scale:  scale,
		Source: automl.ProvenanceBackendDefault,
		Notes:  notes,
	}
}

func autoMLIntRangeSpec(name string, minValue int, maxValue int, notes string) automl.HyperparameterParameterSpec {
	minFloatValue := float64(minValue)
	maxFloatValue := float64(maxValue)
	return automl.HyperparameterParameterSpec{
		Name:   name,
		Type:   automl.ParameterInteger,
		Min:    &minFloatValue,
		Max:    &maxFloatValue,
		Source: automl.ProvenanceBackendDefault,
		Notes:  notes,
	}
}

func autoMLIntChoicesSpec(name string, choices []int, notes string) automl.HyperparameterParameterSpec {
	return automl.HyperparameterParameterSpec{
		Name:       name,
		Type:       automl.ParameterInteger,
		IntChoices: append([]int(nil), choices...),
		Source:     automl.ProvenanceBackendDefault,
		Notes:      notes,
	}
}

func defaultAutoMLBatchSizeChoices(current int) []int {
	supported := []int{4, 8, 16, 32, 64, 128}
	if current <= 0 {
		return []int{8, 16, 32}
	}
	lower := maxInt(4, current/2)
	upper := maxInt(lower, current*2)
	choices := []int{}
	for _, choice := range supported {
		if choice >= lower && choice <= upper {
			choices = append(choices, choice)
		}
	}
	if len(choices) > 0 {
		return choices
	}
	nearest := supported[0]
	bestDistance := absInt(current - nearest)
	for _, choice := range supported[1:] {
		if distance := absInt(current - choice); distance < bestDistance {
			nearest = choice
			bestDistance = distance
		}
	}
	return []int{nearest}
}

func defaultAutoMLPatienceChoices(maxEpochs int) []int {
	limit := minInt(8, maxInt(2, maxEpochs/2))
	choices := []int{}
	for value := 2; value <= limit; value++ {
		choices = append(choices, value)
	}
	return choices
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func prepareAutoMLExperiment(experiment plans.PlannedExperiment, index int, defaultSampler string) (plans.PlannedExperiment, error) {
	return prepareAutoMLExperimentWithHistory(experiment, index, defaultSampler, nil)
}

func prepareAutoMLExperimentWithHistory(experiment plans.PlannedExperiment, index int, defaultSampler string, history []automl.OptimizerTrial) (plans.PlannedExperiment, error) {
	if experiment.AutoML == nil || !experiment.AutoML.Enabled {
		return experiment, nil
	}
	strategy := automlStrategyContext(experiment)
	searchSpace := experiment.AutoML.SearchSpace
	if searchSpace == nil || len(searchSpace.Parameters) == 0 {
		defaultSpace, err := automl.DefaultSearchSpace(experiment.AutoML.Intent.AllowedParameters, strategy)
		if err != nil {
			return experiment, fmt.Errorf("%w: proposed experiment %d AutoML intent is invalid: %s", store.ErrInvalidRequest, index, err.Error())
		}
		searchSpace = &defaultSpace
		experiment.AutoML.SearchSpace = searchSpace
	}
	if err := automl.ValidateSearchSpace(*searchSpace, strategy); err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML search space is invalid: %s", store.ErrInvalidRequest, index, err.Error())
	}
	samplerName := strings.TrimSpace(experiment.AutoML.Sampler)
	if samplerName == "" {
		samplerName = defaultSampler
	}
	samplerName = normalizeAutoMLSampler(samplerName)
	sampler, err := automl.NewSampler(samplerName)
	if err != nil {
		return experiment, fmt.Errorf("%w: proposed experiment %d %s", store.ErrInvalidRequest, index, err.Error())
	}
	seed := experiment.AutoML.Seed
	if seed == 0 {
		seed = automlSeedForExperiment(experiment, index)
	}
	suggestion, err := sampler.Suggest(context.Background(), automl.SuggestRequest{
		SearchSpace:     *searchSpace,
		StrategyContext: strategy,
		History:         history,
		Seed:            seed,
	})
	if err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML suggestion failed: %s", store.ErrInvalidRequest, index, err.Error())
	}
	for _, spec := range searchSpace.Parameters {
		if _, ok := suggestion.Values[spec.Name]; ok {
			continue
		}
		if _, ok := suggestion.Values[automl.NormalizeParameterName(spec.Name)]; ok {
			continue
		}
		clearAutoMLValueFromExperiment(&experiment, spec.Name)
	}
	for name, value := range suggestion.Values {
		if err := applyAutoMLValueToExperiment(&experiment, name, value); err != nil {
			experiment.AutoML.ValidationStatus = "invalid"
			experiment.AutoML.ValidationErrors = []string{err.Error()}
			return experiment, fmt.Errorf("%w: proposed experiment %d %s", store.ErrInvalidRequest, index, err.Error())
		}
	}
	if err := automl.ValidateSuggestion(suggestion, *searchSpace, automlStrategyContext(experiment)); err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML suggestion is invalid: %s", store.ErrInvalidRequest, index, err.Error())
	}
	finalValues, provenance := autoMLFinalValues(experiment, suggestion.Provenance)
	suggestion.FinalValues = finalValues
	suggestion.ValidationStatus = "valid"
	experiment.AutoML.Sampler = samplerName
	experiment.AutoML.Seed = seed
	experiment.AutoML.Suggestion = &suggestion
	experiment.AutoML.FinalValues = finalValues
	experiment.AutoML.ValueProvenance = provenance
	experiment.AutoML.StrategySnapshot = autoMLStrategySnapshot(experiment)
	experiment.AutoML.ValidationStatus = "valid"
	experiment.AutoML.ValidationErrors = []string{}
	return experiment, nil
}

func clearAutoMLValueFromExperiment(experiment *plans.PlannedExperiment, name string) {
	switch automl.NormalizeParameterName(name) {
	case "optimizer_momentum":
		experiment.OptimizerMomentum = 0
	case "scheduler_step_size":
		experiment.SchedulerStepSize = 0
	case "scheduler_gamma":
		experiment.SchedulerGamma = 0
	}
}

func applyAutoMLValueToExperiment(experiment *plans.PlannedExperiment, name string, value any) error {
	switch automl.NormalizeParameterName(name) {
	case "learning_rate":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl learning_rate must be numeric")
		}
		experiment.LearningRate = number
	case "weight_decay":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl weight_decay must be numeric")
		}
		experiment.WeightDecay = number
	case "dropout":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl dropout must be numeric")
		}
		experiment.Dropout = number
	case "optimizer_momentum":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl optimizer_momentum must be numeric")
		}
		experiment.OptimizerMomentum = number
	case "scheduler_step_size":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl scheduler_step_size must be an integer")
		}
		experiment.SchedulerStepSize = integer
	case "scheduler_gamma":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl scheduler_gamma must be numeric")
		}
		experiment.SchedulerGamma = number
	case "label_smoothing":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl label_smoothing must be numeric")
		}
		experiment.LabelSmoothing = number
	case "gradient_clip_norm":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl gradient_clip_norm must be numeric")
		}
		experiment.GradientClipNorm = number
	case "batch_size":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl batch_size must be an integer")
		}
		experiment.BatchSize = integer
	case "epochs":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl epochs must be an integer")
		}
		experiment.Epochs = integer
	case "early_stopping_patience":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl early_stopping_patience must be an integer")
		}
		experiment.EarlyStoppingPatience = integer
	case "optimizer":
		experiment.Optimizer = strings.ToLower(strings.TrimSpace(fmt.Sprint(value)))
	case "scheduler":
		experiment.Scheduler = strings.ToLower(strings.TrimSpace(fmt.Sprint(value)))
	case "augmentation_policy_config.magnitude":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.magnitude must be an integer")
		}
		config.Magnitude = integer
	case "augmentation_policy_config.num_ops":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.num_ops must be an integer")
		}
		config.NumOps = integer
	case "augmentation_policy_config.num_magnitude_bins":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.num_magnitude_bins must be an integer")
		}
		config.NumMagnitudeBins = integer
	case "augmentation_policy_config.probability":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.probability must be numeric")
		}
		config.Probability = number
	case "augmentation_policy_config.alpha":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.alpha must be numeric")
		}
		config.Alpha = number
	case "class_balancing_config.effective_number_beta":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl class_balancing_config.effective_number_beta must be numeric")
		}
		if experiment.ClassBalancingConfig == nil {
			experiment.ClassBalancingConfig = map[string]any{}
		}
		experiment.ClassBalancingConfig["effective_number_beta"] = number
	case "class_balancing_config.focal_loss_gamma":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl class_balancing_config.focal_loss_gamma must be numeric")
		}
		if experiment.ClassBalancingConfig == nil {
			experiment.ClassBalancingConfig = map[string]any{}
		}
		experiment.ClassBalancingConfig["focal_loss_gamma"] = number
	default:
		return fmt.Errorf("automl cannot apply unsupported parameter %q", name)
	}
	return nil
}

func requireAutoMLAugmentationConfig(experiment *plans.PlannedExperiment) (*plans.AugmentationPolicyConfig, error) {
	if experiment.AugmentationPolicyConfig == nil || strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType) == "" {
		return nil, fmt.Errorf("automl structured augmentation parameters require LLM-selected augmentation_policy_config.policy_type")
	}
	return experiment.AugmentationPolicyConfig, nil
}

func automlStrategyContext(experiment plans.PlannedExperiment) automl.StrategyContext {
	policyType := ""
	if experiment.AugmentationPolicyConfig != nil {
		policyType = experiment.AugmentationPolicyConfig.PolicyType
	}
	return automl.StrategyContext{
		Model:                  experiment.Model,
		Template:               experiment.Template,
		Optimizer:              experiment.Optimizer,
		Scheduler:              experiment.Scheduler,
		AugmentationPolicy:     experiment.AugmentationPolicy,
		AugmentationPolicyType: policyType,
		ClassBalancing:         experiment.ClassBalancing,
	}
}

func autoMLStrategySnapshot(experiment plans.PlannedExperiment) map[string]any {
	snapshot := map[string]any{
		"template":            experiment.Template,
		"model":               experiment.Model,
		"mechanism":           experiment.Mechanism,
		"intervention":        experiment.Intervention,
		"image_size":          experiment.ImageSize,
		"resolution_strategy": experiment.ResolutionStrategy,
		"augmentation_policy": experiment.AugmentationPolicy,
		"class_balancing":     experiment.ClassBalancing,
		"sampling_strategy":   experiment.SamplingStrategy,
		"pretrained":          experiment.Pretrained,
		"freeze_backbone":     experiment.FreezeBackbone,
		"fine_tune_strategy":  experiment.FineTuneStrategy,
	}
	if experiment.Preprocessing != nil {
		snapshot["preprocessing"] = experiment.Preprocessing
	}
	if experiment.AugmentationPolicyConfig != nil {
		snapshot["augmentation_policy_config_policy_type"] = experiment.AugmentationPolicyConfig.PolicyType
	}
	return compactNonEmptyMap(snapshot)
}

func autoMLFinalValues(experiment plans.PlannedExperiment, sampled map[string]automl.HyperparameterProvenance) (map[string]any, map[string]automl.HyperparameterProvenance) {
	values := map[string]any{}
	provenance := map[string]automl.HyperparameterProvenance{}
	for _, capability := range automl.DefaultCapabilityRegistry().Capabilities() {
		value, ok := autoMLParameterValue(experiment, capability.Name)
		if !ok {
			continue
		}
		values[capability.Name] = value
		if source, ok := sampled[capability.Name]; ok {
			provenance[capability.Name] = source
		} else {
			provenance[capability.Name] = automl.ProvenanceLLM
		}
	}
	return values, provenance
}

func autoMLParameterValue(experiment plans.PlannedExperiment, name string) (any, bool) {
	switch automl.NormalizeParameterName(name) {
	case "learning_rate":
		return experiment.LearningRate, experiment.LearningRate > 0
	case "weight_decay":
		return experiment.WeightDecay, true
	case "dropout":
		return experiment.Dropout, experiment.Dropout > 0
	case "optimizer_momentum":
		return experiment.OptimizerMomentum, experiment.OptimizerMomentum > 0
	case "scheduler_step_size":
		return experiment.SchedulerStepSize, experiment.SchedulerStepSize > 0
	case "scheduler_gamma":
		return experiment.SchedulerGamma, experiment.SchedulerGamma > 0
	case "label_smoothing":
		return experiment.LabelSmoothing, experiment.LabelSmoothing > 0
	case "gradient_clip_norm":
		return experiment.GradientClipNorm, experiment.GradientClipNorm > 0
	case "batch_size":
		return experiment.BatchSize, experiment.BatchSize > 0
	case "epochs":
		return experiment.Epochs, experiment.Epochs > 0
	case "early_stopping_patience":
		return experiment.EarlyStoppingPatience, experiment.EarlyStoppingPatience >= 0
	case "optimizer":
		if strings.TrimSpace(experiment.Optimizer) == "" {
			return "adamw", true
		}
		return experiment.Optimizer, true
	case "scheduler":
		if strings.TrimSpace(experiment.Scheduler) == "" {
			return "none", true
		}
		return experiment.Scheduler, true
	case "augmentation_policy_config.magnitude":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Magnitude, true
	case "augmentation_policy_config.num_ops":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.NumOps, true
	case "augmentation_policy_config.num_magnitude_bins":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.NumMagnitudeBins, true
	case "augmentation_policy_config.probability":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Probability, true
	case "augmentation_policy_config.alpha":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Alpha, true
	case "class_balancing_config.effective_number_beta":
		if experiment.ClassBalancingConfig == nil {
			return nil, false
		}
		value, ok := experiment.ClassBalancingConfig["effective_number_beta"]
		return value, ok
	case "class_balancing_config.focal_loss_gamma":
		if experiment.ClassBalancingConfig == nil {
			return nil, false
		}
		value, ok := experiment.ClassBalancingConfig["focal_loss_gamma"]
		return value, ok
	default:
		return nil, false
	}
}

func automlSeedForExperiment(experiment plans.PlannedExperiment, index int) int64 {
	blob, _ := json.Marshal(map[string]any{
		"index":              index,
		"strategy_snapshot":  autoMLStrategySnapshot(experiment),
		"search_space":       experiment.AutoML.SearchSpace,
		"allowed_parameters": experiment.AutoML.Intent.AllowedParameters,
		"sampler":            experiment.AutoML.Sampler,
	})
	sum := sha256.Sum256(blob)
	var seed int64
	for i := 0; i < 8; i++ {
		seed = (seed << 8) + int64(sum[i])
	}
	if seed < 0 {
		seed = -seed
	}
	if seed == 0 {
		seed = int64(index + 1)
	}
	return seed
}

func compactNonEmptyMap(input map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range input {
		switch typed := value.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				out[key] = typed
			}
		case int:
			if typed != 0 {
				out[key] = typed
			}
		case bool:
			if typed {
				out[key] = typed
			}
		case nil:
		default:
			out[key] = value
		}
	}
	return out
}

func (s *Server) persistAutoMLForPlan(plan plans.ExperimentPlan) error {
	for index, experiment := range plan.Experiments {
		if experiment.AutoML == nil || !experiment.AutoML.Enabled || experiment.AutoML.Suggestion == nil {
			continue
		}
		study, err := s.store.CreateOptimizerStudy(automl.OptimizerStudy{
			ProjectID:        plan.ProjectID,
			PlanID:           plan.ID,
			DatasetID:        plan.DatasetID,
			SourceDecisionID: plan.SourceDecisionID,
			ExperimentIndex:  index,
			Model:            experiment.Model,
			Intent:           experiment.AutoML.Intent,
			Sampler:          experiment.AutoML.Sampler,
			Seed:             experiment.AutoML.Seed,
			SearchSpace:      *experiment.AutoML.SearchSpace,
			StrategySnapshot: experiment.AutoML.StrategySnapshot,
		})
		if err != nil {
			return err
		}
		suggestion := experiment.AutoML.Suggestion
		_, err = s.store.CreateOptimizerSuggestion(automl.OptimizerSuggestion{
			StudyID:          study.ID,
			ProjectID:        plan.ProjectID,
			PlanID:           plan.ID,
			DatasetID:        plan.DatasetID,
			ExperimentIndex:  index,
			Model:            experiment.Model,
			Sampler:          suggestion.Sampler,
			Seed:             suggestion.Seed,
			Values:           suggestion.Values,
			FinalValues:      suggestion.FinalValues,
			Provenance:       suggestion.Provenance,
			ValidationStatus: suggestion.ValidationStatus,
			ValidationErrors: suggestion.ValidationErrors,
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) automlSuggestionsByExperiment(planID string) map[int]automl.OptimizerSuggestion {
	suggestions, err := s.store.ListPlanOptimizerSuggestions(planID)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML suggestions failed for plan %s: %v", planID, err)
		}
		return map[int]automl.OptimizerSuggestion{}
	}
	out := map[int]automl.OptimizerSuggestion{}
	for _, suggestion := range suggestions {
		if _, exists := out[suggestion.ExperimentIndex]; exists {
			continue
		}
		out[suggestion.ExperimentIndex] = suggestion
	}
	return out
}

func automlJobSummary(experiment plans.PlannedExperiment, suggestion automl.OptimizerSuggestion) map[string]any {
	return map[string]any{
		"enabled":             true,
		"sampler":             suggestion.Sampler,
		"study_id":            suggestion.StudyID,
		"suggestion_id":       suggestion.ID,
		"seed":                suggestion.Seed,
		"values":              suggestion.Values,
		"final_values":        suggestion.FinalValues,
		"provenance":          suggestion.Provenance,
		"validation_status":   suggestion.ValidationStatus,
		"validation_errors":   suggestion.ValidationErrors,
		"strategy_snapshot":   autoMLStrategySnapshot(experiment),
		"llm_remains_planner": true,
	}
}

func (s *Server) observeAutoMLTrialForJob(job jobs.ExperimentJob) error {
	suggestionID := jobConfigString(job.Config, "automl_suggestion_id")
	if suggestionID == "" {
		return nil
	}
	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		return err
	}
	targetMetric := jobConfigString(job.Config, "target_metric")
	if targetMetric == "" {
		targetMetric = "macro_f1"
	}
	status := summary.Status
	if status == "" {
		status = job.Status
	}
	score := plannerTargetMetricValue(targetMetric, summary)
	if status != jobs.StatusSucceeded {
		score = 0
	}
	metrics := map[string]any{
		"best_macro_f1":        summary.BestMacroF1,
		"best_accuracy":        summary.BestAccuracy,
		"final_train_loss":     summary.FinalTrainLoss,
		"final_val_loss":       summary.FinalValLoss,
		"epochs_completed":     summary.EpochsCompleted,
		"runtime_seconds":      summary.RuntimeSeconds,
		"estimated_cost_usd":   summary.EstimatedCostUSD,
		"hyperparameters":      automlHyperparametersFromJob(job),
		"automl_summary":       job.Config["automl_summary"],
		"train_validation_gap": summary.FinalValLoss - summary.FinalTrainLoss,
	}
	if evaluation, err := s.store.GetTrainingRunEvaluation(job.ID); err == nil {
		if diagnostics, ok := evaluation.HolisticScores["training_diagnostics"]; ok {
			metrics["training_diagnostics"] = diagnostics
		}
		if gap, ok := evaluation.HolisticScores["train_validation_gap"]; ok {
			metrics["train_validation_gap"] = gap
		}
	}
	_, err = s.store.UpsertOptimizerTrial(automl.OptimizerTrial{
		StudyID:      jobConfigString(job.Config, "automl_study_id"),
		SuggestionID: suggestionID,
		ProjectID:    job.ProjectID,
		PlanID:       jobConfigString(job.Config, "plan_id"),
		DatasetID:    jobConfigString(job.Config, "dataset_id"),
		JobID:        job.ID,
		Status:       status,
		TargetMetric: targetMetric,
		Score:        score,
		Metrics:      metrics,
		Error:        job.Error,
	})
	return err
}

func automlHyperparametersFromJob(job jobs.ExperimentJob) map[string]any {
	if summary, ok := job.Config["automl_summary"].(map[string]any); ok {
		if finalValues, ok := summary["final_values"].(map[string]any); ok {
			return finalValues
		}
	}
	out := map[string]any{}
	for _, key := range []string{
		"learning_rate",
		"weight_decay",
		"dropout",
		"optimizer_momentum",
		"scheduler_step_size",
		"scheduler_gamma",
		"label_smoothing",
		"gradient_clip_norm",
		"batch_size",
		"epochs",
		"early_stopping_patience",
		"optimizer",
		"scheduler",
		"augmentation_policy_config",
		"class_balancing_config",
	} {
		if value, ok := job.Config[key]; ok {
			out[key] = value
		}
	}
	return out
}

func (s *Server) optimizerFeedbackSummary(studyID string, targetMetric string) *automl.OptimizerFeedbackSummary {
	studyID = strings.TrimSpace(studyID)
	if studyID == "" {
		return nil
	}
	trials, err := s.store.ListStudyOptimizerTrials(studyID)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML trials failed for study %s: %v", studyID, err)
		}
		return nil
	}
	summary := automl.BuildFeedbackSummary(studyID, targetMetric, trials)
	return &summary
}

func (s *Server) optimizerFeedbackSummariesForProject(projectID string, targetMetric string) []automl.OptimizerFeedbackSummary {
	trials, err := s.store.ListProjectOptimizerTrials(projectID, 100)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list project AutoML trials failed for project %s: %v", projectID, err)
		}
		return nil
	}
	byStudy := map[string][]automl.OptimizerTrial{}
	for _, trial := range trials {
		if strings.TrimSpace(trial.StudyID) == "" {
			continue
		}
		byStudy[trial.StudyID] = append(byStudy[trial.StudyID], trial)
	}
	out := []automl.OptimizerFeedbackSummary{}
	for studyID, studyTrials := range byStudy {
		out = append(out, automl.BuildFeedbackSummary(studyID, targetMetric, studyTrials))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].TrialCount > out[j].TrialCount
	})
	if len(out) > 5 {
		out = out[:5]
	}
	return out
}

func configString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

func addDatasetMaterializationConfig(config map[string]any, policy execution.WorkerRequirementPolicy) {
	if policy.DatasetID == "" && policy.DatasetCacheKey == "" {
		return
	}
	if policy.DatasetChecksum != "" {
		config["dataset_checksum_sha256"] = policy.DatasetChecksum
	}
	materialization := map[string]any{
		"dataset_id":                policy.DatasetID,
		"dataset_checksum_sha256":   policy.DatasetChecksum,
		"dataset_cache_key":         policy.DatasetCacheKey,
		"status":                    policy.DatasetMaterializationStatus,
		"cold_cache_policy":         policy.ColdCachePolicy,
		"max_concurrent_jobs":       policy.MaxConcurrentJobs,
		"max_cold_materializations": policy.MaxColdDatasetMaterializations,
	}
	config["dataset_materialization"] = materialization
}

func configInt(config map[string]any, key string) (int, bool) {
	switch value := config[key].(type) {
	case int:
		return value, true
	case int64:
		return int(value), true
	case float64:
		return int(value), true
	default:
		return 0, false
	}
}

func latestExperimentPlan(projectPlans []plans.ExperimentPlan) (plans.ExperimentPlan, bool) {
	if len(projectPlans) == 0 {
		return plans.ExperimentPlan{}, false
	}

	sort.Slice(projectPlans, func(i, j int) bool {
		return projectPlans[i].CreatedAt.After(projectPlans[j].CreatedAt)
	})

	return projectPlans[0], true
}

func followUpPlanForDecision(projectPlans []plans.ExperimentPlan, decisionID string) (plans.ExperimentPlan, bool) {
	for _, plan := range projectPlans {
		if plan.SourceDecisionID == decisionID {
			return plan, true
		}
	}

	return plans.ExperimentPlan{}, false
}

func plannedExperimentsFromPayload(payload map[string]any) ([]plans.PlannedExperiment, error) {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return nil, fmt.Errorf("%w: reviewer decision does not include proposed_experiments", store.ErrInvalidRequest)
	}

	blob, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments could not be encoded", store.ErrInvalidRequest)
	}

	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments has an invalid shape", store.ErrInvalidRequest)
	}

	if len(experiments) == 0 {
		return nil, fmt.Errorf("%w: reviewer proposed no follow-up experiments", store.ErrInvalidRequest)
	}
	if len(experiments) > maxLLMPlannerExperiments {
		return nil, fmt.Errorf("%w: proposed_experiments has %d experiments, max is %d", store.ErrInvalidRequest, len(experiments), maxLLMPlannerExperiments)
	}

	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return nil, err
		}
	}

	return experiments, nil
}

func plannedExperimentsFromPayloadLenient(payload map[string]any) ([]plans.PlannedExperiment, error) {
	value, ok := payload["proposed_experiments"]
	if !ok || value == nil {
		return nil, fmt.Errorf("%w: reviewer decision does not include proposed_experiments", store.ErrInvalidRequest)
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return nil, err
	}
	return experiments, nil
}

func validateLLMPlannerStoredMechanismContract(decision decisions.AgentDecision, experiments []plans.PlannedExperiment) error {
	if decision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
		return nil
	}
	return validatePlannedExperimentMechanismContract(experiments, payloadStringSlice(decision.Payload, "evidence_used"))
}

func plannerExperimentsWithProposalMechanisms(recommendation agents.ExperimentPlanningRecommendation) ([]plans.PlannedExperiment, error) {
	experiments := append([]plans.PlannedExperiment(nil), recommendation.ProposedExperiments...)
	if len(experiments) == 0 {
		return experiments, nil
	}
	return attachProposalMechanismsToExperiments(experiments, recommendation.ProposalMechanisms)
}

func plannedExperimentsWithStoredProposalMechanisms(payload map[string]any, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, error) {
	mechanisms, ok, err := plannerProposalMechanismsFromPayload(payload)
	if err != nil {
		return nil, err
	}
	if !ok {
		return experiments, nil
	}
	return attachProposalMechanismsToExperiments(experiments, mechanisms)
}

func plannerProposalMechanismsFromPayload(payload map[string]any) ([]agents.PlannerProposalMechanism, bool, error) {
	value, ok := payload["proposal_mechanisms"]
	if !ok || value == nil {
		return nil, false, nil
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, false, fmt.Errorf("%w: proposal_mechanisms could not be encoded", store.ErrInvalidRequest)
	}
	var mechanisms []agents.PlannerProposalMechanism
	if err := json.Unmarshal(blob, &mechanisms); err != nil {
		return nil, false, fmt.Errorf("%w: proposal_mechanisms has an invalid shape", store.ErrInvalidRequest)
	}
	return mechanisms, true, nil
}

func attachProposalMechanismsToExperiments(experiments []plans.PlannedExperiment, mechanisms []agents.PlannerProposalMechanism) ([]plans.PlannedExperiment, error) {
	out := append([]plans.PlannedExperiment(nil), experiments...)
	mechanismsByIndex := map[int]agents.PlannerProposalMechanism{}
	for index, mechanism := range mechanisms {
		if mechanism.ExperimentIndex < 0 || mechanism.ExperimentIndex >= len(out) {
			return nil, fmt.Errorf("%w: proposal_mechanisms[%d] has invalid experiment_index %d", store.ErrInvalidRequest, index, mechanism.ExperimentIndex)
		}
		if _, exists := mechanismsByIndex[mechanism.ExperimentIndex]; exists {
			return nil, fmt.Errorf("%w: proposal_mechanisms duplicate experiment_index %d", store.ErrInvalidRequest, mechanism.ExperimentIndex)
		}
		mechanismsByIndex[mechanism.ExperimentIndex] = mechanism
	}
	for index := range out {
		mechanism, ok := mechanismsByIndex[index]
		if !ok {
			continue
		}
		out[index].Mechanism = mechanism.Mechanism
		out[index].Intervention = mechanism.Intervention
		out[index].EvidenceUsed = append([]string(nil), mechanism.EvidenceUsed...)
		out[index].ExpectedEffect = mechanism.ExpectedEffect
	}
	return out, nil
}

func validateLLMPlannerMechanismContract(experiments []plans.PlannedExperiment, evidenceUsed []string) error {
	return validatePlannedExperimentMechanismContract(experiments, evidenceUsed)
}

func validatePlannedExperimentMechanismContract(experiments []plans.PlannedExperiment, planEvidence []string) error {
	if len(experiments) == 0 {
		return nil
	}
	for index, experiment := range experiments {
		mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
		if mechanism == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing mechanism", store.ErrInvalidRequest, index)
		}
		if !allowedExperimentValue(mechanism, allowedPlannerMechanisms()) {
			return fmt.Errorf("%w: proposed experiment %d has unsupported mechanism %q", store.ErrInvalidRequest, index, experiment.Mechanism)
		}
		if strings.TrimSpace(experiment.Intervention) == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing intervention", store.ErrInvalidRequest, index)
		}
		if strings.TrimSpace(experiment.ExpectedEffect) == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing expected_effect", store.ErrInvalidRequest, index)
		}
		if len(nonEmptyStringValues(experiment.EvidenceUsed)) == 0 && len(nonEmptyStringValues(planEvidence)) == 0 {
			return fmt.Errorf("%w: proposed experiment %d is missing evidence_used", store.ErrInvalidRequest, index)
		}
	}
	return nil
}

func validateMechanismDatasetEvidence(profile map[string]any, experiments []plans.PlannedExperiment, planEvidence []string) error {
	violations := []string{}
	for index, experiment := range experiments {
		mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
		if mechanism == "" {
			continue
		}
		evidenceText := experimentMechanismEvidenceText(experiment, planEvidence)
		switch mechanism {
		case "class_imbalance", "minority_targeting":
			if !classBalancingConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s requires class_balancing or sampling_strategy", index, mechanism))
			}
			if !profileOrDiagnosisHasClassImbalanceEvidence(profile, evidenceText) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s requires dataset imbalance, per-class error, or minority-failure evidence", index, mechanism))
			}
		case "bbox_crop_ablation":
			if !bboxCropConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism bbox_crop_ablation requires bbox crop preprocessing", index))
			}
			if !profileHasBBoxEvidence(profile) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism bbox_crop_ablation requires backend-profiled bbox/annotation evidence", index))
			}
		case "resolution_crop":
			if !resolutionCropConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism resolution_crop requires image size, resolution strategy, or crop preprocessing changes", index))
			}
			if resolutionCropNeedsEvidence(experiment) && !profileOrDiagnosisHasResolutionCropEvidence(profile, evidenceText) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism resolution_crop requires object-scale, fine-grained, dimension, crop, or visual-trait evidence", index))
			}
		case "augmentation_auto":
			if !autoAugmentationConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism augmentation_auto requires structured randaugment, trivialaugment, or autoaugment policy config", index))
			}
		case "augmentation_mixed_sample":
			if !mixedSampleAugmentationConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism augmentation_mixed_sample requires structured MixUp or CutMix augmentation policy config", index))
			}
		case "label_noise_audit", "hard_example_audit":
			if !labelQualityAuditExperiment(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s is report-only and must use template %s instead of creating a training job", index, mechanism, jobs.TemplateLabelQualityAudit))
			}
		case "distillation":
			violations = append(violations, fmt.Sprintf("experiment %d mechanism distillation is not schedulable until teacher-artifact validation and worker support are enabled", index))
		case "deployment_latency":
			if !containsAnyText(evidenceText, "latency", "runtime", "cost", "edge", "live", "small", "compact", "mobile") {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism deployment_latency requires deployment, latency, runtime, cost, or compact-model evidence", index))
			}
		}
	}
	if len(violations) > 0 {
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, strings.Join(violations, "; "))
	}
	return nil
}

func experimentMechanismEvidenceText(experiment plans.PlannedExperiment, planEvidence []string) string {
	parts := append([]string{}, planEvidence...)
	parts = append(parts, experiment.EvidenceUsed...)
	parts = append(parts,
		experiment.Intervention,
		experiment.ExpectedEffect,
		experiment.Reason,
		experiment.Strategy,
	)
	return strings.ToLower(strings.Join(parts, " "))
}

func classBalancingConfigured(experiment plans.PlannedExperiment) bool {
	return nonDefaultText(experiment.ClassBalancing, "none") || nonDefaultText(experiment.SamplingStrategy, "none")
}

func bboxCropConfigured(experiment plans.PlannedExperiment) bool {
	if experiment.Preprocessing == nil {
		return false
	}
	return containsAnyText(strings.ToLower(experiment.Preprocessing.CropStrategy+" "+experiment.Preprocessing.BBoxMode+" "+experiment.Preprocessing.ResizeStrategy), "bbox", "box")
}

func resolutionCropConfigured(experiment plans.PlannedExperiment) bool {
	if experiment.ImageSize > 0 || nonDefaultText(experiment.ResolutionStrategy, "fixed") {
		return true
	}
	if experiment.Preprocessing == nil {
		return false
	}
	return nonDefaultText(experiment.Preprocessing.ResizeStrategy, "squash") ||
		nonDefaultText(experiment.Preprocessing.CropStrategy, "none") ||
		nonDefaultText(experiment.Preprocessing.Normalization, "imagenet")
}

func resolutionCropNeedsEvidence(experiment plans.PlannedExperiment) bool {
	if experiment.ImageSize > 256 {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(experiment.ResolutionStrategy), "high_resolution_ablation") {
		return true
	}
	if experiment.Preprocessing == nil {
		return false
	}
	return containsAnyText(strings.ToLower(experiment.Preprocessing.CropStrategy+" "+experiment.Preprocessing.ResizeStrategy), "crop", "bbox", "aspect")
}

func autoAugmentationConfigured(experiment plans.PlannedExperiment) bool {
	policy := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy))
	if containsAnyText(policy, "randaugment", "trivialaugment", "autoaugment") {
		return true
	}
	if experiment.AugmentationPolicyConfig == nil {
		return false
	}
	policyType := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType))
	return policyType == "randaugment" || policyType == "trivialaugment" || policyType == "trivialaugmentwide" || policyType == "autoaugment"
}

func mixedSampleAugmentationConfigured(experiment plans.PlannedExperiment) bool {
	policy := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy))
	if policy == "mixup" || policy == "cutmix" {
		return true
	}
	if experiment.AugmentationPolicyConfig == nil {
		return false
	}
	policyType := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType))
	return policyType == "mixup" || policyType == "cutmix"
}

func labelQualityAuditExperiment(experiment plans.PlannedExperiment) bool {
	if !strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return false
	}
	mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
	return mechanism == "label_noise_audit" || mechanism == "hard_example_audit"
}

func profileOrDiagnosisHasClassImbalanceEvidence(profile map[string]any, evidenceText string) bool {
	if profileFloat(profile, "imbalance_ratio") >= 1.5 {
		return true
	}
	distribution := profileMap(profile, "class_distribution")
	if len(distribution) == 0 {
		distribution = profileMap(profile, "images_per_class")
	}
	if classDistributionImbalanceRatio(distribution) >= 1.5 {
		return true
	}
	return containsAnyText(evidenceText, "class_imbalance", "class imbalance", "minority", "rare class", "per-class", "per class", "macro-f1 trails accuracy", "macro f1 trails accuracy")
}

func classDistributionImbalanceRatio(distribution map[string]any) float64 {
	if len(distribution) == 0 {
		return 0
	}
	minCount := math.MaxFloat64
	maxCount := 0.0
	for _, value := range distribution {
		count := payloadNumber(value)
		if count <= 0 {
			continue
		}
		if count < minCount {
			minCount = count
		}
		if count > maxCount {
			maxCount = count
		}
	}
	if minCount == math.MaxFloat64 || minCount <= 0 {
		return 0
	}
	return maxCount / minCount
}

func profileHasBBoxEvidence(profile map[string]any) bool {
	if profileBool(profile, "bbox_available") || profileBool(profile, "annotations_available") ||
		profileInt(profile, "bbox_annotations_count") > 0 || payloadNumber(profile["bbox_count"]) > 0 ||
		artifactCountsHaveBBoxEvidence(profileMap(profile, "artifact_counts")) {
		return true
	}
	metadata := profileMap(profile, "metadata_summary")
	if metadataSummaryHasBBoxEvidence(metadata) {
		return true
	}
	if metadataSummaryHasBBoxEvidence(profileMap(profile, "agent_safe_metadata_summary")) ||
		metadataSummaryHasBBoxEvidence(profileMap(profile, "normalized_metadata_summary")) {
		return true
	}
	visualTraits := profileMap(profile, "visual_trait_summary")
	if payloadNumber(visualTraits["bbox_count"]) > 0 {
		return true
	}
	traits := profileStringSlice(profile, "dataset_traits")
	for _, trait := range traits {
		if containsAnyText(strings.ToLower(trait), "bbox", "bounding box", "annotation") {
			return true
		}
	}
	for _, artifact := range profileMapSlice(profile, "artifacts") {
		blob, _ := json.Marshal(artifact)
		if containsAnyText(strings.ToLower(string(blob)), "bbox", "bounding_box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func metadataSummaryHasBBoxEvidence(summary map[string]any) bool {
	if len(summary) == 0 {
		return false
	}
	if metadataBool(summary, "bbox_available") || metadataBool(summary, "annotations_available") ||
		payloadNumber(summary["bbox_annotations_count"]) > 0 ||
		payloadNumber(summary["bbox_annotation_count"]) > 0 ||
		payloadNumber(summary["bbox_sample_count"]) > 0 ||
		payloadNumber(summary["bbox_count"]) > 0 ||
		payloadNumber(summary["bbox_coverage_ratio"]) > 0 ||
		artifactCountsHaveBBoxEvidence(profileMap(summary, "artifact_counts")) {
		return true
	}
	annotationCounts := profileMap(summary, "annotation_counts")
	if payloadNumber(annotationCounts["bbox"]) > 0 || payloadNumber(annotationCounts["bounding_box"]) > 0 {
		return true
	}
	capabilities := profileMap(summary, "capabilities")
	if metadataBool(capabilities, "bbox") || metadataBool(capabilities, "bbox_annotations") ||
		metadataBool(capabilities, "bbox_crop") || metadataBool(capabilities, "object_detection") {
		return true
	}
	return false
}

func artifactCountsHaveBBoxEvidence(counts map[string]any) bool {
	for key, value := range counts {
		if payloadNumber(value) <= 0 {
			continue
		}
		if containsAnyText(strings.ToLower(strings.TrimSpace(key)), "bbox", "bounding_box", "bounding box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func profileOrDiagnosisHasResolutionCropEvidence(profile map[string]any, evidenceText string) bool {
	if containsAnyText(
		evidenceText,
		"small object",
		"small_objects",
		"large object",
		"large_objects",
		"object scale",
		"fine-grained",
		"fine grained",
		"fine_grained_similarity",
		"crop mismatch",
		"crop_bbox_useful",
		"background dominance",
		"background_dominance",
		"aspect ratio",
		"variable dimensions",
		"image dimension",
		"orientation_sensitive",
		"domain_shift_possible",
		"color_texture_signal",
	) {
		return true
	}
	traits := profileStringSlice(profile, "dataset_traits")
	for _, trait := range traits {
		if containsAnyText(strings.ToLower(trait), "small object", "object scale", "fine-grained", "fine grained", "background", "crop", "aspect", "variable dimension", "high resolution") {
			return true
		}
	}
	widthMin := profileInt(profile, "width_min")
	widthMax := profileInt(profile, "width_max")
	heightMin := profileInt(profile, "height_min")
	heightMax := profileInt(profile, "height_max")
	if widthMax >= 512 || heightMax >= 512 {
		return true
	}
	if widthMin > 0 && widthMax > widthMin*2 {
		return true
	}
	if heightMin > 0 && heightMax > heightMin*2 {
		return true
	}
	return false
}

func payloadNumber(value any) float64 {
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case float64:
		return typed
	case float32:
		return float64(typed)
	case json.Number:
		out, _ := typed.Float64()
		return out
	default:
		return 0
	}
}

func nonDefaultText(value string, defaults ...string) bool {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return false
	}
	for _, fallback := range defaults {
		if normalized == strings.ToLower(strings.TrimSpace(fallback)) {
			return false
		}
	}
	return true
}

func containsAnyText(value string, needles ...string) bool {
	value = strings.ToLower(value)
	for _, needle := range needles {
		if strings.Contains(value, strings.ToLower(strings.TrimSpace(needle))) {
			return true
		}
	}
	return false
}

func validatePlannedExperiment(experiment plans.PlannedExperiment, index int) error {
	if strings.TrimSpace(experiment.Template) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing template", store.ErrInvalidRequest, index)
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		if !labelQualityAuditExperiment(experiment) {
			return fmt.Errorf("%w: proposed experiment %d template %s requires mechanism label_noise_audit or hard_example_audit", store.ErrInvalidRequest, index, jobs.TemplateLabelQualityAudit)
		}
		if strings.TrimSpace(experiment.Intervention) == "" {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing intervention", store.ErrInvalidRequest, index)
		}
		if strings.TrimSpace(experiment.ExpectedEffect) == "" {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing expected_effect", store.ErrInvalidRequest, index)
		}
		if len(nonEmptyStringValues(experiment.EvidenceUsed)) == 0 {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing evidence_used", store.ErrInvalidRequest, index)
		}
		return nil
	}
	if disallowedDirectExperimentTemplate(experiment.Template) {
		return fmt.Errorf("%w: proposed experiment %d uses control-plane job template %q; experiments cannot directly schedule backend-owned worker jobs", store.ErrInvalidRequest, index, experiment.Template)
	}
	if strings.TrimSpace(experiment.Model) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing model", store.ErrInvalidRequest, index)
	}
	if !supportedModelNames()[strings.ToLower(strings.TrimSpace(experiment.Model))] {
		return fmt.Errorf("%w: proposed experiment %d uses unsupported model %q", store.ErrInvalidRequest, index, experiment.Model)
	}
	if experiment.Epochs < 1 || experiment.Epochs > 100 {
		return fmt.Errorf("%w: proposed experiment %d must have 1-100 epochs", store.ErrInvalidRequest, index)
	}
	if experiment.BatchSize < 1 || experiment.BatchSize > 512 {
		return fmt.Errorf("%w: proposed experiment %d must have batch_size 1-512", store.ErrInvalidRequest, index)
	}
	if experiment.LearningRate <= 0 || experiment.LearningRate > 1 {
		return fmt.Errorf("%w: proposed experiment %d must have learning_rate in (0, 1]", store.ErrInvalidRequest, index)
	}
	if experiment.ImageSize < 0 || experiment.ImageSize > 1024 {
		return fmt.Errorf("%w: proposed experiment %d image_size must be at most 1024", store.ErrInvalidRequest, index)
	}
	if experiment.ResolutionStrategy != "" && !allowedExperimentValue(experiment.ResolutionStrategy, allowedResolutionStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported resolution_strategy %q", store.ErrInvalidRequest, index, experiment.ResolutionStrategy)
	}
	if experiment.Preprocessing != nil {
		if err := validatePreprocessingConfig(*experiment.Preprocessing, index); err != nil {
			return err
		}
	}
	if experiment.Optimizer != "" && !allowedExperimentValue(experiment.Optimizer, allowedOptimizers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported optimizer %q", store.ErrInvalidRequest, index, experiment.Optimizer)
	}
	if experiment.Scheduler != "" && !allowedExperimentValue(experiment.Scheduler, allowedSchedulers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported scheduler %q", store.ErrInvalidRequest, index, experiment.Scheduler)
	}
	if experiment.WeightDecay < 0 || experiment.WeightDecay > 1 {
		return fmt.Errorf("%w: proposed experiment %d weight_decay must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	if experiment.Dropout < 0 || experiment.Dropout > 0.7 {
		return fmt.Errorf("%w: proposed experiment %d dropout must be between 0 and 0.7", store.ErrInvalidRequest, index)
	}
	if experiment.OptimizerMomentum < 0 || experiment.OptimizerMomentum > 0.99 {
		return fmt.Errorf("%w: proposed experiment %d optimizer_momentum must be between 0 and 0.99", store.ErrInvalidRequest, index)
	}
	if experiment.OptimizerMomentum > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		return fmt.Errorf("%w: proposed experiment %d optimizer_momentum is only supported with optimizer sgd", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerStepSize < 0 || experiment.SchedulerStepSize > 100 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_step_size must be between 1 and 100 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerStepSize > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		return fmt.Errorf("%w: proposed experiment %d scheduler_step_size is only supported with scheduler step", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma < 0 || experiment.SchedulerGamma > 0.95 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma must be between 0.05 and 0.95 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma > 0 && experiment.SchedulerGamma < 0.05 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma must be between 0.05 and 0.95 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma is only supported with scheduler step", store.ErrInvalidRequest, index)
	}
	if experiment.LabelSmoothing < 0 || experiment.LabelSmoothing > 0.3 {
		return fmt.Errorf("%w: proposed experiment %d label_smoothing must be between 0 and 0.3", store.ErrInvalidRequest, index)
	}
	if experiment.GradientClipNorm < 0 || experiment.GradientClipNorm > 10 {
		return fmt.Errorf("%w: proposed experiment %d gradient_clip_norm must be between 0 and 10", store.ErrInvalidRequest, index)
	}
	if err := validateAugmentationConfig(experiment.Augmentation, index); err != nil {
		return err
	}
	if experiment.AugmentationPolicy != "" && !allowedExperimentValue(experiment.AugmentationPolicy, allowedAugmentationPolicies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation_policy %q", store.ErrInvalidRequest, index, experiment.AugmentationPolicy)
	}
	if experiment.AugmentationPolicyConfig != nil {
		if err := validateAugmentationPolicyConfig(*experiment.AugmentationPolicyConfig, index); err != nil {
			return err
		}
	}
	if experiment.ClassBalancing != "" && !allowedExperimentValue(experiment.ClassBalancing, allowedClassBalancingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported class_balancing %q", store.ErrInvalidRequest, index, experiment.ClassBalancing)
	}
	if err := validateClassBalancingConfig(experiment.ClassBalancing, experiment.ClassBalancingConfig, index); err != nil {
		return err
	}
	if experiment.SamplingStrategy != "" && !allowedExperimentValue(experiment.SamplingStrategy, allowedSamplingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported sampling_strategy %q", store.ErrInvalidRequest, index, experiment.SamplingStrategy)
	}
	if experiment.EarlyStoppingPatience < 0 || experiment.EarlyStoppingPatience > 50 {
		return fmt.Errorf("%w: proposed experiment %d early_stopping_patience must be between 0 and 50", store.ErrInvalidRequest, index)
	}
	if experiment.FineTuneStrategy != "" {
		switch strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)) {
		case "head_only", "last_block", "full":
		default:
			return fmt.Errorf("%w: proposed experiment %d has unsupported fine_tune_strategy %q", store.ErrInvalidRequest, index, experiment.FineTuneStrategy)
		}
	}
	if err := validateExperimentAutoML(experiment, index); err != nil {
		return err
	}
	return nil
}

func validateExperimentAutoML(experiment plans.PlannedExperiment, index int) error {
	if experiment.AutoML == nil || !experiment.AutoML.Enabled {
		return nil
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return fmt.Errorf("%w: proposed experiment %d cannot use AutoML for report-only label-quality audit jobs", store.ErrInvalidRequest, index)
	}
	strategy := automlStrategyContext(experiment)
	if experiment.AutoML.SearchSpace != nil && len(experiment.AutoML.SearchSpace.Parameters) > 0 {
		if err := automl.ValidateSearchSpace(*experiment.AutoML.SearchSpace, strategy); err != nil {
			return fmt.Errorf("%w: proposed experiment %d has invalid AutoML search space: %s", store.ErrInvalidRequest, index, err.Error())
		}
		return nil
	}
	if len(experiment.AutoML.Intent.AllowedParameters) == 0 {
		return fmt.Errorf("%w: proposed experiment %d AutoML requires a search_space or intent.allowed_parameters", store.ErrInvalidRequest, index)
	}
	if _, err := automl.DefaultSearchSpace(experiment.AutoML.Intent.AllowedParameters, strategy); err != nil {
		return fmt.Errorf("%w: proposed experiment %d has invalid AutoML intent: %s", store.ErrInvalidRequest, index, err.Error())
	}
	return nil
}

func validateClassBalancingConfig(strategy string, config map[string]any, index int) error {
	if len(config) == 0 {
		return nil
	}
	for key, value := range config {
		normalized := strings.ToLower(strings.TrimSpace(key))
		switch normalized {
		case "effective_number_beta":
			if !effectiveNumberClassBalancing(strategy) {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.effective_number_beta is only supported with effective_number_loss class balancing", store.ErrInvalidRequest, index)
			}
			beta := payloadNumber(value)
			if beta < 0.9 || beta > 0.99999 {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.effective_number_beta must be between 0.9 and 0.99999", store.ErrInvalidRequest, index)
			}
		case "focal_loss_gamma":
			if !strings.EqualFold(strings.TrimSpace(strategy), "focal_loss") {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.focal_loss_gamma is only supported with focal_loss class balancing", store.ErrInvalidRequest, index)
			}
			gamma := payloadNumber(value)
			if gamma < 0.5 || gamma > 5 {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.focal_loss_gamma must be between 0.5 and 5", store.ErrInvalidRequest, index)
			}
		default:
			return fmt.Errorf("%w: proposed experiment %d has unsupported class_balancing_config key %q", store.ErrInvalidRequest, index, key)
		}
	}
	return nil
}

func disallowedDirectExperimentTemplate(template string) bool {
	switch strings.ToLower(strings.TrimSpace(template)) {
	case jobs.TemplateProfileDataset,
		jobs.TemplateTrainExperiment,
		jobs.TemplateExportChampion,
		jobs.TemplateChampionDemoPrediction,
		jobs.TemplateGenerateVisualExemplars,
		jobs.TemplateAnalyzeDatasetVisuals:
		return true
	default:
		return false
	}
}

func effectiveNumberClassBalancing(strategy string) bool {
	switch strings.ToLower(strings.TrimSpace(strategy)) {
	case "effective_number", "effective_number_loss", "effective_number_class_balanced_loss", "class_balanced_effective_number":
		return true
	default:
		return false
	}
}

func validatePreprocessingConfig(preprocessing plans.Preprocessing, index int) error {
	if preprocessing.ResizeStrategy != "" && !allowedExperimentValue(preprocessing.ResizeStrategy, allowedResizeStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.resize_strategy %q", store.ErrInvalidRequest, index, preprocessing.ResizeStrategy)
	}
	if preprocessing.Normalization != "" && !allowedExperimentValue(preprocessing.Normalization, allowedNormalizations()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.normalization %q", store.ErrInvalidRequest, index, preprocessing.Normalization)
	}
	if preprocessing.CropStrategy != "" && !allowedExperimentValue(preprocessing.CropStrategy, allowedCropStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.crop_strategy %q", store.ErrInvalidRequest, index, preprocessing.CropStrategy)
	}
	if preprocessing.BBoxMode != "" && !allowedExperimentValue(preprocessing.BBoxMode, allowedBBoxModes()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.bbox_mode %q", store.ErrInvalidRequest, index, preprocessing.BBoxMode)
	}
	return nil
}

func validateAugmentationConfig(augmentation map[string]any, index int) error {
	for key, value := range augmentation {
		normalized := strings.ToLower(strings.TrimSpace(key))
		if !allowedExperimentValue(normalized, allowedAugmentationKeys()) {
			return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation key %q", store.ErrInvalidRequest, index, key)
		}
		switch value.(type) {
		case bool, int, int64, float64, string:
		default:
			return fmt.Errorf("%w: proposed experiment %d augmentation.%s must be a bool, number, or string", store.ErrInvalidRequest, index, key)
		}
	}
	return nil
}

func validateAugmentationPolicyConfig(config plans.AugmentationPolicyConfig, index int) error {
	policyType := strings.ToLower(strings.TrimSpace(config.PolicyType))
	if policyType == "" {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.policy_type is required", store.ErrInvalidRequest, index)
	}
	if !allowedExperimentValue(policyType, allowedStructuredAugmentationPolicyTypes()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation_policy_config.policy_type %q", store.ErrInvalidRequest, index, config.PolicyType)
	}
	if config.Magnitude < 0 || config.Magnitude > 15 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.magnitude must be between 0 and 15", store.ErrInvalidRequest, index)
	}
	if config.NumOps < 0 || config.NumOps > 3 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.num_ops must be between 0 and 3", store.ErrInvalidRequest, index)
	}
	if config.NumMagnitudeBins < 0 || config.NumMagnitudeBins > 31 || (config.NumMagnitudeBins > 0 && config.NumMagnitudeBins < 2) {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.num_magnitude_bins must be between 2 and 31 when set", store.ErrInvalidRequest, index)
	}
	if config.Probability < 0 || config.Probability > 1 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.probability must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	if config.Alpha < 0 || config.Alpha > 1 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.alpha must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	return nil
}

func allowedExperimentValue(value string, allowed map[string]bool) bool {
	return allowed[strings.ToLower(strings.TrimSpace(value))]
}

func allowedPlannerMechanisms() map[string]bool {
	return map[string]bool{
		"stop_select_champion":      true,
		"baseline_control":          true,
		"architecture_challenge":    true,
		"capacity_finetune":         true,
		"optimizer_scheduler":       true,
		"regularization":            true,
		"augmentation_basic":        true,
		"augmentation_auto":         true,
		"augmentation_mixed_sample": true,
		"class_imbalance":           true,
		"minority_targeting":        true,
		"resolution_crop":           true,
		"bbox_crop_ablation":        true,
		"label_noise_audit":         true,
		"hard_example_audit":        true,
		"deployment_latency":        true,
		"distillation":              true,
	}
}

func allowedOptimizers() map[string]bool {
	return map[string]bool{"adamw": true, "adam": true, "sgd": true}
}

func allowedSchedulers() map[string]bool {
	return map[string]bool{"none": true, "cosine": true, "step": true}
}

func allowedResolutionStrategies() map[string]bool {
	return map[string]bool{
		"fixed":                    true,
		"low_latency":              true,
		"compare_224_256":          true,
		"high_resolution_ablation": true,
	}
}

func allowedResizeStrategies() map[string]bool {
	return map[string]bool{
		"squash":                 true,
		"preserve_aspect_pad":    true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
	}
}

func allowedNormalizations() map[string]bool {
	return map[string]bool{"imagenet": true, "dataset": true, "none": true}
}

func allowedCropStrategies() map[string]bool {
	return map[string]bool{
		"none":                   true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
		"bbox_crop_ablation":     true,
	}
}

func allowedBBoxModes() map[string]bool {
	return map[string]bool{
		"ignore":                      true,
		"crop_if_available":           true,
		"crop_and_compare_full_image": true,
		"use_boxes_as_metadata":       true,
	}
}

func allowedAugmentationPolicies() map[string]bool {
	return map[string]bool{
		"none":               true,
		"light":              true,
		"moderate":           true,
		"strong":             true,
		"custom":             true,
		"basic":              true,
		"randaugment":        true,
		"trivialaugment":     true,
		"trivialaugmentwide": true,
		"autoaugment":        true,
		"mixup":              true,
		"cutmix":             true,
	}
}

func allowedStructuredAugmentationPolicyTypes() map[string]bool {
	return map[string]bool{
		"none":               true,
		"basic":              true,
		"randaugment":        true,
		"trivialaugment":     true,
		"trivialaugmentwide": true,
		"autoaugment":        true,
		"mixup":              true,
		"cutmix":             true,
	}
}

func allowedAugmentationKeys() map[string]bool {
	return map[string]bool{
		"horizontal_flip": true,
		"vertical_flip":   true,
		"color_jitter":    true,
		"random_crop":     true,
		"random_rotation": true,
		"random_erasing":  true,
	}
}

func allowedClassBalancingStrategies() map[string]bool {
	return map[string]bool{
		"none":                                 true,
		"weighted_loss":                        true,
		"class_weighted_loss":                  true,
		"effective_number":                     true,
		"effective_number_loss":                true,
		"effective_number_class_balanced_loss": true,
		"class_balanced_effective_number":      true,
		"class_balanced_sampler":               true,
		"weighted_random_sampler":              true,
		"focal_loss":                           true,
	}
}

func allowedSamplingStrategies() map[string]bool {
	return map[string]bool{
		"none":                    true,
		"class_balanced_sampler":  true,
		"weighted_random_sampler": true,
	}
}

type visualExemplarCaps struct {
	MaxTotalImages int   `json:"max_total_images"`
	MaxPerClass    int   `json:"max_per_class"`
	MaxBytes       int64 `json:"max_bytes"`
}

func visualExemplarCapsFromQuery(c *gin.Context, defaultTotal int, defaultPerClass int, defaultBytes int64) visualExemplarCaps {
	caps := visualExemplarCaps{
		MaxTotalImages: defaultTotal,
		MaxPerClass:    defaultPerClass,
		MaxBytes:       defaultBytes,
	}
	caps.MaxTotalImages = clampQueryInt(c, "max_total_images", caps.MaxTotalImages, 1, 50)
	caps.MaxPerClass = clampQueryInt(c, "max_per_class", caps.MaxPerClass, 1, 8)
	caps.MaxBytes = int64(clampQueryInt(c, "max_bytes", int(caps.MaxBytes), 1, 5_000_000))
	return caps
}

func clampQueryInt(c *gin.Context, key string, fallback int, minValue int, maxValue int) int {
	raw := strings.TrimSpace(c.Query(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func cappedVisualExemplars(profile map[string]any, caps visualExemplarCaps, keys ...string) []datasets.VisualExemplar {
	out := []datasets.VisualExemplar{}
	perClass := map[string]int{}
	var totalBytes int64

	for _, key := range keys {
		entries := profileEntries(profile[key])
		if len(entries) == 0 {
			continue
		}
		for _, entry := range entries {
			exemplar, ok := visualExemplarFromProfileEntry(entry)
			if !ok || exemplar.URI == "" {
				continue
			}
			className := exemplar.ClassName
			if className == "" {
				className = exemplar.Label
			}
			if className == "" {
				className = "unknown"
			}
			if perClass[className] >= caps.MaxPerClass {
				continue
			}
			nextBytes := totalBytes + maxInt64(exemplar.SizeBytes, 0)
			if exemplar.SizeBytes > 0 && nextBytes > caps.MaxBytes {
				continue
			}
			exemplar.ClassName = className
			out = append(out, exemplar)
			perClass[className]++
			if exemplar.SizeBytes > 0 {
				totalBytes = nextBytes
			}
			if len(out) >= caps.MaxTotalImages {
				return out
			}
		}
	}
	return out
}

func profileEntries(value any) []any {
	switch entries := value.(type) {
	case []any:
		return entries
	case []map[string]any:
		out := make([]any, 0, len(entries))
		for _, entry := range entries {
			out = append(out, entry)
		}
		return out
	default:
		return nil
	}
}

func visualExemplarFromProfileEntry(entry any) (datasets.VisualExemplar, bool) {
	blob, err := json.Marshal(entry)
	if err != nil {
		return datasets.VisualExemplar{}, false
	}
	var exemplar datasets.VisualExemplar
	if err := json.Unmarshal(blob, &exemplar); err != nil {
		return datasets.VisualExemplar{}, false
	}
	if exemplar.URI == "" {
		var raw map[string]any
		if err := json.Unmarshal(blob, &raw); err == nil {
			exemplar.URI = firstString(raw, "image_uri", "url", "path", "storage_uri")
			if exemplar.ClassName == "" {
				exemplar.ClassName = firstString(raw, "class", "class_name", "label")
			}
		}
	}
	return exemplar, true
}

func championHeldoutDemoImageProfile(champion runs.ProjectChampion) map[string]any {
	out := map[string]any{}
	objective := payloadMap(champion.Evaluation, "objective_profile")
	for _, key := range []string{"heldout_demo_images", "demo_images", "test_images"} {
		if value, ok := objective[key]; ok {
			out[key] = value
		}
		if value, ok := champion.DeploymentProfile[key]; ok {
			out[key] = value
		}
	}
	return out
}

func testOnlyVisualExemplars(exemplars []datasets.VisualExemplar) []datasets.VisualExemplar {
	out := make([]datasets.VisualExemplar, 0, len(exemplars))
	for _, exemplar := range exemplars {
		split := strings.ToLower(strings.TrimSpace(exemplar.Split))
		if split == "test" || split == "heldout" || split == "holdout" {
			out = append(out, exemplar)
		}
	}
	return out
}

func championDemoImageMetadata(profile map[string]any, imageURI string) (string, string, map[string]any, bool) {
	imageURI = strings.TrimSpace(imageURI)
	if imageURI == "" {
		return "", "", nil, false
	}
	for _, key := range []string{"demo_images", "visual_exemplars", "exemplars"} {
		for _, entry := range profileEntries(profile[key]) {
			exemplar, ok := visualExemplarFromProfileEntry(entry)
			if !ok || strings.TrimSpace(exemplar.URI) != imageURI {
				continue
			}
			trueLabel := exemplar.Label
			if trueLabel == "" {
				trueLabel = exemplar.ClassName
			}
			metadata := map[string]any{}
			for key, value := range exemplar.Metadata {
				metadata[key] = value
			}
			if exemplar.ClassName != "" {
				metadata["class_name"] = exemplar.ClassName
			}
			if exemplar.Label != "" {
				metadata["label"] = exemplar.Label
			}
			if exemplar.Split != "" {
				metadata["split"] = exemplar.Split
			}
			if exemplar.Width > 0 {
				metadata["width"] = exemplar.Width
			}
			if exemplar.Height > 0 {
				metadata["height"] = exemplar.Height
			}
			if exemplar.SizeBytes > 0 {
				metadata["size_bytes"] = exemplar.SizeBytes
			}
			if exemplar.MimeType != "" {
				metadata["mime_type"] = exemplar.MimeType
			}
			if exemplar.Description != "" {
				metadata["description"] = exemplar.Description
			}
			return exemplar.ID, trueLabel, metadata, true
		}
	}
	return "", "", nil, false
}

func normalizeChampionExportFormat(format string) string {
	format = strings.ToLower(strings.TrimSpace(format))
	if format == "" {
		format = "onnx"
	}
	switch format {
	case "onnx", "torchscript", "pytorch", "safetensors":
		return format
	default:
		return ""
	}
}

func championArtifactURI(deploymentProfile map[string]any) string {
	if artifactURI := firstString(deploymentProfile, "artifact_uri", "onnx_artifact_uri", "model_artifact_uri", "export_artifact_uri", "checkpoint_uri"); artifactURI != "" {
		return artifactURI
	}
	return championArtifactURIFromEvaluation(payloadMap(deploymentProfile, "model_profile"))
}

func championArtifactURIFromEvaluation(modelProfile map[string]any) string {
	return firstString(modelProfile, "onnx_artifact_uri", "artifact_uri", "model_artifact_uri", "export_artifact_uri", "checkpoint_uri")
}

func artifactMatchesChampionExportFormat(artifactURI string, format string) bool {
	normalized := strings.ToLower(strings.TrimSpace(artifactURI))
	switch format {
	case "onnx":
		return strings.HasSuffix(normalized, ".onnx")
	case "torchscript":
		return strings.HasSuffix(normalized, ".torchscript.pt") || strings.HasSuffix(normalized, ".torchscript")
	case "pytorch":
		return strings.HasSuffix(normalized, ".pt") || strings.HasSuffix(normalized, ".pth")
	case "safetensors":
		return strings.HasSuffix(normalized, ".safetensors")
	default:
		return false
	}
}

func championExportMetadata(champion runs.ProjectChampion, format string, requestMetadata map[string]any) map[string]any {
	metadata := map[string]any{
		"format":             format,
		"source_job_id":      champion.JobID,
		"selection_reason":   champion.SelectionReason,
		"metrics":            champion.Metrics,
		"evaluation":         champion.Evaluation,
		"deployment_profile": champion.DeploymentProfile,
	}
	for key, value := range requestMetadata {
		metadata[key] = value
	}
	return metadata
}

func normalizeChampionExportResultStatus(status string) string {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case runs.ChampionExportStatusReady:
		return runs.ChampionExportStatusReady
	case runs.ChampionExportStatusFailed:
		return runs.ChampionExportStatusFailed
	case runs.ChampionExportStatusPendingArtifact:
		return runs.ChampionExportStatusPendingArtifact
	default:
		return ""
	}
}

func normalizeChampionDemoPredictionResultStatus(status string) string {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case runs.ChampionDemoPredictionStatusSucceeded:
		return runs.ChampionDemoPredictionStatusSucceeded
	case runs.ChampionDemoPredictionStatusFailed:
		return runs.ChampionDemoPredictionStatusFailed
	case runs.ChampionDemoPredictionStatusRuntimeUnavailable:
		return runs.ChampionDemoPredictionStatusRuntimeUnavailable
	default:
		return ""
	}
}

func championExportManifestPath(metadata map[string]any) string {
	manifest, _ := metadata["manifest"].(map[string]any)
	if manifestPath := firstString(manifest, "manifest_path", "local_manifest_path"); manifestPath != "" {
		return manifestPath
	}
	return firstString(metadata, "manifest_path", "local_manifest_path", "export_manifest_path")
}

func championDatasetID(champion runs.ProjectChampion, job jobs.ExperimentJob) string {
	if champion.DatasetID != "" {
		return champion.DatasetID
	}
	return jobConfigString(job.Config, "dataset_id")
}

func usableChampionExport(dataStore store.Store, projectID string, championID string) (runs.ChampionExport, bool) {
	exports, err := dataStore.ListProjectChampionExports(projectID)
	if err != nil {
		return runs.ChampionExport{}, false
	}
	for _, export := range exports {
		if export.ChampionID == championID && export.Status == runs.ChampionExportStatusReady && strings.TrimSpace(export.ArtifactURI) != "" {
			return export, true
		}
	}
	return runs.ChampionExport{}, false
}

func latestVisualAnalysisFromList(analyses []datasets.DatasetVisualAnalysis) *datasets.DatasetVisualAnalysis {
	if len(analyses) == 0 {
		return nil
	}
	latest := analyses[0]
	for _, analysis := range analyses[1:] {
		if analysis.AnalysisVersion > latest.AnalysisVersion ||
			(analysis.AnalysisVersion == latest.AnalysisVersion && analysis.CreatedAt.After(latest.CreatedAt)) {
			latest = analysis
		}
	}
	return &latest
}

func (s *Server) maybeQueueInitialDatasetVisualAnalysis(datasetID string) (bool, error) {
	if !visualAnalysisAutomationEnabled() {
		return false, nil
	}
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return false, err
	}
	if dataset.Status != datasets.StatusProfiled {
		return false, nil
	}
	if _, err := s.store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID); err == nil {
		return false, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	policy, err := s.datasetVisualAnalysisRunPolicy(dataset, datasets.VisualTriggerInitialProfile, visualAnalysisDeficiencyAssessment{})
	if err != nil {
		return false, err
	}
	if policy.ActiveJobID != "" {
		return true, nil
	}
	if !policy.RunAllowed {
		return false, nil
	}
	_, _, err = s.queueDatasetVisualAnalysis(dataset, datasets.VisualTriggerInitialProfile, map[string]any{
		"source": "dataset_profile_update",
	}, "", 0, 0)
	return err == nil, err
}

func (s *Server) maybeQueueDeficiencyDatasetVisualAnalysis(evaluation runs.TrainingRunEvaluation) error {
	if !visualAnalysisAutomationEnabled() {
		return nil
	}
	if strings.TrimSpace(evaluation.DatasetID) == "" {
		return nil
	}
	dataset, err := s.store.GetDataset(evaluation.DatasetID)
	if err != nil {
		return err
	}
	assessment, err := s.assessDatasetVisualDeficiency(dataset, evaluation)
	if err != nil {
		return err
	}
	policy, err := s.datasetVisualAnalysisRunPolicy(dataset, datasets.VisualTriggerDeficiencyReanalysis, assessment)
	if err != nil {
		return err
	}
	if !policy.RunAllowed {
		return nil
	}
	triggerDetails := copyPayloadMap(assessment.Details)
	triggerDetails["source"] = "training_run_evaluation"
	triggerDetails["job_id"] = evaluation.JobID
	triggerDetails["plan_id"] = evaluation.PlanID
	triggerDetails["deficiency_triggers"] = assessment.Triggers
	triggerDetails["deficiency_severity"] = assessment.Severity
	triggerDetails["cooldown_seconds"] = policy.CooldownSeconds
	triggerDetails["runs_for_profile"] = policy.RunsForProfile
	_, _, err = s.queueDatasetVisualAnalysis(dataset, datasets.VisualTriggerDeficiencyReanalysis, triggerDetails, "", 0, 0)
	return err
}

func (s *Server) datasetVisualAnalysisRunPolicy(dataset datasets.Dataset, requestedTrigger datasets.VisualReanalysisTrigger, assessment visualAnalysisDeficiencyAssessment) (datasetVisualAnalysisRerunPolicy, error) {
	cooldown := visualAnalysisCooldownDuration()
	maxRuns := visualAnalysisMaxRunsPerProfile()
	profileFingerprint := datasetProfileFingerprint(dataset.Profile)
	policy := datasetVisualAnalysisRerunPolicy{
		Enabled:              true,
		AutomationEnabled:    visualAnalysisAutomationEnabled(),
		CooldownSeconds:      int(cooldown.Seconds()),
		MaxRunsPerProfile:    maxRuns,
		ProfileFingerprint:   profileFingerprint,
		DeficiencyTriggers:   append([]string(nil), assessment.Triggers...),
		DeficiencySeverity:   roundDiagnosticFloat(assessment.Severity),
		ManualRunAllowed:     true,
		InitialRunAllowed:    true,
		DeficiencyRunAllowed: assessment.Eligible,
	}

	if dataset.Status != datasets.StatusProfiled {
		policy.disableAll("dataset must be profiled before visual analysis can run")
		return policy.withRequestedTrigger(requestedTrigger), nil
	}

	analyses, err := s.store.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		return policy, err
	}
	var latest *datasets.DatasetVisualAnalysis
	currentProfileRuns := 0
	currentProfileAcceptedRuns := 0
	hasAcceptedForCurrentProfile := false
	for _, analysis := range analyses {
		if !visualAnalysisMatchesProfileFingerprint(analysis.ProfileFingerprint, profileFingerprint) {
			continue
		}
		currentProfileRuns++
		if analysis.ValidationStatus == datasets.VisualValidationStatusAccepted {
			currentProfileAcceptedRuns++
			hasAcceptedForCurrentProfile = true
		}
		if latest == nil || analysis.CreatedAt.After(latest.CreatedAt) {
			copy := analysis
			latest = &copy
		}
	}
	policy.RunsForProfile = currentProfileRuns
	policy.AcceptedRunsForProfile = currentProfileAcceptedRuns
	if latest != nil {
		policy.LatestAnalysisID = latest.ID
		createdAt := latest.CreatedAt
		policy.LatestAnalysisCreatedAt = &createdAt
		policy.LatestAnalysisValidation = latest.ValidationStatus
	}

	activeJob, hasActiveJob, err := s.activeDatasetVisualAnalysisJob(dataset)
	if err != nil {
		return policy, err
	}
	if hasActiveJob {
		policy.ActiveJobID = activeJob.ID
		policy.ActiveJobStatus = activeJob.Status
		policy.disableAll(fmt.Sprintf("visual analysis job %s is already %s", activeJob.ID, strings.ToLower(activeJob.Status)))
		return policy.withRequestedTrigger(requestedTrigger), nil
	}

	if maxRuns > 0 && currentProfileRuns >= maxRuns {
		policy.disableAll(fmt.Sprintf("visual analysis has reached the per-profile cap of %d run(s)", maxRuns))
		return policy.withRequestedTrigger(requestedTrigger), nil
	}

	if latest != nil && cooldown > 0 {
		nextAllowedAt := latest.CreatedAt.Add(cooldown)
		if time.Now().UTC().Before(nextAllowedAt) {
			policy.NextAllowedAt = &nextAllowedAt
			policy.disableAll(fmt.Sprintf("visual analysis rerun is cooling down until %s", nextAllowedAt.UTC().Format(time.RFC3339)))
			return policy.withRequestedTrigger(requestedTrigger), nil
		}
	}

	if hasAcceptedForCurrentProfile {
		policy.InitialRunAllowed = false
	}
	if !policy.AutomationEnabled {
		policy.InitialRunAllowed = false
		policy.DeficiencyRunAllowed = false
	}
	if !assessment.Eligible {
		policy.DeficiencyRunAllowed = false
		if len(assessment.Triggers) == 0 {
			policy.Reason = "No severe visual-analysis deficiency trigger is currently active."
		}
	}

	return policy.withRequestedTrigger(requestedTrigger), nil
}

func (policy datasetVisualAnalysisRerunPolicy) withRequestedTrigger(trigger datasets.VisualReanalysisTrigger) datasetVisualAnalysisRerunPolicy {
	switch trigger {
	case datasets.VisualTriggerInitialProfile:
		policy.RunAllowed = policy.InitialRunAllowed
		if !policy.RunAllowed && policy.DisabledReason == "" {
			policy.DisabledReason = "initial visual analysis is not allowed for the current profile"
		}
	case datasets.VisualTriggerDeficiencyReanalysis:
		policy.RunAllowed = policy.DeficiencyRunAllowed
		if !policy.RunAllowed && policy.DisabledReason == "" {
			policy.DisabledReason = "deficiency visual reanalysis requires severe post-training evidence and enabled automation"
		}
	default:
		policy.RunAllowed = policy.ManualRunAllowed
		if !policy.RunAllowed && policy.DisabledReason == "" {
			policy.DisabledReason = "manual visual analysis rerun is not currently allowed"
		}
	}
	return policy
}

func (policy *datasetVisualAnalysisRerunPolicy) disableAll(reason string) {
	policy.ManualRunAllowed = false
	policy.InitialRunAllowed = false
	policy.DeficiencyRunAllowed = false
	policy.RunAllowed = false
	policy.DisabledReason = reason
}

func (s *Server) activeDatasetVisualAnalysisJob(dataset datasets.Dataset) (jobs.ExperimentJob, bool, error) {
	projectJobs, err := s.store.ListProjectJobs(dataset.ProjectID)
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	for _, job := range projectJobs {
		if job.Template != jobs.TemplateAnalyzeDatasetVisuals {
			continue
		}
		switch job.Status {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
		default:
			continue
		}
		if jobConfigString(job.Config, "dataset_id") == dataset.ID {
			return job, true, nil
		}
	}
	return jobs.ExperimentJob{}, false, nil
}

func visualAnalysisMatchesProfileFingerprint(analysisFingerprint string, currentFingerprint string) bool {
	analysisFingerprint = strings.TrimSpace(analysisFingerprint)
	currentFingerprint = strings.TrimSpace(currentFingerprint)
	return analysisFingerprint == "" || analysisFingerprint == currentFingerprint
}

func visualAnalysisAutomationEnabled() bool {
	if value, ok := envFlagValue("MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED"); ok {
		return value
	}
	return envFlag("MODEL_EXPRESS_VISUAL_LLM_ENABLED", false) || strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY")) != ""
}

func visualAnalysisCooldownDuration() time.Duration {
	minutes := envInt("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", visualAnalysisDefaultCooldownMinutes, 0, 24*60*30)
	return time.Duration(minutes) * time.Minute
}

func visualAnalysisMaxRunsPerProfile() int {
	return envInt("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", visualAnalysisDefaultMaxRunsPerProfile, 1, 20)
}

func (s *Server) assessDatasetVisualDeficiency(dataset datasets.Dataset, evaluation runs.TrainingRunEvaluation) (visualAnalysisDeficiencyAssessment, error) {
	assessment := visualAnalysisDeficiencyAssessment{
		Details: map[string]any{
			"dataset_id":          dataset.ID,
			"profile_fingerprint": datasetProfileFingerprint(dataset.Profile),
		},
	}
	project, err := s.store.GetProject(dataset.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return assessment, err
	}
	targetMetric := "macro_f1"
	if evaluation.PlanID != "" {
		if plan, planErr := s.store.GetExperimentPlan(evaluation.PlanID); planErr == nil && plan.TargetMetric != "" {
			targetMetric = plan.TargetMetric
		} else if planErr != nil && !errors.Is(planErr, store.ErrNotFound) {
			return assessment, planErr
		}
	}
	if metric := payloadString(evaluation.ObjectiveProfile, "target_metric"); metric != "" {
		targetMetric = metric
	}
	assessment.Details["target_metric"] = normalizedPlannerTargetMetric(targetMetric)

	summary, hasSummary, err := s.trainingSummaryForVisualDeficiency(evaluation)
	if err != nil {
		return assessment, err
	}
	if hasSummary {
		macroThreshold := envFloat("MODEL_EXPRESS_VISUAL_ANALYSIS_LOW_MACRO_F1_THRESHOLD", visualAnalysisDefaultLowMacroF1Threshold, 0.05, 0.95)
		if summary.BestMacroF1 > 0 && summary.BestMacroF1 < macroThreshold {
			severity := 0.65
			if summary.BestMacroF1 < macroThreshold-0.10 {
				severity = 0.82
			}
			assessment.addTrigger("low_macro_f1", severity, fmt.Sprintf("best_macro_f1 %.3f is below %.3f", summary.BestMacroF1, macroThreshold))
		}
		if summary.BestAccuracy > 0 && summary.BestMacroF1 > 0 && summary.BestAccuracy-summary.BestMacroF1 >= 0.12 {
			assessment.addTrigger("accuracy_macro_f1_gap", 0.68, fmt.Sprintf("accuracy %.3f exceeds macro-F1 %.3f by %.3f", summary.BestAccuracy, summary.BestMacroF1, summary.BestAccuracy-summary.BestMacroF1))
		}
		assessment.Details["best_macro_f1"] = summary.BestMacroF1
		assessment.Details["best_accuracy"] = summary.BestAccuracy
	}

	worstLabel, worstRecall := worstPerClassRecall(evaluation.PerClassMetrics)
	if worstLabel != "" {
		assessment.Details["worst_class"] = worstLabel
		assessment.Details["worst_class_recall"] = worstRecall
		recallThreshold := envFloat("MODEL_EXPRESS_VISUAL_ANALYSIS_WORST_RECALL_THRESHOLD", visualAnalysisDefaultWorstRecallThreshold, 0.05, 0.95)
		if worstRecall > 0 && worstRecall < recallThreshold {
			severity := 0.67
			if worstRecall < recallThreshold-0.15 {
				severity = 0.86
			}
			assessment.addTrigger("worst_class_recall_failure", severity, fmt.Sprintf("worst class %s recall/F1 %.3f is below %.3f", worstLabel, worstRecall, recallThreshold))
		}
	}

	if pair, ratio := topConfusionPairRatio(evaluation.ConfusionMatrix); pair != "" {
		assessment.Details["top_confusion_pair"] = pair
		assessment.Details["top_confusion_ratio"] = ratio
		confusionThreshold := envFloat("MODEL_EXPRESS_VISUAL_ANALYSIS_TOP_CONFUSION_THRESHOLD", visualAnalysisDefaultConfusionThreshold, 0.05, 0.95)
		if ratio >= confusionThreshold {
			assessment.addTrigger("persistent_top_confusion", 0.72, fmt.Sprintf("top confusion %s accounts for %.3f of evaluated samples", pair, ratio))
		}
	}

	if payloadBool(evaluation.HolisticScores, "visual_hypotheses_contradicted") || len(payloadStringSlice(evaluation.HolisticScores, "contradicted_visual_hypotheses")) > 0 {
		assessment.addTrigger("contradicted_visual_hypotheses", 0.82, "training evaluation contradicted one or more accepted visual hypotheses")
	}

	if dataset.ProjectID != "" {
		projectPlans, planErr := s.store.ListProjectExperimentPlans(dataset.ProjectID)
		if planErr != nil && !errors.Is(planErr, store.ErrNotFound) {
			return assessment, planErr
		}
		summaries, summaryErr := s.store.ListProjectTrainingRunSummaries(dataset.ProjectID)
		if summaryErr != nil && !errors.Is(summaryErr, store.ErrNotFound) {
			return assessment, summaryErr
		}
		evaluations, evalErr := s.store.ListProjectTrainingRunEvaluations(dataset.ProjectID)
		if evalErr != nil && !errors.Is(evalErr, store.ErrNotFound) {
			return assessment, evalErr
		}
		if len(projectPlans) > 0 && len(summaries) > 0 {
			_, _, _, noImprovementRounds, _ := experimentPlannerPerformanceContext(targetMetric, projectPlans, summaries, evaluations, projectObjectiveContext(project.Goal), evaluation.PlanID)
			assessment.Details["no_improvement_rounds"] = noImprovementRounds
			if noImprovementRounds >= plannerNoImprovementRoundsToSelect {
				assessment.addTrigger("repeated_no_improvement_rounds", 0.88, fmt.Sprintf("%d consecutive completed follow-up plan(s) did not improve the champion meaningfully", noImprovementRounds))
			}
		}
	}

	assessment.Triggers = uniqueStrings(assessment.Triggers)
	assessment.Eligible = assessment.Severity >= 0.75 || len(assessment.Triggers) >= 2
	assessment.Details["deficiency_triggers"] = assessment.Triggers
	assessment.Details["deficiency_severity"] = roundDiagnosticFloat(assessment.Severity)
	assessment.Details["eligible"] = assessment.Eligible
	return assessment, nil
}

func (assessment *visualAnalysisDeficiencyAssessment) addTrigger(trigger string, severity float64, detail string) {
	if strings.TrimSpace(trigger) == "" {
		return
	}
	assessment.Triggers = append(assessment.Triggers, strings.TrimSpace(trigger))
	if assessment.Details == nil {
		assessment.Details = map[string]any{}
	}
	if detail != "" {
		assessment.Details[trigger] = detail
	}
	if severity > assessment.Severity {
		assessment.Severity = severity
	}
}

func (s *Server) trainingSummaryForVisualDeficiency(evaluation runs.TrainingRunEvaluation) (runs.TrainingRunSummary, bool, error) {
	if evaluation.JobID == "" {
		return runs.TrainingRunSummary{}, false, nil
	}
	summary, err := s.store.GetTrainingRunSummary(evaluation.JobID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return runs.TrainingRunSummary{}, false, nil
		}
		return runs.TrainingRunSummary{}, false, err
	}
	return summary, true, nil
}

func worstPerClassRecall(metrics map[string]any) (string, float64) {
	worstLabel := ""
	worstRecall := 0.0
	for label, value := range metrics {
		normalizedLabel := strings.ToLower(strings.TrimSpace(label))
		if normalizedLabel == "" || normalizedLabel == "accuracy" || strings.Contains(normalizedLabel, "avg") {
			continue
		}
		stats := mapFromAny(value)
		recall := payloadFloat(stats, "recall")
		if recall <= 0 {
			recall = payloadFloat(stats, "f1-score")
		}
		if recall <= 0 {
			recall = payloadFloat(stats, "f1")
		}
		if recall <= 0 {
			continue
		}
		if worstLabel == "" || recall < worstRecall {
			worstLabel = label
			worstRecall = recall
		}
	}
	return worstLabel, worstRecall
}

func topConfusionPairRatio(matrix [][]int) (string, float64) {
	total := 0
	topCount := 0
	topI := -1
	topJ := -1
	for i, row := range matrix {
		for j, count := range row {
			if count < 0 {
				continue
			}
			total += count
			if i != j && count > topCount {
				topCount = count
				topI = i
				topJ = j
			}
		}
	}
	if total <= 0 || topCount <= 0 || topI < 0 || topJ < 0 {
		return "", 0
	}
	return fmt.Sprintf("%d->%d", topI, topJ), float64(topCount) / float64(total)
}

func (s *Server) queueDatasetVisualAnalysis(dataset datasets.Dataset, trigger datasets.VisualReanalysisTrigger, triggerDetails map[string]any, provider string, maxImages int, highDetailCap int) (jobs.ExperimentJob, execution.ExecutionEvent, error) {
	if !allowedVisualTrigger(trigger) {
		return jobs.ExperimentJob{}, execution.ExecutionEvent{}, fmt.Errorf("%w: unsupported visual analysis trigger_reason %q", store.ErrInvalidRequest, trigger)
	}
	if triggerDetails == nil {
		triggerDetails = map[string]any{}
	}
	if maxImages <= 0 {
		maxImages = 48
	}
	if maxImages > 64 {
		maxImages = 64
	}
	if highDetailCap <= 0 {
		highDetailCap = 6
	}
	if highDetailCap > 8 {
		highDetailCap = 8
	}
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" {
		provider = "local"
	}
	config := map[string]any{
		"dataset_id":             dataset.ID,
		"dataset_name":           dataset.Name,
		"trigger_reason":         string(trigger),
		"trigger_details":        triggerDetails,
		"provider":               provider,
		"max_total_images":       maxImages,
		"max_images":             maxImages,
		"max_high_detail_images": highDetailCap,
		"high_detail_cap":        highDetailCap,
		"profile_fingerprint":    datasetProfileFingerprint(dataset.Profile),
		"evidence_only":          true,
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return jobs.ExperimentJob{}, execution.ExecutionEvent{}, err
	}
	if len(metadataSummary) > 0 {
		config["agent_safe_metadata_summary"] = metadataSummary
		config["metadata_summary"] = metadataSummary
		if importID, _ := metadataSummary["import_id"].(string); strings.TrimSpace(importID) != "" {
			config["metadata_import_id"] = strings.TrimSpace(importID)
		}
	}
	job, err := s.ensureOpenJob(dataset.ProjectID, jobs.TemplateAnalyzeDatasetVisuals, config, func(job jobs.ExperimentJob) bool {
		return jobConfigString(job.Config, "dataset_id") == dataset.ID &&
			jobConfigString(job.Config, "trigger_reason") == string(trigger)
	})
	if err != nil {
		return jobs.ExperimentJob{}, execution.ExecutionEvent{}, err
	}
	event, err := s.store.CreateExecutionEvent(dataset.ProjectID, "", execution.EventDatasetVisualAnalysisQueued, "Dataset visual analysis queued.", map[string]any{
		"dataset_id":          dataset.ID,
		"job_id":              job.ID,
		"trigger_reason":      string(trigger),
		"max_images":          maxImages,
		"high_detail_cap":     highDetailCap,
		"profile_fingerprint": config["profile_fingerprint"],
		"evidence_only":       true,
	})
	if err != nil {
		return jobs.ExperimentJob{}, execution.ExecutionEvent{}, err
	}
	return job, event, nil
}

func validateDatasetVisualAnalysisResult(dataset datasets.Dataset, analysis *datasets.DatasetVisualAnalysis, manifest []datasets.VisualSampleManifestItem, rawOutput string, inputContext map[string]any) []string {
	validationErrors := []string{}
	if analysis.DatasetID != dataset.ID {
		validationErrors = append(validationErrors, "dataset_id must match request dataset")
	}
	if analysis.ProjectID != "" && analysis.ProjectID != dataset.ProjectID {
		validationErrors = append(validationErrors, "project_id must match dataset project")
	}
	if analysis.SchemaVersion != datasets.VisualAnalysisSchemaVersion {
		validationErrors = append(validationErrors, "schema_version must be dataset_visual_analysis_v1")
	}
	if !allowedVisualTrigger(analysis.TriggerReason) {
		validationErrors = append(validationErrors, "trigger_reason is unsupported")
	}
	if analysis.ImagesAnalyzed <= 0 {
		validationErrors = append(validationErrors, "images_analyzed must be positive")
	}
	if analysis.ImagesAnalyzed > 64 {
		validationErrors = append(validationErrors, "images_analyzed exceeds backend cap 64")
	}
	if len(manifest) == 0 {
		validationErrors = append(validationErrors, "sample_manifest is required")
	}
	if len(manifest) > 64 {
		validationErrors = append(validationErrors, "sample_manifest exceeds backend cap 64")
	}
	if analysis.ImagesAnalyzed > len(manifest) {
		validationErrors = append(validationErrors, "images_analyzed cannot exceed sample_manifest length")
	}
	if analysis.TotalImages < analysis.ImagesAnalyzed {
		validationErrors = append(validationErrors, "total_images cannot be smaller than images_analyzed")
	}
	if analysis.Confidence != "" && !allowedVisualConfidence(analysis.Confidence) {
		validationErrors = append(validationErrors, "confidence must be low, medium, or high")
	}
	imageIDs := map[string]bool{}
	for _, item := range manifest {
		imageID := strings.TrimSpace(item.ImageID)
		if imageID == "" {
			validationErrors = append(validationErrors, "sample_manifest image_id is required")
			continue
		}
		if imageIDs[imageID] {
			validationErrors = append(validationErrors, "sample_manifest image_id values must be unique")
		}
		imageIDs[imageID] = true
	}
	validationErrors = append(validationErrors, validateVisualCoverageReport(analysis.CoverageReport, analysis.ImagesAnalyzed)...)
	for index, trait := range analysis.VisualTraits {
		if !allowedVisualTrait(trait.Trait) {
			validationErrors = append(validationErrors, fmt.Sprintf("visual_traits[%d].trait is unsupported", index))
		}
		if trait.Confidence != "" && !allowedVisualConfidence(trait.Confidence) {
			validationErrors = append(validationErrors, fmt.Sprintf("visual_traits[%d].confidence is unsupported", index))
		}
		validationErrors = append(validationErrors, validateExampleImageIDs(fmt.Sprintf("visual_traits[%d]", index), trait.ExampleImageIDs, imageIDs)...)
	}
	for index, item := range analysis.ClassesToWatch {
		if strings.TrimSpace(item.ClassName) == "" {
			validationErrors = append(validationErrors, fmt.Sprintf("classes_to_watch[%d].class_name is required", index))
		}
		if item.Confidence != "" && !allowedVisualConfidence(item.Confidence) {
			validationErrors = append(validationErrors, fmt.Sprintf("classes_to_watch[%d].confidence is unsupported", index))
		}
		validationErrors = append(validationErrors, validateExampleImageIDs(fmt.Sprintf("classes_to_watch[%d]", index), item.ExampleImageIDs, imageIDs)...)
	}
	for index := range analysis.PreprocessingHypotheses {
		validationErrors = append(validationErrors, validateVisualPreprocessingHypothesis(dataset.Profile, &analysis.PreprocessingHypotheses[index], index)...)
	}
	for index, caution := range analysis.Cautions {
		if caution.Confidence != "" && !allowedVisualConfidence(caution.Confidence) {
			validationErrors = append(validationErrors, fmt.Sprintf("cautions[%d].confidence is unsupported", index))
		}
		if caution.Severity != "" && !allowedVisualSeverity(caution.Severity) {
			validationErrors = append(validationErrors, fmt.Sprintf("cautions[%d].severity is unsupported", index))
		}
		validationErrors = append(validationErrors, validateExampleImageIDs(fmt.Sprintf("cautions[%d]", index), caution.ExampleImageIDs, imageIDs)...)
	}
	if reasons := unsafeVisualAnalysisContentReasons(rawOutput); len(reasons) > 0 {
		validationErrors = append(validationErrors, visualAnalysisUnsafeContentError("raw visual output contains forbidden paths or execution authority", reasons))
	}
	if manifestJSON, err := json.Marshal(manifest); err == nil {
		if reasons := unsafeVisualAnalysisContentReasons(string(manifestJSON)); len(reasons) > 0 {
			validationErrors = append(validationErrors, visualAnalysisUnsafeContentError("sample_manifest contains forbidden paths or image bytes", reasons))
		}
	}
	if inputContextJSON, err := json.Marshal(inputContext); err == nil {
		if reasons := unsafeVisualAnalysisContentReasons(string(inputContextJSON)); len(reasons) > 0 {
			validationErrors = append(validationErrors, visualAnalysisUnsafeContentError("input_context contains forbidden paths or image bytes", reasons))
		}
	}
	if analysisJSON, err := json.Marshal(analysis); err == nil {
		if reasons := unsafeVisualAnalysisContentReasons(string(analysisJSON)); len(reasons) > 0 {
			validationErrors = append(validationErrors, visualAnalysisUnsafeContentError("visual analysis contains forbidden paths or execution authority", reasons))
		}
	}
	return uniqueStrings(validationErrors)
}

func validateVisualCoverageReport(report datasets.VisualCoverageReport, imagesAnalyzed int) []string {
	validationErrors := []string{}
	if report.ImagesAnalyzed > 0 && report.ImagesAnalyzed != imagesAnalyzed {
		validationErrors = append(validationErrors, "coverage_report.images_analyzed must match images_analyzed")
	}
	if report.ImagesAvailable > 0 && report.ImagesAvailable < imagesAnalyzed {
		validationErrors = append(validationErrors, "coverage_report.images_available cannot be smaller than images_analyzed")
	}
	if report.ClassesCovered < 0 || report.ClassesTotal < 0 || report.ClassesCovered > report.ClassesTotal {
		validationErrors = append(validationErrors, "coverage_report class counts are inconsistent")
	}
	if report.ClassCoverageRatio < 0 || report.ClassCoverageRatio > 1 {
		validationErrors = append(validationErrors, "coverage_report.class_coverage_ratio must be between 0 and 1")
	}
	if report.HighDetailImageCount < 0 || report.HighDetailImageCount > imagesAnalyzed {
		validationErrors = append(validationErrors, "coverage_report.high_detail_image_count is out of bounds")
	}
	perClassTotal := 0
	for className, count := range report.PerClassCounts {
		if strings.TrimSpace(className) == "" || count < 0 {
			validationErrors = append(validationErrors, "coverage_report.per_class_counts contains invalid entries")
		}
		perClassTotal += count
	}
	if perClassTotal > 0 && perClassTotal > imagesAnalyzed {
		validationErrors = append(validationErrors, "coverage_report.per_class_counts exceeds images_analyzed")
	}
	return validationErrors
}

func validateVisualPreprocessingHypothesis(profile map[string]any, hypothesis *datasets.PreprocessingHypothesis, index int) []string {
	validationErrors := []string{}
	hypothesis.ID = strings.TrimSpace(hypothesis.ID)
	hypothesis.Mechanism = strings.ToLower(strings.TrimSpace(hypothesis.Mechanism))
	hypothesis.SupportStatus = strings.ToLower(strings.TrimSpace(hypothesis.SupportStatus))
	if hypothesis.SupportStatus == "" {
		hypothesis.SupportStatus = "needs_backend_validation"
	}
	if hypothesis.ID == "" {
		validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].id is required", index))
	}
	if !allowedPlannerMechanisms()[hypothesis.Mechanism] {
		validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].mechanism is unsupported", index))
	}
	if !allowedVisualSupportStatus(hypothesis.SupportStatus) {
		validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].support_status is unsupported", index))
	}
	if hypothesis.Confidence != "" && !allowedVisualConfidence(hypothesis.Confidence) {
		validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].confidence is unsupported", index))
	}
	hasBBoxEvidence := profileHasBBoxEvidence(profile)
	if hypothesis.Mechanism == "bbox_crop_ablation" && !hasBBoxEvidence {
		markVisualHypothesisUnsupported(hypothesis, "bbox_crop_ablation requires backend-profiled bbox evidence; retained as evidence only")
	}
	if hypothesis.SupportStatus == "unsupported" {
		sanitizeUnsupportedVisualHypothesis(hypothesis)
		return validationErrors
	}
	if hypothesis.SuggestedPreprocessing != nil {
		if err := validatePreprocessingConfig(*hypothesis.SuggestedPreprocessing, index); err != nil {
			validationErrors = append(validationErrors, err.Error())
		}
		if visualPreprocessingRequiresBBoxEvidence(*hypothesis.SuggestedPreprocessing) && !hasBBoxEvidence {
			markVisualHypothesisUnsupported(hypothesis, "suggested bbox preprocessing requires backend-profiled bbox evidence; retained as evidence only")
			sanitizeUnsupportedVisualHypothesis(hypothesis)
			return validationErrors
		}
	}
	for _, imageSize := range hypothesis.SuggestedImageSizes {
		if imageSize < 96 || imageSize > 384 {
			validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].suggested_image_sizes must be between 96 and 384", index))
			break
		}
	}
	if hypothesis.SuggestedAugmentationPolicy != "" && !allowedExperimentValue(hypothesis.SuggestedAugmentationPolicy, allowedAugmentationPolicies()) {
		validationErrors = append(validationErrors, fmt.Sprintf("preprocessing_hypotheses[%d].suggested_augmentation_policy is unsupported", index))
	}
	if hypothesis.SuggestedAugmentationConfig != nil {
		if err := validateAugmentationPolicyConfig(*hypothesis.SuggestedAugmentationConfig, index); err != nil {
			validationErrors = append(validationErrors, err.Error())
		}
	}
	return validationErrors
}

func visualPreprocessingRequiresBBoxEvidence(preprocessing plans.Preprocessing) bool {
	return containsAnyText(
		strings.ToLower(preprocessing.ResizeStrategy+" "+preprocessing.CropStrategy+" "+preprocessing.BBoxMode),
		"bbox",
		"box",
	)
}

func markVisualHypothesisUnsupported(hypothesis *datasets.PreprocessingHypothesis, reason string) {
	hypothesis.SupportStatus = "unsupported"
	hypothesis.UnsupportedReason = appendVisualUnsupportedReason(hypothesis.UnsupportedReason, reason)
}

func appendVisualUnsupportedReason(existing string, reason string) string {
	existing = strings.TrimSpace(existing)
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return existing
	}
	if existing == "" {
		return reason
	}
	if strings.Contains(strings.ToLower(existing), strings.ToLower(reason)) {
		return existing
	}
	return existing + "; " + reason
}

func sanitizeUnsupportedVisualHypothesis(hypothesis *datasets.PreprocessingHypothesis) {
	hypothesis.SuggestedPreprocessing = nil
	hypothesis.SuggestedImageSizes = nil
	hypothesis.SuggestedAugmentationPolicy = ""
	hypothesis.SuggestedAugmentationConfig = nil
	if strings.TrimSpace(hypothesis.UnsupportedReason) == "" {
		hypothesis.UnsupportedReason = "Unsupported visual hypothesis retained as evidence only; no executable preprocessing was persisted."
	}
}

func validateExampleImageIDs(prefix string, ids []string, manifestIDs map[string]bool) []string {
	validationErrors := []string{}
	for _, id := range ids {
		if !manifestIDs[strings.TrimSpace(id)] {
			validationErrors = append(validationErrors, prefix+".example_image_ids contains an image_id not present in sample_manifest")
		}
	}
	return validationErrors
}

func visualAnalysisInvocationContext(inputContext map[string]any, manifest []datasets.VisualSampleManifestItem) map[string]any {
	context := map[string]any{
		"sample_manifest":                  sanitizeVisualAnalysisSampleManifestForContext(manifest),
		"sample_manifest_count":            len(manifest),
		"raw_images_included":              false,
		"raw_images_included_for_planner":  false,
		"raw_visual_output_included":       false,
		"raw_visual_output_for_planner":    false,
		"visual_agent_prompt_for_planner":  false,
		"visual_agent_stream_is_separate":  true,
		"planner_receives_compressed_only": true,
		"evidence_only":                    true,
	}
	for key, value := range inputContext {
		normalizedKey := strings.ToLower(strings.TrimSpace(key))
		if reservedVisualInvocationContextKey(normalizedKey) || !allowedVisualInvocationContextKey(normalizedKey) {
			continue
		}
		sanitized, ok := sanitizeVisualInvocationContextValue(normalizedKey, value)
		if !ok {
			continue
		}
		context[normalizedKey] = sanitized
	}
	return context
}

func reservedVisualInvocationContextKey(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "sample_manifest", "raw_images_included", "raw_images_included_for_planner",
		"raw_visual_output_included", "raw_visual_output_for_planner",
		"visual_agent_prompt_for_planner", "visual_agent_stream_is_separate",
		"planner_receives_compressed_only", "evidence_only",
		"raw_output", "input_messages", "images", "image_inputs", "data_base64":
		return true
	default:
		return false
	}
}

func allowedVisualInvocationContextKey(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "caps", "pack_summary", "sample_manifest_summary", "repair":
		return true
	default:
		return false
	}
}

func sanitizeVisualAnalysisSampleManifestForContext(manifest []datasets.VisualSampleManifestItem) []map[string]any {
	out := make([]map[string]any, 0, len(manifest))
	for _, item := range manifest {
		entry := map[string]any{
			"image_id":        strings.TrimSpace(item.ImageID),
			"class_name":      strings.TrimSpace(item.ClassName),
			"class":           strings.TrimSpace(item.ClassLabel),
			"width":           item.Width,
			"height":          item.Height,
			"selection_basis": item.SelectionBasis,
			"detail_level":    strings.TrimSpace(item.DetailLevel),
			"has_bbox":        item.HasBBox,
		}
		if metadata, ok := sanitizeVisualInvocationContextValue("metadata", item.Metadata); ok {
			entry["metadata"] = metadata
		}
		if bbox, ok := sanitizeVisualInvocationContextValue("bbox", item.BBox); ok {
			entry["bbox"] = bbox
		}
		out = append(out, entry)
	}
	return out
}

func sanitizeVisualInvocationContextValue(key string, value any) (any, bool) {
	if unsafeVisualTelemetryKey(key) {
		return nil, false
	}
	switch typed := value.(type) {
	case nil:
		return nil, true
	case string:
		if containsUnsafeVisualAnalysisContent(typed) {
			return nil, false
		}
		return typed, true
	case bool:
		return typed, true
	case int:
		return typed, true
	case int8:
		return typed, true
	case int16:
		return typed, true
	case int32:
		return typed, true
	case int64:
		return typed, true
	case uint:
		return typed, true
	case uint8:
		return typed, true
	case uint16:
		return typed, true
	case uint32:
		return typed, true
	case uint64:
		return typed, true
	case float32:
		return typed, true
	case float64:
		return typed, true
	case json.Number:
		return typed, true
	case []string:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if containsUnsafeVisualAnalysisContent(item) {
				continue
			}
			out = append(out, item)
		}
		return out, true
	case []any:
		out := make([]any, 0, len(typed))
		for _, item := range typed {
			if sanitized, ok := sanitizeVisualInvocationContextValue("", item); ok {
				out = append(out, sanitized)
			}
		}
		return out, true
	case map[string]any:
		out := map[string]any{}
		for nestedKey, nestedValue := range typed {
			normalizedKey := strings.ToLower(strings.TrimSpace(nestedKey))
			if sanitized, ok := sanitizeVisualInvocationContextValue(normalizedKey, nestedValue); ok {
				out[normalizedKey] = sanitized
			}
		}
		return out, true
	case map[string]string:
		out := map[string]any{}
		for nestedKey, nestedValue := range typed {
			normalizedKey := strings.ToLower(strings.TrimSpace(nestedKey))
			if sanitized, ok := sanitizeVisualInvocationContextValue(normalizedKey, nestedValue); ok {
				out[normalizedKey] = sanitized
			}
		}
		return out, true
	case map[string]int:
		out := map[string]any{}
		for nestedKey, nestedValue := range typed {
			normalizedKey := strings.ToLower(strings.TrimSpace(nestedKey))
			if sanitized, ok := sanitizeVisualInvocationContextValue(normalizedKey, nestedValue); ok {
				out[normalizedKey] = sanitized
			}
		}
		return out, true
	default:
		blob, err := json.Marshal(typed)
		if err != nil || containsUnsafeVisualAnalysisContent(string(blob)) {
			return nil, false
		}
		return typed, true
	}
}

func unsafeVisualTelemetryKey(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "data_base64",
		"image_inputs",
		"images",
		"image",
		"raw_image",
		"raw_images",
		"raw_image_bytes",
		"image_bytes",
		"path",
		"file_path",
		"local_path",
		"source_path",
		"relative_path",
		"absolute_path",
		"storage_uri",
		"uri",
		"url",
		"input_messages",
		"raw_output":
		return true
	default:
		return false
	}
}

func datasetProfileFingerprint(profile map[string]any) string {
	blob, _ := json.Marshal(profile)
	sum := sha256.Sum256(blob)
	return fmt.Sprintf("%x", sum[:])
}

func datasetProfileSchemaVersion(profile map[string]any) string {
	for _, key := range []string{"schema_version", "profile_schema_version"} {
		if value := strings.TrimSpace(profileString(profile, key)); value != "" {
			return value
		}
	}
	return ""
}

func allowedVisualTrigger(trigger datasets.VisualReanalysisTrigger) bool {
	switch trigger {
	case datasets.VisualTriggerInitialProfile, datasets.VisualTriggerDeficiencyReanalysis, datasets.VisualTriggerManual:
		return true
	default:
		return false
	}
}

func allowedVisualConfidence(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "low", "medium", "high":
		return true
	default:
		return false
	}
}

func allowedVisualSeverity(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "low", "medium", "high":
		return true
	default:
		return false
	}
}

func allowedVisualSupportStatus(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "supported", "unsupported", "needs_backend_validation":
		return true
	default:
		return false
	}
}

func allowedVisualTrait(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "small_objects", "large_objects", "background_dominance", "lighting_variation", "blur",
		"fine_grained_similarity", "color_texture_signal", "crop_bbox_useful", "visual_ambiguity",
		"orientation_sensitive", "text_or_watermark", "domain_shift_possible":
		return true
	default:
		return false
	}
}

func containsUnsafeVisualAnalysisContent(value string) bool {
	return len(unsafeVisualAnalysisContentReasons(value)) > 0
}

func unsafeVisualAnalysisContentReasons(value string) []string {
	lower := strings.ToLower(value)
	checks := []struct {
		token  string
		reason string
	}{
		{"file://", "file URI"},
		{"s3://", "object-storage URI"},
		{"data:image", "inline image data URI"},
		{"\"data_base64\"", "base64 image field"},
		{"\"image_bytes\"", "image bytes field"},
		{"\"raw_image\"", "raw image field"},
		{"\"raw_images\"", "raw images field"},
		{"\"raw_image_bytes\"", "raw image bytes field"},
		{"\"image_inputs\"", "image inputs field"},
		{"c:\\", "Windows local path"},
		{"\\users\\", "Windows user path"},
		{"\\\\", "UNC path"},
		{"/users/", "local user path"},
		{"/home/", "local home path"},
		{"/tmp/", "temporary local path"},
		{"/var/", "system local path"},
		{"aws_secret", "secret marker"},
		{"secret_access_key", "secret marker"},
		{"proposed_experiments", "direct experiment proposal"},
		{"\"jobs\"", "job authority field"},
		{"\"commands\"", "command authority field"},
		{"shell command", "shell command text"},
		{"execute this", "execution instruction"},
		{"create job", "job creation instruction"},
		{"schedule job", "job scheduling instruction"},
		{"mutate dataset", "dataset mutation instruction"},
		{"delete files", "file deletion instruction"},
		{"relabel", "label mutation instruction"},
		{"labels_to_change", "label mutation field"},
	}
	reasons := []string{}
	for _, check := range checks {
		if strings.Contains(lower, check.token) {
			reasons = append(reasons, check.reason)
		}
	}
	return uniqueStrings(reasons)
}

func visualAnalysisUnsafeContentError(prefix string, reasons []string) string {
	reasons = nonEmptyStringValues(reasons)
	if len(reasons) == 0 {
		return prefix
	}
	return prefix + ": " + strings.Join(reasons, ", ")
}

func (s *Server) ensureOpenJob(projectID string, template string, config map[string]any, matches func(jobs.ExperimentJob) bool) (jobs.ExperimentJob, error) {
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	for _, job := range projectJobs {
		if job.Template != template {
			continue
		}
		if job.Status != jobs.StatusQueued && job.Status != jobs.StatusAssigned && job.Status != jobs.StatusRunning {
			continue
		}
		if matches(job) {
			return job, nil
		}
	}
	return s.store.CreateJob(projectID, template, config)
}

func jobConfigString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return strings.TrimSpace(value)
}

func copyMap(values map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range values {
		out[key] = value
	}
	return out
}

func validateVisualExemplarPack(exemplars []datasets.VisualExemplar, caps visualExemplarCaps) error {
	if len(exemplars) > caps.MaxTotalImages {
		return fmt.Errorf("%w: exemplar pack exceeds max_total_images %d", store.ErrInvalidRequest, caps.MaxTotalImages)
	}
	perClass := map[string]int{}
	var totalBytes int64
	seen := map[string]bool{}
	for _, exemplar := range exemplars {
		uri := strings.TrimSpace(exemplar.URI)
		if uri == "" {
			return fmt.Errorf("%w: exemplar uri is required", store.ErrInvalidRequest)
		}
		className := strings.TrimSpace(exemplar.ClassName)
		if className == "" {
			className = strings.TrimSpace(exemplar.Label)
		}
		if className == "" {
			return fmt.Errorf("%w: exemplar class_name or label is required", store.ErrInvalidRequest)
		}
		if exemplar.SizeBytes < 0 {
			return fmt.Errorf("%w: exemplar size_bytes cannot be negative", store.ErrInvalidRequest)
		}
		if exemplar.SizeBytes > caps.MaxBytes {
			return fmt.Errorf("%w: one exemplar exceeds max byte budget", store.ErrInvalidRequest)
		}
		totalBytes += exemplar.SizeBytes
		if totalBytes > caps.MaxBytes {
			return fmt.Errorf("%w: exemplar pack exceeds max byte budget %d", store.ErrInvalidRequest, caps.MaxBytes)
		}
		perClass[className]++
		if perClass[className] > caps.MaxPerClass {
			return fmt.Errorf("%w: exemplar pack exceeds max_per_class %d", store.ErrInvalidRequest, caps.MaxPerClass)
		}
		key := className + "\x00" + uri
		if seen[key] {
			return fmt.Errorf("%w: duplicate exemplar uri for class %s", store.ErrInvalidRequest, className)
		}
		seen[key] = true
	}
	return nil
}

func mergeVisualExemplars(existing []datasets.VisualExemplar, incoming []datasets.VisualExemplar, caps visualExemplarCaps) []datasets.VisualExemplar {
	merged := append([]datasets.VisualExemplar(nil), existing...)
	index := map[string]int{}
	for i, exemplar := range merged {
		index[visualExemplarKey(exemplar)] = i
	}
	for _, exemplar := range incoming {
		exemplar.URI = strings.TrimSpace(exemplar.URI)
		if exemplar.ClassName == "" {
			exemplar.ClassName = exemplar.Label
		}
		key := visualExemplarKey(exemplar)
		if existingIndex, ok := index[key]; ok {
			merged[existingIndex] = exemplar
			continue
		}
		index[key] = len(merged)
		merged = append(merged, exemplar)
	}
	if err := validateVisualExemplarPack(merged, caps); err == nil {
		return merged
	}
	return cappedVisualExemplarList(merged, caps)
}

func cappedVisualExemplarList(exemplars []datasets.VisualExemplar, caps visualExemplarCaps) []datasets.VisualExemplar {
	profile := map[string]any{"items": visualExemplarsToProfileValues(exemplars)}
	return cappedVisualExemplars(profile, caps, "items")
}

func visualExemplarKey(exemplar datasets.VisualExemplar) string {
	className := exemplar.ClassName
	if className == "" {
		className = exemplar.Label
	}
	return strings.TrimSpace(className) + "\x00" + strings.TrimSpace(exemplar.URI)
}

func visualExemplarsToProfileValues(exemplars []datasets.VisualExemplar) []map[string]any {
	out := make([]map[string]any, 0, len(exemplars))
	for _, exemplar := range exemplars {
		entry := map[string]any{
			"uri":        strings.TrimSpace(exemplar.URI),
			"class_name": strings.TrimSpace(exemplar.ClassName),
		}
		if exemplar.ID != "" {
			entry["id"] = exemplar.ID
		}
		if exemplar.Label != "" {
			entry["label"] = exemplar.Label
		}
		if exemplar.Width > 0 {
			entry["width"] = exemplar.Width
		}
		if exemplar.Height > 0 {
			entry["height"] = exemplar.Height
		}
		if exemplar.SizeBytes > 0 {
			entry["size_bytes"] = exemplar.SizeBytes
		}
		if exemplar.MimeType != "" {
			entry["mime_type"] = exemplar.MimeType
		}
		if exemplar.Split != "" {
			entry["split"] = exemplar.Split
		}
		if exemplar.Description != "" {
			entry["description"] = exemplar.Description
		}
		if len(exemplar.Metadata) > 0 {
			entry["metadata"] = exemplar.Metadata
		}
		out = append(out, entry)
	}
	return out
}

func firstString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := values[key]; ok {
			if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	return ""
}

func maxInt64(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func recommendedWorkersForExperiments(experiments []plans.PlannedExperiment) int {
	if len(experiments) < 1 {
		return 1
	}
	return len(experiments)
}

func estimateFollowUpMinutes(experiments []plans.PlannedExperiment) int {
	maxEpochs := 1
	for _, experiment := range experiments {
		if experiment.Epochs > maxEpochs {
			maxEpochs = experiment.Epochs
		}
	}

	return max(5, maxEpochs*6)
}

func automationSettingsFromEnv() settings.AutomationSettings {
	defaultProvider := os.Getenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER")
	if defaultProvider == "" {
		defaultProvider = "local"
	}
	agentMode := llm.NormalizeAgentMode(os.Getenv("MODEL_EXPRESS_AGENT_MODE"))

	return settings.AutomationSettings{
		AutoReviewExperiments:   envFlag("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", false),
		AutoScheduleFollowUps:   envFlag("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", false),
		AutoExecutePlans:        envFlag("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", os.Getenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER") != ""),
		MaxFollowUpRounds:       maxAutoFollowUpRoundsFromEnvForMode(agentMode),
		DefaultTrainingProvider: defaultProvider,
		DefaultGPUType:          os.Getenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE"),
		LLMEnabled:              envFlag("MODEL_EXPRESS_LLM_ENABLED", false),
		AgentMode:               agentMode,
		LLMProvider:             defaultLLMProviderFromEnv(),
		LLMModel:                defaultLLMModelFromEnv(),
		AutoMLEnabled:           envFlag("MODEL_EXPRESS_AUTOML_ENABLED", false),
		AutoMLSampler:           defaultAutoMLSamplerFromEnv(),
		UpdatedAt:               time.Now().UTC(),
	}
}

func (s *Server) currentAutomationSettings() settings.AutomationSettings {
	s.settingsMu.RLock()
	defer s.settingsMu.RUnlock()

	return s.automationSettings
}

func (s *Server) setAutomationSettings(automationSettings settings.AutomationSettings) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()

	s.automationSettings = automationSettings
}

func applyAutomationSettingsUpdate(current settings.AutomationSettings, update settings.AutomationSettingsUpdate) (settings.AutomationSettings, error) {
	if update.AutoReviewExperiments != nil {
		current.AutoReviewExperiments = *update.AutoReviewExperiments
	}
	if update.AutoScheduleFollowUps != nil {
		current.AutoScheduleFollowUps = *update.AutoScheduleFollowUps
	}
	if update.AutoExecutePlans != nil {
		current.AutoExecutePlans = *update.AutoExecutePlans
	}
	if update.MaxFollowUpRounds != nil {
		if *update.MaxFollowUpRounds < 0 {
			return settings.AutomationSettings{}, fmt.Errorf("%w: max_followup_rounds must be at least 0", store.ErrInvalidRequest)
		}
		current.MaxFollowUpRounds = *update.MaxFollowUpRounds
	}
	if update.DefaultTrainingProvider != nil {
		current.DefaultTrainingProvider = strings.ToLower(strings.TrimSpace(*update.DefaultTrainingProvider))
		if current.DefaultTrainingProvider == "" {
			current.DefaultTrainingProvider = "local"
		}
	}
	if update.DefaultGPUType != nil {
		current.DefaultGPUType = strings.TrimSpace(*update.DefaultGPUType)
	}
	if update.LLMEnabled != nil {
		current.LLMEnabled = *update.LLMEnabled
	}
	if update.AgentMode != nil {
		current.AgentMode = llm.NormalizeAgentMode(*update.AgentMode)
	}
	if update.LLMProvider != nil {
		current.LLMProvider = normalizeLLMProvider(*update.LLMProvider)
	}
	if update.LLMModel != nil {
		current.LLMModel = strings.TrimSpace(*update.LLMModel)
	}
	if update.AutoMLEnabled != nil {
		current.AutoMLEnabled = *update.AutoMLEnabled
	}
	if update.AutoMLSampler != nil {
		current.AutoMLSampler = normalizeAutoMLSampler(*update.AutoMLSampler)
	}
	if current.AgentMode == "" {
		current.AgentMode = llm.AgentModePropose
	}
	if current.LLMProvider == "" {
		current.LLMProvider = llm.ProviderOpenAI
	}
	if current.AutoMLSampler == "" {
		current.AutoMLSampler = automl.SamplerSeededRandom
	}
	if _, err := automl.NewSampler(current.AutoMLSampler); err != nil {
		return settings.AutomationSettings{}, fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}

	return current, nil
}

func (s *Server) defaultExecuteExperimentPlanRequest() executeExperimentPlanRequest {
	automationSettings := s.currentAutomationSettings()
	provider := automationSettings.DefaultTrainingProvider
	if provider == "" {
		provider = "local"
	}

	return executeExperimentPlanRequest{
		Provider: provider,
		GPUType:  automationSettings.DefaultGPUType,
	}
}

func (s *Server) shouldAutoReviewExperimentJobs() bool {
	return s.currentAutomationSettings().AutoReviewExperiments
}

func (s *Server) shouldAutoScheduleFollowUps() bool {
	return s.currentAutomationSettings().AutoScheduleFollowUps
}

func maxAutoFollowUpRoundsFromEnv() int {
	return maxAutoFollowUpRoundsFromEnvForMode(llm.NormalizeAgentMode(os.Getenv("MODEL_EXPRESS_AGENT_MODE")))
}

func maxAutoFollowUpRoundsFromEnvForMode(agentMode string) int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS"))
	if value == "" {
		if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
			return plannerAutonomousMaxFollowUpRounds
		}
		return plannerDefaultMaxFollowUpRounds
	}

	rounds, err := strconv.Atoi(value)
	if err != nil || rounds < 0 {
		if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
			return plannerAutonomousMaxFollowUpRounds
		}
		return plannerDefaultMaxFollowUpRounds
	}

	return rounds
}

func plannerMinimumMeaningfulImprovementFromEnv(agentMode string) float64 {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_MIN_MEANINGFUL_DELTA"))
	if value != "" {
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
		return plannerAutonomousMeaningfulImprovement
	}
	return plannerMinimumMeaningfulImprovement
}

func terminalPlannerGuardsEnabled() bool {
	return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}

func terminalPlannerGuardsEnabledForMode(agentMode string) bool {
	if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
		return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", true)
	}
	return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}

func maxAutoWorkersFromEnv() int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_AUTO_WORKERS"))
	if value == "" {
		return 4
	}
	count, err := strconv.Atoi(value)
	if err != nil || count < 1 {
		return 4
	}
	return count
}

func (s *Server) maxAutoFollowUpRounds() int {
	return s.currentAutomationSettings().MaxFollowUpRounds
}

func (s *Server) shouldAutoExecuteExperimentPlans() bool {
	return s.currentAutomationSettings().AutoExecutePlans
}

func (s *Server) shouldRunLLMAgents() bool {
	return s.currentAutomationSettings().LLMEnabled
}

func defaultLLMProviderFromEnv() string {
	if provider := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_PROVIDER")); provider != "" {
		return normalizeLLMProvider(provider)
	}
	if provider := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER")); provider != "" {
		return normalizeLLMProvider(provider)
	}
	return llm.ProviderOpenAI
}

func defaultLLMModelFromEnv() string {
	if model := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_MODEL")); model != "" {
		return model
	}
	return strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL"))
}

func defaultAutoMLSamplerFromEnv() string {
	return normalizeAutoMLSampler(os.Getenv("MODEL_EXPRESS_AUTOML_SAMPLER"))
}

func normalizeAutoMLSampler(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return automl.SamplerSeededRandom
	}
	switch normalized {
	case "random", "random_search":
		return automl.SamplerSeededRandom
	case "grid_search":
		return automl.SamplerGrid
	case "adaptive", "bayesian", "bayesian_optimizer":
		return automl.SamplerAdaptiveBayesian
	default:
		return normalized
	}
}

func normalizeLLMProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case llm.ProviderOpenAICompatible:
		return llm.ProviderOpenAICompatible
	case llm.ProviderLocal:
		return llm.ProviderLocal
	default:
		return llm.ProviderOpenAI
	}
}

func envFlag(name string, defaultValue bool) bool {
	if value, ok := envFlagValue(name); ok {
		return value
	}

	return defaultValue
}

func envFlagValue(name string) (bool, bool) {
	return envFlagValueFromString(os.Getenv(name))
}

func envFlagValueFromString(input string) (bool, bool) {
	value := strings.ToLower(strings.TrimSpace(input))
	switch value {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	}

	return false, false
}

func envInt(name string, defaultValue int, minValue int, maxValue int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func queryInt(c *gin.Context, key string, defaultValue int, minValue int, maxValue int) int {
	value := strings.TrimSpace(c.Query(key))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func envFloat(name string, defaultValue float64, minValue float64, maxValue float64) float64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func bindJSON(c *gin.Context, value any) bool {
	if err := c.ShouldBindJSON(value); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}

	return true
}

func writeStoreError(c *gin.Context, err error) {
	if errors.Is(err, store.ErrNotFound) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	if errors.Is(err, store.ErrInvalidRequest) {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
}
