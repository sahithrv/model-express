package api

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

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
	raw := map[string]any{}
	_ = json.Unmarshal(blob, &raw)
	if exemplar.URI == "" {
		if len(raw) > 0 {
			exemplar.URI = firstString(raw, "image_uri", "url", "path", "storage_uri")
			if exemplar.ClassName == "" {
				exemplar.ClassName = firstString(raw, "class", "class_name", "label")
			}
		}
	}
	if len(raw) > 0 {
		if exemplar.Metadata == nil {
			exemplar.Metadata = map[string]any{}
		}
		for _, key := range []string{"image_uri", "preview_uri", "thumbnail_uri", "original_image_uri", "source_artifact_uri"} {
			if value := strings.TrimSpace(firstString(raw, key)); value != "" {
				if _, exists := exemplar.Metadata[key]; !exists {
					exemplar.Metadata[key] = value
				}
			}
		}
	}
	return exemplar, true
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
	return envFlag("MODEL_EXPRESS_VISUAL_LLM_ENABLED", false) || visualLLMAPIKeyConfigured()
}

func visualLLMAPIKeyConfigured() bool {
	if strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY")) != "" {
		return true
	}
	provider := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER")))
	if provider == "" {
		provider = llm.ProviderOpenAI
	}
	return provider == llm.ProviderOpenAI && strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != ""
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
		provider = s.defaultVisualAnalysisProvider()
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
