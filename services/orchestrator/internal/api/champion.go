package api

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"sort"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

const championDemoOriginalUnavailableCode = "ORIGINAL_IMAGE_UNAVAILABLE_FOR_DEMO"

type createChampionExportRequest struct {
	Format      string         `json:"format"`
	ArtifactURI string         `json:"artifact_uri"`
	Metadata    map[string]any `json:"metadata"`
}

type createChampionDemoPredictionRequest struct {
	ImageURI            string         `json:"image_uri" binding:"required"`
	ImageID             string         `json:"image_id"`
	TrueLabel           string         `json:"true_label"`
	ImageMetadata       map[string]any `json:"image_metadata"`
	TopK                int            `json:"top_k"`
	ConfidenceThreshold *float64       `json:"confidence_threshold"`
	IOUThreshold        *float64       `json:"iou_threshold"`
	MaxDetections       int            `json:"max_detections"`
}

type createLocalChampionDemoPredictionRequest struct {
	ImageURI       string                    `json:"image_uri" binding:"required"`
	ImageID        string                    `json:"image_id"`
	TrueLabel      string                    `json:"true_label"`
	Status         string                    `json:"status"`
	PredictedLabel string                    `json:"predicted_label"`
	Confidence     *float64                  `json:"confidence"`
	TopK           []runs.DemoPredictionTopK `json:"top_k"`
	LatencyMS      *float64                  `json:"latency_ms"`
	Correct        *bool                     `json:"correct"`
	Error          string                    `json:"error"`
	ImageMetadata  map[string]any            `json:"image_metadata"`
}

type createChampionFeedbackRequest struct {
	PredictionID       string         `json:"prediction_id"`
	ImageURI           string         `json:"image_uri"`
	ImageID            string         `json:"image_id"`
	Rating             string         `json:"rating" binding:"required"`
	Message            string         `json:"message"`
	PredictionSnapshot map[string]any `json:"prediction_snapshot"`
	MetricsSnapshot    map[string]any `json:"metrics_snapshot"`
	Metadata           map[string]any `json:"metadata"`
}

type championExportResultRequest struct {
	Status            string         `json:"status"`
	ArtifactURI       string         `json:"artifact_uri"`
	ManifestURI       string         `json:"manifest_uri"`
	Metadata          map[string]any `json:"metadata"`
	ValidationErrors  []string       `json:"validation_errors"`
	Error             string         `json:"error"`
	TrainingAttemptID string         `json:"training_attempt_id"`
}

type championDemoPredictionResultRequest struct {
	Status            string                    `json:"status"`
	PredictedLabel    string                    `json:"predicted_label"`
	TrueLabel         string                    `json:"true_label"`
	Confidence        *float64                  `json:"confidence"`
	TopK              []runs.DemoPredictionTopK `json:"top_k"`
	LatencyMS         *float64                  `json:"latency_ms"`
	Correct           *bool                     `json:"correct"`
	Error             string                    `json:"error"`
	ImageMetadata     map[string]any            `json:"image_metadata"`
	TrainingAttemptID string                    `json:"training_attempt_id"`
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
	requestedChampionJobID := championJobID
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
	if requestedChampionJobID == "" {
		requestedChampionJobID = championJobID
	}
	if championJobID == "" {
		return fmt.Errorf("%w: SELECT_CHAMPION decision is missing champion_job_id", store.ErrInvalidRequest)
	}

	summary, err := s.store.GetTrainingRunSummary(championJobID)
	if err != nil {
		return err
	}
	goalText := ""
	if project, err := s.store.GetProject(projectID); err == nil {
		goalText = project.Goal
	}
	targetMetric := payloadString(decision.Payload, "target_metric")
	if targetMetric == "" && summary.PlanID != "" {
		if plan, err := s.store.GetExperimentPlan(summary.PlanID); err == nil && plan.TargetMetric != "" {
			targetMetric = plan.TargetMetric
		}
	}
	if targetMetric == "" {
		targetMetric = "macro_f1"
	}
	selectionReview, err := s.reviewChampionSelection(projectID, championJobID, targetMetric, goalText)
	if err != nil {
		return err
	}
	if selectionReview.SelectedJobID != "" {
		championJobID = selectionReview.SelectedJobID
		summary = selectionReview.SelectedSummary
	}
	job, err := s.store.GetJob(championJobID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return err
	}

	evaluationPayload := map[string]any{}
	championEvaluation := runs.TrainingRunEvaluation{}
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
		championEvaluation = evaluation
		if payload, payloadErr := mapFromStruct(evaluation); payloadErr == nil {
			evaluationPayload = payload
		}
		deploymentProfile["model_profile"] = evaluation.ModelProfile
		deploymentProfile["holistic_scores"] = evaluation.HolisticScores
		deploymentProfile["diagnostics"] = "available"
		if unexportable, reason := championModelProfileUnexportable(evaluation.ModelProfile); unexportable {
			deploymentProfile["export_status"] = "simulated_unexportable"
			deploymentProfile["deployment_ready"] = false
			deploymentProfile["export_validation_errors"] = []string{reason}
		} else {
			if artifactURI := championArtifactURIFromEvaluation(evaluation.ModelProfile); artifactURI != "" {
				deploymentProfile["artifact_uri"] = artifactURI
				if artifactMatchesChampionExportFormat(artifactURI, "onnx") {
					deploymentProfile["onnx_artifact_uri"] = artifactURI
				}
				deploymentProfile["export_status"] = firstString(evaluation.ModelProfile, "export_status")
			}
			if manifestURI := firstString(evaluation.ModelProfile, "export_manifest_uri", "manifest_uri"); manifestURI != "" {
				deploymentProfile["export_manifest_uri"] = manifestURI
			}
			if manifest := payloadMap(evaluation.ModelProfile, "export_manifest"); len(manifest) > 0 {
				deploymentProfile["export_manifest"] = manifest
			}
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
	if selectionReview.Overridden {
		overrideReason := fmt.Sprintf("Backend champion guard selected %s instead of requested %s because holistic validation/test readiness %.3f beat %.3f.",
			selectionReview.SelectedJobID,
			selectionReview.RequestedJobID,
			selectionReview.SelectedScore,
			selectionReview.RequestedScore,
		)
		if selectionReason == "" {
			selectionReason = overrideReason
		} else {
			selectionReason = strings.TrimSpace(selectionReason + " " + overrideReason)
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
	addDetectionChampionMetrics(metrics, championEvaluation)
	if fallbackSelection {
		metrics["selection_source"] = "terminal_stop_best_available"
	} else {
		metrics["selection_source"] = "agent_decision"
	}
	if selectionReview.SelectedJobID != "" {
		if selectionReview.Overridden {
			metrics["selection_source"] = "backend_holistic_override"
		}
		metrics["requested_champion_job_id"] = requestedChampionJobID
		metrics["requested_deployment_readiness_score"] = roundDiagnosticFloat(selectionReview.RequestedScore)
		metrics["deployment_readiness_score"] = roundDiagnosticFloat(selectionReview.SelectedScore)
		metrics["selection_overridden"] = selectionReview.Overridden
		metrics["selection_score_breakdown"] = selectionReview.SelectedBreakdown
		metrics["selection_candidates"] = selectionReview.Candidates
		deploymentProfile["selection_score"] = selectionReview.SelectedBreakdown
		deploymentProfile["selection_guard"] = map[string]any{
			"requested_champion_job_id": requestedChampionJobID,
			"selected_champion_job_id":  selectionReview.SelectedJobID,
			"requested_score":           roundDiagnosticFloat(selectionReview.RequestedScore),
			"selected_score":            roundDiagnosticFloat(selectionReview.SelectedScore),
			"override_min_delta":        championSelectionOverrideMinDelta,
			"overridden":                selectionReview.Overridden,
		}
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

	if _, err := s.store.CreateExecutionEvent(projectID, summary.PlanID, execution.EventChampionSelected, fmt.Sprintf("Champion selected: %s for project %s.", championJobID, projectID), map[string]any{
		"champion_id":        champion.ID,
		"champion_job_id":    champion.JobID,
		"requested_job_id":   requestedChampionJobID,
		"source_decision_id": decision.ID,
		"selection_source":   metrics["selection_source"],
		"selection_score":    metrics["deployment_readiness_score"],
		"selection_overrode": selectionReview.Overridden,
		"model":              metrics["model"],
	}); err != nil {
		log.Printf("record champion selected event failed for project %s champion %s: %v", projectID, champion.ID, err)
	}
	if _, err := s.ensureChampionExport(projectID, champion, job, "onnx", "", nil); err != nil {
		log.Printf("auto champion ONNX export request failed for project %s champion %s job %s: %v", projectID, champion.ID, champion.JobID, err)
		failedExport, failedErr := s.ensureFailedChampionExportState(projectID, champion, "onnx", err)
		if failedErr != nil {
			log.Printf("record failed champion export state failed for project %s champion %s: %v", projectID, champion.ID, failedErr)
		}
		s.recordChampionExportFailureEvent(projectID, champion, failedExport.ID, "onnx", err)
	}
	return nil
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

type terminalChampionSelectionOptions struct {
	DecisionSource    string
	Trigger           string
	EventType         string
	Rationale         string
	EventMessage      string
	NoChampionMessage string
	Payload           map[string]any
}

func (s *Server) selectBestAvailableChampionForTerminalPlanStop(plan plans.ExperimentPlan, opts terminalChampionSelectionOptions) (bool, error) {
	if plan.ID == "" {
		return false, nil
	}
	if _, err := s.store.GetProjectChampion(plan.ProjectID); err == nil {
		return true, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return false, err
	}

	decisionSource := strings.TrimSpace(opts.DecisionSource)
	if decisionSource == "" {
		decisionSource = "terminal_best_available"
	}
	decisionsForProject, err := s.store.ListProjectAgentDecisions(plan.ProjectID)
	if err != nil {
		return false, err
	}
	for _, decision := range decisionsForProject {
		if decision.PlanID == plan.ID && decision.Payload["decision_source"] == decisionSource {
			if err := s.persistProjectChampionFromDecision(plan.ProjectID, decision); err != nil {
				return false, err
			}
			return true, nil
		}
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	goalText := ""
	if project, err := s.store.GetProject(plan.ProjectID); err == nil {
		goalText = project.Goal
	} else if !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	objectiveContext := projectObjectiveContext(goalText)
	best, ok := bestSuccessfulTrainingSummaryForObjective(plan.TargetMetric, summaries, evaluations, objectiveContext)
	eventType := opts.EventType
	if eventType == "" {
		eventType = execution.EventAgentOutcomeRecorded
	}
	basePayload := terminalChampionSelectionPayload(opts.Payload, decisionSource, opts.Trigger, plan.TargetMetric)
	if !ok {
		payload := clonePayload(basePayload)
		payload["exportable"] = false
		payload["selected_best_available_model"] = false
		message := strings.TrimSpace(opts.NoChampionMessage)
		if message == "" {
			message = fmt.Sprintf("Terminal automation stopped for plan %s, but no successful model is available to select as champion.", plan.ID)
		}
		_, eventErr := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, eventType, message, payload)
		return false, eventErr
	}

	evaluationsByJob := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		evaluationsByJob[evaluation.JobID] = evaluation
	}
	score := holisticRunScore(plan.TargetMetric, best, evaluationsByJob[best.JobID], objectiveContext)
	decisionPayload := clonePayload(basePayload)
	decisionPayload["auto_executable"] = true
	decisionPayload["champion_job_id"] = best.JobID
	decisionPayload["champion_model"] = best.Model
	decisionPayload["champion_score"] = roundDiagnosticFloat(score)
	decisionPayload["champion_macro_f1"] = roundDiagnosticFloat(best.BestMacroF1)
	decisionPayload["champion_accuracy"] = roundDiagnosticFloat(best.BestAccuracy)
	decisionPayload["champion_estimated_cost_usd"] = roundDiagnosticFloat(best.EstimatedCostUSD)
	decisionPayload["champion_runtime_seconds"] = roundDiagnosticFloat(best.RuntimeSeconds)
	decisionPayload["selected_best_available_model"] = true

	rationale := strings.TrimSpace(opts.Rationale)
	if rationale == "" {
		rationale = fmt.Sprintf("Terminal automation stopped additional training for plan %s, so the backend selected the best successful model available: %s.", plan.ID, best.JobID)
	}
	decision, err := s.store.CreateAgentDecision(plan.ProjectID, plan.ID, decisions.TypeSelectChampion, rationale, decisionPayload)
	if err != nil {
		return false, err
	}
	if err := s.persistProjectChampionFromDecision(plan.ProjectID, decision); err != nil {
		return false, err
	}

	eventPayload := clonePayload(decisionPayload)
	if _, ok := eventPayload["source_decision_id"]; !ok {
		eventPayload["source_decision_id"] = decision.ID
	}
	eventPayload["terminal_decision_id"] = decision.ID
	eventPayload["exportable"] = true
	message := strings.TrimSpace(opts.EventMessage)
	if message == "" {
		message = fmt.Sprintf("Terminal automation stopped for plan %s and selected best available champion %s.", plan.ID, best.JobID)
	}
	_, err = s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, eventType, message, eventPayload)
	return true, err
}

func terminalChampionSelectionPayload(base map[string]any, decisionSource string, trigger string, targetMetric string) map[string]any {
	payload := clonePayload(base)
	payload["decision_source"] = decisionSource
	if trigger != "" {
		payload["trigger"] = trigger
	}
	if targetMetric != "" {
		payload["target_metric"] = targetMetric
	}
	return payload
}

func clonePayload(base map[string]any) map[string]any {
	payload := map[string]any{}
	for key, value := range base {
		payload[key] = value
	}
	return payload
}

type championSelectionReview struct {
	RequestedJobID     string
	SelectedJobID      string
	SelectedSummary    runs.TrainingRunSummary
	RequestedScore     float64
	SelectedScore      float64
	RequestedBreakdown map[string]any
	SelectedBreakdown  map[string]any
	Candidates         []map[string]any
	Overridden         bool
}

func (s *Server) reviewChampionSelection(projectID, requestedJobID, targetMetric, goalText string) (championSelectionReview, error) {
	review := championSelectionReview{RequestedJobID: requestedJobID}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return review, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return review, err
	}
	evaluationsByJob := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		evaluationsByJob[evaluation.JobID] = evaluation
	}

	objectiveContext := projectObjectiveContext(goalText)
	var requested runs.TrainingRunSummary
	requestedFound := false
	var best runs.TrainingRunSummary
	bestScore := 0.0
	hasBest := false
	requestedSucceeded := false
	candidates := make([]map[string]any, 0, len(summaries))
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			if summary.JobID == requestedJobID {
				requested = summary
				requestedFound = true
			}
			continue
		}
		evaluation := evaluationsByJob[summary.JobID]
		score, breakdown := holisticRunScoreBreakdown(targetMetric, summary, evaluation, objectiveContext)
		candidate := championSelectionCandidatePayload(summary, breakdown)
		candidates = append(candidates, candidate)
		if summary.JobID == requestedJobID {
			requested = summary
			requestedFound = true
			requestedSucceeded = true
			review.RequestedScore = score
			review.RequestedBreakdown = breakdown
		}
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			review.SelectedBreakdown = breakdown
			hasBest = true
		}
	}
	sort.SliceStable(candidates, func(left, right int) bool {
		return payloadFloat(candidates[left], "deployment_readiness_score") > payloadFloat(candidates[right], "deployment_readiness_score")
	})
	if len(candidates) > 8 {
		candidates = candidates[:8]
	}
	review.Candidates = candidates
	if !hasBest {
		return review, nil
	}
	if !requestedFound {
		return review, nil
	}
	if review.RequestedBreakdown == nil {
		review.RequestedScore, review.RequestedBreakdown = holisticRunScoreBreakdown(targetMetric, requested, evaluationsByJob[requested.JobID], objectiveContext)
	}

	selected := requested
	selectedScore := review.RequestedScore
	selectedBreakdown := review.RequestedBreakdown
	if best.JobID != requested.JobID && (!requestedSucceeded || bestScore >= review.RequestedScore+championSelectionOverrideMinDelta) {
		selected = best
		selectedScore = bestScore
		selectedBreakdown = review.SelectedBreakdown
		review.Overridden = true
	}
	review.SelectedSummary = selected
	review.SelectedJobID = selected.JobID
	review.SelectedScore = selectedScore
	review.SelectedBreakdown = selectedBreakdown
	return review, nil
}

func championSelectionCandidatePayload(summary runs.TrainingRunSummary, breakdown map[string]any) map[string]any {
	return map[string]any{
		"job_id":                     summary.JobID,
		"model":                      summary.Model,
		"status":                     summary.Status,
		"deployment_readiness_score": roundDiagnosticFloat(payloadFloat(breakdown, "score")),
		"quality_score":              roundDiagnosticFloat(payloadFloat(breakdown, "quality_score")),
		"validation_metric_score":    roundDiagnosticFloat(payloadFloat(breakdown, "validation_metric_score")),
		"heldout_metric_score":       roundDiagnosticFloat(payloadFloat(breakdown, "heldout_metric_score")),
		"loss_health_score":          roundDiagnosticFloat(payloadFloat(breakdown, "loss_health_score")),
		"best_macro_f1":              roundDiagnosticFloat(summary.BestMacroF1),
		"best_accuracy":              roundDiagnosticFloat(summary.BestAccuracy),
		"final_train_loss":           roundDiagnosticFloat(summary.FinalTrainLoss),
		"final_val_loss":             roundDiagnosticFloat(summary.FinalValLoss),
		"estimated_cost_usd":         roundDiagnosticFloat(summary.EstimatedCostUSD),
		"runtime_seconds":            roundDiagnosticFloat(summary.RuntimeSeconds),
	}
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
	if strings.TrimSpace(req.ArtifactURI) != "" {
		writeStoreError(c, fmt.Errorf("%w: artifact_uri cannot be supplied by export requests; worker provenance is required", store.ErrInvalidRequest))
		return
	}

	export, err := s.ensureChampionExport(projectID, champion, job, format, req.ArtifactURI, req.Metadata)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, export)
}

func (s *Server) ensureChampionExport(
	projectID string,
	champion runs.ProjectChampion,
	championJob jobs.ExperimentJob,
	format string,
	requestArtifactURI string,
	requestMetadata map[string]any,
) (runs.ChampionExport, error) {
	requestArtifactURI = strings.TrimSpace(requestArtifactURI)
	sourceArtifactURI := requestArtifactURI
	if sourceArtifactURI == "" {
		sourceArtifactURI = championArtifactURIForFormat(champion.DeploymentProfile, format)
	}
	artifactURI := ""
	profileReady := championDeploymentProfileHasReadyExport(champion.DeploymentProfile, format, sourceArtifactURI)
	if requestArtifactURI != "" || (artifactMatchesChampionExportFormat(sourceArtifactURI, format) && trustedChampionExportArtifactURI(sourceArtifactURI) && profileReady) {
		artifactURI = sourceArtifactURI
	}
	status := runs.ChampionExportStatusPending
	validationErrors := []string{}
	if sourceArtifactURI == "" {
		status = runs.ChampionExportStatusPendingArtifact
		validationErrors = append(validationErrors, "selected champion has no exportable artifact URI yet")
	} else if artifactURI != "" {
		status = runs.ChampionExportStatusReady
	} else if artifactMatchesChampionExportFormat(sourceArtifactURI, format) {
		validationErrors = append(validationErrors, fmt.Sprintf("selected champion artifact needs a valid worker export manifest, passed self-test, and empty validation errors before READY %s export; worker export required", format))
	} else {
		validationErrors = append(validationErrors, fmt.Sprintf("selected champion artifact does not match requested %s export; worker export required", format))
	}

	metadata := championExportMetadata(champion, format, requestMetadata)
	if sourceArtifactURI != "" && sourceArtifactURI != artifactURI {
		metadata["source_artifact_uri"] = sourceArtifactURI
	}
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
		return runs.ChampionExport{}, err
	}
	datasetID := championDatasetID(champion, championJob)
	if datasetID != "" && export.Status != runs.ChampionExportStatusReady {
		exportJobConfig := map[string]any{
			"dataset_id":          datasetID,
			"champion_id":         champion.ID,
			"champion_job_id":     champion.JobID,
			"export_id":           export.ID,
			"format":              export.Format,
			"artifact_uri":        artifactURI,
			"source_artifact_uri": sourceArtifactURI,
			"metadata":            metadata,
		}
		if _, err := s.ensureOpenJob(champion.ProjectID, jobs.TemplateExportChampion, exportJobConfig, func(existing jobs.ExperimentJob) bool {
			return jobConfigString(existing.Config, "export_id") == export.ID
		}); err != nil {
			if failedExport, updateErr := s.markChampionExportFailed(export, err); updateErr == nil {
				export = failedExport
				s.recordChampionExportFailureEvent(projectID, champion, export.ID, format, err)
			} else {
				log.Printf("mark champion export failed state failed for export %s: %v", export.ID, updateErr)
			}
			return runs.ChampionExport{}, err
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

	return export, nil
}

func (s *Server) ensureFailedChampionExportState(projectID string, champion runs.ProjectChampion, format string, cause error) (runs.ChampionExport, error) {
	exports, err := s.store.ListProjectChampionExports(projectID)
	if err == nil {
		for _, export := range exports {
			if export.ChampionID == champion.ID && export.Format == format {
				return s.markChampionExportFailed(export, cause)
			}
		}
	} else if !errors.Is(err, store.ErrNotFound) {
		return runs.ChampionExport{}, err
	}
	return s.store.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:        projectID,
		ChampionID:       champion.ID,
		JobID:            champion.JobID,
		Status:           runs.ChampionExportStatusFailed,
		Format:           format,
		Metadata:         championExportMetadata(champion, format, nil),
		ValidationErrors: []string{cause.Error()},
	})
}

func (s *Server) markChampionExportFailed(export runs.ChampionExport, cause error) (runs.ChampionExport, error) {
	validationErrors := append([]string(nil), export.ValidationErrors...)
	if !containsString(validationErrors, cause.Error()) {
		validationErrors = append(validationErrors, cause.Error())
	}
	return s.store.UpdateChampionExport(export.ID, runs.ChampionExportUpdate{
		Status:           runs.ChampionExportStatusFailed,
		Metadata:         export.Metadata,
		ValidationErrors: validationErrors,
	})
}

func (s *Server) recordChampionExportFailureEvent(projectID string, champion runs.ProjectChampion, exportID string, format string, cause error) {
	if exportID != "" {
		events, err := s.store.ListProjectExecutionEvents(projectID, 50)
		if err == nil {
			for _, event := range events {
				if event.EventType == execution.EventExecutionFailed &&
					payloadString(event.Payload, "export_id") == exportID &&
					payloadString(event.Payload, "error") == cause.Error() {
					return
				}
			}
		} else if !errors.Is(err, store.ErrNotFound) {
			log.Printf("check champion export failure event failed for project %s champion %s: %v", projectID, champion.ID, err)
		}
	}
	payload := map[string]any{
		"champion_id":     champion.ID,
		"champion_job_id": champion.JobID,
		"export_id":       exportID,
		"format":          format,
		"status":          runs.ChampionExportStatusFailed,
		"error":           cause.Error(),
	}
	if _, err := s.store.CreateExecutionEvent(projectID, champion.PlanID, execution.EventExecutionFailed, fmt.Sprintf("Champion export setup failed for job %s.", champion.JobID), payload); err != nil {
		log.Printf("record champion export failure event failed for project %s champion %s: %v", projectID, champion.ID, err)
	}
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
	sourceOfTruth := "champion.evaluation.heldout_demo_images"
	if len(exemplars) == 0 {
		exemplars = testOnlyVisualExemplars(cappedVisualExemplars(dataset.Profile, caps, "demo_images", "visual_exemplars", "test_images"))
		sourceOfTruth = "datasets.profile"
	}

	c.JSON(http.StatusOK, gin.H{
		"project_id":      projectID,
		"dataset_id":      dataset.ID,
		"champion_job_id": champion.JobID,
		"source_of_truth": sourceOfTruth,
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

func (s *Server) createProjectChampionDemoPredictionLocalResult(c *gin.Context) {
	var req createLocalChampionDemoPredictionRequest
	if !bindJSON(c, &req) {
		return
	}
	imageURI := strings.TrimSpace(req.ImageURI)
	if imageURI == "" {
		writeStoreError(c, fmt.Errorf("%w: image_uri is required", store.ErrInvalidRequest))
		return
	}
	status := normalizeChampionDemoPredictionResultStatus(req.Status)
	if status == "" {
		writeStoreError(c, fmt.Errorf("%w: local prediction status must be SUCCEEDED, FAILED, or RUNTIME_UNAVAILABLE", store.ErrInvalidRequest))
		return
	}
	if status == runs.ChampionDemoPredictionStatusSucceeded && strings.TrimSpace(req.PredictedLabel) == "" && !championDemoPredictionHasDetectionMetadata(req.ImageMetadata) {
		writeStoreError(c, fmt.Errorf("%w: predicted_label is required for successful local prediction", store.ErrInvalidRequest))
		return
	}
	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	imageMetadata := copyPayloadMap(req.ImageMetadata)
	imageMetadata["local_runtime"] = true
	if strings.TrimSpace(payloadString(imageMetadata, "inference_transport")) == "" {
		imageMetadata["inference_transport"] = "mission_control_local_runtime"
	}
	prediction, err := s.store.CreateChampionDemoPrediction(runs.ChampionDemoPredictionCreate{
		ProjectID:      champion.ProjectID,
		ChampionID:     champion.ID,
		JobID:          champion.JobID,
		DatasetID:      champion.DatasetID,
		ImageURI:       imageURI,
		ImageID:        strings.TrimSpace(req.ImageID),
		ImageMetadata:  imageMetadata,
		Status:         status,
		PredictedLabel: strings.TrimSpace(req.PredictedLabel),
		TrueLabel:      strings.TrimSpace(req.TrueLabel),
		Confidence:     req.Confidence,
		TopK:           req.TopK,
		LatencyMS:      req.LatencyMS,
		Correct:        req.Correct,
		Error:          strings.TrimSpace(req.Error),
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if _, err := s.store.CreateExecutionEvent(champion.ProjectID, champion.PlanID, execution.EventChampionDemoPrediction, fmt.Sprintf("Local champion demo prediction recorded for job %s.", champion.JobID), map[string]any{
		"champion_id":   champion.ID,
		"prediction_id": prediction.ID,
		"job_id":        champion.JobID,
		"status":        prediction.Status,
		"image_uri":     prediction.ImageURI,
		"local_runtime": true,
	}); err != nil {
		log.Printf("record local champion demo prediction event failed: %v", err)
	}
	c.JSON(http.StatusCreated, gin.H{"prediction": prediction, "runtime_available": status == runs.ChampionDemoPredictionStatusSucceeded})
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
	if req.ConfidenceThreshold != nil && (*req.ConfidenceThreshold < 0 || *req.ConfidenceThreshold > 1) {
		writeStoreError(c, fmt.Errorf("%w: confidence_threshold must be between 0 and 1", store.ErrInvalidRequest))
		return
	}
	if req.IOUThreshold != nil && (*req.IOUThreshold < 0 || *req.IOUThreshold > 1) {
		writeStoreError(c, fmt.Errorf("%w: iou_threshold must be between 0 and 1", store.ErrInvalidRequest))
		return
	}
	if req.MaxDetections < 0 || req.MaxDetections > 1000 {
		writeStoreError(c, fmt.Errorf("%w: max_detections must be between 0 and 1000", store.ErrInvalidRequest))
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
	imageID := strings.TrimSpace(req.ImageID)
	trueLabel := strings.TrimSpace(req.TrueLabel)
	imageMetadata := map[string]any{}
	for key, value := range req.ImageMetadata {
		imageMetadata[key] = value
	}
	if req.ConfidenceThreshold != nil {
		imageMetadata["confidence_threshold"] = *req.ConfidenceThreshold
	}
	if req.IOUThreshold != nil {
		imageMetadata["iou_threshold"] = *req.IOUThreshold
	}
	if req.MaxDetections > 0 {
		imageMetadata["max_detections"] = req.MaxDetections
	}
	matchedStoredDemoImage := false
	if matchedImageID, matchedTrueLabel, matchedMetadata, ok := championDemoImageMetadata(championHeldoutDemoImageProfile(champion), imageURI); ok {
		matchedStoredDemoImage = true
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
	} else if err == nil {
		if matchedImageID, matchedTrueLabel, matchedMetadata, ok := championDemoImageMetadata(dataset.Profile, imageURI); ok {
			matchedStoredDemoImage = true
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
	backendImageURI := championDemoInferenceImageURI(imageURI, imageMetadata)
	if backendImageURI != imageURI {
		imageMetadata["requested_image_uri"] = imageURI
		imageMetadata["backend_image_uri"] = backendImageURI
	}
	storedOriginalUnavailable := matchedStoredDemoImage && !championDemoHasOriginalInferenceURI(imageMetadata)
	predictionStatus := runs.ChampionDemoPredictionStatusRuntimeUnavailable
	predictionError := "champion demo inference is local-only in Mission Control; run local Python inference and record results through /champion/demo-predictions/local-result"
	if storedOriginalUnavailable {
		imageMetadata["error_code"] = championDemoOriginalUnavailableCode
		predictionStatus = runs.ChampionDemoPredictionStatusFailed
		predictionError = championDemoOriginalUnavailableCode + ": stored demo images need original_image_uri or source_artifact_uri for inference; preview_uri and thumbnail_uri are display-only"
	} else {
		imageMetadata["local_runtime_required"] = true
		imageMetadata["legacy_queued_demo_disabled"] = true
		if strings.TrimSpace(payloadString(imageMetadata, "inference_transport")) == "" {
			imageMetadata["inference_transport"] = "mission_control_local_python"
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
		Status:        predictionStatus,
		TrueLabel:     trueLabel,
		TopK:          []runs.DemoPredictionTopK{},
		Error:         predictionError,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	runtimeAvailable := false
	if _, err := s.store.CreateExecutionEvent(champion.ProjectID, champion.PlanID, execution.EventChampionDemoPrediction, fmt.Sprintf("Champion demo prediction requested for job %s.", champion.JobID), map[string]any{
		"champion_id":   champion.ID,
		"prediction_id": prediction.ID,
		"job_id":        champion.JobID,
		"status":        prediction.Status,
		"image_uri":     prediction.ImageURI,
		"local_only":    true,
	}); err != nil {
		log.Printf("record champion demo prediction event failed: %v", err)
	}

	c.JSON(http.StatusAccepted, gin.H{
		"prediction":        prediction,
		"runtime_available": runtimeAvailable,
		"contract": gin.H{
			"champion_job_id": champion.JobID,
			"image_uri":       imageURI,
			"local_only":      true,
			"top_k":           req.TopK,
			"returns":         []string{"predicted_label", "true_label", "confidence", "top_k", "latency_ms", "correct", "image_metadata.detections"},
		},
	})
}

func (s *Server) listProjectChampionFeedback(c *gin.Context) {
	feedback, err := s.store.ListProjectChampionFeedback(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"feedback": feedback})
}

func (s *Server) createProjectChampionFeedback(c *gin.Context) {
	projectID := c.Param("id")
	var req createChampionFeedbackRequest
	if !bindJSON(c, &req) {
		return
	}

	rating := normalizeChampionFeedbackRating(req.Rating)
	if rating == "" {
		writeStoreError(c, fmt.Errorf("%w: rating must be good, mediocre, or bad", store.ErrInvalidRequest))
		return
	}
	champion, err := s.store.GetProjectChampion(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	predictionSnapshot := copyPayloadMap(req.PredictionSnapshot)
	predictionID := strings.TrimSpace(req.PredictionID)
	imageURI := strings.TrimSpace(req.ImageURI)
	imageID := strings.TrimSpace(req.ImageID)
	if predictionID != "" && !strings.HasPrefix(predictionID, "local-") {
		if prediction, ok, err := s.findProjectChampionDemoPrediction(projectID, predictionID); err != nil {
			writeStoreError(c, err)
			return
		} else if ok {
			if len(predictionSnapshot) == 0 {
				if payload, payloadErr := mapFromStruct(prediction); payloadErr == nil {
					predictionSnapshot = payload
				}
			}
			if imageURI == "" {
				imageURI = prediction.ImageURI
			}
			if imageID == "" {
				imageID = prediction.ImageID
			}
		}
	}
	if imageURI == "" {
		imageURI = firstString(predictionSnapshot, "image_uri", "uri")
	}
	if imageID == "" {
		imageID = firstString(predictionSnapshot, "image_id", "id")
	}

	metricsSnapshot := championFeedbackMetricsSnapshot(champion)
	if len(req.MetricsSnapshot) > 0 {
		metricsSnapshot["user_supplied_metrics"] = copyPayloadMap(req.MetricsSnapshot)
	}
	metadata := copyPayloadMap(req.Metadata)
	metadata["source"] = "champion_test_feedback"
	metadata["rating_scale"] = []string{runs.ChampionFeedbackRatingGood, runs.ChampionFeedbackRatingMediocre, runs.ChampionFeedbackRatingBad}

	created, err := s.store.CreateChampionFeedback(runs.ChampionFeedbackCreate{
		ProjectID:          champion.ProjectID,
		ChampionID:         champion.ID,
		PredictionID:       predictionID,
		JobID:              champion.JobID,
		DatasetID:          champion.DatasetID,
		ImageURI:           imageURI,
		ImageID:            imageID,
		Rating:             rating,
		Message:            strings.TrimSpace(req.Message),
		PredictionSnapshot: predictionSnapshot,
		MetricsSnapshot:    metricsSnapshot,
		Metadata:           metadata,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		ProjectID: champion.ProjectID,
		DatasetID: champion.DatasetID,
		PlanID:    champion.PlanID,
		JobID:     champion.JobID,
		AgentName: "human_champion_feedback",
		Kind:      memory.KindChampionFeedback,
		Summary:   championFeedbackMemorySummary(created, champion),
		Payload: map[string]any{
			"outcome_status":       rating,
			"human_rating":         rating,
			"human_message":        created.Message,
			"champion_job_id":      champion.JobID,
			"champion_model":       payloadString(champion.Metrics, "model"),
			"prediction_id":        created.PredictionID,
			"prediction_snapshot":  created.PredictionSnapshot,
			"metrics_snapshot":     created.MetricsSnapshot,
			"deployment_feedback":  true,
			"primary_mechanism":    "model_selection",
			"intervention":         "champion_export_feedback",
			"lesson":               championFeedbackLesson(created, champion),
			"accepted_for_memory":  true,
			"accepted_for_vector":  true,
			"rating_scale_version": "champion_feedback_v1",
		},
		Tags: []string{"champion_feedback", rating, "user_reported"},
	}); err == nil {
		s.indexMemoryCard(c.Request.Context(), memory.NewAgentMemoryCard(record))
	} else {
		log.Printf("record champion feedback memory failed: %v", err)
	}

	if _, err := s.store.CreateExecutionEvent(champion.ProjectID, champion.PlanID, execution.EventChampionFeedbackRecorded, fmt.Sprintf("Champion feedback recorded as %s for job %s.", rating, champion.JobID), map[string]any{
		"champion_id":     champion.ID,
		"champion_job_id": champion.JobID,
		"feedback_id":     created.ID,
		"prediction_id":   created.PredictionID,
		"rating":          created.Rating,
	}); err != nil {
		log.Printf("record champion feedback event failed: %v", err)
	}

	c.JSON(http.StatusCreated, gin.H{"feedback": created})
}

func (s *Server) findProjectChampionDemoPrediction(projectID string, predictionID string) (runs.ChampionDemoPrediction, bool, error) {
	predictions, err := s.store.ListProjectChampionDemoPredictions(projectID)
	if err != nil {
		return runs.ChampionDemoPrediction{}, false, err
	}
	for _, prediction := range predictions {
		if prediction.ID == predictionID {
			return prediction, true, nil
		}
	}
	return runs.ChampionDemoPrediction{}, false, nil
}

func normalizeChampionFeedbackRating(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "good", "up", "thumbs_up", "thumbsup", "positive", "yes":
		return runs.ChampionFeedbackRatingGood
	case "mediocre", "neutral", "mixed", "okay", "ok", "meh":
		return runs.ChampionFeedbackRatingMediocre
	case "bad", "down", "thumbs_down", "thumbsdown", "negative", "no":
		return runs.ChampionFeedbackRatingBad
	default:
		return ""
	}
}

func championFeedbackMetricsSnapshot(champion runs.ProjectChampion) map[string]any {
	return map[string]any{
		"champion_id":        champion.ID,
		"champion_job_id":    champion.JobID,
		"champion_metrics":   copyPayloadMap(champion.Metrics),
		"evaluation":         copyPayloadMap(champion.Evaluation),
		"deployment_profile": copyPayloadMap(champion.DeploymentProfile),
	}
}

func championFeedbackMemorySummary(feedback runs.ChampionFeedback, champion runs.ProjectChampion) string {
	model := payloadString(champion.Metrics, "model")
	if model == "" {
		model = champion.JobID
	}
	message := strings.TrimSpace(feedback.Message)
	if message != "" {
		return fmt.Sprintf("User rated champion %s (%s) as %s after export/demo testing: %s", champion.JobID, model, feedback.Rating, message)
	}
	return fmt.Sprintf("User rated champion %s (%s) as %s after export/demo testing.", champion.JobID, model, feedback.Rating)
}

func championFeedbackLesson(feedback runs.ChampionFeedback, champion runs.ProjectChampion) string {
	model := payloadString(champion.Metrics, "model")
	if model == "" {
		model = champion.JobID
	}
	switch feedback.Rating {
	case runs.ChampionFeedbackRatingBad:
		return fmt.Sprintf("Treat %s metrics as insufficient for deployment unless held-out/custom image behavior improves.", model)
	case runs.ChampionFeedbackRatingMediocre:
		return fmt.Sprintf("Treat %s as a borderline champion; future runs should improve user-visible held-out/custom predictions, not only aggregate metrics.", model)
	default:
		return fmt.Sprintf("User-visible testing supported %s as a useful champion; similar metric and preprocessing profiles may be deployable.", model)
	}
}

func (s *Server) reportChampionExportResult(c *gin.Context) {
	var req championExportResultRequest
	if !bindJSON(c, &req) {
		return
	}
	job, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"champion_export_result",
		map[string]any{"status": req.Status},
	)
	if !ok {
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
	req.Metadata = championExportResultMetadata(req)
	existingExport, err := s.findProjectChampionExport(job.ProjectID, exportID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if status == runs.ChampionExportStatusReady {
		if err := validateChampionExportReadyResult(job, existingExport, req); err != nil {
			writeStoreError(c, err)
			return
		}
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
	if status == runs.ChampionExportStatusReady {
		s.closeRemoteTrainingSession(job, jobs.StatusSucceeded)
	} else if status == runs.ChampionExportStatusFailed {
		s.closeRemoteTrainingSession(job, jobs.StatusFailed)
	}
	c.JSON(http.StatusOK, gin.H{"export": export, "job": job})
}

func (s *Server) findProjectChampionExport(projectID string, exportID string) (runs.ChampionExport, error) {
	exports, err := s.store.ListProjectChampionExports(projectID)
	if err != nil {
		return runs.ChampionExport{}, err
	}
	for _, export := range exports {
		if export.ID == strings.TrimSpace(exportID) {
			return export, nil
		}
	}
	return runs.ChampionExport{}, store.ErrNotFound
}

func validateChampionExportReadyResult(job jobs.ExperimentJob, export runs.ChampionExport, req championExportResultRequest) error {
	artifactURI := strings.TrimSpace(req.ArtifactURI)
	if !artifactMatchesChampionExportFormat(artifactURI, export.Format) {
		return fmt.Errorf("%w: artifact_uri does not match requested champion export format", store.ErrInvalidRequest)
	}
	if !trustedChampionExportArtifactURI(artifactURI) && !strings.HasPrefix(strings.ToLower(artifactURI), "file://") {
		return fmt.Errorf("%w: artifact_uri must be a trusted S3 artifact or worker-owned file URI", store.ErrInvalidRequest)
	}
	manifest := payloadMap(req.Metadata, "manifest")
	if len(manifest) == 0 {
		return fmt.Errorf("%w: worker export manifest is required for READY export result", store.ErrInvalidRequest)
	}
	if payloadString(manifest, "schema_version") != "champion_export_manifest_v1" {
		return fmt.Errorf("%w: worker export manifest schema is invalid", store.ErrInvalidRequest)
	}
	if !championExportManifestHasCreatedArtifact(manifest, export.Format) {
		return fmt.Errorf("%w: worker export manifest does not include a created artifact for the requested format", store.ErrInvalidRequest)
	}
	if !championExportManifestHasProvenance(manifest, job.ID, export.ID) {
		return fmt.Errorf("%w: worker export manifest provenance does not match the export job", store.ErrInvalidRequest)
	}
	if championExportManifestSelfTestFailed(manifest) {
		return fmt.Errorf("%w: worker export manifest ONNX self-test failed", store.ErrInvalidRequest)
	}
	return nil
}

func championExportManifestHasCreatedArtifact(manifest map[string]any, format string) bool {
	artifacts, ok := manifest["artifacts"].([]any)
	if !ok {
		return false
	}
	for _, item := range artifacts {
		artifact := mapFromAny(item)
		if len(artifact) == 0 {
			continue
		}
		if !strings.EqualFold(payloadString(artifact, "status"), "created") {
			continue
		}
		artifactFormat := strings.ToLower(strings.TrimSpace(firstNonEmptyString(
			payloadString(artifact, "format"),
			payloadString(artifact, "artifact_format"),
		)))
		if artifactFormat == "" {
			artifactURI := firstNonEmptyString(
				payloadString(artifact, "uri"),
				payloadString(artifact, "artifact_uri"),
				payloadString(artifact, "path"),
			)
			if artifactMatchesChampionExportFormat(artifactURI, format) {
				return true
			}
			continue
		}
		if artifactFormat == strings.ToLower(strings.TrimSpace(format)) ||
			(format == "pytorch" && (artifactFormat == "framework_native" || artifactFormat == "framework_native_checkpoint")) {
			return true
		}
	}
	return false
}

func championExportResultMetadata(req championExportResultRequest) map[string]any {
	metadata := copyPayloadMap(req.Metadata)
	manifestURI := strings.TrimSpace(req.ManifestURI)
	if manifestURI != "" {
		if _, ok := metadata["manifest_uri"]; !ok {
			metadata["manifest_uri"] = manifestURI
		}
		if _, ok := metadata["export_manifest_uri"]; !ok {
			metadata["export_manifest_uri"] = manifestURI
		}
	}
	return metadata
}

func championExportManifestSelfTestFailed(manifest map[string]any) bool {
	metadata := payloadMap(manifest, "metadata")
	selfTest := payloadMap(metadata, "export_self_test")
	status := strings.ToLower(strings.TrimSpace(payloadString(selfTest, "status")))
	return status == "failed"
}

func championExportManifestHasProvenance(manifest map[string]any, exportJobID string, exportID string) bool {
	metadata := payloadMap(manifest, "metadata")
	if championExportProvenanceMatches(payloadMap(metadata, "provenance"), exportJobID, exportID) {
		return true
	}
	artifacts, ok := manifest["artifacts"].([]any)
	if !ok {
		return false
	}
	for _, item := range artifacts {
		artifact := mapFromAny(item)
		if len(artifact) == 0 {
			continue
		}
		if championExportProvenanceMatches(payloadMap(artifact, "provenance"), exportJobID, exportID) {
			return true
		}
	}
	return false
}

func championExportProvenanceMatches(provenance map[string]any, exportJobID string, exportID string) bool {
	if len(provenance) == 0 {
		return false
	}
	if payloadString(provenance, "schema_version") != "worker_artifact_provenance_v1" ||
		payloadString(provenance, "generated_by") != "model-express-worker" {
		return false
	}
	source := payloadString(provenance, "source")
	if source != "worker_generated" && source != "worker_controlled_copy" && source != "controlled_legacy_manifest_fallback" {
		return false
	}
	return payloadString(provenance, "export_job_id") == strings.TrimSpace(exportJobID) ||
		payloadString(provenance, "source_export_id") == strings.TrimSpace(exportID)
}

func (s *Server) reportChampionDemoPredictionResult(c *gin.Context) {
	var req championDemoPredictionResultRequest
	if !bindJSON(c, &req) {
		return
	}
	job, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"champion_demo_prediction_result",
		map[string]any{"status": req.Status},
	)
	if !ok {
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
	if status == runs.ChampionDemoPredictionStatusSucceeded && strings.TrimSpace(req.PredictedLabel) == "" && !championDemoPredictionHasDetectionMetadata(req.ImageMetadata) {
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
	} else {
		message := strings.TrimSpace(req.Error)
		if message == "" {
			message = "champion demo prediction failed"
		}
		job, err = s.store.FailJob(job.ID, message)
	}
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if status == runs.ChampionDemoPredictionStatusSucceeded {
		s.closeRemoteTrainingSession(job, jobs.StatusSucceeded)
	} else {
		s.closeRemoteTrainingSession(job, jobs.StatusFailed)
	}
	c.JSON(http.StatusOK, gin.H{"prediction": prediction, "job": job})
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
	for _, key := range []string{"heldout_demo_images", "demo_images", "test_images", "visual_exemplars", "exemplars"} {
		for _, entry := range profileEntries(profile[key]) {
			exemplar, ok := visualExemplarFromProfileEntry(entry)
			if !ok || !championDemoImageURIMatches(exemplar, imageURI) {
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

func championDemoImageURIMatches(exemplar datasets.VisualExemplar, imageURI string) bool {
	imageURI = strings.TrimSpace(imageURI)
	for _, candidate := range championDemoImageURICandidates(exemplar) {
		if strings.TrimSpace(candidate) == imageURI {
			return true
		}
	}
	return false
}

func championDemoImageURICandidates(exemplar datasets.VisualExemplar) []string {
	candidates := []string{exemplar.URI}
	for _, key := range []string{"image_uri", "uri", "preview_uri", "thumbnail_uri", "original_image_uri", "source_artifact_uri"} {
		if value := firstString(exemplar.Metadata, key); value != "" {
			candidates = append(candidates, value)
		}
	}
	return candidates
}

func championDemoInferenceImageURI(requestedImageURI string, metadata map[string]any) string {
	for _, key := range []string{"original_image_uri", "source_artifact_uri"} {
		if uri := strings.TrimSpace(firstString(metadata, key)); uri != "" {
			return uri
		}
	}
	return requestedImageURI
}

func championDemoHasOriginalInferenceURI(metadata map[string]any) bool {
	for _, key := range []string{"original_image_uri", "source_artifact_uri"} {
		if strings.TrimSpace(firstString(metadata, key)) != "" {
			return true
		}
	}
	return false
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

func championArtifactURIForFormat(deploymentProfile map[string]any, format string) string {
	modelProfile := payloadMap(deploymentProfile, "model_profile")
	for _, artifactURI := range championArtifactURICandidatesForFormat(deploymentProfile, modelProfile, format) {
		if artifactMatchesChampionExportFormat(artifactURI, format) {
			return artifactURI
		}
	}
	return championArtifactURI(deploymentProfile)
}

func championArtifactURICandidatesForFormat(deploymentProfile map[string]any, modelProfile map[string]any, format string) []string {
	switch format {
	case "onnx":
		return []string{
			firstString(deploymentProfile, "onnx_artifact_uri"),
			firstString(modelProfile, "onnx_artifact_uri"),
			firstString(deploymentProfile, "artifact_uri"),
			firstString(modelProfile, "artifact_uri"),
		}
	case "torchscript":
		return []string{
			firstString(deploymentProfile, "torchscript_artifact_uri", "torchscript_uri"),
			firstString(modelProfile, "torchscript_artifact_uri", "torchscript_uri"),
			firstString(deploymentProfile, "artifact_uri"),
			firstString(modelProfile, "artifact_uri"),
		}
	case "pytorch":
		return []string{
			firstString(deploymentProfile, "checkpoint_uri", "pytorch_uri", "model_uri"),
			firstString(modelProfile, "checkpoint_uri", "pytorch_uri", "model_uri"),
			firstString(deploymentProfile, "artifact_uri"),
			firstString(modelProfile, "artifact_uri"),
		}
	case "safetensors":
		return []string{
			firstString(deploymentProfile, "safetensors_artifact_uri", "safetensors_uri"),
			firstString(modelProfile, "safetensors_artifact_uri", "safetensors_uri"),
			firstString(deploymentProfile, "artifact_uri"),
			firstString(modelProfile, "artifact_uri"),
		}
	default:
		return []string{championArtifactURI(deploymentProfile)}
	}
}

func championArtifactURIFromEvaluation(modelProfile map[string]any) string {
	return firstString(
		modelProfile,
		"onnx_artifact_uri",
		"onnx_uri",
		"artifact_uri",
		"model_artifact_uri",
		"export_artifact_uri",
		"checkpoint_uri",
		"torchscript_artifact_uri",
		"torchscript_uri",
		"pytorch_uri",
		"model_uri",
		"safetensors_artifact_uri",
		"safetensors_uri",
	)
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

func trustedChampionExportArtifactURI(artifactURI string) bool {
	parsed, err := url.Parse(strings.TrimSpace(artifactURI))
	if err != nil || parsed.Scheme != "s3" {
		return false
	}
	key := strings.TrimLeft(parsed.EscapedPath(), "/")
	key, _ = url.PathUnescape(key)
	return strings.HasPrefix(strings.TrimLeft(key, "/"), "model-express/artifacts/")
}

func championModelProfileUnexportable(modelProfile map[string]any) (bool, string) {
	if value, ok := modelProfile["exportable"].(bool); ok && !value {
		return true, "model profile is marked unexportable"
	}
	if payloadBool(modelProfile, "simulation") || payloadBool(modelProfile, "simulated_training") {
		return true, "local YOLO simulator output is not a deployable trained artifact"
	}
	status := strings.ToLower(strings.TrimSpace(firstString(modelProfile, "export_status", "artifact_profile_status")))
	if strings.Contains(status, "simulated") || strings.Contains(status, "unexportable") || strings.Contains(status, "simulation_only") {
		return true, "model profile export status is simulation-only"
	}
	runtime := strings.ToLower(strings.TrimSpace(firstString(modelProfile, "runtime")))
	if strings.Contains(runtime, "simulated") {
		return true, "model profile runtime is simulated"
	}
	return false, ""
}

func championDeploymentProfileHasReadyExport(deploymentProfile map[string]any, format string, artifactURI string) bool {
	if !artifactMatchesChampionExportFormat(artifactURI, format) || !trustedChampionExportArtifactURI(artifactURI) {
		return false
	}
	modelProfile := payloadMap(deploymentProfile, "model_profile")
	if unexportable, _ := championModelProfileUnexportable(modelProfile); unexportable {
		return false
	}
	manifest := payloadMap(deploymentProfile, "export_manifest")
	if len(manifest) == 0 {
		manifest = payloadMap(modelProfile, "export_manifest")
	}
	if len(manifest) == 0 || payloadString(manifest, "schema_version") != "champion_export_manifest_v1" {
		return false
	}
	if !championExportManifestHasCreatedArtifact(manifest, format) {
		return false
	}
	if championExportManifestSelfTestFailed(manifest) {
		return false
	}
	if !championExportManifestHasWorkerProvenance(manifest) {
		return false
	}
	if !championProfileExportStatusReady(deploymentProfile) && !championProfileExportStatusReady(modelProfile) {
		return false
	}
	return championExportValidationErrorsEmpty(deploymentProfile, modelProfile, manifest)
}

func championProfileExportStatusReady(profile map[string]any) bool {
	if len(profile) == 0 {
		return false
	}
	for _, key := range []string{"export_status", "status"} {
		if championExportStatusReadyValue(profile[key]) {
			return true
		}
	}
	return false
}

func championExportStatusReadyValue(value any) bool {
	switch typed := value.(type) {
	case string:
		normalized := strings.ToLower(strings.TrimSpace(typed))
		switch normalized {
		case "ready", "succeeded", "success", "created", "self_test_passed":
			return true
		default:
			return false
		}
	case map[string]any:
		return championExportStatusReadyValue(payloadString(typed, "status"))
	default:
		mapped := mapFromAny(value)
		return len(mapped) > 0 && championExportStatusReadyValue(payloadString(mapped, "status"))
	}
}

func championExportValidationErrorsEmpty(containers ...map[string]any) bool {
	for _, container := range containers {
		if len(nonEmptyStringValues(payloadStringSlice(container, "validation_errors"))) > 0 ||
			len(nonEmptyStringValues(payloadStringSlice(container, "export_validation_errors"))) > 0 {
			return false
		}
		metadata := payloadMap(container, "metadata")
		if len(metadata) > 0 && !championExportValidationErrorsEmpty(metadata) {
			return false
		}
		provenance := payloadMap(container, "provenance")
		if len(provenance) > 0 && len(nonEmptyStringValues(payloadStringSlice(provenance, "validation_errors"))) > 0 {
			return false
		}
	}
	return true
}

func championExportManifestHasWorkerProvenance(manifest map[string]any) bool {
	metadata := payloadMap(manifest, "metadata")
	if championExportWorkerProvenance(payloadMap(metadata, "provenance")) {
		return true
	}
	artifacts, ok := manifest["artifacts"].([]any)
	if !ok {
		return false
	}
	for _, item := range artifacts {
		artifact := mapFromAny(item)
		if championExportWorkerProvenance(payloadMap(artifact, "provenance")) {
			return true
		}
	}
	return false
}

func championExportWorkerProvenance(provenance map[string]any) bool {
	return payloadString(provenance, "schema_version") == "worker_artifact_provenance_v1" &&
		payloadString(provenance, "generated_by") == "model-express-worker"
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

func championDemoPredictionHasDetectionMetadata(metadata map[string]any) bool {
	if len(metadata) == 0 {
		return false
	}
	taskType := strings.ToLower(strings.TrimSpace(payloadString(metadata, "task_type")))
	if strings.Contains(taskType, "object_detection") || strings.Contains(taskType, "detect") {
		return true
	}
	if _, ok := metadata["detections"]; ok {
		return true
	}
	if _, ok := metadata["detection_count"]; ok {
		return true
	}
	return false
}

func championExportManifestPath(metadata map[string]any) string {
	manifest, _ := metadata["manifest"].(map[string]any)
	if manifestPath := firstString(manifest, "manifest_path", "local_manifest_path", "manifest_uri", "export_manifest_uri"); manifestPath != "" {
		return manifestPath
	}
	return firstString(metadata, "manifest_path", "local_manifest_path", "export_manifest_path", "manifest_uri", "export_manifest_uri")
}

func championDatasetID(champion runs.ProjectChampion, job jobs.ExperimentJob) string {
	if champion.DatasetID != "" {
		return champion.DatasetID
	}
	return jobConfigString(job.Config, "dataset_id")
}

func usableChampionExport(dataStore store.Store, champion runs.ProjectChampion) (runs.ChampionExport, bool) {
	exports, err := dataStore.ListProjectChampionExports(champion.ProjectID)
	if err != nil {
		return runs.ChampionExport{}, false
	}
	for _, export := range exports {
		if export.ChampionID == champion.ID &&
			export.Status == runs.ChampionExportStatusReady &&
			strings.TrimSpace(export.ArtifactURI) != "" &&
			artifactMatchesChampionExportFormat(export.ArtifactURI, export.Format) {
			return export, true
		}
	}
	if export, ok := championDeploymentProfileExport(champion); ok {
		return export, true
	}
	return runs.ChampionExport{}, false
}

func championDeploymentProfileExport(champion runs.ProjectChampion) (runs.ChampionExport, bool) {
	artifactURI := championArtifactURI(champion.DeploymentProfile)
	if artifactURI == "" {
		return runs.ChampionExport{}, false
	}
	if !trustedChampionExportArtifactURI(artifactURI) {
		return runs.ChampionExport{}, false
	}
	format := championExportFormatFromArtifactURI(artifactURI)
	if format == "" {
		format = "pytorch"
	}
	if !championDeploymentProfileHasReadyExport(champion.DeploymentProfile, format, artifactURI) {
		return runs.ChampionExport{}, false
	}
	metadata := championDeploymentExportMetadata(champion, format)
	return runs.ChampionExport{
		ID:          champion.ID + "-deployment-artifact",
		ProjectID:   champion.ProjectID,
		ChampionID:  champion.ID,
		JobID:       champion.JobID,
		Status:      runs.ChampionExportStatusReady,
		Format:      format,
		ArtifactURI: artifactURI,
		Metadata:    metadata,
	}, true
}

func championDeploymentExportMetadata(champion runs.ProjectChampion, format string) map[string]any {
	deploymentProfile := champion.DeploymentProfile
	modelProfile := payloadMap(deploymentProfile, "model_profile")
	metadata := championExportMetadata(champion, format, map[string]any{
		"deployment_profile": deploymentProfile,
		"model_profile":      modelProfile,
	})
	if manifest := payloadMap(deploymentProfile, "export_manifest"); len(manifest) > 0 {
		metadata["manifest"] = manifest
	} else if manifest := payloadMap(modelProfile, "export_manifest"); len(manifest) > 0 {
		metadata["manifest"] = manifest
	}
	if manifestURI := firstString(deploymentProfile, "export_manifest_uri", "manifest_uri"); manifestURI != "" {
		metadata["manifest_uri"] = manifestURI
	} else if manifestURI := firstString(modelProfile, "export_manifest_uri", "manifest_uri"); manifestURI != "" {
		metadata["manifest_uri"] = manifestURI
	}
	return metadata
}

func championExportFormatFromArtifactURI(artifactURI string) string {
	switch {
	case artifactMatchesChampionExportFormat(artifactURI, "onnx"):
		return "onnx"
	case artifactMatchesChampionExportFormat(artifactURI, "torchscript"):
		return "torchscript"
	case artifactMatchesChampionExportFormat(artifactURI, "safetensors"):
		return "safetensors"
	case artifactMatchesChampionExportFormat(artifactURI, "pytorch"):
		return "pytorch"
	default:
		return ""
	}
}
