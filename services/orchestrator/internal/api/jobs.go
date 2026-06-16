package api

import (
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

type recordModalCallRequest struct {
	TrainingAttemptID         string         `json:"training_attempt_id"`
	ModalFunctionCallObjectID string         `json:"modal_function_call_object_id"`
	ModalFunctionCallID       string         `json:"modal_function_call_id"`
	ModalInputID              string         `json:"modal_input_id"`
	CancelStatus              string         `json:"cancel_status"`
	RequestedGPUType          string         `json:"requested_gpu_type"`
	EffectiveGPUType          string         `json:"effective_gpu_type"`
	MemoryMB                  int            `json:"memory_mb"`
	RequestedBatchSize        int            `json:"requested_batch_size"`
	EffectiveBatchSize        int            `json:"effective_batch_size"`
	BatchSizePolicy           string         `json:"batch_size_policy"`
	ModalResourceSignature    string         `json:"modal_resource_signature"`
	ModalResources            map[string]any `json:"modal_resources"`
}

type reportMetricRequest struct {
	Epoch             int                `json:"epoch" binding:"required"`
	Metrics           map[string]float64 `json:"metrics" binding:"required"`
	TrainingAttemptID string             `json:"training_attempt_id"`
}

type completeJobRequest struct {
	MLflowRunID       string `json:"mlflow_run_id"`
	TrainingAttemptID string `json:"training_attempt_id"`
}

type failJobRequest struct {
	Error                  string         `json:"error" binding:"required"`
	Retryable              bool           `json:"retryable"`
	TrainingAttemptID      string         `json:"training_attempt_id"`
	FailureClass           string         `json:"failure_class"`
	FailureType            string         `json:"failure_type"`
	OOM                    bool           `json:"oom"`
	OOMKind                string         `json:"oom_kind"`
	RequestedGPUType       string         `json:"requested_gpu_type"`
	EffectiveGPUType       string         `json:"effective_gpu_type"`
	MemoryMB               int            `json:"memory_mb"`
	RequestedBatchSize     int            `json:"requested_batch_size"`
	EffectiveBatchSize     int            `json:"effective_batch_size"`
	BatchSizePolicy        string         `json:"batch_size_policy"`
	ModalResourceSignature string         `json:"modal_resource_signature"`
	ModalFunctionCallID    string         `json:"modal_function_call_id"`
	ModalInputID           string         `json:"modal_input_id"`
	ModalResources         map[string]any `json:"modal_resources"`
}

type reportTrainingRunSummaryRequest = runs.TrainingRunSummaryUpdate
type reportTrainingRunEvaluationRequest = runs.TrainingRunEvaluationUpdate

func (s *Server) upsertTrainingRunSummary(c *gin.Context) {
	var req reportTrainingRunSummaryRequest
	if !bindJSON(c, &req) {
		return
	}
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"training_run_summary",
		trainingResourcePayload(req.TrainingAttemptID, req.RequestedGPUType, req.EffectiveGPUType, req.ModalResourceSignature),
	); !ok {
		return
	}

	summary, err := s.store.UpsertTrainingRunSummary(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if err := s.observeTrainingRunMaterialization(c.Param("id"), summary); err != nil {
		log.Printf("training run materialization observation failed for job %s: %v", c.Param("id"), err)
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

func (s *Server) observeTrainingRunMaterialization(jobID string, summary runs.TrainingRunSummary) error {
	if len(summary.DatasetMaterialization) == 0 {
		return nil
	}
	status := trainingRunMaterializationRequirementStatus(summary.DatasetMaterialization)
	if status == "" {
		return nil
	}
	job, err := s.store.GetJob(jobID)
	if err != nil {
		return err
	}
	if job.Template != jobs.TemplateTrainExperiment {
		return nil
	}
	planID := strings.TrimSpace(summary.PlanID)
	if planID == "" {
		planID = jobConfigString(job.Config, "plan_id")
	}
	if planID == "" {
		return nil
	}
	provider := strings.ToLower(strings.TrimSpace(summary.Provider))
	if provider == "" {
		provider = strings.ToLower(jobConfigString(job.Config, "provider"))
	}
	cacheKey := strings.TrimSpace(payloadString(summary.DatasetMaterialization, "dataset_materialization_cache_key"))
	if cacheKey == "" {
		cacheKey = strings.TrimSpace(payloadString(summary.DatasetMaterialization, "dataset_cache_key"))
	}
	requirements, err := s.store.ListProjectWorkerRequirements(job.ProjectID)
	if err != nil {
		return err
	}
	for _, requirement := range requirements {
		if requirement.PlanID != planID {
			continue
		}
		if provider != "" {
			requirementProvider := strings.ToLower(strings.TrimSpace(requirement.Provider))
			requirementGPU := strings.ToLower(strings.TrimSpace(requirement.GPUType))
			if requirementProvider != provider && requirementGPU != provider {
				continue
			}
		}
		if cacheKey != "" && requirement.DatasetCacheKey != "" && requirement.DatasetCacheKey != cacheKey {
			continue
		}
		if requirement.DatasetMaterializationStatus == status {
			continue
		}
		nextStatus := status
		if _, err := s.store.UpdateWorkerRequirement(
			requirement.ID,
			execution.WorkerRequirementUpdate{DatasetMaterializationStatus: &nextStatus},
		); err != nil {
			return err
		}
	}
	return nil
}

func trainingRunMaterializationRequirementStatus(materialization map[string]any) string {
	reuseStatus := strings.ToLower(strings.TrimSpace(payloadString(materialization, "dataset_prewarm_reuse_status")))
	if strings.HasPrefix(reuseStatus, "staging_only") {
		return execution.DatasetMaterializationStagingOnly
	}
	if explicitFalseBool(materialization, "dataset_prewarm_reusable_for_training") {
		return execution.DatasetMaterializationStagingOnly
	}
	status := strings.ToLower(strings.TrimSpace(payloadString(materialization, "dataset_materialization_status")))
	switch status {
	case "hit", "hit_after_wait", "materialized":
		return execution.DatasetMaterializationWarm
	case "materializing", "checking":
		return execution.DatasetMaterializationMaterializing
	}
	if payloadBool(materialization, "dataset_materialization_cache_hit") {
		return execution.DatasetMaterializationWarm
	}
	if payloadBool(materialization, "dataset_materialization_cache_miss") {
		return execution.DatasetMaterializationMaterializing
	}
	return ""
}

func explicitFalseBool(payload map[string]any, key string) bool {
	value, ok := payload[key]
	if !ok {
		return false
	}
	switch typed := value.(type) {
	case bool:
		return !typed
	case string:
		parsed, ok := envFlagValueFromString(typed)
		return ok && !parsed
	default:
		return false
	}
}

func (s *Server) upsertTrainingRunEvaluation(c *gin.Context) {
	var req reportTrainingRunEvaluationRequest
	if !bindJSON(c, &req) {
		return
	}
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"training_run_evaluation",
		trainingResourcePayload(req.TrainingAttemptID, req.RequestedGPUType, req.EffectiveGPUType, req.ModalResourceSignature),
	); !ok {
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
	projectID := c.Param("id")
	limit := queryInt(c, "limit", defaultTrainingSummariesLimit, 1, maxTrainingSummariesLimit)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	items, err := s.store.ListProjectTrainingRunSummariesPage(projectID, store.PageOptions{Limit: limit + 1, Offset: offset})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	summaries, hasMore := pageHasMore(items, limit)
	summaries = s.reconcileTrainingSummaryTerminalStatus(projectID, summaries)
	c.JSON(http.StatusOK, pagedListPayload("summaries", summaries, limit, offset, hasMore))
}

func (s *Server) reconcileTrainingSummaryTerminalStatus(projectID string, summaries []runs.TrainingRunSummary) []runs.TrainingRunSummary {
	if len(summaries) == 0 {
		return summaries
	}
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		log.Printf("training summary status reconciliation failed for project %s: %v", projectID, err)
		return summaries
	}
	jobStatusByID := map[string]string{}
	for _, job := range projectJobs {
		if jobStatusIsTerminal(job.Status) {
			jobStatusByID[job.ID] = job.Status
		}
	}
	if len(jobStatusByID) == 0 {
		return summaries
	}
	reconciled := append([]runs.TrainingRunSummary(nil), summaries...)
	for index := range reconciled {
		if status, ok := jobStatusByID[reconciled[index].JobID]; ok {
			reconciled[index].Status = status
		}
	}
	return reconciled
}

func (s *Server) listProjectTrainingRunEvaluations(c *gin.Context) {
	limit := queryInt(c, "limit", defaultTrainingEvaluationsLimit, 1, maxTrainingEvaluationsLimit)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	items, err := s.store.ListProjectTrainingRunEvaluationsPage(c.Param("id"), store.PageOptions{Limit: limit + 1, Offset: offset})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	evaluations, hasMore := pageHasMore(items, limit)
	if queryBool(c, "compact") {
		evaluations = compactTrainingRunEvaluations(evaluations)
	}
	c.JSON(http.StatusOK, pagedListPayload("evaluations", evaluations, limit, offset, hasMore))
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

func compactTrainingRunEvaluations(evaluations []runs.TrainingRunEvaluation) []runs.TrainingRunEvaluation {
	out := append([]runs.TrainingRunEvaluation(nil), evaluations...)
	for index := range out {
		if out[index].HolisticScores != nil {
			out[index].HolisticScores = copyPayloadMap(out[index].HolisticScores)
		}
		if len(out[index].PerClassMetrics) > 20 {
			out[index].PerClassMetrics = map[string]any{"_truncated": true, "class_count": len(out[index].PerClassMetrics)}
		}
		if len(out[index].ConfusionMatrix) > 20 {
			out[index].ConfusionMatrix = nil
			if out[index].HolisticScores == nil {
				out[index].HolisticScores = map[string]any{}
			}
			out[index].HolisticScores["confusion_matrix_truncated"] = true
		}
	}
	return out
}

func (s *Server) recordJobModalCall(c *gin.Context) {
	var req recordModalCallRequest
	if !bindJSON(c, &req) {
		return
	}
	payload := trainingResourcePayload(req.TrainingAttemptID, req.RequestedGPUType, req.EffectiveGPUType, req.ModalResourceSignature)
	payload["modal_function_call_object_id"] = strings.TrimSpace(req.ModalFunctionCallObjectID)
	payload["modal_function_call_id"] = strings.TrimSpace(req.ModalFunctionCallID)
	payload["modal_input_id"] = strings.TrimSpace(req.ModalInputID)
	payload["cancel_status"] = strings.TrimSpace(req.CancelStatus)
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"modal_call",
		payload,
	); !ok {
		return
	}
	patch := map[string]any{
		"modal_function_call_object_id": strings.TrimSpace(req.ModalFunctionCallObjectID),
		"modal_function_call_id":        strings.TrimSpace(req.ModalFunctionCallID),
		"modal_input_id":                strings.TrimSpace(req.ModalInputID),
		"modal_call_cancel_status":      firstNonEmptyString(req.CancelStatus, "active"),
		"modal_call_recorded_at":        time.Now().UTC().Format(time.RFC3339Nano),
	}
	if req.TrainingAttemptID != "" {
		patch["training_attempt_id"] = strings.TrimSpace(req.TrainingAttemptID)
	}
	if req.RequestedGPUType != "" {
		patch["requested_gpu_type"] = strings.TrimSpace(req.RequestedGPUType)
	}
	if req.EffectiveGPUType != "" {
		patch["effective_gpu_type"] = strings.TrimSpace(req.EffectiveGPUType)
	}
	if req.ModalResourceSignature != "" {
		patch["modal_resource_signature"] = strings.TrimSpace(req.ModalResourceSignature)
	}
	if len(req.ModalResources) > 0 {
		patch["modal_resources"] = mergePayloadMap(req.ModalResources, map[string]any{
			"modal_function_call_object_id": strings.TrimSpace(req.ModalFunctionCallObjectID),
			"modal_function_call_id":        strings.TrimSpace(req.ModalFunctionCallID),
			"modal_input_id":                strings.TrimSpace(req.ModalInputID),
			"modal_call_cancel_status":      firstNonEmptyString(req.CancelStatus, "active"),
		})
	}
	job, err := s.store.UpdateJobConfig(c.Param("id"), patch)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, job)
}

func (s *Server) completeJob(c *gin.Context) {
	var req completeJobRequest
	if !bindJSON(c, &req) {
		return
	}
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"complete",
		map[string]any{"mlflow_run_id": req.MLflowRunID},
	); !ok {
		return
	}

	job, err := s.store.CompleteJob(c.Param("id"), req.MLflowRunID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	s.closeRemoteTrainingSession(job, jobs.StatusSucceeded)

	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusSucceeded,
		}); err != nil {
			log.Printf("post-complete training summary update failed for job %s: %v", job.ID, err)
		}
		s.enqueueTrainingTerminalHooks(job)
	}
	s.updateWorkerRequirementDemandAfterTerminalJob(job)

	c.JSON(http.StatusOK, job)
}

func (s *Server) failJob(c *gin.Context) {
	var req failJobRequest
	if !bindJSON(c, &req) {
		return
	}
	callbackPayload := failureCallbackEventPayload(req)
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"fail",
		callbackPayload,
	); !ok {
		return
	}

	if req.Retryable {
		currentJob, err := s.store.GetJob(c.Param("id"))
		if err != nil {
			writeStoreError(c, err)
			return
		}
		if jobStatusIsTerminal(currentJob.Status) {
			c.JSON(http.StatusOK, currentJob)
			return
		}
		retryOptions, retryDecision := s.retryOptionsForFailure(currentJob, req)
		job, requeued, err := s.store.RetryJob(c.Param("id"), req.Error, retryOptions)
		if err != nil {
			writeStoreError(c, err)
			return
		}
		diagnostics.Event("warn", "job_retryable_failure", map[string]any{
			"job_id":        job.ID,
			"project_id":    job.ProjectID,
			"worker_id":     job.WorkerID,
			"template":      job.Template,
			"attempt":       job.Attempt,
			"max_attempts":  job.MaxAttempts,
			"requeued":      requeued,
			"error":         req.Error,
			"failure_class": req.FailureClass,
			"oom_kind":      req.OOMKind,
			"retry_guard":   retryDecision.Status,
		})
		s.recordRetryableJobFailureEvent(job, requeued, req.Error, retryDecision)
		if job.Template == jobs.TemplateTrainExperiment {
			status := jobs.StatusQueued
			if !requeued {
				status = jobs.StatusFailed
			}
			if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
				Status: status,
			}); err != nil {
				log.Printf("retryable failure training summary update failed for job %s: %v", job.ID, err)
			}
			if !requeued {
				s.enqueueTrainingTerminalHooks(job)
				s.updateWorkerRequirementDemandAfterTerminalJob(job)
				s.closeRemoteTrainingSession(job, jobs.StatusFailed)
			}
		}
		if !requeued && job.Template != jobs.TemplateTrainExperiment {
			s.closeRemoteTrainingSession(job, jobs.StatusFailed)
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

	currentJob, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if jobStatusIsTerminal(currentJob.Status) {
		c.JSON(http.StatusOK, currentJob)
		return
	}

	job, err := s.store.FailJob(c.Param("id"), req.Error)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	s.closeRemoteTrainingSession(job, jobs.StatusFailed)
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
			log.Printf("failed-job training summary update failed for job %s: %v", job.ID, err)
		}
		s.enqueueTrainingTerminalHooks(job)
		s.updateWorkerRequirementDemandAfterTerminalJob(job)
	}
	if job.Template == jobs.TemplateAnalyzeDatasetVisuals && jobConfigString(job.Config, "trigger_reason") == string(datasets.VisualTriggerInitialProfile) {
		if err := s.createInitialPlanForDataset(jobConfigString(job.Config, "dataset_id")); err != nil {
			writeStoreError(c, err)
			return
		}
	}

	c.JSON(http.StatusOK, job)
}

type retryFailureDecision struct {
	Status              string
	Reason              string
	FailureClass        string
	OOMKind             string
	ResourceSignature   string
	PreviousGPUType     string
	NextGPUType         string
	EffectiveBatchSize  int
	MemoryMB            int
	RepeatedSignature   bool
	EscalationExhausted bool
}

func (s *Server) validateJobCallback(c *gin.Context, jobID string, trainingAttemptID string, callbackType string, payload map[string]any) (jobs.ExperimentJob, bool) {
	job, err := s.store.GetJob(jobID)
	if err != nil {
		writeStoreError(c, err)
		return jobs.ExperimentJob{}, false
	}
	attemptID := strings.TrimSpace(trainingAttemptID)
	if attemptID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "training_attempt_id is required"})
		return job, false
	}
	if !secureTokenEqual(callbackTokenFromRequest(c), s.callbackToken(job.ID, attemptID)) {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid callback token"})
		return job, false
	}
	activeAttemptID := strings.TrimSpace(jobConfigString(job.Config, "active_attempt_id"))
	if !jobStatusIsTerminal(job.Status) && activeAttemptID != "" && activeAttemptID == attemptID {
		if rejected := s.rejectInactiveRemoteTrainingSession(c, job, attemptID); rejected {
			return job, false
		}
		return job, true
	}

	s.recordStaleJobCallback(job, attemptID, activeAttemptID, callbackType, payload)
	c.JSON(http.StatusConflict, gin.H{
		"error":               "stale training attempt",
		"status":              "stale_attempt",
		"job_id":              job.ID,
		"training_attempt_id": attemptID,
		"active_attempt_id":   activeAttemptID,
		"job_status":          job.Status,
	})
	return job, false
}

func (s *Server) rejectInactiveRemoteTrainingSession(c *gin.Context, job jobs.ExperimentJob, attemptID string) bool {
	session := payloadMap(job.Config, "remote_training_session")
	if len(session) == 0 || payloadString(session, "training_attempt_id") != strings.TrimSpace(attemptID) {
		return false
	}
	sessionID := payloadString(session, "id")
	if sessionID != "" {
		record, err := s.store.GetRemoteTrainingSession(sessionID)
		if err == nil {
			if time.Now().UTC().After(record.ExpiresAt) || time.Now().UTC().Equal(record.ExpiresAt) {
				s.closeRemoteTrainingSession(job, runs.RemoteTrainingSessionStatusExpired)
				c.JSON(http.StatusUnauthorized, gin.H{"error": "remote training session expired"})
				return true
			}
			if store.NormalizeRemoteTrainingSessionStatus(record.Status) != runs.RemoteTrainingSessionStatusActive {
				c.JSON(http.StatusUnauthorized, gin.H{"error": "remote training session is not active"})
				return true
			}
		} else if !errors.Is(err, store.ErrNotFound) {
			writeStoreError(c, err)
			return true
		}
	}
	if remoteTrainingSessionExpired(job.Config, attemptID) {
		s.closeRemoteTrainingSession(job, runs.RemoteTrainingSessionStatusExpired)
		c.JSON(http.StatusUnauthorized, gin.H{"error": "remote training session expired"})
		return true
	}
	if status := store.NormalizeRemoteTrainingSessionStatus(payloadString(session, "status")); status != runs.RemoteTrainingSessionStatusActive {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "remote training session is not active"})
		return true
	}
	return false
}

func (s *Server) recordStaleJobCallback(job jobs.ExperimentJob, attemptID string, activeAttemptID string, callbackType string, payload map[string]any) {
	eventPayload := copyPayloadMap(payload)
	eventPayload["job_id"] = job.ID
	eventPayload["callback_type"] = callbackType
	eventPayload["training_attempt_id"] = attemptID
	eventPayload["active_attempt_id"] = activeAttemptID
	eventPayload["job_status"] = job.Status
	eventPayload["job_attempt"] = job.Attempt
	planID := jobConfigString(job.Config, "plan_id")
	if _, eventErr := s.store.CreateExecutionEvent(
		job.ProjectID,
		planID,
		execution.EventJobStaleCallbackIgnored,
		fmt.Sprintf("Ignored stale %s callback for job %s.", callbackType, job.ID),
		eventPayload,
	); eventErr != nil {
		log.Printf("record stale job callback event failed: %v", eventErr)
	}
	diagnostics.Event("warn", "job_stale_callback_ignored", eventPayload)
}

func jobStatusIsTerminal(status string) bool {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case jobs.StatusSucceeded, jobs.StatusFailed:
		return true
	default:
		return false
	}
}

func trainingResourcePayload(trainingAttemptID string, requestedGPU string, effectiveGPU string, signature string) map[string]any {
	payload := map[string]any{}
	if strings.TrimSpace(trainingAttemptID) != "" {
		payload["training_attempt_id"] = strings.TrimSpace(trainingAttemptID)
	}
	if strings.TrimSpace(requestedGPU) != "" {
		payload["requested_gpu_type"] = strings.TrimSpace(requestedGPU)
	}
	if strings.TrimSpace(effectiveGPU) != "" {
		payload["effective_gpu_type"] = strings.TrimSpace(effectiveGPU)
	}
	if strings.TrimSpace(signature) != "" {
		payload["modal_resource_signature"] = strings.TrimSpace(signature)
	}
	return payload
}

func failureCallbackEventPayload(req failJobRequest) map[string]any {
	payload := trainingResourcePayload(req.TrainingAttemptID, req.RequestedGPUType, req.EffectiveGPUType, req.ModalResourceSignature)
	payload["retryable"] = req.Retryable
	payload["failure_class"] = strings.TrimSpace(req.FailureClass)
	payload["failure_type"] = strings.TrimSpace(req.FailureType)
	payload["oom"] = req.OOM
	payload["oom_kind"] = strings.TrimSpace(req.OOMKind)
	payload["requested_batch_size"] = req.RequestedBatchSize
	payload["effective_batch_size"] = req.EffectiveBatchSize
	payload["memory_mb"] = req.MemoryMB
	payload["modal_function_call_id"] = strings.TrimSpace(req.ModalFunctionCallID)
	payload["modal_input_id"] = strings.TrimSpace(req.ModalInputID)
	return payload
}

func (s *Server) retryOptionsForFailure(job jobs.ExperimentJob, req failJobRequest) (store.RetryJobOptions, retryFailureDecision) {
	decision := retryFailureDecision{
		Status:             "standard_retry",
		FailureClass:       strings.ToLower(strings.TrimSpace(firstNonEmptyString(req.FailureClass, req.FailureType))),
		OOMKind:            strings.TrimSpace(req.OOMKind),
		ResourceSignature:  modalFailureResourceSignature(job, req),
		PreviousGPUType:    normalizeModalGPUType(firstNonEmptyString(req.EffectiveGPUType, req.RequestedGPUType, jobConfigString(job.Config, "gpu_type"))),
		EffectiveBatchSize: firstPositiveInt(req.EffectiveBatchSize, req.RequestedBatchSize, jobConfigInt(job.Config, "batch_size")),
		MemoryMB:           firstPositiveInt(req.MemoryMB, modalDefaultMemoryMB(firstNonEmptyString(req.EffectiveGPUType, req.RequestedGPUType, jobConfigString(job.Config, "gpu_type")), modalJobIsDetection(job.Config))),
	}
	if !confirmedOOMFailure(req) {
		return store.RetryJobOptions{}, decision
	}

	config := copyPayloadMap(job.Config)
	if config == nil {
		config = map[string]any{}
	}
	history := modalOOMRetryHistory(config)
	decision.FailureClass = "oom"
	decision.Status = "oom_gpu_escalation_pending"
	decision.RepeatedSignature = modalOOMHistoryContains(history, decision.ResourceSignature)
	history = append(history, map[string]any{
		"attempt":                  job.Attempt,
		"training_attempt_id":      strings.TrimSpace(req.TrainingAttemptID),
		"modal_resource_signature": decision.ResourceSignature,
		"effective_gpu_type":       decision.PreviousGPUType,
		"effective_batch_size":     decision.EffectiveBatchSize,
		"memory_mb":                decision.MemoryMB,
		"oom_kind":                 strings.TrimSpace(req.OOMKind),
		"modal_function_call_id":   strings.TrimSpace(req.ModalFunctionCallID),
		"modal_input_id":           strings.TrimSpace(req.ModalInputID),
		"recorded_at":              time.Now().UTC().Format(time.RFC3339Nano),
	})
	config[modalOOMRetryHistoryKey] = history

	if decision.RepeatedSignature {
		decision.Status = "oom_retry_blocked_same_resource"
		decision.Reason = "confirmed OOM already recorded for this GPU/batch/memory signature"
		return store.RetryJobOptions{Config: config, ForceFail: true}, decision
	}

	nextGPU, nextMemory := nextModalGPUForOOM(decision.PreviousGPUType, decision.EffectiveBatchSize, history, modalJobIsDetection(job.Config))
	if nextGPU == "" {
		decision.Status = "oom_escalation_exhausted"
		decision.Reason = "no untried GPU tier remains for the confirmed OOM batch signature"
		decision.EscalationExhausted = true
		return store.RetryJobOptions{Config: config, ForceFail: true}, decision
	}

	decision.NextGPUType = nextGPU
	decision.MemoryMB = nextMemory
	config["gpu_type"] = nextGPU
	config["resource_attempt"] = len(history) + 1
	config["modal_retry"] = map[string]any{
		"reason":                      "confirmed_oom_gpu_escalation",
		"previous_gpu_type":           decision.PreviousGPUType,
		"next_gpu_type":               nextGPU,
		"preserved_batch_size":        decision.EffectiveBatchSize,
		"previous_resource_signature": decision.ResourceSignature,
	}
	config["modal_resources"] = mergePayloadMap(payloadMap(config, "modal_resources"), map[string]any{
		"requested_gpu_type":       nextGPU,
		"effective_gpu_type":       nextGPU,
		"memory_mb":                nextMemory,
		"requested_batch_size":     decision.EffectiveBatchSize,
		"effective_batch_size":     decision.EffectiveBatchSize,
		"batch_size_policy":        "preserved",
		"resource_attempt":         len(history) + 1,
		"previous_oom_signature":   decision.ResourceSignature,
		"retry_reason":             "confirmed_oom_gpu_escalation",
		"modal_function_options":   map[string]any{"gpu": nextGPU, "memory": nextMemory},
		"modal_resource_signature": modalResourceSignature(nextGPU, decision.EffectiveBatchSize, nextMemory),
	})
	decision.Status = "oom_gpu_escalated"
	decision.Reason = "confirmed OOM escalated to next untried Modal GPU tier"
	return store.RetryJobOptions{Config: config}, decision
}

func confirmedOOMFailure(req failJobRequest) bool {
	if req.OOM {
		return true
	}
	failureClass := strings.ToLower(strings.TrimSpace(firstNonEmptyString(req.FailureClass, req.FailureType)))
	if failureClass == "oom" || failureClass == "out_of_memory" {
		return true
	}
	message := strings.ToLower(req.Error)
	return strings.Contains(message, "cuda out of memory") ||
		strings.Contains(message, "outofmemoryerror") ||
		strings.Contains(message, "out of memory") ||
		strings.Contains(message, "exit code 137") ||
		strings.Contains(message, "oom killed")
}

func modalFailureResourceSignature(job jobs.ExperimentJob, req failJobRequest) string {
	if signature := strings.TrimSpace(req.ModalResourceSignature); signature != "" {
		return signature
	}
	gpuType := firstNonEmptyString(req.EffectiveGPUType, req.RequestedGPUType, jobConfigString(job.Config, "gpu_type"), "T4")
	batchSize := firstPositiveInt(req.EffectiveBatchSize, req.RequestedBatchSize, jobConfigInt(job.Config, "batch_size"))
	memoryMB := firstPositiveInt(req.MemoryMB, modalDefaultMemoryMB(gpuType, modalJobIsDetection(job.Config)))
	return modalResourceSignature(gpuType, batchSize, memoryMB)
}

func modalOOMRetryHistory(config map[string]any) []map[string]any {
	raw, ok := config[modalOOMRetryHistoryKey].([]any)
	if !ok {
		if typed, typedOK := config[modalOOMRetryHistoryKey].([]map[string]any); typedOK {
			return append([]map[string]any(nil), typed...)
		}
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if typed := mapFromAny(item); len(typed) > 0 {
			out = append(out, typed)
		}
	}
	return out
}

func modalOOMHistoryContains(history []map[string]any, signature string) bool {
	signature = strings.TrimSpace(signature)
	if signature == "" {
		return false
	}
	for _, item := range history {
		if strings.TrimSpace(payloadString(item, "modal_resource_signature")) == signature ||
			strings.TrimSpace(payloadString(item, "resource_signature")) == signature {
			return true
		}
	}
	return false
}

func nextModalGPUForOOM(currentGPU string, batchSize int, history []map[string]any, detectionJob bool) (string, int) {
	current := normalizeModalGPUType(currentGPU)
	start := -1
	for index, candidate := range modalGPUEscalationLadder {
		if normalizeModalGPUType(candidate) == current {
			start = index
			break
		}
	}
	for _, candidate := range modalGPUEscalationLadder[start+1:] {
		memoryMB := modalDefaultMemoryMB(candidate, detectionJob)
		signature := modalResourceSignature(candidate, batchSize, memoryMB)
		if modalOOMHistoryContains(history, signature) {
			continue
		}
		return candidate, memoryMB
	}
	return "", 0
}

func modalResourceSignature(gpuType string, batchSize int, memoryMB int) string {
	return fmt.Sprintf(
		"gpu=%s|batch=%d|memory_mb=%d",
		normalizeModalGPUType(gpuType),
		maxInt(batchSize, 0),
		maxInt(memoryMB, 0),
	)
}

func normalizeModalGPUType(value string) string {
	normalized := strings.ToUpper(strings.ReplaceAll(strings.TrimSpace(value), "_", "-"))
	switch normalized {
	case "A100-40G":
		return "A100-40GB"
	case "A100-80G":
		return "A100-80GB"
	case "":
		return "T4"
	default:
		return normalized
	}
}

func modalDefaultMemoryMB(gpuType string, detectionJob bool) int {
	defaults := map[string]int{
		"T4":        16384,
		"L4":        24576,
		"A10":       32768,
		"L40S":      65536,
		"A100":      65536,
		"A100-40GB": 65536,
		"A100-80GB": 98304,
	}
	memoryMB := defaults[normalizeModalGPUType(gpuType)]
	if memoryMB == 0 {
		memoryMB = 24576
	}
	if detectionJob && memoryMB < 24576 {
		return 24576
	}
	return memoryMB
}

func modalJobIsDetection(config map[string]any) bool {
	model := strings.ToLower(strings.TrimSpace(jobConfigString(config, "model")))
	return strings.ToLower(strings.TrimSpace(jobConfigString(config, "task_type"))) == "object_detection" ||
		strings.ToLower(strings.TrimSpace(jobConfigString(config, "model_kind"))) == "ultralytics_yolo_detector" ||
		strings.HasPrefix(model, "yolo")
}

func jobConfigInt(config map[string]any, key string) int {
	switch value := config[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case string:
		parsed, _ := strconv.Atoi(strings.TrimSpace(value))
		return parsed
	default:
		return 0
	}
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func mergePayloadMap(base map[string]any, overlay map[string]any) map[string]any {
	out := copyPayloadMap(base)
	if out == nil {
		out = map[string]any{}
	}
	for key, value := range overlay {
		out[key] = value
	}
	return out
}

func (s *Server) recordRetryableJobFailureEvent(job jobs.ExperimentJob, requeued bool, message string, retryDecision retryFailureDecision) {
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
		"retry_guard": map[string]any{
			"status":                   retryDecision.Status,
			"reason":                   retryDecision.Reason,
			"failure_class":            retryDecision.FailureClass,
			"oom_kind":                 retryDecision.OOMKind,
			"resource_signature":       retryDecision.ResourceSignature,
			"previous_gpu_type":        retryDecision.PreviousGPUType,
			"next_gpu_type":            retryDecision.NextGPUType,
			"effective_batch_size":     retryDecision.EffectiveBatchSize,
			"memory_mb":                retryDecision.MemoryMB,
			"repeated_signature":       retryDecision.RepeatedSignature,
			"escalation_exhausted":     retryDecision.EscalationExhausted,
			"same_combo_retry_blocked": retryDecision.Status == "oom_retry_blocked_same_resource",
		},
	}); err != nil {
		log.Printf("record retryable job failure event failed: %v", err)
	}
}

func (s *Server) updateWorkerRequirementDemandAfterTerminalJob(job jobs.ExperimentJob) {
	if job.Template != jobs.TemplateTrainExperiment {
		return
	}
	planID := strings.TrimSpace(jobConfigString(job.Config, "plan_id"))
	if planID == "" {
		return
	}
	requirements, err := s.store.ListProjectWorkerRequirements(job.ProjectID)
	if err != nil {
		log.Printf("worker requirement terminal refresh failed for job %s: %v", job.ID, err)
		return
	}
	projectJobs, err := s.store.ListProjectJobs(job.ProjectID)
	if err != nil {
		log.Printf("worker requirement terminal job list failed for job %s: %v", job.ID, err)
		return
	}
	for _, requirement := range requirements {
		if requirement.PlanID != planID || !workerRequirementHasDemandStatus(requirement.Status) {
			continue
		}
		if !workerRequirementMatchesJobProvider(requirement, job) {
			continue
		}
		if openTrainingJobCountForRequirement(projectJobs, requirement) > 0 {
			continue
		}
		status := execution.WorkerRequirementSatisfied
		updated, updateErr := s.store.UpdateWorkerRequirement(
			requirement.ID,
			execution.WorkerRequirementUpdate{Status: &status},
		)
		if updateErr != nil {
			log.Printf("worker requirement satisfy failed for job %s requirement %s: %v", job.ID, requirement.ID, updateErr)
			continue
		}
		s.recordWorkerRequirementStatusEvent(updated)
	}
}

func workerRequirementHasDemandStatus(status string) bool {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case execution.WorkerRequirementPending, execution.WorkerRequirementStarting, execution.WorkerRequirementActive:
		return true
	default:
		return false
	}
}

func openTrainingJobCountForRequirement(experimentJobs []jobs.ExperimentJob, requirement execution.WorkerRequirement) int {
	count := 0
	for _, job := range experimentJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		if strings.TrimSpace(jobConfigString(job.Config, "plan_id")) != requirement.PlanID {
			continue
		}
		if !workerRequirementMatchesJobProvider(requirement, job) {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(job.Status)) {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
			count++
		}
	}
	return count
}

func workerRequirementMatchesJobProvider(requirement execution.WorkerRequirement, job jobs.ExperimentJob) bool {
	requirementProvider := strings.ToLower(strings.TrimSpace(requirement.Provider))
	requirementGPU := strings.ToLower(strings.TrimSpace(requirement.GPUType))
	jobProvider := strings.ToLower(strings.TrimSpace(jobConfigString(job.Config, "provider")))
	jobGPU := strings.ToLower(strings.TrimSpace(jobConfigString(job.Config, "gpu_type")))
	if jobProvider != "" && requirementProvider != "" && jobProvider != requirementProvider {
		return false
	}
	if jobProvider == "modal" && requirementProvider == "modal" {
		return true
	}
	if jobGPU != "" && requirementGPU != "" && jobGPU != requirementGPU {
		return false
	}
	if jobProvider != "" && requirementGPU != "" && jobProvider == requirementGPU {
		return true
	}
	if jobGPU != "" && requirementProvider != "" && jobGPU == requirementProvider {
		return true
	}
	return true
}

func (s *Server) augmentPolledJob(job *jobs.ExperimentJob, pollProvider string) *jobs.ExperimentJob {
	if job == nil {
		return nil
	}
	out := *job
	out.Config = copyPayloadMap(job.Config)
	attemptID := jobConfigString(out.Config, "active_attempt_id")
	if attemptID != "" {
		out.Config["callback_token"] = s.callbackToken(out.ID, attemptID)
		out.Config["callback_token_header"] = callbackTokenHeader
	}
	if pollJobUsesModal(out, pollProvider) && attemptID != "" {
		session, persist := s.remoteTrainingSessionForPolledJob(out, attemptID)
		out.Config["remote_training_session"] = session
		if persist {
			if _, err := s.store.UpdateJobConfig(out.ID, map[string]any{"remote_training_session": session}); err != nil {
				log.Printf("persist remote training session failed for job %s: %v", out.ID, err)
			}
		}
	}
	return &out
}

func pollJobUsesModal(job jobs.ExperimentJob, pollProvider string) bool {
	provider := strings.ToLower(strings.TrimSpace(firstNonEmptyString(jobConfigString(job.Config, "provider"), pollProvider)))
	return provider == "modal"
}

func (s *Server) remoteTrainingSessionForPolledJob(job jobs.ExperimentJob, attemptID string) (map[string]any, bool) {
	existing := payloadMap(job.Config, "remote_training_session")
	if payloadString(existing, "training_attempt_id") == attemptID && payloadString(existing, "id") != "" {
		session := copyPayloadMap(existing)
		persist := false
		if payloadString(session, "status") == "" {
			session["status"] = "active"
			persist = true
		}
		if _, ok := session["redacted_state"]; !ok {
			session["redacted_state"] = remoteTrainingSessionRedactedState(job)
			persist = true
		}
		if payloadString(session, "expires_at") == "" || remoteTrainingSessionExpired(map[string]any{"remote_training_session": session}, attemptID) {
			session["expires_at"] = remoteTrainingSessionExpiresAt()
			session["storage_scope"] = remoteTrainingSessionStorageScope(job)
			persist = true
		}
		if payloadString(session, "storage_prefix") == "" {
			session["storage_prefix"] = remoteTrainingSessionStoragePrefix(job)
			persist = true
		}
		if _, ok := session["storage_scope"]; !ok {
			session["storage_scope"] = remoteTrainingSessionStorageScope(job)
			persist = true
		}
		s.upsertRemoteTrainingSessionRecord(job, attemptID, session)
		return session, persist
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)
	session := map[string]any{
		"id":                  remoteTrainingSessionID(job.ID, attemptID),
		"training_attempt_id": attemptID,
		"status":              "active",
		"created_at":          now,
		"updated_at":          now,
		"expires_at":          remoteTrainingSessionExpiresAt(),
		"storage_prefix":      remoteTrainingSessionStoragePrefix(job),
		"storage_scope":       remoteTrainingSessionStorageScope(job),
		"redacted_state":      remoteTrainingSessionRedactedState(job),
	}
	if publicCallbackURL := remoteTrainingSessionPublicCallbackURL(job); publicCallbackURL != "" {
		session["public_callback_url"] = publicCallbackURL
	}
	if publicStorageURL := remoteTrainingSessionPublicStorageURL(job); publicStorageURL != "" {
		session["public_storage_url"] = publicStorageURL
	}
	s.upsertRemoteTrainingSessionRecord(job, attemptID, session)
	return session, true
}

func (s *Server) upsertRemoteTrainingSessionRecord(job jobs.ExperimentJob, attemptID string, session map[string]any) {
	expiresAt, err := time.Parse(time.RFC3339Nano, payloadString(session, "expires_at"))
	if err != nil {
		expiresAt = time.Now().UTC().Add(remoteTrainingSessionTTL())
	}
	record := runs.RemoteTrainingSession{
		ID:                    payloadString(session, "id"),
		ProjectID:             job.ProjectID,
		JobID:                 job.ID,
		TrainingAttemptID:     strings.TrimSpace(attemptID),
		Status:                payloadString(session, "status"),
		CallbackTokenHash:     remoteTrainingSessionTokenHash(s.callbackToken(job.ID, attemptID)),
		OrchestratorPublicURL: payloadString(session, "public_callback_url"),
		StoragePublicURL:      payloadString(session, "public_storage_url"),
		StoragePrefix:         payloadString(session, "storage_prefix"),
		StorageScope:          payloadMap(session, "storage_scope"),
		Metadata:              payloadMap(session, "redacted_state"),
		ExpiresAt:             expiresAt,
	}
	if _, err := s.store.UpsertRemoteTrainingSession(record); err != nil {
		log.Printf("upsert remote training session failed for job %s: %v", job.ID, err)
	}
}

func remoteTrainingSessionTokenHash(token string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(token)))
	return "sha256:" + base64.RawURLEncoding.EncodeToString(sum[:])
}

func remoteTrainingSessionID(jobID string, attemptID string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(jobID) + "\x00" + strings.TrimSpace(attemptID)))
	return "rts_" + base64.RawURLEncoding.EncodeToString(sum[:12])
}

func remoteTrainingSessionTTL() time.Duration {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_REMOTE_TRAINING_SESSION_TTL_SECONDS"))
	if value == "" {
		return 6 * time.Hour
	}
	seconds, err := strconv.Atoi(value)
	if err != nil || seconds <= 0 {
		return 6 * time.Hour
	}
	duration := time.Duration(seconds) * time.Second
	if duration < 5*time.Minute {
		return 5 * time.Minute
	}
	if duration > 24*time.Hour {
		return 24 * time.Hour
	}
	return duration
}

func remoteTrainingSessionExpiresAt() string {
	return time.Now().UTC().Add(remoteTrainingSessionTTL()).Format(time.RFC3339Nano)
}

func remoteTrainingSessionExpired(config map[string]any, attemptID string) bool {
	session := payloadMap(config, "remote_training_session")
	if len(session) == 0 || payloadString(session, "training_attempt_id") != strings.TrimSpace(attemptID) {
		return false
	}
	expiresAt := payloadString(session, "expires_at")
	if expiresAt == "" {
		return false
	}
	parsed, err := time.Parse(time.RFC3339Nano, expiresAt)
	if err != nil {
		return true
	}
	return !time.Now().UTC().Before(parsed)
}

func remoteTrainingSessionStoragePrefix(job jobs.ExperimentJob) string {
	return "model-express/artifacts/" + safeRemoteStoragePart(job.ID)
}

func remoteTrainingSessionStorageScope(job jobs.ExperimentJob) map[string]any {
	prefix := remoteTrainingSessionStoragePrefix(job)
	return map[string]any{
		"artifact_write_prefixes": []string{prefix + "/"},
		"artifact_read_prefixes":  []string{prefix + "/"},
		"expires_at":              remoteTrainingSessionExpiresAt(),
	}
}

func safeRemoteStoragePart(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "unknown"
	}
	var builder strings.Builder
	lastDash := false
	for _, r := range value {
		allowed := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' || r == '.'
		if allowed {
			builder.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			builder.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(builder.String(), "-.")
	if out == "" {
		return "unknown"
	}
	return out
}

func remoteTrainingSessionPublicCallbackURL(job jobs.ExperimentJob) string {
	return firstNonEmptyString(
		jobConfigString(job.Config, "public_callback_url"),
		jobConfigString(job.Config, "orchestrator_public_url"),
		jobConfigString(job.Config, "modal_orchestrator_url"),
		os.Getenv("MODAL_ORCHESTRATOR_URL"),
	)
}

func remoteTrainingSessionPublicStorageURL(job jobs.ExperimentJob) string {
	return firstNonEmptyString(
		jobConfigString(job.Config, "public_storage_url"),
		jobConfigString(job.Config, "storage_public_url"),
		jobConfigString(job.Config, "modal_s3_endpoint_url"),
		os.Getenv("MODAL_S3_ENDPOINT_URL"),
	)
}

func remoteTrainingSessionRedactedState(job jobs.ExperimentJob) map[string]any {
	callbackURL := remoteTrainingSessionPublicCallbackURL(job)
	storageURL := remoteTrainingSessionPublicStorageURL(job)
	return map[string]any{
		"provider":                       "modal",
		"public_callback_url_configured": callbackURL != "",
		"public_callback_host_kind":      publicURLHostKind(callbackURL),
		"public_storage_url_configured":  storageURL != "",
		"public_storage_host_kind":       publicURLHostKind(storageURL),
		"modal_call_recorded":            jobConfigString(job.Config, "modal_function_call_object_id") != "",
		"modal_cancel_status":            jobConfigString(job.Config, "modal_call_cancel_status"),
	}
}

func publicURLHostKind(rawURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Hostname() == "" {
		return ""
	}
	host := strings.ToLower(parsed.Hostname())
	switch {
	case host == "localhost" || host == "::1" || strings.HasPrefix(host, "127."):
		return "loopback"
	case strings.Contains(host, "trycloudflare.com"):
		return "cloudflare_tunnel"
	default:
		return "public"
	}
}

func (s *Server) closeRemoteTrainingSession(job jobs.ExperimentJob, status string) {
	existing := payloadMap(job.Config, "remote_training_session")
	if len(existing) == 0 {
		return
	}
	normalizedStatus := store.NormalizeRemoteTrainingSessionStatus(status)
	next := copyPayloadMap(existing)
	now := time.Now().UTC().Format(time.RFC3339Nano)
	currentStatus := store.NormalizeRemoteTrainingSessionStatus(payloadString(next, "status"))
	if currentStatus == runs.RemoteTrainingSessionStatusClosed ||
		currentStatus == runs.RemoteTrainingSessionStatusExpired ||
		currentStatus == runs.RemoteTrainingSessionStatusFailed {
		if sessionID := payloadString(next, "id"); sessionID != "" {
			if _, err := s.store.CloseRemoteTrainingSession(sessionID, normalizedStatus); err != nil && !errors.Is(err, store.ErrNotFound) {
				log.Printf("close remote training session record failed for job %s: %v", job.ID, err)
			}
		}
		return
	}
	next["status"] = normalizedStatus
	next["updated_at"] = now
	if normalizedStatus == runs.RemoteTrainingSessionStatusClosed ||
		normalizedStatus == runs.RemoteTrainingSessionStatusExpired ||
		normalizedStatus == runs.RemoteTrainingSessionStatusFailed {
		next["closed_at"] = now
	}
	if _, err := s.store.UpdateJobConfig(job.ID, map[string]any{"remote_training_session": next}); err != nil {
		log.Printf("close remote training session failed for job %s: %v", job.ID, err)
	}
	if sessionID := payloadString(next, "id"); sessionID != "" {
		if _, err := s.store.CloseRemoteTrainingSession(sessionID, normalizedStatus); err != nil && !errors.Is(err, store.ErrNotFound) {
			log.Printf("close remote training session record failed for job %s: %v", job.ID, err)
		}
	}
}

func defaultProviderPollFallbackTemplates() []string {
	return []string{
		jobs.TemplateProfileDataset,
		jobs.TemplateAnalyzeDatasetVisuals,
		jobs.TemplateExportChampion,
		jobs.TemplateChampionDemoPrediction,
		jobs.TemplateGenerateVisualExemplars,
	}
}

func (s *Server) cancelPlanActiveExecutionByID(planID string, req cancelExecutionRequest) (cancelExecutionResponse, error) {
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return cancelExecutionResponse{}, err
	}
	reason := cancellationReason(req.Reason)
	response := cancelExecutionResponse{
		ExecutionID: "plan:" + plan.ID,
		Target: map[string]string{
			"project_id": plan.ProjectID,
			"plan_id":    plan.ID,
		},
		Status:        "CANCELLING_BY_USER",
		Message:       fmt.Sprintf("Cancellation requested for plan %s.", plan.ID),
		ModalCalls:    []cancelModalCallResult{},
		Compatibility: cancellationCompatibility(),
	}
	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventExecutionCancellationRequested, response.Message, map[string]any{
		"reason":                 reason,
		"promote_best_available": req.PromoteBestAvailable,
		"terminate_remote_work":  req.TerminateRemoteWork,
	}); err != nil {
		return cancelExecutionResponse{}, err
	}

	projectJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return cancelExecutionResponse{}, err
	}
	cancelMessage := fmt.Sprintf("cancelled by user: %s", reason)
	for _, job := range projectJobs {
		if strings.TrimSpace(jobConfigString(job.Config, "plan_id")) != plan.ID {
			continue
		}
		if jobStatusIsTerminal(job.Status) {
			response.AlreadyTerminalJobs++
			continue
		}
		modalCall := modalCallCancelResultForJob(job, req.TerminateRemoteWork)
		response.ModalCalls = append(response.ModalCalls, modalCall)
		if req.TerminateRemoteWork && modalCall.ModalFunctionCallObjectID != "" {
			if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventRemoteWorkCancelRequested, fmt.Sprintf("Requested Modal cancellation for job %s.", job.ID), map[string]any{
				"job_id":                        job.ID,
				"training_attempt_id":           modalCall.TrainingAttemptID,
				"modal_function_call_object_id": modalCall.ModalFunctionCallObjectID,
				"modal_function_call_id":        modalCall.ModalFunctionCallID,
				"modal_input_id":                modalCall.ModalInputID,
				"terminate_remote_work":         true,
			}); err != nil {
				return cancelExecutionResponse{}, err
			}
		}
		if strings.ToUpper(strings.TrimSpace(job.Status)) == jobs.StatusQueued {
			response.QueuedJobsCancelled++
		} else {
			response.ActiveJobsMarkedCancelling++
		}
		s.closeRemoteTrainingSession(job, runs.RemoteTrainingSessionStatusClosing)
		cancelledJob, err := s.store.FailJob(job.ID, cancelMessage)
		if err != nil {
			return cancelExecutionResponse{}, err
		}
		cancelledJob, err = s.store.UpdateJobConfig(cancelledJob.ID, cancellationJobConfigPatch(reason, modalCall, req.TerminateRemoteWork))
		if err != nil {
			return cancelExecutionResponse{}, err
		}
		if cancelledJob.Template == jobs.TemplateTrainExperiment {
			if _, err := s.store.UpsertTrainingRunSummary(cancelledJob.ID, runs.TrainingRunSummaryUpdate{
				Status: jobs.StatusFailed,
			}); err != nil {
				return cancelExecutionResponse{}, err
			}
		}
		response.Jobs = append(response.Jobs, cancelledJob)
	}

	updatedRequirements, err := s.cancelPlanWorkerRequirements(plan, reason)
	if err != nil {
		return cancelExecutionResponse{}, err
	}
	response.WorkerRequirements = updatedRequirements
	if req.PromoteBestAvailable {
		best, err := s.selectBestAvailableChampionForUserCancelledPlan(plan, reason)
		if err != nil {
			return cancelExecutionResponse{}, err
		}
		response.BestAvailableModel = best
	} else {
		response.BestAvailableModel = cancelBestAvailableModel{
			Exportable: false,
			Reason:     "best_available_promotion_disabled",
		}
	}
	response.Status = "CANCELLED_BY_USER"
	response.Message = fmt.Sprintf("Cancelled plan %s: %d queued job(s), %d active job(s), %d worker requirement(s).", plan.ID, response.QueuedJobsCancelled, response.ActiveJobsMarkedCancelling, len(response.WorkerRequirements))
	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventExecutionCancelled, response.Message, map[string]any{
		"reason":                               reason,
		"queued_jobs_cancelled":                response.QueuedJobsCancelled,
		"active_jobs_marked_cancelling":        response.ActiveJobsMarkedCancelling,
		"already_terminal_jobs":                response.AlreadyTerminalJobs,
		"worker_requirement_ids":               workerRequirementIDs(response.WorkerRequirements),
		"modal_calls":                          response.ModalCalls,
		"best_available_model":                 response.BestAvailableModel,
		"job_cancelled_status_enabled":         false,
		"late_callbacks_ignored_by_attempt_id": true,
	}); err != nil {
		return cancelExecutionResponse{}, err
	}
	return response, nil
}

func (s *Server) activePlanIDsForProject(projectID string) ([]string, error) {
	seen := map[string]bool{}
	out := []string{}
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return nil, err
	}
	for _, job := range projectJobs {
		if jobStatusIsTerminal(job.Status) {
			continue
		}
		planID := strings.TrimSpace(jobConfigString(job.Config, "plan_id"))
		if planID == "" || seen[planID] {
			continue
		}
		seen[planID] = true
		out = append(out, planID)
	}
	requirements, err := s.store.ListProjectWorkerRequirements(projectID)
	if err != nil {
		return nil, err
	}
	for _, requirement := range requirements {
		if !workerRequirementHasDemandStatus(requirement.Status) {
			continue
		}
		planID := strings.TrimSpace(requirement.PlanID)
		if planID == "" || seen[planID] {
			continue
		}
		seen[planID] = true
		out = append(out, planID)
	}
	sort.Strings(out)
	return out, nil
}

func (s *Server) cancelPlanWorkerRequirements(plan plans.ExperimentPlan, reason string) ([]execution.WorkerRequirement, error) {
	requirements, err := s.store.ListProjectWorkerRequirements(plan.ProjectID)
	if err != nil {
		return nil, err
	}
	out := []execution.WorkerRequirement{}
	cancelled := execution.WorkerRequirementCancelled
	message := fmt.Sprintf("cancelled by user: %s", reason)
	for _, requirement := range requirements {
		if requirement.PlanID != plan.ID || !workerRequirementHasDemandStatus(requirement.Status) {
			continue
		}
		updated, err := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{
			Status:    &cancelled,
			LastError: &message,
		})
		if err != nil {
			return nil, err
		}
		s.recordWorkerRequirementStatusEvent(updated)
		out = append(out, updated)
	}
	return out, nil
}

func cancellationCompatibility() map[string]bool {
	return map[string]bool{
		"job_cancelled_status_enabled":         false,
		"late_callbacks_ignored_by_attempt_id": true,
	}
}

func cancellationReason(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return "user_requested"
	}
	return reason
}

func modalCallCancelResultForJob(job jobs.ExperimentJob, terminateRemoteWork bool) cancelModalCallResult {
	status := "not_started"
	if strings.ToUpper(strings.TrimSpace(job.Status)) != jobs.StatusQueued {
		status = "call_id_not_recorded"
	}
	objectID := firstNonEmptyString(
		jobConfigString(job.Config, "modal_function_call_object_id"),
		payloadString(payloadMap(job.Config, "modal_resources"), "modal_function_call_object_id"),
	)
	if objectID != "" {
		if terminateRemoteWork {
			status = "cancel_requested"
		} else {
			status = "recorded_not_cancelled"
		}
	}
	return cancelModalCallResult{
		JobID:                     job.ID,
		TrainingAttemptID:         jobConfigString(job.Config, "active_attempt_id"),
		ModalFunctionCallObjectID: objectID,
		ModalFunctionCallID: firstNonEmptyString(
			jobConfigString(job.Config, "modal_function_call_id"),
			payloadString(payloadMap(job.Config, "modal_resources"), "modal_function_call_id"),
		),
		ModalInputID: firstNonEmptyString(
			jobConfigString(job.Config, "modal_input_id"),
			payloadString(payloadMap(job.Config, "modal_resources"), "modal_input_id"),
		),
		CancelStatus: status,
	}
}

func cancellationJobConfigPatch(reason string, modalCall cancelModalCallResult, terminateRemoteWork bool) map[string]any {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	patch := map[string]any{
		"cancel_requested":      true,
		"cancel_requested_at":   now,
		"cancel_reason":         reason,
		"failure_class":         "cancelled",
		"modal_cancel_status":   modalCall.CancelStatus,
		"terminate_remote_work": terminateRemoteWork,
		"worker_stop_requested": true,
	}
	if modalCall.ModalFunctionCallObjectID != "" {
		patch["modal_function_call_object_id"] = modalCall.ModalFunctionCallObjectID
	}
	if modalCall.ModalFunctionCallID != "" {
		patch["modal_function_call_id"] = modalCall.ModalFunctionCallID
	}
	if modalCall.ModalInputID != "" {
		patch["modal_input_id"] = modalCall.ModalInputID
	}
	if modalCall.TrainingAttemptID != "" {
		patch["cancelled_training_attempt_id"] = modalCall.TrainingAttemptID
	}
	return patch
}

func workerRequirementIDs(requirements []execution.WorkerRequirement) []string {
	out := make([]string, 0, len(requirements))
	for _, requirement := range requirements {
		out = append(out, requirement.ID)
	}
	return out
}

func (s *Server) selectBestAvailableChampionForUserCancelledPlan(plan plans.ExperimentPlan, reason string) (cancelBestAvailableModel, error) {
	if champion, err := s.store.GetProjectChampion(plan.ProjectID); err == nil {
		return cancelBestAvailableModel{
			JobID:                   champion.JobID,
			Exportable:              true,
			ChampionSelectionSource: userCancelChampionDecisionSource,
			Score:                   payloadFloat(champion.Metrics, "selection_score"),
			Reason:                  "existing_champion_available",
		}, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return cancelBestAvailableModel{}, err
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil {
		return cancelBestAvailableModel{}, err
	}
	planSummaries := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == plan.ID {
			planSummaries = append(planSummaries, summary)
		}
	}
	best, ok := bestSuccessfulTrainingSummaryForObjective(plan.TargetMetric, planSummaries, nil, agents.ProjectObjectiveContext{})
	if !ok {
		if _, eventErr := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventExecutionCancelled, "Run stopped; no exportable completed model is available yet.", map[string]any{
			"decision_source": userCancelChampionDecisionSource,
			"reason":          reason,
			"exportable":      false,
		}); eventErr != nil {
			return cancelBestAvailableModel{}, eventErr
		}
		return cancelBestAvailableModel{
			Exportable: false,
			Reason:     "no_successful_completed_model",
		}, nil
	}
	score := holisticRunScore(plan.TargetMetric, best, runs.TrainingRunEvaluation{}, agents.ProjectObjectiveContext{})
	decision, err := s.store.CreateAgentDecision(
		plan.ProjectID,
		plan.ID,
		decisions.TypeSelectChampion,
		fmt.Sprintf("User stopped plan %s, so the backend selected the best completed model available before cancellation: %s.", plan.ID, best.JobID),
		map[string]any{
			"decision_source":               userCancelChampionDecisionSource,
			"target_metric":                 plan.TargetMetric,
			"champion_job_id":               best.JobID,
			"champion_model":                best.Model,
			"champion_score":                roundDiagnosticFloat(score),
			"champion_macro_f1":             roundDiagnosticFloat(best.BestMacroF1),
			"champion_accuracy":             roundDiagnosticFloat(best.BestAccuracy),
			"champion_estimated_cost_usd":   roundDiagnosticFloat(best.EstimatedCostUSD),
			"champion_runtime_seconds":      roundDiagnosticFloat(best.RuntimeSeconds),
			"selected_best_available_model": true,
			"user_cancel_reason":            reason,
		},
	)
	if err != nil {
		return cancelBestAvailableModel{}, err
	}
	if err := s.persistProjectChampionFromDecision(plan.ProjectID, decision); err != nil {
		return cancelBestAvailableModel{}, err
	}
	return cancelBestAvailableModel{
		JobID:                   best.JobID,
		Exportable:              true,
		ChampionSelectionSource: userCancelChampionDecisionSource,
		Model:                   best.Model,
		Score:                   roundDiagnosticFloat(score),
		Reason:                  "selected_best_completed_model",
	}, nil
}

func (s *Server) createJob(c *gin.Context) {
	var req createJobRequest
	if !bindJSON(c, &req) {
		return
	}

	if req.Config == nil {
		req.Config = map[string]any{}
	}
	if err := validateCreateJobRequest(req); err != nil {
		writeStoreError(c, err)
		return
	}

	job, err := s.store.CreateJob(c.Param("id"), req.Template, req.Config)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, job)
}

func validateCreateJobRequest(req createJobRequest) error {
	if !validDirectJobTemplate(req.Template) {
		return fmt.Errorf("%w: unsupported job template %q", store.ErrInvalidRequest, req.Template)
	}
	if err := validateCreateJobConfigValue(req.Config, "config"); err != nil {
		return err
	}
	return nil
}

func validDirectJobTemplate(template string) bool {
	switch strings.ToLower(strings.TrimSpace(template)) {
	case jobs.TemplateProfileDataset,
		jobs.TemplateTrainExperiment,
		jobs.TemplateLabelQualityAudit,
		jobs.TemplateExportChampion,
		jobs.TemplateChampionDemoPrediction,
		jobs.TemplateGenerateVisualExemplars,
		jobs.TemplateAnalyzeDatasetVisuals:
		return true
	default:
		return false
	}
}

func validateCreateJobConfigValue(value any, location string) error {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			field := strings.TrimSpace(key)
			if unsafeCreateJobConfigKey(field) {
				return fmt.Errorf("%w: job config field %s cannot be supplied through the public job endpoint", store.ErrInvalidRequest, location+"."+field)
			}
			if err := validateCreateJobConfigValue(item, location+"."+field); err != nil {
				return err
			}
		}
	case []any:
		for index, item := range typed {
			if err := validateCreateJobConfigValue(item, fmt.Sprintf("%s[%d]", location, index)); err != nil {
				return err
			}
		}
	case string:
		if unsafeCreateJobConfigString(typed) {
			return fmt.Errorf("%w: job config value %s contains an unsafe path or URI", store.ErrInvalidRequest, location)
		}
	}
	return nil
}

func unsafeCreateJobConfigKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	if normalized == "" {
		return false
	}
	return strings.Contains(normalized, "path") ||
		strings.Contains(normalized, "uri") ||
		strings.Contains(normalized, "url") ||
		strings.Contains(normalized, "endpoint") ||
		strings.Contains(normalized, "token") ||
		strings.Contains(normalized, "secret") ||
		strings.Contains(normalized, "password") ||
		strings.Contains(normalized, "credential") ||
		strings.Contains(normalized, "access_key")
}

func unsafeCreateJobConfigString(value string) bool {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return false
	}
	normalized := strings.ToLower(trimmed)
	if strings.Contains(normalized, "file://") ||
		strings.Contains(normalized, "s3://") ||
		strings.Contains(normalized, "http://") ||
		strings.Contains(normalized, "https://") ||
		strings.Contains(normalized, "..") ||
		strings.HasPrefix(trimmed, "/") ||
		strings.HasPrefix(trimmed, `\`) {
		return true
	}
	return len(trimmed) >= 3 &&
		((trimmed[1] == ':' && (trimmed[2] == '\\' || trimmed[2] == '/')) ||
			(trimmed[0] == '\\' && trimmed[1] == '\\'))
}

func (s *Server) listProjectJobs(c *gin.Context) {
	limit := queryInt(c, "limit", defaultProjectJobsLimit, 1, maxProjectJobsLimit)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	items, err := s.store.ListProjectJobsPage(c.Param("id"), store.PageOptions{Limit: limit + 1, Offset: offset})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	jobs, hasMore := pageHasMore(items, limit)
	c.JSON(http.StatusOK, pagedListPayload("jobs", jobs, limit, offset, hasMore))
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
	if _, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"metrics",
		map[string]any{"epoch": req.Epoch},
	); !ok {
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
	limit := queryInt(c, "limit", defaultJobMetricsLimit, 1, maxJobMetricsLimit)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	items, err := s.store.ListJobMetricsPage(c.Param("id"), store.PageOptions{Limit: limit + 1, Offset: offset})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	metrics, hasMore := pageLatestWindowHasMore(items, limit)
	c.JSON(http.StatusOK, pagedListPayload("metrics", metrics, limit, offset, hasMore))
}
