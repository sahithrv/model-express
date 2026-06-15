package api

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/workers"
)

type registerWorkerRequest struct {
	ProjectID string `json:"project_id" binding:"required"`
	Name      string `json:"name" binding:"required"`
	GPUType   string `json:"gpu_type"`
}

type dispatcherEventRequest struct {
	EventType string         `json:"event_type" binding:"required"`
	PlanID    string         `json:"plan_id"`
	Message   string         `json:"message"`
	Payload   map[string]any `json:"payload"`
}

type pollJobRequest struct {
	Provider                            string   `json:"provider"`
	Templates                           []string `json:"templates"`
	IncludeUnspecifiedProviderTemplates []string `json:"include_unspecified_provider_templates"`
}

type pollJobResponse struct {
	Job *jobs.ExperimentJob `json:"job"`
}

func (s *Server) ensureWorkerRequirementForPlanJobs(
	plan plans.ExperimentPlan,
	provider string,
	gpuType string,
	queuedJobs []jobs.ExperimentJob,
	source string,
	requestedMaxConcurrentJobs int,
) (*execution.WorkerRequirement, error) {
	openJobCount := openTrainingJobCount(queuedJobs)
	if openJobCount == 0 {
		return nil, nil
	}
	if provider == "" {
		provider = "local"
	}
	provider = normalizeTrainingProvider(provider)
	targetCount := s.targetWorkerCountForPlanExecution(plan, openJobCount, requestedMaxConcurrentJobs)
	activeWorkerCount := s.activeOrStartingWorkersForProject(plan.ProjectID, provider, gpuType)
	requirementPolicy, err := s.workerRequirementPolicyForPlan(plan, provider, targetCount)
	if err != nil {
		return nil, err
	}
	if requestedMaxConcurrentJobs > 0 {
		requirementPolicy.MaxConcurrentJobs = targetCount
	}
	if requestedMaxConcurrentJobs <= 0 {
		if policy := costPolicyForSettings(s.currentAutomationSettings()); policy.Enabled && policy.MaxConcurrentJobs > 0 && requirementPolicy.MaxConcurrentJobs > policy.MaxConcurrentJobs {
			requirementPolicy.MaxConcurrentJobs = policy.MaxConcurrentJobs
		}
	}
	requirement, created, err := s.store.UpsertWorkerRequirement(
		plan.ProjectID,
		plan.ID,
		provider,
		gpuType,
		targetCount,
		source,
		requirementPolicy,
	)
	if err != nil {
		return nil, err
	}
	if activeWorkerCount >= requirement.TargetCount {
		active := execution.WorkerRequirementActive
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active})
		if updateErr != nil {
			return nil, updateErr
		}
		requirement = updated
	} else if requirement.Status == execution.WorkerRequirementActive {
		pending := execution.WorkerRequirementPending
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &pending})
		if updateErr != nil {
			return nil, updateErr
		}
		requirement = updated
	}
	eventType := execution.EventWorkersRequired
	if !created {
		eventType = execution.EventWorkerScalingUpdated
	}
	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, eventType, fmt.Sprintf("Execution targets %d worker(s) for %d open job(s).", requirement.TargetCount, openJobCount), map[string]any{
		"worker_requirement_id":         requirement.ID,
		"target_count":                  requirement.TargetCount,
		"open_job_count":                openJobCount,
		"active_worker_count":           activeWorkerCount,
		"provider":                      requirement.Provider,
		"gpu_type":                      requirement.GPUType,
		"source":                        source,
		"requested_max_concurrent_jobs": requestedMaxConcurrentJobs,
		"max_concurrent_jobs":           requirement.MaxConcurrentJobs,
	}); err != nil {
		return nil, err
	}
	return &requirement, nil
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

type costModePolicy struct {
	Enabled           bool
	Mode              string
	BudgetCapUSD      float64
	SpentUSD          float64
	MaxConcurrentJobs int
	MaxPreviewTrials  int
	MaxFullTrials     int
	ExportPolicy      string
	BatchMaxTrials    int
	PreviewCount      int
	FullCount         int
	Skipped           []map[string]any
}

func (s *Server) costPolicyForPlan(plan plans.ExperimentPlan) (costModePolicy, error) {
	settings := s.currentAutomationSettings()
	policy := costPolicyForSettings(settings)
	if !policy.Enabled {
		return policy, nil
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return costModePolicy{}, err
	}
	for _, summary := range summaries {
		policy.SpentUSD += summary.EstimatedCostUSD
	}
	return policy, nil
}

func costPolicyForSettings(automationSettings settings.AutomationSettings) costModePolicy {
	mode := normalizeCostMode(automationSettings.CostMode)
	policy := costModePolicy{
		Enabled:      costModesEnabled(),
		Mode:         mode,
		BudgetCapUSD: automationSettings.BudgetCapUSD,
		ExportPolicy: "champion_only",
	}
	switch mode {
	case "prototype":
		policy.MaxConcurrentJobs = 1
		policy.MaxPreviewTrials = 3
		policy.MaxFullTrials = 1
		policy.BatchMaxTrials = 3
	case "quality":
		policy.MaxConcurrentJobs = 4
		policy.MaxPreviewTrials = 8
		policy.MaxFullTrials = 4
		policy.BatchMaxTrials = 8
		policy.ExportPolicy = "top_candidates"
	default:
		policy.Mode = "balanced"
		policy.MaxConcurrentJobs = 2
		policy.MaxPreviewTrials = 6
		policy.MaxFullTrials = 2
		policy.BatchMaxTrials = 5
		policy.ExportPolicy = "champion_plus_final_candidate"
	}
	if policy.BudgetCapUSD < 0 {
		policy.BudgetCapUSD = 0
	}
	return policy
}

func (policy *costModePolicy) AllowTrainingJob(tier string) (bool, string) {
	if !policy.Enabled {
		return true, ""
	}
	if policy.BudgetCapUSD > 0 && policy.SpentUSD >= policy.BudgetCapUSD && tier == "full" {
		return false, "budget_cap_reached"
	}
	if tier == "full" {
		if policy.MaxFullTrials > 0 && policy.FullCount >= policy.MaxFullTrials {
			return false, "cost_mode_full_trial_limit"
		}
		policy.FullCount++
		return true, ""
	}
	if policy.MaxPreviewTrials > 0 && policy.PreviewCount >= policy.MaxPreviewTrials {
		return false, "cost_mode_preview_trial_limit"
	}
	policy.PreviewCount++
	return true, ""
}

func (policy *costModePolicy) Skip(index int, tier string, reason string) {
	if policy.Skipped == nil {
		policy.Skipped = []map[string]any{}
	}
	policy.Skipped = append(policy.Skipped, map[string]any{
		"experiment_index": index,
		"training_tier":    tier,
		"reason":           reason,
		"cost_mode":        policy.Mode,
		"budget_cap_usd":   policy.BudgetCapUSD,
		"spent_usd":        policy.SpentUSD,
	})
}

func (policy costModePolicy) Payload() map[string]any {
	if !policy.Enabled {
		return nil
	}
	return map[string]any{
		"enabled":             true,
		"cost_mode":           policy.Mode,
		"budget_cap_usd":      policy.BudgetCapUSD,
		"spent_usd":           policy.SpentUSD,
		"max_concurrent_jobs": policy.MaxConcurrentJobs,
		"max_preview_trials":  policy.MaxPreviewTrials,
		"max_full_trials":     policy.MaxFullTrials,
		"export_policy":       policy.ExportPolicy,
		"batch_max_trials":    policy.BatchMaxTrials,
		"preview_count":       policy.PreviewCount,
		"full_count":          policy.FullCount,
		"skipped":             policy.Skipped,
	}
}

func (s *Server) recordCostPolicySkippedJobs(plan plans.ExperimentPlan, policy costModePolicy) error {
	if !policy.Enabled || len(policy.Skipped) == 0 {
		return nil
	}
	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventCostBudgetBlocked, "Cost mode or budget cap blocked queued training job(s).", policy.Payload()); err != nil {
		return err
	}
	_, err := s.selectBestAvailableChampionAfterCostPolicyStop(plan, policy, "queue_policy_block")
	return err
}

func (s *Server) selectBestAvailableChampionAfterCostPolicyStop(plan plans.ExperimentPlan, policy costModePolicy, trigger string) (bool, error) {
	if !policy.Enabled || len(policy.Skipped) == 0 {
		return false, nil
	}
	return s.selectBestAvailableChampionForCostStoppedPlan(plan, policy.Payload(), trigger)
}

func (s *Server) selectBestAvailableChampionForCostStoppedPlan(plan plans.ExperimentPlan, costPolicyPayload map[string]any, trigger string) (bool, error) {
	if plan.ID == "" {
		return false, nil
	}
	if _, err := s.store.GetProjectChampion(plan.ProjectID); err == nil {
		return true, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	decisionsForProject, err := s.store.ListProjectAgentDecisions(plan.ProjectID)
	if err != nil {
		return false, err
	}
	for _, decision := range decisionsForProject {
		if decision.PlanID == plan.ID && decision.Payload["decision_source"] == costPolicyChampionDecisionSource {
			if err := s.persistProjectChampionFromDecision(plan.ProjectID, decision); err != nil {
				return false, err
			}
			return true, nil
		}
	}
	openJobs, err := s.hasOpenTrainingJobForPlan(plan.ProjectID, plan.ID)
	if err != nil {
		return false, err
	}
	if openJobs {
		return false, nil
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil {
		return false, err
	}
	best, ok := bestSuccessfulTrainingSummaryForObjective(plan.TargetMetric, summaries, nil, agents.ProjectObjectiveContext{})
	if !ok {
		_, eventErr := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventCostBudgetBlocked, "Cost policy stopped training, but no successful model is available to export yet.", map[string]any{
			"decision_source": costPolicyChampionDecisionSource,
			"trigger":         trigger,
			"cost_policy":     costPolicyPayload,
			"exportable":      false,
		})
		return false, eventErr
	}
	score := holisticRunScore(plan.TargetMetric, best, runs.TrainingRunEvaluation{}, agents.ProjectObjectiveContext{})
	payload := map[string]any{
		"decision_source":               costPolicyChampionDecisionSource,
		"auto_executable":               true,
		"target_metric":                 plan.TargetMetric,
		"champion_job_id":               best.JobID,
		"champion_model":                best.Model,
		"champion_score":                roundDiagnosticFloat(score),
		"champion_macro_f1":             roundDiagnosticFloat(best.BestMacroF1),
		"champion_accuracy":             roundDiagnosticFloat(best.BestAccuracy),
		"champion_estimated_cost_usd":   roundDiagnosticFloat(best.EstimatedCostUSD),
		"champion_runtime_seconds":      roundDiagnosticFloat(best.RuntimeSeconds),
		"budget_stop_trigger":           trigger,
		"cost_policy":                   costPolicyPayload,
		"selected_best_available_model": true,
	}
	rationale := fmt.Sprintf("Cost mode or budget cap stopped additional training for plan %s, so the backend selected the best successful model available within the budget: %s.", plan.ID, best.JobID)
	decision, err := s.store.CreateAgentDecision(plan.ProjectID, plan.ID, decisions.TypeSelectChampion, rationale, payload)
	if err != nil {
		return false, err
	}
	if err := s.persistProjectChampionFromDecision(plan.ProjectID, decision); err != nil {
		return false, err
	}
	return true, nil
}

func (s *Server) hasOpenTrainingJobForPlan(projectID string, planID string) (bool, error) {
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return false, err
	}
	for _, job := range projectJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		if configString(job.Config, "plan_id") != planID {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(job.Status)) {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
			return true, nil
		}
	}
	return false, nil
}

func (s *Server) selectBestAvailableChampionIfCostStoppedAfterTrainingJob(job jobs.ExperimentJob) (bool, error) {
	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil || summary.PlanID == "" {
		return false, nil
	}
	plan, err := s.store.GetExperimentPlan(summary.PlanID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	payload, ok, err := s.costPolicyStopPayloadForPlan(job.ProjectID, plan.ID)
	if err != nil || !ok {
		return false, err
	}
	return s.selectBestAvailableChampionForCostStoppedPlan(plan, payload, "terminal_training_hook")
}

func (s *Server) costPolicyStopPayloadForPlan(projectID string, planID string) (map[string]any, bool, error) {
	events, err := s.store.ListProjectExecutionEvents(projectID, 500)
	if err != nil {
		return nil, false, err
	}
	for _, event := range events {
		if event.PlanID != planID || event.EventType != execution.EventCostBudgetBlocked {
			continue
		}
		if event.Payload["decision_source"] == costPolicyChampionDecisionSource {
			continue
		}
		if skipped, ok := event.Payload["skipped"].([]map[string]any); ok && len(skipped) > 0 {
			return event.Payload, true, nil
		}
		if skipped, ok := event.Payload["skipped"].([]any); ok && len(skipped) > 0 {
			return event.Payload, true, nil
		}
	}
	return nil, false, nil
}

func (s *Server) targetWorkerCountForPlan(plan plans.ExperimentPlan, openJobCount int) int {
	return s.targetWorkerCountForPlanExecution(plan, openJobCount, 0)
}

func (s *Server) targetWorkerCountForPlanExecution(plan plans.ExperimentPlan, openJobCount int, requestedMaxConcurrentJobs int) int {
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
	if requestedMaxConcurrentJobs > 0 {
		if targetCount > requestedMaxConcurrentJobs {
			targetCount = requestedMaxConcurrentJobs
		}
		return targetCount
	}
	if policy := costPolicyForSettings(s.currentAutomationSettings()); policy.Enabled && policy.MaxConcurrentJobs > 0 && targetCount > policy.MaxConcurrentJobs {
		targetCount = policy.MaxConcurrentJobs
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
	if provider == persistentGPUProviderName {
		policy.ColdCachePolicy = execution.ColdCachePolicySingleMaterialization
		policy.MaxColdDatasetMaterializations = 1
		if checksum != "" || policy.DatasetCacheKey != "" {
			policy.DatasetMaterializationStatus = execution.DatasetMaterializationCold
		}
	}
	return policy
}

const persistentGPUProviderName = "persistent_gpu"

func addCostPolicyConfig(config map[string]any, policy costModePolicy, tier string) {
	if !policy.Enabled {
		return
	}
	config["cost_mode"] = policy.Mode
	config["budget_cap_usd"] = policy.BudgetCapUSD
	config["export_policy"] = policy.ExportPolicy
	config["cost_policy"] = map[string]any{
		"enabled":             true,
		"cost_mode":           policy.Mode,
		"training_tier":       tier,
		"budget_cap_usd":      policy.BudgetCapUSD,
		"spent_usd":           policy.SpentUSD,
		"max_concurrent_jobs": policy.MaxConcurrentJobs,
		"max_preview_trials":  policy.MaxPreviewTrials,
		"max_full_trials":     policy.MaxFullTrials,
		"export_policy":       policy.ExportPolicy,
		"batch_max_trials":    policy.BatchMaxTrials,
	}
}

func addPersistentGPUConfig(config map[string]any, provider string, dataset datasets.Dataset, policy execution.WorkerRequirementPolicy) {
	if provider != persistentGPUProviderName {
		return
	}
	cacheRoot := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT"))
	config["persistent_gpu"] = map[string]any{
		"provider":                 persistentGPUProviderName,
		"cache_root":               cacheRoot,
		"dataset_cache_key":        policy.DatasetCacheKey,
		"dataset_checksum_sha256":  policy.DatasetChecksum,
		"storage_uri_fingerprint":  storageURIFingerprint(dataset.StorageURI),
		"materialization_status":   policy.DatasetMaterializationStatus,
		"cost_queue_metadata_kind": "persistent_disk_gpu_v1",
	}
}

func trainingTierForExperiment(experiment plans.PlannedExperiment) string {
	for _, value := range []string{experiment.Strategy, experiment.Mechanism, experiment.Intervention, experiment.Reason} {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if normalized == "" {
			continue
		}
		if strings.Contains(normalized, "champion_validation") || strings.Contains(normalized, "champion validation") {
			return "champion_validation"
		}
		if strings.Contains(normalized, "full") || strings.Contains(normalized, "promoted") || strings.Contains(normalized, "final") {
			return "full"
		}
		if strings.Contains(normalized, "preview") || strings.Contains(normalized, "trial") {
			return "preview"
		}
	}
	return "preview"
}

func normalizeTrainingProvider(provider string) string {
	normalized := strings.ToLower(strings.TrimSpace(provider))
	switch normalized {
	case "", "local":
		return "local"
	case "modal":
		return "modal"
	case "persistent_gpu", "persistent-gpu", "persistent_disk", "persistent-disk":
		return persistentGPUProviderName
	default:
		return normalized
	}
}

func validateTrainingProviderConfigured(provider string) error {
	if provider != persistentGPUProviderName {
		return nil
	}
	if !persistentGPUProviderEnabled() {
		return fmt.Errorf("%w: persistent_gpu provider requires MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER=1", store.ErrInvalidRequest)
	}
	if strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT")) == "" {
		return fmt.Errorf("%w: persistent_gpu provider requires MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", store.ErrInvalidRequest)
	}
	return nil
}

func persistentGPUProviderEnabled() bool {
	return envFlag("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", false)
}

func costModesEnabled() bool {
	return envFlag("MODEL_EXPRESS_COST_MODES", false)
}

func normalizeCostMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "prototype", "cheap", "prototype/cheap":
		return "prototype"
	case "quality":
		return "quality"
	default:
		return "balanced"
	}
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
	case execution.WorkerRequirementSatisfied:
		eventType = execution.EventWorkerScalingUpdated
		message = fmt.Sprintf("Worker requirement satisfied for plan %s; no open training jobs remain.", requirement.PlanID)
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
		execution.WorkerRequirementSatisfied,
		execution.WorkerRequirementFailed,
		execution.WorkerRequirementCancelled:
		return true
	default:
		return false
	}
}

func validDatasetMaterializationStatus(status string) bool {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case execution.DatasetMaterializationUnknown,
		execution.DatasetMaterializationCold,
		execution.DatasetMaterializationMaterializing,
		execution.DatasetMaterializationWarm,
		execution.DatasetMaterializationStagingOnly:
		return true
	default:
		return false
	}
}

func validDispatcherEventType(eventType string) bool {
	switch strings.ToUpper(strings.TrimSpace(eventType)) {
	case execution.EventDispatcherStatus, execution.EventDispatcherIdleExit:
		return true
	default:
		return false
	}
}

func defaultDispatcherEventMessage(eventType string) string {
	switch strings.ToUpper(strings.TrimSpace(eventType)) {
	case execution.EventDispatcherIdleExit:
		return "Modal dispatcher exited after an idle zero-demand window."
	default:
		return "Modal dispatcher lifecycle status updated."
	}
}

func dispatcherEventPayload(payload map[string]any) map[string]any {
	out := map[string]any{"dispatcher": "modal"}
	allowed := []string{
		"worker_id",
		"previous_slot_count",
		"slot_count",
		"desired_slot_count",
		"registered_slot_count",
		"active_slot_count",
		"idle_seconds",
		"idle_exit_seconds",
		"reason",
	}
	for _, key := range allowed {
		if value, ok := smallDispatcherPayloadValue(payload[key]); ok {
			out[key] = value
		}
	}
	return out
}

func smallDispatcherPayloadValue(value any) (any, bool) {
	switch typed := value.(type) {
	case nil:
		return nil, false
	case string:
		text := strings.TrimSpace(typed)
		if text == "" {
			return "", false
		}
		return activitySafeText(text, 120), true
	case bool:
		return typed, true
	case int:
		return typed, true
	case int64:
		return typed, true
	case float64:
		return typed, true
	case float32:
		return typed, true
	default:
		return nil, false
	}
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
	if req.DatasetMaterializationStatus != nil {
		normalizedStatus := strings.ToUpper(strings.TrimSpace(*req.DatasetMaterializationStatus))
		if !validDatasetMaterializationStatus(normalizedStatus) {
			writeStoreError(c, fmt.Errorf("%w: invalid dataset materialization status", store.ErrInvalidRequest))
			return
		}
		req.DatasetMaterializationStatus = &normalizedStatus
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

func (s *Server) reportProjectDispatcherEvent(c *gin.Context) {
	var req dispatcherEventRequest
	if !bindJSON(c, &req) {
		return
	}
	eventType := strings.ToUpper(strings.TrimSpace(req.EventType))
	if !validDispatcherEventType(eventType) {
		writeStoreError(c, fmt.Errorf("%w: invalid dispatcher event type", store.ErrInvalidRequest))
		return
	}
	message := strings.TrimSpace(req.Message)
	if message == "" {
		message = defaultDispatcherEventMessage(eventType)
	}
	event, err := s.store.CreateExecutionEvent(
		c.Param("id"),
		strings.TrimSpace(req.PlanID),
		eventType,
		message,
		dispatcherEventPayload(req.Payload),
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"event": event})
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
	includeUnspecified := req.IncludeUnspecifiedProviderTemplates
	if strings.TrimSpace(req.Provider) != "" && len(includeUnspecified) == 0 {
		includeUnspecified = defaultProviderPollFallbackTemplates()
	}
	job, err := s.store.PollJob(c.Param("id"), store.JobPollFilter{
		Provider:                            req.Provider,
		Templates:                           req.Templates,
		IncludeUnspecifiedProviderTemplates: includeUnspecified,
	})
	if err == nil {
		c.JSON(http.StatusOK, pollJobResponse{Job: s.augmentPolledJob(job, req.Provider)})
		return
	}

	if errors.Is(err, store.ErrNoJob) {
		c.JSON(http.StatusOK, pollJobResponse{Job: nil})
		return
	}

	writeStoreError(c, err)
}
