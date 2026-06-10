package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/store"
)

const (
	activityDefaultLimit      = 12
	activityMaxLimit          = 50
	activityDefaultIntervalMS = 5000
)

type agentActivityEvent struct {
	ID        string         `json:"id"`
	ProjectID string         `json:"project_id"`
	PlanID    string         `json:"plan_id,omitempty"`
	JobID     string         `json:"job_id,omitempty"`
	Type      string         `json:"type"`
	Severity  string         `json:"severity"`
	Title     string         `json:"title"`
	Message   string         `json:"message"`
	Status    string         `json:"status"`
	CreatedAt time.Time      `json:"created_at"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

var (
	activityStorageURIRe  = regexp.MustCompile(`(?i)\b(?:s3|gs|file|minio|http|https)://[^\s,;"')\]}]+`)
	activityWindowsPathRe = regexp.MustCompile(`(?i)\b[A-Z]:\\[^\s,;"')\]}]+`)
	activityUnixPathRe    = regexp.MustCompile(`(^|\s)/(?:Users|home|tmp|var|mnt|data|datasets|artifacts|workspace|app|srv)[^\s,;"')\]}]+`)
	activityBase64Re      = regexp.MustCompile(`\b[A-Za-z0-9+/]{80,}={0,2}\b`)
	activitySecretRe      = regexp.MustCompile(`(?i)\b(?:sk|pk|rk|xox[baprs]?)-[A-Za-z0-9_-]{16,}\b`)
	activityWhitespaceRe  = regexp.MustCompile(`\s+`)
)

func (s *Server) streamProjectActivityEvents(c *gin.Context) {
	projectID := c.Param("id")
	limit := activityLimitFromQuery(c.DefaultQuery("limit", strconv.Itoa(activityDefaultLimit)))
	interval, _ := strconv.Atoi(c.DefaultQuery("interval_ms", strconv.Itoa(activityDefaultIntervalMS)))
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
	c.Writer.WriteString("retry: 2000\n\n")
	c.Writer.Flush()

	lastID := c.GetHeader("Last-Event-ID")
	delivered := map[string]bool{}
	ticker := time.NewTicker(time.Duration(interval) * time.Millisecond)
	defer ticker.Stop()

	send := func() bool {
		events, err := s.listProjectActivityEvents(projectID, limit)
		if err != nil {
			c.SSEvent("stream_error", gin.H{"error": "activity stream unavailable"})
			c.Writer.Flush()
			return false
		}
		sort.SliceStable(events, func(i, j int) bool {
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
			c.SSEvent("activity_event", event)
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

func (s *Server) listProjectActivityEvents(projectID string, limit int) ([]agentActivityEvent, error) {
	limit = activityClampLimit(limit)
	sourceLimit := limit * 4
	if sourceLimit < 50 {
		sourceLimit = 50
	}

	executionEvents, err := s.store.ListProjectExecutionEvents(projectID, sourceLimit)
	if err != nil {
		return nil, err
	}
	projectJobs, err := s.store.ListProjectJobsPage(projectID, store.PageOptions{Limit: sourceLimit, Offset: 0})
	if err != nil {
		return nil, err
	}
	agentInvocations, err := s.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{Limit: sourceLimit})
	if err != nil {
		return nil, err
	}
	agentDecisions, err := s.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		return nil, err
	}

	out := []agentActivityEvent{}
	for _, event := range executionEvents {
		out = append(out, activityFromExecutionEvent(event))
	}
	activeWork := activityHasActiveWork(projectJobs, agentInvocations)
	for _, invocation := range agentInvocations {
		if event, ok := activityFromAgentInvocation(invocation, activeWork); ok {
			out = append(out, event)
		}
	}
	for index, decision := range agentDecisions {
		if index >= sourceLimit {
			break
		}
		out = append(out, activityFromAgentDecision(decision))
	}
	out = append(out, activityFromJobs(projectID, projectJobs)...)

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID > out[j].ID
		}
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	out = dedupeActivityEvents(out)
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func activityFromExecutionEvent(event execution.ExecutionEvent) agentActivityEvent {
	activity := agentActivityEvent{
		ID:        activityID("execution", event.ID),
		ProjectID: event.ProjectID,
		PlanID:    event.PlanID,
		Type:      "system.event",
		Severity:  "info",
		Title:     "System event",
		Message:   activitySafeText(event.Message, 220),
		Status:    "active",
		CreatedAt: event.CreatedAt,
		Metadata:  activityMetadataFromPayload(event.Payload),
	}
	activity.JobID = activityMetadataString(event.Payload, "job_id")
	if activity.JobID == "" {
		activity.JobID = activityMetadataString(event.Payload, "champion_job_id")
	}

	switch event.EventType {
	case execution.EventAgentStarted:
		activity.Type = "planner.started"
		activity.Severity = "info"
		activity.Title = "Planner started"
		activity.Message = "Reading completed runs, memories, evaluations, and project context."
		activity.Status = "active"
	case execution.EventAgentRecommendationRecorded:
		activity.Type = "planner.decision_recorded"
		activity.Severity = "success"
		activity.Title = "Valid decision recorded"
		activity.Status = "succeeded"
	case execution.EventAgentOutcomeRecorded:
		activity.Type = "agent.outcome_recorded"
		activity.Severity = "success"
		activity.Title = "Agent outcome recorded"
		activity.Status = "succeeded"
		validationStatus := strings.ToLower(activityMetadataString(event.Payload, "backend_validation_status"))
		if validationStatus == "blocked" {
			activity.Type = "planner.blocked"
			activity.Severity = "warning"
			activity.Title = "Planner blocked"
			activity.Status = "blocked"
			activity.Message = activitySafeText(firstNonEmpty(
				activityMetadataString(event.Payload, "backend_validation_error"),
				activityMetadataString(event.Payload, "reason"),
				event.Message,
			), 220)
		} else if validationStatus == "failed" {
			activity.Type = "planner.validation_failed"
			activity.Severity = "error"
			activity.Title = "Planner validation failed"
			activity.Status = "failed"
		}
	case execution.EventAgentFailed:
		activity.Type = "agent.failed"
		activity.Severity = "error"
		activity.Title = "Agent failed"
		activity.Status = "failed"
	case execution.EventJobsQueued:
		openCount := activityMetadataInt(event.Payload, "open_job_count")
		jobCount := len(activityMetadataStringSlice(event.Payload, "job_ids"))
		if jobCount == 0 {
			jobCount = openCount
		}
		activity.Type = "plan.queued"
		activity.Severity = "success"
		activity.Title = "Plan queued"
		activity.Status = "active"
		if openCount == 0 {
			activity.Severity = "info"
			activity.Title = "Plan checked"
			activity.Status = "waiting"
		}
		activity.Metadata["job_count"] = jobCount
	case execution.EventWorkersRequired, execution.EventWorkerScalingUpdated:
		activity.Type = "workers.required"
		activity.Severity = "info"
		activity.Title = "Waiting for workers"
		activity.Status = "waiting"
	case execution.EventWorkersStarting:
		activity.Type = "workers.starting"
		activity.Severity = "info"
		activity.Title = "Workers starting"
		activity.Status = "active"
	case execution.EventWorkersActive:
		activity.Type = "workers.active"
		activity.Severity = "success"
		activity.Title = "Workers active"
		activity.Status = "active"
	case execution.EventDispatcherStatus:
		activity.Type = "dispatcher.status"
		activity.Severity = "info"
		activity.Title = "Modal dispatcher status"
		activity.Status = "active"
		if activityMetadataInt(event.Payload, "slot_count") == 0 {
			activity.Status = "waiting"
		}
	case execution.EventDispatcherIdleExit:
		activity.Type = "dispatcher.idle_exit"
		activity.Severity = "success"
		activity.Title = "Modal dispatcher idle"
		activity.Status = "succeeded"
	case execution.EventJobRetryQueued:
		requeued := activityMetadataBool(event.Payload, "requeued")
		activity.Type = "job.retrying"
		activity.Severity = "warning"
		activity.Title = "Retrying job"
		activity.Status = "active"
		if !requeued {
			activity.Type = "job.failed"
			activity.Severity = "error"
			activity.Title = "Job attempts exhausted"
			activity.Status = "failed"
		}
	case execution.EventExecutionFailed:
		activity.Type = "system.failed"
		activity.Severity = "error"
		activity.Title = "Execution failed"
		activity.Status = "failed"
	case execution.EventMemoryRetrievalLogged:
		retrievedCount := activityMetadataInt(event.Payload, "retrieved_count")
		purpose := activitySafeIdentifier(activityMetadataString(event.Payload, "purpose"))
		activity.Type = "memory.retrieval_logged"
		activity.Severity = "info"
		activity.Title = "Memory retrieval checked"
		activity.Message = fmt.Sprintf("Retrieved %d candidate memory card(s)%s.", retrievedCount, activityPurposeSuffix(purpose))
		activity.Status = "succeeded"
	case execution.EventChampionSelected:
		activity.Type = "champion.selected"
		activity.Severity = "success"
		activity.Title = "Selected champion"
		activity.Status = "succeeded"
	case execution.EventChampionFeedbackRecorded:
		activity.Type = "champion.feedback_recorded"
		activity.Severity = "info"
		activity.Title = "Champion feedback recorded"
		activity.Status = "succeeded"
	case execution.EventDatasetVisualAnalysisQueued:
		activity.Type = "dataset.visual_analysis_queued"
		activity.Severity = "info"
		activity.Title = "Visual analysis queued"
		activity.Status = "active"
	case execution.EventDatasetVisualAnalysisResult:
		activity.Type = "dataset.visual_analysis_recorded"
		activity.Severity = "success"
		activity.Title = "Visual analysis recorded"
		activity.Status = "succeeded"
	case execution.EventExperimentationReopened:
		activity.Type = "project.reopened"
		activity.Severity = "success"
		activity.Title = "Experimentation reopened"
		activity.Status = "active"
	}
	return activity
}

func activityFromAgentInvocation(invocation memory.AgentInvocation, activeWork bool) (agentActivityEvent, bool) {
	validationStatus := strings.ToLower(strings.TrimSpace(invocation.ValidationStatus))
	outcomeStatus := strings.ToLower(activityMetadataString(invocation.DownstreamOutcome, "backend_validation_status"))
	if validationStatus != memory.InvocationValidationInvalid && validationStatus != memory.InvocationValidationFailed && outcomeStatus != "rejected" {
		return agentActivityEvent{}, false
	}

	willRetry := activityMetadataBool(invocation.DownstreamOutcome, "will_retry")
	reason := firstNonEmpty(
		activityMetadataString(invocation.DownstreamOutcome, "backend_validation_error"),
		invocation.ValidationError,
		"backend validation rejected the draft",
	)
	agentName := activitySafeIdentifier(invocation.AgentName)
	eventType := "agent.validation_rejected"
	title := "Agent draft rejected"
	if invocation.AgentName == agents.ExperimentPlannerAgentName {
		eventType = "planner.validation_rejected"
		title = "Planner draft rejected"
	}

	status := "blocked"
	severity := "warning"
	message := "Draft rejected: " + activitySafeText(reason, 180) + "."
	if willRetry {
		status = "active"
		title += "; retrying"
		message = "Draft rejected: " + activitySafeText(reason, 180) + "; retrying with validation feedback."
	} else if activeWork {
		status = "waiting"
		message = "Draft rejected: " + activitySafeText(reason, 180) + "; waiting while other project work continues."
	} else if validationStatus == memory.InvocationValidationFailed {
		status = "failed"
		severity = "error"
		title = "Agent validation failed"
		if invocation.AgentName == agents.ExperimentPlannerAgentName {
			title = "Planner validation failed"
		}
	}

	metadata := map[string]any{
		"agent_name":        agentName,
		"validation_status": firstNonEmpty(outcomeStatus, validationStatus),
		"will_retry":        willRetry,
		"validation_error":  activitySafeText(reason, 180),
	}
	if retryAttempt, ok := activityOptionalInt(invocation.DownstreamOutcome, "retry_attempt"); ok {
		metadata["retry_attempt"] = retryAttempt
	}

	idVariant := fmt.Sprintf("%s:%t:%s:%s", validationStatus, willRetry, status, reason)
	return agentActivityEvent{
		ID:        activityID("invocation", invocation.ID, activityShortHash(idVariant)),
		ProjectID: invocation.ProjectID,
		PlanID:    invocation.PlanID,
		JobID:     invocation.JobID,
		Type:      eventType,
		Severity:  severity,
		Title:     title,
		Message:   activitySafeText(message, 240),
		Status:    status,
		CreatedAt: invocation.CreatedAt,
		Metadata:  metadata,
	}, true
}

func activityFromAgentDecision(decision decisions.AgentDecision) agentActivityEvent {
	decisionType := strings.ToUpper(strings.TrimSpace(decision.DecisionType))
	metadata := map[string]any{
		"decision_type": decisionType,
	}
	if targetMetric := activityMetadataString(decision.Payload, "target_metric"); targetMetric != "" {
		metadata["target_metric"] = targetMetric
	}
	if status := activityMetadataString(decision.Payload, "backend_validation_status"); status != "" {
		metadata["backend_validation_status"] = status
	}
	if retryCount, ok := activityOptionalInt(decision.Payload, "validation_retry_count"); ok {
		metadata["validation_retry_count"] = retryCount
	}
	if models := activityDecisionModels(decision.Payload); len(models) > 0 {
		metadata["models"] = models
		metadata["experiment_count"] = len(models)
	}

	title := "Agent decision recorded"
	eventType := "planner.decision_recorded"
	status := "succeeded"
	severity := "success"
	switch decisionType {
	case decisions.TypeAddExperiments:
		title = "Valid decision recorded"
	case decisions.TypeSelectChampion:
		title = "Champion decision recorded"
		eventType = "champion.decision_recorded"
	case decisions.TypeStopProject:
		title = "Stopped by planner"
		eventType = "planner.stopped"
		status = "blocked"
		severity = "info"
	case decisions.TypeWait:
		title = "Waiting for more jobs"
		eventType = "planner.waiting"
		status = "waiting"
		severity = "info"
	case decisions.TypeReopenExperimentation:
		title = "Experimentation reopened"
		eventType = "project.reopened"
		status = "active"
	}

	message := firstNonEmpty(
		activityMetadataString(decision.Payload, "summary"),
		decision.Rationale,
		decisionType,
	)
	return agentActivityEvent{
		ID:        activityID("decision", decision.ID),
		ProjectID: decision.ProjectID,
		PlanID:    decision.PlanID,
		Type:      eventType,
		Severity:  severity,
		Title:     title,
		Message:   activitySafeText(message, 220),
		Status:    status,
		CreatedAt: decision.CreatedAt,
		Metadata:  metadata,
	}
}

func activityFromJobs(projectID string, projectJobs []jobs.ExperimentJob) []agentActivityEvent {
	if len(projectJobs) == 0 {
		return nil
	}
	counts := activityJobCounts(projectJobs)
	latestJob := latestActivityJob(projectJobs)
	createdAt := activityJobTimestamp(latestJob)
	metadata := map[string]any{
		"job_count":       len(projectJobs),
		"queued_count":    counts[jobs.StatusQueued],
		"assigned_count":  counts[jobs.StatusAssigned],
		"running_count":   counts[jobs.StatusRunning],
		"completed_count": counts[jobs.StatusSucceeded],
		"failed_count":    counts[jobs.StatusFailed],
	}
	activeCount := counts[jobs.StatusAssigned] + counts[jobs.StatusRunning]
	openCount := activeCount + counts[jobs.StatusQueued]
	title := "Jobs completed"
	status := "succeeded"
	severity := "success"
	message := fmt.Sprintf("%d job(s) completed.", counts[jobs.StatusSucceeded])
	if activeCount > 0 {
		title = "Jobs running"
		status = "active"
		severity = "info"
		message = fmt.Sprintf("%d job(s) running, %d queued.", activeCount, counts[jobs.StatusQueued])
	} else if counts[jobs.StatusQueued] > 0 {
		title = "Jobs queued"
		status = "waiting"
		severity = "info"
		message = fmt.Sprintf("%d job(s) queued; waiting for workers.", counts[jobs.StatusQueued])
	} else if counts[jobs.StatusFailed] > 0 && counts[jobs.StatusSucceeded] == 0 {
		title = "Jobs failed"
		status = "failed"
		severity = "error"
		message = fmt.Sprintf("%d job(s) failed.", counts[jobs.StatusFailed])
	}
	if openCount == 0 && counts[jobs.StatusSucceeded] > 0 && counts[jobs.StatusFailed] > 0 {
		message = fmt.Sprintf("%d succeeded, %d failed.", counts[jobs.StatusSucceeded], counts[jobs.StatusFailed])
	}
	if latestJob.ID != "" {
		metadata["latest_job_id"] = latestJob.ID
		metadata["latest_job_status"] = latestJob.Status
	}

	summaryVariant := fmt.Sprintf("%v:%s:%s", metadata, latestJob.ID, latestJob.Status)
	out := []agentActivityEvent{{
		ID:        activityID("jobs", activityShortHash(summaryVariant)),
		ProjectID: projectID,
		JobID:     latestJob.ID,
		Type:      "jobs.status_counts",
		Severity:  severity,
		Title:     title,
		Message:   message,
		Status:    status,
		CreatedAt: createdAt,
		Metadata:  metadata,
	}}

	sortedJobs := append([]jobs.ExperimentJob(nil), projectJobs...)
	sort.SliceStable(sortedJobs, func(i, j int) bool {
		return activityJobTimestamp(sortedJobs[i]).After(activityJobTimestamp(sortedJobs[j]))
	})
	for index, job := range sortedJobs {
		if index >= 3 {
			break
		}
		out = append(out, activityFromJob(projectID, job))
	}
	return out
}

func activityFromJob(projectID string, job jobs.ExperimentJob) agentActivityEvent {
	status := strings.ToUpper(strings.TrimSpace(job.Status))
	eventType := "job.queued"
	title := "Job queued"
	severity := "info"
	activityStatus := "waiting"
	switch status {
	case jobs.StatusAssigned, jobs.StatusRunning:
		eventType = "job.running"
		title = "Job running"
		activityStatus = "active"
	case jobs.StatusSucceeded:
		eventType = "job.completed"
		title = "Job completed"
		severity = "success"
		activityStatus = "succeeded"
	case jobs.StatusFailed:
		eventType = "job.failed"
		title = "Job failed"
		severity = "error"
		activityStatus = "failed"
	}
	metadata := map[string]any{
		"job_status": status,
		"template":   activitySafeIdentifier(job.Template),
		"attempt":    job.Attempt,
	}
	if job.MaxAttempts > 0 {
		metadata["max_attempts"] = job.MaxAttempts
	}
	return agentActivityEvent{
		ID:        activityID("job", job.ID, strings.ToLower(status)),
		ProjectID: projectID,
		PlanID:    activitySafeIdentifier(jobConfigString(job.Config, "plan_id")),
		JobID:     job.ID,
		Type:      eventType,
		Severity:  severity,
		Title:     title,
		Message:   fmt.Sprintf("Job %s is %s.", job.ID, strings.ToLower(status)),
		Status:    activityStatus,
		CreatedAt: activityJobTimestamp(job),
		Metadata:  metadata,
	}
}

func activityMetadataFromPayload(payload map[string]any) map[string]any {
	out := map[string]any{}
	allowed := []string{
		"agent_name",
		"decision_id",
		"source_decision_id",
		"decision_type",
		"job_id",
		"job_ids",
		"worker_requirement_id",
		"open_job_count",
		"active_worker_count",
		"target_count",
		"previous_slot_count",
		"slot_count",
		"desired_slot_count",
		"registered_slot_count",
		"active_slot_count",
		"idle_seconds",
		"idle_exit_seconds",
		"dispatcher",
		"provider",
		"gpu_type",
		"requirement_status",
		"template",
		"attempt",
		"max_attempts",
		"requeued",
		"backend_validation_status",
		"backend_stop_guard",
		"reason",
		"model",
		"selection_source",
		"materialization_status",
		"max_concurrent_jobs",
		"max_cold_dataset_materializations",
		"retry_attempt",
		"will_retry",
		"completed_run_count",
		"memory_count",
		"evaluation_count",
		"purpose",
		"retrieved_count",
		"log_only",
		"cross_project_ok",
	}
	for _, key := range allowed {
		if value, ok := activityMetadataValue(payload[key]); ok {
			out[key] = value
		}
	}
	if validationError := activityMetadataString(payload, "backend_validation_error"); validationError != "" {
		out["validation_error"] = activitySafeText(validationError, 180)
	}
	if errText := firstNonEmpty(activityMetadataString(payload, "error"), activityMetadataString(payload, "last_error")); errText != "" {
		out["error_summary"] = activitySafeText(errText, 180)
	}
	return out
}

func activityPurposeSuffix(purpose string) string {
	switch strings.TrimSpace(purpose) {
	case "planner":
		return " for planner review"
	case "training_monitor":
		return " for training monitor review"
	default:
		if purpose == "" {
			return ""
		}
		return " for " + activitySafeText(strings.ReplaceAll(purpose, "_", " "), 60)
	}
}

func activityMetadataValue(value any) (any, bool) {
	switch typed := value.(type) {
	case nil:
		return nil, false
	case string:
		text := activitySafeText(typed, 120)
		return text, text != ""
	case bool:
		return typed, true
	case int:
		return typed, true
	case int64:
		return typed, true
	case float64:
		if typed == float64(int64(typed)) {
			return int64(typed), true
		}
		return typed, true
	case json.Number:
		if asInt, err := typed.Int64(); err == nil {
			return asInt, true
		}
		if asFloat, err := typed.Float64(); err == nil {
			return asFloat, true
		}
		return nil, false
	case []string:
		return activitySafeStringSlice(typed, 8), len(typed) > 0
	case []any:
		values := []string{}
		for _, item := range typed {
			if text := activitySafeText(fmt.Sprint(item), 80); text != "" {
				values = append(values, text)
			}
			if len(values) >= 8 {
				break
			}
		}
		return values, len(values) > 0
	default:
		return nil, false
	}
}

func activityDecisionModels(payload map[string]any) []string {
	value, ok := payload["proposed_experiments"]
	if !ok || value == nil {
		return nil
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	rows := []map[string]any{}
	if err := json.Unmarshal(blob, &rows); err != nil {
		return nil
	}
	models := []string{}
	seen := map[string]bool{}
	for _, row := range rows {
		model := activitySafeIdentifier(activityMetadataString(row, "model"))
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		models = append(models, model)
		if len(models) >= 6 {
			break
		}
	}
	return models
}

func activityHasActiveWork(projectJobs []jobs.ExperimentJob, invocations []memory.AgentInvocation) bool {
	for _, job := range projectJobs {
		switch strings.ToUpper(strings.TrimSpace(job.Status)) {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
			return true
		}
	}
	for _, invocation := range invocations {
		if activityMetadataBool(invocation.DownstreamOutcome, "will_retry") {
			return true
		}
	}
	return false
}

func activityJobCounts(projectJobs []jobs.ExperimentJob) map[string]int {
	counts := map[string]int{
		jobs.StatusQueued:    0,
		jobs.StatusAssigned:  0,
		jobs.StatusRunning:   0,
		jobs.StatusSucceeded: 0,
		jobs.StatusFailed:    0,
	}
	for _, job := range projectJobs {
		counts[strings.ToUpper(strings.TrimSpace(job.Status))]++
	}
	return counts
}

func latestActivityJob(projectJobs []jobs.ExperimentJob) jobs.ExperimentJob {
	var latest jobs.ExperimentJob
	for _, job := range projectJobs {
		if latest.ID == "" || activityJobTimestamp(job).After(activityJobTimestamp(latest)) {
			latest = job
		}
	}
	return latest
}

func activityJobTimestamp(job jobs.ExperimentJob) time.Time {
	if job.CompletedAt != nil {
		return *job.CompletedAt
	}
	if job.StartedAt != nil {
		return *job.StartedAt
	}
	if !job.CreatedAt.IsZero() {
		return job.CreatedAt
	}
	return time.Now().UTC()
}

func dedupeActivityEvents(events []agentActivityEvent) []agentActivityEvent {
	out := []agentActivityEvent{}
	seen := map[string]bool{}
	for _, event := range events {
		if event.ID == "" || seen[event.ID] {
			continue
		}
		seen[event.ID] = true
		out = append(out, event)
	}
	return out
}

func activityLimitFromQuery(value string) int {
	limit, _ := strconv.Atoi(value)
	return activityClampLimit(limit)
}

func activityClampLimit(limit int) int {
	if limit < 1 {
		return activityDefaultLimit
	}
	if limit > activityMaxLimit {
		return activityMaxLimit
	}
	return limit
}

func activityID(parts ...string) string {
	clean := []string{"activity"}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		clean = append(clean, activitySafeIdentifier(part))
	}
	return strings.Join(clean, "_")
}

func activityShortHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:12]
}

func activitySafeIdentifier(value string) string {
	value = activitySafeText(value, 80)
	var builder strings.Builder
	for _, r := range value {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' || r == '.' {
			builder.WriteRune(r)
			continue
		}
		if unicode.IsSpace(r) {
			builder.WriteRune('_')
		}
	}
	return strings.Trim(builder.String(), "_.-")
}

func activitySafeText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	value = activitySecretRe.ReplaceAllString(value, "[redacted_secret]")
	value = activityBase64Re.ReplaceAllString(value, "[redacted_blob]")
	value = activityStorageURIRe.ReplaceAllString(value, "[redacted_uri]")
	value = activityWindowsPathRe.ReplaceAllString(value, "[redacted_path]")
	value = activityUnixPathRe.ReplaceAllString(value, "$1[redacted_path]")
	value = activityWhitespaceRe.ReplaceAllString(value, " ")
	value = strings.TrimSpace(value)
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	if maxRunes <= 1 {
		return string(runes[:maxRunes])
	}
	return strings.TrimSpace(string(runes[:maxRunes-1])) + "..."
}

func activitySafeStringSlice(values []string, limit int) []string {
	out := []string{}
	for _, value := range values {
		text := activitySafeText(value, 80)
		if text == "" {
			continue
		}
		out = append(out, text)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func activityMetadataString(payload map[string]any, key string) string {
	value, ok := payload[key]
	if !ok || value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	case json.Number:
		return typed.String()
	default:
		return ""
	}
}

func activityMetadataStringSlice(payload map[string]any, key string) []string {
	values := payloadStringSlice(payload, key)
	return activitySafeStringSlice(values, 8)
}

func activityMetadataInt(payload map[string]any, key string) int {
	if value, ok := activityOptionalInt(payload, key); ok {
		return value
	}
	return 0
}

func activityOptionalInt(payload map[string]any, key string) (int, bool) {
	value, ok := payload[key]
	if !ok || value == nil {
		return 0, false
	}
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		return int(typed), true
	case json.Number:
		asInt, err := typed.Int64()
		if err == nil {
			return int(asInt), true
		}
		return 0, false
	default:
		return 0, false
	}
}

func activityMetadataBool(payload map[string]any, key string) bool {
	return payloadBool(payload, key)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
