package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

const (
	projectLiveStateSchema       = "project_live_state.v1"
	projectLiveStateStaleAfter   = 60 * time.Second
	projectLiveStateMaxSafeBytes = 16 * 1024
	projectLiveStateQueryBudget  = 6
)

var liveStateMetadataTokenPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]*$`)

type projectLiveStateEnvelope struct {
	SchemaVersion           string                           `json:"schema_version"`
	ProjectID               string                           `json:"project_id"`
	OperationalState        string                           `json:"operational_state"`
	TaxonomyVersion         int                              `json:"taxonomy_version"`
	CurrentStage            string                           `json:"current_stage,omitempty"`
	NextExpectedStage       string                           `json:"next_expected_stage,omitempty"`
	MixedJobs               bool                             `json:"mixed_jobs"`
	Stale                   bool                             `json:"stale"`
	BlockedReasonCode       string                           `json:"blocked_reason_code,omitempty"`
	Jobs                    store.LiveStateJobCounts         `json:"jobs"`
	Workers                 store.LiveStateWorkerCounts      `json:"workers"`
	WorkerRequirements      store.LiveStateRequirementCounts `json:"worker_requirements"`
	ActiveProgress          []projectLiveStateProgress       `json:"active_progress"`
	ActiveProgressTotal     int                              `json:"active_progress_total"`
	ActiveProgressTruncated bool                             `json:"active_progress_truncated"`
	LastHeartbeatAt         *time.Time                       `json:"last_heartbeat_at,omitempty"`
	LatestImportantEvent    *executionEventV2Envelope        `json:"latest_important_event,omitempty"`
	SnapshotCursor          int64                            `json:"snapshot_cursor"`
	SnapshotRevision        string                           `json:"snapshot_revision,omitempty"`
}

type projectLiveStateProgress struct {
	JobID            string         `json:"job_id"`
	Attempt          int            `json:"attempt"`
	TaxonomyVersion  int            `json:"taxonomy_version"`
	Stage            string         `json:"stage"`
	DetailCode       string         `json:"detail_code,omitempty"`
	Status           string         `json:"status"`
	Current          *int64         `json:"current,omitempty"`
	Total            *int64         `json:"total,omitempty"`
	Unit             string         `json:"unit,omitempty"`
	Message          string         `json:"message,omitempty"`
	Revision         int64          `json:"revision"`
	HeartbeatAt      time.Time      `json:"heartbeat_at"`
	UpdatedAt        time.Time      `json:"updated_at"`
	ElapsedStartedAt time.Time      `json:"elapsed_started_at"`
	Stale            bool           `json:"stale"`
	Metadata         map[string]any `json:"metadata"`
}

func (s *Server) getProjectLiveState(c *gin.Context) {
	startedAt := time.Now()
	bytesBefore := c.Writer.Size()
	snapshot, err := s.store.GetProjectLiveState(c.Request.Context(), c.Param("id"))
	if err != nil {
		if c.Request.Context().Err() != nil {
			return
		}
		diagnostics.Event("warn", "project_live_state_read", map[string]any{
			"duration_ms":      time.Since(startedAt).Milliseconds(),
			"response_bytes":   0,
			"store_call_count": 1,
			"query_count":      projectLiveStateQueryBudget,
			"error_count":      1,
			"reason_code":      "snapshot_read_failed",
		})
		writeStoreError(c, err)
		return
	}

	response := projectLiveStateProjection(snapshot)
	revision := projectLiveStateRevision(response)
	response.SnapshotRevision = revision
	etag := `"` + revision + `"`
	c.Header("Cache-Control", "no-cache")
	c.Header("ETag", etag)
	if ifNoneMatchContains(c.GetHeader("If-None-Match"), etag) {
		c.Status(http.StatusNotModified)
		recordProjectLiveStateDiagnostic(startedAt, bytesBefore, c.Writer.Size(), response, "not_modified")
		return
	}
	c.JSON(http.StatusOK, response)
	recordProjectLiveStateDiagnostic(startedAt, bytesBefore, c.Writer.Size(), response, "snapshot")
}

func recordProjectLiveStateDiagnostic(
	startedAt time.Time,
	bytesBefore int,
	bytesAfter int,
	response projectLiveStateEnvelope,
	reasonCode string,
) {
	diagnostics.Event("info", "project_live_state_read", map[string]any{
		"duration_ms":           time.Since(startedAt).Milliseconds(),
		"response_bytes":        activityResponseByteDelta(bytesBefore, bytesAfter),
		"store_call_count":      1,
		"query_count":           projectLiveStateQueryBudget,
		"active_progress_count": len(response.ActiveProgress),
		"job_count":             response.Jobs.Total,
		"error_count":           0,
		"reason_code":           reasonCode,
	})
}

func projectLiveStateProjection(snapshot store.ProjectLiveStateSnapshot) projectLiveStateEnvelope {
	response := projectLiveStateEnvelope{
		SchemaVersion:           projectLiveStateSchema,
		ProjectID:               activitySafeIdentifier(snapshot.ProjectID),
		TaxonomyVersion:         jobs.ProgressTaxonomyVersion,
		Jobs:                    snapshot.Jobs,
		Workers:                 snapshot.Workers,
		WorkerRequirements:      snapshot.Requirements,
		ActiveProgress:          make([]projectLiveStateProgress, 0, len(snapshot.ActiveProgress)),
		ActiveProgressTotal:     snapshot.ActiveProgressTotal,
		ActiveProgressTruncated: snapshot.ActiveProgressTotal > len(snapshot.ActiveProgress),
		SnapshotCursor:          snapshot.SnapshotCursor,
		LastHeartbeatAt:         cloneAPITimePointer(snapshot.LastWorkerHeartbeat),
	}

	for _, active := range snapshot.ActiveProgress {
		heartbeat := active.Progress.HeartbeatAt.UTC()
		for _, candidate := range []*time.Time{active.WorkerHeartbeatAt, active.LeaseHeartbeatAt} {
			if candidate != nil && candidate.After(heartbeat) {
				heartbeat = candidate.UTC()
			}
		}
		stale := (active.JobStatus == jobs.StatusAssigned || active.JobStatus == jobs.StatusRunning) &&
			snapshot.ObservedAt.Sub(heartbeat) > projectLiveStateStaleAfter
		progress := active.Progress
		response.ActiveProgress = append(response.ActiveProgress, projectLiveStateProgress{
			JobID:            activitySafeIdentifier(progress.JobID),
			Attempt:          progress.Attempt,
			TaxonomyVersion:  progress.TaxonomyVersion,
			Stage:            liveStateStage(progress.Stage),
			DetailCode:       liveStateToken(progress.DetailCode),
			Status:           liveStateToken(progress.Status),
			Current:          cloneAPIInt64Pointer(progress.Current),
			Total:            cloneAPIInt64Pointer(progress.Total),
			Unit:             liveStateToken(progress.Unit),
			Message:          activitySafeText(progress.Message, 220),
			Revision:         progress.Revision,
			HeartbeatAt:      progress.HeartbeatAt.UTC(),
			UpdatedAt:        progress.UpdatedAt.UTC(),
			ElapsedStartedAt: active.ElapsedStartedAt.UTC(),
			Stale:            stale,
			Metadata:         projectLiveStateProgressMetadata(progress.Metadata),
		})
		response.Stale = response.Stale || stale
		if response.LastHeartbeatAt == nil || heartbeat.After(*response.LastHeartbeatAt) {
			value := heartbeat
			response.LastHeartbeatAt = &value
		}
	}

	if snapshot.LatestEvent != nil {
		projected := executionEventV2Projection(*snapshot.LatestEvent)
		response.LatestImportantEvent = &projected
	}
	response.CurrentStage = projectLiveStateCurrentStage(response)
	response.NextExpectedStage = projectLiveStateNextStage(response.CurrentStage)
	response.MixedJobs = projectLiveStateMixedJobs(response.Jobs)
	response.OperationalState, response.BlockedReasonCode = projectLiveStateOperationalState(response)
	return response
}

func projectLiveStateOperationalState(response projectLiveStateEnvelope) (string, string) {
	openJobs := response.Jobs.Queued + response.Jobs.Retrying + response.Jobs.Assigned + response.Jobs.Running
	if openJobs > 0 && response.WorkerRequirements.Failed > 0 {
		return "blocked", "worker_requirement_failed"
	}
	if openJobs > 0 && response.WorkerRequirements.Cancelled > 0 {
		return "blocked", "worker_requirement_cancelled"
	}
	if response.LatestImportantEvent != nil {
		if response.LatestImportantEvent.EventType == execution.EventCostBudgetBlocked && openJobs > 0 {
			return "blocked", "cost_budget_blocked"
		}
		if activityMetadataString(response.LatestImportantEvent.Metadata, "backend_validation_status") == "blocked" && openJobs == 0 {
			return "blocked", "backend_validation_blocked"
		}
	}
	if response.Stale {
		return "stale", ""
	}
	if response.Jobs.Assigned+response.Jobs.Running > 0 {
		return "active", ""
	}
	if response.Jobs.Retrying > 0 {
		return "retrying", ""
	}
	if response.Jobs.Queued > 0 {
		return "queued", ""
	}
	terminal := response.Jobs.Succeeded + response.Jobs.Failed + response.Jobs.Cancelled
	if response.Jobs.Total > 0 && terminal == response.Jobs.Total {
		return "terminal", ""
	}
	return "idle", ""
}

func projectLiveStateCurrentStage(response projectLiveStateEnvelope) string {
	if len(response.ActiveProgress) > 0 && jobs.IsJobProgressStage(response.ActiveProgress[0].Stage) {
		return response.ActiveProgress[0].Stage
	}
	if response.Jobs.Assigned+response.Jobs.Running > 0 {
		return jobs.ProgressStageWorkerStarting
	}
	if response.Jobs.Queued+response.Jobs.Retrying > 0 {
		return jobs.ProgressStageQueued
	}
	if response.Jobs.Failed > 0 {
		return jobs.ProgressStageFailed
	}
	if response.Jobs.Cancelled > 0 {
		return jobs.ProgressStageCancelled
	}
	if response.Jobs.Succeeded > 0 {
		return jobs.ProgressStageCompleted
	}
	return ""
}

func projectLiveStateNextStage(stage string) string {
	switch stage {
	case jobs.ProgressStageQueued:
		return jobs.ProgressStageWorkerStarting
	case jobs.ProgressStageWorkerStarting:
		return jobs.ProgressStageRemoteScheduled
	case jobs.ProgressStageRemoteScheduled:
		return jobs.ProgressStageEnvironmentStarting
	case jobs.ProgressStageEnvironmentStarting:
		return jobs.ProgressStageDatasetMaterializing
	case jobs.ProgressStageDatasetMaterializing:
		return jobs.ProgressStageDataLoading
	case jobs.ProgressStageDataLoading:
		return jobs.ProgressStageModelInitializing
	case jobs.ProgressStageModelInitializing:
		return jobs.ProgressStageTraining
	case jobs.ProgressStageTraining:
		return jobs.ProgressStageEvaluating
	case jobs.ProgressStageEvaluating:
		return jobs.ProgressStageExporting
	case jobs.ProgressStageExporting:
		return jobs.ProgressStageFinalizing
	case jobs.ProgressStageFinalizing:
		return jobs.ProgressStageCompleted
	default:
		return ""
	}
}

func projectLiveStateMixedJobs(counts store.LiveStateJobCounts) bool {
	buckets := 0
	for _, count := range []int{
		counts.Queued,
		counts.Retrying,
		counts.Assigned + counts.Running,
		counts.Succeeded + counts.Failed + counts.Cancelled,
	} {
		if count > 0 {
			buckets++
		}
	}
	return buckets > 1
}

func projectLiveStateProgressMetadata(metadata map[string]any) map[string]any {
	allowed := map[string]bool{}
	for _, key := range jobs.WorkerJobProgressMetadataKeys() {
		allowed[key] = true
	}
	out := map[string]any{}
	for key, value := range metadata {
		if !allowed[key] {
			continue
		}
		if key == "early_stopped" {
			if typed, ok := value.(bool); ok {
				out[key] = typed
			}
			continue
		}
		if typed, ok := value.(string); ok {
			if token := liveStateToken(typed); token != "" {
				out[key] = token
			}
		}
	}
	return out
}

func liveStateStage(value string) string {
	value = liveStateToken(value)
	if !jobs.IsJobProgressStage(value) {
		return ""
	}
	return value
}

func liveStateToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" || len([]byte(value)) > jobs.JobProgressMaxDetailCodeBytes || !liveStateMetadataTokenPattern.MatchString(value) {
		return ""
	}
	return value
}

func projectLiveStateRevision(response projectLiveStateEnvelope) string {
	response.SnapshotRevision = ""
	encoded, _ := json.Marshal(response)
	digest := sha256.Sum256(encoded)
	return "live-" + hex.EncodeToString(digest[:12])
}

func ifNoneMatchContains(header string, etag string) bool {
	for _, value := range strings.Split(header, ",") {
		if strings.TrimSpace(value) == etag || strings.TrimSpace(value) == "*" {
			return true
		}
	}
	return false
}

func cloneAPIInt64Pointer(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneAPITimePointer(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	cloned := value.UTC()
	return &cloned
}
