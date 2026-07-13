package jobs

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
)

const (
	ProgressTaxonomyVersion = 1

	ProgressStageQueued                 = "queued"
	ProgressStageWorkerStarting         = "worker_starting"
	ProgressStageRemoteScheduled        = "remote_scheduled"
	ProgressStageEnvironmentStarting    = "environment_starting"
	ProgressStageDatasetMaterializing   = "dataset_materializing"
	ProgressStageDataLoading            = "data_loading"
	ProgressStageModelInitializing      = "model_initializing"
	ProgressStageTraining               = "training"
	ProgressStageEvaluating             = "evaluating"
	ProgressStageExporting              = "exporting"
	ProgressStageFinalizing             = "finalizing"
	ProgressStageCompleted              = "completed"
	ProgressStageFailed                 = "failed"
	ProgressStageCancelled              = "cancelled"
	ProgressStatusQueued                = "queued"
	ProgressStatusRunning               = "running"
	ProgressStatusCompleted             = "completed"
	ProgressStatusFailed                = "failed"
	ProgressStatusCancelled             = "cancelled"
	JobProgressMaxDetailCodeBytes       = 64
	JobProgressMaxUnitBytes             = 32
	JobProgressMaxMessageBytes          = 512
	JobProgressMaxMetadataEntries       = 16
	JobProgressMaxMetadataKeyBytes      = 64
	JobProgressMaxMetadataStringBytes   = 256
	JobProgressMaxMetadataArrayElements = 8
	JobProgressMaxMetadataJSONBytes     = 2048
	// Reserve one revision so an authoritative terminal transition can always
	// supersede the largest accepted worker revision without integer overflow.
	JobProgressMaxRevision int64 = 1<<63 - 2
)

var (
	progressCodePattern        = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]*$`)
	progressUnsafeTextPattern  = regexp.MustCompile(`(?i)(?:\b(?:s3|gs|https?|file)://|[a-z]:\\|(?:^|[[:space:]])/(?:[^[:space:]]+)|\b(?:bearer|authorization)[[:space:]]+|\b(?:sk|pk)-[a-z0-9_-]{8,})`)
	workerProgressMetadataKeys = map[string]struct{}{
		"cache_status":   {},
		"early_stopped":  {},
		"execution_mode": {},
		"framework":      {},
		"provider":       {},
		"resource_class": {},
		"task_type":      {},
	}
	workerProgressUnits = map[string]struct{}{
		"batch":   {},
		"batches": {},
		"byte":    {},
		"bytes":   {},
		"epoch":   {},
		"epochs":  {},
		"item":    {},
		"items":   {},
		"sample":  {},
		"samples": {},
		"step":    {},
		"steps":   {},
		"percent": {},
	}
)

// JobProgress is the replaceable current snapshot for one job attempt. Older
// attempt rows remain addressable, while callers select the apparent current
// snapshot using the attempt owned by the job lifecycle.
type JobProgress struct {
	ProjectID       string         `json:"project_id"`
	JobID           string         `json:"job_id"`
	Attempt         int            `json:"attempt"`
	TaxonomyVersion int            `json:"taxonomy_version"`
	Stage           string         `json:"stage"`
	DetailCode      string         `json:"detail_code,omitempty"`
	Status          string         `json:"status"`
	Current         *int64         `json:"current,omitempty"`
	Total           *int64         `json:"total,omitempty"`
	Unit            string         `json:"unit,omitempty"`
	Message         string         `json:"message,omitempty"`
	Revision        int64          `json:"revision"`
	HeartbeatAt     time.Time      `json:"heartbeat_at"`
	UpdatedAt       time.Time      `json:"updated_at"`
	Metadata        map[string]any `json:"metadata"`
}

// JobProgressUpsert omits identities and receipt timestamps so stores, not
// workers, remain authoritative for project ownership and staleness clocks.
type JobProgressUpsert struct {
	Attempt         int            `json:"attempt"`
	TaxonomyVersion int            `json:"taxonomy_version"`
	Stage           string         `json:"stage"`
	DetailCode      string         `json:"detail_code,omitempty"`
	Status          string         `json:"status"`
	Current         *int64         `json:"current,omitempty"`
	Total           *int64         `json:"total,omitempty"`
	Unit            string         `json:"unit,omitempty"`
	Message         string         `json:"message,omitempty"`
	Revision        int64          `json:"revision"`
	Metadata        map[string]any `json:"metadata,omitempty"`
}

type JobProgressReportResult struct {
	Progress     JobProgress
	Updated      bool
	EventCreated bool
}

// NormalizeJobProgressUpsert canonicalizes taxonomy tokens, validates stage
// and status compatibility, and copies the bounded metadata envelope.
func NormalizeJobProgressUpsert(update JobProgressUpsert) (JobProgressUpsert, error) {
	update.Stage = strings.ToLower(strings.TrimSpace(update.Stage))
	update.Status = strings.ToLower(strings.TrimSpace(update.Status))
	update.DetailCode = strings.ToLower(strings.TrimSpace(update.DetailCode))
	update.Unit = strings.ToLower(strings.TrimSpace(update.Unit))
	update.Message = strings.TrimSpace(update.Message)

	if update.Attempt < 0 {
		return JobProgressUpsert{}, fmt.Errorf("attempt must be nonnegative")
	}
	if update.TaxonomyVersion != ProgressTaxonomyVersion {
		return JobProgressUpsert{}, fmt.Errorf("unsupported taxonomy version %d", update.TaxonomyVersion)
	}
	if !IsJobProgressStage(update.Stage) {
		return JobProgressUpsert{}, fmt.Errorf("unsupported progress stage %q", update.Stage)
	}
	if !IsJobProgressStatus(update.Status) {
		return JobProgressUpsert{}, fmt.Errorf("unsupported progress status %q", update.Status)
	}
	if !progressStageAllowsStatus(update.Stage, update.Status) {
		return JobProgressUpsert{}, fmt.Errorf("progress stage %q is incompatible with status %q", update.Stage, update.Status)
	}
	if update.Revision < 0 || update.Revision > JobProgressMaxRevision {
		return JobProgressUpsert{}, fmt.Errorf("revision must be between 0 and %d", JobProgressMaxRevision)
	}
	if err := validateOptionalProgressCode("detail code", update.DetailCode, JobProgressMaxDetailCodeBytes); err != nil {
		return JobProgressUpsert{}, err
	}
	if err := validateOptionalProgressCode("unit", update.Unit, JobProgressMaxUnitBytes); err != nil {
		return JobProgressUpsert{}, err
	}
	if len(update.Message) > JobProgressMaxMessageBytes {
		return JobProgressUpsert{}, fmt.Errorf("message exceeds %d bytes", JobProgressMaxMessageBytes)
	}
	if strings.ContainsAny(update.Message, "\x00\r\n") {
		return JobProgressUpsert{}, fmt.Errorf("message contains control characters")
	}
	if err := validateJobProgressSafeText("message", update.Message); err != nil {
		return JobProgressUpsert{}, err
	}
	if update.Current != nil && *update.Current < 0 {
		return JobProgressUpsert{}, fmt.Errorf("current must be nonnegative")
	}
	if update.Total != nil && *update.Total < 0 {
		return JobProgressUpsert{}, fmt.Errorf("total must be nonnegative")
	}
	if update.Current != nil && update.Total != nil && *update.Current > *update.Total {
		return JobProgressUpsert{}, fmt.Errorf("current must not exceed total")
	}

	metadata, err := normalizeJobProgressMetadata(update.Metadata)
	if err != nil {
		return JobProgressUpsert{}, err
	}
	update.Metadata = metadata
	update.Current = copyProgressInt64(update.Current)
	update.Total = copyProgressInt64(update.Total)
	return update, nil
}

// NormalizeWorkerJobProgressUpsert applies the narrower callback contract on
// top of the server lifecycle snapshot contract. Workers can publish only
// observations from worker_starting through finalizing; terminal stages remain
// exclusively backend-owned. Metadata is a closed allowlist so a callback can
// never turn the progress record into an arbitrary payload store.
func NormalizeWorkerJobProgressUpsert(update JobProgressUpsert) (JobProgressUpsert, error) {
	normalized, err := NormalizeJobProgressUpsert(update)
	if err != nil {
		return JobProgressUpsert{}, err
	}
	if !IsWorkerJobProgressStage(normalized.Stage) {
		return JobProgressUpsert{}, fmt.Errorf("workers cannot report progress stage %q", normalized.Stage)
	}
	if normalized.Status != ProgressStatusRunning {
		return JobProgressUpsert{}, fmt.Errorf("worker progress status must be %q", ProgressStatusRunning)
	}
	if normalized.Revision < 1 {
		return JobProgressUpsert{}, fmt.Errorf("worker progress revision must be positive")
	}
	if (normalized.Current == nil) != (normalized.Total == nil) {
		return JobProgressUpsert{}, fmt.Errorf("current and total must be reported together")
	}
	if normalized.Current != nil && normalized.Unit == "" {
		return JobProgressUpsert{}, fmt.Errorf("unit is required when current and total are reported")
	}
	if normalized.Current == nil && normalized.Unit != "" {
		return JobProgressUpsert{}, fmt.Errorf("unit requires current or total")
	}
	if normalized.Unit != "" {
		if _, ok := workerProgressUnits[normalized.Unit]; !ok {
			return JobProgressUpsert{}, fmt.Errorf("worker progress unit %q is not supported", normalized.Unit)
		}
	}
	for key, value := range normalized.Metadata {
		if _, ok := workerProgressMetadataKeys[key]; !ok {
			return JobProgressUpsert{}, fmt.Errorf("metadata key %q is not allowed for worker progress", key)
		}
		if key == "early_stopped" {
			if _, ok := value.(bool); !ok {
				return JobProgressUpsert{}, fmt.Errorf("metadata %q must be a boolean", key)
			}
			continue
		}
		token, ok := value.(string)
		if !ok || token == "" || len(token) > JobProgressMaxDetailCodeBytes || !progressCodePattern.MatchString(token) {
			return JobProgressUpsert{}, fmt.Errorf("metadata %q must be a bounded lowercase safe token", key)
		}
	}
	return normalized, nil
}

func IsWorkerJobProgressStage(stage string) bool {
	switch stage {
	case ProgressStageWorkerStarting,
		ProgressStageRemoteScheduled,
		ProgressStageEnvironmentStarting,
		ProgressStageDatasetMaterializing,
		ProgressStageDataLoading,
		ProgressStageModelInitializing,
		ProgressStageTraining,
		ProgressStageEvaluating,
		ProgressStageExporting,
		ProgressStageFinalizing:
		return true
	default:
		return false
	}
}

func WorkerJobProgressMetadataKeys() []string {
	return []string{
		"cache_status",
		"early_stopped",
		"execution_mode",
		"framework",
		"provider",
		"resource_class",
		"task_type",
	}
}

func IsJobProgressStage(stage string) bool {
	switch stage {
	case ProgressStageQueued,
		ProgressStageWorkerStarting,
		ProgressStageRemoteScheduled,
		ProgressStageEnvironmentStarting,
		ProgressStageDatasetMaterializing,
		ProgressStageDataLoading,
		ProgressStageModelInitializing,
		ProgressStageTraining,
		ProgressStageEvaluating,
		ProgressStageExporting,
		ProgressStageFinalizing,
		ProgressStageCompleted,
		ProgressStageFailed,
		ProgressStageCancelled:
		return true
	default:
		return false
	}
}

func IsTerminalJobProgressStage(stage string) bool {
	switch stage {
	case ProgressStageCompleted, ProgressStageFailed, ProgressStageCancelled:
		return true
	default:
		return false
	}
}

func IsJobProgressStatus(status string) bool {
	switch status {
	case ProgressStatusQueued, ProgressStatusRunning, ProgressStatusCompleted, ProgressStatusFailed, ProgressStatusCancelled:
		return true
	default:
		return false
	}
}

func progressStageAllowsStatus(stage string, status string) bool {
	switch stage {
	case ProgressStageQueued:
		return status == ProgressStatusQueued
	case ProgressStageCompleted:
		return status == ProgressStatusCompleted
	case ProgressStageFailed:
		return status == ProgressStatusFailed
	case ProgressStageCancelled:
		return status == ProgressStatusCancelled
	default:
		return status == ProgressStatusRunning
	}
}

func validateOptionalProgressCode(name string, value string, maxBytes int) error {
	if value == "" {
		return nil
	}
	if len(value) > maxBytes {
		return fmt.Errorf("%s exceeds %d bytes", name, maxBytes)
	}
	if !progressCodePattern.MatchString(value) {
		return fmt.Errorf("%s must be a lowercase safe token", name)
	}
	return nil
}

func normalizeJobProgressMetadata(metadata map[string]any) (map[string]any, error) {
	if metadata == nil {
		return map[string]any{}, nil
	}
	if len(metadata) > JobProgressMaxMetadataEntries {
		return nil, fmt.Errorf("metadata exceeds %d entries", JobProgressMaxMetadataEntries)
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if len(key) == 0 || len(key) > JobProgressMaxMetadataKeyBytes || !progressCodePattern.MatchString(key) {
			return nil, fmt.Errorf("metadata key %q must be a bounded lowercase safe token", key)
		}
		if jobProgressSensitiveMetadataKey(key) {
			return nil, fmt.Errorf("metadata key %q is not allowed in safe progress", key)
		}
		normalized, err := normalizeJobProgressMetadataValue(value)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		out[key] = normalized
	}
	encoded, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("marshal metadata: %w", err)
	}
	if len(encoded) > JobProgressMaxMetadataJSONBytes {
		return nil, fmt.Errorf("metadata exceeds %d encoded bytes", JobProgressMaxMetadataJSONBytes)
	}
	return out, nil
}

func normalizeJobProgressMetadataValue(value any) (any, error) {
	switch typed := value.(type) {
	case string:
		if len(typed) > JobProgressMaxMetadataStringBytes || strings.ContainsAny(typed, "\x00\r\n") {
			return nil, fmt.Errorf("string value is not bounded safe text")
		}
		if err := validateJobProgressSafeText("metadata string", typed); err != nil {
			return nil, err
		}
		return typed, nil
	case bool:
		return typed, nil
	case int:
		return typed, nil
	case int8:
		return typed, nil
	case int16:
		return typed, nil
	case int32:
		return typed, nil
	case int64:
		return typed, nil
	case uint:
		return typed, nil
	case uint8:
		return typed, nil
	case uint16:
		return typed, nil
	case uint32:
		return typed, nil
	case uint64:
		return typed, nil
	case float32:
		if math.IsNaN(float64(typed)) || math.IsInf(float64(typed), 0) {
			return nil, fmt.Errorf("number must be finite")
		}
		return typed, nil
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) {
			return nil, fmt.Errorf("number must be finite")
		}
		return typed, nil
	case []string:
		if len(typed) > JobProgressMaxMetadataArrayElements {
			return nil, fmt.Errorf("array exceeds %d elements", JobProgressMaxMetadataArrayElements)
		}
		out := make([]string, len(typed))
		for index, item := range typed {
			if len(item) > JobProgressMaxMetadataStringBytes || strings.ContainsAny(item, "\x00\r\n") {
				return nil, fmt.Errorf("array string at index %d is not bounded safe text", index)
			}
			if err := validateJobProgressSafeText("metadata array string", item); err != nil {
				return nil, fmt.Errorf("array string at index %d: %w", index, err)
			}
			out[index] = item
		}
		return out, nil
	case []any:
		if len(typed) > JobProgressMaxMetadataArrayElements {
			return nil, fmt.Errorf("array exceeds %d elements", JobProgressMaxMetadataArrayElements)
		}
		out := make([]any, len(typed))
		for index, item := range typed {
			normalized, err := normalizeJobProgressMetadataValue(item)
			if err != nil {
				return nil, fmt.Errorf("array value at index %d: %w", index, err)
			}
			if _, nested := normalized.([]any); nested {
				return nil, fmt.Errorf("nested arrays are not allowed")
			}
			if _, nested := normalized.([]string); nested {
				return nil, fmt.Errorf("nested arrays are not allowed")
			}
			out[index] = normalized
		}
		return out, nil
	default:
		return nil, fmt.Errorf("value type %T is not allowed", value)
	}
}

func validateJobProgressSafeText(name string, value string) error {
	if progressUnsafeTextPattern.MatchString(value) {
		return fmt.Errorf("%s contains a secret, storage URI, or filesystem path", name)
	}
	return nil
}

func jobProgressSensitiveMetadataKey(key string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(key, "-", "_"))
	for _, marker := range []string{
		"token", "secret", "password", "credential", "api_key", "authorization", "cookie",
		"storage_uri", "path", "prompt", "raw_output", "parsed_output",
	} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}

func copyProgressInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
