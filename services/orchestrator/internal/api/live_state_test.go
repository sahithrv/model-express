package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

func TestProjectLiveStateOperationalStates(t *testing.T) {
	now := time.Now().UTC()
	progress := func(stage string, heartbeat time.Time) store.LiveStateActiveProgress {
		return store.LiveStateActiveProgress{
			JobStatus:        jobs.StatusRunning,
			ElapsedStartedAt: heartbeat.Add(-5 * time.Minute),
			Progress: jobs.JobProgress{
				JobID: "job_1", Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
				Stage: stage, Status: jobs.ProgressStatusRunning, Revision: 3,
				HeartbeatAt: heartbeat, UpdatedAt: heartbeat, Metadata: map[string]any{},
			},
		}
	}
	tests := []struct {
		name          string
		snapshot      store.ProjectLiveStateSnapshot
		wantState     string
		wantStage     string
		wantNext      string
		wantMixed     bool
		wantStale     bool
		blockedReason string
	}{
		{name: "queued", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Queued: 1}}, wantState: "queued", wantStage: jobs.ProgressStageQueued, wantNext: jobs.ProgressStageWorkerStarting},
		{name: "active", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Running: 1}, ActiveProgress: []store.LiveStateActiveProgress{progress(jobs.ProgressStageTraining, now)}}, wantState: "active", wantStage: jobs.ProgressStageTraining, wantNext: jobs.ProgressStageEvaluating},
		{name: "retrying", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Retrying: 1}}, wantState: "retrying", wantStage: jobs.ProgressStageQueued, wantNext: jobs.ProgressStageWorkerStarting},
		{name: "stale", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Running: 1}, ActiveProgress: []store.LiveStateActiveProgress{progress(jobs.ProgressStageTraining, now.Add(-2*projectLiveStateStaleAfter))}}, wantState: "stale", wantStage: jobs.ProgressStageTraining, wantNext: jobs.ProgressStageEvaluating, wantStale: true},
		{name: "blocked", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Queued: 1}, Requirements: store.LiveStateRequirementCounts{Failed: 1}}, wantState: "blocked", wantStage: jobs.ProgressStageQueued, wantNext: jobs.ProgressStageWorkerStarting, blockedReason: "worker_requirement_failed"},
		{name: "mixed", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 2, Running: 1, Succeeded: 1}, ActiveProgress: []store.LiveStateActiveProgress{progress(jobs.ProgressStageEvaluating, now)}}, wantState: "active", wantStage: jobs.ProgressStageEvaluating, wantNext: jobs.ProgressStageExporting, wantMixed: true},
		{name: "terminal failed", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 2, Succeeded: 1, Failed: 1}}, wantState: "terminal", wantStage: jobs.ProgressStageFailed},
		{name: "terminal cancelled", snapshot: store.ProjectLiveStateSnapshot{Jobs: store.LiveStateJobCounts{Total: 1, Cancelled: 1}}, wantState: "terminal", wantStage: jobs.ProgressStageCancelled},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.snapshot.ProjectID = "project_1"
			test.snapshot.ObservedAt = now
			test.snapshot.ActiveProgressTotal = len(test.snapshot.ActiveProgress)
			got := projectLiveStateProjection(test.snapshot)
			if got.OperationalState != test.wantState || got.CurrentStage != test.wantStage || got.NextExpectedStage != test.wantNext {
				t.Fatalf("state/stage/next = %q/%q/%q, want %q/%q/%q", got.OperationalState, got.CurrentStage, got.NextExpectedStage, test.wantState, test.wantStage, test.wantNext)
			}
			if got.MixedJobs != test.wantMixed || got.Stale != test.wantStale || got.BlockedReasonCode != test.blockedReason {
				t.Fatalf("flags = mixed:%t stale:%t blocked:%q", got.MixedJobs, got.Stale, got.BlockedReasonCode)
			}
			for _, active := range got.ActiveProgress {
				if active.TaxonomyVersion != jobs.ProgressTaxonomyVersion || !jobs.IsJobProgressStage(active.Stage) {
					t.Fatalf("invalid progress taxonomy: %#v", active)
				}
				if active.ElapsedStartedAt.IsZero() || !active.ElapsedStartedAt.Before(active.UpdatedAt) {
					t.Fatalf("active progress lost elapsed start: %#v", active)
				}
			}
		})
	}
}

func TestProjectLiveStateEndpointIsBoundedRedactedAndCacheable(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("live", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 180; index++ {
		job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
			"dataset_id":  dataset.ID,
			"prompt":      strings.Repeat("private prompt ", 100),
			"storage_uri": "s3://secret-bucket/dataset.zip",
			"raw_config":  map[string]any{"secret": "value"},
		})
		if err != nil {
			t.Fatal(err)
		}
		if index == 0 {
			if _, err := memoryStore.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
				Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
				Stage: jobs.ProgressStageDataLoading, Status: jobs.ProgressStatusRunning, Revision: 5,
				DetailCode: "simulator_data_loading",
				Metadata: map[string]any{
					"provider":       "local",
					"execution_mode": "local_simulator",
					"model":          "private_model_name",
					"raw_config":     "private_config",
				},
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	latestEvent, err := memoryStore.CreateExecutionEvent(
		project.ID,
		"",
		execution.EventDispatcherStatus,
		"Reading s3://secret-bucket/dataset.zip from /tmp/private/config.json",
		map[string]any{
			"status":      "active",
			"provider":    "local",
			"prompt":      "private prompt",
			"storage_uri": "s3://secret-bucket/dataset.zip",
			"raw_config":  map[string]any{"secret": "value"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	counting := &liveStateCountingStore{Store: memoryStore}
	router := NewRouter(counting)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/live-state", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("live state status=%d body=%s", response.Code, response.Body.String())
	}
	if counting.calls != 1 {
		t.Fatalf("handler live-state store calls = %d, want 1", counting.calls)
	}
	if response.Body.Len() > projectLiveStateMaxSafeBytes {
		t.Fatalf("live state body = %d bytes, want <= %d", response.Body.Len(), projectLiveStateMaxSafeBytes)
	}
	body := response.Body.String()
	for _, forbidden := range []string{"private prompt", "secret-bucket", "/tmp/private", "storage_uri", "raw_config", latestEvent.IdempotencyKey} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("live state leaked %q: %s", forbidden, body)
		}
	}
	for _, required := range []string{`"provider":"local"`, `"execution_mode":"local_simulator"`, `"active_progress_truncated":true`, `"snapshot_cursor"`, `"elapsed_started_at"`, `"idempotency_key":"ik_`} {
		if !strings.Contains(body, required) {
			t.Fatalf("live state missing %s: %s", required, body)
		}
	}
	etag := response.Header().Get("ETag")
	if etag == "" {
		t.Fatal("live state did not return an ETag")
	}

	notModified := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/live-state", nil)
	request.Header.Set("If-None-Match", etag)
	router.ServeHTTP(notModified, request)
	if notModified.Code != http.StatusNotModified || notModified.Body.Len() != 0 {
		t.Fatalf("conditional live state = %d %q", notModified.Code, notModified.Body.String())
	}

	var decoded projectLiveStateEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.SchemaVersion != projectLiveStateSchema || decoded.SnapshotRevision == "" || decoded.SnapshotCursor < 1 {
		t.Fatalf("live state identity/cursor = %#v", decoded)
	}
	if len(decoded.ActiveProgress) != store.ProjectLiveStateProgressLimit || decoded.ActiveProgressTotal != 180 {
		t.Fatalf("bounded active progress = %d/%d", len(decoded.ActiveProgress), decoded.ActiveProgressTotal)
	}
	if decoded.LatestImportantEvent == nil || !regexp.MustCompile(`^ik_[0-9a-f]{64}$`).MatchString(decoded.LatestImportantEvent.IdempotencyKey) {
		t.Fatalf("latest event omitted opaque idempotency identity: %#v", decoded.LatestImportantEvent)
	}
	for _, active := range decoded.ActiveProgress {
		if active.ElapsedStartedAt.IsZero() {
			t.Fatalf("active progress omitted elapsed start: %#v", active)
		}
	}
}

type liveStateCountingStore struct {
	store.Store
	calls int
}

func (s *liveStateCountingStore) GetProjectLiveState(ctx context.Context, projectID string) (store.ProjectLiveStateSnapshot, error) {
	s.calls++
	return s.Store.GetProjectLiveState(ctx, projectID)
}
