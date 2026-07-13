package store

import (
	"errors"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/jobs"
)

func TestPostgresJobProgressUpsertPreservesRevisionAndTerminalAuthority(t *testing.T) {
	query := upsertJobProgressQuery()
	for _, required := range []string{
		"ON CONFLICT (job_id, attempt) DO UPDATE",
		"EXCLUDED.revision > job_progress.revision",
		"GREATEST(EXCLUDED.revision, job_progress.revision + 1)",
		"job_progress.stage NOT IN ('completed', 'failed', 'cancelled')",
		"EXCLUDED.stage IN ('completed', 'failed', 'cancelled')",
		"NOT EXISTS (SELECT 1 FROM upserted)",
	} {
		if !strings.Contains(query, required) {
			t.Fatalf("job-progress upsert query omitted %q", required)
		}
	}
}

func TestMemoryJobProgressIsAttemptScopedRevisionedAndImmutableAfterTerminal(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("progress", "test")
	dataset, _ := s.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	job, err := s.CreateJob(project.ID, jobs.TemplateProfileDataset, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}

	queued, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageQueued, Status: jobs.ProgressStatusQueued,
		Revision: 0, Metadata: map[string]any{"reason": "created"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if queued.ProjectID != project.ID || queued.JobID != job.ID || queued.Attempt != 1 {
		t.Fatalf("snapshot identity=%#v", queued)
	}

	stale, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning,
		Revision: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stale.Stage != jobs.ProgressStageQueued || !stale.UpdatedAt.Equal(queued.UpdatedAt) {
		t.Fatalf("same revision changed snapshot: before=%#v after=%#v", queued, stale)
	}
	worker, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning,
		Revision: 100,
	})
	if err != nil || worker.Revision != 100 {
		t.Fatalf("worker revision snapshot=%#v err=%v", worker, err)
	}

	completed, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageCompleted, Status: jobs.ProgressStatusCompleted,
		Revision: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if completed.Revision != 101 {
		t.Fatalf("authoritative terminal revision did not advance monotonically: %#v", completed)
	}
	regressed, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageTraining, Status: jobs.ProgressStatusRunning,
		Revision: 102,
	})
	if err != nil {
		t.Fatal(err)
	}
	if regressed.Stage != jobs.ProgressStageCompleted || regressed.Revision != completed.Revision {
		t.Fatalf("terminal snapshot regressed: %#v", regressed)
	}

	next, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 2, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageQueued, Status: jobs.ProgressStatusQueued,
		Revision: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if next.Attempt != 2 || next.Stage != jobs.ProgressStageQueued {
		t.Fatalf("fresh attempt snapshot=%#v", next)
	}
	prior, err := s.GetJobProgress(job.ID, 1)
	if err != nil || prior.Stage != jobs.ProgressStageCompleted {
		t.Fatalf("prior attempt was not preserved: %#v err=%v", prior, err)
	}
}

func TestMemoryJobProgressUsesServerTimeAndCopiesMetadata(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("progress", "clock")
	dataset, _ := s.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	job, err := s.CreateJob(project.ID, jobs.TemplateProfileDataset, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	metadata := map[string]any{"labels": []string{"one"}}
	before := time.Now().UTC()
	stored, err := s.UpsertJobProgress(job.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageWorkerStarting, Status: jobs.ProgressStatusRunning,
		Revision: 2, Metadata: metadata,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stored.HeartbeatAt.Before(before) || !stored.HeartbeatAt.Equal(stored.UpdatedAt) {
		t.Fatalf("unexpected receipt timestamps: %#v", stored)
	}
	metadata["labels"].([]string)[0] = "mutated"
	stored.Metadata["labels"].([]any)[0] = "also-mutated"
	got, err := s.GetJobProgress(job.ID, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata["labels"].([]any)[0] != "one" {
		t.Fatalf("stored metadata was caller-mutable: %#v", got.Metadata)
	}
	if _, err := s.GetJobProgress(job.ID, -1); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("negative attempt err=%v", err)
	}
}
