package store

import (
	"context"
	"sort"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/workers"
)

const ProjectLiveStateProgressLimit = 8

type LiveStateJobCounts struct {
	Total     int `json:"total"`
	Queued    int `json:"queued"`
	Retrying  int `json:"retrying"`
	Assigned  int `json:"assigned"`
	Running   int `json:"running"`
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
	Cancelled int `json:"cancelled"`
}

type LiveStateWorkerCounts struct {
	Total   int `json:"total"`
	Idle    int `json:"idle"`
	Running int `json:"running"`
	Offline int `json:"offline"`
	Stale   int `json:"stale"`
}

type LiveStateRequirementCounts struct {
	Pending   int `json:"pending"`
	Starting  int `json:"starting"`
	Active    int `json:"active"`
	Satisfied int `json:"satisfied"`
	Failed    int `json:"failed"`
	Cancelled int `json:"cancelled"`
}

// LiveStateActiveProgress contains only the current lifecycle-owned attempt.
// Job configuration is used by the store to select that attempt but is never
// included in the API projection.
type LiveStateActiveProgress struct {
	Progress          jobs.JobProgress
	JobStatus         string
	JobCreatedAt      time.Time
	WorkerHeartbeatAt *time.Time
	LeaseHeartbeatAt  *time.Time
}

// ProjectLiveStateSnapshot is the bounded, race-safe store projection used by
// Mission Control. It intentionally excludes plans, evaluations, histories,
// job configurations, errors, storage locations, and arbitrary metadata.
type ProjectLiveStateSnapshot struct {
	ProjectID           string
	ObservedAt          time.Time
	SnapshotCursor      int64
	Jobs                LiveStateJobCounts
	Workers             LiveStateWorkerCounts
	Requirements        LiveStateRequirementCounts
	LastWorkerHeartbeat *time.Time
	ActiveProgress      []LiveStateActiveProgress
	ActiveProgressTotal int
	LatestEvent         *execution.ExecutionEvent
}

func (s *MemoryStore) GetProjectLiveState(ctx context.Context, projectID string) (ProjectLiveStateSnapshot, error) {
	if err := ctx.Err(); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	if _, ok := s.projects[projectID]; !ok {
		return ProjectLiveStateSnapshot{}, ErrNotFound
	}

	now := time.Now().UTC()
	snapshot := ProjectLiveStateSnapshot{
		ProjectID:      projectID,
		ObservedAt:     now,
		SnapshotCursor: s.nextExecutionEventSequence,
	}

	candidates := make([]liveStateMemoryCandidate, 0, ProjectLiveStateProgressLimit)
	for _, job := range s.jobs {
		if job.ProjectID != projectID {
			continue
		}
		progress, hasProgress := s.jobProgress[jobProgressKey(job.ID, activeJobProgressAttempt(job))]
		cancelled := hasProgress && progress.Stage == jobs.ProgressStageCancelled
		addLiveStateJobCount(&snapshot.Jobs, job, cancelled)
		if !liveStateJobIsOpen(job) || !hasProgress {
			continue
		}
		snapshot.ActiveProgressTotal++
		candidate := liveStateMemoryCandidate{job: job, progress: cloneJobProgress(progress)}
		workerID := job.WorkerID
		if workerID == "" {
			workerID = job.LeaseOwnerWorkerID
		}
		if worker, ok := s.workers[workerID]; ok {
			heartbeat := worker.LastHeartbeat.UTC()
			candidate.workerHeartbeat = &heartbeat
		}
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool {
		leftPriority := liveStateJobPriority(candidates[i].job)
		rightPriority := liveStateJobPriority(candidates[j].job)
		if leftPriority != rightPriority {
			return leftPriority < rightPriority
		}
		if !candidates[i].job.CreatedAt.Equal(candidates[j].job.CreatedAt) {
			return candidates[i].job.CreatedAt.Before(candidates[j].job.CreatedAt)
		}
		return candidates[i].job.ID < candidates[j].job.ID
	})
	if len(candidates) > ProjectLiveStateProgressLimit {
		candidates = candidates[:ProjectLiveStateProgressLimit]
	}
	for _, candidate := range candidates {
		snapshot.ActiveProgress = append(snapshot.ActiveProgress, LiveStateActiveProgress{
			Progress:          candidate.progress,
			JobStatus:         candidate.job.Status,
			JobCreatedAt:      candidate.job.CreatedAt,
			WorkerHeartbeatAt: cloneTimePointer(candidate.workerHeartbeat),
			LeaseHeartbeatAt:  cloneTimePointer(candidate.job.LeaseLastHeartbeatAt),
		})
	}

	for _, worker := range s.workers {
		if worker.ProjectID != projectID {
			continue
		}
		addLiveStateWorkerCount(&snapshot.Workers, worker, now)
		if snapshot.LastWorkerHeartbeat == nil || worker.LastHeartbeat.After(*snapshot.LastWorkerHeartbeat) {
			heartbeat := worker.LastHeartbeat.UTC()
			snapshot.LastWorkerHeartbeat = &heartbeat
		}
	}
	for _, requirement := range s.workerRequirements {
		if requirement.ProjectID == projectID {
			addLiveStateRequirementCount(&snapshot.Requirements, requirement.Status)
		}
	}
	for _, event := range s.executionEvents {
		if event.ProjectID != projectID || (snapshot.LatestEvent != nil && event.Sequence <= snapshot.LatestEvent.Sequence) {
			continue
		}
		projected := event
		projected.Message = boundedExecutionEventProjectionText(event.Message, 512)
		projected.Payload = executionEventStreamPayloadProjection(event.Payload)
		projected.IdempotencyKey = ""
		snapshot.LatestEvent = &projected
	}
	return snapshot, nil
}

type liveStateMemoryCandidate struct {
	job             jobs.ExperimentJob
	progress        jobs.JobProgress
	workerHeartbeat *time.Time
}

func liveStateJobIsOpen(job jobs.ExperimentJob) bool {
	return job.Status == jobs.StatusQueued || job.Status == jobs.StatusAssigned || job.Status == jobs.StatusRunning
}

func liveStateJobPriority(job jobs.ExperimentJob) int {
	switch job.Status {
	case jobs.StatusRunning:
		return 0
	case jobs.StatusAssigned:
		return 1
	case jobs.StatusQueued:
		if job.Attempt > 0 {
			return 2
		}
		return 3
	default:
		return 4
	}
}

func addLiveStateJobCount(counts *LiveStateJobCounts, job jobs.ExperimentJob, cancelled bool) {
	counts.Total++
	switch job.Status {
	case jobs.StatusQueued:
		if job.Attempt > 0 {
			counts.Retrying++
		} else {
			counts.Queued++
		}
	case jobs.StatusAssigned:
		counts.Assigned++
	case jobs.StatusRunning:
		counts.Running++
	case jobs.StatusSucceeded:
		counts.Succeeded++
	case jobs.StatusFailed:
		if cancelled {
			counts.Cancelled++
		} else {
			counts.Failed++
		}
	}
}

func addLiveStateWorkerCount(counts *LiveStateWorkerCounts, worker workers.Worker, now time.Time) {
	counts.Total++
	switch worker.Status {
	case workers.StatusIdle:
		counts.Idle++
	case workers.StatusRunning:
		counts.Running++
	case workers.StatusOffline:
		counts.Offline++
	}
	if worker.Status != workers.StatusOffline && now.Sub(worker.LastHeartbeat) > workers.HeartbeatLimit {
		counts.Stale++
	}
}

func addLiveStateRequirementCount(counts *LiveStateRequirementCounts, status string) {
	switch status {
	case execution.WorkerRequirementPending:
		counts.Pending++
	case execution.WorkerRequirementStarting:
		counts.Starting++
	case execution.WorkerRequirementActive:
		counts.Active++
	case execution.WorkerRequirementSatisfied:
		counts.Satisfied++
	case execution.WorkerRequirementFailed:
		counts.Failed++
	case execution.WorkerRequirementCancelled:
		counts.Cancelled++
	}
}

func cloneTimePointer(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	cloned := value.UTC()
	return &cloned
}
