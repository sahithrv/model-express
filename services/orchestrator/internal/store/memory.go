package store

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/workers"
)

var (
	ErrNotFound = errors.New("not found")
	ErrNoJob    = errors.New("no job available")
)

type MemoryStore struct {
	mu sync.Mutex

	nextID   uint64
	projects map[string]projects.Project
	workers  map[string]workers.Worker
	jobs     map[string]jobs.ExperimentJob
	metrics  map[string][]jobs.EpochMetric
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		projects: make(map[string]projects.Project),
		workers:  make(map[string]workers.Worker),
		jobs:     make(map[string]jobs.ExperimentJob),
		metrics:  make(map[string][]jobs.EpochMetric),
	}
}

func (s *MemoryStore) CreateProject(name string, goal string) projects.Project {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC()
	project := projects.Project{
		ID:        s.newID("project"),
		Name:      name,
		Goal:      goal,
		Status:    projects.StatusCreated,
		CreatedAt: now,
		UpdatedAt: now,
	}

	s.projects[project.ID] = project
	return project
}

func (s *MemoryStore) GetProject(id string) (projects.Project, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	project, ok := s.projects[id]
	if !ok {
		return projects.Project{}, ErrNotFound
	}

	return project, nil
}

func (s *MemoryStore) ListProjects() []projects.Project {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]projects.Project, 0, len(s.projects))
	for _, project := range s.projects {
		out = append(out, project)
	}

	return out
}

func (s *MemoryStore) RegisterWorker(name string, gpuType string) workers.Worker {
	s.mu.Lock()
	defer s.mu.Unlock()

	worker := workers.Worker{
		ID:            s.newID("worker"),
		Name:          name,
		Status:        workers.StatusIdle,
		GPUType:       gpuType,
		LastHeartbeat: time.Now().UTC(),
	}

	s.workers[worker.ID] = worker
	return worker
}

func (s *MemoryStore) HeartbeatWorker(id string) (workers.Worker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	worker, ok := s.workers[id]
	if !ok {
		return workers.Worker{}, ErrNotFound
	}

	worker.LastHeartbeat = time.Now().UTC()
	s.workers[id] = worker

	return worker, nil
}

func (s *MemoryStore) PollJob(workerID string) (*jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	worker, ok := s.workers[workerID]
	if !ok {
		return nil, ErrNotFound
	}

	if worker.CurrentJobID != "" {
		job := s.jobs[worker.CurrentJobID]
		return &job, nil
	}

	for id, job := range s.jobs {
		if job.Status != jobs.StatusQueued {
			continue
		}

		now := time.Now().UTC()
		job.WorkerID = workerID
		job.Status = jobs.StatusAssigned
		job.StartedAt = &now
		s.jobs[id] = job

		worker.Status = workers.StatusRunning
		worker.CurrentJobID = job.ID
		worker.LastHeartbeat = now
		s.workers[workerID] = worker

		return &job, nil
	}

	worker.LastHeartbeat = time.Now().UTC()
	s.workers[workerID] = worker

	return nil, ErrNoJob
}

func (s *MemoryStore) CreateJob(projectID string, template string, config map[string]any) (jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return jobs.ExperimentJob{}, ErrNotFound
	}

	job := jobs.ExperimentJob{
		ID:        s.newID("job"),
		ProjectID: projectID,
		Template:  template,
		Status:    jobs.StatusQueued,
		Config:    config,
		CreatedAt: time.Now().UTC(),
	}

	s.jobs[job.ID] = job
	return job, nil
}

func (s *MemoryStore) GetJob(id string) (jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[id]
	if !ok {
		return jobs.ExperimentJob{}, ErrNotFound
	}

	return job, nil
}

func (s *MemoryStore) ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[jobID]
	if !ok {
		return jobs.EpochMetric{}, ErrNotFound
	}

	if job.Status == jobs.StatusAssigned {
		job.Status = jobs.StatusRunning
		s.jobs[jobID] = job
	}

	metric := jobs.EpochMetric{
		JobID:     jobID,
		Epoch:     epoch,
		Metrics:   values,
		CreatedAt: time.Now().UTC(),
	}

	s.metrics[jobID] = append(s.metrics[jobID], metric)
	return metric, nil
}

func (s *MemoryStore) CompleteJob(jobID string, mlflowRunID string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusSucceeded, mlflowRunID, "")
}

func (s *MemoryStore) FailJob(jobID string, message string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusFailed, "", message)
}

func (s *MemoryStore) finishJob(jobID string, status string, mlflowRunID string, message string) (jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[jobID]
	if !ok {
		return jobs.ExperimentJob{}, ErrNotFound
	}

	now := time.Now().UTC()
	job.Status = status
	job.MLflowRunID = mlflowRunID
	job.Error = message
	job.CompletedAt = &now
	s.jobs[jobID] = job

	if job.WorkerID != "" {
		if worker, ok := s.workers[job.WorkerID]; ok {
			worker.Status = workers.StatusIdle
			worker.CurrentJobID = ""
			worker.LastHeartbeat = now
			s.workers[worker.ID] = worker
		}
	}

	return job, nil
}

func (s *MemoryStore) newID(prefix string) string {
	s.nextID++
	return fmt.Sprintf("%s_%d", prefix, s.nextID)
}
