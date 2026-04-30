package store

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/workers"
)

type MemoryStore struct {
	mu sync.Mutex

	nextID             uint64
	projects           map[string]projects.Project
	datasets           map[string]datasets.Dataset
	workers            map[string]workers.Worker
	jobs               map[string]jobs.ExperimentJob
	metrics            map[string][]jobs.EpochMetric
	plans              map[string]plans.ExperimentPlan
	summaries          map[string]runs.TrainingRunSummary
	decisions          map[string]decisions.AgentDecision
	workerRequirements map[string]execution.WorkerRequirement
	executionEvents    map[string]execution.ExecutionEvent
	agentMemoryRecords map[string]memory.AgentMemoryRecord
	agentInvocations   map[string]memory.AgentInvocation
	automationSettings *settings.AutomationSettings
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		projects:           make(map[string]projects.Project),
		datasets:           make(map[string]datasets.Dataset),
		workers:            make(map[string]workers.Worker),
		jobs:               make(map[string]jobs.ExperimentJob),
		metrics:            make(map[string][]jobs.EpochMetric),
		plans:              make(map[string]plans.ExperimentPlan),
		summaries:          make(map[string]runs.TrainingRunSummary),
		decisions:          make(map[string]decisions.AgentDecision),
		workerRequirements: make(map[string]execution.WorkerRequirement),
		executionEvents:    make(map[string]execution.ExecutionEvent),
		agentMemoryRecords: make(map[string]memory.AgentMemoryRecord),
		agentInvocations:   make(map[string]memory.AgentInvocation),
	}
}

func (s *MemoryStore) CreateProject(name string, goal string) (projects.Project, error) {
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
	return project, nil
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

func (s *MemoryStore) ListProjects() ([]projects.Project, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]projects.Project, 0, len(s.projects))
	for _, project := range s.projects {
		out = append(out, project)
	}

	return out, nil
}

func (s *MemoryStore) CreateDataset(projectID string, name string, storageURI string, checksumSHA256 string, sizeBytes int64) (datasets.Dataset, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return datasets.Dataset{}, ErrNotFound
	}

	dataset := datasets.Dataset{
		ID:             s.newID("dataset"),
		ProjectID:      projectID,
		Name:           name,
		StorageURI:     storageURI,
		ChecksumSHA256: checksumSHA256,
		SizeBytes:      sizeBytes,
		Profile:        map[string]any{},
		Status:         datasets.StatusRegistered,
		CreatedAt:      time.Now().UTC(),
	}

	s.datasets[dataset.ID] = dataset
	return dataset, nil
}

func (s *MemoryStore) GetDataset(id string) (datasets.Dataset, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dataset, ok := s.datasets[id]
	if !ok {
		return datasets.Dataset{}, ErrNotFound
	}

	return dataset, nil
}

func (s *MemoryStore) ListProjectDatasets(projectID string) ([]datasets.Dataset, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []datasets.Dataset{}
	for _, dataset := range s.datasets {
		if dataset.ProjectID == projectID {
			out = append(out, dataset)
		}
	}

	return out, nil
}

func (s *MemoryStore) UpdateDatasetProfile(id string, profile map[string]any) (datasets.Dataset, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dataset, ok := s.datasets[id]
	if !ok {
		return datasets.Dataset{}, ErrNotFound
	}

	now := time.Now().UTC()
	dataset.Profile = profile
	dataset.Status = datasets.StatusProfiled
	dataset.ProfiledAt = &now
	s.datasets[id] = dataset

	return dataset, nil
}

func (s *MemoryStore) RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return workers.Worker{}, ErrNotFound
	}
	if !s.projectHasDataset(projectID) {
		return workers.Worker{}, fmt.Errorf("%w: project must have a dataset before workers or jobs can be created", ErrInvalidRequest)
	}

	worker := workers.Worker{
		ID:            s.newID("worker"),
		ProjectID:     projectID,
		Name:          name,
		Status:        workers.StatusIdle,
		GPUType:       gpuType,
		LastHeartbeat: time.Now().UTC(),
	}

	s.workers[worker.ID] = worker
	return worker, nil
}

func (s *MemoryStore) ListWorkers() ([]workers.Worker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]workers.Worker, 0, len(s.workers))
	for _, worker := range s.workers {
		out = append(out, worker)
	}

	return out, nil
}

func (s *MemoryStore) ListProjectWorkers(projectID string) ([]workers.Worker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []workers.Worker{}
	for _, worker := range s.workers {
		if worker.ProjectID == projectID {
			out = append(out, worker)
		}
	}

	return out, nil
}

func (s *MemoryStore) GetWorker(workerID string) (workers.Worker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	worker, ok := s.workers[workerID]
	if !ok {
		return workers.Worker{}, ErrNotFound
	}

	return worker, nil
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
		if job.ProjectID != worker.ProjectID {
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
	if err := s.requireDatasetConfig(projectID, config); err != nil {
		return jobs.ExperimentJob{}, err
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

func (s *MemoryStore) ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []jobs.ExperimentJob{}
	for _, job := range s.jobs {
		if job.ProjectID == projectID {
			out = append(out, job)
		}
	}

	return out, nil
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

func (s *MemoryStore) ListJobMetrics(jobID string) ([]jobs.EpochMetric, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.jobs[jobID]; !ok {
		return nil, ErrNotFound
	}

	out := append([]jobs.EpochMetric(nil), s.metrics[jobID]...)
	return out, nil
}

func (s *MemoryStore) UpsertTrainingRunSummary(jobID string, update runs.TrainingRunSummaryUpdate) (runs.TrainingRunSummary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[jobID]
	if !ok {
		return runs.TrainingRunSummary{}, ErrNotFound
	}

	now := time.Now().UTC()
	summary, ok := s.summaries[jobID]
	if !ok {
		summary = newTrainingRunSummaryFromJob(job, now)
	}

	applyTrainingRunSummaryUpdate(&summary, update, now)
	s.summaries[jobID] = summary
	return summary, nil
}

func (s *MemoryStore) GetTrainingRunSummary(jobID string) (runs.TrainingRunSummary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	summary, ok := s.summaries[jobID]
	if !ok {
		return runs.TrainingRunSummary{}, ErrNotFound
	}

	return summary, nil
}

func (s *MemoryStore) ListProjectTrainingRunSummaries(projectID string) ([]runs.TrainingRunSummary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []runs.TrainingRunSummary{}
	for _, summary := range s.summaries {
		if summary.ProjectID == projectID {
			out = append(out, summary)
		}
	}

	return out, nil
}

func (s *MemoryStore) CreateAgentDecision(projectID string, planID string, decisionType string, rationale string, payload map[string]any) (decisions.AgentDecision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return decisions.AgentDecision{}, ErrNotFound
	}
	if payload == nil {
		payload = map[string]any{}
	}

	decision := decisions.AgentDecision{
		ID:           s.newID("decision"),
		ProjectID:    projectID,
		PlanID:       planID,
		DecisionType: decisionType,
		Rationale:    rationale,
		Payload:      payload,
		CreatedAt:    time.Now().UTC(),
	}

	s.decisions[decision.ID] = decision
	return decision, nil
}

func (s *MemoryStore) ListProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []decisions.AgentDecision{}
	for _, decision := range s.decisions {
		if decision.ProjectID == projectID {
			out = append(out, decision)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})

	return out, nil
}

func (s *MemoryStore) GetAutomationSettings() (settings.AutomationSettings, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.automationSettings == nil {
		return settings.AutomationSettings{}, ErrNotFound
	}

	return *s.automationSettings, nil
}

func (s *MemoryStore) SaveAutomationSettings(automationSettings settings.AutomationSettings) (settings.AutomationSettings, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	automationSettings.UpdatedAt = time.Now().UTC()
	s.automationSettings = &automationSettings

	return automationSettings, nil
}

func (s *MemoryStore) UpsertWorkerRequirement(projectID string, planID string, provider string, gpuType string, targetCount int, source string) (execution.WorkerRequirement, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return execution.WorkerRequirement{}, false, ErrNotFound
	}
	if targetCount < 1 {
		return execution.WorkerRequirement{}, false, fmt.Errorf("%w: target_count must be at least 1", ErrInvalidRequest)
	}

	now := time.Now().UTC()
	for id, requirement := range s.workerRequirements {
		if requirement.ProjectID == projectID && requirement.PlanID == planID {
			requirement.Provider = provider
			requirement.GPUType = gpuType
			requirement.TargetCount = targetCount
			requirement.Source = source
			requirement.UpdatedAt = now
			s.workerRequirements[id] = requirement
			return requirement, false, nil
		}
	}

	requirement := execution.WorkerRequirement{
		ID:          s.newID("worker_requirement"),
		ProjectID:   projectID,
		PlanID:      planID,
		Provider:    provider,
		GPUType:     gpuType,
		TargetCount: targetCount,
		Status:      execution.WorkerRequirementPending,
		Source:      source,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	s.workerRequirements[requirement.ID] = requirement
	return requirement, true, nil
}

func (s *MemoryStore) ListProjectWorkerRequirements(projectID string) ([]execution.WorkerRequirement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []execution.WorkerRequirement{}
	for _, requirement := range s.workerRequirements {
		if requirement.ProjectID == projectID {
			out = append(out, requirement)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

func (s *MemoryStore) UpdateWorkerRequirement(id string, update execution.WorkerRequirementUpdate) (execution.WorkerRequirement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	requirement, ok := s.workerRequirements[id]
	if !ok {
		return execution.WorkerRequirement{}, ErrNotFound
	}
	if update.Status != nil {
		requirement.Status = *update.Status
	}
	if update.LastError != nil {
		requirement.LastError = *update.LastError
	}
	requirement.UpdatedAt = time.Now().UTC()
	s.workerRequirements[id] = requirement
	return requirement, nil
}

func (s *MemoryStore) CreateExecutionEvent(projectID string, planID string, eventType string, message string, payload map[string]any) (execution.ExecutionEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return execution.ExecutionEvent{}, ErrNotFound
	}
	if payload == nil {
		payload = map[string]any{}
	}

	event := execution.ExecutionEvent{
		ID:        s.newID("execution_event"),
		ProjectID: projectID,
		PlanID:    planID,
		EventType: eventType,
		Message:   message,
		Payload:   payload,
		CreatedAt: time.Now().UTC(),
	}
	s.executionEvents[event.ID] = event
	return event, nil
}

func (s *MemoryStore) ListProjectExecutionEvents(projectID string, limit int) ([]execution.ExecutionEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}
	if limit <= 0 {
		limit = 50
	}

	out := []execution.ExecutionEvent{}
	for _, event := range s.executionEvents {
		if event.ProjectID == projectID {
			out = append(out, event)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func (s *MemoryStore) CreateAgentMemoryRecord(record memory.AgentMemoryRecord) (memory.AgentMemoryRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[record.ProjectID]; !ok {
		return memory.AgentMemoryRecord{}, ErrNotFound
	}
	if record.Payload == nil {
		record.Payload = map[string]any{}
	}
	if record.Tags == nil {
		record.Tags = []string{}
	}
	record.ID = s.newID("memory")
	record.CreatedAt = time.Now().UTC()
	s.agentMemoryRecords[record.ID] = record
	return record, nil
}

func (s *MemoryStore) ListProjectAgentMemoryRecords(projectID string, filter memory.AgentMemoryFilter) ([]memory.AgentMemoryRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	out := []memory.AgentMemoryRecord{}
	for _, record := range s.agentMemoryRecords {
		if record.ProjectID == projectID && memoryRecordMatchesFilter(record, filter) {
			out = append(out, record)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	if len(out) > filter.Limit {
		out = out[:filter.Limit]
	}
	return out, nil
}

func (s *MemoryStore) CreateAgentInvocation(invocation memory.AgentInvocation) (memory.AgentInvocation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[invocation.ProjectID]; !ok {
		return memory.AgentInvocation{}, ErrNotFound
	}
	if invocation.InputMessages == nil {
		invocation.InputMessages = []map[string]string{}
	}
	if invocation.InputContext == nil {
		invocation.InputContext = map[string]any{}
	}
	if invocation.ParsedOutput == nil {
		invocation.ParsedOutput = map[string]any{}
	}
	if invocation.HumanFeedback == nil {
		invocation.HumanFeedback = map[string]any{}
	}
	if invocation.DownstreamOutcome == nil {
		invocation.DownstreamOutcome = map[string]any{}
	}
	invocation.ID = s.newID("agent_invocation")
	invocation.CreatedAt = time.Now().UTC()
	s.agentInvocations[invocation.ID] = invocation
	return invocation, nil
}

func (s *MemoryStore) ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	out := []memory.AgentInvocation{}
	for _, invocation := range s.agentInvocations {
		if invocation.ProjectID == projectID && agentInvocationMatchesFilter(invocation, filter) {
			out = append(out, invocation)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	if len(out) > filter.Limit {
		out = out[:filter.Limit]
	}
	return out, nil
}

func (s *MemoryStore) CreateExperimentPlan(projectID string, datasetID string, targetMetric string, recommendedWorkers int, estimatedMinutes int, experiments []plans.PlannedExperiment, warnings []string, sourceDecisionID string) (plans.ExperimentPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return plans.ExperimentPlan{}, ErrNotFound
	}
	if err := s.requireDatasetBelongsToProject(projectID, datasetID); err != nil {
		return plans.ExperimentPlan{}, err
	}
	if targetMetric == "" {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: target_metric is required", ErrInvalidRequest)
	}
	if recommendedWorkers < 1 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: recommended_workers must be at least 1", ErrInvalidRequest)
	}
	if estimatedMinutes < 1 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: estimated_minutes must be at least 1", ErrInvalidRequest)
	}
	if len(experiments) == 0 {
		return plans.ExperimentPlan{}, fmt.Errorf("%w: at least one planned experiment is required", ErrInvalidRequest)
	}
	if warnings == nil {
		warnings = []string{}
	}

	plan := plans.ExperimentPlan{
		ID:                 s.newID("plan"),
		ProjectID:          projectID,
		DatasetID:          datasetID,
		Status:             plans.StatusProposed,
		SourceDecisionID:   sourceDecisionID,
		TargetMetric:       targetMetric,
		RecommendedWorkers: recommendedWorkers,
		EstimatedMinutes:   estimatedMinutes,
		Experiments:        append([]plans.PlannedExperiment(nil), experiments...),
		Warnings:           append([]string(nil), warnings...),
		CreatedAt:          time.Now().UTC(),
	}

	s.plans[plan.ID] = plan
	return plan, nil
}

func (s *MemoryStore) GetExperimentPlan(id string) (plans.ExperimentPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	plan, ok := s.plans[id]
	if !ok {
		return plans.ExperimentPlan{}, ErrNotFound
	}

	return plan, nil
}

func (s *MemoryStore) ListProjectExperimentPlans(projectID string) ([]plans.ExperimentPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}

	out := []plans.ExperimentPlan{}
	for _, plan := range s.plans {
		if plan.ProjectID == projectID {
			out = append(out, plan)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})

	return out, nil
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

func (s *MemoryStore) projectHasDataset(projectID string) bool {
	for _, dataset := range s.datasets {
		if dataset.ProjectID == projectID {
			return true
		}
	}

	return false
}

func (s *MemoryStore) requireDatasetConfig(projectID string, config map[string]any) error {
	value, ok := config["dataset_id"]
	if !ok {
		return fmt.Errorf("%w: job config must include dataset_id", ErrInvalidRequest)
	}

	datasetID, ok := value.(string)
	if !ok || datasetID == "" {
		return fmt.Errorf("%w: dataset_id must be a non-empty string", ErrInvalidRequest)
	}

	return s.requireDatasetBelongsToProject(projectID, datasetID)
}

func (s *MemoryStore) requireDatasetBelongsToProject(projectID string, datasetID string) error {
	if datasetID == "" {
		return fmt.Errorf("%w: dataset_id is required", ErrInvalidRequest)
	}

	dataset, ok := s.datasets[datasetID]
	if !ok || dataset.ProjectID != projectID {
		return fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}

	return nil
}

func newTrainingRunSummaryFromJob(job jobs.ExperimentJob, now time.Time) runs.TrainingRunSummary {
	provider := memoryConfigString(job.Config, "provider")
	if provider == "" {
		provider = "local"
	}

	return runs.TrainingRunSummary{
		JobID:     job.ID,
		ProjectID: job.ProjectID,
		PlanID:    memoryConfigString(job.Config, "plan_id"),
		DatasetID: memoryConfigString(job.Config, "dataset_id"),
		Model:     memoryConfigString(job.Config, "model"),
		Provider:  provider,
		GPUType:   memoryConfigString(job.Config, "gpu_type"),
		Status:    job.Status,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func applyTrainingRunSummaryUpdate(summary *runs.TrainingRunSummary, update runs.TrainingRunSummaryUpdate, now time.Time) {
	if update.Model != "" {
		summary.Model = update.Model
	}
	if update.Provider != "" {
		summary.Provider = update.Provider
	}
	if update.GPUType != "" {
		summary.GPUType = update.GPUType
	}
	if update.Status != "" {
		summary.Status = update.Status
	}
	if update.RuntimeSeconds != nil {
		summary.RuntimeSeconds = *update.RuntimeSeconds
	}
	if update.EstimatedCostUSD != nil {
		summary.EstimatedCostUSD = *update.EstimatedCostUSD
	}
	if update.BestMacroF1 != nil {
		summary.BestMacroF1 = *update.BestMacroF1
	}
	if update.BestAccuracy != nil {
		summary.BestAccuracy = *update.BestAccuracy
	}
	if update.FinalTrainLoss != nil {
		summary.FinalTrainLoss = *update.FinalTrainLoss
	}
	if update.FinalValLoss != nil {
		summary.FinalValLoss = *update.FinalValLoss
	}
	if update.EpochsCompleted != nil {
		summary.EpochsCompleted = *update.EpochsCompleted
	}
	if update.ModalFunctionCallID != "" {
		summary.ModalFunctionCallID = update.ModalFunctionCallID
	}
	if update.ModalInputID != "" {
		summary.ModalInputID = update.ModalInputID
	}

	summary.UpdatedAt = now
}

func memoryConfigString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

func memoryRecordMatchesFilter(record memory.AgentMemoryRecord, filter memory.AgentMemoryFilter) bool {
	if filter.DatasetID != "" && record.DatasetID != filter.DatasetID {
		return false
	}
	if filter.PlanID != "" && record.PlanID != filter.PlanID {
		return false
	}
	if filter.JobID != "" && record.JobID != filter.JobID {
		return false
	}
	if filter.Kind != "" && record.Kind != filter.Kind {
		return false
	}
	return true
}

func agentInvocationMatchesFilter(invocation memory.AgentInvocation, filter memory.AgentInvocationFilter) bool {
	if filter.DatasetID != "" && invocation.DatasetID != filter.DatasetID {
		return false
	}
	if filter.PlanID != "" && invocation.PlanID != filter.PlanID {
		return false
	}
	if filter.JobID != "" && invocation.JobID != filter.JobID {
		return false
	}
	if filter.AgentName != "" && invocation.AgentName != filter.AgentName {
		return false
	}
	return true
}
