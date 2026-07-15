package store

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/execution"
)

func executionSpecFromConfig(jobID, projectID string, config map[string]any, createdAt time.Time) (execution.JobExecutionSpec, bool) {
	payload, ok := config[execution.ExecutionSpecConfigKey].(map[string]any)
	if !ok || payload["schema_version"] != execution.ExecutionSpecSchemaVersionV1 {
		return execution.JobExecutionSpec{}, false
	}
	accepted, ok := payload["accepted_config"].(map[string]any)
	if !ok {
		return execution.JobExecutionSpec{}, false
	}
	var artifactPlan execution.ArtifactPlanV1
	if artifactPayload, ok := payload["artifact_plan"].(map[string]any); ok {
		artifactBlob, err := json.Marshal(artifactPayload)
		if err != nil {
			return execution.JobExecutionSpec{}, false
		}
		if err := json.Unmarshal(artifactBlob, &artifactPlan); err != nil {
			return execution.JobExecutionSpec{}, false
		}
	}
	return execution.JobExecutionSpec{
		JobID: jobID, ProjectID: projectID,
		SchemaVersion:     stringValue(payload["schema_version"]),
		CapabilityVersion: stringValue(payload["capability_version"]),
		Task:              stringValue(payload["task"]), Runner: stringValue(payload["runner"]),
		RequestedConfigHash: stringValue(payload["requested_config_hash"]),
		AcceptedSpecHash:    stringValue(payload["accepted_spec_hash"]),
		AcceptedSpec:        cloneJSONMap(accepted), ArtifactPlan: artifactPlan,
		ArtifactPlanHash: artifactPlan.ArtifactPlanHash, CreatedAt: createdAt,
	}, true
}

func artifactPlanFromStoredJobConfig(config map[string]any) (execution.ArtifactPlanV1, bool, error) {
	var value any
	if spec, ok := config[execution.ExecutionSpecConfigKey].(map[string]any); ok {
		value = spec["artifact_plan"]
	}
	if value == nil {
		value = config[execution.ArtifactPlanConfigKey]
	}
	if value == nil {
		return execution.ArtifactPlanV1{}, false, nil
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return execution.ArtifactPlanV1{}, true, err
	}
	var plan execution.ArtifactPlanV1
	if err := json.Unmarshal(blob, &plan); err != nil {
		return execution.ArtifactPlanV1{}, true, err
	}
	return plan, true, nil
}

func artifactPlanExecutionIdentity(config map[string]any) (string, string) {
	task, runner := "", ""
	if spec, ok := config[execution.ExecutionSpecConfigKey].(map[string]any); ok {
		task = stringValue(spec["task"])
		runner = stringValue(spec["runner"])
	}
	if task == "" {
		task = stringValue(config["task_type"])
	}
	if task == "" {
		task = stringValue(config["task"])
	}
	if runner == "" {
		runner = stringValue(config["runner"])
	}
	if task == "" {
		task = "image_classification"
	}
	if runner == "" {
		runner = "modal_torchvision"
	}
	return task, runner
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func copyExecutionSpec(spec execution.JobExecutionSpec) execution.JobExecutionSpec {
	spec.AcceptedSpec = cloneJSONMap(spec.AcceptedSpec)
	spec.ArtifactPlan.Artifacts = append([]execution.ArtifactDirective(nil), spec.ArtifactPlan.Artifacts...)
	for index := range spec.ArtifactPlan.Artifacts {
		spec.ArtifactPlan.Artifacts[index].ExecutionRequirements = append([]string(nil), spec.ArtifactPlan.Artifacts[index].ExecutionRequirements...)
	}
	spec.ArtifactPlan.FallbackFormats = append([]string(nil), spec.ArtifactPlan.FallbackFormats...)
	spec.ArtifactPlan.RequiredWorkerCapabilities.PolicyContractVersions = append([]string(nil), spec.ArtifactPlan.RequiredWorkerCapabilities.PolicyContractVersions...)
	spec.ArtifactPlan.RequiredWorkerCapabilities.ArtifactPlanVersions = append([]string(nil), spec.ArtifactPlan.RequiredWorkerCapabilities.ArtifactPlanVersions...)
	return spec
}

func copyAttemptRecord(record execution.AttemptExecutionRecord) execution.AttemptExecutionRecord {
	record.LatestRealizedConfig = cloneJSONMap(record.LatestRealizedConfig)
	record.AdjustmentReasonCodes = append([]string(nil), record.AdjustmentReasonCodes...)
	if record.FidelityVerdict != nil {
		value := *record.FidelityVerdict
		record.FidelityVerdict = &value
	}
	record.Observations = append([]execution.RealizationObservation(nil), record.Observations...)
	for index := range record.Observations {
		record.Observations[index].RealizedConfig = cloneJSONMap(record.Observations[index].RealizedConfig)
		record.Observations[index].FrameworkArguments = cloneJSONMap(record.Observations[index].FrameworkArguments)
		record.Observations[index].Evidence = cloneJSONMap(record.Observations[index].Evidence)
		record.Observations[index].AdjustmentReasonCodes = append(
			[]string(nil),
			record.Observations[index].AdjustmentReasonCodes...,
		)
	}
	return record
}

func (s *MemoryStore) GetJobExecutionRecord(jobID string) (execution.ExecutionRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	spec, ok := s.jobExecutionSpecs[jobID]
	if !ok {
		return execution.ExecutionRecord{}, ErrNotFound
	}
	attempts := []execution.AttemptExecutionRecord{}
	for _, record := range s.attemptExecutions {
		if record.JobID == jobID {
			record.Observations = append([]execution.RealizationObservation(nil), s.realizationObservations[record.ID]...)
			attempts = append(attempts, copyAttemptRecord(record))
		}
	}
	sortAttemptRecords(attempts)
	return execution.ExecutionRecord{AcceptedSpec: copyExecutionSpec(spec), Attempts: attempts}, nil
}

func (s *MemoryStore) ListProjectExecutionRecords(projectID string, options PageOptions) ([]execution.ExecutionRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}
	specs := []execution.JobExecutionSpec{}
	for _, spec := range s.jobExecutionSpecs {
		if spec.ProjectID == projectID {
			specs = append(specs, spec)
		}
	}
	sortExecutionSpecs(specs)
	start, end := pageBounds(len(specs), options)
	out := make([]execution.ExecutionRecord, 0, end-start)
	for _, spec := range specs[start:end] {
		attempts := []execution.AttemptExecutionRecord{}
		for _, record := range s.attemptExecutions {
			if record.JobID == spec.JobID {
				record.Observations = append([]execution.RealizationObservation(nil), s.realizationObservations[record.ID]...)
				attempts = append(attempts, copyAttemptRecord(record))
			}
		}
		sortAttemptRecords(attempts)
		out = append(out, execution.ExecutionRecord{AcceptedSpec: copyExecutionSpec(spec), Attempts: attempts})
	}
	return out, nil
}

func (s *MemoryStore) CreateAttemptExecutionRecord(jobID, attemptID string, attemptNumber int) (execution.AttemptExecutionRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.createAttemptExecutionRecordLocked(jobID, attemptID, attemptNumber)
}

func (s *MemoryStore) createAttemptExecutionRecordLocked(jobID, attemptID string, attemptNumber int) (execution.AttemptExecutionRecord, error) {
	spec, ok := s.jobExecutionSpecs[jobID]
	if !ok {
		return execution.AttemptExecutionRecord{}, ErrNotFound
	}
	for _, record := range s.attemptExecutions {
		if record.JobID == jobID && (record.AttemptID == attemptID || record.AttemptNumber == attemptNumber) {
			return copyAttemptRecord(record), nil
		}
	}
	now := time.Now().UTC()
	record := execution.AttemptExecutionRecord{ID: s.newID("attempt_execution"), JobID: jobID, ProjectID: spec.ProjectID, AttemptID: attemptID, AttemptNumber: attemptNumber, LifecycleStatus: execution.ExecutionLifecyclePending, CreatedAt: now, UpdatedAt: now}
	s.attemptExecutions[record.ID] = record
	return copyAttemptRecord(record), nil
}

func (s *MemoryStore) AppendRealizationObservation(jobID string, create execution.RealizationObservationCreate) (execution.RealizationObservation, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := execution.ValidateObservation(create); err != nil {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	create.FrameworkArguments = execution.RedactSensitiveMap(create.FrameworkArguments)
	create.Evidence = execution.RedactSensitiveMap(create.Evidence)
	spec, ok := s.jobExecutionSpecs[jobID]
	if !ok {
		return execution.RealizationObservation{}, false, ErrNotFound
	}
	var record execution.AttemptExecutionRecord
	found := false
	for _, candidate := range s.attemptExecutions {
		if candidate.JobID == jobID && candidate.AttemptID == create.AttemptID {
			record = candidate
			found = true
			break
		}
	}
	if !found {
		return execution.RealizationObservation{}, false, ErrNotFound
	}
	for _, existing := range s.realizationObservations[record.ID] {
		if existing.IdempotencyKey == create.IdempotencyKey {
			return existing, false, nil
		}
	}
	if record.LifecycleStatus == execution.ExecutionLifecycleFinalized {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: attempt realization is already finalized", ErrInvalidRequest)
	}
	realized, hash, verdict, adjustmentReasonCodes, err := execution.DeriveRealization(spec, create)
	if err != nil {
		return execution.RealizationObservation{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	now := time.Now().UTC()
	stage := strings.ToUpper(strings.TrimSpace(create.Stage))
	lifecycle := execution.ExecutionLifecycleInitialized
	if stage == execution.ExecutionObservationFinalized {
		lifecycle = execution.ExecutionLifecycleFinalized
	}
	observation := execution.RealizationObservation{ID: s.newID("realization"), AttemptRecordID: record.ID, AttemptID: create.AttemptID, SchemaVersion: execution.ExecutionObservationSchemaV1, Stage: stage, IdempotencyKey: create.IdempotencyKey, RealizedConfig: cloneJSONMap(realized), FrameworkArguments: cloneJSONMap(create.FrameworkArguments), Evidence: cloneJSONMap(create.Evidence), AdjustmentPolicy: create.AdjustmentPolicy, AdjustmentReasonCodes: append([]string(nil), adjustmentReasonCodes...), Simulated: create.Simulated, RealizedEffectiveHash: hash, FidelityVerdict: verdict, CreatedAt: now}
	s.realizationObservations[record.ID] = append(s.realizationObservations[record.ID], observation)
	verdictCopy := verdict
	record.LifecycleStatus = lifecycle
	record.FidelityVerdict = &verdictCopy
	record.RealizedEffectiveHash = hash
	record.AdjustmentReasonCodes = append([]string(nil), adjustmentReasonCodes...)
	record.LatestRealizedConfig = cloneJSONMap(realized)
	record.UpdatedAt = now
	s.attemptExecutions[record.ID] = record
	return observation, true, nil
}

func (s *MemoryStore) MarkAttemptNotRealized(jobID, attemptID string) (execution.AttemptExecutionRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.markAttemptNotRealizedLocked(jobID, attemptID)
}

func (s *MemoryStore) markAttemptNotRealizedLocked(jobID, attemptID string) (execution.AttemptExecutionRecord, error) {
	for id, record := range s.attemptExecutions {
		if record.JobID == jobID && record.AttemptID == attemptID {
			if record.LifecycleStatus == execution.ExecutionLifecyclePending {
				record.LifecycleStatus = execution.ExecutionLifecycleNotRealized
				record.UpdatedAt = time.Now().UTC()
				s.attemptExecutions[id] = record
			}
			return copyAttemptRecord(record), nil
		}
	}
	return execution.AttemptExecutionRecord{}, ErrNotFound
}

func sortAttemptRecords(values []execution.AttemptExecutionRecord) {
	sort.Slice(values, func(i, j int) bool { return values[i].AttemptNumber < values[j].AttemptNumber })
}

func sortExecutionSpecs(values []execution.JobExecutionSpec) {
	sort.Slice(values, func(i, j int) bool { return values[i].CreatedAt.After(values[j].CreatedAt) })
}

func jsonMap(value map[string]any) ([]byte, error) {
	if value == nil {
		value = map[string]any{}
	}
	return json.Marshal(value)
}

func cloneJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil
	}
	return out
}
