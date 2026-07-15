package store

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/policies"
)

func (s *MemoryStore) CreateCompatibilityProfile(input policies.CompatibilityProfile) (policies.CompatibilityProfile, error) {
	if err := policies.ValidateProfileIdentity(input.ProfileKey, input.SemanticVersion); err != nil {
		return policies.CompatibilityProfile{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	document, canonical, hash, err := policies.NormalizeCompatibilityProfileDocument(input.Document)
	if err != nil {
		return policies.CompatibilityProfile{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	if err := policies.ValidateMetadata(document.Metadata); err != nil {
		return policies.CompatibilityProfile{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if input.OwnerAccountID != "" && !s.accountExistsLocked(input.OwnerAccountID) {
		return policies.CompatibilityProfile{}, ErrNotFound
	}
	key := strings.TrimSpace(input.ProfileKey) + "@" + strings.TrimSpace(input.SemanticVersion)
	if _, exists := s.policyProfiles[key]; exists {
		return policies.CompatibilityProfile{}, fmt.Errorf("%w: compatibility profile %s", policies.ErrImmutableConflict, key)
	}
	now := time.Now().UTC()
	created := policies.CompatibilityProfile{
		ID:              s.newID("compatibility_profile"),
		ProfileKey:      strings.TrimSpace(input.ProfileKey),
		SemanticVersion: strings.TrimSpace(input.SemanticVersion),
		SchemaVersion:   document.SchemaVersion,
		CatalogVersion:  document.CatalogVersion,
		Document:        document,
		CanonicalJSON:   canonical,
		DocumentHash:    hash,
		OwnerAccountID:  strings.TrimSpace(input.OwnerAccountID),
		CreatedAt:       now,
		CreatedBy:       defaultPolicyActor(input.CreatedBy),
	}
	s.policyProfiles[key] = policies.CloneCompatibilityProfile(created)
	return policies.CloneCompatibilityProfile(created), nil
}

func (s *MemoryStore) GetCompatibilityProfile(profileKey string, semanticVersion string) (policies.CompatibilityProfile, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile, ok := s.policyProfiles[strings.TrimSpace(profileKey)+"@"+strings.TrimSpace(semanticVersion)]
	if !ok {
		return policies.CompatibilityProfile{}, ErrNotFound
	}
	return policies.CloneCompatibilityProfile(profile), nil
}

func (s *MemoryStore) CreateExperimentPolicyVersion(input policies.PolicyVersion) (policies.PolicyVersion, error) {
	document, canonical, hash, err := policies.NormalizePolicyDocument(input.Document)
	if err != nil {
		return policies.PolicyVersion{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	if err := policies.ValidateMetadata(document.Metadata); err != nil {
		return policies.PolicyVersion{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	ownerAccountID := strings.TrimSpace(input.OwnerAccountID)
	if ownerAccountID == "" {
		ownerAccountID = policies.LocalDefaultAccountID
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.accountExistsLocked(ownerAccountID) {
		return policies.PolicyVersion{}, ErrNotFound
	}
	s.nextPolicyRevision++
	created := policies.PolicyVersion{
		ID:             s.newID("policy_version"),
		OwnerAccountID: ownerAccountID,
		SchemaVersion:  document.SchemaVersion,
		Revision:       s.nextPolicyRevision,
		Document:       document,
		CanonicalJSON:  canonical,
		DocumentHash:   hash,
		CreatedAt:      time.Now().UTC(),
		CreatedBy:      defaultPolicyActor(input.CreatedBy),
	}
	s.policyVersions[created.ID] = policies.ClonePolicyVersion(created)
	return policies.ClonePolicyVersion(created), nil
}

func (s *MemoryStore) GetExperimentPolicyVersion(id string) (policies.PolicyVersion, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	version, ok := s.policyVersions[strings.TrimSpace(id)]
	if !ok {
		return policies.PolicyVersion{}, ErrNotFound
	}
	return policies.ClonePolicyVersion(version), nil
}

func (s *MemoryStore) SetExperimentPolicyBinding(write policies.BindingWrite) (policies.Binding, error) {
	write.SubjectID = strings.TrimSpace(write.SubjectID)
	write.PolicyVersionID = strings.TrimSpace(write.PolicyVersionID)
	if write.SubjectID == "" || write.PolicyVersionID == "" || policies.ScopeRank(write.Scope) > policies.ScopeRank(policies.ScopeRun) {
		return policies.Binding{}, fmt.Errorf("%w: valid policy binding scope, subject, and version are required", ErrInvalidRequest)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	version, ok := s.policyVersions[write.PolicyVersionID]
	if !ok {
		return policies.Binding{}, ErrNotFound
	}
	accountID, err := s.policySubjectAccountLocked(write.Scope, write.SubjectID)
	if err != nil {
		return policies.Binding{}, err
	}
	if accountID != version.OwnerAccountID {
		return policies.Binding{}, fmt.Errorf("%w: policy version and binding subject have different account owners", ErrInvalidRequest)
	}

	var active *policies.Binding
	var maxRevision int64
	for id, candidate := range s.policyBindings {
		if candidate.Scope != write.Scope || candidate.SubjectID != write.SubjectID {
			continue
		}
		if candidate.Revision > maxRevision {
			maxRevision = candidate.Revision
		}
		if candidate.Active {
			copyCandidate := candidate
			copyCandidate.ID = id
			active = &copyCandidate
		}
	}
	currentRevision := int64(0)
	if active != nil {
		currentRevision = active.Revision
	}
	if write.ExpectedRevision != nil && *write.ExpectedRevision != currentRevision {
		return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision %d", policies.ErrRevisionConflict, *write.ExpectedRevision, currentRevision)
	}

	now := time.Now().UTC()
	supersedesID := ""
	if active != nil {
		supersedesID = active.ID
		active.Active = false
		active.SupersededAt = &now
		s.policyBindings[active.ID] = policies.CloneBinding(*active)
	}
	created := policies.Binding{
		ID:              s.newID("policy_binding"),
		Scope:           write.Scope,
		SubjectID:       write.SubjectID,
		PolicyVersionID: write.PolicyVersionID,
		Revision:        maxRevision + 1,
		Active:          true,
		SupersedesID:    supersedesID,
		CreatedAt:       now,
		CreatedBy:       defaultPolicyActor(write.CreatedBy),
	}
	s.policyBindings[created.ID] = policies.CloneBinding(created)
	s.markQueuedJobsPolicyPendingLocked(write.Scope, write.SubjectID)
	return policies.CloneBinding(created), nil
}

func (s *MemoryStore) ClearExperimentPolicyBinding(scope policies.Scope, subjectID string, expectedRevision int64) (policies.Binding, error) {
	subjectID = strings.TrimSpace(subjectID)
	if subjectID == "" || policies.ScopeRank(scope) > policies.ScopeRank(policies.ScopeRun) {
		return policies.Binding{}, fmt.Errorf("%w: valid policy binding scope and subject are required", ErrInvalidRequest)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.policySubjectAccountLocked(scope, subjectID); err != nil {
		return policies.Binding{}, err
	}
	for id, binding := range s.policyBindings {
		if binding.Scope != scope || binding.SubjectID != subjectID || !binding.Active {
			continue
		}
		if binding.Revision != expectedRevision {
			return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision %d", policies.ErrRevisionConflict, expectedRevision, binding.Revision)
		}
		now := time.Now().UTC()
		binding.Active = false
		binding.SupersededAt = &now
		s.policyBindings[id] = policies.CloneBinding(binding)
		s.markQueuedJobsPolicyPendingLocked(scope, subjectID)
		return policies.CloneBinding(binding), nil
	}
	if expectedRevision != 0 {
		return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision 0", policies.ErrRevisionConflict, expectedRevision)
	}
	return policies.Binding{}, ErrNotFound
}

func (s *MemoryStore) ListActiveExperimentPolicyBindings(scope policies.ScopeContext) ([]policies.Binding, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	wanted := map[policies.Scope]string{
		policies.ScopeAccount: strings.TrimSpace(scope.AccountID),
		policies.ScopeProject: strings.TrimSpace(scope.ProjectID),
		policies.ScopeDataset: strings.TrimSpace(scope.DatasetID),
		policies.ScopeRun:     strings.TrimSpace(scope.ExperimentJobID),
	}
	out := []policies.Binding{}
	for _, binding := range s.policyBindings {
		if !binding.Active || wanted[binding.Scope] == "" || wanted[binding.Scope] != binding.SubjectID {
			continue
		}
		out = append(out, policies.CloneBinding(binding))
	}
	sort.Slice(out, func(i, j int) bool {
		if policies.ScopeRank(out[i].Scope) != policies.ScopeRank(out[j].Scope) {
			return policies.ScopeRank(out[i].Scope) < policies.ScopeRank(out[j].Scope)
		}
		return out[i].Revision < out[j].Revision
	})
	return out, nil
}

func (s *MemoryStore) CreateExperimentPolicyEvaluation(input policies.Evaluation) (policies.Evaluation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.createExperimentPolicyEvaluationLocked(input)
}

func (s *MemoryStore) createExperimentPolicyEvaluationLocked(input policies.Evaluation) (policies.Evaluation, error) {
	normalizeEvaluationSlices(&input)
	if err := policies.ValidateEvaluation(input); err != nil {
		return policies.Evaluation{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	if err := s.validatePolicyEvaluationOwnershipLocked(input); err != nil {
		return policies.Evaluation{}, err
	}
	if input.ID != "" {
		if _, exists := s.policyEvaluations[input.ID]; exists {
			return policies.Evaluation{}, fmt.Errorf("%w: policy evaluation %s", policies.ErrImmutableConflict, input.ID)
		}
	} else {
		input.ID = s.newID("policy_evaluation")
	}
	input.CreatedAt = time.Now().UTC()
	input.ActorID = defaultPolicyActor(input.ActorID)
	created := policies.CloneEvaluation(input)
	s.policyEvaluations[created.ID] = created
	return policies.CloneEvaluation(created), nil
}

func (s *MemoryStore) GetExperimentPolicyEvaluation(id string) (policies.Evaluation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	evaluation, ok := s.policyEvaluations[strings.TrimSpace(id)]
	if !ok {
		return policies.Evaluation{}, ErrNotFound
	}
	return policies.CloneEvaluation(evaluation), nil
}

func (s *MemoryStore) ListExperimentPolicyEvaluations(projectID string) ([]policies.Evaluation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.projects[projectID]; !ok {
		return nil, ErrNotFound
	}
	out := []policies.Evaluation{}
	for _, evaluation := range s.policyEvaluations {
		if evaluation.ProjectID == projectID {
			out = append(out, policies.CloneEvaluation(evaluation))
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if !out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].CreatedAt.Before(out[j].CreatedAt)
		}
		return out[i].ID < out[j].ID
	})
	return out, nil
}

func (s *MemoryStore) accountExistsLocked(accountID string) bool {
	if accountID == policies.LocalDefaultAccountID {
		return true
	}
	for _, project := range s.projects {
		if project.AccountID == accountID {
			return true
		}
	}
	return false
}

func (s *MemoryStore) policySubjectAccountLocked(scope policies.Scope, subjectID string) (string, error) {
	switch scope {
	case policies.ScopeAccount:
		if !s.accountExistsLocked(subjectID) {
			return "", ErrNotFound
		}
		return subjectID, nil
	case policies.ScopeProject:
		project, ok := s.projects[subjectID]
		if !ok {
			return "", ErrNotFound
		}
		return project.AccountID, nil
	case policies.ScopeDataset:
		dataset, ok := s.datasets[subjectID]
		if !ok {
			return "", ErrNotFound
		}
		project, ok := s.projects[dataset.ProjectID]
		if !ok {
			return "", ErrNotFound
		}
		return project.AccountID, nil
	case policies.ScopeRun:
		job, ok := s.jobs[subjectID]
		if !ok {
			return "", ErrNotFound
		}
		project, ok := s.projects[job.ProjectID]
		if !ok {
			return "", ErrNotFound
		}
		return project.AccountID, nil
	default:
		return "", fmt.Errorf("%w: unsupported policy binding scope %q", ErrInvalidRequest, scope)
	}
}

func (s *MemoryStore) validatePolicyEvaluationOwnershipLocked(evaluation policies.Evaluation) error {
	if evaluation.AccountID != "" && !s.accountExistsLocked(evaluation.AccountID) {
		return ErrNotFound
	}
	if evaluation.ProjectID != "" {
		project, ok := s.projects[evaluation.ProjectID]
		if !ok {
			return ErrNotFound
		}
		if evaluation.AccountID != "" && evaluation.AccountID != project.AccountID {
			return fmt.Errorf("%w: evaluation account does not own project", ErrInvalidRequest)
		}
	}
	if evaluation.DatasetID != "" {
		dataset, ok := s.datasets[evaluation.DatasetID]
		if !ok {
			return ErrNotFound
		}
		if evaluation.ProjectID != "" && dataset.ProjectID != evaluation.ProjectID {
			return fmt.Errorf("%w: evaluation dataset does not belong to project", ErrInvalidRequest)
		}
	}
	if evaluation.JobID != "" {
		job, ok := s.jobs[evaluation.JobID]
		if !ok {
			return ErrNotFound
		}
		if evaluation.ProjectID != "" && job.ProjectID != evaluation.ProjectID {
			return fmt.Errorf("%w: evaluation job does not belong to project", ErrInvalidRequest)
		}
	}
	return nil
}

func defaultPolicyActor(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "system"
	}
	return value
}

var _ policies.Repository = (*MemoryStore)(nil)
