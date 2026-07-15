package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgconn"

	"model-express/services/orchestrator/internal/policies"
)

func (s *PostgresStore) CreateCompatibilityProfile(input policies.CompatibilityProfile) (policies.CompatibilityProfile, error) {
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
	var owner any
	if strings.TrimSpace(input.OwnerAccountID) != "" {
		if err := s.requireAccount(input.OwnerAccountID); err != nil {
			return policies.CompatibilityProfile{}, err
		}
		owner = strings.TrimSpace(input.OwnerAccountID)
	}
	query := `
		INSERT INTO compatibility_profiles (
			profile_key, semantic_version, schema_version, catalog_version,
			document, document_hash, owner_account_id, created_by
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING ` + compatibilityProfileSelectColumns()
	profile, err := scanCompatibilityProfile(s.db.QueryRowContext(
		context.Background(), query,
		strings.TrimSpace(input.ProfileKey), strings.TrimSpace(input.SemanticVersion),
		document.SchemaVersion, document.CatalogVersion, canonical, hash, owner,
		defaultPolicyActor(input.CreatedBy),
	))
	if isUniqueViolation(err) {
		return policies.CompatibilityProfile{}, fmt.Errorf("%w: compatibility profile %s@%s", policies.ErrImmutableConflict, input.ProfileKey, input.SemanticVersion)
	}
	return profile, err
}

func (s *PostgresStore) GetCompatibilityProfile(profileKey string, semanticVersion string) (policies.CompatibilityProfile, error) {
	return scanCompatibilityProfile(s.db.QueryRowContext(context.Background(), `
		SELECT `+compatibilityProfileSelectColumns()+`
		FROM compatibility_profiles
		WHERE profile_key = $1 AND semantic_version = $2
	`, strings.TrimSpace(profileKey), strings.TrimSpace(semanticVersion)))
}

func (s *PostgresStore) CreateExperimentPolicyVersion(input policies.PolicyVersion) (policies.PolicyVersion, error) {
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
	if err := s.requireAccount(ownerAccountID); err != nil {
		return policies.PolicyVersion{}, err
	}
	query := `
		INSERT INTO experiment_policy_versions (
			owner_account_id, schema_version, document, document_hash, created_by
		)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING ` + policyVersionSelectColumns()
	return scanPolicyVersion(s.db.QueryRowContext(
		context.Background(), query, ownerAccountID, document.SchemaVersion,
		canonical, hash, defaultPolicyActor(input.CreatedBy),
	))
}

func (s *PostgresStore) GetExperimentPolicyVersion(id string) (policies.PolicyVersion, error) {
	return scanPolicyVersion(s.db.QueryRowContext(context.Background(), `
		SELECT `+policyVersionSelectColumns()+`
		FROM experiment_policy_versions
		WHERE id = $1
	`, strings.TrimSpace(id)))
}

func (s *PostgresStore) SetExperimentPolicyBinding(write policies.BindingWrite) (policies.Binding, error) {
	write.SubjectID = strings.TrimSpace(write.SubjectID)
	write.PolicyVersionID = strings.TrimSpace(write.PolicyVersionID)
	column, err := bindingSubjectColumn(write.Scope)
	if err != nil || write.SubjectID == "" || write.PolicyVersionID == "" {
		return policies.Binding{}, fmt.Errorf("%w: valid policy binding scope, subject, and version are required", ErrInvalidRequest)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return policies.Binding{}, err
	}
	defer tx.Rollback()
	lockKey := string(write.Scope) + ":" + write.SubjectID
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, lockKey); err != nil {
		return policies.Binding{}, err
	}

	var versionOwner string
	if err := tx.QueryRowContext(ctx, `
		SELECT owner_account_id FROM experiment_policy_versions WHERE id = $1
	`, write.PolicyVersionID).Scan(&versionOwner); err != nil {
		return policies.Binding{}, normalizeSQLError(err)
	}
	subjectAccount, err := policySubjectAccountTx(ctx, tx, write.Scope, write.SubjectID)
	if err != nil {
		return policies.Binding{}, err
	}
	if versionOwner != subjectAccount {
		return policies.Binding{}, fmt.Errorf("%w: policy version and binding subject have different account owners", ErrInvalidRequest)
	}

	active, activeErr := scanPolicyBinding(tx.QueryRowContext(ctx, `
		SELECT `+policyBindingSelectColumns()+`
		FROM experiment_policy_bindings
		WHERE scope = $1 AND `+column+` = $2 AND active
		FOR UPDATE
	`, string(write.Scope), write.SubjectID))
	if activeErr != nil && !errors.Is(activeErr, ErrNotFound) {
		return policies.Binding{}, activeErr
	}
	currentRevision := int64(0)
	if activeErr == nil {
		currentRevision = active.Revision
	}
	if write.ExpectedRevision != nil && *write.ExpectedRevision != currentRevision {
		return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision %d", policies.ErrRevisionConflict, *write.ExpectedRevision, currentRevision)
	}
	var maxRevision int64
	if err := tx.QueryRowContext(ctx, `
		SELECT COALESCE(MAX(revision), 0)
		FROM experiment_policy_bindings
		WHERE scope = $1 AND `+column+` = $2
	`, string(write.Scope), write.SubjectID).Scan(&maxRevision); err != nil {
		return policies.Binding{}, err
	}
	supersedesID := any(nil)
	if activeErr == nil {
		supersedesID = active.ID
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_policy_bindings
			SET active = false, superseded_at = now()
			WHERE id = $1 AND active
		`, active.ID); err != nil {
			return policies.Binding{}, err
		}
	}
	created, err := scanPolicyBinding(tx.QueryRowContext(ctx, `
		INSERT INTO experiment_policy_bindings (
			scope, policy_version_id, `+column+`, revision, active, supersedes_id, created_by
		)
		VALUES ($1, $2, $3, $4, true, $5, $6)
		RETURNING `+policyBindingSelectColumns()+`
	`, string(write.Scope), write.PolicyVersionID, write.SubjectID, maxRevision+1, supersedesID, defaultPolicyActor(write.CreatedBy)))
	if err != nil {
		if isUniqueViolation(err) {
			return policies.Binding{}, policies.ErrRevisionConflict
		}
		return policies.Binding{}, err
	}
	if err := tx.Commit(); err != nil {
		return policies.Binding{}, err
	}
	return created, nil
}

func (s *PostgresStore) ClearExperimentPolicyBinding(scope policies.Scope, subjectID string, expectedRevision int64) (policies.Binding, error) {
	subjectID = strings.TrimSpace(subjectID)
	column, err := bindingSubjectColumn(scope)
	if err != nil || subjectID == "" {
		return policies.Binding{}, fmt.Errorf("%w: valid policy binding scope and subject are required", ErrInvalidRequest)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return policies.Binding{}, err
	}
	defer tx.Rollback()
	lockKey := string(scope) + ":" + subjectID
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, lockKey); err != nil {
		return policies.Binding{}, err
	}
	if _, err := policySubjectAccountTx(ctx, tx, scope, subjectID); err != nil {
		return policies.Binding{}, err
	}
	active, err := scanPolicyBinding(tx.QueryRowContext(ctx, `
		SELECT `+policyBindingSelectColumns()+`
		FROM experiment_policy_bindings
		WHERE scope = $1 AND `+column+` = $2 AND active
		FOR UPDATE
	`, string(scope), subjectID))
	if errors.Is(err, ErrNotFound) {
		if expectedRevision != 0 {
			return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision 0", policies.ErrRevisionConflict, expectedRevision)
		}
		return policies.Binding{}, ErrNotFound
	}
	if err != nil {
		return policies.Binding{}, err
	}
	if active.Revision != expectedRevision {
		return policies.Binding{}, fmt.Errorf("%w: expected revision %d, current revision %d", policies.ErrRevisionConflict, expectedRevision, active.Revision)
	}
	cleared, err := scanPolicyBinding(tx.QueryRowContext(ctx, `
		UPDATE experiment_policy_bindings
		SET active = false, superseded_at = now()
		WHERE id = $1 AND active
		RETURNING `+policyBindingSelectColumns()+`
	`, active.ID))
	if err != nil {
		return policies.Binding{}, err
	}
	if err := tx.Commit(); err != nil {
		return policies.Binding{}, err
	}
	return cleared, nil
}

func (s *PostgresStore) ListActiveExperimentPolicyBindings(scope policies.ScopeContext) ([]policies.Binding, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+policyBindingSelectColumns()+`
		FROM experiment_policy_bindings
		WHERE active AND (
			(scope = 'account' AND account_id = NULLIF($1, ''))
			OR (scope = 'project' AND project_id = NULLIF($2, ''))
			OR (scope = 'dataset' AND dataset_id = NULLIF($3, ''))
			OR (scope = 'run' AND experiment_job_id = NULLIF($4, ''))
		)
		ORDER BY CASE scope
			WHEN 'account' THEN 0 WHEN 'project' THEN 1 WHEN 'dataset' THEN 2 ELSE 3 END,
			revision ASC
	`, strings.TrimSpace(scope.AccountID), strings.TrimSpace(scope.ProjectID), strings.TrimSpace(scope.DatasetID), strings.TrimSpace(scope.ExperimentJobID))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []policies.Binding{}
	for rows.Next() {
		binding, err := scanPolicyBinding(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, binding)
	}
	return out, rows.Err()
}

func (s *PostgresStore) CreateExperimentPolicyEvaluation(input policies.Evaluation) (policies.Evaluation, error) {
	normalizeEvaluationSlices(&input)
	if err := policies.ValidateEvaluation(input); err != nil {
		return policies.Evaluation{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	if err := s.validatePolicyEvaluationOwnership(input); err != nil {
		return policies.Evaluation{}, err
	}
	profilesJSON, err := json.Marshal(input.CompatibilityProfiles)
	if err != nil {
		return policies.Evaluation{}, err
	}
	sourcesJSON, err := json.Marshal(input.PolicySources)
	if err != nil {
		return policies.Evaluation{}, err
	}
	requestedJSON, err := json.Marshal(input.RequestedCapabilityUses)
	if err != nil {
		return policies.Evaluation{}, err
	}
	effectiveJSON, err := json.Marshal(input.EffectiveCapabilityUses)
	if err != nil {
		return policies.Evaluation{}, err
	}
	reasonsJSON, err := json.Marshal(input.ReasonCodes)
	if err != nil {
		return policies.Evaluation{}, err
	}
	findingsJSON, err := json.Marshal(input.Findings)
	if err != nil {
		return policies.Evaluation{}, err
	}
	columns := "operation, decision, account_id, project_id, dataset_id, plan_id, job_id, agent_invocation_id, champion_export_id, candidate_config_hash, catalog_version, compatibility_profile_refs, policy_sources, effective_snapshot, effective_policy_hash, requested_capability_uses, effective_capability_uses, reason_codes, findings, actor_id, request_id"
	placeholders := "$1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''), NULLIF($7, ''), NULLIF($8, ''), NULLIF($9, ''), $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21"
	args := []any{
		input.Operation, input.Decision, input.AccountID, input.ProjectID, input.DatasetID,
		input.PlanID, input.JobID, input.AgentInvocationID, input.ChampionExportID,
		input.CandidateConfigHash, input.CatalogVersion, profilesJSON, sourcesJSON,
		input.EffectiveSnapshot, input.EffectivePolicyHash, requestedJSON, effectiveJSON,
		reasonsJSON, findingsJSON, defaultPolicyActor(input.ActorID), input.RequestID,
	}
	if strings.TrimSpace(input.ID) != "" {
		columns = "id, " + columns
		placeholders = "$1, " + shiftSQLPlaceholders(placeholders, 1)
		args = append([]any{strings.TrimSpace(input.ID)}, args...)
	}
	query := `INSERT INTO experiment_policy_evaluations (` + columns + `)
		VALUES (` + placeholders + `)
		RETURNING ` + policyEvaluationSelectColumns()
	created, err := scanPolicyEvaluation(s.db.QueryRowContext(context.Background(), query, args...))
	if isUniqueViolation(err) {
		return policies.Evaluation{}, fmt.Errorf("%w: policy evaluation %s", policies.ErrImmutableConflict, input.ID)
	}
	return created, err
}

func (s *PostgresStore) GetExperimentPolicyEvaluation(id string) (policies.Evaluation, error) {
	return scanPolicyEvaluation(s.db.QueryRowContext(context.Background(), `
		SELECT `+policyEvaluationSelectColumns()+`
		FROM experiment_policy_evaluations
		WHERE id = $1
	`, strings.TrimSpace(id)))
}

func (s *PostgresStore) ListExperimentPolicyEvaluations(projectID string) ([]policies.Evaluation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+policyEvaluationSelectColumns()+`
		FROM experiment_policy_evaluations
		WHERE project_id = $1
		ORDER BY created_at ASC, id ASC
	`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []policies.Evaluation{}
	for rows.Next() {
		evaluation, err := scanPolicyEvaluation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, evaluation)
	}
	return out, rows.Err()
}

func scanCompatibilityProfile(row rowScanner) (policies.CompatibilityProfile, error) {
	var profile policies.CompatibilityProfile
	var documentJSON []byte
	var owner sql.NullString
	if err := row.Scan(
		&profile.ID, &profile.ProfileKey, &profile.SemanticVersion, &profile.SchemaVersion,
		&profile.CatalogVersion, &documentJSON, &profile.DocumentHash, &owner,
		&profile.CreatedAt, &profile.CreatedBy,
	); err != nil {
		return policies.CompatibilityProfile{}, normalizeSQLError(err)
	}
	if err := json.Unmarshal(documentJSON, &profile.Document); err != nil {
		return policies.CompatibilityProfile{}, fmt.Errorf("unmarshal compatibility profile: %w", err)
	}
	profile.CanonicalJSON = append([]byte(nil), documentJSON...)
	if owner.Valid {
		profile.OwnerAccountID = owner.String
	}
	return profile, nil
}

func scanPolicyVersion(row rowScanner) (policies.PolicyVersion, error) {
	var version policies.PolicyVersion
	var documentJSON []byte
	if err := row.Scan(
		&version.ID, &version.OwnerAccountID, &version.SchemaVersion, &version.Revision,
		&documentJSON, &version.DocumentHash, &version.CreatedAt, &version.CreatedBy,
	); err != nil {
		return policies.PolicyVersion{}, normalizeSQLError(err)
	}
	if err := json.Unmarshal(documentJSON, &version.Document); err != nil {
		return policies.PolicyVersion{}, fmt.Errorf("unmarshal experiment policy version: %w", err)
	}
	version.CanonicalJSON = append([]byte(nil), documentJSON...)
	return version, nil
}

func scanPolicyBinding(row rowScanner) (policies.Binding, error) {
	var binding policies.Binding
	var scope string
	var accountID, projectID, datasetID, jobID sql.NullString
	var supersedesID sql.NullString
	var supersededAt sql.NullTime
	if err := row.Scan(
		&binding.ID, &scope, &binding.PolicyVersionID,
		&accountID, &projectID, &datasetID, &jobID,
		&binding.Revision, &binding.Active, &supersedesID, &supersededAt,
		&binding.CreatedAt, &binding.CreatedBy,
	); err != nil {
		return policies.Binding{}, normalizeSQLError(err)
	}
	binding.Scope = policies.Scope(scope)
	switch binding.Scope {
	case policies.ScopeAccount:
		binding.SubjectID = accountID.String
	case policies.ScopeProject:
		binding.SubjectID = projectID.String
	case policies.ScopeDataset:
		binding.SubjectID = datasetID.String
	case policies.ScopeRun:
		binding.SubjectID = jobID.String
	}
	if supersedesID.Valid {
		binding.SupersedesID = supersedesID.String
	}
	if supersededAt.Valid {
		binding.SupersededAt = &supersededAt.Time
	}
	return binding, nil
}

func scanPolicyEvaluation(row rowScanner) (policies.Evaluation, error) {
	var evaluation policies.Evaluation
	var accountID, projectID, datasetID, planID, jobID, invocationID, exportID sql.NullString
	var profilesJSON, sourcesJSON, snapshotJSON, requestedJSON, effectiveJSON, reasonsJSON, findingsJSON []byte
	if err := row.Scan(
		&evaluation.ID, &evaluation.Operation, &evaluation.Decision,
		&accountID, &projectID, &datasetID, &planID, &jobID, &invocationID, &exportID,
		&evaluation.CandidateConfigHash, &evaluation.CatalogVersion,
		&profilesJSON, &sourcesJSON, &snapshotJSON, &evaluation.EffectivePolicyHash,
		&requestedJSON, &effectiveJSON, &reasonsJSON, &findingsJSON,
		&evaluation.ActorID, &evaluation.RequestID, &evaluation.CreatedAt,
	); err != nil {
		return policies.Evaluation{}, normalizeSQLError(err)
	}
	evaluation.AccountID = accountID.String
	evaluation.ProjectID = projectID.String
	evaluation.DatasetID = datasetID.String
	evaluation.PlanID = planID.String
	evaluation.JobID = jobID.String
	evaluation.AgentInvocationID = invocationID.String
	evaluation.ChampionExportID = exportID.String
	evaluation.EffectiveSnapshot = append([]byte(nil), snapshotJSON...)
	for _, target := range []struct {
		data []byte
		out  any
	}{
		{profilesJSON, &evaluation.CompatibilityProfiles},
		{sourcesJSON, &evaluation.PolicySources},
		{requestedJSON, &evaluation.RequestedCapabilityUses},
		{effectiveJSON, &evaluation.EffectiveCapabilityUses},
		{reasonsJSON, &evaluation.ReasonCodes},
		{findingsJSON, &evaluation.Findings},
	} {
		if err := json.Unmarshal(target.data, target.out); err != nil {
			return policies.Evaluation{}, fmt.Errorf("unmarshal policy evaluation: %w", err)
		}
	}
	return evaluation, nil
}

func compatibilityProfileSelectColumns() string {
	return "id, profile_key, semantic_version, schema_version, catalog_version, document, document_hash, owner_account_id, created_at, created_by"
}

func policyVersionSelectColumns() string {
	return "id, owner_account_id, schema_version, revision, document, document_hash, created_at, created_by"
}

func policyBindingSelectColumns() string {
	return "id, scope, policy_version_id, account_id, project_id, dataset_id, experiment_job_id, revision, active, supersedes_id, superseded_at, created_at, created_by"
}

func policyEvaluationSelectColumns() string {
	return "id, operation, decision, account_id, project_id, dataset_id, plan_id, job_id, agent_invocation_id, champion_export_id, candidate_config_hash, catalog_version, compatibility_profile_refs, policy_sources, effective_snapshot, effective_policy_hash, requested_capability_uses, effective_capability_uses, reason_codes, findings, actor_id, request_id, created_at"
}

func bindingSubjectColumn(scope policies.Scope) (string, error) {
	switch scope {
	case policies.ScopeAccount:
		return "account_id", nil
	case policies.ScopeProject:
		return "project_id", nil
	case policies.ScopeDataset:
		return "dataset_id", nil
	case policies.ScopeRun:
		return "experiment_job_id", nil
	default:
		return "", fmt.Errorf("unsupported policy binding scope %q", scope)
	}
}

func policySubjectAccountTx(ctx context.Context, tx *sql.Tx, scope policies.Scope, subjectID string) (string, error) {
	var query string
	switch scope {
	case policies.ScopeAccount:
		query = `SELECT id FROM accounts WHERE id = $1`
	case policies.ScopeProject:
		query = `SELECT account_id FROM projects WHERE id = $1`
	case policies.ScopeDataset:
		query = `SELECT p.account_id FROM datasets d JOIN projects p ON p.id = d.project_id WHERE d.id = $1`
	case policies.ScopeRun:
		query = `SELECT p.account_id FROM experiment_jobs j JOIN projects p ON p.id = j.project_id WHERE j.id = $1`
	default:
		return "", fmt.Errorf("%w: unsupported policy binding scope %q", ErrInvalidRequest, scope)
	}
	var accountID string
	if err := tx.QueryRowContext(ctx, query, subjectID).Scan(&accountID); err != nil {
		return "", normalizeSQLError(err)
	}
	return accountID, nil
}

func (s *PostgresStore) requireAccount(accountID string) error {
	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `SELECT EXISTS(SELECT 1 FROM accounts WHERE id = $1)`, strings.TrimSpace(accountID)).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return ErrNotFound
	}
	return nil
}

func (s *PostgresStore) validatePolicyEvaluationOwnership(evaluation policies.Evaluation) error {
	if evaluation.AccountID != "" {
		if err := s.requireAccount(evaluation.AccountID); err != nil {
			return err
		}
	}
	if evaluation.ProjectID != "" {
		project, err := s.GetProject(evaluation.ProjectID)
		if err != nil {
			return err
		}
		if evaluation.AccountID != "" && project.AccountID != evaluation.AccountID {
			return fmt.Errorf("%w: evaluation account does not own project", ErrInvalidRequest)
		}
	}
	if evaluation.DatasetID != "" {
		dataset, err := s.GetDataset(evaluation.DatasetID)
		if err != nil {
			return err
		}
		if evaluation.ProjectID != "" && dataset.ProjectID != evaluation.ProjectID {
			return fmt.Errorf("%w: evaluation dataset does not belong to project", ErrInvalidRequest)
		}
	}
	if evaluation.JobID != "" {
		job, err := s.GetJob(evaluation.JobID)
		if err != nil {
			return err
		}
		if evaluation.ProjectID != "" && job.ProjectID != evaluation.ProjectID {
			return fmt.Errorf("%w: evaluation job does not belong to project", ErrInvalidRequest)
		}
	}
	return nil
}

func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
}

func shiftSQLPlaceholders(input string, amount int) string {
	// Replace from the largest placeholder down so $1 cannot corrupt $10.
	for index := 64; index >= 1; index-- {
		input = strings.ReplaceAll(input, fmt.Sprintf("$%d", index), fmt.Sprintf("$%d", index+amount))
	}
	return input
}

var _ policies.Repository = (*PostgresStore)(nil)
