package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/runs"
)

func (s *MemoryStore) UpsertRemoteTrainingSession(session runs.RemoteTrainingSession) (runs.RemoteTrainingSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.upsertRemoteTrainingSessionLocked(session)
}

func (s *MemoryStore) GetRemoteTrainingSession(id string) (runs.RemoteTrainingSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	session, ok := s.remoteSessions[strings.TrimSpace(id)]
	if !ok {
		return runs.RemoteTrainingSession{}, ErrNotFound
	}
	return copyRemoteTrainingSession(session), nil
}

func (s *MemoryStore) CloseRemoteTrainingSession(id string, status string) (runs.RemoteTrainingSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	session, ok := s.remoteSessions[strings.TrimSpace(id)]
	if !ok {
		return runs.RemoteTrainingSession{}, ErrNotFound
	}
	updated := closeRemoteTrainingSessionRecord(session, status, time.Now().UTC())
	s.remoteSessions[session.ID] = updated
	return copyRemoteTrainingSession(updated), nil
}

func (s *MemoryStore) upsertRemoteTrainingSessionLocked(session runs.RemoteTrainingSession) (runs.RemoteTrainingSession, error) {
	normalized, err := normalizeRemoteTrainingSession(session, time.Now().UTC())
	if err != nil {
		return runs.RemoteTrainingSession{}, err
	}
	existing, ok := s.remoteSessions[normalized.ID]
	if ok && remoteTrainingSessionStatusTerminal(existing.Status) {
		return copyRemoteTrainingSession(existing), nil
	}
	if ok {
		normalized.CreatedAt = existing.CreatedAt
		if existing.Status == runs.RemoteTrainingSessionStatusClosing {
			normalized.Status = existing.Status
		}
		if existing.ClosedAt != nil {
			closedAt := *existing.ClosedAt
			normalized.ClosedAt = &closedAt
		}
	}
	s.remoteSessions[normalized.ID] = normalized
	return copyRemoteTrainingSession(normalized), nil
}

func (s *PostgresStore) UpsertRemoteTrainingSession(session runs.RemoteTrainingSession) (runs.RemoteTrainingSession, error) {
	normalized, err := normalizeRemoteTrainingSession(session, time.Now().UTC())
	if err != nil {
		return runs.RemoteTrainingSession{}, err
	}
	storageScopeJSON, err := json.Marshal(normalized.StorageScope)
	if err != nil {
		return runs.RemoteTrainingSession{}, fmt.Errorf("marshal remote training session storage scope: %w", err)
	}
	metadataJSON, err := json.Marshal(normalized.Metadata)
	if err != nil {
		return runs.RemoteTrainingSession{}, fmt.Errorf("marshal remote training session metadata: %w", err)
	}
	const query = `
		INSERT INTO remote_training_sessions (
			id, project_id, job_id, training_attempt_id, status, callback_token_hash,
			orchestrator_public_url, storage_public_url, storage_prefix, storage_scope, metadata,
			created_at, updated_at, expires_at, closed_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		ON CONFLICT (id) DO UPDATE SET
			project_id = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.project_id
				ELSE EXCLUDED.project_id
			END,
			job_id = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.job_id
				ELSE EXCLUDED.job_id
			END,
			training_attempt_id = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.training_attempt_id
				ELSE EXCLUDED.training_attempt_id
			END,
			status = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.status
				WHEN remote_training_sessions.status = 'closing'
					THEN remote_training_sessions.status
				ELSE EXCLUDED.status
			END,
			callback_token_hash = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.callback_token_hash
				ELSE EXCLUDED.callback_token_hash
			END,
			orchestrator_public_url = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.orchestrator_public_url
				ELSE EXCLUDED.orchestrator_public_url
			END,
			storage_public_url = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.storage_public_url
				ELSE EXCLUDED.storage_public_url
			END,
			storage_prefix = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.storage_prefix
				ELSE EXCLUDED.storage_prefix
			END,
			storage_scope = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.storage_scope
				ELSE EXCLUDED.storage_scope
			END,
			metadata = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.metadata
				ELSE EXCLUDED.metadata
			END,
			updated_at = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.updated_at
				ELSE EXCLUDED.updated_at
			END,
			expires_at = CASE
				WHEN remote_training_sessions.status IN ('closed', 'expired', 'failed')
					THEN remote_training_sessions.expires_at
				ELSE EXCLUDED.expires_at
			END
		RETURNING id, project_id, job_id, training_attempt_id, status, callback_token_hash,
			orchestrator_public_url, storage_public_url, storage_prefix, storage_scope, metadata,
			created_at, updated_at, expires_at, closed_at
	`
	return scanRemoteTrainingSession(s.db.QueryRowContext(
		context.Background(),
		query,
		normalized.ID,
		normalized.ProjectID,
		normalized.JobID,
		normalized.TrainingAttemptID,
		normalized.Status,
		normalized.CallbackTokenHash,
		normalized.OrchestratorPublicURL,
		normalized.StoragePublicURL,
		normalized.StoragePrefix,
		storageScopeJSON,
		metadataJSON,
		normalized.CreatedAt,
		normalized.UpdatedAt,
		normalized.ExpiresAt,
		normalized.ClosedAt,
	))
}

func (s *PostgresStore) GetRemoteTrainingSession(id string) (runs.RemoteTrainingSession, error) {
	const query = `
		SELECT id, project_id, job_id, training_attempt_id, status, callback_token_hash,
			orchestrator_public_url, storage_public_url, storage_prefix, storage_scope, metadata,
			created_at, updated_at, expires_at, closed_at
		FROM remote_training_sessions
		WHERE id = $1
	`
	return scanRemoteTrainingSession(s.db.QueryRowContext(context.Background(), query, strings.TrimSpace(id)))
}

func (s *PostgresStore) CloseRemoteTrainingSession(id string, status string) (runs.RemoteTrainingSession, error) {
	return closeRemoteTrainingSessionID(s.db, context.Background(), strings.TrimSpace(id), status, time.Now().UTC())
}

type remoteTrainingSessionExecQueryer interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func closeRemoteTrainingSessionID(db remoteTrainingSessionExecQueryer, ctx context.Context, id string, status string, now time.Time) (runs.RemoteTrainingSession, error) {
	normalizedStatus := NormalizeRemoteTrainingSessionStatus(status)
	if normalizedStatus == runs.RemoteTrainingSessionStatusActive {
		normalizedStatus = runs.RemoteTrainingSessionStatusClosed
	}
	const query = `
		UPDATE remote_training_sessions
		SET status = CASE
				WHEN status IN ('closed', 'expired', 'failed') THEN status
				ELSE $2
			END,
			closed_at = CASE
				WHEN status IN ('closed', 'expired', 'failed') THEN closed_at
				WHEN $2 IN ('closed', 'expired', 'failed') THEN $3
				ELSE closed_at
			END,
			updated_at = CASE
				WHEN status IN ('closed', 'expired', 'failed') THEN updated_at
				ELSE $3
			END
		WHERE id = $1
		RETURNING id, project_id, job_id, training_attempt_id, status, callback_token_hash,
			orchestrator_public_url, storage_public_url, storage_prefix, storage_scope, metadata,
			created_at, updated_at, expires_at, closed_at
	`
	return scanRemoteTrainingSession(db.QueryRowContext(ctx, query, strings.TrimSpace(id), normalizedStatus, now))
}

func normalizeRemoteTrainingSession(session runs.RemoteTrainingSession, now time.Time) (runs.RemoteTrainingSession, error) {
	session.ID = strings.TrimSpace(session.ID)
	session.ProjectID = strings.TrimSpace(session.ProjectID)
	session.JobID = strings.TrimSpace(session.JobID)
	session.TrainingAttemptID = strings.TrimSpace(session.TrainingAttemptID)
	if session.ID == "" {
		return runs.RemoteTrainingSession{}, fmt.Errorf("%w: remote training session id is required", ErrInvalidRequest)
	}
	if session.ProjectID == "" || session.JobID == "" || session.TrainingAttemptID == "" {
		return runs.RemoteTrainingSession{}, fmt.Errorf("%w: remote training session project, job, and attempt are required", ErrInvalidRequest)
	}
	session.Status = NormalizeRemoteTrainingSessionStatus(session.Status)
	session.CallbackTokenHash = strings.TrimSpace(session.CallbackTokenHash)
	session.OrchestratorPublicURL = strings.TrimSpace(session.OrchestratorPublicURL)
	session.StoragePublicURL = strings.TrimSpace(session.StoragePublicURL)
	session.StoragePrefix = strings.TrimSpace(session.StoragePrefix)
	if session.StorageScope == nil {
		session.StorageScope = map[string]any{}
	} else {
		session.StorageScope = copyAnyMap(session.StorageScope)
	}
	if session.Metadata == nil {
		session.Metadata = map[string]any{}
	} else {
		session.Metadata = copyAnyMap(session.Metadata)
	}
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = now
	}
	if session.ExpiresAt.IsZero() {
		session.ExpiresAt = now.Add(6 * time.Hour)
	}
	if session.ClosedAt != nil {
		closedAt := session.ClosedAt.UTC()
		session.ClosedAt = &closedAt
	}
	return session, nil
}

func closeRemoteTrainingSessionRecord(session runs.RemoteTrainingSession, status string, now time.Time) runs.RemoteTrainingSession {
	out := copyRemoteTrainingSession(session)
	if remoteTrainingSessionStatusTerminal(out.Status) {
		return out
	}
	normalized := NormalizeRemoteTrainingSessionStatus(status)
	if normalized == runs.RemoteTrainingSessionStatusActive {
		normalized = runs.RemoteTrainingSessionStatusClosed
	}
	out.Status = normalized
	out.UpdatedAt = now
	if remoteTrainingSessionStatusTerminal(normalized) && out.ClosedAt == nil {
		closedAt := now
		out.ClosedAt = &closedAt
	}
	return out
}

func NormalizeRemoteTrainingSessionStatus(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case runs.RemoteTrainingSessionStatusActive:
		return runs.RemoteTrainingSessionStatusActive
	case runs.RemoteTrainingSessionStatusClosing, "cancelling", "canceling":
		return runs.RemoteTrainingSessionStatusClosing
	case runs.RemoteTrainingSessionStatusClosed, "complete", "completed", "succeeded", "success", "cancelled", "canceled":
		return runs.RemoteTrainingSessionStatusClosed
	case runs.RemoteTrainingSessionStatusExpired, "timeout", "timed_out", "lease_expired", "superseded":
		return runs.RemoteTrainingSessionStatusExpired
	case runs.RemoteTrainingSessionStatusFailed, "failure", "error":
		return runs.RemoteTrainingSessionStatusFailed
	default:
		upper := strings.ToUpper(strings.TrimSpace(status))
		switch upper {
		case "SUCCEEDED":
			return runs.RemoteTrainingSessionStatusClosed
		case "FAILED":
			return runs.RemoteTrainingSessionStatusFailed
		default:
			return runs.RemoteTrainingSessionStatusActive
		}
	}
}

func remoteTrainingSessionStatusTerminal(status string) bool {
	switch NormalizeRemoteTrainingSessionStatus(status) {
	case runs.RemoteTrainingSessionStatusClosed, runs.RemoteTrainingSessionStatusExpired, runs.RemoteTrainingSessionStatusFailed:
		return true
	default:
		return false
	}
}

func remoteTrainingSessionIDFromConfig(config map[string]any) string {
	value, ok := config["remote_training_session"]
	if !ok {
		return ""
	}
	session, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	return strings.TrimSpace(fmt.Sprint(session["id"]))
}

func (s *MemoryStore) closeRemoteTrainingSessionForJobConfigLocked(config map[string]any, status string, now time.Time) {
	sessionID := remoteTrainingSessionIDFromConfig(config)
	if sessionID == "" {
		return
	}
	session, ok := s.remoteSessions[sessionID]
	if !ok {
		return
	}
	s.remoteSessions[sessionID] = closeRemoteTrainingSessionRecord(session, status, now)
}

func closeRemoteTrainingSessionForJobConfigTx(ctx context.Context, tx *sql.Tx, config map[string]any, status string, now time.Time) error {
	sessionID := remoteTrainingSessionIDFromConfig(config)
	if sessionID == "" {
		return nil
	}
	if _, err := closeRemoteTrainingSessionID(tx, ctx, sessionID, status, now); err != nil && err != ErrNotFound {
		return err
	}
	return nil
}

func copyRemoteTrainingSession(session runs.RemoteTrainingSession) runs.RemoteTrainingSession {
	out := session
	out.StorageScope = copyAnyMap(session.StorageScope)
	out.Metadata = copyAnyMap(session.Metadata)
	if session.ClosedAt != nil {
		closedAt := *session.ClosedAt
		out.ClosedAt = &closedAt
	}
	return out
}

func scanRemoteTrainingSession(row rowScanner) (runs.RemoteTrainingSession, error) {
	var session runs.RemoteTrainingSession
	var storageScopeJSON []byte
	var metadataJSON []byte
	var closedAt sql.NullTime
	if err := row.Scan(
		&session.ID,
		&session.ProjectID,
		&session.JobID,
		&session.TrainingAttemptID,
		&session.Status,
		&session.CallbackTokenHash,
		&session.OrchestratorPublicURL,
		&session.StoragePublicURL,
		&session.StoragePrefix,
		&storageScopeJSON,
		&metadataJSON,
		&session.CreatedAt,
		&session.UpdatedAt,
		&session.ExpiresAt,
		&closedAt,
	); err != nil {
		if errorsIsNoRows(err) {
			return runs.RemoteTrainingSession{}, ErrNotFound
		}
		return runs.RemoteTrainingSession{}, err
	}
	if len(storageScopeJSON) > 0 {
		if err := json.Unmarshal(storageScopeJSON, &session.StorageScope); err != nil {
			return runs.RemoteTrainingSession{}, fmt.Errorf("unmarshal remote training session storage scope: %w", err)
		}
	}
	if session.StorageScope == nil {
		session.StorageScope = map[string]any{}
	}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &session.Metadata); err != nil {
			return runs.RemoteTrainingSession{}, fmt.Errorf("unmarshal remote training session metadata: %w", err)
		}
	}
	if session.Metadata == nil {
		session.Metadata = map[string]any{}
	}
	if closedAt.Valid {
		value := closedAt.Time
		session.ClosedAt = &value
	}
	return session, nil
}

func errorsIsNoRows(err error) bool {
	return err == sql.ErrNoRows
}
