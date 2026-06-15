package store

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/memory"
)

func (s *PostgresStore) UpsertMemoryEmbedding(record memory.MemoryEmbeddingRecord) (memory.MemoryEmbeddingRecord, error) {
	normalized, err := normalizeMemoryEmbeddingRecord(record, time.Now().UTC())
	if err != nil {
		return memory.MemoryEmbeddingRecord{}, err
	}
	if err := s.requireProject(normalized.ProjectID); err != nil {
		return memory.MemoryEmbeddingRecord{}, err
	}
	vectorText, err := postgresVectorLiteral(normalized.Embedding)
	if err != nil {
		return memory.MemoryEmbeddingRecord{}, err
	}
	summaryJSON, err := json.Marshal(emptyMapIfNil(normalized.SummaryCard))
	if err != nil {
		return memory.MemoryEmbeddingRecord{}, fmt.Errorf("marshal memory embedding summary_card: %w", err)
	}
	metadataJSON, err := json.Marshal(emptyMapIfNil(normalized.Metadata))
	if err != nil {
		return memory.MemoryEmbeddingRecord{}, fmt.Errorf("marshal memory embedding metadata: %w", err)
	}

	query := fmt.Sprintf(`
		INSERT INTO agent_memory_embeddings (
			id, source_table, source_id, project_id, dataset_id, plan_id, job_id, invocation_id,
			kind, scope, embedding_model, embedding_dimensions, embedding, embedding_text, embedding_text_hash,
			summary_card, metadata, quality_score, outcome_score
		)
		VALUES (
			COALESCE(NULLIF($1, ''), 'memory_embedding_' || nextval('agent_memory_embedding_id_seq')),
			$2, $3, $4, $5, $6, $7, $8,
			$9, $10, $11, $12, $13::vector, $14, $15,
			$16, $17, $18, $19
		)
		ON CONFLICT (source_table, source_id, embedding_model) DO UPDATE SET
			project_id = EXCLUDED.project_id,
			dataset_id = EXCLUDED.dataset_id,
			plan_id = EXCLUDED.plan_id,
			job_id = EXCLUDED.job_id,
			invocation_id = EXCLUDED.invocation_id,
			kind = EXCLUDED.kind,
			scope = EXCLUDED.scope,
			embedding_dimensions = EXCLUDED.embedding_dimensions,
			embedding = EXCLUDED.embedding,
			embedding_text = EXCLUDED.embedding_text,
			embedding_text_hash = EXCLUDED.embedding_text_hash,
			summary_card = EXCLUDED.summary_card,
			metadata = EXCLUDED.metadata,
			quality_score = EXCLUDED.quality_score,
			outcome_score = EXCLUDED.outcome_score,
			updated_at = now()
		RETURNING %s
	`, memoryEmbeddingSelectColumns())
	return scanMemoryEmbeddingRecord(s.db.QueryRowContext(
		context.Background(),
		query,
		normalized.ID,
		normalized.SourceTable,
		normalized.SourceID,
		normalized.ProjectID,
		normalized.DatasetID,
		normalized.PlanID,
		normalized.JobID,
		normalized.InvocationID,
		normalized.Kind,
		normalized.Scope,
		normalized.EmbeddingModel,
		normalized.EmbeddingDimensions,
		vectorText,
		normalized.EmbeddingText,
		normalized.EmbeddingTextHash,
		summaryJSON,
		metadataJSON,
		normalized.QualityScore,
		normalized.OutcomeScore,
	))
}

func (s *PostgresStore) GetMemoryEmbeddingSourceState(projectID string, sourceTable string, sourceID string, embeddingModel string) (memory.MemoryEmbeddingSourceState, error) {
	if strings.TrimSpace(projectID) == "" {
		return memory.MemoryEmbeddingSourceState{}, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(projectID); err != nil {
		return memory.MemoryEmbeddingSourceState{}, err
	}
	const query = `
		SELECT source_table, source_id, project_id, embedding_model, embedding_dimensions, embedding_text_hash, updated_at
		FROM agent_memory_embeddings
		WHERE project_id = $1 AND source_table = $2 AND source_id = $3 AND embedding_model = $4
		LIMIT 1
	`
	var state memory.MemoryEmbeddingSourceState
	if err := s.db.QueryRowContext(context.Background(), query, projectID, sourceTable, sourceID, embeddingModel).Scan(
		&state.SourceTable,
		&state.SourceID,
		&state.ProjectID,
		&state.EmbeddingModel,
		&state.EmbeddingDimensions,
		&state.EmbeddingTextHash,
		&state.UpdatedAt,
	); err != nil {
		return memory.MemoryEmbeddingSourceState{}, normalizeSQLError(err)
	}
	return state, nil
}

func (s *PostgresStore) CreateMemoryEmbeddingUsageEvent(event memory.MemoryEmbeddingUsageEvent) (memory.MemoryEmbeddingUsageEvent, error) {
	if strings.TrimSpace(event.ProjectID) == "" {
		return memory.MemoryEmbeddingUsageEvent{}, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(event.ProjectID); err != nil {
		return memory.MemoryEmbeddingUsageEvent{}, err
	}
	now := time.Now().UTC()
	event.Purpose = strings.TrimSpace(event.Purpose)
	event.RetrievalPurpose = strings.TrimSpace(event.RetrievalPurpose)
	event.SourceTable = strings.TrimSpace(event.SourceTable)
	event.SourceID = strings.TrimSpace(event.SourceID)
	event.EmbeddingModel = strings.TrimSpace(event.EmbeddingModel)
	event.SkipReason = strings.TrimSpace(event.SkipReason)
	event.SourceTextHash = strings.TrimSpace(event.SourceTextHash)
	event.QueryHash = strings.TrimSpace(event.QueryHash)
	if event.CreatedAt.IsZero() {
		event.CreatedAt = now
	}
	metadataJSON, err := json.Marshal(emptyMapIfNil(event.Metadata))
	if err != nil {
		return memory.MemoryEmbeddingUsageEvent{}, fmt.Errorf("marshal memory embedding usage metadata: %w", err)
	}
	usageJSON, err := json.Marshal(emptyMapIfNil(event.ProviderUsage))
	if err != nil {
		return memory.MemoryEmbeddingUsageEvent{}, fmt.Errorf("marshal memory embedding usage provider_usage: %w", err)
	}
	query := `
		INSERT INTO memory_embedding_usage_events (
			id, project_id, dataset_id, plan_id, job_id, invocation_id, purpose, retrieval_purpose,
			source_table, source_id, embedding_model, embedding_dimensions, input_bytes, provider_call_count,
			retrieved_count, injected, log_only, cached, skipped, skip_reason, source_text_hash, query_hash,
			provider_usage, metadata, created_at
		)
		VALUES (
			COALESCE(NULLIF($1, ''), 'memory_embedding_usage_event_' || nextval('memory_embedding_usage_event_id_seq')),
			$2, $3, $4, $5, $6, $7, $8,
			$9, $10, $11, $12, $13, $14,
			$15, $16, $17, $18, $19, $20, $21, $22,
			$23::jsonb, $24::jsonb, $25
		)
		RETURNING id, project_id, dataset_id, plan_id, job_id, invocation_id, purpose, retrieval_purpose,
			source_table, source_id, embedding_model, embedding_dimensions, input_bytes, provider_call_count,
			retrieved_count, injected, log_only, cached, skipped, skip_reason, source_text_hash, query_hash,
			provider_usage, metadata, created_at
	`
	return scanMemoryEmbeddingUsageEvent(s.db.QueryRowContext(
		context.Background(),
		query,
		event.ID,
		event.ProjectID,
		event.DatasetID,
		event.PlanID,
		event.JobID,
		event.InvocationID,
		event.Purpose,
		event.RetrievalPurpose,
		event.SourceTable,
		event.SourceID,
		event.EmbeddingModel,
		event.EmbeddingDimensions,
		event.InputBytes,
		event.ProviderCallCount,
		event.RetrievedCount,
		event.Injected,
		event.LogOnly,
		event.Cached,
		event.Skipped,
		event.SkipReason,
		event.SourceTextHash,
		event.QueryHash,
		usageJSON,
		metadataJSON,
		event.CreatedAt,
	))
}

func (s *PostgresStore) ListProjectMemoryEmbeddingUsageEvents(projectID string, limit int) ([]memory.MemoryEmbeddingUsageEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 25
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, project_id, dataset_id, plan_id, job_id, invocation_id, purpose, retrieval_purpose,
			source_table, source_id, embedding_model, embedding_dimensions, input_bytes, provider_call_count,
			retrieved_count, injected, log_only, cached, skipped, skip_reason, source_text_hash, query_hash,
			provider_usage, metadata, created_at
		FROM memory_embedding_usage_events
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []memory.MemoryEmbeddingUsageEvent{}
	for rows.Next() {
		event, err := scanMemoryEmbeddingUsageEvent(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, rows.Err()
}

func (s *PostgresStore) CountProjectMemoryEmbeddingUsageEvents(projectID string, purpose string, since time.Time) (int, error) {
	if err := s.requireProject(projectID); err != nil {
		return 0, err
	}
	var count int
	query := `
		SELECT COALESCE(SUM(provider_call_count), 0)
		FROM memory_embedding_usage_events
		WHERE project_id = $1
	`
	args := []any{projectID}
	if strings.TrimSpace(purpose) != "" {
		query += " AND purpose = $2"
		args = append(args, purpose)
	}
	if !since.IsZero() {
		if strings.TrimSpace(purpose) != "" {
			query += fmt.Sprintf(" AND created_at >= $%d", len(args)+1)
		} else {
			query += fmt.Sprintf(" AND created_at >= $%d", len(args)+1)
		}
		args = append(args, since)
	}
	if err := s.db.QueryRowContext(context.Background(), query, args...).Scan(&count); err != nil {
		return 0, err
	}
	return count, nil
}

func (s *PostgresStore) GetMemoryRetrievalQueryCache(projectID string, datasetID string, purpose string, embeddingModel string, embeddingDimensions int, normalizedQueryHash string) (memory.MemoryRetrievalQueryCacheRecord, error) {
	if err := s.requireProject(projectID); err != nil {
		return memory.MemoryRetrievalQueryCacheRecord{}, err
	}
	const query = `
		SELECT id, project_id, dataset_id, purpose, embedding_model, embedding_dimensions, normalized_query_hash,
			query_text, embedding_json, created_at, updated_at, expires_at
		FROM memory_retrieval_query_cache
		WHERE project_id = $1
			AND dataset_id = $2
			AND purpose = $3
			AND embedding_model = $4
			AND embedding_dimensions = $5
			AND normalized_query_hash = $6
			AND expires_at > now()
		ORDER BY updated_at DESC
		LIMIT 1
	`
	var record memory.MemoryRetrievalQueryCacheRecord
	var embeddingJSON []byte
	if err := s.db.QueryRowContext(context.Background(), query, projectID, datasetID, purpose, embeddingModel, embeddingDimensions, normalizedQueryHash).Scan(
		&record.ID,
		&record.ProjectID,
		&record.DatasetID,
		&record.Purpose,
		&record.EmbeddingModel,
		&record.EmbeddingDimensions,
		&record.NormalizedQueryHash,
		&record.QueryText,
		&embeddingJSON,
		&record.CreatedAt,
		&record.UpdatedAt,
		&record.ExpiresAt,
	); err != nil {
		return memory.MemoryRetrievalQueryCacheRecord{}, normalizeSQLError(err)
	}
	if len(embeddingJSON) > 0 {
		if err := json.Unmarshal(embeddingJSON, &record.Embedding); err != nil {
			return memory.MemoryRetrievalQueryCacheRecord{}, fmt.Errorf("unmarshal memory retrieval query cache embedding: %w", err)
		}
	}
	return record, nil
}

func (s *PostgresStore) UpsertMemoryRetrievalQueryCache(record memory.MemoryRetrievalQueryCacheRecord) (memory.MemoryRetrievalQueryCacheRecord, error) {
	if err := s.requireProject(record.ProjectID); err != nil {
		return memory.MemoryRetrievalQueryCacheRecord{}, err
	}
	now := time.Now().UTC()
	if record.CreatedAt.IsZero() {
		record.CreatedAt = now
	}
	if record.ExpiresAt.IsZero() {
		record.ExpiresAt = now.Add(time.Hour)
	}
	embeddingJSON, err := json.Marshal(append([]float32(nil), record.Embedding...))
	if err != nil {
		return memory.MemoryRetrievalQueryCacheRecord{}, fmt.Errorf("marshal memory retrieval query cache embedding: %w", err)
	}
	query := `
		INSERT INTO memory_retrieval_query_cache (
			id, project_id, dataset_id, purpose, embedding_model, embedding_dimensions, normalized_query_hash,
			query_text, embedding_json, created_at, updated_at, expires_at
		)
		VALUES (
			COALESCE(NULLIF($1, ''), 'memory_retrieval_query_cache_' || nextval('memory_retrieval_query_cache_id_seq')),
			$2, $3, $4, $5, $6, $7, $8, $9::jsonb, COALESCE($10, now()), $11, $12
		)
		ON CONFLICT (project_id, dataset_id, purpose, embedding_model, embedding_dimensions, normalized_query_hash)
		DO UPDATE SET
			query_text = EXCLUDED.query_text,
			embedding_json = EXCLUDED.embedding_json,
			updated_at = EXCLUDED.updated_at,
			expires_at = EXCLUDED.expires_at
		RETURNING id, project_id, dataset_id, purpose, embedding_model, embedding_dimensions, normalized_query_hash,
			query_text, embedding_json, created_at, updated_at, expires_at
	`
	var out memory.MemoryRetrievalQueryCacheRecord
	var returnedEmbeddingJSON []byte
	if err := s.db.QueryRowContext(context.Background(), query,
		record.ID,
		record.ProjectID,
		record.DatasetID,
		record.Purpose,
		record.EmbeddingModel,
		record.EmbeddingDimensions,
		record.NormalizedQueryHash,
		record.QueryText,
		embeddingJSON,
		record.CreatedAt,
		now,
		record.ExpiresAt,
	).Scan(
		&out.ID,
		&out.ProjectID,
		&out.DatasetID,
		&out.Purpose,
		&out.EmbeddingModel,
		&out.EmbeddingDimensions,
		&out.NormalizedQueryHash,
		&out.QueryText,
		&returnedEmbeddingJSON,
		&out.CreatedAt,
		&out.UpdatedAt,
		&out.ExpiresAt,
	); err != nil {
		return memory.MemoryRetrievalQueryCacheRecord{}, err
	}
	if len(returnedEmbeddingJSON) > 0 {
		if err := json.Unmarshal(returnedEmbeddingJSON, &out.Embedding); err != nil {
			return memory.MemoryRetrievalQueryCacheRecord{}, fmt.Errorf("unmarshal memory retrieval query cache embedding: %w", err)
		}
	}
	out.Embedding = append([]float32(nil), out.Embedding...)
	return out, nil
}

func (s *PostgresStore) SearchMemoryEmbeddings(query memory.MemoryRetrievalQuery) ([]memory.MemoryRetrievalResult, error) {
	if strings.TrimSpace(query.ProjectID) == "" {
		return nil, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(query.ProjectID); err != nil {
		return nil, err
	}
	limit := memoryRetrievalLimit(query.Limit)
	candidateLimit := memorySearchCandidateLimit(limit)
	embeddingModel := strings.TrimSpace(query.EmbeddingModel)
	embeddingDimensions := query.EmbeddingDimensions
	if embeddingDimensions <= 0 && len(query.Embedding) > 0 {
		embeddingDimensions = len(query.Embedding)
	}
	orderBy := "updated_at DESC"
	args := []any{
		query.CrossProjectOK,
		query.ProjectID,
		query.DatasetID,
		candidateLimit,
		embeddingModel,
		embeddingDimensions,
	}
	if len(query.Embedding) > 0 {
		vectorText, err := postgresVectorLiteral(query.Embedding)
		if err != nil {
			return nil, fmt.Errorf("format memory retrieval query vector: %w", err)
		}
		orderBy = "embedding <=> $7::vector ASC, updated_at DESC"
		args = append(args, vectorText)
	}
	querySQL := fmt.Sprintf(`
		SELECT %s
		FROM agent_memory_embeddings
		WHERE ($1 OR project_id = $2)
			AND ($3 = '' OR $1 OR dataset_id = $3)
			AND ($5 = '' OR embedding_model = $5)
			AND ($6 = 0 OR embedding_dimensions = $6)
		ORDER BY %s
		LIMIT $4
	`, memoryEmbeddingSelectColumns(), orderBy)
	rows, err := s.db.QueryContext(
		context.Background(),
		querySQL,
		args...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	candidates := []memoryRetrievalCandidate{}
	for rows.Next() {
		record, err := scanMemoryEmbeddingRecord(rows)
		if err != nil {
			return nil, err
		}
		if !memoryEmbeddingMatchesQuery(record, query) {
			continue
		}
		semantic, structured, score, reason := scoreMemoryEmbedding(record, query)
		candidates = append(candidates, memoryRetrievalCandidate{
			result: memory.MemoryRetrievalResult{
				SourceTable:     record.SourceTable,
				SourceID:        record.SourceID,
				ProjectID:       record.ProjectID,
				DatasetID:       record.DatasetID,
				Kind:            record.Kind,
				Score:           score,
				SemanticScore:   semantic,
				StructuredScore: structured,
				RetrievalReason: reason,
				SummaryCard:     copyAnyMap(record.SummaryCard),
				Metadata:        copyAnyMap(record.Metadata),
			},
			updatedAt: record.UpdatedAt,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sortMemoryRetrievalCandidates(candidates)
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	out := make([]memory.MemoryRetrievalResult, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, candidate.result)
	}
	return out, nil
}

func (s *PostgresStore) CountMemoryEmbeddings(projectID string, datasetID string, embeddingModel string) (int, error) {
	if err := s.requireProject(projectID); err != nil {
		return 0, err
	}
	const query = `
		SELECT COUNT(*)
		FROM agent_memory_embeddings
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR embedding_model = $3)
	`
	var count int
	if err := s.db.QueryRowContext(context.Background(), query, projectID, datasetID, embeddingModel).Scan(&count); err != nil {
		return 0, err
	}
	return count, nil
}

func (s *PostgresStore) ListUnembeddedMemorySources(projectID string, embeddingModel string, embeddingDimensions int, limit int) ([]memory.EmbeddableMemoryCard, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit = unembeddedMemorySourceLimit(limit)
	candidates := []embeddableMemorySourceCandidate{}
	embedded, err := s.listEmbeddedMemorySourceStates(projectID, embeddingModel)
	if err != nil {
		return nil, err
	}

	const memoryQuery = `
		SELECT id, invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags, created_at
		FROM agent_memory_records records
		WHERE records.project_id = $1
		ORDER BY records.created_at DESC
		LIMIT $2
	`
	memoryRows, err := s.db.QueryContext(context.Background(), memoryQuery, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer memoryRows.Close()
	for memoryRows.Next() {
		record, err := scanAgentMemoryRecord(memoryRows)
		if err != nil {
			return nil, err
		}
		card, ok := memory.BuildAgentMemoryCard(record)
		if !ok || !memoryCardShouldBeIndexed(card) || !memorySourceNeedsEmbedding(card, embedded, embeddingDimensions) {
			continue
		}
		candidates = append(candidates, embeddableMemorySourceCandidate{card: card, createdAt: record.CreatedAt})
	}
	if err := memoryRows.Err(); err != nil {
		return nil, err
	}

	scorecardQuery := fmt.Sprintf(`
		SELECT %s
		FROM strategy_scorecards scorecards
		WHERE scorecards.project_id = $1
		ORDER BY scorecards.created_at DESC
		LIMIT $2
	`, strategyScorecardSelectColumns())
	scorecardRows, err := s.db.QueryContext(context.Background(), scorecardQuery, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer scorecardRows.Close()
	for scorecardRows.Next() {
		scorecard, err := scanStrategyScorecard(scorecardRows)
		if err != nil {
			return nil, err
		}
		card, ok := memory.BuildStrategyScorecardCard(scorecard)
		if !ok || !memoryCardShouldBeIndexed(card) || !memorySourceNeedsEmbedding(card, embedded, embeddingDimensions) {
			continue
		}
		candidates = append(candidates, embeddableMemorySourceCandidate{card: card, createdAt: scorecard.CreatedAt})
	}
	if err := scorecardRows.Err(); err != nil {
		return nil, err
	}

	const datasetQuery = `
		SELECT id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
		FROM datasets
		WHERE project_id = $1
			AND status = $2
		ORDER BY profiled_at DESC NULLS LAST, created_at DESC
		LIMIT $3
	`
	datasetRows, err := s.db.QueryContext(context.Background(), datasetQuery, projectID, datasets.StatusProfiled, limit)
	if err != nil {
		return nil, err
	}
	defer datasetRows.Close()
	for datasetRows.Next() {
		dataset, err := scanDataset(datasetRows)
		if err != nil {
			return nil, err
		}
		card, ok := memory.BuildDatasetProfileCard(dataset)
		if !ok || !memoryCardShouldBeIndexed(card) || !memorySourceNeedsEmbedding(card, embedded, embeddingDimensions) {
			continue
		}
		createdAt := dataset.CreatedAt
		if dataset.ProfiledAt != nil {
			createdAt = *dataset.ProfiledAt
		}
		candidates = append(candidates, embeddableMemorySourceCandidate{card: card, createdAt: createdAt})
	}
	if err := datasetRows.Err(); err != nil {
		return nil, err
	}

	visualAnalysisQuery := fmt.Sprintf(`
		SELECT %s
		FROM dataset_visual_analyses
		WHERE project_id = $1
			AND validation_status = $2
		ORDER BY created_at DESC
		LIMIT $3
	`, datasetVisualAnalysisSelectColumns())
	visualRows, err := s.db.QueryContext(context.Background(), visualAnalysisQuery, projectID, datasets.VisualValidationStatusAccepted, limit)
	if err != nil {
		return nil, err
	}
	defer visualRows.Close()
	for visualRows.Next() {
		analysis, err := scanDatasetVisualAnalysis(visualRows)
		if err != nil {
			return nil, err
		}
		card, ok := memory.BuildDatasetVisualAnalysisCard(analysis)
		if ok && memoryCardShouldBeIndexed(card) && memorySourceNeedsEmbedding(card, embedded, embeddingDimensions) {
			candidates = append(candidates, embeddableMemorySourceCandidate{card: card, createdAt: analysis.CreatedAt})
		}
		for _, card := range memory.BuildDatasetPreprocessingHypothesisCards(analysis) {
			if !memoryCardShouldBeIndexed(card) || !memorySourceNeedsEmbedding(card, embedded, embeddingDimensions) {
				continue
			}
			candidates = append(candidates, embeddableMemorySourceCandidate{card: card, createdAt: analysis.CreatedAt})
		}
	}
	if err := visualRows.Err(); err != nil {
		return nil, err
	}

	sortEmbeddableMemorySourceCandidates(candidates)
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	out := make([]memory.EmbeddableMemoryCard, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, candidate.card)
	}
	return out, nil
}

func (s *PostgresStore) listEmbeddedMemorySourceStates(projectID string, embeddingModel string) (map[string]memory.MemoryEmbeddingRecord, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT source_table, source_id, embedding_dimensions, embedding_text_hash
		FROM agent_memory_embeddings
		WHERE project_id = $1
			AND ($2 = '' OR embedding_model = $2)
	`, projectID, embeddingModel)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := map[string]memory.MemoryEmbeddingRecord{}
	for rows.Next() {
		var sourceTable string
		var sourceID string
		var embeddingDimensions int
		var embeddingTextHash string
		if err := rows.Scan(&sourceTable, &sourceID, &embeddingDimensions, &embeddingTextHash); err != nil {
			return nil, normalizeSQLError(err)
		}
		out[memoryEmbeddingSourceKey(sourceTable, sourceID)] = memory.MemoryEmbeddingRecord{
			SourceTable:         sourceTable,
			SourceID:            sourceID,
			EmbeddingDimensions: embeddingDimensions,
			EmbeddingTextHash:   embeddingTextHash,
		}
	}
	return out, rows.Err()
}
