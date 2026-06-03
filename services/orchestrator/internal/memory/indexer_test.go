package memory

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/embeddings"
)

var errIndexerTestNotFound = errors.New("not found")

func TestMemoryIndexerIndexesValidCardsAndSkipsMalformedOrUnsafeCards(t *testing.T) {
	store := newIndexerTestStore([]EmbeddableMemoryCard{
		validPlanningOutcomeCard("memory_valid"),
		{
			CardVersion: MemoryCardVersion,
			SourceTable: SourceAgentMemoryRecord,
			ProjectID:   "project_1",
			Kind:        KindPlanningOutcome,
			Scope:       ScopeProject,
			Text:        "missing source id",
			Metadata:    map[string]any{"outcome": "improved_champion"},
		},
		{
			CardVersion: MemoryCardVersion,
			SourceTable: SourceAgentMemoryRecord,
			SourceID:    "memory_unsupported",
			ProjectID:   "project_1",
			Kind:        KindModelRanking,
			Scope:       ScopeProject,
			Text:        "model ranking is not part of the PR 3 indexing slice",
		},
		{
			CardVersion: MemoryCardVersion,
			SourceTable: SourceStrategyScorecard,
			SourceID:    "scorecard_pending",
			ProjectID:   "project_1",
			Kind:        "strategy_scorecard",
			Scope:       ScopeProject,
			Text:        "pending strategy scorecard",
			Metadata:    map[string]any{"outcome": "pending"},
		},
		{
			CardVersion: MemoryCardVersion,
			SourceTable: SourceAgentMemoryRecord,
			SourceID:    "memory_raw",
			ProjectID:   "project_1",
			Kind:        KindTrainingEvaluation,
			Scope:       ScopeJob,
			Text:        "raw output should not be embedded",
			SummaryCard: map[string]any{"raw_output": strings.Repeat("x", 100)},
			Metadata:    map[string]any{"outcome": "failed"},
		},
	})
	indexer := NewIndexer(store, embeddings.NewFakeClient(4), readyIndexerConfig())

	result, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() error = %v", err)
	}

	if result.SourcesListed != 5 {
		t.Fatalf("SourcesListed = %d, want 5", result.SourcesListed)
	}
	if result.Indexed != 1 || result.Skipped != 4 {
		t.Fatalf("Indexed/Skipped = %d/%d, want 1/4: %#v", result.Indexed, result.Skipped, result)
	}
	if len(store.records) != 1 {
		t.Fatalf("stored embeddings = %d, want 1", len(store.records))
	}
	record := store.records["agent_memory_records\x00memory_valid\x00fake-model"]
	if record.SourceID != "memory_valid" {
		t.Fatalf("stored record SourceID = %q", record.SourceID)
	}
	if record.EmbeddingText != "class balancing improved minority recall" {
		t.Fatalf("EmbeddingText = %q", record.EmbeddingText)
	}
	if len(record.Embedding) != 4 || record.EmbeddingDimensions != 4 {
		t.Fatalf("embedding dimensions = %d/%d, want 4", record.EmbeddingDimensions, len(record.Embedding))
	}
}

func TestMemoryIndexerBackfillIsIdempotent(t *testing.T) {
	store := newIndexerTestStore([]EmbeddableMemoryCard{validPlanningOutcomeCard("memory_once")})
	indexer := NewIndexer(store, embeddings.NewFakeClient(4), readyIndexerConfig())

	first, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() first error = %v", err)
	}
	second, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() second error = %v", err)
	}

	if first.Indexed != 1 {
		t.Fatalf("first Indexed = %d, want 1", first.Indexed)
	}
	if second.Indexed != 0 || second.SourcesListed != 0 {
		t.Fatalf("second result = %#v, want no remaining sources", second)
	}
	if store.upsertCalls != 1 {
		t.Fatalf("upsertCalls = %d, want 1", store.upsertCalls)
	}
}

func TestMemoryIndexerBackfillCapsAutomaticBatchSize(t *testing.T) {
	sources := make([]EmbeddableMemoryCard, 0, 60)
	for i := 0; i < 60; i++ {
		card := validPlanningOutcomeCard(fmt.Sprintf("memory_%d", i))
		card.Text = fmt.Sprintf("class balancing improved minority recall %d", i)
		sources = append(sources, card)
	}
	store := newIndexerTestStore(sources)
	indexer := NewIndexer(store, embeddings.NewFakeClient(4), embeddings.Config{
		EmbeddingsEnabled: true,
		Provider:          embeddings.ProviderLocal,
		BaseURL:           "http://embedding.local/v1",
		Model:             "fake-model",
		Dimensions:        4,
		BackfillLimit:     1000,
	})

	result, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() error = %v", err)
	}
	if result.SourcesListed != 50 {
		t.Fatalf("SourcesListed = %d, want capped batch of 50", result.SourcesListed)
	}
	if result.Indexed != 50 || result.Skipped != 0 {
		t.Fatalf("Indexed/Skipped = %d/%d, want 50/0: %#v", result.Indexed, result.Skipped, result)
	}
	if store.upsertCalls != 50 {
		t.Fatalf("upsertCalls = %d, want 50", store.upsertCalls)
	}
}

func TestMemoryIndexerSkipsUnchangedSourceCardsWithoutCallingProviderAgain(t *testing.T) {
	store := newIndexerTestStore(nil)
	client := &countingEmbeddingClient{delegate: embeddings.NewFakeClient(4)}
	indexer := NewIndexer(store, client, readyIndexerConfig())
	card := validPlanningOutcomeCard("memory_reused")

	first, err := indexer.IndexCards(context.Background(), []EmbeddableMemoryCard{card})
	if err != nil {
		t.Fatalf("IndexCards() first error = %v", err)
	}
	second, err := indexer.IndexCards(context.Background(), []EmbeddableMemoryCard{card})
	if err != nil {
		t.Fatalf("IndexCards() second error = %v", err)
	}

	if first.Indexed != 1 || second.Indexed != 0 {
		t.Fatalf("unexpected index results: first=%#v second=%#v", first, second)
	}
	if client.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", client.calls)
	}
	if len(store.usageEvents) != 2 {
		t.Fatalf("usage events = %d, want 2", len(store.usageEvents))
	}
	if store.usageEvents[1].ProviderCallCount != 0 || !store.usageEvents[1].Skipped {
		t.Fatalf("expected unchanged-card usage event to be skipped without provider call, got %#v", store.usageEvents[1])
	}
}

func TestMemoryIndexerSkipsSourceIndexingWhenDailyCapReached(t *testing.T) {
	store := newIndexerTestStore(nil)
	store.usageEvents = append(store.usageEvents, MemoryEmbeddingUsageEvent{
		ProjectID:         "project_1",
		Purpose:           EmbeddingUsagePurposeSourceIndex,
		ProviderCallCount: 1,
		CreatedAt:         time.Now().UTC(),
	})
	client := &countingEmbeddingClient{delegate: embeddings.NewFakeClient(4)}
	indexer := NewIndexer(store, client, embeddings.Config{
		EmbeddingsEnabled: true,
		Provider:          embeddings.ProviderLocal,
		BaseURL:           "http://embedding.local/v1",
		Model:             "fake-model",
		Dimensions:        4,
		MaxCallsPerDay:    1,
	})
	card := validPlanningOutcomeCard("memory_capped")

	result, err := indexer.IndexCards(context.Background(), []EmbeddableMemoryCard{card})
	if err != nil {
		t.Fatalf("IndexCards() error = %v", err)
	}
	if result.Indexed != 0 || result.Skipped != 1 {
		t.Fatalf("unexpected index result: %#v", result)
	}
	if client.calls != 0 {
		t.Fatalf("provider calls = %d, want 0", client.calls)
	}
	if len(store.usageEvents) != 2 || !store.usageEvents[1].Skipped {
		t.Fatalf("expected cap skip usage event, got %#v", store.usageEvents)
	}
	if got := store.usageEvents[1].SkipReason; !strings.Contains(got, "cap reached") {
		t.Fatalf("skip reason = %q, want cap reached", got)
	}
}

func TestMemoryIndexerMissingConfigNoopsWithoutStoreAccess(t *testing.T) {
	store := newIndexerTestStore([]EmbeddableMemoryCard{validPlanningOutcomeCard("memory_valid")})
	indexer := NewIndexer(store, embeddings.NewFakeClient(4), embeddings.Config{
		EmbeddingsEnabled: true,
		Provider:          embeddings.ProviderOpenAI,
		BaseURL:           embeddings.DefaultOpenAIBaseURL,
		Dimensions:        4,
	})

	result, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() error = %v", err)
	}
	if !result.Disabled || !strings.Contains(result.NoopReason, "MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL") {
		t.Fatalf("result = %#v, want missing-model no-op", result)
	}
	if store.listCalls != 0 || store.upsertCalls != 0 {
		t.Fatalf("store calls = list %d upsert %d, want 0/0", store.listCalls, store.upsertCalls)
	}
}

func TestMemoryIndexerDisabledNoopsWithoutClientOrStore(t *testing.T) {
	indexer := NewIndexer(nil, nil, embeddings.Config{})

	result, err := indexer.BackfillProject(context.Background(), "project_1")
	if err != nil {
		t.Fatalf("BackfillProject() error = %v", err)
	}
	if !result.Disabled || result.NoopReason != embeddings.ErrDisabled.Error() {
		t.Fatalf("result = %#v, want disabled no-op", result)
	}
}

type indexerTestStore struct {
	sources     []EmbeddableMemoryCard
	records     map[string]MemoryEmbeddingRecord
	usageEvents []MemoryEmbeddingUsageEvent
	listCalls   int
	upsertCalls int
}

func newIndexerTestStore(sources []EmbeddableMemoryCard) *indexerTestStore {
	return &indexerTestStore{
		sources: sources,
		records: map[string]MemoryEmbeddingRecord{},
	}
}

func (s *indexerTestStore) ListUnembeddedMemorySources(projectID string, embeddingModel string, embeddingDimensions int, limit int) ([]EmbeddableMemoryCard, error) {
	s.listCalls++
	out := []EmbeddableMemoryCard{}
	for _, card := range s.sources {
		if card.ProjectID != projectID || s.hasEmbedding(card, embeddingModel, embeddingDimensions) {
			continue
		}
		out = append(out, card)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

func (s *indexerTestStore) GetMemoryEmbeddingSourceState(projectID string, sourceTable string, sourceID string, embeddingModel string) (MemoryEmbeddingSourceState, error) {
	record, ok := s.records[embeddingSourceKey(sourceTable, sourceID, embeddingModel)]
	if !ok || record.ProjectID != projectID {
		return MemoryEmbeddingSourceState{}, errIndexerTestNotFound
	}
	return MemoryEmbeddingSourceState{
		SourceTable:         record.SourceTable,
		SourceID:            record.SourceID,
		ProjectID:           record.ProjectID,
		EmbeddingModel:      record.EmbeddingModel,
		EmbeddingDimensions: record.EmbeddingDimensions,
		EmbeddingTextHash:   record.EmbeddingTextHash,
		UpdatedAt:           record.UpdatedAt,
	}, nil
}

func (s *indexerTestStore) UpsertMemoryEmbedding(record MemoryEmbeddingRecord) (MemoryEmbeddingRecord, error) {
	s.upsertCalls++
	key := embeddingRecordKey(record)
	if existing, ok := s.records[key]; ok {
		record.ID = existing.ID
		record.CreatedAt = existing.CreatedAt
	} else {
		record.ID = "embedding_" + record.SourceID
		record.CreatedAt = time.Now().UTC()
	}
	if record.EmbeddingTextHash == "" {
		record.EmbeddingTextHash = HashEmbeddingText(record.EmbeddingText)
	}
	record.UpdatedAt = time.Now().UTC()
	s.records[key] = record
	return record, nil
}

func (s *indexerTestStore) CreateMemoryEmbeddingUsageEvent(event MemoryEmbeddingUsageEvent) (MemoryEmbeddingUsageEvent, error) {
	s.usageEvents = append(s.usageEvents, event)
	return event, nil
}

func (s *indexerTestStore) CountProjectMemoryEmbeddingUsageEvents(projectID string, purpose string, since time.Time) (int, error) {
	count := 0
	for _, event := range s.usageEvents {
		if event.ProjectID != projectID {
			continue
		}
		if purpose != "" && event.Purpose != purpose {
			continue
		}
		if !since.IsZero() && event.CreatedAt.Before(since) {
			continue
		}
		count += event.ProviderCallCount
	}
	return count, nil
}

func (s *indexerTestStore) GetMemoryRetrievalQueryCache(projectID string, datasetID string, purpose string, embeddingModel string, embeddingDimensions int, normalizedQueryHash string) (MemoryRetrievalQueryCacheRecord, error) {
	return MemoryRetrievalQueryCacheRecord{}, errIndexerTestNotFound
}

func (s *indexerTestStore) UpsertMemoryRetrievalQueryCache(record MemoryRetrievalQueryCacheRecord) (MemoryRetrievalQueryCacheRecord, error) {
	return record, nil
}

func (s *indexerTestStore) hasEmbedding(card EmbeddableMemoryCard, embeddingModel string, embeddingDimensions int) bool {
	record, ok := s.records[embeddingSourceKey(card.SourceTable, card.SourceID, embeddingModel)]
	if !ok {
		return false
	}
	if embeddingDimensions > 0 && record.EmbeddingDimensions != embeddingDimensions {
		return false
	}
	return record.EmbeddingTextHash == HashEmbeddingText(card.Text)
}

func embeddingRecordKey(record MemoryEmbeddingRecord) string {
	return embeddingSourceKey(record.SourceTable, record.SourceID, record.EmbeddingModel)
}

func embeddingSourceKey(sourceTable string, sourceID string, embeddingModel string) string {
	return sourceTable + "\x00" + sourceID + "\x00" + embeddingModel
}

func validPlanningOutcomeCard(sourceID string) EmbeddableMemoryCard {
	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceAgentMemoryRecord,
		SourceID:     sourceID,
		ProjectID:    "project_1",
		DatasetID:    "dataset_1",
		Kind:         KindPlanningOutcome,
		Scope:        ScopeDataset,
		Text:         "class balancing improved minority recall",
		SummaryCard:  map[string]any{"lesson": "class balancing helped"},
		Metadata:     map[string]any{"outcome": "improved_champion", "mechanism": "class_balancing", "accepted_for_vector_memory": true},
		QualityScore: 0.8,
		OutcomeScore: 1,
	}
}

func readyIndexerConfig() embeddings.Config {
	return embeddings.Config{
		EmbeddingsEnabled: true,
		Provider:          embeddings.ProviderLocal,
		BaseURL:           "http://embedding.local/v1",
		Model:             "fake-model",
		Dimensions:        4,
		BackfillLimit:     10,
	}
}

type countingEmbeddingClient struct {
	delegate embeddings.Client
	calls    int
}

func (c *countingEmbeddingClient) Embed(ctx context.Context, req embeddings.EmbedRequest) (embeddings.EmbedResult, error) {
	c.calls++
	return c.delegate.Embed(ctx, req)
}
