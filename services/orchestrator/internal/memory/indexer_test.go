package memory

import (
	"context"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/embeddings"
)

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
	listCalls   int
	upsertCalls int
}

func newIndexerTestStore(sources []EmbeddableMemoryCard) *indexerTestStore {
	return &indexerTestStore{
		sources: sources,
		records: map[string]MemoryEmbeddingRecord{},
	}
}

func (s *indexerTestStore) ListUnembeddedMemorySources(projectID string, limit int) ([]EmbeddableMemoryCard, error) {
	s.listCalls++
	out := []EmbeddableMemoryCard{}
	for _, card := range s.sources {
		if card.ProjectID != projectID || s.hasEmbedding(card) {
			continue
		}
		out = append(out, card)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
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
	record.UpdatedAt = time.Now().UTC()
	s.records[key] = record
	return record, nil
}

func (s *indexerTestStore) hasEmbedding(card EmbeddableMemoryCard) bool {
	sourceKey := card.SourceTable + "\x00" + card.SourceID + "\x00"
	for key := range s.records {
		if strings.HasPrefix(key, sourceKey) {
			return true
		}
	}
	return false
}

func embeddingRecordKey(record MemoryEmbeddingRecord) string {
	return record.SourceTable + "\x00" + record.SourceID + "\x00" + record.EmbeddingModel
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
