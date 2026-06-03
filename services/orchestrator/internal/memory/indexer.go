package memory

import (
	"context"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/embeddings"
)

const (
	MaxEmbeddingTextRunes     = 12000
	maxCardMapEntries         = 100
	maxCardSliceEntries       = 100
	maxAutomaticBackfillLimit = 50
)

type MemoryEmbeddingStore interface {
	ListUnembeddedMemorySources(projectID string, embeddingModel string, embeddingDimensions int, limit int) ([]EmbeddableMemoryCard, error)
	GetMemoryEmbeddingSourceState(projectID string, sourceTable string, sourceID string, embeddingModel string) (MemoryEmbeddingSourceState, error)
	UpsertMemoryEmbedding(record MemoryEmbeddingRecord) (MemoryEmbeddingRecord, error)
	CreateMemoryEmbeddingUsageEvent(event MemoryEmbeddingUsageEvent) (MemoryEmbeddingUsageEvent, error)
	CountProjectMemoryEmbeddingUsageEvents(projectID string, purpose string, since time.Time) (int, error)
}

type MemoryIndexer struct {
	store  MemoryEmbeddingStore
	client embeddings.Client
	config embeddings.Config
}

type MemoryIndexResult struct {
	Disabled      bool              `json:"disabled"`
	NoopReason    string            `json:"noop_reason,omitempty"`
	SourcesListed int               `json:"sources_listed"`
	Indexed       int               `json:"indexed"`
	Skipped       int               `json:"skipped"`
	Skips         []MemoryIndexSkip `json:"skips,omitempty"`
}

type MemoryIndexSkip struct {
	SourceTable string `json:"source_table,omitempty"`
	SourceID    string `json:"source_id,omitempty"`
	Kind        string `json:"kind,omitempty"`
	Reason      string `json:"reason"`
}

func NewIndexer(store MemoryEmbeddingStore, client embeddings.Client, config embeddings.Config) *MemoryIndexer {
	if client == nil {
		client = embeddings.NewClient(config)
	}
	return &MemoryIndexer{
		store:  store,
		client: client,
		config: config.Normalized(),
	}
}

func (i *MemoryIndexer) BackfillProject(ctx context.Context, projectID string) (MemoryIndexResult, error) {
	if reason := i.noopReason(); reason != "" {
		return MemoryIndexResult{Disabled: true, NoopReason: reason}, nil
	}
	projectID = strings.TrimSpace(projectID)
	if projectID == "" {
		return MemoryIndexResult{}, fmt.Errorf("project_id is required for memory embedding backfill")
	}
	limit := i.config.Normalized().BackfillLimit
	if limit <= 0 || limit > maxAutomaticBackfillLimit {
		limit = maxAutomaticBackfillLimit
	}
	sources, err := i.store.ListUnembeddedMemorySources(projectID, i.config.Model, i.config.Dimensions, limit)
	if err != nil {
		return MemoryIndexResult{}, err
	}
	result, err := i.IndexCards(ctx, sources)
	result.SourcesListed = len(sources)
	return result, err
}

func (i *MemoryIndexer) IndexCards(ctx context.Context, cards []EmbeddableMemoryCard) (MemoryIndexResult, error) {
	result := MemoryIndexResult{SourcesListed: len(cards)}
	if reason := i.noopReason(); reason != "" {
		result.Disabled = true
		result.NoopReason = reason
		return result, nil
	}
	config := i.config.Normalized()
	for _, card := range cards {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if reason := memoryCardIndexSkipReason(card); reason != "" {
			result.Skipped++
			result.Skips = append(result.Skips, MemoryIndexSkip{
				SourceTable: card.SourceTable,
				SourceID:    card.SourceID,
				Kind:        card.Kind,
				Reason:      reason,
			})
			_ = i.recordSourceIndexUsage(card, 0, false, true, false, reason, nil, MemoryEmbeddingRecord{})
			continue
		}

		if capExceeded, capReason := i.sourceIndexCapExceeded(card.ProjectID); capExceeded {
			result.Skipped++
			result.Skips = append(result.Skips, MemoryIndexSkip{
				SourceTable: card.SourceTable,
				SourceID:    card.SourceID,
				Kind:        card.Kind,
				Reason:      capReason,
			})
			_ = i.recordSourceIndexUsage(card, 0, false, true, true, capReason, nil, MemoryEmbeddingRecord{})
			continue
		}

		state, err := i.store.GetMemoryEmbeddingSourceState(card.ProjectID, card.SourceTable, card.SourceID, config.Model)
		if err == nil && state.EmbeddingDimensions == config.Dimensions && strings.EqualFold(state.EmbeddingTextHash, HashEmbeddingText(card.Text)) {
			result.Skipped++
			reason := "unchanged source card already embedded for model and dimensions"
			result.Skips = append(result.Skips, MemoryIndexSkip{
				SourceTable: card.SourceTable,
				SourceID:    card.SourceID,
				Kind:        card.Kind,
				Reason:      reason,
			})
			_ = i.recordSourceIndexUsage(card, 0, false, true, false, reason, nil, MemoryEmbeddingRecord{})
			continue
		}

		embedded, err := i.client.Embed(ctx, embeddings.EmbedRequest{
			Model:      config.Model,
			Text:       card.Text,
			Dimensions: config.Dimensions,
		})
		if err != nil {
			usageErr := i.recordSourceIndexUsage(card, 1, false, false, false, err.Error(), nil, MemoryEmbeddingRecord{})
			if usageErr != nil {
				return result, fmt.Errorf("record source indexing usage %s/%s: %w", card.SourceTable, card.SourceID, usageErr)
			}
			return result, fmt.Errorf("embed memory card %s/%s: %w", card.SourceTable, card.SourceID, err)
		}
		if len(embedded.Vector) == 0 {
			return result, fmt.Errorf("embed memory card %s/%s: provider returned empty vector", card.SourceTable, card.SourceID)
		}
		dimensions := embedded.Dimensions
		if dimensions <= 0 {
			dimensions = len(embedded.Vector)
		}
		if dimensions != len(embedded.Vector) {
			return result, fmt.Errorf("embed memory card %s/%s: dimensions = %d, vector length = %d", card.SourceTable, card.SourceID, dimensions, len(embedded.Vector))
		}
		model := strings.TrimSpace(embedded.Model)
		if model == "" {
			model = config.Model
		}

		record, err := i.store.UpsertMemoryEmbedding(MemoryEmbeddingRecord{
			SourceTable:         strings.TrimSpace(card.SourceTable),
			SourceID:            strings.TrimSpace(card.SourceID),
			ProjectID:           strings.TrimSpace(card.ProjectID),
			DatasetID:           strings.TrimSpace(card.DatasetID),
			PlanID:              strings.TrimSpace(card.PlanID),
			JobID:               strings.TrimSpace(card.JobID),
			InvocationID:        strings.TrimSpace(card.InvocationID),
			Kind:                strings.TrimSpace(card.Kind),
			Scope:               strings.TrimSpace(card.Scope),
			EmbeddingModel:      model,
			EmbeddingDimensions: dimensions,
			Embedding:           append([]float32(nil), embedded.Vector...),
			EmbeddingText:       strings.TrimSpace(card.Text),
			EmbeddingTextHash:   HashEmbeddingText(card.Text),
			SummaryCard:         copyCardMap(card.SummaryCard),
			Metadata:            copyCardMap(card.Metadata),
			QualityScore:        card.QualityScore,
			OutcomeScore:        card.OutcomeScore,
		})
		if err != nil {
			_ = i.recordSourceIndexUsage(card, 1, false, false, false, err.Error(), embedded.Usage, MemoryEmbeddingRecord{})
			return result, fmt.Errorf("upsert memory embedding %s/%s: %w", card.SourceTable, card.SourceID, err)
		}
		if usageErr := i.recordSourceIndexUsage(card, 1, false, false, false, "", embedded.Usage, record); usageErr != nil {
			return result, fmt.Errorf("record source indexing usage %s/%s: %w", card.SourceTable, card.SourceID, usageErr)
		}
		result.Indexed++
	}
	return result, nil
}

func (i *MemoryIndexer) noopReason() string {
	if i == nil {
		return "memory indexer is not configured"
	}
	if err := i.config.Normalized().ReadyForIndexing(); err != nil {
		return err.Error()
	}
	if i.client == nil {
		return "memory embedding client is not configured"
	}
	if i.store == nil {
		return "memory embedding store is not configured"
	}
	return ""
}

func (i *MemoryIndexer) sourceIndexCapExceeded(projectID string) (bool, string) {
	if i == nil || i.store == nil {
		return false, ""
	}
	projectID = strings.TrimSpace(projectID)
	if projectID == "" {
		return false, ""
	}
	config := i.config.Normalized()
	if config.MaxCallsPerDay <= 0 {
		return false, ""
	}
	since := time.Now().UTC().Truncate(24 * time.Hour)
	count, err := i.store.CountProjectMemoryEmbeddingUsageEvents(projectID, EmbeddingUsagePurposeSourceIndex, since)
	if err != nil {
		return false, ""
	}
	if count >= config.MaxCallsPerDay {
		return true, "source indexing embedding call cap reached for today"
	}
	return false, ""
}

func (i *MemoryIndexer) recordSourceIndexUsage(card EmbeddableMemoryCard, providerCallCount int, cached bool, skipped bool, budgetExceeded bool, skipReason string, providerUsage map[string]any, upserted MemoryEmbeddingRecord) error {
	if i == nil || i.store == nil {
		return nil
	}
	metadata := map[string]any{
		"provider_call_count": providerCallCount,
		"cached":              cached,
		"skipped":             skipped,
		"budget_exhausted":    budgetExceeded,
	}
	if len(providerUsage) > 0 {
		metadata["provider_usage"] = providerUsage
	}
	if upserted.ID != "" {
		metadata["embedding_id"] = upserted.ID
		metadata["embedding_model"] = upserted.EmbeddingModel
	}
	event := MemoryEmbeddingUsageEvent{
		ProjectID:           strings.TrimSpace(card.ProjectID),
		DatasetID:           strings.TrimSpace(card.DatasetID),
		PlanID:              strings.TrimSpace(card.PlanID),
		JobID:               strings.TrimSpace(card.JobID),
		InvocationID:        strings.TrimSpace(card.InvocationID),
		Purpose:             EmbeddingUsagePurposeSourceIndex,
		SourceTable:         strings.TrimSpace(card.SourceTable),
		SourceID:            strings.TrimSpace(card.SourceID),
		EmbeddingModel:      strings.TrimSpace(upserted.EmbeddingModel),
		EmbeddingDimensions: upserted.EmbeddingDimensions,
		InputBytes:          len([]byte(card.Text)),
		ProviderCallCount:   providerCallCount,
		Skipped:             skipped,
		SkipReason:          skipReason,
		SourceTextHash:      HashEmbeddingText(card.Text),
		ProviderUsage:       providerUsage,
		Metadata:            metadata,
	}
	if upserted.ID == "" {
		event.EmbeddingModel = strings.TrimSpace(i.config.Model)
		event.EmbeddingDimensions = i.config.Dimensions
	}
	if event.SourceTable == "" || event.SourceID == "" || event.ProjectID == "" {
		return nil
	}
	_, err := i.store.CreateMemoryEmbeddingUsageEvent(event)
	return err
}

func memoryCardIndexSkipReason(card EmbeddableMemoryCard) string {
	if strings.TrimSpace(card.CardVersion) != "" && strings.TrimSpace(card.CardVersion) != MemoryCardVersion {
		return "unsupported memory card version"
	}
	if strings.TrimSpace(card.SourceTable) == "" {
		return "source_table is required"
	}
	if strings.TrimSpace(card.SourceID) == "" {
		return "source_id is required"
	}
	if strings.TrimSpace(card.ProjectID) == "" {
		return "project_id is required"
	}
	if strings.TrimSpace(card.Kind) == "" {
		return "kind is required"
	}
	if strings.TrimSpace(card.Text) == "" {
		return "embedding text is required"
	}
	if runeCount(card.Text) > MaxEmbeddingTextRunes {
		return "embedding text exceeds compact card limit"
	}
	if !supportedMemoryCardKind(card) {
		return "unsupported memory card source or kind"
	}
	if !cardBoolDefault(card.Metadata, "accepted_for_vector_memory", true) {
		return "card is not accepted for vector memory"
	}
	if outcome := normalizedOutcome(card); outcome == "pending" || outcome == "invalidated" {
		return "card outcome is not eligible for indexing"
	}
	if hasUnsafeEmbeddingFields(card.SummaryCard) || hasUnsafeEmbeddingFields(card.Metadata) {
		return "card contains raw or unbounded fields"
	}
	if containsUnsafeURI(card.Text) {
		return "embedding text contains raw artifact URI"
	}
	return ""
}

func supportedMemoryCardKind(card EmbeddableMemoryCard) bool {
	sourceTable := strings.TrimSpace(card.SourceTable)
	kind := strings.TrimSpace(card.Kind)
	switch sourceTable {
	case SourceStrategyScorecard:
		return kind == "strategy_scorecard"
	case SourceAgentMemoryRecord:
		switch kind {
		case KindPlanningOutcome, KindPlanningFeedback, KindTrainingEvaluation, KindChampionFeedback:
			return true
		default:
			return false
		}
	case SourceDatasetProfile:
		return kind == KindDatasetProfile
	case SourceDatasetVisualAnalysis:
		return kind == KindDatasetVisualAnalysis
	case SourceDatasetPreprocessing:
		return kind == KindDatasetPreprocessingHypothesis
	default:
		return false
	}
}

func normalizedOutcome(card EmbeddableMemoryCard) string {
	for _, values := range []map[string]any{card.Metadata, card.SummaryCard} {
		for _, key := range []string{"outcome", "outcome_status"} {
			if outcome := strings.ToLower(strings.TrimSpace(fmt.Sprint(values[key]))); outcome != "" && outcome != "<nil>" {
				return outcome
			}
		}
	}
	return ""
}

func hasUnsafeEmbeddingFields(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		if len(typed) > maxCardMapEntries {
			return true
		}
		for key, nested := range typed {
			if unsafeEmbeddingKey(key) || hasUnsafeEmbeddingFields(nested) {
				return true
			}
		}
	case []any:
		if len(typed) > maxCardSliceEntries {
			return true
		}
		for _, nested := range typed {
			if hasUnsafeEmbeddingFields(nested) {
				return true
			}
		}
	case []string:
		return len(typed) > maxCardSliceEntries
	case []map[string]any:
		if len(typed) > maxCardSliceEntries {
			return true
		}
		for _, nested := range typed {
			if hasUnsafeEmbeddingFields(nested) {
				return true
			}
		}
	}
	return false
}

func unsafeEmbeddingKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	if normalized == "" || strings.Contains(normalized, "excluded") {
		return false
	}
	switch normalized {
	case "raw_prompt", "raw_output", "input_messages", "input_context", "full_context",
		"epoch_metrics", "full_epoch_arrays", "full_manifest", "manifest", "manifest_records",
		"image_uri", "image_uris", "image_url", "image_urls", "storage_uri", "storage_uris",
		"source_rows", "raw_preview", "raw_payload", "raw_profile", "visual_samples":
		return true
	}
	for _, fragment := range []string{
		"raw_prompt", "raw_output", "full_context", "full_manifest", "epoch_array",
		"image_uri", "image_url", "storage_uri",
	} {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	return false
}

func containsUnsafeURI(text string) bool {
	lower := strings.ToLower(text)
	for _, marker := range []string{"file://", "s3://", "gs://", "data:image/"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func cardBoolDefault(values map[string]any, key string, fallback bool) bool {
	value, ok := values[key]
	if !ok {
		return fallback
	}
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		switch strings.ToLower(strings.TrimSpace(typed)) {
		case "true", "yes", "1":
			return true
		case "false", "no", "0":
			return false
		default:
			return fallback
		}
	default:
		return fallback
	}
}

func runeCount(value string) int {
	count := 0
	for range value {
		count++
	}
	return count
}

func copyCardMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
