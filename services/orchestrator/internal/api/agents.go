package api

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/embeddings"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

func (s *Server) retrievePlannerMemory(ctx context.Context, input agents.ExperimentPlannerInput) []memory.MemoryRetrievalResult {
	if !memoryRetrievalEnabled() {
		return nil
	}
	query := memory.MemoryRetrievalQuery{
		ProjectID:      input.Project.ID,
		DatasetID:      input.Dataset.ID,
		AgentName:      agents.ExperimentPlannerAgentName,
		Purpose:        "experiment_planner",
		Text:           plannerMemoryRetrievalText(input),
		Kinds:          plannerMemoryRetrievalKinds(),
		Mechanisms:     plannerMemoryRetrievalMechanisms(input),
		DatasetTraits:  uniqueStrings(append(input.DatasetInsights.DatasetTraits, datasetProfileTraits(input.Dataset.Profile)...)),
		Objective:      strings.Join([]string{input.ObjectiveContext.PrimaryObjective, input.ObjectiveContext.GoalText, input.SourcePlan.TargetMetric}, " "),
		Limit:          memoryRetrievalMaxCards(),
		CrossProjectOK: memoryRetrievalCrossProjectOK(),
	}
	results, usage := s.searchRetrievedMemory(ctx, query, input.SourcePlan.ID, "")
	s.logMemoryRetrieval("planner", input.Project.ID, input.Dataset.ID, input.SourcePlan.ID, "", results, usage)
	if usage.LogOnly {
		return nil
	}
	return results
}

func (s *Server) retrieveTrainingMonitorMemory(ctx context.Context, plan plans.ExperimentPlan, job jobs.ExperimentJob, summary runs.TrainingRunSummary, objective agents.ProjectObjectiveContext) []memory.MemoryRetrievalResult {
	if !memoryRetrievalEnabled() {
		return nil
	}
	mechanism := strings.TrimSpace(jobConfigString(job.Config, "mechanism"))
	if mechanism == "" {
		mechanism = strings.TrimSpace(jobConfigString(job.Config, "intervention"))
	}
	query := memory.MemoryRetrievalQuery{
		ProjectID:      job.ProjectID,
		DatasetID:      summary.DatasetID,
		AgentName:      agents.TrainingMonitorAgentName,
		Purpose:        "training_monitor",
		Text:           trainingMonitorMemoryRetrievalText(plan, job, summary, objective),
		Kinds:          []string{memory.KindTrainingEvaluation, memory.KindPlanningOutcome, "strategy_scorecard"},
		Mechanisms:     uniqueStrings([]string{mechanism}),
		Objective:      strings.Join([]string{objective.PrimaryObjective, objective.GoalText, plan.TargetMetric}, " "),
		Limit:          minInt(memoryRetrievalMaxCards(), trainingMonitorMemoryRetrievalMaxCards()),
		CrossProjectOK: memoryRetrievalCrossProjectOK(),
	}
	results, usage := s.searchRetrievedMemory(ctx, query, plan.ID, job.ID)
	s.logMemoryRetrieval("training_monitor", job.ProjectID, summary.DatasetID, plan.ID, job.ID, results, usage)
	if usage.LogOnly {
		return nil
	}
	return results
}

func (s *Server) searchRetrievedMemory(ctx context.Context, query memory.MemoryRetrievalQuery, planID string, jobID string) ([]memory.MemoryRetrievalResult, memory.MemoryEmbeddingUsageEvent) {
	query.Text = strings.TrimSpace(query.Text)
	usage := memory.MemoryEmbeddingUsageEvent{
		ProjectID:           strings.TrimSpace(query.ProjectID),
		DatasetID:           strings.TrimSpace(query.DatasetID),
		PlanID:              strings.TrimSpace(planID),
		JobID:               strings.TrimSpace(jobID),
		Purpose:             memory.EmbeddingUsagePurposeRetrievalQuery,
		RetrievalPurpose:    strings.TrimSpace(query.Purpose),
		EmbeddingModel:      strings.TrimSpace(query.EmbeddingModel),
		EmbeddingDimensions: query.EmbeddingDimensions,
		InputBytes:          len([]byte(query.Text)),
		QueryHash:           memory.HashRetrievalQueryText(query.Text),
		Metadata: map[string]any{
			"agent_name":       strings.TrimSpace(query.AgentName),
			"cross_project_ok": query.CrossProjectOK,
			"limit":            query.Limit,
		},
	}
	if query.ProjectID == "" || query.Text == "" {
		usage.Skipped = true
		usage.SkipReason = "project_id and text are required"
		return nil, usage
	}
	if query.Limit <= 0 {
		query.Limit = memoryRetrievalMaxCards()
	}
	usage.Metadata["limit"] = query.Limit
	query, usage = s.withMemoryQueryEmbedding(ctx, query, usage)
	results, err := s.store.SearchMemoryEmbeddings(query)
	if err != nil {
		log.Printf("memory retrieval failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, err)
		usage.Skipped = true
		usage.SkipReason = err.Error()
		usage.LogOnly = memoryRetrievalLogOnly()
		usage.RetrievedCount = 0
		if _, recordErr := s.store.CreateMemoryEmbeddingUsageEvent(usage); recordErr != nil {
			log.Printf("memory retrieval usage event failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, recordErr)
		}
		return nil, usage
	}
	results = filterMemoryRetrievalResults(results, memoryRetrievalMinScore(), query.Limit)
	usage.LogOnly = memoryRetrievalLogOnly()
	usage.RetrievedCount = len(results)
	usage.Injected = !usage.LogOnly && len(results) > 0
	if _, recordErr := s.store.CreateMemoryEmbeddingUsageEvent(usage); recordErr != nil {
		log.Printf("memory retrieval usage event failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, recordErr)
	}
	return results, usage
}

func (s *Server) withMemoryQueryEmbedding(ctx context.Context, query memory.MemoryRetrievalQuery, usage memory.MemoryEmbeddingUsageEvent) (memory.MemoryRetrievalQuery, memory.MemoryEmbeddingUsageEvent) {
	config := embeddings.ConfigFromEnv()
	usage.EmbeddingModel = strings.TrimSpace(config.Model)
	usage.EmbeddingDimensions = config.Dimensions
	usage.LogOnly = memoryRetrievalLogOnly()
	usage.InputBytes = len([]byte(query.Text))
	usage.QueryHash = memory.HashRetrievalQueryText(query.Text)

	readyErr := config.ReadyForIndexing()
	shouldEmbed := config.EmbeddingsEnabled && readyErr == nil
	if !config.EmbeddingsEnabled {
		usage.Skipped = true
		usage.SkipReason = "retrieval embeddings disabled"
	} else if readyErr != nil {
		usage.Skipped = true
		usage.SkipReason = readyErr.Error()
	}
	if usage.LogOnly && !memoryRetrievalLogOnlyEmbeddings() {
		shouldEmbed = false
		usage.Skipped = true
		usage.SkipReason = "log-only retrieval uses lexical fallback"
	}
	if shouldEmbed {
		if count, err := s.store.CountMemoryEmbeddings(query.ProjectID, query.DatasetID, config.Model); err == nil && count < memoryRetrievalMinIndexedCards() {
			shouldEmbed = false
			usage.Skipped = true
			usage.SkipReason = "too few indexed cards for semantic retrieval"
		}
	}
	if shouldEmbed && memoryRetrievalCapReached(s.store, query.ProjectID) {
		shouldEmbed = false
		usage.Skipped = true
		usage.SkipReason = "retrieval embedding call cap reached for today"
	}
	if !shouldEmbed {
		return query, usage
	}

	normalizedQuery := memory.NormalizeRetrievalQueryText(query.Text)
	queryHash := memory.HashRetrievalQueryText(normalizedQuery)
	usage.QueryHash = queryHash
	if queryHash != "" {
		if cached, err := s.store.GetMemoryRetrievalQueryCache(query.ProjectID, query.DatasetID, usage.RetrievalPurpose, config.Model, config.Dimensions, queryHash); err == nil {
			query.EmbeddingModel = cached.EmbeddingModel
			query.EmbeddingDimensions = cached.EmbeddingDimensions
			query.Embedding = append([]float32(nil), cached.Embedding...)
			usage.Cached = true
			return query, usage
		}
	}

	result, err := embeddings.NewClient(config).Embed(ctx, embeddings.EmbedRequest{
		Model:      config.Model,
		Text:       query.Text,
		Dimensions: config.Dimensions,
	})
	if err != nil {
		log.Printf("memory retrieval embedding failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, err)
		usage.Skipped = true
		usage.SkipReason = err.Error()
		return query, usage
	}
	query.EmbeddingModel = result.Model
	query.EmbeddingDimensions = result.Dimensions
	query.Embedding = result.Vector
	usage.ProviderCallCount = 1
	usage.ProviderUsage = result.Usage
	if ttl := memoryRetrievalQueryCacheTTL(); ttl > 0 && queryHash != "" {
		if _, err := s.store.UpsertMemoryRetrievalQueryCache(memory.MemoryRetrievalQueryCacheRecord{
			ProjectID:           query.ProjectID,
			DatasetID:           query.DatasetID,
			Purpose:             usage.RetrievalPurpose,
			EmbeddingModel:      query.EmbeddingModel,
			EmbeddingDimensions: query.EmbeddingDimensions,
			NormalizedQueryHash: queryHash,
			QueryText:           normalizedQuery,
			Embedding:           append([]float32(nil), result.Vector...),
			ExpiresAt:           time.Now().UTC().Add(ttl),
		}); err != nil {
			log.Printf("memory retrieval query cache upsert failed for project %s purpose %s: %v", query.ProjectID, query.Purpose, err)
		}
	}
	return query, usage
}

func (s *Server) logMemoryRetrieval(purpose string, projectID string, datasetID string, planID string, jobID string, results []memory.MemoryRetrievalResult, usage memory.MemoryEmbeddingUsageEvent) {
	payload := map[string]any{
		"purpose":          purpose,
		"project_id":       projectID,
		"dataset_id":       datasetID,
		"retrieved_count":  len(results),
		"retrieved_cards":  memoryRetrievalDiagnosticCards(results),
		"log_only":         usage.LogOnly,
		"injected":         usage.Injected,
		"cached":           usage.Cached,
		"skipped":          usage.Skipped,
		"skip_reason":      usage.SkipReason,
		"embedding_model":  usage.EmbeddingModel,
		"embedding_dim":    usage.EmbeddingDimensions,
		"input_bytes":      usage.InputBytes,
		"query_hash":       usage.QueryHash,
		"provider_calls":   usage.ProviderCallCount,
		"cross_project_ok": memoryRetrievalCrossProjectOK(),
	}
	if planID != "" {
		payload["plan_id"] = planID
	}
	if jobID != "" {
		payload["job_id"] = jobID
	}
	if usage.RetrievalPurpose != "" {
		payload["retrieval_purpose"] = usage.RetrievalPurpose
	}
	diagnostics.Event("info", "memory_retrieval", payload)

	message := fmt.Sprintf("Memory retrieval checked for %s; found %d candidate card(s).", purpose, len(results))
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventMemoryRetrievalLogged, message, payload); err != nil {
		log.Printf("memory retrieval execution event failed for project %s purpose %s: %v", projectID, purpose, err)
	}
}

func memoryRetrievalDiagnosticCards(results []memory.MemoryRetrievalResult) []map[string]any {
	limit := minInt(len(results), 12)
	out := make([]map[string]any, 0, limit)
	for _, result := range results[:limit] {
		card := map[string]any{
			"source_table":     result.SourceTable,
			"source_id":        result.SourceID,
			"project_id":       result.ProjectID,
			"dataset_id":       result.DatasetID,
			"kind":             result.Kind,
			"score":            roundDiagnosticFloat(result.Score),
			"semantic_score":   roundDiagnosticFloat(result.SemanticScore),
			"structured_score": roundDiagnosticFloat(result.StructuredScore),
			"retrieval_reason": compactDiagnosticMemoryText(result.RetrievalReason, 240),
		}
		if summary := memoryRetrievalDiagnosticSummary(result.SummaryCard); len(summary) > 0 {
			card["summary_card"] = summary
		}
		if metadata := memoryRetrievalDiagnosticSummary(result.Metadata); len(metadata) > 0 {
			card["metadata"] = metadata
		}
		out = append(out, card)
	}
	return out
}

func memoryRetrievalDiagnosticSummary(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	allowed := []string{
		"kind",
		"outcome",
		"outcome_status",
		"mechanism",
		"mechanism_group",
		"strategy_type",
		"intervention",
		"action",
		"model",
		"model_family",
		"target_metric",
		"lesson",
		"compact_lesson",
		"summary",
		"compact_summary",
		"recommendation_summary",
		"preprocessing_hypothesis",
		"dataset_traits",
		"training_dynamics",
		"dynamics",
	}
	out := map[string]any{}
	for _, key := range allowed {
		value, ok := values[key]
		if !ok {
			continue
		}
		if compact, ok := compactDiagnosticMemoryValue(value); ok {
			out[key] = compact
		}
		if len(out) >= 10 {
			break
		}
	}
	return out
}

func compactDiagnosticMemoryValue(value any) (any, bool) {
	switch typed := value.(type) {
	case string:
		text := compactDiagnosticMemoryText(typed, 300)
		return text, text != ""
	case []string:
		out := []string{}
		for _, item := range typed {
			text := compactDiagnosticMemoryText(item, 120)
			if text != "" {
				out = append(out, text)
			}
			if len(out) >= 8 {
				break
			}
		}
		return out, len(out) > 0
	case []any:
		out := []string{}
		for _, item := range typed {
			text := compactDiagnosticMemoryText(fmt.Sprint(item), 120)
			if text != "" {
				out = append(out, text)
			}
			if len(out) >= 8 {
				break
			}
		}
		return out, len(out) > 0
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, bool:
		return typed, true
	case float32:
		return roundDiagnosticFloat(float64(typed)), true
	case float64:
		return roundDiagnosticFloat(typed), true
	default:
		return nil, false
	}
}

func compactDiagnosticMemoryText(value string, limit int) string {
	text := activitySafeText(value, limit)
	if text == "" {
		return ""
	}
	return text
}

func (s *Server) indexMemoryCard(ctx context.Context, card memory.EmbeddableMemoryCard) {
	if !embeddings.ConfigFromEnv().EmbeddingsEnabled {
		return
	}
	result, err := memory.NewIndexer(s.store, nil, embeddings.ConfigFromEnv()).IndexCards(ctx, []memory.EmbeddableMemoryCard{card})
	if err != nil {
		log.Printf("memory card indexing failed for %s/%s: %v", card.SourceTable, card.SourceID, err)
		return
	}
	if result.Disabled && result.NoopReason != "" {
		log.Printf("memory card indexing skipped for %s/%s: %s", card.SourceTable, card.SourceID, result.NoopReason)
	}
}

func plannerMemoryRetrievalText(input agents.ExperimentPlannerInput) string {
	return strings.Join(uniqueStrings([]string{
		input.Project.Goal,
		input.Dataset.Name,
		input.DatasetInsights.Summary,
		input.DatasetInsights.TaskType,
		strings.Join(input.DatasetInsights.DatasetTraits, " "),
		input.SourcePlan.TargetMetric,
		input.ObjectiveContext.PrimaryObjective,
		input.ObjectiveContext.GoalText,
		strings.Join(input.DeterministicDiagnosis.RecommendedFailureModes, " "),
		strings.Join(input.DeterministicDiagnosis.Evidence, " "),
		plannerChampionMemoryText(input.CurrentChampion),
	}), "\n")
}

func trainingMonitorMemoryRetrievalText(plan plans.ExperimentPlan, job jobs.ExperimentJob, summary runs.TrainingRunSummary, objective agents.ProjectObjectiveContext) string {
	return strings.Join(uniqueStrings([]string{
		summary.Model,
		memoryModelFamily(summary.Model),
		summary.Status,
		job.Template,
		jobConfigString(job.Config, "mechanism"),
		jobConfigString(job.Config, "intervention"),
		plan.TargetMetric,
		objective.PrimaryObjective,
		objective.GoalText,
		fmt.Sprintf("best_macro_f1 %.4f best_accuracy %.4f final_train_loss %.4f final_val_loss %.4f epochs %d", summary.BestMacroF1, summary.BestAccuracy, summary.FinalTrainLoss, summary.FinalValLoss, summary.EpochsCompleted),
	}), "\n")
}

func plannerMemoryRetrievalKinds() []string {
	return []string{
		"strategy_scorecard",
		memory.KindPlanningOutcome,
		memory.KindPlanningFeedback,
		memory.KindTrainingEvaluation,
		memory.KindDatasetProfile,
		memory.KindDatasetVisualAnalysis,
		memory.KindDatasetPreprocessingHypothesis,
	}
}

func plannerMemoryRetrievalMechanisms(input agents.ExperimentPlannerInput) []string {
	values := append([]string{}, input.DeterministicDiagnosis.RecommendedFailureModes...)
	for _, scorecard := range input.StrategyScorecards {
		values = append(values, scorecard.Mechanism)
	}
	return uniqueStrings(values)
}

func plannerChampionMemoryText(champion *agents.ExperimentChampion) string {
	if champion == nil {
		return ""
	}
	return strings.Join(uniqueStrings([]string{champion.Model, champion.TargetMetric, fmt.Sprintf("score %.4f", champion.Score)}), " ")
}

func filterMemoryRetrievalResults(results []memory.MemoryRetrievalResult, minScore float64, limit int) []memory.MemoryRetrievalResult {
	out := []memory.MemoryRetrievalResult{}
	for _, result := range results {
		if result.Score < minScore {
			continue
		}
		out = append(out, result)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func memoryRetrievalEnabled() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", false)
}

func memoryRetrievalCrossProjectOK() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_CROSS_PROJECT_ENABLED", false)
}

func memoryRetrievalLogOnly() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", true)
}

func memoryRetrievalLogOnlyEmbeddings() bool {
	return envFlag("MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS", false)
}

func memoryRetrievalMaxCards() int {
	return envInt("MODEL_EXPRESS_MEMORY_RETRIEVAL_MAX_CARDS", memoryRetrievalDefaultMaxCards, 1, 50)
}

func trainingMonitorMemoryRetrievalMaxCards() int {
	return minInt(memoryRetrievalMaxCards(), 8)
}

func memoryRetrievalMinScore() float64 {
	return envFloat("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", memoryRetrievalDefaultMinScore, 0, 1)
}

func memoryRetrievalMinIndexedCards() int {
	return envInt("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_INDEXED_CARDS", 3, 1, 500)
}

func memoryRetrievalQueryCacheTTL() time.Duration {
	seconds := envInt("MODEL_EXPRESS_MEMORY_RETRIEVAL_QUERY_CACHE_TTL_SECONDS", 3600, 0, 24*3600)
	if seconds <= 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}

func memoryRetrievalCapReached(storer interface {
	CountProjectMemoryEmbeddingUsageEvents(projectID string, purpose string, since time.Time) (int, error)
}, projectID string) bool {
	config := embeddings.ConfigFromEnv()
	if config.MaxCallsPerDay <= 0 {
		return false
	}
	since := time.Now().UTC().Truncate(24 * time.Hour)
	count, err := storer.CountProjectMemoryEmbeddingUsageEvents(projectID, memory.EmbeddingUsagePurposeRetrievalQuery, since)
	if err != nil {
		return false
	}
	return count >= config.MaxCallsPerDay
}

func memoryModelFamily(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.Contains(normalized, "mobilenet"):
		return "mobilenet"
	case strings.Contains(normalized, "efficientnet"):
		return "efficientnet"
	case strings.Contains(normalized, "resnet"):
		return "resnet"
	case strings.Contains(normalized, "regnet"):
		return "regnet"
	case strings.Contains(normalized, "convnext"):
		return "convnext"
	case strings.Contains(normalized, "swin"):
		return "swin"
	case strings.Contains(normalized, "vit"):
		return "vit"
	case strings.Contains(normalized, "yolo"):
		return "yolo"
	default:
		return normalized
	}
}

func (s *Server) listProjectAgentDecisions(c *gin.Context) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"decisions": agentDecisions})
}

func (s *Server) listProjectAgentMemoryRecords(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	records, err := s.store.ListProjectAgentMemoryRecords(c.Param("id"), memory.AgentMemoryFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		Kind:      c.Query("kind"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"records": records})
}

func (s *Server) backfillProjectMemoryEmbeddings(c *gin.Context) {
	projectID := c.Param("id")
	if _, err := s.store.GetProject(projectID); err != nil {
		writeStoreError(c, err)
		return
	}
	result, err := memory.NewIndexer(s.store, nil, embeddings.ConfigFromEnv()).BackfillProject(c.Request.Context(), projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, result)
}

func (s *Server) listProjectAgentInvocations(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	invocations, err := s.store.ListProjectAgentInvocations(c.Param("id"), memory.AgentInvocationFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		AgentName: c.Query("agent_name"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"invocations": agentInvocationResponseRows(invocations)})
}

func (s *Server) getProjectTelemetrySummary(c *gin.Context) {
	projectID := c.Param("id")
	if _, err := s.store.GetProject(projectID); err != nil {
		writeStoreError(c, err)
		return
	}

	limit := queryInt(c, "limit", 1000, 1, 5000)
	invocations, err := s.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{Limit: limit})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	usageEvents, err := s.store.ListProjectMemoryEmbeddingUsageEvents(projectID, limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"project_id":                    projectID,
		"generated_at":                  time.Now().UTC(),
		"limit":                         limit,
		"agent_invocations":             agentInvocationResponseRows(invocations),
		"memory_embedding_usage_events": usageEvents,
	})
}

type agentInvocationSummary struct {
	ID                string    `json:"id"`
	ProjectID         string    `json:"project_id"`
	DatasetID         string    `json:"dataset_id,omitempty"`
	PlanID            string    `json:"plan_id,omitempty"`
	JobID             string    `json:"job_id,omitempty"`
	AgentName         string    `json:"agent_name"`
	AgentVersion      string    `json:"agent_version,omitempty"`
	PromptVersion     string    `json:"prompt_version,omitempty"`
	Provider          string    `json:"provider,omitempty"`
	Model             string    `json:"model,omitempty"`
	ValidationStatus  string    `json:"validation_status"`
	ValidationError   string    `json:"validation_error,omitempty"`
	AcceptedForMemory bool      `json:"accepted_for_memory"`
	CreatedAt         time.Time `json:"created_at"`
}

func agentInvocationResponseRows(invocations []memory.AgentInvocation) any {
	if envFlag("MODEL_EXPRESS_ENABLE_RAW_AGENT_AUDIT", false) {
		return invocations
	}
	out := make([]agentInvocationSummary, 0, len(invocations))
	for _, invocation := range invocations {
		out = append(out, agentInvocationSummary{
			ID:                invocation.ID,
			ProjectID:         invocation.ProjectID,
			DatasetID:         invocation.DatasetID,
			PlanID:            invocation.PlanID,
			JobID:             invocation.JobID,
			AgentName:         invocation.AgentName,
			AgentVersion:      invocation.AgentVersion,
			PromptVersion:     invocation.PromptVersion,
			Provider:          invocation.Provider,
			Model:             invocation.Model,
			ValidationStatus:  invocation.ValidationStatus,
			ValidationError:   invocation.ValidationError,
			AcceptedForMemory: invocation.AcceptedForMemory,
			CreatedAt:         invocation.CreatedAt,
		})
	}
	return out
}

func (s *Server) listProjectStrategyScorecards(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	scorecards, err := s.store.ListProjectStrategyScorecards(c.Param("id"), limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"scorecards": scorecards})
}
