package memory

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/strategies"
)

const (
	SourceDatasetProfile        = "datasets"
	SourceDatasetVisualAnalysis = "dataset_visual_analyses"
	SourceDatasetPreprocessing  = "dataset_preprocessing_hypotheses"
)

const (
	KindDatasetProfile                 = "dataset_profile"
	KindDatasetVisualAnalysis          = "dataset_visual_analysis"
	KindDatasetPreprocessingHypothesis = "dataset_preprocessing_hypothesis"
)

type DistilledMemoryCard struct {
	SourceTable  string   `json:"source_table"`
	SourceID     string   `json:"source_id"`
	ProjectID    string   `json:"project_id"`
	DatasetID    string   `json:"dataset_id,omitempty"`
	PlanID       string   `json:"plan_id,omitempty"`
	JobID        string   `json:"job_id,omitempty"`
	InvocationID string   `json:"invocation_id,omitempty"`
	Kind         string   `json:"kind"`
	Outcome      string   `json:"outcome,omitempty"`
	Mechanism    string   `json:"mechanism,omitempty"`
	AppliesWhen  []string `json:"applies_when,omitempty"`
	Lesson       string   `json:"lesson"`
	EvidenceUsed []string `json:"evidence_used,omitempty"`
	Confidence   float64  `json:"confidence,omitempty"`
	ValueScore   float64  `json:"value_score,omitempty"`
	Summary      string   `json:"summary,omitempty"`
}

func NewAgentMemoryCard(record AgentMemoryRecord) EmbeddableMemoryCard {
	card, _ := BuildAgentMemoryCard(record)
	return card
}

func BuildAgentMemoryCard(record AgentMemoryRecord) (EmbeddableMemoryCard, bool) {
	payload := emptyMap(record.Payload)
	outcome := firstNonEmptyString(payload, "outcome_status", "outcome", "decision_type")
	lesson := firstNonEmptyString(payload, "lesson")
	mechanism := firstNonEmptyString(payload, "primary_mechanism", "mechanism", "mechanism_group")
	intervention := firstNonEmptyString(payload, "intervention")
	planningMode := firstNonEmptyString(payload, "planning_mode")
	hypothesis := firstNonEmptyString(payload, "hypothesis")
	decisionType := firstNonEmptyString(payload, "decision_type")
	evidenceUsed := stringsFromAny(payload["evidence_used"], 8)
	expectedFailures := stringsFromAny(payload["expected_failure_modes"], 8)
	changedVariables := stringsFromAny(payload["changed_variables"], 8)
	if mechanism == "" || intervention == "" {
		mechanism, intervention = mechanismInterventionFromPayload(payload, mechanism, intervention)
	}
	models := proposedModelsFromPayload(payload["proposed_experiments"])
	bestModel := bestModelFromPayload(payload["actual_best_run"])
	if bestModel == "" {
		bestModel = firstNonEmptyString(payload, "best_model", "model")
	}
	rejectedOptions := rejectedOptionSummaries(payload["rejected_options"])
	tags := cleanStrings(record.Tags, 12)

	summary := map[string]any{
		"source":        SourceAgentMemoryRecord,
		"memory_id":     record.ID,
		"agent_name":    record.AgentName,
		"kind":          record.Kind,
		"summary":       strings.TrimSpace(record.Summary),
		"tags":          tags,
		"outcome":       outcome,
		"decision_type": decisionType,
		"lesson":        lesson,
		"hypothesis":    hypothesis,
		"evidence":      evidenceUsed,
		"best_model":    bestModel,
		"models":        models,
		"rejections":    rejectedOptions,
		"planning_mode": planningMode,
		"mechanism":     mechanism,
		"intervention":  intervention,
	}
	removeEmpty(summary)

	metadata := map[string]any{
		"source_table":                  SourceAgentMemoryRecord,
		"source_id":                     record.ID,
		"invocation_id":                 record.InvocationID,
		"agent_name":                    record.AgentName,
		"memory_kind":                   record.Kind,
		"plan_id":                       record.PlanID,
		"job_id":                        record.JobID,
		"outcome":                       outcome,
		"decision_type":                 decisionType,
		"planning_mode":                 planningMode,
		"mechanism":                     mechanism,
		"intervention":                  intervention,
		"evidence_used":                 evidenceUsed,
		"expected_failure_modes":        expectedFailures,
		"changed_variables":             changedVariables,
		"best_model":                    bestModel,
		"models":                        models,
		"tags":                          tags,
		"expected_delta_vs_champion":    floatValue(payload["expected_delta_vs_champion"]),
		"actual_delta_vs_champion":      floatValue(payload["actual_delta_vs_champion"]),
		"total_cost_usd":                floatValue(payload["total_cost_usd"]),
		"total_runtime_seconds":         floatValue(payload["total_runtime_seconds"]),
		"rank_score":                    floatValue(payload["rank_score"]),
		"recommended_action_type":       recommendedActionType(payload["recommended_action"]),
		"rejected_option_count":         len(rejectedOptions),
		"accepted_for_vector_memory":    true,
		"raw_payload_excluded":          true,
		"embedding_card_schema_version": MemoryCardVersion,
	}
	removeEmpty(metadata)

	text := deterministicMemoryText([]textField{
		{"source", SourceAgentMemoryRecord},
		{"kind", record.Kind},
		{"agent", record.AgentName},
		{"summary", record.Summary},
		{"outcome", outcome},
		{"decision_type", decisionType},
		{"lesson", lesson},
		{"hypothesis", hypothesis},
		{"planning_mode", planningMode},
		{"mechanism", mechanism},
		{"intervention", intervention},
		{"evidence", strings.Join(evidenceUsed, " | ")},
		{"expected_failures", strings.Join(expectedFailures, " | ")},
		{"changed_variables", strings.Join(changedVariables, ", ")},
		{"best_model", bestModel},
		{"models", strings.Join(models, ", ")},
		{"tags", strings.Join(tags, ", ")},
		{"rejections", strings.Join(rejectedOptions, " | ")},
	})

	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceAgentMemoryRecord,
		SourceID:     record.ID,
		ProjectID:    record.ProjectID,
		DatasetID:    record.DatasetID,
		PlanID:       record.PlanID,
		JobID:        record.JobID,
		InvocationID: record.InvocationID,
		Kind:         record.Kind,
		Scope:        memoryScope(record.DatasetID, record.PlanID, record.JobID),
		Text:         text,
		SummaryCard:  summary,
		Metadata:     metadata,
		QualityScore: qualityScoreFromPayload(payload),
		OutcomeScore: outcomeScore(outcome),
	}, strings.TrimSpace(record.ID) != ""
}

func NewStrategyScorecardMemoryCard(scorecard strategies.StrategyScorecard) EmbeddableMemoryCard {
	card, _ := BuildStrategyScorecardCard(scorecard)
	return card
}

func BuildDistilledMemoryCardFromAgentMemoryRecord(record AgentMemoryRecord) (DistilledMemoryCard, bool) {
	payload := emptyMap(record.Payload)
	outcome := firstNonEmptyString(payload, "outcome_status", "outcome", "decision_type")
	lesson := firstNonEmptyString(payload, "lesson")
	mechanism := firstNonEmptyString(payload, "primary_mechanism", "mechanism", "mechanism_group")
	intervention := firstNonEmptyString(payload, "intervention")
	if mechanism == "" || intervention == "" {
		mechanism, intervention = mechanismInterventionFromPayload(payload, mechanism, intervention)
	}
	if lesson == "" {
		lesson = firstNonEmptyString(payload, "summary", "hypothesis", "expected_effect")
	}
	if lesson == "" {
		lesson = strings.TrimSpace(record.Summary)
	}
	if strings.TrimSpace(record.ID) == "" || strings.TrimSpace(lesson) == "" {
		return DistilledMemoryCard{}, false
	}

	appliesWhen := distilledMemoryAppliesWhenFromValues([]map[string]any{
		payload,
		{
			"planning_mode": firstNonEmptyString(payload, "planning_mode"),
			"task_type":     firstNonEmptyString(payload, "task_type"),
			"mechanism":     mechanism,
			"intervention":  intervention,
		},
	})
	evidenceUsed := distilledMemoryEvidenceIDs(record.ID, []map[string]any{payload}, 6)
	if len(evidenceUsed) == 0 {
		evidenceUsed = []string{record.ID}
	}
	summary := strings.TrimSpace(record.Summary)

	return DistilledMemoryCard{
		SourceTable:  SourceAgentMemoryRecord,
		SourceID:     record.ID,
		ProjectID:    record.ProjectID,
		DatasetID:    record.DatasetID,
		PlanID:       record.PlanID,
		JobID:        record.JobID,
		InvocationID: record.InvocationID,
		Kind:         record.Kind,
		Outcome:      outcome,
		Mechanism:    normalizeDistilledMemoryMechanism(mechanism),
		AppliesWhen:  appliesWhen,
		Lesson:       compactDistilledMemoryText(lesson, 220),
		EvidenceUsed: evidenceUsed,
		Confidence:   clampScore(qualityScoreFromPayload(payload)),
		ValueScore:   outcomeScore(outcome),
		Summary:      compactDistilledMemoryText(summary, 180),
	}, true
}

func BuildDistilledMemoryCardFromStrategyScorecard(scorecard strategies.StrategyScorecard) (DistilledMemoryCard, bool) {
	lesson := strings.TrimSpace(scorecard.Lesson)
	if lesson == "" {
		lesson = strings.TrimSpace(scorecard.ExpectedEffect)
	}
	if strings.TrimSpace(scorecard.ID) == "" || strings.TrimSpace(lesson) == "" {
		return DistilledMemoryCard{}, false
	}
	records := []map[string]any{
		{
			"diagnosis_triggers": scorecard.DiagnosisTriggers,
			"dataset_traits":     scorecard.DatasetTraits,
			"objective_profile":  scorecard.ObjectiveProfile,
			"planning_mode":      scorecard.PlanningMode,
			"strategy_type":      scorecard.StrategyType,
			"tags":               scorecard.Tags,
			"mechanism":          scorecard.Mechanism,
			"intervention":       scorecard.Intervention,
			"expected_effect":    scorecard.ExpectedEffect,
		},
		scorecard.ProposedChanges,
	}
	appliesWhen := distilledMemoryAppliesWhenFromValues(records)
	evidenceUsed := distilledMemoryEvidenceIDs(scorecard.ID, []map[string]any{
		{
			"source_decision_id": scorecard.SourceDecisionID,
			"source_plan_id":     scorecard.SourcePlanID,
			"followup_plan_id":   scorecard.FollowUpPlanID,
			"scorecard_id":       scorecard.ID,
		},
	}, 6)
	if len(evidenceUsed) == 0 {
		evidenceUsed = []string{scorecard.ID}
	}
	return DistilledMemoryCard{
		SourceTable:  SourceStrategyScorecard,
		SourceID:     scorecard.ID,
		ProjectID:    scorecard.ProjectID,
		DatasetID:    scorecard.DatasetID,
		PlanID:       scorecard.SourcePlanID,
		Kind:         "strategy_scorecard",
		Outcome:      scorecard.Outcome,
		Mechanism:    normalizeDistilledMemoryMechanism(scorecard.Mechanism),
		AppliesWhen:  appliesWhen,
		Lesson:       compactDistilledMemoryText(lesson, 220),
		EvidenceUsed: evidenceUsed,
		Confidence:   clampScore(scorecardQualityScore(scorecard)),
		ValueScore:   outcomeScore(scorecard.Outcome),
		Summary:      compactDistilledMemoryText(scorecard.StrategyType, 160),
	}, true
}

func BuildDistilledMemoryCardFromRetrievalResult(result MemoryRetrievalResult) (DistilledMemoryCard, bool) {
	summaryCard := emptyMap(result.SummaryCard)
	metadata := emptyMap(result.Metadata)
	records := []map[string]any{summaryCard, metadata}
	sourceID := strings.TrimSpace(result.SourceID)
	if sourceID == "" {
		sourceID = distilledMemoryFirstString(records, "source_id", "memory_id", "scorecard_id", "id", "source_plan_id", "source_decision_id", "plan_id", "job_id", "invocation_id")
	}
	if sourceID == "" {
		return DistilledMemoryCard{}, false
	}
	lesson := distilledMemoryFirstString(records, "lesson", "compact_lesson", "summary", "compact_summary", "recommendation_summary", "preprocessing_hypothesis", "training_dynamics", "dynamics")
	if lesson == "" {
		lesson = strings.TrimSpace(result.RetrievalReason)
	}
	if strings.TrimSpace(lesson) == "" {
		lesson = distilledMemoryFirstString(records, "expected_effect", "intervention")
	}
	if strings.TrimSpace(lesson) == "" {
		return DistilledMemoryCard{}, false
	}
	mechanism := normalizeDistilledMemoryMechanism(distilledMemoryFirstString(records, "mechanism", "mechanism_group", "strategy_type", "selected_mechanism"))
	outcome := distilledMemoryFirstString(records, "outcome", "outcome_status", "decision_type", "status")
	appliesWhen := distilledMemoryAppliesWhenFromValues(records)
	evidenceUsed := distilledMemoryEvidenceIDs(sourceID, records, 6)
	if len(evidenceUsed) == 0 {
		evidenceUsed = []string{sourceID}
	}
	confidence := clampScore(result.Score)
	if confidence == 0 {
		confidence = clampScore(result.SemanticScore)
	}
	return DistilledMemoryCard{
		SourceTable:  strings.TrimSpace(result.SourceTable),
		SourceID:     sourceID,
		ProjectID:    strings.TrimSpace(result.ProjectID),
		DatasetID:    strings.TrimSpace(result.DatasetID),
		Kind:         strings.TrimSpace(result.Kind),
		Outcome:      outcome,
		Mechanism:    mechanism,
		AppliesWhen:  appliesWhen,
		Lesson:       compactDistilledMemoryText(lesson, 220),
		EvidenceUsed: evidenceUsed,
		Confidence:   confidence,
		ValueScore:   clampScore(outcomeScore(outcome)),
		Summary:      compactDistilledMemoryText(distilledMemoryFirstString(records, "summary", "compact_summary"), 180),
	}, true
}

func BuildStrategyScorecardCard(scorecard strategies.StrategyScorecard) (EmbeddableMemoryCard, bool) {
	tags := cleanStrings(scorecard.Tags, 12)
	diagnosisTriggers := cleanStrings(scorecard.DiagnosisTriggers, 12)
	evidenceUsed := cleanStrings(scorecard.EvidenceUsed, 8)
	models := stringsFromAny(scorecard.ProposedChanges["models"], 12)
	if len(models) == 0 {
		models = stringsFromAny(scorecard.ProposedChanges["proposed_models"], 12)
	}

	summary := map[string]any{
		"source":             SourceStrategyScorecard,
		"scorecard_id":       scorecard.ID,
		"strategy_type":      scorecard.StrategyType,
		"planning_mode":      scorecard.PlanningMode,
		"mechanism":          scorecard.Mechanism,
		"intervention":       scorecard.Intervention,
		"diagnosis_triggers": diagnosisTriggers,
		"expected_effect":    scorecard.ExpectedEffect,
		"outcome":            scorecard.Outcome,
		"lesson":             scorecard.Lesson,
		"expected_delta":     scorecard.ExpectedDelta,
		"actual_delta":       scorecard.ActualDelta,
		"cost_usd":           scorecard.CostUSD,
		"runtime_seconds":    scorecard.RuntimeSeconds,
		"models":             models,
		"tags":               tags,
	}
	removeEmpty(summary)

	metadata := map[string]any{
		"source_table":                  SourceStrategyScorecard,
		"source_id":                     scorecard.ID,
		"source_decision_id":            scorecard.SourceDecisionID,
		"source_plan_id":                scorecard.SourcePlanID,
		"followup_plan_id":              scorecard.FollowUpPlanID,
		"strategy_type":                 scorecard.StrategyType,
		"planning_mode":                 scorecard.PlanningMode,
		"mechanism":                     scorecard.Mechanism,
		"intervention":                  scorecard.Intervention,
		"diagnosis_triggers":            diagnosisTriggers,
		"evidence_used":                 evidenceUsed,
		"expected_effect":               scorecard.ExpectedEffect,
		"dataset_traits":                compactMap(scorecard.DatasetTraits, 12),
		"objective_profile":             compactMap(scorecard.ObjectiveProfile, 12),
		"outcome":                       scorecard.Outcome,
		"expected_delta":                scorecard.ExpectedDelta,
		"actual_delta":                  scorecard.ActualDelta,
		"confidence_before":             scorecard.ConfidenceBefore,
		"confidence_after":              scorecard.ConfidenceAfter,
		"cost_usd":                      scorecard.CostUSD,
		"runtime_seconds":               scorecard.RuntimeSeconds,
		"models":                        models,
		"tags":                          tags,
		"accepted_for_vector_memory":    scorecard.Outcome != strategies.OutcomeInvalidated,
		"raw_proposed_changes_excluded": true,
		"embedding_card_schema_version": MemoryCardVersion,
	}
	removeEmpty(metadata)

	text := deterministicMemoryText([]textField{
		{"source", SourceStrategyScorecard},
		{"strategy_type", scorecard.StrategyType},
		{"planning_mode", scorecard.PlanningMode},
		{"mechanism", scorecard.Mechanism},
		{"intervention", scorecard.Intervention},
		{"diagnosis_triggers", strings.Join(diagnosisTriggers, ", ")},
		{"evidence", strings.Join(evidenceUsed, " | ")},
		{"expected_effect", scorecard.ExpectedEffect},
		{"outcome", scorecard.Outcome},
		{"lesson", scorecard.Lesson},
		{"expected_delta", formatFloat(scorecard.ExpectedDelta)},
		{"actual_delta", formatFloat(scorecard.ActualDelta)},
		{"cost_usd", formatFloat(scorecard.CostUSD)},
		{"runtime_seconds", formatFloat(scorecard.RuntimeSeconds)},
		{"models", strings.Join(models, ", ")},
		{"tags", strings.Join(tags, ", ")},
	})

	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceStrategyScorecard,
		SourceID:     scorecard.ID,
		ProjectID:    scorecard.ProjectID,
		DatasetID:    scorecard.DatasetID,
		PlanID:       scorecard.SourcePlanID,
		Kind:         "strategy_scorecard",
		Scope:        memoryScope(scorecard.DatasetID, scorecard.SourcePlanID, ""),
		Text:         text,
		SummaryCard:  summary,
		Metadata:     metadata,
		QualityScore: scorecardQualityScore(scorecard),
		OutcomeScore: outcomeScore(scorecard.Outcome),
	}, strings.TrimSpace(scorecard.ID) != ""
}

func NewDatasetProfileMemoryCard(dataset datasets.Dataset) EmbeddableMemoryCard {
	card, _ := BuildDatasetProfileCard(dataset)
	return card
}

func BuildDatasetProfileCard(dataset datasets.Dataset) (EmbeddableMemoryCard, bool) {
	profile := emptyMap(dataset.Profile)
	if strings.TrimSpace(dataset.ID) == "" || len(profile) == 0 {
		return EmbeddableMemoryCard{}, false
	}
	taskType := datasetProfileString(profile, "task_type")
	if taskType == "" {
		taskType = "image_classification"
	}
	classCount := datasetProfileInt(profile, "class_count")
	imageCount := firstPositiveInt(
		datasetProfileInt(profile, "total_images"),
		datasetProfileInt(profile, "image_count"),
	)
	imbalanceRatio := datasetProfileFloat(profile, "imbalance_ratio")
	if imbalanceRatio == 0 {
		imbalanceRatio = imbalanceRatioFromDistribution(firstNonEmptyMap(
			datasetProfileMap(profile, "class_distribution"),
			datasetProfileMap(profile, "images_per_class"),
		))
	}
	corruptFileCount := firstPositiveInt(
		datasetProfileInt(profile, "corrupt_file_count"),
		datasetProfileInt(profile, "corrupt_image_count"),
	)
	metadataSummary := mergeDatasetSafeMaps(
		datasetSafeMetadataSummary(datasetProfileMap(profile, "metadata_summary")),
		datasetSafeMetadataSummary(datasetProfileMap(profile, "agent_safe_metadata_summary")),
		datasetSafeMetadataSummary(datasetProfileMap(profile, "normalized_metadata_summary")),
	)
	metadataCapabilities := datasetMetadataCapabilities(profile, metadataSummary)
	artifactTypes := datasetArtifactTypes(profile)
	datasetTraits := datasetProfileTraits(profile, metadataCapabilities)
	classNames := datasetSafeStrings(stringsFromAny(profile["class_names"], 12), 12)
	classCountBucket := classCountBucket(classCount)
	imageCountBucket := imageCountBucket(imageCount)
	imbalanceBucket := imbalanceBucket(imbalanceRatio)
	dimensionPattern := datasetDimensionPattern(profile, datasetTraits)
	visualTraits := datasetProfileVisualTraitSummary(profile)
	labelQualitySummary := datasetLabelQualitySummary(profile)
	labelQualityText := labelQualitySummaryText(labelQualitySummary)
	bboxAvailable := datasetHasBBoxEvidence(profile, metadataSummary)

	summary := map[string]any{
		"source":                SourceDatasetProfile,
		"dataset_id":            dataset.ID,
		"dataset_name":          datasetSafeText(dataset.Name, 120),
		"task_type":             taskType,
		"class_count_bucket":    classCountBucket,
		"image_count_bucket":    imageCountBucket,
		"imbalance_bucket":      imbalanceBucket,
		"dimension_pattern":     dimensionPattern,
		"dataset_traits":        datasetTraits,
		"metadata_capabilities": metadataCapabilities,
		"visual_traits":         visualTraits,
		"label_quality":         labelQualitySummary,
		"class_names":           classNames,
	}
	removeEmpty(summary)

	metadata := map[string]any{
		"source_table":                  SourceDatasetProfile,
		"source_id":                     dataset.ID,
		"profile_schema_version":        datasetProfileString(profile, "schema_version"),
		"task_type":                     taskType,
		"class_count":                   classCount,
		"image_count":                   imageCount,
		"class_count_bucket":            classCountBucket,
		"image_count_bucket":            imageCountBucket,
		"imbalance_ratio":               roundedFloat(imbalanceRatio),
		"imbalance_bucket":              imbalanceBucket,
		"corrupt_file_count":            corruptFileCount,
		"dimension_pattern":             dimensionPattern,
		"dataset_traits":                datasetTraits,
		"metadata_capabilities":         metadataCapabilities,
		"metadata_summary":              metadataSummary,
		"artifact_types":                artifactTypes,
		"visual_traits":                 visualTraits,
		"label_quality":                 labelQualitySummary,
		"bbox_available":                bboxAvailable,
		"accepted_for_vector_memory":    true,
		"raw_profile_excluded":          true,
		"raw_manifest_excluded":         true,
		"image_uris_excluded":           true,
		"embedding_card_schema_version": MemoryCardVersion,
	}
	removeEmpty(metadata)

	text := deterministicMemoryText([]textField{
		{"source", SourceDatasetProfile},
		{"kind", KindDatasetProfile},
		{"task_type", taskType},
		{"class_count_bucket", classCountBucket},
		{"image_count_bucket", imageCountBucket},
		{"imbalance_bucket", imbalanceBucket},
		{"dimension_pattern", dimensionPattern},
		{"metadata_capabilities", strings.Join(metadataCapabilities, ", ")},
		{"artifact_types", strings.Join(artifactTypes, ", ")},
		{"dataset_traits", strings.Join(datasetTraits, ", ")},
		{"visual_traits", strings.Join(visualTraits, ", ")},
		{"label_quality", strings.Join(labelQualityText, " | ")},
		{"class_names", strings.Join(classNames, ", ")},
	})

	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceDatasetProfile,
		SourceID:     dataset.ID,
		ProjectID:    dataset.ProjectID,
		DatasetID:    dataset.ID,
		Kind:         KindDatasetProfile,
		Scope:        ScopeDataset,
		Text:         text,
		SummaryCard:  summary,
		Metadata:     metadata,
		QualityScore: datasetProfileQualityScore(dataset, metadataSummary, visualTraits),
		OutcomeScore: 0,
	}, true
}

func NewDatasetVisualAnalysisMemoryCard(analysis datasets.DatasetVisualAnalysis) EmbeddableMemoryCard {
	card, _ := BuildDatasetVisualAnalysisCard(analysis)
	return card
}

func BuildDatasetVisualAnalysisCard(analysis datasets.DatasetVisualAnalysis) (EmbeddableMemoryCard, bool) {
	if strings.TrimSpace(analysis.ID) == "" || strings.TrimSpace(analysis.DatasetID) == "" ||
		!strings.EqualFold(strings.TrimSpace(analysis.ValidationStatus), datasets.VisualValidationStatusAccepted) {
		return EmbeddableMemoryCard{}, false
	}
	visualTraits := datasetVisualTraitSummaries(analysis.VisualTraits, 12)
	classesToWatch := datasetClassWatchSummaries(analysis.ClassesToWatch, 8)
	hypotheses := datasetPreprocessingHypothesisSummaries(analysis.PreprocessingHypotheses, 8)
	hypothesisText := datasetPreprocessingHypothesisText(analysis.PreprocessingHypotheses, 8)
	cautions := datasetVisualCautionSummaries(analysis.Cautions, 6)
	limitations := datasetSafeStrings(analysis.Limitations, 6)
	coverage := datasetVisualCoverageSummary(analysis.CoverageReport)

	summary := map[string]any{
		"source":                      SourceDatasetVisualAnalysis,
		"analysis_id":                 analysis.ID,
		"dataset_id":                  analysis.DatasetID,
		"dataset_name":                datasetSafeText(analysis.DatasetName, 120),
		"trigger_reason":              string(analysis.TriggerReason),
		"validation_status":           analysis.ValidationStatus,
		"confidence":                  analysis.Confidence,
		"coverage":                    coverage,
		"visual_traits":               visualTraits,
		"classes_to_watch":            classesToWatch,
		"preprocessing_hypotheses":    hypotheses,
		"cautions":                    cautions,
		"limitations":                 limitations,
		"profile_fingerprint":         datasetSafeText(analysis.ProfileFingerprint, 120),
		"accepted_visual_evidence":    true,
		"backend_validation_required": true,
	}
	removeEmpty(summary)

	metadata := map[string]any{
		"source_table":                      SourceDatasetVisualAnalysis,
		"source_id":                         analysis.ID,
		"source_job_id":                     analysis.SourceJobID,
		"source_invocation_id":              analysis.SourceInvocationID,
		"profile_schema_version":            analysis.ProfileSchemaVersion,
		"profile_fingerprint":               datasetSafeText(analysis.ProfileFingerprint, 120),
		"analysis_version":                  analysis.AnalysisVersion,
		"trigger_reason":                    string(analysis.TriggerReason),
		"validation_status":                 analysis.ValidationStatus,
		"confidence":                        analysis.Confidence,
		"total_images":                      analysis.TotalImages,
		"images_analyzed":                   analysis.ImagesAnalyzed,
		"coverage":                          coverage,
		"visual_traits":                     visualTraits,
		"classes_to_watch":                  classesToWatch,
		"preprocessing_hypotheses":          hypotheses,
		"cautions":                          cautions,
		"limitations":                       limitations,
		"accepted_for_vector_memory":        true,
		"visual_evidence_advisory_only":     true,
		"backend_validation_still_required": true,
		"raw_visual_samples_excluded":       true,
		"image_uris_excluded":               true,
		"embedding_card_schema_version":     MemoryCardVersion,
	}
	removeEmpty(metadata)

	text := deterministicMemoryText([]textField{
		{"source", SourceDatasetVisualAnalysis},
		{"kind", KindDatasetVisualAnalysis},
		{"trigger_reason", string(analysis.TriggerReason)},
		{"validation_status", analysis.ValidationStatus},
		{"confidence", analysis.Confidence},
		{"coverage", datasetVisualCoverageText(coverage)},
		{"visual_traits", strings.Join(visualTraits, " | ")},
		{"classes_to_watch", strings.Join(classesToWatch, " | ")},
		{"preprocessing_hypotheses", strings.Join(hypothesisText, " | ")},
		{"cautions", strings.Join(cautions, " | ")},
		{"limitations", strings.Join(limitations, " | ")},
	})

	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceDatasetVisualAnalysis,
		SourceID:     analysis.ID,
		ProjectID:    analysis.ProjectID,
		DatasetID:    analysis.DatasetID,
		JobID:        analysis.SourceJobID,
		InvocationID: analysis.SourceInvocationID,
		Kind:         KindDatasetVisualAnalysis,
		Scope:        ScopeDataset,
		Text:         text,
		SummaryCard:  summary,
		Metadata:     metadata,
		QualityScore: datasetVisualAnalysisQualityScore(analysis),
		OutcomeScore: 0,
	}, true
}

func BuildDatasetPreprocessingHypothesisCards(analysis datasets.DatasetVisualAnalysis) []EmbeddableMemoryCard {
	if strings.TrimSpace(analysis.ID) == "" || strings.TrimSpace(analysis.DatasetID) == "" ||
		!strings.EqualFold(strings.TrimSpace(analysis.ValidationStatus), datasets.VisualValidationStatusAccepted) {
		return nil
	}
	cards := []EmbeddableMemoryCard{}
	for index, hypothesis := range analysis.PreprocessingHypotheses {
		card, ok := buildDatasetPreprocessingHypothesisCard(analysis, hypothesis, index)
		if ok {
			cards = append(cards, card)
		}
	}
	return cards
}

func buildDatasetPreprocessingHypothesisCard(analysis datasets.DatasetVisualAnalysis, hypothesis datasets.PreprocessingHypothesis, index int) (EmbeddableMemoryCard, bool) {
	sourceID := datasetHypothesisSourceID(analysis.ID, hypothesis.ID, index)
	hypothesisSummary := compactPreprocessingHypothesis(hypothesis)
	if sourceID == "" || len(hypothesisSummary) == 0 {
		return EmbeddableMemoryCard{}, false
	}
	hypothesisText := compactPreprocessingHypothesisText(hypothesis)
	coverage := datasetVisualCoverageSummary(analysis.CoverageReport)
	metadata := map[string]any{
		"source_table":                      SourceDatasetPreprocessing,
		"source_id":                         sourceID,
		"parent_analysis_id":                analysis.ID,
		"source_job_id":                     analysis.SourceJobID,
		"source_invocation_id":              analysis.SourceInvocationID,
		"profile_fingerprint":               datasetSafeText(analysis.ProfileFingerprint, 120),
		"analysis_version":                  analysis.AnalysisVersion,
		"trigger_reason":                    string(analysis.TriggerReason),
		"validation_status":                 analysis.ValidationStatus,
		"confidence":                        hypothesis.Confidence,
		"support_status":                    hypothesis.SupportStatus,
		"mechanism":                         datasetSafeText(hypothesis.Mechanism, 120),
		"coverage":                          coverage,
		"hypothesis":                        hypothesisSummary,
		"accepted_for_vector_memory":        true,
		"visual_evidence_advisory_only":     true,
		"backend_validation_still_required": true,
		"raw_visual_samples_excluded":       true,
		"image_uris_excluded":               true,
		"embedding_card_schema_version":     MemoryCardVersion,
	}
	removeEmpty(metadata)
	summary := map[string]any{
		"source":                      SourceDatasetPreprocessing,
		"analysis_id":                 analysis.ID,
		"hypothesis":                  hypothesisSummary,
		"validation_status":           analysis.ValidationStatus,
		"backend_validation_required": true,
	}
	removeEmpty(summary)
	text := deterministicMemoryText([]textField{
		{"source", SourceDatasetPreprocessing},
		{"kind", KindDatasetPreprocessingHypothesis},
		{"trigger_reason", string(analysis.TriggerReason)},
		{"validation_status", analysis.ValidationStatus},
		{"hypothesis", hypothesisText},
		{"coverage", datasetVisualCoverageText(coverage)},
	})
	return EmbeddableMemoryCard{
		CardVersion:  MemoryCardVersion,
		SourceTable:  SourceDatasetPreprocessing,
		SourceID:     sourceID,
		ProjectID:    analysis.ProjectID,
		DatasetID:    analysis.DatasetID,
		JobID:        analysis.SourceJobID,
		InvocationID: analysis.SourceInvocationID,
		Kind:         KindDatasetPreprocessingHypothesis,
		Scope:        ScopeDataset,
		Text:         text,
		SummaryCard:  summary,
		Metadata:     metadata,
		QualityScore: datasetConfidenceScore(hypothesis.Confidence),
		OutcomeScore: preprocessingSupportScore(hypothesis.SupportStatus),
	}, true
}

type textField struct {
	key   string
	value string
}

func deterministicMemoryText(fields []textField) string {
	var builder strings.Builder
	for _, field := range fields {
		value := strings.TrimSpace(field.value)
		if value == "" {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(field.key)
		builder.WriteString(": ")
		builder.WriteString(value)
	}
	return builder.String()
}

func memoryScope(datasetID string, planID string, jobID string) string {
	switch {
	case strings.TrimSpace(jobID) != "":
		return ScopeJob
	case strings.TrimSpace(planID) != "":
		return ScopePlan
	case strings.TrimSpace(datasetID) != "":
		return ScopeDataset
	default:
		return ScopeProject
	}
}

func outcomeScore(outcome string) float64 {
	switch strings.ToLower(strings.TrimSpace(outcome)) {
	case strategies.OutcomeImprovedChampion:
		return 1
	case strategies.OutcomeMinorImprovement:
		return 0.65
	case strategies.OutcomeNoImprovement:
		return -0.45
	case strategies.OutcomeFailed:
		return -0.75
	case strategies.OutcomeInvalidated:
		return -1
	case strategies.OutcomePending:
		return 0.05
	default:
		return 0
	}
}

func qualityScoreFromPayload(payload map[string]any) float64 {
	for _, key := range []string{"rank_score", "confidence", "confidence_after", "confidence_before"} {
		if value, ok := numericValue(payload[key]); ok {
			return clampScore(value)
		}
	}
	if action, ok := payload["recommended_action"].(map[string]any); ok {
		if value, ok := numericValue(action["confidence"]); ok {
			return clampScore(value)
		}
	}
	return 0
}

func scorecardQualityScore(scorecard strategies.StrategyScorecard) float64 {
	if scorecard.ConfidenceAfter > 0 {
		return clampScore(scorecard.ConfidenceAfter)
	}
	return clampScore(scorecard.ConfidenceBefore)
}

func clampScore(value float64) float64 {
	if value < -1 {
		return -1
	}
	if value > 1 {
		return 1
	}
	return value
}

func emptyMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

func compactMap(value map[string]any, limit int) map[string]any {
	if len(value) == 0 || limit <= 0 {
		return nil
	}
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if len(keys) > limit {
		keys = keys[:limit]
	}
	out := make(map[string]any, len(keys))
	for _, key := range keys {
		out[key] = value[key]
	}
	return out
}

func removeEmpty(values map[string]any) {
	for key, value := range values {
		switch typed := value.(type) {
		case string:
			if strings.TrimSpace(typed) == "" {
				delete(values, key)
			}
		case []string:
			if len(typed) == 0 {
				delete(values, key)
			}
		case []any:
			if len(typed) == 0 {
				delete(values, key)
			}
		case map[string]any:
			if len(typed) == 0 {
				delete(values, key)
			}
		case nil:
			delete(values, key)
		}
	}
}

func distilledMemoryAppliesWhenFromValues(records []map[string]any) []string {
	values := []string{}
	for _, record := range records {
		if len(record) == 0 {
			continue
		}
		values = append(values, stringsFromAny(record["applies_when"], 6)...)
		values = append(values, stringsFromAny(record["diagnosis_triggers"], 6)...)
		values = append(values, stringsFromAny(record["dataset_traits"], 6)...)
		values = append(values, stringsFromAny(record["tags"], 6)...)
		for _, key := range []string{"planning_mode", "task_type", "model_family", "mechanism", "strategy_type"} {
			if text := firstNonEmptyString(record, key); text != "" {
				values = append(values, text)
			}
		}
		for _, key := range []string{"dataset_traits", "objective_profile"} {
			if text := distilledMemoryValueText(record[key]); text != "" {
				values = append(values, text)
			}
		}
	}
	return distilledMemoryCleanStrings(values, 8)
}

func distilledMemoryEvidenceIDs(seed string, records []map[string]any, limit int) []string {
	values := []string{}
	for _, record := range records {
		for _, key := range []string{
			"source_id",
			"memory_id",
			"scorecard_id",
			"source_decision_id",
			"source_plan_id",
			"followup_plan_id",
			"follow_up_plan_id",
			"plan_id",
			"job_id",
			"invocation_id",
			"decision_id",
		} {
			if text := strings.TrimSpace(firstNonEmptyString(record, key)); text != "" {
				values = append(values, text)
			}
		}
	}
	if seed != "" {
		values = append([]string{seed}, values...)
	}
	return distilledMemoryCleanStrings(values, limit)
}

func distilledMemoryFirstString(records []map[string]any, keys ...string) string {
	for _, key := range keys {
		for _, record := range records {
			if len(record) == 0 {
				continue
			}
			if text := compactDistilledMemoryText(distilledMemoryValueText(record[key]), 220); text != "" {
				return text
			}
		}
	}
	return ""
}

func distilledMemoryValueText(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []string:
		return strings.Join(typed, " ")
	case []any:
		values := []string{}
		for _, item := range typed {
			if text := distilledMemoryValueText(item); text != "" {
				values = append(values, text)
			}
		}
		return strings.Join(values, " ")
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		values := []string{}
		for _, key := range keys {
			if text := distilledMemoryValueText(typed[key]); text != "" {
				values = append(values, key+"="+text)
			}
		}
		return strings.Join(values, "; ")
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}

func distilledMemoryCleanStrings(values []string, limit int) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized == "" {
			continue
		}
		key := strings.ToLower(normalized)
		if key == "<nil>" || key == "nil" || key == "null" {
			continue
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, normalized)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func compactDistilledMemoryText(value string, limit int) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	if limit > 0 && len(trimmed) > limit {
		trimmed = strings.TrimSpace(trimmed[:limit])
	}
	return trimmed
}

func normalizeDistilledMemoryMechanism(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func firstNonEmptyString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := values[key].(string); ok {
			if normalized := strings.TrimSpace(value); normalized != "" {
				return normalized
			}
		}
	}
	return ""
}

func bestModelFromPayload(value any) string {
	switch typed := value.(type) {
	case map[string]any:
		return firstNonEmptyString(typed, "model", "best_model")
	default:
		return ""
	}
}

func recommendedActionType(value any) string {
	action, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	return firstNonEmptyString(action, "type")
}

func mechanismInterventionFromPayload(payload map[string]any, mechanism string, intervention string) (string, string) {
	for _, key := range []string{"proposal_mechanisms", "candidate_hypotheses"} {
		for _, item := range mapsFromAny(payload[key]) {
			if mechanism == "" {
				mechanism = firstNonEmptyString(item, "mechanism", "mechanism_group")
			}
			if intervention == "" {
				intervention = firstNonEmptyString(item, "intervention")
			}
			if mechanism != "" && intervention != "" {
				return mechanism, intervention
			}
		}
	}
	return mechanism, intervention
}

func proposedModelsFromPayload(value any) []string {
	models := []string{}
	for _, item := range mapsFromAny(value) {
		if model := firstNonEmptyString(item, "model"); model != "" {
			models = append(models, model)
		}
	}
	return cleanStrings(models, 12)
}

func rejectedOptionSummaries(value any) []string {
	out := []string{}
	for _, item := range mapsFromAny(value) {
		option := firstNonEmptyString(item, "option")
		reason := firstNonEmptyString(item, "reason")
		if option == "" && reason == "" {
			continue
		}
		switch {
		case option != "" && reason != "":
			out = append(out, option+" - "+reason)
		case option != "":
			out = append(out, option)
		default:
			out = append(out, reason)
		}
	}
	return cleanStrings(out, 8)
}

func mapsFromAny(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		return typed
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if mapped, ok := item.(map[string]any); ok {
				out = append(out, mapped)
			}
		}
		return out
	default:
		return nil
	}
}

func stringsFromAny(value any, limit int) []string {
	switch typed := value.(type) {
	case []string:
		return cleanStrings(typed, limit)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		return cleanStrings(out, limit)
	case string:
		return cleanStrings([]string{typed}, limit)
	default:
		return nil
	}
}

func cleanStrings(values []string, limit int) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized == "" {
			continue
		}
		key := strings.ToLower(normalized)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, normalized)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func floatValue(value any) any {
	if number, ok := numericValue(value); ok {
		return number
	}
	return nil
}

func numericValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case json.Number:
		parsed, err := strconv.ParseFloat(typed.String(), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

func formatFloat(value float64) string {
	if value == 0 {
		return ""
	}
	return fmt.Sprintf("%.4g", value)
}

func datasetProfileString(profile map[string]any, key string) string {
	value, _ := profile[key].(string)
	return datasetSafeText(value, 120)
}

func datasetProfileInt(profile map[string]any, key string) int {
	value, ok := datasetNumericValue(profile[key])
	if !ok {
		return 0
	}
	return int(value)
}

func datasetProfileFloat(profile map[string]any, key string) float64 {
	value, ok := datasetNumericValue(profile[key])
	if !ok {
		return 0
	}
	return value
}

func datasetProfileBool(profile map[string]any, key string) bool {
	return datasetBoolValue(profile[key])
}

func datasetProfileMap(profile map[string]any, key string) map[string]any {
	return datasetMapFromAny(profile[key])
}

func datasetProfileMapSlice(profile map[string]any, key string) []map[string]any {
	return datasetMapSliceFromAny(profile[key])
}

func datasetMapFromAny(value any) map[string]any {
	switch typed := value.(type) {
	case map[string]any:
		return typed
	case map[string]bool:
		out := make(map[string]any, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]int:
		out := make(map[string]any, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]float64:
		out := make(map[string]any, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	default:
		return map[string]any{}
	}
}

func datasetMapSliceFromAny(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		return typed
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			mapped := datasetMapFromAny(item)
			if len(mapped) > 0 {
				out = append(out, mapped)
			}
		}
		return out
	default:
		return nil
	}
}

func datasetNumericValue(value any) (float64, bool) {
	if number, ok := numericValue(value); ok {
		return number, true
	}
	text, ok := value.(string)
	if !ok {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
	return parsed, err == nil
}

func datasetBoolValue(value any) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		normalized := strings.ToLower(strings.TrimSpace(typed))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyMap(values ...map[string]any) map[string]any {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func datasetSafeText(value string, limit int) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" || datasetUnsafeText(trimmed) {
		return ""
	}
	if limit > 0 && len(trimmed) > limit {
		if limit <= 3 {
			return trimmed[:limit]
		}
		return strings.TrimSpace(trimmed[:limit-3]) + "..."
	}
	return trimmed
}

func datasetSafeStrings(values []string, limit int) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		safe := datasetSafeText(value, 180)
		if safe == "" {
			continue
		}
		key := strings.ToLower(safe)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, safe)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetUnsafeText(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, "://") || strings.Contains(lower, `:\`) || strings.HasPrefix(lower, "/") {
		return true
	}
	if strings.Contains(lower, "/") && memoryContainsAnyText(lower, ".csv", ".json", ".xml", ".txt", ".tsv", ".jpg", ".jpeg", ".png", ".parquet", ".yaml", ".yml") {
		return true
	}
	return memoryContainsAnyText(lower, "raw_preview", "raw row", "source row", "sidecar content", "storage uri", "local path")
}

func memoryContainsAnyText(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func classCountBucket(value int) string {
	switch {
	case value <= 0:
		return "unknown_classes"
	case value == 2:
		return "binary_classes"
	case value <= 10:
		return "few_classes"
	case value <= 50:
		return "medium_classes"
	case value <= 200:
		return "many_classes"
	default:
		return "very_many_classes"
	}
}

func imageCountBucket(value int) string {
	switch {
	case value <= 0:
		return "unknown_images"
	case value < 100:
		return "tiny_dataset"
	case value < 500:
		return "small_dataset"
	case value < 2000:
		return "medium_dataset"
	case value < 10000:
		return "large_dataset"
	default:
		return "very_large_dataset"
	}
}

func imbalanceBucket(value float64) string {
	switch {
	case value <= 0:
		return "unknown_imbalance"
	case value < 1.5:
		return "balanced"
	case value < 3:
		return "moderate_imbalance"
	case value < 10:
		return "high_imbalance"
	default:
		return "extreme_imbalance"
	}
}

func imbalanceRatioFromDistribution(distribution map[string]any) float64 {
	if len(distribution) == 0 {
		return 0
	}
	minCount := 0.0
	maxCount := 0.0
	for _, value := range distribution {
		count, ok := datasetNumericValue(value)
		if !ok || count <= 0 {
			continue
		}
		if minCount == 0 || count < minCount {
			minCount = count
		}
		if count > maxCount {
			maxCount = count
		}
	}
	if minCount <= 0 {
		return 0
	}
	return maxCount / minCount
}

func datasetDimensionPattern(profile map[string]any, traits []string) string {
	traitText := strings.ToLower(strings.Join(traits, " "))
	if memoryContainsAnyText(traitText, "variable_image_dimensions", "variable image dimensions") {
		if memoryContainsAnyText(traitText, "high_resolution", "high resolution") {
			return "variable_high_resolution"
		}
		if memoryContainsAnyText(traitText, "low_resolution", "low resolution") {
			return "variable_low_resolution"
		}
		return "variable_image_dimensions"
	}
	maxDimension := maxPositiveFloat(
		datasetProfileFloat(profile, "width_max"),
		datasetProfileFloat(profile, "height_max"),
		datasetDimensionStat(profile, "width", "max"),
		datasetDimensionStat(profile, "height", "max"),
		datasetDimensionStat(profile, "width", "median"),
		datasetDimensionStat(profile, "height", "median"),
	)
	minDimension := minPositiveFloat(
		datasetProfileFloat(profile, "width_min"),
		datasetProfileFloat(profile, "height_min"),
		datasetDimensionStat(profile, "width", "min"),
		datasetDimensionStat(profile, "height", "min"),
	)
	resolution := "standard_resolution"
	switch {
	case maxDimension <= 0:
		resolution = "unknown_dimensions"
	case maxDimension <= 160:
		resolution = "low_resolution"
	case maxDimension >= 512:
		resolution = "high_resolution"
	}
	if minDimension > 0 && maxDimension > 0 && maxDimension/minDimension >= 2 {
		if resolution == "high_resolution" {
			return "variable_high_resolution"
		}
		if resolution == "low_resolution" {
			return "variable_low_resolution"
		}
		return "variable_image_dimensions"
	}
	if memoryContainsAnyText(traitText, "high_resolution", "high resolution") {
		return "high_resolution"
	}
	if memoryContainsAnyText(traitText, "low_resolution", "low resolution") {
		return "low_resolution"
	}
	return resolution
}

func datasetDimensionStat(profile map[string]any, dimension string, field string) float64 {
	stats := datasetProfileMap(profile, "image_dimension_stats")
	dimensionStats := datasetMapFromAny(stats[dimension])
	if value, ok := datasetNumericValue(dimensionStats[field]); ok {
		return value
	}
	return 0
}

func maxPositiveFloat(values ...float64) float64 {
	out := 0.0
	for _, value := range values {
		if value > out {
			out = value
		}
	}
	return out
}

func minPositiveFloat(values ...float64) float64 {
	out := 0.0
	for _, value := range values {
		if value > 0 && (out == 0 || value < out) {
			out = value
		}
	}
	return out
}

func datasetSafeMetadataSummary(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	allowed := map[string]bool{
		"annotation_counts":        true,
		"annotations_available":    true,
		"artifact_counts":          true,
		"bbox_annotation_count":    true,
		"bbox_annotations_count":   true,
		"bbox_available":           true,
		"bbox_count":               true,
		"bbox_coverage_ratio":      true,
		"bbox_sample_count":        true,
		"capabilities":             true,
		"class_count":              true,
		"detected_formats":         true,
		"error_source_count":       true,
		"errors":                   true,
		"formats":                  true,
		"import_id":                true,
		"import_version":           true,
		"labeled_sample_count":     true,
		"metadata_available":       true,
		"missing_label_count":      true,
		"official_split_available": true,
		"parsed_source_count":      true,
		"sample_count":             true,
		"source_count":             true,
		"source_formats":           true,
		"source_kind":              true,
		"split_counts":             true,
		"status":                   true,
		"unsupported_source_count": true,
		"warnings":                 true,
	}
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := map[string]any{}
	for _, key := range keys {
		normalized := datasetMetadataKey(key)
		if !allowed[normalized] || datasetUnsafeMetadataKey(normalized) {
			continue
		}
		if value, ok := datasetSafeMetadataValue(normalized, input[key]); ok {
			out[normalized] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func datasetSafeMetadataValue(key string, value any) (any, bool) {
	switch typed := value.(type) {
	case string:
		safe := datasetSafeText(typed, 160)
		return safe, safe != ""
	case bool:
		return typed, true
	case int, int32, int64, float32, float64, json.Number:
		if number, ok := datasetNumericValue(typed); ok {
			return roundedFloat(number), true
		}
	case []string:
		safe := datasetSafeStrings(typed, 12)
		return safe, len(safe) > 0
	case []any:
		if datasetMetadataIsIssueList(key) {
			issues := datasetIssueSummaries(typed, 6)
			return issues, len(issues) > 0
		}
		values := []string{}
		for _, item := range typed {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
		safe := datasetSafeStrings(values, 12)
		return safe, len(safe) > 0
	case map[string]any, map[string]bool, map[string]int, map[string]float64:
		if !datasetMetadataAllowsLeafMap(key) {
			return nil, false
		}
		out := datasetSafeLeafMap(datasetMapFromAny(typed), 16)
		return out, len(out) > 0
	}
	return nil, false
}

func datasetMetadataAllowsLeafMap(key string) bool {
	switch datasetMetadataKey(key) {
	case "annotation_counts", "artifact_counts", "capabilities", "split_counts":
		return true
	default:
		return false
	}
}

func datasetMetadataIsIssueList(key string) bool {
	switch datasetMetadataKey(key) {
	case "warnings", "errors":
		return true
	default:
		return false
	}
}

func datasetSafeLeafMap(input map[string]any, limit int) map[string]any {
	if len(input) == 0 {
		return nil
	}
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if limit > 0 && len(keys) > limit {
		keys = keys[:limit]
	}
	out := map[string]any{}
	for _, key := range keys {
		normalized := datasetMetadataKey(key)
		if normalized == "" || datasetUnsafeMetadataKey(normalized) {
			continue
		}
		switch value := input[key].(type) {
		case bool:
			out[normalized] = value
		case string:
			if safe := datasetSafeText(value, 120); safe != "" {
				out[normalized] = safe
			}
		default:
			if number, ok := datasetNumericValue(value); ok {
				out[normalized] = roundedFloat(number)
			}
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func datasetIssueSummaries(values []any, limit int) []string {
	out := []string{}
	for _, value := range values {
		mapped := datasetMapFromAny(value)
		if len(mapped) == 0 {
			continue
		}
		code := datasetSafeText(stringFromAny(mapped["code"]), 80)
		severity := datasetSafeText(stringFromAny(mapped["severity"]), 40)
		message := datasetSafeText(stringFromAny(mapped["message"]), 160)
		parts := []string{}
		if severity != "" {
			parts = append(parts, severity)
		}
		if code != "" {
			parts = append(parts, code)
		}
		if message != "" {
			parts = append(parts, message)
		}
		if len(parts) == 0 {
			continue
		}
		out = append(out, strings.Join(parts, ": "))
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetMetadataKey(key string) string {
	normalized := strings.ToLower(strings.TrimSpace(key))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	return normalized
}

func datasetUnsafeMetadataKey(key string) bool {
	normalized := datasetMetadataKey(key)
	if strings.HasSuffix(normalized, "_count") || strings.HasSuffix(normalized, "_counts") || strings.HasSuffix(normalized, "_ratio") {
		return false
	}
	if normalized == "source_count" || normalized == "source_counts" || normalized == "source_formats" ||
		normalized == "source_kind" || normalized == "parsed_source_count" ||
		normalized == "unsupported_source_count" || normalized == "error_source_count" {
		return false
	}
	return memoryContainsAnyText(
		normalized,
		"path",
		"uri",
		"url",
		"raw",
		"preview",
		"content",
		"sidecar",
		"storage",
		"checksum",
		"filename",
		"file_name",
		"manifest_record",
		"manifest_row",
		"source_row",
		"example_image",
		"image_id",
	)
}

func mergeDatasetSafeMaps(values ...map[string]any) map[string]any {
	out := map[string]any{}
	for _, value := range values {
		for key, item := range value {
			out[key] = item
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func datasetMetadataCapabilities(profile map[string]any, summary map[string]any) []string {
	capabilities := map[string]bool{}
	if len(summary) > 0 || datasetProfileBool(profile, "metadata_available") {
		capabilities["metadata_available"] = true
	}
	if datasetHasBBoxEvidence(profile, summary) {
		capabilities["bbox_available"] = true
		capabilities["bbox_annotations"] = true
	}
	if datasetBoolValue(summary["official_split_available"]) {
		capabilities["official_splits"] = true
	}
	for key, value := range datasetMapFromAny(summary["capabilities"]) {
		if datasetBoolValue(value) {
			capabilities[datasetMetadataKey(key)] = true
		}
	}
	for key, value := range datasetMapFromAny(summary["annotation_counts"]) {
		if count, ok := datasetNumericValue(value); ok && count > 0 {
			switch datasetMetadataKey(key) {
			case "bbox", "bounding_box", "bounding_boxes":
				capabilities["bbox_annotations"] = true
			case "attribute", "attributes":
				capabilities["attribute_annotations"] = true
			case "keypoint", "keypoints":
				capabilities["keypoint_annotations"] = true
			}
		}
	}
	for key, value := range datasetMapFromAny(summary["artifact_counts"]) {
		if count, ok := datasetNumericValue(value); ok && count > 0 {
			addArtifactCapability(capabilities, datasetMetadataKey(key))
		}
	}
	for _, artifactType := range datasetArtifactTypes(profile) {
		addArtifactCapability(capabilities, artifactType)
	}
	return sortedTrueKeys(capabilities, 16)
}

func addArtifactCapability(capabilities map[string]bool, artifactType string) {
	switch datasetMetadataKey(artifactType) {
	case datasets.ArtifactBoundingBoxes, "bbox", "bounding_box", "annotation_xml", "annotation_json":
		capabilities["bbox_available"] = true
	case datasets.ArtifactLabelsCSV, "classification_labels":
		capabilities["classification_labels"] = true
	case datasets.ArtifactSplitFile:
		capabilities["official_splits"] = true
	case datasets.ArtifactMetadataFolder:
		capabilities["metadata_available"] = true
	case datasets.ArtifactClassHierarchy:
		capabilities["class_hierarchy"] = true
	}
}

func datasetArtifactTypes(profile map[string]any) []string {
	seen := map[string]bool{}
	for _, artifact := range datasetProfileMapSlice(profile, "artifacts") {
		artifactType := datasetMetadataKey(stringFromAny(artifact["artifact_type"]))
		if artifactType != "" && !datasetUnsafeMetadataKey(artifactType) {
			seen[artifactType] = true
		}
	}
	return sortedTrueKeys(seen, 16)
}

func datasetProfileTraits(profile map[string]any, metadataCapabilities []string) []string {
	values := datasetSafeStrings(stringsFromAny(profile["dataset_traits"], 24), 24)
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		key := strings.ToLower(value)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, value)
	}
	for _, capability := range metadataCapabilities {
		switch capability {
		case "bbox_available", "metadata_available", "official_splits", "classification_labels", "class_hierarchy":
			if !seen[capability] {
				seen[capability] = true
				out = append(out, capability)
			}
		}
	}
	return cleanStrings(out, 24)
}

func datasetHasBBoxEvidence(profile map[string]any, summary map[string]any) bool {
	if datasetProfileBool(profile, "bbox_available") || datasetProfileBool(profile, "annotations_available") ||
		datasetProfileInt(profile, "bbox_annotations_count") > 0 || datasetProfileFloat(profile, "bbox_count") > 0 {
		return true
	}
	if metadataSummaryHasBBoxEvidence(summary) ||
		metadataSummaryHasBBoxEvidence(datasetProfileMap(profile, "metadata_summary")) ||
		metadataSummaryHasBBoxEvidence(datasetProfileMap(profile, "agent_safe_metadata_summary")) ||
		metadataSummaryHasBBoxEvidence(datasetProfileMap(profile, "normalized_metadata_summary")) {
		return true
	}
	visualTraits := datasetProfileMap(profile, "visual_trait_summary")
	if count, ok := datasetNumericValue(visualTraits["bbox_count"]); ok && count > 0 {
		return true
	}
	for _, trait := range stringsFromAny(profile["dataset_traits"], 32) {
		if memoryContainsAnyText(strings.ToLower(trait), "bbox", "bounding box", "annotation") {
			return true
		}
	}
	for _, artifactType := range datasetArtifactTypes(profile) {
		if memoryContainsAnyText(artifactType, "bbox", "bounding_box", "bounding_boxes", "annotation_xml", "annotation_json") {
			return true
		}
	}
	return false
}

func metadataSummaryHasBBoxEvidence(summary map[string]any) bool {
	if len(summary) == 0 {
		return false
	}
	if datasetBoolValue(summary["bbox_available"]) || datasetBoolValue(summary["annotations_available"]) {
		return true
	}
	for _, key := range []string{"bbox_annotations_count", "bbox_annotation_count", "bbox_sample_count", "bbox_count", "bbox_coverage_ratio"} {
		if value, ok := datasetNumericValue(summary[key]); ok && value > 0 {
			return true
		}
	}
	for key, value := range datasetMapFromAny(summary["annotation_counts"]) {
		if count, ok := datasetNumericValue(value); ok && count > 0 &&
			memoryContainsAnyText(datasetMetadataKey(key), "bbox", "bounding_box") {
			return true
		}
	}
	for key, value := range datasetMapFromAny(summary["artifact_counts"]) {
		if count, ok := datasetNumericValue(value); ok && count > 0 &&
			memoryContainsAnyText(datasetMetadataKey(key), "bbox", "bounding_box", "annotation") {
			return true
		}
	}
	for key, value := range datasetMapFromAny(summary["capabilities"]) {
		if datasetBoolValue(value) && memoryContainsAnyText(datasetMetadataKey(key), "bbox", "bounding_box", "object_detection") {
			return true
		}
	}
	return false
}

func datasetProfileVisualTraitSummary(profile map[string]any) []string {
	visualTraits := datasetProfileMap(profile, "visual_trait_summary")
	if len(visualTraits) == 0 {
		return nil
	}
	allowed := map[string]bool{
		"background_dominance":     true,
		"bbox_count":               true,
		"blur_likelihood":          true,
		"crop_plausibility":        true,
		"fine_grained_possible":    true,
		"lighting_variation":       true,
		"object_area_ratio_median": true,
		"object_scale":             true,
		"sampled_image_count":      true,
	}
	keys := make([]string, 0, len(visualTraits))
	for key := range visualTraits {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := []string{}
	for _, key := range keys {
		normalized := datasetMetadataKey(key)
		if !allowed[normalized] || datasetUnsafeMetadataKey(normalized) {
			continue
		}
		switch value := visualTraits[key].(type) {
		case bool:
			if value {
				out = append(out, normalized+"=true")
			}
		case string:
			if safe := datasetSafeText(value, 80); safe != "" {
				out = append(out, normalized+"="+safe)
			}
		default:
			if number, ok := datasetNumericValue(value); ok && (number > 0 || strings.HasSuffix(normalized, "_ratio_median")) {
				out = append(out, normalized+"="+formatAnyNumber(number))
			}
		}
		if len(out) >= 12 {
			break
		}
	}
	return out
}

func datasetLabelQualitySummary(profile map[string]any) map[string]any {
	audit := datasetProfileMap(profile, "label_quality_audit")
	if len(audit) == 0 {
		audits := datasetMapSliceFromAny(profile["label_quality_audits"])
		if len(audits) > 0 {
			audit = audits[0]
		}
	}
	if len(audit) == 0 {
		return nil
	}
	summary := datasetMapFromAny(audit["summary"])
	findings := datasetMapSliceFromAny(audit["findings"])
	out := map[string]any{
		"audit_type":           datasetSafeText(stringFromAny(audit["audit_type"]), 80),
		"status":               datasetSafeText(stringFromAny(audit["status"]), 80),
		"report_only":          datasetBoolValue(audit["report_only"]),
		"mechanism":            datasetSafeText(stringFromAny(audit["mechanism"]), 80),
		"intervention":         datasetSafeText(stringFromAny(audit["intervention"]), 160),
		"expected_effect":      datasetSafeText(stringFromAny(audit["expected_effect"]), 180),
		"finding_count":        len(findings),
		"class_count":          intFromMap(summary, "class_count"),
		"image_count":          intFromMap(summary, "image_count"),
		"imbalance_ratio":      roundedFloat(floatFromMap(summary, "imbalance_ratio")),
		"minority_class_count": intFromMap(summary, "minority_class_count"),
		"majority_class_count": intFromMap(summary, "majority_class_count"),
		"finding_kinds":        labelQualityFindingKinds(findings),
	}
	removeEmpty(out)
	if len(out) == 0 {
		return nil
	}
	return out
}

func labelQualityFindingKinds(findings []map[string]any) []string {
	seen := map[string]bool{}
	for _, finding := range findings {
		kind := datasetMetadataKey(stringFromAny(finding["kind"]))
		if kind != "" && !datasetUnsafeMetadataKey(kind) {
			seen[kind] = true
		}
	}
	return sortedTrueKeys(seen, 8)
}

func labelQualitySummaryText(summary map[string]any) []string {
	if len(summary) == 0 {
		return nil
	}
	fields := []string{}
	for _, key := range []string{"audit_type", "status", "mechanism", "expected_effect"} {
		if value := stringFromAny(summary[key]); value != "" {
			fields = append(fields, key+"="+value)
		}
	}
	if kinds := stringsFromAny(summary["finding_kinds"], 8); len(kinds) > 0 {
		fields = append(fields, "finding_kinds="+strings.Join(kinds, ", "))
	}
	return fields
}

func datasetVisualTraitSummaries(traits []datasets.VisualTrait, limit int) []string {
	out := []string{}
	for _, trait := range traits {
		name := datasetSafeText(trait.Trait, 80)
		level := datasetSafeText(trait.Level, 60)
		if name == "" && level == "" {
			continue
		}
		parts := []string{}
		if name != "" && level != "" {
			parts = append(parts, name+"="+level)
		} else if name != "" {
			parts = append(parts, name)
		} else {
			parts = append(parts, level)
		}
		if confidence := datasetSafeText(trait.Confidence, 40); confidence != "" {
			parts = append(parts, "confidence="+confidence)
		}
		if classes := datasetSafeStrings(trait.AffectedClasses, 4); len(classes) > 0 {
			parts = append(parts, "classes="+strings.Join(classes, ", "))
		}
		if evidence := datasetSafeStrings(trait.Evidence, 3); len(evidence) > 0 {
			parts = append(parts, "evidence="+strings.Join(evidence, "; "))
		}
		if notes := datasetSafeText(trait.Notes, 140); notes != "" {
			parts = append(parts, "notes="+notes)
		}
		out = append(out, strings.Join(parts, ", "))
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetClassWatchSummaries(classes []datasets.ClassWatchItem, limit int) []string {
	out := []string{}
	for _, item := range classes {
		className := datasetSafeText(item.ClassName, 80)
		reason := datasetSafeText(item.Reason, 160)
		if className == "" && reason == "" {
			continue
		}
		parts := []string{}
		if className != "" {
			parts = append(parts, "class="+className)
		}
		if reason != "" {
			parts = append(parts, "reason="+reason)
		}
		if related := datasetSafeStrings(item.RelatedClasses, 4); len(related) > 0 {
			parts = append(parts, "related="+strings.Join(related, ", "))
		}
		if evidence := datasetSafeStrings(item.Evidence, 3); len(evidence) > 0 {
			parts = append(parts, "evidence="+strings.Join(evidence, "; "))
		}
		if confidence := datasetSafeText(item.Confidence, 40); confidence != "" {
			parts = append(parts, "confidence="+confidence)
		}
		out = append(out, strings.Join(parts, ", "))
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetVisualCautionSummaries(cautions []datasets.VisualCaution, limit int) []string {
	out := []string{}
	for _, caution := range cautions {
		operation := datasetSafeText(caution.Operation, 80)
		reason := datasetSafeText(caution.Reason, 160)
		if operation == "" && reason == "" {
			continue
		}
		parts := []string{}
		if operation != "" {
			parts = append(parts, "operation="+operation)
		}
		if reason != "" {
			parts = append(parts, "reason="+reason)
		}
		if severity := datasetSafeText(caution.Severity, 40); severity != "" {
			parts = append(parts, "severity="+severity)
		}
		if confidence := datasetSafeText(caution.Confidence, 40); confidence != "" {
			parts = append(parts, "confidence="+confidence)
		}
		if classes := datasetSafeStrings(caution.AffectedClasses, 4); len(classes) > 0 {
			parts = append(parts, "classes="+strings.Join(classes, ", "))
		}
		out = append(out, strings.Join(parts, ", "))
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetPreprocessingHypothesisSummaries(hypotheses []datasets.PreprocessingHypothesis, limit int) []map[string]any {
	out := []map[string]any{}
	for _, hypothesis := range hypotheses {
		summary := compactPreprocessingHypothesis(hypothesis)
		if len(summary) == 0 {
			continue
		}
		out = append(out, summary)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func datasetPreprocessingHypothesisText(hypotheses []datasets.PreprocessingHypothesis, limit int) []string {
	out := []string{}
	for _, hypothesis := range hypotheses {
		text := compactPreprocessingHypothesisText(hypothesis)
		if text == "" {
			continue
		}
		out = append(out, text)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func compactPreprocessingHypothesis(hypothesis datasets.PreprocessingHypothesis) map[string]any {
	summary := map[string]any{
		"id":                            datasetSafeText(hypothesis.ID, 80),
		"mechanism":                     datasetSafeText(hypothesis.Mechanism, 100),
		"summary":                       datasetSafeText(hypothesis.Summary, 220),
		"evidence":                      datasetSafeStrings(hypothesis.Evidence, 4),
		"suggested_preprocessing":       suggestedPreprocessingMap(hypothesis),
		"suggested_image_sizes":         positiveIntSlice(hypothesis.SuggestedImageSizes, 4),
		"suggested_augmentation_policy": datasetSafeText(hypothesis.SuggestedAugmentationPolicy, 80),
		"suggested_augmentation_config": suggestedAugmentationConfigMap(hypothesis),
		"expected_effect":               datasetSafeText(hypothesis.ExpectedEffect, 180),
		"risk":                          datasetSafeText(hypothesis.Risk, 160),
		"confidence":                    datasetSafeText(hypothesis.Confidence, 60),
		"support_status":                datasetSafeText(hypothesis.SupportStatus, 80),
		"unsupported_reason":            datasetSafeText(hypothesis.UnsupportedReason, 180),
	}
	removeEmpty(summary)
	return summary
}

func compactPreprocessingHypothesisText(hypothesis datasets.PreprocessingHypothesis) string {
	fields := []string{}
	addTextPart := func(key string, value string) {
		if safe := datasetSafeText(value, 180); safe != "" {
			fields = append(fields, key+"="+safe)
		}
	}
	addTextPart("id", hypothesis.ID)
	addTextPart("mechanism", hypothesis.Mechanism)
	addTextPart("summary", hypothesis.Summary)
	if evidence := datasetSafeStrings(hypothesis.Evidence, 4); len(evidence) > 0 {
		fields = append(fields, "evidence="+strings.Join(evidence, "; "))
	}
	if preprocessing := suggestedPreprocessingText(hypothesis); preprocessing != "" {
		fields = append(fields, "preprocessing="+preprocessing)
	}
	if sizes := positiveIntSlice(hypothesis.SuggestedImageSizes, 4); len(sizes) > 0 {
		fields = append(fields, "image_sizes="+intSliceText(sizes))
	}
	addTextPart("augmentation_policy", hypothesis.SuggestedAugmentationPolicy)
	addTextPart("expected_effect", hypothesis.ExpectedEffect)
	addTextPart("risk", hypothesis.Risk)
	addTextPart("confidence", hypothesis.Confidence)
	addTextPart("support_status", hypothesis.SupportStatus)
	addTextPart("unsupported_reason", hypothesis.UnsupportedReason)
	return strings.Join(fields, ", ")
}

func suggestedPreprocessingMap(hypothesis datasets.PreprocessingHypothesis) map[string]any {
	if hypothesis.SuggestedPreprocessing == nil {
		return nil
	}
	preprocessing := hypothesis.SuggestedPreprocessing
	out := map[string]any{
		"resize_strategy": datasetSafeText(preprocessing.ResizeStrategy, 80),
		"normalization":   datasetSafeText(preprocessing.Normalization, 80),
		"crop_strategy":   datasetSafeText(preprocessing.CropStrategy, 80),
		"bbox_mode":       datasetSafeText(preprocessing.BBoxMode, 80),
	}
	if preprocessing.UseDatasetNormalization {
		out["use_dataset_normalization"] = true
	}
	removeEmpty(out)
	if len(out) == 0 {
		return nil
	}
	return out
}

func suggestedPreprocessingText(hypothesis datasets.PreprocessingHypothesis) string {
	preprocessing := suggestedPreprocessingMap(hypothesis)
	if len(preprocessing) == 0 {
		return ""
	}
	keys := make([]string, 0, len(preprocessing))
	for key := range preprocessing {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := []string{}
	for _, key := range keys {
		parts = append(parts, key+"="+fmt.Sprint(preprocessing[key]))
	}
	return strings.Join(parts, "; ")
}

func suggestedAugmentationConfigMap(hypothesis datasets.PreprocessingHypothesis) map[string]any {
	if hypothesis.SuggestedAugmentationConfig == nil {
		return nil
	}
	config := hypothesis.SuggestedAugmentationConfig
	out := map[string]any{
		"policy_type":        datasetSafeText(config.PolicyType, 80),
		"magnitude":          positiveInt(config.Magnitude),
		"num_ops":            positiveInt(config.NumOps),
		"num_magnitude_bins": positiveInt(config.NumMagnitudeBins),
		"probability":        positiveFloat(config.Probability),
		"alpha":              positiveFloat(config.Alpha),
	}
	removeEmpty(out)
	if len(out) == 0 {
		return nil
	}
	return out
}

func datasetVisualCoverageSummary(coverage datasets.VisualCoverageReport) map[string]any {
	out := map[string]any{
		"selection_strategy":      datasetSafeText(coverage.SelectionStrategy, 80),
		"selection_basis":         datasetSafeStrings(coverage.SelectionBasis, 6),
		"images_available":        coverage.ImagesAvailable,
		"images_analyzed":         coverage.ImagesAnalyzed,
		"classes_total":           coverage.ClassesTotal,
		"classes_covered":         coverage.ClassesCovered,
		"class_coverage_ratio":    roundedFloat(coverage.ClassCoverageRatio),
		"hard_example_count":      coverage.HardExampleCount,
		"edge_case_count":         coverage.EdgeCaseCount,
		"high_detail_image_count": coverage.HighDetailImageCount,
		"limitations":             datasetSafeStrings(coverage.Limitations, 4),
	}
	removeEmpty(out)
	return out
}

func datasetVisualCoverageText(coverage map[string]any) string {
	if len(coverage) == 0 {
		return ""
	}
	fields := []string{}
	for _, key := range []string{"images_analyzed", "classes_covered", "classes_total", "class_coverage_ratio", "hard_example_count", "edge_case_count"} {
		if value, ok := coverage[key]; ok {
			fields = append(fields, key+"="+fmt.Sprint(value))
		}
	}
	return strings.Join(fields, ", ")
}

func datasetHypothesisSourceID(analysisID string, hypothesisID string, index int) string {
	analysisID = strings.TrimSpace(analysisID)
	if analysisID == "" {
		return ""
	}
	hypothesisID = strings.TrimSpace(hypothesisID)
	if hypothesisID == "" {
		hypothesisID = fmt.Sprintf("hypothesis_%d", index+1)
	}
	hypothesisID = strings.ReplaceAll(hypothesisID, " ", "_")
	return analysisID + "#" + hypothesisID
}

func datasetProfileQualityScore(dataset datasets.Dataset, metadataSummary map[string]any, visualTraits []string) float64 {
	score := 0.35
	if strings.EqualFold(dataset.Status, datasets.StatusProfiled) {
		score += 0.15
	}
	if dataset.ProfiledAt != nil {
		score += 0.05
	}
	if datasetProfileInt(dataset.Profile, "class_count") > 0 {
		score += 0.1
	}
	if firstPositiveInt(datasetProfileInt(dataset.Profile, "total_images"), datasetProfileInt(dataset.Profile, "image_count")) > 0 {
		score += 0.1
	}
	if len(metadataSummary) > 0 {
		score += 0.1
	}
	if len(visualTraits) > 0 {
		score += 0.1
	}
	return clampScore(score)
}

func datasetVisualAnalysisQualityScore(analysis datasets.DatasetVisualAnalysis) float64 {
	score := datasetConfidenceScore(analysis.Confidence)
	if analysis.CoverageReport.ClassCoverageRatio > 0 {
		score += clampScore(analysis.CoverageReport.ClassCoverageRatio) * 0.15
	}
	if analysis.ImagesAnalyzed > 0 {
		score += 0.05
	}
	return clampScore(score)
}

func datasetConfidenceScore(confidence string) float64 {
	switch strings.ToLower(strings.TrimSpace(confidence)) {
	case "very_high", "high", "strong":
		return 0.85
	case "medium", "moderate":
		return 0.6
	case "low", "weak":
		return 0.3
	case "":
		return 0.45
	default:
		return 0.5
	}
}

func preprocessingSupportScore(status string) float64 {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "supported", "backend_supported", "validated":
		return 0.35
	case "needs_backend_validation", "advisory":
		return 0.15
	case "unsupported", "rejected":
		return -0.35
	default:
		return 0
	}
}

func positiveInt(value int) any {
	if value > 0 {
		return value
	}
	return nil
}

func positiveFloat(value float64) any {
	if value > 0 {
		return roundedFloat(value)
	}
	return nil
}

func positiveIntSlice(values []int, limit int) []int {
	out := []int{}
	seen := map[int]bool{}
	for _, value := range values {
		if value <= 0 || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func intSliceText(values []int) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, strconv.Itoa(value))
	}
	return strings.Join(out, ", ")
}

func intFromMap(values map[string]any, key string) any {
	value, ok := datasetNumericValue(values[key])
	if !ok {
		return nil
	}
	return int(value)
}

func floatFromMap(values map[string]any, key string) float64 {
	value, ok := datasetNumericValue(values[key])
	if !ok {
		return 0
	}
	return value
}

func stringFromAny(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return fmt.Sprint(typed)
	}
}

func sortedTrueKeys(values map[string]bool, limit int) []string {
	keys := make([]string, 0, len(values))
	for key, ok := range values {
		if ok && strings.TrimSpace(key) != "" {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	if limit > 0 && len(keys) > limit {
		keys = keys[:limit]
	}
	return keys
}

func formatAnyNumber(value float64) string {
	if value == float64(int(value)) {
		return strconv.Itoa(int(value))
	}
	return fmt.Sprintf("%.4g", value)
}

func roundedFloat(value float64) float64 {
	if value == 0 {
		return 0
	}
	rounded, err := strconv.ParseFloat(fmt.Sprintf("%.4g", value), 64)
	if err != nil {
		return value
	}
	return rounded
}
