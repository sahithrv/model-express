package store

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/strategies"
)

func TestMemoryStoreStrategyScorecardHydratesMechanismFields(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("scorecard project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}

	scorecard, err := store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:      project.ID,
		DatasetID:      "dataset_1",
		FollowUpPlanID: "plan_followup",
		ProposedChanges: map[string]any{
			"proposal_mechanisms": []any{
				map[string]any{
					"mechanism":          "resolution_crop",
					"intervention":       "preserve aspect ratio with 256px random resized crops",
					"diagnosis_triggers": []any{"variable_dimensions", "object_scale"},
					"evidence_used":      []any{"image_dimension_stats show high variance"},
					"expected_effect":    "Improve robustness to object scale and crop variation.",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateStrategyScorecard() error = %v", err)
	}
	if scorecard.Mechanism != "resolution_crop" {
		t.Fatalf("Mechanism = %q, want resolution_crop", scorecard.Mechanism)
	}
	if scorecard.Intervention != "preserve aspect ratio with 256px random resized crops" {
		t.Fatalf("Intervention = %q", scorecard.Intervention)
	}
	if !reflect.DeepEqual(scorecard.DiagnosisTriggers, []string{"variable_dimensions", "object_scale"}) {
		t.Fatalf("DiagnosisTriggers = %#v", scorecard.DiagnosisTriggers)
	}
	if !reflect.DeepEqual(scorecard.EvidenceUsed, []string{"image_dimension_stats show high variance"}) {
		t.Fatalf("EvidenceUsed = %#v", scorecard.EvidenceUsed)
	}
	if scorecard.ExpectedEffect != "Improve robustness to object scale and crop variation." {
		t.Fatalf("ExpectedEffect = %q", scorecard.ExpectedEffect)
	}

	updated, err := store.UpdateStrategyScorecardOutcomeByFollowUpPlan("plan_followup", strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:     0.04,
		ConfidenceAfter: 0.7,
		Outcome:         strategies.OutcomeMinorImprovement,
		Lesson:          "Resolution and crop changes helped.",
		Tags:            []string{"resolution_crop", "minor_improvement"},
	})
	if err != nil {
		t.Fatalf("UpdateStrategyScorecardOutcomeByFollowUpPlan() error = %v", err)
	}
	if updated.Mechanism != scorecard.Mechanism || updated.Intervention != scorecard.Intervention || updated.ExpectedEffect != scorecard.ExpectedEffect {
		t.Fatalf("update should preserve mechanism fields, got %#v", updated)
	}
}

func TestMemoryStoreDatasetVisualAnalysisCRUD(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("visual project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}
	dataset, err := store.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 1024)
	if err != nil {
		t.Fatalf("CreateDataset() error = %v", err)
	}

	first, err := store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      project.ID,
		DatasetID:      dataset.ID,
		SchemaVersion:  datasets.VisualAnalysisSchemaVersion,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		TotalImages:    100,
		ImagesAnalyzed: 32,
		CoverageReport: datasets.VisualCoverageReport{ImagesAnalyzed: 32, ClassesTotal: 4, ClassesCovered: 3},
		Confidence:     "medium",
		VisualTraits:   []datasets.VisualTrait{{Trait: "blur", Confidence: "low", Evidence: []string{"soft edges in samples"}}},
	})
	if err != nil {
		t.Fatalf("CreateDatasetVisualAnalysis() error = %v", err)
	}
	if first.AnalysisVersion != 1 || first.ValidationStatus != datasets.VisualValidationStatusAccepted {
		t.Fatalf("first analysis version/status = %d/%s", first.AnalysisVersion, first.ValidationStatus)
	}

	rejected, err := store.RejectDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:        project.ID,
		DatasetID:        dataset.ID,
		SchemaVersion:    datasets.VisualAnalysisSchemaVersion,
		TriggerReason:    datasets.VisualTriggerManual,
		ValidationErrors: []string{"bad image reference"},
	})
	if err != nil {
		t.Fatalf("RejectDatasetVisualAnalysis() error = %v", err)
	}
	if rejected.AnalysisVersion != 2 || rejected.ValidationStatus != datasets.VisualValidationStatusRejected {
		t.Fatalf("rejected analysis version/status = %d/%s", rejected.AnalysisVersion, rejected.ValidationStatus)
	}

	latest, err := store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      project.ID,
		DatasetID:      dataset.ID,
		SchemaVersion:  datasets.VisualAnalysisSchemaVersion,
		TriggerReason:  datasets.VisualTriggerDeficiencyReanalysis,
		TotalImages:    100,
		ImagesAnalyzed: 48,
		CoverageReport: datasets.VisualCoverageReport{ImagesAnalyzed: 48, ClassesTotal: 4, ClassesCovered: 4},
		Confidence:     "high",
	})
	if err != nil {
		t.Fatalf("CreateDatasetVisualAnalysis() latest error = %v", err)
	}
	found, err := store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("GetLatestAcceptedDatasetVisualAnalysis() error = %v", err)
	}
	if found.ID != latest.ID {
		t.Fatalf("latest accepted ID = %s, want %s", found.ID, latest.ID)
	}
	analyses, err := store.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		t.Fatalf("ListDatasetVisualAnalyses() error = %v", err)
	}
	if len(analyses) != 3 || analyses[0].ID != latest.ID || analyses[2].ID != first.ID {
		t.Fatalf("unexpected visual analysis ordering: %#v", analyses)
	}
}

func TestMemoryStoreMemoryEmbeddingUpsertAndSearch(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("memory project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}
	otherProject, err := store.CreateProject("other project", "maximize accuracy")
	if err != nil {
		t.Fatalf("CreateProject() other error = %v", err)
	}

	first, err := store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "memory_1",
		ProjectID:           project.ID,
		DatasetID:           "dataset_1",
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.1, 0.2, 0.3},
		EmbeddingText:       "class balancing improved minority class recall",
		SummaryCard:         map[string]any{"lesson": "class balancing helped"},
		Metadata:            map[string]any{"mechanism": "class_balancing", "outcome": strategies.OutcomeImprovedChampion, "accepted_for_vector_memory": true},
		QualityScore:        0.8,
		OutcomeScore:        1,
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() first error = %v", err)
	}
	if first.ID == "" {
		t.Fatalf("UpsertMemoryEmbedding() did not assign ID")
	}

	second, err := store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "memory_2",
		ProjectID:           project.ID,
		Kind:                memory.KindTrainingEvaluation,
		Scope:               memory.ScopeProject,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.4, 0.5, 0.6},
		EmbeddingText:       "training dynamics plateaued after early epochs",
		Metadata:            map[string]any{"agent_name": "training_monitor", "accepted_for_vector_memory": true},
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() second error = %v", err)
	}
	_, err = store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "memory_other",
		ProjectID:           otherProject.ID,
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeProject,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.7, 0.8, 0.9},
		EmbeddingText:       "class balancing from another project",
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() other project error = %v", err)
	}

	updated, err := store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		ID:                  "ignored_new_id",
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            first.SourceID,
		ProjectID:           project.ID,
		DatasetID:           "dataset_1",
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.9, 0.2, 0.1},
		EmbeddingText:       "class balancing and weighted sampling improved minority recall",
		Metadata:            map[string]any{"mechanism": "class_balancing", "accepted_for_vector_memory": true},
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() update error = %v", err)
	}
	if updated.ID != first.ID {
		t.Fatalf("upsert should preserve ID for same source/model, got %q want %q", updated.ID, first.ID)
	}
	if second.ID == updated.ID {
		t.Fatalf("distinct source should keep distinct ID")
	}
	_, err = store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "memory_3",
		ProjectID:           project.ID,
		DatasetID:           "dataset_1",
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0, 1, 0},
		EmbeddingText:       "minority class balancing was attempted with low confidence",
		Metadata:            map[string]any{"mechanism": "class_balancing", "accepted_for_vector_memory": true},
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() competing vector error = %v", err)
	}

	results, err := store.SearchMemoryEmbeddings(memory.MemoryRetrievalQuery{
		ProjectID:           project.ID,
		DatasetID:           "dataset_1",
		Text:                "minority class balancing",
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.9, 0.2, 0.1},
		Kinds:               []string{memory.KindPlanningOutcome},
		Mechanisms:          []string{"class_balancing"},
		Limit:               5,
	})
	if err != nil {
		t.Fatalf("SearchMemoryEmbeddings() error = %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("SearchMemoryEmbeddings() returned %d results, want 2: %#v", len(results), results)
	}
	if results[0].SourceID != first.SourceID {
		t.Fatalf("SearchMemoryEmbeddings() SourceID = %q, want %q", results[0].SourceID, first.SourceID)
	}
	if results[0].Score <= 0 || results[0].SemanticScore <= 0 || results[0].StructuredScore <= 0 {
		t.Fatalf("expected positive retrieval scores, got %#v", results[0])
	}
	if !strings.Contains(results[0].RetrievalReason, "vector match") {
		t.Fatalf("expected vector retrieval reason, got %q", results[0].RetrievalReason)
	}

	crossAgentResults, err := store.SearchMemoryEmbeddings(memory.MemoryRetrievalQuery{
		ProjectID: project.ID,
		AgentName: "experiment_planner",
		Text:      "training dynamics plateau",
		Kinds:     []string{memory.KindTrainingEvaluation},
		Limit:     5,
	})
	if err != nil {
		t.Fatalf("SearchMemoryEmbeddings() cross-agent error = %v", err)
	}
	if len(crossAgentResults) == 0 || crossAgentResults[0].SourceID != second.SourceID {
		t.Fatalf("agent_name should describe requester, not filter source agent: %#v", crossAgentResults)
	}
}

func TestMemoryStoreListUnembeddedMemorySources(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("source project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}

	record, err := store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		ProjectID: project.ID,
		DatasetID: "dataset_1",
		AgentName: "experiment_planner",
		Kind:      memory.KindPlanningOutcome,
		Summary:   "Class balancing produced a useful follow-up.",
		Payload: map[string]any{
			"lesson":         "Class balancing helped minority classes.",
			"mechanism":      "class_balancing",
			"outcome_status": strategies.OutcomeImprovedChampion,
		},
	})
	if err != nil {
		t.Fatalf("CreateAgentMemoryRecord() error = %v", err)
	}
	scorecard, err := store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:    project.ID,
		DatasetID:    "dataset_1",
		Mechanism:    "resolution_crop",
		Intervention: "increase image size with random resized crops",
		Outcome:      strategies.OutcomeMinorImprovement,
		Lesson:       "Resolution crop helped small objects.",
	})
	if err != nil {
		t.Fatalf("CreateStrategyScorecard() error = %v", err)
	}

	sources, err := store.ListUnembeddedMemorySources(project.ID, "test-embedding", 3, 10)
	if err != nil {
		t.Fatalf("ListUnembeddedMemorySources() error = %v", err)
	}
	if len(sources) != 2 {
		t.Fatalf("ListUnembeddedMemorySources() returned %d sources, want 2: %#v", len(sources), sources)
	}
	var recordSource memory.EmbeddableMemoryCard
	for _, source := range sources {
		if source.SourceTable == memory.SourceAgentMemoryRecord && source.SourceID == record.ID {
			recordSource = source
			break
		}
	}
	if recordSource.SourceID == "" {
		t.Fatalf("expected listed source for memory record %s in %#v", record.ID, sources)
	}

	_, err = store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            record.ID,
		ProjectID:           project.ID,
		DatasetID:           "dataset_1",
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.1, 0.2, 0.3},
		EmbeddingText:       recordSource.Text,
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() error = %v", err)
	}

	sources, err = store.ListUnembeddedMemorySources(project.ID, "test-embedding", 3, 10)
	if err != nil {
		t.Fatalf("ListUnembeddedMemorySources() after embedding error = %v", err)
	}
	if len(sources) != 1 {
		t.Fatalf("ListUnembeddedMemorySources() returned %d sources, want 1: %#v", len(sources), sources)
	}
	if sources[0].SourceTable != memory.SourceStrategyScorecard || sources[0].SourceID != scorecard.ID {
		t.Fatalf("remaining source = %s/%s, want scorecard %s", sources[0].SourceTable, sources[0].SourceID, scorecard.ID)
	}
}

func TestMemoryStoreListUnembeddedMemorySourcesIncludesDatasetAndVisualCards(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("dataset source project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}
	dataset, err := store.CreateDataset(project.ID, "small object birds", "s3://bucket/not-indexed", "", 0)
	if err != nil {
		t.Fatalf("CreateDataset() error = %v", err)
	}
	dataset, err = store.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"class_count":  4,
		"total_images": 120,
		"metadata_summary": map[string]any{
			"bbox_available": true,
		},
		"dataset_traits": []any{"small_objects", "fine_grained"},
	})
	if err != nil {
		t.Fatalf("UpdateDatasetProfile() error = %v", err)
	}
	analysis, err := store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:     project.ID,
		DatasetID:     dataset.ID,
		Confidence:    "high",
		TriggerReason: datasets.VisualTriggerInitialProfile,
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAnalyzed:     24,
			ClassesTotal:       4,
			ClassesCovered:     4,
			ClassCoverageRatio: 1,
		},
		VisualTraits: []datasets.VisualTrait{{
			Trait:      "object_scale",
			Level:      "small",
			Confidence: "high",
			Evidence:   []string{"objects are small"},
		}},
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{{
			ID:        "bbox_crop",
			Mechanism: "bbox_crop_ablation",
			Summary:   "Bounding-box crop may reduce background noise.",
			Evidence:  []string{"objects are small"},
			SuggestedPreprocessing: &plans.Preprocessing{
				CropStrategy: "bbox_crop",
				BBoxMode:     "use_if_available",
			},
			ExpectedEffect: "Improve object focus.",
			Confidence:     "high",
			SupportStatus:  "supported",
		}},
	})
	if err != nil {
		t.Fatalf("CreateDatasetVisualAnalysis() error = %v", err)
	}

	sources, err := store.ListUnembeddedMemorySources(project.ID, "test-embedding", 3, 10)
	if err != nil {
		t.Fatalf("ListUnembeddedMemorySources() error = %v", err)
	}
	found := map[string]memory.EmbeddableMemoryCard{}
	for _, source := range sources {
		found[memoryEmbeddingSourceKey(source.SourceTable, source.SourceID)] = source
	}
	for _, want := range []struct {
		sourceTable string
		sourceID    string
	}{
		{memory.SourceDatasetProfile, dataset.ID},
		{memory.SourceDatasetVisualAnalysis, analysis.ID},
		{memory.SourceDatasetPreprocessing, analysis.ID + "#bbox_crop"},
	} {
		if _, ok := found[memoryEmbeddingSourceKey(want.sourceTable, want.sourceID)]; !ok {
			t.Fatalf("missing source %s/%s from %#v", want.sourceTable, want.sourceID, sources)
		}
	}

	_, err = store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceDatasetProfile,
		SourceID:            dataset.ID,
		ProjectID:           project.ID,
		DatasetID:           dataset.ID,
		Kind:                memory.KindDatasetProfile,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{0.1, 0.2, 0.3},
		EmbeddingText:       found[memoryEmbeddingSourceKey(memory.SourceDatasetProfile, dataset.ID)].Text,
	})
	if err != nil {
		t.Fatalf("UpsertMemoryEmbedding() dataset error = %v", err)
	}
	sources, err = store.ListUnembeddedMemorySources(project.ID, "test-embedding", 3, 10)
	if err != nil {
		t.Fatalf("ListUnembeddedMemorySources() after dataset embedding error = %v", err)
	}
	for _, source := range sources {
		if source.SourceTable == memory.SourceDatasetProfile && source.SourceID == dataset.ID {
			t.Fatalf("embedded dataset profile source was returned: %#v", sources)
		}
	}
}

func TestPostgresVectorLiteralSafety(t *testing.T) {
	literal, err := postgresVectorLiteral([]float32{0.25, -1, 3.5})
	if err != nil {
		t.Fatalf("postgresVectorLiteral() error = %v", err)
	}
	if literal != "[0.25,-1,3.5]" {
		t.Fatalf("postgresVectorLiteral() = %q", literal)
	}
	parsed, err := parsePostgresVectorLiteral(literal)
	if err != nil {
		t.Fatalf("parsePostgresVectorLiteral() error = %v", err)
	}
	if !reflect.DeepEqual(parsed, []float32{0.25, -1, 3.5}) {
		t.Fatalf("parsePostgresVectorLiteral() = %#v", parsed)
	}
	if _, err := postgresVectorLiteral([]float32{float32(math.NaN())}); err == nil {
		t.Fatalf("postgresVectorLiteral() should reject NaN")
	}
	if _, err := parsePostgresVectorLiteral("[1,NaN]"); err == nil {
		t.Fatalf("parsePostgresVectorLiteral() should reject NaN")
	}
}

func TestVectorMemoryMigrationIsIdempotent(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/005_vector_memory.sql")
	if err != nil {
		t.Fatalf("read vector memory migration: %v", err)
	}
	sqlText := string(sqlBytes)
	for _, snippet := range []string{
		"CREATE EXTENSION IF NOT EXISTS vector",
		"CREATE SEQUENCE IF NOT EXISTS agent_memory_embedding_id_seq",
		"CREATE TABLE IF NOT EXISTS agent_memory_embeddings",
		"ALTER TABLE agent_memory_embeddings ADD COLUMN IF NOT EXISTS plan_id",
		"CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_project_kind_updated",
		"UNIQUE (source_table, source_id, embedding_model)",
	} {
		if !strings.Contains(sqlText, snippet) {
			t.Fatalf("migration missing idempotent snippet %q", snippet)
		}
	}
}

func TestMemoryStoreDatasetMetadataImportActiveSwapAndBundle(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("metadata project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}
	dataset, err := store.CreateDataset(project.ID, "birds", "s3://bucket/birds.zip", "sha", 2048)
	if err != nil {
		t.Fatalf("CreateDataset() error = %v", err)
	}

	first, err := store.CreateDatasetMetadataImport(datasets.DatasetMetadataImport{
		DatasetID: dataset.ID,
		Status:    datasets.MetadataImportStatusSucceeded,
		Active:    true,
		Sources: []datasets.DatasetMetadataSource{{
			ID:             "source_csv",
			DatasetID:      dataset.ID,
			RelativePath:   "labels.csv",
			DetectedFormat: datasets.MetadataFormatCSVManifest,
			Status:         datasets.MetadataSourceStatusParsed,
		}},
		Classes: []datasets.DatasetClass{{
			DatasetID: dataset.ID,
			ClassKey:  "cat",
			ClassName: "cat",
			SourceID:  "source_csv",
		}},
		ManifestRecords: []datasets.DatasetManifestRecord{{
			DatasetID:    dataset.ID,
			SampleKey:    "cat/one.jpg",
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: "cat/one.jpg",
			LabelKey:     "cat",
			LabelName:    "cat",
			Split:        "train",
			SourceID:     "source_csv",
		}},
	})
	if err != nil {
		t.Fatalf("CreateDatasetMetadataImport() first error = %v", err)
	}
	if !first.Active || first.ImportVersion != 1 {
		t.Fatalf("first import active/version = %t/%d", first.Active, first.ImportVersion)
	}

	second, err := store.CreateDatasetMetadataImport(datasets.DatasetMetadataImport{
		DatasetID: dataset.ID,
		Status:    datasets.MetadataImportStatusSucceeded,
		Active:    true,
		Sources: []datasets.DatasetMetadataSource{{
			ID:             "source_voc",
			DatasetID:      dataset.ID,
			RelativePath:   "annotations/one.xml",
			DetectedFormat: datasets.MetadataFormatPascalVOC,
			Status:         datasets.MetadataSourceStatusParsed,
		}},
		Classes: []datasets.DatasetClass{{
			DatasetID: dataset.ID,
			ClassKey:  "dog",
			ClassName: "dog",
			SourceID:  "source_voc",
		}},
		ManifestRecords: []datasets.DatasetManifestRecord{{
			DatasetID:    dataset.ID,
			SampleKey:    "dog/one.jpg",
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: "dog/one.jpg",
			LabelKey:     "dog",
			LabelName:    "dog",
			Split:        "test",
			SourceID:     "source_voc",
		}},
		Annotations: []datasets.DatasetAnnotation{{
			DatasetID:      dataset.ID,
			SampleKey:      "dog/one.jpg",
			AnnotationType: datasets.MetadataAnnotationBBox,
			LabelKey:       "dog",
			LabelName:      "dog",
			BBox:           map[string]any{"xmin": 1, "ymin": 2, "xmax": 10, "ymax": 20},
			SourceID:       "source_voc",
		}},
	})
	if err != nil {
		t.Fatalf("CreateDatasetMetadataImport() second error = %v", err)
	}
	if second.ImportVersion != 2 || !second.Active {
		t.Fatalf("second import active/version = %t/%d", second.Active, second.ImportVersion)
	}
	active, err := store.GetActiveDatasetMetadataImport(dataset.ID)
	if err != nil {
		t.Fatalf("GetActiveDatasetMetadataImport() error = %v", err)
	}
	if active.ID != second.ID {
		t.Fatalf("active import = %s, want %s", active.ID, second.ID)
	}
	reloadedFirst, err := store.GetDatasetMetadataImport(first.ID)
	if err != nil {
		t.Fatalf("GetDatasetMetadataImport(first) error = %v", err)
	}
	if reloadedFirst.Active {
		t.Fatal("first import should have been deactivated")
	}
	bundle, err := store.GetDatasetMetadataBundle(dataset.ID, "", true, 10, 0)
	if err != nil {
		t.Fatalf("GetDatasetMetadataBundle() error = %v", err)
	}
	if bundle.ImportID != second.ID || len(bundle.ManifestRecords) != 1 || len(bundle.Annotations) != 1 {
		t.Fatalf("unexpected bundle: %#v", bundle)
	}
	if !active.AgentSafeSummary.Capabilities["bbox_annotations"] {
		t.Fatalf("expected active safe summary bbox capability: %#v", active.AgentSafeSummary)
	}
}

func TestMemoryStoreAgentInvocationPersistsLLMUsage(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("usage project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}

	usage := llm.Usage{
		InputTokens:       42,
		OutputTokens:      8,
		TotalTokens:       50,
		CachedInputTokens: 11,
		ReasoningTokens:   3,
		RequestModel:      "test-model",
		APIStyle:          llm.APIStyleResponses,
		ToolRounds:        2,
	}
	inputContext := map[string]any{
		"invocation_runtime": map[string]any{
			"api_style": llm.APIStyleResponses,
			"llm_usage": usage,
		},
	}

	stored, err := store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         project.ID,
		AgentName:         "experiment_planner",
		InputMessages:     []map[string]string{{"role": "user", "content": "hello"}},
		InputContext:      inputContext,
		RawOutput:         `{"ok":true}`,
		ParsedOutput:      map[string]any{"ok": true},
		ValidationStatus:  memory.InvocationValidationValid,
		AcceptedForMemory: true,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
	if err != nil {
		t.Fatalf("CreateAgentInvocation() error = %v", err)
	}

	reloaded, err := store.GetAgentInvocation(stored.ID)
	if err != nil {
		t.Fatalf("GetAgentInvocation() error = %v", err)
	}
	runtime, ok := reloaded.InputContext["invocation_runtime"].(map[string]any)
	if !ok {
		t.Fatalf("expected invocation runtime in input context, got %#v", reloaded.InputContext)
	}
	encodedUsage, ok := runtime["llm_usage"].(llm.Usage)
	if !ok {
		t.Fatalf("expected stored llm usage struct, got %#v", runtime["llm_usage"])
	}
	if encodedUsage.InputTokens != usage.InputTokens || encodedUsage.OutputTokens != usage.OutputTokens || encodedUsage.TotalTokens != usage.TotalTokens {
		t.Fatalf("unexpected stored usage: %#v", encodedUsage)
	}
	if encodedUsage.CachedInputTokens != usage.CachedInputTokens || encodedUsage.ReasoningTokens != usage.ReasoningTokens {
		t.Fatalf("unexpected stored usage breakdown: %#v", encodedUsage)
	}
	if encodedUsage.RequestModel != usage.RequestModel || encodedUsage.APIStyle != usage.APIStyle || encodedUsage.ToolRounds != usage.ToolRounds {
		t.Fatalf("unexpected stored usage metadata: %#v", encodedUsage)
	}
}

func TestScanAgentInvocationPreservesLLMUsage(t *testing.T) {
	createdAt := time.Date(2026, time.June, 2, 12, 0, 0, 0, time.UTC)
	row := fakeAgentInvocationRow{values: []any{
		"agent_invocation_1",
		"project_1",
		"dataset_1",
		"plan_1",
		"job_1",
		"experiment_planner",
		"v1",
		"prompt_v1",
		"openai",
		"test-model",
		[]byte(`[{"role":"user","content":"hello"}]`),
		[]byte(`{"invocation_runtime":{"llm_usage":{"input_tokens":14,"output_tokens":6,"total_tokens":20,"cached_input_tokens":4,"reasoning_tokens":2,"request_model":"test-model","api_style":"responses","tool_rounds":1}}}`),
		`{"ok":true}`,
		[]byte(`{"ok":true}`),
		"valid",
		"",
		true,
		[]byte(`{}`),
		[]byte(`{}`),
		createdAt,
	}}

	invocation, err := scanAgentInvocation(row)
	if err != nil {
		t.Fatalf("scanAgentInvocation() error = %v", err)
	}
	runtime, ok := invocation.InputContext["invocation_runtime"].(map[string]any)
	if !ok {
		t.Fatalf("expected invocation runtime, got %#v", invocation.InputContext)
	}
	usage, ok := runtime["llm_usage"].(map[string]any)
	if !ok {
		t.Fatalf("expected llm usage map, got %#v", runtime["llm_usage"])
	}
	if usage["input_tokens"] != float64(14) || usage["output_tokens"] != float64(6) || usage["total_tokens"] != float64(20) {
		t.Fatalf("unexpected scanned usage tokens: %#v", usage)
	}
	if usage["cached_input_tokens"] != float64(4) || usage["reasoning_tokens"] != float64(2) {
		t.Fatalf("unexpected scanned usage breakdown: %#v", usage)
	}
	if usage["request_model"] != "test-model" || usage["api_style"] != "responses" || usage["tool_rounds"] != float64(1) {
		t.Fatalf("unexpected scanned usage metadata: %#v", usage)
	}
}

type fakeAgentInvocationRow struct {
	values []any
}

func (r fakeAgentInvocationRow) Scan(dest ...any) error {
	if len(dest) != len(r.values) {
		return fmt.Errorf("expected %d scan destinations, got %d", len(r.values), len(dest))
	}
	for i, value := range r.values {
		if err := assignScanValue(dest[i], value); err != nil {
			return fmt.Errorf("scan value %d: %w", i, err)
		}
	}
	return nil
}

func assignScanValue(dest any, value any) error {
	destination := reflect.ValueOf(dest)
	if destination.Kind() != reflect.Ptr || destination.IsNil() {
		return fmt.Errorf("destination is not a pointer")
	}
	target := destination.Elem()
	if !target.CanSet() {
		return fmt.Errorf("destination cannot be set")
	}
	if value == nil {
		target.Set(reflect.Zero(target.Type()))
		return nil
	}
	source := reflect.ValueOf(value)
	if source.Type().AssignableTo(target.Type()) {
		target.Set(source)
		return nil
	}
	if source.Type().ConvertibleTo(target.Type()) {
		target.Set(source.Convert(target.Type()))
		return nil
	}
	return fmt.Errorf("cannot assign %T to %T", value, dest)
}
