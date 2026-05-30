package store

import (
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
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
