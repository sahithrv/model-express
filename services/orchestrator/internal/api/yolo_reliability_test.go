package api

import (
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestYOLOEligibilityRequiresYOLOSpecificEvidence(t *testing.T) {
	bboxOnly := datasets.Dataset{Profile: map[string]any{
		"task_type":                  "object_detection",
		"object_detection_available": true,
		"metadata_summary": map[string]any{
			"available":                   true,
			"bbox_available":              true,
			"object_detection_available":  true,
			"bbox_count":                  42,
			"bbox_per_class":              map[string]any{"bird": 42},
			"yolo_available":              false,
			"yolo_dataset_config_present": false,
		},
		"dataset_traits": []any{"bbox_available", "object_detection"},
	}}
	if datasetHasYOLODetectionEvidence(bboxOnly, map[string]any{}) {
		t.Fatal("bbox-only metadata must not be treated as YOLO-trainable evidence")
	}
	detector := testExperiment("yolo11n.pt", 6)
	detector.Template = "yolo11_detection"
	if err := validateExperimentDatasetCompatibility(detector, bboxOnly, 0); err == nil {
		t.Fatal("expected bbox-only dataset to reject YOLO detector experiment")
	}

	realYOLO := datasets.Dataset{Profile: map[string]any{
		"task_type": "object_detection",
		"metadata_summary": map[string]any{
			"yolo_available":        true,
			"yolo_format":           true,
			"yolo_config_count":     1,
			"yolo_label_file_count": 8,
		},
		"dataset_traits": []any{"yolo_format", "object_detection"},
	}}
	if !datasetHasYOLODetectionEvidence(realYOLO, map[string]any{}) {
		t.Fatal("expected real YOLO metadata to be YOLO-trainable evidence")
	}
	if err := validateExperimentDatasetCompatibility(detector, realYOLO, 0); err != nil {
		t.Fatalf("expected YOLO dataset to accept detector experiment: %v", err)
	}
}

func TestDetectionScoringPreservesZeroYOLOMetrics(t *testing.T) {
	summary := runs.TrainingRunSummary{BestMacroF1: 0.91, BestAccuracy: 0.92}
	evaluation := runs.TrainingRunEvaluation{
		ObjectiveProfile: map[string]any{
			"heldout_test_map50_95":  0.0,
			"heldout_test_map50":     0.0,
			"heldout_test_precision": 0.0,
			"heldout_test_recall":    0.0,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"AP50_95": 0.0, "AP50": 0.0, "precision": 0.0, "recall": 0.0},
			"dog":       map[string]any{"AP50_95": 0.0, "AP50": 0.0, "precision": 0.0, "recall": 0.0},
			"macro avg": map[string]any{"AP50_95": 0.0, "AP50": 0.0, "precision": 0.0, "recall": 0.0},
		},
		HolisticScores: map[string]any{
			"detection_metrics": map[string]any{
				"mAP50_95":  0.0,
				"mAP50":     0.0,
				"precision": 0.0,
				"recall":    0.0,
			},
		},
	}

	if score := validationMetricScore("mAP50_95", summary, evaluation); score != 0 {
		t.Fatalf("expected zero validation detection score, got %.6f", score)
	}
	heldout, hasHeldout := heldoutMetricScore("mAP50_95", evaluation.ObjectiveProfile)
	if !hasHeldout || heldout != 0 {
		t.Fatalf("expected present zero heldout detection score, score=%.6f present=%v", heldout, hasHeldout)
	}
	perClass, hasPerClass := perClassMetricScore(evaluation.PerClassMetrics)
	if !hasPerClass || perClass != 0 {
		t.Fatalf("expected present zero per-class detection score, score=%.6f present=%v", perClass, hasPerClass)
	}
	metrics := map[string]any{}
	addDetectionChampionMetrics(metrics, evaluation)
	for _, key := range []string{"best_map50_95", "best_map50", "best_precision", "best_recall", "primary_metric_value"} {
		value, ok := payloadFloatValue(metrics, key)
		if !ok || value != 0 {
			t.Fatalf("expected %s to be present as zero, got value=%.6f present=%v in %#v", key, value, ok, metrics)
		}
	}
}

func TestEnsureChampionExportQueuesWhenTrustedONNXLacksManifest(t *testing.T) {
	server, memoryStore, projectID, datasetID, job := championExportFixture(t)
	artifactURI := "s3://bucket/model-express/artifacts/" + job.ID + "/model.onnx"
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID: projectID,
		DatasetID: datasetID,
		JobID:     job.ID,
		DeploymentProfile: map[string]any{
			"onnx_artifact_uri": artifactURI,
			"model_profile": map[string]any{
				"onnx_artifact_uri": artifactURI,
				"export_status":     "READY",
			},
		},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	export, err := server.ensureChampionExport(projectID, champion, job, "onnx", "", nil)
	if err != nil {
		t.Fatalf("ensure champion export: %v", err)
	}
	if export.Status == runs.ChampionExportStatusReady || export.ArtifactURI != "" {
		t.Fatalf("bare ONNX URI without manifest must stay pending, got %#v", export)
	}
	if len(export.ValidationErrors) == 0 || !strings.Contains(export.ValidationErrors[0], "valid worker export manifest") {
		t.Fatalf("expected manifest validation error, got %#v", export.ValidationErrors)
	}
	assertExportJobQueued(t, memoryStore, projectID, export.ID)
}

func TestSimulatedYOLOProfileCannotBecomeDeployableChampionExport(t *testing.T) {
	server, memoryStore, projectID, datasetID, job := championExportFixture(t)
	artifactURI := "s3://bucket/model-express/artifacts/" + job.ID + "/model.onnx"
	manifest := validChampionExportManifest("onnx", artifactURI)
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID: projectID,
		DatasetID: datasetID,
		JobID:     job.ID,
		DeploymentProfile: map[string]any{
			"onnx_artifact_uri": artifactURI,
			"export_status":     "READY",
			"export_manifest":   manifest,
			"model_profile": map[string]any{
				"onnx_artifact_uri": artifactURI,
				"export_status":     "READY",
				"export_manifest":   manifest,
				"simulation":        true,
				"exportable":        false,
			},
		},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if export, ok := championDeploymentProfileExport(champion); ok {
		t.Fatalf("simulated YOLO profile must not produce deployable profile export: %#v", export)
	}
	export, err := server.ensureChampionExport(projectID, champion, job, "onnx", "", nil)
	if err != nil {
		t.Fatalf("ensure champion export: %v", err)
	}
	if export.Status == runs.ChampionExportStatusReady || export.ArtifactURI != "" {
		t.Fatalf("simulated YOLO profile must not auto-ready export, got %#v", export)
	}
}

func championExportFixture(t *testing.T) (*Server, *store.MemoryStore, string, string, jobs.ExperimentJob) {
	t.Helper()
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "train detector")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/dataset.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	completed, err := memoryStore.CompleteJob(job.ID, "run_1")
	if err != nil {
		t.Fatalf("complete job: %v", err)
	}
	return server, memoryStore, project.ID, dataset.ID, completed
}

func assertExportJobQueued(t *testing.T, memoryStore store.Store, projectID string, exportID string) {
	t.Helper()
	projectJobs, err := memoryStore.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateExportChampion && jobConfigString(job.Config, "export_id") == exportID {
			return
		}
	}
	t.Fatalf("expected export_champion job for export %s, got %#v", exportID, projectJobs)
}

func validChampionExportManifest(format string, artifactURI string) map[string]any {
	provenance := map[string]any{
		"schema_version":  "worker_artifact_provenance_v1",
		"generated_by":    "model-express-worker",
		"source":          "worker_generated",
		"artifact_format": format,
	}
	return map[string]any{
		"schema_version": "champion_export_manifest_v1",
		"metadata": map[string]any{
			"format":           format,
			"provenance":       provenance,
			"export_self_test": map[string]any{"status": "passed"},
		},
		"artifacts": []any{
			map[string]any{
				"format":     format,
				"status":     "created",
				"uri":        artifactURI,
				"provenance": provenance,
			},
		},
	}
}
