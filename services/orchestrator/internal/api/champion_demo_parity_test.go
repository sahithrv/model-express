package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestChampionDemoImagesExposeYOLOHeldoutEvaluationImages(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	originalURI := "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png"
	if _, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best detection score",
		Evaluation: map[string]any{
			"objective_profile": map[string]any{
				"task_type": "object_detection",
				"heldout_demo_images": []map[string]any{
					{
						"id":                  "val:cat",
						"image_id":            "cat.png",
						"uri":                 originalURI,
						"image_uri":           originalURI,
						"preview_uri":         "data:image/jpeg;base64,THUMB",
						"thumbnail_uri":       "data:image/jpeg;base64,THUMB",
						"original_image_uri":  originalURI,
						"source_artifact_uri": originalURI,
						"class_name":          "cat",
						"label":               "cat",
						"split":               "val",
						"metadata": map[string]any{
							"source":              "heldout_test",
							"demo_source_type":    "heldout_test_original_artifact",
							"parity_safe":         true,
							"task_type":           "object_detection",
							"object_count":        1,
							"source_artifact_uri": originalURI,
						},
					},
				},
			},
		},
	}); err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/champion/demo-images", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var payload struct {
		SourceOfTruth string                    `json:"source_of_truth"`
		Images        []datasets.VisualExemplar `json:"images"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.SourceOfTruth != "champion.evaluation.heldout_demo_images" {
		t.Fatalf("expected champion evaluation source, got %q", payload.SourceOfTruth)
	}
	if len(payload.Images) != 1 {
		t.Fatalf("expected one YOLO heldout demo image, got %#v", payload.Images)
	}
	image := payload.Images[0]
	if image.URI != originalURI || image.ClassName != "cat" || image.Split != "val" {
		t.Fatalf("expected YOLO heldout image to be exposed, got %#v", image)
	}
	if image.Metadata["task_type"] != "object_detection" || image.Metadata["source_artifact_uri"] != originalURI {
		t.Fatalf("expected YOLO demo metadata, got %#v", image.Metadata)
	}
}
func TestChampionDemoPredictionCarriesHeldoutImageMetadataToWorkerJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
		Evaluation: map[string]any{
			"objective_profile": map[string]any{
				"heldout_demo_images": []map[string]any{
					{
						"id":                 "test:cat",
						"image_id":           "cat.png",
						"uri":                "data:image/png;base64,AAAA",
						"original_image_uri": "data:image/png;base64,AAAA",
						"class_name":         "cat",
						"label":              "cat",
						"split":              "test",
						"metadata": map[string]any{
							"source":           "heldout_test",
							"demo_source_type": "heldout_test_original_bytes",
							"parity_safe":      true,
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "file:///exports/champion.onnx",
	}); err != nil {
		t.Fatalf("create export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"data:image/png;base64,AAAA","top_k":3}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction runs.ChampionDemoPrediction `json:"prediction"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Prediction.TrueLabel != "cat" || payload.Prediction.ImageID != "test:cat" {
		t.Fatalf("expected heldout image metadata on prediction, got %#v", payload.Prediction)
	}
	if payload.Prediction.Status != runs.ChampionDemoPredictionStatusRuntimeUnavailable {
		t.Fatalf("expected local-only runtime unavailable prediction, got %#v", payload.Prediction)
	}
	if payload.Prediction.ImageMetadata["demo_source_type"] != "heldout_test_original_bytes" || payload.Prediction.ImageMetadata["parity_safe"] != true {
		t.Fatalf("expected parity-safe heldout metadata, got %#v", payload.Prediction.ImageMetadata)
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateChampionDemoPrediction {
			t.Fatalf("legacy demo endpoint must not create worker prediction job, got %#v", job)
		}
	}
}

func TestChampionDemoPredictionUsesOriginalArtifactForCompactHeldoutImage(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	thumbnailURI := "data:image/jpeg;base64,THUMB"
	originalURI := "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png"
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
		Evaluation: map[string]any{
			"objective_profile": map[string]any{
				"heldout_demo_images": []map[string]any{
					{
						"id":                  "test:cat",
						"image_id":            "cat.png",
						"uri":                 originalURI,
						"image_uri":           originalURI,
						"preview_uri":         thumbnailURI,
						"thumbnail_uri":       thumbnailURI,
						"original_image_uri":  originalURI,
						"source_artifact_uri": originalURI,
						"class_name":          "cat",
						"label":               "cat",
						"split":               "test",
						"metadata": map[string]any{
							"source":              "heldout_test",
							"demo_source_type":    "heldout_test_original_artifact",
							"parity_safe":         true,
							"source_artifact_uri": originalURI,
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "file:///exports/champion.onnx",
	}); err != nil {
		t.Fatalf("create export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"`+thumbnailURI+`","top_k":3}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction runs.ChampionDemoPrediction `json:"prediction"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Prediction.TrueLabel != "cat" || payload.Prediction.ImageURI != thumbnailURI {
		t.Fatalf("expected requested thumbnail to stay on prediction with heldout label, got %#v", payload.Prediction)
	}
	if payload.Prediction.Status != runs.ChampionDemoPredictionStatusRuntimeUnavailable {
		t.Fatalf("expected local-only runtime unavailable prediction, got %#v", payload.Prediction)
	}
	if payload.Prediction.ImageMetadata["requested_image_uri"] != thumbnailURI || payload.Prediction.ImageMetadata["backend_image_uri"] != originalURI {
		t.Fatalf("expected thumbnail request and backend original URI metadata, got %#v", payload.Prediction.ImageMetadata)
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateChampionDemoPrediction {
			t.Fatalf("legacy demo endpoint must not create worker prediction job, got %#v", job)
		}
	}
}

func TestChampionDemoPredictionRejectsThumbnailOnlyHeldoutImage(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	thumbnailURI := "data:image/jpeg;base64,THUMB"
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
		Evaluation: map[string]any{
			"objective_profile": map[string]any{
				"heldout_demo_images": []map[string]any{
					{
						"id":            "test:cat",
						"image_id":      "cat.png",
						"uri":           thumbnailURI,
						"image_uri":     thumbnailURI,
						"preview_uri":   thumbnailURI,
						"thumbnail_uri": thumbnailURI,
						"class_name":    "cat",
						"label":         "cat",
						"split":         "test",
						"metadata": map[string]any{
							"source":           "heldout_test",
							"demo_source_type": "heldout_test_thumbnail_preview",
							"preview_uri":      thumbnailURI,
							"thumbnail_uri":    thumbnailURI,
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "file:///exports/champion.onnx",
	}); err != nil {
		t.Fatalf("create export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"`+thumbnailURI+`","top_k":3}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction       runs.ChampionDemoPrediction `json:"prediction"`
		RuntimeAvailable bool                        `json:"runtime_available"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.RuntimeAvailable || payload.Prediction.Status != runs.ChampionDemoPredictionStatusFailed {
		t.Fatalf("expected failed non-runtime prediction, got %#v", payload)
	}
	if !strings.Contains(payload.Prediction.Error, championDemoOriginalUnavailableCode) {
		t.Fatalf("expected original unavailable error, got %q", payload.Prediction.Error)
	}
	if payload.Prediction.ImageMetadata["error_code"] != championDemoOriginalUnavailableCode {
		t.Fatalf("expected error code metadata, got %#v", payload.Prediction.ImageMetadata)
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateChampionDemoPrediction {
			t.Fatalf("expected no worker prediction job for thumbnail-only heldout image, got %#v", job)
		}
	}
}
