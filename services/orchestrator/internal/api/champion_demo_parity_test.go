package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

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
						"id":         "test:cat",
						"image_id":   "cat.png",
						"uri":        "data:image/png;base64,AAAA",
						"class_name": "cat",
						"label":      "cat",
						"split":      "test",
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
	predictionJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateChampionDemoPrediction)
	imageMetadata, ok := predictionJob.Config["image_metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected image_metadata in prediction job config, got %#v", predictionJob.Config)
	}
	if imageMetadata["demo_source_type"] != "heldout_test_original_bytes" || imageMetadata["parity_safe"] != true {
		t.Fatalf("expected parity-safe heldout metadata, got %#v", imageMetadata)
	}
}
