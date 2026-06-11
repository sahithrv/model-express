package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestValidatePlannedExperimentRejectsDirectVisualAnalysisTemplate(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 6)
	experiment.Template = "analyze_dataset_visuals"

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "template") {
		t.Fatalf("expected direct visual-analysis template to be rejected, got %v", err)
	}
}

func TestDatasetVisualAnalysisResultPersistsAcceptedEvidenceAndInvocation(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"class_count":  2,
		"total_images": 4,
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	router := NewRouter(memoryStore)
	body, _ := json.Marshal(validVisualAnalysisPayload(dataset))
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}

	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	if latest.ValidationStatus != datasets.VisualValidationStatusAccepted || latest.SourceInvocationID == "" {
		t.Fatalf("expected accepted persisted analysis with invocation, got %#v", latest)
	}
	invocation, err := memoryStore.GetAgentInvocation(latest.SourceInvocationID)
	if err != nil {
		t.Fatalf("get invocation: %v", err)
	}
	if invocation.AgentName != datasets.VisualAnalysisAgentName || invocation.ValidationStatus != memory.InvocationValidationValid {
		t.Fatalf("unexpected invocation: %#v", invocation)
	}
}

func TestDatasetVisualAnalysisResultRejectsExecutionAuthority(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4, "class_count": 2})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["raw_output"] = `{"proposed_experiments":[{"template":"train_experiment"}]}`
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected rejected result status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	analyses, err := memoryStore.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		t.Fatalf("list visual analyses: %v", err)
	}
	if len(analyses) != 1 || analyses[0].ValidationStatus != datasets.VisualValidationStatusRejected {
		t.Fatalf("expected rejected visual analysis row, got %#v", analyses)
	}
}

func TestDatasetVisualAnalysisResultRejectsSampleManifestPathLeakage(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4, "class_count": 2})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["sample_manifest"] = []map[string]any{
		{"image_id": "img_1", "class_name": "daisy", "width": 100, "height": 100, "selection_basis": []string{"class_representative"}, "metadata": map[string]any{"path": "s3://bucket/daisy/1.jpg"}},
		{"image_id": "img_2", "class_name": "rose", "width": 120, "height": 100, "selection_basis": []string{"class_representative"}},
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected rejected path-leaking result status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
}

func TestDatasetVisualAnalysisResultDowngradesBBoxHypothesisWithoutProfileEvidence(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4, "class_count": 2})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["preprocessing_hypotheses"] = append(payload["preprocessing_hypotheses"].([]map[string]any), map[string]any{
		"id":              "vh_002",
		"mechanism":       "bbox_crop_ablation",
		"summary":         "Try bbox crops if annotations exist.",
		"evidence":        []string{"Subjects look small in sampled images."},
		"expected_effect": "Improve foreground focus if backend validates bbox annotations.",
		"confidence":      "medium",
		"support_status":  "needs_backend_validation",
		"suggested_preprocessing": map[string]any{
			"crop_strategy": "bbox_crop_ablation",
			"bbox_mode":     "crop_and_compare_full_image",
		},
	})
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted evidence-only bbox result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}

	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	var bboxHypothesis *datasets.PreprocessingHypothesis
	for index := range latest.PreprocessingHypotheses {
		if latest.PreprocessingHypotheses[index].ID == "vh_002" {
			bboxHypothesis = &latest.PreprocessingHypotheses[index]
			break
		}
	}
	if bboxHypothesis == nil {
		t.Fatalf("expected downgraded bbox hypothesis to be retained, got %#v", latest.PreprocessingHypotheses)
	}
	if bboxHypothesis.SupportStatus != "unsupported" || bboxHypothesis.SuggestedPreprocessing != nil {
		t.Fatalf("expected bbox hypothesis to be evidence-only unsupported, got %#v", bboxHypothesis)
	}
	if !strings.Contains(bboxHypothesis.UnsupportedReason, "backend-profiled bbox evidence") {
		t.Fatalf("expected bbox unsupported reason to cite backend evidence, got %q", bboxHypothesis.UnsupportedReason)
	}
}

func TestDatasetVisualAnalysisResultKeepsBBoxHypothesisWithProfileEvidence(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"total_images": 4,
		"class_count":  2,
		"metadata_summary": map[string]any{
			"artifact_counts": map[string]any{"annotation_json": 2},
		},
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["preprocessing_hypotheses"] = append(payload["preprocessing_hypotheses"].([]map[string]any), map[string]any{
		"id":              "vh_002",
		"mechanism":       "bbox_crop_ablation",
		"summary":         "Compare bbox crops against full-image training.",
		"evidence":        []string{"Backend profile includes annotation artifacts."},
		"expected_effect": "Improve foreground focus without changing labels.",
		"confidence":      "medium",
		"support_status":  "needs_backend_validation",
		"suggested_preprocessing": map[string]any{
			"crop_strategy": "bbox_crop_ablation",
			"bbox_mode":     "crop_and_compare_full_image",
		},
	})
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted bbox-backed result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	bboxHypothesis := latest.PreprocessingHypotheses[len(latest.PreprocessingHypotheses)-1]
	if bboxHypothesis.SupportStatus != "needs_backend_validation" || bboxHypothesis.SuggestedPreprocessing == nil {
		t.Fatalf("expected bbox hypothesis to remain planner-visible but gated, got %#v", bboxHypothesis)
	}
}

func TestDatasetVisualAnalysisResultKeepsBBoxHypothesisWithActiveMetadataEvidence(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "cub", "s3://bucket/cub.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4, "class_count": 1})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	if _, err := memoryStore.CreateDatasetMetadataImport(datasets.DatasetMetadataImport{
		DatasetID:  dataset.ID,
		ProjectID:  project.ID,
		Status:     datasets.MetadataImportStatusSucceeded,
		SourceKind: datasets.MetadataSourceKindWorkerDiscovery,
		Active:     true,
		Classes: []datasets.DatasetClass{{
			DatasetID:  dataset.ID,
			ClassKey:   "bird",
			ClassName:  "bird",
			ClassIndex: intPtr(1),
		}, {
			DatasetID:  dataset.ID,
			ClassKey:   "other",
			ClassName:  "other",
			ClassIndex: intPtr(2),
		}},
		ManifestRecords: []datasets.DatasetManifestRecord{{
			DatasetID:    dataset.ID,
			SampleKey:    "images/bird/one.jpg",
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: "images/bird/one.jpg",
			LabelKey:     "bird",
			LabelName:    "bird",
		}, {
			DatasetID:    dataset.ID,
			SampleKey:    "images/other/two.jpg",
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: "images/other/two.jpg",
			LabelKey:     "other",
			LabelName:    "other",
		}},
		Annotations: []datasets.DatasetAnnotation{{
			DatasetID:      dataset.ID,
			SampleKey:      "images/bird/one.jpg",
			AnnotationType: datasets.MetadataAnnotationBBox,
			BBox:           map[string]any{"xmin": 1, "ymin": 2, "xmax": 20, "ymax": 30},
		}},
	}); err != nil {
		t.Fatalf("create metadata import: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["preprocessing_hypotheses"] = append(payload["preprocessing_hypotheses"].([]map[string]any), map[string]any{
		"id":              "vh_002",
		"mechanism":       "bbox_crop_ablation",
		"summary":         "Compare bbox crops against full-image training.",
		"evidence":        []string{"Active normalized metadata import reports bbox annotations."},
		"expected_effect": "Improve foreground focus without changing labels.",
		"confidence":      "medium",
		"support_status":  "needs_backend_validation",
		"suggested_preprocessing": map[string]any{
			"crop_strategy": "bbox_crop_ablation",
			"bbox_mode":     "crop_and_compare_full_image",
		},
	})
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted active-metadata bbox result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	bboxHypothesis := latest.PreprocessingHypotheses[len(latest.PreprocessingHypotheses)-1]
	if bboxHypothesis.SupportStatus != "needs_backend_validation" || bboxHypothesis.SuggestedPreprocessing == nil {
		t.Fatalf("expected active metadata to preserve bbox hypothesis, got %#v", bboxHypothesis)
	}
}

func TestDatasetVisualAnalysisResultAcceptsRealisticLLMOutputContract(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"total_images": 128,
		"class_count":  3,
		"metadata_summary": map[string]any{
			"artifact_counts": map[string]any{"annotation_json": 2},
		},
		"visual_trait_summary": map[string]any{
			"bbox_count": 2,
		},
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := realisticVisualLLMOutput(dataset)
	rawOutput, _ := json.Marshal(payload)
	payload["project_id"] = dataset.ProjectID
	payload["provider"] = "fake"
	payload["model"] = "fake-vision"
	payload["source_job_id"] = "job_visual_contract"
	payload["raw_output"] = string(rawOutput)
	payload["sample_manifest"] = realisticVisualSampleManifest()
	payload["input_context"] = map[string]any{
		"caps":                    map[string]any{"max_total_images": 48, "max_image_bytes": 350000},
		"sample_manifest_summary": map[string]any{"images_analyzed": 4},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected realistic LLM output to pass visual validation, got %d: %s", resp.Code, resp.Body.String())
	}

	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	if latest.ValidationStatus != datasets.VisualValidationStatusAccepted || len(latest.ValidationErrors) != 0 {
		t.Fatalf("expected accepted analysis without validation errors, got %#v", latest)
	}
	if len(latest.PreprocessingHypotheses) != 4 {
		t.Fatalf("expected all realistic hypotheses to persist, got %#v", latest.PreprocessingHypotheses)
	}
	byID := map[string]datasets.PreprocessingHypothesis{}
	for _, hypothesis := range latest.PreprocessingHypotheses {
		byID[hypothesis.ID] = hypothesis
	}
	if byID["vh_001"].SuggestedPreprocessing == nil || byID["vh_001"].SuggestedPreprocessing.CropStrategy != "center_crop" {
		t.Fatalf("expected resolution crop preprocessing to persist, got %#v", byID["vh_001"])
	}
	if byID["vh_002"].SuggestedAugmentationConfig == nil || byID["vh_002"].SuggestedAugmentationConfig.PolicyType != "mixup" {
		t.Fatalf("expected MixUp config to persist, got %#v", byID["vh_002"])
	}
	if byID["vh_003"].SupportStatus == "unsupported" || byID["vh_003"].SuggestedPreprocessing == nil {
		t.Fatalf("expected bbox hypothesis to stay gated but accepted with backend evidence, got %#v", byID["vh_003"])
	}
	if byID["vh_004"].SuggestedAugmentationConfig == nil || byID["vh_004"].SuggestedAugmentationConfig.PolicyType != "randaugment" {
		t.Fatalf("expected RandAugment config to persist, got %#v", byID["vh_004"])
	}
}

func TestProfileHasBBoxEvidenceUsesProfiledCounts(t *testing.T) {
	if !profileHasBBoxEvidence(map[string]any{
		"visual_trait_summary": map[string]any{"bbox_count": 3},
	}) {
		t.Fatal("expected parsed bbox trait count to satisfy bbox evidence")
	}
	if !profileHasBBoxEvidence(map[string]any{
		"metadata_summary": map[string]any{
			"artifact_counts": map[string]any{"annotation_json": 1},
		},
	}) {
		t.Fatal("expected annotation artifact count to satisfy bbox evidence")
	}
	if !profileHasBBoxEvidence(map[string]any{
		"agent_safe_metadata_summary": map[string]any{
			"bbox_annotation_count": 4,
			"annotation_counts":     map[string]any{"bbox": 4},
		},
	}) {
		t.Fatal("expected normalized safe metadata bbox counts to satisfy bbox evidence")
	}
	if profileHasBBoxEvidence(map[string]any{
		"metadata_summary": map[string]any{
			"artifact_counts": map[string]any{"labels_csv": 1},
		},
	}) {
		t.Fatal("labels-only artifact counts should not satisfy bbox evidence")
	}
}

func TestDatasetVisualAnalysisResultAllowsFalseRawImageBoundaryFlags(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"total_images": 4,
		"class_count":  2,
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["input_context"] = map[string]any{
		"sample_manifest_summary":         map[string]any{"images_analyzed": 2},
		"caps":                            map[string]any{"max_total_images": 48, "max_image_bytes": 350000, "max_total_bytes": 8000000},
		"raw_images_included":             false,
		"raw_images_included_for_planner": false,
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted boundary flags status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
}

func TestUnsafeVisualAnalysisContentReasonsExplainMatchedMarker(t *testing.T) {
	if reasons := unsafeVisualAnalysisContentReasons(`{"raw_images_included_for_planner":false,"raw_images_included":false}`); len(reasons) != 0 {
		t.Fatalf("false raw-image boundary flags should not be unsafe, got %v", reasons)
	}
	if reasons := unsafeVisualAnalysisContentReasons(`{"max_image_bytes":350000,"prepared_bytes":12345}`); len(reasons) != 0 {
		t.Fatalf("safe byte-count telemetry should not be unsafe, got %v", reasons)
	}
	reasons := unsafeVisualAnalysisContentReasons(`{"image_inputs":[{"data_base64":"abc","image_bytes":"raw"}],"path":"C:\\Users\\demo\\image.jpg"}`)
	blob := strings.Join(reasons, " ")
	for _, want := range []string{"image inputs field", "base64 image field", "image bytes field", "Windows local path"} {
		if !strings.Contains(blob, want) {
			t.Fatalf("expected reason %q in %v", want, reasons)
		}
	}
}

func TestDatasetVisualAnalysisResultSanitizesRawImageInputContext(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4, "class_count": 2})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["input_context"] = map[string]any{
		"caps":       map[string]any{"max_image_bytes": 350000},
		"raw_images": []map[string]any{{"image_id": "img_1", "data_base64": "abc"}},
		"unsafe":     "C:\\Users\\demo\\image.jpg",
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted result with sanitized telemetry status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	invocation, err := memoryStore.GetAgentInvocation(latest.SourceInvocationID)
	if err != nil {
		t.Fatalf("get invocation: %v", err)
	}
	contextJSON, _ := json.Marshal(invocation.InputContext)
	contextText := string(contextJSON)
	if strings.Contains(contextText, "data_base64") || strings.Contains(contextText, "Users") || strings.Contains(contextText, `"raw_images":[`) {
		t.Fatalf("expected unsafe telemetry to be dropped, got %s", contextText)
	}
	caps, ok := invocation.InputContext["caps"].(map[string]any)
	if !ok || caps["max_image_bytes"] == nil {
		t.Fatalf("expected safe caps to be retained, got %#v", invocation.InputContext)
	}
}

func TestVisualAnalysisInvocationContextPreservesBoundaryFlags(t *testing.T) {
	context := visualAnalysisInvocationContext(map[string]any{
		"caps":                            map[string]any{"max_total_images": 48, "max_image_bytes": 350000},
		"raw_images_included":             true,
		"raw_visual_output_included":      true,
		"visual_agent_prompt_for_planner": true,
		"image_inputs":                    []map[string]any{{"data_base64": "abc"}},
		"unsafe":                          "file:///tmp/image.jpg",
	}, []datasets.VisualSampleManifestItem{{ImageID: "img_1"}})

	if context["raw_images_included"] != false || context["raw_visual_output_included"] != false || context["visual_agent_prompt_for_planner"] != false {
		t.Fatalf("expected reserved boundary flags to remain false, got %#v", context)
	}
	if _, ok := context["unsafe"]; ok {
		t.Fatalf("expected unsafe telemetry key to be dropped, got %#v", context)
	}
	if _, ok := context["image_inputs"]; ok {
		t.Fatalf("expected image_inputs telemetry key to be dropped, got %#v", context)
	}
	caps, ok := context["caps"].(map[string]any)
	if !ok || caps["max_image_bytes"] == nil || context["visual_agent_stream_is_separate"] != true {
		t.Fatalf("expected safe telemetry and separate-stream marker, got %#v", context)
	}
}

func TestRunDatasetVisualAnalysisQueuesBackendOwnedJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	metadataImport, err := memoryStore.CreateDatasetMetadataImport(datasets.DatasetMetadataImport{
		DatasetID:  dataset.ID,
		ProjectID:  project.ID,
		Status:     datasets.MetadataImportStatusSucceeded,
		SourceKind: datasets.MetadataSourceKindWorkerDiscovery,
		Active:     true,
		ManifestRecords: []datasets.DatasetManifestRecord{{
			DatasetID:    dataset.ID,
			SampleKey:    "images/daisy/one.jpg",
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: "images/daisy/one.jpg",
			LabelKey:     "daisy",
			LabelName:    "daisy",
		}},
		Annotations: []datasets.DatasetAnnotation{{
			DatasetID:      dataset.ID,
			SampleKey:      "images/daisy/one.jpg",
			AnnotationType: datasets.MetadataAnnotationBBox,
			BBox:           map[string]any{"xmin": 1, "ymin": 2, "xmax": 20, "ymax": 30},
		}},
	})
	if err != nil {
		t.Fatalf("create metadata import: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected run status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	found := false
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateAnalyzeDatasetVisuals && job.Config["dataset_id"] == dataset.ID {
			found = true
			if job.Config["metadata_import_id"] != metadataImport.ID {
				t.Fatalf("expected active metadata import id in visual job config, got %#v", job.Config)
			}
			summary, ok := job.Config["agent_safe_metadata_summary"].(map[string]any)
			if !ok || payloadNumber(summary["bbox_annotation_count"]) <= 0 {
				t.Fatalf("expected active safe metadata summary with bbox evidence, got %#v", job.Config)
			}
		}
	}
	if !found {
		t.Fatalf("expected analyze_dataset_visuals job, got %#v", projectJobs)
	}
}

func TestRunDatasetVisualAnalysisDefaultsToConfiguredTrainingProvider(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER", "modal")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected run status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}

	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateAnalyzeDatasetVisuals {
			if job.Config["provider"] != "modal" {
				t.Fatalf("expected visual analysis job to inherit modal provider, got %#v", job.Config)
			}
			return
		}
	}
	t.Fatalf("expected analyze_dataset_visuals job, got %#v", projectJobs)
}

func TestRunDatasetVisualAnalysisRespectsCooldownAndMaxRuns(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "60")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "3")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	if _, err := memoryStore.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:          project.ID,
		DatasetID:          dataset.ID,
		DatasetName:        dataset.Name,
		ProfileFingerprint: datasetProfileFingerprint(dataset.Profile),
		SchemaVersion:      datasets.VisualAnalysisSchemaVersion,
		TriggerReason:      datasets.VisualTriggerInitialProfile,
		TotalImages:        4,
		ImagesAnalyzed:     2,
		Confidence:         "medium",
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAvailable:    4,
			ImagesAnalyzed:     2,
			ClassesTotal:       2,
			ClassesCovered:     2,
			ClassCoverageRatio: 1,
		},
	}); err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusConflict {
		t.Fatalf("expected cooldown conflict, got %d: %s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "cooling down") {
		t.Fatalf("expected cooldown policy in response, got %s", resp.Body.String())
	}

	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "1")
	req = httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusConflict {
		t.Fatalf("expected max-run conflict, got %d: %s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "per-profile cap") {
		t.Fatalf("expected max-run policy in response, got %s", resp.Body.String())
	}
}

func TestTrainingRunEvaluationQueuesDeficiencyVisualReanalysis(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "3")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 40, "class_count": 2}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create training job: %v", err)
	}
	runtimeSeconds := 10.0
	costUSD := 0.1
	macroF1 := 0.42
	accuracy := 0.72
	epochs := 3
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:            "mobilenet_v3_small",
		Provider:         "local",
		Status:           jobs.StatusSucceeded,
		RuntimeSeconds:   &runtimeSeconds,
		EstimatedCostUSD: &costUSD,
		BestMacroF1:      &macroF1,
		BestAccuracy:     &accuracy,
		EpochsCompleted:  &epochs,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}

	router := NewRouter(memoryStore)
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)

	body := `{
		"training_attempt_id":"` + callbackAttemptID(t, assigned) + `",
		"objective_profile":{"target_metric":"macro_f1"},
		"per_class_metrics":{"rare":{"recall":0.18,"f1-score":0.16},"common":{"recall":0.91,"f1-score":0.88}},
		"confusion_matrix":[[12,18],[2,30]],
		"model_profile":{},
		"holistic_scores":{},
		"recommendation_summary":"low rare-class recall"
	}`
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-evaluation", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected evaluation accepted, got %d: %s", resp.Code, resp.Body.String())
	}

	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	found := false
	for _, candidate := range projectJobs {
		if candidate.Template != jobs.TemplateAnalyzeDatasetVisuals {
			continue
		}
		found = true
		if candidate.Config["trigger_reason"] != string(datasets.VisualTriggerDeficiencyReanalysis) {
			t.Fatalf("expected deficiency trigger, got %#v", candidate.Config)
		}
		triggerDetails, _ := candidate.Config["trigger_details"].(map[string]any)
		triggers, _ := triggerDetails["deficiency_triggers"].([]string)
		if len(triggers) == 0 {
			t.Fatalf("expected deficiency triggers in job config, got %#v", candidate.Config)
		}
	}
	if !found {
		t.Fatalf("expected deficiency visual-analysis job, got %#v", projectJobs)
	}
}

func TestListDatasetVisualAnalysesExposesRerunPolicy(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/datasets/"+dataset.ID+"/visual-analyses", nil)
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected list status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	policy, ok := payload["rerun_policy"].(map[string]any)
	if !ok {
		t.Fatalf("expected rerun_policy object, got %#v", payload)
	}
	if allowed, _ := policy["manual_run_allowed"].(bool); !allowed {
		t.Fatalf("expected manual rerun to be allowed, got %#v", policy)
	}
	if policy["profile_fingerprint"] == "" {
		t.Fatalf("expected profile fingerprint in policy, got %#v", policy)
	}
}

func TestProfileUpdateQueuesVisualAnalysisBeforeInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}

	body, _ := json.Marshal(map[string]any{"profile": visualGateProfile()})
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/profile", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected profile status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}

	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 0 {
		t.Fatalf("expected initial plan to wait for visual analysis, got %#v", plans)
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if !hasVisualAnalysisJobForDataset(projectJobs, dataset.ID) {
		t.Fatalf("expected initial visual-analysis job before planning, got %#v", projectJobs)
	}
}

func TestInitialVisualAnalysisResultCreatesInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, visualGateProfile()); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	body, _ := json.Marshal(validVisualAnalysisPayload(dataset))
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected visual result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 1 || plans[0].DatasetID != dataset.ID {
		t.Fatalf("expected visual result to unblock initial plan, got %#v", plans)
	}
}

func TestInitialVisualAnalysisFailureFallsBackToInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}

	body, _ := json.Marshal(map[string]any{"profile": visualGateProfile()})
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/profile", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected profile status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	var visualJob jobs.ExperimentJob
	for _, candidate := range projectJobs {
		if candidate.Template == jobs.TemplateAnalyzeDatasetVisuals {
			visualJob = candidate
			break
		}
	}
	if visualJob.ID == "" {
		t.Fatalf("expected visual job, got %#v", projectJobs)
	}

	router := NewRouter(memoryStore)
	worker, err := memoryStore.RegisterWorker(project.ID, "visual worker", "")
	if err != nil {
		t.Fatalf("register visual worker: %v", err)
	}
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)

	req = httptest.NewRequest(http.MethodPost, "/jobs/"+visualJob.ID+"/fail", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assigned)+`","error":"visual llm unavailable"}`))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected visual fail fallback status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 1 || plans[0].DatasetID != dataset.ID {
		t.Fatalf("expected visual failure to fall back to initial plan, got %#v", plans)
	}
}

func visualGateProfile() map[string]any {
	return map[string]any{
		"schema_version":        "dataset_profile_v1",
		"task_type":             "image_classification",
		"class_count":           2,
		"total_images":          120,
		"imbalance_ratio":       1.2,
		"corrupt_image_count":   0,
		"class_names":           []string{"daisy", "rose"},
		"class_distribution":    map[string]int{"daisy": 60, "rose": 60},
		"image_dimension_stats": map[string]any{"width": map[string]any{"median": 224}, "height": map[string]any{"median": 224}},
	}
}

func hasVisualAnalysisJobForDataset(projectJobs []jobs.ExperimentJob, datasetID string) bool {
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateAnalyzeDatasetVisuals && job.Config["dataset_id"] == datasetID {
			return true
		}
	}
	return false
}

func realisticVisualLLMOutput(dataset datasets.Dataset) map[string]any {
	return map[string]any{
		"schema_version":  datasets.VisualAnalysisSchemaVersion,
		"dataset_id":      dataset.ID,
		"dataset_name":    dataset.Name,
		"total_images":    128,
		"images_analyzed": 4,
		"trigger_reason":  "initial_profile",
		"confidence":      "medium",
		"coverage_report": map[string]any{
			"selection_strategy":      "deterministic_risk_and_representative_sampling",
			"selection_basis":         []string{"class_representative", "aspect_ratio_outlier", "bbox_object_scale_outlier"},
			"images_available":        128,
			"images_analyzed":         4,
			"classes_total":           3,
			"classes_covered":         3,
			"class_coverage_ratio":    1,
			"per_class_counts":        map[string]int{"daisy": 2, "rose": 1, "tulip": 1},
			"hard_example_count":      1,
			"edge_case_count":         2,
			"high_detail_image_count": 2,
			"limitations":             []string{"Bounded visual sample, not full dataset inspection."},
		},
		"visual_traits": []map[string]any{
			{
				"trait":             "small_objects",
				"level":             "medium",
				"confidence":        "medium",
				"evidence":          []string{"img_001 and img_004 show foreground objects occupying a small region."},
				"example_image_ids": []string{"img_001", "img_004"},
				"affected_classes":  []string{"daisy", "tulip"},
			},
			{
				"trait":             "background_dominance",
				"level":             "high",
				"confidence":        "medium",
				"evidence":          []string{"Several sampled images contain broad background around the subject."},
				"example_image_ids": []string{"img_001", "img_003"},
			},
			{
				"trait":             "fine_grained_similarity",
				"level":             "medium",
				"confidence":        "low",
				"evidence":          []string{"Rose and tulip examples share similar color and petal texture cues."},
				"example_image_ids": []string{"img_002", "img_003"},
				"affected_classes":  []string{"rose", "tulip"},
			},
			{
				"trait":             "crop_bbox_useful",
				"level":             "medium",
				"confidence":        "medium",
				"evidence":          []string{"Manifest bbox metadata indicates object-centered crops are plausible."},
				"example_image_ids": []string{"img_004"},
			},
		},
		"classes_to_watch": []map[string]any{
			{
				"class_name":        "rose",
				"reason":            "Texture and color overlap with tulip examples.",
				"related_classes":   []string{"tulip"},
				"evidence":          []string{"img_002 and img_003 share warm petal colors and soft boundaries."},
				"example_image_ids": []string{"img_002", "img_003"},
				"confidence":        "low",
			},
		},
		"preprocessing_hypotheses": []map[string]any{
			{
				"id":                    "vh_001",
				"mechanism":             "resolution_crop",
				"summary":               "Compare preserve-aspect padding and moderate image size for variable framing.",
				"evidence":              []string{"Aspect ratio and foreground scale vary across img_001 through img_004.", "Background dominance appears high in multiple samples."},
				"example_image_ids":     []string{"img_001", "img_003", "img_004"},
				"suggested_image_sizes": []int{256, 288},
				"suggested_preprocessing": map[string]any{
					"resize_strategy": "preserve_aspect_pad",
					"normalization":   "imagenet",
					"crop_strategy":   "center_crop",
					"bbox_mode":       "ignore",
				},
				"expected_effect": "Reduce shape distortion and make foreground scale more consistent.",
				"risk":            "May increase latency compared with the current 224 input.",
				"confidence":      "medium",
				"support_status":  "needs_backend_validation",
			},
			{
				"id":                                   "vh_002",
				"mechanism":                            "augmentation_mixed_sample",
				"summary":                              "Try MixUp for visually similar flower classes.",
				"evidence":                             []string{"Rose and tulip samples have overlapping color and petal texture cues."},
				"example_image_ids":                    []string{"img_002", "img_003"},
				"suggested_augmentation_policy":        "mixup",
				"suggested_augmentation_policy_config": map[string]any{"policy_type": "mixup", "probability": 0.45, "alpha": 0.3},
				"expected_effect":                      "Improve calibration for visually similar classes.",
				"risk":                                 "Can soften labels too much on a small dataset.",
				"confidence":                           "low",
				"support_status":                       "needs_backend_validation",
			},
			{
				"id":                "vh_003",
				"mechanism":         "bbox_crop_ablation",
				"summary":           "Compare bbox-centered crop against full-image training when backend annotations exist.",
				"evidence":          []string{"img_004 includes bbox metadata and a small foreground object."},
				"example_image_ids": []string{"img_004"},
				"suggested_preprocessing": map[string]any{
					"crop_strategy": "bbox_crop_ablation",
					"bbox_mode":     "crop_and_compare_full_image",
					"normalization": "imagenet",
				},
				"expected_effect": "Check whether reducing background area improves foreground class signal.",
				"risk":            "Only valid when backend-profiled bbox annotations are available.",
				"confidence":      "medium",
				"support_status":  "needs_backend_validation",
			},
			{
				"id":                                   "vh_004",
				"mechanism":                            "augmentation_auto",
				"summary":                              "Use a light RandAugment search for brightness and texture variation.",
				"evidence":                             []string{"Lighting and texture vary across the sampled flowers."},
				"example_image_ids":                    []string{"img_001", "img_002", "img_003"},
				"suggested_augmentation_policy":        "randaugment",
				"suggested_augmentation_policy_config": map[string]any{"policy_type": "randaugment", "magnitude": 6, "num_ops": 2, "num_magnitude_bins": 15, "probability": 0.75},
				"expected_effect":                      "Improve robustness to bounded visual variation.",
				"risk":                                 "Aggressive color changes may hurt fine-grained flower cues.",
				"confidence":                           "medium",
				"support_status":                       "needs_backend_validation",
			},
		},
		"cautions": []map[string]any{
			{
				"operation":         "vertical_flip",
				"reason":            "Some flowers have orientation cues in the sampled images.",
				"severity":          "medium",
				"confidence":        "medium",
				"example_image_ids": []string{"img_001", "img_004"},
			},
			{
				"operation":         "strong_color_jitter",
				"reason":            "Color appears class-informative for rose and tulip samples.",
				"severity":          "medium",
				"confidence":        "low",
				"affected_classes":  []string{"rose", "tulip"},
				"example_image_ids": []string{"img_002", "img_003"},
			},
		},
		"limitations": []string{
			"Visual evidence comes from a bounded sample and should not be treated as labels.",
			"Backend validation must approve every runnable preprocessing or augmentation field.",
		},
	}
}

func realisticVisualSampleManifest() []map[string]any {
	return []map[string]any{
		{"image_id": "img_001", "class_name": "daisy", "width": 640, "height": 480, "selection_basis": []string{"class_representative"}},
		{"image_id": "img_002", "class_name": "rose", "width": 512, "height": 512, "selection_basis": []string{"class_representative"}},
		{"image_id": "img_003", "class_name": "tulip", "width": 480, "height": 640, "selection_basis": []string{"aspect_ratio_outlier"}},
		{
			"image_id":        "img_004",
			"class_name":      "daisy",
			"width":           800,
			"height":          600,
			"selection_basis": []string{"bbox_object_scale_outlier"},
			"has_bbox":        true,
			"bbox":            map[string]any{"xmin": 220, "ymin": 160, "xmax": 520, "ymax": 430},
		},
	}
}

func validVisualAnalysisPayload(dataset datasets.Dataset) map[string]any {
	return map[string]any{
		"schema_version":  datasets.VisualAnalysisSchemaVersion,
		"project_id":      dataset.ProjectID,
		"dataset_id":      dataset.ID,
		"dataset_name":    dataset.Name,
		"total_images":    4,
		"images_analyzed": 2,
		"trigger_reason":  "initial_profile",
		"confidence":      "medium",
		"provider":        "fake",
		"model":           "fake-vision",
		"source_job_id":   "job_visual",
		"raw_output":      `{"schema_version":"dataset_visual_analysis_v1"}`,
		"sample_manifest": []map[string]any{
			{"image_id": "img_1", "class_name": "daisy", "width": 100, "height": 100, "selection_basis": []string{"class_representative"}},
			{"image_id": "img_2", "class_name": "rose", "width": 120, "height": 100, "selection_basis": []string{"class_representative"}},
		},
		"coverage_report": map[string]any{
			"selection_strategy":      "deterministic_risk_and_representative_sampling",
			"selection_basis":         []string{"class_representative"},
			"images_available":        4,
			"images_analyzed":         2,
			"classes_total":           2,
			"classes_covered":         2,
			"class_coverage_ratio":    1,
			"per_class_counts":        map[string]int{"daisy": 1, "rose": 1},
			"high_detail_image_count": 0,
			"limitations":             []string{"Sample is bounded."},
		},
		"visual_traits": []map[string]any{
			{
				"trait":             "background_dominance",
				"level":             "medium",
				"confidence":        "medium",
				"evidence":          []string{"Several samples have broad background regions."},
				"example_image_ids": []string{"img_1"},
			},
		},
		"classes_to_watch": []map[string]any{
			{
				"class_name":        "daisy",
				"reason":            "Similar texture in sampled petals.",
				"evidence":          []string{"Petal textures overlap."},
				"example_image_ids": []string{"img_1"},
				"confidence":        "low",
			},
		},
		"preprocessing_hypotheses": []map[string]any{
			{
				"id":                    "vh_001",
				"mechanism":             "resolution_crop",
				"summary":               "Compare preserve_aspect_pad against squash resize.",
				"evidence":              []string{"Aspect ratios vary."},
				"expected_effect":       "Reduce shape distortion.",
				"confidence":            "medium",
				"support_status":        "supported",
				"suggested_image_sizes": []int{224},
				"suggested_preprocessing": map[string]any{
					"resize_strategy": "preserve_aspect_pad",
					"normalization":   "imagenet",
					"crop_strategy":   "none",
					"bbox_mode":       "ignore",
				},
			},
		},
		"cautions": []map[string]any{
			{"operation": "vertical_flip", "reason": "Orientation may matter.", "severity": "medium", "confidence": "medium", "example_image_ids": []string{"img_2"}},
		},
		"limitations": []string{"No experiment should be scheduled from this output without backend validation."},
	}
}

func intPtr(value int) *int {
	return &value
}
