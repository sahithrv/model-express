package api

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/store"
)

func datasetPlanningInsights(dataset datasets.Dataset, agentSafeMetadataSummary map[string]any) agents.DatasetPlanningInsights {
	profile := dataset.Profile
	taskType := profileString(profile, "task_type")
	if taskType == "" {
		taskType = "image_classification"
	}
	classCount := profileInt(profile, "class_count")
	totalImages := profileInt(profile, "total_images")
	imageCount := profileInt(profile, "image_count")
	if metadataClassCount := int(payloadNumber(agentSafeMetadataSummary["class_count"])); metadataClassCount > classCount {
		classCount = metadataClassCount
	}
	if metadataSampleCount := int(payloadNumber(agentSafeMetadataSummary["sample_count"])); metadataSampleCount > 0 {
		if totalImages == 0 {
			totalImages = metadataSampleCount
		}
		if imageCount == 0 {
			imageCount = metadataSampleCount
		}
	}
	if totalImages == 0 {
		totalImages = imageCount
	}
	if imageCount == 0 {
		imageCount = totalImages
	}
	imbalanceRatio := profileFloat(profile, "imbalance_ratio")
	corruptImageCount := profileInt(profile, "corrupt_image_count")
	corruptFileCount := profileInt(profile, "corrupt_file_count")
	if corruptImageCount == 0 {
		corruptImageCount = corruptFileCount
	}
	if corruptFileCount == 0 {
		corruptFileCount = corruptImageCount
	}
	widthMin := profileInt(profile, "width_min")
	widthMax := profileInt(profile, "width_max")
	heightMin := profileInt(profile, "height_min")
	heightMax := profileInt(profile, "height_max")
	classDistribution := profileMap(profile, "class_distribution")
	if len(classDistribution) == 0 {
		classDistribution = profileMap(profile, "images_per_class")
	}
	imageDimensionStats := profileMap(profile, "image_dimension_stats")
	splitSummary := profileMap(profile, "split_summary")
	metadataSummary := safeLegacyMetadataSummary(profileMap(profile, "metadata_summary"))
	if len(agentSafeMetadataSummary) > 0 {
		metadataSummary = mergeMetadataEvidenceSummary(metadataSummary, agentSafeMetadataSummary)
	}
	yoloDetectionEvidence := profileHasYOLODetectionEvidence(profile) || metadataSummaryHasYOLOEvidence(metadataSummary)
	if yoloDetectionEvidence {
		taskType = "object_detection"
	}
	leakageWarnings := profileStringSlice(profile, "leakage_warnings")
	datasetTraits := profileStringSlice(profile, "dataset_traits")
	artifacts := profileMapSlice(profile, "artifacts")

	constraints := []string{}
	recommendedPreprocessing := []string{"normalize with ImageNet statistics for transfer learning"}
	recommendedAugmentations := []string{}
	recommendedMetrics := []string{"accuracy", "macro_f1"}
	liveInferencePriorities := []string{
		"Prefer compact architectures when quality is close so the final model can classify live images with low latency.",
		"Only increase image_size when prior results show a meaningful quality gain over the deployment cost.",
	}
	if yoloDetectionEvidence {
		recommendedMetrics = []string{"mAP50_95", "mAP50", "precision", "recall", "box_loss", "cls_loss", "dfl_loss", "latency_p95_ms"}
		recommendedPreprocessing = []string{"preserve YOLO normalized bounding boxes and official train/val/test splits", "use 640px detector inputs as the default baseline"}
		recommendedAugmentations = append(recommendedAugmentations, "YOLO-safe mosaic/scale/color augmentation once the detection worker is enabled")
		liveInferencePriorities = []string{
			"Prefer YOLO nano/small detectors first for live streams, then challenge with medium only when p95 latency remains inside budget.",
			"Rank detector champions by validation/test detection loss, mAP, recall, export parity, p95 latency, and frame-level stability.",
		}
		constraints = append(constraints, "YOLO object-detection files detected; keep bounding-box supervision and do not collapse this dataset into image classification.")
	}

	if totalImages == 0 {
		constraints = append(constraints, "Dataset has not been profiled yet; use conservative transfer-learning defaults and prioritize profiling before aggressive search.")
	} else if totalImages < 500 {
		if yoloDetectionEvidence {
			constraints = append(constraints, "Small detection dataset; preserve official splits, watch recall/mAP variance, and avoid trusting a single validation slice.")
		} else {
			constraints = append(constraints, "Small dataset; avoid overfitting and prefer stronger augmentation, early stopping, and regularization.")
			recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter", "random_crop")
		}
	} else if totalImages < 2000 {
		if yoloDetectionEvidence {
			constraints = append(constraints, "Medium-small detection dataset; compare nano/small YOLO detectors before raising capacity.")
		} else {
			constraints = append(constraints, "Medium-small dataset; compare efficient transfer models with moderate augmentation.")
			recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter")
		}
	}
	if imbalanceRatio >= 1.5 {
		constraints = append(constraints, fmt.Sprintf("Class imbalance detected (ratio %.2f); optimize macro-F1 and test class balancing.", imbalanceRatio))
		recommendedPreprocessing = append(recommendedPreprocessing, "class balancing with weighted_loss")
		recommendedMetrics = append(recommendedMetrics, "per_class_f1")
	}
	if corruptImageCount > 0 {
		constraints = append(constraints, fmt.Sprintf("%d corrupt image(s) were detected; clean or skip them before trusting final metrics.", corruptImageCount))
	}
	if len(leakageWarnings) > 0 {
		constraints = append(constraints, leakageWarnings...)
	}
	if metadataSummaryHasBBoxEvidence(metadataSummary) || containsString(datasetTraits, "bbox_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "bbox_crop_if_available as an ablation against full-image training")
	}
	if metadataBool(metadataSummary, "metadata_available") || containsString(datasetTraits, "metadata_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "preserve metadata artifacts for controlled preprocessing ablations")
	}
	if widthMax > 0 && heightMax > 0 {
		maxDimension := widthMax
		if heightMax > maxDimension {
			maxDimension = heightMax
		}
		minDimension := widthMin
		if heightMin > 0 && (minDimension == 0 || heightMin < minDimension) {
			minDimension = heightMin
		}
		if yoloDetectionEvidence {
			if maxDimension >= 512 {
				recommendedPreprocessing = append(recommendedPreprocessing, "start YOLO detector training at 640 image size and only raise resolution when small-object recall needs it")
			} else if maxDimension <= 160 {
				recommendedPreprocessing = append(recommendedPreprocessing, "verify small-input upscaling does not inflate false positives before trusting live detections")
			}
		} else if maxDimension >= 512 {
			recommendedPreprocessing = append(recommendedPreprocessing, "compare 224 and 256 image_size before trying larger inputs")
		} else if maxDimension <= 160 {
			recommendedPreprocessing = append(recommendedPreprocessing, "avoid unnecessary upscaling beyond 224 unless validation gains justify latency")
		}
		if minDimension > 0 && maxDimension > minDimension*2 {
			if yoloDetectionEvidence {
				constraints = append(constraints, "Large variation in detection image dimensions; preserve box coordinates and use detector-native letterbox resizing.")
			} else {
				constraints = append(constraints, "Large variation in image dimensions; prefer resize plus random crop to improve robustness.")
				recommendedAugmentations = append(recommendedAugmentations, "random_crop")
			}
		}
	}
	if len(recommendedAugmentations) == 0 {
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip if class semantics allow it")
	}

	summary := fmt.Sprintf(
		"%s dataset with %d classes, %d images, imbalance ratio %.2f, and %d corrupt image(s).",
		taskType,
		classCount,
		totalImages,
		imbalanceRatio,
		corruptImageCount,
	)

	return agents.DatasetPlanningInsights{
		Summary:                  summary,
		TaskType:                 taskType,
		ClassCount:               classCount,
		TotalImages:              totalImages,
		ImageCount:               imageCount,
		ClassDistribution:        classDistribution,
		ImbalanceRatio:           imbalanceRatio,
		CorruptImageCount:        corruptImageCount,
		CorruptFileCount:         corruptFileCount,
		WidthMin:                 widthMin,
		WidthMax:                 widthMax,
		HeightMin:                heightMin,
		HeightMax:                heightMax,
		ImageDimensionStats:      imageDimensionStats,
		SplitSummary:             splitSummary,
		MetadataSummary:          metadataSummary,
		AgentSafeMetadataSummary: agentSafeMetadataSummary,
		LeakageWarnings:          leakageWarnings,
		DatasetTraits:            datasetTraits,
		Artifacts:                artifacts,
		Constraints:              uniqueStrings(constraints),
		RecommendedPreprocessing: uniqueStrings(recommendedPreprocessing),
		RecommendedAugmentations: uniqueStrings(recommendedAugmentations),
		RecommendedMetrics:       uniqueStrings(recommendedMetrics),
		LiveInferencePriorities:  liveInferencePriorities,
	}
}

func (s *Server) activeAgentSafeDatasetMetadataSummary(dataset datasets.Dataset) (map[string]any, error) {
	var metadataImport datasets.DatasetMetadataImport
	var err error
	switch getter := any(s.store).(type) {
	case activeDatasetMetadataImportGetterWithContext:
		metadataImport, err = getter.GetActiveDatasetMetadataImport(context.Background(), dataset.ID)
	case activeDatasetMetadataImportGetter:
		metadataImport, err = getter.GetActiveDatasetMetadataImport(dataset.ID)
	default:
		return nil, nil
	}
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}
	if metadataImport.DatasetID != "" && metadataImport.DatasetID != dataset.ID {
		return nil, fmt.Errorf("%w: metadata import dataset_id does not match planner dataset", store.ErrInvalidRequest)
	}
	if metadataImport.ProjectID != "" && metadataImport.ProjectID != dataset.ProjectID {
		return nil, fmt.Errorf("%w: metadata import project_id does not match planner project", store.ErrInvalidRequest)
	}
	summary := metadataImport.AgentSafeSummary
	if summary.DatasetID == "" {
		summary.DatasetID = metadataImport.DatasetID
	}
	if summary.ImportID == "" {
		summary.ImportID = metadataImport.ID
	}
	if summary.Status == "" {
		summary.Status = metadataImport.Status
	}
	if summary.ImportVersion == 0 {
		summary.ImportVersion = metadataImport.ImportVersion
	}
	if summary.SourceKind == "" {
		summary.SourceKind = metadataImport.SourceKind
	}
	return agentSafeDatasetMetadataSummaryMap(summary), nil
}

func agentSafeDatasetMetadataSummaryMap(summary datasets.AgentSafeDatasetMetadataSummary) map[string]any {
	out := map[string]any{}
	addString := func(key string, value string) {
		if strings.TrimSpace(value) != "" {
			out[key] = strings.TrimSpace(value)
		}
	}
	addInt := func(key string, value int) {
		if value > 0 {
			out[key] = value
		}
	}
	addFloat := func(key string, value float64) {
		if value > 0 && isFiniteFloat(value) {
			out[key] = roundDiagnosticFloat(value)
		}
	}
	addString("dataset_id", summary.DatasetID)
	addString("import_id", summary.ImportID)
	addString("status", summary.Status)
	addInt("import_version", summary.ImportVersion)
	addString("source_kind", summary.SourceKind)
	addInt("source_count", summary.SourceCount)
	addInt("parsed_source_count", summary.ParsedSourceCount)
	addInt("unsupported_source_count", summary.UnsupportedSourceCount)
	addInt("error_source_count", summary.ErrorSourceCount)
	if len(summary.SourceFormats) > 0 {
		out["source_formats"] = uniqueStrings(summary.SourceFormats)
	}
	addInt("class_count", summary.ClassCount)
	addInt("sample_count", summary.SampleCount)
	addInt("labeled_sample_count", summary.LabeledSampleCount)
	addInt("missing_label_count", summary.MissingLabelCount)
	if len(summary.SplitCounts) > 0 {
		out["split_counts"] = copyStringIntMap(summary.SplitCounts)
	}
	if summary.OfficialSplitAvailable {
		out["official_split_available"] = true
	}
	if len(summary.AnnotationCounts) > 0 {
		out["annotation_counts"] = copyStringIntMap(summary.AnnotationCounts)
	}
	addInt("bbox_annotation_count", summary.BBoxAnnotationCount)
	addInt("bbox_sample_count", summary.BBoxSampleCount)
	addFloat("bbox_coverage_ratio", summary.BBoxCoverageRatio)
	if len(summary.Warnings) > 0 {
		out["warnings"] = metadataIssueSummaries(summary.Warnings, 8)
	}
	if len(summary.Errors) > 0 {
		out["errors"] = metadataIssueSummaries(summary.Errors, 8)
	}
	if len(summary.Capabilities) > 0 {
		out["capabilities"] = copyStringBoolMap(summary.Capabilities)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func metadataIssueSummaries(issues []datasets.MetadataIssue, limit int) []map[string]any {
	out := []map[string]any{}
	for _, issue := range issues {
		item := map[string]any{}
		if strings.TrimSpace(issue.Severity) != "" {
			item["severity"] = strings.TrimSpace(issue.Severity)
		}
		if strings.TrimSpace(issue.Code) != "" {
			item["code"] = strings.TrimSpace(issue.Code)
		}
		if strings.TrimSpace(issue.Message) != "" && !metadataSummaryTextLooksRaw(issue.Message) {
			item["message"] = strings.TrimSpace(issue.Message)
		}
		if issue.Count > 0 {
			item["count"] = issue.Count
		}
		if len(item) == 0 {
			continue
		}
		out = append(out, item)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func safeLegacyMetadataSummary(input map[string]any) map[string]any {
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
		"bbox_per_class":           true,
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
		"yolo_available":           true,
		"yolo_label_count":         true,
		"yolo_label_file_count":    true,
		"yolo_summary":             true,
	}
	out := map[string]any{}
	for key, value := range input {
		normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(key)), "-", "_")
		if !allowed[normalized] || metadataSummaryKeyLooksRaw(normalized) {
			continue
		}
		if normalized == "yolo_summary" {
			if summary := safeYOLOSummary(mapFromAny(value)); len(summary) > 0 {
				out[normalized] = summary
			}
			continue
		}
		if text, ok := value.(string); ok && metadataSummaryTextLooksRaw(text) {
			continue
		}
		out[normalized] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func safeYOLOSummary(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	allowed := map[string]bool{
		"available":         true,
		"bbox_count":        true,
		"bbox_per_class":    true,
		"box_distribution":  true,
		"class_ids":         true,
		"class_names":       true,
		"config_count":      true,
		"empty_label_count": true,
		"format":            true,
		"image_count":       true,
		"label_file_count":  true,
		"missing_labels":    true,
		"split_counts":      true,
		"split_summary":     true,
	}
	out := map[string]any{}
	for key, value := range input {
		normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(key)), "-", "_")
		if !allowed[normalized] || metadataSummaryKeyLooksRaw(normalized) {
			continue
		}
		if text, ok := value.(string); ok && metadataSummaryTextLooksRaw(text) {
			continue
		}
		out[normalized] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func mergeMetadataEvidenceSummary(base map[string]any, overlay map[string]any) map[string]any {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}
	out := copyPayloadMap(base)
	for key, value := range overlay {
		out[key] = value
	}
	return out
}

func profileWithAgentSafeMetadataSummary(profile map[string]any, summary map[string]any) map[string]any {
	if len(summary) == 0 {
		return profile
	}
	out := copyPayloadMap(profile)
	out["agent_safe_metadata_summary"] = summary
	out["metadata_summary"] = mergeMetadataEvidenceSummary(safeLegacyMetadataSummary(profileMap(profile, "metadata_summary")), summary)
	if classCount := int(payloadNumber(summary["class_count"])); classCount > profileInt(out, "class_count") {
		out["class_count"] = classCount
	}
	if sampleCount := int(payloadNumber(summary["sample_count"])); sampleCount > 0 {
		if profileInt(out, "total_images") == 0 {
			out["total_images"] = sampleCount
		}
		if profileInt(out, "image_count") == 0 {
			out["image_count"] = sampleCount
		}
	}
	if metadataSummaryHasBBoxEvidence(summary) {
		out["bbox_available"] = true
	}
	out["metadata_available"] = true
	return out
}

func metadataSummaryKeyLooksRaw(key string) bool {
	normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(key)), "-", "_")
	if strings.HasSuffix(normalized, "_count") || strings.HasSuffix(normalized, "_counts") || strings.HasSuffix(normalized, "_ratio") {
		return false
	}
	return containsAnyText(normalized, "path", "uri", "url", "raw", "preview", "content", "sidecar", "storage", "checksum", "filename", "file_name", "manifest_record", "source_row")
}

func metadataSummaryTextLooksRaw(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, "://") || strings.Contains(lower, `:\`) || strings.HasPrefix(lower, "/") {
		return true
	}
	if strings.Contains(lower, "/") && containsAnyText(lower, ".csv", ".json", ".xml", ".txt", ".tsv", ".jpg", ".jpeg", ".png", ".parquet", ".yaml", ".yml") {
		return true
	}
	return containsAnyText(lower, "raw_preview", "raw row", "source row", "sidecar content", "storage uri", "local path")
}

func copyStringIntMap(input map[string]int) map[string]int {
	out := make(map[string]int, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func copyStringBoolMap(input map[string]bool) map[string]bool {
	out := make(map[string]bool, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func projectObjectiveContext(goal string) agents.ProjectObjectiveContext {
	goalText := strings.TrimSpace(goal)
	normalized := strings.ToLower(goalText)
	context := agents.ProjectObjectiveContext{
		GoalText:             goalText,
		PrimaryObjective:     "balanced_quality",
		MetricPreferences:    []string{"macro_f1", "accuracy"},
		DeploymentPriorities: []string{"explain quality, cost, and runtime tradeoffs"},
		Constraints:          []string{},
		RankingWeights: map[string]float64{
			"macro_f1":           0.35,
			"accuracy":           0.25,
			"per_class_behavior": 0.15,
			"latency":            0.10,
			"training_cost":      0.08,
			"runtime":            0.07,
		},
	}

	if containsAny(normalized, "live", "real-time", "realtime", "instant", "fast", "quick", "low latency") {
		context.PrimaryObjective = "low_latency_live_service"
		context.DeploymentPriorities = append(context.DeploymentPriorities, "treat inference latency under roughly 25ms as acceptable for live use", "use latency as a tiebreaker when quality is close")
		context.Constraints = append(context.Constraints, "allow stronger quality challengers when expected or observed latency remains within the live budget")
		context.RankingWeights["latency"] = 0.08
		context.RankingWeights["model_size"] = 0.04
		context.RankingWeights["macro_f1"] = 0.40
		context.RankingWeights["accuracy"] = 0.26
	}
	if containsAny(normalized, "cheap", "budget", "cost", "low cost", "inexpensive") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "budget_sensitive"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer lower training and inference cost when quality is close")
		context.RankingWeights["training_cost"] = 0.18
		context.RankingWeights["runtime"] = 0.10
	}
	if containsAny(normalized, "accurate", "accuracy", "best", "quality", "high quality") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "quality_first"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "allow stronger models when they produce meaningful quality gains")
		context.RankingWeights["macro_f1"] = 0.45
		context.RankingWeights["accuracy"] = 0.30
		context.RankingWeights["per_class_behavior"] = 0.18
	}
	if containsAny(normalized, "imbalanced", "minority", "rare class", "rare classes", "fair", "per-class", "per class") {
		context.MetricPreferences = append(context.MetricPreferences, "per_class_f1", "recall_by_class")
		context.DeploymentPriorities = append(context.DeploymentPriorities, "avoid selecting a model that hides weak minority-class behavior behind average accuracy")
		context.RankingWeights["per_class_behavior"] = 0.24
		context.RankingWeights["macro_f1"] = 0.40
	}
	if containsAny(normalized, "mobile", "edge", "browser", "desktop", "cpu") {
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer compact CPU-friendly models only when quality is close or latency exceeds the live budget")
		context.Constraints = append(context.Constraints, "do not reject quality challengers solely for being larger when latency remains acceptable")
		context.RankingWeights["model_size"] = maxFloat(context.RankingWeights["model_size"], 0.08)
		context.RankingWeights["latency"] = maxFloat(context.RankingWeights["latency"], 0.10)
	}

	context.MetricPreferences = uniqueStrings(context.MetricPreferences)
	context.DeploymentPriorities = uniqueStrings(context.DeploymentPriorities)
	context.Constraints = uniqueStrings(context.Constraints)
	return context
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}
