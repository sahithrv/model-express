package api

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

func supportedModelCatalog() []agents.SupportedModelSpec {
	return []agents.SupportedModelSpec{
		classificationModelSpec("mobilenet_v3_small", "mobilenet", "fast_live", 224, 50, "very_fast", "fast live baseline and compact champion refinement"),
		classificationModelSpec("mobilenet_v3_large", "mobilenet", "fast_live", 224, 80, "fast", "higher-capacity MobileNet challenger for live use"),
		classificationModelSpec("efficientnet_b0", "efficientnet", "fast_live", 224, 80, "fast", "strong quality/latency baseline"),
		classificationModelSpec("regnet_y_400mf", "regnet", "fast_live", 224, 100, "fast", "compact architecture challenger"),
		classificationModelSpec("efficientnet_b1", "efficientnet", "balanced", 240, 150, "medium", "balanced quality challenger"),
		classificationModelSpec("efficientnet_b2", "efficientnet", "balanced", 260, 250, "medium", "stronger quality challenger when budget allows"),
		classificationModelSpec("resnet18", "resnet", "balanced", 224, 100, "medium", "stable control architecture"),
		classificationModelSpec("resnet34", "resnet", "balanced", 224, 150, "medium_slow", "larger ResNet comparison"),
		classificationModelSpec("convnext_tiny", "convnext", "quality_challenger", 224, 300, "slow", "quality-first challenger"),
		classificationModelSpec("swin_t", "swin", "quality_challenger", 224, 500, "slow", "transformer challenger for larger datasets"),
		classificationModelSpec("vit_b_16", "vit", "quality_challenger", 224, 800, "slowest", "explicit quality-first experiments on larger datasets"),
	}
}

func classificationModelSpec(name, family, deploymentTier string, imageSize, minImages int, latencyClass, recommendedUse string) agents.SupportedModelSpec {
	return agents.SupportedModelSpec{
		Name:                  name,
		Family:                family,
		TaskType:              "image_classification",
		ModelKind:             "torchvision_classifier",
		DeploymentTier:        deploymentTier,
		DefaultImageSize:      imageSize,
		MinRecommendedImages:  minImages,
		SupportsTransfer:      true,
		TrainingEnabled:       true,
		ExpectedLatencyClass:  latencyClass,
		RecommendedUse:        recommendedUse,
		SupportsFineTuneModes: []string{"head_only", "last_block", "full"},
	}
}

func supportedModelCatalogForDataset(dataset datasets.Dataset, agentSafeMetadataSummary map[string]any) []agents.SupportedModelSpec {
	catalog := append([]agents.SupportedModelSpec{}, supportedModelCatalog()...)
	if datasetHasYOLODetectionEvidence(dataset, agentSafeMetadataSummary) {
		catalog = append(catalog, supportedYOLODetectorModelCatalog()...)
	}
	return catalog
}

func supportedYOLODetectorModelCatalog() []agents.SupportedModelSpec {
	return []agents.SupportedModelSpec{
		yoloDetectorModelSpec("yolo11n.pt", "realtime_detector", "very_fast", "nano COCO-pretrained detector for the first live-stream baseline"),
		yoloDetectorModelSpec("yolo11s.pt", "realtime_detector", "fast", "small COCO-pretrained detector with stronger accuracy while staying live-friendly"),
		yoloDetectorModelSpec("yolo11m.pt", "quality_detector", "medium", "medium COCO-pretrained detector for quality challengers when latency budget allows"),
		yoloDetectorModelSpec("yolo11l.pt", "quality_detector", "slow", "large COCO-pretrained detector for high-accuracy offline comparison"),
		yoloDetectorModelSpec("yolo11x.pt", "quality_detector", "slowest", "extra-large COCO-pretrained detector for upper-bound detection quality studies"),
	}
}

func yoloDetectorModelSpec(name, deploymentTier, latencyClass, recommendedUse string) agents.SupportedModelSpec {
	return agents.SupportedModelSpec{
		Name:                 name,
		Family:               "yolo11",
		TaskType:             "object_detection",
		ModelKind:            "ultralytics_yolo_detector",
		PretrainedWeights:    name,
		DeploymentTier:       deploymentTier,
		DefaultImageSize:     640,
		MinRecommendedImages: 100,
		SupportsTransfer:     true,
		TrainingEnabled:      true,
		ExpectedLatencyClass: latencyClass,
		RecommendedUse:       recommendedUse + "; schedule only for YOLO object-detection datasets.",
	}
}

func supportedModelNames() map[string]bool {
	out := map[string]bool{}
	for _, spec := range supportedModelCatalog() {
		out[strings.ToLower(spec.Name)] = true
	}
	for _, spec := range supportedYOLODetectorModelCatalog() {
		out[strings.ToLower(spec.Name)] = true
	}
	return out
}

func supportedModelSpecByName(model string) (agents.SupportedModelSpec, bool) {
	normalized := strings.ToLower(strings.TrimSpace(model))
	for _, spec := range append(supportedModelCatalog(), supportedYOLODetectorModelCatalog()...) {
		if strings.ToLower(spec.Name) == normalized {
			return spec, true
		}
	}
	return agents.SupportedModelSpec{}, false
}

func datasetHasYOLODetectionEvidence(dataset datasets.Dataset, agentSafeMetadataSummary map[string]any) bool {
	return profileHasYOLODetectionEvidence(dataset.Profile) || metadataSummaryHasYOLOEvidence(agentSafeMetadataSummary)
}

func profileHasYOLODetectionEvidence(profile map[string]any) bool {
	if profileBool(profile, "yolo_available") || profileBool(profile, "yolo_format") {
		return true
	}
	if containsString(profileStringSlice(profile, "dataset_traits"), "yolo_format") {
		return true
	}
	return metadataSummaryHasYOLOEvidence(profileMap(profile, "metadata_summary")) ||
		metadataSummaryHasYOLOEvidence(profileMap(profile, "agent_safe_metadata_summary")) ||
		metadataSummaryHasYOLOEvidence(profileMap(profile, "normalized_metadata_summary"))
}

func metadataSummaryHasYOLOEvidence(summary map[string]any) bool {
	if len(summary) == 0 {
		return false
	}
	format := strings.ToLower(strings.TrimSpace(profileString(summary, "format")))
	formatIsYOLO := format == "yolo" || metadataBool(summary, "yolo_format")
	if metadataBool(summary, "yolo_available") || formatIsYOLO {
		return true
	}
	if payloadNumber(summary["yolo_config_count"]) > 0 || payloadNumber(summary["yolo_dataset_config_count"]) > 0 ||
		payloadNumber(summary["yolo_label_file_count"]) > 0 || payloadNumber(summary["yolo_label_count"]) > 0 {
		return true
	}
	if formatIsYOLO && (payloadNumber(summary["config_count"]) > 0 || payloadNumber(summary["label_file_count"]) > 0 || payloadNumber(summary["label_count"]) > 0) {
		return true
	}
	if nested := profileMap(summary, "yolo_summary"); len(nested) > 0 && metadataSummaryHasYOLOEvidence(nested) {
		return true
	}
	counts := profileMap(summary, "artifact_counts")
	if payloadNumber(counts["yolo_dataset_config"]) > 0 || payloadNumber(counts["yolo_label_file"]) > 0 {
		return true
	}
	capabilities := profileMap(summary, "capabilities")
	return metadataBool(capabilities, "yolo") || metadataBool(capabilities, "yolo_format")
}

func experimentSignaturesForPlans(projectPlans []plans.ExperimentPlan) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			signature := experimentSignature(experiment)
			if seen[signature] {
				continue
			}
			seen[signature] = true
			out = append(out, signature)
		}
	}
	sort.Strings(out)
	return out
}

func validateNovelProposedExperiments(experiments []plans.PlannedExperiment, projectPlans []plans.ExperimentPlan) error {
	existing := map[string]bool{}
	existingMechanisms := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			existing[experimentSignature(experiment)] = true
			existingMechanisms[experimentMechanismSignature(experiment)] = true
		}
	}

	proposed := map[string]bool{}
	proposedMechanisms := map[string]bool{}
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return err
		}
		signature := experimentSignature(experiment)
		if existing[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates an existing experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		if proposed[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates another proposed experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		mechanismSignature := experimentMechanismSignature(experiment)
		if existingMechanisms[mechanismSignature] {
			return fmt.Errorf("%w: proposed experiment %d only changes minor tuning knobs for an already tested experiment mechanism", store.ErrInvalidRequest, index)
		}
		if proposedMechanisms[mechanismSignature] {
			return fmt.Errorf("%w: proposed experiment %d only changes minor tuning knobs relative to another proposed experiment", store.ErrInvalidRequest, index)
		}
		proposed[signature] = true
		proposedMechanisms[mechanismSignature] = true
	}
	return nil
}

func filterNovelPlannedExperiments(experiments []plans.PlannedExperiment, projectPlans []plans.ExperimentPlan) ([]plans.PlannedExperiment, []string) {
	existing := map[string]bool{}
	existingMechanisms := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			existing[experimentSignature(experiment)] = true
			existingMechanisms[experimentMechanismSignature(experiment)] = true
		}
	}

	out := []plans.PlannedExperiment{}
	warnings := []string{}
	proposed := map[string]bool{}
	proposedMechanisms := map[string]bool{}
	for index, experiment := range experiments {
		signature := experimentSignature(experiment)
		mechanismSignature := experimentMechanismSignature(experiment)
		switch {
		case existing[signature] || proposed[signature]:
			warnings = append(warnings, fmt.Sprintf("Skipped follow-up experiment %d because it duplicated an existing experiment signature.", index))
		case existingMechanisms[mechanismSignature] || proposedMechanisms[mechanismSignature]:
			warnings = append(warnings, fmt.Sprintf("Skipped follow-up experiment %d because it only changed minor tuning knobs for an already tested mechanism.", index))
		default:
			out = append(out, experiment)
			proposed[signature] = true
			proposedMechanisms[mechanismSignature] = true
		}
	}
	return out, warnings
}

func experimentSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	augmentationPolicyConfigBlob, _ := json.Marshal(experiment.AugmentationPolicyConfig)
	classBalancingConfigBlob, _ := json.Marshal(experiment.ClassBalancingConfig)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	automlBlob, _ := json.Marshal(experiment.AutoML)
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.Epochs),
		strconv.Itoa(experiment.BatchSize),
		strconv.FormatFloat(experiment.LearningRate, 'g', -1, 64),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		strconv.FormatFloat(experiment.Dropout, 'g', -1, 64),
		strconv.FormatFloat(experiment.OptimizerMomentum, 'g', -1, 64),
		strconv.Itoa(experiment.SchedulerStepSize),
		strconv.FormatFloat(experiment.SchedulerGamma, 'g', -1, 64),
		strconv.FormatFloat(experiment.LabelSmoothing, 'g', -1, 64),
		strconv.FormatFloat(experiment.GradientClipNorm, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		string(augmentationPolicyConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		string(classBalancingConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.Itoa(experiment.EarlyStoppingPatience),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
		string(automlBlob),
	}, ":")
}

func experimentMechanismSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	augmentationPolicyConfigBlob, _ := json.Marshal(experiment.AugmentationPolicyConfig)
	classBalancingConfigBlob, _ := json.Marshal(experiment.ClassBalancingConfig)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	automlSearchBlob, _ := json.Marshal(nil)
	if experiment.AutoML != nil {
		automlSearchBlob, _ = json.Marshal(experiment.AutoML.SearchSpace)
	}
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		strconv.FormatFloat(experiment.Dropout, 'g', -1, 64),
		strconv.FormatFloat(experiment.OptimizerMomentum, 'g', -1, 64),
		strconv.Itoa(experiment.SchedulerStepSize),
		strconv.FormatFloat(experiment.SchedulerGamma, 'g', -1, 64),
		strconv.FormatFloat(experiment.LabelSmoothing, 'g', -1, 64),
		strconv.FormatFloat(experiment.GradientClipNorm, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		string(augmentationPolicyConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		string(classBalancingConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
		string(automlSearchBlob),
	}, ":")
}

func plannedExperimentsFromPayload(payload map[string]any) ([]plans.PlannedExperiment, error) {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return nil, fmt.Errorf("%w: reviewer decision does not include proposed_experiments", store.ErrInvalidRequest)
	}

	blob, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments could not be encoded", store.ErrInvalidRequest)
	}

	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments has an invalid shape", store.ErrInvalidRequest)
	}

	if len(experiments) == 0 {
		return nil, fmt.Errorf("%w: reviewer proposed no follow-up experiments", store.ErrInvalidRequest)
	}
	if len(experiments) > maxLLMPlannerExperiments {
		return nil, fmt.Errorf("%w: proposed_experiments has %d experiments, max is %d", store.ErrInvalidRequest, len(experiments), maxLLMPlannerExperiments)
	}

	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return nil, err
		}
	}

	return experiments, nil
}

func plannedExperimentsFromPayloadLenient(payload map[string]any) ([]plans.PlannedExperiment, error) {
	value, ok := payload["proposed_experiments"]
	if !ok || value == nil {
		return nil, fmt.Errorf("%w: reviewer decision does not include proposed_experiments", store.ErrInvalidRequest)
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return nil, err
	}
	return experiments, nil
}

func validateLLMPlannerStoredMechanismContract(decision decisions.AgentDecision, experiments []plans.PlannedExperiment) error {
	if decision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
		return nil
	}
	return validatePlannedExperimentMechanismContract(experiments, payloadStringSlice(decision.Payload, "evidence_used"))
}

func plannerExperimentsWithProposalMechanisms(recommendation agents.ExperimentPlanningRecommendation) ([]plans.PlannedExperiment, error) {
	experiments := append([]plans.PlannedExperiment(nil), recommendation.ProposedExperiments...)
	if len(experiments) == 0 {
		return experiments, nil
	}
	return attachProposalMechanismsToExperiments(experiments, recommendation.ProposalMechanisms)
}

func plannerExperimentsWithProposalMechanismsRelaxed(recommendation agents.ExperimentPlanningRecommendation) ([]plans.PlannedExperiment, []string) {
	experiments := append([]plans.PlannedExperiment(nil), recommendation.ProposedExperiments...)
	if len(experiments) == 0 {
		return experiments, nil
	}
	return attachProposalMechanismsToExperimentsRelaxed(experiments, recommendation.ProposalMechanisms)
}

func plannedExperimentsWithStoredProposalMechanisms(payload map[string]any, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, error) {
	mechanisms, ok, err := plannerProposalMechanismsFromPayload(payload)
	if err != nil {
		return nil, err
	}
	if !ok {
		return experiments, nil
	}
	return attachProposalMechanismsToExperiments(experiments, mechanisms)
}

func plannerProposalMechanismsFromPayload(payload map[string]any) ([]agents.PlannerProposalMechanism, bool, error) {
	value, ok := payload["proposal_mechanisms"]
	if !ok || value == nil {
		return nil, false, nil
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, false, fmt.Errorf("%w: proposal_mechanisms could not be encoded", store.ErrInvalidRequest)
	}
	var mechanisms []agents.PlannerProposalMechanism
	if err := json.Unmarshal(blob, &mechanisms); err != nil {
		return nil, false, fmt.Errorf("%w: proposal_mechanisms has an invalid shape", store.ErrInvalidRequest)
	}
	return mechanisms, true, nil
}

func attachProposalMechanismsToExperiments(experiments []plans.PlannedExperiment, mechanisms []agents.PlannerProposalMechanism) ([]plans.PlannedExperiment, error) {
	out := append([]plans.PlannedExperiment(nil), experiments...)
	mechanismsByIndex := map[int]agents.PlannerProposalMechanism{}
	for index, mechanism := range mechanisms {
		if mechanism.ExperimentIndex < 0 || mechanism.ExperimentIndex >= len(out) {
			return nil, fmt.Errorf("%w: proposal_mechanisms[%d] has invalid experiment_index %d", store.ErrInvalidRequest, index, mechanism.ExperimentIndex)
		}
		if _, exists := mechanismsByIndex[mechanism.ExperimentIndex]; exists {
			return nil, fmt.Errorf("%w: proposal_mechanisms duplicate experiment_index %d", store.ErrInvalidRequest, mechanism.ExperimentIndex)
		}
		mechanismsByIndex[mechanism.ExperimentIndex] = mechanism
	}
	for index := range out {
		mechanism, ok := mechanismsByIndex[index]
		if !ok {
			continue
		}
		out[index].Mechanism = mechanism.Mechanism
		out[index].Intervention = mechanism.Intervention
		out[index].EvidenceUsed = append([]string(nil), mechanism.EvidenceUsed...)
		out[index].ExpectedEffect = mechanism.ExpectedEffect
	}
	return out, nil
}

func attachProposalMechanismsToExperimentsRelaxed(experiments []plans.PlannedExperiment, mechanisms []agents.PlannerProposalMechanism) ([]plans.PlannedExperiment, []string) {
	out := append([]plans.PlannedExperiment(nil), experiments...)
	if len(out) == 0 {
		return out, nil
	}
	if len(mechanisms) == 0 {
		return out, []string{plannerRelaxedValidationWarningText("proposal_mechanisms missing; using proposed experiment fields as-is")}
	}

	warnings := []string{}
	seen := map[int]bool{}
	for index, mechanism := range mechanisms {
		if mechanism.ExperimentIndex < 0 || mechanism.ExperimentIndex >= len(out) {
			warnings = append(warnings, plannerRelaxedValidationWarningText(fmt.Sprintf("proposal_mechanisms[%d] has invalid experiment_index %d", index, mechanism.ExperimentIndex)))
			continue
		}
		if seen[mechanism.ExperimentIndex] {
			warnings = append(warnings, plannerRelaxedValidationWarningText(fmt.Sprintf("proposal_mechanisms duplicate experiment_index %d", mechanism.ExperimentIndex)))
			continue
		}
		seen[mechanism.ExperimentIndex] = true
		if strings.TrimSpace(mechanism.Mechanism) != "" {
			out[mechanism.ExperimentIndex].Mechanism = mechanism.Mechanism
		}
		if strings.TrimSpace(mechanism.Intervention) != "" {
			out[mechanism.ExperimentIndex].Intervention = mechanism.Intervention
		}
		if len(nonEmptyStringValues(mechanism.EvidenceUsed)) > 0 {
			out[mechanism.ExperimentIndex].EvidenceUsed = append([]string(nil), mechanism.EvidenceUsed...)
		}
		if strings.TrimSpace(mechanism.ExpectedEffect) != "" {
			out[mechanism.ExperimentIndex].ExpectedEffect = mechanism.ExpectedEffect
		}
	}
	for index := range out {
		if !seen[index] {
			warnings = append(warnings, plannerRelaxedValidationWarningText(fmt.Sprintf("proposal_mechanisms missing experiment_index %d; using proposed experiment fields as-is", index)))
		}
	}
	return out, uniqueStrings(warnings)
}

func validateLLMPlannerMechanismContract(experiments []plans.PlannedExperiment, evidenceUsed []string) error {
	return validatePlannedExperimentMechanismContract(experiments, evidenceUsed)
}

func validatePlannedExperimentMechanismContract(experiments []plans.PlannedExperiment, planEvidence []string) error {
	if len(experiments) == 0 {
		return nil
	}
	for index, experiment := range experiments {
		mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
		if mechanism == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing mechanism", store.ErrInvalidRequest, index)
		}
		if !allowedExperimentValue(mechanism, allowedPlannerMechanisms()) {
			return fmt.Errorf("%w: proposed experiment %d has unsupported mechanism %q", store.ErrInvalidRequest, index, experiment.Mechanism)
		}
		if strings.TrimSpace(experiment.Intervention) == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing intervention", store.ErrInvalidRequest, index)
		}
		if strings.TrimSpace(experiment.ExpectedEffect) == "" {
			return fmt.Errorf("%w: proposed experiment %d is missing expected_effect", store.ErrInvalidRequest, index)
		}
		if len(nonEmptyStringValues(experiment.EvidenceUsed)) == 0 && len(nonEmptyStringValues(planEvidence)) == 0 {
			return fmt.Errorf("%w: proposed experiment %d is missing evidence_used", store.ErrInvalidRequest, index)
		}
	}
	return nil
}

func validateMechanismDatasetEvidence(profile map[string]any, experiments []plans.PlannedExperiment, planEvidence []string) error {
	violations := []string{}
	for index, experiment := range experiments {
		mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
		if mechanism == "" {
			continue
		}
		evidenceText := experimentMechanismEvidenceText(experiment, planEvidence)
		switch mechanism {
		case "class_imbalance", "minority_targeting":
			if !classBalancingConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s requires class_balancing or sampling_strategy", index, mechanism))
			}
			if !profileOrDiagnosisHasClassImbalanceEvidence(profile, evidenceText) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s requires dataset imbalance, per-class error, or minority-failure evidence", index, mechanism))
			}
		case "bbox_crop_ablation":
			if !bboxCropConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism bbox_crop_ablation requires bbox crop preprocessing", index))
			}
			if !profileHasBBoxEvidence(profile) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism bbox_crop_ablation requires backend-profiled bbox/annotation evidence", index))
			}
		case "resolution_crop":
			if !resolutionCropConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism resolution_crop requires image size, resolution strategy, or crop preprocessing changes", index))
			}
			if resolutionCropNeedsEvidence(experiment) && !profileOrDiagnosisHasResolutionCropEvidence(profile, evidenceText) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism resolution_crop requires object-scale, fine-grained, dimension, crop, or visual-trait evidence", index))
			}
		case "augmentation_auto":
			if !autoAugmentationConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism augmentation_auto requires structured randaugment, trivialaugment, or autoaugment policy config", index))
			}
		case "augmentation_mixed_sample":
			if !mixedSampleAugmentationConfigured(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism augmentation_mixed_sample requires structured MixUp or CutMix augmentation policy config", index))
			}
		case "label_noise_audit", "hard_example_audit":
			if !labelQualityAuditExperiment(experiment) {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism %s is report-only and must use template %s instead of creating a training job", index, mechanism, jobs.TemplateLabelQualityAudit))
			}
		case "distillation":
			violations = append(violations, fmt.Sprintf("experiment %d mechanism distillation is not schedulable until teacher-artifact validation and worker support are enabled", index))
		case "deployment_latency":
			if !containsAnyText(evidenceText, "latency", "runtime", "cost", "edge", "live", "small", "compact", "mobile") {
				violations = append(violations, fmt.Sprintf("experiment %d mechanism deployment_latency requires deployment, latency, runtime, cost, or compact-model evidence", index))
			}
		}
	}
	if len(violations) > 0 {
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, strings.Join(violations, "; "))
	}
	return nil
}

func experimentMechanismEvidenceText(experiment plans.PlannedExperiment, planEvidence []string) string {
	parts := append([]string{}, planEvidence...)
	parts = append(parts, experiment.EvidenceUsed...)
	parts = append(parts,
		experiment.Intervention,
		experiment.ExpectedEffect,
		experiment.Reason,
		experiment.Strategy,
	)
	return strings.ToLower(strings.Join(parts, " "))
}

func classBalancingConfigured(experiment plans.PlannedExperiment) bool {
	return nonDefaultText(experiment.ClassBalancing, "none") || nonDefaultText(experiment.SamplingStrategy, "none")
}

func bboxCropConfigured(experiment plans.PlannedExperiment) bool {
	if experiment.Preprocessing == nil {
		return false
	}
	return containsAnyText(strings.ToLower(experiment.Preprocessing.CropStrategy+" "+experiment.Preprocessing.BBoxMode+" "+experiment.Preprocessing.ResizeStrategy), "bbox", "box")
}

func resolutionCropConfigured(experiment plans.PlannedExperiment) bool {
	if experiment.ImageSize > 0 || nonDefaultText(experiment.ResolutionStrategy, "fixed") {
		return true
	}
	if experiment.Preprocessing == nil {
		return false
	}
	return nonDefaultText(experiment.Preprocessing.ResizeStrategy, "squash") ||
		nonDefaultText(experiment.Preprocessing.CropStrategy, "none") ||
		nonDefaultText(experiment.Preprocessing.Normalization, "imagenet")
}

func resolutionCropNeedsEvidence(experiment plans.PlannedExperiment) bool {
	if experiment.ImageSize > 256 {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(experiment.ResolutionStrategy), "high_resolution_ablation") {
		return true
	}
	if experiment.Preprocessing == nil {
		return false
	}
	return containsAnyText(strings.ToLower(experiment.Preprocessing.CropStrategy+" "+experiment.Preprocessing.ResizeStrategy), "crop", "bbox", "aspect")
}

func autoAugmentationConfigured(experiment plans.PlannedExperiment) bool {
	policy := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy))
	if containsAnyText(policy, "randaugment", "trivialaugment", "autoaugment") {
		return true
	}
	if experiment.AugmentationPolicyConfig == nil {
		return false
	}
	policyType := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType))
	return policyType == "randaugment" || policyType == "trivialaugment" || policyType == "trivialaugmentwide" || policyType == "autoaugment"
}

func mixedSampleAugmentationConfigured(experiment plans.PlannedExperiment) bool {
	policy := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy))
	if policy == "mixup" || policy == "cutmix" {
		return true
	}
	if experiment.AugmentationPolicyConfig == nil {
		return false
	}
	policyType := strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType))
	return policyType == "mixup" || policyType == "cutmix"
}

func labelQualityAuditExperiment(experiment plans.PlannedExperiment) bool {
	if !strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return false
	}
	mechanism := strings.ToLower(strings.TrimSpace(experiment.Mechanism))
	return mechanism == "label_noise_audit" || mechanism == "hard_example_audit"
}

func profileOrDiagnosisHasClassImbalanceEvidence(profile map[string]any, evidenceText string) bool {
	if profileFloat(profile, "imbalance_ratio") >= 1.5 {
		return true
	}
	distribution := profileMap(profile, "class_distribution")
	if len(distribution) == 0 {
		distribution = profileMap(profile, "images_per_class")
	}
	if classDistributionImbalanceRatio(distribution) >= 1.5 {
		return true
	}
	return containsAnyText(evidenceText, "class_imbalance", "class imbalance", "minority", "rare class", "per-class", "per class", "macro-f1 trails accuracy", "macro f1 trails accuracy")
}

func classDistributionImbalanceRatio(distribution map[string]any) float64 {
	if len(distribution) == 0 {
		return 0
	}
	minCount := math.MaxFloat64
	maxCount := 0.0
	for _, value := range distribution {
		count := payloadNumber(value)
		if count <= 0 {
			continue
		}
		if count < minCount {
			minCount = count
		}
		if count > maxCount {
			maxCount = count
		}
	}
	if minCount == math.MaxFloat64 || minCount <= 0 {
		return 0
	}
	return maxCount / minCount
}

func profileHasBBoxEvidence(profile map[string]any) bool {
	if profileBool(profile, "bbox_available") || profileBool(profile, "annotations_available") ||
		profileInt(profile, "bbox_annotations_count") > 0 || payloadNumber(profile["bbox_count"]) > 0 ||
		artifactCountsHaveBBoxEvidence(profileMap(profile, "artifact_counts")) {
		return true
	}
	metadata := profileMap(profile, "metadata_summary")
	if metadataSummaryHasBBoxEvidence(metadata) {
		return true
	}
	if metadataSummaryHasBBoxEvidence(profileMap(profile, "agent_safe_metadata_summary")) ||
		metadataSummaryHasBBoxEvidence(profileMap(profile, "normalized_metadata_summary")) {
		return true
	}
	visualTraits := profileMap(profile, "visual_trait_summary")
	if payloadNumber(visualTraits["bbox_count"]) > 0 {
		return true
	}
	traits := profileStringSlice(profile, "dataset_traits")
	for _, trait := range traits {
		if containsAnyText(strings.ToLower(trait), "bbox", "bounding box", "annotation") {
			return true
		}
	}
	for _, artifact := range profileMapSlice(profile, "artifacts") {
		blob, _ := json.Marshal(artifact)
		if containsAnyText(strings.ToLower(string(blob)), "bbox", "bounding_box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func metadataSummaryHasBBoxEvidence(summary map[string]any) bool {
	if len(summary) == 0 {
		return false
	}
	if metadataBool(summary, "bbox_available") || metadataBool(summary, "annotations_available") ||
		payloadNumber(summary["bbox_annotations_count"]) > 0 ||
		payloadNumber(summary["bbox_annotation_count"]) > 0 ||
		payloadNumber(summary["bbox_sample_count"]) > 0 ||
		payloadNumber(summary["bbox_count"]) > 0 ||
		payloadNumber(summary["bbox_coverage_ratio"]) > 0 ||
		artifactCountsHaveBBoxEvidence(profileMap(summary, "artifact_counts")) {
		return true
	}
	annotationCounts := profileMap(summary, "annotation_counts")
	if payloadNumber(annotationCounts["bbox"]) > 0 || payloadNumber(annotationCounts["bounding_box"]) > 0 {
		return true
	}
	capabilities := profileMap(summary, "capabilities")
	if metadataBool(capabilities, "bbox") || metadataBool(capabilities, "bbox_annotations") ||
		metadataBool(capabilities, "bbox_crop") || metadataBool(capabilities, "object_detection") {
		return true
	}
	return false
}

func artifactCountsHaveBBoxEvidence(counts map[string]any) bool {
	for key, value := range counts {
		if payloadNumber(value) <= 0 {
			continue
		}
		if containsAnyText(strings.ToLower(strings.TrimSpace(key)), "bbox", "bounding_box", "bounding box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func profileOrDiagnosisHasResolutionCropEvidence(profile map[string]any, evidenceText string) bool {
	if containsAnyText(
		evidenceText,
		"small object",
		"small_objects",
		"large object",
		"large_objects",
		"object scale",
		"fine-grained",
		"fine grained",
		"fine_grained_similarity",
		"crop mismatch",
		"crop_bbox_useful",
		"background dominance",
		"background_dominance",
		"aspect ratio",
		"variable dimensions",
		"image dimension",
		"orientation_sensitive",
		"domain_shift_possible",
		"color_texture_signal",
	) {
		return true
	}
	traits := profileStringSlice(profile, "dataset_traits")
	for _, trait := range traits {
		if containsAnyText(strings.ToLower(trait), "small object", "object scale", "fine-grained", "fine grained", "background", "crop", "aspect", "variable dimension", "high resolution") {
			return true
		}
	}
	widthMin := profileInt(profile, "width_min")
	widthMax := profileInt(profile, "width_max")
	heightMin := profileInt(profile, "height_min")
	heightMax := profileInt(profile, "height_max")
	if widthMax >= 512 || heightMax >= 512 {
		return true
	}
	if widthMin > 0 && widthMax > widthMin*2 {
		return true
	}
	if heightMin > 0 && heightMax > heightMin*2 {
		return true
	}
	return false
}

func payloadNumber(value any) float64 {
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case float64:
		return typed
	case float32:
		return float64(typed)
	case json.Number:
		out, _ := typed.Float64()
		return out
	default:
		return 0
	}
}

func nonDefaultText(value string, defaults ...string) bool {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return false
	}
	for _, fallback := range defaults {
		if normalized == strings.ToLower(strings.TrimSpace(fallback)) {
			return false
		}
	}
	return true
}

func containsAnyText(value string, needles ...string) bool {
	value = strings.ToLower(value)
	for _, needle := range needles {
		if strings.Contains(value, strings.ToLower(strings.TrimSpace(needle))) {
			return true
		}
	}
	return false
}

func validatePlannedExperiment(experiment plans.PlannedExperiment, index int) error {
	if strings.TrimSpace(experiment.Template) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing template", store.ErrInvalidRequest, index)
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		if !labelQualityAuditExperiment(experiment) {
			return fmt.Errorf("%w: proposed experiment %d template %s requires mechanism label_noise_audit or hard_example_audit", store.ErrInvalidRequest, index, jobs.TemplateLabelQualityAudit)
		}
		if strings.TrimSpace(experiment.Intervention) == "" {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing intervention", store.ErrInvalidRequest, index)
		}
		if strings.TrimSpace(experiment.ExpectedEffect) == "" {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing expected_effect", store.ErrInvalidRequest, index)
		}
		if len(nonEmptyStringValues(experiment.EvidenceUsed)) == 0 {
			return fmt.Errorf("%w: proposed experiment %d audit job is missing evidence_used", store.ErrInvalidRequest, index)
		}
		return nil
	}
	if disallowedDirectExperimentTemplate(experiment.Template) {
		return fmt.Errorf("%w: proposed experiment %d uses control-plane job template %q; experiments cannot directly schedule backend-owned worker jobs", store.ErrInvalidRequest, index, experiment.Template)
	}
	if strings.TrimSpace(experiment.Model) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing model", store.ErrInvalidRequest, index)
	}
	modelSpec, ok := supportedModelSpecByName(experiment.Model)
	if !ok {
		return fmt.Errorf("%w: proposed experiment %d uses unsupported model %q", store.ErrInvalidRequest, index, experiment.Model)
	}
	if !modelSpec.TrainingEnabled {
		return fmt.Errorf("%w: proposed experiment %d uses detector model %q before the %s training worker is enabled", store.ErrInvalidRequest, index, experiment.Model, modelSpec.ModelKind)
	}
	if experiment.Epochs < 1 || experiment.Epochs > 100 {
		return fmt.Errorf("%w: proposed experiment %d must have 1-100 epochs", store.ErrInvalidRequest, index)
	}
	if experiment.BatchSize < 1 || experiment.BatchSize > 512 {
		return fmt.Errorf("%w: proposed experiment %d must have batch_size 1-512", store.ErrInvalidRequest, index)
	}
	if experiment.LearningRate <= 0 || experiment.LearningRate > 1 {
		return fmt.Errorf("%w: proposed experiment %d must have learning_rate in (0, 1]", store.ErrInvalidRequest, index)
	}
	maxImageSize := 1024
	if modelSpec.TaskType == "object_detection" {
		maxImageSize = 1280
	}
	if experiment.ImageSize < 0 || experiment.ImageSize > maxImageSize {
		return fmt.Errorf("%w: proposed experiment %d image_size must be at most %d", store.ErrInvalidRequest, index, maxImageSize)
	}
	if experiment.ResolutionStrategy != "" && !allowedExperimentValue(experiment.ResolutionStrategy, allowedResolutionStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported resolution_strategy %q", store.ErrInvalidRequest, index, experiment.ResolutionStrategy)
	}
	if experiment.Preprocessing != nil {
		if err := validatePreprocessingConfig(*experiment.Preprocessing, index); err != nil {
			return err
		}
	}
	if experiment.Optimizer != "" && !allowedExperimentValue(experiment.Optimizer, allowedOptimizers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported optimizer %q", store.ErrInvalidRequest, index, experiment.Optimizer)
	}
	if experiment.Scheduler != "" && !allowedExperimentValue(experiment.Scheduler, allowedSchedulers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported scheduler %q", store.ErrInvalidRequest, index, experiment.Scheduler)
	}
	if experiment.WeightDecay < 0 || experiment.WeightDecay > 1 {
		return fmt.Errorf("%w: proposed experiment %d weight_decay must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	if experiment.Dropout < 0 || experiment.Dropout > 0.7 {
		return fmt.Errorf("%w: proposed experiment %d dropout must be between 0 and 0.7", store.ErrInvalidRequest, index)
	}
	if experiment.OptimizerMomentum < 0 || experiment.OptimizerMomentum > 0.99 {
		return fmt.Errorf("%w: proposed experiment %d optimizer_momentum must be between 0 and 0.99", store.ErrInvalidRequest, index)
	}
	if experiment.OptimizerMomentum > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		return fmt.Errorf("%w: proposed experiment %d optimizer_momentum is only supported with optimizer sgd", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerStepSize < 0 || experiment.SchedulerStepSize > 100 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_step_size must be between 1 and 100 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerStepSize > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		return fmt.Errorf("%w: proposed experiment %d scheduler_step_size is only supported with scheduler step", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma < 0 || experiment.SchedulerGamma > 0.95 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma must be between 0.05 and 0.95 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma > 0 && experiment.SchedulerGamma < 0.05 {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma must be between 0.05 and 0.95 when set", store.ErrInvalidRequest, index)
	}
	if experiment.SchedulerGamma > 0 && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		return fmt.Errorf("%w: proposed experiment %d scheduler_gamma is only supported with scheduler step", store.ErrInvalidRequest, index)
	}
	if experiment.LabelSmoothing < 0 || experiment.LabelSmoothing > 0.3 {
		return fmt.Errorf("%w: proposed experiment %d label_smoothing must be between 0 and 0.3", store.ErrInvalidRequest, index)
	}
	if experiment.GradientClipNorm < 0 || experiment.GradientClipNorm > 10 {
		return fmt.Errorf("%w: proposed experiment %d gradient_clip_norm must be between 0 and 10", store.ErrInvalidRequest, index)
	}
	if err := validateAugmentationConfig(experiment.Augmentation, index); err != nil {
		return err
	}
	if experiment.AugmentationPolicy != "" && !allowedExperimentValue(experiment.AugmentationPolicy, allowedAugmentationPolicies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation_policy %q", store.ErrInvalidRequest, index, experiment.AugmentationPolicy)
	}
	if experiment.AugmentationPolicyConfig != nil {
		if err := validateAugmentationPolicyConfig(*experiment.AugmentationPolicyConfig, index); err != nil {
			return err
		}
	}
	if experiment.ClassBalancing != "" && !allowedExperimentValue(experiment.ClassBalancing, allowedClassBalancingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported class_balancing %q", store.ErrInvalidRequest, index, experiment.ClassBalancing)
	}
	if err := validateClassBalancingConfig(experiment.ClassBalancing, experiment.ClassBalancingConfig, index); err != nil {
		return err
	}
	if experiment.SamplingStrategy != "" && !allowedExperimentValue(experiment.SamplingStrategy, allowedSamplingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported sampling_strategy %q", store.ErrInvalidRequest, index, experiment.SamplingStrategy)
	}
	if experiment.EarlyStoppingPatience < 0 || experiment.EarlyStoppingPatience > 50 {
		return fmt.Errorf("%w: proposed experiment %d early_stopping_patience must be between 0 and 50", store.ErrInvalidRequest, index)
	}
	if experiment.FineTuneStrategy != "" {
		switch strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)) {
		case "head_only", "last_block", "full":
		default:
			return fmt.Errorf("%w: proposed experiment %d has unsupported fine_tune_strategy %q", store.ErrInvalidRequest, index, experiment.FineTuneStrategy)
		}
	}
	if err := validateExperimentAutoML(experiment, index); err != nil {
		return err
	}
	return nil
}

func validateExperimentDatasetCompatibility(experiment plans.PlannedExperiment, dataset datasets.Dataset, index int) error {
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return nil
	}
	modelSpec, ok := supportedModelSpecByName(experiment.Model)
	if !ok {
		return validatePlannedExperiment(experiment, index)
	}
	template := strings.ToLower(strings.TrimSpace(experiment.Template))
	yoloEvidence := datasetHasYOLODetectionEvidence(dataset, map[string]any{})
	if modelSpec.TaskType == "object_detection" {
		if !yoloEvidence {
			return fmt.Errorf("%w: proposed experiment %d uses detector model %q but the dataset has no YOLO detection evidence", store.ErrInvalidRequest, index, experiment.Model)
		}
		if template != "" && !containsAnyText(template, "yolo", "detect") {
			return fmt.Errorf("%w: proposed experiment %d uses detector model %q with non-detection template %q", store.ErrInvalidRequest, index, experiment.Model, experiment.Template)
		}
		return nil
	}
	if yoloEvidence {
		return fmt.Errorf("%w: proposed experiment %d uses classifier model %q for a YOLO object-detection dataset", store.ErrInvalidRequest, index, experiment.Model)
	}
	if containsAnyText(template, "yolo", "detect") {
		return fmt.Errorf("%w: proposed experiment %d uses detection template %q with classifier model %q", store.ErrInvalidRequest, index, experiment.Template, experiment.Model)
	}
	return nil
}

func (s *Server) validateExperimentsDatasetCompatibility(datasetID string, experiments []plans.PlannedExperiment) error {
	if strings.TrimSpace(datasetID) == "" {
		return nil
	}
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return err
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return err
	}
	if len(metadataSummary) > 0 {
		dataset.Profile = profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary)
	}
	for index, experiment := range experiments {
		if err := validateExperimentDatasetCompatibility(experiment, dataset, index); err != nil {
			return err
		}
	}
	return nil
}

func validateExperimentAutoML(experiment plans.PlannedExperiment, index int) error {
	if experiment.AutoML == nil || !experiment.AutoML.Enabled {
		return nil
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return fmt.Errorf("%w: proposed experiment %d cannot use AutoML for report-only label-quality audit jobs", store.ErrInvalidRequest, index)
	}
	strategy := automlStrategyContext(experiment)
	if experiment.AutoML.SearchSpace != nil && len(experiment.AutoML.SearchSpace.Parameters) > 0 {
		if err := automl.ValidateSearchSpace(*experiment.AutoML.SearchSpace, strategy); err != nil {
			return fmt.Errorf("%w: proposed experiment %d has invalid AutoML search space: %s", store.ErrInvalidRequest, index, err.Error())
		}
		return nil
	}
	if len(experiment.AutoML.Intent.AllowedParameters) == 0 {
		return fmt.Errorf("%w: proposed experiment %d AutoML requires a search_space or intent.allowed_parameters", store.ErrInvalidRequest, index)
	}
	if _, err := automl.DefaultSearchSpace(experiment.AutoML.Intent.AllowedParameters, strategy); err != nil {
		return fmt.Errorf("%w: proposed experiment %d has invalid AutoML intent: %s", store.ErrInvalidRequest, index, err.Error())
	}
	return nil
}

func validateClassBalancingConfig(strategy string, config map[string]any, index int) error {
	if len(config) == 0 {
		return nil
	}
	for key, value := range config {
		normalized := strings.ToLower(strings.TrimSpace(key))
		switch normalized {
		case "effective_number_beta":
			if !effectiveNumberClassBalancing(strategy) {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.effective_number_beta is only supported with effective_number_loss class balancing", store.ErrInvalidRequest, index)
			}
			beta := payloadNumber(value)
			if beta < 0.9 || beta > 0.99999 {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.effective_number_beta must be between 0.9 and 0.99999", store.ErrInvalidRequest, index)
			}
		case "focal_loss_gamma":
			if !strings.EqualFold(strings.TrimSpace(strategy), "focal_loss") {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.focal_loss_gamma is only supported with focal_loss class balancing", store.ErrInvalidRequest, index)
			}
			gamma := payloadNumber(value)
			if gamma < 0.5 || gamma > 5 {
				return fmt.Errorf("%w: proposed experiment %d class_balancing_config.focal_loss_gamma must be between 0.5 and 5", store.ErrInvalidRequest, index)
			}
		default:
			return fmt.Errorf("%w: proposed experiment %d has unsupported class_balancing_config key %q", store.ErrInvalidRequest, index, key)
		}
	}
	return nil
}

func disallowedDirectExperimentTemplate(template string) bool {
	switch strings.ToLower(strings.TrimSpace(template)) {
	case jobs.TemplateProfileDataset,
		jobs.TemplateTrainExperiment,
		jobs.TemplateExportChampion,
		jobs.TemplateChampionDemoPrediction,
		jobs.TemplateGenerateVisualExemplars,
		jobs.TemplateAnalyzeDatasetVisuals:
		return true
	default:
		return false
	}
}

func effectiveNumberClassBalancing(strategy string) bool {
	switch strings.ToLower(strings.TrimSpace(strategy)) {
	case "effective_number", "effective_number_loss", "effective_number_class_balanced_loss", "class_balanced_effective_number":
		return true
	default:
		return false
	}
}

func validatePreprocessingConfig(preprocessing plans.Preprocessing, index int) error {
	if preprocessing.ResizeStrategy != "" && !allowedExperimentValue(preprocessing.ResizeStrategy, allowedResizeStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.resize_strategy %q", store.ErrInvalidRequest, index, preprocessing.ResizeStrategy)
	}
	if preprocessing.Normalization != "" && !allowedExperimentValue(preprocessing.Normalization, allowedNormalizations()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.normalization %q", store.ErrInvalidRequest, index, preprocessing.Normalization)
	}
	if preprocessing.CropStrategy != "" && !allowedExperimentValue(preprocessing.CropStrategy, allowedCropStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.crop_strategy %q", store.ErrInvalidRequest, index, preprocessing.CropStrategy)
	}
	if preprocessing.BBoxMode != "" && !allowedExperimentValue(preprocessing.BBoxMode, allowedBBoxModes()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.bbox_mode %q", store.ErrInvalidRequest, index, preprocessing.BBoxMode)
	}
	return nil
}

func validateAugmentationConfig(augmentation map[string]any, index int) error {
	for key, value := range augmentation {
		normalized := strings.ToLower(strings.TrimSpace(key))
		if !allowedExperimentValue(normalized, allowedAugmentationKeys()) {
			return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation key %q", store.ErrInvalidRequest, index, key)
		}
		switch value.(type) {
		case bool, int, int64, float64, string:
		default:
			return fmt.Errorf("%w: proposed experiment %d augmentation.%s must be a bool, number, or string", store.ErrInvalidRequest, index, key)
		}
	}
	return nil
}

func validateAugmentationPolicyConfig(config plans.AugmentationPolicyConfig, index int) error {
	policyType := strings.ToLower(strings.TrimSpace(config.PolicyType))
	if policyType == "" {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.policy_type is required", store.ErrInvalidRequest, index)
	}
	if !allowedExperimentValue(policyType, allowedStructuredAugmentationPolicyTypes()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation_policy_config.policy_type %q", store.ErrInvalidRequest, index, config.PolicyType)
	}
	if config.Magnitude < 0 || config.Magnitude > 15 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.magnitude must be between 0 and 15", store.ErrInvalidRequest, index)
	}
	if config.NumOps < 0 || config.NumOps > 3 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.num_ops must be between 0 and 3", store.ErrInvalidRequest, index)
	}
	if config.NumMagnitudeBins < 0 || config.NumMagnitudeBins > 31 || (config.NumMagnitudeBins > 0 && config.NumMagnitudeBins < 2) {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.num_magnitude_bins must be between 2 and 31 when set", store.ErrInvalidRequest, index)
	}
	if config.Probability < 0 || config.Probability > 1 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.probability must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	if config.Alpha < 0 || config.Alpha > 1 {
		return fmt.Errorf("%w: proposed experiment %d augmentation_policy_config.alpha must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	return nil
}

func allowedExperimentValue(value string, allowed map[string]bool) bool {
	return allowed[strings.ToLower(strings.TrimSpace(value))]
}

func allowedPlannerMechanisms() map[string]bool {
	return map[string]bool{
		"stop_select_champion":      true,
		"baseline_control":          true,
		"architecture_challenge":    true,
		"capacity_finetune":         true,
		"optimizer_scheduler":       true,
		"regularization":            true,
		"augmentation_basic":        true,
		"augmentation_auto":         true,
		"augmentation_mixed_sample": true,
		"class_imbalance":           true,
		"minority_targeting":        true,
		"resolution_crop":           true,
		"bbox_crop_ablation":        true,
		"label_noise_audit":         true,
		"hard_example_audit":        true,
		"deployment_latency":        true,
		"distillation":              true,
	}
}

func allowedOptimizers() map[string]bool {
	return map[string]bool{"adamw": true, "adam": true, "sgd": true}
}

func allowedSchedulers() map[string]bool {
	return map[string]bool{"none": true, "cosine": true, "step": true}
}

func allowedResolutionStrategies() map[string]bool {
	return map[string]bool{
		"fixed":                    true,
		"low_latency":              true,
		"compare_224_256":          true,
		"high_resolution_ablation": true,
	}
}

func allowedResizeStrategies() map[string]bool {
	return map[string]bool{
		"squash":                 true,
		"preserve_aspect_pad":    true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
		"letterbox":              true,
		"yolo_letterbox":         true,
	}
}

func allowedNormalizations() map[string]bool {
	return map[string]bool{"imagenet": true, "dataset": true, "none": true}
}

func allowedCropStrategies() map[string]bool {
	return map[string]bool{
		"none":                   true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
		"bbox_crop_ablation":     true,
	}
}

func allowedBBoxModes() map[string]bool {
	return map[string]bool{
		"ignore":                      true,
		"crop_if_available":           true,
		"crop_and_compare_full_image": true,
		"use_boxes_as_metadata":       true,
	}
}

func allowedAugmentationPolicies() map[string]bool {
	return map[string]bool{
		"none":               true,
		"light":              true,
		"moderate":           true,
		"strong":             true,
		"custom":             true,
		"basic":              true,
		"randaugment":        true,
		"trivialaugment":     true,
		"trivialaugmentwide": true,
		"autoaugment":        true,
		"mixup":              true,
		"cutmix":             true,
	}
}

func allowedStructuredAugmentationPolicyTypes() map[string]bool {
	return map[string]bool{
		"none":               true,
		"basic":              true,
		"randaugment":        true,
		"trivialaugment":     true,
		"trivialaugmentwide": true,
		"autoaugment":        true,
		"mixup":              true,
		"cutmix":             true,
	}
}

func allowedAugmentationKeys() map[string]bool {
	return map[string]bool{
		"horizontal_flip": true,
		"vertical_flip":   true,
		"color_jitter":    true,
		"random_crop":     true,
		"random_rotation": true,
		"random_erasing":  true,
	}
}

func allowedClassBalancingStrategies() map[string]bool {
	return map[string]bool{
		"none":                                 true,
		"weighted_loss":                        true,
		"class_weighted_loss":                  true,
		"effective_number":                     true,
		"effective_number_loss":                true,
		"effective_number_class_balanced_loss": true,
		"class_balanced_effective_number":      true,
		"class_balanced_sampler":               true,
		"weighted_random_sampler":              true,
		"focal_loss":                           true,
	}
}

func allowedSamplingStrategies() map[string]bool {
	return map[string]bool{
		"none":                    true,
		"class_balanced_sampler":  true,
		"weighted_random_sampler": true,
	}
}

func recommendedWorkersForExperiments(experiments []plans.PlannedExperiment) int {
	if len(experiments) < 1 {
		return 1
	}
	return len(experiments)
}

func estimateFollowUpMinutes(experiments []plans.PlannedExperiment) int {
	maxEpochs := 1
	for _, experiment := range experiments {
		if experiment.Epochs > maxEpochs {
			maxEpochs = experiment.Epochs
		}
	}

	return max(5, maxEpochs*6)
}
