package agents

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const plannerSnapshotMaxBackendGatedMethods = 12

func plannerBackendGatedMethods(input ExperimentPlannerInput) []PlannerBackendGatedMethod {
	out := []PlannerBackendGatedMethod{}
	add := func(card PlannerBackendGatedMethod) {
		card.Mechanism = normalizeMechanism(card.Mechanism)
		if !plannerIsBackendGatedMechanism(card.Mechanism) {
			return
		}
		requirements := plannerBackendGatedRequirements(card.Mechanism)
		card.RequiredConcreteFields = requirements.concreteFields
		card.RequiredBackendEvidence = requirements.backendEvidence
		card.PlannerUse = requirements.plannerUse
		card.Summary = strings.TrimSpace(card.Summary)
		if card.Summary == "" {
			card.Summary = fmt.Sprintf("%s is proposal-only guidance until backend validation receives concrete config and evidence.", card.Mechanism)
		}
		card.Evidence = cappedStrings(card.Evidence, 6)
		card.MissingRequirements = plannerBackendGatedMissingRequirements(input, card)
		card.SchedulingAuthority = false
		if card.ProposalStatus == "" {
			card.ProposalStatus = plannerBackendGatedProposalStatus(card.Mechanism, card.MissingRequirements)
		}
		card.SupportedConfigHints = compactAnyMap(card.SupportedConfigHints, 8)

		for index := range out {
			if out[index].Mechanism != card.Mechanism {
				continue
			}
			out[index].Evidence = mergeCappedStrings(out[index].Evidence, card.Evidence, 6)
			out[index].MissingRequirements = mergeCappedStrings(out[index].MissingRequirements, card.MissingRequirements, 8)
			out[index].RequiredConcreteFields = mergeCappedStrings(out[index].RequiredConcreteFields, card.RequiredConcreteFields, 8)
			out[index].RequiredBackendEvidence = mergeCappedStrings(out[index].RequiredBackendEvidence, card.RequiredBackendEvidence, 8)
			out[index].SupportedConfigHints = mergePlannerConfigHints(out[index].SupportedConfigHints, card.SupportedConfigHints)
			out[index].ProposalStatus = plannerBackendGatedProposalStatus(out[index].Mechanism, out[index].MissingRequirements)
			return
		}

		out = append(out, card)
	}

	if input.VisualExemplarContext != nil {
		source := strings.TrimSpace(input.VisualExemplarContext.Source)
		if source == "" {
			source = "visual_analysis"
		}
		for _, hypothesis := range capPlannerPreprocessingHypotheses(input.VisualExemplarContext.PreprocessingHypotheses, plannerVisualMaxHypotheses) {
			mechanism := normalizeMechanism(hypothesis.Mechanism)
			if !plannerIsBackendGatedMechanism(mechanism) {
				continue
			}
			evidence := append([]string{}, hypothesis.Evidence...)
			if strings.TrimSpace(hypothesis.SupportStatus) != "" {
				evidence = append(evidence, "visual support status "+strings.ToLower(strings.TrimSpace(hypothesis.SupportStatus)))
			}
			if strings.TrimSpace(hypothesis.ExpectedEffect) != "" {
				evidence = append(evidence, "expected effect: "+hypothesis.ExpectedEffect)
			}
			add(PlannerBackendGatedMethod{
				Mechanism:            mechanism,
				Source:               source,
				SourceID:             strings.TrimSpace(hypothesis.ID),
				Summary:              hypothesis.Summary,
				Evidence:             evidence,
				SupportedConfigHints: plannerVisualHypothesisConfigHints(hypothesis),
			})
		}
	}

	perClass := plannerPerClassErrorCard(input)
	if plannerHasClassImbalanceSignal(input, perClass) {
		add(PlannerBackendGatedMethod{
			Mechanism: "class_imbalance",
			Source:    "deterministic_diagnosis",
			Summary:   "Class imbalance evidence can guide weighted loss or sampler proposals.",
			Evidence:  plannerClassImbalanceEvidence(input, perClass),
			SupportedConfigHints: map[string]any{
				"class_balancing_options": []string{"weighted_loss", "class_weighted_loss", "class_balanced_sampler", "focal_loss", "effective_number_loss"},
				"sampling_strategy_options": []string{
					"class_balanced_sampler",
					"weighted_random_sampler",
				},
			},
		})
	}
	if plannerHasMinorityTargetingSignal(input, perClass) {
		add(PlannerBackendGatedMethod{
			Mechanism: "minority_targeting",
			Source:    "per_class_error_card",
			Summary:   "Weak minority or per-class behavior can guide targeted sampling or loss proposals.",
			Evidence:  plannerClassImbalanceEvidence(input, perClass),
			SupportedConfigHints: map[string]any{
				"class_balancing_options": []string{"weighted_loss", "focal_loss", "effective_number_loss"},
				"sampling_strategy_options": []string{
					"class_balanced_sampler",
					"weighted_random_sampler",
				},
			},
		})
	}
	if plannerHasResolutionCropSignal(input) {
		add(PlannerBackendGatedMethod{
			Mechanism: "resolution_crop",
			Source:    "dataset_profile",
			Summary:   "Dataset dimensions or visual traits suggest a resolution/crop ablation may be worth proposing.",
			Evidence:  plannerResolutionCropEvidence(input),
			SupportedConfigHints: map[string]any{
				"resolution_strategy_options": []string{"compare_224_256", "high_resolution_ablation"},
				"image_size_options":          []int{224, 256, 320},
				"preprocessing_options":       []string{"preserve_aspect_pad", "center_crop", "random_resized_crop"},
			},
		})
	}
	if plannerInputHasBBoxEvidence(input) {
		add(PlannerBackendGatedMethod{
			Mechanism: "bbox_crop_ablation",
			Source:    "dataset_metadata",
			Summary:   "Backend metadata or profile contains bbox annotation evidence, so bbox crop can be proposed with concrete preprocessing.",
			Evidence:  plannerBBoxEvidence(input),
			SupportedConfigHints: map[string]any{
				"preprocessing_options": []plans.Preprocessing{
					{ResizeStrategy: "bbox_crop_if_available", CropStrategy: "bbox_crop_ablation", BBoxMode: "crop_if_available", Normalization: "imagenet"},
				},
			},
		})
	}
	if plannerRecommendedAutoAugmentation(input) {
		add(PlannerBackendGatedMethod{
			Mechanism: "augmentation_auto",
			Source:    "dataset_profile",
			Summary:   "Dataset recommendations mention a structured automatic augmentation policy.",
			Evidence:  plannerAugmentationRecommendationEvidence(input, "randaugment", "trivialaugment", "trivialaugmentwide", "autoaugment"),
			SupportedConfigHints: map[string]any{
				"augmentation_policy_options": []string{"randaugment", "trivialaugment", "trivialaugmentwide", "autoaugment"},
				"augmentation_policy_config":  plans.AugmentationPolicyConfig{PolicyType: "randaugment", Magnitude: 7, NumOps: 2},
			},
		})
	}
	if plannerRecommendedMixedSampleAugmentation(input) {
		add(PlannerBackendGatedMethod{
			Mechanism: "augmentation_mixed_sample",
			Source:    "dataset_profile",
			Summary:   "Dataset recommendations mention MixUp or CutMix style mixed-sample augmentation.",
			Evidence:  plannerAugmentationRecommendationEvidence(input, "mixup", "cutmix"),
			SupportedConfigHints: map[string]any{
				"augmentation_policy_options": []string{"mixup", "cutmix"},
				"augmentation_policy_config":  plans.AugmentationPolicyConfig{PolicyType: "mixup", Probability: 0.5, Alpha: 0.2},
			},
		})
	}

	labelQuality := plannerLabelQualityCard(input)
	if labelQuality.AuditRecommended {
		add(PlannerBackendGatedMethod{
			Mechanism: "label_noise_audit",
			Source:    "label_quality_card",
			Summary:   "Label-quality signals can be proposed only as a report-only label audit.",
			Evidence:  labelQuality.Evidence,
			SupportedConfigHints: map[string]any{
				"template":    "label_quality_audit",
				"report_only": true,
			},
		})
		if len(labelQuality.AsymmetricConfusions) > 0 || len(perClass.WorstClasses) > 0 {
			add(PlannerBackendGatedMethod{
				Mechanism: "hard_example_audit",
				Source:    "label_quality_card",
				Summary:   "Hard-example signals can be proposed only as a report-only label audit.",
				Evidence:  labelQuality.Evidence,
				SupportedConfigHints: map[string]any{
					"template":    "label_quality_audit",
					"report_only": true,
				},
			})
		}
	}

	deployment := plannerDeploymentCard(input)
	if plannerHasDeploymentLatencySignal(input, deployment) {
		add(PlannerBackendGatedMethod{
			Mechanism: "deployment_latency",
			Source:    "objective_context",
			Summary:   "Latency, runtime, cost, live, or compact-model evidence can guide deployment-latency proposals.",
			Evidence:  plannerDeploymentLatencyEvidence(input, deployment),
			SupportedConfigHints: map[string]any{
				"preferred_deployment_models": plannerLatencyFriendlyModels(input.ModelCatalog),
				"resolution_strategy_options": []string{"low_latency", "fixed"},
			},
		})
	}

	if plannerDistillationMentioned(input) {
		add(PlannerBackendGatedMethod{
			Mechanism:      "distillation",
			ProposalStatus: "future_blocked",
			Source:         "planning_context",
			Summary:        "Distillation is retained only as a future option until teacher artifacts and worker support exist.",
			Evidence:       []string{"distillation or teacher model was mentioned in planning context"},
		})
	}

	if len(out) > plannerSnapshotMaxBackendGatedMethods {
		out = out[:plannerSnapshotMaxBackendGatedMethods]
	}
	return out
}

type plannerBackendGatedRequirement struct {
	concreteFields  []string
	backendEvidence []string
	plannerUse      string
}

func plannerBackendGatedRequirements(mechanism string) plannerBackendGatedRequirement {
	switch mechanism {
	case "bbox_crop_ablation":
		return plannerBackendGatedRequirement{
			concreteFields: []string{"bbox crop preprocessing", "preprocessing.crop_strategy or preprocessing.bbox_mode"},
			backendEvidence: []string{
				"dataset profile bbox/annotation evidence",
				"worker-readable bounding boxes or compatible annotations",
			},
			plannerUse: "Use as a crop-method candidate only after adding concrete bbox preprocessing and citing backend bbox evidence.",
		}
	case "resolution_crop":
		return plannerBackendGatedRequirement{
			concreteFields: []string{"image_size, resolution_strategy, or preprocessing resize/crop change"},
			backendEvidence: []string{
				"object-scale, fine-grained, dimension, crop, or accepted visual-trait evidence",
			},
			plannerUse: "Use to shape an image-size, resize, crop, or high-resolution ablation with concrete config.",
		}
	case "augmentation_auto":
		return plannerBackendGatedRequirement{
			concreteFields:  []string{"augmentation_policy or augmentation_policy_config using randaugment, trivialaugment, trivialaugmentwide, or autoaugment"},
			backendEvidence: []string{"dataset, visual, or diagnosis evidence explaining why structured augmentation should help"},
			plannerUse:      "Use only when the final experiment names a supported automatic augmentation policy/config.",
		}
	case "augmentation_mixed_sample":
		return plannerBackendGatedRequirement{
			concreteFields:  []string{"augmentation_policy or augmentation_policy_config using mixup or cutmix", "probability and alpha within backend bounds when configured"},
			backendEvidence: []string{"dataset, visual, or diagnosis evidence explaining why mixed-sample augmentation should help"},
			plannerUse:      "Use only when the final experiment names a supported MixUp or CutMix policy/config.",
		}
	case "class_imbalance", "minority_targeting":
		return plannerBackendGatedRequirement{
			concreteFields: []string{"class_balancing or sampling_strategy"},
			backendEvidence: []string{
				"dataset imbalance, minority-class, per-class-error, or macro-F1-vs-accuracy evidence",
			},
			plannerUse: "Use for weighted loss, focal loss, or sampler proposals tied to macro-F1/minority recall evidence.",
		}
	case "label_noise_audit", "hard_example_audit":
		return plannerBackendGatedRequirement{
			concreteFields:  []string{"template=label_quality_audit", "report-only audit payload"},
			backendEvidence: []string{"label-quality, hard-example, per-class, or high-confidence-error evidence"},
			plannerUse:      "Use only as a report-only audit recommendation; never mutate labels or schedule a training experiment.",
		}
	case "deployment_latency":
		return plannerBackendGatedRequirement{
			concreteFields:  []string{"compact or latency-oriented candidate design", "evidence_used cites latency, runtime, cost, live, edge, or mobile constraints"},
			backendEvidence: []string{"objective or run evidence with latency, runtime, cost, live, edge, compact, or mobile constraints"},
			plannerUse:      "Use to compare compact or low-latency candidates against quality tradeoffs.",
		}
	case "distillation":
		return plannerBackendGatedRequirement{
			concreteFields:  []string{"teacher artifact selection", "worker distillation support"},
			backendEvidence: []string{"validated teacher model artifact and worker-side distillation support"},
			plannerUse:      "Do not schedule; retain only as a future blocked option.",
		}
	default:
		return plannerBackendGatedRequirement{}
	}
}

func plannerBackendGatedMissingRequirements(input ExperimentPlannerInput, card PlannerBackendGatedMethod) []string {
	missing := []string{}
	switch card.Mechanism {
	case "bbox_crop_ablation":
		if !plannerCardHasBBoxConfig(card) {
			missing = append(missing, "concrete bbox crop preprocessing")
		}
		if !plannerInputHasBBoxEvidence(input) {
			missing = append(missing, "backend-profiled bbox/annotation evidence")
		}
	case "resolution_crop":
		if !plannerCardHasResolutionConfig(card) {
			missing = append(missing, "concrete image_size, resolution_strategy, or preprocessing resize/crop change")
		}
		if !plannerHasResolutionCropEvidence(input, card.Evidence) {
			missing = append(missing, "object-scale, fine-grained, dimension, crop, or visual-trait evidence")
		}
	case "augmentation_auto":
		if !plannerCardHasAutoAugmentationConfig(card) {
			missing = append(missing, "structured randaugment, trivialaugment, trivialaugmentwide, or autoaugment config")
		}
	case "augmentation_mixed_sample":
		if !plannerCardHasMixedSampleConfig(card) {
			missing = append(missing, "structured mixup or cutmix config")
		}
	case "class_imbalance", "minority_targeting":
		if !plannerCardHasClassBalancingConfig(card) {
			missing = append(missing, "concrete class_balancing or sampling_strategy")
		}
		if !plannerHasClassImbalanceEvidence(input, card.Evidence) {
			missing = append(missing, "dataset imbalance, minority, per-class, or macro-F1-vs-accuracy evidence")
		}
	case "label_noise_audit", "hard_example_audit":
		if !plannerLabelQualityCard(input).AuditRecommended {
			missing = append(missing, "label-quality or hard-example audit evidence")
		}
		if !plannerCardHasLabelAuditConfig(card) {
			missing = append(missing, "report-only label_quality_audit template")
		}
	case "deployment_latency":
		if !plannerCardHasDeploymentConfig(card) {
			missing = append(missing, "compact or latency-oriented candidate design")
		}
		if !plannerHasDeploymentLatencyEvidence(input, card.Evidence) {
			missing = append(missing, "latency, runtime, cost, live, edge, compact, or mobile evidence")
		}
	case "distillation":
		missing = append(missing, "teacher-artifact validation", "worker-side distillation support")
	}
	return cappedStrings(missing, 8)
}

func plannerBackendGatedProposalStatus(mechanism string, missing []string) string {
	if mechanism == "distillation" {
		return "future_blocked"
	}
	if len(missing) > 0 {
		return "backend_validation_required"
	}
	return "proposal_only"
}

func plannerIsBackendGatedMechanism(mechanism string) bool {
	switch normalizeMechanism(mechanism) {
	case "bbox_crop_ablation", "resolution_crop", "augmentation_auto", "augmentation_mixed_sample",
		"class_imbalance", "minority_targeting", "label_noise_audit", "hard_example_audit",
		"deployment_latency", "distillation":
		return true
	default:
		return false
	}
}

func plannerVisualHypothesisConfigHints(hypothesis datasets.PreprocessingHypothesis) map[string]any {
	hints := map[string]any{}
	if hypothesis.SuggestedPreprocessing != nil {
		hints["preprocessing"] = *hypothesis.SuggestedPreprocessing
	}
	if len(hypothesis.SuggestedImageSizes) > 0 {
		hints["suggested_image_sizes"] = capInts(hypothesis.SuggestedImageSizes, 3)
	}
	if strings.TrimSpace(hypothesis.SuggestedAugmentationPolicy) != "" {
		hints["augmentation_policy"] = strings.ToLower(strings.TrimSpace(hypothesis.SuggestedAugmentationPolicy))
	}
	if hypothesis.SuggestedAugmentationConfig != nil {
		hints["augmentation_policy_config"] = *hypothesis.SuggestedAugmentationConfig
	}
	if len(hints) == 0 {
		return nil
	}
	return hints
}

func plannerCardHasBBoxConfig(card PlannerBackendGatedMethod) bool {
	text := strings.ToLower(plannerPreprocessingHintText(card.SupportedConfigHints))
	return containsAnyText(text, "bbox", "box")
}

func plannerCardHasResolutionConfig(card PlannerBackendGatedMethod) bool {
	if len(card.SupportedConfigHints) == 0 {
		return false
	}
	if _, ok := card.SupportedConfigHints["image_size"]; ok {
		return true
	}
	if _, ok := card.SupportedConfigHints["suggested_image_sizes"]; ok {
		return true
	}
	if value := strings.ToLower(stringFromAny(card.SupportedConfigHints["resolution_strategy"])); value != "" && value != "fixed" {
		return true
	}
	return strings.TrimSpace(plannerPreprocessingHintText(card.SupportedConfigHints)) != ""
}

func plannerCardHasAutoAugmentationConfig(card PlannerBackendGatedMethod) bool {
	policy := strings.ToLower(stringFromAny(card.SupportedConfigHints["augmentation_policy"]))
	if containsAnyText(policy, "randaugment", "trivialaugment", "autoaugment") {
		return true
	}
	policyType := strings.ToLower(plannerAugmentationPolicyConfigType(card.SupportedConfigHints["augmentation_policy_config"]))
	return policyType == "randaugment" || policyType == "trivialaugment" || policyType == "trivialaugmentwide" || policyType == "autoaugment"
}

func plannerCardHasMixedSampleConfig(card PlannerBackendGatedMethod) bool {
	policy := strings.ToLower(stringFromAny(card.SupportedConfigHints["augmentation_policy"]))
	if policy == "mixup" || policy == "cutmix" {
		return true
	}
	policyType := strings.ToLower(plannerAugmentationPolicyConfigType(card.SupportedConfigHints["augmentation_policy_config"]))
	return policyType == "mixup" || policyType == "cutmix"
}

func plannerCardHasClassBalancingConfig(card PlannerBackendGatedMethod) bool {
	if nonDefaultText(stringFromAny(card.SupportedConfigHints["class_balancing"]), "none") {
		return true
	}
	return nonDefaultText(stringFromAny(card.SupportedConfigHints["sampling_strategy"]), "none")
}

func plannerCardHasLabelAuditConfig(card PlannerBackendGatedMethod) bool {
	template := strings.ToLower(stringFromAny(card.SupportedConfigHints["template"]))
	reportOnly, _ := card.SupportedConfigHints["report_only"].(bool)
	return template == "label_quality_audit" && reportOnly
}

func plannerCardHasDeploymentConfig(card PlannerBackendGatedMethod) bool {
	if len(stringsFromAny(card.SupportedConfigHints["preferred_deployment_models"])) > 0 {
		return true
	}
	return nonDefaultText(stringFromAny(card.SupportedConfigHints["resolution_strategy"]), "fixed")
}

func plannerPreprocessingHintText(hints map[string]any) string {
	value := hints["preprocessing"]
	switch typed := value.(type) {
	case plans.Preprocessing:
		return strings.Join([]string{typed.ResizeStrategy, typed.CropStrategy, typed.BBoxMode, typed.Normalization}, " ")
	case *plans.Preprocessing:
		if typed == nil {
			return ""
		}
		return strings.Join([]string{typed.ResizeStrategy, typed.CropStrategy, typed.BBoxMode, typed.Normalization}, " ")
	case map[string]any:
		return strings.Join([]string{
			stringFromAny(typed["resize_strategy"]),
			stringFromAny(typed["crop_strategy"]),
			stringFromAny(typed["bbox_mode"]),
			stringFromAny(typed["normalization"]),
		}, " ")
	default:
		blob, _ := json.Marshal(value)
		return string(blob)
	}
}

func plannerAugmentationPolicyConfigType(value any) string {
	switch typed := value.(type) {
	case plans.AugmentationPolicyConfig:
		return strings.TrimSpace(typed.PolicyType)
	case *plans.AugmentationPolicyConfig:
		if typed == nil {
			return ""
		}
		return strings.TrimSpace(typed.PolicyType)
	case map[string]any:
		return stringFromAny(typed["policy_type"])
	default:
		return ""
	}
}

func plannerHasClassImbalanceSignal(input ExperimentPlannerInput, perClass PlannerPerClassErrorCard) bool {
	return input.DatasetInsights.ImbalanceRatio >= 1.5 ||
		input.DeterministicDiagnosis.ClassImbalanceScore >= 0.45 ||
		perClass.ImbalanceActive ||
		plannerClassDistributionImbalanceRatio(input.DatasetInsights.ClassDistribution) >= 1.5 ||
		plannerProfileFloat(input.Dataset.Profile, "imbalance_ratio") >= 1.5
}

func plannerHasMinorityTargetingSignal(input ExperimentPlannerInput, perClass PlannerPerClassErrorCard) bool {
	return input.DeterministicDiagnosis.MinorityClassFailureScore >= 0.45 ||
		perClass.MinorityFailureActive ||
		perClass.ClassBalancingUseful ||
		weakWorstClass(perClass.WorstClasses)
}

func plannerHasClassImbalanceEvidence(input ExperimentPlannerInput, evidence []string) bool {
	if plannerHasClassImbalanceSignal(input, plannerPerClassErrorCard(input)) {
		return true
	}
	if plannerClassDistributionImbalanceRatio(plannerMap(input.Dataset.Profile, "class_distribution")) >= 1.5 ||
		plannerClassDistributionImbalanceRatio(plannerMap(input.Dataset.Profile, "images_per_class")) >= 1.5 {
		return true
	}
	return containsAnyText(strings.ToLower(strings.Join(evidence, " ")), "class imbalance", "class_imbalance", "minority", "rare class", "per-class", "per class", "macro-f1", "macro f1")
}

func plannerClassImbalanceEvidence(input ExperimentPlannerInput, perClass PlannerPerClassErrorCard) []string {
	evidence := append([]string{}, perClass.Evidence...)
	if input.DatasetInsights.ImbalanceRatio > 0 {
		evidence = append(evidence, fmt.Sprintf("dataset imbalance ratio %.2f", input.DatasetInsights.ImbalanceRatio))
	}
	if input.DeterministicDiagnosis.ClassImbalanceScore > 0 {
		evidence = append(evidence, fmt.Sprintf("class imbalance score %.2f", input.DeterministicDiagnosis.ClassImbalanceScore))
	}
	if input.DeterministicDiagnosis.MinorityClassFailureScore > 0 {
		evidence = append(evidence, fmt.Sprintf("minority failure score %.2f", input.DeterministicDiagnosis.MinorityClassFailureScore))
	}
	return cappedStrings(evidence, 6)
}

func plannerHasResolutionCropSignal(input ExperimentPlannerInput) bool {
	return plannerHasResolutionCropEvidence(input, nil)
}

func plannerHasResolutionCropEvidence(input ExperimentPlannerInput, evidence []string) bool {
	textParts := append([]string{}, evidence...)
	textParts = append(textParts, input.DatasetInsights.DatasetTraits...)
	textParts = append(textParts, input.DatasetInsights.RecommendedPreprocessing...)
	if input.VisualExemplarContext != nil {
		textParts = append(textParts, input.VisualExemplarContext.ObservedTraits...)
		for _, hypothesis := range input.VisualExemplarContext.PreprocessingHypotheses {
			textParts = append(textParts, hypothesis.Summary, hypothesis.ExpectedEffect)
			textParts = append(textParts, hypothesis.Evidence...)
		}
	}
	text := strings.ToLower(strings.Join(textParts, " "))
	if containsAnyText(text, "small object", "small_objects", "large object", "object scale", "fine-grained", "fine grained", "crop", "bbox", "aspect ratio", "variable dimensions", "background dominance", "high resolution", "image dimension") {
		return true
	}
	if input.DatasetInsights.WidthMax >= 512 || input.DatasetInsights.HeightMax >= 512 {
		return true
	}
	if input.DatasetInsights.WidthMin > 0 && input.DatasetInsights.WidthMax > input.DatasetInsights.WidthMin*2 {
		return true
	}
	if input.DatasetInsights.HeightMin > 0 && input.DatasetInsights.HeightMax > input.DatasetInsights.HeightMin*2 {
		return true
	}
	widthMin := plannerProfileFloat(input.Dataset.Profile, "width_min")
	widthMax := plannerProfileFloat(input.Dataset.Profile, "width_max")
	heightMin := plannerProfileFloat(input.Dataset.Profile, "height_min")
	heightMax := plannerProfileFloat(input.Dataset.Profile, "height_max")
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

func plannerResolutionCropEvidence(input ExperimentPlannerInput) []string {
	evidence := []string{}
	if input.DatasetInsights.WidthMax > 0 || input.DatasetInsights.HeightMax > 0 {
		evidence = append(evidence, fmt.Sprintf("dataset dimensions width %d-%d height %d-%d", input.DatasetInsights.WidthMin, input.DatasetInsights.WidthMax, input.DatasetInsights.HeightMin, input.DatasetInsights.HeightMax))
	}
	for _, value := range append(input.DatasetInsights.DatasetTraits, input.DatasetInsights.RecommendedPreprocessing...) {
		if containsAnyText(strings.ToLower(value), "small object", "crop", "aspect", "resolution", "fine") {
			evidence = append(evidence, value)
		}
	}
	if input.VisualExemplarContext != nil {
		for _, value := range input.VisualExemplarContext.ObservedTraits {
			if containsAnyText(strings.ToLower(value), "object", "crop", "bbox", "aspect", "background", "fine") {
				evidence = append(evidence, value)
			}
		}
	}
	return cappedStrings(evidence, 6)
}

func plannerProfileHasBBoxEvidence(profile map[string]any) bool {
	if plannerProfileBool(profile, "bbox_available") || plannerProfileBool(profile, "annotations_available") ||
		plannerProfileFloat(profile, "bbox_annotations_count") > 0 || plannerProfileFloat(profile, "bbox_count") > 0 ||
		plannerArtifactCountsHaveBBoxEvidence(plannerMap(profile, "artifact_counts")) {
		return true
	}
	metadata := plannerMap(profile, "metadata_summary")
	if plannerMetadataSummaryHasBBoxEvidence(metadata) {
		return true
	}
	if plannerMetadataSummaryHasBBoxEvidence(plannerMap(profile, "agent_safe_metadata_summary")) ||
		plannerMetadataSummaryHasBBoxEvidence(plannerMap(profile, "normalized_metadata_summary")) {
		return true
	}
	visualTraits := plannerMap(profile, "visual_trait_summary")
	if plannerProfileFloat(visualTraits, "bbox_count") > 0 {
		return true
	}
	for _, trait := range append(plannerStringSlice(profile, "dataset_traits"), plannerStringSlice(metadata, "dataset_traits")...) {
		if containsAnyText(strings.ToLower(trait), "bbox", "bounding box", "annotation") {
			return true
		}
	}
	for _, artifact := range plannerMapSlice(profile["artifacts"]) {
		blob, _ := json.Marshal(artifact)
		if containsAnyText(strings.ToLower(string(blob)), "bbox", "bounding_box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func plannerInputHasBBoxEvidence(input ExperimentPlannerInput) bool {
	return plannerProfileHasBBoxEvidence(input.Dataset.Profile) ||
		plannerMetadataSummaryHasBBoxEvidence(input.DatasetInsights.MetadataSummary) ||
		plannerMetadataSummaryHasBBoxEvidence(input.DatasetInsights.AgentSafeMetadataSummary)
}

func plannerBBoxEvidence(input ExperimentPlannerInput) []string {
	evidence := []string{}
	if plannerMetadataSummaryHasBBoxEvidence(input.DatasetInsights.AgentSafeMetadataSummary) {
		if count := plannerProfileFloat(input.DatasetInsights.AgentSafeMetadataSummary, "bbox_annotation_count"); count > 0 {
			evidence = append(evidence, fmt.Sprintf("normalized metadata safe summary reports %.0f bbox annotation(s)", count))
		} else if count := plannerProfileFloat(input.DatasetInsights.AgentSafeMetadataSummary, "bbox_sample_count"); count > 0 {
			evidence = append(evidence, fmt.Sprintf("normalized metadata safe summary reports %.0f sample(s) with bbox annotations", count))
		} else {
			evidence = append(evidence, "normalized metadata safe summary includes bbox annotation evidence")
		}
	}
	if plannerMetadataSummaryHasBBoxEvidence(input.DatasetInsights.MetadataSummary) {
		evidence = append(evidence, "legacy metadata summary includes bbox/annotation evidence")
	}
	if plannerProfileHasBBoxEvidence(input.Dataset.Profile) {
		evidence = append(evidence, "dataset profile includes bbox/annotation evidence")
	}
	return cappedStrings(evidence, 6)
}

func plannerMetadataSummaryHasBBoxEvidence(summary map[string]any) bool {
	if len(summary) == 0 {
		return false
	}
	if plannerProfileBool(summary, "bbox_available") || plannerProfileBool(summary, "annotations_available") ||
		plannerProfileFloat(summary, "bbox_annotations_count") > 0 ||
		plannerProfileFloat(summary, "bbox_annotation_count") > 0 ||
		plannerProfileFloat(summary, "bbox_sample_count") > 0 ||
		plannerProfileFloat(summary, "bbox_count") > 0 ||
		plannerProfileFloat(summary, "bbox_coverage_ratio") > 0 ||
		plannerArtifactCountsHaveBBoxEvidence(plannerMap(summary, "artifact_counts")) {
		return true
	}
	annotationCounts := plannerMap(summary, "annotation_counts")
	if plannerProfileFloat(annotationCounts, "bbox") > 0 || plannerProfileFloat(annotationCounts, "bounding_box") > 0 {
		return true
	}
	capabilities := plannerMap(summary, "capabilities")
	if plannerProfileBool(capabilities, "bbox") || plannerProfileBool(capabilities, "bbox_annotations") ||
		plannerProfileBool(capabilities, "bbox_crop") || plannerProfileBool(capabilities, "object_detection") {
		return true
	}
	return false
}

func plannerArtifactCountsHaveBBoxEvidence(counts map[string]any) bool {
	for key, value := range counts {
		count, _ := anyFloat(value)
		if count <= 0 {
			continue
		}
		if containsAnyText(strings.ToLower(strings.TrimSpace(key)), "bbox", "bounding_box", "bounding box", "annotation", "coco", "voc") {
			return true
		}
	}
	return false
}

func plannerRecommendedAutoAugmentation(input ExperimentPlannerInput) bool {
	return plannerAugmentationRecommendationContains(input, "randaugment", "trivialaugment", "trivialaugmentwide", "autoaugment")
}

func plannerRecommendedMixedSampleAugmentation(input ExperimentPlannerInput) bool {
	return plannerAugmentationRecommendationContains(input, "mixup", "cutmix")
}

func plannerAugmentationRecommendationContains(input ExperimentPlannerInput, terms ...string) bool {
	text := strings.ToLower(strings.Join(append(append([]string{}, input.DatasetInsights.RecommendedAugmentations...), input.DatasetInsights.DatasetTraits...), " "))
	return containsAnyText(text, terms...)
}

func plannerAugmentationRecommendationEvidence(input ExperimentPlannerInput, terms ...string) []string {
	evidence := []string{}
	for _, value := range input.DatasetInsights.RecommendedAugmentations {
		if containsAnyText(strings.ToLower(value), terms...) {
			evidence = append(evidence, value)
		}
	}
	return cappedStrings(evidence, 6)
}

func plannerHasDeploymentLatencySignal(input ExperimentPlannerInput, deployment PlannerDeploymentCard) bool {
	return plannerObjectiveHasLatencyEvidence(input.ObjectiveContext) ||
		input.DeterministicDiagnosis.LatencyPenalty >= 0.45 ||
		deployment.BestLatencyMS > 0 ||
		deployment.ChampionLatencyMS > 0
}

func plannerHasDeploymentLatencyEvidence(input ExperimentPlannerInput, evidence []string) bool {
	if plannerObjectiveHasLatencyEvidence(input.ObjectiveContext) || input.DeterministicDiagnosis.LatencyPenalty >= 0.45 {
		return true
	}
	if containsAnyText(strings.ToLower(strings.Join(evidence, " ")), "latency", "runtime", "cost", "live", "edge", "compact", "mobile") {
		return true
	}
	for _, evaluation := range append(append([]runs.TrainingRunEvaluation(nil), input.PlanEvaluations...), input.PriorEvaluations...) {
		if firstPositivePayloadFloat(evaluation.ModelProfile, "estimated_pipeline_latency_ms", "estimated_latency_ms", "latency_ms", "p50_latency_ms", "inference_latency_ms") > 0 {
			return true
		}
	}
	for _, summary := range append(append([]runs.TrainingRunSummary(nil), input.PlanSummaries...), input.PriorSummaries...) {
		if summary.RuntimeSeconds > 0 || summary.EstimatedCostUSD > 0 {
			return true
		}
	}
	return false
}

func plannerDeploymentLatencyEvidence(input ExperimentPlannerInput, deployment PlannerDeploymentCard) []string {
	evidence := []string{}
	if plannerObjectiveHasLatencyEvidence(input.ObjectiveContext) {
		evidence = append(evidence, "objective context includes latency, runtime, cost, live, edge, compact, or mobile constraints")
	}
	if input.DeterministicDiagnosis.LatencyPenalty > 0 {
		evidence = append(evidence, fmt.Sprintf("latency penalty %.2f", input.DeterministicDiagnosis.LatencyPenalty))
	}
	if deployment.BestLatencyMS > 0 {
		evidence = append(evidence, fmt.Sprintf("best model latency %.1fms", deployment.BestLatencyMS))
	}
	if deployment.ChampionLatencyMS > 0 {
		evidence = append(evidence, fmt.Sprintf("champion latency %.1fms", deployment.ChampionLatencyMS))
	}
	for _, summary := range append(append([]runs.TrainingRunSummary(nil), input.PlanSummaries...), input.PriorSummaries...) {
		if summary.RuntimeSeconds > 0 {
			evidence = append(evidence, fmt.Sprintf("run %s runtime %.1fs", summary.JobID, summary.RuntimeSeconds))
			break
		}
		if summary.EstimatedCostUSD > 0 {
			evidence = append(evidence, fmt.Sprintf("run %s cost %.3f", summary.JobID, summary.EstimatedCostUSD))
			break
		}
	}
	return cappedStrings(evidence, 6)
}

func plannerObjectiveHasLatencyEvidence(context ProjectObjectiveContext) bool {
	parts := []string{context.GoalText, context.PrimaryObjective}
	parts = append(parts, context.MetricPreferences...)
	parts = append(parts, context.DeploymentPriorities...)
	parts = append(parts, context.Constraints...)
	for key, weight := range context.RankingWeights {
		if weight > 0 {
			parts = append(parts, key)
		}
	}
	return containsAnyText(strings.ToLower(strings.Join(parts, " ")), "latency", "runtime", "cost", "edge", "live", "real-time", "realtime", "fast", "compact", "mobile")
}

func plannerLatencyFriendlyModels(catalog []SupportedModelSpec) []string {
	models := []string{}
	for _, spec := range catalog {
		text := strings.ToLower(strings.Join([]string{spec.Name, spec.Family, spec.DeploymentTier, spec.ExpectedLatencyClass, spec.RecommendedUse}, " "))
		if containsAnyText(text, "low", "fast", "mobile", "compact", "edge") {
			models = append(models, spec.Name)
		}
	}
	sort.Strings(models)
	return cappedStrings(models, 6)
}

func plannerDistillationMentioned(input ExperimentPlannerInput) bool {
	parts := []string{input.Project.Goal, input.ObjectiveContext.GoalText, input.ObjectiveContext.PrimaryObjective}
	for _, memory := range append(append([]PlannerStrategyMemory{}, input.SuccessfulStrategyMemory...), input.FailedStrategyMemory...) {
		parts = append(parts, memory.Lesson)
		parts = append(parts, memory.Tags...)
	}
	for _, option := range input.RejectedStrategyMemory {
		parts = append(parts, option.Option, option.Reason, option.Evidence)
		parts = append(parts, option.AppliesWhen...)
	}
	return containsAnyText(strings.ToLower(strings.Join(parts, " ")), "distill", "distillation", "teacher model", "teacher artifact")
}

func plannerMap(values map[string]any, key string) map[string]any {
	return plannerAnyMap(values[key])
}

func plannerAnyMap(value any) map[string]any {
	switch typed := value.(type) {
	case map[string]any:
		return typed
	case map[string]float64:
		out := map[string]any{}
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]int:
		out := map[string]any{}
		for key, value := range typed {
			out[key] = value
		}
		return out
	default:
		return nil
	}
}

func plannerMapSlice(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		return typed
	case []any:
		out := []map[string]any{}
		for _, item := range typed {
			if converted := plannerAnyMap(item); converted != nil {
				out = append(out, converted)
			}
		}
		return out
	default:
		return nil
	}
}

func plannerStringSlice(values map[string]any, key string) []string {
	return stringsFromAny(values[key])
}

func plannerProfileFloat(values map[string]any, key string) float64 {
	value, _ := anyFloat(values[key])
	return value
}

func plannerProfileBool(values map[string]any, key string) bool {
	switch value := values[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func plannerClassDistributionImbalanceRatio(distribution map[string]any) float64 {
	minCount := 0.0
	maxCount := 0.0
	for _, value := range distribution {
		count, _ := anyFloat(value)
		if count <= 0 {
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

func mergePlannerConfigHints(left map[string]any, right map[string]any) map[string]any {
	if len(left) == 0 {
		return right
	}
	for key, value := range right {
		if _, exists := left[key]; !exists {
			left[key] = value
		}
	}
	return compactAnyMap(left, 8)
}

func mergeCappedStrings(left []string, right []string, limit int) []string {
	return cappedStrings(append(append([]string{}, left...), right...), limit)
}
