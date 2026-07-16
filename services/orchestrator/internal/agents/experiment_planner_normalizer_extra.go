package agents

import "strings"

func prunePlannerEmptyExperimentFields(experiment map[string]any, normalizations *[]string, path string) {
	if experiment == nil {
		return
	}
	changed := false
	changed = prunePlannerEmptyFields(experiment, []string{
		"mechanism",
		"intervention",
		"expected_effect",
		"resolution_strategy",
		"optimizer",
		"scheduler",
		"augmentation_policy",
		"class_balancing",
		"sampling_strategy",
		"strategy",
		"fine_tune_strategy",
	}, path, normalizations) || changed
	changed = prunePlannerEmptyFields(experiment, []string{"evidence_used"}, path, normalizations) || changed
	changed = prunePlannerEmptyFields(experiment, []string{
		"image_size",
		"weight_decay",
		"dropout",
		"optimizer_momentum",
		"scheduler_step_size",
		"scheduler_gamma",
		"label_smoothing",
		"gradient_clip_norm",
		"early_stopping_patience",
		"pretrained",
		"freeze_backbone",
	}, path, normalizations) || changed

	if preprocessing, ok := experiment["preprocessing"].(map[string]any); ok {
		prunePlannerEmptyFields(preprocessing, []string{"resize_strategy", "normalization", "crop_strategy", "bbox_mode", "use_dataset_normalization"}, path+".preprocessing", normalizations)
		if plannerMapHasOnlyEmptyValues(preprocessing) {
			delete(experiment, "preprocessing")
			changed = true
			*normalizations = append(*normalizations, path+".preprocessing empty optional object omitted")
		}
	}
	if config, ok := experiment["augmentation_policy_config"].(map[string]any); ok {
		prunePlannerEmptyFields(config, []string{"policy_type", "magnitude", "num_ops", "num_magnitude_bins", "probability", "alpha"}, path+".augmentation_policy_config", normalizations)
		if plannerMapHasOnlyEmptyValues(config) {
			delete(experiment, "augmentation_policy_config")
			changed = true
			*normalizations = append(*normalizations, path+".augmentation_policy_config empty optional object omitted")
		}
	}
	for _, field := range []string{"augmentation", "class_balancing_config"} {
		if config, ok := experiment[field].(map[string]any); ok && plannerMapHasOnlyEmptyValues(config) {
			delete(experiment, field)
			changed = true
			*normalizations = append(*normalizations, path+"."+field+" empty optional object omitted")
		}
	}
	if changed {
		*normalizations = append(*normalizations, path+" empty optional experiment fields omitted")
	}
}

func prunePlannerEmptyProposedChanges(changes map[string]any, normalizations *[]string, path string) {
	if changes == nil {
		return
	}
	changed := false
	for key, value := range changes {
		if strings.TrimSpace(key) == "" || plannerValueIsEmpty(value) {
			delete(changes, key)
			changed = true
		}
	}
	if changed {
		*normalizations = append(*normalizations, path+" empty entries omitted")
	}
}

func prunePlannerEmptyFields(root map[string]any, fields []string, path string, normalizations *[]string) bool {
	changed := false
	for _, field := range fields {
		value, ok := root[field]
		if !ok || !plannerValueIsEmpty(value) {
			continue
		}
		delete(root, field)
		changed = true
		*normalizations = append(*normalizations, path+"."+field+" empty optional field omitted")
	}
	return changed
}

func plannerMapHasOnlyEmptyValues(values map[string]any) bool {
	if len(values) == 0 {
		return true
	}
	for _, value := range values {
		if !plannerValueIsEmpty(value) {
			return false
		}
	}
	return true
}

func plannerValueIsEmpty(value any) bool {
	switch typed := value.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(typed) == ""
	case []string:
		return len(nonEmptyStrings(typed)) == 0
	case []any:
		for _, item := range typed {
			if !plannerValueIsEmpty(item) {
				return false
			}
		}
		return true
	case map[string]any:
		return plannerMapHasOnlyEmptyValues(typed)
	default:
		return false
	}
}
