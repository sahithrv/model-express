package api

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func (s *Server) getAutoMLCapabilities(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"enabled":                   s.currentAutomationSettings().AutoMLEnabled,
		"default_sampler":           s.currentAutomationSettings().AutoMLSampler,
		"capabilities":              automl.DefaultCapabilityRegistry().Capabilities(),
		"strategy_fields_forbidden": []string{"model", "template", "preprocessing", "resolution_strategy", "image_size", "augmentation_policy", "augmentation_policy_config.policy_type", "class_balancing", "sampling_strategy", "pretrained", "freeze_backbone", "fine_tune_strategy"},
		"scheduling_authority":      false,
	})
}

func (s *Server) prepareAutoMLExperiments(experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string, error) {
	return s.prepareAutoMLExperimentsForProject("", experiments)
}

func (s *Server) prepareAutoMLExperimentsForProject(projectID string, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string, error) {
	out := append([]plans.PlannedExperiment(nil), experiments...)
	warnings := []string{}
	automationSettings := s.currentAutomationSettings()
	history := []automl.OptimizerTrial{}
	if strings.TrimSpace(projectID) != "" && automationSettings.AutoMLEnabled {
		var err error
		history, err = s.store.ListProjectOptimizerTrials(projectID, 200)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML history failed for project %s: %v", projectID, err)
			history = nil
		}
	}
	for index := range out {
		if out[index].AutoML == nil && automationSettings.AutoMLEnabled && !strings.EqualFold(strings.TrimSpace(out[index].Template), jobs.TemplateLabelQualityAudit) {
			out[index].AutoML = defaultBackendAutoMLForExperiment(out[index], automationSettings.AutoMLSampler)
			warnings = append(warnings, fmt.Sprintf("AutoML auto-enabled for experiment %d; backend sampled concrete hyperparameters inside the LLM-selected strategy.", index))
		}
		if out[index].AutoML == nil || !out[index].AutoML.Enabled {
			continue
		}
		if !automationSettings.AutoMLEnabled {
			out[index].AutoML.ValidationStatus = "disabled"
			out[index].AutoML.Enabled = false
			warnings = append(warnings, fmt.Sprintf("AutoML disabled for experiment %d; using LLM-provided concrete hyperparameters.", index))
			continue
		}
		prepared, err := prepareAutoMLExperimentWithHistory(out[index], index, automationSettings.AutoMLSampler, history)
		if err != nil {
			return nil, warnings, err
		}
		out[index] = prepared
		warnings = append(warnings, fmt.Sprintf("AutoML sampled %d hyperparameter(s) for experiment %d using %s.", len(prepared.AutoML.Suggestion.Values), index, prepared.AutoML.Sampler))
	}
	return out, warnings, nil
}

func defaultBackendAutoMLForExperiment(experiment plans.PlannedExperiment, defaultSampler string) *automl.ExperimentAutoML {
	searchSpace := defaultBackendAutoMLSearchSpace(experiment)
	return &automl.ExperimentAutoML{
		Enabled:     true,
		Sampler:     normalizeAutoMLSampler(defaultSampler),
		SearchSpace: &searchSpace,
		Intent: automl.ExperimentIntent{
			Summary:           "Backend default AutoML samples concrete hyperparameters only; LLM-owned strategy fields are frozen.",
			PlanningMode:      strings.TrimSpace(experiment.Strategy),
			ExplorationIntent: "backend_default_hyperparameter_sampling",
			Goals: []string{
				"sample executable hyperparameters inside the validated experiment strategy",
				"preserve LLM-selected model, preprocessing, augmentation policy, class balancing, and fine-tuning choices",
			},
			AllowedParameters:   autoMLSearchSpaceParameterNames(searchSpace),
			StrategyDescription: strings.TrimSpace(experiment.Reason),
		},
	}
}

func defaultBackendAutoMLSearchSpace(experiment plans.PlannedExperiment) automl.HyperparameterSearchSpace {
	params := []automl.HyperparameterParameterSpec{}

	lrMin := 1e-6
	lrMax := 1e-2
	lrNotes := "broad log-scale learning-rate search for Adam-like optimizers"
	if strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		lrMin = 1e-5
		lrMax = 1e-1
		lrNotes = "broad log-scale learning-rate search for SGD"
	}
	params = append(params, autoMLFloatSpec("learning_rate", lrMin, lrMax, automl.SearchScaleLog, lrNotes))

	params = append(params, autoMLFloatSpec("weight_decay", 0, 0.3, automl.SearchScaleLinear, "broad regularization strength"))

	params = append(params, autoMLIntChoicesSpec("batch_size", defaultAutoMLBatchSizeChoices(experiment.BatchSize), "full worker-supported batch-size sweep"))

	epochBase := experiment.Epochs
	if epochBase < 3 {
		epochBase = 10
	}
	epochMin := 3
	epochMax := minInt(80, maxInt(24, epochBase*3))
	if epochMax < epochMin {
		epochMax = epochMin
	}
	params = append(params, autoMLIntRangeSpec("epochs", epochMin, epochMax, "broad training budget with early stopping"))
	params = append(params, autoMLIntChoicesSpec("early_stopping_patience", defaultAutoMLPatienceChoices(epochMax), "early-stopping patience sweep"))

	params = append(params, autoMLFloatSpec("dropout", 0, 0.7, automl.SearchScaleLinear, "full head regularization range"))
	params = append(params, autoMLFloatSpec("label_smoothing", 0, 0.3, automl.SearchScaleLinear, "classification regularization"))
	params = append(params, autoMLFloatSpec("gradient_clip_norm", 0, 10, automl.SearchScaleLinear, "gradient stability"))

	if strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		params = append(params, autoMLFloatSpec("optimizer_momentum", 0, 0.99, automl.SearchScaleLinear, "SGD momentum"))
	}
	if strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		stepMax := minInt(100, maxInt(2, epochMax))
		params = append(params, autoMLIntRangeSpec("scheduler_step_size", 1, stepMax, "step scheduler cadence"))
		params = append(params, autoMLFloatSpec("scheduler_gamma", 0.05, 0.95, automl.SearchScaleLinear, "step scheduler decay"))
	}

	if experiment.AugmentationPolicyConfig != nil {
		switch strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType)) {
		case "randaugment":
			params = append(params,
				autoMLIntRangeSpec("augmentation_policy_config.magnitude", 0, 15, "numeric strength for the LLM-selected randaugment policy"),
				autoMLIntRangeSpec("augmentation_policy_config.num_ops", 1, 3, "operation count for the LLM-selected randaugment policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"),
			)
		case "trivialaugment", "trivialaugmentwide":
			params = append(params,
				autoMLIntRangeSpec("augmentation_policy_config.num_magnitude_bins", 2, 31, "magnitude bins for the LLM-selected augmentation policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"),
			)
		case "autoaugment":
			params = append(params, autoMLFloatSpec("augmentation_policy_config.probability", 0, 1, automl.SearchScaleLinear, "application probability for the LLM-selected augmentation policy"))
		case "mixup", "cutmix":
			params = append(params,
				autoMLFloatSpec("augmentation_policy_config.alpha", 0, 1, automl.SearchScaleLinear, "mixing strength for the LLM-selected mixed-sample policy"),
				autoMLFloatSpec("augmentation_policy_config.probability", 0, 1, automl.SearchScaleLinear, "application probability for the LLM-selected mixed-sample policy"),
			)
		}
	}

	if effectiveNumberClassBalancing(experiment.ClassBalancing) {
		params = append(params, autoMLFloatSpec("class_balancing_config.effective_number_beta", 0.9, 0.99999, automl.SearchScaleLinear, "effective-number loss beta"))
	}
	if strings.EqualFold(strings.TrimSpace(experiment.ClassBalancing), "focal_loss") {
		params = append(params, autoMLFloatSpec("class_balancing_config.focal_loss_gamma", 0.5, 5, automl.SearchScaleLinear, "focal loss gamma"))
	}

	return automl.HyperparameterSearchSpace{Parameters: params}
}

func autoMLSearchSpaceParameterNames(space automl.HyperparameterSearchSpace) []string {
	names := make([]string, 0, len(space.Parameters))
	for _, spec := range space.Parameters {
		names = append(names, spec.Name)
	}
	return names
}

func autoMLFloatSpec(name string, minValue float64, maxValue float64, scale automl.SearchScale, notes string) automl.HyperparameterParameterSpec {
	return automl.HyperparameterParameterSpec{
		Name:   name,
		Type:   automl.ParameterFloat,
		Min:    &minValue,
		Max:    &maxValue,
		Scale:  scale,
		Source: automl.ProvenanceBackendDefault,
		Notes:  notes,
	}
}

func autoMLIntRangeSpec(name string, minValue int, maxValue int, notes string) automl.HyperparameterParameterSpec {
	minFloatValue := float64(minValue)
	maxFloatValue := float64(maxValue)
	return automl.HyperparameterParameterSpec{
		Name:   name,
		Type:   automl.ParameterInteger,
		Min:    &minFloatValue,
		Max:    &maxFloatValue,
		Source: automl.ProvenanceBackendDefault,
		Notes:  notes,
	}
}

func autoMLIntChoicesSpec(name string, choices []int, notes string) automl.HyperparameterParameterSpec {
	return automl.HyperparameterParameterSpec{
		Name:       name,
		Type:       automl.ParameterInteger,
		IntChoices: append([]int(nil), choices...),
		Source:     automl.ProvenanceBackendDefault,
		Notes:      notes,
	}
}

func defaultAutoMLBatchSizeChoices(current int) []int {
	return []int{4, 8, 16, 32, 64, 128}
}

func defaultAutoMLPatienceChoices(maxEpochs int) []int {
	limit := minInt(50, maxInt(0, maxEpochs/2))
	candidates := []int{0, 2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 50}
	choices := make([]int, 0, len(candidates))
	for _, value := range candidates {
		if value <= limit {
			choices = append(choices, value)
		}
	}
	if len(choices) == 0 {
		return []int{0}
	}
	return choices
}

func prepareAutoMLExperiment(experiment plans.PlannedExperiment, index int, defaultSampler string) (plans.PlannedExperiment, error) {
	return prepareAutoMLExperimentWithHistory(experiment, index, defaultSampler, nil)
}

func prepareAutoMLExperimentWithHistory(experiment plans.PlannedExperiment, index int, defaultSampler string, history []automl.OptimizerTrial) (plans.PlannedExperiment, error) {
	if experiment.AutoML == nil || !experiment.AutoML.Enabled {
		return experiment, nil
	}
	strategy := automlStrategyContext(experiment)
	searchSpace := experiment.AutoML.SearchSpace
	if searchSpace == nil || len(searchSpace.Parameters) == 0 {
		defaultSpace, err := automl.DefaultSearchSpace(experiment.AutoML.Intent.AllowedParameters, strategy)
		if err != nil {
			return experiment, fmt.Errorf("%w: proposed experiment %d AutoML intent is invalid: %s", store.ErrInvalidRequest, index, err.Error())
		}
		searchSpace = &defaultSpace
		experiment.AutoML.SearchSpace = searchSpace
	}
	if err := automl.ValidateSearchSpace(*searchSpace, strategy); err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML search space is invalid: %s", store.ErrInvalidRequest, index, err.Error())
	}
	samplerName := strings.TrimSpace(experiment.AutoML.Sampler)
	if samplerName == "" {
		samplerName = defaultSampler
	}
	samplerName = normalizeAutoMLSampler(samplerName)
	sampler, err := automl.NewSampler(samplerName)
	if err != nil {
		return experiment, fmt.Errorf("%w: proposed experiment %d %s", store.ErrInvalidRequest, index, err.Error())
	}
	seed := experiment.AutoML.Seed
	if seed == 0 {
		seed = automlSeedForExperiment(experiment, index)
	}
	suggestion, err := sampler.Suggest(context.Background(), automl.SuggestRequest{
		SearchSpace:     *searchSpace,
		StrategyContext: strategy,
		History:         history,
		Seed:            seed,
	})
	if err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML suggestion failed: %s", store.ErrInvalidRequest, index, err.Error())
	}
	for _, spec := range searchSpace.Parameters {
		if _, ok := suggestion.Values[spec.Name]; ok {
			continue
		}
		if _, ok := suggestion.Values[automl.NormalizeParameterName(spec.Name)]; ok {
			continue
		}
		clearAutoMLValueFromExperiment(&experiment, spec.Name)
	}
	for name, value := range suggestion.Values {
		if err := applyAutoMLValueToExperiment(&experiment, name, value); err != nil {
			experiment.AutoML.ValidationStatus = "invalid"
			experiment.AutoML.ValidationErrors = []string{err.Error()}
			return experiment, fmt.Errorf("%w: proposed experiment %d %s", store.ErrInvalidRequest, index, err.Error())
		}
	}
	if err := automl.ValidateSuggestion(suggestion, *searchSpace, automlStrategyContext(experiment)); err != nil {
		experiment.AutoML.ValidationStatus = "invalid"
		experiment.AutoML.ValidationErrors = []string{err.Error()}
		return experiment, fmt.Errorf("%w: proposed experiment %d AutoML suggestion is invalid: %s", store.ErrInvalidRequest, index, err.Error())
	}
	finalValues, provenance := autoMLFinalValues(experiment, suggestion.Provenance)
	suggestion.FinalValues = finalValues
	suggestion.ValidationStatus = "valid"
	experiment.AutoML.Sampler = samplerName
	experiment.AutoML.Seed = seed
	experiment.AutoML.Suggestion = &suggestion
	experiment.AutoML.FinalValues = finalValues
	experiment.AutoML.ValueProvenance = provenance
	experiment.AutoML.StrategySnapshot = autoMLStrategySnapshot(experiment)
	experiment.AutoML.ValidationStatus = "valid"
	experiment.AutoML.ValidationErrors = []string{}
	return experiment, nil
}

func clearAutoMLValueFromExperiment(experiment *plans.PlannedExperiment, name string) {
	switch automl.NormalizeParameterName(name) {
	case "optimizer_momentum":
		experiment.OptimizerMomentum = 0
	case "scheduler_step_size":
		experiment.SchedulerStepSize = 0
	case "scheduler_gamma":
		experiment.SchedulerGamma = 0
	}
}

func applyAutoMLValueToExperiment(experiment *plans.PlannedExperiment, name string, value any) error {
	switch automl.NormalizeParameterName(name) {
	case "learning_rate":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl learning_rate must be numeric")
		}
		experiment.LearningRate = number
	case "weight_decay":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl weight_decay must be numeric")
		}
		experiment.WeightDecay = number
	case "dropout":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl dropout must be numeric")
		}
		experiment.Dropout = number
	case "optimizer_momentum":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl optimizer_momentum must be numeric")
		}
		experiment.OptimizerMomentum = number
	case "scheduler_step_size":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl scheduler_step_size must be an integer")
		}
		experiment.SchedulerStepSize = integer
	case "scheduler_gamma":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl scheduler_gamma must be numeric")
		}
		experiment.SchedulerGamma = number
	case "label_smoothing":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl label_smoothing must be numeric")
		}
		experiment.LabelSmoothing = number
	case "gradient_clip_norm":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl gradient_clip_norm must be numeric")
		}
		experiment.GradientClipNorm = number
	case "batch_size":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl batch_size must be an integer")
		}
		experiment.BatchSize = integer
	case "epochs":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl epochs must be an integer")
		}
		experiment.Epochs = integer
	case "early_stopping_patience":
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl early_stopping_patience must be an integer")
		}
		experiment.EarlyStoppingPatience = integer
	case "optimizer":
		experiment.Optimizer = strings.ToLower(strings.TrimSpace(fmt.Sprint(value)))
	case "scheduler":
		experiment.Scheduler = strings.ToLower(strings.TrimSpace(fmt.Sprint(value)))
	case "augmentation_policy_config.magnitude":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.magnitude must be an integer")
		}
		config.Magnitude = integer
	case "augmentation_policy_config.num_ops":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.num_ops must be an integer")
		}
		config.NumOps = integer
	case "augmentation_policy_config.num_magnitude_bins":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		integer, ok := automl.IntValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.num_magnitude_bins must be an integer")
		}
		config.NumMagnitudeBins = integer
	case "augmentation_policy_config.probability":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.probability must be numeric")
		}
		config.Probability = number
	case "augmentation_policy_config.alpha":
		config, err := requireAutoMLAugmentationConfig(experiment)
		if err != nil {
			return err
		}
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl augmentation_policy_config.alpha must be numeric")
		}
		config.Alpha = number
	case "class_balancing_config.effective_number_beta":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl class_balancing_config.effective_number_beta must be numeric")
		}
		if experiment.ClassBalancingConfig == nil {
			experiment.ClassBalancingConfig = map[string]any{}
		}
		experiment.ClassBalancingConfig["effective_number_beta"] = number
	case "class_balancing_config.focal_loss_gamma":
		number, ok := automl.NumberValue(value)
		if !ok {
			return fmt.Errorf("automl class_balancing_config.focal_loss_gamma must be numeric")
		}
		if experiment.ClassBalancingConfig == nil {
			experiment.ClassBalancingConfig = map[string]any{}
		}
		experiment.ClassBalancingConfig["focal_loss_gamma"] = number
	default:
		return fmt.Errorf("automl cannot apply unsupported parameter %q", name)
	}
	return nil
}

func requireAutoMLAugmentationConfig(experiment *plans.PlannedExperiment) (*plans.AugmentationPolicyConfig, error) {
	if experiment.AugmentationPolicyConfig == nil || strings.TrimSpace(experiment.AugmentationPolicyConfig.PolicyType) == "" {
		return nil, fmt.Errorf("automl structured augmentation parameters require LLM-selected augmentation_policy_config.policy_type")
	}
	return experiment.AugmentationPolicyConfig, nil
}

func automlStrategyContext(experiment plans.PlannedExperiment) automl.StrategyContext {
	policyType := ""
	if experiment.AugmentationPolicyConfig != nil {
		policyType = experiment.AugmentationPolicyConfig.PolicyType
	}
	return automl.StrategyContext{
		Model:                  experiment.Model,
		Template:               experiment.Template,
		Optimizer:              experiment.Optimizer,
		Scheduler:              experiment.Scheduler,
		AugmentationPolicy:     experiment.AugmentationPolicy,
		AugmentationPolicyType: policyType,
		ClassBalancing:         experiment.ClassBalancing,
	}
}

func autoMLStrategySnapshot(experiment plans.PlannedExperiment) map[string]any {
	snapshot := map[string]any{
		"template":            experiment.Template,
		"model":               experiment.Model,
		"mechanism":           experiment.Mechanism,
		"intervention":        experiment.Intervention,
		"image_size":          experiment.ImageSize,
		"resolution_strategy": experiment.ResolutionStrategy,
		"augmentation_policy": experiment.AugmentationPolicy,
		"class_balancing":     experiment.ClassBalancing,
		"sampling_strategy":   experiment.SamplingStrategy,
		"pretrained":          experiment.Pretrained,
		"freeze_backbone":     experiment.FreezeBackbone,
		"fine_tune_strategy":  experiment.FineTuneStrategy,
	}
	if experiment.Preprocessing != nil {
		snapshot["preprocessing"] = experiment.Preprocessing
	}
	if experiment.AugmentationPolicyConfig != nil {
		snapshot["augmentation_policy_config_policy_type"] = experiment.AugmentationPolicyConfig.PolicyType
	}
	return compactNonEmptyMap(snapshot)
}

func autoMLFinalValues(experiment plans.PlannedExperiment, sampled map[string]automl.HyperparameterProvenance) (map[string]any, map[string]automl.HyperparameterProvenance) {
	values := map[string]any{}
	provenance := map[string]automl.HyperparameterProvenance{}
	for _, capability := range automl.DefaultCapabilityRegistry().Capabilities() {
		value, ok := autoMLParameterValue(experiment, capability.Name)
		if !ok {
			continue
		}
		values[capability.Name] = value
		if source, ok := sampled[capability.Name]; ok {
			provenance[capability.Name] = source
		} else {
			provenance[capability.Name] = automl.ProvenanceLLM
		}
	}
	return values, provenance
}

func autoMLParameterValue(experiment plans.PlannedExperiment, name string) (any, bool) {
	switch automl.NormalizeParameterName(name) {
	case "learning_rate":
		return experiment.LearningRate, experiment.LearningRate > 0
	case "weight_decay":
		return experiment.WeightDecay, true
	case "dropout":
		return experiment.Dropout, experiment.Dropout > 0
	case "optimizer_momentum":
		return experiment.OptimizerMomentum, experiment.OptimizerMomentum > 0
	case "scheduler_step_size":
		return experiment.SchedulerStepSize, experiment.SchedulerStepSize > 0
	case "scheduler_gamma":
		return experiment.SchedulerGamma, experiment.SchedulerGamma > 0
	case "label_smoothing":
		return experiment.LabelSmoothing, experiment.LabelSmoothing > 0
	case "gradient_clip_norm":
		return experiment.GradientClipNorm, experiment.GradientClipNorm > 0
	case "batch_size":
		return experiment.BatchSize, experiment.BatchSize > 0
	case "epochs":
		return experiment.Epochs, experiment.Epochs > 0
	case "early_stopping_patience":
		return experiment.EarlyStoppingPatience, experiment.EarlyStoppingPatience >= 0
	case "optimizer":
		if strings.TrimSpace(experiment.Optimizer) == "" {
			return "adamw", true
		}
		return experiment.Optimizer, true
	case "scheduler":
		if strings.TrimSpace(experiment.Scheduler) == "" {
			return "none", true
		}
		return experiment.Scheduler, true
	case "augmentation_policy_config.magnitude":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Magnitude, true
	case "augmentation_policy_config.num_ops":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.NumOps, true
	case "augmentation_policy_config.num_magnitude_bins":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.NumMagnitudeBins, true
	case "augmentation_policy_config.probability":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Probability, true
	case "augmentation_policy_config.alpha":
		if experiment.AugmentationPolicyConfig == nil {
			return nil, false
		}
		return experiment.AugmentationPolicyConfig.Alpha, true
	case "class_balancing_config.effective_number_beta":
		if experiment.ClassBalancingConfig == nil {
			return nil, false
		}
		value, ok := experiment.ClassBalancingConfig["effective_number_beta"]
		return value, ok
	case "class_balancing_config.focal_loss_gamma":
		if experiment.ClassBalancingConfig == nil {
			return nil, false
		}
		value, ok := experiment.ClassBalancingConfig["focal_loss_gamma"]
		return value, ok
	default:
		return nil, false
	}
}

func automlSeedForExperiment(experiment plans.PlannedExperiment, index int) int64 {
	blob, _ := json.Marshal(map[string]any{
		"index":              index,
		"strategy_snapshot":  autoMLStrategySnapshot(experiment),
		"search_space":       experiment.AutoML.SearchSpace,
		"allowed_parameters": experiment.AutoML.Intent.AllowedParameters,
		"sampler":            experiment.AutoML.Sampler,
	})
	sum := sha256.Sum256(blob)
	var seed int64
	for i := 0; i < 8; i++ {
		seed = (seed << 8) + int64(sum[i])
	}
	if seed < 0 {
		seed = -seed
	}
	if seed == 0 {
		seed = int64(index + 1)
	}
	return seed
}

func compactNonEmptyMap(input map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range input {
		switch typed := value.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				out[key] = typed
			}
		case int:
			if typed != 0 {
				out[key] = typed
			}
		case bool:
			if typed {
				out[key] = typed
			}
		case nil:
		default:
			out[key] = value
		}
	}
	return out
}

func (s *Server) persistAutoMLForPlan(plan plans.ExperimentPlan) error {
	for index, experiment := range plan.Experiments {
		if experiment.AutoML == nil || !experiment.AutoML.Enabled || experiment.AutoML.Suggestion == nil {
			continue
		}
		study, err := s.store.CreateOptimizerStudy(automl.OptimizerStudy{
			ProjectID:        plan.ProjectID,
			PlanID:           plan.ID,
			DatasetID:        plan.DatasetID,
			SourceDecisionID: plan.SourceDecisionID,
			ExperimentIndex:  index,
			Model:            experiment.Model,
			Intent:           experiment.AutoML.Intent,
			Sampler:          experiment.AutoML.Sampler,
			Seed:             experiment.AutoML.Seed,
			SearchSpace:      *experiment.AutoML.SearchSpace,
			StrategySnapshot: experiment.AutoML.StrategySnapshot,
		})
		if err != nil {
			return err
		}
		suggestion := experiment.AutoML.Suggestion
		_, err = s.store.CreateOptimizerSuggestion(automl.OptimizerSuggestion{
			StudyID:          study.ID,
			ProjectID:        plan.ProjectID,
			PlanID:           plan.ID,
			DatasetID:        plan.DatasetID,
			ExperimentIndex:  index,
			Model:            experiment.Model,
			Sampler:          suggestion.Sampler,
			Seed:             suggestion.Seed,
			Values:           suggestion.Values,
			FinalValues:      suggestion.FinalValues,
			Provenance:       suggestion.Provenance,
			ValidationStatus: suggestion.ValidationStatus,
			ValidationErrors: suggestion.ValidationErrors,
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) automlSuggestionsByExperiment(planID string) map[int]automl.OptimizerSuggestion {
	suggestions, err := s.store.ListPlanOptimizerSuggestions(planID)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML suggestions failed for plan %s: %v", planID, err)
		}
		return map[int]automl.OptimizerSuggestion{}
	}
	out := map[int]automl.OptimizerSuggestion{}
	for _, suggestion := range suggestions {
		if _, exists := out[suggestion.ExperimentIndex]; exists {
			continue
		}
		out[suggestion.ExperimentIndex] = suggestion
	}
	return out
}

func automlJobSummary(experiment plans.PlannedExperiment, suggestion automl.OptimizerSuggestion) map[string]any {
	return map[string]any{
		"enabled":             true,
		"sampler":             suggestion.Sampler,
		"study_id":            suggestion.StudyID,
		"suggestion_id":       suggestion.ID,
		"seed":                suggestion.Seed,
		"values":              suggestion.Values,
		"final_values":        suggestion.FinalValues,
		"provenance":          suggestion.Provenance,
		"validation_status":   suggestion.ValidationStatus,
		"validation_errors":   suggestion.ValidationErrors,
		"strategy_snapshot":   autoMLStrategySnapshot(experiment),
		"llm_remains_planner": true,
	}
}

func (s *Server) observeAutoMLTrialForJob(job jobs.ExperimentJob) error {
	suggestionID := jobConfigString(job.Config, "automl_suggestion_id")
	if suggestionID == "" {
		return nil
	}
	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		return err
	}
	targetMetric := jobConfigString(job.Config, "target_metric")
	if targetMetric == "" {
		targetMetric = "macro_f1"
	}
	status := summary.Status
	if status == "" {
		status = job.Status
	}
	evaluation := runs.TrainingRunEvaluation{}
	if storedEvaluation, err := s.store.GetTrainingRunEvaluation(job.ID); err == nil {
		evaluation = storedEvaluation
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}
	targetMetricScore := plannerTargetMetricValue(targetMetric, summary, evaluation)
	score := targetMetricScore
	if evaluation.JobID != "" {
		score = holisticRunScore(targetMetric, summary, evaluation, agents.ProjectObjectiveContext{})
	}
	if status != jobs.StatusSucceeded {
		score = 0
	}
	metrics := map[string]any{
		"best_macro_f1":        summary.BestMacroF1,
		"best_accuracy":        summary.BestAccuracy,
		"target_metric_score":  targetMetricScore,
		"trial_score_basis":    "loss_heavy_deployment_readiness",
		"final_train_loss":     summary.FinalTrainLoss,
		"final_val_loss":       summary.FinalValLoss,
		"epochs_completed":     summary.EpochsCompleted,
		"runtime_seconds":      summary.RuntimeSeconds,
		"estimated_cost_usd":   summary.EstimatedCostUSD,
		"hyperparameters":      automlHyperparametersFromJob(job),
		"automl_summary":       job.Config["automl_summary"],
		"train_validation_gap": summary.FinalValLoss - summary.FinalTrainLoss,
	}
	if evaluation.JobID != "" {
		metrics["deployment_readiness_score"] = score
		if diagnostics, ok := evaluation.HolisticScores["training_diagnostics"]; ok {
			metrics["training_diagnostics"] = diagnostics
		}
		if gap, ok := evaluation.HolisticScores["train_validation_gap"]; ok {
			metrics["train_validation_gap"] = gap
		}
		if heldoutLoss := payloadFloat(evaluation.ObjectiveProfile, "heldout_test_loss"); heldoutLoss > 0 {
			metrics["heldout_test_loss"] = heldoutLoss
		}
	}
	_, err = s.store.UpsertOptimizerTrial(automl.OptimizerTrial{
		StudyID:      jobConfigString(job.Config, "automl_study_id"),
		SuggestionID: suggestionID,
		ProjectID:    job.ProjectID,
		PlanID:       jobConfigString(job.Config, "plan_id"),
		DatasetID:    jobConfigString(job.Config, "dataset_id"),
		JobID:        job.ID,
		Status:       status,
		TargetMetric: targetMetric,
		Score:        score,
		Metrics:      metrics,
		Error:        job.Error,
	})
	return err
}

func automlHyperparametersFromJob(job jobs.ExperimentJob) map[string]any {
	if summary, ok := job.Config["automl_summary"].(map[string]any); ok {
		if finalValues, ok := summary["final_values"].(map[string]any); ok {
			return finalValues
		}
	}
	out := map[string]any{}
	for _, key := range []string{
		"learning_rate",
		"weight_decay",
		"dropout",
		"optimizer_momentum",
		"scheduler_step_size",
		"scheduler_gamma",
		"label_smoothing",
		"gradient_clip_norm",
		"batch_size",
		"epochs",
		"early_stopping_patience",
		"optimizer",
		"scheduler",
		"augmentation_policy_config",
		"class_balancing_config",
	} {
		if value, ok := job.Config[key]; ok {
			out[key] = value
		}
	}
	return out
}

func (s *Server) optimizerFeedbackSummary(studyID string, targetMetric string) *automl.OptimizerFeedbackSummary {
	studyID = strings.TrimSpace(studyID)
	if studyID == "" {
		return nil
	}
	trials, err := s.store.ListStudyOptimizerTrials(studyID)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list AutoML trials failed for study %s: %v", studyID, err)
		}
		return nil
	}
	summary := automl.BuildFeedbackSummary(studyID, targetMetric, trials)
	return &summary
}

func (s *Server) optimizerFeedbackSummariesForProject(projectID string, targetMetric string) []automl.OptimizerFeedbackSummary {
	trials, err := s.store.ListProjectOptimizerTrials(projectID, 100)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Printf("list project AutoML trials failed for project %s: %v", projectID, err)
		}
		return nil
	}
	byStudy := map[string][]automl.OptimizerTrial{}
	for _, trial := range trials {
		if strings.TrimSpace(trial.StudyID) == "" {
			continue
		}
		byStudy[trial.StudyID] = append(byStudy[trial.StudyID], trial)
	}
	out := []automl.OptimizerFeedbackSummary{}
	for studyID, studyTrials := range byStudy {
		out = append(out, automl.BuildFeedbackSummary(studyID, targetMetric, studyTrials))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].TrialCount > out[j].TrialCount
	})
	if len(out) > 5 {
		out = out[:5]
	}
	return out
}
