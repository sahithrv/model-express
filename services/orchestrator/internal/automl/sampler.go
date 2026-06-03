package automl

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

type SeededRandomSampler struct{}

func (s SeededRandomSampler) Suggest(_ context.Context, request SuggestRequest) (HyperparameterSuggestion, error) {
	if err := ValidateSearchSpace(request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	seed := request.Seed
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))
	values := map[string]any{}
	provenance := map[string]HyperparameterProvenance{}
	for _, spec := range orderedSpecs(request.SearchSpace.Parameters) {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if !conditionActive(parameterCondition(spec, capability), request.StrategyContext, values) {
			continue
		}
		value, err := sampleValue(rng, spec)
		if err != nil {
			return HyperparameterSuggestion{}, err
		}
		values[spec.Name] = value
		provenance[spec.Name] = ProvenanceRandomSearch
	}
	suggestion := HyperparameterSuggestion{
		StudyID:          request.StudyID,
		Sampler:          SamplerSeededRandom,
		Seed:             seed,
		Values:           values,
		Provenance:       provenance,
		ValidationStatus: "valid",
		CreatedAt:        time.Now().UTC(),
	}
	if err := ValidateSuggestion(suggestion, request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	return suggestion, nil
}

func (s SeededRandomSampler) Observe(context.Context, ObserveRequest) error {
	return nil
}

type GridSampler struct{}

func (s GridSampler) Suggest(_ context.Context, request SuggestRequest) (HyperparameterSuggestion, error) {
	if err := ValidateSearchSpace(request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	values := map[string]any{}
	provenance := map[string]HyperparameterProvenance{}
	index := len(request.History)
	for _, spec := range orderedSpecs(request.SearchSpace.Parameters) {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if !conditionActive(parameterCondition(spec, capability), request.StrategyContext, values) {
			continue
		}
		value, err := gridValue(index, spec)
		if err != nil {
			return HyperparameterSuggestion{}, err
		}
		values[spec.Name] = value
		provenance[spec.Name] = ProvenanceGridSearch
	}
	suggestion := HyperparameterSuggestion{
		StudyID:          request.StudyID,
		Sampler:          SamplerGrid,
		Seed:             request.Seed,
		Values:           values,
		Provenance:       provenance,
		ValidationStatus: "valid",
		CreatedAt:        time.Now().UTC(),
	}
	if err := ValidateSuggestion(suggestion, request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	return suggestion, nil
}

func (s GridSampler) Observe(context.Context, ObserveRequest) error {
	return nil
}

type AdaptiveBayesianSampler struct{}

func (s AdaptiveBayesianSampler) Suggest(_ context.Context, request SuggestRequest) (HyperparameterSuggestion, error) {
	if err := ValidateSearchSpace(request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	seed := request.Seed
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))
	bestValues := bestObservedHyperparameters(request.History)
	values := map[string]any{}
	provenance := map[string]HyperparameterProvenance{}
	for _, spec := range orderedSpecs(request.SearchSpace.Parameters) {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if !conditionActive(parameterCondition(spec, capability), request.StrategyContext, values) {
			continue
		}
		value, err := adaptiveValue(rng, spec, bestValues[NormalizeParameterName(spec.Name)])
		if err != nil {
			return HyperparameterSuggestion{}, err
		}
		values[spec.Name] = value
		provenance[spec.Name] = ProvenanceBayesianOptimizer
	}
	suggestion := HyperparameterSuggestion{
		StudyID:          request.StudyID,
		Sampler:          SamplerAdaptiveBayesian,
		Seed:             seed,
		Values:           values,
		Provenance:       provenance,
		ValidationStatus: "valid",
		CreatedAt:        time.Now().UTC(),
	}
	if err := ValidateSuggestion(suggestion, request.SearchSpace, request.StrategyContext); err != nil {
		return HyperparameterSuggestion{}, err
	}
	return suggestion, nil
}

func (s AdaptiveBayesianSampler) Observe(context.Context, ObserveRequest) error {
	return nil
}

func NewSampler(name string) (Optimizer, error) {
	switch NormalizeParameterName(name) {
	case "", SamplerSeededRandom, "random", "random_search":
		return SeededRandomSampler{}, nil
	case SamplerGrid, "grid_search":
		return GridSampler{}, nil
	case SamplerAdaptiveBayesian, "adaptive", "bayesian", "bayesian_optimizer":
		return AdaptiveBayesianSampler{}, nil
	default:
		return nil, fmt.Errorf("unsupported AutoML sampler %q", name)
	}
}

func sampleValue(rng *rand.Rand, spec HyperparameterParameterSpec) (any, error) {
	switch spec.Type {
	case ParameterFloat:
		minValue, maxValue := numericBounds(spec, HyperparameterCapability{})
		if spec.Min == nil || spec.Max == nil {
			capability, ok := DefaultCapabilityRegistry().Capability(spec.Name)
			if !ok {
				return nil, fmt.Errorf("unsupported AutoML parameter %q", spec.Name)
			}
			minValue, maxValue = numericBounds(spec, capability)
		}
		var value float64
		if spec.Scale == SearchScaleLog {
			value = math.Exp(math.Log(minValue) + rng.Float64()*(math.Log(maxValue)-math.Log(minValue)))
		} else {
			value = minValue + rng.Float64()*(maxValue-minValue)
		}
		return boundedFloatValue(value, minValue, maxValue, spec.Step), nil
	case ParameterInteger:
		if len(spec.IntChoices) > 0 {
			return spec.IntChoices[rng.Intn(len(spec.IntChoices))], nil
		}
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		minValue, maxValue := numericBounds(spec, capability)
		minInt := int(math.Ceil(minValue))
		maxInt := int(math.Floor(maxValue))
		if maxInt < minInt {
			return nil, fmt.Errorf("automl integer parameter %q has empty bounds", spec.Name)
		}
		return minInt + rng.Intn(maxInt-minInt+1), nil
	case ParameterCategorical:
		if len(spec.Choices) == 0 {
			return nil, fmt.Errorf("automl categorical parameter %q has no choices", spec.Name)
		}
		return spec.Choices[rng.Intn(len(spec.Choices))], nil
	default:
		return nil, fmt.Errorf("unsupported AutoML parameter type %q", spec.Type)
	}
}

func orderedSpecs(specs []HyperparameterParameterSpec) []HyperparameterParameterSpec {
	out := make([]HyperparameterParameterSpec, 0, len(specs))
	for _, spec := range specs {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if parameterCondition(spec, capability) == nil {
			out = append(out, spec)
		}
	}
	for _, spec := range specs {
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		if parameterCondition(spec, capability) != nil {
			out = append(out, spec)
		}
	}
	return out
}

func gridValue(index int, spec HyperparameterParameterSpec) (any, error) {
	switch spec.Type {
	case ParameterFloat:
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		minValue, maxValue := numericBounds(spec, capability)
		position := index % 5
		if spec.Scale == SearchScaleLog {
			if minValue <= 0 {
				return nil, fmt.Errorf("automl log parameter %q must have min > 0", spec.Name)
			}
			value := math.Exp(math.Log(minValue) + float64(position)/4*(math.Log(maxValue)-math.Log(minValue)))
			return boundedFloatValue(value, minValue, maxValue, spec.Step), nil
		}
		value := minValue + float64(position)/4*(maxValue-minValue)
		return boundedFloatValue(value, minValue, maxValue, spec.Step), nil
	case ParameterInteger:
		if len(spec.IntChoices) > 0 {
			return spec.IntChoices[index%len(spec.IntChoices)], nil
		}
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		minValue, maxValue := numericBounds(spec, capability)
		minInt := int(math.Ceil(minValue))
		maxInt := int(math.Floor(maxValue))
		if maxInt < minInt {
			return nil, fmt.Errorf("automl integer parameter %q has empty bounds", spec.Name)
		}
		return minInt + index%(maxInt-minInt+1), nil
	case ParameterCategorical:
		if len(spec.Choices) == 0 {
			return nil, fmt.Errorf("automl categorical parameter %q has no choices", spec.Name)
		}
		return spec.Choices[index%len(spec.Choices)], nil
	default:
		return nil, fmt.Errorf("unsupported AutoML parameter type %q", spec.Type)
	}
}

func roundStep(value float64, step *float64) float64 {
	if step == nil || *step <= 0 {
		return value
	}
	return math.Round(value / *step) * *step
}

func boundedFloatValue(value float64, minValue float64, maxValue float64, step *float64) float64 {
	return clampFloat(roundStep(value, step), minValue, maxValue)
}

func adaptiveValue(rng *rand.Rand, spec HyperparameterParameterSpec, best any) (any, error) {
	if best == nil {
		return sampleValue(rng, spec)
	}
	switch spec.Type {
	case ParameterFloat:
		bestNumber, ok := NumberValue(best)
		if !ok {
			return sampleValue(rng, spec)
		}
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		minValue, maxValue := numericBounds(spec, capability)
		if spec.Scale == SearchScaleLog {
			if minValue <= 0 || bestNumber <= 0 {
				return sampleValue(rng, spec)
			}
			logMin := math.Log(minValue)
			logMax := math.Log(maxValue)
			center := math.Log(bestNumber)
			radius := math.Max((logMax-logMin)*0.18, 0.05)
			value := math.Exp(clampFloat(center+(rng.Float64()*2-1)*radius, logMin, logMax))
			return boundedFloatValue(value, minValue, maxValue, spec.Step), nil
		}
		span := maxValue - minValue
		value := clampFloat(bestNumber+(rng.Float64()*2-1)*span*0.18, minValue, maxValue)
		return boundedFloatValue(value, minValue, maxValue, spec.Step), nil
	case ParameterInteger:
		bestInt, ok := IntValue(best)
		if !ok {
			return sampleValue(rng, spec)
		}
		if len(spec.IntChoices) > 0 {
			bestIndex := 0
			for index, choice := range spec.IntChoices {
				if choice == bestInt {
					bestIndex = index
					break
				}
			}
			minIndex := maxInt(0, bestIndex-1)
			maxIndex := minInt(len(spec.IntChoices)-1, bestIndex+1)
			return spec.IntChoices[minIndex+rng.Intn(maxIndex-minIndex+1)], nil
		}
		capability, _ := DefaultCapabilityRegistry().Capability(spec.Name)
		minValue, maxValue := numericBounds(spec, capability)
		minValueInt := int(math.Ceil(minValue))
		maxValueInt := int(math.Floor(maxValue))
		radius := maxInt(1, int(math.Round(float64(maxValueInt-minValueInt)*0.18)))
		return minInt(maxValueInt, maxInt(minValueInt, bestInt-radius+rng.Intn(radius*2+1))), nil
	case ParameterCategorical:
		bestText := fmt.Sprint(best)
		for _, choice := range spec.Choices {
			if NormalizeParameterName(choice) == NormalizeParameterName(bestText) && rng.Float64() < 0.75 {
				return choice, nil
			}
		}
		return sampleValue(rng, spec)
	default:
		return sampleValue(rng, spec)
	}
}

func bestObservedHyperparameters(history []OptimizerTrial) map[string]any {
	var best *OptimizerTrial
	for index := range history {
		trial := &history[index]
		if !strings.EqualFold(strings.TrimSpace(trial.Status), "SUCCEEDED") {
			continue
		}
		if best == nil || trial.Score > best.Score {
			best = trial
		}
	}
	if best == nil {
		return nil
	}
	values := map[string]any{}
	for key, value := range mapFromAny(best.Metrics["hyperparameters"]) {
		values[NormalizeParameterName(key)] = value
	}
	return values
}

func clampFloat(value float64, minValue float64, maxValue float64) float64 {
	return math.Max(minValue, math.Min(maxValue, value))
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}
