package automl

import "context"

const (
	SamplerSeededRandom     = "seeded_random"
	SamplerGrid             = "grid"
	SamplerAdaptiveBayesian = "adaptive_bayesian"
)

type SuggestRequest struct {
	StudyID         string
	SearchSpace     HyperparameterSearchSpace
	StrategyContext StrategyContext
	History         []OptimizerTrial
	Seed            int64
}

type ObserveRequest struct {
	Trial OptimizerTrial
}

type Optimizer interface {
	Suggest(context.Context, SuggestRequest) (HyperparameterSuggestion, error)
	Observe(context.Context, ObserveRequest) error
}
