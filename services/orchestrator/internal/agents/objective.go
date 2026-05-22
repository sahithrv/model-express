package agents

type ProjectObjectiveContext struct {
	GoalText             string             `json:"goal_text"`
	PrimaryObjective     string             `json:"primary_objective"`
	MetricPreferences    []string           `json:"metric_preferences"`
	DeploymentPriorities []string           `json:"deployment_priorities"`
	Constraints          []string           `json:"constraints"`
	RankingWeights       map[string]float64 `json:"ranking_weights"`
}

type SupportedModelSpec struct {
	Name                  string   `json:"name"`
	Family                string   `json:"family"`
	DeploymentTier        string   `json:"deployment_tier"`
	DefaultImageSize      int      `json:"default_image_size"`
	MinRecommendedImages  int      `json:"min_recommended_images"`
	SupportsTransfer      bool     `json:"supports_transfer"`
	ExpectedLatencyClass  string   `json:"expected_latency_class"`
	RecommendedUse        string   `json:"recommended_use"`
	SupportsFineTuneModes []string `json:"supports_fine_tune_modes"`
}
