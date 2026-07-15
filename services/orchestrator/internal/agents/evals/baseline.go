package evals

import (
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

const PlannerRubricBaselineSchemaVersionV1 = "planner_rubric_baseline_v1"

type PlannerCriticalTolerance struct {
	MaxScenarioRegressions      int `json:"max_scenario_regressions"`
	MaxUnexpectedMutationPasses int `json:"max_unexpected_mutation_passes"`
}

type PlannerQualityTolerance struct {
	MaxCorrectnessCheckRegressions int     `json:"max_correctness_check_regressions"`
	MinimumScenarioPassRate        float64 `json:"minimum_scenario_pass_rate"`
}

type PlannerEfficiencyTolerance struct {
	MaxResponseByteIncreasePerScenario int `json:"max_response_byte_increase_per_scenario"`
	MaxTotalResponseByteIncrease       int `json:"max_total_response_byte_increase"`
}

type PlannerBaselineTolerances struct {
	Critical   PlannerCriticalTolerance   `json:"critical"`
	Quality    PlannerQualityTolerance    `json:"quality"`
	Efficiency PlannerEfficiencyTolerance `json:"efficiency"`
}

type PlannerRubricBaseline struct {
	SchemaVersion string                        `json:"schema_version"`
	ReviewNote    string                        `json:"review_note"`
	Tolerances    PlannerBaselineTolerances     `json:"tolerances"`
	Artifact      PlannerRubricBaselineArtifact `json:"artifact"`
}

type PlannerRubricBaselineArtifact struct {
	SchemaVersion string                          `json:"schema_version"`
	Scenarios     []PlannerRubricBaselineScenario `json:"scenarios"`
	Summary       PlannerRubricSummary            `json:"summary"`
}

type PlannerRubricBaselineScenario struct {
	FixtureName        string                          `json:"fixture_name"`
	Description        string                          `json:"description"`
	Coverage           []string                        `json:"coverage"`
	TaskType           string                          `json:"task_type"`
	ResponseBytes      int                             `json:"response_bytes"`
	DecisionType       string                          `json:"decision_type"`
	BackendSchedulable bool                            `json:"backend_schedulable"`
	Passed             bool                            `json:"passed"`
	SafetyPassed       bool                            `json:"safety_passed"`
	CorrectnessChecks  int                             `json:"correctness_checks"`
	Mutations          []PlannerRubricBaselineMutation `json:"mutations,omitempty"`
}

type PlannerRubricBaselineMutation struct {
	Name     string `json:"name"`
	Detected bool   `json:"detected"`
}

type PlannerBaselineChange struct {
	FixtureName string `json:"fixture_name"`
	Dimension   string `json:"dimension"`
	Regression  bool   `json:"regression"`
	Explanation string `json:"explanation"`
}

type PlannerBaselineComparison struct {
	SchemaVersion            string                    `json:"schema_version"`
	Passed                   bool                      `json:"passed"`
	CriticalPassed           bool                      `json:"critical_passed"`
	QualityPassed            bool                      `json:"quality_passed"`
	EfficiencyPassed         bool                      `json:"efficiency_passed"`
	EfficiencyGated          bool                      `json:"efficiency_gated"`
	Tolerances               PlannerBaselineTolerances `json:"tolerances"`
	Changes                  []PlannerBaselineChange   `json:"changes"`
	Warnings                 []PlannerBaselineChange   `json:"warnings,omitempty"`
	CriticalRegressions      int                       `json:"critical_regressions"`
	QualityRegressions       int                       `json:"quality_regressions"`
	UnexpectedMutationPasses int                       `json:"unexpected_mutation_passes"`
	ResponseByteIncrease     int                       `json:"response_byte_increase"`
}

type PlannerBaselineGateOptions struct {
	GateEfficiency bool
}

//go:embed baselines/*.json
var plannerBaselineFS embed.FS

func DefaultPlannerBaselineTolerances() PlannerBaselineTolerances {
	return PlannerBaselineTolerances{
		Critical: PlannerCriticalTolerance{
			MaxScenarioRegressions:      0,
			MaxUnexpectedMutationPasses: 0,
		},
		Quality: PlannerQualityTolerance{
			MaxCorrectnessCheckRegressions: 0,
			MinimumScenarioPassRate:        1,
		},
		Efficiency: PlannerEfficiencyTolerance{
			MaxResponseByteIncreasePerScenario: 256,
			MaxTotalResponseByteIncrease:       2048,
		},
	}
}

func BuildPlannerRubricBaseline(artifact PlannerRubricArtifact) PlannerRubricBaseline {
	return PlannerRubricBaseline{
		SchemaVersion: PlannerRubricBaselineSchemaVersionV1,
		ReviewNote:    "Review every scenario-level change and its explanation before replacing this file.",
		Tolerances:    DefaultPlannerBaselineTolerances(),
		Artifact:      plannerRubricBaselineSnapshot(artifact),
	}
}

func LoadCheckedPlannerRubricBaseline() (PlannerRubricBaseline, error) {
	blob, err := plannerBaselineFS.ReadFile("baselines/planner_rubric_v1.json")
	if err != nil {
		return PlannerRubricBaseline{}, err
	}
	return decodePlannerRubricBaseline(blob)
}

func LoadPlannerRubricBaseline(path string) (PlannerRubricBaseline, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return PlannerRubricBaseline{}, err
	}
	return decodePlannerRubricBaseline(blob)
}

func decodePlannerRubricBaseline(blob []byte) (PlannerRubricBaseline, error) {
	var baseline PlannerRubricBaseline
	if err := json.Unmarshal(blob, &baseline); err != nil {
		return PlannerRubricBaseline{}, err
	}
	if baseline.SchemaVersion != PlannerRubricBaselineSchemaVersionV1 {
		return PlannerRubricBaseline{}, fmt.Errorf("unsupported planner rubric baseline schema %q", baseline.SchemaVersion)
	}
	return baseline, nil
}

func WritePlannerRubricBaseline(path string, artifact PlannerRubricArtifact) error {
	baseline := BuildPlannerRubricBaseline(artifact)
	blob, err := json.MarshalIndent(baseline, "", "  ")
	if err != nil {
		return err
	}
	blob = append(blob, '\n')
	return os.WriteFile(path, blob, 0o644)
}

func ComparePlannerRubricBaseline(baseline PlannerRubricBaseline, current PlannerRubricArtifact) PlannerBaselineComparison {
	return ComparePlannerRubricBaselineWithOptions(baseline, current, PlannerBaselineGateOptions{})
}

func ComparePlannerRubricBaselineWithOptions(baseline PlannerRubricBaseline, current PlannerRubricArtifact, options PlannerBaselineGateOptions) PlannerBaselineComparison {
	comparison := PlannerBaselineComparison{
		SchemaVersion:    PlannerRubricBaselineSchemaVersionV1,
		CriticalPassed:   true,
		QualityPassed:    true,
		EfficiencyPassed: true,
		EfficiencyGated:  options.GateEfficiency,
		Tolerances:       baseline.Tolerances,
	}
	baselineByName := plannerBaselineScenariosByName(baseline.Artifact.Scenarios)
	currentSnapshot := plannerRubricBaselineSnapshot(current)
	currentByName := plannerBaselineScenariosByName(currentSnapshot.Scenarios)
	names := make([]string, 0, len(baselineByName)+len(currentByName))
	seen := map[string]bool{}
	for name := range baselineByName {
		seen[name] = true
		names = append(names, name)
	}
	for name := range currentByName {
		if !seen[name] {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	for _, name := range names {
		before, hadBefore := baselineByName[name]
		after, hasAfter := currentByName[name]
		switch {
		case !hadBefore:
			comparison.addChange(name, "critical", true, fmt.Sprintf("New scenario requires baseline review: %s", after.Description))
			comparison.CriticalRegressions++
			continue
		case !hasAfter:
			comparison.addChange(name, "critical", true, fmt.Sprintf("Scenario was removed: %s", before.Description))
			comparison.CriticalRegressions++
			continue
		}
		description := firstNonEmptyBaselineText(after.Description, before.Description, name)
		if before.Passed && !after.Passed {
			comparison.addChange(name, "critical", true, fmt.Sprintf("Scenario no longer passes: %s", description))
			comparison.CriticalRegressions++
		}
		if before.BackendSchedulable && !after.BackendSchedulable {
			comparison.addChange(name, "critical", true, fmt.Sprintf("Backend schedulability regressed: %s", description))
			comparison.CriticalRegressions++
		}
		if before.DecisionType != after.DecisionType {
			comparison.addChange(name, "critical", true, fmt.Sprintf("Decision changed from %s to %s: %s", before.DecisionType, after.DecisionType, description))
			comparison.CriticalRegressions++
		}
		if before.SafetyPassed && !after.SafetyPassed {
			comparison.addChange(name, "critical", true, fmt.Sprintf("Safety check regressed: %s", description))
			comparison.CriticalRegressions++
		}
		beforeMutations := plannerMutationsByName(before.Mutations)
		for _, mutation := range after.Mutations {
			if previous, ok := beforeMutations[mutation.Name]; ok && previous.Detected && !mutation.Detected {
				comparison.addChange(name, "critical", true, fmt.Sprintf("Mutation %q is no longer detected: %s", mutation.Name, description))
				comparison.UnexpectedMutationPasses++
			}
		}
		if after.CorrectnessChecks < before.CorrectnessChecks {
			comparison.addChange(name, "quality", true, fmt.Sprintf("Correctness checks decreased from %d to %d: %s", before.CorrectnessChecks, after.CorrectnessChecks, description))
			comparison.QualityRegressions++
		}
		if delta := after.ResponseBytes - before.ResponseBytes; delta != 0 {
			regression := delta > baseline.Tolerances.Efficiency.MaxResponseByteIncreasePerScenario
			comparison.addChange(name, "efficiency", regression, fmt.Sprintf("Response bytes changed by %+d (%d to %d): %s", delta, before.ResponseBytes, after.ResponseBytes, description))
			if delta > 0 {
				comparison.ResponseByteIncrease += delta
			}
		}
	}

	passRate := 1.0
	if current.Summary.ScenarioCount > 0 {
		passRate = float64(current.Summary.PassedCount) / float64(current.Summary.ScenarioCount)
	}
	if passRate < baseline.Tolerances.Quality.MinimumScenarioPassRate {
		comparison.QualityRegressions++
		comparison.addChange("_aggregate", "quality", true, fmt.Sprintf("Scenario pass rate %.3f is below the %.3f floor.", passRate, baseline.Tolerances.Quality.MinimumScenarioPassRate))
	}
	comparison.CriticalPassed = comparison.CriticalRegressions <= baseline.Tolerances.Critical.MaxScenarioRegressions &&
		comparison.UnexpectedMutationPasses <= baseline.Tolerances.Critical.MaxUnexpectedMutationPasses
	comparison.QualityPassed = comparison.QualityRegressions <= baseline.Tolerances.Quality.MaxCorrectnessCheckRegressions
	comparison.EfficiencyPassed = comparison.ResponseByteIncrease <= baseline.Tolerances.Efficiency.MaxTotalResponseByteIncrease
	for _, change := range comparison.Changes {
		if change.Dimension == "efficiency" && change.Regression {
			comparison.EfficiencyPassed = false
			comparison.Warnings = append(comparison.Warnings, change)
		}
	}
	comparison.Passed = comparison.CriticalPassed && comparison.QualityPassed &&
		(!comparison.EfficiencyGated || comparison.EfficiencyPassed)
	return comparison
}

func (comparison *PlannerBaselineComparison) addChange(fixtureName, dimension string, regression bool, explanation string) {
	comparison.Changes = append(comparison.Changes, PlannerBaselineChange{
		FixtureName: fixtureName,
		Dimension:   dimension,
		Regression:  regression,
		Explanation: strings.TrimSpace(explanation),
	})
}

func plannerBaselineScenariosByName(scenarios []PlannerRubricBaselineScenario) map[string]PlannerRubricBaselineScenario {
	out := make(map[string]PlannerRubricBaselineScenario, len(scenarios))
	for _, scenario := range scenarios {
		out[scenario.FixtureName] = scenario
	}
	return out
}

func plannerMutationsByName(mutations []PlannerRubricBaselineMutation) map[string]PlannerRubricBaselineMutation {
	out := make(map[string]PlannerRubricBaselineMutation, len(mutations))
	for _, mutation := range mutations {
		out[mutation.Name] = mutation
	}
	return out
}

func plannerRubricBaselineSnapshot(artifact PlannerRubricArtifact) PlannerRubricBaselineArtifact {
	snapshot := PlannerRubricBaselineArtifact{
		SchemaVersion: artifact.SchemaVersion,
		Summary:       artifact.Summary,
	}
	for _, scenario := range artifact.Scenarios {
		baselineScenario := PlannerRubricBaselineScenario{
			FixtureName:        scenario.FixtureName,
			Description:        scenario.Description,
			Coverage:           append([]string(nil), scenario.Coverage...),
			TaskType:           scenario.TaskType,
			ResponseBytes:      scenario.ResponseBytes,
			DecisionType:       scenario.Score.DecisionType,
			BackendSchedulable: scenario.Score.BackendSchedulable,
			Passed:             scenario.Score.Passed,
			SafetyPassed:       scenario.Score.SafetyPassed,
			CorrectnessChecks:  scenario.Score.CorrectnessChecks,
		}
		for _, mutation := range scenario.Mutations {
			baselineScenario.Mutations = append(baselineScenario.Mutations, PlannerRubricBaselineMutation{
				Name:     mutation.Name,
				Detected: mutation.Detected,
			})
		}
		snapshot.Scenarios = append(snapshot.Scenarios, baselineScenario)
	}
	return snapshot
}

func firstNonEmptyBaselineText(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return "scenario changed"
}
