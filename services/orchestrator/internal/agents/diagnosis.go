package agents

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const projectTrajectoryRecentWindow = 3

type PlannerDiagnosis struct {
	OverfittingScore           float64  `json:"overfitting_score"`
	UnderfittingScore          float64  `json:"underfitting_score"`
	PlateauScore               float64  `json:"plateau_score"`
	InstabilityScore           float64  `json:"instability_score"`
	ClassImbalanceScore        float64  `json:"class_imbalance_score"`
	MinorityClassFailureScore  float64  `json:"minority_class_failure_score"`
	GeneralizationGap          float64  `json:"generalization_gap"`
	BestMetricDeltaVsChampion  float64  `json:"best_metric_delta_vs_champion"`
	CostEfficiencyScore        float64  `json:"cost_efficiency_score"`
	LatencyPenalty             float64  `json:"latency_penalty"`
	ImprovementStagnationScore float64  `json:"improvement_stagnation_score"`
	RecommendedFailureModes    []string `json:"recommended_failure_modes"`
	DeterministicDiagnosisUsed []string `json:"deterministic_diagnosis_used"`
	Evidence                   []string `json:"evidence"`
}

type PlannerProjectTrajectoryCard struct {
	CompletedTrainingRuns  int                       `json:"completed_training_runs"`
	CompletedPlannerRounds int                       `json:"completed_planner_rounds"`
	FirstSuccessfulScore   float64                   `json:"first_successful_score"`
	CurrentChampionScore   float64                   `json:"current_champion_score"`
	AbsoluteChampionGain   float64                   `json:"absolute_champion_gain"`
	GainPerCompletedRun    float64                   `json:"gain_per_completed_run"`
	RecentBestDelta        float64                   `json:"recent_best_delta"`
	MinimumUsefulDelta     float64                   `json:"minimum_useful_delta"`
	NoImprovementRounds    int                       `json:"no_improvement_rounds"`
	DecisionPressure       string                    `json:"decision_pressure"`
	MechanismOutcomes      []PlannerMechanismOutcome `json:"mechanism_outcomes"`
	BlockedMechanisms      []string                  `json:"blocked_mechanisms"`
	Warnings               []string                  `json:"warnings"`
}

type PlannerMechanismOutcome struct {
	Mechanism           string   `json:"mechanism"`
	AttemptCount        int      `json:"attempt_count"`
	PlanCount           int      `json:"plan_count"`
	BestScore           float64  `json:"best_score"`
	BestDeltaVsPrior    float64  `json:"best_delta_vs_prior_champion"`
	RecentBestDelta     float64  `json:"recent_best_delta"`
	TotalCostUSD        float64  `json:"total_cost_usd"`
	TotalRuntimeSeconds float64  `json:"total_runtime_seconds"`
	Status              string   `json:"status"`
	ExhaustionReason    string   `json:"exhaustion_reason,omitempty"`
	AllowedNextOnlyWith []string `json:"allowed_next_only_with,omitempty"`
}

func ComputePlannerDiagnosis(input ExperimentPlannerInput) PlannerDiagnosis {
	targetMetric := normalizedDiagnosisMetric(input.SourcePlan.TargetMetric)
	bestScore := 0.0
	hasBest := false
	totalCost := 0.0
	totalRuntime := 0.0
	overfitScores := []float64{}
	underfitScores := []float64{}
	gaps := []float64{}
	plateauScores := []float64{}
	instabilityScores := []float64{}

	for _, summary := range input.PlanSummaries {
		score := diagnosisSummaryMetric(summary, targetMetric)
		if strings.EqualFold(summary.Status, jobs.StatusSucceeded) && (!hasBest || score > bestScore) {
			bestScore = score
			hasBest = true
		}
		totalCost += summary.EstimatedCostUSD
		totalRuntime += summary.RuntimeSeconds

		lossGap := summary.FinalValLoss - summary.FinalTrainLoss
		if summary.FinalValLoss > 0 || summary.FinalTrainLoss > 0 {
			gaps = append(gaps, lossGap)
			overfit := clamp01((lossGap - 0.08) / 0.45)
			if summary.FinalTrainLoss > 0 && summary.FinalTrainLoss < 0.35 && summary.FinalValLoss > 0.55 {
				overfit = maxDiagnosis(overfit, 0.75)
			}
			overfitScores = append(overfitScores, overfit)

			underfit := 0.0
			if summary.FinalTrainLoss > 0.65 && summary.FinalValLoss > 0.65 && score < 0.62 {
				underfit = clamp01((0.62 - score) / 0.45)
				underfit = maxDiagnosis(underfit, clamp01((summary.FinalTrainLoss-0.55)/0.65))
			}
			underfitScores = append(underfitScores, underfit)
		}

		if metrics := input.PlanMetrics[summary.JobID]; len(metrics) > 0 {
			plateauScores = append(plateauScores, diagnosisPlateauScore(metrics, targetMetric))
			instabilityScores = append(instabilityScores, diagnosisInstabilityScore(metrics, targetMetric))
		}
	}

	deltaVsChampion := 0.0
	if hasBest {
		if input.CurrentChampion != nil && input.CurrentChampion.JobID != "" {
			deltaVsChampion = bestScore - input.CurrentChampion.Score
		} else {
			deltaVsChampion = bestScore
		}
	}

	classImbalanceScore := clamp01((input.DatasetInsights.ImbalanceRatio - 1.0) / 4.0)
	minorityFailureScore, minorityEvidence := diagnosisMinorityFailure(input.PlanEvaluations)
	if minorityFailureScore == 0 && classImbalanceScore > 0.45 && hasBest {
		for _, summary := range input.PlanSummaries {
			if summary.BestAccuracy-summary.BestMacroF1 > 0.10 {
				minorityFailureScore = maxDiagnosis(minorityFailureScore, clamp01((summary.BestAccuracy-summary.BestMacroF1)/0.30))
			}
		}
	}

	latencyPenalty, latencyEvidence := diagnosisLatencyPenalty(input.PlanEvaluations, input.ObjectiveContext)
	costEfficiency := diagnosisCostEfficiency(deltaVsChampion, totalCost, totalRuntime)
	stagnationScore := maxDiagnosis(clamp01(float64(input.NoImprovementRounds)/3.0), averageScore(plateauScores)*0.85)

	diagnosis := PlannerDiagnosis{
		OverfittingScore:           roundDiagnosis(averageScore(overfitScores)),
		UnderfittingScore:          roundDiagnosis(averageScore(underfitScores)),
		PlateauScore:               roundDiagnosis(averageScore(plateauScores)),
		InstabilityScore:           roundDiagnosis(averageScore(instabilityScores)),
		ClassImbalanceScore:        roundDiagnosis(classImbalanceScore),
		MinorityClassFailureScore:  roundDiagnosis(minorityFailureScore),
		GeneralizationGap:          roundDiagnosis(averageScore(gaps)),
		BestMetricDeltaVsChampion:  roundDiagnosis(deltaVsChampion),
		CostEfficiencyScore:        roundDiagnosis(costEfficiency),
		LatencyPenalty:             roundDiagnosis(latencyPenalty),
		ImprovementStagnationScore: roundDiagnosis(stagnationScore),
	}

	diagnosis.RecommendedFailureModes = diagnosisFailureModes(diagnosis)
	diagnosis.DeterministicDiagnosisUsed = diagnosisSignals(diagnosis)
	diagnosis.Evidence = diagnosisEvidence(input, diagnosis, bestScore, totalCost, totalRuntime, minorityEvidence, latencyEvidence)
	return diagnosis
}

func ComputeProjectTrajectoryDiagnosis(input ExperimentPlannerInput) PlannerProjectTrajectoryCard {
	targetMetric := normalizedDiagnosisMetric(input.SourcePlan.TargetMetric)
	minimumUsefulDelta := math.Max(input.MinimumMeaningfulImprovement, 0.010)
	jobsByID := projectTrajectoryJobsByID(input)
	plansByID := projectTrajectoryPlansByID(input)
	runs := projectTrajectoryRuns(input, jobsByID, plansByID, targetMetric)
	successfulRuns := projectTrajectorySuccessfulRuns(runs)
	terminalRuns := 0
	for _, run := range runs {
		if projectTrajectoryTerminalStatus(run.Status) {
			terminalRuns++
		}
	}

	card := PlannerProjectTrajectoryCard{
		CompletedTrainingRuns:  terminalRuns,
		CompletedPlannerRounds: projectTrajectoryPlannerRoundCount(input, runs),
		MinimumUsefulDelta:     roundDiagnosis(minimumUsefulDelta),
		NoImprovementRounds:    input.NoImprovementRounds,
	}
	if len(successfulRuns) > 0 {
		card.FirstSuccessfulScore = roundDiagnosis(successfulRuns[0].Score)
		card.CurrentChampionScore = roundDiagnosis(projectTrajectoryChampionScore(input, successfulRuns))
		card.AbsoluteChampionGain = roundDiagnosis(card.CurrentChampionScore - card.FirstSuccessfulScore)
		if terminalRuns > 0 {
			card.GainPerCompletedRun = roundDiagnosis(card.AbsoluteChampionGain / float64(terminalRuns))
		}
		card.RecentBestDelta = roundDiagnosis(projectTrajectoryRecentBestDelta(successfulRuns, projectTrajectoryRecentWindow))
	}

	explicitlyBlocked := mechanismsFromRejectedOptions(input.RejectedStrategyMemory)
	outcomes := projectTrajectoryMechanismOutcomes(runs, plansByID, explicitlyBlocked)
	blocked := map[string]bool{}
	for index := range outcomes {
		status, reason := mechanismStatusFromOutcome(outcomes[index], card)
		if explicitlyBlocked[outcomes[index].Mechanism] {
			status = "blocked"
			reason = "Rejected strategy memory blocks repeating this mechanism."
		}
		outcomes[index].Status = status
		if status == "exhausted" || status == "blocked" {
			blocked[outcomes[index].Mechanism] = true
			outcomes[index].ExhaustionReason = reason
			outcomes[index].AllowedNextOnlyWith = projectTrajectoryAllowedNextOnlyWith(outcomes[index].Mechanism)
		}
	}
	card.MechanismOutcomes = outcomes
	card.BlockedMechanisms = capSortedMapKeys(blocked, plannerSnapshotMaxMechanisms)
	card.DecisionPressure = computeDecisionPressure(card)
	card.Warnings = projectTrajectoryWarnings(card, input, explicitlyBlocked)
	return card
}

func mechanismStatusFromOutcome(outcome PlannerMechanismOutcome, trajectory PlannerProjectTrajectoryCard) (string, string) {
	if outcome.AttemptCount == 0 {
		return "unexplored", ""
	}
	usefulDelta := math.Max(trajectory.MinimumUsefulDelta, 0.010)
	if outcome.Mechanism == "architecture_challenge" && outcome.AttemptCount >= 8 && outcome.RecentBestDelta < usefulDelta {
		return "exhausted", "Repeated architecture/backbone attempts produced low recent champion uplift."
	}
	if outcome.AttemptCount >= 6 && outcome.RecentBestDelta < usefulDelta && outcome.BestDeltaVsPrior < 0.015 {
		return "exhausted", "Repeated attempts produced low recent champion uplift."
	}
	if outcome.BestDeltaVsPrior >= usefulDelta {
		return "promising", "Mechanism has produced useful champion uplift."
	}
	return "active", "Mechanism is not yet exhausted."
}

func computeDecisionPressure(card PlannerProjectTrajectoryCard) string {
	if card.CompletedTrainingRuns >= 15 && card.GainPerCompletedRun > 0 && card.GainPerCompletedRun < 0.003 {
		return "champion_confirmation_or_non_architecture_pivot"
	}
	if card.NoImprovementRounds >= 2 {
		return "non_exhausted_mechanism_or_stop"
	}
	return "normal"
}

type projectTrajectoryRun struct {
	JobID         string
	PlanID        string
	Model         string
	Mechanism     string
	Status        string
	Score         float64
	CostUSD       float64
	RuntimeSecs   float64
	CompletedAt   time.Time
	CreatedAt     time.Time
	InputPosition int
}

type projectTrajectoryMechanismAccumulator struct {
	outcome           PlannerMechanismOutcome
	planIDs           map[string]bool
	bestDeltaObserved bool
	recentDeltas      []float64
}

func projectTrajectoryJobsByID(input ExperimentPlannerInput) map[string]jobs.ExperimentJob {
	out := map[string]jobs.ExperimentJob{}
	for _, job := range append(append([]jobs.ExperimentJob(nil), input.PriorJobs...), input.PlanJobs...) {
		if strings.TrimSpace(job.ID) == "" {
			continue
		}
		out[job.ID] = job
	}
	return out
}

func projectTrajectoryPlansByID(input ExperimentPlannerInput) map[string]plans.ExperimentPlan {
	out := map[string]plans.ExperimentPlan{}
	for _, plan := range input.PriorPlans {
		if strings.TrimSpace(plan.ID) == "" {
			continue
		}
		out[plan.ID] = plan
	}
	if strings.TrimSpace(input.SourcePlan.ID) != "" {
		out[input.SourcePlan.ID] = input.SourcePlan
	}
	return out
}

func projectTrajectoryRuns(input ExperimentPlannerInput, jobsByID map[string]jobs.ExperimentJob, plansByID map[string]plans.ExperimentPlan, targetMetric string) []projectTrajectoryRun {
	summaries := append(append([]runs.TrainingRunSummary(nil), input.PriorSummaries...), input.PlanSummaries...)
	byJobID := map[string]projectTrajectoryRun{}
	anonymous := []projectTrajectoryRun{}
	for index, summary := range summaries {
		job := jobsByID[summary.JobID]
		planID := strings.TrimSpace(summary.PlanID)
		if planID == "" {
			planID = plannerConfigString(job.Config, "plan_id")
		}
		mechanism := projectTrajectoryMechanism(summary, job, plansByID[planID])
		run := projectTrajectoryRun{
			JobID:         summary.JobID,
			PlanID:        planID,
			Model:         projectTrajectoryModel(summary, job),
			Mechanism:     mechanism,
			Status:        summary.Status,
			Score:         diagnosisSummaryMetric(summary, targetMetric),
			CostUSD:       summary.EstimatedCostUSD,
			RuntimeSecs:   summary.RuntimeSeconds,
			CompletedAt:   projectTrajectoryCompletedAt(summary, job),
			CreatedAt:     projectTrajectoryCreatedAt(summary, job),
			InputPosition: index,
		}
		if strings.TrimSpace(run.JobID) == "" {
			anonymous = append(anonymous, run)
			continue
		}
		if existing, ok := byJobID[run.JobID]; !ok || projectTrajectoryRunNewer(run, existing) {
			byJobID[run.JobID] = run
		}
	}
	out := make([]projectTrajectoryRun, 0, len(byJobID)+len(anonymous))
	for _, run := range byJobID {
		out = append(out, run)
	}
	out = append(out, anonymous...)
	sort.SliceStable(out, func(i, j int) bool {
		return projectTrajectoryRunBefore(out[i], out[j])
	})
	return out
}

func projectTrajectorySuccessfulRuns(runs []projectTrajectoryRun) []projectTrajectoryRun {
	out := []projectTrajectoryRun{}
	for _, run := range runs {
		if strings.EqualFold(run.Status, jobs.StatusSucceeded) {
			out = append(out, run)
		}
	}
	return out
}

func projectTrajectoryTerminalStatus(status string) bool {
	return strings.EqualFold(status, jobs.StatusSucceeded) || strings.EqualFold(status, jobs.StatusFailed)
}

func projectTrajectoryChampionScore(input ExperimentPlannerInput, successfulRuns []projectTrajectoryRun) float64 {
	if input.CurrentChampion != nil && input.CurrentChampion.Score > 0 {
		return input.CurrentChampion.Score
	}
	best := 0.0
	for _, run := range successfulRuns {
		if run.Score > best {
			best = run.Score
		}
	}
	return best
}

func projectTrajectoryRecentBestDelta(successfulRuns []projectTrajectoryRun, window int) float64 {
	if len(successfulRuns) < 2 {
		return 0
	}
	if window < 1 {
		window = 1
	}
	start := len(successfulRuns) - window
	if start < 0 {
		start = 0
	}
	recentBest := successfulRuns[start].Score
	for _, run := range successfulRuns[start:] {
		if run.Score > recentBest {
			recentBest = run.Score
		}
	}
	priorBest := successfulRuns[0].Score
	if start == 0 {
		return recentBest - successfulRuns[0].Score
	}
	for _, run := range successfulRuns[:start] {
		if run.Score > priorBest {
			priorBest = run.Score
		}
	}
	return recentBest - priorBest
}

func projectTrajectoryPlannerRoundCount(input ExperimentPlannerInput, runs []projectTrajectoryRun) int {
	planIDs := map[string]bool{}
	for _, plan := range input.PriorPlans {
		if strings.TrimSpace(plan.ID) != "" {
			planIDs[plan.ID] = true
		}
	}
	if strings.TrimSpace(input.SourcePlan.ID) != "" {
		planIDs[input.SourcePlan.ID] = true
	}
	for _, run := range runs {
		if strings.TrimSpace(run.PlanID) != "" {
			planIDs[run.PlanID] = true
		}
	}
	return len(planIDs)
}

func projectTrajectoryMechanismOutcomes(runs []projectTrajectoryRun, plansByID map[string]plans.ExperimentPlan, explicitlyBlocked map[string]bool) []PlannerMechanismOutcome {
	accumulators := map[string]*projectTrajectoryMechanismAccumulator{}
	for mechanism := range explicitlyBlocked {
		projectTrajectoryAccumulator(accumulators, mechanism)
	}
	for _, plan := range plansByID {
		for _, experiment := range plan.Experiments {
			mechanism := normalizeMechanism(experiment.Mechanism)
			if mechanism == "" {
				mechanism = inferExperimentMechanismTaxonomy(experiment)
			}
			if mechanism == "" {
				continue
			}
			accumulator := projectTrajectoryAccumulator(accumulators, mechanism)
			if strings.TrimSpace(plan.ID) != "" {
				accumulator.planIDs[plan.ID] = true
			}
		}
	}

	hasPriorBest := false
	priorBest := 0.0
	for _, run := range runs {
		if strings.TrimSpace(run.Mechanism) == "" {
			if strings.EqualFold(run.Status, jobs.StatusSucceeded) && (!hasPriorBest || run.Score > priorBest) {
				priorBest = run.Score
				hasPriorBest = true
			}
			continue
		}
		accumulator := projectTrajectoryAccumulator(accumulators, run.Mechanism)
		accumulator.outcome.AttemptCount++
		accumulator.outcome.TotalCostUSD += run.CostUSD
		accumulator.outcome.TotalRuntimeSeconds += run.RuntimeSecs
		if strings.TrimSpace(run.PlanID) != "" {
			accumulator.planIDs[run.PlanID] = true
		}
		if strings.EqualFold(run.Status, jobs.StatusSucceeded) {
			if run.Score > accumulator.outcome.BestScore {
				accumulator.outcome.BestScore = run.Score
			}
			if hasPriorBest {
				delta := run.Score - priorBest
				if !accumulator.bestDeltaObserved || delta > accumulator.outcome.BestDeltaVsPrior {
					accumulator.outcome.BestDeltaVsPrior = delta
					accumulator.bestDeltaObserved = true
				}
				accumulator.recentDeltas = append(accumulator.recentDeltas, delta)
			} else {
				accumulator.recentDeltas = append(accumulator.recentDeltas, 0)
			}
		}
		if strings.EqualFold(run.Status, jobs.StatusSucceeded) && (!hasPriorBest || run.Score > priorBest) {
			priorBest = run.Score
			hasPriorBest = true
		}
	}

	out := make([]PlannerMechanismOutcome, 0, len(accumulators))
	for _, accumulator := range accumulators {
		outcome := accumulator.outcome
		outcome.PlanCount = len(accumulator.planIDs)
		outcome.BestScore = roundDiagnosis(outcome.BestScore)
		outcome.BestDeltaVsPrior = roundDiagnosis(outcome.BestDeltaVsPrior)
		outcome.RecentBestDelta = roundDiagnosis(projectTrajectoryRecentDeltaFromValues(accumulator.recentDeltas, projectTrajectoryRecentWindow))
		outcome.TotalCostUSD = roundDiagnosis(outcome.TotalCostUSD)
		outcome.TotalRuntimeSeconds = roundDiagnosis(outcome.TotalRuntimeSeconds)
		out = append(out, outcome)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].AttemptCount != out[j].AttemptCount {
			return out[i].AttemptCount > out[j].AttemptCount
		}
		if out[i].BestScore != out[j].BestScore {
			return out[i].BestScore > out[j].BestScore
		}
		return out[i].Mechanism < out[j].Mechanism
	})
	if len(out) > plannerSnapshotMaxMechanisms {
		out = out[:plannerSnapshotMaxMechanisms]
	}
	return out
}

func projectTrajectoryAccumulator(accumulators map[string]*projectTrajectoryMechanismAccumulator, mechanism string) *projectTrajectoryMechanismAccumulator {
	normalized := normalizeMechanism(mechanism)
	if normalized == "" {
		normalized = "baseline_control"
	}
	accumulator, ok := accumulators[normalized]
	if !ok {
		accumulator = &projectTrajectoryMechanismAccumulator{
			outcome: PlannerMechanismOutcome{
				Mechanism: normalized,
				Status:    "unexplored",
			},
			planIDs: map[string]bool{},
		}
		accumulators[normalized] = accumulator
	}
	return accumulator
}

func projectTrajectoryRecentDeltaFromValues(values []float64, window int) float64 {
	if len(values) == 0 {
		return 0
	}
	if window < 1 {
		window = 1
	}
	start := len(values) - window
	if start < 0 {
		start = 0
	}
	best := values[start]
	for _, value := range values[start:] {
		if value > best {
			best = value
		}
	}
	return best
}

func projectTrajectoryMechanism(summary runs.TrainingRunSummary, job jobs.ExperimentJob, plan plans.ExperimentPlan) string {
	experiment := plannerExperimentFromJob(job)
	if mechanism := normalizeMechanism(experiment.Mechanism); mechanism != "" {
		return mechanism
	}
	for _, key := range []string{"mechanism", "mechanism_group", "strategy_type"} {
		if mechanism := normalizeMechanism(plannerConfigString(job.Config, key)); mechanism != "" {
			return mechanism
		}
	}
	if planned, ok := projectTrajectoryPlanExperiment(summary, job, plan); ok {
		if mechanism := normalizeMechanism(planned.Mechanism); mechanism != "" {
			return mechanism
		}
		return inferExperimentMechanismTaxonomy(planned)
	}
	if strings.TrimSpace(summary.Model) != "" && strings.TrimSpace(experiment.Model) == "" {
		experiment.Model = summary.Model
	}
	return inferExperimentMechanismTaxonomy(experiment)
}

func projectTrajectoryPlanExperiment(summary runs.TrainingRunSummary, job jobs.ExperimentJob, plan plans.ExperimentPlan) (plans.PlannedExperiment, bool) {
	if len(plan.Experiments) == 0 {
		return plans.PlannedExperiment{}, false
	}
	if index, ok := diagnosisConfigInt(job.Config, "experiment_index"); ok && index >= 0 && index < len(plan.Experiments) {
		return plan.Experiments[index], true
	}
	if len(plan.Experiments) == 1 {
		return plan.Experiments[0], true
	}
	model := strings.ToLower(strings.TrimSpace(summary.Model))
	if model == "" {
		model = strings.ToLower(strings.TrimSpace(plannerConfigString(job.Config, "model")))
	}
	if model == "" {
		return plans.PlannedExperiment{}, false
	}
	for _, experiment := range plan.Experiments {
		if strings.EqualFold(strings.TrimSpace(experiment.Model), model) {
			return experiment, true
		}
	}
	return plans.PlannedExperiment{}, false
}

func projectTrajectoryModel(summary runs.TrainingRunSummary, job jobs.ExperimentJob) string {
	if strings.TrimSpace(summary.Model) != "" {
		return strings.TrimSpace(summary.Model)
	}
	return plannerConfigString(job.Config, "model")
}

func projectTrajectoryCompletedAt(summary runs.TrainingRunSummary, job jobs.ExperimentJob) time.Time {
	if job.CompletedAt != nil && !job.CompletedAt.IsZero() {
		return *job.CompletedAt
	}
	if !summary.UpdatedAt.IsZero() {
		return summary.UpdatedAt
	}
	return time.Time{}
}

func projectTrajectoryCreatedAt(summary runs.TrainingRunSummary, job jobs.ExperimentJob) time.Time {
	if !summary.CreatedAt.IsZero() {
		return summary.CreatedAt
	}
	return job.CreatedAt
}

func projectTrajectoryRunNewer(left projectTrajectoryRun, right projectTrajectoryRun) bool {
	leftTime := projectTrajectorySortTime(left)
	rightTime := projectTrajectorySortTime(right)
	if !leftTime.Equal(rightTime) {
		return leftTime.After(rightTime)
	}
	return left.InputPosition > right.InputPosition
}

func projectTrajectoryRunBefore(left projectTrajectoryRun, right projectTrajectoryRun) bool {
	leftTime := projectTrajectorySortTime(left)
	rightTime := projectTrajectorySortTime(right)
	if !leftTime.Equal(rightTime) {
		if leftTime.IsZero() {
			return false
		}
		if rightTime.IsZero() {
			return true
		}
		return leftTime.Before(rightTime)
	}
	return left.InputPosition < right.InputPosition
}

func projectTrajectorySortTime(run projectTrajectoryRun) time.Time {
	if !run.CompletedAt.IsZero() {
		return run.CompletedAt
	}
	return run.CreatedAt
}

func projectTrajectoryAllowedNextOnlyWith(mechanism string) []string {
	switch normalizeMechanism(mechanism) {
	case "architecture_challenge":
		return []string{"paired_non_architecture_mechanism", "new_backend_evidence", "champion_confirmation"}
	default:
		return []string{"paired_non_exhausted_mechanism", "new_backend_evidence", "champion_confirmation"}
	}
}

func projectTrajectoryWarnings(card PlannerProjectTrajectoryCard, input ExperimentPlannerInput, explicitlyBlocked map[string]bool) []string {
	warnings := []string{}
	if card.CompletedTrainingRuns == 0 {
		warnings = append(warnings, "no completed training runs are available for project trajectory diagnosis")
	}
	if input.MinimumMeaningfulImprovement < 0.010 {
		warnings = append(warnings, "minimum useful delta was floored at 0.010")
	}
	if len(card.BlockedMechanisms) > 0 {
		warnings = append(warnings, fmt.Sprintf("blocked or exhausted mechanisms: %s", strings.Join(card.BlockedMechanisms, ",")))
	}
	if card.DecisionPressure != "normal" {
		warnings = append(warnings, fmt.Sprintf("decision pressure: %s", card.DecisionPressure))
	}
	if card.NoImprovementRounds > 0 {
		warnings = append(warnings, fmt.Sprintf("%d no-improvement follow-up round(s) are part of project trajectory", card.NoImprovementRounds))
	}
	for _, mechanism := range capSortedMapKeys(explicitlyBlocked, 4) {
		warnings = append(warnings, fmt.Sprintf("rejected strategy memory blocks %s", mechanism))
	}
	return cappedStrings(warnings, 8)
}

func diagnosisConfigInt(config map[string]any, key string) (int, bool) {
	switch value := config[key].(type) {
	case int:
		return value, true
	case int64:
		return int(value), true
	case float64:
		return int(value), true
	case float32:
		return int(value), true
	case jsonNumber:
		out, err := value.Float64()
		if err != nil {
			return 0, false
		}
		return int(out), true
	default:
		return 0, false
	}
}

func diagnosisSummaryMetric(summary runs.TrainingRunSummary, targetMetric string) float64 {
	if targetMetric == "accuracy" {
		return summary.BestAccuracy
	}
	return summary.BestMacroF1
}

func diagnosisPlateauScore(metrics []jobs.EpochMetric, targetMetric string) float64 {
	values := orderedMetricValues(metrics, targetMetric)
	if len(values) < 4 {
		return 0
	}
	bestBeforeTail := values[0]
	for _, value := range values[:len(values)-3] {
		if value > bestBeforeTail {
			bestBeforeTail = value
		}
	}
	bestTail := values[len(values)-3]
	for _, value := range values[len(values)-3:] {
		if value > bestTail {
			bestTail = value
		}
	}
	improvement := bestTail - bestBeforeTail
	if improvement >= 0.015 {
		return 0
	}
	return clamp01((0.015 - improvement) / 0.04)
}

func diagnosisInstabilityScore(metrics []jobs.EpochMetric, targetMetric string) float64 {
	values := orderedMetricValues(metrics, targetMetric)
	if len(values) < 4 {
		return 0
	}
	diffs := []float64{}
	signChanges := 0
	lastSign := 0
	for index := 1; index < len(values); index++ {
		diff := values[index] - values[index-1]
		diffs = append(diffs, math.Abs(diff))
		sign := 0
		if diff > 0.005 {
			sign = 1
		} else if diff < -0.005 {
			sign = -1
		}
		if sign != 0 && lastSign != 0 && sign != lastSign {
			signChanges++
		}
		if sign != 0 {
			lastSign = sign
		}
	}
	return clamp01(averageScore(diffs)/0.055 + float64(signChanges)*0.12)
}

func orderedMetricValues(metrics []jobs.EpochMetric, targetMetric string) []float64 {
	ordered := append([]jobs.EpochMetric(nil), metrics...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Epoch < ordered[j].Epoch
	})
	values := []float64{}
	for _, metric := range ordered {
		value, ok := metric.Metrics[targetMetric]
		if !ok && targetMetric != "macro_f1" {
			value, ok = metric.Metrics["macro_f1"]
		}
		if !ok && targetMetric != "accuracy" {
			value, ok = metric.Metrics["accuracy"]
		}
		if ok {
			values = append(values, value)
		}
	}
	return values
}

func diagnosisMinorityFailure(evaluations []runs.TrainingRunEvaluation) (float64, string) {
	worstRecall := 1.0
	worstLabel := ""
	found := false
	for _, evaluation := range evaluations {
		for label, raw := range evaluation.PerClassMetrics {
			normalizedLabel := strings.ToLower(strings.TrimSpace(label))
			if normalizedLabel == "" || strings.Contains(normalizedLabel, "avg") || normalizedLabel == "accuracy" {
				continue
			}
			stats, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			recall := diagnosisPayloadFloat(stats, "recall")
			if recall == 0 {
				recall = diagnosisPayloadFloat(stats, "f1-score")
			}
			if recall == 0 {
				recall = diagnosisPayloadFloat(stats, "f1")
			}
			if recall <= 0 {
				continue
			}
			found = true
			if recall < worstRecall {
				worstRecall = recall
				worstLabel = label
			}
		}
	}
	if !found {
		return 0, ""
	}
	score := clamp01((0.68 - worstRecall) / 0.48)
	if score == 0 {
		return 0, ""
	}
	return score, fmt.Sprintf("worst per-class recall/F1 is %.3f for %s", worstRecall, worstLabel)
}

func diagnosisLatencyPenalty(evaluations []runs.TrainingRunEvaluation, objective ProjectObjectiveContext) (float64, string) {
	maxLatency := 0.0
	for _, evaluation := range evaluations {
		latency := firstPositivePayloadFloat(evaluation.ModelProfile, "estimated_latency_ms", "latency_ms", "p50_latency_ms", "inference_latency_ms")
		if latency > maxLatency {
			maxLatency = latency
		}
	}
	if maxLatency <= 0 {
		return 0, ""
	}
	threshold := 160.0
	if objective.PrimaryObjective == "low_latency_live_service" {
		threshold = 80.0
	}
	penalty := clamp01((maxLatency - threshold) / threshold)
	if penalty == 0 {
		return 0, ""
	}
	return penalty, fmt.Sprintf("max estimated latency is %.1fms", maxLatency)
}

func diagnosisCostEfficiency(deltaVsChampion float64, totalCost float64, totalRuntime float64) float64 {
	if totalCost <= 0 && totalRuntime <= 0 {
		if deltaVsChampion > 0 {
			return 0.7
		}
		return 0.4
	}
	costPressure := clamp01(totalCost/0.50 + totalRuntime/3600.0)
	if deltaVsChampion <= 0 {
		return roundDiagnosis(maxDiagnosis(0, 0.45-costPressure*0.35))
	}
	return clamp01(0.45 + deltaVsChampion*8 - costPressure*0.25)
}

func diagnosisFailureModes(diagnosis PlannerDiagnosis) []string {
	modes := []string{}
	if diagnosis.OverfittingScore >= 0.55 {
		modes = append(modes, "overfitting")
	}
	if diagnosis.UnderfittingScore >= 0.55 {
		modes = append(modes, "underfitting")
	}
	if diagnosis.PlateauScore >= 0.55 {
		modes = append(modes, "plateau")
	}
	if diagnosis.InstabilityScore >= 0.55 {
		modes = append(modes, "instability")
	}
	if diagnosis.ClassImbalanceScore >= 0.45 || diagnosis.MinorityClassFailureScore >= 0.45 {
		modes = append(modes, "class_imbalance")
	}
	if diagnosis.MinorityClassFailureScore >= 0.55 {
		modes = append(modes, "minority_class_failure")
	}
	if diagnosis.CostEfficiencyScore > 0 && diagnosis.CostEfficiencyScore < 0.35 {
		modes = append(modes, "poor_cost_efficiency")
	}
	if diagnosis.LatencyPenalty >= 0.45 {
		modes = append(modes, "latency_penalty")
	}
	if diagnosis.ImprovementStagnationScore >= 0.55 {
		modes = append(modes, "improvement_stagnation")
	}
	return uniqueDiagnosisStrings(modes)
}

func diagnosisSignals(diagnosis PlannerDiagnosis) []string {
	signals := []string{}
	if diagnosis.OverfittingScore > 0 {
		signals = append(signals, fmt.Sprintf("overfitting_score=%.3f", diagnosis.OverfittingScore))
	}
	if diagnosis.UnderfittingScore > 0 {
		signals = append(signals, fmt.Sprintf("underfitting_score=%.3f", diagnosis.UnderfittingScore))
	}
	if diagnosis.PlateauScore > 0 {
		signals = append(signals, fmt.Sprintf("plateau_score=%.3f", diagnosis.PlateauScore))
	}
	if diagnosis.InstabilityScore > 0 {
		signals = append(signals, fmt.Sprintf("instability_score=%.3f", diagnosis.InstabilityScore))
	}
	if diagnosis.ClassImbalanceScore > 0 {
		signals = append(signals, fmt.Sprintf("class_imbalance_score=%.3f", diagnosis.ClassImbalanceScore))
	}
	if diagnosis.MinorityClassFailureScore > 0 {
		signals = append(signals, fmt.Sprintf("minority_class_failure_score=%.3f", diagnosis.MinorityClassFailureScore))
	}
	if diagnosis.BestMetricDeltaVsChampion != 0 {
		signals = append(signals, fmt.Sprintf("best_metric_delta_vs_champion=%.3f", diagnosis.BestMetricDeltaVsChampion))
	}
	if diagnosis.LatencyPenalty > 0 {
		signals = append(signals, fmt.Sprintf("latency_penalty=%.3f", diagnosis.LatencyPenalty))
	}
	if diagnosis.ImprovementStagnationScore > 0 {
		signals = append(signals, fmt.Sprintf("improvement_stagnation_score=%.3f", diagnosis.ImprovementStagnationScore))
	}
	return signals
}

func diagnosisEvidence(input ExperimentPlannerInput, diagnosis PlannerDiagnosis, bestScore float64, totalCost float64, totalRuntime float64, minorityEvidence string, latencyEvidence string) []string {
	evidence := []string{
		fmt.Sprintf("plan has %d summaries and %d evaluation payloads", len(input.PlanSummaries), len(input.PlanEvaluations)),
		fmt.Sprintf("best source-plan metric is %.3f with delta %.3f vs champion", bestScore, diagnosis.BestMetricDeltaVsChampion),
		fmt.Sprintf("source-plan cost %.3f USD and runtime %.0fs", totalCost, totalRuntime),
	}
	if diagnosis.GeneralizationGap != 0 {
		evidence = append(evidence, fmt.Sprintf("average validation/train loss gap is %.3f", diagnosis.GeneralizationGap))
	}
	if input.DatasetInsights.ImbalanceRatio > 0 {
		evidence = append(evidence, fmt.Sprintf("dataset imbalance ratio is %.2f", input.DatasetInsights.ImbalanceRatio))
	}
	if minorityEvidence != "" {
		evidence = append(evidence, minorityEvidence)
	}
	if latencyEvidence != "" {
		evidence = append(evidence, latencyEvidence)
	}
	if input.NoImprovementRounds > 0 {
		evidence = append(evidence, fmt.Sprintf("%d no-improvement follow-up round(s)", input.NoImprovementRounds))
	}
	return uniqueDiagnosisStrings(evidence)
}

func normalizedDiagnosisMetric(metric string) string {
	normalized := strings.ToLower(strings.TrimSpace(metric))
	if normalized == "accuracy" {
		return "accuracy"
	}
	return "macro_f1"
}

func diagnosisPayloadFloat(payload map[string]any, key string) float64 {
	switch value := payload[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case jsonNumber:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func firstPositivePayloadFloat(payload map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value := diagnosisPayloadFloat(payload, key); value > 0 {
			return value
		}
	}
	return 0
}

func averageScore(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func maxDiagnosis(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func roundDiagnosis(value float64) float64 {
	return math.Round(value*1000) / 1000
}

func uniqueDiagnosisStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}
