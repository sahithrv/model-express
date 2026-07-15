package api

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/strategies"
)

func plannerStrategyMemory(records []memory.AgentMemoryRecord) ([]agents.PlannerStrategyMemory, []agents.PlannerStrategyMemory) {
	successful := []agents.PlannerStrategyMemory{}
	failed := []agents.PlannerStrategyMemory{}
	for _, record := range records {
		if record.Kind != memory.KindPlanningOutcome {
			continue
		}
		entry := plannerStrategyMemoryFromRecord(record)
		if !entry.LearningEligible {
			continue
		}
		switch entry.OutcomeStatus {
		case agents.ExperimentPlanningOutcomeImprovedChampion, agents.ExperimentPlanningOutcomeMinorImprovement:
			if len(successful) < 6 {
				successful = append(successful, entry)
			}
		case agents.ExperimentPlanningOutcomeNoImprovement, agents.ExperimentPlanningOutcomeFailed:
			if len(failed) < 6 {
				failed = append(failed, entry)
			}
		}
	}
	return successful, failed
}

func plannerRejectedOptions(records []memory.AgentMemoryRecord, diagnosis agents.PlannerDiagnosis) []agents.RejectedPlannerOption {
	out := []agents.RejectedPlannerOption{}
	failureModes := map[string]bool{}
	for _, mode := range diagnosis.RecommendedFailureModes {
		failureModes[strings.ToLower(strings.TrimSpace(mode))] = true
	}
	for _, record := range records {
		if record.Kind != memory.KindPlanningFeedback && record.Kind != memory.KindPlanningOutcome {
			continue
		}
		value, ok := record.Payload["rejected_options"]
		if !ok {
			continue
		}
		blob, err := json.Marshal(value)
		if err != nil {
			continue
		}
		var options []agents.RejectedPlannerOption
		if err := json.Unmarshal(blob, &options); err != nil {
			continue
		}
		for _, option := range options {
			if !rejectedOptionApplies(option, failureModes) {
				continue
			}
			out = append(out, option)
			if len(out) >= 8 {
				return out
			}
		}
	}
	return out
}

func rejectedOptionApplies(option agents.RejectedPlannerOption, failureModes map[string]bool) bool {
	if len(option.AppliesWhen) == 0 || len(failureModes) == 0 {
		return true
	}
	for _, condition := range option.AppliesWhen {
		normalized := strings.ToLower(strings.TrimSpace(condition))
		if failureModes[normalized] {
			return true
		}
	}
	return false
}

func plannerStrategyScorecards(scorecards []strategies.StrategyScorecard, datasetID string) []agents.PlannerStrategyScorecard {
	out := []agents.PlannerStrategyScorecard{}
	for _, scorecard := range scorecards {
		evidenceEligible := scorecard.EvidenceEligible
		fidelityVerdicts := append([]string(nil), scorecard.FidelityVerdicts...)
		if len(fidelityVerdicts) == 0 {
			fidelityVerdicts = []string{execution.ExecutionVerdictUnverified}
			evidenceEligible = legacyExecutionEvidencePolicy() == execution.LegacyEvidencePolicyAllow
		}
		if !evidenceEligible {
			continue
		}
		if scorecard.DatasetID != datasetID && len(out) >= 6 {
			continue
		}
		out = append(out, agents.PlannerStrategyScorecard{
			ID:                        scorecard.ID,
			DatasetID:                 scorecard.DatasetID,
			SourceDecisionID:          scorecard.SourceDecisionID,
			SourcePlanID:              scorecard.SourcePlanID,
			FollowUpPlanID:            scorecard.FollowUpPlanID,
			StrategyType:              scorecard.StrategyType,
			PlanningMode:              scorecard.PlanningMode,
			Mechanism:                 scorecard.Mechanism,
			Intervention:              scorecard.Intervention,
			DiagnosisTriggers:         scorecard.DiagnosisTriggers,
			EvidenceUsed:              scorecard.EvidenceUsed,
			ExpectedEffect:            scorecard.ExpectedEffect,
			DatasetTraits:             scorecard.DatasetTraits,
			ObjectiveProfile:          scorecard.ObjectiveProfile,
			ProposedChanges:           scorecard.ProposedChanges,
			ExpectedDelta:             scorecard.ExpectedDelta,
			ActualDelta:               scorecard.ActualDelta,
			ConfidenceBefore:          scorecard.ConfidenceBefore,
			ConfidenceAfter:           scorecard.ConfidenceAfter,
			CostUSD:                   scorecard.CostUSD,
			RuntimeSeconds:            scorecard.RuntimeSeconds,
			Outcome:                   scorecard.Outcome,
			Lesson:                    scorecard.Lesson,
			Tags:                      scorecard.Tags,
			FidelityVerdicts:          fidelityVerdicts,
			EvidenceEligible:          evidenceEligible,
			RequestedMechanism:        scorecard.RequestedMechanism,
			RealizedMechanismIdentity: scorecard.RealizedMechanismIdentity,
		})
		if len(out) >= 10 {
			break
		}
	}
	return out
}

func plannerStrategyMemoryFromRecord(record memory.AgentMemoryRecord) agents.PlannerStrategyMemory {
	bestModel := ""
	if champion, ok := experimentChampionFromPayload(record.Payload["actual_best_run"]); ok {
		bestModel = champion.Model
	}
	learningEligible := legacyExecutionEvidencePolicy() == execution.LegacyEvidencePolicyAllow
	if _, ok := record.Payload["evidence_eligible_run_count"]; ok {
		learningEligible = payloadFloat(record.Payload, "evidence_eligible_run_count") > 0
	}
	return agents.PlannerStrategyMemory{
		MemoryID:                record.ID,
		OutcomeStatus:           payloadString(record.Payload, "outcome_status"),
		Lesson:                  payloadString(record.Payload, "lesson"),
		BestModel:               bestModel,
		ActualDeltaVsChampion:   payloadFloat(record.Payload, "actual_delta_vs_champion"),
		ExpectedDeltaVsChampion: payloadFloat(record.Payload, "expected_delta_vs_champion"),
		TotalCostUSD:            payloadFloat(record.Payload, "total_cost_usd"),
		TotalRuntimeSeconds:     payloadFloat(record.Payload, "total_runtime_seconds"),
		ProposedModels:          proposedModelsFromPayload(record.Payload),
		Tags:                    record.Tags,
		FidelityVerdict:         firstNonEmptyString(firstOutcomeFidelityVerdict(record.Payload), execution.ExecutionVerdictUnverified),
		LearningEligible:        learningEligible,
	}
}

func firstOutcomeFidelityVerdict(payload map[string]any) string {
	values, ok := payload["execution_evidence"].([]any)
	if !ok || len(values) == 0 {
		return ""
	}
	if item, ok := values[0].(map[string]any); ok {
		return payloadString(item, "fidelity_verdict")
	}
	return ""
}

func proposedModelsFromPayload(payload map[string]any) []string {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return []string{}
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return []string{}
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return []string{}
	}
	models := []string{}
	for _, experiment := range experiments {
		if strings.TrimSpace(experiment.Model) != "" {
			models = append(models, experiment.Model)
		}
	}
	return uniqueStrings(models)
}

func agentDecisionByID(agentDecisions []decisions.AgentDecision, decisionID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.ID == decisionID {
			return decision, true
		}
	}
	return decisions.AgentDecision{}, false
}

func experimentPlanningOutcomeForPlan(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	executionEvidenceByJob map[string]agents.ExperimentExecutionEvidence,
) (agents.ExperimentPlanningOutcome, error) {
	return experimentPlanningOutcomeForPlanWithTerminalCount(
		sourceDecision, followUpPlan, projectPlans, summaries, evaluations, objectiveContext,
		executionEvidenceByJob, len(summariesForPlanID(summaries, followUpPlan.ID)),
	)
}

func experimentPlanningOutcomeForPlanWithTerminalCount(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	executionEvidenceByJob map[string]agents.ExperimentExecutionEvidence,
	terminalExperimentCount int,
) (agents.ExperimentPlanningOutcome, error) {
	planSummaries := summariesForPlanID(summaries, followUpPlan.ID)
	eligibleSummaries := learningEligibleSummaries(planSummaries, executionEvidenceByJob)
	eligibleProjectSummaries := learningEligibleSummaries(summaries, executionEvidenceByJob)
	eligibleEvaluations := evaluationsForEligibleSummaries(evaluations, eligibleProjectSummaries)
	evaluationsByJob := evaluationsByJobID(evaluations)
	proposedExperiments, err := plannedExperimentsFromPayload(sourceDecision.Payload)
	if err != nil {
		proposedExperiments = []plans.PlannedExperiment{}
	}

	baselineChampion := baselineChampionForPlannerOutcome(sourceDecision, followUpPlan, projectPlans, eligibleProjectSummaries, eligibleEvaluations, objectiveContext, executionEvidenceByJob)
	bestSummary, hasBest := bestSuccessfulTrainingSummaryForObjective(followUpPlan.TargetMetric, eligibleSummaries, evaluationsForPlanID(eligibleEvaluations, followUpPlan.ID), objectiveContext)

	var actualBest *agents.ExperimentChampion
	actualDelta := 0.0
	if hasBest {
		best := experimentChampionFromSummaryWithEvaluation(followUpPlan.TargetMetric, bestSummary, evaluationsByJob[bestSummary.JobID], objectiveContext)
		if evidence, ok := executionEvidenceByJob[bestSummary.JobID]; ok {
			best.ExecutionEvidence = &evidence
		}
		actualBest = &best
		if baselineChampion != nil {
			actualDelta = best.Score - baselineChampion.Score
		} else {
			actualDelta = best.Score
		}
	}

	expectedDelta := payloadFloat(sourceDecision.Payload, "expected_delta_vs_champion")
	metExpectedDelta := hasBest && actualDelta > plannerMinimumMeaningfulImprovement
	if expectedDelta > 0 {
		metExpectedDelta = hasBest && actualDelta >= expectedDelta
	}
	outcomeStatus := plannerOutcomeStatus(actualDelta, hasBest)
	if !hasBest && successfulSummaryCount(planSummaries) > 0 {
		outcomeStatus = agents.ExperimentPlanningOutcomeExecutionIneligible
	}
	planEvidence := make([]agents.ExperimentExecutionEvidence, 0, len(planSummaries))
	for _, summary := range planSummaries {
		if evidence, ok := executionEvidenceByJob[summary.JobID]; ok {
			planEvidence = append(planEvidence, evidence)
		}
	}
	sort.Slice(planEvidence, func(i, j int) bool { return planEvidence[i].JobID < planEvidence[j].JobID })
	outcome := agents.ExperimentPlanningOutcome{
		OutcomeType:              "planner_followup_result",
		OutcomeStatus:            outcomeStatus,
		SourceDecisionID:         sourceDecision.ID,
		SourcePlanID:             sourceDecision.PlanID,
		FollowUpPlanID:           followUpPlan.ID,
		BaselineChampion:         baselineChampion,
		ActualBestRun:            actualBest,
		ExpectedDeltaVsChampion:  expectedDelta,
		ActualDeltaVsChampion:    actualDelta,
		MetExpectedDelta:         metExpectedDelta,
		TotalCostUSD:             totalSummaryCost(planSummaries),
		TotalRuntimeSeconds:      totalSummaryRuntime(planSummaries),
		TerminalRunCount:         terminalExperimentCount,
		SuccessfulRunCount:       successfulSummaryCount(planSummaries),
		FailedRunCount:           failedSummaryCount(planSummaries),
		ProposedExperiments:      proposedExperiments,
		CompletedAt:              time.Now().UTC(),
		ExecutionEvidence:        planEvidence,
		EvidenceEligibleRunCount: len(eligibleSummaries),
	}
	outcome.Lesson = plannerOutcomeLesson(followUpPlan.TargetMetric, outcome)
	return outcome, nil
}

func baselineChampionForPlannerOutcome(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	executionEvidenceByJob map[string]agents.ExperimentExecutionEvidence,
) *agents.ExperimentChampion {
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["current_champion"]); ok {
		if attachEligibleChampionEvidence(champion, executionEvidenceByJob) {
			return champion
		}
	}
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["source_plan_baseline_champion"]); ok {
		if attachEligibleChampionEvidence(champion, executionEvidenceByJob) {
			return champion
		}
	}
	evaluationsByJob := evaluationsByJobID(evaluations)
	if summary, ok := bestSuccessfulTrainingSummaryBeforePlanForObjective(followUpPlan.TargetMetric, projectPlans, summaries, evaluations, objectiveContext, followUpPlan.ID); ok {
		champion := experimentChampionFromSummaryWithEvaluation(followUpPlan.TargetMetric, summary, evaluationsByJob[summary.JobID], objectiveContext)
		return &champion
	}
	return nil
}

func attachEligibleChampionEvidence(champion *agents.ExperimentChampion, evidenceByJob map[string]agents.ExperimentExecutionEvidence) bool {
	if champion == nil {
		return false
	}
	evidence, ok := evidenceByJob[champion.JobID]
	if !ok {
		return legacyExecutionEvidencePolicy() == execution.LegacyEvidencePolicyAllow
	}
	if !evidence.LearningEligible {
		return false
	}
	evidenceCopy := evidence
	champion.ExecutionEvidence = &evidenceCopy
	return true
}

func experimentChampionFromPayload(value any) (*agents.ExperimentChampion, bool) {
	if value == nil {
		return nil, false
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var champion agents.ExperimentChampion
	if err := json.Unmarshal(blob, &champion); err != nil {
		return nil, false
	}
	if champion.JobID == "" {
		return nil, false
	}
	return &champion, true
}

func plannerOutcomeStatus(actualDelta float64, hasBest bool) string {
	if !hasBest {
		return agents.ExperimentPlanningOutcomeFailed
	}
	if actualDelta > plannerMinimumMeaningfulImprovement {
		return agents.ExperimentPlanningOutcomeImprovedChampion
	}
	if actualDelta > 0 {
		return agents.ExperimentPlanningOutcomeMinorImprovement
	}
	return agents.ExperimentPlanningOutcomeNoImprovement
}

func plannerOutcomeConfidence(outcome agents.ExperimentPlanningOutcome) float64 {
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return minFloat(0.95, 0.70+maxFloat(0, outcome.ActualDeltaVsChampion)*4)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return 0.55
	case agents.ExperimentPlanningOutcomeFailed:
		return 0.20
	default:
		return 0.35
	}
}

func plannerOutcomeLesson(targetMetric string, outcome agents.ExperimentPlanningOutcome) string {
	metric := "deployment readiness"
	if outcome.OutcomeStatus == agents.ExperimentPlanningOutcomeExecutionIneligible {
		return fmt.Sprintf("Planner follow-up plan %s produced results, but none were eligible learning evidence under execution-fidelity policy; do not credit its requested mechanism.", outcome.FollowUpPlanID)
	}
	if outcome.OutcomeStatus == agents.ExperimentPlanningOutcomeFailed {
		return fmt.Sprintf("Planner follow-up plan %s produced no successful runs after %.3f total cost; avoid repeating this failed strategy without changing the setup.", outcome.FollowUpPlanID, outcome.TotalCostUSD)
	}
	bestModel := ""
	if outcome.ActualBestRun != nil {
		bestModel = outcome.ActualBestRun.Model
	}
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return fmt.Sprintf("Planner follow-up plan %s improved the champion with %s by %.3f %s; similar strategy changes are worth reusing.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return fmt.Sprintf("Planner follow-up plan %s only slightly improved the champion with %s by %.3f %s, below the meaningful threshold %.3f; treat this as weak evidence.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, plannerMinimumMeaningfulImprovement)
	default:
		return fmt.Sprintf("Planner follow-up plan %s failed to beat the prior champion; best run %s trailed by %.3f %s after %.3f total cost.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, outcome.TotalCostUSD)
	}
}

func plannerOutcomeTags(outcome agents.ExperimentPlanningOutcome) []string {
	tags := []string{"planner_outcome", outcome.OutcomeStatus}
	if outcome.MetExpectedDelta {
		tags = append(tags, "met_expected_delta")
	} else {
		tags = append(tags, "missed_expected_delta")
	}
	if outcome.ActualBestRun != nil && outcome.ActualBestRun.Model != "" {
		tags = append(tags, strings.ToLower(strings.TrimSpace(outcome.ActualBestRun.Model)))
	}
	for _, evidence := range outcome.ExecutionEvidence {
		if evidence.FidelityVerdict != "" {
			tags = append(tags, strings.ToLower(evidence.FidelityVerdict))
		}
	}
	if outcome.EvidenceEligibleRunCount == 0 {
		tags = append(tags, "execution_evidence_ineligible")
	}
	return tags
}

func primaryOutcomeExecutionEvidence(outcome agents.ExperimentPlanningOutcome) agents.ExperimentExecutionEvidence {
	if outcome.ActualBestRun != nil && outcome.ActualBestRun.ExecutionEvidence != nil {
		return *outcome.ActualBestRun.ExecutionEvidence
	}
	if len(outcome.ExecutionEvidence) > 0 {
		return outcome.ExecutionEvidence[0]
	}
	return agents.ExperimentExecutionEvidence{}
}

func outcomeExecutionFidelityVerdicts(outcome agents.ExperimentPlanningOutcome) []string {
	values := []string{}
	for _, evidence := range outcome.ExecutionEvidence {
		if evidence.FidelityVerdict != "" {
			values = append(values, evidence.FidelityVerdict)
		}
	}
	values = uniqueStrings(values)
	sort.Strings(values)
	return values
}

func totalSummaryCost(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.EstimatedCostUSD
	}
	return total
}

func totalSummaryRuntime(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.RuntimeSeconds
	}
	return total
}

func successfulSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusSucceeded {
			count++
		}
	}
	return count
}

func failedSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusFailed {
			count++
		}
	}
	return count
}
