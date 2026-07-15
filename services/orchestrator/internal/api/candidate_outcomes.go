package api

import (
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func (s *Server) finalizeCandidateOutcomesForPlan(planID string) (bool, error) {
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return false, err
	}
	if plan.SourceDecisionID == "" {
		return false, nil
	}
	candidates, err := s.store.ListDecisionCandidateProvenance(plan.SourceDecisionID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	if len(candidates) == 0 {
		return false, nil
	}
	projectJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return false, err
	}
	planJobs := candidateJobsForPlan(projectJobs, plan.ID)
	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return false, err
	}
	events, err := s.store.ListProjectExecutionEvents(plan.ProjectID, 500)
	if err != nil {
		return false, err
	}
	skipped := skippedCandidateExperimentReasons(events, plan.ID)
	summaryByJob := map[string]runs.TrainingRunSummary{}
	for _, summary := range summaries {
		summaryByJob[summary.JobID] = summary
	}
	evaluationByJob := evaluationsByJobID(evaluations)

	updates := make([]calibration.CandidateOutcomeUpdate, 0, len(candidates))
	for _, candidate := range candidates {
		if !candidate.Selected || candidate.SelectedExperimentIndex == nil {
			continue
		}
		experimentIndex := *candidate.SelectedExperimentIndex
		if experimentIndex < 0 || experimentIndex >= len(plan.Experiments) {
			continue
		}
		update := calibration.CandidateOutcomeUpdate{
			CandidateIndex: candidate.CandidateIndex,
			FollowUpPlanID: plan.ID,
			ExperimentID:   calibration.ExperimentLineageID(plan.ID, experimentIndex),
		}
		job, mapped := unambiguousCandidateJob(candidate, planJobs[experimentIndex])
		if !mapped {
			if reason, ok := skipped[experimentIndex]; len(planJobs[experimentIndex]) == 0 && ok {
				update = skippedCandidateOutcome(update, reason)
			}
			updates = append(updates, update)
			continue
		}
		update.JobID = candidateStringPointer(job.ID)
		terminalState, terminal := candidateJobTerminalState(job)
		if !terminal {
			updates = append(updates, update)
			continue
		}
		update.TerminalState = candidateStringPointer(terminalState)
		if summary, ok := summaryByJob[job.ID]; ok {
			if terminalState == calibration.CandidateTerminalSucceeded || summary.EstimatedCostUSD > 0 {
				update.CostUSD = candidateFloatPointer(summary.EstimatedCostUSD)
			}
			if terminalState == calibration.CandidateTerminalSucceeded || summary.RuntimeSeconds > 0 {
				update.RuntimeSeconds = candidateFloatPointer(summary.RuntimeSeconds)
			}
		}
		eligibility, attemptID := s.candidateExecutionEligibility(candidate, job)
		if attemptID != "" {
			update.AttemptID = candidateStringPointer(attemptID)
		}
		if eligibility.RealizedEffectiveHash != "" {
			update.RealizedEffectiveHash = candidateStringPointer(eligibility.RealizedEffectiveHash)
		}
		if terminalState != calibration.CandidateTerminalSucceeded {
			reason := "terminal_" + strings.ToLower(terminalState)
			if eligibility.EligibilityReason == execution.EvidenceReasonMismatched || eligibility.EligibilityReason == execution.EvidenceReasonSimulated {
				reason = eligibility.EligibilityReason
			}
			update = unobservedCandidateOutcome(update, reason)
			updates = append(updates, update)
			continue
		}
		summary, hasSummary := summaryByJob[job.ID]
		if !hasSummary || strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			update = unobservedCandidateOutcome(update, "successful_job_without_succeeded_summary")
			updates = append(updates, update)
			continue
		}
		if !eligibility.LearningEligible {
			reason := eligibility.EligibilityReason
			if reason == "" {
				reason = execution.EvidenceReasonVerdictNotEvidenceEligible
			}
			update = unobservedCandidateOutcome(update, reason)
			updates = append(updates, update)
			continue
		}
		actualScore, observed := observedScoreForFrozenForecast(candidate.Forecast, summary, evaluationByJob[job.ID])
		if !observed {
			update = unobservedCandidateOutcome(update, "frozen_forecast_score_unobserved")
			updates = append(updates, update)
			continue
		}
		actualDelta := actualScore - candidate.Forecast.BaselineScore
		eligible := true
		reason := eligibility.EligibilityReason
		update.OutcomeStatus = calibration.CandidateOutcomeObserved
		update.ActualScore = candidateFloatPointer(actualScore)
		update.ActualDelta = candidateFloatPointer(actualDelta)
		update.CalibrationEligible = &eligible
		update.EligibilityReason = &reason
		updates = append(updates, update)
	}
	updated, err := s.store.FinalizeCandidateOutcomes(plan.SourceDecisionID, updates)
	if err != nil {
		return false, err
	}
	return candidatePlanExperimentsTerminal(plan, planJobs, updated), nil
}

func (s *Server) finalizeCandidateOutcomesAfterNonTrainingJob(job jobs.ExperimentJob) {
	if job.Template == jobs.TemplateTrainExperiment {
		return
	}
	planID := jobConfigString(job.Config, "plan_id")
	if planID == "" {
		return
	}
	complete, err := s.finalizeCandidateOutcomesForPlan(planID)
	if err != nil {
		log.Printf("candidate outcome finalization failed for plan %s after job %s: %v", planID, job.ID, err)
		return
	}
	if !complete {
		return
	}
	plan, err := s.store.GetExperimentPlan(planID)
	if err == nil {
		err = s.recordExperimentPlannerOutcomeForPlan(plan)
	}
	if err != nil {
		log.Printf("candidate plan aggregate finalization failed for plan %s after job %s: %v", planID, job.ID, err)
	}
}

func candidateJobsForPlan(projectJobs []jobs.ExperimentJob, planID string) map[int][]jobs.ExperimentJob {
	byExperiment := map[int][]jobs.ExperimentJob{}
	for _, job := range projectJobs {
		if job.Template != jobs.TemplateTrainExperiment && job.Template != jobs.TemplateLabelQualityAudit {
			continue
		}
		if jobConfigString(job.Config, "plan_id") != planID {
			continue
		}
		index, ok := configInt(job.Config, "experiment_index")
		if !ok || index < 0 {
			continue
		}
		byExperiment[index] = append(byExperiment[index], job)
	}
	for index := range byExperiment {
		sort.Slice(byExperiment[index], func(i, j int) bool {
			return byExperiment[index][i].CreatedAt.Before(byExperiment[index][j].CreatedAt)
		})
	}
	return byExperiment
}

func unambiguousCandidateJob(candidate calibration.CandidateProvenance, candidates []jobs.ExperimentJob) (jobs.ExperimentJob, bool) {
	if candidate.JobID != nil {
		for _, job := range candidates {
			if job.ID == *candidate.JobID {
				return job, true
			}
		}
		return jobs.ExperimentJob{}, false
	}
	compatible := make([]jobs.ExperimentJob, 0, len(candidates))
	for _, job := range candidates {
		hash := acceptedSpecHashFromJob(job)
		if hash == "" || hash == candidate.AcceptedSpecHash {
			compatible = append(compatible, job)
		}
	}
	if len(compatible) == 1 {
		return compatible[0], true
	}
	return jobs.ExperimentJob{}, false
}

func (s *Server) candidateExecutionEligibility(candidate calibration.CandidateProvenance, job jobs.ExperimentJob) (execution.EvidenceEligibility, string) {
	record, err := s.store.GetJobExecutionRecord(job.ID)
	if err != nil {
		return execution.DeriveEvidenceEligibility(nil, execution.LegacyEvidencePolicyVisibleOnly), candidateAttemptIDFromJob(job)
	}
	eligibility := execution.DeriveEvidenceEligibility(&record, execution.LegacyEvidencePolicyVisibleOnly)
	if record.AcceptedSpec.RequestedConfigHash != candidate.RequestedConfigHash || record.AcceptedSpec.AcceptedSpecHash != candidate.AcceptedSpecHash {
		eligibility.LearningEligible = false
		eligibility.AutomaticChampionEligible = false
		eligibility.EligibilityReason = "candidate_execution_identity_mismatch"
	}
	attemptID := ""
	if attempt, ok := latestCandidateAttempt(record.Attempts); ok {
		attemptID = attempt.AttemptID
	} else {
		attemptID = candidateAttemptIDFromJob(job)
	}
	return eligibility, attemptID
}

func candidateAttemptIDFromJob(job jobs.ExperimentJob) string {
	if job.Attempt < 1 {
		return ""
	}
	return fmt.Sprintf("%s:attempt-%d", job.ID, job.Attempt)
}

func latestCandidateAttempt(attempts []execution.AttemptExecutionRecord) (execution.AttemptExecutionRecord, bool) {
	if len(attempts) == 0 {
		return execution.AttemptExecutionRecord{}, false
	}
	latest := attempts[0]
	for _, attempt := range attempts[1:] {
		if attempt.AttemptNumber > latest.AttemptNumber || (attempt.AttemptNumber == latest.AttemptNumber && attempt.UpdatedAt.After(latest.UpdatedAt)) {
			latest = attempt
		}
	}
	return latest, true
}

func observedScoreForFrozenForecast(forecast calibration.CandidateForecastContract, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) (float64, bool) {
	if forecast.ScoreVersion != calibration.CandidateForecastScoreVersionV1 {
		return 0, false
	}
	basis := strings.ToLower(strings.TrimSpace(forecast.ScoreBasis))
	target := normalizedPlannerTargetMetric(forecast.ForecastTarget)
	if basis == "loss_heavy_deployment_readiness" || basis == "deployment_readiness_score" {
		return roundDiagnosticFloat(holisticRunScore(forecast.ForecastTarget, summary, evaluation, agents.ProjectObjectiveContext{})), true
	}
	if basis == target+"_score" {
		return roundDiagnosticFloat(plannerTargetMetricValue(forecast.ForecastTarget, summary, evaluation)), true
	}
	return 0, false
}

func candidateJobTerminalState(job jobs.ExperimentJob) (string, bool) {
	switch strings.ToUpper(strings.TrimSpace(job.Status)) {
	case jobs.StatusSucceeded:
		return calibration.CandidateTerminalSucceeded, true
	case jobs.StatusFailed:
		if strings.EqualFold(jobConfigString(job.Config, "failure_class"), "cancelled") || jobConfigString(job.Config, "cancel_reason") != "" {
			return calibration.CandidateTerminalCancelled, true
		}
		return calibration.CandidateTerminalFailed, true
	default:
		return "", false
	}
}

func skippedCandidateExperimentReasons(events []execution.ExecutionEvent, planID string) map[int]string {
	out := map[int]string{}
	for _, event := range events {
		if event.PlanID != planID {
			continue
		}
		if event.EventType == execution.EventCostBudgetBlocked {
			switch skipped := event.Payload["skipped"].(type) {
			case []map[string]any:
				for _, item := range skipped {
					if index, ok := configInt(item, "experiment_index"); ok && index >= 0 {
						out[index] = "budget_skipped"
					}
				}
			case []any:
				for _, value := range skipped {
					if item, ok := value.(map[string]any); ok {
						if index, ok := configInt(item, "experiment_index"); ok && index >= 0 {
							out[index] = "budget_skipped"
						}
					}
				}
			}
			continue
		}
		if event.EventType == execution.EventExecutionValidationReported {
			index, ok := configInt(event.Payload, "experiment_index")
			if !ok || index < 0 {
				continue
			}
			var report execution.ExecutionValidationReport
			if err := decodeCandidateProvenancePayload(event.Payload["report"], &report); err == nil && report.AcceptedDuplicate.Skip {
				out[index] = "accepted_spec_duplicate_skipped"
			}
		}
	}
	return out
}

func skippedCandidateOutcome(update calibration.CandidateOutcomeUpdate, reason string) calibration.CandidateOutcomeUpdate {
	update.TerminalState = candidateStringPointer(calibration.CandidateTerminalSkipped)
	return unobservedCandidateOutcome(update, reason)
}

func unobservedCandidateOutcome(update calibration.CandidateOutcomeUpdate, reason string) calibration.CandidateOutcomeUpdate {
	eligible := false
	update.OutcomeStatus = calibration.CandidateOutcomeUnobserved
	update.CalibrationEligible = &eligible
	update.EligibilityReason = candidateStringPointer(reason)
	return update
}

func candidatePlanExperimentsTerminal(plan plans.ExperimentPlan, jobsByExperiment map[int][]jobs.ExperimentJob, candidates []calibration.CandidateProvenance) bool {
	if plan.ID == "" || len(plan.Experiments) == 0 {
		return false
	}
	skipped := map[int]bool{}
	for _, candidate := range candidates {
		if candidate.SelectedExperimentIndex != nil && candidate.TerminalState != nil && *candidate.TerminalState == calibration.CandidateTerminalSkipped {
			skipped[*candidate.SelectedExperimentIndex] = true
		}
	}
	for index := range plan.Experiments {
		planJobs := jobsByExperiment[index]
		if len(planJobs) == 0 {
			if !skipped[index] {
				return false
			}
			continue
		}
		for _, job := range planJobs {
			if _, terminal := candidateJobTerminalState(job); !terminal {
				return false
			}
		}
	}
	return true
}

func candidateStringPointer(value string) *string {
	value = strings.TrimSpace(value)
	return &value
}

func candidateFloatPointer(value float64) *float64 {
	return &value
}
