package api

import (
	"errors"
	"os"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func legacyExecutionEvidencePolicy() string {
	return execution.NormalizeLegacyEvidencePolicy(os.Getenv("MODEL_EXPRESS_LEGACY_EXECUTION_EVIDENCE_POLICY"))
}

func (s *Server) executionEvidenceForJobs(projectJobs []jobs.ExperimentJob) (map[string]agents.ExperimentExecutionEvidence, error) {
	out := make(map[string]agents.ExperimentExecutionEvidence, len(projectJobs))
	for _, job := range projectJobs {
		var record *execution.ExecutionRecord
		stored, err := s.store.GetJobExecutionRecord(job.ID)
		if err == nil {
			record = &stored
		} else if !errors.Is(err, store.ErrNotFound) {
			return nil, err
		}
		eligibility := execution.DeriveEvidenceEligibility(record, legacyExecutionEvidencePolicy())
		requestedMechanism := requestedMechanismForJob(job)
		out[job.ID] = agents.ExperimentExecutionEvidence{
			JobID:                     job.ID,
			PlanID:                    jobConfigString(job.Config, "plan_id"),
			Model:                     jobConfigString(job.Config, "model"),
			JobStatus:                 job.Status,
			SchemaVersion:             eligibility.SchemaVersion,
			CapabilityVersion:         eligibility.CapabilityVersion,
			LifecycleStatus:           eligibility.LifecycleStatus,
			FidelityVerdict:           eligibility.FidelityVerdict,
			RequestedConfigHash:       eligibility.RequestedConfigHash,
			AcceptedSpecHash:          eligibility.AcceptedSpecHash,
			RealizedEffectiveHash:     eligibility.RealizedEffectiveHash,
			RequestedMechanism:        requestedMechanism,
			RealizedMechanismIdentity: eligibility.RealizedEffectiveHash,
			AdjustmentReasonCodes:     eligibility.AdjustmentReasonCodes,
			LearningEligible:          eligibility.LearningEligible,
			AutomaticChampionEligible: eligibility.AutomaticChampionEligible,
			EligibilityReason:         eligibility.EligibilityReason,
		}
	}
	return out, nil
}

func requestedMechanismForJob(job jobs.ExperimentJob) string {
	spec := payloadMap(job.Config, execution.ExecutionSpecConfigKey)
	requested := payloadMap(spec, "requested_config")
	for _, source := range []map[string]any{requested, job.Config} {
		for _, key := range []string{"mechanism", "mechanism_group", "strategy"} {
			if value := strings.TrimSpace(payloadString(source, key)); value != "" {
				return value
			}
		}
	}
	return ""
}

func executionEvidenceList(evidenceByJob map[string]agents.ExperimentExecutionEvidence) []agents.ExperimentExecutionEvidence {
	out := make([]agents.ExperimentExecutionEvidence, 0, len(evidenceByJob))
	for _, evidence := range evidenceByJob {
		out = append(out, evidence)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].JobID < out[j].JobID })
	if len(out) > 20 {
		out = out[len(out)-20:]
	}
	return out
}

func learningEligibleSummaries(summaries []runs.TrainingRunSummary, evidenceByJob map[string]agents.ExperimentExecutionEvidence) []runs.TrainingRunSummary {
	out := make([]runs.TrainingRunSummary, 0, len(summaries))
	for _, summary := range summaries {
		if evidenceByJob[summary.JobID].LearningEligible {
			out = append(out, summary)
		}
	}
	return out
}

func automaticChampionEligibleSummaries(summaries []runs.TrainingRunSummary, evidenceByJob map[string]agents.ExperimentExecutionEvidence) []runs.TrainingRunSummary {
	out := make([]runs.TrainingRunSummary, 0, len(summaries))
	for _, summary := range summaries {
		if evidenceByJob[summary.JobID].AutomaticChampionEligible {
			out = append(out, summary)
		}
	}
	return out
}

func evaluationsForEligibleSummaries(evaluations []runs.TrainingRunEvaluation, summaries []runs.TrainingRunSummary) []runs.TrainingRunEvaluation {
	eligible := make(map[string]bool, len(summaries))
	for _, summary := range summaries {
		eligible[summary.JobID] = true
	}
	out := make([]runs.TrainingRunEvaluation, 0, len(evaluations))
	for _, evaluation := range evaluations {
		if eligible[evaluation.JobID] {
			out = append(out, evaluation)
		}
	}
	return out
}

func attachPlannerExecutionEvidence(
	current *agents.ExperimentChampion,
	baseline *agents.ExperimentChampion,
	deltas []agents.ExperimentRunDelta,
	evidenceByJob map[string]agents.ExperimentExecutionEvidence,
) {
	for _, champion := range []*agents.ExperimentChampion{current, baseline} {
		if champion == nil {
			continue
		}
		if evidence, ok := evidenceByJob[champion.JobID]; ok {
			evidenceCopy := evidence
			champion.ExecutionEvidence = &evidenceCopy
		}
	}
	for index := range deltas {
		if evidence, ok := evidenceByJob[deltas[index].JobID]; ok {
			evidenceCopy := evidence
			deltas[index].ExecutionEvidence = &evidenceCopy
		}
	}
}
