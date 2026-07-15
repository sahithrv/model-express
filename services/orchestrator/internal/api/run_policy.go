package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

const (
	policyOperationScheduleRun = "schedule_run"
	policyOperationCreateJob   = "create_job"
	policyOperationRetryRun    = "retry_run"
	policyOperationRequeueRun  = "requeue_run"
	policyOperationDispatchRun = "dispatch_run"
)

func policyControlledJobTemplate(template string) bool {
	switch strings.ToLower(strings.TrimSpace(template)) {
	case jobs.TemplateTrainExperiment, jobs.TemplateExportChampion:
		return true
	default:
		return false
	}
}

func (s *Server) evaluateJobPolicy(projectID, jobID, template string, config map[string]any, operation string) (policies.Evaluation, error) {
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return policies.Evaluation{}, err
	}
	datasetID := strings.TrimSpace(configString(config, "dataset_id"))
	planID := strings.TrimSpace(configString(config, "plan_id"))
	task, runner, err := s.jobPolicyTaskRunner(template, config, datasetID)
	if err != nil {
		return policies.Evaluation{}, err
	}
	effective, resolveErr := policies.NewResolver(s.store).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, DatasetID: datasetID,
		ExperimentJobID: strings.TrimSpace(jobID), Task: task, Runner: runner,
	})
	if effective.EffectivePolicyHash == "" {
		return policies.Evaluation{}, resolveErr
	}

	var evaluation policies.Evaluation
	var evaluateErr error
	switch strings.ToLower(strings.TrimSpace(template)) {
	case jobs.TemplateTrainExperiment:
		if resolveErr != nil {
			evaluation, err = policies.EvaluationFromEffectivePolicy(effective, operation, "system", "")
			evaluateErr = resolveErr
			break
		}
		experiment, decodeErr := plannedExperimentFromJobConfig(config)
		if decodeErr != nil {
			return policies.Evaluation{}, decodeErr
		}
		evaluation, evaluateErr = policies.EvaluateProposal(effective, operation, []plans.PlannedExperiment{experiment})
	case jobs.TemplateExportChampion:
		if resolveErr != nil {
			evaluation, err = policies.EvaluationFromEffectivePolicy(effective, operation, "system", "")
			evaluateErr = resolveErr
			break
		}
		format := firstNonEmptyString(configString(config, "format"), configString(config, "export_format"), configString(config, "requested_format"), "onnx")
		evaluation, evaluateErr = policies.EvaluateCapabilityUses(effective, operation, config, []policies.CapabilityUse{{
			Catalog: "export_formats", ID: format, FieldPath: "config.format", Origin: "explicit",
		}})
	default:
		// Non-training orchestration jobs do not consume proposal capabilities.
		// They still receive a dispatch snapshot so the atomic policy-head check
		// can share one correctness boundary for the whole queue.
		evaluation, evaluateErr = policies.EvaluateCapabilityUses(effective, operation, config, nil)
	}
	if err != nil {
		return policies.Evaluation{}, err
	}
	evaluation.ProjectID = project.ID
	evaluation.DatasetID = datasetID
	evaluation.PlanID = planID
	evaluation.JobID = strings.TrimSpace(jobID)
	evaluation.ActorID = "system"
	return evaluation, evaluateErr
}

func (s *Server) jobPolicyTaskRunner(template string, config map[string]any, datasetID string) (string, string, error) {
	spec := payloadMap(config, execution.ExecutionSpecConfigKey)
	task := firstNonEmptyString(payloadString(spec, "task"), configString(config, "task_type"), configString(config, "task"))
	if entry, ok := catalog.Resolve("tasks", task); ok {
		task = entry.ID
	} else {
		task = ""
	}
	var dataset datasets.Dataset
	if datasetID != "" {
		loaded, err := s.store.GetDataset(datasetID)
		if err != nil {
			return "", "", err
		}
		dataset = loaded
		if task == "" {
			if entry, ok := catalog.Resolve("tasks", profileString(dataset.Profile, "task_type")); ok {
				task = entry.ID
			}
		}
	}
	if task == "" {
		if model, ok := catalog.Resolve("models", configString(config, "model")); ok && len(model.Tasks) == 1 {
			task = model.Tasks[0]
		}
	}
	if task == "" && strings.EqualFold(strings.TrimSpace(template), jobs.TemplateTrainExperiment) {
		task = "image_classification"
	}
	runner := payloadString(spec, "runner")
	if runner == "" && strings.EqualFold(strings.TrimSpace(template), jobs.TemplateTrainExperiment) {
		provider := normalizeTrainingProvider(firstNonEmptyString(configString(config, "provider"), "local"))
		resolved, err := executionRunnerFor(provider, task)
		if err != nil {
			return "", "", err
		}
		runner = resolved
	}
	if strings.EqualFold(strings.TrimSpace(template), jobs.TemplateExportChampion) && runner == "" {
		if sourceJobID := configString(config, "champion_job_id"); sourceJobID != "" {
			if source, err := s.store.GetJob(sourceJobID); err == nil {
				sourceSpec := payloadMap(source.Config, execution.ExecutionSpecConfigKey)
				runner = payloadString(sourceSpec, "runner")
				if task == "" {
					task = payloadString(sourceSpec, "task")
				}
			}
		}
	}
	return task, runner, nil
}

func plannedExperimentFromJobConfig(config map[string]any) (plans.PlannedExperiment, error) {
	materialized := copyPayloadMap(config)
	if accepted := payloadMap(payloadMap(config, execution.ExecutionSpecConfigKey), "accepted_config"); len(accepted) > 0 {
		for key, value := range accepted {
			materialized[key] = value
		}
	}
	blob, err := json.Marshal(materialized)
	if err != nil {
		return plans.PlannedExperiment{}, fmt.Errorf("marshal job policy intent: %w", err)
	}
	var experiment plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiment); err != nil {
		return plans.PlannedExperiment{}, fmt.Errorf("decode job policy intent: %w", err)
	}
	return experiment, nil
}

func (s *Server) recordJobPolicyEvaluation(projectID, jobID, template string, config map[string]any, operation string) (policies.Evaluation, error) {
	evaluation, evaluateErr := s.evaluateJobPolicy(projectID, jobID, template, config, operation)
	if evaluation.EffectivePolicyHash == "" {
		return policies.Evaluation{}, evaluateErr
	}
	created, createErr := s.store.CreateExperimentPolicyEvaluation(evaluation)
	if createErr != nil {
		return policies.Evaluation{}, createErr
	}
	if evaluateErr != nil {
		return created, policyErrorWithEvaluation(evaluateErr, created)
	}
	return created, nil
}

func policyReferenceForEvaluation(evaluation policies.Evaluation) policies.PersistenceReference {
	return policies.PersistenceReference{
		EvaluationID: evaluation.ID, EffectivePolicyHash: evaluation.EffectivePolicyHash,
		Status: policyEligibilityForEvaluation(evaluation),
	}
}

func policyEligibilityForEvaluation(evaluation policies.Evaluation) string {
	if evaluation.Decision == policies.DecisionAllowed {
		return jobs.PolicyEligibilityAllowed
	}
	return jobs.PolicyEligibilityBlocked
}

func policyErrorWithEvaluation(err error, evaluation policies.Evaluation) error {
	var policyErr *policies.PolicyError
	if errors.As(err, &policyErr) {
		policyErr.PolicyEvaluationID = evaluation.ID
		policyErr.EffectivePolicyHash = evaluation.EffectivePolicyHash
	}
	return err
}

func (s *Server) recordJobPolicyActivity(job jobs.ExperimentJob, evaluation policies.Evaluation, eventType, message string) {
	reasonCodes := make([]string, 0, len(evaluation.ReasonCodes))
	for _, code := range evaluation.ReasonCodes {
		reasonCodes = append(reasonCodes, string(code))
	}
	sort.Strings(reasonCodes)
	if _, err := s.store.CreateExecutionEvent(job.ProjectID, firstNonEmptyString(job.PlanID, configString(job.Config, "plan_id")), eventType, message, map[string]any{
		"job_id": job.ID, "template": job.Template,
		"policy_evaluation_id":      evaluation.ID,
		"effective_policy_hash":     evaluation.EffectivePolicyHash,
		"policy_eligibility_status": policyEligibilityForEvaluation(evaluation),
		"reason_codes":              reasonCodes,
	}); err != nil {
		log.Printf("record job policy activity failed for job %s: %v", job.ID, err)
	}
}

func (s *Server) recordPlanPolicyActivity(projectID, planID string, evaluation policies.Evaluation, message string) {
	reasonCodes := make([]string, 0, len(evaluation.ReasonCodes))
	for _, code := range evaluation.ReasonCodes {
		reasonCodes = append(reasonCodes, string(code))
	}
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventJobPolicyBlocked, message, map[string]any{
		"policy_evaluation_id":      evaluation.ID,
		"effective_policy_hash":     evaluation.EffectivePolicyHash,
		"policy_eligibility_status": jobs.PolicyEligibilityBlocked,
		"reason_codes":              reasonCodes,
	}); err != nil {
		log.Printf("record plan policy activity failed for plan %s: %v", planID, err)
	}
}

func (s *Server) createJobWithCurrentPolicy(projectID, template string, config map[string]any, operation string) (jobs.ExperimentJob, error) {
	if !policyControlledJobTemplate(template) {
		return s.store.CreateJob(projectID, template, config)
	}
	evaluation, err := s.recordJobPolicyEvaluation(projectID, "", template, config, operation)
	if err != nil {
		placeholder := jobs.ExperimentJob{ProjectID: projectID, PlanID: configString(config, "plan_id"), Template: template, Config: config}
		s.recordJobPolicyActivity(placeholder, evaluation, execution.EventJobPolicyBlocked, "Job creation blocked by the current experiment policy.")
		return jobs.ExperimentJob{}, err
	}
	return s.store.CreateJobWithOptions(projectID, template, config, store.CreateJobOptions{PolicyReference: policyReferenceForEvaluation(evaluation)})
}

func (s *Server) recordPlanSchedulePolicy(plan plans.ExperimentPlan, dataset datasets.Dataset, provider string) (policies.Evaluation, error) {
	project, err := s.store.GetProject(plan.ProjectID)
	if err != nil {
		return policies.Evaluation{}, err
	}
	task := ""
	for _, experiment := range plan.Experiments {
		if model, ok := catalog.Resolve("models", experiment.Model); ok && len(model.Tasks) == 1 {
			if task == "" {
				task = model.Tasks[0]
			}
			if task != model.Tasks[0] {
				return policies.Evaluation{}, fmt.Errorf("%w: plan mixes task-specific models", store.ErrInvalidRequest)
			}
		}
	}
	if task == "" {
		if entry, ok := catalog.Resolve("tasks", profileString(dataset.Profile, "task_type")); ok {
			task = entry.ID
		}
	}
	if task == "" {
		task = "image_classification"
	}
	runner, err := executionRunnerFor(provider, task)
	if err != nil {
		return policies.Evaluation{}, err
	}
	effective, resolveErr := policies.NewResolver(s.store).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, DatasetID: dataset.ID, Task: task, Runner: runner,
	})
	if effective.EffectivePolicyHash == "" {
		return policies.Evaluation{}, resolveErr
	}
	var evaluation policies.Evaluation
	var evaluateErr error
	if resolveErr != nil {
		evaluation, err = policies.EvaluationFromEffectivePolicy(effective, policyOperationScheduleRun, "system", "")
		evaluateErr = resolveErr
	} else {
		evaluation, evaluateErr = policies.EvaluateProposal(effective, policyOperationScheduleRun, plan.Experiments)
	}
	if err != nil {
		return policies.Evaluation{}, err
	}
	evaluation.ProjectID = plan.ProjectID
	evaluation.DatasetID = plan.DatasetID
	evaluation.PlanID = plan.ID
	evaluation.ActorID = "system"
	created, createErr := s.store.CreateExperimentPolicyEvaluation(evaluation)
	if createErr != nil {
		return policies.Evaluation{}, createErr
	}
	if evaluateErr != nil {
		s.recordPlanPolicyActivity(plan.ProjectID, plan.ID, created, "Stored plan execution blocked by the current experiment policy.")
		return created, policyErrorWithEvaluation(evaluateErr, created)
	}
	return created, nil
}
