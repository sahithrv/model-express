package api

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

type policyQueueImpact struct {
	Queued  int `json:"queued"`
	Allowed int `json:"allowed"`
	Pending int `json:"pending"`
	Blocked int `json:"blocked"`
}

type experimentPolicyAdminResponse struct {
	Scope           policies.Scope           `json:"scope"`
	SubjectID       string                   `json:"subject_id"`
	ActiveBinding   *policies.Binding        `json:"active_binding,omitempty"`
	PolicyVersion   *policies.PolicyVersion  `json:"policy_version,omitempty"`
	EffectivePolicy policies.EffectivePolicy `json:"effective_policy"`
	QueueImpact     policyQueueImpact        `json:"queue_impact"`
}

type experimentPolicyUpdateRequest struct {
	Document         policies.PolicyDocument `json:"document"`
	ExpectedRevision *int64                  `json:"expected_revision"`
}

type policyAdminTarget struct {
	Scope     policies.Scope
	SubjectID string
	Context   policies.ScopeContext
}

func (s *Server) listCompatibilityProfiles(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"profiles": policies.BuiltinCompatibilityProfiles()})
}

func (s *Server) getAccountExperimentPolicy(c *gin.Context) {
	s.getExperimentPolicyAdmin(c, policies.ScopeAccount)
}

func (s *Server) updateAccountExperimentPolicy(c *gin.Context) {
	s.updateExperimentPolicyAdmin(c, policies.ScopeAccount)
}

func (s *Server) clearAccountExperimentPolicy(c *gin.Context) {
	s.clearExperimentPolicyAdmin(c, policies.ScopeAccount)
}

func (s *Server) getProjectExperimentPolicy(c *gin.Context) {
	s.getExperimentPolicyAdmin(c, policies.ScopeProject)
}

func (s *Server) updateProjectExperimentPolicy(c *gin.Context) {
	s.updateExperimentPolicyAdmin(c, policies.ScopeProject)
}

func (s *Server) clearProjectExperimentPolicy(c *gin.Context) {
	s.clearExperimentPolicyAdmin(c, policies.ScopeProject)
}

func (s *Server) getDatasetExperimentPolicy(c *gin.Context) {
	s.getExperimentPolicyAdmin(c, policies.ScopeDataset)
}

func (s *Server) updateDatasetExperimentPolicy(c *gin.Context) {
	s.updateExperimentPolicyAdmin(c, policies.ScopeDataset)
}

func (s *Server) clearDatasetExperimentPolicy(c *gin.Context) {
	s.clearExperimentPolicyAdmin(c, policies.ScopeDataset)
}

func (s *Server) getRunExperimentPolicy(c *gin.Context) {
	s.getExperimentPolicyAdmin(c, policies.ScopeRun)
}

func (s *Server) updateRunExperimentPolicy(c *gin.Context) {
	s.updateExperimentPolicyAdmin(c, policies.ScopeRun)
}

func (s *Server) clearRunExperimentPolicy(c *gin.Context) {
	s.clearExperimentPolicyAdmin(c, policies.ScopeRun)
}

func (s *Server) getExperimentPolicyAdmin(c *gin.Context, scope policies.Scope) {
	target, err := s.resolvePolicyAdminTarget(c, scope)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	response, err := s.experimentPolicyAdminResponse(target)
	if err != nil {
		writePolicyPreviewError(c, response.EffectivePolicy, err)
		return
	}
	c.JSON(http.StatusOK, response)
}

func (s *Server) updateExperimentPolicyAdmin(c *gin.Context, scope policies.Scope) {
	target, err := s.resolvePolicyAdminTarget(c, scope)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	var request experimentPolicyUpdateRequest
	if !bindJSON(c, &request) {
		return
	}
	for _, ref := range request.Document.ProfileRefs {
		if _, err := s.store.GetCompatibilityProfile(ref.ID, ref.Version); err != nil {
			writePolicyPreviewError(c, policies.EffectivePolicy{}, &policies.PolicyError{
				Code:    policies.ReasonProfileVersionUnavailable,
				Message: "compatibility profile " + ref.ID + "@" + ref.Version + " is unavailable",
				Findings: []policies.Finding{{
					Code: policies.ReasonProfileVersionUnavailable, ProfileKey: ref.ID, ProfileVersion: ref.Version,
					Origin: "profile_ref", Remediation: "Choose an available immutable compatibility profile version.",
				}},
			})
			return
		}
	}
	version, err := s.store.CreateExperimentPolicyVersion(policies.PolicyVersion{
		OwnerAccountID: target.Context.AccountID, Document: request.Document, CreatedBy: "mission_control",
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if _, err := s.store.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: target.Scope, SubjectID: target.SubjectID, PolicyVersionID: version.ID,
		ExpectedRevision: request.ExpectedRevision, CreatedBy: "mission_control",
	}); err != nil {
		writeStoreError(c, err)
		return
	}
	response, err := s.experimentPolicyAdminResponse(target)
	if err != nil {
		writePolicyPreviewError(c, response.EffectivePolicy, err)
		return
	}
	c.JSON(http.StatusOK, response)
}

func (s *Server) clearExperimentPolicyAdmin(c *gin.Context, scope policies.Scope) {
	target, err := s.resolvePolicyAdminTarget(c, scope)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	revision, err := strconv.ParseInt(strings.TrimSpace(c.Query("expected_revision")), 10, 64)
	if err != nil || revision < 1 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "expected_revision must be a positive integer"})
		return
	}
	if _, err := s.store.ClearExperimentPolicyBinding(target.Scope, target.SubjectID, revision); err != nil {
		writeStoreError(c, err)
		return
	}
	response, err := s.experimentPolicyAdminResponse(target)
	if err != nil {
		writePolicyPreviewError(c, response.EffectivePolicy, err)
		return
	}
	c.JSON(http.StatusOK, response)
}

func (s *Server) listProjectExperimentPolicyAudit(c *gin.Context) {
	projectID := strings.TrimSpace(c.Param("id"))
	if _, err := s.store.GetProject(projectID); err != nil {
		writeStoreError(c, err)
		return
	}
	evaluations, err := s.store.ListExperimentPolicyEvaluations(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"evaluations": evaluations})
}

func (s *Server) resolvePolicyAdminTarget(c *gin.Context, scope policies.Scope) (policyAdminTarget, error) {
	target := policyAdminTarget{Scope: scope, Context: policies.ScopeContext{AccountID: policies.LocalDefaultAccountID}}
	switch scope {
	case policies.ScopeAccount:
		target.SubjectID = policies.LocalDefaultAccountID
	case policies.ScopeProject:
		project, err := s.store.GetProject(strings.TrimSpace(c.Param("id")))
		if err != nil {
			return policyAdminTarget{}, err
		}
		target.SubjectID = project.ID
		target.Context.AccountID, target.Context.ProjectID = project.AccountID, project.ID
	case policies.ScopeDataset:
		dataset, err := s.store.GetDataset(strings.TrimSpace(c.Param("id")))
		if err != nil {
			return policyAdminTarget{}, err
		}
		project, err := s.store.GetProject(dataset.ProjectID)
		if err != nil {
			return policyAdminTarget{}, err
		}
		target.SubjectID = dataset.ID
		target.Context.AccountID, target.Context.ProjectID, target.Context.DatasetID = project.AccountID, project.ID, dataset.ID
		if task, ok := dataset.Profile["task_type"].(string); ok {
			target.Context.Task = strings.TrimSpace(task)
		}
	case policies.ScopeRun:
		job, err := s.store.GetJob(strings.TrimSpace(c.Param("id")))
		if err != nil {
			return policyAdminTarget{}, err
		}
		project, err := s.store.GetProject(job.ProjectID)
		if err != nil {
			return policyAdminTarget{}, err
		}
		target.SubjectID = job.ID
		target.Context.AccountID, target.Context.ProjectID = project.AccountID, project.ID
		target.Context.DatasetID, target.Context.ExperimentJobID = job.DatasetID, job.ID
		target.Context.Task = configString(job.Config, "task_type")
		if spec := payloadMap(job.Config, "execution_spec_v1"); len(spec) > 0 {
			target.Context.Runner = configString(spec, "runner")
		}
	default:
		return policyAdminTarget{}, store.ErrInvalidRequest
	}
	return target, nil
}

func (s *Server) experimentPolicyAdminResponse(target policyAdminTarget) (experimentPolicyAdminResponse, error) {
	response := experimentPolicyAdminResponse{Scope: target.Scope, SubjectID: target.SubjectID}
	bindings, err := s.store.ListActiveExperimentPolicyBindings(target.Context)
	if err != nil {
		return response, err
	}
	for _, binding := range bindings {
		if binding.Scope != target.Scope || binding.SubjectID != target.SubjectID {
			continue
		}
		bindingCopy := binding
		response.ActiveBinding = &bindingCopy
		version, getErr := s.store.GetExperimentPolicyVersion(binding.PolicyVersionID)
		if getErr != nil {
			return response, getErr
		}
		response.PolicyVersion = &version
		break
	}
	response.QueueImpact, err = s.policyQueueImpact(target)
	if err != nil {
		return response, err
	}
	response.EffectivePolicy, err = policies.NewResolver(s.store).Resolve(target.Context)
	return response, err
}

func (s *Server) policyQueueImpact(target policyAdminTarget) (policyQueueImpact, error) {
	impact := policyQueueImpact{}
	projects, err := s.store.ListProjects()
	if err != nil {
		return impact, err
	}
	for _, project := range projects {
		if project.AccountID != target.Context.AccountID || (target.Context.ProjectID != "" && project.ID != target.Context.ProjectID) {
			continue
		}
		projectJobs, err := s.store.ListProjectJobs(project.ID)
		if err != nil {
			return impact, err
		}
		for _, job := range projectJobs {
			if job.Status != jobs.StatusQueued || (target.Scope == policies.ScopeDataset && job.DatasetID != target.SubjectID) || (target.Scope == policies.ScopeRun && job.ID != target.SubjectID) {
				continue
			}
			impact.Queued++
			if job.PolicyEligibilityStatus == jobs.PolicyEligibilityPending {
				impact.Pending++
			}
			evaluation, evaluateErr := s.evaluateJobPolicy(
				job.ProjectID, job.ID, job.Template, job.Config, policyOperationDispatchRun,
			)
			if evaluateErr != nil || evaluation.Decision != policies.DecisionAllowed {
				impact.Blocked++
			} else {
				impact.Allowed++
			}
		}
	}
	return impact, nil
}

func policyConflict(err error) bool {
	return errors.Is(err, policies.ErrRevisionConflict) || errors.Is(err, policies.ErrImmutableConflict)
}
