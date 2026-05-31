# Model Express LLM Decision Quality Upgrade — Implementation PR Plan

**Audience:** solo developer + multiple coding subagents  
**Source reviewed:** uploaded `orchestrator.zip` backend  
**Primary goal:** prevent low-yield autonomous model shopping and force mechanism-level, evidence-backed, backend-ranked experiment decisions.  
**Safety invariant:** the LLM only proposes JSON. The Go backend validates, ranks, stores, schedules, and executes.

---

## Implementation status - 2026-05-31

Implemented in the orchestrator backend:

```text
- Mechanism metadata is preserved when completed jobs are reconstructed into planner experiments.
- Planner contexts now carry project_trajectory_card.
- Project trajectory diagnosis computes completed runs, planner rounds, champion gain, gain per run, recent best delta, mechanism outcomes, blocked mechanisms, warnings, and decision pressure.
- Repeated low-yield architecture_challenge attempts are marked exhausted and blocked.
- ADD_EXPERIMENTS now requires candidate_hypotheses and always runs through FinalizePlannerRecommendation.
- Direct LLM proposed_experiments are draft-only for ADD_EXPERIMENTS; backend ranking populates final proposed_experiments and proposal_mechanisms.
- Dry-run validation uses the same finalizer path as scheduling.
- Autonomous defaults are more conservative: terminal guards default on, useful delta defaults to 0.010, and max follow-up rounds defaults to 3 unless overridden.
- Deterministic replay evals now include the plateau_backbone_lottery fixture.
- Mission Control now has a read-only Decision Quality panel for project trajectory pressure, blocked mechanisms, gain per run, candidate selection/rejection counts, and top rejection reasons.
```

Verified:

```text
go test ./internal/agents ./internal/agents/evals ./internal/api -count=1
go test ./internal/api
go test ./... (services/orchestrator)
npm run build (apps/mission-control)
```

Deferred from this implementation:

```text
- Broader docs outside docs/me_ground_truth.md and this plan.
- Semantic/vector memory, multi-agent debate, tree search, planner fine-tuning, prompt caching, and new queue/workflow infrastructure.
```

---

## 0. Executive summary

Note: the status section above reflects the current repository state after implementation. The original plan text below is retained as design history and may describe gaps that are now closed.

Your current code already contains many strong pieces: deterministic diagnosis, strategy scorecards, candidate hypotheses, rejected option memory, bounded tools, dry-run validation, validation retry, and rich prompt cards. The underwhelming run pattern you observed — roughly `0.701 → 0.743` after 20+ model trainings — is not mainly a lack-of-memory problem yet.

The main implementation gap is that the backend does not yet enforce a project-wide, outcome-aware rule like:

```text
Architecture/backbone exploration is exhausted for this project.
Do not schedule another architecture-only experiment unless new evidence unlocks it.
```

The second gap is that deterministic candidate ranking is currently bypassable. In `ExperimentPlannerAgent.PlanWithTrace`, backend ranking only runs when `proposed_experiments` is empty and `candidate_hypotheses` exists. If the LLM returns `proposed_experiments` directly, the LLM can effectively choose the scheduled batch, subject to validation.

The third gap is evaluation. You need replay fixtures that fail when the planner tries another backbone/model-family batch after a 20-run low-yield plateau.

This plan decomposes the fix into small PRs that can be executed by separate subagents and merged sequentially.

---

## 1. Verified code facts from the uploaded backend

These facts shape the PR plan:

1. `internal/agents/diagnosis.go` has `ComputePlannerDiagnosis(input ExperimentPlannerInput) PlannerDiagnosis`, but it primarily analyzes `input.PlanSummaries`, `input.PlanMetrics`, and source-plan context rather than the entire project trajectory.
2. `ExperimentPlannerInput` already includes rich project history: `PriorPlans`, `PriorJobs`, `PriorSummaries`, `PriorEvaluations`, `StrategyScorecards`, `RejectedStrategyMemory`, `ExistingExperimentSignatures`, and `NoImprovementRounds`.
3. `PlanWithTrace` currently runs `RankPlannerCandidateHypotheses(...)` only when `len(recommendation.ProposedExperiments) == 0 && len(recommendation.CandidateHypotheses) > 0`.
4. `plannerValidateCandidateExperimentsTool(...)` repeats the same conditional ranking behavior.
5. `candidate_ranking.go` has useful penalties for duplicates, tiny-only changes, high cost without evidence, same-mechanism minor variants, and weak mechanism support.
6. `architectureMechanismSupported(diagnosis PlannerDiagnosis)` currently treats plateau or stagnation as sufficient support for architecture challenge:

   ```go
   return diagnosis.UnderfittingScore >= 0.55 ||
       diagnosis.PlateauScore >= 0.60 ||
       diagnosis.ImprovementStagnationScore >= 0.60
   ```

   That is risky after many architecture attempts. Plateau should often trigger a pivot, not another backbone sweep.

7. `plannerMinimumMeaningfulImprovement` is currently `0.005`, which is too permissive for autonomous follow-up loops.
8. `terminalPlannerGuardsEnabled()` defaults to `false`.
9. `addOptionalExperimentConfig(...)` writes `mechanism`, `intervention`, `evidence_used`, and `expected_effect` into job config, but `plannerExperimentFromJob(...)` reconstructs many fields from job config without preserving these mechanism fields.
10. Jobs do not have a direct `PlanID` field; plan linkage is stored in `job.Config["plan_id"]` and `job.Config["experiment_index"]`.

---

## 2. Target behavior

Given a project trajectory like this:

```text
completed training runs: 20+
planner rounds: 6+
first successful Macro-F1: 0.701
current best Macro-F1: 0.743
absolute gain: +0.042
gain per run: about +0.002
repeated families: EfficientNet, MobileNet, ResNet, RegNet
recent champion improvement: negligible
```

The backend should compute:

```json
{
  "decision_pressure": "champion_confirmation_or_non_architecture_pivot",
  "blocked_mechanisms": ["architecture_challenge"],
  "mechanism_outcomes": [
    {
      "mechanism": "architecture_challenge",
      "attempt_count": 15,
      "status": "exhausted",
      "exhaustion_reason": "Repeated backbone/model-family attempts produced low recent champion uplift."
    }
  ]
}
```

Then the system should allow only:

```text
- champion confirmation / seed replicate
- shadow holdout or confirmation evaluation, if supported
- class imbalance / minority-targeting intervention
- label-quality or hard-example audit
- preprocessing / crop / resolution intervention
- SELECT_CHAMPION
- STOP_PROJECT
- WAIT
```

And reject:

```text
- another EfficientNet-only sweep
- another architecture-only challenger
- small epoch/LR/batch-size drift
- high-cost training with expected delta below the project noise/usefulness floor
```

---

## 3. Merge plan overview

| PR | Name | Primary owner | Can run in parallel? | Risk | Main outcome |
|---:|---|---|---|---|---|
| 1 | Preserve mechanism metadata from completed jobs | Backend subagent A | Yes | Low | Historical job ledger keeps `mechanism`, `intervention`, `evidence_used`, `expected_effect`. |
| 2 | Add project trajectory card and mechanism outcome computation | Backend subagent A | After PR 1 preferred | Medium | Backend computes full-project ROI and mechanism exhaustion signals. |
| 3 | Add mechanism exhaustion governor to candidate ranking | Backend subagent B | After PR 2 | Medium | Exhausted mechanisms are rejected or heavily penalized. |
| 4 | Make backend finalizer mandatory for `ADD_EXPERIMENTS` | Backend subagent B | After PR 3 | High | LLM cannot bypass deterministic ranking by returning direct `proposed_experiments`. |
| 5 | Update planner prompt/schema to candidate-first, governor-aware planning | Prompt/schema subagent C | After PR 4 | Medium | LLM proposes candidates; backend selects final experiments. |
| 6 | Make dry-run validation use the same finalizer as scheduling | Tooling subagent C | After PR 4 | Low/Medium | Planner repair loop receives real backend rejection reasons. |
| 7 | Safer autonomous defaults and dynamic experiment caps | Backend subagent A/B | After PR 2 | Medium | Autonomous loops become harder to run indefinitely after weak improvement. |
| 8 | Replay eval harness and plateau fixture | Eval subagent D | After PR 4, can begin earlier | Medium | A regression test catches the exact failure mode from the screenshot. |
| 9 | Docs and optional Mission Control decision-quality panel | Docs/UI subagent E | After PRs 2–8 | Low | Future maintainers can see why plans are accepted/rejected. |

Recommended merge order:

```text
PR 1 → PR 2 → PR 3 → PR 4 → PR 5 → PR 6 → PR 7 → PR 8 → PR 9
```

If using multiple subagents, let PR 1/2, PR 3/4, PR 5/6, and PR 8 be developed on separate branches, but merge them in the order above.

---

## 4. Shared interfaces to coordinate subagents

Add these types early so subagents can build against stable names.

### 4.1 `PlannerProjectTrajectoryCard`

Location:

```text
internal/agents/diagnosis.go
```

Suggested structs:

```go
type PlannerProjectTrajectoryCard struct {
    CompletedTrainingRuns       int                       `json:"completed_training_runs"`
    CompletedPlannerRounds      int                       `json:"completed_planner_rounds"`
    FirstSuccessfulScore        float64                   `json:"first_successful_score"`
    CurrentChampionScore        float64                   `json:"current_champion_score"`
    AbsoluteChampionGain        float64                   `json:"absolute_champion_gain"`
    GainPerCompletedRun         float64                   `json:"gain_per_completed_run"`
    RecentBestDelta             float64                   `json:"recent_best_delta"`
    MinimumUsefulDelta          float64                   `json:"minimum_useful_delta"`
    NoImprovementRounds         int                       `json:"no_improvement_rounds"`
    DecisionPressure            string                    `json:"decision_pressure"`
    MechanismOutcomes           []PlannerMechanismOutcome `json:"mechanism_outcomes"`
    BlockedMechanisms           []string                  `json:"blocked_mechanisms"`
    Warnings                    []string                  `json:"warnings"`
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
    Status              string   `json:"status"` // unexplored | active | promising | exhausted | blocked
    ExhaustionReason    string   `json:"exhaustion_reason,omitempty"`
    AllowedNextOnlyWith []string `json:"allowed_next_only_with,omitempty"`
}
```

Add to `ExperimentPlannerInput`:

```go
ProjectTrajectory PlannerProjectTrajectoryCard
```

Add to `PlannerContextSnapshot`:

```go
ProjectTrajectoryCard PlannerProjectTrajectoryCard `json:"project_trajectory_card"`
```

### 4.2 Finalizer API

Location:

```text
internal/agents/candidate_ranking.go
```

```go
func FinalizePlannerRecommendation(
    input ExperimentPlannerInput,
    recommendation ExperimentPlanningRecommendation,
) (ExperimentPlanningRecommendation, error)
```

Contract:

```text
- Non-ADD_EXPERIMENTS decisions pass through unchanged.
- ADD_EXPERIMENTS must include candidate_hypotheses.
- Backend ranking is authoritative.
- Backend populates proposed_experiments and proposal_mechanisms.
- If no candidate survives, return an error.
```

### 4.3 Mechanism governor helpers

Location:

```text
internal/agents/candidate_ranking.go
```

```go
func mechanismExhausted(input ExperimentPlannerInput, mechanism string) (bool, string)
func projectDecisionPressure(input ExperimentPlannerInput) string
func effectiveMaxPlannerExperiments(input ExperimentPlannerInput) int
```

### 4.4 Replay eval package

Location:

```text
internal/agents/evals/
```

```go
type PlannerReplayFixture struct {
    Name     string                 `json:"name"`
    Input    map[string]any         `json:"input"`
    Expected PlannerReplayExpected  `json:"expected"`
}

type PlannerReplayExpected struct {
    ForbiddenMechanisms             []string `json:"forbidden_mechanisms"`
    AllowedDecisions                []string `json:"allowed_decisions"`
    AllowedAddExperimentMechanisms  []string `json:"allowed_add_experiment_mechanisms"`
    MaxSelectedExperiments          int      `json:"max_selected_experiments"`
}

type PlannerReplayScores struct {
    SchemaValid                     bool `json:"schema_valid"`
    BackendValidationPassed         bool `json:"backend_validation_passed"`
    CandidateRankingApplied         bool `json:"candidate_ranking_applied"`
    AvoidedBlockedMechanisms        bool `json:"avoided_blocked_mechanisms"`
    AvoidedDuplicateSignatures      bool `json:"avoided_duplicate_signatures"`
    AvoidedArchitectureAfterPlateau bool `json:"avoided_architecture_after_plateau"`
    ExpectedValueAboveFloor         bool `json:"expected_value_above_floor"`
    EvidencePresent                 bool `json:"evidence_present"`
    SelectedExperimentCountOK       bool `json:"selected_experiment_count_ok"`
}
```

---

# PR 1 — Preserve mechanism metadata from completed jobs

## Why this PR exists

Your execution path already stores useful LLM mechanism metadata into job config via `addOptionalExperimentConfig(...)`:

```text
mechanism
intervention
evidence_used
expected_effect
```

But `plannerExperimentFromJob(...)` does not currently reconstruct those fields. This weakens the completed experiment ledger and makes mechanism-outcome analysis less reliable.

## Files to modify

```text
internal/agents/experiment_planner_llm.go
internal/agents/experiment_planner_llm_test.go
```

## Implementation steps

1. In `plannerExperimentFromJob(job jobs.ExperimentJob)`, populate:

   ```go
   Mechanism:      plannerConfigString(config, "mechanism"),
   Intervention:   plannerConfigString(config, "intervention"),
   EvidenceUsed:   plannerConfigStringSlice(config, "evidence_used"),
   ExpectedEffect: plannerConfigString(config, "expected_effect"),
   ```

2. Add helper:

   ```go
   func plannerConfigStringSlice(config map[string]any, key string) []string {
       raw, ok := config[key]
       if !ok || raw == nil {
           return nil
       }
       switch values := raw.(type) {
       case []string:
           return nonEmptyStrings(values)
       case []any:
           out := []string{}
           for _, value := range values {
               if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
                   out = append(out, strings.TrimSpace(text))
               }
           }
           return out
       case string:
           if strings.TrimSpace(values) == "" {
               return nil
           }
           return []string{strings.TrimSpace(values)}
       default:
           return nil
       }
   }
   ```

3. Preserve existing fallback inference:

   ```go
   if strings.TrimSpace(experiment.Mechanism) == "" {
       experiment.Mechanism = plannerExperimentMechanism(experiment, "")
   }
   ```

## Tests

Add:

```go
func TestPlannerExperimentFromJobPreservesMechanismMetadata(t *testing.T) {}
func TestPlannerExperimentFromJobInfersMechanismWhenMetadataMissing(t *testing.T) {}
```

Test fixture:

```go
job := jobs.ExperimentJob{
    ID: "job_1",
    Config: map[string]any{
        "model": "efficientnet_b2",
        "mechanism": "class_imbalance",
        "intervention": "weighted_cross_entropy",
        "evidence_used": []any{"per_class_errors", "macro_f1_gap"},
        "expected_effect": "Improve minority-class recall.",
    },
}
```

Expected:

```text
experiment.Mechanism == "class_imbalance"
experiment.Intervention == "weighted_cross_entropy"
experiment.EvidenceUsed contains refs
experiment.ExpectedEffect not empty
```

## Definition of done

```text
go test ./internal/agents
```

No behavior change except richer reconstructed ledger fields.

---

# PR 2 — Add project trajectory card and mechanism outcome computation

## Why this PR exists

`ComputePlannerDiagnosis(...)` is source-plan focused. For autonomous loops, you need a project-level card that sees all prior plans/jobs/summaries and tells the planner/backend whether a mechanism has become low-yield.

## Files to modify

```text
internal/agents/diagnosis.go
internal/agents/experiment_planner_llm.go
internal/agents/experiment_planner_llm_test.go
internal/api/handlers.go
internal/api/handlers_test.go
```

## Implementation steps

### Step 1 — Add types

Add `PlannerProjectTrajectoryCard` and `PlannerMechanismOutcome` as defined in section 4.

Add to `ExperimentPlannerInput`:

```go
ProjectTrajectory PlannerProjectTrajectoryCard
```

Add to `PlannerContextSnapshot`:

```go
ProjectTrajectoryCard PlannerProjectTrajectoryCard `json:"project_trajectory_card"`
```

Add this in the prompt context builder where other cards are populated:

```go
ProjectTrajectoryCard: input.ProjectTrajectory,
```

### Step 2 — Add builder

In `diagnosis.go`:

```go
func ComputeProjectTrajectoryDiagnosis(input ExperimentPlannerInput) PlannerProjectTrajectoryCard
```

Use:

```text
input.PriorPlans
input.PriorJobs
input.PriorSummaries
input.CurrentChampion
input.NoImprovementRounds
input.MinimumMeaningfulImprovement
```

Algorithm:

1. Normalize target metric from `input.SourcePlan.TargetMetric`.
2. Sort completed successful summaries by completed time if available, otherwise created time/job order.
3. Compute:

   ```text
   CompletedTrainingRuns
   CompletedPlannerRounds
   FirstSuccessfulScore
   CurrentChampionScore
   AbsoluteChampionGain
   GainPerCompletedRun
   RecentBestDelta
   MinimumUsefulDelta = max(input.MinimumMeaningfulImprovement, 0.010)
   NoImprovementRounds
   ```

4. Build `jobID -> summary` and `jobID -> reconstructed experiment` using `plannerExperimentFromJob(job)`.
5. Infer mechanism from stored mechanism first, then fallback to `inferExperimentMechanismTaxonomy(experiment)`.
6. Aggregate by mechanism:

   ```text
   AttemptCount
   PlanCount
   BestScore
   BestDeltaVsPrior
   RecentBestDelta
   TotalCostUSD
   TotalRuntimeSeconds
   Status
   ExhaustionReason
   ```

7. Return blocked mechanisms and warnings.

### Step 3 — Initial exhaustion rules

Ship deterministic rules first. Keep them simple and testable.

```go
func mechanismStatusFromOutcome(outcome PlannerMechanismOutcome, trajectory PlannerProjectTrajectoryCard) (status string, reason string) {
    usefulDelta := math.Max(trajectory.MinimumUsefulDelta, 0.010)

    if outcome.Mechanism == "architecture_challenge" &&
        outcome.AttemptCount >= 8 &&
        outcome.RecentBestDelta < usefulDelta {
        return "exhausted", "Repeated architecture/backbone attempts produced low recent champion uplift."
    }

    if outcome.AttemptCount >= 6 &&
        outcome.RecentBestDelta < usefulDelta &&
        outcome.BestDeltaVsPrior < 0.015 {
        return "exhausted", "Repeated attempts produced low recent champion uplift."
    }

    if outcome.BestDeltaVsPrior >= usefulDelta {
        return "promising", "Mechanism has produced useful champion uplift."
    }

    return "active", "Mechanism is not yet exhausted."
}
```

### Step 4 — Decision pressure

Add decision pressure rules:

```go
func computeDecisionPressure(card PlannerProjectTrajectoryCard) string {
    if card.CompletedTrainingRuns >= 15 && card.GainPerCompletedRun > 0 && card.GainPerCompletedRun < 0.003 {
        return "champion_confirmation_or_non_architecture_pivot"
    }
    if card.NoImprovementRounds >= 2 {
        return "non_exhausted_mechanism_or_stop"
    }
    return "normal"
}
```

### Step 5 — Wire into API input builder

In `internal/api/handlers.go`, in `buildExperimentPlannerInput(...)`:

1. Keep current `partialInput` and `deterministicDiagnosis` flow.
2. Build the full input as today.
3. Compute trajectory using the full input or a near-full input.
4. Set `ProjectTrajectory` before returning.

Pseudocode:

```go
input := agents.ExperimentPlannerInput{
    // existing fields...
    MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovementFromEnv(automationSettings.AgentMode),
    PriorPlans: projectPlans,
    PriorJobs: projectJobs,
    PriorSummaries: summaries,
    PriorEvaluations: evaluations,
}
input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
return input, true, nil
```

If `automationSettings.AgentMode` is not available in `buildExperimentPlannerInput`, keep the default in this PR and add mode-specific thresholds in PR 7.

## Tests

Add:

```go
func TestComputeProjectTrajectoryDiagnosisCountsAllPriorRuns(t *testing.T) {}
func TestComputeProjectTrajectoryDiagnosisComputesGainPerRun(t *testing.T) {}
func TestComputeProjectTrajectoryDiagnosisExhaustsArchitectureAfterLowYieldSweep(t *testing.T) {}
func TestPlannerContextIncludesProjectTrajectoryCard(t *testing.T) {}
func TestBuildExperimentPlannerInputPopulatesProjectTrajectory(t *testing.T) {}
```

Use a fixture approximating the screenshot:

```text
22 completed runs
first successful score 0.701
current champion score 0.743
15+ architecture_challenge attempts
recent architecture delta 0.000
```

Expected:

```text
ProjectTrajectory.CompletedTrainingRuns >= 20
ProjectTrajectory.AbsoluteChampionGain around 0.042
ProjectTrajectory.GainPerCompletedRun around 0.002
ProjectTrajectory.BlockedMechanisms includes architecture_challenge
architecture_challenge status == exhausted
DecisionPressure == champion_confirmation_or_non_architecture_pivot
```

## Definition of done

```text
go test ./internal/agents ./internal/api
```

The planner context snapshot now includes a compact project-level trajectory card.

---

# PR 3 — Add mechanism exhaustion governor to candidate ranking

## Why this PR exists

A card alone is advisory. The backend must enforce it.

This PR makes exhausted mechanisms hard-rejected or heavily penalized before scheduling.

## Files to modify

```text
internal/agents/candidate_ranking.go
internal/agents/experiment_planner_llm_test.go
```

## Implementation steps

### Step 1 — Add helper

```go
func mechanismExhausted(input ExperimentPlannerInput, mechanism string) (bool, string) {
    normalized := normalizeMechanism(mechanism)
    for _, blocked := range input.ProjectTrajectory.BlockedMechanisms {
        if normalizeMechanism(blocked) == normalized {
            return true, "mechanism is blocked by project trajectory governor"
        }
    }
    for _, outcome := range input.ProjectTrajectory.MechanismOutcomes {
        if normalizeMechanism(outcome.Mechanism) == normalized && strings.EqualFold(outcome.Status, "exhausted") {
            if strings.TrimSpace(outcome.ExhaustionReason) != "" {
                return true, outcome.ExhaustionReason
            }
            return true, "mechanism is exhausted by project trajectory governor"
        }
    }
    return false, ""
}
```

### Step 2 — Enforce in `scorePlannerCandidate`

Near the top of `scorePlannerCandidate`, after shape and mechanism validation:

```go
if exhausted, reason := mechanismExhausted(input, candidate.Mechanism); exhausted {
    ranking.Rejected = true
    ranking.Score = 0
    ranking.Reasons = append(ranking.Reasons, reason)
    return ranking
}
```

### Step 3 — Fix architecture support logic

Current logic allows plateau/stagnation to support architecture challenge. Replace it with an input-aware version.

Change:

```go
func architectureMechanismSupported(diagnosis PlannerDiagnosis) bool
```

to:

```go
func architectureMechanismSupported(input ExperimentPlannerInput) bool {
    if exhausted, _ := mechanismExhausted(input, "architecture_challenge"); exhausted {
        return false
    }
    d := input.DeterministicDiagnosis
    if d.UnderfittingScore >= 0.60 {
        return true
    }
    // Plateau only supports architecture if there is also underfitting/capacity evidence.
    return d.PlateauScore >= 0.70 && d.UnderfittingScore >= 0.45
}
```

Update callers:

```go
diagnosisMatchesMechanism(input ExperimentPlannerInput, mechanism string, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool
```

Instead of passing only `PlannerDiagnosis`.

### Step 4 — Tighten plateau bonus

In `candidateDiagnosisAlignment`, avoid rewarding architecture/model shopping from plateau alone. Keep the plateau bonus for scheduler, augmentation, preprocessing, image/resolution, but not generic `model` unless architecture is still supported.

Change the text check from:

```go
containsAnyText(text, "model", "family", "scheduler", "augmentation", "preprocess", "image")
```

to something like:

```go
containsAnyText(text, "scheduler", "augmentation", "preprocess", "image", "resolution", "crop", "sampler")
```

## Tests

Add:

```go
func TestCandidateRankingRejectsExhaustedMechanism(t *testing.T) {}
func TestCandidateRankingRejectsExhaustedArchitectureChallenge(t *testing.T) {}
func TestArchitectureMechanismNotSupportedByPlateauAloneAfterExhaustion(t *testing.T) {}
func TestArchitectureMechanismSupportedByStrongUnderfittingWhenNotExhausted(t *testing.T) {}
func TestPlateauStillSupportsNonArchitecturePivot(t *testing.T) {}
```

## Definition of done

```text
go test ./internal/agents
```

In plateau fixture, architecture-only candidates should be rejected with a reason containing `exhausted` or `project trajectory governor`.

---

# PR 4 — Make backend finalizer mandatory for `ADD_EXPERIMENTS`

## Why this PR exists

Current behavior:

```go
if len(recommendation.ProposedExperiments) == 0 && len(recommendation.CandidateHypotheses) > 0 {
    rankings, selected, mechanisms := RankPlannerCandidateHypotheses(...)
    recommendation.ProposedExperiments = selected
}
```

This means direct `proposed_experiments` can bypass deterministic ranking.

New behavior:

```text
LLM proposes candidate_hypotheses.
Backend ranks candidates.
Backend populates proposed_experiments.
Backend validates.
Backend schedules.
```

## Files to modify

```text
internal/agents/candidate_ranking.go
internal/agents/experiment_planner_llm.go
internal/agents/planner_information_tools.go
internal/agents/experiment_planner_llm_test.go
```

## Implementation steps

### Step 1 — Add finalizer

In `candidate_ranking.go`:

```go
func FinalizePlannerRecommendation(input ExperimentPlannerInput, recommendation ExperimentPlanningRecommendation) (ExperimentPlanningRecommendation, error) {
    if !strings.EqualFold(strings.TrimSpace(recommendation.DecisionType), decisions.TypeAddExperiments) {
        return recommendation, nil
    }

    if len(recommendation.CandidateHypotheses) == 0 {
        return recommendation, fmt.Errorf("ADD_EXPERIMENTS requires candidate_hypotheses; backend selects final experiments")
    }

    maxExperiments := effectiveMaxPlannerExperiments(input)
    rankings, selected, mechanisms := RankPlannerCandidateHypotheses(input, recommendation.CandidateHypotheses, maxExperiments)

    recommendation.CandidateRankings = rankings
    recommendation.ProposedExperiments = selected
    recommendation.ProposalMechanisms = mechanisms

    if strings.TrimSpace(recommendation.PlanningMode) == "" && len(selected) > 0 {
        for _, ranking := range rankings {
            if ranking.Selected && strings.TrimSpace(ranking.PlanningMode) != "" {
                recommendation.PlanningMode = ranking.PlanningMode
                break
            }
        }
    }

    if len(selected) == 0 {
        return recommendation, fmt.Errorf("no candidate survived backend ranking and mechanism governor validation")
    }

    return recommendation, nil
}
```

Import `decisions` in `candidate_ranking.go`.

### Step 2 — Add `effectiveMaxPlannerExperiments`

```go
func effectiveMaxPlannerExperiments(input ExperimentPlannerInput) int {
    base := maxPlannerExperiments(input.MaxExperiments)
    pressure := strings.ToLower(strings.TrimSpace(input.ProjectTrajectory.DecisionPressure))
    if pressure == "champion_confirmation_or_non_architecture_pivot" {
        return minInt(base, 2)
    }
    if input.NoImprovementRounds >= 2 {
        return minInt(base, 2)
    }
    if input.FollowUpRound >= 4 {
        return minInt(base, 2)
    }
    return base
}
```

Add a local `minInt` helper if none exists.

### Step 3 — Use finalizer in `PlanWithTrace`

Replace conditional ranking block with:

```go
recommendation, err = FinalizePlannerRecommendation(input, recommendation)
if err != nil {
    trace.ValidationStatus = memory.InvocationValidationInvalid
    trace.ValidationError = err.Error()
    trace.Recommendation = recommendation
    return trace, err
}
```

Then call:

```go
if err := validateExperimentPlanningRecommendation(recommendation, effectiveMaxPlannerExperiments(input)); err != nil {
    ...
}
```

### Step 4 — Decide migration mode

You have two rollout options.

Recommended first rollout:

```text
Phase 1: require candidate_hypotheses for autonomous mode only.
Phase 2: require candidate_hypotheses for all ADD_EXPERIMENTS.
```

Simpler version:

```text
Require candidate_hypotheses for all ADD_EXPERIMENTS immediately.
```

Because your prompt already requests candidate hypotheses, the simpler version should be fine, but it may require test updates.

## Tests

Add:

```go
func TestFinalizePlannerRecommendationRequiresCandidatesForAddExperiments(t *testing.T) {}
func TestFinalizePlannerRecommendationIgnoresDirectProposedExperiments(t *testing.T) {}
func TestFinalizePlannerRecommendationPopulatesBackendSelectedExperiments(t *testing.T) {}
func TestPlanWithTraceAlwaysAppliesBackendFinalizer(t *testing.T) {}
func TestEffectiveMaxPlannerExperimentsShrinksUnderDecisionPressure(t *testing.T) {}
```

Important fixture:

```text
LLM output contains 4 direct architecture-only proposed_experiments.
LLM output also contains candidate_hypotheses with one class_imbalance candidate.
ProjectTrajectory blocks architecture_challenge.
```

Expected:

```text
final recommendation does not contain the direct architecture-only proposals
final selected experiment is the non-blocked ranked candidate
candidate_rankings show rejected architecture candidates
```

## Definition of done

```text
go test ./internal/agents
```

No scheduled `ADD_EXPERIMENTS` plan can bypass backend candidate ranking.

---

# PR 5 — Update planner prompt/schema to candidate-first, governor-aware planning

## Why this PR exists

Once the backend finalizer is authoritative, the prompt should stop asking the LLM to directly author final scheduled experiments. It should ask for a candidate set with mechanism, evidence, expected effect, and executable config.

## Files to modify

```text
internal/agents/experiment_planner_llm.go
internal/agents/experiment_planner_llm_test.go
internal/llm/client.go  # only if schema/structured-output support needs adjustment
```

## Schema changes

Add optional fields to `ExperimentPlanningRecommendation`:

```go
PrimaryMechanism   string                    `json:"primary_mechanism,omitempty"`
GovernorCompliance PlannerGovernorCompliance `json:"governor_compliance,omitempty"`
```

Add:

```go
type PlannerGovernorCompliance struct {
    BlockedMechanismsSeen      []string `json:"blocked_mechanisms_seen"`
    AvoidedBlockedMechanisms   bool     `json:"avoided_blocked_mechanisms"`
    WhyAllowedToContinue       string   `json:"why_allowed_to_continue,omitempty"`
    ExpectedValueJustification string   `json:"expected_value_justification,omitempty"`
}
```

You can validate this later. Do not block this PR on full validator enforcement.

## Prompt changes

In `experimentPlannerJSONRequest(...)`, add a concise contract:

```text
You must choose mechanisms before concrete models/configs.
For ADD_EXPERIMENTS, provide candidate_hypotheses. The backend will rank candidates and populate final proposed_experiments.
Do not rely on direct proposed_experiments to force scheduling; they are treated as draft only.
If project_trajectory_card marks a mechanism as exhausted, do not propose that mechanism.
If architecture_challenge is exhausted, do not propose another backbone/model-family-only experiment.
Plateau alone is not sufficient evidence for architecture_challenge after repeated architecture attempts.
When decision_pressure is champion_confirmation_or_non_architecture_pivot, choose one of: champion confirmation, label/data diagnosis, class imbalance intervention, preprocessing/resolution intervention, SELECT_CHAMPION, STOP_PROJECT, or WAIT.
```

Add an invalid-pattern block:

```text
Invalid shallow proposals:
- Trying another backbone only because it exists in the model catalog.
- Repeating EfficientNet/ResNet/MobileNet variants after architecture_challenge is exhausted.
- Changing epochs, learning rate, or batch size without training-dynamics evidence.
- Proposing high-cost work when expected_metric_impact is below project_trajectory_card.minimum_useful_delta.
```

## Validation change

In `validateExperimentPlanningRecommendation(...)`, for `ADD_EXPERIMENTS`, require:

```go
if len(recommendation.CandidateHypotheses) == 0 {
    return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing candidate_hypotheses")
}
```

This becomes safe after PR 4.

## Tests

Add/update:

```go
func TestPlannerPromptMentionsProjectTrajectoryGovernor(t *testing.T) {}
func TestValidateRecommendationRequiresCandidateHypotheses(t *testing.T) {}
func TestGovernorComplianceShapeAccepted(t *testing.T) {}
```

If tests snapshot the prompt, update snapshots.

## Definition of done

```text
go test ./internal/agents ./internal/llm
```

Planner output is mechanism-first and candidate-first.

---

# PR 6 — Make dry-run validation use the same finalizer as scheduling

## Why this PR exists

The dry-run validation tool should expose the same backend reality that scheduling will enforce. Otherwise the LLM repair loop can pass dry-run but fail later, or vice versa.

## Files to modify

```text
internal/agents/planner_information_tools.go
internal/agents/experiment_planner_llm_test.go
```

## Implementation steps

In `plannerValidateCandidateExperimentsTool(...)`, replace the conditional ranking block with:

```go
recommendation, finalizeErr := FinalizePlannerRecommendation(input, recommendation)
```

Build the result from the finalized recommendation.

If `finalizeErr != nil`, return a dry-run response like:

```json
{
  "validation_status": "invalid",
  "valid": false,
  "validation_error": "no candidate survived backend ranking and mechanism governor validation",
  "candidate_rankings": [...],
  "would_write_rows": false,
  "would_schedule_jobs": false
}
```

Then still run `validateExperimentPlanningRecommendation(...)` on finalized recommendation when finalization succeeds.

## Tests

Add:

```go
func TestDryRunValidationUsesBackendFinalizer(t *testing.T) {}
func TestDryRunValidationReportsExhaustedMechanismRejection(t *testing.T) {}
func TestDryRunValidationNeverSchedulesJobs(t *testing.T) {}
```

## Definition of done

```text
go test ./internal/agents
```

The planner repair loop receives the same rejection reasons that scheduling would enforce.

---

# PR 7 — Safer autonomous defaults and dynamic experiment caps

## Why this PR exists

Your autonomous mode should be more conservative than propose/review mode.

Current code:

```go
plannerMinimumMeaningfulImprovement = 0.005
terminalPlannerGuardsEnabled() defaults false
plannerDefaultMaxFollowUpRounds = 10
```

For a local autonomous loop, those defaults are too permissive.

## Files to modify

```text
internal/api/handlers.go
internal/api/handlers_test.go
```

## Implementation steps

### Step 1 — Add env-based useful delta

Replace direct use of `plannerMinimumMeaningfulImprovement` where practical with:

```go
func plannerMinimumMeaningfulImprovementFromEnv(agentMode string) float64 {
    envKey := "MODEL_EXPRESS_PLANNER_MIN_MEANINGFUL_DELTA"
    if value := strings.TrimSpace(os.Getenv(envKey)); value != "" {
        parsed, err := strconv.ParseFloat(value, 64)
        if err == nil && parsed > 0 {
            return parsed
        }
    }
    if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
        return 0.010
    }
    return plannerMinimumMeaningfulImprovement
}
```

If importing `llm` into that section is awkward because it is already imported, use the string constant available in the file.

### Step 2 — Make terminal guards enabled in autonomous mode

Instead of only:

```go
func terminalPlannerGuardsEnabled() bool {
    return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}
```

Add:

```go
func terminalPlannerGuardsEnabledForMode(agentMode string) bool {
    if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
        return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", true)
    }
    return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}
```

Then gradually replace planner-autonomy call sites with the mode-aware helper.

Start with:

```text
runExperimentPlannerAfterTrainingJob
runExperimentPlannerWithBackendValidationRetry
applyExperimentPlannerStopCriteria
experimentPlannerStopSignals
plannerFollowUpStopReason
validateFollowUpPlanCanExecute
ensureFollowUpPlan
```

### Step 3 — Lower follow-up default for autonomous mode

Current default is `10`. Keep env override but use a safer autonomous default:

```text
autonomous default: 3 or 4
manual/propose default: 10
```

If changing global default risks tests, add a mode-specific helper and only use it in autonomous paths.

## Tests

Add/update:

```go
func TestAutonomousTerminalPlannerGuardsDefaultEnabled(t *testing.T) {}
func TestProposeModeTerminalPlannerGuardsDefaultAdvisory(t *testing.T) {}
func TestAutonomousMinimumMeaningfulImprovementDefaultsToOnePoint(t *testing.T) {}
func TestEnvCanOverrideMinimumMeaningfulImprovement(t *testing.T) {}
func TestAutonomousMaxFollowUpRoundsUsesSaferDefault(t *testing.T) {}
```

Existing test likely to update:

```text
TestExperimentPlannerStopCriteriaKeepsAddExperimentsByDefault
```

Keep that behavior only for non-autonomous/propose mode.

## Definition of done

```text
go test ./internal/api
```

Autonomous runs are safer by default, while manual/propose workflows remain flexible.

---

# PR 8 — Replay eval harness and plateau fixture

## Why this PR exists

You need a regression test that would have caught the screenshot failure before spending 20+ training runs.

This PR adds scenario-level tests over backend validation/finalization logic. Start deterministic; do not call an LLM in CI.

## Files to add

```text
internal/agents/evals/replay.go
internal/agents/evals/scorers.go
internal/agents/evals/replay_test.go
internal/agents/evals/testdata/plateau_backbone_lottery.json
```

## Files to modify

```text
internal/agents/candidate_ranking.go       # expose helpers if needed
internal/agents/experiment_planner_llm.go  # expose validator/finalizer as needed
```

## Fixture

Create `internal/agents/evals/testdata/plateau_backbone_lottery.json`:

```json
{
  "name": "plateau_backbone_lottery",
  "input_summary": {
    "first_successful_macro_f1": 0.701,
    "current_best_macro_f1": 0.743,
    "completed_training_runs": 22,
    "planner_rounds": 6,
    "attempted_models": [
      "efficientnet_b0",
      "efficientnet_b1",
      "efficientnet_b2",
      "mobilenet_v3_large",
      "resnet18",
      "regnet_y_400mf"
    ],
    "dominant_mechanism": "architecture_challenge"
  },
  "expected": {
    "forbidden_mechanisms": ["architecture_challenge"],
    "allowed_decisions": [
      "ADD_EXPERIMENTS",
      "SELECT_CHAMPION",
      "STOP_PROJECT",
      "WAIT"
    ],
    "allowed_add_experiment_mechanisms": [
      "class_imbalance",
      "minority_targeting",
      "resolution_crop",
      "augmentation_auto",
      "augmentation_mixed_sample",
      "label_noise_audit",
      "hard_example_audit",
      "deployment_latency",
      "stop_select_champion"
    ],
    "max_selected_experiments": 2
  }
}
```

## Scorers

Add deterministic scorers:

```go
func ScorePlannerRecommendation(input agents.ExperimentPlannerInput, recommendation agents.ExperimentPlanningRecommendation, expected PlannerReplayExpected) PlannerReplayScores
```

Scorer checks:

```text
schema valid
backend validation passed
candidate ranking applied
no forbidden mechanism selected
no duplicate signatures selected
no architecture_challenge selected after plateau
expected delta above floor unless stop/select/wait
selected experiment count <= expected max
```

## Tests

Add:

```go
func TestReplayPlateauBackboneLotteryRejectsArchitectureChallenge(t *testing.T) {}
func TestReplayPlateauBackboneLotteryAllowsClassImbalancePivot(t *testing.T) {}
func TestReplayPlateauBackboneLotteryAllowsStopOrSelect(t *testing.T) {}
func TestReplayFailsWhenCandidateRankingBypassed(t *testing.T) {}
```

Test cases:

1. Bad recommendation: only architecture candidates. Expected finalizer error or zero selected.
2. Good recommendation: class imbalance candidate. Expected selected.
3. Good recommendation: `SELECT_CHAMPION`. Expected pass.
4. Good recommendation: `WAIT` with reason. Expected pass.

## Definition of done

```text
go test ./internal/agents ./internal/agents/evals
```

A plateau/model-lottery regression test exists and passes.

---

# PR 9 — Docs and optional Mission Control visibility

## Why this PR exists

Once the backend starts rejecting low-value plans, Mission Control and docs should explain why. Otherwise the system will feel arbitrary.

## Files to modify

Docs:

```text
docs/me_ground_truth.md
docs/llm_agent_decision_context.md
docs/llm_decision_quality_report.md
docs/agents_context/llm_decision_intelligence/context.md
docs/plans/model_express_agentic_upgrade_roadmap.md
```

Optional UI:

```text
apps/mission-control/src/App.tsx
```

## Docs updates

Add sections:

```text
- LLM proposal/backend finalization boundary
- Mechanism exhaustion governor
- Project trajectory card
- Candidate-first planner contract
- Autonomous stop/champion-confirmation defaults
- Replay eval fixtures
```

Add a short invariant:

```text
The planner may propose candidates, but the backend selects final experiments. Direct LLM proposed_experiments are treated as draft-only for ADD_EXPERIMENTS.
```

## Optional UI panel

Add a compact decision-quality panel:

```text
Decision pressure: champion_confirmation_or_non_architecture_pivot
Blocked mechanisms: architecture_challenge
Completed training runs: 22
Gain per run: +0.002
Recent best delta: +0.000
Selected candidates: 1 / 5
Rejected candidates: 4
Top rejection reason: mechanism exhausted by project trajectory governor
```

## Definition of done

```text
Docs explain why the agent will stop/pivot instead of running more model-family experiments.
UI, if touched, displays trajectory/governor metadata without adding execution authority.
```

---

## 5. Subagent work packets

Use these as copy/paste prompts for coding subagents.

### Subagent A — Trajectory and autonomous safety

**Branch:** `agentic-pr/trajectory-governor`  
**PRs owned:** PR 1, PR 2, PR 7

Prompt:

```text
You are modifying the Model Express orchestrator backend. Implement project trajectory diagnosis and safer autonomous defaults without giving the LLM execution authority.

Start with PR 1: preserve mechanism metadata in plannerExperimentFromJob. Then implement PR 2: PlannerProjectTrajectoryCard, PlannerMechanismOutcome, ComputeProjectTrajectoryDiagnosis, and wire ProjectTrajectory into ExperimentPlannerInput and PlannerContextSnapshot. Finally implement PR 7: autonomous-mode defaults for terminal planner guards and minimum meaningful improvement.

Do not change scheduling semantics except through backend validation/governor fields. Add unit tests in internal/agents and internal/api. Run go test ./internal/agents ./internal/api.
```

### Subagent B — Candidate finalizer and exhaustion enforcement

**Branch:** `agentic-pr/backend-finalizer`  
**PRs owned:** PR 3, PR 4

Prompt:

```text
You are modifying Model Express candidate ranking. Implement mechanismExhausted, architecture support changes, and FinalizePlannerRecommendation. Backend candidate ranking must be authoritative for ADD_EXPERIMENTS. The LLM may propose JSON candidates but must not bypass ranking by returning direct proposed_experiments.

Reject architecture_challenge when ProjectTrajectory blocks or exhausts it. Plateau alone must not support architecture_challenge after repeated architecture attempts. Add tests that a plateau/model-lottery fixture rejects architecture-only candidates and selects non-blocked candidates.

Preserve the invariant: LLM proposes JSON, backend ranks/validates/stores/schedules.
```

### Subagent C — Prompt/schema/tool consistency

**Branch:** `agentic-pr/prompt-dryrun-consistency`  
**PRs owned:** PR 5, PR 6

Prompt:

```text
Update the Experiment Planner prompt/schema to be candidate-first and mechanism-first. For ADD_EXPERIMENTS, require candidate_hypotheses. Explain that direct proposed_experiments are draft-only and backend ranking is authoritative.

Add governor_compliance fields. Update dry-run validation so plannerValidateCandidateExperimentsTool uses FinalizePlannerRecommendation, returning the same rejection reasons scheduling would enforce. Add tests for prompt contents, candidate_hypotheses validation, and dry-run exhaustion rejection.
```

### Subagent D — Replay evals

**Branch:** `agentic-pr/planner-replay-evals`  
**PR owned:** PR 8

Prompt:

```text
Create internal/agents/evals with deterministic replay fixtures and scorers. Add a plateau_backbone_lottery fixture representing 20+ model-family/backbone runs with only 0.701 → 0.743 Macro-F1 improvement. The eval must fail architecture_challenge candidates after exhaustion and pass class_imbalance/label_quality/preprocessing pivots or stop/select/wait decisions.

Do not call an LLM in CI. Use existing backend finalization and validation functions. Run go test ./internal/agents ./internal/agents/evals.
```

### Subagent E — Docs and Mission Control visibility

**Branch:** `agentic-pr/decision-quality-docs-ui`  
**PR owned:** PR 9

Prompt:

```text
Update Model Express docs to describe the mechanism exhaustion governor, project trajectory card, candidate-first planner contract, and replay evals. Optionally expose decision pressure, blocked mechanisms, gain per run, and candidate rejection reasons in Mission Control. Do not add any LLM execution authority.
```

---

## 6. Recommended test commands

From the orchestrator root:

```bash
go test ./internal/agents
```

For API wiring:

```bash
go test ./internal/api
```

After adding eval package:

```bash
go test ./internal/agents/evals
```

Before merging behavior-changing PRs:

```bash
go test ./...
```

If tests are slow, use focused runs while developing:

```bash
go test ./internal/agents -run 'Test.*Trajectory|Test.*Candidate|Test.*Finalize|Test.*Architecture'
go test ./internal/api -run 'Test.*Planner|Test.*Autonomous|Test.*FollowUp'
```

---

## 7. Acceptance criteria for the full implementation

The final system should satisfy all of these:

```text
1. Every planner context includes project_trajectory_card.
2. Mechanism metadata is preserved from completed jobs.
3. architecture_challenge becomes exhausted after repeated low-yield architecture attempts.
4. Exhausted mechanisms cannot be selected by backend ranking.
5. ADD_EXPERIMENTS cannot bypass backend candidate ranking.
6. Dry-run validation uses the same finalizer as scheduling.
7. Autonomous mode has safer terminal guards and minimum meaningful delta defaults.
8. A replay fixture catches the 20+ model low-yield plateau failure mode.
9. Decision payloads or logs show why candidates were rejected.
10. No LLM receives direct execution authority.
```

Quantitative targets:

```text
accepted architecture-only experiments after architecture_challenge exhaustion: 0
scheduled experiments bypassing candidate ranking: 0
planner prompts with project_trajectory_card: 100%
plateau replay fixture pass rate: 100%
backend validation pass rate after one repair pass: >= 95%
accepted duplicate signatures: 0
accepted exhausted-mechanism candidates: 0
```

---

## 8. Immediate runtime settings while implementing

Until the full governor is merged, run autonomous mode with conservative settings:

```bash
export MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS=true
export MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=3
export MODEL_EXPRESS_PLANNER_MIN_MEANINGFUL_DELTA=0.010
```

This will not fully fix bypassable ranking or missing trajectory diagnosis, but it should reduce runaway low-yield loops.

---

## 9. Techniques to defer

Do not prioritize these until the above PRs are merged and evaluated:

```text
semantic/vector memory
multi-agent debate
tree search / MCTS over plans
fine-tuning the planner
Kafka/Redis/workflow orchestration
large model-routing refactors
prompt caching work
```

Those may be useful later, but they will not fix the observed failure mode as directly as backend-enforced mechanism exhaustion and mandatory candidate finalization.

---

## 10. Final implementation principle

The desired behavior is not:

```text
The LLM becomes smarter and decides better by itself.
```

The desired behavior is:

```text
The LLM proposes structured mechanism candidates.
The backend computes project trajectory and mechanism exhaustion.
The backend ranks and validates candidates.
The backend rejects exhausted, duplicate, shallow, or low-expected-value work.
Only backend-approved plans are stored and scheduled.
```

That keeps Model Express agentic, but not reckless.
