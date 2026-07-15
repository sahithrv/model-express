# Candidate Ranking Correctness Plan

Status: Proposed
Date: 2026-07-09
Track: Ranking
Depends on: None
Blocks: Ranker-v2 shadow evaluation in [Agent Quality Calibration](./04-agent-quality-calibration.md)

## Purpose

Fix the concrete family-diversity selection defect and make backend candidate selection deterministic, testable, and auditable without changing the broader candidate-scoring policy.

## Current Defect

`RankPlannerCandidateHypotheses` in `services/orchestrator/internal/agents/candidate_ranking.go` currently:

1. Scores all candidates.
2. Sorts them once by score.
3. Iterates over `ordered` using a value copy.
4. Subtracts a family penalty from that copy after multiple same-family selections.
5. Selects the candidate anyway without reordering the remaining candidates.

The modified score is neither used for the current choice nor persisted. Therefore, the intended family-diversity penalty has no effect.

## Intended Behavior

Selection should be a deterministic greedy process:

1. Compute each candidate's base score once.
2. Reject invalid, exhausted, duplicate, or below-threshold candidates before selection.
3. For each available slot, compute selection-time adjustments using candidates already selected.
4. Choose the highest adjusted score.
5. Break exact ties using the original candidate index.
6. Persist the selected order and every selection-time adjustment.

A diversity penalty should be a preference, not an absolute family quota. A substantially stronger same-family candidate may still win after the penalty.

## Scope Boundary

This track fixes selection correctness only. It does not recalibrate the current weights for expected gain, novelty, cost, risk, memory, or diagnosis alignment. Those changes belong in the agent-calibration track after reliable outcome data exists.

## PR Breakdown

### Ranking PR 1: Deterministic Diversity-Aware Selector

Size: S
Behavior change: Yes, bug fix

Goal: Replace the ineffective one-pass family penalty with a pure, dynamically adjusted selector.

Scope:

1. Extract selection from base candidate scoring into a focused helper.
2. Keep base scoring, rejection, maximum-experiment pressure, and stable candidate-index tie-breaking unchanged.
3. Recompute only selection-time adjustments for remaining candidates at each slot.
4. Apply the existing family-diversity policy at its intended threshold.
5. Return selected candidate indexes in selection order.
6. Avoid mutating caller-owned candidate or ranking inputs.

Primary areas:

- `services/orchestrator/internal/agents/candidate_ranking.go`
- A focused candidate-ranking test file, or the existing `experiment_planner_llm_test.go`

Required tests:

1. Three high-scoring candidates from one family plus a competitive alternative family.
   - The alternative wins the relevant slot only after the dynamic penalty.
2. A same-family candidate whose adjusted score remains clearly highest.
   - It still wins; diversity is not a hard quota.
3. Equal adjusted scores.
   - Original candidate index breaks the tie.
4. Rejected candidates.
   - They never participate in selection-time adjustment.
5. `maxExperiments` values from one through five.
   - The selector returns `min(maxExperiments, eligibleCandidateCount)`.
   - Add all-rejected and partially eligible candidate cases.
6. Input immutability and repeated execution.
   - Two calls produce byte-equivalent rankings and selection order.
7. Existing mechanism, memory, and multi-fidelity tests.
   - Their results remain unchanged unless they exercised the defective family case.

Acceptance criteria:

- The new regression test fails against the old implementation and passes against the new selector.
- The family penalty can demonstrably change selection order.
- Stable tie-breaking remains deterministic.
- No database or API migration is required.

Rollout:

- Treat changed selections in the affected family-diversity case as expected.
- Run all planner replay fixtures before merge and attach the before/after selection diff to the PR description.
- As an isolated correctness fix, use before/after replay artifacts and normal code revert as rollback rather than a long-lived runtime flag.

Non-goals:

- Changing base score weights or the acceptance threshold.
- Adding empirical priors.
- Altering candidate generation prompts.
- Enforcing a hard maximum per model family.

### Ranking PR 2: Selection Audit Contract And Replay Coverage

Size: M
Depends on: Ranking PR 1

Goal: Make it obvious why the final selected order differs from raw base-score order and prevent regression through the replay harness.

Scope:

1. Extend `CandidateRanking` additively with explicit base score, selected experiment index, and typed selection adjustments.
2. Preserve the existing `score` field as a compatibility alias for base score while new readers use the explicit fields.
3. Add a bounded per-round selection trace so an unselected candidate's changing greedy-round score is not collapsed into one ambiguous value.
4. Define `selection_score` only for the round in which a candidate was selected; leave it absent for unselected candidates.
5. Add a `family_diversity` adjustment reason.
6. Add at least one replay fixture containing three same-family candidates and a viable alternative.
7. Add zero-based `selected_experiment_index` for the explicit candidate-to-proposed-experiment mapping used by later outcome calibration.

Suggested additive shape:

```json
{
  "base_score": 0.74,
  "selection_score": 0.62,
  "selection_order": 2,
  "selected_experiment_index": 1,
  "selection_adjustments": [
    {
      "code": "family_diversity",
      "value": -0.12,
      "detail": "two candidates from this model family were already selected"
    }
  ]
}
```

The recommendation should retain a bounded trace for each greedy round. Selected experiments/rounds are capped at five, but the LLM candidate list is not currently capped. Each round should therefore store the chosen candidate plus a fixed number of highest-scoring competitors, along with `total_candidate_count` and `truncated`. The chosen candidate must never be omitted, and trace-size limits must not alter selection.

Primary areas:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/agents/evals/`

Required tests:

- Candidate-ranking JSON remains backward compatible.
- Selected order is represented exactly once and is contiguous.
- Base score plus selection adjustments equals selection score after rounding.
- Replay detects a bypassed or non-dynamic selector.
- Selected candidate indexes map one-to-one to zero-based proposed-experiment indexes.
- Per-round traces are deterministically ordered and bounded.
- Truncated traces disclose total candidate count and always include each round's selected candidate.

Acceptance criteria:

- Stored audit data can distinguish base quality, rejection, and selection-time diversity.
- Replay scoring fails if family adjustments are calculated but do not affect selection.
- Old stored decision payloads remain valid JSON for existing readers.

### Ranking PR 3: Mission Control Ranking Audit

Size: S
Depends on: Ranking PR 2

Goal: Render the new audit contract without mixing UI changes into the backend correctness PR.

Scope:

1. Add the new optional fields to Mission Control types.
2. Show concise base-score, selection-order, and diversity-adjustment explanations in the Agents drill-down.
3. Keep the full per-round trace behind an expert disclosure.
4. Continue rendering old decisions that contain only `score`, `selected`, and `reasons`.

Primary areas:

- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/features/mission/projectDetailModel.tsx`

Tests and acceptance:

- Old and new ranking fixtures both render.
- The UI does not imply an unselected candidate had one fixed selection score.
- Audit detail remains outside the default Overview.
- Existing Mission Control Node tests and `npm run build` pass.

## Verification Commands

Run at minimum:

```bash
cd services/orchestrator
go test ./internal/agents/...
```

After Ranking PR 3:

```bash
cd apps/mission-control
npm run build
```

## Definition Of Done

This track is complete when:

1. Family diversity affects selection at the point intended by policy.
2. Higher adjusted score always wins with stable tie-breaking.
3. Selection order and adjustments are durable and inspectable.
4. A dedicated regression fixture prevents reintroduction.
5. No unrelated score-weight changes are mixed into the fix.

## Risks

1. Existing projects may receive a different candidate mix.
   - Mitigation: replay before/after comparison and explicit PR release notes.

2. A diversity preference could displace a genuinely superior experiment.
   - Mitigation: use a score penalty rather than a hard quota and test the high-margin case.

3. Adding more score fields could confuse consumers.
   - Mitigation: define `base_score` and `selection_score` precisely and preserve old-field parsing during rollout.

## Future Follow-Up

After candidate-to-outcome calibration exists, replace the fixed diversity value with a versioned policy justified by observed marginal value. That work belongs to the ranker-v2 PRs in the calibration plan.
