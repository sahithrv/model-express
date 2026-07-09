# Empirical Agent Quality Calibration Plan

Status: Proposed
Date: 2026-07-09
Track: Calibration
Depends on: [Candidate Ranking Correctness](./02-candidate-ranking-correctness.md) and verified execution identity from [Experiment Execution Fidelity](./01-experiment-execution-fidelity.md)
Can begin before dependencies: Runtime identity, evaluator fidelity, rubric work, and shadow validation

## Purpose

Replace intuition-driven planner promotion with a reproducible system that measures:

1. Whether recommendations are valid, safe, and evidence-supported.
2. Whether predicted metric changes agree with observed results.
3. Whether a prompt, context, retrieval, validator, or ranker version improves quality at acceptable cost.
4. Whether rollout can be paused or reversed before affecting many projects.

The Experiment Planner is the first target because it schedules expensive work and already has replay infrastructure. Runtime identity and reporting should remain reusable for the Training Monitor and Visual Analysis agents.

## Current Problem

- Replay variants reuse one canned model response, so they do not independently compare generated decision quality.
- Context replay can alter the input passed to the finalizer, unlike a production prompt/context experiment.
- Prompt-size estimates do not consistently use the actual production request.
- The scenario corpus is small.
- Best-variant scoring includes the backend candidate score, which is circular when evaluating that ranker.
- Stored prompt version does not uniquely identify prompt, context, retrieval, validator, ranker, retry, or runtime settings.
- Self-reported expected impact and novelty contribute heavily to candidate score without historical calibration.
- Candidate-to-experiment-to-job outcome lineage is not durable.
- Plan outcome finalization can remain pending when a policy skips an experiment.
- Strict validation defaults to relaxed mode and its configuration is duplicated.

## Evaluation Principles

1. Separate deterministic structural replay from repeated live evaluation.
2. Do not use the ranker score under evaluation as its own quality label.
3. Backend schedulability validation remains a valid structural oracle.
4. Keep full backend input identical across prompt-only/context-only comparisons.
5. Record first-pass validity separately from validity after retries.
6. Compare forecasts only with observed outcomes.
7. Never label an unselected candidate as a failure merely because it was not run.
8. Show sample size, uncertainty, selection bias, and chronological data windows.
9. Shadow new behavior before scheduling with it.
10. Give every promoted policy stable identity and rollback.

## Planner Variant Identity

```text
agent_version
+ static_prompt_version
+ context_builder_version
+ tool_policy_version
+ validator_version_and_mode
+ ranker_version
+ retrieval_policy_version
+ model_and_runtime_settings
= planner_variant_id
```

Attempt group/index, retry reason, measured wall latency, provider usage, and derived cost are invocation facts. Cost must include a `pricing_version`; provider token usage alone is not a cost record.

## Forecast Contract

Candidate forecasts must persist their target at proposal time:

- `forecast_target`
- `metric_direction`
- `score_basis`
- `score_version`
- `baseline_job_id`
- `baseline_score`
- `predicted_delta`
- `prediction_source`

The current generic recommendation confidence is not the probability that a candidate improves the champion. Probability and prediction-interval fields are deferred until a separate forecast-schema proposal defines and evaluates them.

For candidate calibration, `predicted_delta` comes from candidate-level `candidate.expected_metric_impact`, not recommendation-level `expected_delta_vs_champion`. Calibration PR 6 updates the prompt/schema so this field is explicitly expressed in the frozen forecast target and score basis, relative to the stored baseline, with validated direction, units, finite range, and `prediction_source="candidate.expected_metric_impact"`.

## Core Metrics

### Structural And Safety

- Parse success
- First-pass and eventual validation
- Retry rate and attempts per accepted recommendation
- Unsupported/task-incompatible proposal rate
- Duplicate/no-op rate using accepted semantic identity
- Blocked-mechanism or stop-policy escape rate
- Candidate mechanism/signature diversity

### Decision Quality

- Expected decision type on labeled scenarios
- Diagnosis-to-mechanism alignment
- Required evidence coverage and unsupported-claim rate
- Correct stop/select/wait behavior
- Cost/deployment constraint compliance

### Observed Forecast Quality

- Predicted-delta mean absolute error and bias
- Correct improvement direction
- Meaningful-improvement precision and recall
- Coverage: share of forecasts with trustworthy observed outcomes

### Efficiency

- Planner wall latency measured locally
- Provider tokens/tool rounds
- Derived cost with pricing version
- Downstream runtime/cost/failure rate for accepted decisions

## PR Breakdown

### Calibration PR 1: Stable Planner Runtime Identity

Size: M
Behavior change: None

Goal: Attribute every invocation to an exact planner configuration.

Scope:

1. Define and persist `planner_variant_id`.
2. Record prompt, context, tool, validator, ranker, and retrieval versions.
3. Include provider, API style, model, request temperature, reasoning settings, tool budget, and other allowlisted generation parameters in canonical variant identity.
4. Record validation mode, attempt group/index, and retry reason.
5. Measure wall latency locally.
6. Preserve provider usage and derive cost only with a versioned pricing snapshot.
7. Read old rows as legacy/unknown variants.

Tests:

- Equal configurations produce equal IDs; each meaningful variant produces a distinct ID.
- Provider, API style, temperature, and reasoning/tool settings affect identity.
- Retry attempts share a group and have unique indexes.
- Latency is locally measured; cost records name their pricing version.
- Legacy rows remain readable.

Acceptance criteria:

- Every new decision is queryable by exact planner variant.
- No validation or selection behavior changes.

### Calibration PR 2: Faithful Paired Evaluator

Size: M
Depends on: Calibration PR 1

Goal: Compare independently generated variants using production request builders.

Scope:

1. Separate canned structural replay from live paired evaluation.
2. Build every variant request through production prompt/context builders.
3. Generate independently for every variant and repeat.
4. Pass the same complete finalizer input to every prompt/context variant.
5. Record actual request bytes, usage, measured latency, derived cost/pricing version, tool trace, retries, and parsed result.
6. Add an opt-in read-only `planner-eval` command with bounded JSONL output.
7. Keep provider calls explicitly opt-in and budget-capped.

Tests:

- A fake generator proves independent calls per variant/repeat.
- Prompt-only variants receive identical finalizer context.
- Measurements come from the actual built request.
- Evaluation performs no project, plan, job, memory, or champion writes.

Acceptance criteria:

- Offline replay remains deterministic/network-free.
- Paired evaluation can compare generated quality, not only prompt length.

### Calibration PR 3: Rubric Engine And Starter Scenarios

Size: M
Depends on: Calibration PR 2

Goal: Establish non-circular scoring with a small representative set before expanding the corpus.

Scope:

1. Define rubric fields for decision type, allowed/forbidden mechanisms, evidence, task compatibility, stop behavior, and safety.
2. Score first-pass and eventual validity separately.
3. Put correctness/safety before efficiency tie-breakers.
4. Remove the ranker-under-test's own score from quality labels.
5. Retain backend validation as the schedulability oracle.
6. Add starter scenarios for classification, detection, stop/select, and invalid task/config behavior.

Tests:

- Each rubric has one passing and one failing mutation.
- Invalid concise output cannot beat valid output through token savings.
- Scoring does not use the candidate score under evaluation as an oracle.
- Fixture order does not change aggregate results.

Acceptance criteria:

- A deterministic local rubric command produces a reviewable artifact.

### Calibration PR 4: Scenario Corpus And Checked Baseline

Size: M
Depends on: Calibration PR 3

Goal: Expand coverage without overloading the rubric-engine PR.

Scope:

1. Add scenarios for imbalance/minority failure, overfit, underfit, plateau, repeated architecture shopping, latency/cost, duplicates/no-ops, label audit, wait, and memory success/failure/rejection.
2. Cover classification and detection where a mechanism is task-sensitive.
3. Add targeted mutation failures.
4. Check in a baseline artifact with an explicit review/update procedure.
5. Define critical, quality, and efficiency tolerances separately.

Acceptance criteria:

- Every supported planner decision type has representative coverage.
- Baseline changes show per-scenario explanations rather than only one aggregate score.
- Live LLM calls remain optional.

### Calibration PR 5: Centralized Shadow-Strict Validation

Size: M
Depends on: Calibration PR 1 and Fidelity PR 3

Goal: Measure strict validation before changing scheduling.

Scope:

1. Replace duplicated booleans with `relaxed`, `shadow_strict`, or `strict`.
2. In shadow mode, persist typed strict verdicts without blocking the relaxed result.
3. Record first-pass/eventual status and retry outcomes by variant.
4. Measure missing evidence, mechanism mismatch, invalid task/model, and proposal-time no-op accepted specs.
5. Keep execution-time mismatch verdicts in the Fidelity system rather than inferring them here.

Tests:

- All modes use the same strict implementation.
- Shadow returns the relaxed decision while storing the strict verdict.
- Retry attempts remain attributable to one attempt group.
- Proposal-time no-op checks use accepted-spec identity.

Acceptance criteria:

- Production-like traffic can quantify what strict mode would block.
- There is one source of truth for validation mode.

### Calibration PR 6: Candidate Provenance Records

Size: M
Depends on: Calibration PR 1, Ranking PR 2, and Fidelity PR 2

Goal: Persist every accepted recommendation candidate before auto/manual scheduling decisions diverge.

Scope:

1. Add a candidate-provenance migration using the number assigned at merge/rebase time.
2. Create candidate rows when an accepted planner decision is persisted, even when auto-execution is off.
3. Persist the decision and candidate rows in one transaction where store boundaries allow it; otherwise call an idempotent `ensureCandidateProvenance(decision)` on both the new-decision and existing-decision paths.
4. Store invocation, decision, variant, candidate index, requested/accepted hash, task, mechanism, forecast contract, base score, selection trace reference, selected/rejected state, and reasons.
5. Define candidate `predicted_delta` as `candidate.expected_metric_impact`; update the prompt/schema to name its frozen target, baseline, score basis/version, direction, units, and valid range.
6. Keep recommendation-level `expected_delta_vs_champion` separate from candidate forecasts.
7. Store zero-based selected experiment index from Ranking PR 2.
8. Attach follow-up plan, experiment, job, and realized-effective hash later as nullable lineage fields.
9. Keep invalid pre-acceptance attempts in invocation audit only.
10. Preserve existing plan decisions and scorecards.

Tests:

- Propose/manual and auto-execute modes both create identical candidate provenance at decision time.
- Candidate indexes map deterministically to selected experiment indexes.
- Unselected candidates have unknown outcomes, not negative outcomes.
- Requested/accepted hashes are present; realized hash remains null until execution.
- Persistence is idempotent.
- A simulated failure after decision creation is repaired on the existing-decision path without duplicating candidates.
- Candidate forecast sign/range/units are validated against the frozen score basis.

Acceptance criteria:

- Every accepted candidate is traceable to its decision and planner variant before scheduling.

### Calibration PR 7: Candidate Outcome Finalization

Size: M
Depends on: Calibration PR 6 and Fidelity PRs 8-9

Goal: Attach trustworthy observed outcomes and fix skipped-experiment terminal accounting.

Scope:

1. Attach selected experiment/job/attempt and realized-effective hash.
2. Record actual forecast-target score, delta from stored baseline, terminal state, cost, and runtime.
3. Finalize plan aggregates when every experiment is terminal, including skipped/cancelled/failed experiments.
4. Exclude mismatched/simulated execution from calibration cohorts.
5. Backfill only unambiguous historical mappings.

Tests:

- Multi-experiment plans map each candidate to the correct job.
- Budget-skipped experiments become terminal.
- Failed/cancelled jobs do not fabricate metrics.
- Metric direction, score basis, and version are frozen from proposal time.
- Repeated finalization does not double-count cost/outcomes.

Acceptance criteria:

- Predicted versus actual outcome is queryable by variant, task, mechanism, and score version.
- Plans with skipped work no longer remain pending forever.

### Calibration PR 8: Read-Only Calibration Report

Size: M
Depends on: Calibration PR 7

Goal: Report trustworthy performance before changing ranking behavior.

Scope:

1. Report structural, decision, forecast, outcome-coverage, and efficiency metrics.
2. Group by planner variant, task, mechanism, and model family where sample size permits.
3. Include sample count, confidence interval, time window, pricing version, and score version.
4. Suppress undersized cohorts.
5. Use chronological train/evaluation windows for empirical priors.
6. Explicitly disclose selection bias and observed-outcome coverage.
7. Add a bounded read-only CLI/API; UI visualization is separate.

Tests:

- Synthetic cohorts verify MAE, bias, direction, precision/recall, coverage, and interval calculations.
- Maximize/minimize metric directions are correct.
- Unselected candidates never become negative outcomes.
- Queries are bounded by project/time/limit filters.

Acceptance criteria:

- A reviewer can compare variants without manually reading invocation JSON.
- Reports distinguish insufficient evidence from poor results.

### Calibration PR 9: Ranker V2 In Shadow Mode

Size: M
Depends on: Ranking PR 1 and Calibration PR 8

Goal: Reduce direct trust in self-reported impact/novelty without changing scheduling.

Scope:

1. Define a versioned ranker-v2 policy.
2. Blend deterministic evidence with smoothed priors by task/mechanism.
3. Fall back to broad cohorts when narrow cohorts are undersized.
4. Cap self-reported expected impact and novelty influence.
5. Compute v1/v2 ordering and selections side by side while scheduling with v1.
6. Persist overlap, ordering changes, selection-set changes, and score components.
7. Freeze prior data windows before evaluation windows.

Tests:

- Empty/small cohorts use documented fallbacks.
- One outlier cannot dominate a prior.
- Shadow scoring is deterministic and cannot schedule.
- Dynamic family diversity still applies after base scoring.

Acceptance criteria:

- Shadow analysis reports ordering behavior and outcomes only for candidates executed under v1/overlap.
- It does not claim outcome quality for v2-only counterfactual selections.

### Calibration PR 10: Required Deterministic Evaluation Gate

Size: S
Depends on: Calibration PR 4

Goal: Make structural/safety regressions merge-blocking without mixing in production default changes.

Scope:

1. Add canned structural replay/rubric evaluation to CI or the required verification workflow.
2. Compare with the checked baseline using explicit critical/quality/efficiency tolerances.
3. Block critical safety/validation regressions.
4. Warn on cost/token regressions until evidence supports hard gates.
5. Keep repeated live evaluation opt-in.

Acceptance criteria:

- A deliberately unsafe mutation fails the required gate.
- Deterministic replay does not require provider credentials/network access.
- Baseline updates require an explicit reviewed artifact change.

### Calibration PR 11: Strict Validation Default

Size: S
Depends on: Calibration PRs 5 and 10, plus Fidelity PR 11

Goal: Make strict validation the production default after shadow evidence proves safety.

Scope:

1. Define rollout thresholds for eventual validation, retry rate, and unsupported proposals.
2. Require zero unsafe schedules/post-validation escapes, not zero first-pass LLM violations.
3. Change new/cloud examples to strict.
4. Keep relaxed mode as an explicit observable rollback for one compatibility release.
5. Document retry and rollback behavior.

Acceptance criteria:

- Unsupported/task-incompatible proposals cannot schedule.
- Expected first-pass violations can retry within policy.
- Rollback is one configuration change and emits a diagnostic counter.

### Calibration PR 12: Guarded Ranker And Variant Rollout

Size: M
Depends on: Calibration PRs 9-11

Goal: Gather real v2-selected outcomes safely and promote only non-inferior policies.

Scope:

1. Assign projects deterministically to stable cohorts.
2. Roll out one dimension at a time: ranker, prompt, context, or retrieval policy.
3. Start at 5%; gather v2-selected outcomes before advancing to 25%, 50%, and 100%.
4. Define pause/manual-review thresholds.
5. Provide one-switch rollback to the last approved policy.
6. Persist cohort and policy identity with invocations/candidates.
7. Do not delete rollout flags in this PR; cleanup follows an observation window.

Promotion gates:

- Zero safety/task-compatibility regression.
- First-pass/eventual validation non-inferiority.
- Observed outcome non-inferiority at adequate sample size.
- Cost/latency improvement when quality is tied.
- Failure/retry rates within thresholds.

Acceptance criteria:

- Rollout can pause/reverse without data repair.
- Cohort assignment is restart-stable.
- Simultaneous ranker-plus-prompt experiments are prevented unless intentionally factorial.

## Verification Layers

1. **Unit and canned structural replay**: deterministic, offline, required.
2. **Paired live evaluation**: opt-in, repeated, read-only, used before promotion.
3. **Production shadow/outcome evaluation**: real executions and bounded cohorts.

Passing one layer is necessary but not sufficient for the next.

## Migration Coordination

Calibration, Fidelity, and Activity may add migrations in parallel. Assign the next available migration number when a PR rebases for merge; CI should reject duplicate or out-of-order migration filenames.

## Definition Of Done

1. Every decision has stable variant identity.
2. Prompt/context variants use independent generations and real request builders.
3. Representative non-circular replay is required.
4. Accepted candidates map to verified effective outcomes.
5. Reports include sample size, uncertainty, score version, and selection-bias disclosure.
6. Self-reported impact/novelty no longer dominate without empirical support.
7. Strict validation is default with rollback.
8. New policies roll out by guarded stable cohort.

## Risks

1. Outcome data is observational and selection-biased.
   - Mitigation: coverage disclosure, observed-only metrics, chronological splits, and controlled rollout.

2. Small cohorts create unstable priors.
   - Mitigation: minimum sizes, shrinkage, confidence intervals, and broader fallbacks.

3. Fixtures can overfit prompt development.
   - Mitigation: held-out scenarios and production shadow confirmation.

4. Live evaluation can spend money or mutate state.
   - Mitigation: explicit opt-in, budgets, fake-provider tests, and read-only construction.

5. Strict validation can increase retries.
   - Mitigation: shadow measurement, structured feedback, bounded retry, and rollback.

## Non-Goals

- Online reinforcement learning.
- Causal claims for unexecuted candidates.
- Hidden chain-of-thought evaluation.
- Automatic promotion from a tiny cohort.
- Replacing deterministic backend validation with model judgment.
- Probability calibration until a separate forecast probability schema exists.
- Full Training Monitor/Visual Analysis calibration in the first planner rollout.

## Future Follow-Up

1. Add a post-observation cleanup PR to remove obsolete rollout flags.
2. Add evidence-grounding/preprocessing rubrics for Visual Analysis.
3. Add intervention precision and false-alarm metrics for Training Monitor before enabling its LLM path by default.
4. Consider small budget-capped randomized exploration to reduce selection bias.
