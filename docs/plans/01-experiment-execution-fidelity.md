# Experiment Execution Fidelity Plan

Status: Proposed
Date: 2026-07-09
Track: Fidelity
Depends on: None
Blocks: Outcome-based calibration in [Agent Quality Calibration](./04-agent-quality-calibration.md)

## Purpose

Guarantee a machine-verifiable relationship between:

1. The experiment requested by a plan or AutoML search.
2. The semantic configuration accepted before scheduling.
3. The arguments realized by the selected worker and framework.
4. The preprocessing used by evaluation and export.

No non-default field may be silently ignored. Automatic adjustments must be bounded, typed, and visible.

## Current Problem

ModelExpress validates a broad `PlannedExperiment` shape, but its execution paths do not support that field set uniformly.

Concrete examples:

- The YOLO Modal path forwards a narrow set of arguments to `detector.train(...)`; planner-visible optimizer, scheduler, regularization, augmentation, balancing, sampling, and fine-tuning fields may not affect execution.
- Classification image-size validation permits values above the worker's effective cap.
- Go accepts `letterbox`/`yolo_letterbox`, while the classification preprocessing registry does not expose the same strategies.
- Scalar `omitempty` fields and conditional job serialization can replace explicit `false` or `0` with worker defaults.
- Pretrained-weight loading can silently fall back to random initialization.
- Dataset normalization can fall back to ImageNet normalization without making that change first-class.
- AutoML can tune detection fields that the YOLO worker does not apply.
- Novelty signatures can include requested-but-unexecuted values, making equivalent runs appear different.
- Worker preprocessing/export evidence is not fully represented by typed orchestrator updates.

## Identity Model

The system needs three distinct identities:

1. `requested_config_hash`
   - Stable identity of the user's/agent's request after JSON canonicalization.
2. `accepted_spec_hash`
   - Semantic configuration approved before scheduling, after aliases and defaults are resolved.
   - Used for pre-scheduling duplicate/no-op detection.
3. `realized_effective_hash`
   - Framework-semantic configuration actually realized by the worker.
   - Used for outcome grouping and audit.

Attempt, GPU, host, queue, and other infrastructure details must not alter the semantic hash. A training-semantic recovery, such as batch-size reduction, may change the realized hash but must not be credited as a novel planner mechanism.

## Execution Record Lifecycle

Do not model a running attempt as one immutable receipt that is repeatedly overwritten. Use:

- An immutable accepted-spec section created by the server.
- Attempt-scoped, append-only realization observations such as `INITIALIZED` and `FINALIZED`.
- A derived lifecycle status and fidelity verdict.

Lifecycle status:

- `PENDING`: accepted spec exists; no worker realization received.
- `INITIALIZED`: worker reported resolved framework/preprocessing arguments.
- `FINALIZED`: worker reported final realized arguments.
- `NOT_REALIZED`: attempt ended before framework initialization.

Fidelity verdict:

- `MATCHED`: accepted and realized semantics agree after documented normalization.
- `APPROVED_ADJUSTMENT`: a bounded policy changed execution, such as batch-size recovery.
- `MISMATCH`: execution differs without an approved policy; it is not trustworthy learning evidence.
- `UNVERIFIED`: legacy data without a versioned execution record.
- `SIMULATED`: local smoke/simulator output, never represented as verified real training.

For a new versioned attempt, fidelity verdict is null until at least an `INITIALIZED` realization observation exists. `UNVERIFIED` is reserved for legacy records whose execution cannot be reconstructed; it must not be used for a currently pending attempt.

Pretrained-to-random initialization is not an approved resource adjustment. It must fail, or be resubmitted as a newly accepted experiment that explicitly requests untrained initialization.

## Design Principles

1. Reject unsupported settings instead of silently dropping them once enforcement is enabled.
2. Preserve explicit `false`, `0`, and empty-policy semantics.
3. Store the exact allowlisted framework arguments used.
4. Keep accepted and realized identities separate.
5. Treat preprocessing as part of the executable model contract.
6. Roll out verdict collection before completion/champion gating.
7. Keep old jobs readable without fabricating historical truth.

## PR Breakdown

### Fidelity PR 1: Capability Source And Generated Consumers

Size: M
Behavior change: None

Goal: Establish a versioned source of truth for executable configuration.

Scope:

1. Add a canonical capability document covering every `PlannedExperiment` field by task and runner.
2. Classify fields as executed, conditional, metadata-only, or unsupported.
3. Define canonical defaults, aliases, ranges, and reason codes.
4. Generate or embed Go data for the orchestrator.
5. Generate a packaged Python module for local/Modal workers.
6. Add CI/contract checks that generated artifacts match the canonical source.

Primary areas:

- New top-level `contracts/experiment_execution_capabilities.v1.json`
- New `services/orchestrator/internal/execution/`
- New `services/worker/worker/training/execution_capabilities_generated.py`
- Go and Python golden tests

Acceptance criteria:

- Go and Python normalize shared fixtures identically.
- Every plan field has an explicit classification for classification and detection.
- Go does not depend on a repository-relative runtime file.
- Modal packaging includes the generated Python representation.
- No scheduling or worker behavior changes.

### Fidelity PR 2: Presence-Safe Canonical Execution Spec

Size: M
Depends on: Fidelity PR 1
Behavior change: Payload shape only; legacy flat keys remain

Goal: Preserve intent and resolve every new experiment into `execution_spec_v1`.

Scope:

1. Adopt a presence-aware representation for values where omitted, false, and zero differ.
2. Update LLM decoding and plan-to-job serialization accordingly.
3. Resolve aliases/defaults into a canonical accepted semantic spec.
4. Attach capability version, task, runner, requested hash, and accepted-spec hash.
5. Keep legacy flat keys during the compatibility window.
6. Mark older plans/jobs as `legacy_unversioned` at read time; do not rewrite them.

Primary areas:

- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/api/plans.go`
- `services/orchestrator/internal/execution/`

Tests:

- `pretrained=false`, `freeze_backbone=false`, zero momentum, and zero augmentation parameters survive plan-to-job JSON round trips.
- Hashes are stable across map ordering.
- Legacy payloads resolve through documented defaults.
- Accepted hashes exclude attempt/infrastructure metadata.

Acceptance criteria:

- Every newly queued job has a canonical accepted spec.
- Existing workers still receive the legacy fields they require.
- This PR does not claim to know realized worker configuration.

### Fidelity PR 3: Task-Aware Shadow Validation And Planner Capability Card

Size: M
Depends on: Fidelity PR 2

Goal: Detect unsupported/no-op proposals before enforcement and stop encouraging the planner to generate them.

Scope:

1. Validate accepted specs by task, runner, and model family.
2. Add `shadow` and `enforce` modes, initially defaulting to shadow.
3. Emit typed warnings for unsupported, conditional, or normalized-away values.
4. Compute and shadow-report the duplicate/no-op decision that accepted-spec hashes would make; keep the legacy scheduling decision until Fidelity PR 8.
5. Provide the planner a compact capability card rather than duplicating long enum rules in prose.
6. Measure warnings by task, runner, field, and reason code.

Tests:

- Classification and detection reject/warn on different capability sets.
- Equivalent accepted specs have the same accepted hash and the same shadow duplicate verdict.
- Planner context is bounded and contains only current task capabilities.
- Shadow mode does not block existing jobs.

Acceptance criteria:

- The backend can report what enforcement would block.
- New planner calls see the runner's actual supported field set.

### Fidelity PR 4: Task-Scoped AutoML Search Spaces

Size: S
Depends on: Fidelity PR 3

Goal: Prevent AutoML from generating known no-op or unsupported combinations.

Scope:

1. Derive search-space eligibility from the capability contract.
2. Separate classification and detection tunables.
3. Exclude conditional fields when their prerequisites are absent.
4. Record capability version with generated trials.

Primary areas:

- `services/orchestrator/internal/api/automl.go`
- `services/orchestrator/internal/automl/`

Acceptance criteria:

- Detection trials never sample classification-only settings.
- Every generated trial passes task-aware validation.
- Search-space changes are deterministic for a fixed capability version.

### Fidelity PR 5: Attempt Execution Record Plumbing

Size: M
Depends on: Fidelity PR 2
Behavior change: Additive collection only

Goal: Persist accepted specs and optional worker realization observations without enforcing completion yet.

Scope:

1. Add execution-record migrations using the migration number assigned at merge/rebase time.
2. Store the immutable accepted spec at job scope when the job is queued.
3. Create a separate attempt-scoped `PENDING` record when assignment establishes the active attempt ID, including every retry.
4. Store realization observations against that attempt record.
5. Add store models and bounded job/run API reads.
6. Add an attempt-authenticated, idempotent worker callback.
7. Reject stale attempt callbacks.
8. Derive verdicts server-side from allowlisted semantic fields.
9. Wire the local deterministic simulator to finalize its attempt explicitly as `SIMULATED`.

Tests:

- Duplicate callbacks are idempotent.
- Earlier attempts cannot overwrite the active attempt.
- Each retry gets its own pending attempt while retaining the same job-scoped accepted spec.
- Accepted spec remains immutable after scheduling.
- Observation payloads and framework arguments are bounded and redacted.
- A job can still complete during shadow rollout without a final observation.
- Local simulator runs produce `SIMULATED` rather than remaining pending or unverified.

Acceptance criteria:

- PR 5 is safe with old workers that never call the endpoint.
- Pre-initialization failures remain `NOT_REALIZED` or `PENDING`, not falsely `MISMATCH`.
- No champion/completion gating is introduced.

### Fidelity PR 6: Classification Runner Fidelity

Size: M
Depends on: Fidelity PRs 3 and 5

Goal: Make classification consume one resolved spec and report exact realized semantics.

Scope:

1. Route setup through the generated capabilities and Python execution resolver.
2. Align image-size and preprocessing enums with the accepted contract.
3. Honor explicit booleans and zeros.
4. Fail pretrained loading instead of silently switching to random initialization.
5. Fail unavailable dataset normalization unless a separately accepted fallback policy exists.
6. Record exact normalization, augmentations, sampler, loss, optimizer, scheduler, fine-tune behavior, and framework versions.
7. Post `INITIALIZED` and `FINALIZED` observations.
8. In shadow mode, record mismatches without claiming trusted evidence; in runner enforcement mode, fail before expensive training.

Tests:

- Every supported classifier field reaches its implementation.
- Explicit false/zero fixtures remain explicit.
- Pretrained and normalization failures cannot silently change semantics.
- Train/eval transform parity is covered.
- A mocked run produces a deterministic realized hash and verdict.

Acceptance criteria:

- No supported classification field disappears between accepted spec and realization.
- Any semantic fallback requires a new accepted spec.

### Fidelity PR 7: YOLO Runner Fidelity

Size: M
Depends on: Fidelity PRs 3 and 5
Implementation order: Rebase after Fidelity PR 6 if both modify `modal_app.py`

Goal: Translate the accepted detection spec into verified Ultralytics arguments and reject unsupported semantics.

Scope:

1. Add a pure `execution_spec_v1` to Ultralytics-kwargs translator.
2. Map only fields with equivalent semantics in the pinned Ultralytics version.
3. Mark classifier augmentation, class balancing, sampling, and other non-equivalent settings unsupported.
4. Model native letterboxing and native augmentation defaults explicitly.
5. Capture allowlisted realized trainer arguments from the framework.
6. Post `INITIALIZED` and `FINALIZED` observations.
7. Support the same shadow-versus-runner-enforcement modes as classification.

Tests:

- Mocked `YOLO.train(**kwargs)` tests cover every supported mapping.
- Unsupported non-default fields produce typed verdicts and never silently vanish.
- Saved trainer arguments agree with the realization observation.
- Two accepted specs with different framework-semantic hashes produce different kwargs.
- Infrastructure-only changes do not alter the framework-semantic hash.

Acceptance criteria:

- A successful enforced-mode YOLO job has no ignored field.
- Equivalent Ultralytics executions share the same realized semantic hash.

### Fidelity PR 8: Accepted And Realized Identity Semantics

Size: S
Depends on: Fidelity PRs 6-7

Goal: Use the correct identity at each lifecycle stage.

Scope:

1. Cut scheduling-time duplicate/no-op checks over from the legacy requested signature to accepted-spec hash.
2. Persist realized-effective hash only after a realization observation.
3. Keep resource/attempt identity separate from semantic identity.
4. Record accepted-to-realized adjustment reason codes.
5. Ensure approved batch recovery does not become a new planner mechanism solely because the realized hash changed.

Tests:

- Accepted duplicate jobs are caught before scheduling.
- Uninitialized jobs have no fabricated realized hash.
- Infrastructure changes preserve semantic identity.
- Training-semantic changes update realized identity and retain lineage to the accepted spec.

Acceptance criteria:

- APIs and stores never use “effective hash” ambiguously.
- Historical unversioned records remain `UNVERIFIED`.

### Fidelity PR 9: Learning And Champion Eligibility

Size: M
Depends on: Fidelity PR 8

Goal: Prevent unverified or mismatched execution from teaching the planner or becoming a champion.

Scope:

1. Add fidelity status to outcome/scorecard inputs.
2. Exclude `MISMATCH` from learned strategy evidence and automatic champion selection.
3. Keep `UNVERIFIED` historical runs visible but clearly marked and policy-controlled.
4. Preserve requested mechanism and realized mechanism/identity separately.
5. Allow `APPROVED_ADJUSTMENT` only under explicit eligibility rules.

Tests:

- A mechanism is not credited when it was not realized.
- Mismatched runs cannot become automatic champions.
- Simulated runs are never treated as verified real-training evidence.
- Old projects remain readable.

Acceptance criteria:

- Planner history and champion logic distinguish verified, adjusted, mismatched, simulated, and legacy evidence.

### Fidelity PR 10: Export And Inference Parity

Size: M
Depends on: Fidelity PR 8
Can proceed in parallel with: Fidelity PR 9

Goal: Build export/runtime preprocessing from realized execution rather than raw requested job configuration.

Scope:

1. Build export preprocessing manifests from final realization observations.
2. Include capability version, accepted-spec hash, and realized-effective hash.
3. Refuse export when realized preprocessing contradicts the export contract.
4. Add the narrow typed run-update/artifact references needed for preprocessing/export evidence so they are not discarded.
5. Keep full framework arguments in the execution record, not compact run summaries.

Tests:

- Training/evaluation/export preprocessing agrees.
- Export hash references the correct execution record.
- Contradictory normalization or resize semantics block export.
- Compact job/run reads expose only status, hashes, and references.

Acceptance criteria:

- An exported model proves which realized preprocessing contract it requires.
- Large receipt payloads are not duplicated into dashboard summaries.

### Fidelity PR 11: Enforcement And Operational Metrics

Size: S
Depends on: Fidelity PRs 9-10

Goal: Make task-aware execution fidelity the default backend policy.

Scope:

1. Change new-plan validation from shadow to enforce after measured gates pass.
2. Require final realization for successful versioned real-training attempts.
3. Keep old/legacy reads compatible.
4. Add metrics for unsupported proposals, pending/not-realized attempts, mismatch verdicts, approved adjustments, and unverified reads.
5. Keep a documented one-switch shadow rollback for one compatibility release.

Rollout gates:

- Representative classification and YOLO smoke jobs are `MATCHED`.
- New successful real-training runs have zero unexplained mismatches.
- No versioned successful job lacks a final realization.
- Old-worker/new-backend compatibility was exercised before enforcement.

Acceptance criteria:

- Unsupported configurations fail before GPU scheduling.
- Completion and evidence eligibility use the same fidelity verdict.
- Rollback changes behavior without rewriting stored records.

### Fidelity PR 12: Mission Control Execution Audit

Size: S
Depends on: Fidelity PR 11

Goal: Make fidelity status understandable without enlarging the default Overview.

Scope:

1. Add a compact status, capability version, accepted hash, realized hash, and adjustment summary to run details.
2. Add an expandable requested-versus-realized semantic diff in the expert audit view.
3. Render `MATCHED`, `APPROVED_ADJUSTMENT`, `MISMATCH`, `UNVERIFIED`, `SIMULATED`, and not-realized states.
4. Link to full bounded receipt data only on demand.

Tests:

- Old records render as unverified without errors.
- Large framework-argument objects are not loaded for the default project view.
- Mismatch and simulated states cannot be mistaken for verified success.
- `npm run build` and existing Mission Control Node tests pass.

Acceptance criteria:

- An operator can answer what was requested, what ran, and whether the result is trustworthy.

## Migration Coordination

Fidelity, Activity, and Calibration tracks add migrations and may be developed in parallel. Do not reserve the same migration number in multiple open PRs. Assign or renumber to the next available number when each migration-bearing PR rebases for merge, and keep migration filenames strictly ordered on the target branch.

## Rollout Plan

1. Deploy capability/spec work with no behavior change.
2. Enable shadow validation and planner capability context.
3. Deploy execution-record plumbing while old workers remain valid.
4. Wire classification, then rebase and wire YOLO.
5. Collect accepted-versus-realized verdicts on representative workloads.
6. Enable evidence/champion eligibility rules.
7. Enable enforcement only after zero unexplained mismatches.
8. Add the UI audit after backend semantics are stable.

## Definition Of Done

This track is complete when:

1. Every new real-training attempt has a versioned accepted spec and final realization or an explicit not-realized state.
2. Every accepted non-default field is executed or rejected before scheduling.
3. Accepted and realized identities have distinct, documented uses.
4. Duplicate detection uses accepted semantics; outcome learning uses realized semantics.
5. Training, evaluation, and export preprocessing agree.
6. Mismatched/simulated runs cannot silently influence champions or agent learning.
7. Mission Control can explain fidelity without loading large payloads by default.

## Risks

1. Presence-aware fields can affect many plan literals and fixtures.
   - Mitigation: shared golden cases and legacy-default readers.

2. Framework upgrades can change defaults.
   - Mitigation: pin versions and gate upgrades with realized-argument tests.

3. Shadow mode can still spend GPU on a configuration known to mismatch.
   - Mitigation: short shadow window, runner-specific enforcement switches, and fast pre-submission verdicts.

4. Execution records could expose environment details.
   - Mitigation: strict allowlists, bounds, redaction, and references for bulky artifacts.

5. Mixed-version rollout can leave pending records.
   - Mitigation: pending is expected in shadow; enforcement waits until old attempts are terminal.

## Non-Goals

- Adding video tasks or new model families.
- Making classifier-only techniques available to YOLO.
- Replacing framework-native training loops.
- Backfilling unverifiable historical semantics.
- Treating hardware/resource identity as planner novelty.
- Allowing agents to bypass deterministic capability validation.
