# Failure Fixes Plan

Date: June 17, 2026

This plan targets the runtime failure modes that can leave a Model Express user without a usable final model even though parts of the run appear to have completed. The goal is a releasable local v1, not a scheduler rewrite.

## Release Goal

A release is acceptable when a normal local run can move from dataset creation to a selected champion and exportable model, and when common failures become visible, retryable, or degraded instead of silently producing no model.

Release success means:

- A successful `train_experiment` job cannot be marked successful unless the backend has enough persisted output to select and export a model.
- Every terminal training path, including lease recovery, runs the same post-training decision logic.
- LLM or agent failures cannot block deterministic champion fallback.
- A selected champion always has an export state: `PENDING`, `PENDING_ARTIFACT`, `READY`, or `FAILED`.
- Worker startup and polling survive temporary orchestrator downtime.
- Mission Control distinguishes "empty" from "failed to load" for model-critical panels.

Non-goals:

- Replacing the current orchestrator, worker, and Mission Control architecture.
- Building a general distributed scheduler.
- Making LLM agents required for v1.
- Reworking model scoring beyond the minimum needed to prevent no-model dead ends.

## Recommended Sequence

Land these as small PRs in order:

1. P0: Make post-training hooks degrade safely when the LLM planner fails.
2. P0: Run terminal hooks for training jobs failed by lease recovery.
3. P0: Gate training job completion on persisted model outputs.
4. P0: Guarantee champion export state after champion persistence.
5. P1: Add worker and Modal dispatcher retry loops around orchestrator outages.
6. P1: Surface frontend fetch failures and stale state.
7. P1: Add release smoke verification that exercises success, partial failure, and recovery.

This order fixes backend truth first, then process liveness, then UI honesty. The UI should not be asked to compensate for false backend success.

## Shared Contracts

### Training Success Contract

For new `train_experiment` completions, `SUCCEEDED` must mean all of the following are true:

- The callback identity is valid for the active training attempt.
- A training summary already exists for the job and reports `SUCCEEDED`.
- A training evaluation already exists for the job.
- The evaluation model profile contains an exportable model artifact URI recognized by the existing champion/export artifact helpers.

If these are not true, `/jobs/:id/complete` should reject completion with a clear 409-style store error and leave the job non-terminal so the worker can report failure or the lease recovery path can handle it.

This applies only to new terminal transitions. Existing historical projects should still load.

### Terminal Hook Contract

Every terminal `train_experiment` transition must enqueue the same terminal hook path:

- normal `/complete`
- normal `/fail`
- max-attempt failure from lease recovery
- cancellation paths that intentionally select the best available model

Hooks must be idempotent. Re-running the hook must not create duplicate champion decisions or duplicate export jobs.

### Champion Export Contract

After a champion is selected, the project should not be left with a champion and no export row. Export readiness can be blocked, but the block must be explicit:

- `PENDING` when export work has been queued.
- `PENDING_ARTIFACT` when the champion exists but no source artifact is available yet.
- `FAILED` when export creation or validation fails.
- `READY` when a validated artifact is available.

Telemetry event insertion must not be allowed to skip export creation.

### UI Honesty Contract

Mission Control should preserve useful stale data, but it must label stale or failed panel data. Empty because the backend returned zero rows is different from empty because the request failed.

## P0-1: LLM Planner Failure Must Fall Back To Deterministic Selection

### Current Failure

`runPlanningLoopAfterTrainingJob` calls the LLM planner and returns immediately on planner error. That skips the deterministic reviewer and best-available champion fallback.

Relevant files:

- `services/orchestrator/internal/api/agent_followups.go`
- `services/orchestrator/internal/api/agent_runtime.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/api/run_smoke_test.go`

### Fix

Change `runPlanningLoopAfterTrainingJob` so an LLM planner error records a degraded planner failure but still executes deterministic fallback.

Implementation details:

1. Keep `runExperimentPlannerAfterTrainingJob` responsible for LLM invocation diagnostics.
2. In `runPlanningLoopAfterTrainingJob`, replace the early return on planner error with:
   - log and record the planner failure as it does today,
   - call `runAutomaticExperimentReviewAfterTrainingJob(job)`,
   - if deterministic review cannot select a champion, call `selectBestAvailableChampionForTerminalPlanStop` for the job's plan when there are no open jobs.
3. Make the hook outcome explicit:
   - `finished` when planner or fallback produced a terminal decision,
   - `degraded` when fallback ran after a planner failure,
   - `failed` only when no decision could be made and no successful run exists.
4. Do not create a second decision if a planner decision already exists for the plan/job.

### Why This Should Not Cause Other Issues

The deterministic reviewer and best-available champion selector already exist and are used by other terminal-stop paths. This change does not introduce a new decision model; it only ensures the existing fallback is reached after an LLM failure.

The main risk is duplicate decisions. Avoid that by checking existing agent decisions before creating fallback decisions, using the same idempotency patterns already used in planner follow-up handling.

### Tests

Add or extend Go tests in `services/orchestrator/internal/api/handlers_test.go`:

- Seed a project, plan, completed training summary, and evaluation with an artifact.
- Configure the LLM planner to fail or return malformed output.
- Run terminal hooks.
- Assert a champion exists.
- Assert a champion export exists.
- Assert an `AGENT_FAILED` or degraded event exists.
- Assert no duplicate decisions are created if the hook is called twice.

Also extend `TestFakeRunSmokeEndToEndSuccessVisibility` in `services/orchestrator/internal/api/run_smoke_test.go` or add a neighboring smoke test for "LLM failure still selects model".

## P0-2: Lease Recovery Must Trigger Training Terminal Hooks

### Current Failure

Normal `/complete` and `/fail` callbacks enqueue terminal hooks. Lease recovery can mark a max-attempt training job `FAILED`, but it only upserts a failed summary, updates worker demand, and records an execution event. It does not enqueue the terminal hook path.

Relevant files:

- `services/orchestrator/internal/api/lease_recovery.go`
- `services/orchestrator/internal/api/lease_recovery_test.go`
- `services/orchestrator/internal/store/postgres_job_records.go`
- `services/orchestrator/internal/api/jobs.go`

### Fix

In `handleRecoveredExpiredLeaseFailure`, after the failed training summary is persisted and worker demand is updated, enqueue terminal hooks for `train_experiment` jobs.

Implementation details:

1. Call `s.enqueueTrainingTerminalHooks(job)` for recovered terminal `train_experiment` jobs.
2. Keep the existing failed summary update before enqueueing hooks so downstream logic sees terminal state.
3. Keep the existing `EXECUTION_FAILED` event.
4. If hook enqueue fails, record a visible event or log with project, plan, and job IDs. The hook itself should be best-effort but recoverable by a later reconciliation test or manual retry.

### Why This Should Not Cause Other Issues

This makes lease recovery match the normal failure callback path. It should not change queued or requeued lease-expiry behavior because the hook is only for jobs that reached terminal failure after maximum attempts.

The main risk is running hooks twice if recovery is called repeatedly. The job state transition to terminal failure happens once, and terminal hook logic should be idempotent. The tests must explicitly call recovery twice to prove this.

### Tests

Extend `services/orchestrator/internal/api/lease_recovery_test.go`:

- Create a plan with one successful run and one assigned/running training job at max attempts.
- Expire the lease.
- Run recovery.
- Assert the expired job is `FAILED`.
- Assert terminal hooks select the prior successful run as champion.
- Assert a champion export row exists.
- Run recovery again and assert no duplicate champion/export rows.

## P0-3: Training Completion Must Not Manufacture Success

### Current Failure

`completeJob` calls `UpsertTrainingRunSummary` with only `Status: SUCCEEDED`. If no summary exists, the store creates a summary from job config with empty metrics. That can make a job look successful even when no evaluation or artifact exists.

Relevant files:

- `services/orchestrator/internal/api/jobs.go`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres_runs.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/runs/model.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/api/run_smoke_test.go`
- `services/worker/worker/training/local.py`
- `services/worker/worker/training/modal_app.py`

### Fix

For `train_experiment`, validate readiness before accepting `/jobs/:id/complete`.

Implementation details:

1. Add a backend helper near `completeJob`, for example `validateTrainingCompletionReadiness(job jobs.ExperimentJob) error`.
2. The helper should:
   - call `GetTrainingRunSummary(job.ID)`,
   - require `summary.Status == jobs.StatusSucceeded`,
   - call `GetTrainingRunEvaluation(job.ID)`,
   - require an artifact URI from `evaluation.ModelProfile` using the same keys already accepted by champion/export code: `artifact_uri`, `onnx_artifact_uri`, `model_artifact_uri`, `export_artifact_uri`, `checkpoint_uri`, and framework-specific variants already handled by `championArtifactURIFromEvaluation`.
3. In `completeJob`, run this helper after callback validation and before `CompleteJob`.
4. Remove the automatic success-summary creation in the training complete path. If a summary exists, do not overwrite it with an empty update.
5. Return a clear invalid request error when readiness is missing, for example "training completion requires a succeeded summary and exportable evaluation artifact".
6. Keep non-training templates unchanged.

### Why This Should Not Cause Other Issues

The existing local and Modal training paths already post final summary and evaluation before calling `complete_job`, so the intended happy path remains valid. This change blocks only the false-success path where completion arrives without model outputs.

The main compatibility risk is any older or hidden worker path that completes training before posting evaluation. That path would now fail visibly instead of silently producing no model. For release quality, that is the right failure mode.

Existing historical summaries should not be migrated or rejected on read. The gate applies only when a job is transitioning to terminal success.

### Tests

Add Go tests in `services/orchestrator/internal/api/handlers_test.go`:

- Completing a `train_experiment` with no summary returns an error and does not mark the job `SUCCEEDED`.
- Completing with a succeeded summary but no evaluation returns an error.
- Completing with evaluation but no artifact URI returns an error.
- Completing with succeeded summary and artifact-bearing evaluation succeeds.
- Non-training job completion is unaffected.

Extend `TestFakeRunSmokeMidRunFailureStaysVisibleAndRejectsStaleCompletion` or add a new smoke test that proves missing outputs do not become a successful model.

## P0-4: Champion Selection Must Always Produce An Export State

### Current Failure

`persistProjectChampionFromDecision` upserts the champion, then writes a `CHAMPION_SELECTED` event, then calls `ensureChampionExport`. If event creation fails, export creation is skipped. If export creation fails, the error is logged but not persisted as a visible export state.

Relevant files:

- `services/orchestrator/internal/api/champion.go`
- `services/orchestrator/internal/api/champion_scoring.go`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres_champion.go`
- `services/orchestrator/internal/execution/model.go`
- `services/orchestrator/internal/api/handlers_test.go`

### Fix

Make export creation independent from telemetry event success.

Implementation details:

1. In `persistProjectChampionFromDecision`, call `ensureChampionExport` after champion upsert even if `CreateExecutionEvent(EventChampionSelected)` fails.
2. Treat `CHAMPION_SELECTED` event creation as telemetry. Log and continue if it fails.
3. If `ensureChampionExport` fails:
   - create or update a champion export row with `FAILED` when possible,
   - store the error in `validation_errors` or export metadata,
   - record `EXECUTION_FAILED` or a new export-specific event if a new event constant is justified.
4. Keep `PENDING_ARTIFACT` behavior for champions without a source artifact. That state is useful and should remain distinct from `FAILED`.
5. Ensure export creation is idempotent for repeated champion persistence calls.

### Why This Should Not Cause Other Issues

Champion selection is already committed before export creation, so this plan does not add a new partial-commit class. It reduces the existing partial state by ensuring an export record exists after champion selection.

The main risk is extra `FAILED` export rows if transient store errors occur. Avoid this by upserting one export row per champion/job/format, following the existing `UpsertChampionExport` behavior in the store.

### Tests

Add Go tests in `services/orchestrator/internal/api/handlers_test.go`:

- Force `CreateExecutionEvent` to fail after champion upsert and assert export creation is still attempted.
- Force export creation failure and assert the project exposes a `FAILED` export state rather than no export.
- Re-run champion persistence and assert export rows are not duplicated.
- Assert `PENDING_ARTIFACT` still appears when no artifact URI exists.

## P1-5: Workers Must Survive Temporary Orchestrator Downtime

### Current Failure

The local worker and Modal dispatcher can exit when registration or polling fails. Lease recovery helps only after a job is assigned; it does not help queued jobs when no worker remains alive.

Relevant files:

- `services/worker/worker/main.py`
- `services/worker/worker/orchestrator_client.py`
- `services/worker/worker/training/modal_dispatcher.py`
- `services/worker/tests/test_training_modal_helpers.py`
- new worker lifecycle tests under `services/worker/tests/`

### Fix

Add bounded retry/backoff around worker registration, polling, and failure reporting.

Implementation details:

1. In `main.py`, wrap `client.register_worker()` in retry with exponential backoff and jitter.
2. In the main poll loop, catch transport and HTTP errors from `client.poll_job()`, log once per backoff window, sleep, and continue.
3. If job execution raises and `client.fail_job()` also fails, log the failed reporting attempt and continue cautiously. The lease timeout will recover the assigned job if the backend remains unreachable.
4. In `modal_dispatcher.py`, keep the dispatcher process alive when worker registration or `poll_job` fails.
5. Preserve existing callback strictness. Do not make callback identity optional for terminal callbacks.
6. Keep retry defaults conservative so a down orchestrator does not spin CPU or flood logs.

### Why This Should Not Cause Other Issues

The worker already treats orchestrator communication as the source of truth. Retrying registration and polling does not change job semantics; it only prevents process exit on temporary unavailability.

The main risk is a worker continuing after it failed to report a job failure. Lease recovery already exists for this state. The worker should avoid immediately claiming unrelated new work while its current assigned job may still be active from the orchestrator's perspective.

### Tests

Add Python tests:

- Worker startup with the first N register attempts failing eventually succeeds.
- Poll failures do not terminate the loop.
- `fail_job` failure during exception handling does not crash the worker process.
- Modal dispatcher poll failures keep the dispatcher alive and retry.

Run:

- `python -m pytest services/worker/tests`
- `python -m compileall services/worker/worker`

## P1-6: Mission Control Must Show Failed Or Stale Model-Critical Data

### Current Failure

Several project detail fetches catch errors and substitute empty data. Live refresh failures mostly set health to null while old detail can remain on screen. That can make a dead run look quiet or incomplete rather than failed.

Relevant files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/api/missionControlClient.ts`
- `apps/mission-control/src/features/mission/ProjectRoutePanels.tsx`
- `apps/mission-control/src/hooks/useWorkerSupervisor.ts`
- `apps/mission-control/src/vite-env.d.ts` if types need extension

### Fix

Add lightweight per-panel load status for model-critical detail slices.

Implementation details:

1. Track status for:
   - training evaluations,
   - agent decisions,
   - worker requirements,
   - champion exports,
   - demo predictions,
   - feedback,
   - live refresh health.
2. Use a small shape:
   - `status: "available" | "empty" | "stale" | "error"`
   - `message?: string`
   - `last_success_at?: string`
3. Preserve stale data when available, but render a compact stale/error label in the relevant panel.
4. Do not block the whole dashboard when optional data fails.
5. Keep existing champion export stale preservation, but make the stale/error state visible.

### Why This Should Not Cause Other Issues

This is display-state only. It should not change backend calls, job transitions, or export behavior. It reduces false confidence while preserving the current degraded rendering behavior.

The main risk is noisy UI. Keep the copy compact and tied only to panels that affect "do I have a model?" decisions.

### Tests

Add focused frontend tests if the project already has renderer tests for these panels. If not, extract the status merge logic into a small pure helper and test it.

Manual acceptance:

- Mock or force `/training-run-evaluations` to return 500 while jobs and champion load.
- Confirm the evaluations panel shows stale/error state, not a clean empty state.
- Stop the orchestrator during live refresh and confirm the project shows stale timestamp or backend unavailable state.

Run:

- `npm run build` in `apps/mission-control`

## P1-7: Release Smoke Verification

### Purpose

The fixes above are about state contracts. Unit tests are necessary but not enough. Add a small smoke matrix that proves the release cannot silently end without a model in the known failure cases.

Relevant files:

- `services/orchestrator/internal/api/run_smoke_test.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/api/lease_recovery_test.go`
- `services/worker/tests/`
- optionally `docs/plans/final_v1_updates_for_release.md` if this plan becomes part of the release checklist

### Smoke Cases

Add or confirm these cases:

1. Happy path:
   - dataset profiled,
   - plan created,
   - training summary and evaluation reported,
   - job completed,
   - champion selected,
   - export row created.

2. LLM planner failure:
   - training succeeds,
   - planner fails,
   - deterministic fallback selects champion,
   - export row exists.

3. Missing training outputs:
   - completion without summary/evaluation/artifact is rejected,
   - job does not become `SUCCEEDED`,
   - no champion is selected from false success.

4. Lease max-attempt failure:
   - assigned job expires after max attempts,
   - failure summary is visible,
   - terminal hooks run,
   - best previous successful run can be selected.

5. Champion export creation failure:
   - champion exists,
   - export state is `FAILED` or `PENDING_ARTIFACT`,
   - UI/API does not show a champion with no export state.

6. Worker outage:
   - worker starts before orchestrator is available,
   - process stays alive,
   - job is claimed after orchestrator returns.

### Release Commands

Before calling the failure fixes releasable, run the closest equivalent of:

```powershell
cd services/orchestrator
go test ./...
```

```powershell
python -m pytest services/worker/tests
python -m compileall services/worker/worker
```

```powershell
cd apps/mission-control
npm run build
```

Also run the local app manually once:

1. Start infra with `docker compose -f infra/compose.yaml up -d`.
2. Start the Go orchestrator.
3. Start Mission Control.
4. Start or allow Mission Control to supervise workers.
5. Create a project with a small image dataset.
6. Confirm a champion and export record appear.
7. Kill a worker mid-run and confirm recovery is visible.

## Files Touched Summary

| Area | Files |
| --- | --- |
| Terminal hooks and planner fallback | `services/orchestrator/internal/api/agent_followups.go`, `services/orchestrator/internal/api/agent_runtime.go`, `services/orchestrator/internal/api/handlers_test.go`, `services/orchestrator/internal/api/run_smoke_test.go` |
| Lease recovery | `services/orchestrator/internal/api/lease_recovery.go`, `services/orchestrator/internal/api/lease_recovery_test.go` |
| Training completion validation | `services/orchestrator/internal/api/jobs.go`, `services/orchestrator/internal/store/store.go`, `services/orchestrator/internal/store/memory.go`, `services/orchestrator/internal/store/postgres_runs.go`, `services/orchestrator/internal/api/handlers_test.go`, `services/orchestrator/internal/api/run_smoke_test.go` |
| Champion export guarantees | `services/orchestrator/internal/api/champion.go`, `services/orchestrator/internal/store/store.go`, `services/orchestrator/internal/store/memory.go`, `services/orchestrator/internal/store/postgres_champion.go`, `services/orchestrator/internal/execution/model.go`, `services/orchestrator/internal/api/handlers_test.go` |
| Worker retry behavior | `services/worker/worker/main.py`, `services/worker/worker/orchestrator_client.py`, `services/worker/worker/training/modal_dispatcher.py`, `services/worker/tests/` |
| Frontend failure visibility | `apps/mission-control/src/App.tsx`, `apps/mission-control/src/api/missionControlClient.ts`, `apps/mission-control/src/features/mission/ProjectRoutePanels.tsx`, `apps/mission-control/src/hooks/useWorkerSupervisor.ts` |
| Release verification | `services/orchestrator/internal/api/run_smoke_test.go`, `docs/plans/final_v1_updates_for_release.md` if the release checklist should link this plan |

## Risks And Mitigations

### Risk: Completion Gate Blocks A Legitimate Worker

If a worker currently completes before posting evaluation/artifact, P0-3 will expose it as a failure.

Mitigation:

- Verify local and Modal training paths post evaluation before completion.
- Keep the error message actionable.
- Add tests for the current local and Modal callback order.

### Risk: Duplicate Champion Or Export Rows

Terminal hooks may be called from more paths after P0-1 and P0-2.

Mitigation:

- Reuse existing idempotent champion/export upsert behavior.
- Add repeated-hook tests.
- Prefer upsert by champion/job/format rather than create-only logic.

### Risk: UI Becomes Noisy

More visible stale/error state can make the app look less polished.

Mitigation:

- Only show compact status for model-critical panels.
- Preserve stale data when useful.
- Avoid modal alerts or blocking banners for optional data.

### Risk: Worker Retry Loops Hide Persistent Misconfiguration

Retries could make a bad URL or missing orchestrator look like "still trying" forever.

Mitigation:

- Log backoff state with the orchestrator URL.
- Surface worker requirement failures in Mission Control.
- Use bounded or capped backoff and visible status, not silent loops.

## Recommendation

Ship the backend P0 fixes first: planner fallback, lease-recovery hooks, training completion gating, and champion export state. These are the changes that directly prevent false success and no-model terminal states.

Then add worker liveness and frontend honesty. Those improve recovery and diagnosis, but they should not be used as substitutes for backend state correctness.

The releasable version should be considered blocked until the smoke matrix proves:

- no false successful training job without model outputs,
- no terminal training path that bypasses champion selection,
- no selected champion without an export state,
- no worker process death on a temporary orchestrator outage.
