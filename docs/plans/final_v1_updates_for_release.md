# Final V1 Updates For Release

Date: June 16, 2026

This plan is for the last functional hardening pass before a v1/beta demo release. The target is not perfection. The target is that a user can create or reopen a project, see honest run/export state, download a usable final bundle, and recover from common local failures without the app pretending work succeeded.

Visual-only frontend polish is intentionally split into `docs/plans/final-ui-changed.md`.

## Release Bar

Ship v1 only when these are true:

- A selected champion with a created portable bundle can be saved as a `.zip` outside Mission Control.
- Older projects can be selected, their existing champion export records load, and their portable bundles can be saved when the bytes still exist.
- Export/demo worker callbacks do not create false worker failures after the backend has already accepted a result callback.
- Export states distinguish "not ready yet", "blocked/failed", and "downloadable" clearly enough that a user does not wait forever on a dead export.
- Dead assigned/running jobs recover without requiring an operator to start a new worker manually.

## Current Evidence

- Worker-side portable bundle creation exists in `services/worker/worker/exporting/portable_bundle.py`.
- Export result metadata includes `portable_bundle_uri` and `portable_inference_bundle` in `services/worker/worker/champion_jobs.py`.
- Export records are persisted through `champion_exports` and exposed by `GET /projects/:id/champion/exports` in `services/orchestrator/internal/api/champion.go`.
- The router has export list/create/result routes, but no download route in `services/orchestrator/internal/api/router.go`.
- Mission Control renders export records and portable bundle metadata in `apps/mission-control/src/features/mission/ProjectRoutePanels.tsx`, but it has no save/download action.
- Electron preload exposes `loadModelArtifact`, but no general save/download IPC in `apps/mission-control/electron/preload.cjs`.
- Selecting older projects already triggers a forced detail refresh in `apps/mission-control/src/App.tsx`, but champion export fetch failures are swallowed into an empty export list.
- Job lease recovery exists in `RecoverExpiredJobLeases`, but it runs from polling or explicit calls rather than a standalone recovery loop.

## PR V1-1: Save The Final Portable Export Zip

Priority: P0

Affected run path:

- Champion/export/demo
- UI export rendering
- Electron local filesystem handoff

Current failure:

- The worker can create `portable_inference_bundle.zip`, and the UI can display its `file://` location, but the user cannot save it to a normal destination.
- `loadModelArtifact` only supports model artifact extensions, so it cannot be reused safely for `.zip` without broadening the wrong helper.

Attack method:

1. Add an Electron IPC bridge dedicated to saving export artifacts.
   - In `apps/mission-control/electron/preload.cjs`, expose `saveExportArtifact(options)`.
   - In `apps/mission-control/electron/main.cjs`, add `ipcMain.handle("artifact:saveExport", ...)`.
   - Keep it separate from `artifact:loadModel` so model-inference validation stays narrow.

2. Resolve the artifact source safely.
   - Accept `artifactUri`, optional `artifactPath`, `suggestedName`, and `kind`.
   - For local `file://` or plain paths, validate the resolved file is inside existing allowed roots from `configuredLocalArtifactRoots`.
   - Add a download-specific extension allowlist: `.zip`, `.onnx`, `.pt`, `.pth`, `.torchscript`, `.safetensors`, `.json`.
   - For `s3://`, reuse existing S3 client/env handling and stream the object to disk.
   - Reject `http://`, `https://`, parent traversal, directories, and missing files.

3. Save with a native file dialog.
   - Default filename for portable bundles: `model-express-{project_id}-{export_id}-portable-bundle.zip`.
   - Use `dialog.showSaveDialog`.
   - Copy local files with streaming `fs.createReadStream(...).pipe(fs.createWriteStream(...))`.
   - Return `{ canceled, file_path, bytes }`.

4. Add a renderer action.
   - In `apps/mission-control/src/vite-env.d.ts`, type `saveExportArtifact`.
   - In `apps/mission-control/src/App.tsx`, add `downloadPortableBundle(exportRecordOrBundle)` or pass an `onSavePortableBundle` callback to the export panel.
   - In `ChampionExportDemoPanel`, render a compact "Save ZIP" command next to the portable bundle only when:
     - `portableBundle.status` is `created` or `READY`, and
     - `artifact_uri` or `artifact_path` is present.
   - While saving, disable only that button and show a notice with the saved path or the error.

5. Keep model export download secondary.
   - This PR should prioritize the portable `.zip`.
   - A "Save model artifact" button can be added only if it reuses the same `saveExportArtifact` path and stays small.

Tests:

- Add Electron unit tests in `apps/mission-control/electron/main.test.cjs` for:
  - local zip under allowed export root succeeds,
  - local zip outside allowed roots is rejected,
  - unsupported extension is rejected,
  - canceled dialog returns `canceled: true`.
- Add a focused renderer/model test if the callback selection logic is extracted.
- Run `npm run build` in `apps/mission-control`.

Manual acceptance:

- Create or use a champion with a `portable_inference_bundle.zip`.
- Open Export/Demo.
- Click "Save ZIP".
- Confirm the saved file opens as a zip and contains `manifest.json`, `model.onnx`, examples, requirements, README, and parity output when present.

Regression risk:

- Path validation must not weaken existing model artifact loading.
- Do not allow arbitrary local file reads through the renderer.

## PR V1-2: Older Project Export Archive And Honest Load State

Priority: P0

Affected run path:

- Project history
- Champion/export/demo
- UI polling/rendering

Current failure:

- Older projects are listed and selectable, but export fetch errors are converted to `[]` in `refreshProjectDetail`.
- The export panel currently displays only the first four export records, which can hide older downloadable records.
- App startup defaults to the newest project when no project is selected, so older export recovery is possible but not obvious.

Attack method:

1. Stop masking champion export load failures.
   - In `apps/mission-control/src/hooks/useProjectDetail.ts`, add a small `championExportsStatus` shape:
     - `status: "available" | "empty" | "error"`
     - `message: string`
   - In `refreshProjectDetail`, replace `.catch(() => ({ exports: [] }))` for champion exports with an object that preserves the error message.
   - Preserve previous exports only when the champion still matches and the new request failed, and show the stale-data warning.

2. Persist the user's last selected project.
   - Store `selectedProjectId` in `localStorage`.
   - On `refreshProjects`, prefer the stored id if it still exists.
   - Fall back to newest only when the stored id is absent.

3. Add an export archive view inside the existing Export/Demo panel.
   - Show all export records for the selected project, not only `slice(0, 4)`.
   - Group or sort by:
     - downloadable portable bundle first,
     - ready ONNX/model records next,
     - pending/failed records last.
   - Keep this in `ProjectRoutePanels.tsx`; do not add a new route.

4. Make missing bytes visible.
   - If an export record is `READY` but save validation fails because the file is missing, show a notice like:
     - "The export record exists, but the local artifact file is missing. Re-run export for this project."
   - Do not remove the export record from the UI.

5. Add a clear re-run path.
   - Keep "Request ONNX" available for older projects with a champion.
   - After a re-run, force slow-data refresh so the export archive updates immediately.

Tests:

- Add a frontend test around `buildChampionExportDemo` or extracted export sorting to prove portable bundles surface before stale pending records.
- Add a small App-level test only if the selected-project persistence is already testable without large harness work.
- Run `npm run build`.

Manual acceptance:

- Create two projects.
- Export the older project's champion.
- Restart Mission Control.
- Select the older project.
- Confirm the export archive loads, the portable bundle row is visible, and "Save ZIP" works when bytes exist.

Regression risk:

- Avoid making every live refresh fetch slow export data. Keep slow-data fetch behavior bounded and cache-aware.

## PR V1-3: Remove Duplicate Terminal Callbacks From Export And Demo Workers

Priority: P0

Affected run path:

- Worker execution
- Champion export result
- Champion demo prediction result
- Job state machine

Current failure:

- `run_export_champion_job` calls `report_champion_export_result`.
- The backend `reportChampionExportResult` updates the export record and completes/fails the export job for `READY` and `FAILED`.
- The worker then calls `complete_job` or `fail_job` again.
- `run_champion_demo_prediction_job` has the same pattern after `report_champion_demo_prediction_result`.
- Because `validateJobCallback` rejects terminal/stale attempts, the second callback can produce a conflict and make the worker loop treat an already accepted result as a worker exception.

Attack method:

1. Make result callbacks the single terminal path.
   - In `services/worker/worker/champion_jobs.py`, remove the post-result `client.complete_job` and `client.fail_job` calls from:
     - `run_export_champion_job`
     - `run_champion_demo_prediction_job`
   - Return immediately after successful result callback.

2. Keep backend ownership unchanged.
   - Do not move terminal job state out of `reportChampionExportResult`.
   - Do not move terminal job state out of `reportChampionDemoPredictionResult`.

3. Handle result callback failures normally.
   - If `report_*_result` raises, allow the worker main loop to call `fail_job`.
   - That is still a true failure because the backend did not accept the result.

4. Update tests.
   - In `services/worker/tests/test_champion_jobs.py`, change the fake client expectations:
     - export result recorded,
     - no direct `complete_job` call after result,
     - no direct `fail_job` call after result.
   - Add a fake client that raises from `report_champion_export_result` and assert the exception propagates.

Tests:

- `python -m unittest services.worker.tests.test_champion_jobs -v` from repo root if import paths allow it, otherwise from `services/worker`.
- `go test ./...` from `services/orchestrator` to protect backend callback behavior.

Manual acceptance:

- Run an export job and a demo prediction job.
- Confirm worker stays alive after result acceptance.
- Confirm job terminal state is set exactly once by the result callback.

Regression risk:

- Low. This removes duplicate terminal writes and keeps backend result validation authoritative.

## PR V1-4: Make Export Pending/Blocked States Honest

Priority: P1

Affected run path:

- Champion export
- UI export status
- State machine visibility

Current failure:

- `PENDING_ARTIFACT` can mean several different things:
  - export requested before worker work,
  - source artifact not available,
  - dependency/runtime unavailable,
  - worker attempted export but no future work is actually queued.
- The UI can show "waiting" even when the export is effectively blocked and no worker will make progress.

Attack method:

1. Define v1 state semantics in code comments and tests.
   - `PENDING`: export request exists and worker work may still run.
   - `PENDING_ARTIFACT`: source training artifact is not available yet, and retrying later may make sense.
   - `FAILED`: worker attempted export and hit a deterministic blocker that will not fix itself.
   - `READY`: artifact URI is present and validated.

2. Tighten worker status classification.
   - In `services/worker/worker/champion_jobs.py`, return `FAILED` for deterministic blockers:
     - unsafe source artifact,
     - missing worker-visible source path after an export attempt,
     - unsupported conversion dependency for a requested format when no source artifact exists,
     - object-detection export without a trained detector artifact.
   - Keep `PENDING_ARTIFACT` only for cases where a future artifact can reasonably appear.

3. Surface blocked pending exports in UI.
   - In `buildChampionExportDemo`, if status is pending-like but validation errors exist and no active export job exists, include a limitation:
     - "Export is blocked until the source artifact is available or export is re-run."
   - In `ProjectRoutePanels.tsx`, show validation errors directly under the export row.

4. Do not add a new DB status enum for v1.
   - Use existing statuses and better classification.
   - A migration is unnecessary for this release pass.

Tests:

- Worker tests for deterministic blockers mapping to `FAILED`.
- Frontend model test for pending export with validation errors rendering as blocked/waiting limitation.

Manual acceptance:

- Trigger an export with no source artifact.
- Confirm the UI does not imply the zip will appear automatically.
- Confirm re-requesting export remains available.

Regression risk:

- Some older pending rows will still exist. UI handling must remain backward compatible.

## PR V1-5: Add A Lightweight Lease Recovery Safety Net

Priority: P1

Affected run path:

- Job execution
- Worker startup/death
- UI polling

Current failure:

- `RecoverExpiredJobLeases` exists and runs inside `PollJob`, but if all workers die, jobs can stay `ASSIGNED` or `RUNNING` until another poll or explicit recovery call happens.
- This is acceptable for an internal dev loop but weak for a v1/beta demo.

Attack method:

1. Add a small recovery loop in the orchestrator process.
   - In `services/orchestrator/internal/api/router.go` or a small adjacent file, start a background ticker from `newServer`.
   - Default interval: 60 seconds.
   - Env override: `MODEL_EXPRESS_LEASE_RECOVERY_INTERVAL_SECONDS`.
   - Disable only with `0`.

2. Keep the loop boring.
   - On each tick, call `s.store.RecoverExpiredJobLeases(time.Now().UTC())`.
   - Log count and job ids through existing diagnostics.
   - For recovered terminal failures, create execution events where possible.
   - Do not add a new scheduler or queue.

3. Run one startup sweep.
   - Call recovery once after server creation so stale jobs from a previous crash are handled quickly.

4. Keep polling recovery.
   - Do not remove recovery from `PollJob`; it is still useful.

Tests:

- Go unit test with memory store:
  - assigned job with expired lease requeues on recovery tick helper,
  - max-attempt expired job fails,
  - non-expired job is untouched.
- If the ticker itself is hard to test, extract `recoverExpiredLeasesOnce(now)` and test that.

Manual acceptance:

- Start a job, kill the worker, advance or wait past lease expiry in a test/dev setup.
- Confirm the UI stops showing permanent running state after recovery.

Regression risk:

- Avoid a very short default interval. Recovery does DB writes and should not churn.

## PR V1-6: Final Release Smoke For Export Download

Priority: P1

Affected run path:

- End-to-end demo
- Regression confidence

Current failure:

- Existing smoke coverage validates run and export/demo contracts, but not the final "user saved zip outside app" outcome.

Attack method:

1. Add a focused manual-plus-automated checklist.
   - Extend `docs/model_express_end_to_end_checklist.md` or add a small script under `scripts/`.
   - The smoke should verify:
     - project exists,
     - champion exists,
     - export record exists,
     - portable bundle metadata exists,
     - artifact path/URI resolves,
     - saved zip has `manifest.json`.

2. Prefer a fake/local path for automation.
   - Use worker test fixtures or a temp export root to avoid requiring real training.
   - Do not require Modal for this check.

3. Keep video-demo checklist concrete.
   - One known dataset.
   - One project create/import/profile.
   - One plan execution.
   - One champion export.
   - One saved zip.
   - One reopened older project and saved zip.

Tests:

- `go test ./...`
- Worker export/portable bundle tests.
- `npm run build`.

Manual acceptance:

- The demo script can be followed without guessing what "done" means.

## Accepted V1/Beta Gaps

These should not block the demo release unless they directly break the workflows above:

- No full production scheduler.
- No Kafka/Redis/NATS/workflow engine.
- No perfect cross-machine artifact storage story, as long as local v1 download works and missing artifact bytes are honest.
- No full dataset-profile table promotion.
- No perfect UI redesign.
- No complete remote artifact promotion for every Modal edge case.

## Validation Matrix For The Whole Pass

Run after the PRs above land:

- `go test ./...` from `services/orchestrator`
- `python -m unittest discover -s tests -v` from `services/worker`
- `npm run build` from `apps/mission-control`
- Manual local demo:
  - create new project,
  - run/profile/plan/train enough to select champion,
  - request ONNX export,
  - save portable zip,
  - restart app,
  - select older project,
  - save older portable zip.

