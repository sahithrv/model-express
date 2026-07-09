# Progress Reporting And Activity Efficiency Plan

Status: Proposed
Date: 2026-07-09
Track: Activity
Depends on: None
Can proceed in parallel with: Fidelity and Ranking, subject to worker-file sequencing below

## Purpose

Make Mission Control cheaper to keep live and clearer during long-running work, especially before the first training epoch.

The target is one authoritative current-state snapshot plus a durable, cursor-safe transition stream. Mission Control should not reconstruct a small feed from large agent records or broadly refresh the project whenever any event arrives.

## Current Problem

- The activity SSE polls every five seconds during active work and ten seconds while idle.
- Each tick synthesizes events from execution events, jobs, agent invocations, and agent decisions.
- Invocation reads include full messages/context/output records even when the feed needs only a few status fields.
- Decision reads are not bounded at the store query.
- Reconnect searches `Last-Event-ID` only in the latest synthesized window and can miss a larger burst.
- The connection keeps a growing delivered-ID map instead of using a durable cursor.
- Mission Control also performs broad ten/thirty-second refreshes, and SSE events can trigger more broad refreshes.
- Modal stage telemetry is generally unavailable before an epoch summary; YOLO exposes even less before completion.

## Target Data Model

Use two records with different retention semantics:

1. `execution_events`
   - Append-only durable transitions.
   - Monotonic cursor.
   - Written only for meaningful stage/status/progress-boundary changes.
2. `job_progress`
   - Replaceable current snapshot for one active job attempt.
   - Updated by heartbeats without appending an event every time.

The backend owns authoritative queue, retry, cancellation, lease-recovery, completion, and failure states. Workers report observations through `finalizing`; a worker must not mark a job `completed` before summary/evaluation processing succeeds.

## Stable Stage Taxonomy

Version these coarse stages independently from worker-specific detail codes:

1. `queued`
2. `worker_starting`
3. `remote_scheduled`
4. `environment_starting`
5. `dataset_materializing`
6. `data_loading`
7. `model_initializing`
8. `training`
9. `evaluating`
10. `exporting`
11. `finalizing`
12. `completed`
13. `failed`
14. `cancelled`

UI should show elapsed time and last update, not an uncalibrated ETA.

## PR Breakdown

### Activity PR 1: Baseline The Current Live Path

Size: S
Behavior change: Diagnostic only

Goal: Establish measurable before/after targets without logging payload contents.

Scope:

1. Instrument activity duration, events returned, response bytes, store-call count, reconnects, and errors.
2. Instrument Mission Control requests per minute by endpoint category.
3. Record event-to-visible-state latency in bounded diagnostics.
4. Add a deterministic request-count harness instead of flaky wall-clock assertions.
5. Capture active and idle baseline scenarios.

Acceptance criteria:

- Baseline reports store reads/tick, bytes/tick, broad GETs/minute, and visibility latency.
- Metrics contain counts, sizes, durations, and reason codes only.
- No refresh or activity behavior changes.

### Activity PR 2: Slim And Bound Legacy Activity Reads

Size: S
Depends on: Activity PR 1

Goal: Reduce immediate cost while retaining the v1 activity JSON contract.

Scope:

1. Add a lightweight invocation activity projection that excludes full messages, context, raw output, and parsed output.
2. Project only the downstream scalars required by `activityFromAgentInvocation`, including validation status/error, retry status, retry attempt, and completion state.
3. Add SQL- and memory-bounded decision activity queries.
4. Continue bounded job/execution-event reads.
5. Keep mapping and sanitization behavior unchanged.

Tests:

- Store limits are applied before records reach the API.
- A spy store proves only bounded projection calls are used.
- Retry/rejection activity remains present.
- Large prompt/output fixtures never enter the projection.
- Existing sanitization tests remain green.

Acceptance criteria:

- V1 no longer reads full LLM payloads for activity.
- Decision reads are bounded in both stores.
- Baseline metrics show a material read/payload reduction.

### Activity PR 3: Cursor Schema And Raw Event Stream V2

Size: M
Depends on: Activity PR 2

Goal: Make existing execution events cursor-safe before claiming they cover all activity.

Scope:

1. Add a monotonic event sequence using the migration number assigned at merge/rebase time.
2. Backfill existing rows deterministically by `created_at`, then stable event ID.
3. Enforce non-null/unique sequence and index `(project_id, sequence)` after backfill.
4. Add a stable event idempotency key and uniqueness rule; give legacy rows a deterministic key derived from their existing ID.
5. Add `ListProjectExecutionEventsAfter(projectID, cursor, limit)` to both stores.
6. Add a feature-flagged v2 raw-event SSE endpoint using sequence as SSE ID.
7. Project every legacy and new row through the same bounded allowlist/redaction envelope at read time; never serialize stored message/payload JSON directly.
8. Support bounded catch-up pagination, idle keepalive comments, cancellation, and defined invalid/too-old cursor recovery.
9. Remove the in-memory delivered map from the v2 path.
10. Keep v1 unchanged.

Tests:

- Old rows receive stable ordered cursors.
- Reconnect has no duplicate.
- Bursts larger than one page have no gap.
- Same-timestamp ordering is stable.
- Payload bounds/redaction and request cancellation work.
- Historical rows containing storage URIs, paths, or disallowed payload keys are sanitized by the v2 read projection.

Acceptance criteria:

- V2 accurately streams raw execution events.
- Documentation states that v2 is not yet authoritative for synthesized job/agent transitions.
- A backend flag can disable the endpoint without data repair.

### Activity PR 4: Durable Producer Coverage

Size: M
Depends on: Activity PR 3

Goal: Ensure every important synthesized transition has a durable producer before the UI relies on v2.

Scope:

1. Add a typed, allowlisted execution-event creation helper.
2. Give each transition a deterministic idempotency key.
3. Refactor store-owned job transitions so the job mutation and event append occur in the same database transaction, or use a transactional outbox with equivalent guarantees.
4. Emit events from server-owned job queue, assignment, first metric/running, retry, cancellation, completion, failure, lease recovery, and terminal recovery paths.
5. Emit agent validation rejected/retrying/accepted transitions without storing prompt/output payloads.
6. Define stable category, phase, status, severity, job/plan identity, safe message, and bounded metadata.
7. Dual-write during rollout while v1 synthesis remains available.

Tests:

- Each authoritative job transition produces exactly one durable transition event.
- Repeated callbacks/recovery are idempotent.
- A crash/rollback cannot commit the job mutation without its event or vice versa.
- Validation retry events preserve attempt/retry identity.
- Events contain no prompts, raw outputs, storage URIs, or secrets.
- A producer-coverage test maps every v1 activity type to a durable event or a documented non-transition exclusion.

Acceptance criteria:

- V2 can reproduce state-changing v1 activity without querying job/agent history.
- V2 is marked authoritative only after the coverage test passes.

### Activity PR 5: Job Progress Schema And Server Lifecycle

Size: M
Depends on: Activity PR 4

Goal: Add the current-progress store and keep it correct during server-owned lifecycle changes.

Scope:

1. Add `job_progress` using the migration number assigned at merge/rebase time.
2. Store project/job/attempt, taxonomy version, stage, detail code, status, current/total/unit, safe message, revision, heartbeat/update times, and bounded metadata.
3. Initialize/reset progress on queue, assignment, new attempt, and retry.
4. Update authoritative terminal state only after backend completion/failure/cancellation/recovery transitions.
5. Ensure an abandoned worker attempt cannot remain the project's apparent active stage.
6. Refactor/reuse the PR 4 transition writer so the authoritative job row, progress snapshot, and one corresponding event commit in the same transaction; do not append a second lifecycle event.

Tests:

- Retry creates a fresh attempt-scoped snapshot.
- Lease expiry/recovery cannot leave old progress active.
- Completion is recorded only after backend terminal processing.
- Terminal state cannot regress.
- Transaction rollback leaves job row, snapshot, and event all unchanged.

Acceptance criteria:

- Server-owned lifecycle always overrides stale worker observations.
- No worker callback is required yet.

### Activity PR 6: Attempt-Authenticated Progress Callback

Size: S
Depends on: Activity PR 5

Goal: Accept worker observations safely without flooding the event log.

Scope:

1. Add authenticated `POST /jobs/:id/progress` using existing attempt-token validation.
2. Require monotonic attempt revision and idempotent duplicate handling.
3. Reject stale attempts.
4. Use server receipt time for heartbeat/staleness.
5. Upsert snapshot and append a boundary transition event in one transaction.
6. Treat unchanged-stage heartbeat as snapshot-only.
7. Allow workers to report observations through `finalizing`; backend remains terminal authority.

Tests:

- Duplicate/older revisions are safe.
- Stale attempts cannot update active progress.
- Same-stage heartbeat does not create an event.
- Stage/progress-boundary transition creates one event atomically.
- Metadata/message/range limits are enforced.

Acceptance criteria:

- Heartbeat traffic is bounded independently from durable event growth.

### Activity PR 7: Python Progress Reporter And Compatibility

Size: S
Depends on: Activity PR 6

Goal: Add reusable worker-side reporting before integrating individual providers.

Scope:

1. Add `ProgressReporter` with monotonic revisions, throttling, bounded retry, and safe payload construction.
2. Make 404/unsupported progress endpoints non-fatal for new-worker/old-backend compatibility.
3. Add a worker enable/disable flag with documented default and rollback.
4. Ensure reporting failures never fail training.
5. Retain local diagnostics for reporting outages.

Tests:

- Same-stage heartbeat throttling respects the configured interval.
- Transitions bypass heartbeat throttling.
- 404, timeout, and transient failure behavior is bounded/non-fatal.
- Attempt identity and revisions are preserved.

Acceptance criteria:

- Provider integrations can report through one tested client.
- Mixed-version rollout is safe before any provider starts using it.

### Activity PR 8: Modal Classification Progress Integration

Size: M
Depends on: Activity PR 7
Implementation order: Rebase after Fidelity PR 6 because both touch classification setup in `modal_app.py`

Goal: Make classification progress visible before epoch one.

Scope:

1. Report dispatcher/provider submission and remote scheduling observations.
2. Report environment startup, dataset materialization, data loading, model initialization, training epoch `N/M`, evaluation, export, and finalizing.
3. Keep detailed legacy stage telemetry as optional diagnostic dual-write.
4. Do not let the worker publish authoritative completion.

Tests:

- Stage appears before the first epoch.
- Epoch progress is monotonic.
- Callback outage does not fail training.
- Terminal backend transition occurs after summary/evaluation callback processing.

Acceptance criteria:

- A classification cold start no longer appears as generic unexplained running time.

### Activity PR 9: Modal YOLO Progress Parity

Size: S
Depends on: Activity PR 8
Implementation order: Rebase after Fidelity PR 7 because both touch YOLO setup in `modal_app.py`

Goal: Give YOLO the same coarse progress contract.

Scope:

1. Map YOLO callbacks/epochs into the stable stage taxonomy.
2. Report materialization, model initialization, epoch `N/M`, evaluation, export, and finalizing.
3. Preserve framework-specific detail only in bounded detail codes.

Tests and acceptance:

- YOLO reports before epoch one and during epochs.
- The same API/UI taxonomy works for classifier and detection.
- YOLO finalization does not race backend completion processing.

### Activity PR 10: Local And Persistent Provider Parity

Size: S
Depends on: Activity PRs 7-9

Goal: Avoid a Modal-only progress abstraction.

Scope:

1. Integrate local/simulator and persistent-GPU provider paths.
2. Mark simulator/detail semantics explicitly.
3. Share throttling and stage mappings where possible.

Acceptance criteria:

- Every supported provider returns a valid taxonomy version and current stage.
- Provider-specific stages do not leak into the stable contract.

### Activity PR 11: Compact Live-State Endpoint

Size: M
Depends on: Activity PRs 5-6
Can proceed in parallel with: Worker integration PRs 7-10

Goal: Seed Mission Control from one bounded, race-safe current-state response.

Scope:

1. Add `GET /projects/:id/live-state`.
2. Return aggregate job/worker state, active progress, last heartbeat, stale flag, next expected stage, latest important event, and snapshot cursor.
3. Exclude prompts, evaluations, plans, histories, storage paths, and raw configs.
4. Read cursor then snapshots within one repeatable-read transaction, or use an equivalently proven algorithm, so streaming after the cursor cannot miss a committed transition.
5. Use fixed response/query budgets and optional ETag/revision support.

Tests:

- Queued, active, retrying, stale, blocked, mixed-job, and terminal projects are correct.
- A transition racing the snapshot is either represented in the snapshot or returned after its cursor.
- Query count and response size stay bounded.
- Metadata is allowlisted/redacted.

Acceptance criteria:

- One response renders current operational state without broad project detail reads.

### Activity PR 12: Mission Control Incremental Data Layer

Size: M
Depends on: Activity PRs 4 and 11

Goal: Add v2 state consumption without changing the visible layout or disabling fallback polling yet.

Scope:

1. Extract a focused hook/reducer from `App.tsx`.
2. Seed from live state and apply cursor-ordered SSE events.
3. Add a typed event-to-resource invalidation map.
4. Coalesce burst invalidations by resource.
5. Retain current polling and presentation during this PR.
6. Add a client feature flag and automatic fallback for unsupported/404 v2 endpoints.

Tests:

- Snapshot plus subsequent cursor events compose without gaps.
- Duplicate events are idempotent.
- Epoch event invalidates metrics only; terminal/decision/champion events target their resources.
- Old backends select fallback behavior.

Acceptance criteria:

- The incremental model can run in shadow and compare its derived state with current UI state.

### Activity PR 13: Progress Presentation And Polling Cutover

Size: M
Depends on: Activity PR 12 and worker progress PRs 8-10

Goal: Render meaningful stages and stop redundant broad refreshes when v2 is healthy.

Scope:

1. Show stable stage, `N/M`, elapsed, last update, stale/blocked reason, and next expected stage.
2. Disable broad interval refresh while v2 is connected and healthy.
3. Fetch only resources invalidated by typed events.
4. Retain conservative fallback polling and manual refresh when disconnected.
5. Add a client rollback switch that restores current refresh behavior.
6. Do not display speculative ETA.

Tests:

- Connected idle state performs no broad project refresh.
- One epoch event causes at most one targeted metric fetch.
- Burst events coalesce.
- Disconnect restores fallback polling.
- Stale progress is visually distinct from failure.

Acceptance criteria:

- Current stage is visible within one heartbeat interval.
- Request volume drops materially from the PR 1 baseline.
- Existing detail tabs still populate on demand.

### Activity PR 14: Compatibility Rollout And Measurement

Size: S
Depends on: Activity PR 13

Goal: Prove v2 under mixed versions and production-like bursts while retaining v1.

Scope:

1. Track v2/fallback use, reconnect gaps, request volume, query count, bytes, and event latency.
2. Exercise old UI/new backend, new UI/old backend, and old/new worker combinations.
3. Run burst/reconnect soak scenarios.
4. Keep v1 synthesis and legacy stage telemetry for at least one compatibility release.
5. Document go/no-go and rollback thresholds.

Acceptance criteria:

- Active-run API request volume is at least 80% below PR 1 baseline.
- No missed transition in burst/reconnect soak tests.
- V2 activity requires no multi-source synthesis.
- Rollback requires flags only, not data repair.

### Activity PR 15: Legacy Activity Retirement

Size: S
Depends on: Activity PR 14 plus the documented compatibility observation window

Goal: Remove redundant live paths only after v2 adoption is proven.

Scope:

1. Remove v1 synthesis/polling endpoints or retain a clearly deprecated compatibility endpoint according to release policy.
2. Remove frontend broad-refresh fallback only if supported backend versions no longer require it.
3. Remove legacy stage dual-write after telemetry confirms no consumers.
4. Delete rollout flags whose rollback window has expired.

Acceptance criteria:

- Supported-version matrix no longer needs removed paths.
- Request/query targets remain satisfied after cleanup.
- Historical execution events and progress records remain readable.

## Migration Coordination

Activity, Fidelity, and Calibration work may add migrations in parallel. Migration numbers are not reserved in planning documents. Each migration-bearing PR must rebase and take the next available number immediately before merge; CI should reject duplicate or out-of-order filenames.

## Definition Of Done

This track is complete when:

1. Classification and YOLO show useful progress before epoch one.
2. Current state comes from a bounded attempt-aware snapshot.
3. Transitions come from a cursor-safe, producer-complete event log.
4. Snapshot/event writes are atomic where they represent one transition.
5. Heartbeats do not bloat the event log.
6. Connected clients do not broadly poll or refresh on every event.
7. Measured request volume meets the reduction target.
8. Legacy paths are removed only after a compatibility release.

## Risks

1. Snapshot and event state can diverge.
   - Mitigation: transactional writes and snapshot/cursor race tests.

2. Worker death can leave stale progress.
   - Mitigation: server-owned lifecycle/recovery always supersedes worker observations.

3. Incremental UI state can become stale.
   - Mitigation: typed invalidation, cursor recovery, fallback polling, and manual refresh.

4. Taxonomy can become framework-specific.
   - Mitigation: coarse versioned stages plus bounded detail codes.

5. Parallel worker PRs can conflict with Fidelity changes.
   - Mitigation: serialize implementation/rebase in the explicit order listed above.

## Non-Goals

- Redesigning all Mission Control tabs.
- Streaming raw logs, prompts, chain-of-thought, or framework output.
- Providing precise ETA without a calibrated estimator.
- Replacing durable execution events with ephemeral pub/sub only.
