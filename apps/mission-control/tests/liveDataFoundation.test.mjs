import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadFoundation() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  const [contract, reducer, invalidations, polling, refreshCoordinator, projectDetail] = await Promise.all([
    viteServer.ssrLoadModule("/src/features/live/liveStateContract.ts"),
    viteServer.ssrLoadModule("/src/features/live/liveStateReducer.ts"),
    viteServer.ssrLoadModule("/src/features/live/liveInvalidations.ts"),
    viteServer.ssrLoadModule("/src/features/live/livePollingPolicy.ts"),
    viteServer.ssrLoadModule("/src/features/live/liveRefreshCoordinator.ts"),
    viteServer.ssrLoadModule("/src/hooks/useProjectDetail.ts"),
  ]);
  return { ...contract, ...reducer, ...invalidations, ...polling, ...refreshCoordinator, ...projectDetail };
}

after(async () => viteServer?.close());

function progress(overrides = {}) {
  return {
    job_id: "job-a",
    attempt: 1,
    taxonomy_version: 1,
    stage: "training",
    detail_code: "epoch",
    status: "running",
    current: 1,
    total: 4,
    unit: "epochs",
    message: "Training",
    revision: 1,
    heartbeat_at: "2026-07-12T12:00:00.000Z",
    updated_at: "2026-07-12T12:00:00.000Z",
    elapsed_started_at: "2026-07-12T11:55:00.000Z",
    stale: false,
    metadata: { framework: "pytorch", raw_output: "must not survive" },
    ...overrides,
  };
}

function event(sequence, overrides = {}) {
  return {
    schema_version: "execution_event.v2",
    sequence,
    event_id: `event-${sequence}`,
    idempotency_key: `progress-attempt-1-${sequence}`,
    project_id: "project-a",
    event_type: "JOB_PROGRESS_BOUNDARY",
    message: "Training advanced.",
    created_at: `2026-07-12T12:00:${String(sequence % 60).padStart(2, "0")}.000Z`,
    metadata: {
      category: "job",
      phase: "progress",
      status: "running",
      job_id: "job-a",
      attempt: 1,
      taxonomy_version: 1,
      stage: "training",
      detail_code: "epoch",
      revision: sequence,
      current: Math.min(sequence, 4),
      total: Math.max(sequence, 4),
      unit: "epochs",
    },
    ...overrides,
  };
}

function snapshot(cursor = 10, overrides = {}) {
  return {
    schema_version: "project_live_state.v1",
    project_id: "project-a",
    operational_state: "active",
    taxonomy_version: 1,
    current_stage: "training",
    next_expected_stage: "evaluating",
    mixed_jobs: false,
    stale: false,
    jobs: { total: 1, queued: 0, retrying: 0, assigned: 0, running: 1, succeeded: 0, failed: 0, cancelled: 0 },
    workers: { total: 1, idle: 0, running: 1, offline: 0, stale: 0 },
    worker_requirements: { pending: 0, starting: 0, active: 1, satisfied: 0, failed: 0, cancelled: 0 },
    active_progress: [progress()],
    active_progress_total: 1,
    active_progress_truncated: false,
    last_heartbeat_at: "2026-07-12T12:00:00.000Z",
    snapshot_cursor: cursor,
    snapshot_revision: `revision-${cursor}`,
    elapsed_started_at: "2026-07-12T11:55:00.000Z",
    ...overrides,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test("runtime contracts enforce schema, taxonomy, project identity, and bounded allowlists", async () => {
  const { parseExecutionEventV2, parseProjectLiveState } = await loadFoundation();
  const parsedEvent = parseExecutionEventV2(event(7, {
    message: "Read s3://secret-bucket/model.bin with sk-thisisasecrettokenvalue",
    metadata: {
      reason: "See /Users/person/private/data.csv",
      current: 2,
      prompt: "private prompt",
      raw_config: { token: "private" },
      evaluation: "private evaluation",
    },
  }), "project-a");
  assert.equal(parsedEvent.sequence, 7);
  assert.equal(parsedEvent.idempotency_key, "progress-attempt-1-7");
  assert.match(parsedEvent.message, /\[redacted_uri\].*\[redacted_secret\]/);
  assert.deepEqual(Object.keys(parsedEvent.metadata).sort(), ["current", "reason"]);
  assert.match(parsedEvent.metadata.reason, /\[redacted_path\]/);

  const parsedSnapshot = parseProjectLiveState(snapshot(), "project-a");
  assert.equal(parsedSnapshot.current_stage, "training");
  assert.equal(parsedSnapshot.active_progress[0].elapsed_started_at, "2026-07-12T11:55:00.000Z");
  assert.deepEqual(parsedSnapshot.active_progress[0].metadata, { framework: "pytorch" });

  const legacyCurrentOnly = parseProjectLiveState(snapshot(11, {
    active_progress: [progress({ current: 3, total: undefined, unit: "epoch" })],
  }), "project-a");
  assert.equal(legacyCurrentOnly.active_progress[0].current, 3);
  assert.equal(legacyCurrentOnly.active_progress[0].total, undefined);

  const serverTotalOnly = parseProjectLiveState(snapshot(12, {
    active_progress: [progress({ current: undefined, total: 8, unit: undefined })],
  }), "project-a");
  assert.equal(serverTotalOnly.active_progress[0].current, undefined);
  assert.equal(serverTotalOnly.active_progress[0].total, 8);

  assert.throws(
    () => parseProjectLiveState(snapshot(10, { schema_version: "project_live_state.v2" }), "project-a"),
    (error) => error.code === "unsupported_schema",
  );
  assert.throws(
    () => parseProjectLiveState(snapshot(10, { taxonomy_version: 2 }), "project-a"),
    (error) => error.code === "unsupported_taxonomy",
  );
  assert.throws(
    () => parseProjectLiveState(snapshot(), "another-project"),
    (error) => error.code === "project_mismatch",
  );
  assert.deepEqual(
    parseExecutionEventV2(event(8, { metadata: { current: Number.POSITIVE_INFINITY } })).metadata,
    {},
    "invalid allowlisted values are projected out rather than entering client state",
  );
});

test("snapshot and later global cursor events compose once without cursor regression", async () => {
  const {
    applyExecutionEvent,
    createIncrementalLiveState,
    installLiveStateSnapshot,
    parseExecutionEventV2,
    parseProjectLiveState,
  } = await loadFoundation();
  const racingEvent = parseExecutionEventV2(event(10), "project-a");
  let state = createIncrementalLiveState("project-a");
  state = installLiveStateSnapshot(state, parseProjectLiveState(snapshot(10, { latest_important_event: event(10) }), "project-a"));
  assert.equal(state.cursor, 10);

  state = applyExecutionEvent(state, racingEvent);
  assert.equal(state.last_transition, "duplicate_cursor");
  assert.equal(state.applied_event_count, 0, "the snapshot-covered transition is not represented twice");

  state = applyExecutionEvent(state, parseExecutionEventV2(event(13), "project-a"));
  assert.equal(state.cursor, 13, "global sequence jumps belonging to other projects are valid");
  assert.equal(state.applied_event_count, 1);

  const afterApplied = state.snapshot.latest_important_event;
  state = applyExecutionEvent(state, parseExecutionEventV2(event(14, {
    event_id: "a-different-event",
    idempotency_key: "progress-attempt-1-13",
  }), "project-a"));
  assert.equal(state.cursor, 14, "an idempotent replay can safely advance the observed cursor");
  assert.equal(state.last_transition, "duplicate_idempotency_key");
  assert.deepEqual(state.snapshot.latest_important_event, afterApplied);

  state = applyExecutionEvent(state, parseExecutionEventV2(event(13), "project-a"));
  assert.equal(state.cursor, 14, "an older known replay is idempotent and cannot regress the cursor");
  assert.equal(state.last_transition, "duplicate_event");

  state = applyExecutionEvent(state, parseExecutionEventV2(event(12), "project-a"));
  assert.equal(state.cursor, 14);
  assert.equal(state.last_transition, "out_of_order");

  const older = installLiveStateSnapshot(state, parseProjectLiveState(snapshot(11), "project-a"));
  assert.equal(older.cursor, 14);
  assert.equal(older.last_transition, "older_snapshot");
});

test("progress boundaries are attempt/revision monotonic and finalizing remains nonterminal", async () => {
  const {
    applyExecutionEvent,
    createIncrementalLiveState,
    installLiveStateSnapshot,
    parseExecutionEventV2,
    parseProjectLiveState,
  } = await loadFoundation();
  let state = installLiveStateSnapshot(
    createIncrementalLiveState("project-a"),
    parseProjectLiveState(snapshot(20, {
      operational_state: "active",
      active_progress: [progress({ revision: 20 })],
    }), "project-a"),
  );
  state = applyExecutionEvent(state, parseExecutionEventV2(event(21, {
    metadata: {
      job_id: "job-a",
      attempt: 1,
      taxonomy_version: 1,
      stage: "finalizing",
      status: "running",
      revision: 21,
    },
  }), "project-a"));
  assert.equal(state.snapshot.current_stage, "finalizing");
  assert.equal(state.snapshot.operational_state, "active", "worker finalizing is not backend completion");

  state = applyExecutionEvent(state, parseExecutionEventV2(event(22, {
    metadata: {
      job_id: "job-a",
      attempt: 1,
      taxonomy_version: 1,
      stage: "training",
      status: "running",
      revision: 19,
    },
  }), "project-a"));
  assert.equal(state.last_transition, "stale_progress");
  assert.equal(state.snapshot.current_stage, "finalizing");
  assert.equal(state.snapshot.active_progress[0].revision, 21);
});

test("event invalidations are exact and unknown events have no broad fallback", async () => {
  const { parseProjectLiveState, resourcesForExecutionEvent, resourcesForLiveStateReplacement } = await loadFoundation();
  assert.deepEqual(resourcesForExecutionEvent("EPOCH_COMPLETED"), ["metrics"]);
  assert.deepEqual(resourcesForExecutionEvent("JOB_PROGRESS_BOUNDARY"), ["metrics"]);
  assert.deepEqual(resourcesForExecutionEvent("JOB_COMPLETED"), [
    "live_state", "project_index", "jobs", "metrics", "training_results",
  ]);
  assert.deepEqual(resourcesForExecutionEvent("JOB_RETRY_QUEUED_TRANSITION"), ["live_state"]);
  assert.deepEqual(resourcesForExecutionEvent("JOB_LEASE_RECOVERED"), ["live_state"]);
  assert.deepEqual(resourcesForExecutionEvent("WORKERS_ACTIVE"), ["live_state"]);
  assert.deepEqual(resourcesForExecutionEvent("AGENT_DECISION_RECORDED"), ["decisions"]);
  assert.deepEqual(resourcesForExecutionEvent("CHAMPION_SELECTED"), ["champion", "training_results", "champion_exports"]);
  assert.deepEqual(resourcesForExecutionEvent("AGENT_STARTED"), []);
  assert.deepEqual(resourcesForExecutionEvent("A_NEW_UNKNOWN_EVENT"), []);

  const oldWorkerBefore = parseProjectLiveState(snapshot(20, {
    active_progress: [progress({ current: 1, total: undefined, revision: 20, unit: "epoch" })],
  }), "project-a");
  const oldWorkerAfter = parseProjectLiveState(snapshot(20, {
    active_progress: [progress({ current: 2, total: undefined, revision: 21, unit: "epoch" })],
  }), "project-a");
  assert.deepEqual(
    resourcesForLiveStateReplacement(oldWorkerBefore, oldWorkerAfter),
    ["metrics"],
    "compact snapshots retain metrics refreshes for old workers that do not emit epoch events",
  );
  assert.deepEqual(resourcesForLiveStateReplacement(oldWorkerAfter, oldWorkerAfter), []);
});

test("burst invalidations coalesce and in-flight work permits only one follow-up", async () => {
  const { createInvalidationCoalescer } = await loadFoundation();
  const calls = [];
  const first = deferred();
  const coalescer = createInvalidationCoalescer({
    execute: (resource) => {
      calls.push(resource);
      return calls.length === 1 ? first.promise : undefined;
    },
  });

  coalescer.invalidateEvent("EPOCH_COMPLETED");
  coalescer.invalidateEvent("EPOCH_PROGRESS");
  coalescer.invalidate(["metrics", "metrics"]);
  const idle = coalescer.flush();
  await Promise.resolve();
  assert.deepEqual(calls, ["metrics"]);

  coalescer.invalidateEvent("JOB_PROGRESS_BOUNDARY");
  coalescer.invalidateEvent("JOB_PROGRESS_BOUNDARY");
  coalescer.invalidateEvent("JOB_PROGRESS_BOUNDARY");
  assert.deepEqual(coalescer.snapshot().follow_up, ["metrics"]);
  first.resolve();
  await idle;
  assert.deepEqual(calls, ["metrics", "metrics"]);
  assert.deepEqual(coalescer.snapshot().in_flight, []);
});

test("disposing invalidations aborts in-flight requests and ignores later work", async () => {
  const { createInvalidationCoalescer } = await loadFoundation();
  let calls = 0;
  let observedSignal;
  const coalescer = createInvalidationCoalescer({
    execute: (_resource, { signal }) => {
      calls += 1;
      observedSignal = signal;
      return new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }));
    },
  });
  coalescer.invalidate(["jobs"]);
  const idle = coalescer.flush();
  await Promise.resolve();
  coalescer.dispose("project_changed");
  await idle;
  assert.equal(observedSignal.aborted, true);
  coalescer.invalidate(["jobs"]);
  await coalescer.flush();
  assert.equal(calls, 1);
});

test("failed invalidations receive one bounded retry and a later event gets a fresh allowance", async () => {
  const { createInvalidationCoalescer } = await loadFoundation();
  let calls = 0;
  const coalescer = createInvalidationCoalescer({
    maxRetries: 1,
    execute: async () => {
      calls += 1;
      throw new Error("transient targeted failure");
    },
  });

  coalescer.invalidate(["training_results"]);
  await coalescer.flush();
  assert.equal(calls, 2, "the first failed resource request receives one follow-up");

  coalescer.invalidate(["training_results"]);
  await coalescer.flush();
  assert.equal(calls, 4, "a later terminal event gets a fresh bounded retry allowance");
});

test("a late targeted response cannot commit after project cleanup", async () => {
  const { createInvalidationCoalescer } = await loadFoundation();
  const response = deferred();
  let committed = false;
  let observedSignal;
  const coalescer = createInvalidationCoalescer({
    execute: async (_resource, { signal }) => {
      observedSignal = signal;
      await response.promise;
      if (!signal.aborted) committed = true;
    },
  });
  coalescer.invalidate(["jobs"]);
  const idle = coalescer.flush();
  await Promise.resolve();
  coalescer.dispose("project_changed");
  response.resolve();
  await idle;
  assert.equal(observedSignal.aborted, true);
  assert.equal(committed, false);
});

test("empty project detail retains explicit ownership for later targeted commits", async () => {
  const { emptyProjectDetail } = await loadFoundation();
  const detail = emptyProjectDetail("project-a");
  assert.equal(detail.project_id, "project-a");
  assert.deepEqual(detail.jobs, []);
  assert.deepEqual(detail.datasets, []);
  assert.deepEqual(detail.plans, []);
});

test("a broad refresh preempts hanging targeted work without overlap and retains one post-broad target", async () => {
  const { createLiveRefreshCoordinator, projectLiveRefreshScope } = await loadFoundation();
  const coordinator = createLiveRefreshCoordinator();
  const scope = projectLiveRefreshScope("project-a");
  const broadGate = deferred();
  const order = [];
  let activeTargeted = 0;

  const callerController = new AbortController();
  const firstTarget = coordinator.runTargeted(scope, callerController.signal, async (signal) => {
    assert.notEqual(signal, callerController.signal);
    activeTargeted += 1;
    order.push("target-1-start");
    try {
      await new Promise((_, reject) => signal.addEventListener("abort", () => {
        order.push("target-1-abort");
        reject(signal.reason);
      }, { once: true }));
    } finally {
      activeTargeted -= 1;
    }
  });
  const firstTargetOutcome = firstTarget.catch((error) => error?.name);
  await Promise.resolve();
  const broad = coordinator.runBroad(scope, async () => {
    assert.equal(activeTargeted, 0);
    order.push("broad-start");
    await broadGate.promise;
    order.push("broad-end");
  });
  const secondTarget = coordinator.runTargeted(scope, new AbortController().signal, async (signal) => {
    assert.equal(signal.aborted, false);
    activeTargeted += 1;
    order.push("target-2-start");
    order.push("target-2-end");
    activeTargeted -= 1;
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(await firstTargetOutcome, "AbortError");
  assert.deepEqual(order, ["target-1-start", "target-1-abort", "broad-start"]);
  assert.deepEqual(coordinator.snapshot(scope), { broad_active: true, targeted_active: 0 });

  broadGate.resolve();
  await Promise.all([broad, secondTarget]);
  assert.deepEqual(order, [
    "target-1-start",
    "target-1-abort",
    "broad-start",
    "broad-end",
    "target-2-start",
    "target-2-end",
  ]);
  assert.deepEqual(coordinator.snapshot(scope), { broad_active: false, targeted_active: 0 });
});

test("a targeted read waiting behind a broad refresh remains abortable", async () => {
  const { createLiveRefreshCoordinator } = await loadFoundation();
  const coordinator = createLiveRefreshCoordinator();
  const broadGate = deferred();
  const broad = coordinator.runBroad("project:project-a", () => broadGate.promise);
  const controller = new AbortController();
  const targeted = coordinator.runTargeted("project:project-a", controller.signal, async () => undefined);
  controller.abort();
  await assert.rejects(targeted, (error) => error?.name === "AbortError");
  broadGate.resolve();
  await broad;
});

function health(overrides = {}) {
  return {
    endpoint_support: "supported",
    snapshot_ready: true,
    stream_status: "connected",
    cursor_status: "consistent",
    ...overrides,
  };
}

function flags(overrides = {}) {
  return {
    incremental_v2_enabled: true,
    shadow_mode: false,
    rollback_to_legacy: false,
    ...overrides,
  };
}

test("polling cutover, shadow, fallback, unsupported, and rollback policies are deterministic", async () => {
  const { isUnsupportedLiveEndpointStatus, resolveLivePollingPolicy } = await loadFoundation();
  const healthy = resolveLivePollingPolicy({ flags: flags(), health: health(), has_open_work: false });
  assert.equal(healthy.mode, "incremental");
  assert.equal(healthy.run_legacy_polling, false);
  assert.equal(healthy.broad_refresh_interval_ms, null);
  assert.equal(healthy.manual_refresh_enabled, true);
  assert.equal(healthy.detail_tabs_on_demand, true);

  const shadow = resolveLivePollingPolicy({ flags: flags({ shadow_mode: true }), health: health(), has_open_work: true });
  assert.equal(shadow.mode, "shadow");
  assert.equal(shadow.start_incremental_layer, true);
  assert.equal(shadow.use_incremental_presentation, false);
  assert.equal(shadow.broad_refresh_interval_ms, 10_000);

  const disconnected = resolveLivePollingPolicy({
    flags: flags(),
    health: health({ stream_status: "disconnected" }),
    has_open_work: false,
  });
  assert.equal(disconnected.mode, "fallback");
  assert.equal(disconnected.reason_code, "stream_disconnected");
  assert.equal(disconnected.broad_refresh_interval_ms, 30_000);
  assert.equal(disconnected.manual_refresh_enabled, true);
  assert.equal(disconnected.detail_tabs_on_demand, true);

  const cursorFailed = resolveLivePollingPolicy({
    flags: flags(),
    health: health({ cursor_status: "failed" }),
    has_open_work: true,
  });
  assert.equal(cursorFailed.run_legacy_polling, true);
  assert.equal(cursorFailed.reason_code, "cursor_recovery_failed");

  const rollback = resolveLivePollingPolicy({
    flags: flags({ rollback_to_legacy: true }),
    health: health(),
    has_open_work: true,
  });
  assert.equal(rollback.mode, "rollback");
  assert.equal(rollback.start_incremental_layer, false);
  assert.equal(rollback.run_legacy_polling, true);
  assert.equal(rollback.manual_refresh_enabled, true);
  assert.equal(rollback.detail_tabs_on_demand, true);

  for (const status of [404, 405, 501]) assert.equal(isUnsupportedLiveEndpointStatus(status), true);
  assert.equal(isUnsupportedLiveEndpointStatus(410), false);
  const unsupported = resolveLivePollingPolicy({
    flags: flags(),
    health: health({ endpoint_support: "unsupported" }),
    has_open_work: true,
  });
  assert.equal(unsupported.reason_code, "endpoint_unsupported");
  assert.equal(unsupported.start_incremental_layer, false);
});

test("request-count harness proves zero idle broad refresh and material active reduction", async () => {
  const {
    requestReductionPercent,
    resolveLivePollingPolicy,
    simulateLegacyPollingVolume,
    simulateLiveRequestVolume,
  } = await loadFoundation();
  assert.deepEqual(simulateLegacyPollingVolume(60_000, true), {
    polling_ticks: 6,
    broad_requests: 63,
    uncached_requests: 60,
    cached_requests: 3,
  });
  assert.deepEqual(simulateLegacyPollingVolume(60_000, false), {
    polling_ticks: 2,
    broad_requests: 22,
    uncached_requests: 20,
    cached_requests: 2,
  });

  const policy = resolveLivePollingPolicy({ flags: flags(), health: health(), has_open_work: true });
  const oneBurst = simulateLiveRequestVolume({
    duration_ms: 60_000,
    has_open_work: true,
    policy,
    event_bursts: [{ at_ms: 10_000, event_types: ["EPOCH_COMPLETED", "EPOCH_PROGRESS", "JOB_PROGRESS_BOUNDARY"] }],
  });
  assert.equal(oneBurst.broad_requests, 0);
  assert.equal(oneBurst.targeted_requests, 1);
  assert.deepEqual(oneBurst.targeted_by_resource, { metrics: 1 });

  const sixEpochs = simulateLiveRequestVolume({
    duration_ms: 60_000,
    has_open_work: true,
    policy,
    event_bursts: [10_000, 20_000, 30_000, 40_000, 50_000, 60_000]
      .map((at_ms) => ({ at_ms, event_types: ["EPOCH_COMPLETED"] })),
  });
  assert.equal(sixEpochs.broad_requests, 0);
  assert.equal(sixEpochs.targeted_requests, 6);
  assert.equal(sixEpochs.compact_snapshot_requests, 2);
  assert.equal(sixEpochs.stream_probes, 1);
  assert.equal(
    sixEpochs.total_requests,
    11,
    "initial snapshot/probe/stream, two compact state refreshes, and six targeted metric requests",
  );
  assert.ok(requestReductionPercent(63, sixEpochs.total_requests) >= 80);

  const idle = simulateLiveRequestVolume({ duration_ms: 60_000, has_open_work: false, policy });
  assert.equal(idle.broad_refreshes, 0);
  assert.equal(idle.broad_requests, 0);
  assert.equal(idle.compact_snapshot_requests, 0);
  assert.equal(idle.total_requests, 3, "idle performs one snapshot, one cursor probe, and one stream connection");

  const withManualAndDetail = simulateLiveRequestVolume({
    duration_ms: 60_000,
    has_open_work: false,
    policy,
    manual_refresh_requests: 1,
    manual_refresh_request_cost: 5,
    detail_tab_requests: 2,
  });
  assert.equal(withManualAndDetail.manual_refreshes, 1);
  assert.equal(withManualAndDetail.manual_requests, 5);
  assert.equal(withManualAndDetail.detail_tab_requests, 2);
});
