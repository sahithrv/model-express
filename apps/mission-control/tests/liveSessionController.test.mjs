import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadSession() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  return viteServer.ssrLoadModule("/src/features/live/liveSessionController.ts");
}

after(async () => viteServer?.close());

function snapshot(projectId, cursor, overrides = {}) {
  return {
    schema_version: "project_live_state.v1",
    project_id: projectId,
    operational_state: "active",
    taxonomy_version: 1,
    current_stage: "training",
    next_expected_stage: "evaluating",
    mixed_jobs: false,
    stale: false,
    jobs: { total: 1, queued: 0, retrying: 0, assigned: 0, running: 1, succeeded: 0, failed: 0, cancelled: 0 },
    workers: { total: 1, idle: 0, running: 1, offline: 0, stale: 0 },
    worker_requirements: { pending: 0, starting: 0, active: 1, satisfied: 0, failed: 0, cancelled: 0 },
    active_progress: [{
      job_id: `job-${projectId}`,
      attempt: 1,
      taxonomy_version: 1,
      stage: "training",
      status: "running",
      current: 1,
      total: 4,
      unit: "epochs",
      revision: 1,
      heartbeat_at: "2026-07-12T12:00:00.000Z",
      updated_at: "2026-07-12T12:00:00.000Z",
      elapsed_started_at: "2026-07-12T11:55:00.000Z",
      stale: false,
      metadata: {},
    }],
    active_progress_total: 1,
    active_progress_truncated: false,
    last_heartbeat_at: "2026-07-12T12:00:00.000Z",
    snapshot_cursor: cursor,
    snapshot_revision: `live-${cursor}`,
    ...overrides,
  };
}

function event(projectId, sequence, overrides = {}) {
  return {
    schema_version: "execution_event.v2",
    sequence,
    event_id: `event-${sequence}`,
    idempotency_key: `ik_${String(sequence).padStart(64, "0")}`,
    project_id: projectId,
    event_type: "JOB_PROGRESS_BOUNDARY",
    message: "Job progress advanced.",
    created_at: "2026-07-12T12:00:01.000Z",
    metadata: {
      category: "job",
      phase: "progress",
      status: "running",
      job_id: `job-${projectId}`,
      attempt: 1,
      taxonomy_version: 1,
      stage: "training",
      revision: sequence,
      current: 2,
      total: 4,
      unit: "epochs",
    },
    ...overrides,
  };
}

function fakeTimers() {
  const timers = [];
  return {
    timers,
    setTimer(callback, delayMs) {
      const timer = { callback, delayMs, cleared: false };
      timers.push(timer);
      return timer;
    },
    clearTimer(timer) {
      timer.cleared = true;
    },
    runNext() {
      const timer = timers.find((candidate) => !candidate.cleared);
      assert.ok(timer, "expected a pending timer");
      timer.cleared = true;
      timer.callback();
      return timer;
    },
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

function harness(IncrementalLiveSessionController, options = {}) {
  const timers = fakeTimers();
  const states = [];
  const streams = [];
  const events = [];
  const snapshots = options.snapshots ?? [snapshot("project-a", 10)];
  let snapshotIndex = 0;
  const probes = [];
  const controller = new IncrementalLiveSessionController({
    fetchSnapshot: options.fetchSnapshot ?? (async () => snapshots[Math.min(snapshotIndex++, snapshots.length - 1)]),
    probeStream: options.probeStream ?? (async (request) => { probes.push(request); }),
    openStream: (request) => {
      const stream = { ...request, closed: false };
      streams.push(stream);
      return { close: () => { stream.closed = true; } };
    },
    onState: (state) => states.push(state),
    onAppliedEvent: (applied) => events.push(applied),
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
    reconnectBaseDelayMs: 10,
    compactSnapshotIntervalMs: 1_000,
    maxReconnectAttempts: 3,
  });
  return { controller, timers, states, streams, events, probes };
}

test("snapshot cursor is installed atomically and later global cursor jumps apply once", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController);
  h.controller.start("project-a", "primary");
  await flush();
  assert.equal(h.streams.length, 1);
  assert.equal(h.streams[0].cursor, 10);
  h.streams[0].callbacks.onOpen();
  h.streams[0].callbacks.onEvent({
    eventType: "execution_event_v2",
    lastEventId: "13",
    data: JSON.stringify(event("project-a", 13)),
  });
  assert.equal(h.controller.current().live.cursor, 13, "other-project cursor gaps are legal");
  assert.equal(h.events.length, 1);
  h.streams[0].callbacks.onEvent({
    eventType: "execution_event_v2",
    lastEventId: "13",
    data: JSON.stringify(event("project-a", 13)),
  });
  assert.equal(h.events.length, 1);
  assert.equal(h.controller.current().live.cursor, 13);
});

test("reconnect probes and reopens strictly after the last successfully applied cursor", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController);
  h.controller.start("project-a", "primary");
  await flush();
  h.streams[0].callbacks.onOpen();
  h.streams[0].callbacks.onEvent({ eventType: "execution_event_v2", lastEventId: "12", data: JSON.stringify(event("project-a", 12)) });
  h.streams[0].callbacks.onDisconnect("network_error");
  assert.equal(h.controller.current().healthy, false);
  h.timers.runNext();
  await flush();
  assert.equal(h.probes.at(-1).cursor, 12);
  assert.equal(h.streams.at(-1).cursor, 12);
  assert.equal(h.streams[0].closed, true);
});

test("expired cursor recovery fetches a fresh snapshot before reopening", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  let probeCount = 0;
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [snapshot("project-a", 5), snapshot("project-a", 20)],
    probeStream: async (request) => {
      probeCount += 1;
      h.probes.push(request);
      if (probeCount === 2) throw { status: 410 };
    },
  });
  h.controller.start("project-a", "primary");
  await flush();
  h.streams[0].callbacks.onOpen();
  h.streams[0].callbacks.onDisconnect("network_error");
  h.timers.runNext();
  await flush();
  assert.equal(h.controller.current().live.cursor, 20);
  assert.equal(h.streams.at(-1).cursor, 20);
  assert.equal(probeCount, 3);
});

test("repeated cursor errors from the opened stream are bounded by fallback", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [
      snapshot("project-a", 10),
      snapshot("project-a", 20),
      snapshot("project-a", 30),
    ],
  });
  h.controller.start("project-a", "primary");
  await flush();

  for (let attempt = 0; attempt < 3; attempt += 1) {
    const stream = h.streams.at(-1);
    stream.callbacks.onError({ status: 410, reasonCode: "cursor_too_old" });
    await flush();
  }

  assert.equal(h.streams.length, 3, "only two cursor-recovery streams are opened");
  assert.equal(h.controller.current().connection, "fallback");
  assert.equal(h.controller.current().fallback_reason, "cursor_recovery_failed");
  assert.equal(h.controller.current().live.cursor, 30);
});

test("404, 405, and 501 snapshots select unsupported fallback", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  for (const status of [404, 405, 501]) {
    const h = harness(IncrementalLiveSessionController, {
      fetchSnapshot: async () => { throw { status }; },
    });
    h.controller.start("project-a", "primary");
    await flush();
    assert.equal(h.controller.current().connection, "unsupported");
    assert.equal(h.controller.current().supported, false);
    assert.equal(h.controller.current().fallback_reason, "snapshot_unsupported");
  }
});

test("404, 405, and 501 stream probes select unsupported fallback", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  for (const status of [404, 405, 501]) {
    const h = harness(IncrementalLiveSessionController, {
      probeStream: async () => { throw { status }; },
    });
    h.controller.start("project-a", "primary");
    await flush();
    assert.equal(h.controller.current().connection, "unsupported");
    assert.equal(h.controller.current().supported, false);
    assert.equal(h.controller.current().fallback_reason, "stream_unsupported");
    assert.equal(h.streams.length, 0);
  }
});

test("project switch aborts the old snapshot and ignores its late response", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const first = deferred();
  const second = deferred();
  const signals = [];
  let call = 0;
  const h = harness(IncrementalLiveSessionController, {
    fetchSnapshot: ({ signal }) => {
      signals.push(signal);
      call += 1;
      return call === 1 ? first.promise : second.promise;
    },
  });
  h.controller.start("project-a", "primary");
  h.controller.start("project-b", "primary");
  assert.equal(signals[0].aborted, true);
  first.resolve(snapshot("project-a", 99));
  second.resolve(snapshot("project-b", 7));
  await flush();
  assert.equal(h.controller.current().live.project_id, "project-b");
  assert.equal(h.controller.current().live.cursor, 7);
  assert.equal(h.streams.length, 1);
  assert.equal(h.streams[0].projectId, "project-b");
});

test("project switch closes the old stream and ignores its late event", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [snapshot("project-a", 10), snapshot("project-b", 20)],
  });
  h.controller.start("project-a", "primary");
  await flush();
  const oldStream = h.streams[0];
  oldStream.callbacks.onOpen();

  h.controller.start("project-b", "primary");
  await flush();
  assert.equal(oldStream.closed, true);
  oldStream.callbacks.onEvent({
    eventType: "execution_event_v2",
    lastEventId: "13",
    data: JSON.stringify(event("project-a", 13)),
  });
  assert.equal(h.controller.current().live.project_id, "project-b");
  assert.equal(h.controller.current().live.cursor, 20);
  assert.equal(h.events.length, 0);
});

test("out-of-order or malformed events cannot regress the cursor and trigger resnapshot recovery", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [snapshot("project-a", 10), snapshot("project-a", 15), snapshot("project-a", 16)],
  });
  h.controller.start("project-a", "primary");
  await flush();
  h.streams[0].callbacks.onOpen();
  h.streams[0].callbacks.onEvent({ eventType: "execution_event_v2", lastEventId: "14", data: JSON.stringify(event("project-a", 14)) });
  h.streams[0].callbacks.onEvent({ eventType: "execution_event_v2", lastEventId: "12", data: JSON.stringify(event("project-a", 12)) });
  assert.equal(h.controller.current().live.cursor, 14);
  await flush();
  assert.equal(h.controller.current().live.cursor, 15);
  const latest = h.streams.at(-1);
  latest.callbacks.onOpen();
  latest.callbacks.onEvent({ eventType: "execution_event_v2", lastEventId: "16", data: "{not json" });
  await flush();
  assert.equal(h.controller.current().live.cursor, 16);
});

test("repeated malformed events are bounded and select conservative fallback", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [snapshot("project-a", 10), snapshot("project-a", 10), snapshot("project-a", 10)],
  });
  h.controller.start("project-a", "primary");
  await flush();

  for (let attempt = 0; attempt < 3; attempt += 1) {
    const stream = h.streams.at(-1);
    stream.callbacks.onOpen();
    stream.callbacks.onEvent({ eventType: "execution_event_v2", lastEventId: "11", data: "{not json" });
    await flush();
  }

  assert.equal(h.streams.length, 3, "only two replacement streams are opened");
  assert.equal(h.controller.current().connection, "fallback");
  assert.equal(h.controller.current().fallback_reason, "cursor_recovery_failed");
  assert.equal(h.controller.current().live.cursor, 10);
});

test("compact snapshot-only refresh preserves a healthy stream when the cursor is unchanged", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [
      snapshot("project-a", 10),
      snapshot("project-a", 10, { operational_state: "stale", stale: true }),
    ],
  });
  h.controller.start("project-a", "primary");
  await flush();
  h.streams[0].callbacks.onOpen();
  h.timers.runNext();
  await flush();

  assert.equal(h.controller.current().live.snapshot.operational_state, "stale");
  assert.equal(h.controller.current().healthy, true);
  assert.equal(h.streams.length, 1);
  assert.equal(h.probes.length, 1);
  assert.equal(h.streams[0].closed, false);
});

test("manual resync works while connected and rollback never opens v2", async () => {
  const { IncrementalLiveSessionController } = await loadSession();
  const h = harness(IncrementalLiveSessionController, {
    snapshots: [snapshot("project-a", 3), snapshot("project-a", 4)],
  });
  h.controller.start("project-a", "primary");
  await flush();
  h.streams[0].callbacks.onOpen();
  await h.controller.manualResync();
  assert.equal(h.controller.current().live.cursor, 4);
  assert.equal(h.streams.at(-1).cursor, 4);

  const rollback = harness(IncrementalLiveSessionController);
  rollback.controller.start("project-a", "rollback");
  await flush();
  assert.equal(rollback.controller.current().connection, "fallback");
  assert.equal(rollback.streams.length, 0);
});
