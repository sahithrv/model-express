import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadLiveProgress() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  const [panel, viewModel] = await Promise.all([
    viteServer.ssrLoadModule("/src/features/live/LiveProgressPanel.tsx"),
    viteServer.ssrLoadModule("/src/features/live/liveProgressViewModel.ts"),
  ]);
  return { ...panel, ...viewModel };
}

after(async () => {
  await viteServer?.close();
});

const fixedNowMs = Date.parse("2026-07-12T12:30:45.000Z");

function liveState(overrides = {}) {
  const base = {
    schema_version: "project_live_state.v1",
    project_id: "project-safe",
    operational_state: "active",
    taxonomy_version: 1,
    current_stage: "training",
    next_expected_stage: "evaluating",
    mixed_jobs: false,
    stale: false,
    blocked_reason_code: "",
    jobs: {
      total: 1,
      queued: 0,
      retrying: 0,
      assigned: 0,
      running: 1,
      succeeded: 0,
      failed: 0,
      cancelled: 0,
    },
    workers: { total: 1, idle: 0, running: 1, offline: 0, stale: 0 },
    worker_requirements: { pending: 0, starting: 0, active: 1, satisfied: 0, failed: 0, cancelled: 0 },
    active_progress: [activeProgress()],
    active_progress_total: 1,
    active_progress_truncated: false,
    last_heartbeat_at: "2026-07-12T12:30:15.000Z",
    latest_important_event: null,
    snapshot_cursor: 42,
    snapshot_revision: "live-safe",
  };
  return { ...base, ...overrides };
}

function activeProgress(overrides = {}) {
  return {
    job_id: "job-safe",
    attempt: 1,
    taxonomy_version: 1,
    stage: "training",
    detail_code: "epoch",
    status: "running",
    current: 3,
    total: 10,
    unit: "epochs",
    message: "",
    revision: 3,
    heartbeat_at: "2026-07-12T12:30:15.000Z",
    updated_at: "2026-07-12T12:30:15.000Z",
    elapsed_started_at: "2026-07-12T11:28:30.000Z",
    stale: false,
    metadata: {},
    ...overrides,
  };
}

function renderPanel(LiveProgressPanel, props = {}) {
  return renderToStaticMarkup(React.createElement(LiveProgressPanel, {
    snapshot: liveState(),
    connectionState: "connected",
    mode: "primary",
    nowMs: fixedNowMs,
    ...props,
  }));
}

test("active training presents stable stage, epoch, true elapsed, update age, and next stage without an ETA", async () => {
  const { LiveProgressPanel, buildLiveProgressViewModel } = await loadLiveProgress();
  const snapshot = liveState();
  const view = buildLiveProgressViewModel({ snapshot, connectionState: "connected", mode: "primary", nowMs: fixedNowMs });

  assert.equal(view.state, "active");
  assert.equal(view.stageLabel, "Training");
  assert.equal(view.epochLabel, "Epoch 3/10");
  assert.equal(view.elapsedLabel, "1h 2m");
  assert.equal(view.lastUpdateLabel, "30s ago");
  assert.equal(view.nextStageLabel, "Evaluating");
  assert.equal(view.taxonomyLabel, "Taxonomy v1");

  const markup = renderPanel(LiveProgressPanel, { snapshot });
  assert.match(markup, /role="status"/);
  assert.match(markup, /aria-live="polite"/);
  assert.match(markup, /data-live-progress-state="active"/);
  assert.match(markup, />Epoch 3\/10</);
  assert.match(markup, />1h 2m</);
  assert.match(markup, /dateTime="2026-07-12T12:30:15.000Z"/);
  assert.match(markup, />Evaluating</);
  assert.doesNotMatch(markup, /\bETA\b|estimated time|time remaining/i);
});

test("worker finalizing remains active until backend operational state becomes terminal", async () => {
  const { LiveProgressPanel, buildLiveProgressViewModel } = await loadLiveProgress();
  const snapshot = liveState({
    operational_state: "active",
    current_stage: "finalizing",
    next_expected_stage: "completed",
    active_progress: [activeProgress({ stage: "finalizing", detail_code: "artifacts", current: null, total: null, unit: "" })],
  });
  const view = buildLiveProgressViewModel({ snapshot, connectionState: "connected", mode: "primary", nowMs: fixedNowMs });

  assert.equal(view.state, "active");
  assert.equal(view.stateLabel, "Active");
  assert.equal(view.stageLabel, "Finalizing");
  assert.equal(view.nextStageLabel, "Completed");
  assert.equal(view.terminalStage, "");

  const markup = renderPanel(LiveProgressPanel, { snapshot });
  assert.match(markup, /data-live-progress-state="active"/);
  assert.match(markup, /state-active">Active</);
  assert.doesNotMatch(markup, /terminal-completed/);
});

test("backend terminal stage is authoritative for completed, failed, and cancelled work", async () => {
  const { buildLiveProgressViewModel } = await loadLiveProgress();
  for (const terminalStage of ["completed", "failed", "cancelled"]) {
    const snapshot = liveState({
      operational_state: "terminal",
      current_stage: terminalStage,
      next_expected_stage: "",
      active_progress: [],
      jobs: {
        total: 1,
        queued: 0,
        retrying: 0,
        assigned: 0,
        running: 0,
        succeeded: terminalStage === "completed" ? 1 : 0,
        failed: terminalStage === "failed" ? 1 : 0,
        cancelled: terminalStage === "cancelled" ? 1 : 0,
      },
    });
    const view = buildLiveProgressViewModel({ snapshot, connectionState: "connected", mode: "primary", nowMs: fixedNowMs });
    assert.equal(view.state, "terminal");
    assert.equal(view.terminalStage, terminalStage);
    assert.equal(view.stateLabel.toLowerCase(), terminalStage);
    assert.equal(view.nextStageLabel, "");
  }
});

test("queued, retrying, blocked, stale, and mixed states retain server semantics", async () => {
  const { buildLiveProgressViewModel } = await loadLiveProgress();
  const cases = [
    { operational_state: "queued", expected: "Queued" },
    { operational_state: "retrying", expected: "Retrying" },
    {
      operational_state: "blocked",
      blocked_reason_code: "worker_requirement_failed",
      expected: "Blocked",
      reason: "The required worker failed to start.",
    },
    {
      operational_state: "stale",
      stale: true,
      expected: "Stale",
      reason: "The server marked the active heartbeat as stale.",
    },
  ];

  for (const candidate of cases) {
    const view = buildLiveProgressViewModel({
      snapshot: liveState(candidate),
      connectionState: "connected",
      mode: "primary",
      nowMs: fixedNowMs,
    });
    assert.equal(view.stateLabel, candidate.expected);
    if (candidate.reason) assert.equal(view.reasonLabel, candidate.reason);
  }

  const mixed = buildLiveProgressViewModel({
    snapshot: liveState({
      mixed_jobs: true,
      jobs: { total: 3, queued: 1, retrying: 0, assigned: 0, running: 1, succeeded: 1, failed: 0, cancelled: 0 },
    }),
    connectionState: "connected",
    mode: "primary",
    nowMs: fixedNowMs,
  });
  assert.equal(mixed.mixed, true);
  assert.equal(mixed.mixedLabel, "2 open, 1 terminal");
  assert.match(mixed.detail, /multiple lifecycle states/);
});

test("connection state exposes connected, reconnecting, and fallback badges", async () => {
  const { LiveProgressPanel } = await loadLiveProgress();
  const connected = renderPanel(LiveProgressPanel, { connectionState: "connected" });
  const reconnecting = renderPanel(LiveProgressPanel, { connectionState: "reconnecting" });
  const fallback = renderPanel(LiveProgressPanel, { snapshot: null, connectionState: "fallback", mode: "legacy" });

  assert.match(connected, /live-progress-connection-badge connected">Connected</);
  assert.match(reconnecting, /live-progress-connection-badge pending">Reconnecting</);
  assert.match(fallback, /live-progress-connection-badge fallback">Fallback polling</);
  assert.match(fallback, /Legacy polling/);
});

test("stale presentation has a distinct semantic class and visual treatment from failed work", async () => {
  const { LiveProgressPanel } = await loadLiveProgress();
  const stale = renderPanel(LiveProgressPanel, {
    snapshot: liveState({ operational_state: "stale", stale: true }),
  });
  const failed = renderPanel(LiveProgressPanel, {
    snapshot: liveState({ operational_state: "terminal", current_stage: "failed", active_progress: [] }),
  });

  assert.match(stale, /live-progress-panel state-stale/);
  assert.match(stale, /class="stale-reason"/);
  assert.doesNotMatch(stale, /terminal-failed/);
  assert.match(failed, /terminal-failed/);

  const styles = await readFile(path.join(appRoot, "src/styles.css"), "utf8");
  assert.match(styles, /\.live-progress-panel\.state-stale[\s\S]*?repeating-linear-gradient/);
  assert.match(styles, /\.live-progress-panel\.state-blocked,[\s\S]*?\.live-progress-panel\.terminal-failed/);
});
