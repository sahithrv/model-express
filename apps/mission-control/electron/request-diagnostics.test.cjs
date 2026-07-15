const assert = require("node:assert/strict");
const test = require("node:test");

const {
  classifyEndpointCategory,
  createRollingRequestDiagnostics,
  runActivityRolloutReport,
  runBaselineRequestScenario,
} = require("./request-diagnostics.cjs");

test("request diagnostics classify endpoint shapes without retaining resource ids", () => {
  assert.equal(classifyEndpointCategory("/healthz"), "health");
  assert.equal(classifyEndpointCategory("/projects"), "project_index");
  assert.equal(classifyEndpointCategory("/projects/private-project/jobs?limit=100"), "project_live");
  assert.equal(classifyEndpointCategory("/projects/private-project/agent-invocations?limit=8"), "project_history");
  assert.equal(classifyEndpointCategory("/jobs/private-job/metrics?limit=200"), "job_metrics");
  assert.equal(classifyEndpointCategory("/projects/private-project/live-state"), "project_live_state");
  assert.equal(classifyEndpointCategory("/projects/private-project/events/stream/v2?cursor=7"), "execution_event_stream_v2");
  assert.equal(classifyEndpointCategory("/datasets/private-dataset/metadata/summary"), "dataset_detail");
});

test("v2 reasons stay bounded and targeted traffic is never classified as broad", () => {
  let nowMs = 1_000;
  const diagnostics = createRollingRequestDiagnostics({ now: () => nowMs });
  const snapshot = diagnostics.record({
    method: "GET",
    path: "/projects/private/live-state",
    reasonCode: "v2_snapshot",
    statusCode: 200,
  });
  nowMs += 1;
  const targeted = diagnostics.record({
    method: "GET",
    path: "/jobs/private/metrics",
    reasonCode: "targeted_invalidation",
    statusCode: 200,
  });
  assert.equal(snapshot.reason_code, "v2_snapshot");
  assert.equal(targeted.rolling_broad_get_count, 0);
  assert.deepEqual(targeted.reason_code_counts, { targeted_invalidation: 1, v2_snapshot: 1 });
});

test("rolling request diagnostics use a fake clock and bounded second buckets", () => {
  let nowMs = 1_000;
  const diagnostics = createRollingRequestDiagnostics({ now: () => nowMs });

  diagnostics.record({
    method: "GET",
    path: "/projects/project-1/jobs",
    reasonCode: "active_poll",
    statusCode: 200,
    durationMs: 12,
    responseBytes: 120,
  });
  nowMs = 2_000;
  const second = diagnostics.record({
    method: "GET",
    path: "/projects/project-1/agent-decisions",
    reasonCode: "activity_event",
    statusCode: 500,
    durationMs: 20,
    responseBytes: 40,
  });

  assert.equal(second.rolling_request_count, 2);
  assert.equal(second.rolling_broad_get_count, 2);
  assert.equal(second.rolling_error_count, 1);
  assert.equal(second.rolling_response_bytes, 160);
  assert.deepEqual(second.endpoint_category_counts, { project_history: 1, project_live: 1 });
  assert.equal(JSON.stringify(second).includes("project-1"), false, "diagnostics must not retain endpoint ids");

  nowMs = 61_000;
  const expired = diagnostics.snapshot();
  assert.equal(expired.rolling_request_count, 1, "the first second bucket should leave the rolling minute");
  assert.deepEqual(expired.endpoint_category_counts, { project_history: 1 });
});

test("deterministic active and idle baselines report broad GETs per minute", () => {
  const active = runBaselineRequestScenario("active");
  const idle = runBaselineRequestScenario("idle");

  assert.equal(active.refresh_interval_ms, 10_000);
  assert.equal(active.rolling_request_count, 63);
  assert.equal(active.rolling_broad_get_count, 63);
  assert.deepEqual(active.endpoint_category_counts, {
    health: 6,
    job_metrics: 6,
    project_index: 6,
    project_live: 45,
  });
  assert.deepEqual(active.reason_code_counts, { active_poll: 63 });

  assert.equal(idle.refresh_interval_ms, 30_000);
  assert.equal(idle.rolling_request_count, 22);
  assert.equal(idle.rolling_broad_get_count, 22);
  assert.deepEqual(idle.endpoint_category_counts, {
    health: 2,
    job_metrics: 2,
    project_index: 2,
    project_live: 16,
  });
  assert.deepEqual(idle.reason_code_counts, { idle_poll: 22 });
});

test("deprecated activity stream requests remain measurable without counting as broad refreshes", () => {
  let nowMs = 1_000;
  const diagnostics = createRollingRequestDiagnostics({ now: () => nowMs });
  const initial = diagnostics.record({
    method: "GET",
    path: "/projects/resource/activity-stream",
    reasonCode: "stream_initial",
    statusCode: 200,
  });
  nowMs = 2_000;
  const reconnect = diagnostics.record({
    method: "GET",
    path: "/projects/resource/activity-stream",
    reasonCode: "stream_reconnect",
    statusCode: 0,
    failed: true,
  });
  assert.equal(initial.endpoint_category, "activity_stream");
  assert.equal(reconnect.rolling_request_count, 2);
  assert.equal(reconnect.rolling_broad_get_count, 0);
  assert.equal(reconnect.rolling_error_count, 1);
  assert.deepEqual(reconnect.reason_code_counts, { stream_initial: 1, stream_reconnect: 1 });
});

test("rollout gate proves the active v2 request target and zero broad polling", () => {
  const report = runActivityRolloutReport();
  assert.equal(report.schema_version, "activity_rollout_report.v1");
  assert.equal(report.active.baseline_request_count, 63);
  assert.equal(report.active.rolling_request_count, 11);
  assert.equal(report.active.rolling_broad_get_count, 0);
  assert.ok(report.active.request_reduction_percent >= 80);
  assert.deepEqual(report.active.endpoint_category_counts, {
    execution_event_stream_v2: 2,
    job_metrics: 6,
    project_live_state: 3,
  });
  assert.equal(report.idle.rolling_request_count, 3);
  assert.equal(report.idle.rolling_broad_get_count, 0);
  assert.equal(report.go_no_go, true);
});
