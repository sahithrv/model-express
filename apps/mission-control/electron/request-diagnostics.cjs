const REQUEST_WINDOW_SECONDS = 60;
const LIVE_REQUEST_PLAN = require("../src/api/liveRequestPlan.json");

const REQUEST_REASON_CODES = new Set([
  "activity_event",
  "active_poll",
  "fallback_poll",
  "idle_poll",
  "initial_load",
  "manual_refresh",
  "project_change",
  "rollback_poll",
  "stream_initial",
  "stream_reconnect",
  "targeted",
  "targeted_invalidation",
  "v2_cursor_recovery",
  "v2_snapshot",
  "unspecified",
]);

const BROAD_GET_REASON_CODES = new Set([
  "activity_event",
  "active_poll",
  "fallback_poll",
  "idle_poll",
  "initial_load",
  "manual_refresh",
  "project_change",
  "rollback_poll",
]);

const LEGACY_BASELINE_REQUEST_KEYS = [
  "health",
  "projectIndex",
  "workerRequirements",
  "datasets",
  "jobs",
  "plans",
  "trainingRunSummaries",
  "champion",
  "workers",
  "executionEvents",
  "jobMetrics",
];
const BASELINE_FAST_REFRESH_REQUESTS = LEGACY_BASELINE_REQUEST_KEYS.map((key) => LIVE_REQUEST_PLAN[key]).map((entry) => ({
  path: entry.template
    .replace("{project_id}", "project-baseline")
    .replace("{job_id}", "job-baseline"),
  cacheTtlMs: entry.cache_ttl_ms,
}));
const BASELINE_FAST_REFRESH_PATHS = BASELINE_FAST_REFRESH_REQUESTS.map((entry) => entry.path);

function normalizeRequestReason(value) {
  const reason = String(value ?? "").trim().toLowerCase();
  return REQUEST_REASON_CODES.has(reason) ? reason : "unspecified";
}

function classifyEndpointCategory(requestPath) {
  const pathname = safePathname(requestPath);
  if (pathname === "/healthz") return "health";
  if (pathname === "/projects") return "project_index";
  if (pathname === "/settings/automation") return "settings";
  if (/^\/jobs\/[^/]+\/metrics$/.test(pathname)) return "job_metrics";
  if (/^\/projects\/[^/]+\/activity-stream$/.test(pathname)) return "activity_stream";
  if (/^\/projects\/[^/]+\/live-state$/.test(pathname)) return "project_live_state";
  if (/^\/projects\/[^/]+\/events\/stream\/v2$/.test(pathname)) return "execution_event_stream_v2";
  if (
    /^\/projects\/[^/]+\/(?:datasets|jobs|plans|training-run-summaries|workers|worker-requirements|execution-events)$/.test(pathname) ||
    /^\/projects\/[^/]+\/champion$/.test(pathname)
  ) {
    return "project_live";
  }
  if (
    /^\/projects\/[^/]+\/(?:training-run-evaluations|agent-decisions|agent-invocations|agent-memory|telemetry-summary|strategy-scorecards)$/.test(pathname) ||
    /^\/projects\/[^/]+\/champion\//.test(pathname)
  ) {
    return "project_history";
  }
  if (/^\/datasets\/[^/]+\//.test(pathname)) return "dataset_detail";
  if (/^\/jobs\/[^/]+(?:\/|$)/.test(pathname)) return "job_detail";
  if (/^\/projects\/[^/]+(?:\/|$)/.test(pathname)) return "project_other";
  if (/^\/workers(?:\/|$)/.test(pathname) || /^\/worker-requirements\//.test(pathname)) return "worker";
  return "other";
}

function safePathname(requestPath) {
  try {
    return new URL(String(requestPath ?? ""), "http://127.0.0.1").pathname;
  } catch {
    return "";
  }
}

function isBroadGet(method, reasonCode) {
  return String(method ?? "GET").toUpperCase() === "GET" && BROAD_GET_REASON_CODES.has(normalizeRequestReason(reasonCode));
}

function createRollingRequestDiagnostics(options = {}) {
  const now = typeof options.now === "function" ? options.now : Date.now;
  const buckets = new Map();

  function prune(currentSecond) {
    const oldestSecond = currentSecond - REQUEST_WINDOW_SECONDS + 1;
    for (const second of buckets.keys()) {
      if (second < oldestSecond || second > currentSecond) {
        buckets.delete(second);
      }
    }
  }

  function record(sample = {}) {
    const nowMs = finiteNonNegative(now());
    const currentSecond = Math.floor(nowMs / 1000);
    prune(currentSecond);
    let bucket = buckets.get(currentSecond);
    if (!bucket) {
      bucket = emptyBucket();
      buckets.set(currentSecond, bucket);
    }

    const methodCode = normalizedMethodCode(sample.method);
    const reasonCode = normalizeRequestReason(sample.reasonCode);
    const endpointCategory = classifyEndpointCategory(sample.path);
    const responseBytes = finiteNonNegativeInteger(sample.responseBytes);
    const durationMs = finiteNonNegativeInteger(sample.durationMs);
    const statusCode = finiteNonNegativeInteger(sample.statusCode);
    const failed = Boolean(sample.failed) || statusCode === 0 || statusCode >= 400;

    bucket.requestCount += 1;
    bucket.broadGetCount += isBroadGet(methodCode, reasonCode) ? 1 : 0;
    bucket.errorCount += failed ? 1 : 0;
    bucket.responseBytes += responseBytes;
    bucket.durationMs += durationMs;
    incrementCount(bucket.endpointCategoryCounts, endpointCategory);
    incrementCount(bucket.reasonCodeCounts, reasonCode);

    return {
      endpoint_category: endpointCategory,
      reason_code: reasonCode,
      method_code: methodCode,
      status_code: statusCode,
      duration_ms: durationMs,
      response_bytes: responseBytes,
      request_error_count: failed ? 1 : 0,
      ...snapshotAt(nowMs),
    };
  }

  function snapshot() {
    return snapshotAt(finiteNonNegative(now()));
  }

  function snapshotAt(nowMs) {
    const currentSecond = Math.floor(nowMs / 1000);
    prune(currentSecond);
    const summary = {
      window_seconds: REQUEST_WINDOW_SECONDS,
      window_resolution_ms: 1_000,
      rolling_request_count: 0,
      rolling_broad_get_count: 0,
      rolling_error_count: 0,
      rolling_response_bytes: 0,
      rolling_duration_ms: 0,
      endpoint_category_counts: {},
      reason_code_counts: {},
    };
    for (const bucket of buckets.values()) {
      summary.rolling_request_count += bucket.requestCount;
      summary.rolling_broad_get_count += bucket.broadGetCount;
      summary.rolling_error_count += bucket.errorCount;
      summary.rolling_response_bytes += bucket.responseBytes;
      summary.rolling_duration_ms += bucket.durationMs;
      mergeCounts(summary.endpoint_category_counts, bucket.endpointCategoryCounts);
      mergeCounts(summary.reason_code_counts, bucket.reasonCodeCounts);
    }
    summary.endpoint_category_counts = sortedCounts(summary.endpoint_category_counts);
    summary.reason_code_counts = sortedCounts(summary.reason_code_counts);
    return summary;
  }

  return { record, snapshot };
}

function emptyBucket() {
  return {
    requestCount: 0,
    broadGetCount: 0,
    errorCount: 0,
    responseBytes: 0,
    durationMs: 0,
    endpointCategoryCounts: {},
    reasonCodeCounts: {},
  };
}

function incrementCount(counts, key) {
  counts[key] = (counts[key] ?? 0) + 1;
}

function mergeCounts(target, source) {
  for (const [key, count] of Object.entries(source)) {
    target[key] = (target[key] ?? 0) + count;
  }
}

function sortedCounts(counts) {
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function normalizedMethodCode(method) {
  const value = String(method ?? "GET").trim().toUpperCase();
  return ["GET", "HEAD", "POST", "PATCH", "DELETE"].includes(value) ? value : "OTHER";
}

function finiteNonNegative(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : 0;
}

function finiteNonNegativeInteger(value) {
  return Math.round(finiteNonNegative(value));
}

function runBaselineRequestScenario(mode) {
  const active = mode === "active";
  const refreshIntervalMs = active ? 10_000 : 30_000;
  const reasonCode = active ? "active_poll" : "idle_poll";
  let nowMs = 0;
  const cacheExpiresAt = new Map();
  const diagnostics = createRollingRequestDiagnostics({ now: () => nowMs });

  for (nowMs = refreshIntervalMs; nowMs <= 60_000; nowMs += refreshIntervalMs) {
    for (const { path: requestPath, cacheTtlMs } of BASELINE_FAST_REFRESH_REQUESTS) {
      if ((cacheExpiresAt.get(requestPath) ?? 0) > nowMs) {
        continue;
      }
      diagnostics.record({
        method: "GET",
        path: requestPath,
        reasonCode,
        statusCode: 200,
        responseBytes: 0,
        durationMs: 0,
      });
      if (cacheTtlMs > 0) {
        cacheExpiresAt.set(requestPath, nowMs + cacheTtlMs);
      }
    }
  }

  nowMs = 60_000;
  return {
    mode: active ? "active" : "idle",
    refresh_interval_ms: refreshIntervalMs,
    duration_ms: 60_000,
    ...diagnostics.snapshot(),
  };
}

if (require.main === module) {
  process.stdout.write(`${JSON.stringify({
    active: runBaselineRequestScenario("active"),
    idle: runBaselineRequestScenario("idle"),
  }, null, 2)}\n`);
}

module.exports = {
  BASELINE_FAST_REFRESH_PATHS,
  BROAD_GET_REASON_CODES,
  REQUEST_REASON_CODES,
  classifyEndpointCategory,
  createRollingRequestDiagnostics,
  isBroadGet,
  normalizeRequestReason,
  runBaselineRequestScenario,
};
