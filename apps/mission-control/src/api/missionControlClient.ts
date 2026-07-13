import liveRequestPlan from "./liveRequestPlan.json";

export type MissionControlRequestReason =
  | "activity_event"
  | "active_poll"
  | "idle_poll"
  | "initial_load"
  | "manual_refresh"
  | "project_change"
  | "stream_initial"
  | "stream_reconnect"
  | "targeted"
  | "unspecified";

export type RequestOptions = {
  method?: string;
  body?: unknown;
  bypassCache?: boolean;
  cacheTtlMs?: number;
  diagnosticReason?: MissionControlRequestReason;
};

export type CachedGetRequest = {
  expiresAt: number;
  hasValue: boolean;
  promise?: Promise<unknown>;
  value?: unknown;
};

export type OrchestratorHttpErrorResponse = {
  __mission_control_http_error: true;
  status: number;
  statusText?: string;
  message?: string;
  path?: string;
  url?: string;
  payload?: unknown;
};

const expensiveGetCacheTtlMs = 15_000;

export type LiveRequestPlanKey = keyof typeof liveRequestPlan;

export function liveRequestPath(
  key: LiveRequestPlanKey,
  identifiers: { projectId?: string; jobId?: string } = {},
): string {
  return liveRequestPlan[key].template
    .replace("{project_id}", encodeURIComponent(identifiers.projectId ?? ""))
    .replace("{job_id}", encodeURIComponent(identifiers.jobId ?? ""));
}

export function cachedGetRequestTtlMs(path: string): number {
  const normalizedPath = path.split("?")[0] ?? path;
  if (/^\/projects\/[^/]+\/execution-events$/.test(normalizedPath)) {
    return liveRequestPlan.executionEvents.cache_ttl_ms;
  }
  if (/^\/projects\/[^/]+\/(agent-invocations|agent-decisions|agent-memory|strategy-scorecards|training-run-evaluations)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/projects\/[^/]+\/telemetry-summary$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/datasets\/[^/]+\/(visual-analyses|visual-analyses\/latest|metadata\/summary|metadata\/imports)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/projects\/[^/]+\/champion\/(exports|demo-images|demo-predictions|feedback)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  return 0;
}

export function isOrchestratorHttpErrorResponse(value: unknown): value is OrchestratorHttpErrorResponse {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    (value as { __mission_control_http_error?: unknown }).__mission_control_http_error === true &&
    typeof (value as { status?: unknown }).status === "number"
  );
}
