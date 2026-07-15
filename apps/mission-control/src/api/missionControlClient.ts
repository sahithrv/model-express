import liveRequestPlan from "./liveRequestPlan.json";

export type MissionControlRequestReason =
  | "activity_event"
  | "active_poll"
  | "fallback_poll"
  | "idle_poll"
  | "initial_load"
  | "manual_refresh"
  | "project_change"
  | "rollback_poll"
  | "stream_initial"
  | "stream_reconnect"
  | "targeted"
  | "targeted_invalidation"
  | "v2_cursor_recovery"
  | "v2_snapshot"
  | "unspecified";

export type RequestOptions = {
  method?: string;
  body?: unknown;
  bypassCache?: boolean;
  cacheTtlMs?: number;
  diagnosticReason?: MissionControlRequestReason;
  signal?: AbortSignal;
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

export class OrchestratorHttpError extends Error {
  readonly status: number;
  readonly reasonCode: string;
  readonly policy: PolicyErrorDetails | null;

  constructor(response: OrchestratorHttpErrorResponse) {
    const statusText = response.statusText ? ` ${response.statusText}` : "";
    const message = response.message || "request failed";
    const requestPath = response.path ? ` (${response.path})` : "";
    super(`${response.status}${statusText} ${message}${requestPath}`);
    this.name = "OrchestratorHttpError";
    this.status = response.status;
    this.reasonCode = httpErrorReasonCode(response.payload);
    this.policy = policyErrorDetails(response.payload);
  }
}

export type PolicyErrorFinding = {
  code: string;
  catalog?: string;
  id?: string;
  fieldPath?: string;
  scope?: string;
  remediation?: string;
};

export type PolicyErrorDetails = {
  code: string;
  evaluationId?: string;
  effectivePolicyHash?: string;
  blockedDimensions: string[];
  findings: PolicyErrorFinding[];
};

export function isUnsupportedIncrementalStatus(status: number): boolean {
  return status === 404 || status === 405 || status === 501;
}

export function isCursorRecoveryStatus(status: number): boolean {
  return status === 400 || status === 409 || status === 410;
}

export function httpErrorReasonCode(payload: unknown): string {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return "";
  const source = payload as { reason_code?: unknown; code?: unknown };
  const value = typeof source.reason_code === "string" ? source.reason_code : source.code;
  return typeof value === "string" && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(value) ? value : "";
}

function boundedIdentifier(value: unknown, maxLength = 256): string | undefined {
  if (typeof value !== "string" || value.length === 0 || value.length > maxLength) return undefined;
  return /^[A-Za-z0-9_./:@-]+$/.test(value) ? value : undefined;
}

export function policyErrorDetails(payload: unknown): PolicyErrorDetails | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  const source = payload as Record<string, unknown>;
  const code = boundedIdentifier(source.code, 64);
  if (!code?.startsWith("POLICY_")) return null;
  const findings = Array.isArray(source.findings)
    ? source.findings.slice(0, 20).flatMap((value): PolicyErrorFinding[] => {
        if (!value || typeof value !== "object" || Array.isArray(value)) return [];
        const finding = value as Record<string, unknown>;
        const findingCode = boundedIdentifier(finding.code, 64);
        if (!findingCode?.startsWith("POLICY_")) return [];
        return [{
          code: findingCode,
          catalog: boundedIdentifier(finding.catalog),
          id: boundedIdentifier(finding.id),
          fieldPath: boundedIdentifier(finding.field_path),
          scope: boundedIdentifier(finding.scope, 32),
          remediation: typeof finding.remediation === "string" ? finding.remediation.slice(0, 500) : undefined,
        }];
      })
    : [];
  return {
    code,
    evaluationId: boundedIdentifier(source.policy_evaluation_id),
    effectivePolicyHash: boundedIdentifier(source.effective_policy_hash),
    blockedDimensions: Array.isArray(source.blocked_dimensions)
      ? source.blocked_dimensions.slice(0, 30).flatMap((value) => boundedIdentifier(value) ?? [])
      : [],
    findings,
  };
}

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
