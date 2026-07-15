export const PROJECT_LIVE_STATE_SCHEMA_VERSION = "project_live_state.v1" as const;
export const EXECUTION_EVENT_SCHEMA_VERSION = "execution_event.v2" as const;
export const PROGRESS_TAXONOMY_VERSION = 1 as const;
export const MAX_ACTIVE_PROGRESS = 8;

export const STABLE_PROGRESS_STAGES = [
  "queued",
  "worker_starting",
  "remote_scheduled",
  "environment_starting",
  "dataset_materializing",
  "data_loading",
  "model_initializing",
  "training",
  "evaluating",
  "exporting",
  "finalizing",
  "completed",
  "failed",
  "cancelled",
] as const;

export type StableProgressStage = (typeof STABLE_PROGRESS_STAGES)[number];
export type ProgressStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type LiveOperationalState = "idle" | "queued" | "active" | "retrying" | "blocked" | "stale" | "terminal";

export type LiveJobCounts = {
  total: number;
  queued: number;
  retrying: number;
  assigned: number;
  running: number;
  succeeded: number;
  failed: number;
  cancelled: number;
};

export type LiveWorkerCounts = {
  total: number;
  idle: number;
  running: number;
  offline: number;
  stale: number;
};

export type LiveWorkerRequirementCounts = {
  pending: number;
  starting: number;
  active: number;
  satisfied: number;
  failed: number;
  cancelled: number;
};

export type ProgressMetadata = {
  cache_status?: string;
  early_stopped?: boolean;
  execution_mode?: string;
  framework?: string;
  provider?: string;
  resource_class?: string;
  task_type?: string;
};

export type ActiveAttemptProgress = {
  job_id: string;
  attempt: number;
  taxonomy_version: typeof PROGRESS_TAXONOMY_VERSION;
  stage: StableProgressStage;
  detail_code?: string;
  status: ProgressStatus;
  current?: number;
  total?: number;
  unit?: string;
  message?: string;
  revision: number;
  heartbeat_at: string;
  updated_at: string;
  /** Optional rollout field. Older live-state.v1 backends do not return it. */
  elapsed_started_at?: string;
  stale: boolean;
  metadata: ProgressMetadata;
};

export const SAFE_EXECUTION_EVENT_METADATA_KEYS = [
  "category", "phase", "status", "severity", "agent_name", "invocation_id", "decision_id",
  "source_decision_id", "decision_type", "job_id", "attempt_id", "job_ids", "worker_requirement_id",
  "open_job_count", "active_worker_count", "target_count", "previous_slot_count", "slot_count",
  "desired_slot_count", "registered_slot_count", "active_slot_count", "idle_seconds", "idle_exit_seconds",
  "dispatcher", "provider", "gpu_type", "requirement_status", "template", "attempt", "taxonomy_version",
  "stage", "detail_code", "revision", "current", "total", "unit", "max_attempts", "requeued",
  "backend_validation_status", "backend_stop_guard", "reason", "reason_code", "model", "selection_source",
  "materialization_status", "max_concurrent_jobs", "max_cold_dataset_materializations", "retry_attempt",
  "will_retry", "completed_run_count", "memory_count", "evaluation_count", "purpose", "retrieved_count",
  "log_only", "cross_project_ok", "validation_error", "error_summary",
] as const;

export type SafeExecutionEventMetadataKey = (typeof SAFE_EXECUTION_EVENT_METADATA_KEYS)[number];
export type SafeExecutionEventMetadataValue = string | number | boolean | readonly string[];
export type ExecutionEventMetadata = Partial<Record<SafeExecutionEventMetadataKey, SafeExecutionEventMetadataValue>>;

export type ExecutionEventV2 = {
  schema_version: typeof EXECUTION_EVENT_SCHEMA_VERSION;
  sequence: number;
  event_id: string;
  project_id: string;
  plan_id?: string;
  event_type: string;
  message: string;
  created_at: string;
  metadata: ExecutionEventMetadata;
  /** Optional opaque identity. It is deliberately never interpreted or logged. */
  idempotency_key?: string;
};

export type ProjectLiveState = {
  schema_version: typeof PROJECT_LIVE_STATE_SCHEMA_VERSION;
  project_id: string;
  operational_state: LiveOperationalState;
  taxonomy_version: typeof PROGRESS_TAXONOMY_VERSION;
  current_stage?: StableProgressStage;
  next_expected_stage?: StableProgressStage;
  mixed_jobs: boolean;
  stale: boolean;
  blocked_reason_code?: string;
  jobs: LiveJobCounts;
  workers: LiveWorkerCounts;
  worker_requirements: LiveWorkerRequirementCounts;
  active_progress: ActiveAttemptProgress[];
  active_progress_total: number;
  active_progress_truncated: boolean;
  last_heartbeat_at?: string;
  latest_important_event?: ExecutionEventV2;
  snapshot_cursor: number;
  snapshot_revision?: string;
  /** Optional project-level start hint for mixed-version deployments. */
  elapsed_started_at?: string;
};

export type LiveStateContractErrorCode =
  | "malformed"
  | "project_mismatch"
  | "unsupported_schema"
  | "unsupported_taxonomy";

export class LiveStateContractError extends Error {
  readonly code: LiveStateContractErrorCode;

  constructor(code: LiveStateContractErrorCode, message: string) {
    super(message);
    this.name = "LiveStateContractError";
    this.code = code;
  }
}

const stableProgressStageSet = new Set<string>(STABLE_PROGRESS_STAGES);
const operationalStateSet = new Set<string>(["idle", "queued", "active", "retrying", "blocked", "stale", "terminal"]);
const progressStatusSet = new Set<string>(["queued", "running", "completed", "failed", "cancelled"]);
const eventMetadataKeySet = new Set<string>(SAFE_EXECUTION_EVENT_METADATA_KEYS);
const identifierPattern = /^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,199}$/;
const tokenPattern = /^[a-z0-9][a-z0-9_.-]{0,63}$/;
const eventTypePattern = /^[A-Z][A-Z0-9_]{0,95}$/;
const unsafeURI = /\b(?:s3|gs|file|minio|https?):\/\/[^\s,;"')\]}]+/gi;
const unsafeWindowsPath = /\b[A-Z]:\\[^\s,;"')\]}]+/gi;
const unsafeUnixPath = /(^|\s)\/(?:Users|home|tmp|var|mnt|data|datasets|artifacts|workspace|app|srv)[^\s,;"')\]}]+/gi;
const unsafeSecret = /\b(?:sk|pk|rk|xox[baprs]?)-[A-Za-z0-9_-]{12,}\b/gi;
const unsafeBlob = /\b[A-Za-z0-9+/]{80,}={0,2}\b/g;

export function parseProjectLiveState(value: unknown, expectedProjectId = ""): ProjectLiveState {
  const input = objectValue(value, "live-state response");
  const schema = stringValue(input.schema_version, "schema_version", 64);
  if (schema !== PROJECT_LIVE_STATE_SCHEMA_VERSION) {
    throw new LiveStateContractError("unsupported_schema", `Unsupported live-state schema ${schema || "(empty)"}.`);
  }
  const projectId = identifierValue(input.project_id, "project_id");
  if (expectedProjectId && projectId !== expectedProjectId) {
    throw new LiveStateContractError("project_mismatch", "Live-state response belongs to a different project.");
  }
  const taxonomyVersion = nonNegativeInteger(input.taxonomy_version, "taxonomy_version");
  if (taxonomyVersion !== PROGRESS_TAXONOMY_VERSION) {
    throw new LiveStateContractError("unsupported_taxonomy", `Unsupported progress taxonomy ${taxonomyVersion}.`);
  }
  const operationalState = stringValue(input.operational_state, "operational_state", 32);
  if (!operationalStateSet.has(operationalState)) malformed("operational_state is not supported.");
  const activeInput = arrayValue(input.active_progress, "active_progress");
  if (activeInput.length > MAX_ACTIVE_PROGRESS) malformed(`active_progress exceeds ${MAX_ACTIVE_PROGRESS} records.`);
  const activeProgress = activeInput.map((progress, index) => parseActiveAttemptProgress(progress, `active_progress[${index}]`));
  const snapshotCursor = nonNegativeInteger(input.snapshot_cursor, "snapshot_cursor");
  const latestImportantEvent = optionalValue(input.latest_important_event, (event) => parseExecutionEventV2(event, projectId));
  if (latestImportantEvent && latestImportantEvent.sequence > snapshotCursor) {
    malformed("latest_important_event is newer than snapshot_cursor.");
  }
  const activeProgressTotal = nonNegativeInteger(input.active_progress_total, "active_progress_total");
  if (activeProgressTotal < activeProgress.length) malformed("active_progress_total is smaller than active_progress.");

  return {
    schema_version: PROJECT_LIVE_STATE_SCHEMA_VERSION,
    project_id: projectId,
    operational_state: operationalState as LiveOperationalState,
    taxonomy_version: PROGRESS_TAXONOMY_VERSION,
    current_stage: optionalStage(input.current_stage, "current_stage"),
    next_expected_stage: optionalStage(input.next_expected_stage, "next_expected_stage"),
    mixed_jobs: booleanValue(input.mixed_jobs, "mixed_jobs"),
    stale: booleanValue(input.stale, "stale"),
    blocked_reason_code: optionalToken(input.blocked_reason_code, "blocked_reason_code"),
    jobs: parseCounts(input.jobs, "jobs", ["total", "queued", "retrying", "assigned", "running", "succeeded", "failed", "cancelled"]),
    workers: parseCounts(input.workers, "workers", ["total", "idle", "running", "offline", "stale"]),
    worker_requirements: parseCounts(input.worker_requirements, "worker_requirements", ["pending", "starting", "active", "satisfied", "failed", "cancelled"]),
    active_progress: activeProgress,
    active_progress_total: activeProgressTotal,
    active_progress_truncated: booleanValue(input.active_progress_truncated, "active_progress_truncated"),
    last_heartbeat_at: optionalTimestamp(input.last_heartbeat_at, "last_heartbeat_at"),
    latest_important_event: latestImportantEvent,
    snapshot_cursor: snapshotCursor,
    snapshot_revision: optionalIdentifier(input.snapshot_revision, "snapshot_revision"),
    elapsed_started_at: optionalTimestamp(input.elapsed_started_at, "elapsed_started_at"),
  } as ProjectLiveState;
}

export function parseExecutionEventV2(value: unknown, expectedProjectId = ""): ExecutionEventV2 {
  const input = objectValue(value, "execution event");
  const schema = stringValue(input.schema_version, "schema_version", 64);
  if (schema !== EXECUTION_EVENT_SCHEMA_VERSION) {
    throw new LiveStateContractError("unsupported_schema", `Unsupported execution-event schema ${schema || "(empty)"}.`);
  }
  const projectId = identifierValue(input.project_id, "project_id");
  if (expectedProjectId && projectId !== expectedProjectId) {
    throw new LiveStateContractError("project_mismatch", "Execution event belongs to a different project.");
  }
  const eventType = stringValue(input.event_type, "event_type", 96);
  if (!eventTypePattern.test(eventType)) malformed("event_type is not a safe event identifier.");
  return {
    schema_version: EXECUTION_EVENT_SCHEMA_VERSION,
    sequence: nonNegativeInteger(input.sequence, "sequence"),
    event_id: identifierValue(input.event_id, "event_id"),
    project_id: projectId,
    plan_id: optionalIdentifier(input.plan_id, "plan_id"),
    event_type: eventType,
    message: sanitizeBoundedText(stringValue(input.message, "message", 512), 220),
    created_at: timestampValue(input.created_at, "created_at"),
    metadata: projectExecutionEventMetadata(input.metadata),
    idempotency_key: optionalIdentifier(input.idempotency_key, "idempotency_key"),
  };
}

export function projectExecutionEventMetadata(value: unknown): ExecutionEventMetadata {
  if (value === undefined || value === null) return {};
  const input = objectValue(value, "metadata");
  const output: ExecutionEventMetadata = {};
  for (const [key, raw] of Object.entries(input)) {
    if (!eventMetadataKeySet.has(key)) continue;
    const projected = metadataValue(raw);
    if (projected !== undefined) output[key as SafeExecutionEventMetadataKey] = projected;
  }
  return output;
}

export function sanitizeBoundedText(value: string, maxLength: number): string {
  const safe = value
    .replace(unsafeURI, "[redacted_uri]")
    .replace(unsafeWindowsPath, "[redacted_path]")
    .replace(unsafeUnixPath, "$1[redacted_path]")
    .replace(unsafeSecret, "[redacted_secret]")
    .replace(unsafeBlob, "[redacted_blob]")
    .replace(/\s+/g, " ")
    .trim();
  return safe.length > maxLength ? `${safe.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…` : safe;
}

function parseActiveAttemptProgress(value: unknown, path: string): ActiveAttemptProgress {
  const input = objectValue(value, path);
  const taxonomyVersion = nonNegativeInteger(input.taxonomy_version, `${path}.taxonomy_version`);
  if (taxonomyVersion !== PROGRESS_TAXONOMY_VERSION) {
    throw new LiveStateContractError("unsupported_taxonomy", `Unsupported progress taxonomy ${taxonomyVersion}.`);
  }
  const stage = stageValue(input.stage, `${path}.stage`);
  const status = stringValue(input.status, `${path}.status`, 32);
  if (!progressStatusSet.has(status)) malformed(`${path}.status is not supported.`);
  const current = optionalInteger(input.current, `${path}.current`);
  const total = optionalInteger(input.total, `${path}.total`);
  if (current !== undefined && total !== undefined && current > total) malformed(`${path}.current exceeds total.`);
  const unit = optionalToken(input.unit, `${path}.unit`);
  const heartbeatAt = timestampValue(input.heartbeat_at, `${path}.heartbeat_at`);
  const updatedAt = timestampValue(input.updated_at, `${path}.updated_at`);
  return {
    job_id: identifierValue(input.job_id, `${path}.job_id`),
    attempt: nonNegativeInteger(input.attempt, `${path}.attempt`),
    taxonomy_version: PROGRESS_TAXONOMY_VERSION,
    stage,
    detail_code: optionalToken(input.detail_code, `${path}.detail_code`),
    status: status as ProgressStatus,
    current,
    total,
    unit,
    message: input.message === undefined ? undefined : sanitizeBoundedText(stringValue(input.message, `${path}.message`, 512), 220),
    revision: nonNegativeInteger(input.revision, `${path}.revision`),
    heartbeat_at: heartbeatAt,
    updated_at: updatedAt,
    elapsed_started_at: optionalTimestamp(input.elapsed_started_at, `${path}.elapsed_started_at`),
    stale: booleanValue(input.stale, `${path}.stale`),
    metadata: progressMetadata(input.metadata),
  };
}

function progressMetadata(value: unknown): ProgressMetadata {
  if (value === undefined || value === null) return {};
  const input = objectValue(value, "progress metadata");
  const output: ProgressMetadata = {};
  for (const key of ["cache_status", "execution_mode", "framework", "provider", "resource_class", "task_type"] as const) {
    if (typeof input[key] === "string") output[key] = sanitizeBoundedText(input[key] as string, 120);
  }
  if (typeof input.early_stopped === "boolean") output.early_stopped = input.early_stopped;
  return output;
}

function metadataValue(value: unknown): SafeExecutionEventMetadataValue | undefined {
  if (typeof value === "boolean") return value;
  if (typeof value === "number" && Number.isSafeInteger(value)) return value;
  if (typeof value === "string") return sanitizeBoundedText(value, 120);
  if (Array.isArray(value) && value.length <= 8 && value.every((item) => typeof item === "string")) {
    return value.map((item) => sanitizeBoundedText(item, 120));
  }
  return undefined;
}

function parseCounts<K extends string>(value: unknown, path: string, keys: readonly K[]): Record<K, number> {
  const input = objectValue(value, path);
  const output = {} as Record<K, number>;
  for (const key of keys) output[key] = nonNegativeInteger(input[key], `${path}.${key}`);
  return output;
}

function objectValue(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) malformed(`${path} must be an object.`);
  return value as Record<string, unknown>;
}

function arrayValue(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) malformed(`${path} must be an array.`);
  return value as unknown[];
}

function stringValue(value: unknown, path: string, maxLength: number): string {
  if (typeof value !== "string" || value.length > maxLength) malformed(`${path} must be a bounded string.`);
  return value as string;
}

function identifierValue(value: unknown, path: string): string {
  const text = stringValue(value, path, 200).trim();
  if (!identifierPattern.test(text)) malformed(`${path} is not a safe identifier.`);
  return text;
}

function optionalIdentifier(value: unknown, path: string): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return identifierValue(value, path);
}

function optionalToken(value: unknown, path: string): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const text = stringValue(value, path, 64).trim().toLowerCase();
  if (!tokenPattern.test(text)) malformed(`${path} is not a safe token.`);
  return text;
}

function stageValue(value: unknown, path: string): StableProgressStage {
  const text = stringValue(value, path, 64).trim().toLowerCase();
  if (!stableProgressStageSet.has(text)) malformed(`${path} is not a stable progress stage.`);
  return text as StableProgressStage;
}

function optionalStage(value: unknown, path: string): StableProgressStage | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return stageValue(value, path);
}

function nonNegativeInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) malformed(`${path} must be a nonnegative safe integer.`);
  return value as number;
}

function optionalInteger(value: unknown, path: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  return nonNegativeInteger(value, path);
}

function booleanValue(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") malformed(`${path} must be a boolean.`);
  return value as boolean;
}

function timestampValue(value: unknown, path: string): string {
  const text = stringValue(value, path, 64);
  if (!Number.isFinite(Date.parse(text))) malformed(`${path} must be an ISO timestamp.`);
  return text;
}

function optionalTimestamp(value: unknown, path: string): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return timestampValue(value, path);
}

function optionalValue<T>(value: unknown, parse: (input: unknown) => T): T | undefined {
  return value === undefined || value === null ? undefined : parse(value);
}

function malformed(message: string): never {
  throw new LiveStateContractError("malformed", message);
}
