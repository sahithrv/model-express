import {
  MAX_ACTIVE_PROGRESS,
  PROGRESS_TAXONOMY_VERSION,
  STABLE_PROGRESS_STAGES,
  type ActiveAttemptProgress,
  type ExecutionEventMetadata,
  type ExecutionEventV2,
  type ProgressStatus,
  type ProjectLiveState,
  type StableProgressStage,
} from "./liveStateContract";

export const MAX_SEEN_EVENT_IDENTITIES = 128;

export type LiveStateTransition =
  | "idle"
  | "snapshot_installed"
  | "event_applied"
  | "progress_applied"
  | "duplicate_cursor"
  | "duplicate_event"
  | "duplicate_idempotency_key"
  | "out_of_order"
  | "stale_progress"
  | "project_mismatch"
  | "older_snapshot";

export type IncrementalLiveState = {
  project_id: string;
  snapshot: ProjectLiveState | null;
  cursor: number;
  snapshot_installed: boolean;
  cursor_consistent: boolean;
  seen_event_ids: readonly string[];
  seen_idempotency_keys: readonly string[];
  applied_event_count: number;
  duplicate_event_count: number;
  rejected_event_count: number;
  last_transition: LiveStateTransition;
};

export type LiveStateReducerAction =
  | { type: "install_snapshot"; snapshot: ProjectLiveState }
  | { type: "apply_event"; event: ExecutionEventV2 }
  | { type: "reset"; project_id: string };

const stableStageSet = new Set<string>(STABLE_PROGRESS_STAGES);
const progressStatusSet = new Set<string>(["queued", "running", "completed", "failed", "cancelled"]);

export function createIncrementalLiveState(projectId: string): IncrementalLiveState {
  return {
    project_id: projectId,
    snapshot: null,
    cursor: 0,
    snapshot_installed: false,
    cursor_consistent: true,
    seen_event_ids: [],
    seen_idempotency_keys: [],
    applied_event_count: 0,
    duplicate_event_count: 0,
    rejected_event_count: 0,
    last_transition: "idle",
  };
}

export function liveStateReducer(state: IncrementalLiveState, action: LiveStateReducerAction): IncrementalLiveState {
  switch (action.type) {
    case "reset":
      return createIncrementalLiveState(action.project_id);
    case "install_snapshot":
      return installLiveStateSnapshot(state, action.snapshot);
    case "apply_event":
      return applyExecutionEvent(state, action.event);
  }
}

/** Installs the complete snapshot and its cursor in one reducer transition. */
export function installLiveStateSnapshot(
  state: IncrementalLiveState,
  snapshot: ProjectLiveState,
): IncrementalLiveState {
  if (snapshot.project_id !== state.project_id) {
    return { ...state, rejected_event_count: state.rejected_event_count + 1, last_transition: "project_mismatch" };
  }
  if (state.snapshot_installed && snapshot.snapshot_cursor < state.cursor) {
    return { ...state, rejected_event_count: state.rejected_event_count + 1, last_transition: "older_snapshot" };
  }

  const latest = snapshot.latest_important_event;
  return {
    ...state,
    snapshot,
    cursor: snapshot.snapshot_cursor,
    snapshot_installed: true,
    cursor_consistent: true,
    seen_event_ids: latest ? appendBoundedIdentity(state.seen_event_ids, latest.event_id) : state.seen_event_ids,
    seen_idempotency_keys: latest?.idempotency_key
      ? appendBoundedIdentity(state.seen_idempotency_keys, latest.idempotency_key)
      : state.seen_idempotency_keys,
    last_transition: "snapshot_installed",
  };
}

/**
 * Applies a project-filtered event after the snapshot cursor. Execution-event
 * sequences are global, so jumps are valid; only non-increasing cursors are
 * duplicates/out-of-order.
 */
export function applyExecutionEvent(
  state: IncrementalLiveState,
  event: ExecutionEventV2,
): IncrementalLiveState {
  if (event.project_id !== state.project_id) {
    return { ...state, rejected_event_count: state.rejected_event_count + 1, last_transition: "project_mismatch" };
  }
  if (!state.snapshot_installed || !state.snapshot) {
    return { ...state, cursor_consistent: false, rejected_event_count: state.rejected_event_count + 1, last_transition: "out_of_order" };
  }
  const duplicateEvent = state.seen_event_ids.includes(event.event_id);
  const duplicateIdempotency = Boolean(
    event.idempotency_key && state.seen_idempotency_keys.includes(event.idempotency_key),
  );
  if (event.sequence < state.cursor) {
    if (duplicateEvent || duplicateIdempotency) {
      return {
        ...state,
        duplicate_event_count: state.duplicate_event_count + 1,
        last_transition: duplicateEvent ? "duplicate_event" : "duplicate_idempotency_key",
      };
    }
    return { ...state, rejected_event_count: state.rejected_event_count + 1, last_transition: "out_of_order" };
  }
  if (event.sequence === state.cursor) {
    return { ...state, duplicate_event_count: state.duplicate_event_count + 1, last_transition: "duplicate_cursor" };
  }

  const seenEventIds = appendBoundedIdentity(state.seen_event_ids, event.event_id);
  const seenIdempotencyKeys = event.idempotency_key
    ? appendBoundedIdentity(state.seen_idempotency_keys, event.idempotency_key)
    : state.seen_idempotency_keys;

  if (duplicateEvent || duplicateIdempotency) {
    return {
      ...state,
      snapshot: advanceSnapshotCursor(state.snapshot, event.sequence),
      cursor: event.sequence,
      seen_event_ids: seenEventIds,
      seen_idempotency_keys: seenIdempotencyKeys,
      duplicate_event_count: state.duplicate_event_count + 1,
      last_transition: duplicateEvent ? "duplicate_event" : "duplicate_idempotency_key",
    };
  }

  const progressResult = event.event_type === "JOB_PROGRESS_BOUNDARY"
    ? applyProgressBoundary(state.snapshot, event)
    : { snapshot: state.snapshot, transition: "event_applied" as const };
  const snapshot = {
    ...progressResult.snapshot,
    latest_important_event: event,
    snapshot_cursor: event.sequence,
  };
  return {
    ...state,
    snapshot,
    cursor: event.sequence,
    cursor_consistent: true,
    seen_event_ids: seenEventIds,
    seen_idempotency_keys: seenIdempotencyKeys,
    applied_event_count: state.applied_event_count + 1,
    last_transition: progressResult.transition,
  };
}

function applyProgressBoundary(
  snapshot: ProjectLiveState,
  event: ExecutionEventV2,
): { snapshot: ProjectLiveState; transition: "progress_applied" | "stale_progress" | "event_applied" } {
  const metadata = event.metadata;
  const jobId = metadataString(metadata, "job_id");
  const stage = metadataStage(metadata);
  const attempt = metadataInteger(metadata, "attempt");
  const revision = metadataInteger(metadata, "revision");
  const taxonomyVersion = metadataInteger(metadata, "taxonomy_version");
  if (!jobId || !stage || attempt === undefined || revision === undefined || taxonomyVersion !== PROGRESS_TAXONOMY_VERSION) {
    return { snapshot, transition: "event_applied" };
  }

  const index = snapshot.active_progress.findIndex((progress) => progress.job_id === jobId);
  const previous = index >= 0 ? snapshot.active_progress[index] : undefined;
  if (previous && (attempt < previous.attempt || (attempt === previous.attempt && revision <= previous.revision))) {
    return { snapshot, transition: "stale_progress" };
  }
  const current = metadataInteger(metadata, "current");
  const total = metadataInteger(metadata, "total");
  const hasRange = current !== undefined && total !== undefined && current <= total;
  const statusValue = metadataString(metadata, "status");
  const status: ProgressStatus = statusValue && progressStatusSet.has(statusValue)
    ? statusValue as ProgressStatus
    : "running";
  const progress: ActiveAttemptProgress = {
    job_id: jobId,
    attempt,
    taxonomy_version: PROGRESS_TAXONOMY_VERSION,
    stage,
    detail_code: metadataToken(metadata, "detail_code"),
    status,
    current: hasRange ? current : undefined,
    total: hasRange ? total : undefined,
    unit: hasRange ? metadataToken(metadata, "unit") : undefined,
    message: event.message || previous?.message,
    revision,
    heartbeat_at: event.created_at,
    updated_at: event.created_at,
    elapsed_started_at: previous?.elapsed_started_at,
    // Staleness remains server-authoritative until a replacement snapshot.
    stale: previous?.stale ?? snapshot.stale,
    metadata: previous?.metadata ?? {},
  };

  const activeProgress = [...snapshot.active_progress];
  if (index >= 0) activeProgress[index] = progress;
  else activeProgress.unshift(progress);
  const bounded = activeProgress.slice(0, MAX_ACTIVE_PROGRESS);
  const primaryChanged = index <= 0;
  return {
    snapshot: {
      ...snapshot,
      active_progress: bounded,
      active_progress_total: Math.max(snapshot.active_progress_total, bounded.length),
      current_stage: primaryChanged ? stage : snapshot.current_stage,
      last_heartbeat_at: laterTimestamp(snapshot.last_heartbeat_at, event.created_at),
      // Do not derive operational terminal state from a worker observation.
      operational_state: snapshot.operational_state,
      stale: snapshot.stale,
    },
    transition: "progress_applied",
  };
}

function advanceSnapshotCursor(snapshot: ProjectLiveState, cursor: number): ProjectLiveState {
  return cursor > snapshot.snapshot_cursor ? { ...snapshot, snapshot_cursor: cursor } : snapshot;
}

function appendBoundedIdentity(current: readonly string[], value: string): readonly string[] {
  if (!value || current.includes(value)) return current;
  const next = [...current, value];
  return next.length > MAX_SEEN_EVENT_IDENTITIES ? next.slice(next.length - MAX_SEEN_EVENT_IDENTITIES) : next;
}

function metadataString<K extends keyof ExecutionEventMetadata>(metadata: ExecutionEventMetadata, key: K): string {
  const value = metadata[key];
  return typeof value === "string" ? value : "";
}

function metadataInteger<K extends keyof ExecutionEventMetadata>(metadata: ExecutionEventMetadata, key: K): number | undefined {
  const value = metadata[key];
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function metadataToken<K extends keyof ExecutionEventMetadata>(metadata: ExecutionEventMetadata, key: K): string | undefined {
  const value = metadataString(metadata, key).trim().toLowerCase();
  return value && /^[a-z0-9][a-z0-9_.-]{0,63}$/.test(value) ? value : undefined;
}

function metadataStage(metadata: ExecutionEventMetadata): StableProgressStage | undefined {
  const value = metadataString(metadata, "stage").trim().toLowerCase();
  return stableStageSet.has(value) ? value as StableProgressStage : undefined;
}

function laterTimestamp(left: string | undefined, right: string): string {
  if (!left) return right;
  return Date.parse(right) > Date.parse(left) ? right : left;
}
