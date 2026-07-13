import type { ExecutionEventV2, ProjectLiveState } from "./liveStateContract";

export const LIVE_RESOURCE_KEYS = [
  "live_state",
  "project_index",
  "jobs",
  "metrics",
  "training_results",
  "decisions",
  "champion",
  "champion_exports",
  "champion_demo_predictions",
  "champion_feedback",
  "workers",
  "worker_requirements",
  "plans",
  "dataset_visual_analysis",
] as const;

export type LiveResource = (typeof LIVE_RESOURCE_KEYS)[number];

const NONE = [] as const satisfies readonly LiveResource[];
const PROGRESS = ["metrics"] as const satisfies readonly LiveResource[];
const JOB_OPERATIONAL = ["live_state"] as const satisfies readonly LiveResource[];
const JOB_TERMINAL = ["live_state", "project_index", "jobs", "metrics", "training_results"] as const satisfies readonly LiveResource[];
const WORKER_OPERATIONAL = ["live_state"] as const satisfies readonly LiveResource[];
const RETRY_OPERATIONAL = ["live_state"] as const satisfies readonly LiveResource[];
const CHAMPION_CHANGED = ["champion", "training_results", "champion_exports"] as const satisfies readonly LiveResource[];

/**
 * This map is intentionally explicit. Unknown event types resolve to no fetch;
 * they never trigger a broad project-detail refresh.
 */
export const EVENT_RESOURCE_INVALIDATIONS = {
  // Typed durable job/progress transitions.
  JOB_QUEUED: JOB_OPERATIONAL,
  JOB_ASSIGNED: JOB_OPERATIONAL,
  JOB_RUNNING: JOB_OPERATIONAL,
  JOB_RETRY_QUEUED_TRANSITION: RETRY_OPERATIONAL,
  JOB_CANCELLED: JOB_TERMINAL,
  JOB_COMPLETED: JOB_TERMINAL,
  JOB_FAILED: JOB_TERMINAL,
  JOB_LEASE_RECOVERED: RETRY_OPERATIONAL,
  JOB_PROGRESS_BOUNDARY: PROGRESS,

  // Typed durable agent transitions.
  AGENT_VALIDATION_REJECTED: NONE,
  AGENT_VALIDATION_RETRYING: NONE,
  AGENT_VALIDATION_ACCEPTED: NONE,
  AGENT_VALIDATION_FAILED: NONE,
  AGENT_DECISION_RECORDED: ["decisions"],

  // Compatibility event types emitted by older/mixed backends.
  JOBS_QUEUED: JOB_OPERATIONAL,
  JOB_RETRY_QUEUED: RETRY_OPERATIONAL,
  JOB_STALE_CALLBACK_IGNORED: NONE,
  EXECUTION_CANCELLATION_REQUESTED: JOB_OPERATIONAL,
  EXECUTION_CANCELLED: JOB_TERMINAL,
  EXECUTION_FAILED: JOB_TERMINAL,
  REMOTE_WORK_CANCEL_REQUESTED: NONE,
  REMOTE_WORK_CANCEL_FAILED: NONE,
  WORKERS_REQUIRED: WORKER_OPERATIONAL,
  WORKER_SCALING_UPDATED: WORKER_OPERATIONAL,
  WORKERS_STARTING: WORKER_OPERATIONAL,
  WORKERS_ACTIVE: WORKER_OPERATIONAL,
  DISPATCHER_STATUS: WORKER_OPERATIONAL,
  DISPATCHER_IDLE_EXIT: WORKER_OPERATIONAL,
  COST_BUDGET_BLOCKED: ["live_state"],
  CHAMPION_SELECTED: CHAMPION_CHANGED,
  CHAMPION_EXPORT_REQUESTED: ["champion_exports"],
  CHAMPION_DEMO_PREDICTION: ["champion_demo_predictions"],
  CHAMPION_FEEDBACK_RECORDED: ["champion_feedback"],
  AGENT_STARTED: NONE,
  AGENT_RECOMMENDATION_RECORDED: ["decisions"],
  AGENT_OUTCOME_RECORDED: ["decisions"],
  AGENT_FAILED: NONE,
  EXECUTION_VALIDATION_REPORTED: NONE,
  MEMORY_RETRIEVAL_LOGGED: NONE,
  EXPERIMENTATION_REOPENED: ["live_state", "project_index", "plans"],
  DATASET_VISUAL_ANALYSIS_QUEUED: ["dataset_visual_analysis"],
  DATASET_VISUAL_ANALYSIS_RESULT: ["dataset_visual_analysis"],

  // Pre-boundary progress names accepted during mixed worker rollout.
  EPOCH_COMPLETED: PROGRESS,
  EPOCH_PROGRESS: PROGRESS,
  TRAINING_EPOCH_COMPLETED: PROGRESS,
  JOB_PROGRESS: PROGRESS,
  JOB_PROGRESS_UPDATED: PROGRESS,
} as const satisfies Record<string, readonly LiveResource[]>;

export type KnownExecutionEventType = keyof typeof EVENT_RESOURCE_INVALIDATIONS;

export function resourcesForExecutionEvent(event: Pick<ExecutionEventV2, "event_type"> | string): readonly LiveResource[] {
  const eventType = typeof event === "string" ? event : event.event_type;
  return Object.prototype.hasOwnProperty.call(EVENT_RESOURCE_INVALIDATIONS, eventType)
    ? EVENT_RESOURCE_INVALIDATIONS[eventType as KnownExecutionEventType]
    : NONE;
}

/**
 * Legacy workers can advance job_progress through metric reports without a
 * durable epoch event. A compact replacement snapshot turns only that bounded
 * progress-boundary change into the same metrics invalidation used by v2.
 */
export function resourcesForLiveStateReplacement(
  previous: ProjectLiveState | null,
  next: ProjectLiveState,
): readonly LiveResource[] {
  if (!previous || previous.project_id !== next.project_id) return NONE;
  return liveProgressBoundaryIdentity(previous) === liveProgressBoundaryIdentity(next) ? NONE : PROGRESS;
}

export const invalidatedResourcesForEvent = resourcesForExecutionEvent;

export type InvalidationExecutionContext = {
  signal: AbortSignal;
};

export type InvalidationExecutor = (
  resource: LiveResource,
  context: InvalidationExecutionContext,
) => Promise<unknown> | unknown;

export type InvalidationScheduler = (task: () => void) => void | (() => void);

export type InvalidationCoalescerOptions = {
  execute: InvalidationExecutor;
  schedule?: InvalidationScheduler;
  onError?: (error: unknown, resource: LiveResource) => void;
  /** Bounded automatic follow-ups after a transient executor failure. */
  maxRetries?: number;
};

export type InvalidationCoalescerSnapshot = {
  disposed: boolean;
  scheduled: boolean;
  queued: readonly LiveResource[];
  in_flight: readonly LiveResource[];
  follow_up: readonly LiveResource[];
};

export type InvalidationCoalescer = {
  invalidate(resources: Iterable<LiveResource>): void;
  invalidateEvent(event: Pick<ExecutionEventV2, "event_type"> | string): void;
  flush(): Promise<void>;
  dispose(reason?: unknown): void;
  snapshot(): InvalidationCoalescerSnapshot;
};

type ResourceWork = {
  queued: boolean;
  inFlight: boolean;
  followUp: boolean;
  retryCount: number;
  controller: AbortController | null;
};

/**
 * Coalesces by resource. A burst before execution starts becomes one request;
 * any number of invalidations during that request become at most one follow-up.
 */
export function createInvalidationCoalescer(options: InvalidationCoalescerOptions): InvalidationCoalescer {
  const schedule = options.schedule ?? ((task: () => void) => queueMicrotask(task));
  const maxRetries = Math.max(0, Math.min(3, Math.trunc(options.maxRetries ?? 0)));
  const work = new Map<LiveResource, ResourceWork>();
  const idleWaiters = new Set<() => void>();
  let disposed = false;
  let scheduled = false;
  let cancelScheduled: (() => void) | null = null;

  const stateFor = (resource: LiveResource): ResourceWork => {
    let state = work.get(resource);
    if (!state) {
      state = { queued: false, inFlight: false, followUp: false, retryCount: 0, controller: null };
      work.set(resource, state);
    }
    return state;
  };

  const hasWork = () => scheduled || Array.from(work.values()).some((state) => state.queued || state.inFlight || state.followUp);

  const settleIdle = () => {
    if (hasWork()) return;
    for (const resolve of idleWaiters) resolve();
    idleWaiters.clear();
  };

  const start = (resource: LiveResource, state: ResourceWork) => {
    if (disposed || state.inFlight || !state.queued) return;
    state.queued = false;
    state.inFlight = true;
    const controller = new AbortController();
    state.controller = controller;
    let failed = false;
    Promise.resolve()
      .then(() => options.execute(resource, { signal: controller.signal }))
      .catch((error: unknown) => {
        failed = true;
        if (!disposed && !controller.signal.aborted) {
          options.onError?.(error, resource);
          if (state.retryCount < maxRetries) {
            state.retryCount += 1;
            state.followUp = true;
          }
        }
      })
      .finally(() => {
        state.inFlight = false;
        state.controller = null;
        if (disposed) {
          state.queued = false;
          state.followUp = false;
          settleIdle();
          return;
        }
        if (!failed) state.retryCount = 0;
        if (state.followUp) {
          state.followUp = false;
          state.queued = true;
          scheduleDrain();
        } else if (failed) {
          // A later event receives a fresh bounded retry allowance.
          state.retryCount = 0;
        }
        settleIdle();
      });
  };

  const drain = () => {
    scheduled = false;
    cancelScheduled = null;
    if (disposed) {
      settleIdle();
      return;
    }
    for (const [resource, state] of work) start(resource, state);
    settleIdle();
  };

  function scheduleDrain() {
    if (disposed || scheduled) return;
    scheduled = true;
    const cancel = schedule(drain);
    cancelScheduled = typeof cancel === "function" ? cancel : null;
  }

  const invalidate = (resources: Iterable<LiveResource>) => {
    if (disposed) return;
    let changed = false;
    for (const resource of new Set(resources)) {
      const state = stateFor(resource);
      if (state.inFlight) {
        if (!state.followUp) {
          state.followUp = true;
          changed = true;
        }
      } else if (!state.queued) {
        state.queued = true;
        changed = true;
      }
    }
    if (changed) scheduleDrain();
  };

  const flush = (): Promise<void> => {
    if (scheduled) {
      cancelScheduled?.();
      drain();
    } else {
      drain();
    }
    if (!hasWork()) return Promise.resolve();
    return new Promise<void>((resolve) => idleWaiters.add(resolve));
  };

  const dispose = (reason?: unknown) => {
    if (disposed) return;
    disposed = true;
    cancelScheduled?.();
    cancelScheduled = null;
    scheduled = false;
    for (const state of work.values()) {
      state.queued = false;
      state.followUp = false;
      state.controller?.abort(reason);
    }
    settleIdle();
  };

  const snapshot = (): InvalidationCoalescerSnapshot => ({
    disposed,
    scheduled,
    queued: resourcesMatching(work, (state) => state.queued),
    in_flight: resourcesMatching(work, (state) => state.inFlight),
    follow_up: resourcesMatching(work, (state) => state.followUp),
  });

  return {
    invalidate,
    invalidateEvent: (event) => invalidate(resourcesForExecutionEvent(event)),
    flush,
    dispose,
    snapshot,
  };
}

function resourcesMatching(
  work: ReadonlyMap<LiveResource, ResourceWork>,
  predicate: (state: ResourceWork) => boolean,
): LiveResource[] {
  return Array.from(work.entries())
    .filter(([, state]) => predicate(state))
    .map(([resource]) => resource)
    .sort();
}

function liveProgressBoundaryIdentity(snapshot: ProjectLiveState): string {
  return snapshot.active_progress
    .map((progress) => [
      progress.job_id,
      progress.attempt,
      progress.revision,
      progress.stage,
      progress.status,
      progress.current ?? "",
      progress.total ?? "",
    ].join(":"))
    .join("|");
}
