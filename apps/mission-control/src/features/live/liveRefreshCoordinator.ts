export type LiveRefreshCoordinatorSnapshot = {
  broad_active: boolean;
  targeted_active: number;
};

export type LiveRefreshCoordinator = {
  runBroad<T>(scope: string, task: () => Promise<T>): Promise<T>;
  runTargeted<T>(scope: string, signal: AbortSignal, task: (signal: AbortSignal) => Promise<T>): Promise<T>;
  snapshot(scope: string): LiveRefreshCoordinatorSnapshot;
};

type TargetedWork = {
  controller: AbortController;
  completion: Promise<void>;
};

type ScopeState = {
  broad: Promise<void> | null;
  targeted: Set<TargetedWork>;
};

/**
 * Keeps legacy/manual project loads and v2 invalidation fetches from issuing
 * overlapping reads. A broad reservation preempts already-running targeted
 * work and waits for its cancellation; targeted work arriving after the
 * reservation waits for the broad load.
 */
export function createLiveRefreshCoordinator(): LiveRefreshCoordinator {
  const scopes = new Map<string, ScopeState>();

  const stateFor = (scope: string): ScopeState => {
    const key = normalizeScope(scope);
    let state = scopes.get(key);
    if (!state) {
      state = { broad: null, targeted: new Set() };
      scopes.set(key, state);
    }
    return state;
  };

  const releaseIfIdle = (scope: string, state: ScopeState) => {
    if (!state.broad && state.targeted.size === 0) scopes.delete(scope);
  };

  const runBroad = async <T,>(scopeValue: string, task: () => Promise<T>): Promise<T> => {
    const scope = normalizeScope(scopeValue);
    const state = stateFor(scope);
    const priorBroad = state.broad;
    const priorTargeted = [...state.targeted];
    let releaseReservation: () => void = () => undefined;
    const reservation = new Promise<void>((resolve) => {
      releaseReservation = resolve;
    });
    const broadGate = priorBroad
      ? priorBroad.catch(() => undefined).then(() => reservation)
      : reservation;
    state.broad = broadGate;
    for (const pending of priorTargeted) {
      pending.controller.abort(new DOMException("Preempted by a broad refresh.", "AbortError"));
    }
    try {
      await Promise.all([
        priorBroad?.catch(() => undefined),
        ...priorTargeted.map((pending) => pending.completion),
      ]);
      return await task();
    } finally {
      releaseReservation();
      if (state.broad === broadGate) state.broad = null;
      releaseIfIdle(scope, state);
    }
  };

  const runTargeted = async <T,>(
    scopeValue: string,
    signal: AbortSignal,
    task: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    const scope = normalizeScope(scopeValue);
    let state = stateFor(scope);
    while (state.broad) {
      await waitForGate(state.broad, signal);
      throwIfAborted(signal);
      state = stateFor(scope);
    }
    throwIfAborted(signal);

    const controller = new AbortController();
    const relayCallerAbort = () => controller.abort(signal.reason);
    signal.addEventListener("abort", relayCallerAbort, { once: true });
    if (signal.aborted) relayCallerAbort();

    let releaseCompletion: () => void = () => undefined;
    const completion = new Promise<void>((resolve) => {
      releaseCompletion = resolve;
    });
    const work: TargetedWork = { controller, completion };
    state.targeted.add(work);
    try {
      throwIfAborted(controller.signal);
      return await task(controller.signal);
    } finally {
      signal.removeEventListener("abort", relayCallerAbort);
      releaseCompletion();
      state.targeted.delete(work);
      releaseIfIdle(scope, state);
    }
  };

  return {
    runBroad,
    runTargeted,
    snapshot(scopeValue) {
      const state = scopes.get(normalizeScope(scopeValue));
      return {
        broad_active: Boolean(state?.broad),
        targeted_active: state?.targeted.size ?? 0,
      };
    },
  };
}

export function projectLiveRefreshScope(projectId: string): string {
  return `project:${normalizeScope(projectId)}`;
}

function normalizeScope(scope: string): string {
  const value = String(scope ?? "").trim();
  if (!value || value.length > 240) throw new TypeError("Live refresh scope must be a bounded identifier.");
  return value;
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException("Request aborted.", "AbortError");
}

function waitForGate(gate: Promise<void>, signal: AbortSignal): Promise<void> {
  throwIfAborted(signal);
  return new Promise<void>((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      reject(new DOMException("Request aborted.", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    gate.then(
      () => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        resolve();
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}
