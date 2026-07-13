import {
  isCursorRecoveryStatus,
  isUnsupportedIncrementalStatus,
} from "../../api/missionControlClient";
import {
  LiveStateContractError,
  parseExecutionEventV2,
  parseProjectLiveState,
  type ExecutionEventV2,
  type ProjectLiveState,
} from "./liveStateContract";
import {
  applyExecutionEvent,
  createIncrementalLiveState,
  installLiveStateSnapshot,
  type IncrementalLiveState,
} from "./liveStateReducer";

export type IncrementalLiveMode = "off" | "shadow" | "primary" | "rollback";
export type IncrementalConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "recovering"
  | "fallback"
  | "unsupported"
  | "disconnected";

export type IncrementalFallbackReason =
  | "disabled"
  | "rollback"
  | "snapshot_unsupported"
  | "stream_unsupported"
  | "snapshot_failed"
  | "cursor_recovery_failed"
  | "stream_failed"
  | "malformed_snapshot"
  | "malformed_event"
  | "out_of_order";

export type IncrementalSessionState = {
  live: IncrementalLiveState;
  mode: IncrementalLiveMode;
  connection: IncrementalConnectionState;
  supported: boolean | null;
  healthy: boolean;
  fallback_reason: IncrementalFallbackReason | "";
  reconnect_count: number;
};

export type IncrementalStreamCallbacks = {
  onOpen: () => void;
  onEvent: (message: { eventType: string; lastEventId: string; data: string }) => void;
  onError: (failure: { status: number; reasonCode: string }) => void;
  onDisconnect: (reasonCode: string) => void;
};

export type IncrementalStreamHandle = { close: () => void };

export type IncrementalSessionDiagnostic = {
  reason_code:
    | "cursor_recovery"
    | "fallback"
    | "malformed_event"
    | "out_of_order"
    | "rollback"
    | "snapshot"
    | "stream_disconnect";
  outcome_code: "applied" | "connected" | "failed" | "recovered" | "unsupported";
  count?: number;
  duration_ms?: number;
};

export type IncrementalSessionDependencies = {
  fetchSnapshot: (request: {
    projectId: string;
    reason: "v2_snapshot" | "v2_cursor_recovery";
    signal: AbortSignal;
  }) => Promise<unknown>;
  probeStream: (request: {
    projectId: string;
    cursor: number;
    reason: "stream_initial" | "stream_reconnect";
    signal: AbortSignal;
  }) => Promise<void>;
  openStream: (request: {
    projectId: string;
    cursor: number;
    reason: "stream_initial" | "stream_reconnect";
    callbacks: IncrementalStreamCallbacks;
  }) => IncrementalStreamHandle;
  onState: (state: IncrementalSessionState) => void;
  onAppliedEvent: (event: ExecutionEventV2) => void;
  onConnected?: (connection: {
    projectId: string;
    reason: "stream_initial" | "stream_reconnect";
    openedAtMs: number;
  }) => void;
  onDiagnostic?: (diagnostic: IncrementalSessionDiagnostic) => void;
  setTimer?: (callback: () => void, delayMs: number) => unknown;
  clearTimer?: (timer: unknown) => void;
  now?: () => number;
  reconnectBaseDelayMs?: number;
  compactSnapshotIntervalMs?: number;
  maxReconnectAttempts?: number;
};

const defaultReconnectBaseDelayMs = 1_000;
const defaultCompactSnapshotIntervalMs = 30_000;
const defaultMaxReconnectAttempts = 5;
const maxCursorRecoveryAttempts = 2;

export class IncrementalLiveSessionController {
  private readonly deps: Required<Pick<IncrementalSessionDependencies,
    "setTimer" | "clearTimer" | "now" | "reconnectBaseDelayMs" | "compactSnapshotIntervalMs" | "maxReconnectAttempts">>
    & Omit<IncrementalSessionDependencies,
      "setTimer" | "clearTimer" | "now" | "reconnectBaseDelayMs" | "compactSnapshotIntervalMs" | "maxReconnectAttempts">;
  private state: IncrementalSessionState = initialSessionState("", "off");
  private generation = 0;
  private requestController: AbortController | null = null;
  private stream: IncrementalStreamHandle | null = null;
  private reconnectTimer: unknown = null;
  private compactSnapshotTimer: unknown = null;
  private reconnectAttempts = 0;
  private cursorRecoveryAttempts = 0;

  constructor(dependencies: IncrementalSessionDependencies) {
    this.deps = {
      ...dependencies,
      setTimer: dependencies.setTimer ?? ((callback, delayMs) => globalThis.setTimeout(callback, delayMs)),
      clearTimer: dependencies.clearTimer ?? ((timer) => globalThis.clearTimeout(timer as ReturnType<typeof setTimeout>)),
      now: dependencies.now ?? Date.now,
      reconnectBaseDelayMs: dependencies.reconnectBaseDelayMs ?? defaultReconnectBaseDelayMs,
      compactSnapshotIntervalMs: dependencies.compactSnapshotIntervalMs ?? defaultCompactSnapshotIntervalMs,
      maxReconnectAttempts: dependencies.maxReconnectAttempts ?? defaultMaxReconnectAttempts,
    };
  }

  start(projectId: string, mode: IncrementalLiveMode): void {
    this.disposeActiveWork();
    this.generation += 1;
    this.reconnectAttempts = 0;
    this.cursorRecoveryAttempts = 0;
    this.state = initialSessionState(projectId, mode);
    if (!projectId || mode === "off") {
      this.state = { ...this.state, connection: "idle", fallback_reason: "disabled" };
      this.emit();
      return;
    }
    if (mode === "rollback") {
      this.state = { ...this.state, connection: "fallback", fallback_reason: "rollback" };
      this.emit();
      this.diagnostic({ reason_code: "rollback", outcome_code: "applied", count: 1 });
      return;
    }
    this.emit();
    void this.loadSnapshot("initial");
  }

  stop(): void {
    this.disposeActiveWork();
    this.generation += 1;
    this.state = initialSessionState("", "off");
    this.emit();
  }

  current(): IncrementalSessionState {
    return this.state;
  }

  async manualResync(): Promise<void> {
    if (!this.state.live.project_id || this.state.mode === "off" || this.state.mode === "rollback") return;
    await this.loadSnapshot(this.state.healthy ? "compact" : "manual");
  }

  private async loadSnapshot(reason: "initial" | "manual" | "cursor_recovery" | "compact"): Promise<void> {
    const projectId = this.state.live.project_id;
    if (!projectId) return;
    const generation = this.generation;
    const startedAt = this.deps.now();
    const recovering = reason === "cursor_recovery";
    this.abortRequest();
    const controller = new AbortController();
    this.requestController = controller;
    if (reason !== "compact") {
      this.state = {
        ...this.state,
        healthy: false,
        connection: recovering ? "recovering" : "connecting",
      };
      this.emit();
    }
    try {
      const raw = await this.deps.fetchSnapshot({
        projectId,
        reason: recovering ? "v2_cursor_recovery" : "v2_snapshot",
        signal: controller.signal,
      });
      if (!this.isCurrent(generation, projectId) || controller.signal.aborted) return;
      const snapshot = parseProjectLiveState(raw, projectId);
      const cursorBeforeInstall = this.state.live.cursor;
      const nextLive = installLiveStateSnapshot(this.state.live, snapshot);
      if (nextLive.last_transition === "older_snapshot") {
        this.scheduleCompactSnapshot();
        return;
      }
      this.state = {
        ...this.state,
        live: nextLive,
        supported: true,
        fallback_reason: "",
      };
      this.emit();
      this.diagnostic({
        reason_code: recovering ? "cursor_recovery" : "snapshot",
        outcome_code: recovering ? "recovered" : "applied",
        count: 1,
        duration_ms: this.deps.now() - startedAt,
      });
      if (reason === "compact" && nextLive.cursor === cursorBeforeInstall) {
        // Snapshot-only heartbeats/staleness can refresh in place. Preserve the
        // already cursor-safe stream unless the compact snapshot advances it.
        this.scheduleCompactSnapshot();
        return;
      }
      await this.probeAndOpen(reason === "initial" ? "stream_initial" : "stream_reconnect", generation);
    } catch (error) {
      if (!this.isCurrent(generation, projectId) || controller.signal.aborted) return;
      const status = errorStatus(error);
      if (isUnsupportedIncrementalStatus(status)) {
        this.enterFallback("snapshot_unsupported", true);
      } else if (error instanceof LiveStateContractError) {
        this.enterFallback("malformed_snapshot", false);
      } else {
        this.enterFallback(recovering ? "cursor_recovery_failed" : "snapshot_failed", false);
      }
    } finally {
      if (this.requestController === controller) this.requestController = null;
    }
  }

  private async probeAndOpen(
    reason: "stream_initial" | "stream_reconnect",
    generation = this.generation,
  ): Promise<void> {
    const projectId = this.state.live.project_id;
    const cursor = this.state.live.cursor;
    if (!projectId || !this.state.live.snapshot_installed) return;
    this.abortRequest();
    const controller = new AbortController();
    this.requestController = controller;
    try {
      await this.deps.probeStream({ projectId, cursor, reason, signal: controller.signal });
      if (!this.isCurrent(generation, projectId) || controller.signal.aborted) return;
      this.openStream(cursor, reason, generation);
    } catch (error) {
      if (!this.isCurrent(generation, projectId) || controller.signal.aborted) return;
      const status = errorStatus(error);
      if (isUnsupportedIncrementalStatus(status)) {
        this.enterFallback("stream_unsupported", true);
      } else if (isCursorRecoveryStatus(status)) {
        this.diagnostic({ reason_code: "cursor_recovery", outcome_code: "failed", count: 1 });
        if (!this.allowCursorRecovery()) return;
        await this.loadSnapshot("cursor_recovery");
      } else {
        this.scheduleReconnect();
      }
    } finally {
      if (this.requestController === controller) this.requestController = null;
    }
  }

  private openStream(
    cursor: number,
    reason: "stream_initial" | "stream_reconnect",
    generation: number,
  ): void {
    const projectId = this.state.live.project_id;
    if (!this.isCurrent(generation, projectId) || cursor < this.state.live.cursor) return;
    this.closeStream();
    this.state = { ...this.state, healthy: false, connection: reason === "stream_initial" ? "connecting" : "reconnecting" };
    this.emit();
    try {
      this.stream = this.deps.openStream({
        projectId,
        cursor,
        reason,
        callbacks: {
          onOpen: () => {
            if (!this.isCurrent(generation, projectId)) return;
            this.reconnectAttempts = 0;
            this.state = {
              ...this.state,
              connection: "connected",
              supported: true,
              healthy: true,
              fallback_reason: "",
            };
            this.deps.onConnected?.({ projectId, reason, openedAtMs: this.deps.now() });
            this.emit();
            this.diagnostic({ reason_code: "snapshot", outcome_code: "connected", count: 1 });
            this.scheduleCompactSnapshot();
          },
          onEvent: (message) => this.handleStreamEvent(message, generation, projectId),
          onError: (failure) => this.handleStreamFailure(failure.status, failure.reasonCode, generation, projectId),
          onDisconnect: (reasonCode) => this.handleStreamFailure(0, reasonCode, generation, projectId),
        },
      });
    } catch {
      this.handleStreamFailure(0, "stream_open_failed", generation, projectId);
    }
  }

  private handleStreamEvent(
    message: { eventType: string; lastEventId: string; data: string },
    generation: number,
    projectId: string,
  ): void {
    if (!this.isCurrent(generation, projectId)) return;
    if (message.eventType === "stream_error") {
      this.handleStreamFailure(0, "execution_events_read_failed", generation, projectId);
      return;
    }
    if (message.eventType !== "execution_event_v2") {
      this.recoverMalformedEvent();
      return;
    }
    let event: ExecutionEventV2;
    try {
      event = parseExecutionEventV2(JSON.parse(message.data), projectId);
    } catch {
      this.recoverMalformedEvent();
      return;
    }
    if (!message.lastEventId || message.lastEventId !== String(event.sequence)) {
      this.recoverMalformedEvent();
      return;
    }
    const previousCursor = this.state.live.cursor;
    const nextLive = applyExecutionEvent(this.state.live, event);
    this.state = { ...this.state, live: nextLive };
    this.emit();
    if (nextLive.last_transition === "out_of_order" || nextLive.last_transition === "project_mismatch") {
      this.diagnostic({ reason_code: "out_of_order", outcome_code: "failed", count: 1 });
      this.closeStream();
      if (!this.allowCursorRecovery()) return;
      void this.loadSnapshot("cursor_recovery");
      return;
    }
    if (nextLive.cursor <= previousCursor) return;
    this.cursorRecoveryAttempts = 0;
    if (["duplicate_event", "duplicate_idempotency_key"].includes(nextLive.last_transition)) return;
    this.deps.onAppliedEvent(event);
  }

  private recoverMalformedEvent(): void {
    this.diagnostic({ reason_code: "malformed_event", outcome_code: "failed", count: 1 });
    this.closeStream();
    if (!this.allowCursorRecovery()) return;
    this.state = { ...this.state, healthy: false, connection: "recovering", fallback_reason: "malformed_event" };
    this.emit();
    void this.loadSnapshot("cursor_recovery");
  }

  private handleStreamFailure(status: number, _reasonCode: string, generation: number, projectId: string): void {
    if (!this.isCurrent(generation, projectId)) return;
    this.closeStream();
    this.clearCompactSnapshotTimer();
    this.state = { ...this.state, healthy: false, connection: "disconnected" };
    this.emit();
    this.diagnostic({ reason_code: "stream_disconnect", outcome_code: "failed", count: 1 });
    if (isUnsupportedIncrementalStatus(status)) {
      this.enterFallback("stream_unsupported", true);
      return;
    }
    if (isCursorRecoveryStatus(status)) {
      if (!this.allowCursorRecovery()) return;
      void this.loadSnapshot("cursor_recovery");
      return;
    }
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    this.reconnectAttempts += 1;
    if (this.reconnectAttempts > this.deps.maxReconnectAttempts) {
      this.enterFallback("stream_failed", false);
      return;
    }
    this.state = {
      ...this.state,
      healthy: false,
      connection: "reconnecting",
      reconnect_count: this.state.reconnect_count + 1,
    };
    this.emit();
    const generation = this.generation;
    const projectId = this.state.live.project_id;
    const delay = Math.min(10_000, this.deps.reconnectBaseDelayMs * 2 ** Math.max(0, this.reconnectAttempts - 1));
    this.reconnectTimer = this.deps.setTimer(() => {
      this.reconnectTimer = null;
      if (!this.isCurrent(generation, projectId)) return;
      void this.probeAndOpen("stream_reconnect", generation);
    }, delay);
  }

  private allowCursorRecovery(): boolean {
    this.cursorRecoveryAttempts += 1;
    if (this.cursorRecoveryAttempts <= maxCursorRecoveryAttempts) return true;
    this.enterFallback("cursor_recovery_failed", false);
    return false;
  }

  private scheduleCompactSnapshot(): void {
    this.clearCompactSnapshotTimer();
    const snapshot = this.state.live.snapshot;
    if (!this.state.healthy || !snapshot || !snapshotHasOpenWork(snapshot)) return;
    const generation = this.generation;
    const projectId = this.state.live.project_id;
    this.compactSnapshotTimer = this.deps.setTimer(() => {
      this.compactSnapshotTimer = null;
      if (!this.isCurrent(generation, projectId)) return;
      void this.loadSnapshot("compact");
    }, this.deps.compactSnapshotIntervalMs);
  }

  private enterFallback(reason: IncrementalFallbackReason, unsupported: boolean): void {
    this.disposeActiveWork();
    this.state = {
      ...this.state,
      healthy: false,
      supported: unsupported ? false : this.state.supported,
      connection: unsupported ? "unsupported" : "fallback",
      fallback_reason: reason,
    };
    this.emit();
    this.diagnostic({
      reason_code: "fallback",
      outcome_code: unsupported ? "unsupported" : "failed",
      count: 1,
    });
  }

  private disposeActiveWork(): void {
    this.abortRequest();
    this.closeStream();
    this.clearReconnectTimer();
    this.clearCompactSnapshotTimer();
  }

  private abortRequest(): void {
    this.requestController?.abort();
    this.requestController = null;
  }

  private closeStream(): void {
    this.stream?.close();
    this.stream = null;
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer === null) return;
    this.deps.clearTimer(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  private clearCompactSnapshotTimer(): void {
    if (this.compactSnapshotTimer === null) return;
    this.deps.clearTimer(this.compactSnapshotTimer);
    this.compactSnapshotTimer = null;
  }

  private isCurrent(generation: number, projectId: string): boolean {
    return generation === this.generation && projectId === this.state.live.project_id;
  }

  private emit(): void {
    this.deps.onState(this.state);
  }

  private diagnostic(diagnostic: IncrementalSessionDiagnostic): void {
    this.deps.onDiagnostic?.(diagnostic);
  }
}

export function initialSessionState(projectId: string, mode: IncrementalLiveMode): IncrementalSessionState {
  return {
    live: createIncrementalLiveState(projectId),
    mode,
    connection: mode === "off" ? "idle" : "connecting",
    supported: null,
    healthy: false,
    fallback_reason: "",
    reconnect_count: 0,
  };
}

export function snapshotHasOpenWork(snapshot: ProjectLiveState): boolean {
  return snapshot.jobs.queued + snapshot.jobs.retrying + snapshot.jobs.assigned + snapshot.jobs.running > 0;
}

export function errorStatus(error: unknown): number {
  if (!error || typeof error !== "object") return 0;
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" && Number.isInteger(status) && status >= 0 && status <= 599 ? status : 0;
}
