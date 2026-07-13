import type { MissionControlRequestReason } from "./missionControlClient";

export type ExecutionEventStreamFailure = {
  status: number;
  reasonCode: string;
};

export type ExecutionEventStreamCallbacks = {
  onOpen: () => void;
  onEvent: (event: { eventType: string; lastEventId: string; data: string }) => void;
  onError: (failure: ExecutionEventStreamFailure) => void;
  onDisconnect: (reasonCode: string) => void;
};

export type ExecutionEventStreamHandle = {
  close: () => void;
};

export type OpenExecutionEventStreamOptions = ExecutionEventStreamCallbacks & {
  baseUrl: string;
  projectId: string;
  cursor: number;
  diagnosticReason: Extract<MissionControlRequestReason, "stream_initial" | "stream_reconnect">;
};

let streamSequence = 0;

export function executionEventStreamPath(projectId: string, cursor: number): string {
  if (!projectId.trim()) throw new Error("Project id is required for the execution-event stream.");
  if (!Number.isSafeInteger(cursor) || cursor < 0) throw new Error("Execution-event cursor must be a nonnegative safe integer.");
  return `/projects/${encodeURIComponent(projectId)}/events/stream/v2?cursor=${cursor}&limit=100&interval_ms=1000`;
}

export function openExecutionEventStream(options: OpenExecutionEventStreamOptions): ExecutionEventStreamHandle {
  const path = executionEventStreamPath(options.projectId, options.cursor);
  if (supportsEventStreamRelay()) {
    return openRelayedExecutionEventStream(options, path);
  }
  return openBrowserExecutionEventStream(options, path);
}

function supportsEventStreamRelay(): boolean {
  if (typeof window === "undefined") return false;
  const bridge = window.missionControl as unknown as Record<string, unknown> | undefined;
  return Boolean(
    bridge &&
    typeof bridge.openEventStream === "function" &&
    typeof bridge.closeEventStream === "function" &&
    typeof bridge.onEventStreamMessage === "function",
  );
}

function openRelayedExecutionEventStream(
  options: OpenExecutionEventStreamOptions,
  path: string,
): ExecutionEventStreamHandle {
  const streamId = `livev2_${Date.now().toString(36)}_${(++streamSequence).toString(36)}`;
  let closed = false;
  const unsubscribe = window.missionControl.onEventStreamMessage((message) => {
    if (closed || message.stream_id !== streamId) return;
    switch (message.kind) {
      case "open":
        options.onOpen();
        break;
      case "event":
        options.onEvent({
          eventType: boundedToken(message.event_type),
          lastEventId: boundedCursorText(message.last_event_id),
          data: typeof message.data === "string" ? message.data : "",
        });
        break;
      case "disconnect":
        options.onDisconnect(boundedToken(message.reason_code) || "stream_ended");
        break;
      case "error":
        options.onError({
          status: safeStatus(message.status),
          reasonCode: boundedToken(message.reason_code) || "stream_error",
        });
        break;
    }
  });

  window.missionControl.openEventStream({
    streamId,
    baseUrl: options.baseUrl,
    path,
    diagnosticReason: options.diagnosticReason,
  }).catch(() => {
    if (!closed) options.onError({ status: 0, reasonCode: "relay_open_failed" });
  });

  return {
    close: () => {
      if (closed) return;
      closed = true;
      unsubscribe();
      window.missionControl.closeEventStream(streamId).catch(() => undefined);
    },
  };
}

function openBrowserExecutionEventStream(
  options: OpenExecutionEventStreamOptions,
  path: string,
): ExecutionEventStreamHandle {
  if (typeof EventSource === "undefined") {
    queueMicrotask(() => options.onError({ status: 0, reasonCode: "event_source_unavailable" }));
    return { close: () => undefined };
  }
  const url = new URL(path, options.baseUrl);
  const source = new EventSource(url.toString());
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    source.close();
  };
  const handleEvent = (event: MessageEvent) => {
    if (closed) return;
    options.onEvent({
      eventType: boundedToken(event.type),
      lastEventId: boundedCursorText(event.lastEventId),
      data: typeof event.data === "string" ? event.data : "",
    });
  };
  source.onopen = () => {
    if (!closed) options.onOpen();
  };
  source.addEventListener("execution_event_v2", handleEvent);
  source.addEventListener("stream_error", (event) => {
    let reasonCode = "stream_error";
    if (event instanceof MessageEvent && typeof event.data === "string") {
      try {
        const payload = JSON.parse(event.data) as { reason_code?: unknown };
        reasonCode = boundedToken(payload.reason_code) || reasonCode;
      } catch {
        reasonCode = "malformed_stream_error";
      }
    }
    close();
    options.onError({ status: 0, reasonCode });
  });
  source.onerror = () => {
    if (closed) return;
    close();
    options.onDisconnect("event_source_error");
  };
  return { close };
}

function boundedToken(value: unknown): string {
  const token = typeof value === "string" ? value.trim().toLowerCase() : "";
  return token.length <= 128 && /^[a-z][a-z0-9_.-]*$/.test(token) ? token : "";
}

function boundedCursorText(value: unknown): string {
  const cursor = typeof value === "string" ? value.trim() : "";
  return /^\d{1,19}$/.test(cursor) ? cursor : "";
}

function safeStatus(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 599 ? value : 0;
}
