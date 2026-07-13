import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { openExecutionEventStream } from "../api/executionEventStream";
import {
  liveRequestPath,
  type MissionControlRequestReason,
  type RequestOptions,
} from "../api/missionControlClient";
import {
  createInvalidationCoalescer,
  resourcesForExecutionEvent,
  resourcesForLiveStateReplacement,
  type LiveResource,
} from "../features/live/liveInvalidations";
import {
  IncrementalLiveSessionController,
  initialSessionState,
  type IncrementalLiveMode,
  type IncrementalSessionState,
} from "../features/live/liveSessionController";
import type { ExecutionEventV2, ProjectLiveState } from "../features/live/liveStateContract";
import type { LiveDataFeatureFlags } from "../features/live/livePollingPolicy";

export type IncrementalRequest = <T>(path: string, options?: RequestOptions) => Promise<T>;

export type IncrementalResourceFetcher = (
  projectId: string,
  resource: Exclude<LiveResource, "live_state">,
  signal: AbortSignal,
) => Promise<unknown>;

export type UseIncrementalLiveStateOptions = {
  baseUrl: string;
  projectId: string;
  flags: LiveDataFeatureFlags;
  request: IncrementalRequest;
  fetchResource: IncrementalResourceFetcher;
  onAppliedEvent?: (event: ExecutionEventV2) => void;
  onStreamOpen?: (connection: {
    projectId: string;
    reason: "stream_initial" | "stream_reconnect";
    openedAtMs: number;
  }) => void;
};

export type IncrementalLiveStateResult = {
  session: IncrementalSessionState;
  resync: () => Promise<void>;
};

export function useIncrementalLiveState({
  baseUrl,
  projectId,
  flags,
  request,
  fetchResource,
  onAppliedEvent,
  onStreamOpen,
}: UseIncrementalLiveStateOptions): IncrementalLiveStateResult {
  const mode = incrementalModeFromFlags(flags);
  const [session, setSession] = useState<IncrementalSessionState>(() => initialSessionState(projectId, mode));
  const controllerRef = useRef<IncrementalLiveSessionController | null>(null);

  useEffect(() => {
    let controller: IncrementalLiveSessionController;
    let lastSnapshot: ProjectLiveState | null = null;
    const coalescer = createInvalidationCoalescer({
      execute: async (resource, context) => {
        if (resource === "live_state") {
          await controller.manualResync();
          return;
        }
        await fetchResource(projectId, resource, context.signal);
      },
      onError: () => {
        recordIncrementalDiagnostic({
          reason_code: "targeted_invalidation",
          outcome_code: "failed",
          count: 1,
        });
      },
      maxRetries: 1,
    });

    controller = new IncrementalLiveSessionController({
      fetchSnapshot: async ({ projectId: requestedProjectId, reason, signal }) => request<unknown>(
        liveRequestPath("liveState", { projectId: requestedProjectId }),
        { bypassCache: true, diagnosticReason: reason, signal },
      ),
      probeStream: async ({ projectId: requestedProjectId, cursor, reason, signal }) => {
        await request<unknown>(
          `${liveRequestPath("executionEventStreamV2", { projectId: requestedProjectId })}?cursor=${cursor}`,
          { method: "HEAD", diagnosticReason: reason, signal },
        );
      },
      openStream: ({ projectId: requestedProjectId, cursor, reason, callbacks }) => openExecutionEventStream({
        baseUrl,
        projectId: requestedProjectId,
        cursor,
        diagnosticReason: reason,
        ...callbacks,
      }),
      onState: (nextSession) => {
        const snapshot = nextSession.live.snapshot;
        if (snapshot) {
          const replacementResources = resourcesForLiveStateReplacement(lastSnapshot, snapshot);
          if (mode === "primary" && replacementResources.length > 0) {
            // Old workers can advance job_progress through legacy metric reports
            // without a durable epoch event. Compact snapshots retain metrics
            // compatibility while the same resource coalescer deduplicates
            // modern progress events.
            coalescer.invalidate(replacementResources);
            recordIncrementalDiagnostic({
              reason_code: "targeted_invalidation",
              outcome_code: "scheduled",
              count: replacementResources.length,
            });
          }
          lastSnapshot = snapshot;
        }
        setSession(nextSession);
      },
      onConnected: onStreamOpen,
      onAppliedEvent: (event) => {
        const resources = resourcesForExecutionEvent(event);
        // Shadow mode may compact its internal operational snapshot, but must
        // never mutate a visible legacy resource through targeted fetches.
        if (mode === "shadow") {
          if (resources.includes("live_state")) coalescer.invalidate(["live_state"]);
          return;
        }
        if (mode !== "primary") return;
        onAppliedEvent?.(event);
        if (resources.length === 0) return;
        coalescer.invalidate(resources);
        recordIncrementalDiagnostic({
          reason_code: "targeted_invalidation",
          outcome_code: "scheduled",
          count: resources.length,
        });
      },
      onDiagnostic: recordIncrementalDiagnostic,
    });
    controllerRef.current = controller;
    controller.start(projectId, mode);

    return () => {
      coalescer.dispose("incremental_session_disposed");
      controller.stop();
      if (controllerRef.current === controller) controllerRef.current = null;
    };
  }, [baseUrl, fetchResource, mode, onAppliedEvent, onStreamOpen, projectId, request]);

  const visibleSession = useMemo(
    () => session.live.project_id === projectId && session.mode === mode
      ? session
      : initialSessionState(projectId, mode),
    [mode, projectId, session],
  );
  const resync = useCallback(async () => {
    await controllerRef.current?.manualResync();
  }, []);
  return { session: visibleSession, resync };
}

export function incrementalModeFromFlags(flags: LiveDataFeatureFlags): IncrementalLiveMode {
  if (flags.rollback_to_legacy) return "rollback";
  if (!flags.incremental_v2_enabled) return "off";
  return flags.shadow_mode ? "shadow" : "primary";
}

function recordIncrementalDiagnostic(diagnostic: {
  reason_code:
    | "cursor_recovery"
    | "fallback"
    | "malformed_event"
    | "out_of_order"
    | "rollback"
    | "shadow_compare"
    | "snapshot"
    | "stream_disconnect"
    | "targeted_invalidation";
  outcome_code:
    | "applied"
    | "connected"
    | "failed"
    | "matched"
    | "mismatched"
    | "recovered"
    | "scheduled"
    | "unsupported";
  count?: number;
  duration_ms?: number;
}): void {
  if (typeof window === "undefined" || !window.missionControl?.recordIncrementalLiveDiagnostic) return;
  window.missionControl.recordIncrementalLiveDiagnostic(diagnostic).catch(() => undefined);
}

export function requestReasonForFallback(rollback: boolean): MissionControlRequestReason {
  return rollback ? "rollback_poll" : "fallback_poll";
}
