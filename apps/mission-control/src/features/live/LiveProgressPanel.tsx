import { useEffect, useState } from "react";

import type { ProjectLiveState } from "./liveStateContract";
import {
  buildLiveProgressViewModel,
  type LiveProgressConnectionState,
  type LiveProgressMode,
} from "./liveProgressViewModel";

export type LiveProgressPanelProps = {
  snapshot: ProjectLiveState | null;
  connectionState: LiveProgressConnectionState;
  mode: LiveProgressMode;
  nowMs?: number;
};

export function LiveProgressPanel({
  snapshot,
  connectionState,
  mode,
  nowMs,
}: LiveProgressPanelProps) {
  const currentTimeMs = useLiveProgressClock(nowMs);
  const view = buildLiveProgressViewModel({ snapshot, connectionState, mode, nowMs: currentTimeMs });
  const panelClasses = [
    "live-progress-panel",
    `state-${view.state}`,
    `connection-${view.connectionTone}`,
    view.mixed ? "mixed" : "",
    view.terminalStage ? `terminal-${view.terminalStage}` : "",
  ].filter(Boolean).join(" ");

  return (
    <section
      className={panelClasses}
      data-live-progress-state={view.state}
      data-live-connection={connectionState}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-label={view.ariaLabel}
    >
      <div className="live-progress-summary">
        <span className="live-progress-state-mark" aria-hidden="true" />
        <div className="live-progress-copy">
          <small>Live progress</small>
          <strong>{view.stageLabel}</strong>
          <span>{view.detail}</span>
        </div>
      </div>

      <div className="live-progress-badges" aria-label="Progress and connection state">
        <span className={`live-progress-state-badge state-${view.state}`}>{view.stateLabel}</span>
        {view.mixed && <span className="live-progress-state-badge mixed">Mixed jobs</span>}
        <span className={`live-progress-connection-badge ${view.connectionTone}`}>{view.connectionLabel}</span>
        <span className="live-progress-mode-badge">{view.modeLabel}</span>
        {view.taxonomyLabel && <span className="live-progress-taxonomy-badge">{view.taxonomyLabel}</span>}
      </div>

      {view.epochLabel && (
        <div className="live-progress-epoch" aria-label={view.epochLabel}>
          <span>{view.epochLabel}</span>
        </div>
      )}

      {view.facts.length > 0 && (
        <dl className="live-progress-facts">
          {view.facts.map((fact) => (
            <div key={fact.label} className={fact.label === "Stale state" ? "stale-reason" : fact.label === "Blocked reason" ? "blocked-reason" : ""}>
              <dt>{fact.label}</dt>
              <dd>
                {fact.dateTime ? <time dateTime={fact.dateTime}>{fact.value}</time> : fact.value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

function useLiveProgressClock(fixedNowMs?: number): number {
  const [clockMs, setClockMs] = useState(() => fixedNowMs ?? Date.now());

  useEffect(() => {
    if (fixedNowMs !== undefined) {
      setClockMs(fixedNowMs);
      return;
    }
    const timer = window.setInterval(() => setClockMs(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [fixedNowMs]);

  return fixedNowMs ?? clockMs;
}
