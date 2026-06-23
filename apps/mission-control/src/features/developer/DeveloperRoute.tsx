import { Rocket, SquareTerminal } from "lucide-react";

import type { AgentActivityEvent } from "../../types";
import type { ProjectDetail } from "../../hooks/useProjectDetail";

const developerPlanetAssetUrl = new URL("../../../moon2.png", import.meta.url).href;

export type DeveloperDiagnostics = {
  counts: Array<{ label: string; value: string }>;
  summary: string;
};

export function buildDeveloperDiagnostics(detail: ProjectDetail, events: AgentActivityEvent[]): DeveloperDiagnostics {
  return {
    summary: "Expanded operational detail is preserved here for debugging, audit, and mission backup.",
    counts: [
      { label: "Invocations", value: String(detail.agentInvocations.length) },
      { label: "Memory records", value: String(detail.agentMemory.length) },
      { label: "Validation events", value: String(detail.executionEvents.length) },
      { label: "Workers", value: String(detail.workers.length) },
      { label: "Telemetry", value: detail.telemetry ? "Loaded" : "Empty" },
      { label: "Raw events", value: String(events.length) },
    ],
  };
}

export function DeveloperRoute({ diagnostics, onBack }: { diagnostics: DeveloperDiagnostics; onBack: () => void }) {
  return (
    <div className="developer-route-stack">
      <section className="route-planet-hero developer-planet-hero blue">
        <img className="route-planet-asset" src={developerPlanetAssetUrl} alt="" aria-hidden="true" />
        <div className="route-hero-copy">
          <span className="route-hero-icon" aria-hidden="true">
            <SquareTerminal size={22} />
          </span>
          <div>
            <div className="eyebrow">Project systems</div>
            <h3>In-Depth View</h3>
            <p>Review automation state, telemetry, experiment evidence, and the raw audit trail behind each mission.</p>
          </div>
        </div>
        <div className="route-hero-facts">
          {diagnostics.counts.slice(0, 4).map((item) => (
            <span key={`${item.label}-${item.value}`}>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
        <div className="route-hero-actions">
          <button className="command primary compact" type="button" onClick={onBack}>
            <Rocket size={15} />
            Back to Mission
          </button>
        </div>
      </section>

      <section className="developer-intro developer-intro-panel" id="developer-raw-events">
        <div>
          <div className="eyebrow">In-Depth View</div>
          <h3>Technical audit trail</h3>
          <p>{diagnostics.summary}</p>
        </div>
        <div className="developer-counts">
          {diagnostics.counts.map((item) => (
            <span key={item.label}>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}