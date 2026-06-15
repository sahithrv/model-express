import type { AgentActivityEvent } from "../../types";
import type { ProjectDetail } from "../../hooks/useProjectDetail";

export type DeveloperDiagnostics = {
  counts: Array<{ label: string; value: string }>;
  summary: string;
};

export function buildDeveloperDiagnostics(detail: ProjectDetail, events: AgentActivityEvent[]): DeveloperDiagnostics {
  return {
    summary: "Raw operational detail is preserved here for debugging, audit, and demo backup.",
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
    <section className="developer-intro" id="developer-raw-events">
      <div>
        <div className="eyebrow">Developer View</div>
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
      <button className="command primary compact" type="button" onClick={onBack}>
        Back to Mission
      </button>
    </section>
  );
}
