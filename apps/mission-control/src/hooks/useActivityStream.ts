export type ActivityStreamState = "idle" | "connecting" | "connected" | "reconnecting" | "fallback";

const slowProjectRefreshEventTypes = new Set([
  "AGENT_RECOMMENDATION_RECORDED",
  "AGENT_OUTCOME_RECORDED",
  "AGENT_FAILED",
  "CHAMPION_SELECTED",
  "DATASET_VISUAL_ANALYSIS_RESULT",
  "MEMORY_RETRIEVAL_LOGGED",
]);

const slowActivityRefreshTypes = new Set([
  "agent.failed",
  "agent.outcome_recorded",
  "champion.selected",
  "planner.blocked",
  "planner.decision_recorded",
  "planner.stopped",
  "planner.validation_failed",
  "planner.validation_rejected",
  "memory.retrieval_logged",
]);

export function eventNeedsSlowProjectRefresh(event: MessageEvent | Event): boolean {
  if (!(event instanceof MessageEvent) || !event.data) {
    return false;
  }
  try {
    const parsed = JSON.parse(String(event.data)) as { event_type?: string; type?: string };
    return slowProjectRefreshEventTypes.has(String(parsed.event_type ?? "")) || slowActivityRefreshTypes.has(String(parsed.type ?? ""));
  } catch {
    return false;
  }
}
