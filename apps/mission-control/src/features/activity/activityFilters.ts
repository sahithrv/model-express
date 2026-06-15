import type { ActivityFilterKey } from "../mission/workflowTabs";

export const activityFilters: Array<{ key: ActivityFilterKey; label: string }> = [
  { key: "all", label: "All" },
  { key: "decisions", label: "Decisions" },
  { key: "experiments", label: "Experiments" },
  { key: "results", label: "Results" },
  { key: "blockers", label: "Blockers" },
];
