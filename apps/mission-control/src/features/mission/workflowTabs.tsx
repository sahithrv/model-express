import type { ReactNode } from "react";
import { Activity, BarChart3, ClipboardList, Trophy } from "lucide-react";

export type ProjectTabKey = "mission" | "activity" | "results" | "export" | "developer";
export type LegacyProjectTabKey = "overview" | "data" | "experiments" | "agents" | "operations";
export type ProjectTabTarget = ProjectTabKey | LegacyProjectTabKey;
export type ActivityFilterKey = "all" | "decisions" | "experiments" | "results" | "blockers";
export type ProjectWorkflowTabState = "done" | "active" | "waiting" | "blocked";
export type ProjectWorkflowTabBase = { key: Exclude<ProjectTabKey, "developer">; label: string; icon: ReactNode };
export type ProjectWorkflowTab = ProjectWorkflowTabBase & { state: ProjectWorkflowTabState; detail: string };

export const projectTabs: ProjectWorkflowTabBase[] = [
  { key: "mission", label: "Mission", icon: <ClipboardList size={14} /> },
  { key: "activity", label: "Activity", icon: <Activity size={14} /> },
  { key: "results", label: "Results", icon: <BarChart3 size={14} /> },
  { key: "export", label: "Test & Export", icon: <Trophy size={14} /> },
];
