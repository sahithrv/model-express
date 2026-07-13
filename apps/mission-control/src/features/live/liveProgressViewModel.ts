import type {
  ActiveAttemptProgress,
  ProjectLiveState,
  StableProgressStage,
} from "./liveStateContract";

export type LiveProgressMode = "legacy" | "shadow" | "primary";

export type LiveProgressConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "recovering"
  | "fallback"
  | "unsupported"
  | "disconnected";

export type LiveProgressPresentationState =
  | "idle"
  | "queued"
  | "active"
  | "retrying"
  | "blocked"
  | "stale"
  | "terminal";

type ActiveProgress = ActiveAttemptProgress;
export type LiveProgressStage = StableProgressStage;

export type LiveProgressFact = {
  label: string;
  value: string;
  dateTime?: string;
};

export type LiveProgressViewModel = {
  state: LiveProgressPresentationState;
  stateLabel: string;
  stage: LiveProgressStage | "";
  stageLabel: string;
  detail: string;
  connectionLabel: string;
  connectionTone: "connected" | "pending" | "fallback";
  modeLabel: string;
  taxonomyLabel: string;
  epochLabel: string;
  elapsedLabel: string;
  lastUpdateLabel: string;
  lastUpdateAt: string;
  nextStageLabel: string;
  reasonLabel: string;
  mixed: boolean;
  mixedLabel: string;
  terminalStage: "completed" | "failed" | "cancelled" | "";
  facts: LiveProgressFact[];
  ariaLabel: string;
};

export type BuildLiveProgressViewModelOptions = {
  snapshot: ProjectLiveState | null;
  connectionState: LiveProgressConnectionState;
  mode: LiveProgressMode;
  nowMs?: number;
};

const stageLabels: Partial<Record<NonNullable<LiveProgressStage>, string>> = {
  queued: "Queued",
  worker_starting: "Worker starting",
  remote_scheduled: "Remote scheduled",
  environment_starting: "Environment starting",
  dataset_materializing: "Dataset materializing",
  data_loading: "Data loading",
  model_initializing: "Model initializing",
  training: "Training",
  evaluating: "Evaluating",
  exporting: "Exporting",
  finalizing: "Finalizing",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const blockedReasonLabels: Record<string, string> = {
  backend_validation_blocked: "Backend validation blocked this work.",
  cost_budget_blocked: "The project cost budget blocked this work.",
  worker_requirement_cancelled: "The required worker startup was cancelled.",
  worker_requirement_failed: "The required worker failed to start.",
};

export function buildLiveProgressViewModel({
  snapshot,
  connectionState,
  mode,
  nowMs = Date.now(),
}: BuildLiveProgressViewModelOptions): LiveProgressViewModel {
  const connection = connectionPresentation(connectionState);
  const modeLabel = mode === "primary" ? "Incremental" : mode === "shadow" ? "Shadow" : "Legacy";
  if (!snapshot) {
    const stateLabel = connectionState === "fallback" || mode === "legacy" ? "Legacy polling" : "Waiting for live state";
    const detail = connectionState === "fallback"
      ? "The incremental source is unavailable; conservative polling remains active."
      : "Mission Control is waiting for the bounded project snapshot.";
    const ariaLabel = `Live progress. ${stateLabel}. Connection ${connection.label}.`;
    return {
      state: "idle",
      stateLabel,
      stage: "",
      stageLabel: "Progress unavailable",
      detail,
      connectionLabel: connection.label,
      connectionTone: connection.tone,
      modeLabel,
      taxonomyLabel: "",
      epochLabel: "",
      elapsedLabel: "",
      lastUpdateLabel: "",
      lastUpdateAt: "",
      nextStageLabel: "",
      reasonLabel: "",
      mixed: false,
      mixedLabel: "",
      terminalStage: "",
      facts: [],
      ariaLabel,
    };
  }

  // Operational state is backend-owned. In particular, a worker-observed
  // `finalizing` stage must never be promoted to terminal completion here.
  const state = presentationState(snapshot.operational_state);
  const progress = primaryProgress(snapshot);
  const stage = snapshot.current_stage || progress?.stage || "";
  const terminalStage = state === "terminal" && isTerminalStage(stage) ? stage : "";
  const stageLabel = labelForStage(stage);
  const stateLabel = labelForState(state, terminalStage);
  const epochLabel = epochProgressLabel(progress);
  const elapsedLabel = progress?.elapsed_started_at ? formatElapsedDuration(progress.elapsed_started_at, nowMs) : "";
  const lastUpdateAt = latestProgressTimestamp(progress, snapshot.last_heartbeat_at);
  const lastUpdateLabel = lastUpdateAt ? formatRelativeTimestamp(lastUpdateAt, nowMs) : "Unavailable";
  const nextStageLabel = state === "terminal" ? "" : labelForStage(snapshot.next_expected_stage || "");
  const reasonLabel = stateReason(snapshot, state);
  const taxonomyVersion = progress?.taxonomy_version || snapshot.taxonomy_version;
  const taxonomyLabel = taxonomyVersion > 0 ? `Taxonomy v${taxonomyVersion}` : "";
  const mixed = snapshot.mixed_jobs;
  const mixedLabel = mixed ? mixedJobsLabel(snapshot) : "";
  const detail = progressDetail(state, stageLabel, reasonLabel, mixed);
  const facts: LiveProgressFact[] = [];

  if (epochLabel) facts.push({ label: "Epoch", value: epochLabel.replace(/^Epoch\s+/i, "") });
  if (elapsedLabel) facts.push({ label: "Elapsed", value: elapsedLabel });
  facts.push({ label: "Last update", value: lastUpdateLabel, dateTime: lastUpdateAt || undefined });
  if (nextStageLabel) facts.push({ label: "Next expected", value: nextStageLabel });
  if (reasonLabel) facts.push({ label: state === "stale" ? "Stale state" : "Blocked reason", value: reasonLabel });
  if (mixedLabel) facts.push({ label: "Job mix", value: mixedLabel });

  const ariaParts = [
    "Live progress",
    stateLabel,
    `Stage ${stageLabel}`,
    epochLabel,
    elapsedLabel ? `Elapsed ${elapsedLabel}` : "",
    `Last update ${lastUpdateLabel}`,
    reasonLabel,
    nextStageLabel ? `Next expected ${nextStageLabel}` : "",
    mixedLabel,
    `Connection ${connection.label}`,
  ].filter(Boolean);

  return {
    state,
    stateLabel,
    stage,
    stageLabel,
    detail,
    connectionLabel: connection.label,
    connectionTone: connection.tone,
    modeLabel,
    taxonomyLabel,
    epochLabel,
    elapsedLabel,
    lastUpdateLabel,
    lastUpdateAt,
    nextStageLabel,
    reasonLabel,
    mixed,
    mixedLabel,
    terminalStage,
    facts,
    ariaLabel: `${ariaParts.join(". ")}.`,
  };
}

export function labelForStage(stage: LiveProgressStage | ""): string {
  if (!stage) return "Awaiting stage";
  return stageLabels[stage] ?? humanizeToken(String(stage));
}

export function formatElapsedDuration(startedAt: string, nowMs: number): string {
  const startedAtMs = Date.parse(startedAt);
  if (!Number.isFinite(startedAtMs)) return "Unavailable";
  const totalSeconds = Math.max(0, Math.floor((nowMs - startedAtMs) / 1_000));
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function formatRelativeTimestamp(timestamp: string, nowMs: number): string {
  const timestampMs = Date.parse(timestamp);
  if (!Number.isFinite(timestampMs)) return "Unavailable";
  const seconds = Math.max(0, Math.floor((nowMs - timestampMs) / 1_000));
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function presentationState(value: ProjectLiveState["operational_state"]): LiveProgressPresentationState {
  switch (value) {
    case "queued":
    case "active":
    case "retrying":
    case "blocked":
    case "stale":
    case "terminal":
      return value;
    default:
      return "idle";
  }
}

function primaryProgress(snapshot: ProjectLiveState): ActiveProgress | null {
  if (snapshot.active_progress.length === 0) return null;
  const stageMatch = snapshot.current_stage
    ? snapshot.active_progress.find((progress: ActiveProgress) => progress.stage === snapshot.current_stage)
    : undefined;
  return stageMatch ?? snapshot.active_progress[0] ?? null;
}

function labelForState(
  state: LiveProgressPresentationState,
  terminalStage: LiveProgressViewModel["terminalStage"],
): string {
  if (terminalStage) return labelForStage(terminalStage);
  switch (state) {
    case "queued":
      return "Queued";
    case "active":
      return "Active";
    case "retrying":
      return "Retrying";
    case "blocked":
      return "Blocked";
    case "stale":
      return "Stale";
    case "terminal":
      return "Terminal";
    default:
      return "Idle";
  }
}

function progressDetail(
  state: LiveProgressPresentationState,
  stageLabel: string,
  reasonLabel: string,
  mixed: boolean,
): string {
  let detail: string;
  switch (state) {
    case "queued":
      detail = "Work is queued and waiting for backend assignment.";
      break;
    case "active":
      detail = `${stageLabel} is in progress.`;
      break;
    case "retrying":
      detail = "The backend has queued another authoritative attempt.";
      break;
    case "blocked":
      detail = reasonLabel || "The backend marked this project as blocked.";
      break;
    case "stale":
      detail = reasonLabel || "The server marked the active progress heartbeat as stale.";
      break;
    case "terminal":
      detail = `The backend recorded the project as ${stageLabel.toLowerCase()}.`;
      break;
    default:
      detail = "No project work is currently active.";
  }
  return mixed ? `${detail} Jobs are currently in multiple lifecycle states.` : detail;
}

function epochProgressLabel(progress: ActiveProgress | null): string {
  if (!progress || !["epoch", "epochs"].includes(progress.unit ?? "")) return "";
  if (!isNonNegativeInteger(progress.current) || !isNonNegativeInteger(progress.total)) return "";
  return `Epoch ${progress.current}/${progress.total}`;
}

function latestProgressTimestamp(progress: ActiveProgress | null, projectHeartbeat?: string): string {
  const candidates = [progress?.updated_at, progress?.heartbeat_at, projectHeartbeat]
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .map((value) => ({ value, score: Date.parse(value) }))
    .filter((candidate) => Number.isFinite(candidate.score))
    .sort((left, right) => right.score - left.score);
  return candidates[0]?.value ?? "";
}

function stateReason(snapshot: ProjectLiveState, state: LiveProgressPresentationState): string {
  if (state === "stale") return "The server marked the active heartbeat as stale.";
  if (state !== "blocked") return "";
  const reasonCode = snapshot.blocked_reason_code || "";
  return blockedReasonLabels[reasonCode] ?? (reasonCode ? `${humanizeToken(reasonCode)}.` : "The backend marked this work as blocked.");
}

function mixedJobsLabel(snapshot: ProjectLiveState): string {
  const open = snapshot.jobs.queued + snapshot.jobs.retrying + snapshot.jobs.assigned + snapshot.jobs.running;
  const terminal = snapshot.jobs.succeeded + snapshot.jobs.failed + snapshot.jobs.cancelled;
  if (open > 0 && terminal > 0) return `${open} open, ${terminal} terminal`;
  return `${snapshot.jobs.total} jobs across lifecycle states`;
}

function isTerminalStage(stage: LiveProgressStage | ""): stage is "completed" | "failed" | "cancelled" {
  return stage === "completed" || stage === "failed" || stage === "cancelled";
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function humanizeToken(value: string): string {
  const words = value.replace(/[^a-z0-9]+/gi, " ").trim();
  if (!words) return "Unknown";
  return words.charAt(0).toUpperCase() + words.slice(1).toLowerCase();
}

function connectionPresentation(connectionState: LiveProgressConnectionState): {
  label: string;
  tone: LiveProgressViewModel["connectionTone"];
} {
  switch (connectionState) {
    case "connected":
      return { label: "Connected", tone: "connected" };
    case "reconnecting":
      return { label: "Reconnecting", tone: "pending" };
    case "recovering":
      return { label: "Recovering cursor", tone: "pending" };
    case "fallback":
      return { label: "Fallback polling", tone: "fallback" };
    case "unsupported":
      return { label: "Legacy backend", tone: "fallback" };
    case "disconnected":
      return { label: "Disconnected", tone: "fallback" };
    case "connecting":
      return { label: "Connecting", tone: "pending" };
    default:
      return { label: "Idle", tone: "pending" };
  }
}
