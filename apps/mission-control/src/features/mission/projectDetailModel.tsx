import type { ReactNode } from "react";
import { Activity, AlertTriangle, CheckCircle2, X } from "lucide-react";

import { readyONNXExport } from "../../championLocalInference";
import type { ActivityStreamState } from "../../hooks/useActivityStream";
import type { DatasetMetadataDetail, ProjectDetail, VisualAnalysisDetail } from "../../hooks/useProjectDetail";
import { projectTabs, type ActivityFilterKey, type ProjectTabKey, type ProjectTabTarget } from "./workflowTabs";
import {
  formatBytes,
  formatCompactNumber,
  formatCurrency,
  formatDecisionQualityCount,
  formatDecisionQualityMetric,
  formatLatency,
  formatLossGap,
  formatMaybeMetric,
  formatMetricNumber,
  formatPercent,
  formatRelativeTime,
  formatSeconds,
  formatSeedVariance,
  formatTelemetryTokenPair,
  formatTimestamp,
  formatTopKScore,
  formatUnknownValue,
} from "../../utils/formatting";
import { shortValue } from "../../utils/safeDisplay";
import type {
  AgentActivityEvent,
  AgentDecision,
  AgentInvocation,
  AgentMemoryRecord,
  ChampionDemoImage,
  ChampionDemoPrediction,
  ChampionDetection,
  ChampionExport,
  ChampionFeedback,
  Dataset,
  DatasetMetadataImport,
  DatasetMetadataSummary,
  DatasetVisualAnalysis,
  EpochMetric,
  ExecutionEvent,
  ExperimentPlan,
  Health,
  Job,
  PlannedExperiment,
  MemoryEmbeddingUsageEvent,
  MissionControlTelemetryResponse,
  Project,
  ProjectChampion,
  PortableInferenceBundle,
  RetrievedMemoryCard,
  StrategyScorecard,
  TrainingRunEvaluation,
  TrainingRunSummary,
  VisualAnalysisRerunPolicy,
  AutomationSettings,
  Worker,
  WorkerRequirement,
} from "../../types";

export type MetricKey = string;
export const classificationMetricPriority = ["macro_f1", "accuracy", "train_loss", "val_loss"];
export const detectionMetricPriority = [
  "mAP50_95",
  "mAP50",
  "precision",
  "recall",
  "box_loss",
  "cls_loss",
  "dfl_loss",
  "train_loss",
  "val_loss",
];
export const detectionMetricAliases: Record<string, string[]> = {
  mAP50_95: ["mAP50_95", "map50_95", "map"],
  mAP50: ["mAP50", "map50"],
};

export function metricLabel(key: MetricKey) {
  switch (key) {
    case "mAP50_95":
    case "map50_95":
    case "map":
      return "mAP50-95";
    case "mAP50":
    case "map50":
      return "mAP50";
    case "dfl_loss":
      return "DFL loss";
    case "box_loss":
      return "Box loss";
    case "cls_loss":
      return "Class loss";
    case "train_loss":
      return "Train loss";
    case "val_loss":
      return "Val loss";
    case "accuracy":
      return "Accuracy";
    default:
      return key;
  }
}

export function isDetectionJob(job: Job | null) {
  if (!job) return false;
  const config = job.config ?? {};
  const values = [job.template, config.task, config.task_type, config.model, config.architecture, config.framework]
    .map((value) => String(value ?? "").toLowerCase())
    .join(" ");
  return values.includes("object_detection") || values.includes("detection") || values.includes("yolo");
}

export function isDetectionRun(summary: TrainingRunSummary | null, evaluation: TrainingRunEvaluation | null, job: Job | null) {
  if (isDetectionJob(job)) return true;
  const objective = evaluation?.objective_profile ?? {};
  const modelProfile = evaluation?.model_profile ?? {};
  const values = [
    summary?.target_metric,
    recordString(objective, "target_metric"),
    recordString(objective, "task_type"),
    recordString(modelProfile, "task_type"),
    recordString(modelProfile, "model_kind"),
    recordString(modelProfile, "architecture"),
    recordString(modelProfile, "framework"),
  ]
    .map((value) => String(value ?? "").toLowerCase())
    .join(" ");
  return values.includes("object_detection") || values.includes("detection") || values.includes("yolo") || values.includes("map50");
}

export function firstPositiveMetric(values: Array<number | null | undefined>) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return 0;
}

export function yoloMetricFromEvaluation(evaluation: TrainingRunEvaluation | null, metric: "map50_95" | "map50" | "precision" | "recall") {
  const objective = evaluation?.objective_profile ?? {};
  const detectionMetrics = recordObject((evaluation?.holistic_scores ?? {}).detection_metrics);
  switch (metric) {
    case "map50_95":
      return firstPositiveMetric([
        recordNumber(objective, "heldout_test_map50_95"),
        recordNumber(objective, "heldout_test_map"),
        recordNumber(detectionMetrics, "mAP50_95"),
        recordNumber(detectionMetrics, "map50_95"),
        recordNumber(detectionMetrics, "map"),
      ]);
    case "map50":
      return firstPositiveMetric([
        recordNumber(objective, "heldout_test_map50"),
        recordNumber(detectionMetrics, "mAP50"),
        recordNumber(detectionMetrics, "map50"),
      ]);
    case "precision":
      return firstPositiveMetric([recordNumber(objective, "heldout_test_precision"), recordNumber(detectionMetrics, "precision")]);
    case "recall":
      return firstPositiveMetric([recordNumber(objective, "heldout_test_recall"), recordNumber(detectionMetrics, "recall")]);
  }
  return 0;
}

export function runPrimaryMetric(summary: TrainingRunSummary | null, evaluation: TrainingRunEvaluation | null, job: Job | null) {
  if (isDetectionRun(summary, evaluation, job)) {
    const map50_95 = firstPositiveMetric([
      yoloMetricFromEvaluation(evaluation, "map50_95"),
      summary?.best_map50_95,
      summary?.best_macro_f1,
    ]);
    const map50 = firstPositiveMetric([yoloMetricFromEvaluation(evaluation, "map50"), summary?.best_map50, summary?.best_accuracy]);
    return {
      label: "mAP50-95",
      value: map50_95,
      secondaryLabel: "mAP50",
      secondaryValue: map50,
      isDetection: true,
    };
  }
  return {
    label: "Macro-F1",
    value: summary?.best_macro_f1 ?? 0,
    secondaryLabel: "Accuracy",
    secondaryValue: summary?.best_accuracy ?? 0,
    isDetection: false,
  };
}

export function effectiveTrainingRunStatus(summary: TrainingRunSummary | null, job: Job | null) {
  const jobStatus = normalizedStatus(job?.status ?? "");
  if (jobStatus === "SUCCEEDED" || jobStatus === "FAILED") {
    return jobStatus;
  }
  const summaryStatus = String(summary?.status ?? "").trim();
  return summaryStatus ? normalizedStatus(summaryStatus) : "UNKNOWN";
}

export function championPrimaryMetric(
  champion: ProjectChampion | null,
  summary: TrainingRunSummary | null,
  evaluation: TrainingRunEvaluation | null,
  job: Job | null,
) {
  const metric = runPrimaryMetric(summary, evaluation, job);
  if (!champion) return metric;
  const explicitLabel = recordString(champion.metrics, "primary_metric_label");
  const explicitValue = recordNumber(champion.metrics, "primary_metric_value");
  if (explicitValue > 0) {
    return { ...metric, label: explicitLabel || metric.label, value: explicitValue };
  }
  if (metric.isDetection) {
    const map50_95 = firstPositiveMetric([recordNumber(champion.metrics, "best_map50_95"), metric.value]);
    const map50 = firstPositiveMetric([recordNumber(champion.metrics, "best_map50"), metric.secondaryValue]);
    return { ...metric, value: map50_95, secondaryValue: map50 };
  }
  return metric;
}

export function metricTabOptions(metrics: EpochMetric[], job: Job | null): Array<{ key: MetricKey; label: string }> {
  const present = new Set<string>();
  metrics.forEach((metric) => {
    Object.entries(metric.metrics ?? {}).forEach(([key, value]) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        present.add(key);
      }
    });
  });
  const priority = isDetectionJob(job) ? detectionMetricPriority : classificationMetricPriority;
  const prioritizedKeys = priority
    .map((key) => (detectionMetricAliases[key] ?? [key]).find((alias) => present.has(alias)))
    .filter((key): key is string => Boolean(key));
  const prioritizedAliases = new Set(priority.flatMap((key) => detectionMetricAliases[key] ?? [key]));
  const ordered = [
    ...prioritizedKeys,
    ...Array.from(present)
      .filter((key) => !prioritizedAliases.has(key))
      .sort((a, b) => a.localeCompare(b)),
  ];
  return ordered.map((key) => ({ key, label: metricLabel(key) }));
}

export type CancelExecutionResponse = {
  execution_id: string;
  status: string;
  message?: string;
  queued_jobs_cancelled: number;
  active_jobs_marked_cancelling: number;
  already_terminal_jobs?: number;
  modal_calls?: Array<{
    job_id: string;
    training_attempt_id?: string;
    modal_function_call_object_id?: string;
    cancel_status: string;
  }>;
  worker_requirements?: WorkerRequirement[];
  best_available_model?: {
    job_id?: string;
    exportable?: boolean;
    reason?: string;
    champion_selection_source?: string;
  };
};

export type MissionDigestState =
  | "empty"
  | "dataset_needed"
  | "profiling"
  | "plan_ready"
  | "training"
  | "review_needed"
  | "follow_up_ready"
  | "champion_selected"
  | "blocked";

export type MissionTone = "ok" | "warning" | "bad" | "info";

export type MissionHealthItem = {
  id: string;
  label: string;
  value: string;
  tone: MissionTone;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

export type MissionActionKey =
  | "new_project"
  | "refresh"
  | "execute_plan"
  | "review_experiments"
  | "schedule_follow_up"
  | "open_export";

export type MissionNextAction = {
  id: string;
  label: string;
  reason: string;
  priority: "primary" | "secondary";
  disabled?: boolean;
  actionKey?: MissionActionKey;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

export type MissionLiveActivity = {
  status: "idle" | "active" | "waiting" | "blocked" | "failed" | "succeeded";
  label: string;
  detail: string;
  streamState: ActivityStreamState;
  steps: Array<{
    id: string;
    label: string;
    status: "active" | "waiting" | "succeeded" | "failed" | "blocked";
    timestamp?: string;
    targetTab?: ProjectTabTarget;
    targetId?: string;
  }>;
};

export type MissionSignal = {
  id: string;
  label: string;
  detail: string;
  tone: MissionTone;
  timestamp?: string;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

export type MissionChampionSummary = {
  model: string;
  primaryMetricLabel: string;
  primaryMetricValue: string;
  secondaryMetricLabel: string;
  secondaryMetricValue: string;
  cost: string;
  runtime: string;
  latency: string;
  modelSize: string;
  objectiveFit: string;
};

export type MissionDigest = {
  state: MissionDigestState;
  stateLabel: string;
  healthLabel: string;
  headline: string;
  detail: string;
  facts: Array<{ label: string; value: string; tone?: MissionTone }>;
  health: MissionHealthItem[];
  nextActions: MissionNextAction[];
  liveActivity: MissionLiveActivity;
  recentSignals: MissionSignal[];
  champion?: MissionChampionSummary;
};

export type MissionBrief = {
  id: string;
  title: string;
  goal: string;
  statusLabel: string;
  progressLabel: string;
  completedExperiments: number;
  totalExperiments: number;
  bestMetricLabel: string;
  bestMetricValue: string;
  etaLabel: string;
  primaryAction: string;
  blocker: string;
  updatedAt: string;
};

export type AIThinking = {
  state: string;
  observation: string;
  reasoning: string;
  decision: string;
  expectedOutcome: string;
  confidenceLabel: string;
  updatedAt: string;
};

export type MissionStage = {
  id: string;
  label: string;
  detail: string;
  status: "done" | "active" | "waiting" | "blocked";
  timestamp?: string;
  evidence?: string;
};

export type ActivityCardType = "mission" | "observation" | "decision" | "experiment" | "result" | "blocker" | "export";

export type ActivityCardModel = {
  id: string;
  type: ActivityCardType;
  title: string;
  summary: string;
  timestamp: string;
  status: "active" | "waiting" | "succeeded" | "failed" | "blocked";
  evidenceSummary: string;
  resultSummary: string;
  technicalSource: string;
  developerPayloadRef: string;
};

export type ResultsCandidate = {
  rank: number;
  model: string;
  metricLabel: string;
  metricValue: string;
  status: string;
  why: string;
  jobId: string;
};

export type ResultsSummary = {
  hasResults: boolean;
  championModel: string;
  primaryMetricLabel: string;
  primaryMetricValue: string;
  improvementLabel: string;
  exportStatus: string;
  whyItWon: string;
  learningSummary: string[];
  remainingRisks: string[];
  topCandidates: ResultsCandidate[];
};

export type ExportSummary = {
  hasChampion: boolean;
  title: string;
  statusLabel: string;
  readinessLabel: string;
  primaryFormat: string;
  validationStatus: string;
  demoStatus: string;
  includes: string[];
  useCases: string[];
  limitations: string[];
  manifestAvailable: boolean;
};

export type TelemetryWindowKey = "today" | "7d" | "lifetime";

export type TelemetryWindowSummary = {
  key: TelemetryWindowKey;
  label: string;
  calls: number;
  exactCalls: number;
  approxCalls: number;
  validCalls: number;
  invalidCalls: number;
  exactInputTokens: number;
  exactOutputTokens: number;
  approxInputTokens: number;
  approxOutputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  estimatedCostUsd: number;
};

export type TelemetryCountSummary = {
  label: string;
  count: number;
  exactCalls: number;
  approxCalls: number;
  inputTokens: number;
  outputTokens: number;
  approxInputTokens: number;
  approxOutputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  estimatedCostUsd: number;
};

export type TelemetrySectionSummary = {
  name: string;
  calls: number;
  bytes: number;
  approxTokens: number;
  exampleSource?: string;
};

export type TelemetryEmbeddingBreakdownRow = {
  label: string;
  count: number;
  providerCalls: number;
  inputBytes: number;
  estimatedCostUsd: number;
  retrievedCount: number;
  injected: number;
  logOnly: number;
  cached: number;
  skipped: number;
};

export type TelemetryInvocationSummary = {
  id: string;
  createdAt: string;
  agentName: string;
  model: string;
  validationStatus: string;
  usageKind: "exact" | "approximate";
  inputTokens: number;
  outputTokens: number;
  approxInputTokens: number;
  approxOutputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  estimatedCostUsd: number;
  promptBytes: number;
  sections: TelemetrySectionSummary[];
  largestSection: string;
};

export type TelemetryEmbeddingPurposeSummary = {
  purpose: string;
  retrievalPurpose: string;
  count: number;
  providerCalls: number;
  inputBytes: number;
  estimatedCostUsd: number;
  retrievedCount: number;
  injected: number;
  logOnly: number;
  cached: number;
  skipped: number;
  bySourceTable: TelemetryEmbeddingBreakdownRow[];
  byModel: TelemetryEmbeddingBreakdownRow[];
};

export type TelemetrySummary = {
  invocations: AgentInvocation[];
  usageEvents: MemoryEmbeddingUsageEvent[];
  windows: TelemetryWindowSummary[];
  callsByAgent: TelemetryCountSummary[];
  callsByModel: TelemetryCountSummary[];
  topInvocations: TelemetryInvocationSummary[];
  promptSections: TelemetrySectionSummary[];
  embedding: {
    sourceIndex: TelemetryEmbeddingPurposeSummary;
    retrievalQuery: TelemetryEmbeddingPurposeSummary;
    totalUsageEvents: number;
  };
};

export type VisualAnalysisListResponse = {
  visual_analyses?: DatasetVisualAnalysis[];
  dataset_visual_analyses?: DatasetVisualAnalysis[];
  analyses?: DatasetVisualAnalysis[];
  items?: DatasetVisualAnalysis[];
  analysis?: DatasetVisualAnalysis;
  latest?: DatasetVisualAnalysis;
  dataset_visual_analysis?: DatasetVisualAnalysis;
  rerun_policy?: VisualAnalysisRerunPolicy;
  manual_run_supported?: boolean;
};

export type AgentInvocationsResponse = {
  invocations?: AgentInvocation[];
  agent_invocations?: AgentInvocation[];
  items?: AgentInvocation[];
};

export type ProjectDetailRefreshOptions = {
  includeSlowData?: boolean;
  forceSlowData?: boolean;
};

export type DatasetMetadataSummaryResponse =
  | DatasetMetadataSummary
  | {
      summary?: DatasetMetadataSummary;
      metadata_summary?: DatasetMetadataSummary;
      dataset_metadata_summary?: DatasetMetadataSummary;
      agent_safe_summary?: DatasetMetadataSummary;
      import?: DatasetMetadataImport;
      metadata_import?: DatasetMetadataImport;
    };

export type DatasetMetadataImportsResponse =
  | DatasetMetadataImport[]
  | {
      imports?: DatasetMetadataImport[];
      metadata_imports?: DatasetMetadataImport[];
      items?: DatasetMetadataImport[];
      latest?: DatasetMetadataImport;
      active?: DatasetMetadataImport;
      import?: DatasetMetadataImport;
      metadata_import?: DatasetMetadataImport;
    };

export type TimelineItem = {
  label: string;
  detail: string;
  status: "done" | "active" | "waiting" | "blocked";
  timestamp?: string;
};

export type InsightItem = {
  label: string;
  value: string;
  tone?: "good" | "warn" | "bad";
};

export type MetadataCountRow = {
  label: string;
  value: string;
};

export type MetadataStatusDisplay = {
  status: string;
  detail: string;
  facts: InsightItem[];
  sources: string[];
  splitRows: MetadataCountRow[];
  annotationRows: MetadataCountRow[];
  warnings: string[];
  errors: string[];
};

export type ReasoningSection = {
  title: string;
  items: string[];
};

export type ChampionComparisonRow = {
  jobId: string;
  model: string;
  rankScore: number;
  primaryMetricLabel: string;
  primaryMetricValue: number;
  secondaryMetricLabel: string;
  secondaryMetricValue: number;
  runtimeSeconds: number;
  costUsd: number;
  latencyMs: number;
  objectiveFit: number;
  trainValidationGap: number | null;
  divergenceStatus: string;
  seedVariance: number | null;
  seedRunCount: number;
  isChampion: boolean;
};

export type CandidateScoreRow = {
  label: string;
  status: string;
  mechanism: string;
  intervention: string;
  expectedEffect: string;
  validationStatus: string;
  totalScore: number | null;
  reasons: string[];
  memoryReasons: string[];
  memoryHits: RetrievedMemoryDisplay[];
  components: Array<{ label: string; value: number | string }>;
};

export type RetrievedMemoryDisplay = {
  source: string;
  sourceId: string;
  kind: string;
  outcome: string;
  mechanism: string;
  intervention: string;
  summary: string;
  retrievalReason: string;
  score: number | null;
  identifiers: Array<{ label: string; value: string }>;
};

export type MemoryRetrievalProbeSnapshot = {
  id: string;
  purpose: string;
  logOnly: boolean;
  crossProjectOK: boolean;
  retrievedCount: number;
  createdAt: string;
  cards: RetrievedMemoryDisplay[];
};

export type AgentInvocationAuditRow = {
  id: string;
  agentName: string;
  createdAt: string;
  target: string;
  validationStatus: string;
  validationError: string;
  apiStyle: string;
  providerModel: string;
  reasoningEffort: string;
  toolRounds: string;
  toolNames: string[];
  rejectedToolCalls: string[];
  dryRunValidationResults: Array<{ status: string; text: string }>;
  decisionLink: string;
};

export type DecisionChatTurn = {
  decision: AgentDecision;
  question: string;
  opening: string;
  highlights: Array<{ label: string; value: string }>;
  sections: ReasoningSection[];
  retrievedMemory: RetrievedMemoryDisplay[];
  rejections: Array<{ kind: string; text: string }>;
  mechanismCoverage: MechanismCoverageRow[];
  candidateScores: CandidateScoreRow[];
};

export type MechanismCoverageRow = {
  mechanism: string;
  status: string;
  detail: string;
};

export type DecisionQualitySnapshot = {
  decisionId: string;
  decisionType: string;
  createdAt: string;
  source: string;
  decisionPressure: string;
  blockedMechanisms: string[];
  completedTrainingRuns: number | null;
  completedPlannerRounds: number | null;
  gainPerCompletedRun: number | null;
  recentBestDelta: number | null;
  minimumUsefulDelta: number | null;
  selectedCandidates: number;
  totalCandidates: number;
  rejectedCandidates: number;
  topRejectionReason: string;
  exhaustedOutcomes: MechanismCoverageRow[];
  warnings: string[];
};

export type ChampionExportDemo = {
  hasChampion: boolean;
  exportStatus: string;
  exports: ChampionExport[];
  portableBundle?: PortableInferenceBundle;
  projectId: string;
  modelCard: Record<string, unknown>;
  deploymentProfile: Record<string, unknown>;
  modelProfile: Record<string, unknown>;
  useCases: string[];
  limitations: string[];
  preprocessing: string[];
  demoImages: ChampionDemoImage[];
  demoPredictions: ChampionDemoPrediction[];
  feedback: ChampionFeedback[];
};

export type ChampionFeedbackRating = ChampionFeedback["rating"];

export type ChampionFeedbackDraft = {
  rating: ChampionFeedbackRating;
  message: string;
};

export type Notice = {
  kind: "info" | "error";
  text: string;
};

export function NoticeBanner({ notice }: { notice: Notice }) {
  const display = noticeDisplay(notice);

  return (
    <div className={`notice ${notice.kind}`} role={notice.kind === "error" ? "alert" : "status"} aria-live={notice.kind === "error" ? "assertive" : "polite"}>
      <span className="notice-icon" aria-hidden="true">
        {notice.kind === "error" ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
      </span>
      <span className="notice-copy">
        <strong>{display.title}</strong>
        <small title={notice.text}>{display.message}</small>
      </span>
    </div>
  );
}

export function noticeDisplay(notice: Notice) {
  if (notice.kind === "error" && notice.text.includes("Cannot read properties of undefined") && notice.text.includes("request")) {
    return {
      title: "Desktop bridge unavailable",
      message: "The browser preview cannot reach the Mission Control desktop bridge. Open the Electron app for live project actions.",
    };
  }
  if (
    notice.kind === "error" &&
    ((notice.text.includes("401") && notice.text.includes("missing or invalid API token")) ||
      (notice.text.includes("MODEL_EXPRESS_API_TOKEN") && notice.text.includes("LAN or tunnel mode")))
  ) {
    return {
      title: "API token required",
      message: "The backend is in LAN or tunnel mode. Set MODEL_EXPRESS_API_TOKEN in .env.local, or remove the public Modal/tunnel URL for local-only use, then restart the backend and app.",
    };
  }

  return {
    title: notice.kind === "error" ? "Action needs attention" : "Mission update",
    message: notice.text,
  };
}

export type DatasetFolder = {
  token: string;
  path: string;
  name: string;
  preflight?: DatasetUploadPreflight;
};

export type DatasetUploadWarning = {
  code: string;
  message: string;
  threshold?: number;
  value?: number;
};

export type DatasetUploadPreflight = {
  file_count: number;
  uncompressed_size_bytes: number;
  archive_size_bytes: number;
  largest_file?: {
    path: string;
    size_bytes: number;
  } | null;
  warnings: DatasetUploadWarning[];
  errors: DatasetUploadWarning[];
};

export type ScheduleFollowUpResponse = {
  decision: AgentDecision;
  follow_up_plan?: ExperimentPlan;
};

export type AutomationSettingsUpdate = Partial<
  Pick<
    AutomationSettings,
    | "auto_review_experiments"
    | "auto_schedule_followups"
    | "auto_execute_plans"
    | "max_followup_rounds"
    | "default_training_provider"
    | "default_gpu_type"
    | "cost_mode"
    | "budget_cap_usd"
    | "llm_enabled"
    | "agent_mode"
    | "llm_provider"
    | "llm_model"
    | "automl_enabled"
    | "automl_sampler"
  >
>;
export function summarizeTrainingRuns(
  summaries: TrainingRunSummary[],
  evaluations: TrainingRunEvaluation[] = [],
  jobs: Job[] = [],
) {
  const evaluationByJob = new Map(evaluations.map((evaluation) => [evaluation.job_id, evaluation]));
  const jobById = new Map(jobs.map((job) => [job.id, job]));
  const best = summaries.reduce<{ summary: TrainingRunSummary; metric: ReturnType<typeof runPrimaryMetric> } | null>((currentBest, summary) => {
    const metric = runPrimaryMetric(summary, evaluationByJob.get(summary.job_id) ?? null, jobById.get(summary.job_id) ?? null);
    if (!currentBest) return { summary, metric };
    return metric.value > currentBest.metric.value ? { summary, metric } : currentBest;
  }, null);

  return {
    totalCost: summaries.reduce((total, summary) => total + summary.estimated_cost_usd, 0),
    totalRuntimeSeconds: summaries.reduce((total, summary) => total + summary.runtime_seconds, 0),
    bestMacroF1: best?.summary.best_macro_f1 ?? 0,
    bestPrimaryMetricLabel: best?.metric.label ?? "Macro-F1",
    bestPrimaryMetricValue: best?.metric.value ?? 0,
    activeRuns: summaries.filter((summary) =>
      ["RUNNING", "ASSIGNED", "QUEUED"].includes(effectiveTrainingRunStatus(summary, jobById.get(summary.job_id) ?? null)),
    ).length,
  };
}

export function trainingRunCacheSummary(summary: TrainingRunSummary) {
  const materialization = recordObject(summary.dataset_materialization);
  const reuseStatus = recordString(materialization, "dataset_prewarm_reuse_status");
  const status = recordString(materialization, "dataset_materialization_status");
  const cacheRoot = recordString(materialization, "dataset_training_cache_root");
  const cacheKey = recordString(materialization, "dataset_materialization_cache_key");
  const hitMiss = recordBoolean(materialization, "dataset_materialization_cache_hit")
    ? "hit"
    : recordBoolean(materialization, "dataset_materialization_cache_miss")
      ? "miss"
      : "";
  const parts = [
    reuseStatus ? humanizeAuditKey(reuseStatus) : status ? humanizeAuditKey(status) : "",
    hitMiss,
    cacheKey ? shortCacheKey(cacheKey) : "",
    cacheRoot ? "cache root set" : "",
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" - ") : "";
}

export function trainingRunLifecycleChips(summary: TrainingRunSummary) {
  const stageTelemetry = recordObject(summary.stage_telemetry);
  const materialization = recordObject(summary.dataset_materialization);
  const warmPolicy = recordObject(stageTelemetry.warm_container_policy);
  const chips = [
    lifecycleSecondsChip("queue", recordNumber(stageTelemetry, "queue_wait_seconds")),
    lifecycleSecondsChip("materialize", recordNumber(stageTelemetry, "dataset_materialization_seconds")),
    lifecycleSecondsChip("train", recordNumber(stageTelemetry, "active_training_seconds")),
    lifecycleSecondsChip("idle", recordNumber(stageTelemetry, "idle_wait_seconds")),
    lifecycleSecondsChip("scaledown", recordNumber(warmPolicy, "scaledown_window_seconds")),
    trainingRunCacheSummary(summary),
  ].filter(Boolean);
  const reuseStatus = recordString(materialization, "dataset_prewarm_reuse_status");
  if (reuseStatus && !chips.includes(reuseStatus)) chips.push(humanizeAuditKey(reuseStatus));
  return chips.slice(0, 7);
}

export function lifecycleSecondsChip(label: string, seconds: number) {
  return seconds > 0 ? `${label} ${formatSeconds(seconds)}` : "";
}

export function workerRequirementMaterializationSummary(requirement: WorkerRequirement) {
  const parts = [
    requirement.dataset_materialization_status ? humanizeAuditKey(requirement.dataset_materialization_status) : "",
    requirement.max_concurrent_jobs ? `${requirement.max_concurrent_jobs} concurrent` : "",
    requirement.max_cold_dataset_materializations ? `${requirement.max_cold_dataset_materializations} cold` : "",
    requirement.dataset_cache_key ? shortCacheKey(requirement.dataset_cache_key) : "",
  ].filter(Boolean);
  return parts.join(" / ");
}

export function shortCacheKey(value: string) {
  const text = String(value || "").trim();
  if (text.length <= 18) return text;
  const [prefix, rest] = text.split("-", 2);
  if (prefix && rest) return `${prefix}-${rest.slice(0, 8)}`;
  return text.slice(0, 12);
}

export function projectTabFromTarget(tab: ProjectTabTarget): ProjectTabKey {
  if (tab === "overview") return "mission";
  if (tab === "agents") return "activity";
  if (tab === "experiments") return "results";
  if (tab === "data" || tab === "operations") return "developer";
  return tab;
}

export function missionStateLabelFromProject(project: Project) {
  const status = normalizedStatus(project.status || "");
  if (status === "COMPLETED") return "Completed";
  if (status === "FAILED" || status === "BLOCKED") return "Needs input";
  if (status === "RUNNING" || status === "ACTIVE") return "In progress";
  return "Ready";
}

export function missionStateToneFromProject(project: Project) {
  const status = normalizedStatus(project.status || "");
  if (status === "COMPLETED") return "done";
  if (status === "FAILED" || status === "BLOCKED") return "blocked";
  if (status === "RUNNING" || status === "ACTIVE") return "active";
  return "ready";
}

export function buildMissionBrief(
  project: Project | null,
  detail: ProjectDetail,
  digest: MissionDigest,
  automationSettings: AutomationSettings,
): MissionBrief {
  const counts = jobStatusCounts(detail.jobs);
  const completedExperiments = counts.SUCCEEDED + counts.FAILED;
  const latestPlan = detail.plans[0] ?? null;
  const plannedExperiments = latestPlan?.experiments.length ?? 0;
  const totalExperiments = Math.max(detail.jobs.length, plannedExperiments, completedExperiments);
  const runTotals = summarizeTrainingRuns(detail.runSummaries, detail.runEvaluations, detail.jobs);
  const championSummary = detail.champion ? buildMissionChampionSummary(detail) : undefined;
  const bestMetricLabel = championSummary?.primaryMetricLabel || runTotals.bestPrimaryMetricLabel || "Best result";
  const bestMetricValue =
    championSummary?.primaryMetricValue ||
    (runTotals.bestPrimaryMetricValue > 0 ? formatMaybeMetric(runTotals.bestPrimaryMetricValue) : "Pending");
  const activeOrQueued = counts.QUEUED + counts.ASSIGNED + counts.RUNNING;
  const etaLabel =
    activeOrQueued > 0 && latestPlan?.estimated_minutes
      ? `${latestPlan.estimated_minutes} min estimate`
      : detail.champion
        ? "Export handoff ready"
        : automationSettings.auto_execute_plans
          ? "Working automatically"
          : "Waiting for approval";
  const primaryAction = digest.nextActions.find((action) => action.priority === "primary")?.label || "Watch progress";
  const blocker = digest.state === "blocked" ? userFacingActivityText(digest.detail, 160) : "";

  return {
    id: project?.id || "no-mission",
    title: project?.name || "Waiting for a mission",
    goal:
      project?.goal ||
      "Give Model Express a dataset and goal. It will train, compare, refine, and prepare the best model.",
    statusLabel: digest.stateLabel,
    progressLabel:
      totalExperiments > 0
        ? `${completedExperiments}/${totalExperiments} experiments complete`
        : project
          ? "Preparing first experiment"
          : "No mission yet",
    completedExperiments,
    totalExperiments,
    bestMetricLabel,
    bestMetricValue,
    etaLabel,
    primaryAction,
    blocker,
    updatedAt: project?.updated_at || "",
  };
}

export function buildCurrentThinking(project: Project | null, detail: ProjectDetail, digest: MissionDigest): AIThinking {
  const dataset = detail.datasets[0] ?? null;
  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const counts = jobStatusCounts(detail.jobs);
  const activeExperiments = counts.QUEUED + counts.ASSIGNED + counts.RUNNING;
  const latestExperiment = latestPlan?.experiments[0] ?? null;
  const latestActivity = digest.liveActivity;

  if (!project) {
    return {
      state: "Waiting for a mission",
      observation: "No mission is selected yet.",
      reasoning: "Model Express needs a goal and dataset before it can choose experiments.",
      decision: "Create or select a mission.",
      expectedOutcome: "The mission workspace will show progress as soon as setup starts.",
      confidenceLabel: "Ready",
      updatedAt: "",
    };
  }

  if (!dataset) {
    return {
      state: "Waiting for dataset",
      observation: "The mission has a goal but no dataset attached.",
      reasoning: "A dataset review is required before Model Express can choose a training path.",
      decision: "Attach an image dataset to begin.",
      expectedOutcome: "The AI will inspect classes, examples, and label coverage.",
      confidenceLabel: "Needs input",
      updatedAt: project.updated_at,
    };
  }

  if (digest.state === "blocked") {
    return {
      state: "Needs your input",
      observation: userFacingActivityText(digest.headline, 140),
      reasoning: userFacingActivityText(digest.detail, 180),
      decision: "Pause new work until the blocker is resolved.",
      expectedOutcome: "Once resolved, Model Express can continue the mission from the latest safe point.",
      confidenceLabel: "Blocked",
      updatedAt: latestDecision?.created_at || latestActivity.steps[0]?.timestamp || project.updated_at,
    };
  }

  if (detail.champion) {
    const champion = buildMissionChampionSummary(detail);
    return {
      state: "Preparing handoff",
      observation: `${champion?.model || "The selected model"} is currently leading the mission.`,
      reasoning: "The completed evidence supports using this model as the best available candidate.",
      decision: "Prepare the export package and demo validation.",
      expectedOutcome: "You can hand off an export-ready model with an explanation of why it won.",
      confidenceLabel: "Champion selected",
      updatedAt: detail.champion.updated_at || detail.champion.created_at,
    };
  }

  if (latestDecision) {
    const payload = latestDecision.payload ?? {};
    const proposed = firstProposedExperiment(payload);
    const expectedEffect =
      recordFirstString(proposed, ["expected_effect", "expected_metric_effect"]) ||
      recordFirstString(payload, ["expected_effect", "expected_metric_effect", "why_can_beat_champion"]);
    const observation =
      recordFirstString(payload, ["observation", "summary", "diagnosis", "recommendation_summary"]) ||
      latestDecision.rationale ||
      "The AI reviewed the latest experiment evidence.";
    const decision = missionDecisionToUserText(latestDecision, proposed);
    return {
      state: latestActivity.label === "Idle" ? "Reviewing evidence" : latestActivity.label,
      observation: userFacingActivityText(observation, 170),
      reasoning: userFacingActivityText(decisionRationaleSummary(latestDecision), 190),
      decision,
      expectedOutcome: userFacingActivityText(expectedEffect || "Improve the mission metric without losing deployment readiness.", 170),
      confidenceLabel: "Evidence-backed",
      updatedAt: latestDecision.created_at,
    };
  }

  if (activeExperiments > 0) {
    return {
      state: "Running experiments",
      observation: `${activeExperiments} experiment${activeExperiments === 1 ? "" : "s"} are in progress or waiting to start.`,
      reasoning: "Model Express is collecting comparable evidence before choosing a refinement or champion.",
      decision: "Wait for the current experiments to report results.",
      expectedOutcome: "Completed runs will reveal the strongest candidate and the next improvement target.",
      confidenceLabel: "In progress",
      updatedAt: detail.jobs[0]?.completed_at || detail.jobs[0]?.started_at || detail.jobs[0]?.created_at || project.updated_at,
    };
  }

  if (latestPlan && latestExperiment) {
    return {
      state: "Planning first experiments",
      observation: `${latestPlan.experiments.length} experiment${latestPlan.experiments.length === 1 ? "" : "s"} are ready.`,
      reasoning: "The initial batch establishes a baseline and tests the most likely model family for the goal.",
      decision: `Start ${latestExperiment.model || "the first experiment"}.`,
      expectedOutcome: "Produce the first comparable metrics for champion selection.",
      confidenceLabel: "Plan ready",
      updatedAt: latestPlan.created_at,
    };
  }

  return {
    state: missionDatasetProfiled(dataset) ? "Reviewing dataset" : "Profiling dataset",
    observation: missionDatasetProfiled(dataset)
      ? "Dataset structure and label coverage are available."
      : "Model Express is checking the dataset before planning experiments.",
    reasoning: "A dataset review prevents random trials and makes the first experiments evidence-driven.",
    decision: missionDatasetProfiled(dataset) ? "Create the first experiment plan." : "Finish dataset review.",
    expectedOutcome: "The next step will choose experiments that match the mission goal.",
    confidenceLabel: "Reviewing",
    updatedAt: dataset.profiled_at || dataset.created_at || project.updated_at,
  };
}

export function buildMissionStages(
  project: Project | null,
  detail: ProjectDetail,
  digest: MissionDigest,
  exportDemo: ChampionExportDemo,
): MissionStage[] {
  const dataset = detail.datasets[0] ?? null;
  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const counts = jobStatusCounts(detail.jobs);
  const completed = counts.SUCCEEDED + counts.FAILED;
  const active = counts.QUEUED + counts.ASSIGNED + counts.RUNNING;
  const evaluationDone = detail.runEvaluations.length > 0 || detail.agentMemory.some((record) => record.kind === "training_evaluation");
  const followUpPlan = latestDecision ? detail.plans.find((plan) => plan.source_decision_id === latestDecision.id) : null;
  const exportReady = Boolean(readyONNXExport(exportDemo.exports));
  const demoValidated = exportDemo.demoPredictions.some((prediction) => normalizedStatus(prediction.status || "") === "SUCCEEDED");
  const blocked = digest.state === "blocked";

  const status = (done: boolean, activeNow: boolean): MissionStage["status"] =>
    blocked && activeNow ? "blocked" : done ? "done" : activeNow ? "active" : "waiting";

  return [
    {
      id: "created",
      label: "Mission Created",
      detail: project ? "Goal captured and ready for autonomous work." : "Create a mission to begin.",
      status: project ? "done" : "waiting",
      timestamp: project?.created_at,
    },
    {
      id: "dataset",
      label: "Dataset Reviewed",
      detail: dataset
        ? missionDatasetProfiled(dataset)
          ? datasetReviewSummary(dataset)
          : "Checking labels, image counts, and readiness."
        : "Waiting for an image dataset.",
      status: status(Boolean(dataset && missionDatasetProfiled(dataset)), Boolean(dataset && !missionDatasetProfiled(dataset))),
      timestamp: dataset?.profiled_at || dataset?.created_at,
    },
    {
      id: "plan",
      label: "Initial Plan Created",
      detail: latestPlan ? `${latestPlan.experiments.length} experiments selected for the first pass.` : "Starts after dataset review.",
      status: status(Boolean(latestPlan), Boolean(dataset && missionDatasetProfiled(dataset) && !latestPlan)),
      timestamp: latestPlan?.created_at,
    },
    {
      id: "experiments",
      label: "Initial Experiments",
      detail:
        detail.jobs.length > 0
          ? `${completed}/${detail.jobs.length} experiments complete${active > 0 ? `, ${active} still running.` : "."}`
          : "Experiments start after the plan is approved or auto-run.",
      status: status(detail.jobs.length > 0 && active === 0, active > 0 || Boolean(latestPlan && detail.jobs.length === 0)),
      timestamp: detail.jobs[0]?.created_at,
    },
    {
      id: "evaluation",
      label: "Evaluation",
      detail: evaluationDone ? "Completed runs have comparable evidence." : "Evaluation begins when experiments finish.",
      status: status(evaluationDone, completed > 0 && !evaluationDone),
      timestamp: detail.runEvaluations[0]?.created_at,
    },
    {
      id: "refinement",
      label: "Refinement",
      detail: followUpPlan
        ? `${followUpPlan.experiments.length} targeted follow-up experiment${followUpPlan.experiments.length === 1 ? "" : "s"} prepared.`
        : latestDecision
          ? userFacingActivityText(missionDecisionToUserText(latestDecision), 120)
          : "Starts when evaluation reveals a useful improvement target.",
      status: status(Boolean(followUpPlan), Boolean(latestDecision && latestDecision.decision_type === "ADD_EXPERIMENTS" && !followUpPlan)),
      timestamp: followUpPlan?.created_at || latestDecision?.created_at,
    },
    {
      id: "champion",
      label: "Champion Selection",
      detail: detail.champion ? "Best candidate selected for handoff." : "Starts after enough evidence is available.",
      status: status(Boolean(detail.champion), evaluationDone && !detail.champion),
      timestamp: detail.champion?.created_at,
    },
    {
      id: "export",
      label: "Export",
      detail: exportReady ? "ONNX handoff package is ready." : detail.champion ? "Preparing model handoff package." : "Waiting for champion selection.",
      status: status(exportReady, Boolean(detail.champion && !exportReady)),
      timestamp: readyONNXExport(exportDemo.exports)?.completed_at || readyONNXExport(exportDemo.exports)?.updated_at,
    },
    {
      id: "demo",
      label: "Demo Validation",
      detail: demoValidated ? "A demo image has been tested." : "Run a demo image after export readiness.",
      status: status(demoValidated, exportReady && !demoValidated),
      timestamp: exportDemo.demoPredictions[0]?.completed_at || exportDemo.demoPredictions[0]?.created_at,
    },
  ];
}

export function buildActivityFeed(
  project: Project | null,
  detail: ProjectDetail,
  events: AgentActivityEvent[],
  exportDemo: ChampionExportDemo,
): ActivityCardModel[] {
  const cards: ActivityCardModel[] = [];
  const push = (card: ActivityCardModel | null) => {
    if (card) cards.push(card);
  };

  if (project) {
    push({
      id: `mission-${project.id}`,
      type: "mission",
      title: "Mission started",
      summary: project.goal || "Model Express received the mission goal.",
      timestamp: project.created_at,
      status: "succeeded",
      evidenceSummary: "Goal and dataset setup are tracked in the mission workspace.",
      resultSummary: "",
      technicalSource: "Project record",
      developerPayloadRef: project.id,
    });
  }

  for (const event of events) push(activityCardFromEvent(event));
  for (const decision of detail.decisions.slice(0, 8)) push(activityCardFromDecision(decision));
  for (const summary of detail.runSummaries.slice(0, 10)) {
    const job = detail.jobs.find((item) => item.id === summary.job_id) ?? null;
    const evaluation = detail.runEvaluations.find((item) => item.job_id === summary.job_id) ?? null;
    push(activityCardFromRun(summary, evaluation, job));
  }
  if (detail.champion) {
    const champion = buildMissionChampionSummary(detail);
    push({
      id: `champion-${detail.champion.id}`,
      type: "result",
      title: "Best model selected",
      summary: `${champion?.model || "The champion"} leads with ${champion?.primaryMetricValue || "the best current score"} ${champion?.primaryMetricLabel || ""}.`.trim(),
      timestamp: detail.champion.created_at,
      status: "succeeded",
      evidenceSummary: userFacingActivityText(detail.champion.selection_reason || "Selected from completed experiment evidence.", 180),
      resultSummary: "Ready for results review and export preparation.",
      technicalSource: "Champion record",
      developerPayloadRef: detail.champion.id,
    });
  }
  const exportRecord = readyONNXExport(exportDemo.exports);
  if (exportRecord) {
    push({
      id: `export-${exportRecord.id || exportRecord.artifact_uri || "ready"}`,
      type: "export",
      title: "Export package prepared",
      summary: "The handoff package is ready for validation and use.",
      timestamp: exportRecord.completed_at || exportRecord.updated_at || exportRecord.created_at || new Date().toISOString(),
      status: "succeeded",
      evidenceSummary: "ONNX artifact readiness was reported.",
      resultSummary: "Open Export to run a demo image or inspect the handoff package.",
      technicalSource: "Export record",
      developerPayloadRef: exportRecord.id || exportRecord.artifact_uri || "",
    });
  }

  return uniqueBy(
    cards
      .filter((card) => Boolean(card.timestamp))
      .sort((left, right) => Date.parse(right.timestamp) - Date.parse(left.timestamp)),
    (card) => `${card.type}-${card.title}-${card.timestamp}`,
  ).slice(0, 30);
}

export function buildResultsSummary(
  detail: ProjectDetail,
  comparison: ChampionComparisonRow[],
  exportDemo: ChampionExportDemo,
): ResultsSummary {
  const runTotals = summarizeTrainingRuns(detail.runSummaries, detail.runEvaluations, detail.jobs);
  const champion = detail.champion ? buildMissionChampionSummary(detail) : undefined;
  const championRow = comparison.find((row) => row.isChampion) ?? comparison[0] ?? null;
  const baselineRow = comparison.find((row) => !row.isChampion) ?? comparison[comparison.length - 1] ?? null;
  const improvement =
    championRow && baselineRow && championRow.jobId !== baselineRow.jobId
      ? championRow.primaryMetricValue - baselineRow.primaryMetricValue
      : 0;
  const exportReady = Boolean(readyONNXExport(exportDemo.exports));
  const topCandidates = comparison
    .slice()
    .sort((left, right) => right.rankScore - left.rankScore)
    .slice(0, 3)
    .map((row, index) => ({
      rank: index + 1,
      model: row.model || `Candidate ${index + 1}`,
      metricLabel: row.primaryMetricLabel,
      metricValue: formatMaybeMetric(row.primaryMetricValue),
      status: row.isChampion ? "Champion" : row.divergenceStatus || "Candidate",
      why: row.isChampion
        ? "Best balance of mission score and deployment fit."
        : candidateWhyText(row),
      jobId: row.jobId,
    }));
  const learningSummary = resultsLearningSummary(detail);
  const remainingRisks = resultsRemainingRisks(detail, exportDemo);

  return {
    hasResults: detail.runSummaries.length > 0 || Boolean(detail.champion),
    championModel: champion?.model || championRow?.model || (runTotals.bestPrimaryMetricValue > 0 ? "Best candidate emerging" : "No champion yet"),
    primaryMetricLabel: champion?.primaryMetricLabel || championRow?.primaryMetricLabel || runTotals.bestPrimaryMetricLabel,
    primaryMetricValue:
      champion?.primaryMetricValue ||
      (championRow ? formatMaybeMetric(championRow.primaryMetricValue) : runTotals.bestPrimaryMetricValue > 0 ? formatMaybeMetric(runTotals.bestPrimaryMetricValue) : "Pending"),
    improvementLabel:
      improvement > 0
        ? `+${improvement.toFixed(3)} over nearest baseline`
        : detail.runSummaries.length > 1
          ? "Baseline comparison available"
          : "Baseline pending",
    exportStatus: exportReady ? "Export ready" : detail.champion ? "Preparing export" : "Waiting",
    whyItWon:
      detail.champion?.selection_reason
        ? userFacingActivityText(detail.champion.selection_reason, 220)
        : championRow
          ? "This candidate currently has the strongest mission score among completed experiments."
          : "Model Express is still collecting evidence before choosing a champion.",
    learningSummary,
    remainingRisks,
    topCandidates,
  };
}

export function buildExportSummary(detail: ProjectDetail, exportDemo: ChampionExportDemo): ExportSummary {
  const readyExport = readyONNXExport(exportDemo.exports);
  const hasChampion = Boolean(detail.champion);
  const validationErrors = exportDemo.exports.flatMap((item) => item.validation_errors ?? []);
  const demoSucceeded = exportDemo.demoPredictions.some((prediction) => normalizedStatus(prediction.status || "") === "SUCCEEDED");
  const includes = [
    readyExport ? "ONNX model" : "ONNX model request",
    "Preprocessing contract",
    "Label map",
    "Model card",
    exportDemo.demoImages.length > 0 ? "Held-out demo images" : "Demo image slot",
  ];
  return {
    hasChampion,
    title: hasChampion ? "Model handoff package" : "Waiting for champion",
    statusLabel: readyExport ? "Ready" : hasChampion ? "Preparing" : "Pending",
    readinessLabel: readyExport
      ? "The package has a ready ONNX artifact and can be validated with a demo image."
      : hasChampion
        ? "The champion is selected; prepare the ONNX package when you are ready."
        : "Export begins after Model Express selects the best model.",
    primaryFormat: readyExport?.format?.toUpperCase() || "ONNX",
    validationStatus: validationErrors.length > 0 ? "Needs review" : readyExport ? "Passed" : "Not run",
    demoStatus: demoSucceeded ? "Demo passed" : exportDemo.demoImages.length > 0 ? "Demo available" : "Waiting for image",
    includes,
    useCases: exportDemo.useCases.length > 0 ? exportDemo.useCases.map((item) => userFacingActivityText(item, 140)) : ["Production or prototype inference handoff."],
    limitations: exportDemo.limitations.length > 0 ? exportDemo.limitations.map((item) => userFacingActivityText(item, 140)) : ["Validate on your deployment data before broad rollout."],
    manifestAvailable: exportDemo.exports.some((item) => Object.keys(recordObject(recordObject(item.metadata).manifest)).length > 0),
  };
}

export function userFacingActivityText(value: string, maxLength = 180) {
  const text = activitySafeDisplayText(value, maxLength)
    .replace(/\borchestrator\b/gi, "AI engine")
    .replace(/\bcontrol plane\b/gi, "mission workspace")
    .replace(/\bplanner\b/gi, "AI")
    .replace(/\bagents?\b/gi, "AI")
    .replace(/\bLLM\b/g, "AI")
    .replace(/\bmemory retrieval\b/gi, "review of previous experiments")
    .replace(/\bmemory\b/gi, "prior evidence")
    .replace(/\binvocations?\b/gi, "model calls")
    .replace(/\btelemetry\b/gi, "usage details")
    .replace(/\bworkers?\b/gi, "training capacity")
    .replace(/\bjobs?\b/gi, "experiments")
    .replace(/\bbackend gate\b/gi, "safety check")
    .replace(/\bbackend\b/gi, "system");
  return text || "Model Express is updating the mission.";
}

export function userFacingActionLabel(value: string) {
  return value
    .replace(/\bOpen Workers\b/gi, "Review training capacity")
    .replace(/\bOpen Agents\b/gi, "Review AI work")
    .replace(/\bOpen Operations\b/gi, "View technical details")
    .replace(/\bOpen Runs\b/gi, "View results")
    .replace(/\bOpen Experiments\b/gi, "Review experiments");
}

export function firstProposedExperiment(payload: Record<string, unknown>): Record<string, unknown> {
  const proposed =
    Array.isArray(payload.proposed_experiments) ? payload.proposed_experiments :
    Array.isArray(payload.proposedExperiments) ? payload.proposedExperiments :
    [];
  return recordObject(proposed[0]);
}

export function missionDecisionToUserText(decision: AgentDecision, proposed = firstProposedExperiment(decision.payload ?? {})) {
  const decisionType = normalizedStatus(decision.decision_type);
  const model = recordString(proposed, "model");
  const mechanism = recordFirstString(proposed, ["mechanism", "selected_mechanism", "strategy"]);
  const intervention = recordString(proposed, "intervention");
  const imageSize = recordNumber(proposed, "image_size");
  if (decisionType === "ADD_EXPERIMENTS") {
    const pieces = [
      "Test",
      model || "a follow-up experiment",
      imageSize ? `at ${imageSize}px` : "",
      intervention || mechanism ? `for ${intervention || mechanism}` : "",
    ].filter(Boolean);
    return userFacingActivityText(pieces.join(" "), 150);
  }
  if (decisionType.includes("CHAMPION")) return "Select the strongest completed model as champion.";
  if (decisionType.includes("STOP")) return "Stop additional experiments and prepare the result.";
  if (decisionType.includes("WAIT")) return "Wait for more experiment evidence before changing course.";
  return userFacingActivityText(decisionHistorySummary(decision), 150);
}

export function decisionRationaleSummary(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  return (
    recordFirstString(payload, [
      "rationale",
      "reason",
      "summary",
      "why_can_beat_champion",
      "deterministic_diagnosis_used",
      "recommendation_summary",
    ]) ||
    decision.rationale ||
    "The AI compared current evidence against the mission goal and selected the next useful step."
  );
}

export function datasetReviewSummary(dataset: Dataset) {
  const profile = recordObject(dataset.profile);
  const totalImages = recordNumber(profile, "total_images");
  const classCount = recordNumber(profile, "class_count");
  const pieces = [
    totalImages ? `${totalImages} images` : "",
    classCount ? `${classCount} classes` : "",
    "ready for experiment planning",
  ].filter(Boolean);
  return pieces.join(", ");
}

export function activityCardFromEvent(event: AgentActivityEvent): ActivityCardModel {
  const text = `${event.type} ${event.title} ${event.message}`.toLowerCase();
  const status = activityStatus(event.status) as ActivityCardModel["status"];
  let type: ActivityCardType = "mission";
  let title = "Mission update";
  if (text.includes("champion")) {
    type = "result";
    title = "Best model selected";
  } else if (text.includes("export")) {
    type = "export";
    title = "Export prepared";
  } else if (text.includes("blocked") || text.includes("rejected") || text.includes("failed")) {
    type = "blocker";
    title = "Proposed work was blocked";
  } else if (text.includes("decision") || text.includes("recommendation") || text.includes("plan")) {
    type = "decision";
    title = "Decision recorded";
  } else if (text.includes("training") || text.includes("job") || text.includes("run") || text.includes("queued")) {
    type = "experiment";
    title = status === "succeeded" ? "Experiment finished" : "Experiment started";
  } else if (text.includes("dataset") || text.includes("profile") || text.includes("memory") || text.includes("retrieval")) {
    type = "observation";
    title = text.includes("dataset") ? "Dataset observation" : "AI reviewed previous experiments";
  }
  return {
    id: `event-${event.id}`,
    type,
    title,
    summary: userFacingActivityText(event.message || event.title || title, 220),
    timestamp: event.created_at,
    status,
    evidenceSummary: `Updated ${formatRelativeTime(event.created_at)}`,
    resultSummary: status === "succeeded" ? "Completed" : status === "blocked" || status === "failed" ? "Needs review" : "In progress",
    technicalSource: "Project event",
    developerPayloadRef: event.id,
  };
}

export function activityCardFromDecision(decision: AgentDecision): ActivityCardModel {
  const rejections = decisionRejections(decision);
  const blocked = rejections.length > 0;
  return {
    id: `decision-${decision.id}`,
    type: blocked ? "blocker" : "decision",
    title: blocked ? "Proposed experiment was blocked" : "Decision recorded",
    summary: missionDecisionToUserText(decision),
    timestamp: decision.created_at,
    status: blocked ? "blocked" : "succeeded",
    evidenceSummary: userFacingActivityText(decisionRationaleSummary(decision), 160),
    resultSummary: blocked ? userFacingActivityText(rejections[0]?.text || "The proposal did not pass safety checks.", 140) : "Next step is stored for the mission.",
    technicalSource: "AI decision",
    developerPayloadRef: decision.id,
  };
}

export function activityCardFromRun(
  summary: TrainingRunSummary,
  evaluation: TrainingRunEvaluation | null,
  job: Job | null,
): ActivityCardModel {
  const runStatus = effectiveTrainingRunStatus(summary, job);
  const status = activityStatus(runStatus);
  const primary = runPrimaryMetric(summary, evaluation, job);
  const terminal = ["SUCCEEDED", "FAILED"].includes(runStatus);
  return {
    id: `run-${summary.job_id}`,
    type: terminal ? "result" : "experiment",
    title: terminal ? "Experiment finished" : "Experiment started",
    summary: `${summary.model || "Candidate model"} ${terminal ? "finished" : "is running"} with ${primary.label} ${formatMaybeMetric(primary.value)}.`,
    timestamp: summary.updated_at || summary.created_at,
    status: status as ActivityCardModel["status"],
    evidenceSummary: `${summary.epochs_completed} epochs completed`,
    resultSummary: primary.value > 0 ? `${formatMaybeMetric(primary.value)} ${primary.label}` : "Metric pending",
    technicalSource: "Training result",
    developerPayloadRef: summary.job_id,
  };
}

export function activityCardMatchesFilter(card: ActivityCardModel, filter: ActivityFilterKey) {
  if (filter === "all") return true;
  if (filter === "decisions") return card.type === "decision";
  if (filter === "experiments") return card.type === "experiment";
  if (filter === "results") return card.type === "result" || card.type === "export";
  if (filter === "blockers") return card.type === "blocker" || card.status === "blocked" || card.status === "failed";
  return true;
}

export function candidateWhyText(row: ChampionComparisonRow) {
  const details = [
    row.latencyMs > 0 ? `${formatLatency(row.latencyMs)} latency` : "",
    row.costUsd > 0 ? `${formatCurrency(row.costUsd)} cost` : "",
    row.objectiveFit > 0 ? `${formatMaybeMetric(row.objectiveFit)} mission fit` : "",
  ].filter(Boolean);
  return details.length > 0 ? `Strong candidate with ${details.join(", ")}.` : "Comparable result from a completed experiment.";
}

export function resultsLearningSummary(detail: ProjectDetail) {
  const latestDecision = detail.decisions[0] ?? null;
  const latestEvaluation = detail.runEvaluations[0] ?? null;
  const latestMemory = detail.agentMemory[0] ?? null;
  const items = uniqueStrings([
    latestDecision ? userFacingActivityText(decisionRationaleSummary(latestDecision), 160) : "",
    latestEvaluation?.recommendation_summary ? userFacingActivityText(latestEvaluation.recommendation_summary, 160) : "",
    latestMemory?.summary ? userFacingActivityText(latestMemory.summary, 160) : "",
    detail.runSummaries.length > 0 ? `${detail.runSummaries.length} experiment${detail.runSummaries.length === 1 ? "" : "s"} produced comparable evidence.` : "",
  ]);
  return items.length > 0 ? items.slice(0, 4) : ["Learning summary will appear after the first completed experiment."];
}

export function resultsRemainingRisks(detail: ProjectDetail, exportDemo: ChampionExportDemo) {
  const latestDecision = detail.decisions[0] ?? null;
  const risks = uniqueStrings([
    ...exportDemo.limitations.map((item) => userFacingActivityText(item, 140)),
    ...stringArrayPayload(latestDecision?.payload.risks).map((item) => userFacingActivityText(item, 140)),
    detail.jobs.some((job) => normalizedStatus(job.status) === "FAILED")
      ? "Some experiments failed and should be reviewed before relying on the result."
      : "",
    exportDemo.demoPredictions.length === 0 ? "Run a demo image before deployment handoff." : "",
  ]);
  return risks.length > 0 ? risks.slice(0, 4) : ["No major risk has been summarized yet; validate on real deployment data."];
}

export function buildMissionDigest({
  health,
  selectedProject,
  detail,
  automationSettings,
  activityStreamState,
  visibleActivityEvents,
  loading,
}: {
  health: Health | null;
  selectedProject: Project | null;
  detail: ProjectDetail;
  automationSettings: AutomationSettings;
  activityStreamState: ActivityStreamState;
  visibleActivityEvents: AgentActivityEvent[];
  loading: boolean;
}): MissionDigest {
  const dataset = detail.datasets[0] ?? null;
  const datasetProfiled = missionDatasetProfiled(dataset);
  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const latestDecisionHasFollowUpPlan = latestDecision
    ? detail.plans.some((plan) => plan.source_decision_id === latestDecision.id)
    : false;
  const counts = jobStatusCounts(detail.jobs);
  const queuedJobs = counts.QUEUED;
  const activeJobs = counts.ASSIGNED + counts.RUNNING;
  const terminalJobs = counts.SUCCEEDED + counts.FAILED;
  const runTotals = summarizeTrainingRuns(detail.runSummaries, detail.runEvaluations, detail.jobs);
  const workerSummary = missionWorkerSummary(detail.workers, detail.workerRequirements);
  const latestTerminalTimestamp = Math.max(
    ...detail.jobs
      .filter((job) => ["SUCCEEDED", "FAILED"].includes(normalizedStatus(job.status)))
      .map((job) => timestampSortScore(job.completed_at || job.started_at || job.created_at)),
    0,
  );
  const latestDecisionTimestamp = timestampSortScore(latestDecision?.created_at);
  const latestDecisionStale = terminalJobs > 0 && (!latestDecision || latestDecisionTimestamp < latestTerminalTimestamp);
  const validationStatus = latestDecision
    ? recordFirstString(latestDecision.payload ?? {}, ["backend_validation_status", "validation_status", "planner_validation_status"]).toLowerCase()
    : "";
  const blockingDecision =
    latestDecision &&
    decisionRejections(latestDecision).length > 0 &&
    validationStatus &&
    !["accepted", "approved", "valid"].includes(validationStatus);
  const activeFailedEvent = visibleActivityEvents.find(
    (event) => ["failed", "blocked"].includes(activityStatus(event.status)) && !isChampionSelectedGuardActivity(event),
  );
  const failedWithoutProgress = counts.FAILED > 0 && counts.SUCCEEDED === 0 && activeJobs === 0 && queuedJobs === 0;
  const orchestratorUnhealthy = Boolean(health && health.status !== "ok");
  const workerCapacityBlocked = workerSummary.failed && (queuedJobs > 0 || activeJobs > 0);
  const hardBlocked = orchestratorUnhealthy || Boolean(activeFailedEvent) || Boolean(blockingDecision) || workerCapacityBlocked || failedWithoutProgress;

  let state: MissionDigestState = "plan_ready";
  if (hardBlocked) {
    state = "blocked";
  } else if (!selectedProject) {
    state = "empty";
  } else if (!dataset) {
    state = "dataset_needed";
  } else if (!datasetProfiled) {
    state = "profiling";
  } else if (queuedJobs > 0 || activeJobs > 0) {
    state = "training";
  } else if (latestDecision?.decision_type === "ADD_EXPERIMENTS" && !latestDecisionHasFollowUpPlan) {
    state = "follow_up_ready";
  } else if (detail.champion) {
    state = "champion_selected";
  } else if (latestDecisionStale || (terminalJobs > 0 && !latestDecision)) {
    state = "review_needed";
  } else if (latestPlan) {
    state = "plan_ready";
  } else {
    state = datasetProfiled ? "plan_ready" : "profiling";
  }

  const facts = missionFacts({
    state,
    dataset,
    datasetProfiled,
    latestPlan,
    latestDecision,
    runTotals,
    queuedJobs,
    activeJobs,
    terminalJobs,
    workerSummary,
    automationSettings,
  });
  const liveActivity = buildMissionLiveActivity({ detail, streamState: activityStreamState, visibleActivityEvents });
  const stateCopy = missionStateCopy({
    state,
    dataset,
    latestPlan,
    latestDecision,
    runTotals,
    queuedJobs,
    activeJobs,
    terminalJobs,
    workerSummary,
    workerCapacityBlocked,
    orchestratorUnhealthy,
    activeFailedEvent,
    blockingDecision: Boolean(blockingDecision),
  });

  return {
    state,
    stateLabel: missionStateLabel(state),
    healthLabel: hardBlocked ? "Needs attention" : missionHealthLabel(health),
    headline: stateCopy.headline,
    detail: stateCopy.detail,
    facts,
    health: buildMissionHealth({
      health,
      dataset,
      datasetProfiled,
      counts,
      workerSummary,
      liveActivity,
      detail,
      latestPlan,
      latestDecision,
    }),
    nextActions: buildMissionNextActions({
      state,
      loading,
      latestPlan,
      latestDecision,
      latestDecisionHasFollowUpPlan,
      selectedProject,
      dataset,
      queuedJobs,
      activeJobs,
      workerSummary,
      workerCapacityBlocked,
      orchestratorUnhealthy,
      blockingDecision: Boolean(blockingDecision),
    }),
    liveActivity,
    recentSignals: buildMissionSignals({
      state,
      health,
      dataset,
      datasetProfiled,
      latestPlan,
      latestDecision,
      counts,
      detail,
      visibleActivityEvents,
      activeFailedEvent,
      blockingDecision: Boolean(blockingDecision),
    }),
    champion: detail.champion ? buildMissionChampionSummary(detail) : undefined,
  };
}

export function buildMissionLiveActivity({
  detail,
  streamState,
  visibleActivityEvents,
}: {
  detail: ProjectDetail;
  streamState: ActivityStreamState;
  visibleActivityEvents: AgentActivityEvent[];
}): MissionLiveActivity {
  const counts = jobStatusCounts(detail.jobs);
  const eventSteps = visibleActivityEvents.slice(0, 5).map(missionLiveStepFromEvent);
  const fallbackSteps: MissionLiveActivity["steps"] = [];
  if (eventSteps.length === 0 && detail.agentMemory[0]) {
    fallbackSteps.push({
      id: `memory-${detail.agentMemory[0].id}`,
      label: "Retrieving memories",
      status: "succeeded",
      timestamp: detail.agentMemory[0].created_at,
      targetTab: "agents",
      targetId: "memory-retrieval-probe",
    });
  }
  if (eventSteps.length === 0 && detail.decisions[0]) {
    fallbackSteps.push({
      id: `decision-${detail.decisions[0].id}`,
      label: "Writing decision",
      status: "succeeded",
      timestamp: detail.decisions[0].created_at,
      targetTab: "agents",
      targetId: "agent-decisions",
    });
  }
  const steps = uniqueBy([...eventSteps, ...fallbackSteps], (step) => step.label).slice(0, 5);
  const activeStep = steps.find((step) => ["active", "waiting", "blocked", "failed"].includes(step.status));
  const latestStep = activeStep ?? steps[0] ?? null;
  const runningJobs = counts.ASSIGNED + counts.RUNNING;

  if (runningJobs > 0) {
    return {
      status: "active",
      label: "Waiting for training results",
      detail: `${runningJobs} training job${runningJobs === 1 ? "" : "s"} active; workers are expected to report metrics.`,
      streamState,
      steps: missionEnsureLiveStep(steps, {
        id: "jobs-running-live",
        label: "Waiting for training results",
        status: "active",
        targetTab: "experiments",
        targetId: "runs",
      }),
    };
  }
  if (counts.QUEUED > 0) {
    return {
      status: "waiting",
      label: "Scheduling workers",
      detail: `${counts.QUEUED} queued job${counts.QUEUED === 1 ? "" : "s"} waiting for worker capacity.`,
      streamState,
      steps: missionEnsureLiveStep(steps, {
        id: "jobs-queued-live",
        label: "Scheduling workers",
        status: "waiting",
        targetTab: "operations",
        targetId: "workers",
      }),
    };
  }
  if (latestStep) {
    return {
      status: latestStep.status,
      label: latestStep.label,
      detail: missionLiveActivityDetail(latestStep.label, latestStep.status),
      streamState,
      steps,
    };
  }
  if (streamState === "connecting" || streamState === "connected") {
    return {
      status: streamState === "connecting" ? "waiting" : "idle",
      label: streamState === "connecting" ? "Connecting" : "Listening",
      detail: streamState === "connecting" ? "Opening the live activity stream." : "No agentic work is active right now.",
      streamState,
      steps: [],
    };
  }
  if (streamState === "reconnecting" || streamState === "fallback") {
    return {
      status: "waiting",
      label: streamState === "reconnecting" ? "Reconnecting" : "Polling",
      detail: "Mission Control is still refreshing project state while the stream is unavailable.",
      streamState,
      steps: [],
    };
  }
  return {
    status: "idle",
    label: "Idle",
    detail: "No planner, memory, validation, worker, or training activity is active.",
    streamState,
    steps: [],
  };
}

export function missionDatasetProfiled(dataset: Dataset | null) {
  if (!dataset) return false;
  if (dataset.profiled_at) return true;
  if (normalizedStatus(dataset.status) === "PROFILED") return true;
  const profile = recordObject(dataset.profile);
  return recordNumber(profile, "total_images") > 0 && recordNumber(profile, "class_count") > 0;
}

export function missionWorkerSummary(workers: Worker[], requirements: WorkerRequirement[]) {
  const activeWorkers = workers.filter((worker) => worker.current_job_id || ["ACTIVE", "RUNNING", "BUSY"].includes(normalizedStatus(worker.status))).length;
  const availableWorkers = workers.filter((worker) => {
    const status = normalizedStatus(worker.status);
    return !["FAILED", "OFFLINE", "ERROR", "STOPPED", "CANCELLED"].includes(status);
  }).length;
  const failedWorkers = workers.filter((worker) => ["FAILED", "OFFLINE", "ERROR", "STOPPED", "CANCELLED"].includes(normalizedStatus(worker.status))).length;
  const failedRequirements = requirements.filter((requirement) =>
    ["FAILED", "ERROR", "BLOCKED", "CANCELLED"].includes(normalizedStatus(requirement.status)),
  ).length;
  const pendingRequirements = requirements.filter((requirement) =>
    ["PENDING", "REQUESTED", "WORKERS_REQUIRED", "STARTING", "WORKERS_STARTING"].includes(normalizedStatus(requirement.status)),
  ).length;
  return {
    totalWorkers: workers.length,
    activeWorkers,
    availableWorkers,
    failedWorkers,
    pendingRequirements,
    failedRequirements,
    failed: failedWorkers > 0 || failedRequirements > 0,
  };
}

export function missionFacts({
  state,
  dataset,
  datasetProfiled,
  latestPlan,
  latestDecision,
  runTotals,
  queuedJobs,
  activeJobs,
  terminalJobs,
  workerSummary,
  automationSettings,
}: {
  state: MissionDigestState;
  dataset: Dataset | null;
  datasetProfiled: boolean;
  latestPlan: ExperimentPlan | null;
  latestDecision: AgentDecision | null;
  runTotals: ReturnType<typeof summarizeTrainingRuns>;
  queuedJobs: number;
  activeJobs: number;
  terminalJobs: number;
  workerSummary: ReturnType<typeof missionWorkerSummary>;
  automationSettings: AutomationSettings;
}): MissionDigest["facts"] {
  const profile = recordObject(dataset?.profile);
  const totalImages = recordNumber(profile, "total_images");
  const classCount = recordNumber(profile, "class_count");
  const facts: MissionDigest["facts"] = [];
  if (dataset) {
    facts.push({
      label: "Dataset",
      value: datasetProfiled
        ? [totalImages ? `${totalImages} images` : "", classCount ? `${classCount} classes` : ""].filter(Boolean).join(" / ") || "Profile ready"
        : "Profiling pending",
      tone: datasetProfiled ? "ok" : "warning",
    });
  } else {
    facts.push({ label: "Dataset", value: state === "empty" ? "No project selected" : "Upload needed", tone: "warning" });
  }
  if (latestPlan) {
    facts.push({
      label: "Plan",
      value: `${latestPlan.experiments.length} experiment${latestPlan.experiments.length === 1 ? "" : "s"} / ${latestPlan.target_metric}`,
      tone: "info",
    });
  } else {
    facts.push({ label: "Plan", value: datasetProfiled ? "Waiting for proposal" : "Needs profile", tone: datasetProfiled ? "warning" : "info" });
  }
  if (activeJobs > 0 || queuedJobs > 0 || terminalJobs > 0) {
    facts.push({
      label: "Runs",
      value: activeJobs > 0
        ? `${activeJobs} active, ${queuedJobs} queued`
        : queuedJobs > 0
          ? `${queuedJobs} queued`
          : `${terminalJobs} completed`,
      tone: activeJobs > 0 ? "ok" : queuedJobs > 0 ? "warning" : "info",
    });
  } else {
    facts.push({ label: "Runs", value: "Not launched", tone: latestPlan ? "warning" : "info" });
  }
  if (runTotals.bestPrimaryMetricValue > 0) {
    facts.push({ label: `Best ${runTotals.bestPrimaryMetricLabel}`, value: formatMaybeMetric(runTotals.bestPrimaryMetricValue), tone: "ok" });
  } else if (workerSummary.availableWorkers > 0 || workerSummary.pendingRequirements > 0) {
    facts.push({
      label: "Workers",
      value: workerSummary.activeWorkers > 0
        ? `${workerSummary.activeWorkers} active`
        : workerSummary.pendingRequirements > 0
          ? "Starting"
          : `${workerSummary.availableWorkers} available`,
      tone: workerSummary.failed ? "bad" : "info",
    });
  } else {
    facts.push({
      label: "Automation",
      value: automationSettings.auto_execute_plans ? "Auto execute on" : "Manual execute",
      tone: automationSettings.auto_execute_plans ? "ok" : "info",
    });
  }
  if (latestDecision && facts.length < 4) {
    facts.push({ label: "Decision", value: missionDecisionLabel(latestDecision), tone: decisionRejections(latestDecision).length > 0 ? "warning" : "info" });
  }
  return facts.slice(0, 4);
}

export function missionStateCopy({
  state,
  dataset,
  latestPlan,
  latestDecision,
  runTotals,
  queuedJobs,
  activeJobs,
  terminalJobs,
  workerSummary,
  workerCapacityBlocked,
  orchestratorUnhealthy,
  activeFailedEvent,
  blockingDecision,
}: {
  state: MissionDigestState;
  dataset: Dataset | null;
  latestPlan: ExperimentPlan | null;
  latestDecision: AgentDecision | null;
  runTotals: ReturnType<typeof summarizeTrainingRuns>;
  queuedJobs: number;
  activeJobs: number;
  terminalJobs: number;
  workerSummary: ReturnType<typeof missionWorkerSummary>;
  workerCapacityBlocked: boolean;
  orchestratorUnhealthy: boolean;
  activeFailedEvent?: AgentActivityEvent;
  blockingDecision: boolean;
}) {
  if (state === "blocked") {
    if (orchestratorUnhealthy) {
      return {
        headline: "Mission Control cannot reach the orchestrator.",
        detail: "Refresh the connection or verify the orchestrator URL before scheduling more work.",
      };
    }
    if (workerCapacityBlocked) {
      return {
        headline: "Worker supervision needs attention.",
        detail: "A worker or requirement reported a failure. Open Operations for the exact worker state.",
      };
    }
    if (blockingDecision || activeFailedEvent) {
      return {
        headline: "Agentic work is blocked.",
        detail: "A planner, validation, or execution failure needs review in the agent and operations drill-downs.",
      };
    }
    return {
      headline: "Training did not make usable progress.",
      detail: "The latest runs failed before a successful result was reported. Review the run and worker details.",
    };
  }
  if (state === "empty") {
    return {
      headline: "No project is selected.",
      detail: "Create or choose a project with an image-folder dataset to start profiling and planning.",
    };
  }
  if (state === "dataset_needed") {
    return {
      headline: "This project needs a dataset.",
      detail: "Mission Control cannot profile, plan, or train until a dataset is attached.",
    };
  }
  if (state === "profiling") {
    return {
      headline: "Dataset profiling is in progress.",
      detail: "The dataset is waiting for class counts and image statistics before experiments are planned.",
    };
  }
  if (state === "training") {
    return {
      headline: activeJobs > 0 ? "Experiments are training." : "Experiments are queued.",
      detail: activeJobs > 0
        ? `${activeJobs} training job${activeJobs === 1 ? "" : "s"} running${runTotals.bestPrimaryMetricValue > 0 ? `; best ${runTotals.bestPrimaryMetricLabel} is ${formatMaybeMetric(runTotals.bestPrimaryMetricValue)}.` : "."}`
        : `${queuedJobs} queued job${queuedJobs === 1 ? "" : "s"} waiting for worker capacity.`,
    };
  }
  if (state === "follow_up_ready") {
    return {
      headline: "The planner proposed follow-up experiments.",
      detail: "Backend validation has a stored decision ready for scheduling through the existing follow-up action.",
    };
  }
  if (state === "champion_selected") {
    return {
      headline: "A champion has been selected.",
      detail: `The best reported ${runTotals.bestPrimaryMetricLabel} is ${formatMaybeMetric(runTotals.bestPrimaryMetricValue)}. Export, demo, or compare it against completed runs.`,
    };
  }
  if (state === "review_needed") {
    return {
      headline: "Completed runs need review.",
      detail: `${terminalJobs} terminal run${terminalJobs === 1 ? "" : "s"} are ready for the experiment reviewer to compare.`,
    };
  }
  if (latestPlan) {
    return {
      headline: "An experiment plan is ready.",
      detail: `${latestPlan.experiments.length} experiment${latestPlan.experiments.length === 1 ? "" : "s"} target ${latestPlan.target_metric}; launch them or inspect the plan first.`,
    };
  }
  return {
    headline: latestDecision ? "Mission Control is waiting for the next plan." : "The profiled dataset is ready for planning.",
    detail: "Open Experiments to inspect planning state and start the next backend-backed action.",
  };
}

export function buildMissionHealth({
  health,
  dataset,
  datasetProfiled,
  counts,
  workerSummary,
  liveActivity,
  detail,
  latestPlan,
  latestDecision,
}: {
  health: Health | null;
  dataset: Dataset | null;
  datasetProfiled: boolean;
  counts: ReturnType<typeof jobStatusCounts>;
  workerSummary: ReturnType<typeof missionWorkerSummary>;
  liveActivity: MissionLiveActivity;
  detail: ProjectDetail;
  latestPlan: ExperimentPlan | null;
  latestDecision: AgentDecision | null;
}): MissionHealthItem[] {
  const runActive = counts.ASSIGNED + counts.RUNNING;
  const items: MissionHealthItem[] = [
    {
      id: "orchestrator",
      label: "Orchestrator",
      value: health?.status === "ok" ? "Online" : health ? "Offline" : "Checking",
      tone: health?.status === "ok" ? "ok" : health ? "bad" : "warning",
      targetTab: "operations",
      targetId: "operations",
    },
    {
      id: "dataset",
      label: "Dataset",
      value: dataset ? (datasetProfiled ? "Profiled" : "Profiling") : "Needed",
      tone: dataset ? (datasetProfiled ? "ok" : "warning") : "warning",
      targetTab: "data",
      targetId: "data",
    },
  ];
  const candidates: MissionHealthItem[] = [];
  if (latestPlan || detail.jobs.length > 0 || detail.champion) {
    candidates.push({
      id: "runs",
      label: "Runs",
      value: counts.FAILED > 0 && counts.SUCCEEDED === 0 && runActive === 0 && counts.QUEUED === 0
        ? "Failed"
        : runActive > 0
          ? `${runActive} active`
          : counts.QUEUED > 0
            ? `${counts.QUEUED} queued`
            : counts.SUCCEEDED > 0
              ? `${counts.SUCCEEDED} done`
              : "Not launched",
      tone: counts.FAILED > 0 && counts.SUCCEEDED === 0 ? "bad" : runActive > 0 ? "ok" : counts.QUEUED > 0 ? "warning" : "info",
      targetTab: "experiments",
      targetId: "runs",
    });
  }
  if (detail.workers.length > 0 || detail.workerRequirements.length > 0 || counts.QUEUED > 0 || runActive > 0) {
    candidates.push({
      id: "workers",
      label: "Workers",
      value: workerSummary.failed
        ? "Needs review"
        : counts.QUEUED > 0 && workerSummary.availableWorkers === 0
          ? "No active workers"
          : workerSummary.activeWorkers > 0
            ? `${workerSummary.activeWorkers} active`
            : workerSummary.pendingRequirements > 0
              ? "Starting"
              : workerSummary.availableWorkers > 0
                ? `${workerSummary.availableWorkers} available`
                : "Idle",
      tone: workerSummary.failed || (counts.QUEUED > 0 && workerSummary.availableWorkers === 0) ? "warning" : workerSummary.activeWorkers > 0 ? "ok" : "info",
      targetTab: "operations",
      targetId: "workers",
    });
  }
  if (
    liveActivity.steps.length > 0 ||
    detail.agentInvocations.length > 0 ||
    detail.decisions.length > 0 ||
    detail.agentMemory.length > 0 ||
    latestDecision
  ) {
    candidates.push({
      id: "agents",
      label: "Agents",
      value: liveActivity.status === "failed" || liveActivity.status === "blocked"
        ? "Needs review"
        : liveActivity.status === "active"
          ? "Active"
          : liveActivity.status === "waiting"
            ? "Waiting"
            : "Idle",
      tone: liveActivity.status === "failed" || liveActivity.status === "blocked" ? "bad" : liveActivity.status === "waiting" ? "warning" : liveActivity.status === "active" ? "ok" : "info",
      targetTab: "agents",
      targetId: "agent-activity",
    });
  }
  return [...items, ...candidates.sort((left, right) => missionToneRank(left.tone) - missionToneRank(right.tone)).slice(0, 2)];
}

export function buildMissionNextActions({
  state,
  loading,
  latestPlan,
  latestDecision,
  latestDecisionHasFollowUpPlan,
  selectedProject,
  dataset,
  queuedJobs,
  activeJobs,
  workerSummary,
  workerCapacityBlocked,
  orchestratorUnhealthy,
  blockingDecision,
}: {
  state: MissionDigestState;
  loading: boolean;
  latestPlan: ExperimentPlan | null;
  latestDecision: AgentDecision | null;
  latestDecisionHasFollowUpPlan: boolean;
  selectedProject: Project | null;
  dataset: Dataset | null;
  queuedJobs: number;
  activeJobs: number;
  workerSummary: ReturnType<typeof missionWorkerSummary>;
  workerCapacityBlocked: boolean;
  orchestratorUnhealthy: boolean;
  blockingDecision: boolean;
}): MissionNextAction[] {
  const action = (
    id: string,
    label: string,
    reason: string,
    priority: MissionNextAction["priority"],
    options: Partial<MissionNextAction> = {},
  ): MissionNextAction => ({ id, label, reason, priority, ...options });

  if (state === "blocked") {
    const primary = orchestratorUnhealthy
      ? action("refresh-orchestrator", "Refresh connection", "Mission Control needs a healthy orchestrator before work can continue.", "primary", {
          actionKey: "refresh",
          disabled: loading,
        })
      : workerCapacityBlocked || (queuedJobs > 0 && workerSummary.availableWorkers === 0)
        ? action("open-workers", "Open Workers", "Worker state explains why queued work is not moving.", "primary", {
            targetTab: "operations",
            targetId: "workers",
          })
        : action("open-agent-review", "Open Agents", blockingDecision ? "Backend validation detail is in the agent audit trail." : "The failure detail is in the live activity and decision views.", "primary", {
            targetTab: "agents",
            targetId: "agent-activity",
          });
    return [
      primary,
      action("open-operations", "Open Operations", "Inspect workers, automation settings, and execution events.", "secondary", {
        targetTab: "operations",
        targetId: "automation-timeline",
      }),
      action("open-runs", "Open Runs", "Check the latest job and run summaries.", "secondary", {
        targetTab: "experiments",
        targetId: "runs",
      }),
    ];
  }
  if (state === "empty" || state === "dataset_needed") {
    return [
      action("new-project", state === "empty" ? "New Project" : "Create project with dataset", "Dataset upload is part of the project setup flow.", "primary", {
        actionKey: "new_project",
        disabled: loading,
      }),
      action("refresh-projects", "Refresh", "Reload projects and orchestrator state.", "secondary", {
        actionKey: "refresh",
        disabled: loading,
      }),
    ];
  }
  if (state === "profiling") {
    return [
      action("open-data", "Open Data", "Dataset profile, metadata, and visual analysis live there.", "primary", {
        targetTab: "data",
        targetId: "data",
      }),
      action("open-operations", "Open Operations", "Check profiling workers and execution events if profiling stalls.", "secondary", {
        targetTab: "operations",
        targetId: "workers",
      }),
    ];
  }
  if (state === "plan_ready") {
    return [
      latestPlan
        ? action("execute-plan", "Execute Plan", "The latest plan has not launched training jobs yet.", "primary", {
            actionKey: "execute_plan",
            disabled: loading || !selectedProject || !dataset,
          })
        : action("open-experiments", "Open Experiments", "Inspect planning state and create or review the next plan.", "primary", {
            targetTab: "experiments",
            targetId: "plans",
          }),
      action("inspect-plan", "Inspect Plan", "Review experiment templates, metrics, and warnings before launch.", "secondary", {
        targetTab: "experiments",
        targetId: "plans",
      }),
      action("open-agents", "Open Agents", "Review planner decisions and backend gates.", "secondary", {
        targetTab: "agents",
        targetId: "agent-decisions",
      }),
    ];
  }
  if (state === "training") {
    const queuedWithoutWorkers = queuedJobs > 0 && activeJobs === 0 && workerSummary.availableWorkers === 0;
    return [
      queuedWithoutWorkers
        ? action("open-workers", "Open Workers", "Queued runs need worker capacity before training can start.", "primary", {
            targetTab: "operations",
            targetId: "workers",
          })
        : action("open-runs", "Open Runs", "Training summaries and metrics show current progress.", "primary", {
            targetTab: "experiments",
            targetId: "runs",
          }),
      action("open-activity", "Open Activity", "Follow planner, worker, and validation events in detail.", "secondary", {
        targetTab: "agents",
        targetId: "agent-activity",
      }),
      action("open-operations", "Open Operations", "Inspect worker requirements and execution events.", "secondary", {
        targetTab: "operations",
        targetId: "automation-timeline",
      }),
    ];
  }
  if (state === "review_needed") {
    return [
      action("review-experiments", "Review Experiments", "Completed runs need the reviewer to choose stop, champion, or follow-up.", "primary", {
        actionKey: "review_experiments",
        disabled: loading || !selectedProject,
      }),
      action("open-runs", "Open Runs", "Compare completed training summaries before review.", "secondary", {
        targetTab: "experiments",
        targetId: "runs",
      }),
    ];
  }
  if (state === "follow_up_ready") {
    return [
      action("schedule-follow-up", "Schedule Follow-up", "The latest decision proposed additional backend-validated experiments.", "primary", {
        actionKey: "schedule_follow_up",
        disabled: loading || !latestDecision || latestDecisionHasFollowUpPlan,
      }),
      action("open-agents", "Open Agents", "Read the decision summary and backend gate outcome.", "secondary", {
        targetTab: "agents",
        targetId: "agent-decisions",
      }),
      action("open-plan", "Open Experiments", "Inspect current and follow-up plans.", "secondary", {
        targetTab: "experiments",
        targetId: "plans",
      }),
    ];
  }
  return [
    action("open-export", "Open Export", "Use, demo, export, or validate the selected champion.", "primary", {
      actionKey: "open_export",
      targetTab: "export",
      targetId: "export-demo",
    }),
    action("compare-runs", "Compare Runs", "See why this champion beat other completed runs.", "secondary", {
      targetTab: "experiments",
      targetId: "champion-comparison",
    }),
    action("open-agents", "Open Agents", "Review the decision trail behind the champion.", "secondary", {
      targetTab: "agents",
      targetId: "agent-decisions",
    }),
  ];
}

export function buildMissionSignals({
  state,
  health,
  dataset,
  datasetProfiled,
  latestPlan,
  latestDecision,
  counts,
  detail,
  visibleActivityEvents,
  activeFailedEvent,
  blockingDecision,
}: {
  state: MissionDigestState;
  health: Health | null;
  dataset: Dataset | null;
  datasetProfiled: boolean;
  latestPlan: ExperimentPlan | null;
  latestDecision: AgentDecision | null;
  counts: ReturnType<typeof jobStatusCounts>;
  detail: ProjectDetail;
  visibleActivityEvents: AgentActivityEvent[];
  activeFailedEvent?: AgentActivityEvent;
  blockingDecision: boolean;
}): MissionSignal[] {
  const signals: MissionSignal[] = [];
  const push = (signal: MissionSignal) => signals.push(signal);
  if (health && health.status !== "ok") {
    push({
      id: "orchestrator-unhealthy",
      label: "Orchestrator offline",
      detail: "Refresh or check the configured orchestrator URL.",
      tone: "bad",
      targetTab: "operations",
      targetId: "operations",
    });
  }
  if (activeFailedEvent) {
    push({
      id: `activity-${activeFailedEvent.id}`,
      label: missionActivityLabelForEvent(activeFailedEvent),
      detail: "Open the full activity trail for failure context.",
      tone: "bad",
      timestamp: activeFailedEvent.created_at,
      targetTab: "agents",
      targetId: "agent-activity",
    });
  }
  if (blockingDecision && latestDecision) {
    push({
      id: `decision-blocked-${latestDecision.id}`,
      label: "Validation blocked",
      detail: "Backend gate details are available in Agents.",
      tone: "bad",
      timestamp: latestDecision.created_at,
      targetTab: "agents",
      targetId: "agent-decisions",
    });
  }
  for (const event of visibleActivityEvents.slice(0, 4)) {
    const status = activityStatus(event.status);
    if (!["active", "waiting", "succeeded", "failed", "blocked"].includes(status)) continue;
    const target = missionActivityTargetForEvent(event);
    push({
      id: `event-${event.id}`,
      label: missionActivityLabelForEvent(event),
      detail: missionSignalDetailForEvent(event),
      tone: status === "failed" || status === "blocked" ? "bad" : status === "waiting" ? "warning" : status === "succeeded" ? "ok" : "info",
      timestamp: event.created_at,
      ...target,
    });
  }
  if (counts.ASSIGNED + counts.RUNNING > 0 || counts.QUEUED > 0) {
    push({
      id: "jobs-active",
      label: counts.ASSIGNED + counts.RUNNING > 0 ? "Training active" : "Jobs queued",
      detail: counts.ASSIGNED + counts.RUNNING > 0
        ? `${counts.ASSIGNED + counts.RUNNING} active, ${counts.QUEUED} queued.`
        : `${counts.QUEUED} queued job${counts.QUEUED === 1 ? "" : "s"} waiting for workers.`,
      tone: counts.ASSIGNED + counts.RUNNING > 0 ? "ok" : "warning",
      targetTab: "experiments",
      targetId: "runs",
    });
  }
  if (detail.champion) {
    push({
      id: "champion-selected",
      label: "Champion selected",
      detail: "Export and demo actions are ready.",
      tone: "ok",
      timestamp: detail.champion.created_at,
      targetTab: "export",
      targetId: "export-demo",
    });
  } else if (latestDecision) {
    push({
      id: `decision-${latestDecision.id}`,
      label: missionDecisionLabel(latestDecision),
      detail: decisionRejections(latestDecision).length > 0 ? "Backend gate warnings are visible in Agents." : "Latest project decision is recorded.",
      tone: decisionRejections(latestDecision).length > 0 ? "warning" : "info",
      timestamp: latestDecision.created_at,
      targetTab: "agents",
      targetId: "agent-decisions",
    });
  } else if (latestPlan) {
    push({
      id: `plan-${latestPlan.id}`,
      label: "Plan ready",
      detail: `${latestPlan.experiments.length} experiment${latestPlan.experiments.length === 1 ? "" : "s"} proposed.`,
      tone: "info",
      timestamp: latestPlan.created_at,
      targetTab: "experiments",
      targetId: "plans",
    });
  } else if (dataset) {
    push({
      id: `dataset-${dataset.id}`,
      label: datasetProfiled ? "Dataset profiled" : "Dataset profiling",
      detail: datasetProfiled ? "Class counts and image statistics are available." : "Waiting on profile results before planning.",
      tone: datasetProfiled ? "ok" : "warning",
      timestamp: dataset.profiled_at || dataset.created_at,
      targetTab: "data",
      targetId: "data",
    });
  }
  if (signals.length === 0 && state === "empty") {
    push({
      id: "empty",
      label: "Ready for setup",
      detail: "Create a project with a dataset folder.",
      tone: "info",
    });
  }
  return uniqueBy(signals, (signal) => signal.label).slice(0, 5);
}

export function buildMissionChampionSummary(detail: ProjectDetail): MissionChampionSummary | undefined {
  const champion = detail.champion;
  if (!champion) return undefined;
  const summary = detail.runSummaries.find((item) => item.job_id === champion.job_id) ?? null;
  const evaluation = detail.runEvaluations.find((item) => item.job_id === champion.job_id) ?? null;
  const modelProfile = championModelProfile(champion);
  const holisticScores = evaluation?.holistic_scores ?? {};
  const objectiveFit = recordNumber(holisticScores, "overall_score") || recordNumber(holisticScores, "objective_fit");
  const job = detail.jobs.find((item) => item.id === champion.job_id) ?? null;
  const primaryMetric = championPrimaryMetric(champion, summary, evaluation, job);
  const cost = recordNumber(champion.metrics, "estimated_cost_usd") || summary?.estimated_cost_usd || 0;
  const runtime = recordNumber(champion.metrics, "runtime_seconds") || summary?.runtime_seconds || 0;
  return {
    model: recordString(champion.metrics, "model") || summary?.model || "Selected champion",
    primaryMetricLabel: primaryMetric.label,
    primaryMetricValue: formatMaybeMetric(primaryMetric.value),
    secondaryMetricLabel: primaryMetric.secondaryLabel,
    secondaryMetricValue: formatMaybeMetric(primaryMetric.secondaryValue),
    cost: formatCurrency(cost),
    runtime: formatSeconds(runtime),
    latency: formatLatency(modelProfile.estimated_latency_ms),
    modelSize: formatModelSize(modelProfile),
    objectiveFit: formatMaybeMetric(objectiveFit),
  };
}

export function missionLiveStepFromEvent(event: AgentActivityEvent): MissionLiveActivity["steps"][number] {
  const championGuard = isChampionSelectedGuardActivity(event);
  return {
    id: `event-${event.id}`,
    label: missionActivityLabelForEvent(event),
    status: championGuard ? "succeeded" : missionLiveStatus(event.status),
    timestamp: event.created_at,
    ...missionActivityTargetForEvent(event),
  };
}

export function missionLiveStatus(value: string): MissionLiveActivity["steps"][number]["status"] {
  const status = activityStatus(value);
  if (["active", "waiting", "succeeded", "failed", "blocked"].includes(status)) {
    return status as MissionLiveActivity["steps"][number]["status"];
  }
  return "active";
}

export function missionActivityLabelForEvent(event: AgentActivityEvent) {
  const status = activityStatus(event.status);
  const text = `${event.type} ${event.title} ${event.message}`.toLowerCase();
  if (isChampionSelectedGuardActivity(event)) return "Champion selected";
  if (text.includes("memory") || text.includes("retrieval")) return "Retrieving memories";
  if (text.includes("dataset") || text.includes("profile")) return "Reading dataset profile";
  if (text.includes("candidate") || text.includes("ranking") || text.includes("rank")) return "Ranking candidates";
  if (text.includes("validation") || text.includes("gate") || text.includes("rejected") || text.includes("blocked")) {
    return status === "blocked" || status === "failed" ? "Validation blocked" : "Validating plan";
  }
  if (text.includes("worker")) {
    if (text.includes("starting") || text.includes("active")) return "Starting workers";
    return "Waiting for workers";
  }
  if (text.includes("queued") || text.includes("schedule") || text.includes("follow-up") || text.includes("followup")) return "Scheduling workers";
  if (text.includes("job") || text.includes("training") || text.includes("run")) return "Waiting for training results";
  if (text.includes("champion")) return "Selecting champion";
  if (text.includes("decision") || text.includes("outcome") || text.includes("recorded") || text.includes("recommendation")) return "Writing decision";
  if (text.includes("planner") || text.includes("reviewer") || text.includes("agent")) return "Thinking";
  return "Thinking";
}

export function missionActivityTargetForEvent(event: AgentActivityEvent): Pick<MissionSignal, "targetTab" | "targetId"> {
  const text = `${event.type} ${event.title}`.toLowerCase();
  if (text.includes("memory")) return { targetTab: "agents", targetId: "memory-retrieval-probe" };
  if (text.includes("worker")) return { targetTab: "operations", targetId: "workers" };
  if (text.includes("job") || text.includes("training") || text.includes("run")) return { targetTab: "experiments", targetId: "runs" };
  if (text.includes("champion")) return { targetTab: "export", targetId: "export-demo" };
  if (text.includes("plan.queued")) return { targetTab: "operations", targetId: "automation-timeline" };
  if (text.includes("validation") || text.includes("decision") || text.includes("planner") || text.includes("agent")) {
    return { targetTab: "agents", targetId: "agent-decisions" };
  }
  return { targetTab: "agents", targetId: "agent-activity" };
}

export function missionLiveActivityDetail(label: string, status: MissionLiveActivity["status"]) {
  if (label === "Retrieving memories") return "The agent is checking compact prior run and strategy memory.";
  if (label === "Reading dataset profile") return "The planner is using class counts and dataset statistics.";
  if (label === "Ranking candidates") return "The planner is comparing candidate experiments.";
  if (label === "Validating plan") return "Backend gates are checking the proposed plan.";
  if (label === "Validation blocked") return "A backend gate stopped unsafe or invalid work.";
  if (label === "Scheduling workers") return "Mission Control is handing validated work to workers.";
  if (label === "Waiting for workers") return "Queued work is waiting for worker capacity.";
  if (label === "Waiting for training results") return "Training jobs are expected to report metrics.";
  if (label === "Writing decision") return "The agent is recording a compact project decision.";
  if (label === "Champion selected") return "A champion is selected; follow-up scheduling requires reopen.";
  if (label === "Selecting champion") return "Completed runs are being promoted into a champion outcome.";
  if (status === "failed") return "The latest agentic step failed; open Agents for the audit trail.";
  if (status === "blocked") return "The latest agentic step is blocked by a visible gate.";
  if (status === "waiting") return "Mission Control is waiting for the next backend or worker update.";
  if (status === "succeeded") return "The latest agentic step completed.";
  return "The agentic system is working through the next high-level step.";
}

export function missionEnsureLiveStep(
  steps: MissionLiveActivity["steps"],
  step: MissionLiveActivity["steps"][number],
) {
  if (steps.some((item) => item.label === step.label)) return steps;
  return [step, ...steps].slice(0, 5);
}

export function missionSignalDetailForEvent(event: AgentActivityEvent) {
  const label = missionActivityLabelForEvent(event);
  if (label === "Champion selected") {
    const metadata = recordObject(event.metadata);
    return activitySafeDisplayText(
      recordString(metadata, "reason") || "Champion selected; reopen experimentation to schedule more.",
      180,
    );
  }
  if (label === "Validation blocked") return "Open Agents for backend gate detail.";
  if (label === "Retrieving memories") return "Memory lookup completed without exposing raw payloads.";
  if (label === "Scheduling workers" || label === "Waiting for workers") return "Worker state is available in Operations.";
  if (label === "Waiting for training results") return "Run progress is available in Experiments.";
  if (label === "Writing decision") return "Decision summary is available in Agents.";
  if (label === "Selecting champion") return "Champion details are available in Export.";
  return "Open the drill-down for the full audit trail.";
}

export function missionDecisionLabel(decision: AgentDecision) {
  const value = decision.decision_type.toLowerCase().replace(/_/g, " ");
  return value.replace(/\b\w/g, (match) => match.toUpperCase());
}

export function missionStateLabel(state: MissionDigestState) {
  switch (state) {
    case "empty":
      return "No Project";
    case "dataset_needed":
      return "Dataset Needed";
    case "profiling":
      return "Profiling";
    case "plan_ready":
      return "Plan Ready";
    case "training":
      return "Training";
    case "review_needed":
      return "Review Needed";
    case "follow_up_ready":
      return "Follow-up Ready";
    case "champion_selected":
      return "Champion Selected";
    case "blocked":
      return "Blocked";
  }
}

export function missionHealthLabel(health: Health | null) {
  if (!health) return "Checking";
  return health.status === "ok" ? "Healthy" : "Unhealthy";
}

export function missionToneRank(tone: MissionTone) {
  if (tone === "bad") return 0;
  if (tone === "warning") return 1;
  if (tone === "info") return 2;
  return 3;
}

export const activityStorageUriPattern = /\b(?:s3|gs|file|minio|https?):\/\/[^\s,;"')\]}]+/gi;
export const activityWindowsPathPattern = /\b[A-Z]:\\[^\s,;"')\]}]+/gi;
export const activityUnixPathPattern = /(^|\s)\/(?:Users|home|tmp|var|mnt|data|datasets|artifacts|workspace|app|srv)[^\s,;"')\]}]+/gi;
export const activityBase64Pattern = /\b[A-Za-z0-9+/]{80,}={0,2}\b/g;
export const activitySecretPattern = /\b(?:sk|pk|rk|xox[baprs]?)-[A-Za-z0-9_-]{16,}\b/gi;

export function activityEventFromMessage(event: MessageEvent): AgentActivityEvent | null {
  try {
    const parsed = recordObject(JSON.parse(String(event.data)));
    const id = recordString(parsed, "id");
    const projectId = recordString(parsed, "project_id");
    const type = recordString(parsed, "type");
    const createdAt = recordString(parsed, "created_at");
    if (!id || !projectId || !type || !createdAt) return null;
    return {
      id,
      project_id: projectId,
      plan_id: recordString(parsed, "plan_id") || undefined,
      job_id: recordString(parsed, "job_id") || undefined,
      type,
      severity: activitySeverity(recordString(parsed, "severity")),
      title: activitySafeDisplayText(recordString(parsed, "title"), 96) || "Activity",
      message: activitySafeDisplayText(recordString(parsed, "message"), 240),
      status: activityStatus(recordString(parsed, "status")),
      created_at: createdAt,
      metadata: activityMetadataObject(parsed.metadata),
    };
  } catch {
    return null;
  }
}

export function mergeActivityEvents(current: AgentActivityEvent[], event: AgentActivityEvent) {
  const byId = new Map<string, AgentActivityEvent>();
  for (const item of current) byId.set(item.id, item);
  byId.set(event.id, event);
  return Array.from(byId.values())
    .sort((left, right) => activityTimestamp(right) - activityTimestamp(left))
    .slice(0, 12);
}

export function buildFallbackActivityEvents(projectId: string, detail: ProjectDetail): AgentActivityEvent[] {
  if (!projectId) return [];
  const events: AgentActivityEvent[] = [
    ...detail.executionEvents.map(fallbackActivityFromExecutionEvent),
    ...detail.agentInvocations.flatMap((invocation) => fallbackActivityFromInvocation(invocation, detail.jobs)),
    ...fallbackActivityFromJobs(projectId, detail.jobs),
  ];
  return events
    .filter((event) => event.project_id === projectId)
    .sort((left, right) => activityTimestamp(right) - activityTimestamp(left))
    .slice(0, 12);
}

export function fallbackActivityFromExecutionEvent(event: ExecutionEvent): AgentActivityEvent {
  const payload = recordObject(event.payload);
  const jobId = recordString(payload, "job_id") || recordString(payload, "champion_job_id");
  const base: AgentActivityEvent = {
    id: `fallback_execution_${event.id}`,
    project_id: event.project_id,
    plan_id: event.plan_id,
    job_id: jobId || undefined,
    type: "system.event",
    severity: "info",
    title: "System event",
    message: activitySafeDisplayText(event.message, 220),
    status: "active",
    created_at: event.created_at,
    metadata: fallbackActivityMetadata(payload),
  };

  switch (event.event_type) {
    case "AGENT_STARTED":
      return { ...base, type: "planner.started", title: "Planner started", message: "Reading completed runs, memories, evaluations, and project context." };
    case "AGENT_RECOMMENDATION_RECORDED":
      return { ...base, type: "planner.decision_recorded", severity: "success", title: "Valid decision recorded", status: "succeeded" };
    case "AGENT_OUTCOME_RECORDED": {
      const validationStatus = recordString(payload, "backend_validation_status").toLowerCase();
      if (validationStatus === "blocked") {
        if (recordString(payload, "backend_stop_guard").toLowerCase() === "champion_selected_guard") {
          return {
            ...base,
            type: "planner.champion_guard",
            severity: "info",
            title: "Champion selected",
            status: "succeeded",
            message: activitySafeDisplayText(recordString(payload, "reason") || event.message, 220),
          };
        }
        return {
          ...base,
          type: "planner.blocked",
          severity: "warning",
          title: "Planner blocked",
          status: "blocked",
          message: activitySafeDisplayText(recordString(payload, "backend_validation_error") || recordString(payload, "reason") || event.message, 220),
        };
      }
      if (validationStatus === "failed") {
        return { ...base, type: "planner.validation_failed", severity: "error", title: "Planner validation failed", status: "failed" };
      }
      return { ...base, type: "agent.outcome_recorded", severity: "success", title: "Agent outcome recorded", status: "succeeded" };
    }
    case "AGENT_FAILED":
      return { ...base, type: "agent.failed", severity: "error", title: "Agent failed", status: "failed" };
    case "JOBS_QUEUED": {
      const openCount = recordNumber(payload, "open_job_count");
      return {
        ...base,
        type: "plan.queued",
        severity: openCount > 0 ? "success" : "info",
        title: openCount > 0 ? "Plan queued" : "Plan checked",
        status: openCount > 0 ? "active" : "waiting",
      };
    }
    case "WORKERS_REQUIRED":
    case "WORKER_SCALING_UPDATED":
      return { ...base, type: "workers.required", title: "Waiting for workers", status: "waiting" };
    case "WORKERS_STARTING":
      return { ...base, type: "workers.starting", title: "Workers starting", status: "active" };
    case "WORKERS_ACTIVE":
      return { ...base, type: "workers.active", severity: "success", title: "Workers active", status: "active" };
    case "DISPATCHER_STATUS":
      return {
        ...base,
        type: "dispatcher.status",
        title: "Modal dispatcher status",
        status: recordNumber(payload, "slot_count") === 0 ? "waiting" : "active",
      };
    case "DISPATCHER_IDLE_EXIT":
      return { ...base, type: "dispatcher.idle_exit", severity: "success", title: "Modal dispatcher idle", status: "succeeded" };
    case "JOB_RETRY_QUEUED": {
      const requeued = recordBoolean(payload, "requeued");
      return {
        ...base,
        type: requeued ? "job.retrying" : "job.failed",
        severity: requeued ? "warning" : "error",
        title: requeued ? "Retrying job" : "Job attempts exhausted",
        status: requeued ? "active" : "failed",
      };
    }
    case "EXECUTION_FAILED":
      return { ...base, type: "system.failed", severity: "error", title: "Execution failed", status: "failed" };
    case "MEMORY_RETRIEVAL_LOGGED": {
      const retrievedCount = recordNumber(payload, "retrieved_count");
      const purpose = recordString(payload, "purpose");
      return {
        ...base,
        type: "memory.retrieval_logged",
        title: "Memory retrieval checked",
        message: `Retrieved ${retrievedCount} candidate memory card(s)${memoryPurposeSuffix(purpose)}.`,
        status: "succeeded",
      };
    }
    case "CHAMPION_SELECTED":
      return { ...base, type: "champion.selected", severity: "success", title: "Selected champion", status: "succeeded" };
    default:
      return base;
  }
}

export function fallbackActivityFromInvocation(invocation: AgentInvocation, jobs: Job[]): AgentActivityEvent[] {
  const downstream = recordObject(invocation.downstream_outcome);
  const validationStatus = (invocation.validation_status || "").toLowerCase();
  const outcomeStatus = recordString(downstream, "backend_validation_status").toLowerCase();
  if (validationStatus !== "invalid" && validationStatus !== "failed" && outcomeStatus !== "rejected") return [];
  const willRetry = recordBoolean(downstream, "will_retry");
  const activeWork = jobs.some((job) => ["QUEUED", "ASSIGNED", "RUNNING"].includes(normalizedStatus(job.status))) || willRetry;
  const reason = recordString(downstream, "backend_validation_error") || invocation.validation_error || "backend validation rejected the draft";
  const planner = invocation.agent_name === "experiment_planner";
  const status = willRetry ? "active" : activeWork ? "waiting" : validationStatus === "failed" ? "failed" : "blocked";
  const severity = status === "failed" ? "error" : "warning";
  const title = `${planner ? "Planner" : "Agent"} draft rejected${willRetry ? "; retrying" : ""}`;
  const message = willRetry
    ? `Draft rejected: ${activitySafeDisplayText(reason, 180)}; retrying with validation feedback.`
    : `Draft rejected: ${activitySafeDisplayText(reason, 180)}.`;
  return [{
    id: `fallback_invocation_${invocation.id}_${status}`,
    project_id: invocation.project_id,
    plan_id: invocation.plan_id,
    job_id: invocation.job_id,
    type: planner ? "planner.validation_rejected" : "agent.validation_rejected",
    severity,
    title,
    message,
    status,
    created_at: invocation.created_at,
    metadata: {
      agent_name: invocation.agent_name,
      validation_status: outcomeStatus || validationStatus,
      will_retry: willRetry,
      validation_error: activitySafeDisplayText(reason, 180),
    },
  }];
}

export function fallbackActivityFromJobs(projectId: string, jobs: Job[]): AgentActivityEvent[] {
  if (jobs.length === 0) return [];
  const counts = jobStatusCounts(jobs);
  const activeCount = counts.ASSIGNED + counts.RUNNING;
  const latestJob = [...jobs].sort((left, right) => jobActivityTimestamp(right) - jobActivityTimestamp(left))[0];
  const title = activeCount > 0 ? "Jobs running" : counts.QUEUED > 0 ? "Jobs queued" : counts.FAILED > 0 && counts.SUCCEEDED === 0 ? "Jobs failed" : "Jobs completed";
  const status = activeCount > 0 ? "active" : counts.QUEUED > 0 ? "waiting" : counts.FAILED > 0 && counts.SUCCEEDED === 0 ? "failed" : "succeeded";
  const severity = status === "failed" ? "error" : status === "succeeded" ? "success" : "info";
  return [{
    id: `fallback_jobs_${jobs.length}_${counts.QUEUED}_${counts.ASSIGNED}_${counts.RUNNING}_${counts.SUCCEEDED}_${counts.FAILED}_${latestJob?.id ?? "none"}_${latestJob?.status ?? "none"}`,
    project_id: projectId,
    job_id: latestJob?.id,
    type: "jobs.status_counts",
    severity,
    title,
    message: activeCount > 0
      ? `${activeCount} job(s) running, ${counts.QUEUED} queued.`
      : counts.QUEUED > 0
        ? `${counts.QUEUED} job(s) queued; waiting for workers.`
        : `${counts.SUCCEEDED} succeeded, ${counts.FAILED} failed.`,
    status,
    created_at: latestJob?.completed_at || latestJob?.started_at || latestJob?.created_at || new Date().toISOString(),
    metadata: {
      job_count: jobs.length,
      queued_count: counts.QUEUED,
      assigned_count: counts.ASSIGNED,
      running_count: counts.RUNNING,
      completed_count: counts.SUCCEEDED,
      failed_count: counts.FAILED,
    },
  }];
}

export function agentActivityCurrentState(events: AgentActivityEvent[], detail: ProjectDetail, streamState: ActivityStreamState) {
  const retryEvent = events.find((event) => event.type === "planner.validation_rejected" && event.status === "active");
  if (retryEvent) {
    return { title: "Planner is retrying validation", detail: retryEvent.message, status: "active", severity: "warning" };
  }
  const activeJobs = detail.jobs.filter((job) => ["ASSIGNED", "RUNNING"].includes(normalizedStatus(job.status))).length;
  if (activeJobs > 0) {
    return { title: `${activeJobs} job${activeJobs === 1 ? "" : "s"} running`, detail: "Training workers are reporting active job work.", status: "active", severity: "info" };
  }
  const queuedJobs = detail.jobs.filter((job) => normalizedStatus(job.status) === "QUEUED").length;
  if (queuedJobs > 0) {
    return { title: `${queuedJobs} job${queuedJobs === 1 ? "" : "s"} queued`, detail: "Waiting for workers to pick up queued jobs.", status: "waiting", severity: "info" };
  }
  const activeEvent = events.find((event) => event.status === "active" || event.status === "waiting");
  if (activeEvent) {
    return { title: activeEvent.title, detail: activeEvent.message || activityEventSubtitle(activeEvent), status: activeEvent.status, severity: activeEvent.severity };
  }
  if (streamState === "reconnecting") {
    return { title: "Activity stream reconnecting", detail: "Mission Control is using fallback refresh while the SSE stream reconnects.", status: "waiting", severity: "warning" };
  }
  if (streamState === "fallback") {
    return { title: "Activity refresh fallback", detail: "SSE is unavailable; Mission Control is using periodic project refresh.", status: "waiting", severity: "info" };
  }
  const latest = events[0];
  if (latest) {
    return { title: latest.title, detail: latest.message || activityEventSubtitle(latest), status: latest.status, severity: latest.severity };
  }
  return { title: "Waiting for activity", detail: "No planner, worker, or job activity has been recorded yet.", status: "waiting", severity: "info" };
}

export function activitySeverityIcon(severity: string) {
  const normalized = severity.toLowerCase();
  if (normalized === "success") return <CheckCircle2 size={15} />;
  if (normalized === "warning") return <AlertTriangle size={15} />;
  if (normalized === "error") return <X size={15} />;
  return <Activity size={15} />;
}

export function activityStreamBadge(streamState: ActivityStreamState) {
  if (streamState === "connected") return "LIVE";
  if (streamState === "reconnecting") return "RECONNECTING";
  if (streamState === "fallback") return "POLLING";
  if (streamState === "connecting") return "CONNECTING";
  return "IDLE";
}

export function activityEventSubtitle(event: AgentActivityEvent) {
  return [
    event.type,
    event.plan_id ? `plan ${event.plan_id}` : "",
    event.job_id ? `job ${event.job_id}` : "",
    formatRelativeTime(event.created_at),
  ].filter(Boolean).join(" - ");
}

export function activityMetadataRows(metadata: Record<string, unknown> | undefined) {
  const record = activityMetadataObject(metadata);
  return Object.entries(record)
    .map(([key, value]) => ({ label: humanizeAuditKey(key), value: activityMetadataDisplayValue(value) }))
    .filter((row) => row.value !== "")
    .slice(0, 10);
}

export function activityMetadataObject(value: unknown): Record<string, unknown> {
  const record = recordObject(value);
  const out: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(record)) {
    const display = activityMetadataDisplayValue(entry);
    if (display) out[key] = display;
  }
  return out;
}

export function activityMetadataDisplayValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return activitySafeDisplayText(value, 110);
  if (Array.isArray(value)) {
    return value
      .map((item) => activitySafeDisplayText(String(item), 72))
      .filter(Boolean)
      .slice(0, 6)
      .join(", ");
  }
  return "";
}

export function fallbackActivityMetadata(payload: Record<string, unknown>) {
  const allowed = [
    "agent_name",
    "decision_id",
    "source_decision_id",
    "decision_type",
    "job_id",
    "job_ids",
    "worker_requirement_id",
    "open_job_count",
    "active_worker_count",
    "target_count",
    "previous_slot_count",
    "slot_count",
    "desired_slot_count",
    "registered_slot_count",
    "active_slot_count",
    "idle_seconds",
    "idle_exit_seconds",
    "dispatcher",
    "provider",
    "gpu_type",
    "requirement_status",
    "template",
    "attempt",
    "max_attempts",
    "requeued",
    "backend_validation_status",
    "backend_stop_guard",
    "reason",
    "model",
    "selection_source",
    "materialization_status",
    "max_concurrent_jobs",
    "max_cold_dataset_materializations",
    "retry_attempt",
    "will_retry",
    "completed_run_count",
    "memory_count",
    "evaluation_count",
    "purpose",
    "retrieved_count",
    "log_only",
    "cross_project_ok",
  ];
  const out: Record<string, unknown> = {};
  for (const key of allowed) {
    const display = activityMetadataDisplayValue(payload[key]);
    if (display) out[key] = display;
  }
  const validationError = recordString(payload, "backend_validation_error");
  if (validationError) out.validation_error = activitySafeDisplayText(validationError, 180);
  const errorSummary = recordString(payload, "error") || recordString(payload, "last_error");
  if (errorSummary) out.error_summary = activitySafeDisplayText(errorSummary, 180);
  return out;
}

export function activitySafeDisplayText(value: string, maxLength = 160) {
  let text = String(value ?? "").trim();
  if (!text) return "";
  text = text
    .replace(activitySecretPattern, "[redacted_secret]")
    .replace(activityBase64Pattern, "[redacted_blob]")
    .replace(activityStorageUriPattern, "[redacted_uri]")
    .replace(activityWindowsPathPattern, "[redacted_path]")
    .replace(activityUnixPathPattern, "$1[redacted_path]")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 1)).trim()}...`;
}

export function activitySeverity(value: string): AgentActivityEvent["severity"] {
  const normalized = value.toLowerCase();
  if (["info", "warning", "error", "success"].includes(normalized)) return normalized;
  return "info";
}

export function activityStatus(value: string): AgentActivityEvent["status"] {
  const normalized = value.toLowerCase();
  if (["active", "waiting", "succeeded", "failed", "blocked"].includes(normalized)) return normalized;
  return "active";
}

export function isChampionSelectedGuardActivity(event: AgentActivityEvent) {
  const metadata = recordObject(event.metadata);
  const guard = recordString(metadata, "backend_stop_guard").toLowerCase();
  const validationError = recordString(metadata, "validation_error").toLowerCase();
  const text = `${event.type} ${event.title} ${event.message} ${guard} ${validationError}`.toLowerCase();
  return guard === "champion_selected_guard" || text.includes("champion selected guard");
}

export function activityTimestamp(event: AgentActivityEvent) {
  const timestamp = Date.parse(event.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export function jobActivityTimestamp(job: Job) {
  const timestamp = Date.parse(job.completed_at || job.started_at || job.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export function jobStatusCounts(jobs: Job[]) {
  return jobs.reduce(
    (counts, job) => {
      const status = normalizedStatus(job.status);
      if (status in counts) counts[status as keyof typeof counts] += 1;
      return counts;
    },
    { QUEUED: 0, ASSIGNED: 0, RUNNING: 0, SUCCEEDED: 0, FAILED: 0 },
  );
}

export function recordBoolean(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["true", "yes", "1"].includes(value.toLowerCase());
  return false;
}

export function buildExperimentTimeline(project: Project | null, detail: ProjectDetail): TimelineItem[] {
  const firstDataset = detail.datasets[0] ?? null;
  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const followUpPlan = latestDecision
    ? detail.plans.find((plan) => plan.source_decision_id === latestDecision.id)
    : null;
  const completedJobs = detail.jobs.filter((job) => ["SUCCEEDED", "FAILED"].includes(job.status)).length;
  const activeJobs = detail.jobs.filter((job) => ["QUEUED", "ASSIGNED", "RUNNING"].includes(job.status)).length;
  const monitorMemory = detail.agentMemory.find((record) => record.kind === "training_evaluation");

  return [
    {
      label: "Project created",
      detail: project ? project.goal || "Ready for dataset upload and profiling." : "Choose or create a project.",
      status: project ? "done" : "waiting",
      timestamp: project?.created_at,
    },
    {
      label: "Dataset uploaded",
      detail: firstDataset
        ? `${firstDataset.name} - ${formatBytes(firstDataset.size_bytes)}`
        : "Waiting for an image classification folder.",
      status: firstDataset ? "done" : "waiting",
      timestamp: firstDataset?.created_at,
    },
    {
      label: "Dataset profiled",
      detail: firstDataset?.profiled_at
        ? "Class counts and image statistics are available."
        : "Profiling worker has not reported dataset intelligence yet.",
      status: firstDataset?.profiled_at ? "done" : firstDataset ? "active" : "waiting",
      timestamp: firstDataset?.profiled_at,
    },
    {
      label: "Plan created",
      detail: latestPlan
        ? `${latestPlan.experiments.length} experiment(s), target ${latestPlan.target_metric}`
        : "Planner will create an experiment batch after profiling.",
      status: latestPlan ? "done" : firstDataset?.profiled_at ? "active" : "waiting",
      timestamp: latestPlan?.created_at,
    },
    {
      label: "Jobs launched",
      detail: detail.jobs.length > 0 ? `${detail.jobs.length} job(s) in the queue or history.` : "Plan execution has not launched jobs.",
      status: detail.jobs.length > 0 ? "done" : latestPlan ? "active" : "waiting",
      timestamp: detail.jobs[0]?.created_at,
    },
    {
      label: "Jobs completed",
      detail: detail.jobs.length > 0 ? `${completedJobs}/${detail.jobs.length} terminal, ${activeJobs} active.` : "No training jobs yet.",
      status: detail.jobs.length > 0 && activeJobs === 0 ? "done" : activeJobs > 0 ? "active" : "waiting",
      timestamp: detail.jobs.find((job) => job.completed_at)?.completed_at,
    },
    {
      label: "Monitor evaluated",
      detail: monitorMemory ? monitorMemory.summary : "Training monitor memory appears after completed run review.",
      status: monitorMemory ? "done" : completedJobs > 0 ? "active" : "waiting",
      timestamp: monitorMemory?.created_at,
    },
    {
      label: "Planner proposed",
      detail: latestDecision ? decisionHistorySummary(latestDecision) : "Experiment planner has not recorded a project-level decision.",
      status: latestDecision ? "done" : completedJobs > 0 ? "active" : "waiting",
      timestamp: latestDecision?.created_at,
    },
    {
      label: "Backend gate",
      detail: latestDecision
        ? backendGateSummary(latestDecision)
        : "The backend will validate model names, duplicate signatures, planning mode, cost, and novelty.",
      status: latestDecision ? (decisionRejections(latestDecision).length > 0 ? "blocked" : "done") : "waiting",
      timestamp: latestDecision?.created_at,
    },
    {
      label: "Follow-up scheduled",
      detail: followUpPlan
        ? `${followUpPlan.experiments.length} follow-up experiment(s) created.`
        : "No accepted follow-up plan is linked to the latest decision.",
      status: followUpPlan ? "done" : latestDecision?.decision_type === "ADD_EXPERIMENTS" ? "active" : "waiting",
      timestamp: followUpPlan?.created_at,
    },
    {
      label: "Champion selected",
      detail: detail.champion ? detail.champion.selection_reason || detail.champion.job_id : "No project champion has been persisted yet.",
      status: detail.champion ? "done" : "waiting",
      timestamp: detail.champion?.created_at,
    },
  ];
}

export function buildDatasetIntelligence(
  dataset: Dataset | null,
  latestDecision: AgentDecision | null,
  metadataDetail: DatasetMetadataDetail,
) {
  const profile = dataset?.profile ?? {};
  const plannerInsights = recordObject(latestDecision?.payload.dataset_planning_insights);
  const classDistribution = recordObject(profile.images_per_class ?? profile.class_distribution);
  const classRows = Object.entries(classDistribution)
    .map(([name, value]) => ({ name, count: numericValue(value) }))
    .filter((row) => row.count > 0)
    .sort((left, right) => right.count - left.count)
    .slice(0, 8);
  const maxClassCount = Math.max(...classRows.map((row) => row.count), 1);
  const totalImages = recordNumber(profile, "total_images");
  const classCount = recordNumber(profile, "class_count");
  const imbalanceRatio = recordNumber(profile, "imbalance_ratio");
  const corruptCount = recordNumber(profile, "corrupt_image_count");
  const widthMin = recordNumber(profile, "width_min");
  const widthMax = recordNumber(profile, "width_max");
  const heightMin = recordNumber(profile, "height_min");
  const heightMax = recordNumber(profile, "height_max");
  const legacyMetadataSummary = recordObject(profile.metadata_summary);
  const metadataStatus = buildMetadataStatusDisplay(metadataDetail);
  const metadataAvailable =
    Boolean(metadataStatus) || profile.metadata_available === true || Object.keys(legacyMetadataSummary).length > 0;
  const exemplarStatus = exemplarStatusFromProfile(profile);
  const recommendedMetrics = stringArrayPayload(plannerInsights.recommended_metrics);
  const recommendedPreprocessing = stringArrayPayload(plannerInsights.recommended_preprocessing);
  const recommendedAugmentations = stringArrayPayload(plannerInsights.recommended_augmentations);

  const metrics = uniqueStrings([
    ...(recommendedMetrics.length > 0 ? recommendedMetrics : ["accuracy", "macro_f1"]),
    ...(imbalanceRatio >= 1.5 ? ["per_class_recall"] : []),
  ]);
  const preprocessing = uniqueStrings([
    ...(recommendedPreprocessing.length > 0
      ? recommendedPreprocessing
      : ["ImageNet normalization", "transfer learning baseline"]),
    ...(recommendedAugmentations.length > 0 ? recommendedAugmentations.map((item) => `augmentation: ${item}`) : []),
    ...(imbalanceRatio >= 1.5 ? ["weighted loss or balanced sampling"] : []),
    ...(Math.max(widthMax, heightMax) >= 512 ? ["compare 224px and 256px input sizes"] : []),
  ]).slice(0, 6);

  const artifacts = [
    "image folders",
    metadataAvailable ? (metadataStatus ? "backend metadata imported" : "metadata detected") : "metadata not reported",
    corruptCount > 0 ? `${corruptCount} corrupt image(s)` : "no corrupt images reported",
    exemplarStatus,
  ].filter(Boolean);
  const warnings = [
    ...(imbalanceRatio >= 1.5 ? [`Class imbalance ratio ${imbalanceRatio.toFixed(2)}; prefer macro-F1 and per-class checks.`] : []),
    ...(corruptCount > 0 ? [`${corruptCount} corrupt image(s) detected by profiler.`] : []),
    ...(metadataDetail.status === "error" ? [metadataDetail.message] : []),
    ...(widthMin > 0 && heightMin > 0 && Math.max(widthMax, heightMax) > Math.min(widthMin, heightMin) * 2
      ? ["Image dimensions vary widely; resize plus crop is safer for comparison runs."]
      : []),
  ];

  return {
    dataset,
    classRows: classRows.map((row) => ({ ...row, percent: Math.max(4, (row.count / maxClassCount) * 100) })),
    metrics,
    preprocessing,
    artifacts,
    warnings,
    metadataStatus,
    insights: [
      { label: "Images", value: totalImages ? String(totalImages) : "-", tone: totalImages > 0 ? "good" : "warn" },
      { label: "Classes", value: classCount ? String(classCount) : "-", tone: classCount >= 2 ? "good" : "warn" },
      {
        label: "Imbalance",
        value: imbalanceRatio ? `${imbalanceRatio.toFixed(2)}x` : "-",
        tone: imbalanceRatio >= 1.5 ? "warn" : "good",
      },
      {
        label: "Image Sizes",
        value: widthMax && heightMax ? `${widthMin || "?"}-${widthMax}w / ${heightMin || "?"}-${heightMax}h` : "-",
      },
      { label: "Corrupt", value: String(corruptCount), tone: corruptCount > 0 ? "bad" : "good" },
      { label: "Profile", value: dataset?.profiled_at ? "ready" : "pending", tone: dataset?.profiled_at ? "good" : "warn" },
    ] satisfies InsightItem[],
  };
}

export function datasetMetadataSummaryFromResponse(
  value: DatasetMetadataSummaryResponse,
  imports: DatasetMetadataImport[] = [],
): DatasetMetadataSummary | null {
  const record = recordObject(value);
  const importRecords = [
    ...imports,
    ...[record.import, record.metadata_import].map(recordObject).filter(hasDatasetMetadataImportShape),
  ];
  const fallback = datasetMetadataSummaryFallback(record);
  const candidates = [
    recordObject(record.agent_safe_summary),
    recordObject(record.metadata_summary),
    recordObject(record.dataset_metadata_summary),
    recordObject(record.summary),
    hasDatasetMetadataSummaryShape(record) ? record : {},
    datasetMetadataSummaryFromImports(importRecords),
  ].filter((item): item is Record<string, unknown> => Boolean(item) && Object.keys(recordObject(item)).length > 0);

  const summary = candidates.find(hasDatasetMetadataSummaryShape);
  if (!summary) return null;
  return { ...fallback, ...summary } as DatasetMetadataSummary;
}

export function datasetMetadataSummaryFromImports(imports: DatasetMetadataImport[]): DatasetMetadataSummary | null {
  const latestImport = [...imports].sort((left, right) => metadataImportSortScore(right) - metadataImportSortScore(left)).find(Boolean);
  if (!latestImport) return null;
  return datasetMetadataSummaryFromImport(latestImport);
}

export function datasetMetadataSummaryFromImport(metadataImport: DatasetMetadataImport): DatasetMetadataSummary | null {
  const importRecord = recordObject(metadataImport);
  const agentSafeSummary = recordObject(metadataImport.agent_safe_summary);
  const importSummary = recordObject(metadataImport.summary);
  const summary = hasDatasetMetadataSummaryShape(agentSafeSummary) ? agentSafeSummary : importSummary;
  const fallback = datasetMetadataSummaryFallback(importRecord);
  const warnings = metadataImport.warnings ?? (summary.warnings as DatasetMetadataSummary["warnings"]);
  const errors = metadataImport.errors ?? (summary.errors as DatasetMetadataSummary["errors"]);
  if (hasDatasetMetadataSummaryShape(summary)) {
    return { ...fallback, warnings, errors, ...summary } as DatasetMetadataSummary;
  }
  if (hasDatasetMetadataImportShape(importRecord)) {
    return { ...fallback, warnings, errors } as DatasetMetadataSummary;
  }
  return null;
}

export function datasetMetadataImportsFromResponse(value: DatasetMetadataImportsResponse): DatasetMetadataImport[] {
  if (Array.isArray(value)) {
    return value.map((item) => recordObject(item) as DatasetMetadataImport).filter(hasDatasetMetadataImportShape);
  }

  const record = recordObject(value);
  const imports = [
    ...arrayDatasetMetadataImports(record.imports),
    ...arrayDatasetMetadataImports(record.metadata_imports),
    ...arrayDatasetMetadataImports(record.items),
  ];

  for (const key of ["latest", "active", "import", "metadata_import"]) {
    const metadataImport = recordObject(record[key]);
    if (hasDatasetMetadataImportShape(metadataImport)) imports.push(metadataImport as DatasetMetadataImport);
  }

  return uniqueBy(imports, (metadataImport, index) => metadataImport.import_id || metadataImport.id || String(index));
}

export function arrayDatasetMetadataImports(value: unknown): DatasetMetadataImport[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as DatasetMetadataImport).filter(hasDatasetMetadataImportShape);
}

export function hasDatasetMetadataImportShape(value: unknown): value is DatasetMetadataImport {
  const record = recordObject(value);
  return [
    "id",
    "import_id",
    "status",
    "active",
    "summary",
    "agent_safe_summary",
    "warnings",
    "errors",
    "created_at",
    "completed_at",
  ].some((key) => key in record);
}

export function hasDatasetMetadataSummaryShape(value: unknown): value is DatasetMetadataSummary {
  const record = recordObject(value);
  return [
    "status",
    "import_id",
    "source_kinds",
    "source_formats",
    "formats",
    "class_count",
    "sample_count",
    "split_counts",
    "annotation_counts",
    "bbox_coverage",
    "bbox_count",
    "bbox_counts",
    "warnings",
    "errors",
    "official_split_available",
    "unsupported_source_count",
    "created_at",
    "completed_at",
  ].some((key) => key in record);
}

export function datasetMetadataSummaryFallback(record: Record<string, unknown>): DatasetMetadataSummary {
  const fallback: DatasetMetadataSummary = {};
  const status = recordFirstString(record, ["status"]);
  const importId = recordFirstString(record, ["import_id", "id"]);
  const createdAt = recordFirstString(record, ["created_at"]);
  const completedAt = recordFirstString(record, ["completed_at"]);
  const unsupportedSourceCount = recordNumber(record, "unsupported_source_count");
  if (status) fallback.status = status;
  if (importId) fallback.import_id = importId;
  if (createdAt) fallback.created_at = createdAt;
  if (completedAt) fallback.completed_at = completedAt;
  if (typeof record.official_split_available === "boolean") {
    fallback.official_split_available = record.official_split_available;
  }
  if (unsupportedSourceCount > 0) fallback.unsupported_source_count = unsupportedSourceCount;
  return fallback;
}

export function metadataImportSortScore(metadataImport: DatasetMetadataImport) {
  const timestamp = Date.parse(metadataImport.completed_at || metadataImport.created_at || "");
  if (Number.isFinite(timestamp)) return timestamp;
  if (metadataImport.active) return 1;
  return 0;
}

export function buildMetadataStatusDisplay(metadataDetail: DatasetMetadataDetail): MetadataStatusDisplay | null {
  const summary = metadataDetail.summary;
  if (metadataDetail.status !== "available" || !summary) return null;

  const status = metadataSummaryStatus(summary, metadataDetail.imports);
  const classCount = metadataNumber(summary, ["class_count", "classes", "num_classes"]);
  const sampleCount = metadataNumber(summary, ["sample_count", "samples", "image_count", "total_images"]);
  const unsupportedCount = metadataNumber(summary, ["unsupported_source_count", "unsupported_sources"]);
  const errors = metadataIssueSummaries(summary.errors).slice(0, 5);
  const warnings = uniqueStrings([
    ...metadataIssueSummaries(summary.warnings),
    ...(unsupportedCount > 0 ? [`${unsupportedCount} unsupported metadata source(s) skipped.`] : []),
  ]).slice(0, 6);
  const officialSplitAvailable = typeof summary.official_split_available === "boolean" ? summary.official_split_available : null;

  return {
    status,
    detail: metadataDetailText(summary, metadataDetail.imports),
    facts: [
      { label: "Metadata Classes", value: metadataCountValue(classCount), tone: classCount > 0 ? "good" : "warn" },
      { label: "Metadata Samples", value: metadataCountValue(sampleCount), tone: sampleCount > 0 ? "good" : "warn" },
      {
        label: "Official Split",
        value: officialSplitAvailable === null ? "-" : officialSplitAvailable ? "yes" : "no",
        tone: officialSplitAvailable ? "good" : officialSplitAvailable === false ? "warn" : undefined,
      },
      { label: "BBox Evidence", value: metadataBBoxValue(summary), tone: metadataBBoxTone(summary) },
      { label: "Skipped Sources", value: String(unsupportedCount), tone: unsupportedCount > 0 ? "warn" : "good" },
      { label: "Metadata Errors", value: String(errors.length), tone: errors.length > 0 ? "bad" : "good" },
    ],
    sources: metadataSourceTags(summary),
    splitRows: metadataCountRows(summary.split_counts),
    annotationRows: metadataCountRows(summary.annotation_counts),
    warnings,
    errors,
  };
}

export function metadataSummaryStatus(summary: DatasetMetadataSummary, imports: DatasetMetadataImport[]) {
  const summaryStatus = recordFirstString(summary, ["status"]);
  if (summaryStatus) return summaryStatus;
  const importId = recordFirstString(summary, ["import_id"]);
  const matchingImport = imports.find((metadataImport) => metadataImport.import_id === importId || metadataImport.id === importId);
  return matchingImport?.status || imports.find((metadataImport) => metadataImport.active)?.status || "reported";
}

export function metadataDetailText(summary: DatasetMetadataSummary, imports: DatasetMetadataImport[]) {
  const importId = recordFirstString(summary, ["import_id"]);
  const matchingImport = imports.find((metadataImport) => metadataImport.import_id === importId || metadataImport.id === importId);
  const createdAt = recordFirstString(summary, ["created_at"]) || matchingImport?.created_at || "";
  const completedAt = recordFirstString(summary, ["completed_at"]) || matchingImport?.completed_at || "";
  const timestamp = completedAt || createdAt;
  const timeLabel = timestamp ? `${completedAt ? "completed" : "created"} ${formatTimestamp(timestamp)}` : "";
  return [importId ? `import ${importId}` : "agent-safe summary", timeLabel].filter(Boolean).join(" - ");
}

export function metadataSourceTags(summary: DatasetMetadataSummary) {
  const sourceKinds = metadataStringList(summary.source_kinds ?? summary.source_kind);
  const sourceFormats = metadataStringList(summary.source_formats ?? summary.formats ?? summary.detected_formats);
  return uniqueStrings([
    ...sourceKinds.map((kind) => `kind: ${kind}`),
    ...sourceFormats.map((format) => `format: ${format}`),
  ]).slice(0, 8);
}

export function metadataCountRows(value: unknown): MetadataCountRow[] {
  return Object.entries(recordObject(value))
    .map(([label, count]) => ({ label, value: metadataCountText(count) }))
    .filter((row) => row.value !== "")
    .slice(0, 8);
}

export function metadataCountText(value: unknown) {
  const numeric = metadataNumericValue(value);
  if (numeric !== null) return String(numeric);
  const record = recordObject(value);
  const nested = metadataNumber(record, ["count", "sample_count", "annotation_count"]);
  if (nested > 0) return String(nested);
  if (typeof value === "string" && value.trim()) return value.trim();
  return "";
}

export function metadataStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map(metadataStringValue).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => shortAuditText(item, 80))
      .filter(Boolean);
  }
  const record = recordObject(value);
  return Object.entries(record)
    .map(([key, entry]) => {
      const count = metadataNumericValue(entry);
      return count !== null && count > 0 ? `${key} (${count})` : key;
    })
    .filter(Boolean);
}

export function metadataStringValue(value: unknown) {
  if (typeof value === "string") return shortAuditText(value, 80);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  const record = recordObject(value);
  return shortAuditText(
    recordFirstString(record, ["format", "source_format", "detected_format", "declared_format", "kind", "source_kind", "name"]),
    80,
  );
}

export function metadataIssueSummaries(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(metadataIssueSummary).filter(Boolean);
  const record = recordObject(value);
  if (Object.keys(record).length === 0) return [metadataIssueSummary(value)].filter(Boolean);
  if (["message", "reason", "code", "error", "status"].some((key) => key in record)) {
    return [metadataIssueSummary(record)].filter(Boolean);
  }
  return Object.entries(record)
    .map(([key, entry]) => metadataIssueSummary({ code: key, value: entry }))
    .filter(Boolean);
}

export function metadataIssueSummary(value: unknown) {
  if (typeof value === "string") return shortAuditText(value, 180);
  const record = recordObject(value);
  if (Object.keys(record).length === 0) return "";
  const code = recordFirstString(record, ["code", "type", "severity", "status"]);
  const message = recordFirstString(record, ["message", "reason", "summary", "detail", "error"]);
  const kind = recordFirstString(record, ["source_kind", "kind"]);
  const format = recordFirstString(record, ["source_format", "declared_format", "detected_format", "format"]);
  const count = metadataNumericValue(record.count);
  const valueText = metadataPrimitiveText(record.value);
  return shortAuditText(
    [code, message || valueText, kind ? `kind ${kind}` : "", format ? `format ${format}` : "", count ? `${count} item(s)` : ""]
      .filter(Boolean)
      .join(": "),
    180,
  );
}

export function metadataPrimitiveText(value: unknown) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} item(s)`;
  return "";
}

export function metadataNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const numeric = metadataNumericValue(record[key]);
    if (numeric !== null) return numeric;
  }
  return 0;
}

export function metadataNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function metadataCountValue(value: number) {
  return Number.isFinite(value) && value > 0 ? String(value) : "-";
}

export function metadataBBoxValue(summary: DatasetMetadataSummary) {
  const coverage = metadataCoverageText(summary.bbox_coverage ?? summary.bounding_box_coverage ?? summary.bbox_coverage_ratio);
  const bboxCounts = recordObject(summary.bbox_counts);
  const bboxCount =
    metadataNumber(summary, ["bbox_count", "bounding_box_count", "box_count"]) ||
    metadataNumber(bboxCounts, ["bbox", "box", "boxes", "total", "count"]);
  if (coverage && bboxCount > 0) return `${coverage} / ${bboxCount}`;
  if (coverage) return coverage;
  if (bboxCount > 0) return `${bboxCount}`;
  return "-";
}

export function metadataBBoxTone(summary: DatasetMetadataSummary): InsightItem["tone"] {
  const value = metadataBBoxValue(summary);
  if (value === "-") return "warn";
  return "good";
}

export function metadataCoverageText(value: unknown) {
  const numeric = metadataNumericValue(value);
  if (numeric !== null) {
    if (numeric >= 0 && numeric <= 1) return formatPercent(numeric);
    if (numeric <= 100) return `${Math.round(numeric)}%`;
    return String(numeric);
  }
  if (typeof value === "string" && value.trim()) return shortAuditText(value, 40);

  const record = recordObject(value);
  const percent = metadataNumber(record, ["coverage_percent", "percent", "bbox_coverage_percent"]);
  if (percent > 0) return `${Math.round(percent)}%`;
  const ratio = metadataNumber(record, ["coverage_ratio", "sample_coverage_ratio", "bbox_coverage_ratio", "coverage"]);
  if (ratio > 0) return ratio <= 1 ? formatPercent(ratio) : `${Math.round(ratio)}%`;
  const covered = metadataNumber(record, ["annotated_sample_count", "covered_sample_count", "samples_with_bbox"]);
  const total = metadataNumber(record, ["sample_count", "total_sample_count", "total_samples"]);
  if (covered > 0 && total > 0) return formatPercent(covered / total);
  return "";
}

export function visualAnalysesFromResponse(value: unknown): DatasetVisualAnalysis[] {
  if (Array.isArray(value)) {
    return value.map((item) => recordObject(item) as DatasetVisualAnalysis).filter(hasVisualAnalysisShape);
  }

  const record = recordObject(value);
  const analyses = [
    ...arrayVisualAnalyses(record.visual_analyses),
    ...arrayVisualAnalyses(record.dataset_visual_analyses),
    ...arrayVisualAnalyses(record.analyses),
    ...arrayVisualAnalyses(record.items),
  ];

  for (const key of ["analysis", "latest", "dataset_visual_analysis"]) {
    const analysis = recordObject(record[key]);
    if (hasVisualAnalysisShape(analysis)) analyses.push(analysis as DatasetVisualAnalysis);
  }

  if (hasVisualAnalysisShape(record)) analyses.push(record as DatasetVisualAnalysis);
  return analyses;
}

export function visualAnalysisRerunPolicyFromResponse(value: unknown): VisualAnalysisRerunPolicy | null {
  const record = recordObject(value);
  const policy = recordObject(record.rerun_policy);
  return Object.keys(policy).length > 0 ? (policy as VisualAnalysisRerunPolicy) : null;
}

export function arrayVisualAnalyses(value: unknown): DatasetVisualAnalysis[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as DatasetVisualAnalysis).filter(hasVisualAnalysisShape);
}

export function hasVisualAnalysisShape(value: unknown): value is DatasetVisualAnalysis {
  const record = recordObject(value);
  return [
    "coverage_report",
    "visual_traits",
    "preprocessing_hypotheses",
    "cautions",
    "validation_status",
    "trigger_reason",
    "analysis_version",
  ].some((key) => key in record);
}

export function latestVisualAnalysis(analyses: DatasetVisualAnalysis[]) {
  return [...analyses]
    .filter(hasVisualAnalysisShape)
    .sort((left, right) => visualAnalysisSortScore(right) - visualAnalysisSortScore(left))[0] ?? null;
}

export function visualAnalysisSortScore(analysis: DatasetVisualAnalysis) {
  const timestamp = Date.parse(analysis.updated_at || analysis.created_at || "");
  if (Number.isFinite(timestamp)) return timestamp;
  return typeof analysis.analysis_version === "number" ? analysis.analysis_version : 0;
}

export function visualAnalysisFromProfile(profile: Record<string, unknown>) {
  const analyses: DatasetVisualAnalysis[] = [];
  for (const key of ["dataset_visual_analysis", "visual_analysis", "latest_visual_analysis", "visual_dataset_analysis"]) {
    const analysis = recordObject(profile[key]);
    if (hasVisualAnalysisShape(analysis)) analyses.push(analysis as DatasetVisualAnalysis);
  }
  analyses.push(...arrayVisualAnalyses(profile.visual_analyses));
  analyses.push(...arrayVisualAnalyses(profile.dataset_visual_analyses));
  return latestVisualAnalysis(analyses);
}

export function agentInvocationsFromResponse(response: AgentInvocationsResponse): AgentInvocation[] {
  return response.invocations ?? response.agent_invocations ?? response.items ?? [];
}

export function buildAgentInvocationAuditRows(invocations: AgentInvocation[], decisions: AgentDecision[]): AgentInvocationAuditRow[] {
  return invocations.map((invocation) => {
    const inputContext = recordObject(invocation.input_context);
    const runtime = invocationRuntimeRecord(inputContext);
    const downstreamOutcome = recordObject(invocation.downstream_outcome);
    const direct = recordObject(invocation);
    const records = [downstreamOutcome, runtime, inputContext, direct];
    const validationStatus =
      firstAuditString(records, ["backend_validation_status", "validation_status", "planner_validation_status"]) ||
      invocation.validation_status ||
      "unknown";
    const validationError = shortAuditText(
      firstAuditString(records, ["backend_validation_error", "validation_error", "planner_validation_error", "error"]),
      220,
    );
    const provider = invocation.provider || firstAuditString([runtime, inputContext], ["provider", "llm_provider"]);
    const model = invocation.model || firstAuditString([runtime, inputContext], ["model", "llm_model"]);
    const toolNames = uniqueStrings([
      ...auditToolNamesFromValue(firstAuditValue(records, ["tool_names", "toolNames", "tools"])),
      ...auditToolNamesFromValue(firstAuditValue(records, ["tool_calls", "toolCalls", "information_requests"])),
      ...auditToolNamesFromValue(firstAuditValue(records, ["tool_results", "toolResults"])),
    ]).slice(0, 8);

    return {
      id: invocation.id,
      agentName: invocation.agent_name || "agent",
      createdAt: invocation.created_at,
      target: invocationTarget(invocation),
      validationStatus,
      validationError,
      apiStyle: firstAuditString([runtime, inputContext, downstreamOutcome], ["api_style", "llm_api_style", "apiStyle", "style"]) || "-",
      providerModel: [provider, model].filter(Boolean).join(" / ") || "-",
      reasoningEffort:
        firstAuditString([runtime, inputContext, downstreamOutcome], ["reasoning_effort", "reasoningEffort", "effort"]) || "-",
      toolRounds: auditCountValue(firstAuditValue(records, ["tool_rounds", "toolRounds", "rounds", "round_count"])),
      toolNames,
      rejectedToolCalls: auditRejectedToolCallSummaries(
        firstAuditValue(records, ["rejected_tool_calls", "rejectedToolCalls", "tool_rejections"]),
      ),
      dryRunValidationResults: auditDryRunValidationSummaries(
        firstAuditValue(records, ["dry_run_validation_results", "dryRunValidationResults", "dry_run_validation", "validation_results"]),
      ),
      decisionLink: invocationDecisionLink(invocation, records, decisions),
    };
  });
}

export function buildMissionControlTelemetrySummary(
  telemetry: MissionControlTelemetryResponse | null,
  fallbackInvocations: AgentInvocation[],
): TelemetrySummary | null {
  const invocations =
    telemetry?.agent_invocations ??
    telemetry?.invocations ??
    fallbackInvocations ??
    [];
  const usageEvents =
    telemetry?.memory_embedding_usage_events ??
    telemetry?.embedding_usage_events ??
    telemetry?.usage_events ??
    [];
  if (invocations.length === 0 && usageEvents.length === 0) {
    return null;
  }

  const windows = telemetryWindowsFromInvocations(invocations);
  const callsByAgent = telemetryCountSummaryFromInvocations(invocations, (invocation) => invocation.agent_name || "agent");
  const callsByModel = telemetryCountSummaryFromInvocations(invocations, (invocation) => invocation.model || "unknown model");
  const topInvocations = telemetryInvocationSummaries(invocations).slice(0, 6);
  const promptSections = telemetryPromptSectionSummaries(invocations).slice(0, 8);
  const embedding = telemetryEmbeddingSummaryFromUsageEvents(usageEvents);

  return {
    invocations,
    usageEvents,
    windows,
    callsByAgent,
    callsByModel,
    topInvocations,
    promptSections,
    embedding,
  };
}

export function telemetryWindowsFromInvocations(invocations: AgentInvocation[]): TelemetryWindowSummary[] {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const sevenDayStart = new Date(todayStart);
  sevenDayStart.setDate(todayStart.getDate() - 6);
  const windows: Array<{ key: TelemetryWindowKey; label: string; since: Date }> = [
    { key: "today", label: "Today", since: todayStart },
    { key: "7d", label: "7-day", since: sevenDayStart },
    { key: "lifetime", label: "Lifetime", since: new Date(0) },
  ];

  return windows.map((window) => {
    const rows = invocations.filter((invocation) => telemetryInvocationTimestamp(invocation) >= window.since.getTime());
    return telemetryWindowSummary(window.key, window.label, rows);
  });
}

export function telemetryWindowSummary(key: TelemetryWindowKey, label: string, invocations: AgentInvocation[]): TelemetryWindowSummary {
  const rows = invocations.map((invocation) => telemetryInvocationUsage(invocation));
  return {
    key,
    label,
    calls: invocations.length,
    exactCalls: rows.filter((row) => row.usageKind === "exact").length,
    approxCalls: rows.filter((row) => row.usageKind !== "exact").length,
    validCalls: invocations.filter((invocation) => normalizedStatus(invocation.validation_status || "").toLowerCase() === "valid").length,
    invalidCalls: invocations.filter((invocation) => normalizedStatus(invocation.validation_status || "").toLowerCase() !== "valid").length,
    exactInputTokens: sumTelemetry(rows.map((row) => row.inputTokens)),
    exactOutputTokens: sumTelemetry(rows.map((row) => row.outputTokens)),
    approxInputTokens: sumTelemetry(rows.map((row) => row.approxInputTokens)),
    approxOutputTokens: sumTelemetry(rows.map((row) => row.approxOutputTokens)),
    cachedInputTokens: sumTelemetry(rows.map((row) => row.cachedInputTokens)),
    reasoningTokens: sumTelemetry(rows.map((row) => row.reasoningTokens)),
    estimatedCostUsd: sumTelemetry(rows.map((row) => row.estimatedCostUsd)),
  };
}

export function telemetryCountSummaryFromInvocations(
  invocations: AgentInvocation[],
  labelForInvocation: (invocation: AgentInvocation) => string,
): TelemetryCountSummary[] {
  const grouped = new Map<string, TelemetryCountSummary>();
  for (const invocation of invocations) {
    const label = labelForInvocation(invocation) || "unknown";
    const usage = telemetryInvocationUsage(invocation);
    const current = grouped.get(label) ?? {
      label,
      count: 0,
      exactCalls: 0,
      approxCalls: 0,
      inputTokens: 0,
      outputTokens: 0,
      approxInputTokens: 0,
      approxOutputTokens: 0,
      cachedInputTokens: 0,
      reasoningTokens: 0,
      estimatedCostUsd: 0,
    };
    current.count += 1;
    if (usage.usageKind === "exact") {
      current.exactCalls += 1;
      current.inputTokens += usage.inputTokens;
      current.outputTokens += usage.outputTokens;
    } else {
      current.approxCalls += 1;
      current.approxInputTokens += usage.approxInputTokens;
      current.approxOutputTokens += usage.approxOutputTokens;
    }
    current.cachedInputTokens += usage.cachedInputTokens;
    current.reasoningTokens += usage.reasoningTokens;
    current.estimatedCostUsd += usage.estimatedCostUsd;
    grouped.set(label, current);
  }
  return Array.from(grouped.values()).sort((left, right) => right.count - left.count || right.estimatedCostUsd - left.estimatedCostUsd);
}

export function telemetryInvocationSummaries(invocations: AgentInvocation[]): TelemetryInvocationSummary[] {
  return invocations
    .map((invocation) => telemetryInvocationSummary(invocation))
    .sort((left, right) => {
      if (right.estimatedCostUsd !== left.estimatedCostUsd) return right.estimatedCostUsd - left.estimatedCostUsd;
      const rightTokens = right.inputTokens + right.approxInputTokens + right.outputTokens + right.approxOutputTokens;
      const leftTokens = left.inputTokens + left.approxInputTokens + left.outputTokens + left.approxOutputTokens;
      if (rightTokens !== leftTokens) return rightTokens - leftTokens;
      return telemetryTimestampValue(right.createdAt) - telemetryTimestampValue(left.createdAt);
    });
}

export function telemetryInvocationSummary(invocation: AgentInvocation): TelemetryInvocationSummary {
  const usage = telemetryInvocationUsage(invocation);
  const sections = telemetryInvocationSections(invocation);
  const largestSection = sections[0]?.name || "";
  const promptBytes = telemetryInvocationPromptBytes(invocation);
  const runtime = telemetryInvocationRuntime(invocation);
  return {
    id: invocation.id,
    createdAt: invocation.created_at,
    agentName: invocation.agent_name || "agent",
    model: invocation.model || recordString(runtime, "model") || "unknown model",
    validationStatus: invocation.validation_status || "unknown",
    usageKind: usage.usageKind,
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    approxInputTokens: usage.approxInputTokens,
    approxOutputTokens: usage.approxOutputTokens,
    cachedInputTokens: usage.cachedInputTokens,
    reasoningTokens: usage.reasoningTokens,
    estimatedCostUsd: usage.estimatedCostUsd,
    promptBytes,
    sections,
    largestSection,
  };
}

export function telemetryInvocationUsage(
  invocation: AgentInvocation,
): {
  usageKind: "exact" | "approximate";
  inputTokens: number;
  outputTokens: number;
  approxInputTokens: number;
  approxOutputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  estimatedCostUsd: number;
} {
  const runtime = telemetryInvocationRuntime(invocation);
  const usage = recordObject(runtime.llm_usage);
  const exactInputTokens = Math.max(0, numberPayload(usage.input_tokens) ?? 0);
  const exactOutputTokens = Math.max(0, numberPayload(usage.output_tokens) ?? 0);
  const exactTotalTokens = Math.max(0, numberPayload(usage.total_tokens) ?? 0);
  const cachedInputTokens = Math.max(0, numberPayload(usage.cached_input_tokens) ?? 0);
  const reasoningTokens = Math.max(0, numberPayload(usage.reasoning_tokens) ?? 0);
  const hasExactUsage = exactInputTokens > 0 || exactOutputTokens > 0 || exactTotalTokens > 0 || cachedInputTokens > 0 || reasoningTokens > 0;
  const approxInputTokens = hasExactUsage ? 0 : telemetryApproximateTokensFromInput(invocation);
  const approxOutputTokens = hasExactUsage ? 0 : telemetryApproximateTokensFromText(invocation.raw_output || "");
  const pricing = telemetryPricingForModel(invocation.model || recordString(runtime, "model") || recordString(usage, "request_model"));
  const inputCostTokens = hasExactUsage ? exactInputTokens : approxInputTokens;
  const outputCostTokens = hasExactUsage ? exactOutputTokens : approxOutputTokens;
  const estimatedCostUsd =
    pricing.input > 0
      ? ((Math.max(0, inputCostTokens - cachedInputTokens) * pricing.input) +
          (cachedInputTokens * (pricing.cached ?? pricing.input)) +
          (outputCostTokens * pricing.output)) / 1_000_000
      : 0;

  return {
    usageKind: hasExactUsage ? "exact" : "approximate",
    inputTokens: exactInputTokens,
    outputTokens: exactOutputTokens,
    approxInputTokens,
    approxOutputTokens,
    cachedInputTokens,
    reasoningTokens,
    estimatedCostUsd,
  };
}

export function telemetryInvocationRuntime(invocation: AgentInvocation) {
  const inputContext = recordObject(invocation.input_context);
  return recordObject(inputContext.invocation_runtime);
}

export function telemetryPromptSectionSummaries(invocations: AgentInvocation[]): TelemetrySectionSummary[] {
  const grouped = new Map<string, TelemetrySectionSummary>();
  for (const invocation of invocations) {
    for (const section of telemetryInvocationSections(invocation)) {
      const key = section.name || "section";
      const current = grouped.get(key) ?? {
        name: key,
        calls: 0,
        bytes: 0,
        approxTokens: 0,
        exampleSource: section.exampleSource,
      };
      current.calls += section.calls;
      current.bytes += section.bytes;
      current.approxTokens += section.approxTokens;
      if (!current.exampleSource && section.exampleSource) {
        current.exampleSource = section.exampleSource;
      }
      grouped.set(key, current);
    }
  }
  return Array.from(grouped.values()).sort((left, right) => right.bytes - left.bytes || right.approxTokens - left.approxTokens);
}

export function telemetryApproximateTokensFromInput(invocation: AgentInvocation) {
  const messageBytes = byteLengthOfJson(invocation.input_messages ?? []);
  return Math.max(0, Math.ceil(messageBytes / 4));
}

export function telemetryApproximateTokensFromText(text: string) {
  return Math.max(0, Math.ceil(byteLengthOfText(text) / 4));
}

export function telemetryInvocationPromptBytes(invocation: AgentInvocation) {
  return byteLengthOfJson(invocation.input_messages ?? []);
}

export function telemetryInvocationSections(invocation: AgentInvocation) {
  const inputContext = recordObject(invocation.input_context);
  const snapshots = [
    { name: "planner_context_snapshot", snapshot: recordObject(inputContext.planner_context_snapshot) },
    { name: "training_monitor_context_snapshot", snapshot: recordObject(inputContext.training_monitor_context_snapshot) },
  ].filter(({ snapshot }) => Object.keys(snapshot).length > 0);
  if (snapshots.length === 0) {
    return [];
  }

  const sections = snapshots.flatMap(({ name, snapshot }) =>
    telemetrySectionsFromSnapshot(snapshot).map((section) => ({
      ...section,
      exampleSource: section.exampleSource || `${invocation.agent_name || "agent"} - ${name}`,
    })),
  );
  const grouped = new Map<string, TelemetrySectionSummary>();
  for (const section of sections) {
    const key = section.name || "section";
    const current = grouped.get(key) ?? {
      name: key,
      calls: 0,
      bytes: 0,
      approxTokens: 0,
      exampleSource: section.exampleSource,
    };
    current.calls += section.calls;
    current.bytes += section.bytes;
    current.approxTokens += section.approxTokens;
    if (!current.exampleSource && section.exampleSource) {
      current.exampleSource = section.exampleSource;
    }
    grouped.set(key, current);
  }
  return Array.from(grouped.values()).sort((left, right) => right.bytes - left.bytes || right.approxTokens - left.approxTokens);
}

export function telemetrySectionsFromSnapshot(snapshot: Record<string, unknown>) {
  const promptBudget = recordObject(snapshot.prompt_budget ?? snapshot.promptBudget);
  const explicitEstimates = telemetrySectionEstimateList(promptBudget.section_estimates ?? promptBudget.sectionEstimates);
  if (explicitEstimates.length > 0) {
    return explicitEstimates;
  }

  const out: TelemetrySectionSummary[] = [];
  for (const [key, value] of Object.entries(snapshot)) {
    if (["context_version", "prompt_budget"].includes(key)) {
      continue;
    }
    if (value === undefined || value === null) {
      continue;
    }
    const bytes = byteLengthOfJson(value);
    if (bytes <= 0) {
      continue;
    }
    out.push({
      name: key,
      calls: 1,
      bytes,
      approxTokens: Math.max(1, Math.ceil(bytes / 4)),
    });
  }
  return out;
}

export function telemetrySectionEstimateList(value: unknown): TelemetrySectionSummary[] {
  if (Array.isArray(value)) {
    return value.flatMap((item) => telemetrySectionEstimateList(item));
  }
  const record = recordObject(value);
  if (Object.keys(record).length === 0) {
    return [];
  }

  const name = recordString(record, "name") || recordString(record, "section") || recordString(record, "label") || recordString(record, "id");
  const bytes = Math.max(
    0,
    recordNumber(record, "bytes") ||
      recordNumber(record, "byte_estimate") ||
      recordNumber(record, "approx_bytes") ||
      recordNumber(record, "approximate_bytes"),
  );
  const approxTokens = Math.max(
    0,
    recordNumber(record, "approx_tokens") ||
      recordNumber(record, "approximate_tokens") ||
      recordNumber(record, "token_estimate") ||
      recordNumber(record, "approx_token_estimate"),
  );
  const calls = Math.max(
    1,
    recordNumber(record, "calls") || recordNumber(record, "count") || recordNumber(record, "invocations") || 0,
  );
  if (!name && bytes === 0 && approxTokens === 0) {
    return [];
  }
  return [
    {
      name: name || "section",
      calls,
      bytes,
      approxTokens: approxTokens || (bytes > 0 ? Math.max(1, Math.ceil(bytes / 4)) : 0),
      exampleSource: recordString(record, "source") || recordString(record, "example_source") || recordString(record, "exampleSource") || undefined,
    },
  ];
}

export function telemetryEmbeddingSummaryFromUsageEvents(events: MemoryEmbeddingUsageEvent[]) {
  const sourceIndexEvents = events.filter((event) => telemetryEmbeddingPurpose(event) === "source_index");
  const retrievalEvents = events.filter((event) => telemetryEmbeddingPurpose(event) === "retrieval_query");

  return {
    totalUsageEvents: events.length,
    sourceIndex: telemetryEmbeddingPurposeSummary("source_index", sourceIndexEvents),
    retrievalQuery: telemetryEmbeddingPurposeSummary("retrieval_query", retrievalEvents),
  };
}

export function telemetryEmbeddingPurposeSummary(purpose: string, events: MemoryEmbeddingUsageEvent[]): TelemetryEmbeddingPurposeSummary {
  const bySourceTable = telemetryEmbeddingBreakdownSummary(events, (event) => event.source_table || event.source_id || "unknown");
  const byModel = telemetryEmbeddingBreakdownSummary(events, (event) => event.embedding_model || "unknown");
  const retrievalPurpose =
    purpose === "retrieval_query"
      ? uniqueStrings(
          events
            .map((event) => event.retrieval_purpose || recordString(recordObject(event.metadata), "retrieval_purpose"))
            .filter((value): value is string => Boolean(value)),
        ).join(", ")
      : "";
  return {
    purpose,
    retrievalPurpose,
    count: events.length,
    providerCalls: sumTelemetry(events.map((event) => event.provider_call_count || 0)),
    inputBytes: sumTelemetry(events.map((event) => event.input_bytes || 0)),
    estimatedCostUsd: sumTelemetry(events.map((event) => telemetryEmbeddingEstimatedCost(event))),
    retrievedCount: sumTelemetry(events.map((event) => event.retrieved_count || 0)),
    injected: events.filter((event) => event.injected === true).length,
    logOnly: events.filter((event) => event.log_only === true).length,
    cached: events.filter((event) => event.cached === true).length,
    skipped: events.filter((event) => event.skipped === true).length,
    bySourceTable,
    byModel,
  };
}

export function telemetryEmbeddingBreakdownSummary(
  events: MemoryEmbeddingUsageEvent[],
  labelForEvent: (event: MemoryEmbeddingUsageEvent) => string,
): TelemetryEmbeddingBreakdownRow[] {
  const grouped = new Map<string, TelemetryEmbeddingBreakdownRow>();
  for (const event of events) {
    const label = labelForEvent(event) || "unknown";
    const current = grouped.get(label) ?? {
      label,
      count: 0,
      providerCalls: 0,
      inputBytes: 0,
      estimatedCostUsd: 0,
      retrievedCount: 0,
      injected: 0,
      logOnly: 0,
      cached: 0,
      skipped: 0,
    };
    current.count += 1;
    current.providerCalls += Math.max(0, event.provider_call_count || 0);
    current.inputBytes += Math.max(0, event.input_bytes || 0);
    current.estimatedCostUsd += telemetryEmbeddingEstimatedCost(event);
    current.retrievedCount += Math.max(0, event.retrieved_count || 0);
    current.injected += event.injected === true ? 1 : 0;
    current.logOnly += event.log_only === true ? 1 : 0;
    current.cached += event.cached === true ? 1 : 0;
    current.skipped += event.skipped === true ? 1 : 0;
    grouped.set(label, current);
  }
  return Array.from(grouped.values()).sort((left, right) => right.count - left.count || right.inputBytes - left.inputBytes);
}

export function telemetryEmbeddingPurpose(event: MemoryEmbeddingUsageEvent) {
  const purpose = String(event.purpose || "").toLowerCase().trim();
  if (purpose === "source_index" || purpose === "retrieval_query") {
    return purpose;
  }
  if (purpose.includes("source")) return "source_index";
  if (purpose.includes("retrieval")) return "retrieval_query";
  return "source_index";
}

export function telemetryEmbeddingEstimatedCost(event: MemoryEmbeddingUsageEvent) {
  const model = String(event.embedding_model || "").toLowerCase();
  const pricing = telemetryEmbeddingPricingForModel(model);
  const tokens = Math.max(0, event.input_bytes || 0) / 4;
  return (tokens * pricing.input) / 1_000_000;
}

export function telemetryEmbeddingPricingForModel(model: string) {
  if (model.includes("text-embedding-3-large")) {
    return { input: 0.13 };
  }
  if (model.includes("text-embedding-3-small")) {
    return { input: 0.02 };
  }
  return { input: 0.02 };
}

export function telemetryPricingForModel(model: string) {
  const lower = model.toLowerCase();
  if (lower.includes("gpt-5.4-pro")) return { input: 30, cached: 3, output: 180 };
  if (lower.includes("gpt-5.4-mini")) return { input: 0.75, cached: 0.075, output: 4.5 };
  if (lower.includes("gpt-5.4")) return { input: 2.5, cached: 0.25, output: 15 };
  if (lower.includes("gpt-5 mini")) return { input: 0.25, cached: 0.025, output: 2 };
  return { input: 0.75, cached: 0.075, output: 4.5 };
}

export function telemetryInvocationTimestamp(invocation: AgentInvocation) {
  const timestamp = Date.parse(invocation.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export function telemetryTimestampValue(value: string) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export function sumTelemetry(values: number[]) {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
}

export function byteLengthOfText(value: string) {
  return new TextEncoder().encode(value).length;
}

export function byteLengthOfJson(value: unknown) {
  try {
    return byteLengthOfText(JSON.stringify(value) ?? "");
  } catch {
    return 0;
  }
}

export function invocationRuntimeRecord(inputContext: Record<string, unknown>) {
  for (const key of ["invocation_runtime", "runtime", "llm_runtime", "responses_runtime"]) {
    const runtime = recordObject(inputContext[key]);
    if (Object.keys(runtime).length > 0) return runtime;
  }
  return {};
}

export function invocationTarget(invocation: AgentInvocation) {
  const parts = [
    invocation.plan_id ? `plan ${invocation.plan_id}` : "",
    invocation.job_id ? `job ${invocation.job_id}` : "",
    invocation.dataset_id ? `dataset ${invocation.dataset_id}` : "",
  ].filter(Boolean);
  return parts.join(" - ") || (invocation.project_id ? `project ${invocation.project_id}` : "project scope");
}

export function invocationDecisionLink(
  invocation: AgentInvocation,
  records: Record<string, unknown>[],
  decisions: AgentDecision[],
) {
  const directDecisionID = firstAuditString(records, ["decision_id", "final_decision_id"]);
  const sourceDecisionID = firstAuditString(records, ["source_decision_id"]);
  const linkedDecision =
    decisions.find((decision) => decision.id === directDecisionID) ??
    decisions.find((decision) => recordString(decision.payload ?? {}, "invocation_id") === invocation.id) ??
    decisions.find((decision) => decision.id === sourceDecisionID);

  if (linkedDecision) return `${linkedDecision.id} / ${linkedDecision.decision_type}`;
  if (directDecisionID) return directDecisionID;
  if (sourceDecisionID) return `source ${sourceDecisionID}`;
  return "-";
}

export function visualAnalysisFacts(visualAnalysis: VisualAnalysisDetail, activeJob: Job | null): InsightItem[] {
  const analysis = visualAnalysis.analysis;
  const policy = visualAnalysis.rerunPolicy ?? null;
  const coverage = recordObject(analysis?.coverage_report);
  const status = visualAnalysisStatusBadge(visualAnalysis, activeJob);
  const imagesAnalyzed = recordNumber(coverage, "images_analyzed") || analysis?.images_analyzed || 0;
  const imagesAvailable = recordNumber(coverage, "images_available") || analysis?.total_images || 0;
  const classesCovered = recordNumber(coverage, "classes_covered");
  const classesTotal = recordNumber(coverage, "classes_total");
  const classCoverageRatio = numberPayload(coverage.class_coverage_ratio);
  const classCoverage = classesTotal
    ? `${classesCovered}/${classesTotal}`
    : classCoverageRatio !== null
      ? formatPercent(classCoverageRatio)
      : "-";

  return [
    { label: "Status", value: status, tone: visualStatusTone(status) },
    { label: "Trigger", value: analysis?.trigger_reason || (activeJob ? "manual" : "-") },
    {
      label: "Runs/Profile",
      value: visualAnalysisPolicyRunCount(policy),
      tone: policy && policy.max_runs_per_profile && (policy.runs_for_profile ?? 0) >= policy.max_runs_per_profile ? "bad" : "good",
    },
    {
      label: "Rerun",
      value: visualAnalysisPolicyReadiness(policy),
      tone: policy?.manual_run_allowed === false ? "warn" : "good",
    },
    {
      label: "Images",
      value: imagesAvailable ? `${imagesAnalyzed}/${imagesAvailable}` : imagesAnalyzed ? String(imagesAnalyzed) : "-",
      tone: imagesAnalyzed > 0 ? "good" : "warn",
    },
    {
      label: "Class Coverage",
      value: classCoverage,
      tone: classesTotal && classesCovered < classesTotal ? "warn" : classesCovered > 0 ? "good" : "warn",
    },
    { label: "Traits", value: String(analysis?.visual_traits?.length ?? 0) },
    { label: "Hypotheses", value: String(analysis?.preprocessing_hypotheses?.length ?? 0), tone: "warn" },
  ];
}

export function visualStatusTone(status: string): InsightItem["tone"] {
  const normalized = normalizedStatus(status).toLowerCase();
  if (["accepted", "available", "succeeded", "completed", "ready"].includes(normalized)) return "good";
  if (["failed", "error", "rejected", "unsupported"].includes(normalized)) return "bad";
  return "warn";
}

export function visualAnalysisStatusBadge(visualAnalysis: VisualAnalysisDetail, activeJob: Job | null) {
  if (activeJob) return normalizedStatus(activeJob.status);
  if (visualAnalysis.analysis?.validation_status) return normalizedStatus(visualAnalysis.analysis.validation_status);
  if (visualAnalysis.status === "available") return "AVAILABLE";
  if (visualAnalysis.status === "unsupported") return "UNSUPPORTED";
  if (visualAnalysis.status === "error") return "ERROR";
  return "NOT_RECORDED";
}

export function visualAnalysisActiveJob(jobs: Job[], datasetId: string) {
  return jobs.find((job) => {
    const status = normalizedStatus(job.status);
    if (!["QUEUED", "ASSIGNED", "RUNNING", "REQUESTED", "PENDING"].includes(status)) return false;
    const template = job.template.toLowerCase();
    if (!template.includes("visual")) return false;
    const jobDatasetId = typeof job.config.dataset_id === "string" ? job.config.dataset_id : "";
    return !datasetId || !jobDatasetId || jobDatasetId === datasetId;
  }) ?? null;
}

export function visualAnalysisRerunDisabledReason(
  visualAnalysis: VisualAnalysisDetail,
  dataset: Dataset | null,
  activeJob: Job | null,
  loading: boolean,
) {
  if (!dataset) return "Upload a dataset first.";
  if (activeJob) return `Visual analysis job is already ${activeJob.status.toLowerCase()}.`;
  if (!visualAnalysis.manualRunSupported) return "Backend manual visual-analysis endpoint is not available.";
  const policy = visualAnalysis.rerunPolicy;
  if (policy?.manual_run_allowed === false) {
    return policy.disabled_reason || policy.reason || "Manual visual-analysis rerun is currently blocked by backend policy.";
  }
  if (loading) return "Another Mission Control request is in progress.";
  return "";
}

export function visualAnalysisPolicyRunCount(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "-";
  const runs = policy.runs_for_profile ?? 0;
  const maxRuns = policy.max_runs_per_profile ?? 0;
  return maxRuns ? `${runs}/${maxRuns}` : String(runs);
}

export function visualAnalysisPolicyReadiness(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "-";
  if (policy.active_job_status) return normalizedStatus(policy.active_job_status);
  if (policy.manual_run_allowed === false) {
    if (policy.next_allowed_at) return `After ${new Date(policy.next_allowed_at).toLocaleTimeString()}`;
    return "Blocked";
  }
  return "Ready";
}

export function visualAnalysisPolicySummary(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "";
  if (policy.disabled_reason) return policy.disabled_reason;
  if (policy.reason) return policy.reason;
  const triggerCount = policy.deficiency_triggers?.length ?? 0;
  if (triggerCount > 0) return `${triggerCount} deficiency trigger(s) active; backend may queue a rare reanalysis if automation is enabled.`;
  return policy.automation_enabled ? "Automatic initial and deficiency reanalysis are enabled." : "Automatic visual analysis is disabled; manual rerun is still backend-gated.";
}

export function visualAnalysisLimitations(analysis: DatasetVisualAnalysis | null) {
  if (!analysis) return [];
  const coverage = recordObject(analysis.coverage_report);
  return uniqueStrings([
    ...stringArrayPayload(analysis.limitations),
    ...stringArrayPayload(coverage.limitations),
  ]).slice(0, 8);
}

export function visualCoverageSummary(analysis: DatasetVisualAnalysis) {
  const coverage = recordObject(analysis.coverage_report);
  const strategy = recordString(coverage, "selection_strategy") || "bounded visual sample";
  const imagesAnalyzed = recordNumber(coverage, "images_analyzed") || analysis.images_analyzed || 0;
  const imagesAvailable = recordNumber(coverage, "images_available") || analysis.total_images || 0;
  const classesCovered = recordNumber(coverage, "classes_covered");
  const classesTotal = recordNumber(coverage, "classes_total");
  const edgeCases = recordNumber(coverage, "edge_case_count");
  const hardExamples = recordNumber(coverage, "hard_example_count");
  const highDetail = recordNumber(coverage, "high_detail_image_count");
  return [
    strategy,
    imagesAvailable ? `${imagesAnalyzed}/${imagesAvailable} images analyzed` : imagesAnalyzed ? `${imagesAnalyzed} images analyzed` : "",
    classesTotal ? `${classesCovered}/${classesTotal} classes covered` : "",
    edgeCases ? `${edgeCases} edge case(s)` : "",
    hardExamples ? `${hardExamples} hard example(s)` : "",
    highDetail ? `${highDetail} high-detail image(s)` : "",
  ]
    .filter(Boolean)
    .join(" - ");
}

export function visualPerClassCoverageRows(coverage: Record<string, unknown>) {
  const counts = recordObject(coverage.per_class_counts);
  return Object.entries(counts)
    .map(([label, value]) => ({ label, value: String(numericValue(value)) }))
    .filter((row) => row.value !== "0")
    .slice(0, 10);
}

export function buildDecisionChatTurns(decisions: AgentDecision[]): DecisionChatTurn[] {
  return decisions
    .slice()
    .reverse()
    .map((decision, index) => ({
      decision,
      question: decisionChatQuestion(decision, index),
      opening: decisionChatOpening(decision),
      highlights: decisionHighlights(decision),
      sections: decisionReasoningSections(decision).filter((section) => section.title !== "Summary"),
      retrievedMemory: decisionRetrievedMemoryRows(decision),
      rejections: decisionRejections(decision),
      mechanismCoverage: mechanismCoverageRows(decision.payload),
      candidateScores: candidateScoreRows(decision),
    }));
}

export function decisionChatQuestion(decision: AgentDecision, index: number) {
  const payload = decision.payload ?? {};
  const planID =
    decision.plan_id ||
    recordFirstString(payload, ["plan_id", "source_plan_id", "followup_plan_id", "follow_up_plan_id"]) ||
    `decision_${index + 1}`;
  return `What is ${planID}?`;
}

export function decisionChatOpening(decision: AgentDecision) {
  const summary = recordString(decision.payload ?? {}, "summary");
  return uniqueStrings([summary, decision.rationale]).join(" ");
}

export function decisionReasoningSections(decision: AgentDecision): ReasoningSection[] {
  const payload = decision.payload ?? {};
  const diagnosis = recordObject(payload.deterministic_diagnosis);
  const sections: ReasoningSection[] = [];
  const addSection = (title: string, items: string[]) => {
    const filtered = uniqueStrings(items.map((item) => item.trim()).filter(Boolean)).slice(0, 5);
    if (filtered.length > 0) sections.push({ title, items: filtered });
  };

  addSection("Summary", [recordString(payload, "summary"), decision.rationale]);
  addSection("Evidence", [
    ...stringArrayPayload(payload.evidence_used),
    ...stringArrayPayload(payload.deterministic_diagnosis_used),
  ]);
  addSection("Diagnosis", [
    ...stringArrayPayload(diagnosis.recommended_failure_modes),
    ...stringArrayPayload(payload.stop_signals),
  ]);
  addSection("Hypothesis", [
    recordString(payload, "hypothesis"),
    recordString(payload, "why_can_beat_champion"),
    recordString(payload, "success_criteria"),
  ]);
  addSection("Mechanism", mechanismDecisionSummaries(payload));
  addSection("Proposed Experiments", proposedExperimentSummaries(payload.proposed_experiments));
  addSection("Rejected Options", rejectedOptionSummaries(payload.rejected_options));
  addSection("Validation", [
    recordFirstString(payload, ["backend_validation_status", "validation_status", "planner_validation_status"]),
    recordFirstString(payload, ["backend_validation_error", "validation_error", "planner_validation_error"]),
    recordString(payload, "backend_stop_guard"),
  ]);
  addSection("Tradeoffs", [
    recordString(payload, "deployment_tradeoff"),
    ...stringArrayPayload(payload.expected_tradeoffs),
    ...stringArrayPayload(payload.changed_variables).map((item) => `changed ${item}`),
  ]);
  addSection("Risks", [
    ...stringArrayPayload(payload.risks),
    ...stringArrayPayload(payload.expected_failure_modes),
    recordString(payload, "stop_condition"),
  ]);
  const confidence = numberPayload(payload.confidence);
  addSection("Confidence", confidence === null ? [] : [`${Math.round(confidence * 100)}% confidence`]);
  return sections;
}

export function decisionRejections(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  const items: Array<{ kind: string; text: string }> = [];

  for (const option of Array.isArray(payload.rejected_options) ? payload.rejected_options : []) {
    const record = recordObject(option);
    const optionName = recordString(record, "option") || "Planner option";
    const reason = recordString(record, "reason") || recordString(record, "evidence");
    items.push({ kind: classifyRejectionReason(reason || optionName), text: reason ? `${optionName}: ${reason}` : optionName });
  }

  for (const ranking of Array.isArray(payload.candidate_rankings) ? payload.candidate_rankings : []) {
    const record = recordObject(ranking);
    if (record.rejected !== true) continue;
    const reasons = stringArrayPayload(record.reasons);
    const hypothesis = recordString(record, "hypothesis") || recordString(record, "experiment_signature") || "Candidate";
    const reasonText = reasons.slice(0, 2).join("; ") || "Rejected by deterministic backend ranking.";
    items.push({ kind: classifyRejectionReason(reasonText), text: `${hypothesis}: ${reasonText}` });
  }

  const validationError = recordString(payload, "validation_error");
  if (validationError) {
    items.push({ kind: classifyRejectionReason(validationError), text: validationError });
  }

  const backendValidationStatus = recordFirstString(payload, [
    "backend_validation_status",
    "validation_status",
    "planner_validation_status",
  ]);
  if (backendValidationStatus && !["accepted", "approved", "valid"].includes(backendValidationStatus.toLowerCase())) {
    const backendValidationError = recordFirstString(payload, [
      "backend_validation_error",
      "validation_error",
      "planner_validation_error",
    ]);
    items.push({
      kind: classifyRejectionReason(backendValidationError || backendValidationStatus),
      text: backendValidationError ? `${backendValidationStatus}: ${backendValidationError}` : backendValidationStatus,
    });
  }

  const backendStopGuard = recordString(payload, "backend_stop_guard");
  if (backendStopGuard) {
    items.push({ kind: "Stop guard", text: backendStopGuard });
  }

  return items.slice(0, 8);
}

export function buildDecisionQualitySnapshot(decisions: AgentDecision[], invocations: AgentInvocation[]): DecisionQualitySnapshot | null {
  const decision = latestDecisionWithQualityContext(decisions);
  const payload = decision?.payload ?? {};
  const trajectory = firstNonEmptyRecord([
    recordObject(payload.project_trajectory_card),
    recordObject(payload.projectTrajectoryCard),
    latestProjectTrajectoryFromInvocations(invocations),
  ]);

  const rankings = Array.isArray(payload.candidate_rankings) ? payload.candidate_rankings.map(recordObject) : [];
  const candidateHypotheses = Array.isArray(payload.candidate_hypotheses) ? payload.candidate_hypotheses.length : 0;
  const proposedExperiments = Array.isArray(payload.proposed_experiments) ? payload.proposed_experiments.length : 0;
  const selectedCandidates = rankings.filter((ranking) => ranking.selected === true).length || proposedExperiments;
  const rejectedCandidates = rankings.filter((ranking) => ranking.rejected === true).length;
  const totalCandidates = rankings.length || candidateHypotheses || selectedCandidates + rejectedCandidates;
  const topRejectionReason =
    topCandidateRejectionReason(rankings) || (decision ? decisionRejections(decision)[0]?.text ?? "" : "");
  const hasTrajectory = Object.keys(trajectory).length > 0;
  const hasQualityPayload = hasTrajectory || rankings.length > 0 || candidateHypotheses > 0 || proposedExperiments > 0 || Boolean(topRejectionReason);
  if (!decision && !hasTrajectory) return null;
  if (!hasQualityPayload) return null;

  return {
    decisionId: decision?.id ?? "no decision",
    decisionType: decision?.decision_type ?? "TRAJECTORY",
    createdAt: decision?.created_at ?? latestInvocationTimestamp(invocations),
    source: hasTrajectory ? "project trajectory" : "decision payload",
    decisionPressure:
      recordFirstString(trajectory, ["decision_pressure", "DecisionPressure"]) ||
      recordFirstString(payload, ["decision_pressure", "project_decision_pressure"]) ||
      "normal",
    blockedMechanisms: uniqueStrings([
      ...stringArrayPayload(trajectory.blocked_mechanisms),
      ...stringArrayPayload(trajectory.BlockedMechanisms),
    ]),
    completedTrainingRuns: recordFirstNumber(trajectory, ["completed_training_runs", "CompletedTrainingRuns"]),
    completedPlannerRounds: recordFirstNumber(trajectory, ["completed_planner_rounds", "CompletedPlannerRounds"]),
    gainPerCompletedRun: recordFirstNumber(trajectory, ["gain_per_completed_run", "GainPerCompletedRun"]),
    recentBestDelta: recordFirstNumber(trajectory, ["recent_best_delta", "RecentBestDelta"]),
    minimumUsefulDelta: recordFirstNumber(trajectory, ["minimum_useful_delta", "MinimumUsefulDelta"]),
    selectedCandidates,
    totalCandidates,
    rejectedCandidates,
    topRejectionReason,
    exhaustedOutcomes: decisionQualityMechanismOutcomes(trajectory),
    warnings: uniqueStrings([...stringArrayPayload(trajectory.warnings), ...stringArrayPayload(trajectory.Warnings)]).slice(0, 6),
  };
}

export function latestDecisionWithQualityContext(decisions: AgentDecision[]) {
  const sorted = [...decisions].sort((left, right) => timestampSortScore(right.created_at) - timestampSortScore(left.created_at));
  return (
    sorted.find((decision) => Object.keys(recordObject(decision.payload?.project_trajectory_card)).length > 0) ??
    sorted.find((decision) => Array.isArray(decision.payload?.candidate_rankings)) ??
    sorted[0] ??
    null
  );
}

export function latestProjectTrajectoryFromInvocations(invocations: AgentInvocation[]) {
  const sorted = [...invocations].sort((left, right) => timestampSortScore(right.created_at) - timestampSortScore(left.created_at));
  for (const invocation of sorted) {
    if (invocation.agent_name && invocation.agent_name !== "experiment_planner") continue;
    const inputContext = recordObject(invocation.input_context);
    const direct = recordObject(inputContext.project_trajectory_card ?? inputContext.projectTrajectoryCard);
    if (Object.keys(direct).length > 0) return direct;

    const snapshot = recordObject(inputContext.planner_context_snapshot ?? inputContext.plannerContextSnapshot);
    const nested = recordObject(snapshot.project_trajectory_card ?? snapshot.projectTrajectoryCard);
    if (Object.keys(nested).length > 0) return nested;
  }
  return {};
}

export function latestInvocationTimestamp(invocations: AgentInvocation[]) {
  const latest = [...invocations].sort((left, right) => timestampSortScore(right.created_at) - timestampSortScore(left.created_at))[0];
  return latest?.created_at ?? "";
}

export function firstNonEmptyRecord(records: Record<string, unknown>[]) {
  return records.find((record) => Object.keys(record).length > 0) ?? {};
}

export function topCandidateRejectionReason(rankings: Record<string, unknown>[]) {
  for (const ranking of rankings) {
    if (ranking.rejected !== true) continue;
    const reasons = stringArrayPayload(ranking.reasons);
    if (reasons.length > 0) return reasons[0];
    const validationError = recordFirstString(ranking, ["backend_validation_error", "validation_error", "planner_validation_error"]);
    if (validationError) return validationError;
    const status = recordFirstString(ranking, ["backend_validation_status", "validation_status", "planner_validation_status"]);
    if (status) return status;
  }
  return "";
}

export function decisionQualityMechanismOutcomes(trajectory: Record<string, unknown>): MechanismCoverageRow[] {
  const outcomesValue = trajectory.mechanism_outcomes ?? trajectory.MechanismOutcomes;
  const outcomes = Array.isArray(outcomesValue) ? outcomesValue.map(recordObject) : [];
  return outcomes
    .map((record) => {
      const mechanism = recordFirstString(record, ["mechanism", "Mechanism"]);
      const status = recordFirstString(record, ["status", "Status"]);
      return {
        mechanism,
        status: status ? status.toUpperCase() : "OUTCOME",
        detail: decisionQualityOutcomeDetail(record),
      };
    })
    .filter(
      (row) =>
        row.mechanism &&
        ["BLOCKED", "EXHAUSTED"].includes(row.status.toUpperCase()),
    )
    .slice(0, 8);
}

export function decisionQualityOutcomeDetail(record: Record<string, unknown>) {
  const reason = recordFirstString(record, ["exhaustion_reason", "ExhaustionReason", "reason", "detail"]);
  if (reason) return reason;
  const attempts = recordFirstNumber(record, ["attempt_count", "AttemptCount"]);
  const recentDelta = recordFirstNumber(record, ["recent_best_delta", "RecentBestDelta"]);
  const parts = [
    attempts !== null ? `${attempts} attempt(s)` : "",
    recentDelta !== null ? `recent delta ${formatDecisionQualityMetric(recentDelta, true)}` : "",
  ].filter(Boolean);
  return parts.join(" - ") || "governor status";
}

export function recordFirstNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = numberPayload(record[key]);
    if (value !== null) return value;
  }
  return null;
}

export function timestampSortScore(value?: string) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export function buildChampionComparison(
  summaries: TrainingRunSummary[],
  evaluations: TrainingRunEvaluation[],
  jobs: Job[],
  champion: ProjectChampion | null,
): ChampionComparisonRow[] {
  const evaluationByJob = new Map(evaluations.map((evaluation) => [evaluation.job_id, evaluation]));
  const jobById = new Map(jobs.map((job) => [job.id, job]));
  const seedVarianceBySignature = buildSeedVarianceBySignature(summaries, jobById, evaluationByJob);
  const championJobId = champion?.job_id ?? "";
  return summaries
    .map((summary) => {
      const evaluation = evaluationByJob.get(summary.job_id);
      const job = jobById.get(summary.job_id);
      const modelProfile = evaluation?.model_profile ?? {};
      const holisticScores = evaluation?.holistic_scores ?? {};
      const diagnostics = recordObject(holisticScores.training_diagnostics);
      const objectiveFit = recordNumber(holisticScores, "overall_score") || recordNumber(holisticScores, "objective_fit");
      const trainValidationGap =
        numberPayload(diagnostics.train_validation_gap) ?? numberPayload(holisticScores.train_validation_gap);
      const divergenceStatus =
        recordString(diagnostics, "status") || recordString(holisticScores, "divergence_status") || "";
      const signature = job ? experimentComparisonSignature(job.config) : "";
      const seedVariance = signature ? seedVarianceBySignature.get(signature) : undefined;
      const latencyMs = recordNumber(modelProfile, "estimated_latency_ms");
      const primaryMetric = runPrimaryMetric(summary, evaluation ?? null, job ?? null);
      const rankScore = experimentRankScore({
        primaryMetric: primaryMetric.value,
        secondaryMetric: primaryMetric.secondaryValue,
        costUsd: summary.estimated_cost_usd,
        runtimeSeconds: summary.runtime_seconds,
        latencyMs,
        objectiveFit,
        trainValidationGap,
        divergenceStatus,
        seedVariance: seedVariance?.variance ?? null,
      });
      return {
        jobId: summary.job_id,
        model: summary.model,
        rankScore,
        primaryMetricLabel: primaryMetric.label,
        primaryMetricValue: primaryMetric.value,
        secondaryMetricLabel: primaryMetric.secondaryLabel,
        secondaryMetricValue: primaryMetric.secondaryValue,
        runtimeSeconds: summary.runtime_seconds,
        costUsd: summary.estimated_cost_usd,
        latencyMs,
        objectiveFit,
        trainValidationGap,
        divergenceStatus,
        seedVariance: seedVariance?.variance ?? null,
        seedRunCount: seedVariance?.runCount ?? 0,
        isChampion: summary.job_id === championJobId,
      };
    })
    .sort((left, right) => {
      if (left.isChampion !== right.isChampion) return left.isChampion ? -1 : 1;
      return right.rankScore - left.rankScore;
    })
    .slice(0, 8);
}

export function buildSeedVarianceBySignature(
  summaries: TrainingRunSummary[],
  jobById: Map<string, Job>,
  evaluationByJob: Map<string, TrainingRunEvaluation>,
) {
  const groups = new Map<string, Array<{ seed: string; score: number }>>();
  for (const summary of summaries) {
    const job = jobById.get(summary.job_id);
    if (!job || effectiveTrainingRunStatus(summary, job) !== "SUCCEEDED") continue;
    const signature = experimentComparisonSignature(job.config);
    if (!signature) continue;
    const seed = experimentSeed(job.config);
    if (!seed) continue;
    const rows = groups.get(signature) ?? [];
    rows.push({ seed, score: runPrimaryMetric(summary, evaluationByJob.get(summary.job_id) ?? null, job).value });
    groups.set(signature, rows);
  }

  const out = new Map<string, { variance: number; runCount: number }>();
  for (const [signature, rows] of groups.entries()) {
    const uniqueSeeds = new Set(rows.map((row) => row.seed));
    if (rows.length < 2 || uniqueSeeds.size < 2) continue;
    const mean = rows.reduce((sum, row) => sum + row.score, 0) / rows.length;
    const variance = rows.reduce((sum, row) => sum + Math.pow(row.score - mean, 2), 0) / (rows.length - 1);
    out.set(signature, { variance, runCount: rows.length });
  }
  return out;
}

export function experimentRankScore(input: {
  primaryMetric: number;
  secondaryMetric: number;
  costUsd: number;
  runtimeSeconds: number;
  latencyMs: number;
  objectiveFit: number;
  trainValidationGap: number | null;
  divergenceStatus: string;
  seedVariance: number | null;
}) {
  const quality = input.primaryMetric * 0.72 + input.secondaryMetric * 0.28;
  const latencyScore = input.latencyMs ? clamp01(1 - input.latencyMs / 160) : 0.5;
  const costScore = clamp01(1 - input.costUsd / 10);
  const runtimeScore = clamp01(1 - input.runtimeSeconds / 1800);
  const base = input.objectiveFit || quality * 0.74 + latencyScore * 0.12 + costScore * 0.08 + runtimeScore * 0.06;
  const gapPenalty =
    input.trainValidationGap !== null && input.trainValidationGap > 0
      ? Math.min(0.08, input.trainValidationGap * 0.08)
      : 0;
  const divergencePenalty =
    input.divergenceStatus === "diverging" ? 0.07 : input.divergenceStatus === "overfitting_risk" ? 0.04 : 0;
  const seedPenalty = input.seedVariance !== null ? Math.min(0.06, input.seedVariance * 20) : 0;
  return clamp01(base - gapPenalty - divergencePenalty - seedPenalty);
}

export function experimentComparisonSignature(config: Record<string, unknown>) {
  const normalized = normalizeComparisonConfig(config) as Record<string, unknown>;
  return Object.keys(normalized).length > 0 ? stableStringify(normalized) : "";
}

export function normalizeComparisonConfig(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => normalizeComparisonConfig(item));
  if (!value || typeof value !== "object") return value;
  const ignoredKeys = new Set(["seed", "random_seed", "experiment_index", "plan_id", "dataset_id"]);
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !ignoredKeys.has(key))
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, normalizeComparisonConfig(child)]),
  );
}

export function stableStringify(value: unknown): string {
  return JSON.stringify(normalizeComparisonConfig(value));
}

export function experimentSeed(config: Record<string, unknown>) {
  const seed = config.seed ?? config.random_seed;
  if (typeof seed === "number" && Number.isFinite(seed)) return String(seed);
  if (typeof seed === "string" && seed.trim()) return seed.trim();
  return "";
}

export function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export function buildMemoryRetrievalProbeSnapshots(events: ExecutionEvent[]): MemoryRetrievalProbeSnapshot[] {
  return events
    .filter((event) => event.event_type === "MEMORY_RETRIEVAL_LOGGED")
    .map((event) => {
      const payload = recordObject(event.payload);
      const cards = uniqueBy(memoryCardsFromUnknown(payload.retrieved_cards), memoryDisplayKey).slice(0, 12);
      const retrievedCount = recordNumber(payload, "retrieved_count");
      return {
        id: event.id,
        purpose: recordString(payload, "purpose") || "memory_retrieval",
        logOnly: recordBoolean(payload, "log_only"),
        crossProjectOK: recordBoolean(payload, "cross_project_ok"),
        retrievedCount: retrievedCount || cards.length,
        createdAt: event.created_at,
        cards,
      };
    })
    .sort((left, right) => Date.parse(right.createdAt || "") - Date.parse(left.createdAt || ""));
}

export function humanizeMemoryPurpose(value: string) {
  const normalized = value.trim().toLowerCase();
  if (normalized === "planner" || normalized === "experiment_planner") return "Planner retrieval";
  if (normalized === "training_monitor") return "Training monitor retrieval";
  if (!normalized) return "Memory retrieval";
  return `${normalized.replace(/[_-]+/g, " ")} retrieval`;
}

export function memoryPurposeSuffix(value: string) {
  const label = humanizeMemoryPurpose(value).replace(/\s+retrieval$/i, "").toLowerCase();
  return label && label !== "memory" ? ` for ${label}` : "";
}

export function candidateScoreRows(decision: AgentDecision): CandidateScoreRow[] {
  const rankings = Array.isArray(decision.payload?.candidate_rankings) ? decision.payload.candidate_rankings : [];
  return rankings
    .map((item, index) => {
      const record = recordObject(item);
      const components = recordObject(record.score_components);
      const memoryHits = candidateRetrievedMemoryRows(record, components);
      const memoryReasons = candidateMemoryReasons(record, components, memoryHits);
      return {
        label:
          recordString(record, "hypothesis") ||
          recordString(record, "experiment_signature") ||
          recordString(record, "model") ||
          `Candidate ${index + 1}`,
        status: record.selected === true ? "SELECTED" : record.rejected === true ? "REJECTED" : "CANDIDATE",
        mechanism: recordFirstString(record, ["mechanism", "selected_mechanism", "proposal_mechanism"]),
        intervention: recordString(record, "intervention"),
        expectedEffect: recordFirstString(record, ["expected_effect", "expected_metric_effect"]),
        validationStatus: recordFirstString(record, ["backend_validation_status", "validation_status"]),
        totalScore: numberPayload(record.score) ?? numberPayload(record.total_score),
        reasons: stringArrayPayload(record.reasons),
        memoryReasons,
        memoryHits,
        components: candidateScoreComponentRows(components),
      };
    })
    .filter(
      (row) =>
        row.totalScore !== null ||
        row.components.length > 0 ||
        row.reasons.length > 0 ||
        row.memoryReasons.length > 0 ||
        row.memoryHits.length > 0 ||
        Boolean(row.mechanism || row.intervention || row.expectedEffect || row.validationStatus),
    )
    .slice(0, 6);
}

export function decisionRetrievedMemoryRows(decision: AgentDecision): RetrievedMemoryDisplay[] {
  const payload = decision.payload ?? {};
  const snapshot = recordObject(payload.planner_context_snapshot ?? payload.plannerContextSnapshot);
  const monitorSnapshot = recordObject(payload.training_monitor_context_snapshot ?? payload.trainingMonitorContextSnapshot);
  const rows = [
    ...memoryCardsFromUnknown(payload.retrieved_memory),
    ...memoryCardsFromUnknown(payload.retrievedMemory),
    ...memoryCardsFromUnknown(payload.retrieved_run_memory),
    ...memoryCardsFromUnknown(payload.retrievedRunMemory),
    ...memoryCardsFromUnknown(snapshot.retrieved_memory ?? snapshot.retrievedMemory),
    ...memoryCardsFromUnknown(monitorSnapshot.retrieved_run_memory ?? monitorSnapshot.retrievedRunMemory),
  ];
  return uniqueBy(rows, memoryDisplayKey).slice(0, 8);
}

export function candidateRetrievedMemoryRows(
  record: Record<string, unknown>,
  components: Record<string, unknown>,
): RetrievedMemoryDisplay[] {
  const rows = [
    ...memoryCardsFromUnknown(record.retrieved_memory),
    ...memoryCardsFromUnknown(record.retrievedMemory),
    ...memoryCardsFromUnknown(record.retrieved_memory_hits),
    ...memoryCardsFromUnknown(record.retrievedMemoryHits),
    ...memoryCardsFromUnknown(record.memory_hits),
    ...memoryCardsFromUnknown(record.memoryHits),
    ...memoryCardsFromUnknown(record.memory_retrieval_hits),
    ...memoryCardsFromUnknown(record.memory_cards),
    ...memoryCardsFromUnknown(record.retrieved_cards),
    ...memoryCardsFromUnknown(components.retrieved_memory),
    ...memoryCardsFromUnknown(components.retrieved_memory_hits),
    ...memoryCardsFromUnknown(components.memory_hits),
  ];
  return uniqueBy(rows, memoryDisplayKey).slice(0, 4);
}

export function candidateMemoryReasons(
  record: Record<string, unknown>,
  components: Record<string, unknown>,
  memoryHits: RetrievedMemoryDisplay[],
) {
  const explicitReasons = [
    ...memoryReasonStrings(record.retrieved_memory_reasons),
    ...memoryReasonStrings(record.retrievedMemoryReasons),
    ...memoryReasonStrings(record.memory_reasons),
    ...memoryReasonStrings(record.memoryReasons),
    ...memoryReasonStrings(record.retrieval_reasons),
    ...memoryReasonStrings(record.retrievalReasons),
  ];
  const rankingReasons = stringArrayPayload(record.reasons).filter(isMemoryReasonText).map((reason) => safeMemoryText(reason));
  const hitReasons = memoryHits.map((memory) => memory.retrievalReason).filter(Boolean);
  return uniqueStrings([
    ...explicitReasons,
    ...rankingReasons,
    ...memoryScoreComponentSummaries(components),
    ...hitReasons,
  ]).slice(0, 6);
}

export function candidateScoreComponentRows(components: Record<string, unknown>): Array<{ label: string; value: number | string }> {
  const rows: Array<{ label: string; value: number | string }> = [];
  for (const [label, value] of Object.entries(components)) {
    if (typeof value === "number" && Number.isFinite(value)) {
      if (isMemoryComponentLabel(label) && value === 0) continue;
      rows.push({ label, value });
      continue;
    }
    if (typeof value === "string" && value.trim()) {
      const text = safeMemoryText(value, 80);
      if (text) rows.push({ label, value: text });
    }
  }
  return rows
    .sort((left, right) => Number(isMemoryComponentLabel(right.label)) - Number(isMemoryComponentLabel(left.label)))
    .slice(0, 6);
}

export function memoryCardsFromUnknown(value: unknown): RetrievedMemoryDisplay[] {
  return memoryCardsFromUnknownDepth(value, 0);
}

export function memoryCardsFromUnknownDepth(value: unknown, depth: number): RetrievedMemoryDisplay[] {
  if (depth > 3 || value === undefined || value === null) return [];
  if (Array.isArray(value)) {
    return value.flatMap((item) => memoryCardsFromUnknownDepth(item, depth + 1));
  }

  const record = recordObject(value);
  if (Object.keys(record).length === 0) return [];
  if (isMemoryCardLike(record)) {
    const row = memoryDisplayFromRecord(record as RetrievedMemoryCard);
    return row ? [row] : [];
  }

  return Object.entries(record)
    .filter(([key, entry]) => isMemoryContainerKey(key, entry))
    .flatMap(([, entry]) => memoryCardsFromUnknownDepth(entry, depth + 1));
}

export function memoryDisplayFromRecord(record: RetrievedMemoryCard): RetrievedMemoryDisplay | null {
  const summaryCard = recordObject(record.summary_card ?? record.summaryCard);
  const metadata = recordObject(record.metadata);
  const records = [record as Record<string, unknown>, summaryCard, metadata];
  const sourceTable = firstMemoryString(records, ["source_table", "sourceTable", "source", "source_kind"], 80);
  const source = humanizeMemorySource(sourceTable);
  const sourceId = firstMemoryString(records, ["source_id", "sourceID", "memory_id", "scorecard_id", "id"], 120);
  const kind = firstMemoryString(records, ["kind", "memory_kind", "card_kind", "type"], 80);
  const outcome = memoryOutcome(records, kind);
  const mechanism = firstMemoryString(records, ["mechanism", "strategy_type", "strategy", "selected_mechanism"], 120);
  const intervention = firstMemoryString(records, ["intervention", "action", "proposed_change"], 180);
  const summary =
    firstMemoryString(records, ["lesson", "compact_lesson", "summary", "compact_summary"], 260) ||
    firstMemoryString(records, ["recommendation_summary", "dynamics", "training_dynamics", "preprocessing_hypothesis"], 220);
  const retrievalReason =
    firstMemoryString(records, ["retrieval_reason", "reason_for_retrieval", "match_reason", "memory_reason"], 220) ||
    firstMemoryString([record as Record<string, unknown>], ["reason"], 180);
  const score =
    numberPayload(record.score) ??
    numberPayload(record.retrieval_score) ??
    numberPayload(record.semantic_score) ??
    numberPayload(record.structured_score);
  const identifiers = memoryIdentifierRows(records);

  if (!sourceId && !kind && !mechanism && !summary && !retrievalReason) return null;

  return {
    source,
    sourceId,
    kind,
    outcome,
    mechanism,
    intervention,
    summary,
    retrievalReason,
    score,
    identifiers,
  };
}

export function isMemoryCardLike(record: Record<string, unknown>) {
  return [
    "source_id",
    "sourceID",
    "source_table",
    "summary_card",
    "summaryCard",
    "retrieval_reason",
    "reason_for_retrieval",
    "kind",
    "memory_kind",
    "lesson",
    "compact_lesson",
    "compact_summary",
    "outcome",
    "mechanism",
    "intervention",
  ].some((key) => key in record);
}

export function isMemoryContainerKey(key: string, entry: unknown) {
  if (!entry || key.toLowerCase().includes("embedding") || key.toLowerCase().includes("payload")) return false;
  if (Array.isArray(entry)) return true;
  const normalized = key.toLowerCase();
  return (
    normalized.includes("retrieved_memory") ||
    normalized.includes("retrievedrunmemory") ||
    normalized.includes("retrieved_run_memory") ||
    normalized.includes("memory_hits") ||
    normalized.includes("memory_cards") ||
    normalized.includes("retrieved_cards") ||
    normalized.includes("successful_strategy_cards") ||
    normalized.includes("failed_strategy_cards") ||
    normalized.includes("blocked_or_rejected_cards") ||
    normalized.includes("dataset_preprocessing_cards") ||
    normalized.includes("run_dynamics_cards")
  );
}

export function firstMemoryString(records: Record<string, unknown>[], keys: string[], maxLength: number) {
  for (const record of records) {
    for (const key of keys) {
      const value = record[key];
      if (typeof value === "string") {
        const text = safeMemoryText(value, maxLength);
        if (text) return text;
      }
      if ((typeof value === "number" && Number.isFinite(value)) || typeof value === "boolean") {
        return String(value);
      }
    }
  }
  return "";
}

export function memoryIdentifierRows(records: Record<string, unknown>[]) {
  const labels: Array<{ key: string; label: string }> = [
    { key: "source_id", label: "Source ID" },
    { key: "memory_id", label: "Memory ID" },
    { key: "scorecard_id", label: "Scorecard" },
    { key: "decision_id", label: "Decision" },
    { key: "source_decision_id", label: "Decision" },
    { key: "plan_id", label: "Plan" },
    { key: "source_plan_id", label: "Source Plan" },
    { key: "followup_plan_id", label: "Follow-up Plan" },
    { key: "follow_up_plan_id", label: "Follow-up Plan" },
    { key: "job_id", label: "Job" },
    { key: "invocation_id", label: "Invocation" },
    { key: "dataset_id", label: "Dataset" },
  ];
  const rows: Array<{ label: string; value: string }> = [];
  for (const item of labels) {
    const value = firstMemoryString(records, [item.key], 120);
    if (value) rows.push({ label: item.label, value });
  }
  return uniqueBy(rows, (row) => `${row.label}-${row.value}`).slice(0, 6);
}

export function memoryOutcome(records: Record<string, unknown>[], kind: string) {
  const explicit = firstMemoryString(records, ["outcome", "outcome_status", "result"], 80);
  if (explicit) return explicit;
  const text = kind.toLowerCase();
  if (text.includes("rejected") || text.includes("blocked")) return "rejected";
  if (text.includes("failed") || text.includes("failure")) return "failed";
  if (text.includes("no_improvement") || text.includes("no improvement")) return "no_improvement";
  if (text.includes("minor")) return "minor_improvement";
  if (text.includes("success") || text.includes("improved")) return "improved";
  return kind ? "memory" : "retrieved";
}

export function humanizeMemorySource(value: string) {
  const normalized = value.toLowerCase();
  if (!normalized) return "Memory";
  const labels: Record<string, string> = {
    agent_memory_records: "Agent Memory",
    strategy_scorecards: "Strategy Scorecard",
    dataset_profiles: "Dataset Profile",
    dataset_visual_analyses: "Visual Analysis",
    dataset_preprocessing: "Preprocessing Memory",
    training_run_evaluations: "Run Evaluation",
    training_run_summaries: "Run Summary",
  };
  return labels[normalized] || value.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function memoryReasonStrings(value: unknown): string[] {
  if (typeof value === "string") {
    const text = safeMemoryText(value);
    return text ? [text] : [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => memoryReasonStrings(item));
  }
  const record = recordObject(value);
  if (Object.keys(record).length === 0) return [];
  return [
    firstMemoryString([record], ["retrieval_reason", "reason_for_retrieval", "match_reason", "memory_reason", "reason"], 220),
    firstMemoryString([record], ["summary", "lesson"], 220),
  ].filter(Boolean);
}

export function memoryScoreComponentSummaries(components: Record<string, unknown>) {
  return Object.entries(components)
    .filter(([label]) => isMemoryComponentLabel(label))
    .flatMap(([label, value]) => {
      if (typeof value === "number" && Number.isFinite(value) && value !== 0) {
        const sign = value > 0 ? "+" : "";
        return [`${humanizeAuditKey(label)} ${sign}${value.toFixed(3)}`];
      }
      if (typeof value === "string") {
        const text = safeMemoryText(value);
        return text ? [`${humanizeAuditKey(label)}: ${text}`] : [];
      }
      return memoryReasonStrings(value);
    });
}

export function isMemoryComponentLabel(label: string) {
  const normalized = label.toLowerCase();
  return normalized.includes("retrieved_memory") || normalized.includes("memory_similarity") || normalized.includes("memory_score");
}

export function isMemoryReasonText(value: string) {
  const normalized = value.toLowerCase();
  return normalized.includes("retrieved") || normalized.includes("memory") || normalized.includes("scorecard");
}

export function memoryDisplayKey(memory: RetrievedMemoryDisplay) {
  return `${memory.source}-${memory.sourceId}-${memory.kind}-${memory.summary}`;
}

export function safeMemoryText(value: string, maxLength = 220) {
  const text = shortAuditText(value, maxLength);
  const trimmed = text.trim();
  if (!trimmed) return "";
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) return "";
  return trimmed;
}

export function buildChampionExportDemo(detail: ProjectDetail): ChampionExportDemo {
  const champion = detail.champion;
  if (!champion) {
    return {
      hasChampion: false,
      exportStatus: "PENDING",
      exports: [],
      projectId: "",
      modelCard: {},
      deploymentProfile: {},
      modelProfile: {},
      useCases: ["Select a champion first"],
      limitations: [],
      preprocessing: [],
      demoImages: [],
      demoPredictions: [],
      feedback: [],
    };
  }

  const deployment = champion.deployment_profile ?? {};
  const evaluation = champion.evaluation ?? {};
  const modelCard = {
    ...recordObject(deployment.model_card),
    ...recordObject(evaluation.model_card),
  };
  const modelProfile = {
    ...recordObject(deployment.model_profile),
    ...recordObject(evaluation.model_profile),
  };
  const deploymentArtifactCandidates = [
    recordString(deployment, "onnx_artifact_uri"),
    recordString(modelProfile, "onnx_artifact_uri"),
    recordString(deployment, "artifact_uri"),
    recordString(modelProfile, "artifact_uri"),
  ];
  const deploymentONNXArtifact = firstChampionArtifactMatchingFormat(deploymentArtifactCandidates, "onnx");
  const deploymentSourceArtifact = deploymentArtifactCandidates.find(Boolean) || "";
  const deploymentSourceFormat = championExportFormatFromArtifact(deploymentSourceArtifact);
  const deploymentExportManifest = recordObject(deployment.export_manifest);
  const modelExportManifest = recordObject(modelProfile.export_manifest);
  const sourceExportStatus =
    recordString(modelCard, "export_status") ||
    recordString(deployment, "export_status") ||
    recordString(modelProfile, "export_status") ||
    "PENDING_ARTIFACT";
  const sourceValidationErrors = championExportValidationErrorsForFormat(
    stringArrayPayload(modelProfile.export_validation_errors),
    deploymentSourceFormat,
  );
  const exports = [
    ...(deploymentONNXArtifact
      ? [
          {
            id: `${champion.id}-training-onnx`,
            project_id: champion.project_id,
            champion_id: champion.id,
            job_id: champion.job_id,
            status: "READY",
            format: "onnx",
            artifact_uri: deploymentONNXArtifact,
            metadata: {
              manifest: Object.keys(deploymentExportManifest).length > 0 ? deploymentExportManifest : modelExportManifest,
              deployment_profile: deployment,
              model_profile: modelProfile,
            },
          } satisfies ChampionExport,
        ]
      : []),
    ...(deploymentSourceArtifact && deploymentSourceArtifact !== deploymentONNXArtifact
      ? [
          {
            id: `${champion.id}-training-source`,
            project_id: champion.project_id,
            champion_id: champion.id,
            job_id: champion.job_id,
            status: deploymentSourceFormat ? "READY" : sourceExportStatus,
            format: deploymentSourceFormat || "model artifact",
            artifact_uri: deploymentSourceArtifact,
            validation_errors: sourceValidationErrors,
            metadata: {
              manifest: Object.keys(deploymentExportManifest).length > 0 ? deploymentExportManifest : modelExportManifest,
              deployment_profile: deployment,
              model_profile: modelProfile,
            },
          } satisfies ChampionExport,
        ]
      : []),
    ...detail.championExports,
    ...championExportsFromUnknown(champion.champion_exports),
    ...championExportsFromUnknown(deployment.champion_exports),
    ...championExportsFromUnknown(deployment.exports),
    ...championExportsFromUnknown(modelCard.champion_exports),
  ];
  const demoImages = [
    ...detail.championDemoImages,
    ...demoImagesFromUnknown(champion.demo_images),
    ...demoImagesFromUnknown(deployment.demo_images),
    ...demoImagesFromUnknown(deployment.heldout_images),
    ...demoImagesFromUnknown(deployment.test_images),
    ...demoImagesFromUnknown(recordObject(evaluation.objective_profile).heldout_demo_images),
    ...demoImagesFromUnknown(recordObject(evaluation.objective_profile).demo_images),
    ...demoImagesFromUnknown(recordObject(evaluation.objective_profile).test_images),
  ];
  const demoPredictions = [
    ...detail.championDemoPredictions,
    ...demoPredictionsFromUnknown(champion.demo_predictions),
    ...demoPredictionsFromUnknown(deployment.demo_predictions),
    ...demoPredictionsFromUnknown(deployment.predictions),
  ];
  const metadataExportStatus =
    recordString(modelCard, "export_status") ||
    recordString(deployment, "export_status") ||
    "PENDING";
  const uniqueExports = uniqueBy(exports, (item, index) => item.id || item.artifact_uri || item.model_uri || item.download_url || String(index));
  const exportStatus = championExportOverallStatus(uniqueExports, metadataExportStatus);
  const latency = recordNumber(modelProfile, "estimated_latency_ms");
  const modelSize = recordNumber(modelProfile, "model_size_mb") || recordNumber(modelProfile, "estimated_model_size_mb");
  const objectiveContext = recordString(deployment, "objective_context");
  const targetMetric = recordString(deployment, "target_metric");
  const useCases = uniqueStrings([
    ...stringArrayPayload(modelCard.intended_uses),
    recordString(modelCard, "intended_use"),
    objectiveContext ? `${objectiveContext} objective` : "",
    targetMetric ? `optimized for ${targetMetric}` : "",
    latency && latency <= 30 ? "live low-latency inference candidate" : "",
    latency && latency > 30 ? "offline or batch classification candidate" : "",
    modelSize && modelSize <= 50 ? "compact deployment footprint" : "",
    "prototype validation with backend-verified champion metrics",
  ]).slice(0, 6);
  const limitations = uniqueStrings([
    ...stringArrayPayload(modelCard.known_limitations),
    recordString(modelCard, "limitation"),
    exportStatus.toLowerCase() === "pending" ? "Final export artifact is still pending." : "",
    demoPredictions.length === 0 ? "Live demo inference is waiting on backend prediction records or APIs." : "",
  ]).slice(0, 5);
  const preprocessing = uniqueStrings([
    ...stringArrayPayload(modelCard.recommended_preprocessing),
    ...stringArrayPayload(deployment.recommended_preprocessing),
    recordString(modelProfile, "normalization") ? `normalization: ${recordString(modelProfile, "normalization")}` : "",
    recordString(modelProfile, "input_size") ? `input: ${recordString(modelProfile, "input_size")}` : "",
  ]).slice(0, 6);

  const portableBundle = firstPortableInferenceBundle(uniqueExports);

  return {
    hasChampion: true,
    exportStatus,
    exports: uniqueExports,
    portableBundle,
    projectId: champion.project_id,
    modelCard,
    deploymentProfile: deployment,
    modelProfile,
    useCases,
    limitations,
    preprocessing: preprocessing.length > 0 ? preprocessing : ["Use the preprocessing from the winning experiment config."],
    demoImages,
    demoPredictions,
    feedback: detail.championFeedback,
  };
}

function championExportOverallStatus(exports: ChampionExport[], metadataStatus: string) {
  const exportStatus = preferredChampionExportStatus(exports.map((exportRecord) => exportRecord.status || ""));
  if (exportStatus) return exportStatus;
  return preferredChampionExportStatus([metadataStatus]) || "PENDING";
}

function preferredChampionExportStatus(statuses: string[]) {
  const normalized = statuses.map((status) => normalizedStatus(status || "")).filter(Boolean);
  if (normalized.some((status) => status === "READY" || status === "SUCCEEDED")) return "READY";
  if (normalized.includes("RUNNING")) return "RUNNING";
  if (normalized.some((status) => status === "REQUESTED" || status === "QUEUED")) return "QUEUED";
  if (normalized.includes("PENDING")) return "PENDING";
  if (normalized.includes("PENDING_ARTIFACT")) return "PENDING_ARTIFACT";
  if (normalized.includes("FAILED")) return "FAILED";
  return "";
}

export function championFeedbackMetricsSnapshot(detail: ProjectDetail) {
  const champion = detail.champion;
  if (!champion) return {};
  const runSummary = detail.runSummaries.find((summary) => summary.job_id === champion.job_id) ?? null;
  const runEvaluation = detail.runEvaluations.find((evaluation) => evaluation.job_id === champion.job_id) ?? null;
  return {
    champion_id: champion.id,
    champion_job_id: champion.job_id,
    champion_metrics: champion.metrics,
    champion_evaluation: champion.evaluation,
    deployment_profile: champion.deployment_profile,
    run_summary: runSummary,
    run_evaluation: runEvaluation,
  };
}

export function feedbackRatingLabel(rating: string) {
  switch (rating) {
    case "good":
      return "Good";
    case "bad":
      return "Bad";
    default:
      return "Mediocre";
  }
}

export function normalizeChampionFeedbackResponse(response: { feedback?: ChampionFeedback } | ChampionFeedback): ChampionFeedback | null {
  const wrapped = recordObject(response).feedback;
  if (isChampionFeedback(wrapped)) return wrapped;
  if (isChampionFeedback(response)) return response;
  return null;
}

export function isChampionFeedback(value: unknown): value is ChampionFeedback {
  const record = recordObject(value);
  return Boolean(record.id && record.champion_id && record.rating);
}

export function experimentPreprocessingItems(experiment: PlannedExperiment) {
  const preprocessing = experiment.preprocessing ?? {};
  return [
    { label: "resolution", value: experiment.resolution_strategy },
    { label: "resize", value: preprocessing.resize_strategy },
    { label: "normalization", value: preprocessing.normalization },
    { label: "crop", value: preprocessing.crop_strategy },
    { label: "bbox", value: preprocessing.bbox_mode },
    { label: "dataset norm", value: preprocessing.use_dataset_normalization ? "enabled" : "" },
    { label: "augmentation", value: experiment.augmentation_policy },
    { label: "aug policy", value: augmentationPolicyConfigSummary(experiment.augmentation_policy_config) },
    { label: "sampling", value: experiment.sampling_strategy },
    { label: "balancing", value: experiment.class_balancing },
    { label: "balance config", value: classBalancingConfigSummary(experiment.class_balancing_config) },
  ].filter((item): item is { label: string; value: string } => typeof item.value === "string" && item.value.length > 0);
}

export function classBalancingConfigSummary(config: PlannedExperiment["class_balancing_config"]) {
  const record = recordObject(config);
  const beta = recordNumber(record, "effective_number_beta");
  return typeof beta === "number" ? `beta ${formatMetricNumber(beta)}` : "";
}

export function augmentationPolicyConfigSummary(config: PlannedExperiment["augmentation_policy_config"]) {
  if (!config?.policy_type) return "";
  const details = [
    config.magnitude ? `mag ${config.magnitude}` : "",
    config.num_ops ? `${config.num_ops} ops` : "",
    config.num_magnitude_bins ? `${config.num_magnitude_bins} bins` : "",
    typeof config.probability === "number" ? `p ${formatMetricNumber(config.probability)}` : "",
    config.alpha ? `alpha ${formatMetricNumber(config.alpha)}` : "",
  ].filter(Boolean);
  return [config.policy_type, ...details].join(" / ");
}

export function experimentAutoMLItems(experiment: PlannedExperiment) {
  const config = experiment.automl;
  if (!config?.enabled) return [];
  const finalValues = recordObject(config.final_values ?? config.suggestion?.final_values);
  const provenance = recordObject(config.value_provenance ?? config.suggestion?.provenance);
  const tuned = Object.entries(finalValues)
    .slice(0, 8)
    .map(([key, value]) => {
      const source = recordString(provenance, key);
      return `${key}=${formatUnknownValue(value)}${source ? ` (${source})` : ""}`;
    })
    .join(" / ");
  const searchNames = (config.search_space?.parameters ?? [])
    .map((parameter) => parameter.name)
    .filter(Boolean)
    .slice(0, 8)
    .join(", ");
  return [
    { label: "AutoML", value: config.validation_status || "enabled" },
    { label: "sampler", value: config.sampler },
    { label: "search", value: searchNames },
    { label: "suggestion", value: tuned },
  ].filter((item): item is { label: string; value: string } => typeof item.value === "string" && item.value.length > 0);
}

export function experimentMechanismItems(experiment: PlannedExperiment) {
  const evidence = stringArrayPayload(experiment.evidence_used).slice(0, 2).join("; ");
  return [
    { label: "mechanism", value: experiment.mechanism },
    { label: "intervention", value: experiment.intervention },
    { label: "expected effect", value: experiment.expected_effect },
    { label: "evidence", value: evidence },
    { label: "validation", value: experiment.backend_validation_status || experiment.validation_status },
    { label: "validation error", value: experiment.backend_validation_error || experiment.validation_error },
  ].filter((item): item is { label: string; value: string } => typeof item.value === "string" && item.value.length > 0);
}

export function jobMechanismSummary(job: Job) {
  const mechanism = recordString(job.config, "mechanism");
  const intervention = recordString(job.config, "intervention");
  const validation = recordFirstString(job.config, ["backend_validation_status", "validation_status"]);
  const automlSummary = recordObject(job.config.automl_summary);
  const automl = recordString(automlSummary, "suggestion_id") ? `AutoML ${recordString(automlSummary, "sampler") || "suggestion"}` : "";
  return [mechanism ? `mechanism ${mechanism}` : "", intervention, automl, validation ? `validation ${validation}` : ""]
    .filter(Boolean)
    .join(" / ");
}

export function backendGateSummary(decision: AgentDecision) {
  const validationStatus = recordFirstString(decision.payload ?? {}, [
    "backend_validation_status",
    "validation_status",
    "planner_validation_status",
  ]);
  const stopGuard = recordString(decision.payload ?? {}, "backend_stop_guard");
  if (stopGuard) {
    return `Backend stop guard: ${stopGuard}.`;
  }
  if (validationStatus) {
    return `Backend validation status: ${validationStatus}.`;
  }
  const rejections = decisionRejections(decision);
  if (rejections.length > 0) {
    return `${rejections.length} rejected candidate/options visible; stored decision is ${decision.decision_type}.`;
  }
  if (decision.decision_type === "ADD_EXPERIMENTS") {
    return "Accepted for follow-up scheduling when automation settings allow it.";
  }
  return `Accepted project decision: ${decision.decision_type}.`;
}

export function decisionHistorySummary(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  const mechanism = recordFirstString(payload, ["mechanism", "selected_mechanism", "proposal_mechanism"]);
  const intervention = recordString(payload, "intervention");
  const validation = recordFirstString(payload, ["backend_validation_status", "validation_status"]);
  return [mechanism ? `mechanism ${mechanism}` : "", intervention, validation ? `validation ${validation}` : "", decision.rationale]
    .filter(Boolean)
    .join(" - ");
}

export function decisionHighlights(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  const items: Array<{ label: string; value: string }> = [];

  if (typeof payload.champion_model === "string" && payload.champion_model) {
    items.push({ label: "Champion", value: payload.champion_model });
  }

  if (typeof payload.planning_mode === "string" && payload.planning_mode) {
    items.push({ label: "Mode", value: payload.planning_mode });
  }

  const mechanism = recordFirstString(payload, ["mechanism", "selected_mechanism", "proposal_mechanism"]);
  if (mechanism) {
    items.push({ label: "Mechanism", value: mechanism });
  }

  const intervention = recordString(payload, "intervention");
  if (intervention) {
    items.push({ label: "Intervention", value: intervention });
  }

  const expectedEffect = recordFirstString(payload, ["expected_effect", "expected_metric_effect"]);
  if (expectedEffect) {
    items.push({ label: "Expected Effect", value: expectedEffect });
  }

  const validationStatus = recordFirstString(payload, [
    "backend_validation_status",
    "validation_status",
    "planner_validation_status",
  ]);
  if (validationStatus) {
    items.push({ label: "Validation", value: validationStatus });
  }

  const championScore = numberPayload(payload.champion_score);
  if (championScore !== null) {
    items.push({ label: "Score", value: championScore.toFixed(3) });
  }

  const expectedDelta = numberPayload(payload.expected_delta_vs_champion);
  if (expectedDelta !== null) {
    items.push({ label: "Expected Delta", value: expectedDelta.toFixed(3) });
  }

  const diagnosis = recordObject(payload.deterministic_diagnosis);
  const failureModes = stringArrayPayload(diagnosis.recommended_failure_modes);
  if (failureModes.length > 0) {
    items.push({ label: "Diagnosis", value: failureModes.slice(0, 3).join(", ") });
  }

  const rankings = Array.isArray(payload.candidate_rankings) ? payload.candidate_rankings : [];
  if (rankings.length > 0) {
    const selected = rankings.filter((item) => recordObject(item).selected === true).length;
    const rejected = rankings.filter((item) => recordObject(item).rejected === true).length;
    items.push({ label: "Candidates", value: `${selected} selected, ${rejected} rejected` });
  }

  const rejectedOptions = Array.isArray(payload.rejected_options) ? payload.rejected_options.length : 0;
  if (rejectedOptions > 0) {
    items.push({ label: "Rejected Options", value: String(rejectedOptions) });
  }

  const runtimeSeconds = numberPayload(
    payload.champion_runtime_seconds !== undefined
      ? payload.champion_runtime_seconds
      : payload.total_runtime_seconds,
  );
  if (runtimeSeconds !== null) {
    items.push({ label: "Runtime", value: formatSeconds(runtimeSeconds) });
  }

  const reportedRuns = numberPayload(payload.reported_experiments);
  if (reportedRuns !== null) {
    items.push({ label: "Runs", value: String(reportedRuns) });
  }

  const estimatedCost = numberPayload(
    payload.champion_estimated_cost_usd !== undefined
      ? payload.champion_estimated_cost_usd
      : payload.total_estimated_cost,
  );
  if (estimatedCost !== null) {
    items.push({ label: "Budget Used", value: formatCurrency(estimatedCost) });
  }

  if (items.length === 0) {
    items.push({ label: "Decision", value: decision.decision_type });
  }

  return items;
}

export function mechanismDecisionSummaries(payload: Record<string, unknown>) {
  const direct = mechanismSummaryFromRecord(payload);
  const selectedCandidate = (Array.isArray(payload.candidate_rankings) ? payload.candidate_rankings : [])
    .map(recordObject)
    .find((record) => record.selected === true);
  const proposed = Array.isArray(payload.proposed_experiments) ? payload.proposed_experiments.map(recordObject) : [];
  const proposalMechanisms = Array.isArray(payload.proposal_mechanisms) ? payload.proposal_mechanisms.map(recordObject) : [];
  return uniqueStrings([
    direct,
    selectedCandidate ? mechanismSummaryFromRecord(selectedCandidate) : "",
    ...proposalMechanisms.map(mechanismSummaryFromRecord),
    ...proposed.map(mechanismSummaryFromRecord),
  ]);
}

export function mechanismSummaryFromRecord(record: Record<string, unknown>) {
  const mechanism = recordFirstString(record, ["mechanism", "selected_mechanism", "proposal_mechanism"]);
  const intervention = recordString(record, "intervention");
  const expectedEffect = recordFirstString(record, ["expected_effect", "expected_metric_effect"]);
  const evidence = stringArrayPayload(record.evidence_used).slice(0, 2).join("; ");
  const validation = recordFirstString(record, ["backend_validation_status", "validation_status"]);
  return [
    mechanism ? `mechanism ${mechanism}` : "",
    intervention ? `intervention ${intervention}` : "",
    expectedEffect ? `expected ${expectedEffect}` : "",
    evidence ? `evidence ${evidence}` : "",
    validation ? `validation ${validation}` : "",
  ]
    .filter(Boolean)
    .join(" - ");
}

export function mechanismCoverageRows(payload: Record<string, unknown>): MechanismCoverageRow[] {
  const coverage = recordObject(payload.mechanism_coverage ?? payload.mechanism_coverage_card);
  const rows: MechanismCoverageRow[] = [];
  addCoverageRows(rows, "TRIED", coverage.tried ?? coverage.tried_mechanisms ?? coverage.attempted ?? coverage.attempted_mechanisms);
  addCoverageRows(rows, "ELIGIBLE", coverage.eligible ?? coverage.eligible_mechanisms);
  addCoverageRows(rows, "BLOCKED", coverage.blocked ?? coverage.blocked_mechanisms);
  addCoverageRows(rows, "NO_IMPROVEMENT", coverage.no_improvement ?? coverage.no_improvement_mechanisms);
  addCoverageRows(rows, "BEST_RESULT", coverage.best_result ?? coverage.best_results ?? coverage.best_result_by_mechanism);
  return uniqueBy(rows, (row) => `${row.status}-${row.mechanism}-${row.detail}`).slice(0, 10);
}

export function addCoverageRows(rows: MechanismCoverageRow[], status: string, value: unknown) {
  if (Array.isArray(value)) {
    for (const item of value) {
      if (typeof item === "string") {
        rows.push({ mechanism: item, status, detail: status.toLowerCase() });
        continue;
      }
      const record = recordObject(item);
      const mechanism = recordFirstString(record, ["mechanism", "name", "key", "id"]);
      if (!mechanism) continue;
      rows.push({ mechanism, status: recordString(record, "status") || status, detail: coverageDetail(record, status) });
    }
    return;
  }

  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [mechanism, detailValue] of Object.entries(value as Record<string, unknown>)) {
      const detailRecord = recordObject(detailValue);
      const detail =
        Object.keys(detailRecord).length > 0
          ? coverageDetail(detailRecord, status)
          : typeof detailValue === "string"
            ? detailValue
            : typeof detailValue === "number"
              ? String(detailValue)
              : status.toLowerCase();
      rows.push({ mechanism, status, detail });
    }
  }
}

export function coverageDetail(record: Record<string, unknown>, fallback: string) {
  const detail = recordFirstString(record, ["detail", "reason", "outcome", "lesson", "summary"]);
  const score = numberPayload(record.best_score ?? record.score ?? record.metric);
  const validation = recordFirstString(record, ["backend_validation_status", "validation_status"]);
  return [
    detail,
    score !== null ? `score ${score.toFixed(3)}` : "",
    validation ? `validation ${validation}` : "",
  ]
    .filter(Boolean)
    .join("; ") || fallback.toLowerCase();
}

export function automationReviewState(settings: AutomationSettings) {
  if (!settings.auto_execute_plans) {
    return {
      badge: "REVIEW",
      title: "Dry-run review mode",
      detail: [
        settings.auto_review_experiments ? "auto review on" : "manual review",
        settings.auto_schedule_followups ? "follow-up plans may be created" : "manual follow-up scheduling",
        "auto execution off",
      ].join(" - "),
      tone: "review",
    };
  }

  if (settings.agent_mode !== "autonomous") {
    return {
      badge: "PROPOSE",
      title: "Proposal mode",
      detail: "Agents can propose decisions, but autonomous scheduling remains governed by the visible toggles.",
      tone: "wait",
    };
  }

  return {
    badge: "AUTO",
    title: "Autonomous execution visible",
    detail: "Auto execution is enabled; review backend gates and worker events before spending on new plans.",
    tone: "auto",
  };
}

export function numberPayload(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function recordString(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

export function recordFirstString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = recordString(record, key);
    if (value) return value;
  }
  return "";
}

export function recordNumber(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

export function recordObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

export function firstAuditValue(records: Record<string, unknown>[], keys: string[]) {
  for (const record of records) {
    for (const key of keys) {
      const value = record[key];
      if (value !== undefined && value !== null && value !== "") return value;
    }
  }
  return undefined;
}

export function firstAuditString(records: Record<string, unknown>[], keys: string[]) {
  const value = firstAuditValue(records, keys);
  if (typeof value === "string") return shortAuditText(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

export function auditCountValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string" && value.trim()) return shortAuditText(value, 40);
  if (Array.isArray(value)) return String(value.length);

  const record = recordObject(value);
  const count = firstAuditValue([record], ["count", "total", "rounds", "round_count", "tool_rounds"]);
  if (typeof count === "number" && Number.isFinite(count)) return String(count);
  if (typeof count === "string" && count.trim()) return shortAuditText(count, 40);
  return "-";
}

export function auditToolNamesFromValue(value: unknown): string[] {
  if (!value) return [];
  if (typeof value === "string") {
    return value
      .split(/[,\n]/)
      .map((item) => shortAuditText(item, 64))
      .filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.flatMap(auditToolNamesFromValue);
  }

  const record = recordObject(value);
  const functionRecord = recordObject(record.function);
  const direct = firstAuditString([record, functionRecord], ["tool_name", "name", "tool", "function_name"]);
  if (direct) return [direct];
  return Object.keys(record)
    .filter((key) => !isSensitiveAuditKey(key))
    .map((key) => shortAuditText(key, 64))
    .filter(Boolean);
}

export function auditRejectedToolCallSummaries(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => auditRejectedToolCallSummaries(auditRejectionItem(item, index))).slice(0, 5);
  }

  const record = recordObject(value);
  if (Object.keys(record).length === 0) {
    return typeof value === "string" ? [shortAuditText(value)] : [];
  }

  if (hasAuditRejectionShape(record)) {
    const direct = auditRejectionItem(record, 0);
    if (typeof direct === "string") return [direct];
  }
  return Object.entries(record)
    .filter(([key]) => !isSensitiveAuditKey(key))
    .slice(0, 5)
    .map(([key, entry]) => {
      const entryRecord = recordObject(entry);
      const reason =
        firstAuditString([entryRecord], ["reason", "error", "validation_error", "message", "status"]) ||
        auditPrimitiveSummary(entry);
      return shortAuditText(`${humanizeAuditKey(key)}: ${reason || "rejected"}`);
    })
    .filter(Boolean);
}

export function hasAuditRejectionShape(record: Record<string, unknown>) {
  return [
    "tool_name",
    "name",
    "tool",
    "function_name",
    "function",
    "reason",
    "error",
    "validation_error",
    "message",
    "status",
    "rejection",
  ].some((key) => key in record);
}

export function auditRejectionItem(value: unknown, index: number): string | Record<string, unknown> {
  if (typeof value === "string") return shortAuditText(value);
  const record = recordObject(value);
  if (Object.keys(record).length === 0) return {};

  const functionRecord = recordObject(record.function);
  const name =
    firstAuditString([record, functionRecord], ["tool_name", "name", "tool", "function_name"]) ||
    `tool call ${index + 1}`;
  const reason =
    firstAuditString([record], ["reason", "error", "validation_error", "message", "status"]) ||
    firstAuditString([recordObject(record.rejection)], ["reason", "error", "message"]);
  return shortAuditText(`${name}: ${reason || "rejected"}`);
}

export function auditDryRunValidationSummaries(value: unknown): Array<{ status: string; text: string }> {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => auditDryRunValidationItem(item, index)).slice(0, 5);
  }

  const record = recordObject(value);
  if (Object.keys(record).length === 0) {
    const text = auditPrimitiveSummary(value);
    return text ? [{ status: "reported", text }] : [];
  }

  const nested = firstAuditValue([record], ["results", "items", "experiments", "candidates", "validations"]);
  if (Array.isArray(nested)) return auditDryRunValidationSummaries(nested);

  if (hasAuditValidationShape(record)) return auditDryRunValidationItem(record, 0);

  return Object.entries(record)
    .filter(([key]) => !isSensitiveAuditKey(key))
    .slice(0, 5)
    .flatMap(([key, entry], index) => auditDryRunValidationItem(entry, index, humanizeAuditKey(key)));
}

export function auditDryRunValidationItem(value: unknown, index: number, fallbackLabel?: string): Array<{ status: string; text: string }> {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    const text = auditPrimitiveSummary(value);
    return text ? [{ status: "reported", text: fallbackLabel ? `${fallbackLabel}: ${text}` : text }] : [];
  }

  const record = recordObject(value);
  if (Object.keys(record).length === 0) return [];

  if (!hasAuditValidationShape(record)) {
    return Object.entries(record)
      .filter(([key]) => !isSensitiveAuditKey(key))
      .slice(0, 3)
      .map(([key, entry]) => ({
        status: "reported",
        text: shortAuditText(`${fallbackLabel || humanizeAuditKey(key)}: ${auditPrimitiveSummary(entry) || "reported"}`),
      }));
  }

  const status =
    firstAuditString([record], ["backend_validation_status", "validation_status", "status", "result"]) ||
    (record.valid === true ? "valid" : record.valid === false ? "invalid" : "reported");
  const label =
    firstAuditString([record], ["experiment_id", "candidate_id", "model", "template", "name", "tool_name"]) ||
    fallbackLabel ||
    `item ${index + 1}`;
  const mechanism = firstAuditString([record], ["mechanism", "intervention", "decision_type"]);
  const error = firstAuditString([record], ["backend_validation_error", "validation_error", "error", "message", "reason"]);
  const text = [label, mechanism, error].filter(Boolean).join(" - ");
  return [{ status, text: shortAuditText(text || label, 180) }];
}

export function hasAuditValidationShape(record: Record<string, unknown>) {
  return [
    "backend_validation_status",
    "validation_status",
    "status",
    "result",
    "valid",
    "backend_validation_error",
    "validation_error",
    "error",
    "reason",
  ].some((key) => key in record);
}

export function auditPrimitiveSummary(value: unknown) {
  if (typeof value === "string") return shortAuditText(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} item(s)`;
  if (value && typeof value === "object") return "object";
  return "";
}

export function shortAuditText(value: string, maxLength = 160) {
  const collapsed = value.replace(/\s+/g, " ").trim();
  if (!collapsed) return "";
  if (isLikelyEncodedPayload(collapsed)) return "[redacted payload]";
  const redacted = collapsed
    .replace(/\b[a-z][a-z0-9+.-]*:\/\/\S+/gi, "[redacted uri]")
    .replace(/[A-Za-z]:\\[^\s]+/g, "[redacted path]")
    .replace(/\/(?:[^/\s]+\/){2,}[^/\s]+/g, "[redacted path]");
  return redacted.length > maxLength ? `${redacted.slice(0, maxLength - 3)}...` : redacted;
}

export function isLikelyEncodedPayload(value: string) {
  const lower = value.toLowerCase();
  return lower.startsWith("data:image") || lower.includes(";base64,") || (value.length > 180 && /^[A-Za-z0-9+/=]+$/.test(value));
}

export function isSensitiveAuditKey(key: string) {
  const normalized = key.toLowerCase();
  return (
    normalized === "input_messages" ||
    normalized === "raw_output" ||
    normalized === "parsed_output" ||
    normalized === "input_context" ||
    normalized === "content" ||
    normalized === "payload" ||
    normalized.includes("prompt") ||
    normalized.includes("image") ||
    normalized.includes("base64") ||
    normalized.includes("manifest") ||
    normalized.includes("uri") ||
    normalized.includes("url") ||
    normalized.includes("path")
  );
}

export function humanizeAuditKey(key: string) {
  return key.replace(/[_-]+/g, " ");
}

export function normalizedStatus(value: string) {
  return value.trim().toUpperCase() || "PENDING";
}

export function statusToneClass(value?: string) {
  const status = normalizedStatus(value || "PENDING").toLowerCase();
  if (["succeeded", "ready", "correct", "completed", "created"].includes(status)) return "status-good";
  if (["failed", "error", "missed", "runtime_unavailable"].includes(status)) return "status-bad";
  if (["requested", "running", "pending", "pending_artifact", "queued"].includes(status)) return "status-wait";
  return "";
}

export function exportStatusMessage(value?: string) {
  const status = normalizedStatus(value || "PENDING");
  if (status === "READY" || status === "SUCCEEDED") return "export artifact ready";
  if (status === "FAILED") return "export failed";
  if (status === "RUNNING") return "export worker running";
  if (status === "REQUESTED" || status === "QUEUED") return "export queued by backend";
  if (status === "PENDING_ARTIFACT") return "waiting for champion artifact";
  return "artifact URI pending";
}

export function predictionStatusMessage(value?: string) {
  const status = normalizedStatus(value || "PENDING");
  if (status === "SUCCEEDED") return "prediction complete";
  if (status === "FAILED") return "prediction failed";
  if (status === "RUNTIME_UNAVAILABLE") return "runtime unavailable";
  if (status === "RUNNING") return "prediction running";
  if (status === "REQUESTED" || status === "QUEUED") return "prediction queued";
  return "prediction pending";
}

export function objectSummary(record: Record<string, unknown>) {
  const entries = Object.entries(record)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, 4)
    .map(([key, value]) => `${key}: ${shortValue(value)}`);
  return entries.join("; ") || "-";
}

export function exemplarStatusFromProfile(profile: Record<string, unknown>) {
  const directStatus =
    recordString(profile, "visual_exemplar_status") ||
    recordString(profile, "exemplar_generation_status") ||
    recordString(profile, "exemplar_persistence_status");
  if (directStatus) return `exemplars ${directStatus.toLowerCase()}`;

  const audit = recordObject(profile.visual_exemplar_audit ?? profile.exemplar_audit);
  const auditStatus =
    recordString(audit, "status") ||
    recordString(audit, "generation_status") ||
    recordString(audit, "persistence_status");
  if (auditStatus) return `exemplars ${auditStatus.toLowerCase()}`;

  const exemplars = Array.isArray(profile.visual_exemplars) ? profile.visual_exemplars.length : 0;
  if (exemplars > 0) return `${exemplars} visual exemplar(s) exposed`;
  return "";
}

export function stringArrayPayload(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

export function championExportsFromUnknown(value: unknown): ChampionExport[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionExport).filter((item) => Object.keys(item).length > 0);
}

export function firstPortableInferenceBundle(exports: ChampionExport[]): PortableInferenceBundle | undefined {
  for (const exportRecord of exports) {
    const bundle = portableInferenceBundleFromExport(exportRecord);
    if (bundle) return bundle;
  }
  return undefined;
}

export function portableInferenceBundleFromExport(exportRecord: ChampionExport): PortableInferenceBundle | undefined {
  const exportObject = recordObject(exportRecord);
  const metadata = recordObject(exportRecord.metadata);
  const summary = recordObject(metadata.portable_inference_bundle ?? exportObject.portable_inference_bundle);
  const manifest = recordObject(metadata.manifest);
  const artifact = portableBundleArtifactFromManifest(manifest);
  const status =
    recordFirstString(summary, ["status"]) ||
    recordFirstString(artifact, ["status"]) ||
    (recordFirstString(metadata, ["portable_bundle_uri", "portable_inference_bundle_uri"]) ? "created" : "");
  const artifactURI =
    recordFirstString(exportObject, ["portable_bundle_uri", "portable_inference_bundle_uri"]) ||
    recordFirstString(metadata, ["portable_bundle_uri", "portable_inference_bundle_uri"]) ||
    recordFirstString(summary, ["artifact_uri", "uri"]) ||
    recordFirstString(artifact, ["uri", "artifact_uri"]);
  const artifactPath =
    recordFirstString(summary, ["artifact_path", "path"]) ||
    recordFirstString(artifact, ["artifact_path", "local_path", "path"]);
  const contents = uniqueStrings([
    ...stringArrayPayload(summary.contents),
    ...stringArrayPayload(artifact.contents),
  ]);
  const bytes = recordNumber(summary, "bytes") || recordNumber(artifact, "bytes");
  const error = recordFirstString(summary, ["error", "message"]) || recordFirstString(artifact, ["error", "message"]);
  const errorCode = recordFirstString(summary, ["error_code", "code"]) || recordFirstString(artifact, ["error_code", "code"]);
  if (!artifactURI && !artifactPath && !status && contents.length === 0 && !error) return undefined;
  return {
    schema_version: recordFirstString(summary, ["schema_version"]) || "portable_inference_bundle_v1",
    status: status || (artifactURI || artifactPath ? "created" : "unknown"),
    artifact_uri: artifactURI,
    artifact_path: artifactPath || artifactURI,
    uri: artifactURI,
    path: artifactPath,
    bytes: bytes || undefined,
    contents,
    error_code: errorCode,
    error,
  };
}

export function portableBundleArtifactFromManifest(manifest: Record<string, unknown>) {
  const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
  for (const item of artifacts) {
    const artifact = recordObject(item);
    if (recordString(artifact, "format").toLowerCase() === "portable_inference_bundle") {
      return artifact;
    }
  }
  return {};
}

export function championExportExternalData(exportRecord: ChampionExport) {
  const metadata = recordObject(exportRecord.metadata);
  const manifests = [
    recordObject(metadata.manifest),
    recordObject(recordObject(metadata.deployment_profile).export_manifest),
    recordObject(recordObject(metadata.model_profile).export_manifest),
  ].filter((item) => Object.keys(item).length > 0);
  const sidecars = manifests.flatMap((manifest) => {
    const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
    return artifacts.flatMap((item) => {
      const artifact = recordObject(item);
      if (recordString(artifact, "format").toLowerCase() !== "onnx") return [];
      const externalData = Array.isArray(artifact.external_data) ? artifact.external_data : [];
      return externalData
        .map((entry) => {
          const record = recordObject(entry);
          const sidecarPath = recordFirstString(record, ["path", "relative_path", "file_name"]);
          if (!sidecarPath) return null;
          return {
            path: sidecarPath,
            relative_path: recordString(record, "relative_path"),
            uri: recordFirstString(record, ["uri", "artifact_uri"]),
            artifact_path: recordString(record, "artifact_path"),
            local_path: recordString(record, "local_path"),
            file_name: recordString(record, "file_name"),
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
    });
  });
  return uniqueBy(sidecars, (item) => item.path);
}

export function firstChampionArtifactMatchingFormat(values: string[], format: string) {
  return values.find((value) => value && artifactMatchesChampionExportFormat(value, format)) || "";
}

export function championExportFormatFromArtifact(uri: string) {
  if (artifactMatchesChampionExportFormat(uri, "onnx")) return "onnx";
  if (artifactMatchesChampionExportFormat(uri, "torchscript")) return "torchscript";
  if (artifactMatchesChampionExportFormat(uri, "safetensors")) return "safetensors";
  if (artifactMatchesChampionExportFormat(uri, "pytorch")) return "pytorch";
  return "";
}

export function championExportValidationErrorsForFormat(errors: string[], format: string) {
  const normalized = format.toLowerCase();
  if (normalized === "onnx" || !normalized) return errors;
  return errors.filter((error) => !/onnx|onnxscript/i.test(error));
}

export function artifactMatchesChampionExportFormat(uri: string, format: string) {
  const path = artifactComparablePath(uri);
  if (format === "onnx") return path.endsWith(".onnx");
  if (format === "torchscript") return path.endsWith(".torchscript.pt") || path.endsWith(".torchscript");
  if (format === "pytorch") return path.endsWith(".pt") || path.endsWith(".pth");
  if (format === "safetensors") return path.endsWith(".safetensors");
  return false;
}

export function artifactComparablePath(uri: string) {
  const trimmed = uri.trim();
  const fallback = trimmed.split(/[?#]/)[0].toLowerCase();
  try {
    return decodeURIComponent(new URL(trimmed).pathname || fallback).toLowerCase();
  } catch {
    return fallback;
  }
}

export function demoImagesFromUnknown(value: unknown): ChampionDemoImage[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoImage).filter((item) => Object.keys(item).length > 0);
}

export function demoPredictionsFromUnknown(value: unknown): ChampionDemoPrediction[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoPrediction).filter((item) => Object.keys(item).length > 0);
}

export function championExportDemoIsDetection(data: ChampionExportDemo) {
  const records = [
    data.modelProfile,
    data.deploymentProfile,
    data.modelCard,
    ...data.exports.map((item) => recordObject(item.metadata)),
    ...data.exports.map((item) => recordObject(recordObject(item.metadata).manifest)),
  ];
  return records.some((record) => {
    const metadata = recordObject(record);
    const manifestMetadata = recordObject(recordObject(metadata.manifest).metadata);
    const candidates = [metadata, manifestMetadata, recordObject(metadata.model_profile)];
    return candidates.some((candidate) => {
      const modelKind = recordString(candidate, "model_kind").toLowerCase();
      const taskType = recordString(candidate, "task_type").toLowerCase();
      const model = recordString(candidate, "model").toLowerCase();
      return modelKind.includes("detect") || taskType.includes("object_detection") || model.includes("yolo");
    });
  });
}

export function detectionDefaultsFromChampionExportDemo(data: ChampionExportDemo) {
  const records = [
    data.modelProfile,
    data.deploymentProfile,
    data.modelCard,
    ...data.exports.map((item) => recordObject(item.metadata)),
    ...data.exports.map((item) => recordObject(recordObject(item.metadata).manifest)),
    ...data.exports.map((item) => recordObject(recordObject(recordObject(item.metadata).manifest).metadata)),
  ];
  let confidenceThreshold = 0.25;
  let iouThreshold = 0.7;
  for (const record of records) {
    const metadata = recordObject(record);
    const defaults = recordObject(metadata.confidence_threshold_defaults);
    const detection = recordObject(defaults.detection);
    const postprocessing = recordObject(metadata.postprocessing_contract);
    const nms = recordObject(postprocessing.nms);
    const confidence = numericValue(detection.confidence_threshold ?? nms.confidence_threshold ?? postprocessing.confidence_threshold);
    const iou = numericValue(detection.iou_threshold ?? nms.iou_threshold ?? postprocessing.iou_threshold);
    if (confidence > 0 && confidence <= 1) confidenceThreshold = confidence;
    if (iou > 0 && iou <= 1) iouThreshold = iou;
  }
  return {
    isDetection: championExportDemoIsDetection(data),
    confidenceThreshold,
    iouThreshold,
  };
}

export function detectionBoxesFromPrediction(prediction?: ChampionDemoPrediction | null): ChampionDetection[] {
  if (!prediction) return [];
  const metadata = { ...recordObject(prediction.image_metadata), ...recordObject(prediction.metadata) };
  let raw: unknown[] = [];
  if (Array.isArray(prediction.detections)) {
    raw = prediction.detections;
  } else if (Array.isArray(metadata.detections)) {
    raw = metadata.detections;
  } else {
    const detectionResult = recordObject(metadata.detection_result);
    if (Array.isArray(detectionResult.detections)) {
      raw = detectionResult.detections;
    }
  }
  return raw
    .map((item) => normalizeDetection(item))
    .filter((item): item is ChampionDetection => Boolean(item))
    .sort((left, right) => Number(right.confidence ?? right.score ?? 0) - Number(left.confidence ?? left.score ?? 0));
}

export function normalizeDetection(value: unknown): ChampionDetection | null {
  const record = recordObject(value);
  if (Object.keys(record).length === 0) return null;
  const box = normalizedDetectionBox(record as ChampionDetection);
  if (!box) return null;
  const label = recordString(record, "label") || recordString(record, "class_name") || recordString(record, "name");
  const confidence = numericValue(record.confidence ?? record.score);
  const classID = numericValue(record.class_id ?? record.class_index);
  return {
    ...(record as ChampionDetection),
    label: label || (Number.isFinite(classID) ? `class_${classID}` : "object"),
    class_name: label || (Number.isFinite(classID) ? `class_${classID}` : "object"),
    class_id: Number.isFinite(classID) ? Math.round(classID) : undefined,
    confidence: confidence || 0,
    score: confidence || 0,
    box,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    x1: box.x,
    y1: box.y,
    x2: box.x + box.width,
    y2: box.y + box.height,
  };
}

export function normalizedDetectionBox(detection: ChampionDetection): { x: number; y: number; width: number; height: number } | null {
  const box = recordObject(detection.box);
  const x = numericValue(box.x ?? detection.x ?? detection.x1);
  const y = numericValue(box.y ?? detection.y ?? detection.y1);
  const width = numericValue(box.width ?? detection.width);
  const height = numericValue(box.height ?? detection.height);
  if (width > 0 && height > 0) {
    return boundedDetectionBox(x, y, width, height);
  }
  const x1 = numericValue(box.x1 ?? detection.x1);
  const y1 = numericValue(box.y1 ?? detection.y1);
  const x2 = numericValue(box.x2 ?? detection.x2);
  const y2 = numericValue(box.y2 ?? detection.y2);
  if ([x1, y1, x2, y2].every(Number.isFinite)) {
    return boundedDetectionBox(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
  }
  return null;
}

export function boundedDetectionBox(x: number, y: number, width: number, height: number) {
  const left = Math.max(0, Math.min(1, x));
  const top = Math.max(0, Math.min(1, y));
  const boundedWidth = Math.max(0, Math.min(1 - left, width));
  const boundedHeight = Math.max(0, Math.min(1 - top, height));
  if (boundedWidth <= 0 || boundedHeight <= 0) return null;
  return { x: left, y: top, width: boundedWidth, height: boundedHeight };
}

export function predictionPostprocessLatency(prediction?: ChampionDemoPrediction | null) {
  if (!prediction) return 0;
  const metadata = { ...recordObject(prediction.image_metadata), ...recordObject(prediction.metadata) };
  const breakdown = recordObject(metadata.latency_breakdown_ms);
  return numericValue(prediction.postprocess_latency_ms ?? metadata.postprocess_latency_ms ?? breakdown.postprocess);
}

export function normalizeDemoPredictionResponse(value: ChampionDemoPrediction | { prediction?: ChampionDemoPrediction }) {
  const wrapped = recordObject(value).prediction;
  const prediction =
    recordObject(wrapped).id || Object.keys(recordObject(wrapped)).length > 0
      ? (wrapped as ChampionDemoPrediction)
      : (value as ChampionDemoPrediction);
  if (normalizedStatus(prediction.status || "") === "RUNTIME_UNAVAILABLE") {
    return { ...prediction, status: "RUNTIME_UNAVAILABLE", runtime_unavailable: true };
  }
  return prediction;
}

export function attachDemoPredictionPreview(prediction: ChampionDemoPrediction, image: ChampionDemoImage) {
  const previewURI = demoImagePreviewURI(image);
  if (!previewURI) return prediction;
  const metadata = {
    ...recordObject(prediction.image_metadata),
    ...recordObject(prediction.metadata),
    thumbnail_uri: previewURI,
  };
  return { ...prediction, metadata, image_metadata: { ...recordObject(prediction.image_metadata), preview_available: true } };
}

export function demoImageURI(image?: ChampionDemoImage | null) {
  return image?.uri || image?.image_uri || "";
}

export function demoImageInferenceURI(image?: ChampionDemoImage | null) {
  const source = demoImageURI(image);
  if (source && !source.startsWith("s3://")) return source;
  return "";
}

export function demoImagePreviewURI(image?: ChampionDemoImage | null) {
  return image?.preview_uri || image?.thumbnail_uri || image?.uri || image?.image_uri || "";
}

export function demoImageLabel(image?: ChampionDemoImage | null) {
  return image?.true_label || image?.label || image?.class_name || "";
}

export function demoImageDetail(image?: ChampionDemoImage | null) {
  if (!image) return "";
  const uri = demoImageURI(image);
  return [
    image.split,
    image.size_bytes ? formatBytes(image.size_bytes) : "",
    uri.startsWith("data:image") ? "inline test image" : uri,
  ]
    .filter(Boolean)
    .join(" - ");
}

export function demoPredictionRequestMetadata(image: ChampionDemoImage) {
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(recordObject(image.metadata))) {
    if (["path", "source_path", "source_image_path", "original_path", "preview_uri", "thumbnail_uri", "uri", "image_uri"].includes(key.toLowerCase())) continue;
    if (typeof value === "string" && isLikelyEncodedPayload(value)) continue;
    metadata[key] = value;
  }
  if (image.split) metadata.split = image.split;
  if (image.size_bytes) metadata.size_bytes = image.size_bytes;
  if (image.width) metadata.width = image.width;
  if (image.height) metadata.height = image.height;
  if (image.image_id || image.id) metadata.image_id = image.image_id || image.id;
  return metadata;
}

export function isTerminalDemoPredictionStatus(value?: string) {
  return ["SUCCEEDED", "FAILED", "RUNTIME_UNAVAILABLE"].includes(normalizedStatus(value || ""));
}

export function sleep(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export function nextDemoImageIndex(current: number, count: number) {
  if (count < 1) return 0;
  return (current + 1) % count;
}

export function randomDemoImageIndex(current: number, count: number) {
  if (count < 2) return 0;
  let next = Math.floor(Math.random() * count);
  if (next === current) next = (next + 1) % count;
  return next;
}

export function uniqueBy<T>(items: T[], keyForItem: (item: T, index: number) => string) {
  const seen = new Set<string>();
  return items.filter((item, index) => {
    const key = keyForItem(item, index);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function proposedExperimentSummaries(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => {
    const record = recordObject(item);
    const model = recordString(record, "model") || "experiment";
    const strategy = recordString(record, "strategy");
    const mechanism = recordFirstString(record, ["mechanism", "selected_mechanism", "proposal_mechanism"]);
    const intervention = recordString(record, "intervention");
    const expectedEffect = recordFirstString(record, ["expected_effect", "expected_metric_effect"]);
    const validationStatus = recordFirstString(record, ["backend_validation_status", "validation_status"]);
    const reason = recordString(record, "reason");
    const imageSize = recordNumber(record, "image_size");
    const evidence = stringArrayPayload(record.evidence_used).slice(0, 2).join("; ");
    const details = [
      mechanism,
      intervention,
      expectedEffect,
      strategy,
      imageSize ? `${imageSize}px` : "",
      validationStatus ? `validation ${validationStatus}` : "",
      evidence ? `evidence ${evidence}` : "",
      reason,
    ]
      .filter(Boolean)
      .join(" - ");
    return details ? `${model}: ${details}` : model;
  });
}

export function rejectedOptionSummaries(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => {
    const record = recordObject(item);
    const option = recordString(record, "option") || "Rejected option";
    const mechanism = recordString(record, "mechanism");
    const reason = recordString(record, "reason") || recordString(record, "evidence");
    const prefix = mechanism ? `${option} (${mechanism})` : option;
    return reason ? `${prefix}: ${reason}` : prefix;
  });
}

export function classifyRejectionReason(text: string) {
  const normalized = text.toLowerCase();
  if (normalized.includes("stop_guard") || normalized.includes("champion_selected_guard")) return "Stop guard";
  if (normalized.includes("validation")) return "Validation";
  if (normalized.includes("duplicate")) return "Duplicate signature";
  if (normalized.includes("unsupported") || normalized.includes("must be")) return "Unsupported option";
  if (normalized.includes("tiny") || normalized.includes("only epochs") || normalized.includes("learning rate")) {
    return "Too minor";
  }
  if (normalized.includes("cost") || normalized.includes("expensive")) return "Too expensive";
  if (normalized.includes("mode") || normalized.includes("autonomous") || normalized.includes("propose")) {
    return "Planning mode";
  }
  if (normalized.includes("objective") || normalized.includes("latency")) return "Objective fit";
  if (normalized.includes("score below")) return "Low score";
  return "Backend rejected";
}

export function uniqueStrings(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

export function numericValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

export function championModelProfile(champion: ProjectChampion): Record<string, unknown> {
  const deploymentProfile = recordObject(champion.deployment_profile.model_profile);
  if (Object.keys(deploymentProfile).length > 0) return deploymentProfile;
  return recordObject(champion.evaluation.model_profile);
}

export function championConfusionMatrix(champion: ProjectChampion) {
  return normalizedConfusionMatrix(champion.evaluation.confusion_matrix);
}

export function normalizedConfusionMatrix(value: unknown) {
  const matrix = value;
  if (!Array.isArray(matrix)) return [];
  return matrix
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((value) => (typeof value === "number" ? value : Number(value) || 0)));
}

export function perClassMetricRows(metrics: Record<string, unknown>) {
  return Object.entries(metrics)
    .filter(([label]) => !["accuracy", "macro avg", "weighted avg", "micro avg"].includes(label.toLowerCase()))
    .map(([label, value]) => {
      const record = recordObject(value);
      return {
        label,
        precision: numberPayload(record.precision),
        recall: numberPayload(record.recall),
        f1: numberPayload(record["f1-score"]) ?? numberPayload(record.f1),
      };
    })
    .filter((row) => row.precision !== null || row.recall !== null || row.f1 !== null);
}

export function formatModelSize(profile: Record<string, unknown>) {
  const size = recordNumber(profile, "model_size_mb") || recordNumber(profile, "estimated_model_size_mb");
  if (!size) return "-";
  return `${size.toFixed(size < 10 ? 2 : 1)} MB`;
}
