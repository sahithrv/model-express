import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Database,
  DollarSign,
  Eye,
  FolderOpen,
  HardDriveUpload,
  ImageIcon,
  Link2,
  ListRestart,
  MonitorDot,
  Pause,
  Play,
  Plus,
  RefreshCcw,
  Server,
  Shuffle,
  SlidersHorizontal,
  SquareTerminal,
  StepForward,
  MessageSquare,
  Timer,
  ThumbsDown,
  ThumbsUp,
  Trophy,
  Upload,
  X,
} from "lucide-react";
import {
  createChampionLocalRuntime,
  predictChampionImage,
  readyONNXExport,
  type ChampionLocalRuntime,
} from "./championLocalInference";
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
  RetrievedMemoryCard,
  StrategyScorecard,
  TrainingRunEvaluation,
  TrainingRunSummary,
  VisualAnalysisRerunPolicy,
  Worker,
  AutomationSettings,
  WorkerRequirement,
} from "./types";

const defaultBaseUrl = localStorage.getItem("orchestratorUrl") ?? "http://127.0.0.1:8080";
const jobsPerPage = 10;
const activeLiveRefreshIntervalMs = 10_000;
const idleLiveRefreshIntervalMs = 30_000;
const eventRefreshMinIntervalMs = 3_000;
const eventRefreshDebounceMs = 750;
const projectJobsFetchLimit = 100;
const trainingSummariesFetchLimit = 100;
const trainingEvaluationsFetchLimit = 50;
const selectedJobMetricsFetchLimit = 200;

type MetricKey = string;
type ProjectTabKey = "mission" | "activity" | "results" | "export" | "developer";
type LegacyProjectTabKey = "overview" | "data" | "experiments" | "agents" | "operations";
type ProjectTabTarget = ProjectTabKey | LegacyProjectTabKey;
type ActivityFilterKey = "all" | "decisions" | "experiments" | "results" | "blockers";
type ProjectWorkflowTabState = "done" | "active" | "waiting" | "blocked";
type ProjectWorkflowTabBase = { key: Exclude<ProjectTabKey, "developer">; label: string; icon: ReactNode };
type ProjectWorkflowTab = ProjectWorkflowTabBase & { state: ProjectWorkflowTabState; detail: string };

const classificationMetricPriority = ["macro_f1", "accuracy", "train_loss", "val_loss"];
const detectionMetricPriority = [
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
const detectionMetricAliases: Record<string, string[]> = {
  mAP50_95: ["mAP50_95", "map50_95", "map"],
  mAP50: ["mAP50", "map50"],
};

const projectTabs: ProjectWorkflowTabBase[] = [
  { key: "mission", label: "Mission", icon: <ClipboardList size={14} /> },
  { key: "activity", label: "Activity", icon: <Activity size={14} /> },
  { key: "results", label: "Results", icon: <BarChart3 size={14} /> },
  { key: "export", label: "Export", icon: <Trophy size={14} /> },
];

function metricLabel(key: MetricKey) {
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

function isDetectionJob(job: Job | null) {
  if (!job) return false;
  const config = job.config ?? {};
  const values = [job.template, config.task, config.task_type, config.model, config.architecture, config.framework]
    .map((value) => String(value ?? "").toLowerCase())
    .join(" ");
  return values.includes("object_detection") || values.includes("detection") || values.includes("yolo");
}

function isDetectionRun(summary: TrainingRunSummary | null, evaluation: TrainingRunEvaluation | null, job: Job | null) {
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

function firstPositiveMetric(values: Array<number | null | undefined>) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return 0;
}

function yoloMetricFromEvaluation(evaluation: TrainingRunEvaluation | null, metric: "map50_95" | "map50" | "precision" | "recall") {
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

function runPrimaryMetric(summary: TrainingRunSummary | null, evaluation: TrainingRunEvaluation | null, job: Job | null) {
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

function championPrimaryMetric(
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

function metricTabOptions(metrics: EpochMetric[], job: Job | null): Array<{ key: MetricKey; label: string }> {
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

type ProjectDetail = {
  decisions: AgentDecision[];
  datasets: Dataset[];
  telemetry: MissionControlTelemetryResponse | null;
  visualAnalysis: VisualAnalysisDetail;
  datasetMetadata: DatasetMetadataDetail;
  jobs: Job[];
  plans: ExperimentPlan[];
  runSummaries: TrainingRunSummary[];
  runEvaluations: TrainingRunEvaluation[];
  champion: ProjectChampion | null;
  championExports: ChampionExport[];
  championDemoImages: ChampionDemoImage[];
  championDemoPredictions: ChampionDemoPrediction[];
  championFeedback: ChampionFeedback[];
  workers: Worker[];
  workerRequirements: WorkerRequirement[];
  executionEvents: ExecutionEvent[];
  agentInvocations: AgentInvocation[];
  agentMemory: AgentMemoryRecord[];
  strategyScorecards: StrategyScorecard[];
};

type CancelExecutionResponse = {
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

type MissionDigestState =
  | "empty"
  | "dataset_needed"
  | "profiling"
  | "plan_ready"
  | "training"
  | "review_needed"
  | "follow_up_ready"
  | "champion_selected"
  | "blocked";

type MissionTone = "ok" | "warning" | "bad" | "info";

type MissionHealthItem = {
  id: string;
  label: string;
  value: string;
  tone: MissionTone;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

type MissionActionKey =
  | "new_project"
  | "refresh"
  | "execute_plan"
  | "review_experiments"
  | "schedule_follow_up"
  | "open_export";

type MissionNextAction = {
  id: string;
  label: string;
  reason: string;
  priority: "primary" | "secondary";
  disabled?: boolean;
  actionKey?: MissionActionKey;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

type MissionLiveActivity = {
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

type MissionSignal = {
  id: string;
  label: string;
  detail: string;
  tone: MissionTone;
  timestamp?: string;
  targetTab?: ProjectTabTarget;
  targetId?: string;
};

type MissionChampionSummary = {
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

type MissionDigest = {
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

type MissionBrief = {
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

type AIThinking = {
  state: string;
  observation: string;
  reasoning: string;
  decision: string;
  expectedOutcome: string;
  confidenceLabel: string;
  updatedAt: string;
};

type MissionStage = {
  id: string;
  label: string;
  detail: string;
  status: "done" | "active" | "waiting" | "blocked";
  timestamp?: string;
  evidence?: string;
};

type ActivityCardType = "mission" | "observation" | "decision" | "experiment" | "result" | "blocker" | "export";

type ActivityCardModel = {
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

type ResultsCandidate = {
  rank: number;
  model: string;
  metricLabel: string;
  metricValue: string;
  status: string;
  why: string;
  jobId: string;
};

type ResultsSummary = {
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

type ExportSummary = {
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

type DeveloperDiagnostics = {
  counts: Array<{ label: string; value: string }>;
  summary: string;
};

type VisualAnalysisDetail = {
  analysis: DatasetVisualAnalysis | null;
  status: "available" | "empty" | "unsupported" | "error";
  message: string;
  manualRunSupported: boolean;
  rerunPolicy?: VisualAnalysisRerunPolicy | null;
};

type DatasetMetadataDetail = {
  summary: DatasetMetadataSummary | null;
  imports: DatasetMetadataImport[];
  status: "available" | "empty" | "unsupported" | "error";
  message: string;
};

type TelemetryWindowKey = "today" | "7d" | "lifetime";

type TelemetryWindowSummary = {
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

type TelemetryCountSummary = {
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

type TelemetrySectionSummary = {
  name: string;
  calls: number;
  bytes: number;
  approxTokens: number;
  exampleSource?: string;
};

type TelemetryEmbeddingBreakdownRow = {
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

type TelemetryInvocationSummary = {
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

type TelemetryEmbeddingPurposeSummary = {
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

type TelemetrySummary = {
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

type VisualAnalysisListResponse = {
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

type AgentInvocationsResponse = {
  invocations?: AgentInvocation[];
  agent_invocations?: AgentInvocation[];
  items?: AgentInvocation[];
};

type ProjectDetailRefreshOptions = {
  includeSlowData?: boolean;
  forceSlowData?: boolean;
};

type ActivityStreamState = "idle" | "connecting" | "connected" | "reconnecting" | "fallback";

type RequestOptions = {
  method?: string;
  body?: unknown;
  bypassCache?: boolean;
  cacheTtlMs?: number;
};

type CachedGetRequest = {
  expiresAt: number;
  hasValue: boolean;
  promise?: Promise<unknown>;
  value?: unknown;
};

type DatasetMetadataSummaryResponse =
  | DatasetMetadataSummary
  | {
      summary?: DatasetMetadataSummary;
      metadata_summary?: DatasetMetadataSummary;
      dataset_metadata_summary?: DatasetMetadataSummary;
      agent_safe_summary?: DatasetMetadataSummary;
      import?: DatasetMetadataImport;
      metadata_import?: DatasetMetadataImport;
    };

type DatasetMetadataImportsResponse =
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

type OrchestratorHttpErrorResponse = {
  __mission_control_http_error: true;
  status: number;
  statusText?: string;
  message?: string;
  path?: string;
  url?: string;
  payload?: unknown;
};

type TimelineItem = {
  label: string;
  detail: string;
  status: "done" | "active" | "waiting" | "blocked";
  timestamp?: string;
};

type InsightItem = {
  label: string;
  value: string;
  tone?: "good" | "warn" | "bad";
};

type MetadataCountRow = {
  label: string;
  value: string;
};

type MetadataStatusDisplay = {
  status: string;
  detail: string;
  facts: InsightItem[];
  sources: string[];
  splitRows: MetadataCountRow[];
  annotationRows: MetadataCountRow[];
  warnings: string[];
  errors: string[];
};

type ReasoningSection = {
  title: string;
  items: string[];
};

type ChampionComparisonRow = {
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

type CandidateScoreRow = {
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

type RetrievedMemoryDisplay = {
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

type MemoryRetrievalProbeSnapshot = {
  id: string;
  purpose: string;
  logOnly: boolean;
  crossProjectOK: boolean;
  retrievedCount: number;
  createdAt: string;
  cards: RetrievedMemoryDisplay[];
};

type AgentInvocationAuditRow = {
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

type DecisionChatTurn = {
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

type MechanismCoverageRow = {
  mechanism: string;
  status: string;
  detail: string;
};

type DecisionQualitySnapshot = {
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

type ChampionExportDemo = {
  hasChampion: boolean;
  exportStatus: string;
  exports: ChampionExport[];
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

type ChampionFeedbackRating = ChampionFeedback["rating"];

type ChampionFeedbackDraft = {
  rating: ChampionFeedbackRating;
  message: string;
};

type Notice = {
  kind: "info" | "error";
  text: string;
};

function NoticeBanner({ notice }: { notice: Notice }) {
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

function noticeDisplay(notice: Notice) {
  if (notice.kind === "error" && notice.text.includes("Cannot read properties of undefined") && notice.text.includes("request")) {
    return {
      title: "Desktop bridge unavailable",
      message: "The browser preview cannot reach the Mission Control desktop bridge. Open the Electron app for live project actions.",
    };
  }

  return {
    title: notice.kind === "error" ? "Action needs attention" : "Mission update",
    message: notice.text,
  };
}

type DatasetFolder = {
  token: string;
  path: string;
  name: string;
  preflight?: DatasetUploadPreflight;
};

type DatasetUploadWarning = {
  code: string;
  message: string;
  threshold?: number;
  value?: number;
};

type DatasetUploadPreflight = {
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

type ScheduleFollowUpResponse = {
  decision: AgentDecision;
  follow_up_plan?: ExperimentPlan;
};

type AutomationSettingsUpdate = Partial<
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

const defaultAutomationSettings: AutomationSettings = {
  auto_review_experiments: false,
  auto_schedule_followups: false,
  auto_execute_plans: false,
  max_followup_rounds: 2,
  default_training_provider: "local",
  default_gpu_type: "",
  cost_mode: "balanced",
  budget_cap_usd: 0,
  llm_enabled: false,
  agent_mode: "propose",
  llm_provider: "openai",
  llm_model: "",
  automl_enabled: false,
  automl_sampler: "seeded_random",
  updated_at: "",
};

function emptyProjectDetail(message = "Select a dataset to load visual analysis evidence."): ProjectDetail {
  return {
    decisions: [],
    datasets: [],
    telemetry: null,
    visualAnalysis: {
      analysis: null,
      status: "empty",
      message,
      manualRunSupported: false,
    },
    datasetMetadata: {
      summary: null,
      imports: [],
      status: "empty",
      message: "Dataset metadata imports have not been reported.",
    },
    jobs: [],
    plans: [],
    runSummaries: [],
    runEvaluations: [],
    champion: null,
    championExports: [],
    championDemoImages: [],
    championDemoPredictions: [],
    championFeedback: [],
    workers: [],
    workerRequirements: [],
    executionEvents: [],
    agentInvocations: [],
    agentMemory: [],
    strategyScorecards: [],
  };
}

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

const expensiveGetCacheTtlMs = 15_000;

function cachedGetRequestTtlMs(path: string): number {
  const normalizedPath = path.split("?")[0] ?? path;
  if (/^\/projects\/[^/]+\/execution-events$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/projects\/[^/]+\/(agent-invocations|agent-decisions|agent-memory|strategy-scorecards|training-run-evaluations)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/projects\/[^/]+\/telemetry-summary$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/datasets\/[^/]+\/(visual-analyses|visual-analyses\/latest|metadata\/summary|metadata\/imports)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  if (/^\/projects\/[^/]+\/champion\/(exports|demo-images|demo-predictions|feedback)$/.test(normalizedPath)) {
    return expensiveGetCacheTtlMs;
  }
  return 0;
}

function eventNeedsSlowProjectRefresh(event: MessageEvent | Event): boolean {
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

export function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [health, setHealth] = useState<Health | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [detail, setDetail] = useState<ProjectDetail>(() => emptyProjectDetail());
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [metrics, setMetrics] = useState<EpochMetric[]>([]);
  const [automationSettings, setAutomationSettings] = useState<AutomationSettings>(defaultAutomationSettings);
  const [settingsDraft, setSettingsDraft] = useState<AutomationSettings>(defaultAutomationSettings);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [loading, setLoading] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newProjectFolder, setNewProjectFolder] = useState<DatasetFolder | null>(null);
	const [jobPage, setJobPage] = useState(0);
	const [selectedMetricKey, setSelectedMetricKey] = useState<MetricKey>("macro_f1");
	const [activeProjectTab, setActiveProjectTab] = useState<ProjectTabKey>("mission");
  const [activityFilter, setActivityFilter] = useState<ActivityFilterKey>("all");
	const [demoPrediction, setDemoPrediction] = useState<ChampionDemoPrediction | null>(null);
  const [demoPredictionError, setDemoPredictionError] = useState("");
  const [demoPredictionLoading, setDemoPredictionLoading] = useState(false);
  const [selectedDemoImageIndex, setSelectedDemoImageIndex] = useState(0);
  const [customDemoImage, setCustomDemoImage] = useState<ChampionDemoImage | null>(null);
  const [customDemoImageURI, setCustomDemoImageURI] = useState("");
  const [customDemoTrueLabel, setCustomDemoTrueLabel] = useState("");
  const [localInferenceStatus, setLocalInferenceStatus] = useState("not_ready");
  const [localInferenceError, setLocalInferenceError] = useState("");
  const [demoSlideshowEnabled, setDemoSlideshowEnabled] = useState(false);
  const [demoSlideshowIntervalMs, setDemoSlideshowIntervalMs] = useState(5200);
  const [detectionConfidenceThreshold, setDetectionConfidenceThreshold] = useState(0.25);
  const [detectionIouThreshold, setDetectionIouThreshold] = useState(0.7);
  const [championFeedbackDraft, setChampionFeedbackDraft] = useState<ChampionFeedbackDraft | null>(null);
  const [championFeedbackSubmitting, setChampionFeedbackSubmitting] = useState(false);
  const [activityEvents, setActivityEvents] = useState<AgentActivityEvent[]>([]);
  const [activityStreamState, setActivityStreamState] = useState<ActivityStreamState>("idle");
  const localRuntime = useRef<ChampionLocalRuntime | null>(null);
  const demoImagesRef = useRef<ChampionDemoImage[]>([]);
  const demoSlideshowInFlight = useRef(false);
  const supervisingRequirements = useRef<Set<string>>(new Set());
  const activeRequirementEnsureAt = useRef<Map<string, number>>(new Map());
  const eventRefreshInFlight = useRef(false);
  const eventRefreshTimer = useRef<number | null>(null);
  const eventRefreshQueuedSlow = useRef(false);
  const lastEventRefreshAt = useRef(0);
  const liveRefreshInFlight = useRef(false);
  const cachedGetRequests = useRef<Map<string, CachedGetRequest>>(new Map());
  const workerRequirementsRef = useRef<WorkerRequirement[]>([]);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const selectedJob = useMemo(
    () => detail.jobs.find((job) => job.id === selectedJobId) ?? null,
    [detail.jobs, selectedJobId],
  );

  const selectedRunSummary = useMemo(
    () => detail.runSummaries.find((summary) => summary.job_id === selectedJobId) ?? null,
    [detail.runSummaries, selectedJobId],
  );
  const selectedRunEvaluation = useMemo(
    () => detail.runEvaluations.find((evaluation) => evaluation.job_id === selectedJobId) ?? null,
    [detail.runEvaluations, selectedJobId],
  );
  const projectHasOpenWork = useMemo(
    () => detail.jobs.some((job) => ["QUEUED", "ASSIGNED", "RUNNING"].includes(normalizedStatus(job.status))),
    [detail.jobs],
  );
  const selectedMetricOptions = useMemo(
    () => metricTabOptions(metrics, selectedJob),
    [metrics, selectedJob],
  );

  useEffect(() => {
    if (selectedMetricOptions.length === 0) {
      return;
    }
    if (!selectedMetricOptions.some((metric) => metric.key === selectedMetricKey)) {
      setSelectedMetricKey(selectedMetricOptions[0].key);
    }
  }, [selectedMetricKey, selectedMetricOptions]);

  const latestPlan = detail.plans[0] ?? null;
  const stoppablePlan = useMemo(() => {
    const activePlanIds = new Set<string>();
    detail.jobs.forEach((job) => {
      const status = normalizedStatus(job.status);
      if (!["QUEUED", "ASSIGNED", "RUNNING"].includes(status)) return;
      const planId = recordString(job.config, "plan_id");
      if (planId) activePlanIds.add(planId);
    });
    detail.workerRequirements.forEach((requirement) => {
      if (!["PENDING", "STARTING", "ACTIVE"].includes(normalizedStatus(requirement.status))) return;
      if (requirement.plan_id) activePlanIds.add(requirement.plan_id);
    });
    if (latestPlan && activePlanIds.has(latestPlan.id)) {
      return latestPlan;
    }
    return detail.plans.find((plan) => activePlanIds.has(plan.id)) ?? null;
  }, [detail.jobs, detail.plans, detail.workerRequirements, latestPlan]);
  const latestDecision = detail.decisions[0] ?? null;
  const latestDecisionHasFollowUpPlan = latestDecision
    ? detail.plans.some((plan) => plan.source_decision_id === latestDecision.id)
    : false;
  const decisionChatTurns = useMemo(() => buildDecisionChatTurns(detail.decisions), [detail.decisions]);
  const runTotals = useMemo(
    () => summarizeTrainingRuns(detail.runSummaries, detail.runEvaluations, detail.jobs),
    [detail.jobs, detail.runEvaluations, detail.runSummaries],
  );
  const selectedChampionPrimaryMetric = useMemo(() => {
    if (!detail.champion) return null;
    return championPrimaryMetric(
      detail.champion,
      detail.runSummaries.find((summary) => summary.job_id === detail.champion?.job_id) ?? null,
      detail.runEvaluations.find((evaluation) => evaluation.job_id === detail.champion?.job_id) ?? null,
      detail.jobs.find((job) => job.id === detail.champion?.job_id) ?? null,
    );
  }, [detail.champion, detail.jobs, detail.runEvaluations, detail.runSummaries]);
  const timelineItems = useMemo(
    () => buildExperimentTimeline(selectedProject, detail),
    [detail, selectedProject],
  );
  const visibleActivityEvents = useMemo(
    () => (activityEvents.length > 0 ? activityEvents : buildFallbackActivityEvents(selectedProjectId, detail)),
    [activityEvents, detail, selectedProjectId],
  );
  const missionDigest = useMemo(
    () =>
      buildMissionDigest({
        health,
        selectedProject,
        detail,
        automationSettings,
        activityStreamState,
        visibleActivityEvents,
        loading,
      }),
    [activityStreamState, automationSettings, detail, health, loading, selectedProject, visibleActivityEvents],
  );
  const memoryRetrievalProbe = useMemo(
    () => buildMemoryRetrievalProbeSnapshots(detail.executionEvents),
    [detail.executionEvents],
  );
  const datasetIntelligence = useMemo(
    () => buildDatasetIntelligence(detail.datasets[0] ?? null, latestDecision, detail.datasetMetadata),
    [detail.datasetMetadata, detail.datasets, latestDecision],
  );
  const championComparison = useMemo(
    () => buildChampionComparison(detail.runSummaries, detail.runEvaluations, detail.jobs, detail.champion),
    [detail.champion, detail.jobs, detail.runEvaluations, detail.runSummaries],
  );
  const championExportDemo = useMemo(() => buildChampionExportDemo(detail), [detail]);
  const missionBrief = useMemo(
    () => buildMissionBrief(selectedProject, detail, missionDigest, automationSettings),
    [automationSettings, detail, missionDigest, selectedProject],
  );
  const currentThinking = useMemo(
    () => buildCurrentThinking(selectedProject, detail, missionDigest),
    [detail, missionDigest, selectedProject],
  );
  const missionStages = useMemo(
    () => buildMissionStages(selectedProject, detail, missionDigest, championExportDemo),
    [championExportDemo, detail, missionDigest, selectedProject],
  );
  const activityFeed = useMemo(
    () => buildActivityFeed(selectedProject, detail, visibleActivityEvents, championExportDemo),
    [championExportDemo, detail, selectedProject, visibleActivityEvents],
  );
  const resultsSummary = useMemo(
    () => buildResultsSummary(detail, championComparison, championExportDemo),
    [championComparison, championExportDemo, detail],
  );
  const exportSummary = useMemo(
    () => buildExportSummary(detail, championExportDemo),
    [championExportDemo, detail],
  );
  const projectWorkflowTabs = useMemo(
    () =>
      buildProjectWorkflowTabs({
        tabs: projectTabs,
        missionDigest,
        missionStages,
        activityFeed,
        activityStreamState,
        resultsSummary,
        exportSummary,
      }),
    [activityFeed, activityStreamState, exportSummary, missionDigest, missionStages, resultsSummary],
  );
  const developerDiagnostics = useMemo(
    () => buildDeveloperDiagnostics(detail, visibleActivityEvents),
    [detail, visibleActivityEvents],
  );
  const championDetectionDefaults = useMemo(() => detectionDefaultsFromChampionExportDemo(championExportDemo), [championExportDemo]);
  const reviewState = automationReviewState(automationSettings);

  const firstDatasetId = detail.datasets[0]?.id ?? "";
  const jobPageCount = Math.max(1, Math.ceil(detail.jobs.length / jobsPerPage));
  const visibleJobs = detail.jobs.slice(jobPage * jobsPerPage, jobPage * jobsPerPage + jobsPerPage);

  const request = useCallback(
    async <T,>(path: string, options: RequestOptions = {}) => {
      const method = (options.method ?? "GET").toUpperCase();
      const runRequest = async () => {
        const response = await window.missionControl.request<T | OrchestratorHttpErrorResponse>({
          baseUrl,
          path,
          method: options.method,
          body: options.body,
        });
        if (isOrchestratorHttpErrorResponse(response)) {
          const statusText = response.statusText ? ` ${response.statusText}` : "";
          const message = response.message || "request failed";
          const requestPath = response.path ? ` (${response.path})` : "";
          throw new Error(`${response.status}${statusText} ${message}${requestPath}`);
        }
        return response;
      };

      if (method !== "GET") {
        const response = await runRequest();
        cachedGetRequests.current.clear();
        return response;
      }

      const cacheTtlMs = options.bypassCache ? 0 : options.cacheTtlMs ?? cachedGetRequestTtlMs(path);
      const cacheKey = `${baseUrl} ${path}`;
      const now = Date.now();
      const cached = cachedGetRequests.current.get(cacheKey);
      if (cached?.promise) {
        return cached.promise as Promise<T>;
      }
      if (!options.bypassCache && cached?.hasValue && cached.expiresAt > now) {
        return cached.value as T;
      }

      const promise = runRequest();
      cachedGetRequests.current.set(cacheKey, {
        expiresAt: now + cacheTtlMs,
        hasValue: false,
        promise,
      });
      try {
        const response = await promise;
        if (cacheTtlMs > 0) {
          cachedGetRequests.current.set(cacheKey, {
            expiresAt: Date.now() + cacheTtlMs,
            hasValue: true,
            value: response,
          });
        } else {
          const current = cachedGetRequests.current.get(cacheKey);
          if (current?.promise === promise) {
            cachedGetRequests.current.delete(cacheKey);
          }
        }
        return response;
      } catch (error) {
        const current = cachedGetRequests.current.get(cacheKey);
        if (current?.promise === promise) {
          cachedGetRequests.current.delete(cacheKey);
        }
        throw error;
      }
    },
    [baseUrl],
  );

  const refreshProjects = useCallback(async () => {
    const response = await request<{ projects: Project[] }>("/projects");
    setProjects(response.projects);
    if (!selectedProjectId && response.projects.length > 0) {
      setSelectedProjectId(response.projects[0].id);
    }
  }, [request, selectedProjectId]);

  const refreshHealth = useCallback(async () => {
    const response = await request<Health>("/healthz");
    setHealth(response);
  }, [request]);

  const refreshAutomationSettings = useCallback(async () => {
    const response = await request<AutomationSettings>("/settings/automation");
    setAutomationSettings(response);
    setSettingsDraft(response);
  }, [request]);

  const fetchLatestDatasetVisualAnalysis = useCallback(
    async (dataset: Dataset | null, options: Pick<RequestOptions, "bypassCache"> = {}): Promise<VisualAnalysisDetail> => {
      if (!dataset) {
        return {
          analysis: null,
          status: "empty",
          message: "Upload a dataset before visual analysis can run.",
          manualRunSupported: false,
          rerunPolicy: null,
        };
      }

      const profileFallback = visualAnalysisFromProfile(dataset.profile);

      try {
        const response = await request<VisualAnalysisListResponse>(`/datasets/${dataset.id}/visual-analyses`, options);
        const latest = latestVisualAnalysis(visualAnalysesFromResponse(response)) ?? profileFallback;
        return {
          analysis: latest,
          status: latest ? "available" : "empty",
          message: latest
            ? "Latest visual analysis loaded from the backend."
            : "Visual analysis API is available; no analysis has been recorded for this dataset yet.",
          manualRunSupported: response.manual_run_supported !== false,
          rerunPolicy: visualAnalysisRerunPolicyFromResponse(response),
        };
      } catch (listError) {
        try {
          const response = await request<VisualAnalysisListResponse | DatasetVisualAnalysis>(
            `/datasets/${dataset.id}/visual-analyses/latest`,
            options,
          );
          const latest = latestVisualAnalysis(visualAnalysesFromResponse(response)) ?? profileFallback;
          return {
            analysis: latest,
            status: latest ? "available" : "empty",
            message: latest
              ? "Latest visual analysis loaded from the backend."
              : "Visual analysis API is available; no analysis has been recorded for this dataset yet.",
            manualRunSupported: true,
            rerunPolicy: visualAnalysisRerunPolicyFromResponse(response),
          };
        } catch (latestError) {
          if (profileFallback) {
            return {
              analysis: profileFallback,
              status: "available",
              message: "Showing visual analysis stored on the dataset profile; dedicated API endpoints are not available.",
              manualRunSupported: false,
              rerunPolicy: null,
            };
          }

          const error = latestError instanceof Error ? latestError : listError;
          return {
            analysis: null,
            status: isUnsupportedEndpointError(error) ? "unsupported" : "error",
            message: isUnsupportedEndpointError(error)
              ? "This backend does not expose dataset visual-analysis endpoints yet."
              : `Visual analysis lookup failed: ${errorMessage(error)}`,
            manualRunSupported: false,
            rerunPolicy: null,
          };
        }
      }
    },
    [request],
  );

  const fetchLatestDatasetMetadata = useCallback(
    async (dataset: Dataset | null, options: Pick<RequestOptions, "bypassCache"> = {}): Promise<DatasetMetadataDetail> => {
      if (!dataset) {
        return {
          summary: null,
          imports: [],
          status: "empty",
          message: "Upload a dataset before metadata imports can run.",
        };
      }

      const [summaryResult, importsResult] = await Promise.all([
        request<DatasetMetadataSummaryResponse>(`/datasets/${dataset.id}/metadata/summary`, options)
          .then((response) => ({ response }))
          .catch((error: unknown) => ({ error })),
        request<DatasetMetadataImportsResponse>(`/datasets/${dataset.id}/metadata/imports`, options)
          .then((response) => ({ response }))
          .catch((error: unknown) => ({ error })),
      ]);

      const imports =
        "response" in importsResult ? datasetMetadataImportsFromResponse(importsResult.response) : [];
      const summary =
        "response" in summaryResult
          ? datasetMetadataSummaryFromResponse(summaryResult.response, imports)
          : datasetMetadataSummaryFromImports(imports);

      if (summary) {
        return {
          summary,
          imports,
          status: "available",
          message: "Dataset metadata summary loaded from the backend.",
        };
      }

      const summaryError = "error" in summaryResult ? summaryResult.error : null;
      const importsError = "error" in importsResult ? importsResult.error : null;
      const errors = [summaryError, importsError].filter(Boolean);
      if (errors.length > 0 && errors.every(isUnsupportedEndpointError)) {
        return {
          summary: null,
          imports: [],
          status: "unsupported",
          message: "This backend does not expose dataset metadata endpoints yet.",
        };
      }
      if (errors.length > 0 && !errors.every(isUnsupportedEndpointError)) {
        const error = errors.find((item) => !isUnsupportedEndpointError(item)) ?? errors[0];
        return {
          summary: null,
          imports,
          status: "error",
          message: `Metadata status lookup failed: ${errorMessage(error)}`,
        };
      }

      return {
        summary: null,
        imports,
        status: "empty",
        message: "No metadata imports have been recorded for this dataset.",
      };
    },
    [request],
  );

  const refreshProjectDetail = useCallback(
    async (projectId: string, options: ProjectDetailRefreshOptions = {}) => {
      if (!projectId) {
        setDetail(emptyProjectDetail());
        return;
      }
      const includeSlowData = options.includeSlowData ?? true;
      const slowRequestOptions: Pick<RequestOptions, "bypassCache"> = {
        bypassCache: options.forceSlowData ?? false,
      };

      const [
        datasets,
        jobs,
        plans,
        runSummaries,
        champion,
        workers,
        workerRequirements,
        executionEvents,
      ] = await Promise.all([
        request<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`),
        request<{ jobs: Job[] }>(`/projects/${projectId}/jobs?limit=${projectJobsFetchLimit}`),
        request<{ plans: ExperimentPlan[] }>(`/projects/${projectId}/plans`),
        request<{ summaries: TrainingRunSummary[] }>(`/projects/${projectId}/training-run-summaries?limit=${trainingSummariesFetchLimit}`),
        request<{ champion: ProjectChampion | null }>(`/projects/${projectId}/champion`),
        request<{ workers: Worker[] }>(`/projects/${projectId}/workers`),
        request<{ requirements: WorkerRequirement[] }>(`/projects/${projectId}/worker-requirements`),
        request<{ events: ExecutionEvent[] }>(`/projects/${projectId}/execution-events?limit=8`),
      ]);

      const firstDataset = datasets.datasets[0] ?? null;
      const slowData = includeSlowData
        ? await Promise.all([
            request<{ evaluations: TrainingRunEvaluation[] }>(
              `/projects/${projectId}/training-run-evaluations?limit=${trainingEvaluationsFetchLimit}&compact=1`,
              slowRequestOptions,
            ).catch(
              (): { evaluations: TrainingRunEvaluation[] } => ({ evaluations: [] }),
            ),
            request<{ decisions: AgentDecision[] }>(`/projects/${projectId}/agent-decisions`, slowRequestOptions).catch(
              (): { decisions: AgentDecision[] } => ({ decisions: [] }),
            ),
            request<AgentInvocationsResponse>(`/projects/${projectId}/agent-invocations?limit=8`, slowRequestOptions).catch(
              (): AgentInvocationsResponse => ({ invocations: [] }),
            ),
            request<MissionControlTelemetryResponse>(`/projects/${projectId}/telemetry-summary?limit=1000`, slowRequestOptions).catch(
              (): MissionControlTelemetryResponse => ({}),
            ),
            request<{ records: AgentMemoryRecord[] }>(`/projects/${projectId}/agent-memory?limit=6`, slowRequestOptions).catch(
              (): { records: AgentMemoryRecord[] } => ({ records: [] }),
            ),
            request<{ scorecards: StrategyScorecard[] }>(`/projects/${projectId}/strategy-scorecards?limit=6`, slowRequestOptions).catch(
              (): { scorecards: StrategyScorecard[] } => ({ scorecards: [] }),
            ),
            fetchLatestDatasetVisualAnalysis(firstDataset, slowRequestOptions),
            fetchLatestDatasetMetadata(firstDataset, slowRequestOptions),
          ])
        : null;
      const runEvaluations = slowData?.[0];
      const decisions = slowData?.[1];
      const agentInvocations = slowData?.[2];
      const telemetry = slowData?.[3];
      const agentMemory = slowData?.[4];
      const strategyScorecards = slowData?.[5];
      const visualAnalysis = slowData?.[6];
      const datasetMetadata = slowData?.[7];

      const championValue = champion.champion;
      const championSlowData = includeSlowData && championValue
        ? await Promise.all([
            request<{ exports: ChampionExport[] }>(`/projects/${projectId}/champion/exports`, slowRequestOptions).catch(() => ({ exports: [] })),
            request<{ images: ChampionDemoImage[] }>(
              `/projects/${projectId}/champion/demo-images?max_total_images=32&max_per_class=4`,
              slowRequestOptions,
            ).catch(() => ({ images: [] })),
            request<{ predictions?: ChampionDemoPrediction[]; history?: ChampionDemoPrediction[]; demo_predictions?: ChampionDemoPrediction[] }>(
              `/projects/${projectId}/champion/demo-predictions?limit=8`,
              slowRequestOptions,
            ).catch((): { predictions?: ChampionDemoPrediction[]; history?: ChampionDemoPrediction[]; demo_predictions?: ChampionDemoPrediction[] } => ({
              predictions: [],
            })),
            request<{ feedback?: ChampionFeedback[]; items?: ChampionFeedback[] }>(
              `/projects/${projectId}/champion/feedback`,
              slowRequestOptions,
            ).catch((): { feedback?: ChampionFeedback[]; items?: ChampionFeedback[] } => ({ feedback: [] })),
          ])
        : null;
      const championExports = championSlowData?.[0];
      const championDemoImages = championSlowData?.[1];
      const championDemoPredictions = championSlowData?.[2];
      const championFeedback = championSlowData?.[3];

      setDetail((previous) => {
        const previousChampionMatches =
          championValue && previous.champion && previous.champion.job_id === championValue.job_id;
        return {
          decisions: decisions?.decisions ?? previous.decisions,
          datasets: datasets.datasets,
          visualAnalysis: visualAnalysis ?? previous.visualAnalysis,
          datasetMetadata: datasetMetadata ?? previous.datasetMetadata,
          jobs: jobs.jobs,
          plans: plans.plans,
          runSummaries: runSummaries.summaries,
          runEvaluations: runEvaluations?.evaluations ?? previous.runEvaluations,
          champion: championValue,
          championExports: championValue
            ? championExports?.exports ?? (previousChampionMatches ? previous.championExports : [])
            : [],
          championDemoImages: championValue
            ? championDemoImages?.images ?? (previousChampionMatches ? previous.championDemoImages : [])
            : [],
          championDemoPredictions:
            championValue
              ? championDemoPredictions?.predictions ??
                championDemoPredictions?.history ??
                championDemoPredictions?.demo_predictions ??
                (previousChampionMatches ? previous.championDemoPredictions : [])
              : [],
          championFeedback: championValue
            ? championFeedback?.feedback ?? championFeedback?.items ?? (previousChampionMatches ? previous.championFeedback : [])
            : [],
          workers: workers.workers,
          workerRequirements: workerRequirements.requirements,
          executionEvents: executionEvents.events,
          agentInvocations: agentInvocations ? agentInvocationsFromResponse(agentInvocations) : previous.agentInvocations,
          telemetry: telemetry ?? previous.telemetry,
          agentMemory: agentMemory?.records ?? previous.agentMemory,
          strategyScorecards: strategyScorecards?.scorecards ?? previous.strategyScorecards,
        };
      });

      setSelectedJobId((currentJobId) => {
        if (jobs.jobs.length === 0) return "";
        if (jobs.jobs.some((job) => job.id === currentJobId)) return currentJobId;
        return jobs.jobs[0].id;
      });
    },
    [fetchLatestDatasetMetadata, fetchLatestDatasetVisualAnalysis, request],
  );

  const refreshSelectedJobMetrics = useCallback(async () => {
    if (!selectedJobId) {
      setMetrics([]);
      return;
    }

    const response = await request<{ metrics: EpochMetric[] }>(`/jobs/${selectedJobId}/metrics?limit=${selectedJobMetricsFetchLimit}`);
    setMetrics(response.metrics);
  }, [request, selectedJobId]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setNotice(null);
    try {
      await refreshHealth();
      await refreshAutomationSettings();
      await refreshProjects();
      if (selectedProjectId) {
        await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
      }
      await refreshSelectedJobMetrics();
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }, [
    refreshAutomationSettings,
    refreshHealth,
    refreshProjectDetail,
    refreshProjects,
    refreshSelectedJobMetrics,
    selectedProjectId,
  ]);

  const refreshLive = useCallback(async (options: ProjectDetailRefreshOptions = { includeSlowData: false }) => {
    if (liveRefreshInFlight.current) {
      return;
    }
    const includeSlowData = options.includeSlowData ?? false;
    liveRefreshInFlight.current = true;
    try {
      await refreshHealth();
      await refreshProjects();
      if (selectedProjectId) {
        await refreshProjectDetail(selectedProjectId, { includeSlowData, forceSlowData: includeSlowData });
      }
      await refreshSelectedJobMetrics();
    } catch {
      setHealth(null);
    } finally {
      liveRefreshInFlight.current = false;
    }
  }, [refreshHealth, refreshProjectDetail, refreshProjects, refreshSelectedJobMetrics, selectedProjectId]);

  const superviseWorkerRequirements = useCallback(async () => {
    if (!selectedProjectId) return;

    const actionable = workerRequirementsRef.current.filter((requirement) =>
      (requirement.status === "PENDING" || requirement.status === "STARTING" || requirement.status === "ACTIVE") &&
      workerRequirementHasOpenWork(requirement, detail.jobs),
    );
    const now = Date.now();

    for (const requirement of actionable) {
      if (supervisingRequirements.current.has(requirement.id)) {
        continue;
      }
      const alreadyActive = requirement.status === "ACTIVE";
      const lastEnsureAt = activeRequirementEnsureAt.current.get(requirement.id) ?? 0;
      if (alreadyActive && now - lastEnsureAt < 30_000) {
        continue;
      }
      activeRequirementEnsureAt.current.set(requirement.id, now);
      supervisingRequirements.current.add(requirement.id);
      try {
        if (!alreadyActive) {
          await request<WorkerRequirement>(`/worker-requirements/${requirement.id}`, {
            method: "PATCH",
            body: { status: "STARTING", last_error: "" },
          });
        }

        await window.missionControl.ensureProjectWorker({
          projectId: requirement.project_id,
          baseUrl,
          name: `auto-worker-${requirement.project_id}`,
          gpuType: requirement.provider === "modal" ? "modal" : requirement.gpu_type || requirement.provider || "local",
          count: requirement.target_count,
        });

        if (!alreadyActive) {
          await request<WorkerRequirement>(`/worker-requirements/${requirement.id}`, {
            method: "PATCH",
            body: { status: "ACTIVE", last_error: "" },
          });
        }
      } catch (error) {
        await request<WorkerRequirement>(`/worker-requirements/${requirement.id}`, {
          method: "PATCH",
          body: {
            status: "FAILED",
            last_error: error instanceof Error ? error.message : String(error),
          },
        }).catch(() => undefined);
      } finally {
        supervisingRequirements.current.delete(requirement.id);
        refreshProjectDetail(selectedProjectId, { includeSlowData: false }).catch(() => undefined);
      }
    }
  }, [baseUrl, detail.jobs, refreshProjectDetail, request, selectedProjectId]);

  useEffect(() => {
    localStorage.setItem("orchestratorUrl", baseUrl);
  }, [baseUrl]);

  useEffect(() => {
    workerRequirementsRef.current = detail.workerRequirements;
  }, [detail.workerRequirements]);

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (selectedProjectId) {
      setDemoPrediction(null);
      setDemoPredictionError("");
      setSelectedDemoImageIndex(0);
      setCustomDemoImage(null);
      setCustomDemoImageURI("");
      setCustomDemoTrueLabel("");
      setDemoSlideshowEnabled(false);
      setDemoSlideshowIntervalMs(5200);
      setDetectionConfidenceThreshold(0.25);
      setDetectionIouThreshold(0.7);
      setLocalInferenceStatus("not_ready");
      setLocalInferenceError("");
      setActivityEvents([]);
      setActivityStreamState("connecting");
      localRuntime.current = null;
      activeRequirementEnsureAt.current.clear();
      setJobPage(0);
      refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true }).catch((error) =>
        setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
      );
    }
  }, [refreshProjectDetail, selectedProjectId]);

  useEffect(() => {
    setSelectedDemoImageIndex((current) => {
      if (championExportDemo.demoImages.length === 0) return 0;
      return Math.min(current, championExportDemo.demoImages.length - 1);
    });
  }, [championExportDemo.demoImages.length]);

  useEffect(() => {
    demoImagesRef.current = championExportDemo.demoImages;
  }, [championExportDemo.demoImages]);

  useEffect(() => {
    if (readyONNXExport(championExportDemo.exports)) {
      setLocalInferenceStatus((status) => (status === "not_ready" ? "available" : status));
    } else {
      localRuntime.current = null;
      setLocalInferenceStatus("not_ready");
      setLocalInferenceError("");
    }
  }, [championExportDemo.exports]);

  useEffect(() => {
    if (!championDetectionDefaults.isDetection) return;
    setDetectionConfidenceThreshold(championDetectionDefaults.confidenceThreshold);
    setDetectionIouThreshold(championDetectionDefaults.iouThreshold);
  }, [
    championDetectionDefaults.confidenceThreshold,
    championDetectionDefaults.iouThreshold,
    championDetectionDefaults.isDetection,
  ]);

  useEffect(() => {
    if (!demoSlideshowEnabled) return;
    const runNextSlide = () => {
      if (demoSlideshowInFlight.current) return;
      const images = demoImagesRef.current;
      if (images.length === 0) return;
      let imageToRun: ChampionDemoImage | null = null;
      setSelectedDemoImageIndex((current) => {
        const next = nextDemoImageIndex(current, images.length);
        imageToRun = images[next] ?? null;
        return next;
      });
      if (!imageToRun) return;
      demoSlideshowInFlight.current = true;
      runChampionDemoPrediction(imageToRun)
        .catch((error) => setDemoPredictionError(error instanceof Error ? error.message : String(error)))
        .finally(() => {
          demoSlideshowInFlight.current = false;
        });
    };

    runNextSlide();
    const timer = window.setInterval(runNextSlide, demoSlideshowIntervalMs);
    return () => window.clearInterval(timer);
  }, [demoSlideshowEnabled, demoSlideshowIntervalMs, selectedProjectId]);

  useEffect(() => {
    refreshSelectedJobMetrics().catch((error) =>
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
    );
  }, [refreshSelectedJobMetrics]);

  useEffect(() => {
    const interval = projectHasOpenWork ? activeLiveRefreshIntervalMs : idleLiveRefreshIntervalMs;
    const timer = window.setInterval(() => {
      refreshLive();
    }, interval);

    return () => window.clearInterval(timer);
  }, [projectHasOpenWork, refreshLive]);

  useEffect(() => {
    if (!selectedProjectId) {
      setActivityStreamState("idle");
      return;
    }
    if (typeof EventSource === "undefined") {
      setActivityStreamState("fallback");
      return;
    }

    let closed = false;
    setActivityStreamState("connecting");
    const streamUrl = new URL(`/projects/${selectedProjectId}/activity-stream`, baseUrl);
    streamUrl.searchParams.set("limit", "12");
    streamUrl.searchParams.set("interval_ms", projectHasOpenWork ? "5000" : "10000");
    const events = new EventSource(streamUrl.toString());
    const triggerRefresh = (event: MessageEvent | Event) => {
      if (closed) return;
      const includeSlowData = eventNeedsSlowProjectRefresh(event);
      eventRefreshQueuedSlow.current = eventRefreshQueuedSlow.current || includeSlowData;
      if (eventRefreshInFlight.current || eventRefreshTimer.current !== null) return;
      const elapsed = Date.now() - lastEventRefreshAt.current;
      const delay = Math.max(eventRefreshDebounceMs, eventRefreshMinIntervalMs - elapsed);
      eventRefreshTimer.current = window.setTimeout(() => {
        eventRefreshTimer.current = null;
        eventRefreshInFlight.current = true;
        lastEventRefreshAt.current = Date.now();
        const shouldIncludeSlowData = eventRefreshQueuedSlow.current;
        eventRefreshQueuedSlow.current = false;
        refreshLive({ includeSlowData: shouldIncludeSlowData })
          .catch(() => undefined)
          .finally(() => {
            eventRefreshInFlight.current = false;
          });
      }, delay);
    };

    const handleActivityEvent = (event: MessageEvent) => {
      const activity = activityEventFromMessage(event);
      if (activity) {
        setActivityEvents((current) => mergeActivityEvents(current, activity));
      }
      triggerRefresh(event);
    };

    events.onopen = () => {
      if (!closed) setActivityStreamState("connected");
    };
    events.onmessage = (event) => {
      handleActivityEvent(event);
    };
    events.addEventListener("activity_event", handleActivityEvent);
    events.addEventListener("stream_error", () => {
      if (!closed) setActivityStreamState("fallback");
    });
    events.onerror = () => {
      if (!closed) setActivityStreamState("reconnecting");
    };

    return () => {
      closed = true;
      if (eventRefreshTimer.current !== null) {
        window.clearTimeout(eventRefreshTimer.current);
        eventRefreshTimer.current = null;
      }
      events.close();
    };
  }, [baseUrl, projectHasOpenWork, refreshLive, selectedProjectId]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      superviseWorkerRequirements().catch(() => undefined);
    }, 3000);

    superviseWorkerRequirements().catch(() => undefined);
    return () => window.clearInterval(timer);
  }, [superviseWorkerRequirements]);

  async function chooseNewProjectFolder() {
    const folder = await window.missionControl.selectDatasetFolder();
    if (folder) {
      try {
        const preflight = await window.missionControl.preflightDatasetFolder({ datasetToken: folder.token });
        setNewProjectFolder({ ...folder, preflight });
      } catch (error) {
        setNewProjectFolder(folder);
        setNotice({ kind: "error", text: errorMessage(error) });
      }
    }
  }

  async function createProjectWithDataset(formData: FormData) {
    const name = String(formData.get("name") ?? "").trim();
    const goal = String(formData.get("goal") ?? "").trim();

    if (!name || !newProjectFolder) {
      setNotice({ kind: "error", text: "Project name and dataset folder are required." });
      return;
    }
    const uploadWarnings = newProjectFolder.preflight?.warnings ?? [];
    if (uploadWarnings.length > 0) {
      const summary = uploadWarnings.map((warning) => warning.message).join("\n");
      const confirmed = window.confirm(`${summary}\n\nContinue with dataset upload?`);
      if (!confirmed) {
        return;
      }
    }

    setLoading(true);
    setNotice(null);
    try {
      const project = await request<Project>("/projects", {
        method: "POST",
        body: { name, goal },
      });

      const metadata = await window.missionControl.uploadDatasetFolder({
        projectId: project.id,
        datasetToken: newProjectFolder.token,
      });

      await request<Dataset>(`/projects/${project.id}/datasets`, {
        method: "POST",
        body: metadata,
      });

      const profileProvider = automationSettings.default_training_provider === "modal" ? "modal" : "local";
      let workerMessage = profileProvider === "modal" ? "Modal profiling worker started." : "Profiling worker started.";
      try {
        const workerProcess = await window.missionControl.ensureProjectWorker({
          projectId: project.id,
          baseUrl,
          name: `${profileProvider}-profile-worker-${project.id}`,
          gpuType: profileProvider,
        });
        workerMessage = workerProcess.started
          ? workerMessage
          : `${profileProvider === "modal" ? "Modal profiling" : "Profiling"} worker is already running.`;
      } catch (error) {
        workerMessage = `Worker did not start: ${error instanceof Error ? error.message : String(error)}`;
      }

      setSelectedProjectId(project.id);
      setNewProjectFolder(null);
      setNewProjectOpen(false);
      await refreshProjects();
      await refreshProjectDetail(project.id, { includeSlowData: true, forceSlowData: true });
      setNotice({ kind: "info", text: `Created ${project.name} with dataset ${metadata.name}. ${workerMessage}` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function createJob(formData: FormData) {
    if (!selectedProjectId) return;

    const configText = String(formData.get("config") ?? "{}").trim() || "{}";
    const config = JSON.parse(configText) as Record<string, unknown>;

    const job = await request<Job>(`/projects/${selectedProjectId}/jobs`, {
      method: "POST",
      body: {
        template: String(formData.get("template") ?? "profile_dataset"),
        config,
      },
    });

    setSelectedJobId(job.id);
    await refreshProjectDetail(selectedProjectId, { includeSlowData: false });
  }

  async function requestVisualAnalysisRerun() {
    const dataset = detail.datasets[0] ?? null;
    if (!selectedProjectId || !dataset) return;

    setLoading(true);
    setNotice(null);
    try {
      const response = await request<Record<string, unknown>>(`/datasets/${dataset.id}/visual-analyses/run`, {
        method: "POST",
        body: { trigger_reason: "manual" },
      });
      let workerMessage = "Worker is ready to pick it up.";
      try {
        const workerProcess = await window.missionControl.ensureProjectWorker({
          projectId: selectedProjectId,
          baseUrl,
          name: `visual-analysis-worker-${selectedProjectId}`,
          gpuType: "local",
        });
        workerMessage = workerProcess.started
          ? "Started a visual-analysis worker."
          : "Visual-analysis worker is already running.";
      } catch (error) {
        workerMessage = `Queued, but worker did not start: ${errorMessage(error)}`;
      }
      await refreshProjectDetail(selectedProjectId, { includeSlowData: false });
      const responseStatus =
        recordString(response, "status") ||
        recordString(recordObject(response.job), "status") ||
        recordString(recordObject(response.analysis), "validation_status") ||
        "requested";
      setNotice({
        kind: "info",
        text: `Manual visual analysis rerun ${responseStatus.toLowerCase()} for ${dataset.name}. ${workerMessage}`,
      });
    } catch (error) {
      setNotice({
        kind: "error",
        text: isUnsupportedEndpointError(error)
          ? "Manual visual analysis rerun is not supported by this backend yet."
          : errorMessage(error),
      });
    } finally {
      setLoading(false);
    }
  }

  async function executePlan(planId: string) {
    if (!selectedProjectId) return;

    setLoading(true);
    setNotice(null);
    try {
      const plan = detail.plans.find((candidate) => candidate.id === planId);
      const workerCount = Math.max(
        1,
        Math.min(plan?.recommended_workers ?? 1, plan?.experiments.length || 1),
      );

      const response = await request<{ jobs: Job[]; worker_requirement?: WorkerRequirement }>(`/plans/${planId}/execute`, {
        method: "POST",
        body: { provider: "modal", gpu_type: "T4", max_concurrent_jobs: workerCount },
      });

      const targetCount = Math.max(1, response.worker_requirement?.target_count ?? workerCount);
      const workerPool = await window.missionControl.ensureProjectWorker({
        projectId: selectedProjectId,
        baseUrl,
        name: `modal-worker-${selectedProjectId}`,
        gpuType: "modal",
        count: targetCount,
      });

      await refreshProjectDetail(selectedProjectId, { includeSlowData: false });
      setNotice({
        kind: "info",
        text: `Plan execution ensured ${response.jobs.length} experiment jobs across ${workerPool.running_count} workers.`,
      });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function stopActiveRun() {
    if (!selectedProjectId || !stoppablePlan) return;
    const confirmed = window.confirm(`Stop active run for ${stoppablePlan.id}? Queued and active work will be cancelled.`);
    if (!confirmed) return;

    setLoading(true);
    setNotice(null);
    try {
      const response = await request<CancelExecutionResponse>(`/plans/${stoppablePlan.id}/cancel-active-execution`, {
        method: "POST",
        body: {
          reason: "user_requested",
          promote_best_available: true,
          terminate_remote_work: true,
        },
      });
      await window.missionControl.stopProjectWorker({ projectId: selectedProjectId }).catch(() => undefined);
      await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
      const cancelled = (response.queued_jobs_cancelled ?? 0) + (response.active_jobs_marked_cancelling ?? 0);
      const best = response.best_available_model?.exportable ? " Best completed model was selected." : "";
      setNotice({ kind: "info", text: `Stopped run ${stoppablePlan.id}; ${cancelled} job(s) cancelled.${best}` });
    } catch (error) {
      setNotice({ kind: "error", text: errorMessage(error) });
    } finally {
      setLoading(false);
    }
  }

  async function reviewExperiments() {
    if (!selectedProjectId) return;

    setLoading(true);
    setNotice(null);
    try {
      const decision = await request<AgentDecision>(`/projects/${selectedProjectId}/review-experiments`, {
        method: "POST",
        body: {},
      });

      await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
      setNotice({ kind: "info", text: `Reviewer decision: ${decision.decision_type}` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function scheduleFollowUpExperiments() {
    if (!selectedProjectId) return;

    setLoading(true);
    setNotice(null);
    try {
      const response = await request<ScheduleFollowUpResponse>(
        `/projects/${selectedProjectId}/schedule-follow-up-experiments`,
        {
          method: "POST",
          body: {},
        },
      );

      if (!response.follow_up_plan) {
        await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
        setNotice({ kind: "info", text: `No follow-up scheduled. Reviewer decision: ${response.decision.decision_type}` });
        return;
      }

      const plan = response.follow_up_plan;
      const workerCount = Math.max(1, Math.min(plan.recommended_workers, plan.experiments.length || 1));
      const execution = await request<{ jobs: Job[]; worker_requirement?: WorkerRequirement }>(`/plans/${plan.id}/execute`, {
        method: "POST",
        body: { provider: "modal", gpu_type: "T4", max_concurrent_jobs: workerCount },
      });

      const targetCount = Math.max(1, execution.worker_requirement?.target_count ?? workerCount);
      const workerPool = await window.missionControl.ensureProjectWorker({
        projectId: selectedProjectId,
        baseUrl,
        name: `modal-worker-${selectedProjectId}`,
        gpuType: "modal",
        count: targetCount,
      });

      await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
      setNotice({
        kind: "info",
        text: `Scheduled ${execution.jobs.length} follow-up jobs from ${plan.id} across ${workerPool.running_count} workers.`,
      });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function requestChampionExport(format = "onnx") {
    if (!selectedProjectId || !detail.champion) return;

    setLoading(true);
    setNotice(null);
    try {
      const exportRecord = await request<ChampionExport>(`/projects/${selectedProjectId}/champion/exports`, {
        method: "POST",
        body: { format },
      });
      await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true });
      setNotice({ kind: "info", text: `Champion export recorded as ${exportRecord.status || "PENDING"}.` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function chooseChampionDemoImage() {
    try {
      const image = await window.missionControl.selectDemoImage();
      if (!image) return;
      setCustomDemoImage(image);
      setCustomDemoImageURI(demoImageURI(image));
      setCustomDemoTrueLabel(demoImageLabel(image));
      setDemoPrediction(null);
      setDemoPredictionError("");
    } catch (error) {
      setDemoPredictionError(error instanceof Error ? error.message : String(error));
    }
  }

  async function runCustomChampionDemoPrediction() {
    const imageURI = customDemoImageURI.trim();
    if (!imageURI) {
      setDemoPredictionError("Choose an image or enter a worker-visible image URI.");
      return;
    }

    const customImageMatchesPicker = customDemoImage ? imageURI === demoImageURI(customDemoImage) : false;
    await runChampionDemoPrediction({
      ...(customDemoImage ?? {}),
      uri: imageURI,
      image_uri: imageURI,
      thumbnail_uri: customImageMatchesPicker ? customDemoImage?.thumbnail_uri : undefined,
      split: customDemoImage?.split || "custom",
      true_label: customDemoTrueLabel.trim() || customDemoImage?.true_label || customDemoImage?.label || customDemoImage?.class_name,
      label: customDemoTrueLabel.trim() || customDemoImage?.label,
      class_name: customDemoTrueLabel.trim() || customDemoImage?.class_name,
    });
  }

  async function ensureChampionLocalRuntime() {
    const exportRecord = readyONNXExport(championExportDemo.exports);
    if (!exportRecord) {
      throw new Error("No READY ONNX export is available for local UI inference.");
    }
    const artifactURI = exportRecord.artifact_uri || exportRecord.model_uri || exportRecord.download_url || "";
    if (!artifactURI) {
      throw new Error("The READY ONNX export does not expose an artifact URI.");
    }
    if (localRuntime.current?.artifactURI === artifactURI) {
      return localRuntime.current;
    }

    setLocalInferenceStatus("loading");
    setLocalInferenceError("");
    const artifact = await window.missionControl.loadModelArtifact({
      artifactUri: artifactURI,
      externalData: championExportExternalData(exportRecord),
    });
    const runtime = await createChampionLocalRuntime(artifact, {
      exportRecord,
      deploymentProfile: championExportDemo.deploymentProfile,
      modelProfile: championExportDemo.modelProfile,
    });
    localRuntime.current = runtime;
    setLocalInferenceStatus("ready");
    return runtime;
  }

  async function runChampionLocalPrediction(image: ChampionDemoImage) {
    const imageSource = demoImagePreviewURI(image) || demoImageURI(image);
    if (!imageSource || imageSource.startsWith("s3://")) {
      throw new Error("Local UI inference needs an image preview URI or a local uploaded image.");
    }
    const runtime = await ensureChampionLocalRuntime();
    const prediction = await predictChampionImage(runtime, image, imageSource, {
      confidenceThreshold: detectionConfidenceThreshold,
      iouThreshold: detectionIouThreshold,
      maxDetections: 100,
    });
    setDemoPrediction(attachDemoPredictionPreview(prediction, { ...image, thumbnail_uri: imageSource }));
  }

  async function runChampionDemoPrediction(image: ChampionDemoImage) {
    if (!selectedProjectId || !detail.champion) return;

    const imageURI = demoImageURI(image);
    if (!imageURI) {
      setDemoPrediction(null);
      setDemoPredictionError("Demo image has no URI to send to the backend.");
      return;
    }

    setDemoPrediction(null);
    setDemoPredictionError("");
    setDemoPredictionLoading(true);
    try {
      if (readyONNXExport(championExportDemo.exports)) {
        await runChampionLocalPrediction(image);
        return;
      }
      const response = await request<ChampionDemoPrediction | { prediction?: ChampionDemoPrediction }>(
        `/projects/${selectedProjectId}/champion/demo-predictions`,
        {
          method: "POST",
          body: {
            image_uri: imageURI,
            image_id: image.image_id || image.id || "",
            true_label: demoImageLabel(image),
            image_metadata: demoPredictionRequestMetadata(image),
            top_k: 5,
            confidence_threshold: detectionConfidenceThreshold,
            iou_threshold: detectionIouThreshold,
            max_detections: 100,
          },
        },
      );
      const normalized = attachDemoPredictionPreview(normalizeDemoPredictionResponse(response), image);
      setDemoPrediction(normalized);
      if (normalized.id && !isTerminalDemoPredictionStatus(normalized.status)) {
        await pollChampionDemoPrediction(normalized.id, image);
      } else {
        await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true }).catch(() => undefined);
      }
    } catch (error) {
      if (readyONNXExport(championExportDemo.exports)) {
        setLocalInferenceStatus("error");
        setLocalInferenceError(error instanceof Error ? error.message : String(error));
      }
      setDemoPredictionError(error instanceof Error ? error.message : String(error));
    } finally {
      setDemoPredictionLoading(false);
    }
  }

  async function pollChampionDemoPrediction(predictionId: string, image: ChampionDemoImage) {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await sleep(attempt === 0 ? 700 : 1500);
      if (!selectedProjectId) return;

      const response = await request<{
        predictions?: ChampionDemoPrediction[];
        history?: ChampionDemoPrediction[];
        demo_predictions?: ChampionDemoPrediction[];
      }>(`/projects/${selectedProjectId}/champion/demo-predictions?limit=12`);
      const predictions = response.predictions ?? response.history ?? response.demo_predictions ?? [];
      const matched = predictions.find((item) => item.id === predictionId);
      if (!matched) continue;

      const normalized = attachDemoPredictionPreview(normalizeDemoPredictionResponse(matched), image);
      setDemoPrediction(normalized);
      if (isTerminalDemoPredictionStatus(normalized.status)) break;
    }
    await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true }).catch(() => undefined);
  }

  function openChampionFeedback(rating: ChampionFeedbackRating) {
    if (!detail.champion) return;
    setChampionFeedbackDraft({ rating, message: "" });
  }

  async function submitChampionFeedback() {
    if (!selectedProjectId || !detail.champion || !championFeedbackDraft) return;
    const selectedHeldoutImage = championExportDemo.demoImages[selectedDemoImageIndex] ?? null;
    const activeImage = customDemoImage ?? selectedHeldoutImage;
    const imageURI = demoPrediction?.image_uri || (activeImage ? demoImageURI(activeImage) : customDemoImageURI.trim());
    const imageID = demoPrediction?.image_id || activeImage?.image_id || activeImage?.id || "";

    setChampionFeedbackSubmitting(true);
    setNotice(null);
    try {
      const response = await request<{ feedback?: ChampionFeedback } | ChampionFeedback>(`/projects/${selectedProjectId}/champion/feedback`, {
        method: "POST",
        body: {
          prediction_id: demoPrediction?.id || "",
          image_uri: imageURI,
          image_id: imageID,
          rating: championFeedbackDraft.rating,
          message: championFeedbackDraft.message.trim(),
          prediction_snapshot: demoPrediction ? recordObject(demoPrediction) : {},
          metrics_snapshot: championFeedbackMetricsSnapshot(detail),
          metadata: {
            ui_source: "mission_control_champion_test_bench",
            local_inference_status: localInferenceStatus,
            local_inference_error: localInferenceError,
            export_status: championExportDemo.exportStatus,
            selected_image_index: selectedDemoImageIndex,
            custom_image: Boolean(customDemoImageURI.trim()),
          },
        },
      });
      const created = normalizeChampionFeedbackResponse(response);
      if (created) {
        setDetail((previous) => ({
          ...previous,
          championFeedback: [created, ...previous.championFeedback.filter((item) => item.id !== created.id)],
        }));
      }
      setChampionFeedbackDraft(null);
      setNotice({ kind: "info", text: "Champion feedback recorded." });
      await refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true }).catch(() => undefined);
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setChampionFeedbackSubmitting(false);
    }
  }

  function updateSettingsDraft(update: AutomationSettingsUpdate) {
    setSettingsDraft((current) => ({ ...current, ...update }));
  }

  async function saveAutomationSettings() {
    setLoading(true);
    setNotice(null);
    try {
      const response = await request<AutomationSettings>("/settings/automation", {
        method: "PATCH",
        body: {
          auto_review_experiments: settingsDraft.auto_review_experiments,
          auto_schedule_followups: settingsDraft.auto_schedule_followups,
          auto_execute_plans: settingsDraft.auto_execute_plans,
          max_followup_rounds: Math.max(0, Math.trunc(settingsDraft.max_followup_rounds || 0)),
          default_training_provider: settingsDraft.default_training_provider,
          default_gpu_type: settingsDraft.default_gpu_type,
          cost_mode: settingsDraft.cost_mode,
          budget_cap_usd: Math.max(0, Number(settingsDraft.budget_cap_usd || 0)),
          llm_enabled: settingsDraft.llm_enabled,
          agent_mode: settingsDraft.agent_mode,
          llm_provider: settingsDraft.llm_provider,
          llm_model: settingsDraft.llm_model,
          automl_enabled: settingsDraft.automl_enabled,
          automl_sampler: settingsDraft.automl_sampler,
        },
      });

      setAutomationSettings(response);
      setSettingsDraft(response);
      setNotice({ kind: "info", text: "Automation settings updated." });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  function openProjectTab(tab: ProjectTabTarget, targetId?: string) {
    setActiveProjectTab(projectTabFromTarget(tab));
    if (!targetId) return;
    window.requestAnimationFrame(() => {
      document.getElementById(targetId)?.scrollIntoView({ block: "start" });
    });
  }

  async function handleMissionAction(action: MissionNextAction) {
    if (action.disabled) return;
    if (action.actionKey === "new_project") {
      setNewProjectOpen(true);
      return;
    }
    if (action.actionKey === "refresh") {
      await refreshAll();
      return;
    }
    if (action.actionKey === "execute_plan") {
      if (latestPlan) await executePlan(latestPlan.id);
      return;
    }
    if (action.actionKey === "review_experiments") {
      await reviewExperiments();
      return;
    }
    if (action.actionKey === "schedule_follow_up") {
      await scheduleFollowUpExperiments();
      return;
    }
    if (action.actionKey === "open_export") {
      openProjectTab(action.targetTab ?? "export", action.targetId ?? "export-demo");
      return;
    }
    if (action.targetTab) {
      openProjectTab(action.targetTab, action.targetId);
    }
  }

  async function handleSubmit(action: (formData: FormData) => Promise<void>, form: HTMLFormElement) {
    setLoading(true);
    setNotice(null);
    try {
      await action(new FormData(form));
      form.reset();
      setNotice({ kind: "info", text: "Saved" });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="app-chrome">
        <div className="chrome-left">
          <span className="chrome-mark">
            <BrainCircuit size={16} />
          </span>
          <span>
            <strong>Model Express</strong>
            <small>Autonomous ML engineer</small>
          </span>
        </div>
        <div className="chrome-right">
          <span>{health?.status === "ok" ? "AI engine ready" : "AI engine offline"}</span>
        </div>
      </header>

      <aside className="sidebar">
        <label className="field compact">
          <span>Engine URL</span>
          <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
        </label>

        <div className="sidebar-actions">
          <button className="command primary" onClick={() => setNewProjectOpen(true)} disabled={loading}>
            <Plus size={16} />
            New Mission
          </button>
          <button className="icon-command" onClick={refreshAll} disabled={loading} title="Refresh now">
            <RefreshCcw size={16} />
          </button>
        </div>

        <section className="nav-section">
          <div className="section-title mission-queue-title">
            <span>
              <Database size={15} />
              Missions
            </span>
            <small>{projects.length}</small>
          </div>
          <div className="project-list">
            {projects.map((project) => (
              <button
                key={project.id}
                className={project.id === selectedProjectId ? "project active" : "project"}
                aria-current={project.id === selectedProjectId ? "true" : undefined}
                onClick={() => setSelectedProjectId(project.id)}
                title={project.goal || missionStateLabelFromProject(project)}
              >
                <span className={`project-status-dot ${missionStateToneFromProject(project)}`} aria-hidden="true" />
                <span className="project-copy">
                  <strong>{project.name}</strong>
                  <small>{project.goal || "Ready for mission setup"}</small>
                </span>
                <span className={`project-state ${missionStateToneFromProject(project)}`}>{missionStateLabelFromProject(project)}</span>
              </button>
            ))}
            {projects.length === 0 && (
              <div className="project-list-empty">
                <span className="project-empty-mark" aria-hidden="true">
                  <Database size={14} />
                </span>
                <strong>No missions yet</strong>
                <small>Create a mission to start the workflow.</small>
              </div>
            )}
          </div>
        </section>
      </aside>

			<section className="workspace" data-active-tab={activeProjectTab}>
        <header className={`topbar ${activeProjectTab === "developer" ? "developer-topbar" : "mission-command-topbar"}`}>
          <div className="topbar-copy">
            <div className="topbar-kicker">
              <div className="eyebrow">{activeProjectTab === "developer" ? "Developer View" : "Mission"}</div>
              {activeProjectTab !== "developer" && <span className={`mission-state-pill ${missionDigest.state}`}>{missionDigest.stateLabel}</span>}
            </div>
            <h2>{selectedProject ? selectedProject.name : "No Mission Selected"}</h2>
            {activeProjectTab !== "developer" && <p>{selectedProject?.goal || missionDigest.detail}</p>}
          </div>
          <div className="topbar-actions">
            {activeProjectTab !== "developer" && (
              <div className={health?.status === "ok" ? "engine-chip ok" : "engine-chip bad"} aria-label={health?.status === "ok" ? "Engine ready" : "Engine offline"}>
                <span className="engine-chip-light" aria-hidden="true" />
                <Server size={15} />
                <span>
                  <strong>Engine</strong>
                  <small>{health?.status === "ok" ? "Ready" : "Offline"}</small>
                </span>
              </div>
            )}
            {stoppablePlan && (
              <button className="command compact danger" type="button" onClick={stopActiveRun} disabled={loading}>
                <X size={15} />
                Stop Run
              </button>
            )}
            <button
              className={activeProjectTab === "developer" ? "command compact back-to-mission" : "diagnostics-toggle"}
              type="button"
              aria-label={activeProjectTab === "developer" ? "Back to Mission" : "Open Developer View diagnostics"}
              onClick={() => setActiveProjectTab(activeProjectTab === "developer" ? "mission" : "developer")}
            >
              {activeProjectTab === "developer" ? (
                "Back to Mission"
              ) : (
                <>
                  <SquareTerminal size={15} />
                  <span>
                    <strong>Diagnostics</strong>
                    <small>Developer View</small>
                  </span>
                </>
              )}
            </button>
            {activeProjectTab === "developer" && (
              <div className={health?.status === "ok" ? "status ok" : "status bad"}>
                <Server size={16} />
                {health?.status === "ok" ? "ready" : "offline"}
              </div>
            )}
          </div>
        </header>

        {notice && <NoticeBanner notice={notice} />}

        <nav className="section-tabs" aria-label="Project workflow tabs" role="tablist">
          {projectWorkflowTabs.map((tab, index) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              className={`workflow-tab state-${tab.state}${activeProjectTab === tab.key ? " selected active" : ""}`}
              aria-label={`${tab.label}, step ${index + 1}, ${tab.detail}`}
              aria-selected={activeProjectTab === tab.key}
              onClick={() => setActiveProjectTab(tab.key)}
            >
              <span className="workflow-tab-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="workflow-tab-main">
                <span className="workflow-tab-label">
                  <span className="workflow-tab-icon" aria-hidden="true">
                    {tab.icon}
                  </span>
                  <span>{tab.label}</span>
                </span>
                <small>{tab.detail}</small>
              </span>
              <span className="workflow-tab-dot" aria-hidden="true" />
            </button>
          ))}
        </nav>

				<section className="mission-route" id="mission" data-project-tab="mission">
          <MissionRoute
            brief={missionBrief}
            thinking={currentThinking}
            stages={missionStages}
            activity={activityFeed}
            results={resultsSummary}
            exportSummary={exportSummary}
            actions={missionDigest.nextActions}
            onAction={handleMissionAction}
            onOpenTab={openProjectTab}
          />
        </section>

        <section className="activity-route" id="activity" data-project-tab="activity">
          <ActivityRoute
            cards={activityFeed}
            filter={activityFilter}
            onFilterChange={setActivityFilter}
            onOpenDeveloper={() => openProjectTab("developer", "developer-raw-events")}
          />
        </section>

        <section className="results-route" id="results" data-project-tab="results">
          <ResultsRoute
            summary={resultsSummary}
            onSelectCandidate={setSelectedJobId}
            onOpenExport={() => openProjectTab("export", "export-package")}
            onOpenDeveloper={() => openProjectTab("developer", "champion-comparison")}
          />
        </section>

        <section className="export-route" id="export" data-project-tab="export">
          <ExportRoute
            summary={exportSummary}
            data={championExportDemo}
            prediction={demoPrediction}
            predictionError={demoPredictionError}
            predictionLoading={demoPredictionLoading}
            selectedImageIndex={selectedDemoImageIndex}
            customImage={customDemoImage}
            customImageURI={customDemoImageURI}
            customTrueLabel={customDemoTrueLabel}
            localInferenceStatus={localInferenceStatus}
            localInferenceError={localInferenceError}
            slideshowEnabled={demoSlideshowEnabled}
            slideshowIntervalMs={demoSlideshowIntervalMs}
            detectionConfidenceThreshold={detectionConfidenceThreshold}
            detectionIouThreshold={detectionIouThreshold}
            onCustomImageURIChange={setCustomDemoImageURI}
            onCustomTrueLabelChange={setCustomDemoTrueLabel}
            onChooseCustomImage={chooseChampionDemoImage}
            onRunCustomPrediction={runCustomChampionDemoPrediction}
            onToggleSlideshow={() => setDemoSlideshowEnabled((enabled) => !enabled)}
            onSelectImage={(index) => {
              setSelectedDemoImageIndex(index);
              setCustomDemoImage(null);
              setCustomDemoImageURI("");
              setCustomDemoTrueLabel("");
              setDemoSlideshowEnabled(false);
            }}
            onNextImage={() => {
              setCustomDemoImage(null);
              setCustomDemoImageURI("");
              setCustomDemoTrueLabel("");
              setSelectedDemoImageIndex((index) => nextDemoImageIndex(index, championExportDemo.demoImages.length));
            }}
            onRandomImage={() => {
              setCustomDemoImage(null);
              setCustomDemoImageURI("");
              setCustomDemoTrueLabel("");
              setSelectedDemoImageIndex((index) => randomDemoImageIndex(index, championExportDemo.demoImages.length));
            }}
            onRequestExport={() => requestChampionExport("onnx")}
            onRunPrediction={runChampionDemoPrediction}
            onOpenFeedback={openChampionFeedback}
            onSlideshowIntervalChange={setDemoSlideshowIntervalMs}
            onDetectionConfidenceThresholdChange={setDetectionConfidenceThreshold}
            onDetectionIouThresholdChange={setDetectionIouThreshold}
            onOpenDeveloper={() => openProjectTab("developer", "export-demo")}
          />
        </section>

        <section className="developer-route" id="developer" data-project-tab="developer">
          <DeveloperRoute diagnostics={developerDiagnostics} onBack={() => setActiveProjectTab("mission")} />
        </section>

				<section className="content-grid developer-grid">
					<Panel title="Dataset Intelligence" icon={<BarChart3 size={17} />} wide id="data" tab="developer">
            {datasetIntelligence.dataset ? (
              <div className="dataset-intelligence">
                <div className="insight-grid">
                  {datasetIntelligence.insights.map((item) => (
                    <div className={`insight-card ${item.tone ?? ""}`} key={item.label}>
                      <small>{item.label}</small>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>
                {datasetIntelligence.metadataStatus && (
                  <div className="metadata-status-panel">
                    <div className="metadata-status-head">
                      <span>
                        <strong>Metadata Import</strong>
                        <small>{datasetIntelligence.metadataStatus.detail}</small>
                      </span>
                      <Badge value={datasetIntelligence.metadataStatus.status || "reported"} />
                    </div>
                    <div className="metadata-fact-grid">
                      {datasetIntelligence.metadataStatus.facts.map((item) => (
                        <div className={`insight-card ${item.tone ?? ""}`} key={item.label}>
                          <small>{item.label}</small>
                          <strong>{item.value}</strong>
                        </div>
                      ))}
                    </div>
                    {datasetIntelligence.metadataStatus.sources.length > 0 && (
                      <div className="tag-list">
                        {datasetIntelligence.metadataStatus.sources.map((source, index) => (
                          <small key={`${source}-${index}`}>{source}</small>
                        ))}
                      </div>
                    )}
                    {(datasetIntelligence.metadataStatus.splitRows.length > 0 ||
                      datasetIntelligence.metadataStatus.annotationRows.length > 0) && (
                      <div className="metadata-count-grid">
                        {datasetIntelligence.metadataStatus.splitRows.length > 0 && (
                          <div className="metadata-count-block">
                            <strong>Splits</strong>
                            {datasetIntelligence.metadataStatus.splitRows.map((row) => (
                              <span key={row.label}>
                                <small>{row.label}</small>
                                <b>{row.value}</b>
                              </span>
                            ))}
                          </div>
                        )}
                        {datasetIntelligence.metadataStatus.annotationRows.length > 0 && (
                          <div className="metadata-count-block">
                            <strong>Annotations</strong>
                            {datasetIntelligence.metadataStatus.annotationRows.map((row) => (
                              <span key={row.label}>
                                <small>{row.label}</small>
                                <b>{row.value}</b>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    {datasetIntelligence.metadataStatus.warnings.length > 0 && (
                      <div className="warning-list">
                        {datasetIntelligence.metadataStatus.warnings.map((warning, index) => (
                          <span key={`${warning}-${index}`}>
                            <AlertTriangle size={14} />
                            {warning}
                          </span>
                        ))}
                      </div>
                    )}
                    {datasetIntelligence.metadataStatus.errors.length > 0 && (
                      <div className="metadata-error-list">
                        {datasetIntelligence.metadataStatus.errors.map((error, index) => (
                          <span key={`${error}-${index}`}>
                            <AlertTriangle size={14} />
                            {error}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="dataset-intelligence-grid">
                  <div className="class-distribution">
                    <strong>Class Distribution</strong>
                    {datasetIntelligence.classRows.length > 0 ? (
                      datasetIntelligence.classRows.map((row) => (
                        <div className="class-bar-row" key={row.name}>
                          <span>{row.name}</span>
                          <div>
                            <i style={{ width: `${row.percent}%` }} />
                          </div>
                          <small>{row.count}</small>
                        </div>
                      ))
                    ) : (
                      <div className="empty compact">Class counts will appear after profiling.</div>
                    )}
                  </div>
                  <div className="recommendation-stack">
                    <strong>Metrics</strong>
                    <div className="tag-list">
                      {datasetIntelligence.metrics.map((metric) => (
                        <small key={metric}>{metric}</small>
                      ))}
                    </div>
                    <strong>Preprocessing</strong>
                    <div className="recommendation-list">
                      {datasetIntelligence.preprocessing.map((item) => (
                        <span key={item}>{item}</span>
                      ))}
                    </div>
                    <strong>Artifacts</strong>
                    <div className="recommendation-list">
                      {datasetIntelligence.artifacts.map((item) => (
                        <span key={item}>{item}</span>
                      ))}
                    </div>
                  </div>
                </div>
                {datasetIntelligence.warnings.length > 0 && (
                  <div className="warning-list">
                    {datasetIntelligence.warnings.map((warning) => (
                      <span key={warning}>
                        <AlertTriangle size={14} />
                        {warning}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty">Upload a dataset to see class balance, image sizes, artifacts, and metric recommendations.</div>
            )}
          </Panel>

					<Panel title="Visual Dataset Analysis" icon={<Eye size={17} />} wide tab="developer">
            <VisualAnalysisPanel
              dataset={detail.datasets[0] ?? null}
              jobs={detail.jobs}
              loading={loading}
              visualAnalysis={detail.visualAnalysis}
              onRequestRerun={requestVisualAnalysisRerun}
            />
          </Panel>
        </section>

        <section className="content-grid">
					<Panel title="Experiment Timeline" icon={<ListRestart size={17} />} wide id="experiment-timeline" tab="developer">
            <div className="timeline">
              {timelineItems.map((item) => (
                <div className={`timeline-item ${item.status}`} key={item.label}>
                  <span className="timeline-dot" />
                  <div>
                    <strong>{item.label}</strong>
                    <small>{item.detail}</small>
                    {item.timestamp && <small>{new Date(item.timestamp).toLocaleString()}</small>}
                  </div>
                  <Badge value={item.status} />
                </div>
              ))}
            </div>
          </Panel>

					<Panel title="Automation Settings" icon={<SlidersHorizontal size={17} />} wide id="operations" tab="developer">
            <div className="settings-panel">
              <div className="settings-grid">
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.auto_review_experiments}
                    onChange={(event) => updateSettingsDraft({ auto_review_experiments: event.currentTarget.checked })}
                  />
                  <span>
                    <strong>Auto Review</strong>
                    <small>{automationSettings.auto_review_experiments ? "on" : "off"}</small>
                  </span>
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.auto_schedule_followups}
                    onChange={(event) => updateSettingsDraft({ auto_schedule_followups: event.currentTarget.checked })}
                  />
                  <span>
                    <strong>Auto Follow-ups</strong>
                    <small>{automationSettings.auto_schedule_followups ? "on" : "off"}</small>
                  </span>
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.auto_execute_plans}
                    onChange={(event) => updateSettingsDraft({ auto_execute_plans: event.currentTarget.checked })}
                  />
                  <span>
                    <strong>Auto Execute</strong>
                    <small>{automationSettings.auto_execute_plans ? "on" : "off"}</small>
                  </span>
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.llm_enabled}
                    onChange={(event) => updateSettingsDraft({ llm_enabled: event.currentTarget.checked })}
                  />
                  <span>
                    <strong>LLM Agents</strong>
                    <small>{automationSettings.llm_enabled ? "on" : "off"}</small>
                  </span>
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.automl_enabled}
                    onChange={(event) => updateSettingsDraft({ automl_enabled: event.currentTarget.checked })}
                  />
                  <span>
                    <strong>AutoML HPO</strong>
                    <small>{automationSettings.automl_enabled ? "on" : "off"}</small>
                  </span>
                </label>
                <label className="setting-field">
                  <span>Agent Mode</span>
                  <select
                    value={settingsDraft.agent_mode}
                    onChange={(event) => updateSettingsDraft({ agent_mode: event.currentTarget.value })}
                  >
                    <option value="propose">propose</option>
                    <option value="autonomous">autonomous</option>
                  </select>
                </label>
                <label className="setting-field">
                  <span>AutoML Sampler</span>
                  <select
                    value={settingsDraft.automl_sampler}
                    onChange={(event) => updateSettingsDraft({ automl_sampler: event.currentTarget.value })}
                  >
                    <option value="seeded_random">seeded_random</option>
                    <option value="grid">grid</option>
                    <option value="adaptive_bayesian">adaptive_bayesian</option>
                  </select>
                </label>
                <label className="setting-field">
                  <span>Follow-up Rounds</span>
                  <input
                    type="number"
                    min="0"
                    step="1"
                    value={settingsDraft.max_followup_rounds}
                    onChange={(event) =>
                      updateSettingsDraft({ max_followup_rounds: Number.parseInt(event.currentTarget.value, 10) || 0 })
                    }
                  />
                </label>
                <label className="setting-field">
                  <span>Cost Mode</span>
                  <select
                    value={settingsDraft.cost_mode}
                    onChange={(event) => updateSettingsDraft({ cost_mode: event.currentTarget.value })}
                  >
                    <option value="prototype">prototype/cheap</option>
                    <option value="balanced">balanced</option>
                    <option value="quality">quality</option>
                  </select>
                </label>
                <label className="setting-field">
                  <span>Budget Cap</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    value={settingsDraft.budget_cap_usd}
                    onChange={(event) => updateSettingsDraft({ budget_cap_usd: Number.parseFloat(event.currentTarget.value) || 0 })}
                  />
                </label>
                <label className="setting-field">
                  <span>Provider</span>
                  <select
                    value={settingsDraft.default_training_provider}
                    onChange={(event) => updateSettingsDraft({ default_training_provider: event.currentTarget.value })}
                  >
                    <option value="local">local</option>
                    <option value="modal">modal</option>
                    <option value="persistent_gpu">persistent_gpu</option>
                  </select>
                </label>
                <label className="setting-field">
                  <span>LLM Provider</span>
                  <select
                    value={settingsDraft.llm_provider}
                    onChange={(event) => updateSettingsDraft({ llm_provider: event.currentTarget.value })}
                  >
                    <option value="openai">openai</option>
                    <option value="openai_compatible">openai_compatible</option>
                    <option value="local">local</option>
                  </select>
                </label>
                <label className="setting-field">
                  <span>LLM Model</span>
                  <input
                    value={settingsDraft.llm_model}
                    placeholder="model id from .env"
                    onChange={(event) => updateSettingsDraft({ llm_model: event.currentTarget.value })}
                  />
                </label>
                <label className="setting-field">
                  <span>GPU</span>
                  <input
                    value={settingsDraft.default_gpu_type}
                    placeholder="T4"
                    onChange={(event) => updateSettingsDraft({ default_gpu_type: event.currentTarget.value })}
                  />
                </label>
              </div>
              <div className={`review-state ${reviewState.tone}`}>
                <Badge value={reviewState.badge} />
                <span>
                  <strong>{reviewState.title}</strong>
                  <small>{reviewState.detail}</small>
                </span>
              </div>
              <div className="settings-actions">
                <small>
                  Updated {automationSettings.updated_at ? new Date(automationSettings.updated_at).toLocaleString() : "from defaults"}
                </small>
                <button className="command primary" type="button" onClick={saveAutomationSettings} disabled={loading}>
                  <CheckCircle2 size={16} />
                  Apply Settings
                </button>
              </div>
            </div>
          </Panel>

          {detail.champion && (
						<Panel title="Champion Details" icon={<Trophy size={17} />} wide id="champion-detail" tab="developer">
              <div className="champion-panel">
                <div className="champion-head">
                  <span>
                    <strong>{recordString(detail.champion.metrics, "model") || detail.champion.job_id}</strong>
                    <small>
                      {detail.champion.job_id} {detail.champion.plan_id ? `- ${detail.champion.plan_id}` : ""}
                    </small>
                  </span>
                  <Badge value="SELECTED" />
                </div>
                <div className="champion-grid">
                  <div>
                    <small>{selectedChampionPrimaryMetric?.label ?? "Macro-F1"}</small>
                    <strong>{formatMaybeMetric(selectedChampionPrimaryMetric?.value ?? 0)}</strong>
                  </div>
                  <div>
                    <small>{selectedChampionPrimaryMetric?.secondaryLabel ?? "Accuracy"}</small>
                    <strong>{formatMaybeMetric(selectedChampionPrimaryMetric?.secondaryValue ?? 0)}</strong>
                  </div>
                  <div>
                    <small>Cost</small>
                    <strong>{formatCurrency(recordNumber(detail.champion.metrics, "estimated_cost_usd"))}</strong>
                  </div>
                  <div>
                    <small>Runtime</small>
                    <strong>{formatSeconds(recordNumber(detail.champion.metrics, "runtime_seconds"))}</strong>
                  </div>
                  <div>
                    <small>Latency</small>
                    <strong>{formatLatency(championModelProfile(detail.champion).estimated_latency_ms)}</strong>
                  </div>
                  <div>
                    <small>Model Size</small>
                    <strong>{formatModelSize(championModelProfile(detail.champion))}</strong>
                  </div>
                </div>
                <p>{detail.champion.selection_reason || "Champion selected by the agentic review loop."}</p>
                {championConfusionMatrix(detail.champion).length > 0 ? (
                  <div className="confusion-preview">
                    {championConfusionMatrix(detail.champion).slice(0, 6).map((row, rowIndex) => (
                      <div key={`${detail.champion?.id}-row-${rowIndex}`}>
                        {row.slice(0, 6).map((value, columnIndex) => (
                          <span key={`${rowIndex}-${columnIndex}`}>{value}</span>
                        ))}
                      </div>
                    ))}
                  </div>
                ) : (
                  <small className="diagnostic-note">Detailed diagnostics will appear after the worker reports evaluation data.</small>
                )}
              </div>
            </Panel>
          )}

					<Panel title="Champion Export / Demo" icon={<Trophy size={17} />} wide id="export-demo" tab="developer">
            <ChampionExportDemoPanel
              data={championExportDemo}
              prediction={demoPrediction}
              predictionError={demoPredictionError}
              predictionLoading={demoPredictionLoading}
              selectedImageIndex={selectedDemoImageIndex}
              customImage={customDemoImage}
              customImageURI={customDemoImageURI}
              customTrueLabel={customDemoTrueLabel}
              localInferenceStatus={localInferenceStatus}
              localInferenceError={localInferenceError}
              slideshowEnabled={demoSlideshowEnabled}
              slideshowIntervalMs={demoSlideshowIntervalMs}
              detectionConfidenceThreshold={detectionConfidenceThreshold}
              detectionIouThreshold={detectionIouThreshold}
              onCustomImageURIChange={setCustomDemoImageURI}
              onCustomTrueLabelChange={setCustomDemoTrueLabel}
              onChooseCustomImage={chooseChampionDemoImage}
              onRunCustomPrediction={runCustomChampionDemoPrediction}
              onToggleSlideshow={() => setDemoSlideshowEnabled((enabled) => !enabled)}
              onSelectImage={(index) => {
                setSelectedDemoImageIndex(index);
                setCustomDemoImage(null);
                setCustomDemoImageURI("");
                setCustomDemoTrueLabel("");
                setDemoSlideshowEnabled(false);
              }}
              onNextImage={() => {
                setCustomDemoImage(null);
                setCustomDemoImageURI("");
                setCustomDemoTrueLabel("");
                setSelectedDemoImageIndex((index) => nextDemoImageIndex(index, championExportDemo.demoImages.length));
              }}
              onRandomImage={() => {
                setCustomDemoImage(null);
                setCustomDemoImageURI("");
                setCustomDemoTrueLabel("");
                setSelectedDemoImageIndex((index) => randomDemoImageIndex(index, championExportDemo.demoImages.length));
              }}
              onRequestExport={() => requestChampionExport("onnx")}
              onRunPrediction={runChampionDemoPrediction}
              onOpenFeedback={openChampionFeedback}
              onSlideshowIntervalChange={setDemoSlideshowIntervalMs}
              onDetectionConfidenceThresholdChange={setDetectionConfidenceThreshold}
              onDetectionIouThresholdChange={setDetectionIouThreshold}
            />
          </Panel>

					<Panel title="Training Run Summary" icon={<Trophy size={17} />} wide id="runs" tab="developer">
            <div className="run-summary">
              <div className="run-overview">
                <div>
                  <span><DollarSign size={15} /> Estimated Spend</span>
                  <strong>{formatCurrency(runTotals.totalCost)}</strong>
                </div>
                <div>
                  <span><Trophy size={15} /> Best {runTotals.bestPrimaryMetricLabel}</span>
                  <strong>{formatMaybeMetric(runTotals.bestPrimaryMetricValue)}</strong>
                </div>
                <div>
                  <span><Timer size={15} /> GPU Runtime</span>
                  <strong>{formatSeconds(runTotals.totalRuntimeSeconds)}</strong>
                </div>
                <div>
                  <span><Activity size={15} /> Active Runs</span>
                  <strong>{runTotals.activeRuns}</strong>
                </div>
              </div>

              {detail.runSummaries.length > 0 ? (
                <div className="run-table">
                  <div className="run-table-row run-table-head">
                    <span>Model</span>
                    <span>Status</span>
                    <span>Best Metric</span>
                    <span>Cost</span>
                    <span>Runtime</span>
                    <span>Epochs</span>
                  </div>
                  {detail.runSummaries.map((summary) => {
                    const evaluation = detail.runEvaluations.find((item) => item.job_id === summary.job_id) ?? null;
                    const job = detail.jobs.find((item) => item.id === summary.job_id) ?? null;
                    const primaryMetric = runPrimaryMetric(summary, evaluation, job);
                    return (
                      <button
                        className={summary.job_id === selectedJobId ? "run-table-row run-row active" : "run-table-row run-row"}
                        key={summary.job_id}
                        onClick={() => setSelectedJobId(summary.job_id)}
                      >
                        <span>
                          <strong>{summary.model || "unknown"}</strong>
                          <small>{summary.job_id}</small>
                          {trainingRunCacheSummary(summary) && <small>{trainingRunCacheSummary(summary)}</small>}
                        </span>
                        <Badge value={summary.status || "UNKNOWN"} />
                        <strong title={primaryMetric.label}>{formatMaybeMetric(primaryMetric.value)}</strong>
                        <strong>{formatCurrency(summary.estimated_cost_usd)}</strong>
                        <span>{formatSeconds(summary.runtime_seconds)}</span>
                        <span>{summary.epochs_completed}</span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="empty">Training summaries will appear as soon as experiment jobs report their first epoch.</div>
              )}
            </div>
          </Panel>

					<Panel title="Champion Comparison" icon={<Trophy size={17} />} wide id="champion-comparison" tab="developer">
            {championComparison.length > 0 ? (
              <div className="comparison-table">
                <div className="comparison-row comparison-head">
                  <span>Model</span>
                  <span>Rank</span>
                  <span>Primary</span>
                  <span>Secondary</span>
                  <span>Gap</span>
                  <span>Seed Var</span>
                  <span>Runtime</span>
                  <span>Cost</span>
                  <span>Latency</span>
                  <span>Fit</span>
                </div>
                {championComparison.map((row) => (
                  <button
                    className={row.isChampion ? "comparison-row champion" : "comparison-row"}
                    key={row.jobId}
                    onClick={() => setSelectedJobId(row.jobId)}
                  >
                    <span>
                      <strong>{row.model || "unknown"}</strong>
                      <small>{row.isChampion ? "selected champion" : row.jobId}</small>
                    </span>
                    <strong>{formatMaybeMetric(row.rankScore)}</strong>
                    <strong title={row.primaryMetricLabel}>{formatMaybeMetric(row.primaryMetricValue)}</strong>
                    <span title={row.secondaryMetricLabel}>{formatMaybeMetric(row.secondaryMetricValue)}</span>
                    <span title={row.divergenceStatus || undefined}>{formatLossGap(row.trainValidationGap)}</span>
                    <span>{formatSeedVariance(row.seedVariance, row.seedRunCount)}</span>
                    <span>{formatSeconds(row.runtimeSeconds)}</span>
                    <span>{formatCurrency(row.costUsd)}</span>
                    <span>{formatLatency(row.latencyMs)}</span>
                    <span>{formatMaybeMetric(row.objectiveFit)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty">Completed run comparisons will appear once training summaries are reported.</div>
            )}
          </Panel>

					<Panel title="Live Agent Activity" icon={<Activity size={17} />} wide id="agent-activity" tab="developer">
            <AgentActivityPanel events={visibleActivityEvents} streamState={activityStreamState} detail={detail} />
          </Panel>

					<Panel title="Agent Decisions" icon={<BrainCircuit size={17} />} wide id="agent-decisions" tab="developer">
            <div className="decision-panel">
              <div className="decision-actions">
                <div>
                  <strong>Experiment Reviewer</strong>
                  <small>Compares finished runs and records the next project decision.</small>
                </div>
                <div className="decision-buttons">
                  {latestDecision?.decision_type === "ADD_EXPERIMENTS" && !latestDecisionHasFollowUpPlan && (
                    <button className="command primary" onClick={scheduleFollowUpExperiments} disabled={!selectedProjectId || loading}>
                      <Play size={16} />
                      Schedule Follow-up
                    </button>
                  )}
                  <button className="command" onClick={reviewExperiments} disabled={!selectedProjectId || loading}>
                    <BrainCircuit size={16} />
                    Review Experiments
                  </button>
                </div>
              </div>

              {decisionChatTurns.length > 0 ? (
                <AgentDecisionChat turns={decisionChatTurns} />
              ) : (
                <div className="empty">No agent decisions yet. Run the reviewer after experiments finish.</div>
              )}
            </div>
          </Panel>

					<Panel title="Decision Quality" icon={<BarChart3 size={17} />} wide tab="developer">
            <DecisionQualityPanel decisions={detail.decisions} invocations={detail.agentInvocations} />
          </Panel>

					<Panel title="Mission Control Telemetry" icon={<Activity size={17} />} wide tab="developer">
            <MissionControlTelemetryPanel telemetry={detail.telemetry} fallbackInvocations={detail.agentInvocations} />
          </Panel>

					<Panel title="Automation Timeline" icon={<ListRestart size={17} />} wide id="automation-timeline" tab="developer">
            <div className="automation-grid">
              <div className="automation-block">
                <strong>Worker Requirements</strong>
                {detail.workerRequirements.length > 0 ? (
                  <div className="status-list">
                    {detail.workerRequirements.slice(0, 4).map((requirement) => (
                      <div className="status-row" key={requirement.id}>
                        <span>
                          <strong>{requirement.plan_id || "no plan"}</strong>
                          <small>
                            {requirement.target_count} worker(s) - {requirement.provider}
                            {requirement.gpu_type ? `/${requirement.gpu_type}` : ""}
                            {workerRequirementMaterializationSummary(requirement) ? ` - ${workerRequirementMaterializationSummary(requirement)}` : ""}
                          </small>
                        </span>
                        <Badge value={requirement.status} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty compact">No worker requirements yet.</div>
                )}
              </div>
              <div className="automation-block">
                <strong>Execution Events</strong>
                {detail.executionEvents.length > 0 ? (
                  <div className="event-list">
                    {detail.executionEvents.map((event) => (
                      <div className="event-row" key={event.id}>
                        <Badge value={event.event_type} />
                        <span>
                          <strong>{event.message}</strong>
                          <small>{new Date(event.created_at).toLocaleString()}</small>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty compact">No automation events yet.</div>
                )}
              </div>
            </div>
          </Panel>

					<Panel title="Agent Invocation Audit" icon={<SquareTerminal size={17} />} wide tab="developer">
            <AgentInvocationAuditPanel invocations={detail.agentInvocations} decisions={detail.decisions} />
          </Panel>

					<Panel title="Memory Retrieval Probe" icon={<BrainCircuit size={17} />} wide id="memory-retrieval-probe" tab="developer">
            <MemoryRetrievalProbePanel snapshots={memoryRetrievalProbe} />
          </Panel>

					<Panel title="Agent Memory" icon={<BrainCircuit size={17} />} wide id="agent-memory" tab="developer">
            {detail.agentMemory.length > 0 ? (
              <div className="memory-list">
                {detail.agentMemory.map((record) => (
                  <div className="memory-row" key={record.id}>
                    <span>
                      <strong>{record.agent_name}</strong>
                      <small>
                        {record.kind}
                        {record.invocation_id ? ` - ${record.invocation_id}` : ""}
                      </small>
                    </span>
                    <p>{record.summary}</p>
                    <div className="tag-list">
                      {record.tags.slice(0, 5).map((tag) => (
                        <small key={`${record.id}-${tag}`}>{tag}</small>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty">LLM agent recommendations will appear after completed runs.</div>
            )}
          </Panel>

					<Panel title="Experiment Plan" icon={<ClipboardList size={17} />} wide id="plans" tab="developer">
            {latestPlan ? (
              <div className="plan-card">
                <div className="plan-actions">
                  <span>
                    <strong>{latestPlan.id}</strong>
                    <small>{new Date(latestPlan.created_at).toLocaleString()}</small>
                  </span>
                  <button className="command" onClick={() => executePlan(latestPlan.id)} disabled={loading}>
                    <Play size={16} />
                    Execute Plan
                  </button>
                  {stoppablePlan?.id === latestPlan.id && (
                    <button className="command danger" type="button" onClick={stopActiveRun} disabled={loading}>
                      <X size={16} />
                      Stop Run
                    </button>
                  )}
                </div>
                <div className="plan-overview">
                  <div>
                    <small>Status</small>
                    <Badge value={latestPlan.status} />
                  </div>
                  <div>
                    <small>Target Metric</small>
                    <strong>{latestPlan.target_metric}</strong>
                  </div>
                  <div>
                    <small>Workers</small>
                    <strong>{latestPlan.recommended_workers}</strong>
                  </div>
                  <div>
                    <small>Estimate</small>
                    <strong>{latestPlan.estimated_minutes}m</strong>
                  </div>
                </div>
                {!automationSettings.auto_execute_plans && (
                  <div className="review-state plan-review-state review">
                    <Badge value="REVIEW" />
                    <span>
                      <strong>Auto execution disabled</strong>
                      <small>Proposed experiments stay visible here for review until a manual execute action or backend setting runs them.</small>
                    </span>
                  </div>
                )}
                <div className="experiment-list">
                  {latestPlan.experiments.map((experiment, index) => (
                    <div className="experiment-item" key={`${latestPlan.id}-${index}-${experiment.template}-${experiment.model}`}>
                      <span>
                        <strong>{experiment.model || experiment.mechanism || experiment.template}</strong>
                        <small>{[experiment.template, experiment.mechanism].filter(Boolean).join(" - ")}</small>
                      </span>
                      {experiment.template !== "label_quality_audit" && (
                        <>
                          <span>
                            <small>{experiment.epochs} epochs</small>
                            <small>batch {experiment.batch_size}</small>
                          </span>
                          <span>
                            <small>lr</small>
                            <strong>{experiment.learning_rate}</strong>
                          </span>
                        </>
                      )}
                      {(experiment.image_size ||
                        experiment.optimizer ||
                        experiment.scheduler ||
                        experiment.class_balancing ||
                        experiment.dropout ||
                        experiment.label_smoothing ||
                        experiment.gradient_clip_norm) && (
                        <span>
                          {experiment.image_size ? <small>{experiment.image_size}px</small> : null}
                          {experiment.optimizer ? <small>{experiment.optimizer}</small> : null}
                          {experiment.scheduler ? <small>{experiment.scheduler}</small> : null}
                          {experiment.dropout ? <small>dropout {formatMetricNumber(experiment.dropout)}</small> : null}
                          {experiment.label_smoothing ? <small>smooth {formatMetricNumber(experiment.label_smoothing)}</small> : null}
                          {experiment.gradient_clip_norm ? <small>clip {formatMetricNumber(experiment.gradient_clip_norm)}</small> : null}
                          {experiment.class_balancing ? <small>{experiment.class_balancing}</small> : null}
                        </span>
                      )}
                      <p>{experiment.reason}</p>
                      {experimentMechanismItems(experiment).length > 0 && (
                        <div className="experiment-mechanism">
                          {experimentMechanismItems(experiment).map((item) => (
                            <span key={`${latestPlan.id}-${index}-${item.label}`}>
                              <small>{item.label}</small>
                              <strong>{item.value}</strong>
                            </span>
                          ))}
                        </div>
                      )}
                      {experimentPreprocessingItems(experiment).length > 0 && (
                        <div className="experiment-preprocessing">
                          {experimentPreprocessingItems(experiment).map((item) => (
                            <span key={`${latestPlan.id}-${experiment.model}-${item.label}`}>
                              <small>{item.label}</small>
                              <strong>{item.value}</strong>
                            </span>
                          ))}
                        </div>
                      )}
                      {experimentAutoMLItems(experiment).length > 0 && (
                        <div className="experiment-automl">
                          {experimentAutoMLItems(experiment).map((item) => (
                            <span key={`${latestPlan.id}-${index}-automl-${item.label}`}>
                              <small>{item.label}</small>
                              <strong>{item.value}</strong>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                {latestPlan.warnings.length > 0 && (
                  <div className="warning-list">
                    {latestPlan.warnings.map((warning) => (
                      <span key={warning}>{warning}</span>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty">
                {detail.datasets.some((dataset) => dataset.status === "PROFILED")
                  ? "No experiment plan has been proposed yet."
                  : "Waiting for the dataset profiling job to finish."}
              </div>
            )}
          </Panel>

					<Panel title="Manual Job Queue" icon={<Play size={17} />} wide id="manual-job-queue" tab="developer">
            <form
              className="job-create-grid"
              onSubmit={(event) => {
                event.preventDefault();
                handleSubmit(createJob, event.currentTarget);
              }}
            >
              <select name="template" defaultValue="profile_dataset">
                <option value="profile_dataset">profile_dataset</option>
                <option value="mobilenet_transfer">mobilenet_transfer</option>
                <option value="simple_cnn">simple_cnn</option>
                <option value="resnet_transfer">resnet_transfer</option>
              </select>
              <textarea
                key={`${selectedProjectId}-${firstDatasetId}`}
                name="config"
                rows={4}
                defaultValue={JSON.stringify({ dataset_id: firstDatasetId || "dataset_id_here" }, null, 2)}
              />
              <button className="command" disabled={!selectedProjectId || detail.datasets.length === 0 || loading}>
                <Play size={16} />
                Queue
              </button>
            </form>
          </Panel>

					<Panel title="Workers" icon={<MonitorDot size={17} />} wide id="workers" tab="developer">
            <div className="table">
              <div className="table-row table-head">
                <span>Name</span>
                <span>Status</span>
                <span>GPU</span>
                <span>Job</span>
              </div>
              {detail.workers.map((worker) => (
                <div className="table-row" key={worker.id}>
                  <span>
                    <strong>{worker.name}</strong>
                    <small>{worker.id}</small>
                  </span>
                  <Badge value={worker.status} />
                  <span>{worker.gpu_type || "unknown"}</span>
                  <span>{worker.current_job_id || "-"}</span>
                </div>
              ))}
            </div>
          </Panel>

					<Panel title="Datasets" icon={<Database size={17} />} wide tab="developer">
            <div className="table">
              <div className="table-row table-head">
                <span>Name</span>
                <span>Status</span>
                <span>Size</span>
                <span>URI</span>
              </div>
              {detail.datasets.map((dataset) => (
                <div className="table-row dataset-row" key={dataset.id}>
                  <span>
                    <strong>{dataset.name}</strong>
                    <small>{dataset.id}</small>
                  </span>
                  <Badge value={dataset.status} />
                  <span>{formatBytes(dataset.size_bytes)}</span>
                  <span className="mono">{dataset.storage_uri}</span>
                </div>
              ))}
            </div>
          </Panel>

					<Panel title="Recent Jobs" icon={<SquareTerminal size={17} />} wide id="recent-jobs" tab="developer">
            <div className="job-panel-head">
              <span>
                Showing {visibleJobs.length} of {detail.jobs.length}
              </span>
              <div className="pager">
                <button
                  className="icon-command"
                  onClick={() => setJobPage((page) => Math.max(0, page - 1))}
                  disabled={jobPage === 0}
                >
                  Prev
                </button>
                <small>
                  {jobPage + 1} / {jobPageCount}
                </small>
                <button
                  className="icon-command"
                  onClick={() => setJobPage((page) => Math.min(jobPageCount - 1, page + 1))}
                  disabled={jobPage >= jobPageCount - 1}
                >
                  Next
                </button>
              </div>
            </div>
            <div className="job-list paged">
              {visibleJobs.map((job) => (
                <button
                  key={job.id}
                  className={job.id === selectedJobId ? "job active" : "job"}
                  onClick={() => setSelectedJobId(job.id)}
                >
                  <span>
                    <strong>{job.template}</strong>
                    <small>{[job.id, jobMechanismSummary(job)].filter(Boolean).join(" - ")}</small>
                  </span>
                  <span>
                    <small>{job.worker_id || "unassigned"}</small>
                    <small>{new Date(job.created_at).toLocaleTimeString()}</small>
                  </span>
                  <Badge value={job.status} />
                </button>
              ))}
            </div>
          </Panel>

					<Panel title="Run Metrics" icon={<Activity size={17} />} wide id="run-metrics" tab="developer">
            {selectedJob ? (
              <div className="metric-area">
                <div className="selected-job">
                  <span>
                    <strong>{selectedJob.template}</strong>
                    <small>{[selectedJob.id, jobMechanismSummary(selectedJob)].filter(Boolean).join(" - ")}</small>
                  </span>
                  <Badge value={selectedJob.status} />
                </div>
                <div className="metric-toolbar">
                  <div className="metric-tabs">
                    {selectedMetricOptions.map((metric) => (
                      <button
                        key={metric.key}
                        className={selectedMetricKey === metric.key ? "metric-tab active" : "metric-tab"}
                        onClick={() => setSelectedMetricKey(metric.key)}
                      >
                        {metric.label}
                      </button>
                    ))}
                  </div>
                  {selectedRunSummary && (
                    <div className="metric-inline-stats">
                      <span>{formatCurrency(selectedRunSummary.estimated_cost_usd)}</span>
                      <span>{formatSeconds(selectedRunSummary.runtime_seconds)}</span>
                      <span>{selectedRunSummary.epochs_completed} epochs</span>
                      {trainingRunLifecycleChips(selectedRunSummary).map((chip) => (
                        <span key={chip}>{chip}</span>
                      ))}
                      {selectedRunEvaluation && (
                        <span>{formatLatency(recordNumber(selectedRunEvaluation.model_profile, "estimated_latency_ms"))}</span>
                      )}
                    </div>
                  )}
                </div>
                {selectedMetricOptions.length > 0 ? (
                  <MetricChart
                    metrics={metrics}
                    metricKey={selectedMetricKey}
                    label={selectedMetricOptions.find((metric) => metric.key === selectedMetricKey)?.label ?? metricLabel(selectedMetricKey)}
                  />
                ) : (
                  <div className="empty chart-empty">No graphable metrics reported</div>
                )}
                {selectedRunEvaluation && <RunEvaluationDetails evaluation={selectedRunEvaluation} />}
              </div>
            ) : (
              <div className="empty">No job selected</div>
            )}
          </Panel>
        </section>
      </section>

      {newProjectOpen && (
        <div className="modal-backdrop">
          <section className="modal">
            <header>
              <div>
                <div className="eyebrow">New Project</div>
                <h3>Project Dataset Setup</h3>
              </div>
              <button className="icon-command" onClick={() => setNewProjectOpen(false)} disabled={loading}>
                <X size={16} />
              </button>
            </header>
            <form
              className="stack"
              onSubmit={(event) => {
                event.preventDefault();
                createProjectWithDataset(new FormData(event.currentTarget));
              }}
            >
              <div className="new-mission-fields">
                <label>
                  <span>Mission name</span>
                  <input name="name" placeholder="Surface defect classifier" required />
                </label>
                <label>
                  <span>Goal</span>
                  <input name="goal" placeholder="Optimize accuracy, cost, latency, or deployment constraints" />
                </label>
              </div>
              <div className="new-mission-dataset">
                <div className="mission-section-head">
                  <div>
                    <small>Dataset source</small>
                    <strong>Attach an image folder</strong>
                  </div>
                  <Badge value={newProjectFolder ? "selected" : "required"} />
                </div>
                <button className="dataset-picker" type="button" onClick={chooseNewProjectFolder} disabled={loading}>
                  <FolderOpen size={18} />
                  <span>
                    <strong>{newProjectFolder ? newProjectFolder.name : "Choose Folder & Upload"}</strong>
                    <small>{newProjectFolder ? newProjectFolder.path : "Required image dataset folder"}</small>
                  </span>
                  {newProjectFolder && <CheckCircle2 size={18} />}
                </button>
              </div>
              {newProjectFolder?.preflight && (
                <div className={newProjectFolder.preflight.warnings.length > 0 ? "notice-inline warning preflight-card" : "notice-inline preflight-card"}>
                  <div>
                    <strong>{newProjectFolder.preflight.warnings.length > 0 ? "Preflight warning" : "Preflight ready"}</strong>
                    <small>{newProjectFolder.preflight.warnings[0]?.message || "Folder scan completed before upload packaging."}</small>
                  </div>
                  <span>
                    <strong>{newProjectFolder.preflight.file_count.toLocaleString()}</strong>
                    <small>files</small>
                  </span>
                  <span>
                    <strong>{formatBytes(newProjectFolder.preflight.uncompressed_size_bytes)}</strong>
                    <small>before ZIP</small>
                  </span>
                </div>
              )}
              <button className="command primary" disabled={!newProjectFolder || loading}>
                <HardDriveUpload size={16} />
                Create Project
              </button>
            </form>
          </section>
        </div>
      )}
      {championFeedbackDraft && (
        <div className="modal-backdrop">
          <section className="modal">
            <header>
              <div>
                <div className="eyebrow">Champion Feedback</div>
                <h3>{feedbackRatingLabel(championFeedbackDraft.rating)} Output</h3>
              </div>
              <button className="icon-command" onClick={() => setChampionFeedbackDraft(null)} disabled={championFeedbackSubmitting}>
                <X size={16} />
              </button>
            </header>
            <form
              className="stack"
              onSubmit={(event) => {
                event.preventDefault();
                submitChampionFeedback();
              }}
            >
              <label className="field">
                <span>Optional note</span>
                <textarea
                  value={championFeedbackDraft.message}
                  onChange={(event) => setChampionFeedbackDraft((current) => current ? { ...current, message: event.target.value } : current)}
                  placeholder="What did the champion get right or wrong?"
                  rows={4}
                />
              </label>
              <button className="command primary" disabled={championFeedbackSubmitting}>
                <MessageSquare size={16} />
                {championFeedbackSubmitting ? "Recording..." : "Record Feedback"}
              </button>
            </form>
          </section>
        </div>
      )}
    </main>
  );
}

function Panel({
	id,
	title,
	icon,
	wide = false,
	tab,
	children,
}: {
	id?: string;
	title: string;
	icon: ReactNode;
	wide?: boolean;
	tab?: ProjectTabKey;
	children: ReactNode;
}) {
	return (
		<section className={wide ? "panel wide" : "panel"} id={id} data-project-tab={tab}>
			<header>
				<span>{icon}</span>
				<h3>{title}</h3>
      </header>
      {children}
    </section>
  );
}

function MissionRoute({
  brief,
  thinking,
  stages,
  activity,
  results,
  exportSummary,
  actions,
  onAction,
  onOpenTab,
}: {
  brief: MissionBrief;
  thinking: AIThinking;
  stages: MissionStage[];
  activity: ActivityCardModel[];
  results: ResultsSummary;
  exportSummary: ExportSummary;
  actions: MissionNextAction[];
  onAction: (action: MissionNextAction) => void;
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  const primaryAction = actions.find((action) => action.priority === "primary") ?? actions[0];
  const latestActivity = activity[0] ?? null;
  const activeStage = currentMissionStage(stages);
  const completedStageCount = stages.filter((stage) => stage.status === "done").length;
  const flowProgress = missionStageProgress(stages);

  if (brief.id === "no-mission") {
    return <MissionEmptyState brief={brief} stages={stages} actions={actions} onAction={onAction} />;
  }

  return (
    <div className="mission-workspace">
      <section className={`mission-stage-panel mission-flow-board ${activeStage.status}`}>
        <div className="mission-flow-head">
          <div className="mission-now-state">
            <span className={`mission-now-icon ${activeStage.status}`}>{missionStageIcon(activeStage.status)}</span>
            <div>
              <div className="eyebrow">Realtime project state</div>
              <h3>{activeStage.label}</h3>
              <p>{activeStage.detail}</p>
            </div>
          </div>
          <div className="mission-flow-meta">
            <Badge value={brief.statusLabel} />
            <small>
              {completedStageCount}/{stages.length} steps complete
            </small>
            <button className="mission-link-button" type="button" onClick={() => onOpenTab("activity", "activity")}>
              <Activity size={13} />
              Open journal
            </button>
          </div>
        </div>
        <div className="mission-flow-progress" aria-hidden="true">
          <span style={{ width: `${flowProgress}%` }} />
        </div>
        <MissionStageTimeline stages={stages} />
      </section>

      <section className="mission-card">
        <div className="mission-card-head">
          <div>
            <div className="eyebrow">Mission</div>
            <h3>{brief.title}</h3>
            <p>{brief.goal}</p>
          </div>
          <Badge value={brief.statusLabel} />
        </div>
        <div className="mission-card-metrics">
          <span>
            <small>Progress</small>
            <strong>{brief.progressLabel}</strong>
          </span>
          <span>
            <small>{brief.bestMetricLabel}</small>
            <strong>{brief.bestMetricValue}</strong>
          </span>
          <span>
            <small>ETA</small>
            <strong>{brief.etaLabel}</strong>
          </span>
        </div>
        {brief.blocker && (
          <div className="mission-blocker">
            <AlertTriangle size={15} />
            <span>{brief.blocker}</span>
          </div>
        )}
      </section>

      <section className="ai-thinking-card">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">What the AI is doing</div>
            <strong>{thinking.state}</strong>
          </div>
          <small>{thinking.updatedAt ? formatRelativeTime(thinking.updatedAt) : thinking.confidenceLabel}</small>
        </div>
        <div className="thinking-grid">
          <ThinkingRow label="Observation" value={thinking.observation} />
          <ThinkingRow label="Reasoning" value={thinking.reasoning} />
          <ThinkingRow label="Decision" value={thinking.decision} />
          <ThinkingRow label="Expected outcome" value={thinking.expectedOutcome} />
        </div>
      </section>

      <aside className="mission-inspector">
        <section className="result-snapshot-card">
          <div className="mission-section-head">
            <div>
              <div className="eyebrow">Best result</div>
              <strong>{results.championModel}</strong>
            </div>
            <Badge value={exportSummary.statusLabel} />
          </div>
          <div className="champion-outcome-primary">
            <small>{results.primaryMetricLabel}</small>
            <strong>{results.primaryMetricValue}</strong>
          </div>
          <p>{results.whyItWon}</p>
          <button className="command compact" type="button" onClick={() => onOpenTab("results", "results")}>
            View results
          </button>
        </section>

        <section className="next-action-card">
          <div className="eyebrow">Next action</div>
          {primaryAction ? (
            <button
              className="mission-primary-action"
              type="button"
              onClick={() => onAction(primaryAction)}
              disabled={primaryAction.disabled}
            >
              <span>
                <strong>{userFacingActionLabel(primaryAction.label)}</strong>
                <small>{userFacingActivityText(primaryAction.reason, 140)}</small>
              </span>
              <StepForward size={17} />
            </button>
          ) : (
            <div className="empty compact">No action is needed right now.</div>
          )}
        </section>

        <section className="activity-snapshot-card">
          <div className="eyebrow">Latest update</div>
          {latestActivity ? (
            <button className="mission-signal info" type="button" onClick={() => onOpenTab("activity", "activity")}>
              <span>
                <strong>{latestActivity.title}</strong>
                <small>{latestActivity.summary}</small>
              </span>
              <small>{formatRelativeTime(latestActivity.timestamp)}</small>
            </button>
          ) : (
            <div className="empty compact">The work journal will fill in as the mission starts.</div>
          )}
        </section>
      </aside>
    </div>
  );
}

function MissionEmptyState({
  brief,
  stages,
  actions,
  onAction,
}: {
  brief: MissionBrief;
  stages: MissionStage[];
  actions: MissionNextAction[];
  onAction: (action: MissionNextAction) => void;
}) {
  const primary = actions.find((action) => action.priority === "primary") ?? actions[0];
  const secondary = actions.filter((action) => action.id !== primary?.id).slice(0, 1);
  const previewStages = stages.slice(0, 5);

  return (
    <div className="mission-empty-state">
      <section className="mission-empty-hero">
        <div className="mission-empty-copy">
          <span className="mission-empty-mark" aria-hidden="true">
            <BrainCircuit size={22} />
          </span>
          <div>
            <div className="eyebrow">Mission setup</div>
            <h3>{brief.title}</h3>
            <p>{brief.goal}</p>
          </div>
          <div className="mission-empty-actions">
            {primary && (
              <button className="mission-primary-action" type="button" onClick={() => onAction(primary)} disabled={primary.disabled}>
                <span>
                  <strong>{primary.label}</strong>
                  <small>{primary.reason}</small>
                </span>
                <FolderOpen size={17} />
              </button>
            )}
            {secondary.map((action) => (
              <button className="mission-secondary-action" type="button" key={action.id} onClick={() => onAction(action)} disabled={action.disabled}>
                <span>
                  <strong>{action.label}</strong>
                  <small>{action.reason}</small>
                </span>
                <RefreshCcw size={14} />
              </button>
            ))}
          </div>
        </div>
        <div className="mission-empty-panel" aria-label="Mission workflow preview">
          <div className="mission-section-head">
            <div>
              <div className="eyebrow">Run flow</div>
              <strong>From folder to export-ready model</strong>
            </div>
            <Badge value={brief.statusLabel} />
          </div>
          <div className="mission-empty-flow">
            {previewStages.map((stage, index) => (
              <div className="mission-empty-step" key={stage.id}>
                <small>{String(index + 1).padStart(2, "0")}</small>
                <span>{missionStageIcon(index === 0 ? "active" : "waiting")}</span>
                <div>
                  <strong>{stage.label}</strong>
                  <p>{stage.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function ThinkingRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="thinking-row">
      <small>{label}</small>
      <p>{value}</p>
    </div>
  );
}

function MissionStageTimeline({ stages }: { stages: MissionStage[] }) {
  return (
    <div className="mission-stage-timeline" aria-label="Mission workflow">
      {stages.map((stage, index) => (
        <div
          className={`mission-stage ${stage.status}`}
          key={stage.id}
          aria-current={stage.status === "active" || stage.status === "blocked" ? "step" : undefined}
        >
          <span className={`mission-stage-marker ${stage.status}`} aria-hidden="true">
            {missionStageIcon(stage.status)}
          </span>
          <small className="mission-stage-number">{String(index + 1).padStart(2, "0")}</small>
          <div>
            <strong>{stage.label}</strong>
            <small>{stage.detail}</small>
          </div>
          <Badge value={stage.status} />
        </div>
      ))}
    </div>
  );
}

function currentMissionStage(stages: MissionStage[]): MissionStage {
  if (stages.length > 0 && stages.every((stage) => stage.status === "waiting")) {
    return {
      id: "empty",
      label: "Waiting for mission",
      detail: stages[0]?.detail || "Create or select a mission to begin.",
      status: "waiting",
    };
  }

  return (
    stages.find((stage) => stage.status === "blocked") ??
    stages.find((stage) => stage.status === "active") ??
    stages.find((stage) => stage.status === "waiting") ??
    stages[stages.length - 1] ?? {
      id: "empty",
      label: "Waiting for mission",
      detail: "Create or select a mission to begin.",
      status: "waiting",
    }
  );
}

function missionStageProgress(stages: MissionStage[]) {
  if (stages.length === 0) return 0;
  const activeIndex = stages.findIndex((stage) => stage.status === "active" || stage.status === "blocked");
  if (activeIndex >= 0) {
    return Math.round(((activeIndex + 0.55) / stages.length) * 100);
  }
  const completed = stages.filter((stage) => stage.status === "done").length;
  return Math.round((completed / stages.length) * 100);
}

function buildProjectWorkflowTabs({
  tabs,
  missionDigest,
  missionStages,
  activityFeed,
  activityStreamState,
  resultsSummary,
  exportSummary,
}: {
  tabs: ProjectWorkflowTabBase[];
  missionDigest: MissionDigest;
  missionStages: MissionStage[];
  activityFeed: ActivityCardModel[];
  activityStreamState: ActivityStreamState;
  resultsSummary: ResultsSummary;
  exportSummary: ExportSummary;
}): ProjectWorkflowTab[] {
  const stageState = (ids: string[]) => summarizeWorkflowStageState(missionStages.filter((stage) => ids.includes(stage.id)));
  const activityDetail =
    missionDigest.liveActivity.status === "idle" && activityFeed.length > 0
      ? `${activityFeed.length} journal ${activityFeed.length === 1 ? "entry" : "entries"}`
      : missionDigest.liveActivity.label || activityStreamBadge(activityStreamState);
  const details: Record<Exclude<ProjectTabKey, "developer">, { state: ProjectWorkflowTabState; detail: string }> = {
    mission: {
      state: missionDigest.state === "empty" ? "active" : stageState(["created", "dataset", "plan"]),
      detail: missionDigest.stateLabel,
    },
    activity: {
      state: stageState(["experiments", "refinement"]),
      detail: activityDetail,
    },
    results: {
      state: stageState(["evaluation", "champion"]),
      detail: resultsSummary.hasResults
        ? `${resultsSummary.primaryMetricValue} ${resultsSummary.primaryMetricLabel}`
        : "Awaiting evidence",
    },
    export: {
      state: stageState(["export", "demo"]),
      detail: exportSummary.statusLabel,
    },
  };

  return tabs.map((tab) => ({ ...tab, ...details[tab.key] }));
}

function summarizeWorkflowStageState(stages: MissionStage[]): ProjectWorkflowTabState {
  if (stages.some((stage) => stage.status === "blocked")) return "blocked";
  if (stages.some((stage) => stage.status === "active")) return "active";
  if (stages.length > 0 && stages.every((stage) => stage.status === "done")) return "done";
  if (stages.some((stage) => stage.status === "done")) return "active";
  return "waiting";
}

function missionStageIcon(status: MissionStage["status"]): ReactNode {
  if (status === "done") return <CheckCircle2 size={15} />;
  if (status === "active") return <Activity size={15} />;
  if (status === "blocked") return <AlertTriangle size={15} />;
  return <Timer size={15} />;
}

function ActivityRoute({
  cards,
  filter,
  onFilterChange,
  onOpenDeveloper,
}: {
  cards: ActivityCardModel[];
  filter: ActivityFilterKey;
  onFilterChange: (filter: ActivityFilterKey) => void;
  onOpenDeveloper: () => void;
}) {
  const filters: Array<{ key: ActivityFilterKey; label: string }> = [
    { key: "all", label: "All" },
    { key: "decisions", label: "Decisions" },
    { key: "experiments", label: "Experiments" },
    { key: "results", label: "Results" },
    { key: "blockers", label: "Blockers" },
  ];
  const visibleCards = cards.filter((card) => activityCardMatchesFilter(card, filter));

  return (
    <div className="activity-journal">
      <header className="route-heading">
        <div>
          <div className="eyebrow">AI work journal</div>
          <h3>Activity</h3>
          <p>Readable updates from the mission, summarized from project events, decisions, experiments, and results.</p>
        </div>
        <button className="command compact" type="button" onClick={onOpenDeveloper}>
          Developer View
        </button>
      </header>
      <div className="activity-filter-bar" role="tablist" aria-label="Activity filters">
        {filters.map((item) => (
          <button
            key={item.key}
            type="button"
            className={filter === item.key ? "active" : ""}
            onClick={() => onFilterChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="activity-card-list">
        {visibleCards.length > 0 ? (
          visibleCards.map((card) => <ActivityCard key={card.id} card={card} onOpenDeveloper={onOpenDeveloper} />)
        ) : (
          <ActivityEmptyState hasCards={cards.length > 0} filter={filter} />
        )}
      </div>
    </div>
  );
}

function ActivityEmptyState({ hasCards, filter }: { hasCards: boolean; filter: ActivityFilterKey }) {
  const filtered = hasCards && filter !== "all";
  const preview = [
    { label: "Mission created", detail: "Goal, dataset, and setup context are captured." },
    { label: "Planner decision", detail: "The agent explains what it is doing and why." },
    { label: "Experiment result", detail: "Completed runs surface the evidence that matters." },
  ];

  return (
    <section className="activity-empty-state">
      <span className="activity-empty-mark" aria-hidden="true">
        <Activity size={18} />
      </span>
      <div className="activity-empty-copy">
        <div className="eyebrow">Operations journal</div>
        <strong>{filtered ? "No entries match this filter" : "Waiting for the first mission event"}</strong>
        <p>
          {filtered
            ? "Try another journal filter to see decisions, experiments, results, and blockers already captured."
            : "When the mission starts, this stream will summarize planner decisions, worker movement, experiment results, and handoff readiness."}
        </p>
      </div>
      <div className="activity-empty-preview" aria-label="Activity journal preview">
        {preview.map((item, index) => (
          <span key={item.label}>
            <small>{String(index + 1).padStart(2, "0")}</small>
            <span>
              <strong>{item.label}</strong>
              <small>{item.detail}</small>
            </span>
          </span>
        ))}
      </div>
    </section>
  );
}

function ActivityCard({ card, onOpenDeveloper }: { card: ActivityCardModel; onOpenDeveloper: () => void }) {
  return (
    <article className={`activity-card ${card.type} ${card.status}`}>
      <div className="activity-card-icon">
        {card.type === "decision" ? <BrainCircuit size={16} /> : card.type === "result" ? <Trophy size={16} /> : <Activity size={16} />}
      </div>
      <div className="activity-card-body">
        <header>
          <span>
            <small>{new Date(card.timestamp).toLocaleTimeString()}</small>
            <strong>{card.title}</strong>
          </span>
          <Badge value={card.status} />
        </header>
        <p>{card.summary}</p>
        {(card.evidenceSummary || card.resultSummary) && (
          <div className="activity-card-facts">
            {card.evidenceSummary && (
              <span>
                <small>Evidence</small>
                <strong>{card.evidenceSummary}</strong>
              </span>
            )}
            {card.resultSummary && (
              <span>
                <small>Result</small>
                <strong>{card.resultSummary}</strong>
              </span>
            )}
          </div>
        )}
        <details className="activity-details">
          <summary>Technical details</summary>
          <div className="activity-metadata">
            <span>
              <small>Source</small>
              <strong>{card.technicalSource}</strong>
            </span>
            <span>
              <small>Raw payload</small>
              <strong>Available in Developer View</strong>
            </span>
          </div>
          <button className="mission-link-button" type="button" onClick={onOpenDeveloper}>
            Open Developer View
          </button>
        </details>
      </div>
    </article>
  );
}

function ResultsRoute({
  summary,
  onSelectCandidate,
  onOpenExport,
  onOpenDeveloper,
}: {
  summary: ResultsSummary;
  onSelectCandidate: (jobId: string) => void;
  onOpenExport: () => void;
  onOpenDeveloper: () => void;
}) {
  return (
    <div className="results-page">
      <section className="results-hero">
        <div>
          <div className="eyebrow">Current champion</div>
          <h3>{summary.championModel}</h3>
          <p>{summary.whyItWon}</p>
        </div>
        <div className="results-primary-metric">
          <small>{summary.primaryMetricLabel}</small>
          <strong>{summary.primaryMetricValue}</strong>
          <span>{summary.improvementLabel}</span>
        </div>
      </section>

      <section className="results-grid">
        <div className="results-section">
          <div className="mission-section-head">
            <div>
              <div className="eyebrow">Why it won</div>
              <strong>Champion explanation</strong>
            </div>
            <Badge value={summary.exportStatus} />
          </div>
          <p>{summary.whyItWon}</p>
          <button className="command compact" type="button" onClick={onOpenExport}>
            Open export
          </button>
        </div>

        <div className="results-section">
          <div className="eyebrow">Learning summary</div>
          <div className="result-list">
            {summary.learningSummary.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>

        <div className="results-section">
          <div className="eyebrow">Remaining risks</div>
          <div className="result-list warning">
            {summary.remainingRisks.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
      </section>

      <section className="candidate-section">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Top candidates</div>
            <strong>Best three by mission fit</strong>
          </div>
          <button className="command compact" type="button" onClick={onOpenDeveloper}>
            Full comparison
          </button>
        </div>
        <div className="candidate-list">
          {summary.topCandidates.length > 0 ? (
            summary.topCandidates.map((candidate) => (
              <button
                className="candidate-card"
                key={`${candidate.rank}-${candidate.jobId || candidate.model}`}
                type="button"
                onClick={() => candidate.jobId && onSelectCandidate(candidate.jobId)}
              >
                <span>
                  <small>#{candidate.rank}</small>
                  <strong>{candidate.model}</strong>
                  <p>{candidate.why}</p>
                </span>
                <span>
                  <small>{candidate.metricLabel}</small>
                  <strong>{candidate.metricValue}</strong>
                  <Badge value={candidate.status} />
                </span>
              </button>
            ))
          ) : (
            <ResultsEmptyState hasResults={summary.hasResults} />
          )}
        </div>
      </section>
    </div>
  );
}

function ResultsEmptyState({ hasResults }: { hasResults: boolean }) {
  const steps = [
    { label: "Train candidates", detail: "Experiments report comparable metrics." },
    { label: "Score fit", detail: "Model Express weighs accuracy, risk, and handoff readiness." },
    { label: "Select champion", detail: "The best supported model becomes the export candidate." },
  ];

  return (
    <section className="results-empty-state">
      <span className="results-empty-mark" aria-hidden="true">
        <BarChart3 size={18} />
      </span>
      <div className="results-empty-copy">
        <div className="eyebrow">Evidence queue</div>
        <strong>{hasResults ? "Waiting for ranked candidates" : "Waiting for experiment metrics"}</strong>
        <p>
          {hasResults
            ? "Completed runs exist, but Model Express needs enough comparable evidence before ranking the strongest candidates."
            : "Once training jobs finish, this area will summarize the leading models and why each one matters."}
        </p>
      </div>
      <div className="results-empty-steps" aria-label="Results evidence flow">
        {steps.map((step, index) => (
          <span key={step.label}>
            <small>{String(index + 1).padStart(2, "0")}</small>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
          </span>
        ))}
      </div>
    </section>
  );
}

function ExportWaitingState({ readinessLabel }: { readinessLabel: string }) {
  const steps = [
    { label: "Champion selected", detail: "Model Express chooses the strongest supported candidate." },
    { label: "Package prepared", detail: "The ONNX artifact, label map, and model card are assembled." },
    { label: "Demo validated", detail: "A held-out image confirms the handoff path works." },
  ];

  return (
    <section className="export-waiting-state">
      <span className="export-waiting-mark" aria-hidden="true">
        <Trophy size={18} />
      </span>
      <div className="export-waiting-copy">
        <div className="eyebrow">Handoff queue</div>
        <strong>Waiting for a champion model</strong>
        <p>{readinessLabel}</p>
      </div>
      <div className="export-waiting-steps" aria-label="Export readiness flow">
        {steps.map((step, index) => (
          <span key={step.label}>
            <small>{String(index + 1).padStart(2, "0")}</small>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
          </span>
        ))}
      </div>
    </section>
  );
}

function ExportRoute({
  summary,
  data,
  prediction,
  predictionError,
  predictionLoading,
  selectedImageIndex,
  onNextImage,
  onRandomImage,
  onRequestExport,
  onRunPrediction,
  onOpenDeveloper,
}: {
  summary: ExportSummary;
  data: ChampionExportDemo;
  prediction: ChampionDemoPrediction | null;
  predictionError: string;
  predictionLoading: boolean;
  selectedImageIndex: number;
  customImage: ChampionDemoImage | null;
  customImageURI: string;
  customTrueLabel: string;
  localInferenceStatus: string;
  localInferenceError: string;
  slideshowEnabled: boolean;
  slideshowIntervalMs: number;
  detectionConfidenceThreshold: number;
  detectionIouThreshold: number;
  onCustomImageURIChange: (value: string) => void;
  onCustomTrueLabelChange: (value: string) => void;
  onChooseCustomImage: () => void;
  onRunCustomPrediction: () => void;
  onToggleSlideshow: () => void;
  onSelectImage: (index: number) => void;
  onNextImage: () => void;
  onRandomImage: () => void;
  onRequestExport: () => void;
  onRunPrediction: (image: ChampionDemoImage) => void;
  onOpenFeedback: (rating: ChampionFeedbackRating) => void;
  onSlideshowIntervalChange: (value: number) => void;
  onDetectionConfidenceThresholdChange: (value: number) => void;
  onDetectionIouThresholdChange: (value: number) => void;
  onOpenDeveloper: () => void;
}) {
  const selectedImage = data.demoImages[selectedImageIndex] ?? data.demoImages[0] ?? null;
  const previewURI = demoImagePreviewURI(selectedImage);
  const predictionStatus = prediction ? normalizedStatus(prediction.status || "PENDING") : "";
  const confidence = prediction ? numericValue(prediction.confidence) : 0;

  return (
    <div className="export-page" id="export-package">
      <section className="export-package-card">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Export package</div>
            <h3>{summary.title}</h3>
            <p>{summary.readinessLabel}</p>
          </div>
          <Badge value={summary.statusLabel} />
        </div>
        <div className="export-summary-grid">
          <span>
            <small>Format</small>
            <strong>{summary.primaryFormat}</strong>
          </span>
          <span>
            <small>Validation</small>
            <strong>{summary.validationStatus}</strong>
          </span>
          <span>
            <small>Demo</small>
            <strong>{summary.demoStatus}</strong>
          </span>
        </div>
        <div className="export-actions">
          <button className="command primary" type="button" onClick={onRequestExport} disabled={!summary.hasChampion}>
            <HardDriveUpload size={16} />
            Prepare ONNX
          </button>
          <button className="command" type="button" onClick={onOpenDeveloper}>
            Open technical manifest
          </button>
        </div>
      </section>

      {!summary.hasChampion && <ExportWaitingState readinessLabel={summary.readinessLabel} />}

      <section className="handoff-grid">
        <div className="handoff-section">
          <div className="eyebrow">Includes</div>
          <div className="result-list">
            {summary.includes.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
        <div className="handoff-section">
          <div className="eyebrow">Use this when</div>
          <div className="result-list">
            {summary.useCases.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
        <div className="handoff-section">
          <div className="eyebrow">Known limitations</div>
          <div className="result-list warning">
            {summary.limitations.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
      </section>

      <section className="export-demo-simple">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Demo image</div>
            <strong>{demoImageLabel(selectedImage) || "Held-out image"}</strong>
          </div>
          <span>
            <button className="icon-command" type="button" onClick={onRandomImage} disabled={data.demoImages.length < 2} title="Random held-out image">
              <Shuffle size={14} />
            </button>
            <button className="icon-command" type="button" onClick={onNextImage} disabled={data.demoImages.length < 2} title="Next held-out image">
              <StepForward size={14} />
            </button>
            <button
              className="command compact"
              type="button"
              onClick={() => selectedImage && onRunPrediction(selectedImage)}
              disabled={!selectedImage || predictionLoading}
            >
              <Play size={15} />
              Run demo
            </button>
          </span>
        </div>
        <div className="demo-simple-grid">
          <div className="test-image-frame">
            {previewURI ? <img src={previewURI} alt={demoImageLabel(selectedImage) || "demo image"} /> : <div className="test-image-placeholder">No image</div>}
            {prediction && detectionBoxesFromPrediction(prediction).length > 0 && <DetectionOverlay detections={detectionBoxesFromPrediction(prediction)} />}
          </div>
          <div className="demo-result-summary">
            {predictionLoading && <Badge value="RUNNING" />}
            {predictionError && <div className="mission-blocker"><AlertTriangle size={15} /><span>{userFacingActivityText(predictionError, 140)}</span></div>}
            {prediction ? (
              <>
                <span>
                  <small>Prediction</small>
                  <strong>{prediction.predicted_label || predictionStatusMessage(predictionStatus)}</strong>
                </span>
                <span>
                  <small>Confidence</small>
                  <strong>{confidence ? formatTopKScore(confidence) : "-"}</strong>
                </span>
                <span>
                  <small>Latency</small>
                  <strong>{typeof prediction.latency_ms === "number" ? formatLatency(prediction.latency_ms) : "-"}</strong>
                </span>
              </>
            ) : (
              <div className="empty compact">Run a demo image after the package is ready.</div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function DeveloperRoute({ diagnostics, onBack }: { diagnostics: DeveloperDiagnostics; onBack: () => void }) {
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

function MissionOverview({
  digest,
  onAction,
  onOpenTab,
}: {
  digest: MissionDigest;
  onAction: (action: MissionNextAction) => void;
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  return (
    <div className={`mission-overview mission-state-${digest.state}`}>
      <MissionStatusPanel digest={digest} />
      <MissionHealthStrip items={digest.health} onOpenTab={onOpenTab} />
      <MissionNextActions actions={digest.nextActions} onAction={onAction} />
      <LiveAgentActivity activity={digest.liveActivity} onOpenTab={onOpenTab} />
      <MissionSignals signals={digest.recentSignals} onOpenTab={onOpenTab} />
      {digest.champion && <ChampionOutcomeSummary champion={digest.champion} onOpenTab={onOpenTab} />}
    </div>
  );
}

function MissionStatusPanel({ digest }: { digest: MissionDigest }) {
  return (
    <section className="mission-status-panel">
      <div className="mission-status-head">
        <div>
          <div className="eyebrow">What is happening</div>
          <h3>{digest.headline}</h3>
          <p>{digest.detail}</p>
        </div>
        <div className="mission-status-badges">
          <Badge value={digest.stateLabel} />
          <Badge value={digest.healthLabel} />
        </div>
      </div>
      <div className="mission-facts">
        {digest.facts.map((fact) => (
          <span className={fact.tone ?? "info"} key={`${fact.label}-${fact.value}`}>
            <small>{fact.label}</small>
            <strong>{fact.value}</strong>
          </span>
        ))}
      </div>
    </section>
  );
}

function MissionHealthStrip({
  items,
  onOpenTab,
}: {
  items: MissionHealthItem[];
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  return (
    <section className="mission-health-strip" aria-label="Mission health">
      {items.map((item) => (
        <button
          className={`mission-health-chip ${item.tone}`}
          key={item.id}
          type="button"
          onClick={() => item.targetTab && onOpenTab(item.targetTab, item.targetId)}
        >
          <small>{item.label}</small>
          <strong>{item.value}</strong>
        </button>
      ))}
    </section>
  );
}

function MissionNextActions({
  actions,
  onAction,
}: {
  actions: MissionNextAction[];
  onAction: (action: MissionNextAction) => void;
}) {
  const primary = actions.find((action) => action.priority === "primary") ?? actions[0];
  const secondary = actions.filter((action) => action.id !== primary?.id).slice(0, 3);

  return (
    <section className="mission-next-actions">
      <div>
        <div className="eyebrow">What should I do next</div>
        {primary ? (
          <button
            className="mission-primary-action"
            type="button"
            onClick={() => onAction(primary)}
            disabled={primary.disabled}
          >
            <span>
              <strong>{primary.label}</strong>
              <small>{primary.reason}</small>
            </span>
            <StepForward size={17} />
          </button>
        ) : (
          <div className="empty compact">No operator action is needed right now.</div>
        )}
      </div>
      {secondary.length > 0 && (
        <div className="mission-secondary-actions">
          {secondary.map((action) => (
            <button
              className="mission-secondary-action"
              type="button"
              key={action.id}
              onClick={() => onAction(action)}
              disabled={action.disabled}
            >
              <span>
                <strong>{action.label}</strong>
                <small>{action.reason}</small>
              </span>
              <Link2 size={14} />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function LiveAgentActivity({
  activity,
  onOpenTab,
}: {
  activity: MissionLiveActivity;
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  const moving = ["active", "waiting"].includes(activity.status) || ["connecting", "reconnecting", "fallback"].includes(activity.streamState);

  return (
    <section className={`live-agent-activity ${activity.status}`}>
      <div className="live-agent-current">
        <span className={moving ? "live-agent-pulse active" : "live-agent-pulse"} />
        <div>
          <div className="eyebrow">Live LLM activity</div>
          <strong>{activity.label}</strong>
          <small>{activity.detail}</small>
        </div>
        <Badge value={activityStreamBadge(activity.streamState)} />
      </div>
      {activity.steps.length > 0 && (
        <div className="live-agent-steps">
          {activity.steps.map((step) => (
            <button
              className={`live-agent-step ${step.status}`}
              key={step.id}
              type="button"
              onClick={() => onOpenTab(step.targetTab ?? "agents", step.targetId ?? "agent-activity")}
            >
              <span>{step.label}</span>
              <small>{step.timestamp ? formatRelativeTime(step.timestamp) : step.status}</small>
            </button>
          ))}
        </div>
      )}
      <button className="mission-link-button" type="button" onClick={() => onOpenTab("agents", "agent-activity")}>
        Open full activity
      </button>
    </section>
  );
}

function MissionSignals({
  signals,
  onOpenTab,
}: {
  signals: MissionSignal[];
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  return (
    <section className="mission-signals">
      <div className="mission-section-head">
        <div>
          <div className="eyebrow">Recent signals</div>
          <strong>Important changes</strong>
        </div>
      </div>
      <div className="mission-signal-list">
        {signals.length > 0 ? (
          signals.map((signal) => (
            <button
              className={`mission-signal ${signal.tone}`}
              key={signal.id}
              type="button"
              onClick={() => signal.targetTab && onOpenTab(signal.targetTab, signal.targetId)}
            >
              <span>
                <strong>{signal.label}</strong>
                <small>{signal.detail}</small>
              </span>
              <small>{signal.timestamp ? formatRelativeTime(signal.timestamp) : ""}</small>
            </button>
          ))
        ) : (
          <div className="empty compact">No state-changing events have been recorded yet.</div>
        )}
      </div>
    </section>
  );
}

function ChampionOutcomeSummary({
  champion,
  onOpenTab,
}: {
  champion: MissionChampionSummary;
  onOpenTab: (tab: ProjectTabTarget, targetId?: string) => void;
}) {
  const extraFacts = [
    { label: champion.secondaryMetricLabel, value: champion.secondaryMetricValue },
    { label: "Latency", value: champion.latency },
    { label: "Cost", value: champion.cost },
    { label: "Size", value: champion.modelSize },
    { label: "Fit", value: champion.objectiveFit },
  ].filter((fact) => fact.value && fact.value !== "-");

  return (
    <section className="champion-outcome-summary">
      <div className="mission-section-head">
        <div>
          <div className="eyebrow">Outcome summary</div>
          <strong>{champion.model}</strong>
        </div>
        <Badge value="SELECTED" />
      </div>
      <div className="champion-outcome-primary">
        <small>{champion.primaryMetricLabel}</small>
        <strong>{champion.primaryMetricValue}</strong>
      </div>
      {extraFacts.length > 0 && (
        <div className="champion-outcome-facts">
          {extraFacts.slice(0, 4).map((fact) => (
            <span key={`${fact.label}-${fact.value}`}>
              <small>{fact.label}</small>
              <strong>{fact.value}</strong>
            </span>
          ))}
        </div>
      )}
      <div className="champion-outcome-actions">
        <button className="command compact" type="button" onClick={() => onOpenTab("export", "export-demo")}>
          Open Export
        </button>
        <button className="command compact" type="button" onClick={() => onOpenTab("experiments", "champion-comparison")}>
          Compare Runs
        </button>
      </div>
    </section>
  );
}

function AgentActivityPanel({
  events,
  streamState,
  detail,
}: {
  events: AgentActivityEvent[];
  streamState: ActivityStreamState;
  detail: ProjectDetail;
}) {
  const current = agentActivityCurrentState(events, detail, streamState);
  const visibleEvents = events.slice(0, 12);

  return (
    <div className="agent-activity-panel">
      <div className={`activity-current ${current.status}`}>
        <span className={`activity-marker ${current.severity}`}>{activitySeverityIcon(current.severity)}</span>
        <div>
          <strong>{current.title}</strong>
          <small>{current.detail}</small>
        </div>
        <Badge value={activityStreamBadge(streamState)} />
      </div>

      {visibleEvents.length > 0 ? (
        <div className="activity-list">
          {visibleEvents.map((event) => {
            const rows = activityMetadataRows(event.metadata);
            return (
              <div className={`activity-row ${event.severity} ${event.status}`} key={event.id}>
                <span className={`activity-marker ${event.severity}`}>{activitySeverityIcon(event.severity)}</span>
                <div className="activity-body">
                  <div className="activity-row-head">
                    <span>
                      <strong>{event.title}</strong>
                      <small>{activityEventSubtitle(event)}</small>
                    </span>
                    <Badge value={event.status} />
                  </div>
                  {event.message && <p>{activitySafeDisplayText(event.message, 220)}</p>}
                  {rows.length > 0 && (
                    <details className="activity-details">
                      <summary>Details</summary>
                      <div className="activity-metadata">
                        {rows.map((row) => (
                          <span key={`${event.id}-${row.label}`}>
                            <small>{row.label}</small>
                            <strong title={row.value}>{row.value}</strong>
                          </span>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty compact">Agent activity will appear when planning, validation, worker, or job events are recorded.</div>
      )}
    </div>
  );
}

function AgentDecisionChat({ turns }: { turns: DecisionChatTurn[] }) {
  return (
    <div className="decision-chat">
      {turns.map((turn) => (
        <div className="decision-chat-turn" key={turn.decision.id}>
          <div className="message-row user">
            <div className="message-bubble user-bubble">
              <span>{turn.question}</span>
            </div>
          </div>

          <div className="message-row agent">
            <div className="message-avatar">
              <BrainCircuit size={15} />
            </div>
            <div className="message-bubble agent-bubble">
              <div className="decision-message-head">
                <span>
                  <Badge value={turn.decision.decision_type} />
                  <small>{new Date(turn.decision.created_at).toLocaleString()}</small>
                </span>
                <small>{turn.decision.plan_id || "no plan"}</small>
              </div>

              <p>{turn.opening}</p>

              <div className="decision-payload message-facts">
                {turn.highlights.map((item) => (
                  <span key={`${turn.decision.id}-${item.label}`}>
                    <small>{item.label}</small>
                    <strong>{item.value}</strong>
                  </span>
                ))}
              </div>

              {turn.sections.length > 0 && (
                <div className="message-section-list">
                  {turn.sections.map((section) => (
                    <div className="message-section" key={`${turn.decision.id}-${section.title}`}>
                      <strong>{section.title}</strong>
                      {section.items.slice(0, 4).map((item) => (
                        <p key={`${section.title}-${item}`}>{item}</p>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              {turn.mechanismCoverage.length > 0 && (
                <div className="mechanism-coverage-panel compact">
                  <strong>Mechanism Coverage</strong>
                  <div className="mechanism-coverage-list">
                    {turn.mechanismCoverage.slice(0, 6).map((item) => (
                      <div
                        className="mechanism-coverage-row"
                        key={`${turn.decision.id}-${item.status}-${item.mechanism}-${item.detail}`}
                      >
                        <span>
                          <strong>{item.mechanism}</strong>
                          <small>{item.detail}</small>
                        </span>
                        <Badge value={item.status} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {turn.retrievedMemory.length > 0 && <RetrievedMemoryPanel memories={turn.retrievedMemory} />}

              {turn.rejections.length > 0 && (
                <div className="rejection-panel compact">
                  <strong>Backend Gate And Rejections</strong>
                  <div className="rejection-list">
                    {turn.rejections.slice(0, 5).map((item) => (
                      <span key={`${turn.decision.id}-${item.kind}-${item.text}`}>
                        <small>{item.kind}</small>
                        {item.text}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {turn.candidateScores.length > 0 && (
                <div className="candidate-score-panel compact">
                  <strong>Candidate Scores</strong>
                  <div className="candidate-score-list">
                    {turn.candidateScores.slice(0, 4).map((candidate) => (
                      <div className="candidate-score-row" key={`${turn.decision.id}-${candidate.label}`}>
                        <div className="candidate-score-head">
                          <span>
                            <strong>{candidate.label}</strong>
                            <small>
                              {[
                                candidate.mechanism ? `mechanism ${candidate.mechanism}` : "",
                                candidate.intervention,
                                ...candidate.reasons.slice(0, 2),
                              ]
                                .filter(Boolean)
                                .join("; ") || "No rejection reason reported."}
                            </small>
                          </span>
                          <Badge value={candidate.status} />
                        </div>
                        <div className="score-component-list">
                          {[
                            candidate.mechanism ? { label: "Mechanism", value: candidate.mechanism } : null,
                            candidate.expectedEffect ? { label: "Expected Effect", value: candidate.expectedEffect } : null,
                            candidate.validationStatus ? { label: "Validation", value: candidate.validationStatus } : null,
                            candidate.totalScore !== null ? { label: "Total", value: candidate.totalScore.toFixed(3) } : null,
                          ]
                            .filter((item): item is { label: string; value: string } => item !== null)
                            .map((item) => (
                              <span key={`${turn.decision.id}-${candidate.label}-${item.label}`}>
                                <small>{item.label}</small>
                                <strong>{item.value}</strong>
                              </span>
                            ))}
                          {candidate.components.slice(0, 4).map((component) => (
                            <span key={`${turn.decision.id}-${candidate.label}-${component.label}`}>
                              <small>{component.label}</small>
                              <strong>{typeof component.value === "number" ? component.value.toFixed(3) : component.value}</strong>
                            </span>
                          ))}
                        </div>
                        {(candidate.memoryReasons.length > 0 || candidate.memoryHits.length > 0) && (
                          <div className="candidate-memory-list">
                            {candidate.memoryReasons.slice(0, 3).map((reason) => (
                              <span key={`${turn.decision.id}-${candidate.label}-memory-reason-${reason}`}>
                                <small>Retrieved Memory</small>
                                {reason}
                              </span>
                            ))}
                            {candidate.memoryHits.slice(0, 2).map((memory, index) => (
                              <span key={`${turn.decision.id}-${candidate.label}-memory-${memory.sourceId}-${index}`}>
                                <small>{memory.kind || memory.source || "Memory"}</small>
                                {memory.summary || memory.retrievalReason || memory.sourceId}
                                <em>{memory.outcome || memory.sourceId}</em>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function RetrievedMemoryPanel({ memories }: { memories: RetrievedMemoryDisplay[] }) {
  if (memories.length === 0) return null;

  return (
    <div className="retrieved-memory-panel compact">
      <strong>Retrieved Memory</strong>
      <div className="retrieved-memory-list">
        {memories.slice(0, 5).map((memory, index) => (
          <div className="retrieved-memory-row" key={`${memory.source}-${memory.sourceId}-${index}`}>
            <div className="retrieved-memory-head">
              <span>
                <strong>{memory.source || "Memory"}</strong>
                <small>
                  {[memory.kind, memory.mechanism, memory.intervention].filter(Boolean).join(" - ") ||
                    "retrieved decision context"}
                </small>
              </span>
              <Badge value={memory.outcome || memory.kind || "memory"} />
            </div>
            {memory.summary && <p>{memory.summary}</p>}
            {(memory.retrievalReason || memory.score !== null) && (
              <small className="retrieval-reason">
                {[memory.retrievalReason, memory.score !== null ? `score ${memory.score.toFixed(3)}` : ""]
                  .filter(Boolean)
                  .join(" - ")}
              </small>
            )}
            {memory.identifiers.length > 0 && (
              <div className="retrieved-memory-identifiers">
                {memory.identifiers.slice(0, 5).map((identifier) => (
                  <span key={`${memory.sourceId}-${identifier.label}-${identifier.value}`}>
                    <Link2 size={12} />
                    <small>{identifier.label}</small>
                    <strong>{identifier.value}</strong>
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function MemoryRetrievalProbePanel({ snapshots }: { snapshots: MemoryRetrievalProbeSnapshot[] }) {
  if (snapshots.length === 0) {
    return <div className="empty">No memory retrieval diagnostics logged yet.</div>;
  }

  const latest = snapshots[0];

  return (
    <div className="memory-probe-panel">
      <div className="memory-probe-summary">
        <span>
          <strong>{latest.retrievedCount} candidate memory card(s)</strong>
          <small>
            {humanizeMemoryPurpose(latest.purpose)} - {formatRelativeTime(latest.createdAt)}
          </small>
        </span>
        <div className="memory-probe-badges">
          <Badge value={latest.logOnly ? "log only" : "prompt enabled"} />
          {latest.crossProjectOK && <Badge value="cross project" />}
        </div>
      </div>

      <div className="memory-probe-runs">
        {snapshots.slice(0, 4).map((snapshot, index) => (
          <details className="memory-probe-run" key={snapshot.id} open={index === 0}>
            <summary>
              <span>
                <strong>{humanizeMemoryPurpose(snapshot.purpose)}</strong>
                <small>
                  {snapshot.retrievedCount} retrieved - {formatRelativeTime(snapshot.createdAt)}
                </small>
              </span>
              <Badge value={snapshot.logOnly ? "log only" : "prompt enabled"} />
            </summary>
            {snapshot.cards.length > 0 ? (
              <div className="retrieved-memory-list">
                {snapshot.cards.slice(0, 8).map((memory, memoryIndex) => (
                  <div className="retrieved-memory-row" key={`${snapshot.id}-${memory.source}-${memory.sourceId}-${memoryIndex}`}>
                    <div className="retrieved-memory-head">
                      <span>
                        <strong>{memory.source || "Memory"}</strong>
                        <small>
                          {[memory.kind, memory.mechanism, memory.intervention].filter(Boolean).join(" - ") ||
                            "retrieved diagnostic context"}
                        </small>
                      </span>
                      <Badge value={memory.outcome || memory.kind || "memory"} />
                    </div>
                    {memory.summary && <p>{memory.summary}</p>}
                    {(memory.retrievalReason || memory.score !== null) && (
                      <small className="retrieval-reason">
                        {[memory.retrievalReason, memory.score !== null ? `score ${memory.score.toFixed(3)}` : ""]
                          .filter(Boolean)
                          .join(" - ")}
                      </small>
                    )}
                    {memory.identifiers.length > 0 && (
                      <div className="retrieved-memory-identifiers">
                        {memory.identifiers.slice(0, 4).map((identifier) => (
                          <span key={`${snapshot.id}-${memory.sourceId}-${identifier.label}-${identifier.value}`}>
                            <Link2 size={12} />
                            <small>{identifier.label}</small>
                            <strong>{identifier.value}</strong>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty compact">Retrieval ran but found no candidate cards above the active threshold.</div>
            )}
          </details>
        ))}
      </div>
    </div>
  );
}

function DecisionQualityPanel({
  decisions,
  invocations,
}: {
  decisions: AgentDecision[];
  invocations: AgentInvocation[];
}) {
  const snapshot = buildDecisionQualitySnapshot(decisions, invocations);

  if (!snapshot) {
    return <div className="empty">Decision quality metadata will appear after the Experiment Planner records trajectory data.</div>;
  }

  const facts = [
    { label: "Decision Pressure", value: snapshot.decisionPressure || "-" },
    {
      label: "Blocked",
      value: snapshot.blockedMechanisms.length > 0 ? snapshot.blockedMechanisms.join(", ") : "none",
    },
    { label: "Training Runs", value: formatDecisionQualityCount(snapshot.completedTrainingRuns) },
    { label: "Planner Rounds", value: formatDecisionQualityCount(snapshot.completedPlannerRounds) },
    { label: "Gain / Run", value: formatDecisionQualityMetric(snapshot.gainPerCompletedRun, true) },
    { label: "Recent Delta", value: formatDecisionQualityMetric(snapshot.recentBestDelta, true) },
    { label: "Useful Delta", value: formatDecisionQualityMetric(snapshot.minimumUsefulDelta, false) },
    {
      label: "Candidates",
      value: snapshot.totalCandidates > 0 ? `${snapshot.selectedCandidates}/${snapshot.totalCandidates} selected` : "-",
    },
    { label: "Rejected", value: String(snapshot.rejectedCandidates) },
    { label: "Top Rejection", value: snapshot.topRejectionReason || "none" },
  ];

  return (
    <div className="decision-quality-panel">
      <div className="decision-quality-head">
        <span>
          <strong>{snapshot.decisionType}</strong>
          <small>
            {[snapshot.createdAt ? new Date(snapshot.createdAt).toLocaleString() : "", snapshot.decisionId, snapshot.source]
              .filter(Boolean)
              .join(" - ")}
          </small>
        </span>
        <Badge value={snapshot.decisionPressure || "normal"} />
      </div>

      <div className="decision-quality-facts">
        {facts.map((item) => (
          <span key={item.label}>
            <small>{item.label}</small>
            <strong title={item.value}>{item.value}</strong>
          </span>
        ))}
      </div>

      <div className="decision-quality-sections">
        <div className="decision-quality-section">
          <strong>Blocked Mechanisms</strong>
          {snapshot.blockedMechanisms.length > 0 ? (
            <div className="decision-quality-tags">
              {snapshot.blockedMechanisms.map((mechanism) => (
                <small key={mechanism}>{mechanism}</small>
              ))}
            </div>
          ) : (
            <div className="empty compact">No blocked mechanisms reported.</div>
          )}
        </div>

        <div className="decision-quality-section">
          <strong>Governor Outcomes</strong>
          {snapshot.exhaustedOutcomes.length > 0 ? (
            <div className="decision-quality-list">
              {snapshot.exhaustedOutcomes.map((row) => (
                <div className="decision-quality-row" key={`${row.status}-${row.mechanism}-${row.detail}`}>
                  <span>
                    <strong>{row.mechanism}</strong>
                    <small>{row.detail}</small>
                  </span>
                  <Badge value={row.status} />
                </div>
              ))}
            </div>
          ) : (
            <div className="empty compact">No exhausted governor outcomes reported.</div>
          )}
        </div>
      </div>

      {snapshot.warnings.length > 0 && (
        <div className="decision-quality-section warning">
          <strong>Trajectory Warnings</strong>
          <div className="decision-quality-list">
            {snapshot.warnings.slice(0, 4).map((warning) => (
              <div className="decision-quality-row" key={warning}>
                <span>
                  <strong>{warning}</strong>
                </span>
                <Badge value="WARN" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AgentInvocationAuditPanel({
  invocations,
  decisions,
}: {
  invocations: AgentInvocation[];
  decisions: AgentDecision[];
}) {
  const rows = buildAgentInvocationAuditRows(invocations, decisions);

  if (rows.length === 0) {
    return <div className="empty">Agent invocation metadata will appear after LLM calls are recorded.</div>;
  }

  return (
    <div className="agent-audit-list">
      {rows.map((row) => (
        <div className="agent-audit-row" key={row.id}>
          <div className="agent-audit-head">
            <span>
              <strong>{row.agentName}</strong>
              <small>{[row.createdAt ? new Date(row.createdAt).toLocaleString() : "", row.id].filter(Boolean).join(" - ")}</small>
              <small>{row.target}</small>
            </span>
            <Badge value={row.validationStatus} />
          </div>

          <div className="agent-audit-facts">
            {[
              { label: "API Style", value: row.apiStyle },
              { label: "Provider / Model", value: row.providerModel },
              { label: "Reasoning", value: row.reasoningEffort },
              { label: "Tool Rounds", value: row.toolRounds },
              { label: "Tool Names", value: row.toolNames.length > 0 ? row.toolNames.join(", ") : "-" },
              { label: "Decision", value: row.decisionLink },
            ].map((item) => (
              <span key={`${row.id}-${item.label}`}>
                <small>{item.label}</small>
                <strong title={item.value}>{item.value}</strong>
              </span>
            ))}
          </div>

          {row.validationError && (
            <div className="agent-audit-note error">
              <small>Validation Error</small>
              <span>{row.validationError}</span>
            </div>
          )}

          {row.rejectedToolCalls.length > 0 && (
            <div className="agent-audit-subsection rejected">
              <strong>Rejected Tool Calls</strong>
              <div className="agent-audit-result-list">
                {row.rejectedToolCalls.map((item) => (
                  <div className="agent-audit-result" key={`${row.id}-rejected-${item}`}>
                    <Badge value="REJECTED" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {row.dryRunValidationResults.length > 0 && (
            <div className="agent-audit-subsection">
              <strong>Dry-run Validation</strong>
              <div className="agent-audit-result-list">
                {row.dryRunValidationResults.map((item) => (
                  <div className="agent-audit-result" key={`${row.id}-dry-run-${item.status}-${item.text}`}>
                    <Badge value={item.status} />
                    <span>{item.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function MissionControlTelemetryPanel({
  telemetry,
  fallbackInvocations,
}: {
  telemetry: MissionControlTelemetryResponse | null;
  fallbackInvocations: AgentInvocation[];
}) {
  const summary = useMemo(
    () => buildMissionControlTelemetrySummary(telemetry, fallbackInvocations),
    [fallbackInvocations, telemetry],
  );

  if (!summary) {
    return <div className="empty">LLM token, prompt, and embedding telemetry will appear after agent calls are recorded.</div>;
  }

  return (
    <div className="telemetry-panel">
      <div className="insight-grid telemetry-window-grid">
        {summary.windows.map((window) => (
          <div className={`insight-card telemetry-window ${window.key}`} key={window.key}>
            <small>{window.label}</small>
            <strong>{formatCurrency(window.estimatedCostUsd)}</strong>
            <span>{window.calls} calls</span>
            <span>
              {formatTelemetryTokenPair(window.exactInputTokens, window.approxInputTokens)} in
              {window.exactOutputTokens > 0 || window.approxOutputTokens > 0
                ? ` / ${formatTelemetryTokenPair(window.exactOutputTokens, window.approxOutputTokens)} out`
                : ""}
            </span>
            <span>
              {window.cachedInputTokens > 0 ? `${formatCompactNumber(window.cachedInputTokens)} cached` : "no cached tokens"}
              {" - "}
              {window.reasoningTokens > 0 ? `${formatCompactNumber(window.reasoningTokens)} reasoning` : "no reasoning tokens"}
            </span>
            <span>
              {window.validCalls} valid / {window.invalidCalls} invalid
            </span>
          </div>
        ))}
      </div>

      <div className="telemetry-grid">
        <section className="telemetry-block">
          <strong>Calls by Agent</strong>
          <div className="telemetry-list">
            {summary.callsByAgent.length > 0 ? (
              summary.callsByAgent.map((row) => (
                <div className="telemetry-row" key={row.label}>
                  <span>
                    <strong>{row.label}</strong>
                    <small>
                      {row.count} calls - {formatTelemetryTokenPair(row.inputTokens, row.approxInputTokens)} in
                    </small>
                  </span>
                  <div className="telemetry-row-facts">
                    <span>{formatCurrency(row.estimatedCostUsd)}</span>
                    <span>{row.exactCalls} exact</span>
                    <span>{row.approxCalls} approx</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty compact">Agent calls will appear after telemetry data is available.</div>
            )}
          </div>
        </section>

        <section className="telemetry-block">
          <strong>Calls by Model</strong>
          <div className="telemetry-list">
            {summary.callsByModel.length > 0 ? (
              summary.callsByModel.map((row) => (
                <div className="telemetry-row" key={row.label}>
                  <span>
                    <strong>{row.label}</strong>
                    <small>
                      {row.count} calls - {formatTelemetryTokenPair(row.inputTokens, row.approxInputTokens)} in
                    </small>
                  </span>
                  <div className="telemetry-row-facts">
                    <span>{formatCurrency(row.estimatedCostUsd)}</span>
                    <span>{row.cachedInputTokens > 0 ? `${formatCompactNumber(row.cachedInputTokens)} cached` : "no cache"}</span>
                    <span>{row.reasoningTokens > 0 ? `${formatCompactNumber(row.reasoningTokens)} reasoning` : "no reasoning"}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty compact">Model split will appear after telemetry data is available.</div>
            )}
          </div>
        </section>
      </div>

      <div className="telemetry-grid">
        <section className="telemetry-block">
          <strong>Top Token-Heavy Invocations</strong>
          <div className="telemetry-list">
            {summary.topInvocations.length > 0 ? (
              summary.topInvocations.map((row) => (
                <div className="telemetry-invocation" key={row.id}>
                  <div className="telemetry-invocation-head">
                    <span>
                      <strong>{row.agentName}</strong>
                      <small>
                        {row.model || "unknown model"} - {new Date(row.createdAt).toLocaleString()}
                      </small>
                    </span>
                    <Badge value={row.usageKind === "exact" ? "EXACT" : "APPROX"} />
                  </div>
                  <div className="telemetry-invocation-facts">
                    <span>
                      <small>Prompt</small>
                      <strong>{formatTelemetryTokenPair(row.inputTokens, row.approxInputTokens)}</strong>
                    </span>
                    <span>
                      <small>Output</small>
                      <strong>{formatTelemetryTokenPair(row.outputTokens, row.approxOutputTokens)}</strong>
                    </span>
                    <span>
                      <small>Cached</small>
                      <strong>{formatCompactNumber(row.cachedInputTokens)}</strong>
                    </span>
                    <span>
                      <small>Reasoning</small>
                      <strong>{formatCompactNumber(row.reasoningTokens)}</strong>
                    </span>
                    <span>
                      <small>Cost</small>
                      <strong>{formatCurrency(row.estimatedCostUsd)}</strong>
                    </span>
                    <span>
                      <small>Prompt Size</small>
                      <strong>{formatBytes(row.promptBytes)}</strong>
                    </span>
                  </div>
                  <div className="telemetry-section-summary">
                    <small>{row.largestSection || "No section breakdown available"}</small>
                    {row.sections.length > 0 && (
                      <div className="tag-list">
                        {row.sections.slice(0, 3).map((section) => (
                          <small key={`${row.id}-${section.name}`}>
                            {section.name}: {formatCompactNumber(section.approxTokens)}t
                          </small>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty compact">Top invocations will appear once telemetry rows are loaded.</div>
            )}
          </div>
        </section>

        <section className="telemetry-block">
          <strong>Largest Prompt Sections</strong>
          <div className="telemetry-list">
            {summary.promptSections.length > 0 ? (
              summary.promptSections.map((row) => (
                <div className="telemetry-row" key={row.name}>
                  <span>
                    <strong>{row.name}</strong>
                    <small>
                      {row.calls} calls - {formatBytes(row.bytes)}
                    </small>
                  </span>
                  <div className="telemetry-row-facts">
                    <span>{formatCompactNumber(row.approxTokens)} tokens</span>
                    {row.exampleSource && <span title={row.exampleSource}>{row.exampleSource}</span>}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty compact">Prompt section estimates are unavailable for these invocations.</div>
            )}
          </div>
        </section>
      </div>

      <div className="telemetry-grid">
        <section className="telemetry-block">
          <strong>Embedding Source Index</strong>
          <div className="telemetry-list">
            {summary.embedding.sourceIndex.count > 0 ? (
              <>
                <div className="telemetry-row">
                  <span>
                    <strong>Total</strong>
                    <small>
                      {summary.embedding.sourceIndex.count} events - {formatCompactNumber(summary.embedding.sourceIndex.providerCalls)} provider call(s)
                    </small>
                  </span>
                  <div className="telemetry-row-facts">
                    <span>{formatBytes(summary.embedding.sourceIndex.inputBytes)}</span>
                    <span>{formatCurrency(summary.embedding.sourceIndex.estimatedCostUsd)}</span>
                    <span>{summary.embedding.sourceIndex.injected} injected</span>
                    <span>{summary.embedding.sourceIndex.skipped} skipped</span>
                  </div>
                </div>
                {summary.embedding.sourceIndex.bySourceTable.slice(0, 6).map((row) => (
                  <div className="telemetry-row" key={row.label}>
                    <span>
                      <strong>{row.label}</strong>
                      <small>{row.count} indexed source(s)</small>
                    </span>
                    <div className="telemetry-row-facts">
                      <span>{formatCompactNumber(row.providerCalls)} provider calls</span>
                      <span>{formatBytes(row.inputBytes)}</span>
                      <span>{formatCurrency(row.estimatedCostUsd)}</span>
                    </div>
                  </div>
                ))}
                {summary.embedding.sourceIndex.byModel.length > 0 && (
                  <div className="tag-list">
                    {summary.embedding.sourceIndex.byModel.slice(0, 4).map((row) => (
                      <small key={`${row.label}-model`}>
                        {row.label}: {row.count}
                      </small>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="empty compact">No source-index embedding usage events yet.</div>
            )}
          </div>
        </section>

        <section className="telemetry-block">
          <strong>Retrieval Query Usefulness</strong>
          <div className="telemetry-list">
            {summary.embedding.retrievalQuery.count > 0 ? (
              <>
                <div className="telemetry-row">
                  <span>
                    <strong>Total</strong>
                    <small>
                      {summary.embedding.retrievalQuery.count} checks - {formatCompactNumber(summary.embedding.retrievalQuery.providerCalls)} provider call(s)
                    </small>
                  </span>
                  <div className="telemetry-row-facts">
                    <span>{formatBytes(summary.embedding.retrievalQuery.inputBytes)}</span>
                    <span>{formatCurrency(summary.embedding.retrievalQuery.estimatedCostUsd)}</span>
                    <span>{summary.embedding.retrievalQuery.retrievedCount} retrieved</span>
                    <span>{summary.embedding.retrievalQuery.injected} injected</span>
                    <span>{summary.embedding.retrievalQuery.logOnly} log-only</span>
                    <span>{summary.embedding.retrievalQuery.cached} cached</span>
                    <span>{summary.embedding.retrievalQuery.skipped} skipped</span>
                  </div>
                </div>
                {summary.embedding.retrievalQuery.retrievalPurpose && (
                  <div className="telemetry-note">
                    <small>{summary.embedding.retrievalQuery.retrievalPurpose}</small>
                  </div>
                )}
                {summary.embedding.retrievalQuery.bySourceTable.length > 0 && (
                  <div className="telemetry-mini-list">
                    {summary.embedding.retrievalQuery.bySourceTable.slice(0, 6).map((row) => (
                      <div className="telemetry-mini-row" key={row.label}>
                        <strong>{row.label}</strong>
                        <small>
                          {row.count} checks - {row.retrievedCount} retrieved - {row.injected} injected
                        </small>
                        <span>
                          {formatCurrency(row.estimatedCostUsd)} - {formatBytes(row.inputBytes)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {summary.embedding.retrievalQuery.byModel.length > 0 && (
                  <div className="tag-list">
                    {summary.embedding.retrievalQuery.byModel.slice(0, 4).map((row) => (
                      <small key={`${row.label}-retrieval-model`}>
                        {row.label}: {row.count}
                      </small>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="empty compact">No retrieval-query embedding telemetry has been recorded yet.</div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function VisualAnalysisPanel({
  dataset,
  jobs,
  loading,
  visualAnalysis,
  onRequestRerun,
}: {
  dataset: Dataset | null;
  jobs: Job[];
  loading: boolean;
  visualAnalysis: VisualAnalysisDetail;
  onRequestRerun: () => void;
}) {
  const analysis = visualAnalysis.analysis;
  const activeJob = visualAnalysisActiveJob(jobs, dataset?.id ?? "");
  const coverage = recordObject(analysis?.coverage_report);
  const facts = visualAnalysisFacts(visualAnalysis, activeJob);
  const traits = Array.isArray(analysis?.visual_traits) ? analysis.visual_traits.slice(0, 6) : [];
  const hypotheses = Array.isArray(analysis?.preprocessing_hypotheses)
    ? analysis.preprocessing_hypotheses.slice(0, 6)
    : [];
  const cautions = Array.isArray(analysis?.cautions) ? analysis.cautions.slice(0, 6) : [];
  const classesToWatch = Array.isArray(analysis?.classes_to_watch) ? analysis.classes_to_watch.slice(0, 4) : [];
  const limitations = visualAnalysisLimitations(analysis);
  const validationErrors = stringArrayPayload(analysis?.validation_errors);
  const rerunDisabledReason = visualAnalysisRerunDisabledReason(visualAnalysis, dataset, activeJob, loading);
  const canRerun = !rerunDisabledReason;
  const datasetLabel = analysis?.dataset_name || dataset?.name || "No dataset selected";

  return (
    <div className="visual-analysis-panel">
      <div className="visual-analysis-head">
        <span>
          <strong>{datasetLabel}</strong>
          <small>{visualAnalysis.message}</small>
        </span>
        <div className="visual-analysis-actions">
          <Badge value={visualAnalysisStatusBadge(visualAnalysis, activeJob)} />
          <button
            className="command compact"
            type="button"
            onClick={onRequestRerun}
            disabled={!canRerun}
            title={rerunDisabledReason || "Request a manual visual-analysis rerun"}
          >
            <RefreshCcw size={15} />
            Manual Rerun
          </button>
        </div>
      </div>

      <div className="review-state wait visual-advisory">
        <Badge value="EVIDENCE_ONLY" />
        <span>
          <strong>Advisory visual evidence</strong>
          <small>Hypotheses below are observations for backend validation; they are not approved experiments or runnable config.</small>
        </span>
      </div>

      {visualAnalysis.rerunPolicy && (
        <div className="review-state wait visual-rerun-policy">
          <Badge value={visualAnalysis.rerunPolicy.manual_run_allowed === false ? "RERUN_BLOCKED" : "RERUN_READY"} />
          <span>
            <strong>Backend rerun policy</strong>
            <small>{visualAnalysisPolicySummary(visualAnalysis.rerunPolicy)}</small>
          </span>
        </div>
      )}

      <div className="insight-grid visual-facts">
        {facts.map((item) => (
          <div className={`insight-card ${item.tone ?? ""}`} key={item.label}>
            <small>{item.label}</small>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

      {!analysis ? (
        <div className="empty compact">
          {activeJob
            ? `Visual analysis job ${activeJob.status.toLowerCase()} for this dataset.`
            : visualAnalysis.message}
        </div>
      ) : (
        <>
          <div className="visual-analysis-grid">
            <div className="visual-analysis-block">
              <strong>Coverage</strong>
              <p>{visualCoverageSummary(analysis)}</p>
              {stringArrayPayload(coverage.selection_basis).length > 0 && (
                <div className="tag-list">
                  {stringArrayPayload(coverage.selection_basis).slice(0, 8).map((item) => (
                    <small key={item}>{item}</small>
                  ))}
                </div>
              )}
              {visualPerClassCoverageRows(coverage).length > 0 && (
                <div className="visual-mini-table">
                  {visualPerClassCoverageRows(coverage).map((row) => (
                    <span key={row.label}>
                      <small>{row.label}</small>
                      <strong>{row.value}</strong>
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="visual-analysis-block">
              <strong>Classes To Watch</strong>
              {classesToWatch.length > 0 ? (
                <div className="visual-list">
                  {classesToWatch.map((item, index) => (
                    <span key={`${item.class_name || "class"}-${index}`}>
                      <strong>{item.class_name || "class"}</strong>
                      <small>{[item.confidence, item.reason].filter(Boolean).join(" - ") || "watch item"}</small>
                    </span>
                  ))}
                </div>
              ) : (
                <div className="empty compact">No class watch items reported.</div>
              )}
            </div>
          </div>

          <div className="visual-section">
            <strong>Visual Traits</strong>
            {traits.length > 0 ? (
              <div className="visual-card-grid">
                {traits.map((trait, index) => (
                  <div className="visual-card" key={`${trait.trait || "trait"}-${index}`}>
                    <div className="visual-card-head">
                      <span>
                        <strong>{trait.trait || "visual trait"}</strong>
                        <small>{[trait.level, trait.confidence].filter(Boolean).join(" confidence ")}</small>
                      </span>
                      {trait.confidence && <Badge value={trait.confidence} />}
                    </div>
                    {stringArrayPayload(trait.evidence).slice(0, 2).map((item) => (
                      <p key={item}>{item}</p>
                    ))}
                    {trait.notes && <small>{trait.notes}</small>}
                    {stringArrayPayload(trait.affected_classes).length > 0 && (
                      <div className="tag-list">
                        {stringArrayPayload(trait.affected_classes).slice(0, 5).map((item) => (
                          <small key={item}>{item}</small>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty compact">No visual traits reported by the visual agent yet.</div>
            )}
          </div>

          <div className="visual-section">
            <strong>Preprocessing Hypotheses</strong>
            {hypotheses.length > 0 ? (
              <div className="visual-hypothesis-list">
                {hypotheses.map((hypothesis, index) => (
                  <div className="visual-hypothesis" key={hypothesis.id || `${hypothesis.mechanism}-${index}`}>
                    <div className="visual-card-head">
                      <span>
                        <strong>{hypothesis.summary || hypothesis.mechanism || "hypothesis"}</strong>
                        <small>{[hypothesis.mechanism, hypothesis.confidence].filter(Boolean).join(" - ")}</small>
                      </span>
                      <Badge value={hypothesis.support_status || "needs_backend_validation"} />
                    </div>
                    {hypothesis.expected_effect && <p>{hypothesis.expected_effect}</p>}
                    <div className="visual-mini-table">
                      {hypothesis.suggested_image_sizes && hypothesis.suggested_image_sizes.length > 0 && (
                        <span>
                          <small>Image Sizes</small>
                          <strong>{hypothesis.suggested_image_sizes.join(", ")}</strong>
                        </span>
                      )}
                      {hypothesis.suggested_augmentation_policy && (
                        <span>
                          <small>Augmentation</small>
                          <strong>{hypothesis.suggested_augmentation_policy}</strong>
                        </span>
                      )}
                      {hypothesis.suggested_preprocessing && (
                        <span>
                          <small>Preprocessing</small>
                          <strong>{objectSummary(hypothesis.suggested_preprocessing)}</strong>
                        </span>
                      )}
                      {hypothesis.risk && (
                        <span>
                          <small>Risk</small>
                          <strong>{hypothesis.risk}</strong>
                        </span>
                      )}
                    </div>
                    {stringArrayPayload(hypothesis.evidence).length > 0 && (
                      <div className="visual-evidence-list">
                        {stringArrayPayload(hypothesis.evidence).slice(0, 3).map((item) => (
                          <small key={item}>{item}</small>
                        ))}
                      </div>
                    )}
                    {(hypothesis.unsupported_reason || hypothesis.support_status !== "supported") && (
                      <p className="visual-warning">
                        {hypothesis.unsupported_reason || "Requires backend validation before any experiment can be scheduled."}
                      </p>
                    )}
                    <small className="visual-advisory-note">Not an approved experiment.</small>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty compact">No preprocessing hypotheses reported.</div>
            )}
          </div>

          <div className="visual-analysis-grid">
            <div className="visual-analysis-block">
              <strong>Cautions</strong>
              {cautions.length > 0 ? (
                <div className="visual-list caution">
                  {cautions.map((caution, index) => (
                    <span key={`${caution.operation || "operation"}-${index}`}>
                      <strong>{caution.operation || "operation caution"}</strong>
                      <small>{[caution.severity, caution.confidence, caution.reason].filter(Boolean).join(" - ")}</small>
                    </span>
                  ))}
                </div>
              ) : (
                <div className="empty compact">No visual cautions reported.</div>
              )}
            </div>

            <div className="visual-analysis-block">
              <strong>Limitations</strong>
              {limitations.length > 0 ? (
                <div className="warning-list">
                  {limitations.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
              ) : (
                <div className="empty compact">No limitations reported beyond the bounded sample.</div>
              )}
            </div>
          </div>

          {validationErrors.length > 0 && (
            <div className="rejection-panel">
              <strong>Validation Errors</strong>
              <div className="rejection-list">
                {validationErrors.map((item) => (
                  <span key={item}>
                    <small>Visual analysis validation</small>
                    {item}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ChampionExportDemoPanel({
  data,
  prediction,
  predictionError,
  predictionLoading,
  selectedImageIndex,
  customImage,
  customImageURI,
  customTrueLabel,
  localInferenceStatus,
  localInferenceError,
  slideshowEnabled,
  slideshowIntervalMs,
  detectionConfidenceThreshold,
  detectionIouThreshold,
  onCustomImageURIChange,
  onCustomTrueLabelChange,
  onChooseCustomImage,
  onRunCustomPrediction,
  onToggleSlideshow,
  onSelectImage,
  onNextImage,
  onRandomImage,
  onRequestExport,
  onRunPrediction,
  onOpenFeedback,
  onSlideshowIntervalChange,
  onDetectionConfidenceThresholdChange,
  onDetectionIouThresholdChange,
}: {
  data: ChampionExportDemo;
  prediction: ChampionDemoPrediction | null;
  predictionError: string;
  predictionLoading: boolean;
  selectedImageIndex: number;
  customImage: ChampionDemoImage | null;
  customImageURI: string;
  customTrueLabel: string;
  localInferenceStatus: string;
  localInferenceError: string;
  slideshowEnabled: boolean;
  slideshowIntervalMs: number;
  detectionConfidenceThreshold: number;
  detectionIouThreshold: number;
  onCustomImageURIChange: (value: string) => void;
  onCustomTrueLabelChange: (value: string) => void;
  onChooseCustomImage: () => void;
  onRunCustomPrediction: () => void;
  onToggleSlideshow: () => void;
  onSelectImage: (index: number) => void;
  onNextImage: () => void;
  onRandomImage: () => void;
  onRequestExport: () => void;
  onRunPrediction: (image: ChampionDemoImage) => void;
  onOpenFeedback: (rating: ChampionFeedbackRating) => void;
  onSlideshowIntervalChange: (value: number) => void;
  onDetectionConfidenceThresholdChange: (value: number) => void;
  onDetectionIouThresholdChange: (value: number) => void;
}) {
  if (!data.hasChampion) {
    return <div className="empty">Champion export and demo details will appear after the backend selects a champion.</div>;
  }

  const selectedImage = data.demoImages[selectedImageIndex] ?? data.demoImages[0] ?? null;
  const customURI = customImageURI.trim();
  const customImageMatchesPicker = customImage ? customURI === demoImageURI(customImage) : false;
  const customPreviewImage = customImageURI.trim()
    ? ({
        ...(customImage ?? {}),
        uri: customURI,
        image_uri: customURI,
        thumbnail_uri: customImageMatchesPicker ? customImage?.thumbnail_uri : undefined,
        true_label: customTrueLabel.trim() || customImage?.true_label || customImage?.label || customImage?.class_name,
        split: customImage?.split || "custom",
      } satisfies ChampionDemoImage)
    : null;
  const activeImage = customPreviewImage ?? selectedImage;
  const activePreviewURI = demoImagePreviewURI(activeImage);
  const activeImageLabel = demoImageLabel(activeImage);
  const detectorDemo = championExportDemoIsDetection(data) || detectionBoxesFromPrediction(prediction).length > 0;
  const activeDetections = detectionBoxesFromPrediction(prediction);
  const activeFps = prediction?.latency_ms && prediction.latency_ms > 0 ? 1000 / prediction.latency_ms : 0;
  const postprocessLatency = predictionPostprocessLatency(prediction);

  return (
    <div className="export-demo-panel">
      <div className="export-demo-grid">
        <div className="export-block">
          <strong>Export Status</strong>
          <div className="export-status-line">
            <Badge value={data.exportStatus || "PENDING"} />
            <small>{data.exports.length > 0 ? `${data.exports.length} export record(s)` : "No export records exposed yet."}</small>
            <button className="command compact" type="button" onClick={onRequestExport}>
              <HardDriveUpload size={15} />
              Request ONNX
            </button>
          </div>
          {data.exports.length > 0 ? (
            <div className="export-record-list">
              {data.exports.slice(0, 4).map((exportRecord, index) => (
                <div
                  className={`export-record ${statusToneClass(exportRecord.status)}`}
                  key={exportRecord.id || `${exportRecord.format}-${index}`}
                >
                  <span>
                    <strong>{exportRecord.format || "model artifact"}</strong>
                    <small>
                      {exportRecord.artifact_uri ||
                        exportRecord.model_uri ||
                        exportRecord.download_url ||
                        exportStatusMessage(exportRecord.status)}
                    </small>
                  </span>
                  <span>
                    <Badge value={exportRecord.status || "PENDING"} />
                    <small>
                      {exportRecord.size_bytes
                        ? formatBytes(exportRecord.size_bytes)
                        : exportRecord.completed_at ||
                          exportRecord.failed_at ||
                          exportRecord.updated_at ||
                          exportRecord.started_at ||
                          exportRecord.requested_at ||
                          exportRecord.created_at ||
                          ""}
                    </small>
                  </span>
                  {(exportRecord.error || exportRecord.error_message || (exportRecord.validation_errors ?? []).length > 0) && (
                    <p>{exportRecord.error || exportRecord.error_message || exportRecord.validation_errors?.join("; ")}</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="empty compact">No export request has been recorded for this champion yet.</div>
          )}
        </div>

        <div className="export-block">
          <strong>Use Case Fit</strong>
          <div className="recommendation-list">
            {data.useCases.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
          <strong>Preprocessing Contract</strong>
          <div className="recommendation-list">
            {data.preprocessing.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
          {data.limitations.length > 0 && (
            <>
              <strong>Limitations</strong>
              <div className="warning-list">
                {data.limitations.map((item) => (
                  <span key={item}>{item}</span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="champion-test-bench">
        <div className="test-image-stage">
          <div className="test-image-preview">
            {activePreviewURI ? (
              <div className="test-image-frame">
                <img src={activePreviewURI} alt={activeImageLabel || "test image"} />
                {activeDetections.length > 0 && <DetectionOverlay detections={activeDetections} />}
              </div>
            ) : (
              <div className="test-image-placeholder">
                <ImageIcon size={28} />
                <span>No image</span>
              </div>
            )}
          </div>
          <div className="test-image-meta">
            <span>
              <Badge value={activeImage?.split || "TEST"} />
              <strong>{activeImageLabel || activeImage?.image_id || "Select an image"}</strong>
            </span>
            <small>{demoImageDetail(activeImage) || "Held-out image or custom worker-visible URI"}</small>
          </div>
        </div>

        <div className="test-controls">
          <div className="demo-block-head">
            <strong>Champion Test</strong>
            <span>
              <Badge value={localInferenceStatus === "ready" || localInferenceStatus === "available" ? "LOCAL_ONNX" : localInferenceStatus === "loading" ? "LOADING_ONNX" : "WORKER_FALLBACK"} />
              <button className="command compact" type="button" onClick={onToggleSlideshow} disabled={data.demoImages.length < 2 || predictionLoading}>
                {slideshowEnabled ? <Pause size={15} /> : <Play size={15} />}
                {slideshowEnabled ? "Pause" : "Slideshow"}
              </button>
              <button className="icon-command" type="button" onClick={onRandomImage} disabled={data.demoImages.length < 2} title="Random held-out image">
                <Shuffle size={14} />
              </button>
              <button className="icon-command" type="button" onClick={onNextImage} disabled={data.demoImages.length < 2} title="Next held-out image">
                <StepForward size={14} />
              </button>
              <button
                className="command compact"
                type="button"
                onClick={() => selectedImage && onRunPrediction(selectedImage)}
                disabled={!selectedImage || predictionLoading}
              >
                <Play size={15} />
                Predict Held-out
              </button>
            </span>
          </div>

          <div className="custom-image-actions">
            <button className="command compact" type="button" onClick={onChooseCustomImage}>
              <Upload size={15} />
              Choose Image
            </button>
            <button className="command primary compact" type="button" onClick={onRunCustomPrediction} disabled={!customImageURI.trim() || predictionLoading}>
              <Play size={15} />
              Predict Custom
            </button>
          </div>

          <label className="field">
            <span><Link2 size={12} /> Image URI</span>
            <input
              value={customImageURI}
              onChange={(event) => onCustomImageURIChange(event.target.value)}
              placeholder="file://, s3://, or worker-visible path"
            />
          </label>
          <label className="field">
            <span>True label</span>
            <input
              value={customTrueLabel}
              onChange={(event) => onCustomTrueLabelChange(event.target.value)}
              placeholder="optional"
            />
          </label>
          <label className="field compact-range">
            <span><Timer size={12} /> Speed</span>
            <input
              type="range"
              min={1200}
              max={10000}
              step={400}
              value={slideshowIntervalMs}
              onChange={(event) => onSlideshowIntervalChange(Number(event.target.value))}
            />
            <small>{(slideshowIntervalMs / 1000).toFixed(1)}s</small>
          </label>
          {detectorDemo && (
            <div className="detector-controls">
              <label className="field compact-range">
                <span>Confidence</span>
                <input
                  type="range"
                  min={0.01}
                  max={0.95}
                  step={0.01}
                  value={detectionConfidenceThreshold}
                  onChange={(event) => onDetectionConfidenceThresholdChange(Number(event.target.value))}
                />
                <small>{formatTopKScore(detectionConfidenceThreshold)}</small>
              </label>
              <label className="field compact-range">
                <span>IoU</span>
                <input
                  type="range"
                  min={0.1}
                  max={0.95}
                  step={0.01}
                  value={detectionIouThreshold}
                  onChange={(event) => onDetectionIouThresholdChange(Number(event.target.value))}
                />
                <small>{formatTopKScore(detectionIouThreshold)}</small>
              </label>
            </div>
          )}
        </div>

        <div className="test-result-panel">
          <div className="demo-block-head">
            <strong>Prediction Result</strong>
            {predictionLoading && <Badge value="RUNNING" />}
          </div>
          {predictionError && (
            <div className="warning-list">
              <span>{predictionError}</span>
            </div>
          )}
          {localInferenceError && (
            <div className="warning-list">
              <span>{localInferenceError}</span>
            </div>
          )}
          {predictionLoading && <div className="empty compact">{readyONNXExport(data.exports) ? "Running local ONNX inference..." : "Waiting for inference..."}</div>}
          {prediction ? (
            <>
              <PredictionRow prediction={prediction} index={0} />
              {detectorDemo && (
                <div className="detector-live-stats">
                  <span>
                    <small>Detections</small>
                    <strong>{activeDetections.length}</strong>
                  </span>
                  <span>
                    <small>FPS</small>
                    <strong>{activeFps ? activeFps.toFixed(1) : "-"}</strong>
                  </span>
                  <span>
                    <small>Postprocess</small>
                    <strong>{postprocessLatency ? formatLatency(postprocessLatency) : "-"}</strong>
                  </span>
                </div>
              )}
              <div className="feedback-actions" aria-label="Champion feedback">
                <button className="command compact" type="button" onClick={() => onOpenFeedback("good")} disabled={predictionLoading} title="Mark champion output good">
                  <ThumbsUp size={15} />
                  Good
                </button>
                <button className="command compact" type="button" onClick={() => onOpenFeedback("mediocre")} disabled={predictionLoading} title="Mark champion output mediocre">
                  <MessageSquare size={15} />
                  Mediocre
                </button>
                <button className="command compact" type="button" onClick={() => onOpenFeedback("bad")} disabled={predictionLoading} title="Mark champion output bad">
                  <ThumbsDown size={15} />
                  Bad
                </button>
              </div>
            </>
          ) : (
            <div className="empty compact">Run a held-out or custom image to see the champion prediction.</div>
          )}
        </div>
      </div>

      <div className="demo-grid">
        <div className="demo-block">
          <div className="demo-block-head">
            <strong>Held-out Images</strong>
            <span>
              <button className="icon-command" type="button" onClick={onRandomImage} disabled={data.demoImages.length < 2} title="Random demo image">
                <Shuffle size={14} />
              </button>
              <button className="icon-command" type="button" onClick={onNextImage} disabled={data.demoImages.length < 2} title="Next demo image">
                <StepForward size={14} />
              </button>
              <button
                className="command compact"
                type="button"
                onClick={() => selectedImage && onRunPrediction(selectedImage)}
                disabled={!selectedImage || predictionLoading}
              >
                <Play size={15} />
                Predict
              </button>
            </span>
          </div>
          {data.demoImages.length > 0 ? (
            <div className="demo-image-list">
              {data.demoImages.slice(0, 6).map((image, index) => (
                <button
                  className={`demo-image-row ${index === selectedImageIndex ? "selected" : ""}`}
                  key={image.id || image.image_id || `${image.uri}-${index}`}
                  type="button"
                  onClick={() => onSelectImage(index)}
                >
                  {demoImagePreviewURI(image) ? (
                    <img src={demoImagePreviewURI(image)} alt={demoImageLabel(image) || "demo image"} />
                  ) : (
                    <div className="demo-image-placeholder">image</div>
                  )}
                  <span>
                    <strong>{demoImageLabel(image) || image.image_id || "unlabeled"}</strong>
                    <small>{demoImageDetail(image) || "image metadata pending"}</small>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty compact">No held-out/test demo images are exposed by the backend yet.</div>
          )}
        </div>

        <div className="demo-block">
          <strong>Prediction History</strong>
          {data.demoPredictions.length > 0 ? (
            <div className="prediction-list">
              {data.demoPredictions.slice(0, 6).map((predictionRow, index) => (
                <PredictionRow
                  key={predictionRow.id || `${predictionRow.image_id}-${index}`}
                  prediction={predictionRow}
                  index={index}
                />
              ))}
            </div>
          ) : (
            <div className="empty compact">Prediction history will appear if the backend exposes durable demo predictions.</div>
          )}
          {data.feedback.length > 0 && (
            <>
              <strong>Feedback</strong>
              <div className="feedback-history">
                {data.feedback.slice(0, 4).map((item) => (
                  <div className={`feedback-row feedback-${item.rating}`} key={item.id}>
                    <Badge value={feedbackRatingLabel(item.rating)} />
                    <span>
                      <strong>{item.message || "No note added"}</strong>
                      <small>{[item.image_id || item.prediction_id || item.job_id || "", item.created_at || ""].filter(Boolean).join(" - ")}</small>
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function DetectionOverlay({ detections }: { detections: ChampionDetection[] }) {
  const boxes = detections
    .map((detection) => ({ detection, box: normalizedDetectionBox(detection) }))
    .filter((item): item is { detection: ChampionDetection; box: { x: number; y: number; width: number; height: number } } => Boolean(item.box))
    .slice(0, 60);
  if (boxes.length === 0) return null;
  return (
    <div className="detection-overlay" aria-hidden="true">
      {boxes.map(({ detection, box }, index) => {
        const label = detection.label || detection.class_name || `class_${detection.class_id ?? index}`;
        const confidence = Number(detection.confidence ?? detection.score ?? 0);
        const labelBelow = box.y < 0.08;
        return (
          <div
            className={`detection-box${labelBelow ? " label-below" : ""}`}
            key={`${label}-${index}-${box.x}-${box.y}`}
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.width * 100}%`,
              height: `${box.height * 100}%`,
            }}
          >
            <span>
              {label} {formatTopKScore(confidence)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function PredictionRow({ prediction, index }: { prediction: ChampionDemoPrediction; index: number }) {
  const status = normalizedStatus(prediction.status || (prediction.runtime_unavailable ? "RUNTIME_UNAVAILABLE" : "PENDING"));
  const displayLabel = prediction.predicted_label || predictionStatusMessage(status);
  const confidence = numericValue(prediction.confidence);
  const topK = Array.isArray(prediction.top_k) ? prediction.top_k : [];
  const detections = detectionBoxesFromPrediction(prediction);
  const imageMetadata = { ...recordObject(prediction.image_metadata), ...recordObject(prediction.metadata) };
  const imageSrc =
    recordString(imageMetadata, "thumbnail_uri") ||
    recordString(imageMetadata, "preview_uri") ||
    prediction.image_uri;
  const timestamp =
    prediction.completed_at || prediction.updated_at || prediction.started_at || prediction.requested_at || prediction.created_at || "";

  return (
    <div className={`prediction-row ${statusToneClass(status)}`}>
      {imageSrc ? (
        <img className="prediction-thumb" src={imageSrc} alt={prediction.true_label || prediction.predicted_label || "prediction image"} />
      ) : (
        <div className="prediction-thumb placeholder">image</div>
      )}
      <span>
        <strong>{displayLabel}</strong>
        <small>
          {[
            status,
            prediction.true_label ? `true: ${prediction.true_label}` : "true label pending",
            prediction.image_id || prediction.image_uri || "",
          ]
            .filter(Boolean)
            .join(" - ")}
        </small>
      </span>
      <span className="prediction-result-stack">
        <Badge value={status} />
        {typeof prediction.correct === "boolean" && <Badge value={prediction.correct ? "CORRECT" : "MISSED"} />}
        <small>{timestamp}</small>
      </span>
      <div className="prediction-facts">
        <span>
          <small>Confidence</small>
          <strong>{confidence ? formatTopKScore(confidence) : "-"}</strong>
        </span>
        <span>
          <small>Latency</small>
          <strong>{typeof prediction.latency_ms === "number" ? formatLatency(prediction.latency_ms) : "-"}</strong>
        </span>
        <span>
          <small>Correctness</small>
          <strong>{typeof prediction.correct === "boolean" ? (prediction.correct ? "correct" : "missed") : "-"}</strong>
        </span>
      </div>
      {(prediction.error || prediction.error_message) && <p>{prediction.error || prediction.error_message}</p>}
      {topK.length > 0 && (
        <div className="topk-list">
          {topK.slice(0, 5).map((item, topIndex) => (
            <small key={`${prediction.id || index}-${topIndex}`}>
              {item.label || item.class_name || "class"} {formatTopKScore(item.confidence ?? item.probability ?? item.score)}
            </small>
          ))}
        </div>
      )}
      {detections.length > 0 && (
        <div className="detection-chip-list">
          {detections.slice(0, 6).map((detection, detectionIndex) => (
            <small key={`${prediction.id || index}-det-${detectionIndex}`}>
              {detection.label || detection.class_name || `class_${detection.class_id ?? detectionIndex}`}{" "}
              {formatTopKScore(detection.confidence ?? detection.score)}
            </small>
          ))}
        </div>
      )}
    </div>
  );
}

function RunEvaluationDetails({ evaluation }: { evaluation: TrainingRunEvaluation }) {
  const diagnostics = recordObject(evaluation.holistic_scores.training_diagnostics);
  const perClassRows = perClassMetricRows(evaluation.per_class_metrics);
  const matrix = normalizedConfusionMatrix(evaluation.confusion_matrix);

  return (
    <div className="evaluation-details">
      <div className="evaluation-card">
        <strong>Training diagnostics</strong>
        <div className="evaluation-facts">
          <span>
            <small>Status</small>
            <Badge value={recordString(diagnostics, "status") || recordString(evaluation.holistic_scores, "divergence_status") || "stable"} />
          </span>
          <span>
            <small>Loss gap</small>
            <b>{formatLossGap(numberPayload(diagnostics.train_validation_gap) ?? numberPayload(evaluation.holistic_scores.train_validation_gap))}</b>
          </span>
          <span>
            <small>Severity</small>
            <b>{formatMetricNumber(numberPayload(diagnostics.severity))}</b>
          </span>
        </div>
      </div>

      {perClassRows.length > 0 && (
        <div className="evaluation-card">
          <strong>Per-class metrics</strong>
          <div className="per-class-table">
            <div className="per-class-row per-class-head">
              <span>Class</span>
              <span>Prec</span>
              <span>Rec</span>
              <span>F1</span>
            </div>
            {perClassRows.slice(0, 6).map((row) => (
              <div className="per-class-row" key={row.label}>
                <span>{row.label}</span>
                <span>{formatMetricNumber(row.precision)}</span>
                <span>{formatMetricNumber(row.recall)}</span>
                <span>{formatMetricNumber(row.f1)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {matrix.length > 0 && (
        <div className="evaluation-card">
          <strong>Confusion matrix</strong>
          <div className="confusion-preview">
            {matrix.slice(0, 6).map((row, rowIndex) => (
              <div key={`selected-run-matrix-${rowIndex}`}>
                {row.slice(0, 6).map((value, colIndex) => (
                  <span key={`${rowIndex}-${colIndex}`}>{value}</span>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
  return (
    <div className="metric-card">
      <span>{icon}</span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function Badge({ value }: { value: string }) {
  return <span className={`badge ${value.toLowerCase().replace(/[^a-z0-9_-]+/g, "_")}`}>{value}</span>;
}

function MetricChart({ metrics, metricKey, label }: { metrics: EpochMetric[]; metricKey: MetricKey; label: string }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (metrics.length === 0) {
    return <div className="empty chart-empty">No metrics reported</div>;
  }

  const rows = metrics
    .map((metric) => ({ epoch: metric.epoch, value: metric.metrics[metricKey] }))
    .filter((metric): metric is { epoch: number; value: number } => typeof metric.value === "number" && Number.isFinite(metric.value));
  if (rows.length === 0) {
    return <div className="empty chart-empty">No {label} values reported</div>;
  }

  const values = rows.map((metric) => metric.value);
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const range = Math.max(maxValue - minValue, 0.001);

  const width = 760;
  const height = 240;
  const padding = 28;
  const points = rows.map((row, index) => {
    const x =
      rows.length === 1
        ? width / 2
        : padding + (index / (rows.length - 1)) * (width - padding * 2);
    const y = height - padding - ((row.value - minValue) / range) * (height - padding * 2);
    return { x, y, value: row.value, epoch: row.epoch };
  });

  const latest = points[points.length - 1];
  const hovered = hoveredIndex === null ? null : points[hoveredIndex];
  const tooltipWidth = 128;
  const tooltipHeight = 58;
  const tooltipX = hovered
    ? Math.min(Math.max(hovered.x - tooltipWidth / 2, padding), width - padding - tooltipWidth)
    : 0;
  const tooltipY = hovered ? Math.max(8, hovered.y - tooltipHeight - 12) : 0;

  return (
    <div className="chart-wrap">
      <div className="chart-stat">
        <span>{label}</span>
        <strong>{formatChartValue(latest.value)}</strong>
      </div>
      <svg className="metric-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} chart`}>
        <defs>
          <linearGradient id="metric-fill-up" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#00d47e" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#00d47e" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="metric-fill-down" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#ff5967" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#ff5967" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3].map((line) => {
          const y = padding + (line / 3) * (height - padding * 2);
          return <line key={line} className="chart-grid" x1={padding} x2={width - padding} y1={y} y2={y} />;
        })}
        {points.slice(1).map((point, index) => {
          const previous = points[index];
          const direction = point.value >= previous.value ? "up" : "down";
          const baseline = height - padding;
          const fillPath = [
            `M ${previous.x.toFixed(2)} ${baseline.toFixed(2)}`,
            `L ${previous.x.toFixed(2)} ${previous.y.toFixed(2)}`,
            `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`,
            `L ${point.x.toFixed(2)} ${baseline.toFixed(2)}`,
            "Z",
          ].join(" ");

          return <path key={`fill-${previous.epoch}-${point.epoch}`} className={`chart-fill ${direction}`} d={fillPath} />;
        })}
        {points.slice(1).map((point, index) => {
          const previous = points[index];
          const direction = point.value >= previous.value ? "up" : "down";
          return (
            <line
              key={`${previous.epoch}-${point.epoch}`}
              className={`chart-segment ${direction}`}
              x1={previous.x}
              y1={previous.y}
              x2={point.x}
              y2={point.y}
            />
          );
        })}
        {points.map((point, index) => (
          <g key={point.epoch}>
            <circle
              className={hoveredIndex === index ? "chart-dot active" : "chart-dot"}
              cx={point.x}
              cy={point.y}
              r="4"
            />
            <circle
              className="chart-hit"
              cx={point.x}
              cy={point.y}
              r="15"
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
            <text className="chart-label" x={point.x} y={height - 7} textAnchor="middle">
              {point.epoch}
            </text>
          </g>
        ))}
        {hovered && (
          <g className="chart-tooltip" transform={`translate(${tooltipX} ${tooltipY})`}>
            <rect width={tooltipWidth} height={tooltipHeight} rx="7" />
            <text x="10" y="18">
              epoch {hovered.epoch}
            </text>
            <text x="10" y="38" className="chart-tooltip-value">
              {formatChartValue(hovered.value)}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

function formatBytes(value: number) {
  if (!value) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function summarizeTrainingRuns(
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
    activeRuns: summaries.filter((summary) => ["RUNNING", "ASSIGNED", "QUEUED"].includes(summary.status)).length,
  };
}

function trainingRunCacheSummary(summary: TrainingRunSummary) {
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

function trainingRunLifecycleChips(summary: TrainingRunSummary) {
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

function lifecycleSecondsChip(label: string, seconds: number) {
  return seconds > 0 ? `${label} ${formatSeconds(seconds)}` : "";
}

function workerRequirementMaterializationSummary(requirement: WorkerRequirement) {
  const parts = [
    requirement.dataset_materialization_status ? humanizeAuditKey(requirement.dataset_materialization_status) : "",
    requirement.max_concurrent_jobs ? `${requirement.max_concurrent_jobs} concurrent` : "",
    requirement.max_cold_dataset_materializations ? `${requirement.max_cold_dataset_materializations} cold` : "",
    requirement.dataset_cache_key ? shortCacheKey(requirement.dataset_cache_key) : "",
  ].filter(Boolean);
  return parts.join(" / ");
}

function workerRequirementHasOpenWork(requirement: WorkerRequirement, jobs: Job[]) {
  return jobs.some((job) => {
    const status = normalizedStatus(job.status);
    if (!["QUEUED", "ASSIGNED", "RUNNING"].includes(status)) return false;
    const planId = recordString(job.config, "plan_id");
    if (requirement.plan_id && planId !== requirement.plan_id) return false;
    const provider = recordString(job.config, "provider");
    if (requirement.provider && provider && provider !== requirement.provider) return false;
    return true;
  });
}

function shortCacheKey(value: string) {
  const text = String(value || "").trim();
  if (text.length <= 18) return text;
  const [prefix, rest] = text.split("-", 2);
  if (prefix && rest) return `${prefix}-${rest.slice(0, 8)}`;
  return text.slice(0, 12);
}

function projectTabFromTarget(tab: ProjectTabTarget): ProjectTabKey {
  if (tab === "overview") return "mission";
  if (tab === "agents") return "activity";
  if (tab === "experiments") return "results";
  if (tab === "data" || tab === "operations") return "developer";
  return tab;
}

function missionStateLabelFromProject(project: Project) {
  const status = normalizedStatus(project.status || "");
  if (status === "COMPLETED") return "Completed";
  if (status === "FAILED" || status === "BLOCKED") return "Needs input";
  if (status === "RUNNING" || status === "ACTIVE") return "In progress";
  return "Ready";
}

function missionStateToneFromProject(project: Project) {
  const status = normalizedStatus(project.status || "");
  if (status === "COMPLETED") return "done";
  if (status === "FAILED" || status === "BLOCKED") return "blocked";
  if (status === "RUNNING" || status === "ACTIVE") return "active";
  return "ready";
}

function buildMissionBrief(
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

function buildCurrentThinking(project: Project | null, detail: ProjectDetail, digest: MissionDigest): AIThinking {
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

function buildMissionStages(
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

function buildActivityFeed(
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

function buildResultsSummary(
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

function buildExportSummary(detail: ProjectDetail, exportDemo: ChampionExportDemo): ExportSummary {
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

function buildDeveloperDiagnostics(detail: ProjectDetail, events: AgentActivityEvent[]): DeveloperDiagnostics {
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

function userFacingActivityText(value: string, maxLength = 180) {
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

function userFacingActionLabel(value: string) {
  return value
    .replace(/\bOpen Workers\b/gi, "Review training capacity")
    .replace(/\bOpen Agents\b/gi, "Review AI work")
    .replace(/\bOpen Operations\b/gi, "View technical details")
    .replace(/\bOpen Runs\b/gi, "View results")
    .replace(/\bOpen Experiments\b/gi, "Review experiments");
}

function firstProposedExperiment(payload: Record<string, unknown>): Record<string, unknown> {
  const proposed =
    Array.isArray(payload.proposed_experiments) ? payload.proposed_experiments :
    Array.isArray(payload.proposedExperiments) ? payload.proposedExperiments :
    [];
  return recordObject(proposed[0]);
}

function missionDecisionToUserText(decision: AgentDecision, proposed = firstProposedExperiment(decision.payload ?? {})) {
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

function decisionRationaleSummary(decision: AgentDecision) {
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

function datasetReviewSummary(dataset: Dataset) {
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

function activityCardFromEvent(event: AgentActivityEvent): ActivityCardModel {
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

function activityCardFromDecision(decision: AgentDecision): ActivityCardModel {
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

function activityCardFromRun(
  summary: TrainingRunSummary,
  evaluation: TrainingRunEvaluation | null,
  job: Job | null,
): ActivityCardModel {
  const status = activityStatus(summary.status);
  const primary = runPrimaryMetric(summary, evaluation, job);
  const terminal = ["SUCCEEDED", "FAILED"].includes(normalizedStatus(summary.status));
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

function activityCardMatchesFilter(card: ActivityCardModel, filter: ActivityFilterKey) {
  if (filter === "all") return true;
  if (filter === "decisions") return card.type === "decision";
  if (filter === "experiments") return card.type === "experiment";
  if (filter === "results") return card.type === "result" || card.type === "export";
  if (filter === "blockers") return card.type === "blocker" || card.status === "blocked" || card.status === "failed";
  return true;
}

function candidateWhyText(row: ChampionComparisonRow) {
  const details = [
    row.latencyMs > 0 ? `${formatLatency(row.latencyMs)} latency` : "",
    row.costUsd > 0 ? `${formatCurrency(row.costUsd)} cost` : "",
    row.objectiveFit > 0 ? `${formatMaybeMetric(row.objectiveFit)} mission fit` : "",
  ].filter(Boolean);
  return details.length > 0 ? `Strong candidate with ${details.join(", ")}.` : "Comparable result from a completed experiment.";
}

function resultsLearningSummary(detail: ProjectDetail) {
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

function resultsRemainingRisks(detail: ProjectDetail, exportDemo: ChampionExportDemo) {
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

function buildMissionDigest({
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
  const activeFailedEvent = visibleActivityEvents.find((event) => ["failed", "blocked"].includes(activityStatus(event.status)));
  const failedWithoutProgress = counts.FAILED > 0 && counts.SUCCEEDED === 0 && activeJobs === 0 && queuedJobs === 0;
  const orchestratorUnhealthy = Boolean(health && health.status !== "ok");
  const hardBlocked = orchestratorUnhealthy || Boolean(activeFailedEvent) || Boolean(blockingDecision) || workerSummary.failed || failedWithoutProgress;

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

function buildMissionLiveActivity({
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

function missionDatasetProfiled(dataset: Dataset | null) {
  if (!dataset) return false;
  if (dataset.profiled_at) return true;
  if (normalizedStatus(dataset.status) === "PROFILED") return true;
  const profile = recordObject(dataset.profile);
  return recordNumber(profile, "total_images") > 0 && recordNumber(profile, "class_count") > 0;
}

function missionWorkerSummary(workers: Worker[], requirements: WorkerRequirement[]) {
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

function missionFacts({
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

function missionStateCopy({
  state,
  dataset,
  latestPlan,
  latestDecision,
  runTotals,
  queuedJobs,
  activeJobs,
  terminalJobs,
  workerSummary,
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
    if (workerSummary.failed) {
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

function buildMissionHealth({
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

function buildMissionNextActions({
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
      : workerSummary.failed || (queuedJobs > 0 && workerSummary.availableWorkers === 0)
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

function buildMissionSignals({
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

function buildMissionChampionSummary(detail: ProjectDetail): MissionChampionSummary | undefined {
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

function missionLiveStepFromEvent(event: AgentActivityEvent): MissionLiveActivity["steps"][number] {
  return {
    id: `event-${event.id}`,
    label: missionActivityLabelForEvent(event),
    status: missionLiveStatus(event.status),
    timestamp: event.created_at,
    ...missionActivityTargetForEvent(event),
  };
}

function missionLiveStatus(value: string): MissionLiveActivity["steps"][number]["status"] {
  const status = activityStatus(value);
  if (["active", "waiting", "succeeded", "failed", "blocked"].includes(status)) {
    return status as MissionLiveActivity["steps"][number]["status"];
  }
  return "active";
}

function missionActivityLabelForEvent(event: AgentActivityEvent) {
  const status = activityStatus(event.status);
  const text = `${event.type} ${event.title} ${event.message}`.toLowerCase();
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

function missionActivityTargetForEvent(event: AgentActivityEvent): Pick<MissionSignal, "targetTab" | "targetId"> {
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

function missionLiveActivityDetail(label: string, status: MissionLiveActivity["status"]) {
  if (label === "Retrieving memories") return "The agent is checking compact prior run and strategy memory.";
  if (label === "Reading dataset profile") return "The planner is using class counts and dataset statistics.";
  if (label === "Ranking candidates") return "The planner is comparing candidate experiments.";
  if (label === "Validating plan") return "Backend gates are checking the proposed plan.";
  if (label === "Validation blocked") return "A backend gate stopped unsafe or invalid work.";
  if (label === "Scheduling workers") return "Mission Control is handing validated work to workers.";
  if (label === "Waiting for workers") return "Queued work is waiting for worker capacity.";
  if (label === "Waiting for training results") return "Training jobs are expected to report metrics.";
  if (label === "Writing decision") return "The agent is recording a compact project decision.";
  if (label === "Selecting champion") return "Completed runs are being promoted into a champion outcome.";
  if (status === "failed") return "The latest agentic step failed; open Agents for the audit trail.";
  if (status === "blocked") return "The latest agentic step is blocked by a visible gate.";
  if (status === "waiting") return "Mission Control is waiting for the next backend or worker update.";
  if (status === "succeeded") return "The latest agentic step completed.";
  return "The agentic system is working through the next high-level step.";
}

function missionEnsureLiveStep(
  steps: MissionLiveActivity["steps"],
  step: MissionLiveActivity["steps"][number],
) {
  if (steps.some((item) => item.label === step.label)) return steps;
  return [step, ...steps].slice(0, 5);
}

function missionSignalDetailForEvent(event: AgentActivityEvent) {
  const label = missionActivityLabelForEvent(event);
  if (label === "Validation blocked") return "Open Agents for backend gate detail.";
  if (label === "Retrieving memories") return "Memory lookup completed without exposing raw payloads.";
  if (label === "Scheduling workers" || label === "Waiting for workers") return "Worker state is available in Operations.";
  if (label === "Waiting for training results") return "Run progress is available in Experiments.";
  if (label === "Writing decision") return "Decision summary is available in Agents.";
  if (label === "Selecting champion") return "Champion details are available in Export.";
  return "Open the drill-down for the full audit trail.";
}

function missionDecisionLabel(decision: AgentDecision) {
  const value = decision.decision_type.toLowerCase().replace(/_/g, " ");
  return value.replace(/\b\w/g, (match) => match.toUpperCase());
}

function missionStateLabel(state: MissionDigestState) {
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

function missionHealthLabel(health: Health | null) {
  if (!health) return "Checking";
  return health.status === "ok" ? "Healthy" : "Unhealthy";
}

function missionToneRank(tone: MissionTone) {
  if (tone === "bad") return 0;
  if (tone === "warning") return 1;
  if (tone === "info") return 2;
  return 3;
}

const activityStorageUriPattern = /\b(?:s3|gs|file|minio|https?):\/\/[^\s,;"')\]}]+/gi;
const activityWindowsPathPattern = /\b[A-Z]:\\[^\s,;"')\]}]+/gi;
const activityUnixPathPattern = /(^|\s)\/(?:Users|home|tmp|var|mnt|data|datasets|artifacts|workspace|app|srv)[^\s,;"')\]}]+/gi;
const activityBase64Pattern = /\b[A-Za-z0-9+/]{80,}={0,2}\b/g;
const activitySecretPattern = /\b(?:sk|pk|rk|xox[baprs]?)-[A-Za-z0-9_-]{16,}\b/gi;

function activityEventFromMessage(event: MessageEvent): AgentActivityEvent | null {
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

function mergeActivityEvents(current: AgentActivityEvent[], event: AgentActivityEvent) {
  const byId = new Map<string, AgentActivityEvent>();
  for (const item of current) byId.set(item.id, item);
  byId.set(event.id, event);
  return Array.from(byId.values())
    .sort((left, right) => activityTimestamp(right) - activityTimestamp(left))
    .slice(0, 12);
}

function buildFallbackActivityEvents(projectId: string, detail: ProjectDetail): AgentActivityEvent[] {
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

function fallbackActivityFromExecutionEvent(event: ExecutionEvent): AgentActivityEvent {
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

function fallbackActivityFromInvocation(invocation: AgentInvocation, jobs: Job[]): AgentActivityEvent[] {
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

function fallbackActivityFromJobs(projectId: string, jobs: Job[]): AgentActivityEvent[] {
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

function agentActivityCurrentState(events: AgentActivityEvent[], detail: ProjectDetail, streamState: ActivityStreamState) {
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

function activitySeverityIcon(severity: string) {
  const normalized = severity.toLowerCase();
  if (normalized === "success") return <CheckCircle2 size={15} />;
  if (normalized === "warning") return <AlertTriangle size={15} />;
  if (normalized === "error") return <X size={15} />;
  return <Activity size={15} />;
}

function activityStreamBadge(streamState: ActivityStreamState) {
  if (streamState === "connected") return "LIVE";
  if (streamState === "reconnecting") return "RECONNECTING";
  if (streamState === "fallback") return "POLLING";
  if (streamState === "connecting") return "CONNECTING";
  return "IDLE";
}

function activityEventSubtitle(event: AgentActivityEvent) {
  return [
    event.type,
    event.plan_id ? `plan ${event.plan_id}` : "",
    event.job_id ? `job ${event.job_id}` : "",
    formatRelativeTime(event.created_at),
  ].filter(Boolean).join(" - ");
}

function activityMetadataRows(metadata: Record<string, unknown> | undefined) {
  const record = activityMetadataObject(metadata);
  return Object.entries(record)
    .map(([key, value]) => ({ label: humanizeAuditKey(key), value: activityMetadataDisplayValue(value) }))
    .filter((row) => row.value !== "")
    .slice(0, 10);
}

function activityMetadataObject(value: unknown): Record<string, unknown> {
  const record = recordObject(value);
  const out: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(record)) {
    const display = activityMetadataDisplayValue(entry);
    if (display) out[key] = display;
  }
  return out;
}

function activityMetadataDisplayValue(value: unknown) {
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

function fallbackActivityMetadata(payload: Record<string, unknown>) {
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

function activitySafeDisplayText(value: string, maxLength = 160) {
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

function activitySeverity(value: string): AgentActivityEvent["severity"] {
  const normalized = value.toLowerCase();
  if (["info", "warning", "error", "success"].includes(normalized)) return normalized;
  return "info";
}

function activityStatus(value: string): AgentActivityEvent["status"] {
  const normalized = value.toLowerCase();
  if (["active", "waiting", "succeeded", "failed", "blocked"].includes(normalized)) return normalized;
  return "active";
}

function activityTimestamp(event: AgentActivityEvent) {
  const timestamp = Date.parse(event.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function jobActivityTimestamp(job: Job) {
  const timestamp = Date.parse(job.completed_at || job.started_at || job.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function jobStatusCounts(jobs: Job[]) {
  return jobs.reduce(
    (counts, job) => {
      const status = normalizedStatus(job.status);
      if (status in counts) counts[status as keyof typeof counts] += 1;
      return counts;
    },
    { QUEUED: 0, ASSIGNED: 0, RUNNING: 0, SUCCEEDED: 0, FAILED: 0 },
  );
}

function recordBoolean(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["true", "yes", "1"].includes(value.toLowerCase());
  return false;
}

function formatRelativeTime(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  const elapsedSeconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (elapsedSeconds < 60) return `${elapsedSeconds}s ago`;
  const elapsedMinutes = Math.round(elapsedSeconds / 60);
  if (elapsedMinutes < 60) return `${elapsedMinutes}m ago`;
  const elapsedHours = Math.round(elapsedMinutes / 60);
  if (elapsedHours < 24) return `${elapsedHours}h ago`;
  return new Date(timestamp).toLocaleString();
}

function buildExperimentTimeline(project: Project | null, detail: ProjectDetail): TimelineItem[] {
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

function buildDatasetIntelligence(
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

function datasetMetadataSummaryFromResponse(
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

function datasetMetadataSummaryFromImports(imports: DatasetMetadataImport[]): DatasetMetadataSummary | null {
  const latestImport = [...imports].sort((left, right) => metadataImportSortScore(right) - metadataImportSortScore(left)).find(Boolean);
  if (!latestImport) return null;
  return datasetMetadataSummaryFromImport(latestImport);
}

function datasetMetadataSummaryFromImport(metadataImport: DatasetMetadataImport): DatasetMetadataSummary | null {
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

function datasetMetadataImportsFromResponse(value: DatasetMetadataImportsResponse): DatasetMetadataImport[] {
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

function arrayDatasetMetadataImports(value: unknown): DatasetMetadataImport[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as DatasetMetadataImport).filter(hasDatasetMetadataImportShape);
}

function hasDatasetMetadataImportShape(value: unknown): value is DatasetMetadataImport {
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

function hasDatasetMetadataSummaryShape(value: unknown): value is DatasetMetadataSummary {
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

function datasetMetadataSummaryFallback(record: Record<string, unknown>): DatasetMetadataSummary {
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

function metadataImportSortScore(metadataImport: DatasetMetadataImport) {
  const timestamp = Date.parse(metadataImport.completed_at || metadataImport.created_at || "");
  if (Number.isFinite(timestamp)) return timestamp;
  if (metadataImport.active) return 1;
  return 0;
}

function buildMetadataStatusDisplay(metadataDetail: DatasetMetadataDetail): MetadataStatusDisplay | null {
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

function metadataSummaryStatus(summary: DatasetMetadataSummary, imports: DatasetMetadataImport[]) {
  const summaryStatus = recordFirstString(summary, ["status"]);
  if (summaryStatus) return summaryStatus;
  const importId = recordFirstString(summary, ["import_id"]);
  const matchingImport = imports.find((metadataImport) => metadataImport.import_id === importId || metadataImport.id === importId);
  return matchingImport?.status || imports.find((metadataImport) => metadataImport.active)?.status || "reported";
}

function metadataDetailText(summary: DatasetMetadataSummary, imports: DatasetMetadataImport[]) {
  const importId = recordFirstString(summary, ["import_id"]);
  const matchingImport = imports.find((metadataImport) => metadataImport.import_id === importId || metadataImport.id === importId);
  const createdAt = recordFirstString(summary, ["created_at"]) || matchingImport?.created_at || "";
  const completedAt = recordFirstString(summary, ["completed_at"]) || matchingImport?.completed_at || "";
  const timestamp = completedAt || createdAt;
  const timeLabel = timestamp ? `${completedAt ? "completed" : "created"} ${formatTimestamp(timestamp)}` : "";
  return [importId ? `import ${importId}` : "agent-safe summary", timeLabel].filter(Boolean).join(" - ");
}

function metadataSourceTags(summary: DatasetMetadataSummary) {
  const sourceKinds = metadataStringList(summary.source_kinds ?? summary.source_kind);
  const sourceFormats = metadataStringList(summary.source_formats ?? summary.formats ?? summary.detected_formats);
  return uniqueStrings([
    ...sourceKinds.map((kind) => `kind: ${kind}`),
    ...sourceFormats.map((format) => `format: ${format}`),
  ]).slice(0, 8);
}

function metadataCountRows(value: unknown): MetadataCountRow[] {
  return Object.entries(recordObject(value))
    .map(([label, count]) => ({ label, value: metadataCountText(count) }))
    .filter((row) => row.value !== "")
    .slice(0, 8);
}

function metadataCountText(value: unknown) {
  const numeric = metadataNumericValue(value);
  if (numeric !== null) return String(numeric);
  const record = recordObject(value);
  const nested = metadataNumber(record, ["count", "sample_count", "annotation_count"]);
  if (nested > 0) return String(nested);
  if (typeof value === "string" && value.trim()) return value.trim();
  return "";
}

function metadataStringList(value: unknown): string[] {
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

function metadataStringValue(value: unknown) {
  if (typeof value === "string") return shortAuditText(value, 80);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  const record = recordObject(value);
  return shortAuditText(
    recordFirstString(record, ["format", "source_format", "detected_format", "declared_format", "kind", "source_kind", "name"]),
    80,
  );
}

function metadataIssueSummaries(value: unknown): string[] {
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

function metadataIssueSummary(value: unknown) {
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

function metadataPrimitiveText(value: unknown) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} item(s)`;
  return "";
}

function metadataNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const numeric = metadataNumericValue(record[key]);
    if (numeric !== null) return numeric;
  }
  return 0;
}

function metadataNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function metadataCountValue(value: number) {
  return Number.isFinite(value) && value > 0 ? String(value) : "-";
}

function metadataBBoxValue(summary: DatasetMetadataSummary) {
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

function metadataBBoxTone(summary: DatasetMetadataSummary): InsightItem["tone"] {
  const value = metadataBBoxValue(summary);
  if (value === "-") return "warn";
  return "good";
}

function metadataCoverageText(value: unknown) {
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

function formatTimestamp(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  return new Date(timestamp).toLocaleString();
}

function visualAnalysesFromResponse(value: unknown): DatasetVisualAnalysis[] {
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

function visualAnalysisRerunPolicyFromResponse(value: unknown): VisualAnalysisRerunPolicy | null {
  const record = recordObject(value);
  const policy = recordObject(record.rerun_policy);
  return Object.keys(policy).length > 0 ? (policy as VisualAnalysisRerunPolicy) : null;
}

function arrayVisualAnalyses(value: unknown): DatasetVisualAnalysis[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as DatasetVisualAnalysis).filter(hasVisualAnalysisShape);
}

function hasVisualAnalysisShape(value: unknown): value is DatasetVisualAnalysis {
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

function latestVisualAnalysis(analyses: DatasetVisualAnalysis[]) {
  return [...analyses]
    .filter(hasVisualAnalysisShape)
    .sort((left, right) => visualAnalysisSortScore(right) - visualAnalysisSortScore(left))[0] ?? null;
}

function visualAnalysisSortScore(analysis: DatasetVisualAnalysis) {
  const timestamp = Date.parse(analysis.updated_at || analysis.created_at || "");
  if (Number.isFinite(timestamp)) return timestamp;
  return typeof analysis.analysis_version === "number" ? analysis.analysis_version : 0;
}

function visualAnalysisFromProfile(profile: Record<string, unknown>) {
  const analyses: DatasetVisualAnalysis[] = [];
  for (const key of ["dataset_visual_analysis", "visual_analysis", "latest_visual_analysis", "visual_dataset_analysis"]) {
    const analysis = recordObject(profile[key]);
    if (hasVisualAnalysisShape(analysis)) analyses.push(analysis as DatasetVisualAnalysis);
  }
  analyses.push(...arrayVisualAnalyses(profile.visual_analyses));
  analyses.push(...arrayVisualAnalyses(profile.dataset_visual_analyses));
  return latestVisualAnalysis(analyses);
}

function agentInvocationsFromResponse(response: AgentInvocationsResponse): AgentInvocation[] {
  return response.invocations ?? response.agent_invocations ?? response.items ?? [];
}

function buildAgentInvocationAuditRows(invocations: AgentInvocation[], decisions: AgentDecision[]): AgentInvocationAuditRow[] {
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

function buildMissionControlTelemetrySummary(
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

function telemetryWindowsFromInvocations(invocations: AgentInvocation[]): TelemetryWindowSummary[] {
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

function telemetryWindowSummary(key: TelemetryWindowKey, label: string, invocations: AgentInvocation[]): TelemetryWindowSummary {
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

function telemetryCountSummaryFromInvocations(
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

function telemetryInvocationSummaries(invocations: AgentInvocation[]): TelemetryInvocationSummary[] {
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

function telemetryInvocationSummary(invocation: AgentInvocation): TelemetryInvocationSummary {
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

function telemetryInvocationUsage(
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

function telemetryInvocationRuntime(invocation: AgentInvocation) {
  const inputContext = recordObject(invocation.input_context);
  return recordObject(inputContext.invocation_runtime);
}

function telemetryPromptSectionSummaries(invocations: AgentInvocation[]): TelemetrySectionSummary[] {
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

function telemetryApproximateTokensFromInput(invocation: AgentInvocation) {
  const messageBytes = byteLengthOfJson(invocation.input_messages ?? []);
  return Math.max(0, Math.ceil(messageBytes / 4));
}

function telemetryApproximateTokensFromText(text: string) {
  return Math.max(0, Math.ceil(byteLengthOfText(text) / 4));
}

function telemetryInvocationPromptBytes(invocation: AgentInvocation) {
  return byteLengthOfJson(invocation.input_messages ?? []);
}

function telemetryInvocationSections(invocation: AgentInvocation) {
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

function telemetrySectionsFromSnapshot(snapshot: Record<string, unknown>) {
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

function telemetrySectionEstimateList(value: unknown): TelemetrySectionSummary[] {
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

function telemetryEmbeddingSummaryFromUsageEvents(events: MemoryEmbeddingUsageEvent[]) {
  const sourceIndexEvents = events.filter((event) => telemetryEmbeddingPurpose(event) === "source_index");
  const retrievalEvents = events.filter((event) => telemetryEmbeddingPurpose(event) === "retrieval_query");

  return {
    totalUsageEvents: events.length,
    sourceIndex: telemetryEmbeddingPurposeSummary("source_index", sourceIndexEvents),
    retrievalQuery: telemetryEmbeddingPurposeSummary("retrieval_query", retrievalEvents),
  };
}

function telemetryEmbeddingPurposeSummary(purpose: string, events: MemoryEmbeddingUsageEvent[]): TelemetryEmbeddingPurposeSummary {
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

function telemetryEmbeddingBreakdownSummary(
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

function telemetryEmbeddingPurpose(event: MemoryEmbeddingUsageEvent) {
  const purpose = String(event.purpose || "").toLowerCase().trim();
  if (purpose === "source_index" || purpose === "retrieval_query") {
    return purpose;
  }
  if (purpose.includes("source")) return "source_index";
  if (purpose.includes("retrieval")) return "retrieval_query";
  return "source_index";
}

function telemetryEmbeddingEstimatedCost(event: MemoryEmbeddingUsageEvent) {
  const model = String(event.embedding_model || "").toLowerCase();
  const pricing = telemetryEmbeddingPricingForModel(model);
  const tokens = Math.max(0, event.input_bytes || 0) / 4;
  return (tokens * pricing.input) / 1_000_000;
}

function telemetryEmbeddingPricingForModel(model: string) {
  if (model.includes("text-embedding-3-large")) {
    return { input: 0.13 };
  }
  if (model.includes("text-embedding-3-small")) {
    return { input: 0.02 };
  }
  return { input: 0.02 };
}

function telemetryPricingForModel(model: string) {
  const lower = model.toLowerCase();
  if (lower.includes("gpt-5.4-pro")) return { input: 30, cached: 3, output: 180 };
  if (lower.includes("gpt-5.4-mini")) return { input: 0.75, cached: 0.075, output: 4.5 };
  if (lower.includes("gpt-5.4")) return { input: 2.5, cached: 0.25, output: 15 };
  if (lower.includes("gpt-5 mini")) return { input: 0.25, cached: 0.025, output: 2 };
  return { input: 0.75, cached: 0.075, output: 4.5 };
}

function telemetryInvocationTimestamp(invocation: AgentInvocation) {
  const timestamp = Date.parse(invocation.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function telemetryTimestampValue(value: string) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sumTelemetry(values: number[]) {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
}

function byteLengthOfText(value: string) {
  return new TextEncoder().encode(value).length;
}

function byteLengthOfJson(value: unknown) {
  try {
    return byteLengthOfText(JSON.stringify(value) ?? "");
  } catch {
    return 0;
  }
}

function formatCompactNumber(value: number) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs < 1000) {
    return String(Math.round(value));
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatTelemetryTokenPair(exactTokens: number, approxTokens: number) {
  const hasExact = Number.isFinite(exactTokens) && exactTokens > 0;
  const hasApprox = Number.isFinite(approxTokens) && approxTokens > 0;
  if (hasExact && hasApprox) {
    return `${formatCompactNumber(exactTokens)} exact + ~${formatCompactNumber(approxTokens)} approx`;
  }
  if (hasExact) {
    return `${formatCompactNumber(exactTokens)} exact`;
  }
  if (hasApprox) {
    return `~${formatCompactNumber(approxTokens)}`;
  }
  return "0";
}

function invocationRuntimeRecord(inputContext: Record<string, unknown>) {
  for (const key of ["invocation_runtime", "runtime", "llm_runtime", "responses_runtime"]) {
    const runtime = recordObject(inputContext[key]);
    if (Object.keys(runtime).length > 0) return runtime;
  }
  return {};
}

function invocationTarget(invocation: AgentInvocation) {
  const parts = [
    invocation.plan_id ? `plan ${invocation.plan_id}` : "",
    invocation.job_id ? `job ${invocation.job_id}` : "",
    invocation.dataset_id ? `dataset ${invocation.dataset_id}` : "",
  ].filter(Boolean);
  return parts.join(" - ") || (invocation.project_id ? `project ${invocation.project_id}` : "project scope");
}

function invocationDecisionLink(
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

function visualAnalysisFacts(visualAnalysis: VisualAnalysisDetail, activeJob: Job | null): InsightItem[] {
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

function visualStatusTone(status: string): InsightItem["tone"] {
  const normalized = normalizedStatus(status).toLowerCase();
  if (["accepted", "available", "succeeded", "completed", "ready"].includes(normalized)) return "good";
  if (["failed", "error", "rejected", "unsupported"].includes(normalized)) return "bad";
  return "warn";
}

function visualAnalysisStatusBadge(visualAnalysis: VisualAnalysisDetail, activeJob: Job | null) {
  if (activeJob) return normalizedStatus(activeJob.status);
  if (visualAnalysis.analysis?.validation_status) return normalizedStatus(visualAnalysis.analysis.validation_status);
  if (visualAnalysis.status === "available") return "AVAILABLE";
  if (visualAnalysis.status === "unsupported") return "UNSUPPORTED";
  if (visualAnalysis.status === "error") return "ERROR";
  return "NOT_RECORDED";
}

function visualAnalysisActiveJob(jobs: Job[], datasetId: string) {
  return jobs.find((job) => {
    const status = normalizedStatus(job.status);
    if (!["QUEUED", "ASSIGNED", "RUNNING", "REQUESTED", "PENDING"].includes(status)) return false;
    const template = job.template.toLowerCase();
    if (!template.includes("visual")) return false;
    const jobDatasetId = typeof job.config.dataset_id === "string" ? job.config.dataset_id : "";
    return !datasetId || !jobDatasetId || jobDatasetId === datasetId;
  }) ?? null;
}

function visualAnalysisRerunDisabledReason(
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

function visualAnalysisPolicyRunCount(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "-";
  const runs = policy.runs_for_profile ?? 0;
  const maxRuns = policy.max_runs_per_profile ?? 0;
  return maxRuns ? `${runs}/${maxRuns}` : String(runs);
}

function visualAnalysisPolicyReadiness(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "-";
  if (policy.active_job_status) return normalizedStatus(policy.active_job_status);
  if (policy.manual_run_allowed === false) {
    if (policy.next_allowed_at) return `After ${new Date(policy.next_allowed_at).toLocaleTimeString()}`;
    return "Blocked";
  }
  return "Ready";
}

function visualAnalysisPolicySummary(policy: VisualAnalysisRerunPolicy | null) {
  if (!policy) return "";
  if (policy.disabled_reason) return policy.disabled_reason;
  if (policy.reason) return policy.reason;
  const triggerCount = policy.deficiency_triggers?.length ?? 0;
  if (triggerCount > 0) return `${triggerCount} deficiency trigger(s) active; backend may queue a rare reanalysis if automation is enabled.`;
  return policy.automation_enabled ? "Automatic initial and deficiency reanalysis are enabled." : "Automatic visual analysis is disabled; manual rerun is still backend-gated.";
}

function visualAnalysisLimitations(analysis: DatasetVisualAnalysis | null) {
  if (!analysis) return [];
  const coverage = recordObject(analysis.coverage_report);
  return uniqueStrings([
    ...stringArrayPayload(analysis.limitations),
    ...stringArrayPayload(coverage.limitations),
  ]).slice(0, 8);
}

function visualCoverageSummary(analysis: DatasetVisualAnalysis) {
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

function visualPerClassCoverageRows(coverage: Record<string, unknown>) {
  const counts = recordObject(coverage.per_class_counts);
  return Object.entries(counts)
    .map(([label, value]) => ({ label, value: String(numericValue(value)) }))
    .filter((row) => row.value !== "0")
    .slice(0, 10);
}

function buildDecisionChatTurns(decisions: AgentDecision[]): DecisionChatTurn[] {
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

function decisionChatQuestion(decision: AgentDecision, index: number) {
  const payload = decision.payload ?? {};
  const planID =
    decision.plan_id ||
    recordFirstString(payload, ["plan_id", "source_plan_id", "followup_plan_id", "follow_up_plan_id"]) ||
    `decision_${index + 1}`;
  return `What is ${planID}?`;
}

function decisionChatOpening(decision: AgentDecision) {
  const summary = recordString(decision.payload ?? {}, "summary");
  return uniqueStrings([summary, decision.rationale]).join(" ");
}

function decisionReasoningSections(decision: AgentDecision): ReasoningSection[] {
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

function decisionRejections(decision: AgentDecision) {
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

function buildDecisionQualitySnapshot(decisions: AgentDecision[], invocations: AgentInvocation[]): DecisionQualitySnapshot | null {
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

function latestDecisionWithQualityContext(decisions: AgentDecision[]) {
  const sorted = [...decisions].sort((left, right) => timestampSortScore(right.created_at) - timestampSortScore(left.created_at));
  return (
    sorted.find((decision) => Object.keys(recordObject(decision.payload?.project_trajectory_card)).length > 0) ??
    sorted.find((decision) => Array.isArray(decision.payload?.candidate_rankings)) ??
    sorted[0] ??
    null
  );
}

function latestProjectTrajectoryFromInvocations(invocations: AgentInvocation[]) {
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

function latestInvocationTimestamp(invocations: AgentInvocation[]) {
  const latest = [...invocations].sort((left, right) => timestampSortScore(right.created_at) - timestampSortScore(left.created_at))[0];
  return latest?.created_at ?? "";
}

function firstNonEmptyRecord(records: Record<string, unknown>[]) {
  return records.find((record) => Object.keys(record).length > 0) ?? {};
}

function topCandidateRejectionReason(rankings: Record<string, unknown>[]) {
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

function decisionQualityMechanismOutcomes(trajectory: Record<string, unknown>): MechanismCoverageRow[] {
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

function decisionQualityOutcomeDetail(record: Record<string, unknown>) {
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

function recordFirstNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = numberPayload(record[key]);
    if (value !== null) return value;
  }
  return null;
}

function timestampSortScore(value?: string) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function buildChampionComparison(
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

function buildSeedVarianceBySignature(
  summaries: TrainingRunSummary[],
  jobById: Map<string, Job>,
  evaluationByJob: Map<string, TrainingRunEvaluation>,
) {
  const groups = new Map<string, Array<{ seed: string; score: number }>>();
  for (const summary of summaries) {
    const job = jobById.get(summary.job_id);
    if (!job || normalizedStatus(summary.status) !== "SUCCEEDED") continue;
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

function experimentRankScore(input: {
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

function experimentComparisonSignature(config: Record<string, unknown>) {
  const normalized = normalizeComparisonConfig(config) as Record<string, unknown>;
  return Object.keys(normalized).length > 0 ? stableStringify(normalized) : "";
}

function normalizeComparisonConfig(value: unknown): unknown {
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

function stableStringify(value: unknown): string {
  return JSON.stringify(normalizeComparisonConfig(value));
}

function experimentSeed(config: Record<string, unknown>) {
  const seed = config.seed ?? config.random_seed;
  if (typeof seed === "number" && Number.isFinite(seed)) return String(seed);
  if (typeof seed === "string" && seed.trim()) return seed.trim();
  return "";
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function buildMemoryRetrievalProbeSnapshots(events: ExecutionEvent[]): MemoryRetrievalProbeSnapshot[] {
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

function humanizeMemoryPurpose(value: string) {
  const normalized = value.trim().toLowerCase();
  if (normalized === "planner" || normalized === "experiment_planner") return "Planner retrieval";
  if (normalized === "training_monitor") return "Training monitor retrieval";
  if (!normalized) return "Memory retrieval";
  return `${normalized.replace(/[_-]+/g, " ")} retrieval`;
}

function memoryPurposeSuffix(value: string) {
  const label = humanizeMemoryPurpose(value).replace(/\s+retrieval$/i, "").toLowerCase();
  return label && label !== "memory" ? ` for ${label}` : "";
}

function candidateScoreRows(decision: AgentDecision): CandidateScoreRow[] {
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

function decisionRetrievedMemoryRows(decision: AgentDecision): RetrievedMemoryDisplay[] {
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

function candidateRetrievedMemoryRows(
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

function candidateMemoryReasons(
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

function candidateScoreComponentRows(components: Record<string, unknown>): Array<{ label: string; value: number | string }> {
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

function memoryCardsFromUnknown(value: unknown): RetrievedMemoryDisplay[] {
  return memoryCardsFromUnknownDepth(value, 0);
}

function memoryCardsFromUnknownDepth(value: unknown, depth: number): RetrievedMemoryDisplay[] {
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

function memoryDisplayFromRecord(record: RetrievedMemoryCard): RetrievedMemoryDisplay | null {
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

function isMemoryCardLike(record: Record<string, unknown>) {
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

function isMemoryContainerKey(key: string, entry: unknown) {
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

function firstMemoryString(records: Record<string, unknown>[], keys: string[], maxLength: number) {
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

function memoryIdentifierRows(records: Record<string, unknown>[]) {
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

function memoryOutcome(records: Record<string, unknown>[], kind: string) {
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

function humanizeMemorySource(value: string) {
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

function memoryReasonStrings(value: unknown): string[] {
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

function memoryScoreComponentSummaries(components: Record<string, unknown>) {
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

function isMemoryComponentLabel(label: string) {
  const normalized = label.toLowerCase();
  return normalized.includes("retrieved_memory") || normalized.includes("memory_similarity") || normalized.includes("memory_score");
}

function isMemoryReasonText(value: string) {
  const normalized = value.toLowerCase();
  return normalized.includes("retrieved") || normalized.includes("memory") || normalized.includes("scorecard");
}

function memoryDisplayKey(memory: RetrievedMemoryDisplay) {
  return `${memory.source}-${memory.sourceId}-${memory.kind}-${memory.summary}`;
}

function safeMemoryText(value: string, maxLength = 220) {
  const text = shortAuditText(value, maxLength);
  const trimmed = text.trim();
  if (!trimmed) return "";
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) return "";
  return trimmed;
}

function buildChampionExportDemo(detail: ProjectDetail): ChampionExportDemo {
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
  const exportStatus =
    recordString(modelCard, "export_status") ||
    recordString(deployment, "export_status") ||
    exports[0]?.status ||
    "PENDING";
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

  return {
    hasChampion: true,
    exportStatus,
    exports: uniqueBy(exports, (item, index) => item.id || item.artifact_uri || item.model_uri || item.download_url || String(index)),
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

function championFeedbackMetricsSnapshot(detail: ProjectDetail) {
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

function feedbackRatingLabel(rating: string) {
  switch (rating) {
    case "good":
      return "Good";
    case "bad":
      return "Bad";
    default:
      return "Mediocre";
  }
}

function normalizeChampionFeedbackResponse(response: { feedback?: ChampionFeedback } | ChampionFeedback): ChampionFeedback | null {
  const wrapped = recordObject(response).feedback;
  if (isChampionFeedback(wrapped)) return wrapped;
  if (isChampionFeedback(response)) return response;
  return null;
}

function isChampionFeedback(value: unknown): value is ChampionFeedback {
  const record = recordObject(value);
  return Boolean(record.id && record.champion_id && record.rating);
}

function experimentPreprocessingItems(experiment: PlannedExperiment) {
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

function classBalancingConfigSummary(config: PlannedExperiment["class_balancing_config"]) {
  const record = recordObject(config);
  const beta = recordNumber(record, "effective_number_beta");
  return typeof beta === "number" ? `beta ${formatMetricNumber(beta)}` : "";
}

function augmentationPolicyConfigSummary(config: PlannedExperiment["augmentation_policy_config"]) {
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

function experimentAutoMLItems(experiment: PlannedExperiment) {
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

function experimentMechanismItems(experiment: PlannedExperiment) {
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

function jobMechanismSummary(job: Job) {
  const mechanism = recordString(job.config, "mechanism");
  const intervention = recordString(job.config, "intervention");
  const validation = recordFirstString(job.config, ["backend_validation_status", "validation_status"]);
  const automlSummary = recordObject(job.config.automl_summary);
  const automl = recordString(automlSummary, "suggestion_id") ? `AutoML ${recordString(automlSummary, "sampler") || "suggestion"}` : "";
  return [mechanism ? `mechanism ${mechanism}` : "", intervention, automl, validation ? `validation ${validation}` : ""]
    .filter(Boolean)
    .join(" / ");
}

function backendGateSummary(decision: AgentDecision) {
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

function decisionHistorySummary(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  const mechanism = recordFirstString(payload, ["mechanism", "selected_mechanism", "proposal_mechanism"]);
  const intervention = recordString(payload, "intervention");
  const validation = recordFirstString(payload, ["backend_validation_status", "validation_status"]);
  return [mechanism ? `mechanism ${mechanism}` : "", intervention, validation ? `validation ${validation}` : "", decision.rationale]
    .filter(Boolean)
    .join(" - ");
}

function decisionHighlights(decision: AgentDecision) {
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

function mechanismDecisionSummaries(payload: Record<string, unknown>) {
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

function mechanismSummaryFromRecord(record: Record<string, unknown>) {
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

function mechanismCoverageRows(payload: Record<string, unknown>): MechanismCoverageRow[] {
  const coverage = recordObject(payload.mechanism_coverage ?? payload.mechanism_coverage_card);
  const rows: MechanismCoverageRow[] = [];
  addCoverageRows(rows, "TRIED", coverage.tried ?? coverage.tried_mechanisms ?? coverage.attempted ?? coverage.attempted_mechanisms);
  addCoverageRows(rows, "ELIGIBLE", coverage.eligible ?? coverage.eligible_mechanisms);
  addCoverageRows(rows, "BLOCKED", coverage.blocked ?? coverage.blocked_mechanisms);
  addCoverageRows(rows, "NO_IMPROVEMENT", coverage.no_improvement ?? coverage.no_improvement_mechanisms);
  addCoverageRows(rows, "BEST_RESULT", coverage.best_result ?? coverage.best_results ?? coverage.best_result_by_mechanism);
  return uniqueBy(rows, (row) => `${row.status}-${row.mechanism}-${row.detail}`).slice(0, 10);
}

function addCoverageRows(rows: MechanismCoverageRow[], status: string, value: unknown) {
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

function coverageDetail(record: Record<string, unknown>, fallback: string) {
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

function automationReviewState(settings: AutomationSettings) {
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

function numberPayload(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordString(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

function recordFirstString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = recordString(record, key);
    if (value) return value;
  }
  return "";
}

function recordNumber(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

function recordObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function firstAuditValue(records: Record<string, unknown>[], keys: string[]) {
  for (const record of records) {
    for (const key of keys) {
      const value = record[key];
      if (value !== undefined && value !== null && value !== "") return value;
    }
  }
  return undefined;
}

function firstAuditString(records: Record<string, unknown>[], keys: string[]) {
  const value = firstAuditValue(records, keys);
  if (typeof value === "string") return shortAuditText(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function auditCountValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string" && value.trim()) return shortAuditText(value, 40);
  if (Array.isArray(value)) return String(value.length);

  const record = recordObject(value);
  const count = firstAuditValue([record], ["count", "total", "rounds", "round_count", "tool_rounds"]);
  if (typeof count === "number" && Number.isFinite(count)) return String(count);
  if (typeof count === "string" && count.trim()) return shortAuditText(count, 40);
  return "-";
}

function auditToolNamesFromValue(value: unknown): string[] {
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

function auditRejectedToolCallSummaries(value: unknown): string[] {
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

function hasAuditRejectionShape(record: Record<string, unknown>) {
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

function auditRejectionItem(value: unknown, index: number): string | Record<string, unknown> {
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

function auditDryRunValidationSummaries(value: unknown): Array<{ status: string; text: string }> {
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

function auditDryRunValidationItem(value: unknown, index: number, fallbackLabel?: string): Array<{ status: string; text: string }> {
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

function hasAuditValidationShape(record: Record<string, unknown>) {
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

function auditPrimitiveSummary(value: unknown) {
  if (typeof value === "string") return shortAuditText(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} item(s)`;
  if (value && typeof value === "object") return "object";
  return "";
}

function shortAuditText(value: string, maxLength = 160) {
  const collapsed = value.replace(/\s+/g, " ").trim();
  if (!collapsed) return "";
  if (isLikelyEncodedPayload(collapsed)) return "[redacted payload]";
  const redacted = collapsed
    .replace(/\b[a-z][a-z0-9+.-]*:\/\/\S+/gi, "[redacted uri]")
    .replace(/[A-Za-z]:\\[^\s]+/g, "[redacted path]")
    .replace(/\/(?:[^/\s]+\/){2,}[^/\s]+/g, "[redacted path]");
  return redacted.length > maxLength ? `${redacted.slice(0, maxLength - 3)}...` : redacted;
}

function isLikelyEncodedPayload(value: string) {
  const lower = value.toLowerCase();
  return lower.startsWith("data:image") || lower.includes(";base64,") || (value.length > 180 && /^[A-Za-z0-9+/=]+$/.test(value));
}

function isSensitiveAuditKey(key: string) {
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

function humanizeAuditKey(key: string) {
  return key.replace(/[_-]+/g, " ");
}

function normalizedStatus(value: string) {
  return value.trim().toUpperCase() || "PENDING";
}

function statusToneClass(value?: string) {
  const status = normalizedStatus(value || "PENDING").toLowerCase();
  if (["succeeded", "ready", "correct", "completed"].includes(status)) return "status-good";
  if (["failed", "error", "missed", "runtime_unavailable"].includes(status)) return "status-bad";
  if (["requested", "running", "pending", "pending_artifact", "queued"].includes(status)) return "status-wait";
  return "";
}

function exportStatusMessage(value?: string) {
  const status = normalizedStatus(value || "PENDING");
  if (status === "READY" || status === "SUCCEEDED") return "export artifact ready";
  if (status === "FAILED") return "export failed";
  if (status === "RUNNING") return "export worker running";
  if (status === "REQUESTED" || status === "QUEUED") return "export queued by backend";
  if (status === "PENDING_ARTIFACT") return "waiting for champion artifact";
  return "artifact URI pending";
}

function predictionStatusMessage(value?: string) {
  const status = normalizedStatus(value || "PENDING");
  if (status === "SUCCEEDED") return "prediction complete";
  if (status === "FAILED") return "prediction failed";
  if (status === "RUNTIME_UNAVAILABLE") return "runtime unavailable";
  if (status === "RUNNING") return "prediction running";
  if (status === "REQUESTED" || status === "QUEUED") return "prediction queued";
  return "prediction pending";
}

function objectSummary(record: Record<string, unknown>) {
  const entries = Object.entries(record)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, 4)
    .map(([key, value]) => `${key}: ${shortValue(value)}`);
  return entries.join("; ") || "-";
}

function shortValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => shortValue(item)).join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function isUnsupportedEndpointError(error: unknown) {
  const message = errorMessage(error).toLowerCase();
  return (
    message.includes("404") ||
    message.includes("not found") ||
    message.includes("cannot get") ||
    message.includes("unexpected non-whitespace")
  );
}

function isOrchestratorHttpErrorResponse(value: unknown): value is OrchestratorHttpErrorResponse {
  const record = recordObject(value);
  return record.__mission_control_http_error === true && typeof record.status === "number";
}

function exemplarStatusFromProfile(profile: Record<string, unknown>) {
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

function stringArrayPayload(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function championExportsFromUnknown(value: unknown): ChampionExport[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionExport).filter((item) => Object.keys(item).length > 0);
}

function championExportExternalData(exportRecord: ChampionExport) {
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

function firstChampionArtifactMatchingFormat(values: string[], format: string) {
  return values.find((value) => value && artifactMatchesChampionExportFormat(value, format)) || "";
}

function championExportFormatFromArtifact(uri: string) {
  if (artifactMatchesChampionExportFormat(uri, "onnx")) return "onnx";
  if (artifactMatchesChampionExportFormat(uri, "torchscript")) return "torchscript";
  if (artifactMatchesChampionExportFormat(uri, "safetensors")) return "safetensors";
  if (artifactMatchesChampionExportFormat(uri, "pytorch")) return "pytorch";
  return "";
}

function championExportValidationErrorsForFormat(errors: string[], format: string) {
  const normalized = format.toLowerCase();
  if (normalized === "onnx" || !normalized) return errors;
  return errors.filter((error) => !/onnx|onnxscript/i.test(error));
}

function artifactMatchesChampionExportFormat(uri: string, format: string) {
  const path = artifactComparablePath(uri);
  if (format === "onnx") return path.endsWith(".onnx");
  if (format === "torchscript") return path.endsWith(".torchscript.pt") || path.endsWith(".torchscript");
  if (format === "pytorch") return path.endsWith(".pt") || path.endsWith(".pth");
  if (format === "safetensors") return path.endsWith(".safetensors");
  return false;
}

function artifactComparablePath(uri: string) {
  const trimmed = uri.trim();
  const fallback = trimmed.split(/[?#]/)[0].toLowerCase();
  try {
    return decodeURIComponent(new URL(trimmed).pathname || fallback).toLowerCase();
  } catch {
    return fallback;
  }
}

function demoImagesFromUnknown(value: unknown): ChampionDemoImage[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoImage).filter((item) => Object.keys(item).length > 0);
}

function demoPredictionsFromUnknown(value: unknown): ChampionDemoPrediction[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoPrediction).filter((item) => Object.keys(item).length > 0);
}

function championExportDemoIsDetection(data: ChampionExportDemo) {
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

function detectionDefaultsFromChampionExportDemo(data: ChampionExportDemo) {
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

function detectionBoxesFromPrediction(prediction?: ChampionDemoPrediction | null): ChampionDetection[] {
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

function normalizeDetection(value: unknown): ChampionDetection | null {
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

function normalizedDetectionBox(detection: ChampionDetection): { x: number; y: number; width: number; height: number } | null {
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

function boundedDetectionBox(x: number, y: number, width: number, height: number) {
  const left = Math.max(0, Math.min(1, x));
  const top = Math.max(0, Math.min(1, y));
  const boundedWidth = Math.max(0, Math.min(1 - left, width));
  const boundedHeight = Math.max(0, Math.min(1 - top, height));
  if (boundedWidth <= 0 || boundedHeight <= 0) return null;
  return { x: left, y: top, width: boundedWidth, height: boundedHeight };
}

function predictionPostprocessLatency(prediction?: ChampionDemoPrediction | null) {
  if (!prediction) return 0;
  const metadata = { ...recordObject(prediction.image_metadata), ...recordObject(prediction.metadata) };
  const breakdown = recordObject(metadata.latency_breakdown_ms);
  return numericValue(prediction.postprocess_latency_ms ?? metadata.postprocess_latency_ms ?? breakdown.postprocess);
}

function normalizeDemoPredictionResponse(value: ChampionDemoPrediction | { prediction?: ChampionDemoPrediction }) {
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

function attachDemoPredictionPreview(prediction: ChampionDemoPrediction, image: ChampionDemoImage) {
  const previewURI = demoImagePreviewURI(image);
  if (!previewURI) return prediction;
  const metadata = {
    ...recordObject(prediction.image_metadata),
    ...recordObject(prediction.metadata),
    thumbnail_uri: previewURI,
  };
  return { ...prediction, metadata, image_metadata: { ...recordObject(prediction.image_metadata), preview_available: true } };
}

function demoImageURI(image?: ChampionDemoImage | null) {
  return image?.uri || image?.image_uri || "";
}

function demoImagePreviewURI(image?: ChampionDemoImage | null) {
  return image?.preview_uri || image?.thumbnail_uri || image?.uri || image?.image_uri || "";
}

function demoImageLabel(image?: ChampionDemoImage | null) {
  return image?.true_label || image?.label || image?.class_name || "";
}

function demoImageDetail(image?: ChampionDemoImage | null) {
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

function demoPredictionRequestMetadata(image: ChampionDemoImage) {
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(recordObject(image.metadata))) {
    if (["path", "preview_uri", "thumbnail_uri", "uri", "image_uri"].includes(key.toLowerCase())) continue;
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

function isTerminalDemoPredictionStatus(value?: string) {
  return ["SUCCEEDED", "FAILED", "RUNTIME_UNAVAILABLE"].includes(normalizedStatus(value || ""));
}

function sleep(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function nextDemoImageIndex(current: number, count: number) {
  if (count < 1) return 0;
  return (current + 1) % count;
}

function randomDemoImageIndex(current: number, count: number) {
  if (count < 2) return 0;
  let next = Math.floor(Math.random() * count);
  if (next === current) next = (next + 1) % count;
  return next;
}

function uniqueBy<T>(items: T[], keyForItem: (item: T, index: number) => string) {
  const seen = new Set<string>();
  return items.filter((item, index) => {
    const key = keyForItem(item, index);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function proposedExperimentSummaries(value: unknown) {
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

function rejectedOptionSummaries(value: unknown) {
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

function classifyRejectionReason(text: string) {
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

function uniqueStrings(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function formatTopKScore(value: unknown) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 0;
  if (!numeric || !Number.isFinite(numeric)) return "";
  return numeric <= 1 ? `${Math.round(numeric * 100)}%` : numeric.toFixed(3);
}

function numericValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

function championModelProfile(champion: ProjectChampion): Record<string, unknown> {
  const deploymentProfile = recordObject(champion.deployment_profile.model_profile);
  if (Object.keys(deploymentProfile).length > 0) return deploymentProfile;
  return recordObject(champion.evaluation.model_profile);
}

function championConfusionMatrix(champion: ProjectChampion) {
  return normalizedConfusionMatrix(champion.evaluation.confusion_matrix);
}

function normalizedConfusionMatrix(value: unknown) {
  const matrix = value;
  if (!Array.isArray(matrix)) return [];
  return matrix
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((value) => (typeof value === "number" ? value : Number(value) || 0)));
}

function perClassMetricRows(metrics: Record<string, unknown>) {
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

function formatCurrency(value: number) {
  return `$${value.toFixed(value < 1 ? 4 : 2)}`;
}

function formatSeconds(value: number) {
  if (!value) return "0s";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatMaybeMetric(value: number) {
  if (!value) return "-";
  return value.toFixed(3);
}

function formatDecisionQualityCount(value: number | null) {
  return value === null || !Number.isFinite(value) ? "-" : String(value);
}

function formatDecisionQualityMetric(value: number | null, signed: boolean) {
  if (value === null || !Number.isFinite(value)) return "-";
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function formatPercent(value: number) {
  if (!Number.isFinite(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatLossGap(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function formatSeedVariance(value: number | null, runCount: number) {
  if (value === null || runCount < 2 || !Number.isFinite(value)) return "-";
  return `${value.toFixed(5)} (${runCount})`;
}

function formatMetricNumber(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  return value.toFixed(3);
}

function formatLatency(value: unknown) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 0;
  if (!numeric || !Number.isFinite(numeric)) return "-";
  return `${numeric.toFixed(numeric < 10 ? 2 : 1)}ms`;
}

function formatModelSize(profile: Record<string, unknown>) {
  const size = recordNumber(profile, "model_size_mb") || recordNumber(profile, "estimated_model_size_mb");
  if (!size) return "-";
  return `${size.toFixed(size < 10 ? 2 : 1)} MB`;
}

function formatChartValue(value: number) {
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4);
}

function formatUnknownValue(value: unknown) {
  if (typeof value === "number") return formatMetricNumber(value);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  return JSON.stringify(value);
}
