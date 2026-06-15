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
import {
  cachedGetRequestTtlMs,
  isOrchestratorHttpErrorResponse,
  type CachedGetRequest,
  type OrchestratorHttpErrorResponse,
  type RequestOptions,
} from "./api/missionControlClient";
import {
  emptyProjectDetail,
  type DatasetMetadataDetail,
  type ProjectDetail,
  type VisualAnalysisDetail,
} from "./hooks/useProjectDetail";
import { eventNeedsSlowProjectRefresh, type ActivityStreamState } from "./hooks/useActivityStream";
import { useWorkerSupervisor } from "./hooks/useWorkerSupervisor";
import {
  projectTabs,
  type ActivityFilterKey,
  type ProjectTabKey,
  type ProjectTabTarget,
  type ProjectWorkflowTab,
  type ProjectWorkflowTabBase,
  type ProjectWorkflowTabState,
} from "./features/mission/workflowTabs";
import {
  MetricKey,
  CancelExecutionResponse,
  MissionDigestState,
  MissionTone,
  MissionHealthItem,
  MissionActionKey,
  MissionNextAction,
  MissionLiveActivity,
  MissionSignal,
  MissionChampionSummary,
  MissionDigest,
  MissionBrief,
  AIThinking,
  MissionStage,
  ActivityCardType,
  ActivityCardModel,
  ResultsCandidate,
  ResultsSummary,
  ExportSummary,
  TelemetryWindowKey,
  TelemetryWindowSummary,
  TelemetryCountSummary,
  TelemetrySectionSummary,
  TelemetryEmbeddingBreakdownRow,
  TelemetryInvocationSummary,
  TelemetryEmbeddingPurposeSummary,
  TelemetrySummary,
  VisualAnalysisListResponse,
  AgentInvocationsResponse,
  ProjectDetailRefreshOptions,
  DatasetMetadataSummaryResponse,
  DatasetMetadataImportsResponse,
  TimelineItem,
  InsightItem,
  MetadataCountRow,
  MetadataStatusDisplay,
  ReasoningSection,
  ChampionComparisonRow,
  CandidateScoreRow,
  RetrievedMemoryDisplay,
  MemoryRetrievalProbeSnapshot,
  AgentInvocationAuditRow,
  DecisionChatTurn,
  MechanismCoverageRow,
  DecisionQualitySnapshot,
  ChampionExportDemo,
  ChampionFeedbackRating,
  ChampionFeedbackDraft,
  Notice,
  DatasetFolder,
  DatasetUploadWarning,
  DatasetUploadPreflight,
  ScheduleFollowUpResponse,
  AutomationSettingsUpdate,
  classificationMetricPriority,
  detectionMetricPriority,
  detectionMetricAliases,
  activityStorageUriPattern,
  activityWindowsPathPattern,
  activityUnixPathPattern,
  activityBase64Pattern,
  activitySecretPattern,
  metricLabel,
  isDetectionJob,
  isDetectionRun,
  firstPositiveMetric,
  yoloMetricFromEvaluation,
  runPrimaryMetric,
  championPrimaryMetric,
  metricTabOptions,
  NoticeBanner,
  noticeDisplay,
  summarizeTrainingRuns,
  trainingRunCacheSummary,
  trainingRunLifecycleChips,
  lifecycleSecondsChip,
  workerRequirementMaterializationSummary,
  shortCacheKey,
  projectTabFromTarget,
  missionStateLabelFromProject,
  missionStateToneFromProject,
  buildMissionBrief,
  buildCurrentThinking,
  buildMissionStages,
  buildActivityFeed,
  buildResultsSummary,
  buildExportSummary,
  userFacingActivityText,
  userFacingActionLabel,
  firstProposedExperiment,
  missionDecisionToUserText,
  decisionRationaleSummary,
  datasetReviewSummary,
  activityCardFromEvent,
  activityCardFromDecision,
  activityCardFromRun,
  activityCardMatchesFilter,
  candidateWhyText,
  resultsLearningSummary,
  resultsRemainingRisks,
  buildMissionDigest,
  buildMissionLiveActivity,
  missionDatasetProfiled,
  missionWorkerSummary,
  missionFacts,
  missionStateCopy,
  buildMissionHealth,
  buildMissionNextActions,
  buildMissionSignals,
  buildMissionChampionSummary,
  missionLiveStepFromEvent,
  missionLiveStatus,
  missionActivityLabelForEvent,
  missionActivityTargetForEvent,
  missionLiveActivityDetail,
  missionEnsureLiveStep,
  missionSignalDetailForEvent,
  missionDecisionLabel,
  missionStateLabel,
  missionHealthLabel,
  missionToneRank,
  activityEventFromMessage,
  mergeActivityEvents,
  buildFallbackActivityEvents,
  fallbackActivityFromExecutionEvent,
  fallbackActivityFromInvocation,
  fallbackActivityFromJobs,
  agentActivityCurrentState,
  activitySeverityIcon,
  activityStreamBadge,
  activityEventSubtitle,
  activityMetadataRows,
  activityMetadataObject,
  activityMetadataDisplayValue,
  fallbackActivityMetadata,
  activitySafeDisplayText,
  activitySeverity,
  activityStatus,
  activityTimestamp,
  jobActivityTimestamp,
  jobStatusCounts,
  recordBoolean,
  buildExperimentTimeline,
  buildDatasetIntelligence,
  datasetMetadataSummaryFromResponse,
  datasetMetadataSummaryFromImports,
  datasetMetadataSummaryFromImport,
  datasetMetadataImportsFromResponse,
  arrayDatasetMetadataImports,
  hasDatasetMetadataImportShape,
  hasDatasetMetadataSummaryShape,
  datasetMetadataSummaryFallback,
  metadataImportSortScore,
  buildMetadataStatusDisplay,
  metadataSummaryStatus,
  metadataDetailText,
  metadataSourceTags,
  metadataCountRows,
  metadataCountText,
  metadataStringList,
  metadataStringValue,
  metadataIssueSummaries,
  metadataIssueSummary,
  metadataPrimitiveText,
  metadataNumber,
  metadataNumericValue,
  metadataCountValue,
  metadataBBoxValue,
  metadataBBoxTone,
  metadataCoverageText,
  visualAnalysesFromResponse,
  visualAnalysisRerunPolicyFromResponse,
  arrayVisualAnalyses,
  hasVisualAnalysisShape,
  latestVisualAnalysis,
  visualAnalysisSortScore,
  visualAnalysisFromProfile,
  agentInvocationsFromResponse,
  buildAgentInvocationAuditRows,
  buildMissionControlTelemetrySummary,
  telemetryWindowsFromInvocations,
  telemetryWindowSummary,
  telemetryCountSummaryFromInvocations,
  telemetryInvocationSummaries,
  telemetryInvocationSummary,
  telemetryInvocationUsage,
  telemetryInvocationRuntime,
  telemetryPromptSectionSummaries,
  telemetryApproximateTokensFromInput,
  telemetryApproximateTokensFromText,
  telemetryInvocationPromptBytes,
  telemetryInvocationSections,
  telemetrySectionsFromSnapshot,
  telemetrySectionEstimateList,
  telemetryEmbeddingSummaryFromUsageEvents,
  telemetryEmbeddingPurposeSummary,
  telemetryEmbeddingBreakdownSummary,
  telemetryEmbeddingPurpose,
  telemetryEmbeddingEstimatedCost,
  telemetryEmbeddingPricingForModel,
  telemetryPricingForModel,
  telemetryInvocationTimestamp,
  telemetryTimestampValue,
  sumTelemetry,
  byteLengthOfText,
  byteLengthOfJson,
  invocationRuntimeRecord,
  invocationTarget,
  invocationDecisionLink,
  visualAnalysisFacts,
  visualStatusTone,
  visualAnalysisStatusBadge,
  visualAnalysisActiveJob,
  visualAnalysisRerunDisabledReason,
  visualAnalysisPolicyRunCount,
  visualAnalysisPolicyReadiness,
  visualAnalysisPolicySummary,
  visualAnalysisLimitations,
  visualCoverageSummary,
  visualPerClassCoverageRows,
  buildDecisionChatTurns,
  decisionChatQuestion,
  decisionChatOpening,
  decisionReasoningSections,
  decisionRejections,
  buildDecisionQualitySnapshot,
  latestDecisionWithQualityContext,
  latestProjectTrajectoryFromInvocations,
  latestInvocationTimestamp,
  firstNonEmptyRecord,
  topCandidateRejectionReason,
  decisionQualityMechanismOutcomes,
  decisionQualityOutcomeDetail,
  recordFirstNumber,
  timestampSortScore,
  buildChampionComparison,
  buildSeedVarianceBySignature,
  experimentRankScore,
  experimentComparisonSignature,
  normalizeComparisonConfig,
  stableStringify,
  experimentSeed,
  clamp01,
  buildMemoryRetrievalProbeSnapshots,
  humanizeMemoryPurpose,
  memoryPurposeSuffix,
  candidateScoreRows,
  decisionRetrievedMemoryRows,
  candidateRetrievedMemoryRows,
  candidateMemoryReasons,
  candidateScoreComponentRows,
  memoryCardsFromUnknown,
  memoryCardsFromUnknownDepth,
  memoryDisplayFromRecord,
  isMemoryCardLike,
  isMemoryContainerKey,
  firstMemoryString,
  memoryIdentifierRows,
  memoryOutcome,
  humanizeMemorySource,
  memoryReasonStrings,
  memoryScoreComponentSummaries,
  isMemoryComponentLabel,
  isMemoryReasonText,
  memoryDisplayKey,
  safeMemoryText,
  buildChampionExportDemo,
  championFeedbackMetricsSnapshot,
  feedbackRatingLabel,
  normalizeChampionFeedbackResponse,
  isChampionFeedback,
  experimentPreprocessingItems,
  classBalancingConfigSummary,
  augmentationPolicyConfigSummary,
  experimentAutoMLItems,
  experimentMechanismItems,
  jobMechanismSummary,
  backendGateSummary,
  decisionHistorySummary,
  decisionHighlights,
  mechanismDecisionSummaries,
  mechanismSummaryFromRecord,
  mechanismCoverageRows,
  addCoverageRows,
  coverageDetail,
  automationReviewState,
  numberPayload,
  recordString,
  recordFirstString,
  recordNumber,
  recordObject,
  firstAuditValue,
  firstAuditString,
  auditCountValue,
  auditToolNamesFromValue,
  auditRejectedToolCallSummaries,
  hasAuditRejectionShape,
  auditRejectionItem,
  auditDryRunValidationSummaries,
  auditDryRunValidationItem,
  hasAuditValidationShape,
  auditPrimitiveSummary,
  shortAuditText,
  isLikelyEncodedPayload,
  isSensitiveAuditKey,
  humanizeAuditKey,
  normalizedStatus,
  statusToneClass,
  exportStatusMessage,
  predictionStatusMessage,
  objectSummary,
  exemplarStatusFromProfile,
  stringArrayPayload,
  championExportsFromUnknown,
  championExportExternalData,
  firstChampionArtifactMatchingFormat,
  championExportFormatFromArtifact,
  championExportValidationErrorsForFormat,
  artifactMatchesChampionExportFormat,
  artifactComparablePath,
  demoImagesFromUnknown,
  demoPredictionsFromUnknown,
  championExportDemoIsDetection,
  detectionDefaultsFromChampionExportDemo,
  detectionBoxesFromPrediction,
  normalizeDetection,
  normalizedDetectionBox,
  boundedDetectionBox,
  predictionPostprocessLatency,
  normalizeDemoPredictionResponse,
  attachDemoPredictionPreview,
  demoImageURI,
  demoImagePreviewURI,
  demoImageLabel,
  demoImageDetail,
  demoPredictionRequestMetadata,
  isTerminalDemoPredictionStatus,
  sleep,
  nextDemoImageIndex,
  randomDemoImageIndex,
  uniqueBy,
  proposedExperimentSummaries,
  rejectedOptionSummaries,
  classifyRejectionReason,
  uniqueStrings,
  numericValue,
  championModelProfile,
  championConfusionMatrix,
  normalizedConfusionMatrix,
  perClassMetricRows,
  formatModelSize,
} from "./features/mission/projectDetailModel";
import {
  buildProjectWorkflowTabs,
  Panel,
  MissionRoute,
  MissionEmptyState,
  ThinkingRow,
  MissionStageTimeline,
  ActivityRoute,
  ActivityEmptyState,
  ActivityCard,
  ResultsRoute,
  ResultsEmptyState,
  ExportWaitingState,
  ExportRoute,
  MissionOverview,
  MissionStatusPanel,
  MissionHealthStrip,
  MissionNextActions,
  LiveAgentActivity,
  MissionSignals,
  ChampionOutcomeSummary,
  AgentActivityPanel,
  AgentDecisionChat,
  RetrievedMemoryPanel,
  MemoryRetrievalProbePanel,
  DecisionQualityPanel,
  AgentInvocationAuditPanel,
  MissionControlTelemetryPanel,
  VisualAnalysisPanel,
  ChampionExportDemoPanel,
  DetectionOverlay,
  PredictionRow,
  RunEvaluationDetails,
  MetricCard,
  Badge,
  MetricChart,
} from "./features/mission/ProjectRoutePanels";
import {
  buildDeveloperDiagnostics,
  DeveloperRoute,
  type DeveloperDiagnostics,
} from "./features/developer/DeveloperRoute";
import { activityFilters } from "./features/activity/activityFilters";
import { resultsEmptySteps } from "./features/results/resultsEmptySteps";
import { exportWaitingSteps } from "./features/exportDemo/exportWaitingSteps";
import {
  formatBytes,
  formatChartValue,
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
} from "./utils/formatting";
import { errorMessage, isUnsupportedEndpointError, shortValue } from "./utils/safeDisplay";
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
  const eventRefreshInFlight = useRef(false);
  const eventRefreshTimer = useRef<number | null>(null);
  const eventRefreshQueuedSlow = useRef(false);
  const lastEventRefreshAt = useRef(0);
  const liveRefreshInFlight = useRef(false);
  const cachedGetRequests = useRef<Map<string, CachedGetRequest>>(new Map());

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

  const { resetWorkerSupervisor, superviseWorkerRequirements } = useWorkerSupervisor({
    baseUrl,
    selectedProjectId,
    workerRequirements: detail.workerRequirements,
    jobs: detail.jobs,
    request,
    refreshProjectDetail,
  });

  useEffect(() => {
    localStorage.setItem("orchestratorUrl", baseUrl);
  }, [baseUrl]);

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
      resetWorkerSupervisor();
      setJobPage(0);
      refreshProjectDetail(selectedProjectId, { includeSlowData: true, forceSlowData: true }).catch((error) =>
        setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
      );
    }
  }, [refreshProjectDetail, resetWorkerSupervisor, selectedProjectId]);

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

