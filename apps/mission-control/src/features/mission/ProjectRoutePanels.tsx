import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Database,
  Download,
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

import { championLocalInferenceSafety, readyONNXExport, type ChampionLocalRuntime } from "../../championLocalInference";
import { activityFilters } from "../activity/activityFilters";
import { exportWaitingSteps } from "../exportDemo/exportWaitingSteps";
import { resultsEmptySteps } from "../results/resultsEmptySteps";
import type { ActivityStreamState } from "../../hooks/useActivityStream";
import type { DatasetMetadataDetail, ProjectDetail, ProjectDetailLoadStatus, VisualAnalysisDetail } from "../../hooks/useProjectDetail";
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
} from "../../utils/formatting";
import { errorMessage, shortValue } from "../../utils/safeDisplay";
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
} from "../../types";
import {
  projectTabs,
  type ActivityFilterKey,
  type ProjectTabKey,
  type ProjectTabTarget,
  type ProjectWorkflowTab,
  type ProjectWorkflowTabBase,
  type ProjectWorkflowTabState,
} from "./workflowTabs";
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
  ModelImprovementData,
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
  metricTechnicalLabel,
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
  portableBundleDownloadable,
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
  demoImageIsRunnable,
  demoImageURI,
  demoImagePreviewURI,
  demoImageLabel,
  demoImageDetail,
  demoImageCategory,
  demoImageCategoryDetail,
  demoImageCategoryLabel,
  demoImageTrainingPredictionText,
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
} from "./projectDetailModel";

export function Panel({
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

export function MissionRoute({
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
  const trialRows = missionTrialProgressRows(brief.trialProgress);
  const hasTrialProgress = brief.trialProgress.total > 0;

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
        <div className="mission-live-briefing">
          <div className="mission-briefing-copy">
            <div className="eyebrow">Live briefing</div>
            <strong>{brief.plainSummary}</strong>
            <span>{brief.rightNow}</span>
          </div>
          <div className="trial-progress-visual" aria-label="Model trial progress distribution">
            <div className="trial-progress-head">
              <span>
                <small>Model trial progress</small>
                <strong>{hasTrialProgress ? `${brief.trialProgress.completed}/${brief.trialProgress.total} complete` : "Waiting for plan"}</strong>
              </span>
              <Badge value={brief.statusLabel} />
            </div>
            <div className="trial-distribution-bar" aria-hidden="true">
              {trialRows.map((row) => (
                <span
                  className={row.key}
                  key={row.key}
                  style={{ width: `${hasTrialProgress ? Math.max(5, (row.value / Math.max(1, brief.trialProgress.total)) * 100) : 25}%` }}
                />
              ))}
            </div>
            <div className="trial-distribution-table">
              {trialRows.map((row) => (
                <span className={row.key} key={row.key}>
                  <small>{row.label}</small>
                  <strong>{row.value}</strong>
                </span>
              ))}
            </div>
          </div>
          <div className="mission-next-decision">
            <small>Next decision</small>
            <strong>{brief.nextDecision}</strong>
          </div>
        </div>
        <div className="mission-flow-console" aria-label="Mission state console">
          <span className={`mission-console-cell ${activeStage.status}`}>
            <small>Now</small>
            <strong>{activeStage.label}</strong>
          </span>
          <span className="mission-console-cell">
            <small>Next</small>
            <strong>{primaryAction ? userFacingActionLabel(primaryAction.label) : "Monitoring"}</strong>
          </span>
          <button className="mission-console-cell interactive" type="button" onClick={() => onOpenTab("activity", "activity")}>
            <small>Latest</small>
            <strong>{latestActivity ? userFacingActivityText(latestActivity.title, 52) : "No journal entries yet"}</strong>
          </button>
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
            <em>{metricTechnicalLabel(brief.bestMetricLabel)}</em>
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
            <div className="eyebrow">Why this step</div>
            <strong>{thinking.state}</strong>
          </div>
          <small>{thinking.updatedAt ? formatRelativeTime(thinking.updatedAt) : thinking.confidenceLabel}</small>
        </div>
        <div className="thinking-grid">
          <ThinkingRow label="Observation" value={thinking.observation} />
          <ThinkingRow label="Decision" value={thinking.decision} />
          <ThinkingRow label="Impact" value={thinking.expectedOutcome} />
        </div>
        <details className="activity-details">
          <summary>Developer reasoning</summary>
          <ThinkingRow label="Reasoning" value={thinking.reasoning} />
        </details>
      </section>

      <aside className="mission-inspector">
        <section className="result-snapshot-card">
          <div className="mission-section-head">
            <div>
              <div className="eyebrow">Best model so far</div>
              <strong>{results.championModel}</strong>
            </div>
            <Badge value={exportSummary.statusLabel} />
          </div>
          <div className="champion-outcome-primary">
            <small>{results.primaryMetricLabel}</small>
            <strong>{results.primaryMetricValue}</strong>
          </div>
          <p>{results.leaderComparison}</p>
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

export function MissionEmptyState({
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

export function ThinkingRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="thinking-row">
      <small>{label}</small>
      <p>{value}</p>
    </div>
  );
}

export function MissionStageTimeline({ stages }: { stages: MissionStage[] }) {
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
  const latestDoneIndex = stages.reduce(
    (index, stage, stageIndex) => (stage.status === "done" ? stageIndex : index),
    -1,
  );
  const searchStages = latestDoneIndex >= 0 ? stages.slice(latestDoneIndex + 1) : stages;

  if (
    latestDoneIndex === -1 &&
    searchStages.length > 0 &&
    searchStages.every((stage) => stage.status === "waiting")
  ) {
    return {
      id: "empty",
      label: "Waiting for mission",
      detail: stages[0]?.detail || "Create or select a mission to begin.",
      status: "waiting",
    };
  }

  return (
    searchStages.find((stage) => stage.status === "blocked") ??
    searchStages.find((stage) => stage.status === "active") ??
    searchStages.find((stage) => stage.status === "waiting") ??
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
  const latestDoneIndex = stages.reduce(
    (index, stage, stageIndex) => (stage.status === "done" ? stageIndex : index),
    -1,
  );
  const activeIndex = stages
    .slice(Math.max(0, latestDoneIndex + 1))
    .findIndex((stage) => stage.status === "active" || stage.status === "blocked");
  if (activeIndex >= 0) {
    const absoluteActiveIndex = Math.max(0, latestDoneIndex + 1) + activeIndex;
    return Math.round(((absoluteActiveIndex + 0.55) / stages.length) * 100);
  }
  const completed = stages.filter((stage) => stage.status === "done").length;
  return Math.round((completed / stages.length) * 100);
}

export function buildProjectWorkflowTabs({
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

function missionTrialProgressRows(progress: MissionBrief["trialProgress"]) {
  return [
    { key: "completed", label: "Completed", value: progress.completed },
    { key: "running", label: "Running", value: progress.running },
    { key: "waiting", label: "Waiting", value: progress.waiting },
    { key: "failed", label: "Failed", value: progress.failed },
  ];
}

function activityStatusLabel(status: ActivityCardModel["status"]) {
  if (status === "blocked") return "Action required";
  if (status === "waiting") return "Waiting";
  if (status === "active") return "Running";
  if (status === "succeeded") return "Done";
  return "Failed";
}

export function ActivityRoute({
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
        {activityFilters.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={filter === item.key}
            className={filter === item.key ? "active" : ""}
            onClick={() => onFilterChange(item.key)}
          >
            {item.label}
            <small>{cards.filter((card) => activityCardMatchesFilter(card, item.key)).length}</small>
          </button>
        ))}
      </div>
      <div className="activity-timeline-table">
        {visibleCards.length > 0 ? (
          <>
            <div className="activity-timeline-row activity-timeline-head">
              <span>Time</span>
              <span>Event</span>
              <span>Evidence / result</span>
              <span>Impact</span>
              <span>Status</span>
            </div>
            {visibleCards.map((card) => (
              <details className={`activity-timeline-row ${card.type} ${card.status}`} key={card.id}>
                <summary>
                  <span>{formatRelativeTime(card.timestamp)}</span>
                  <span>
                    <strong>{card.title}</strong>
                    <small>{card.type}</small>
                  </span>
                  <span>{card.evidenceSummary || card.resultSummary || "Evidence pending"}</span>
                  <span>{card.summary}</span>
                  <span>
                    <Badge value={activityStatusLabel(card.status)} />
                  </span>
                </summary>
                <div className="activity-timeline-detail">
                  <span>
                    <small>Result</small>
                    <strong>{card.resultSummary || "Result pending"}</strong>
                  </span>
                  <span>
                    <small>Technical source</small>
                    <strong>{card.technicalSource}</strong>
                  </span>
                  <button className="mission-link-button" type="button" onClick={onOpenDeveloper}>
                    Open Developer View
                  </button>
                </div>
              </details>
            ))}
          </>
        ) : (
          <ActivityEmptyState hasCards={cards.length > 0} filter={filter} />
        )}
      </div>
    </div>
  );
}

export function ActivityEmptyState({ hasCards, filter }: { hasCards: boolean; filter: ActivityFilterKey }) {
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

export function ActivityCard({ card, onOpenDeveloper }: { card: ActivityCardModel; onOpenDeveloper: () => void }) {
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

export function ResultsRoute({
  summary,
  modelImprovement,
  onSelectCandidate,
  onOpenExport,
  onOpenDeveloper,
}: {
  summary: ResultsSummary;
  modelImprovement: ModelImprovementData;
  onSelectCandidate: (jobId: string) => void;
  onOpenExport: () => void;
  onOpenDeveloper: () => void;
}) {
  const leader = summary.topCandidates[0] ?? null;
  return (
    <div className="results-page">
      <section className="results-hero">
        <div>
          <div className="eyebrow">Best model so far</div>
          <h3>{summary.championModel}</h3>
          <p>{summary.leaderComparison}</p>
        </div>
        <div className="results-primary-metric">
          <small>{summary.primaryMetricLabel}</small>
          <strong>{summary.primaryMetricValue}</strong>
          <span>{summary.improvementLabel}</span>
        </div>
      </section>

      <ModelImprovementChart data={modelImprovement} />

      <section className="candidate-section">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Leaderboard</div>
            <strong>{leader ? `${leader.model} leads the completed model trials` : "Waiting for comparable model trials"}</strong>
            <small>{summary.scoreContext}</small>
          </div>
          <span className="results-actions">
            <Badge value={summary.exportStatus} />
            <button className="command compact" type="button" onClick={onOpenExport}>
              Test & Export
            </button>
            <button className="command compact" type="button" onClick={onOpenDeveloper}>
              Developer comparison
            </button>
          </span>
        </div>
        <div className="leaderboard-table">
          {summary.topCandidates.length > 0 ? (
            <>
              <div className="leaderboard-row leaderboard-head">
                <span>Model</span>
                <span>Balanced score</span>
                <span>Accuracy</span>
                <span>Latency</span>
                <span>Size</span>
                <span>Cost</span>
                <span>Status</span>
              </div>
              {summary.topCandidates.map((candidate) => (
              <button
                className={candidate.rank === 1 ? "leaderboard-row leader" : "leaderboard-row"}
                key={`${candidate.rank}-${candidate.jobId || candidate.model}`}
                type="button"
                disabled={!candidate.jobId}
                onClick={() => candidate.jobId && onSelectCandidate(candidate.jobId)}
              >
                <span>
                  <small>#{candidate.rank}</small>
                  <strong>{candidate.model}</strong>
                  <small>{candidate.why}</small>
                </span>
                <span>
                  <small>{candidate.metricLabel}</small>
                  <strong>{candidate.metricValue}</strong>
                  <small>{candidate.metricTechnicalLabel}</small>
                </span>
                <span>{candidate.accuracyValue}</span>
                <span>{candidate.latency}</span>
                <span>{candidate.modelSize}</span>
                <span>{candidate.cost}</span>
                <span><Badge value={candidate.status} /></span>
              </button>
              ))}
            </>
          ) : (
            <ResultsEmptyState hasResults={summary.hasResults} />
          )}
        </div>
      </section>

      <section className="results-grid">
        <div className="results-section">
          <div className="eyebrow">Why the leader is ahead</div>
          <p>{summary.whyItWon}</p>
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

      {(summary.perClassRows.length > 0 || summary.confusionMatrix.length > 0) && (
        <section className="results-evidence-grid">
          {summary.perClassRows.length > 0 && (
            <div className="results-section">
              <div className="eyebrow">Per-class evidence</div>
              <div className="results-per-class-table">
                <div className="results-per-class-row head">
                  <span>Class</span>
                  <span>Precision</span>
                  <span>Recall</span>
                  <span>F1</span>
                </div>
                {summary.perClassRows.map((row) => (
                  <div className="results-per-class-row" key={row.label}>
                    <span>{row.label}</span>
                    <span>{row.precision}</span>
                    <span>{row.recall}</span>
                    <span>{row.f1}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {summary.confusionMatrix.length > 0 && (
            <div className="results-section">
              <div className="eyebrow">Confusion preview</div>
              <div className="confusion-preview">
                {summary.confusionMatrix.map((row, rowIndex) => (
                  <div key={`results-matrix-${rowIndex}`}>
                    {row.map((value, columnIndex) => (
                      <span key={`${rowIndex}-${columnIndex}`}>{value}</span>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export function ModelImprovementChart({ data }: { data: ModelImprovementData }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const points = data.points;
  const scoredPoints = points.filter((point) => point.bestScore !== null);
  const bestPoint = scoredPoints.reduce<ModelImprovementData["points"][number] | null>((currentBest, point) => {
    if (!currentBest || (point.bestScore ?? -1) > (currentBest.bestScore ?? -1)) return point;
    return currentBest;
  }, null);
  const latestScoredPoint = scoredPoints[scoredPoints.length - 1] ?? null;

  if (data.state !== "ready") {
    const empty = modelImprovementEmptyCopy(data);
    return (
      <section className="model-improvement-section">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Model Improvement</div>
            <strong>{empty.title}</strong>
            <small>{empty.detail}</small>
          </div>
          <Badge value={empty.badge} />
        </div>
        <div className="model-improvement-empty">
          <span className="results-empty-mark" aria-hidden="true">
            <BarChart3 size={18} />
          </span>
          <div>
            <strong>{empty.title}</strong>
            <p>{empty.detail}</p>
          </div>
        </div>
      </section>
    );
  }

  const width = 840;
  const height = 300;
  const paddingLeft = 50;
  const paddingRight = 24;
  const paddingTop = 26;
  const paddingBottom = 58;
  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;
  const xForIndex = (index: number) =>
    points.length === 1 ? paddingLeft + plotWidth / 2 : paddingLeft + (index / Math.max(1, points.length - 1)) * plotWidth;
  const yForScore = (score: number) => paddingTop + (1 - Math.max(0, Math.min(1, score))) * plotHeight;
  const cumulativePoints = points
    .map((point, index) =>
      point.cumulativeBestScore === null
        ? null
        : { point, x: xForIndex(index), y: yForScore(point.cumulativeBestScore) },
    )
    .filter((point): point is { point: ModelImprovementData["points"][number]; x: number; y: number } => Boolean(point));
  const bestSegments = points.slice(1).flatMap((point, index) => {
    const previous = points[index];
    if (previous.bestScore === null || point.bestScore === null) return [];
    return [
      {
        from: { x: xForIndex(index), y: yForScore(previous.bestScore) },
        to: { x: xForIndex(index + 1), y: yForScore(point.bestScore) },
      },
    ];
  });
  const cumulativePath = modelImprovementLinePath(cumulativePoints);
  const labelEvery = Math.max(1, Math.ceil(points.length / 8));
  const hovered = hoveredIndex === null ? null : points[hoveredIndex] ?? null;
  const hoveredScore = hovered?.bestScore ?? hovered?.cumulativeBestScore ?? 0.5;
  const hoveredX = hoveredIndex === null ? 0 : xForIndex(hoveredIndex);
  const hoveredY = yForScore(hoveredScore);
  const tooltipWidth = 214;
  const tooltipHeight = 106;
  const tooltipX = Math.min(Math.max(hoveredX - tooltipWidth / 2, paddingLeft), width - paddingRight - tooltipWidth);
  const tooltipY = Math.max(8, hoveredY - tooltipHeight - 12);
  const improvementLabel =
    data.improvementDelta === null ? "-" : `${data.improvementDelta >= 0 ? "+" : ""}${formatModelImprovementScore(data.improvementDelta)}`;

  return (
    <section className="model-improvement-section">
      <div className="mission-section-head">
        <div>
          <div className="eyebrow">Model Improvement</div>
          <strong>{bestPoint ? `${bestPoint.planLabel} holds the best score` : "Waiting for scored models"}</strong>
          <small>Best plan score and cumulative best across experiment plans.</small>
        </div>
        <Badge value={`${data.scoredPlanCount}/${data.totalPlans} scored`} />
      </div>

      <div className="model-improvement-stats">
        <span>
          <small>Best score</small>
          <strong>{formatModelImprovementScore(data.bestScore)}</strong>
          <em>{bestPoint?.bestModel || bestPoint?.jobId || "No scored model"}</em>
        </span>
        <span>
          <small>Latest scored plan</small>
          <strong>{latestScoredPoint ? latestScoredPoint.planLabel : "-"}</strong>
          <em>{latestScoredPoint?.source || "Score pending"}</em>
        </span>
        <span>
          <small>Improvement</small>
          <strong>{improvementLabel}</strong>
          <em>from first scored plan</em>
        </span>
      </div>

      <div className="model-improvement-chart-wrap">
        <svg className="model-improvement-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Model improvement by experiment plan">
          <defs>
            <linearGradient id="model-improvement-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#00d47e" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#00d47e" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[0, 0.25, 0.5, 0.75, 1].map((value) => {
            const y = yForScore(value);
            return (
              <g key={value}>
                <line className="model-improvement-grid" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
                <text className="model-improvement-axis-label" x={12} y={y + 4}>
                  {value.toFixed(2)}
                </text>
              </g>
            );
          })}
          <line className="model-improvement-axis" x1={paddingLeft} x2={width - paddingRight} y1={height - paddingBottom} y2={height - paddingBottom} />
          <line className="model-improvement-axis" x1={paddingLeft} x2={paddingLeft} y1={paddingTop} y2={height - paddingBottom} />
          {cumulativePath && (
            <>
              <path className="model-improvement-cumulative-fill" d={`${cumulativePath} L ${cumulativePoints[cumulativePoints.length - 1].x.toFixed(2)} ${(height - paddingBottom).toFixed(2)} L ${cumulativePoints[0].x.toFixed(2)} ${(height - paddingBottom).toFixed(2)} Z`} />
              <path className="model-improvement-line cumulative" d={cumulativePath} />
            </>
          )}
          {bestSegments.map((segment, index) => (
            <line
              className="model-improvement-line best"
              key={`best-${index}`}
              x1={segment.from.x}
              y1={segment.from.y}
              x2={segment.to.x}
              y2={segment.to.y}
            />
          ))}
          {points.map((point, index) => {
            const x = xForIndex(index);
            const slotWidth = points.length === 1 ? plotWidth : plotWidth / Math.max(1, points.length - 1);
            const hitWidth = Math.max(34, Math.min(96, slotWidth));
            const hasScore = point.bestScore !== null;
            const cumulativeY = point.cumulativeBestScore === null ? null : yForScore(point.cumulativeBestScore);
            return (
              <g className="model-improvement-point" key={point.planId || index}>
                <title>{modelImprovementPointTitle(point)}</title>
                {cumulativeY !== null && <circle className="model-improvement-dot cumulative" cx={x} cy={cumulativeY} r="3" />}
                {hasScore ? (
                  <circle className={hoveredIndex === index ? "model-improvement-dot best active" : "model-improvement-dot best"} cx={x} cy={yForScore(point.bestScore ?? 0)} r="5" />
                ) : (
                  <g className="model-improvement-missing" transform={`translate(${x} ${height - paddingBottom + 18})`}>
                    <line x1="-5" x2="5" y1="-5" y2="5" />
                    <line x1="-5" x2="5" y1="5" y2="-5" />
                  </g>
                )}
                {(index % labelEvery === 0 || index === points.length - 1) && (
                  <text className="model-improvement-x-label" x={x} y={height - 14} textAnchor="middle">
                    {point.planIndex}
                  </text>
                )}
                <rect
                  className="model-improvement-hit"
                  x={x - hitWidth / 2}
                  y={paddingTop}
                  width={hitWidth}
                  height={height - paddingTop - 8}
                  tabIndex={0}
                  onFocus={() => setHoveredIndex(index)}
                  onBlur={() => setHoveredIndex(null)}
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}
                />
              </g>
            );
          })}
          {hovered && (
            <g className="model-improvement-tooltip" transform={`translate(${tooltipX} ${tooltipY})`}>
              <rect width={tooltipWidth} height={tooltipHeight} rx="7" />
              {modelImprovementTooltipLines(hovered).map((line, index) => (
                <text key={`${hovered.planId}-${index}`} x="10" y={18 + index * 19} className={index === 1 ? "tooltip-value" : ""}>
                  {line}
                </text>
              ))}
            </g>
          )}
        </svg>
      </div>

      <div className="model-improvement-legend">
        <span><i className="best" /> Best score per plan</span>
        <span><i className="cumulative" /> Cumulative best</span>
        {data.missingScorePlanCount > 0 && <span><i className="missing" /> Missing plan score</span>}
      </div>
      {data.missingScorePlanCount > 0 && (
        <div className="model-improvement-note">
          {data.missingScorePlanCount} completed plan{data.missingScorePlanCount === 1 ? "" : "s"} had no score field, so the best-score line leaves a gap instead of plotting zero.
        </div>
      )}
    </section>
  );
}

function modelImprovementEmptyCopy(data: ModelImprovementData) {
  if (data.state === "no_plans") {
    return {
      title: "No plans yet",
      detail: "The improvement chart appears after Model Express creates an experiment plan.",
      badge: "No plans",
    };
  }
  if (data.state === "no_completed_plans") {
    return {
      title: "No completed plans yet",
      detail: "Plans exist, but no successful training runs have completed for charting.",
      badge: `${data.totalPlans} plan${data.totalPlans === 1 ? "" : "s"}`,
    };
  }
  return {
    title: "Plans exist, but no scored models yet",
    detail: "Completed runs were found, but the available records do not include model score fields. Missing scores stay blank rather than becoming zero.",
    badge: "Scores missing",
  };
}

function modelImprovementLinePath(points: Array<{ x: number; y: number }>) {
  if (points.length === 0) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function formatModelImprovementScore(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  return value.toFixed(3);
}

function shortModelImprovementText(value: unknown, maxLength = 58) {
  const text = shortValue(value);
  return text.length <= maxLength ? text : `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}
function modelImprovementPointTitle(point: ModelImprovementData["points"][number]) {
  return modelImprovementTooltipLines(point).join("; ");
}

function modelImprovementTooltipLines(point: ModelImprovementData["points"][number]) {
  const model = point.bestModel || point.jobId || "model pending";
  const provider = point.provider || "provider pending";
  const completion = point.completedAt ? formatTimestamp(point.completedAt) : "completion time pending";
  if (point.bestScore === null) {
    return [
      `${point.planLabel} (${point.targetMetric || "target pending"})`,
      "Score missing",
      shortModelImprovementText(point.missingReason),
      `${point.completedRunCount}/${point.totalRunCount || point.completedRunCount} completed`,
    ];
  }
  return [
    `${point.planLabel} (${point.targetMetric || "target pending"})`,
    `Score ${formatModelImprovementScore(point.bestScore)}`,
    shortModelImprovementText([model, provider].filter(Boolean).join(" - ")),
    shortModelImprovementText([point.source, point.scoreBasis, completion].filter(Boolean).join(" - ")),
  ];
}
export function ResultsEmptyState({ hasResults }: { hasResults: boolean }) {
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
        {resultsEmptySteps.map((step, index) => (
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

export function ExportWaitingState({ readinessLabel }: { readinessLabel: string }) {
  return (
    <section className="export-waiting-state">
      <span className="export-waiting-mark" aria-hidden="true">
        <Trophy size={18} />
      </span>
      <div className="export-waiting-copy">
        <div className="eyebrow">Test & Export</div>
        <strong>Waiting for the best model</strong>
        <p>{readinessLabel}</p>
      </div>
      <div className="export-waiting-steps" aria-label="Export readiness flow">
        {exportWaitingSteps.map((step, index) => (
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

export function ExportRoute({
  summary,
  data,
  prediction,
  predictionError,
  predictionLoading,
  selectedImageIndex,
  customImage,
  customImageURI,
  customTrueLabel,
  localInferenceStatus,
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
  onSavePortableBundle,
  savingPortableBundle,
  onRunPrediction,
  onOpenFeedback,
  onSlideshowIntervalChange,
  onDetectionConfidenceThresholdChange,
  onDetectionIouThresholdChange,
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
  onSavePortableBundle: (bundle: NonNullable<ChampionExportDemo["portableBundle"]>) => void;
  savingPortableBundle: boolean;
  onRunPrediction: (image: ChampionDemoImage) => void;
  onOpenFeedback: (rating: ChampionFeedbackRating) => void;
  onSlideshowIntervalChange: (value: number) => void;
  onDetectionConfidenceThresholdChange: (value: number) => void;
  onDetectionIouThresholdChange: (value: number) => void;
  onOpenDeveloper: () => void;
}) {
  const selectedImage = data.demoImages[selectedImageIndex] ?? data.demoImages[0] ?? null;
  const selectedImageRunnable = demoImageIsRunnable(selectedImage);
  const customURI = customImageURI.trim();
  const customImageMatchesPicker = customImage ? customURI === demoImageURI(customImage) : false;
  const customPreviewImage = customURI
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
  const previewURI = demoImagePreviewURI(activeImage);
  const predictionStatus = prediction ? normalizedStatus(prediction.status || "PENDING") : "";
  const confidence = prediction ? numericValue(prediction.confidence) : 0;
  const activeDetections = detectionBoxesFromPrediction(prediction);
  const topK = Array.isArray(prediction?.top_k) ? prediction.top_k : [];
  const portableBundle = data.portableBundle;
  const canSavePortableBundle = Boolean(portableBundle && portableBundleDownloadable(portableBundle));
  const localDemoAvailable = localChampionDemoRuntimeAvailable(data);

  if (!summary.hasChampion) {
    return (
      <div className="export-page" id="export-package">
        <ExportWaitingState readinessLabel={summary.readinessLabel} />
      </div>
    );
  }

  return (
    <div className="export-page" id="export-package">
      <section className="export-demo-simple test-export-hero">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Try the best model</div>
            <h3>{demoImageLabel(activeImage) || "Choose a held-out image"}</h3>
            <p>Run an image through the best model so far and check the prediction, confidence, known label, correctness, and latency.</p>
          </div>
          <span>
            <Badge value={predictionLoading ? "Running" : summary.demoStatus} />
            <button className="icon-command" type="button" onClick={onRandomImage} disabled={data.demoImages.length < 2} title="Random held-out image">
              <Shuffle size={14} />
            </button>
            <button className="icon-command" type="button" onClick={onNextImage} disabled={data.demoImages.length < 2} title="Next held-out image">
              <StepForward size={14} />
            </button>
          </span>
        </div>
        <div className="demo-simple-grid">
          <div className="test-image-stage">
            <div className="test-image-preview">
              {previewURI ? (
                <div className="test-image-frame">
                  <img src={previewURI} alt={demoImageLabel(activeImage) || "demo image"} />
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
                <strong>{demoImageLabel(activeImage) || activeImage?.image_id || "Select an image"}</strong>
              </span>
              <small>{demoImageDetail(activeImage) || "Held-out image or custom URI"}</small>
            </div>
          </div>

          <div className="demo-result-summary">
            {predictionError && <div className="mission-blocker"><AlertTriangle size={15} /><span>{userFacingActivityText(predictionError, 150)}</span></div>}
            {predictionLoading && <Badge value="Running" />}
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
                  <small>Known label</small>
                  <strong>{prediction.true_label || demoImageLabel(activeImage) || "Unknown"}</strong>
                </span>
                <span>
                  <small>Correctness</small>
                  <strong>{typeof prediction.correct === "boolean" ? (prediction.correct ? "Correct" : "Missed") : "Not scored"}</strong>
                </span>
                <span>
                  <small>Latency</small>
                  <strong>{typeof prediction.latency_ms === "number" ? formatLatency(prediction.latency_ms) : "-"}</strong>
                </span>
                <TopKConfidenceBars topK={topK} />
                <div className="feedback-actions" aria-label="Prediction feedback">
                  <button className="command compact" type="button" onClick={() => onOpenFeedback("good")} disabled={predictionLoading} title="Mark output good">
                    <ThumbsUp size={15} />
                    Good
                  </button>
                  <button className="command compact" type="button" onClick={() => onOpenFeedback("mediocre")} disabled={predictionLoading} title="Mark output mediocre">
                    <MessageSquare size={15} />
                    Mediocre
                  </button>
                  <button className="command compact" type="button" onClick={() => onOpenFeedback("bad")} disabled={predictionLoading} title="Mark output bad">
                    <ThumbsDown size={15} />
                    Bad
                  </button>
                </div>
              </>
            ) : (
              <div className="empty compact">Run a held-out image or choose your own image to see the prediction.</div>
            )}
          </div>

          <div className="test-controls export-test-controls">
            <div className="demo-block-head">
              <strong>Images</strong>
              <span>
                <button className="command compact" type="button" onClick={onToggleSlideshow} disabled={!localDemoAvailable || data.demoImages.length < 2 || predictionLoading}>
                  {slideshowEnabled ? <Pause size={15} /> : <Play size={15} />}
                  {slideshowEnabled ? "Pause" : "Slideshow"}
                </button>
                <button
                  className="command primary compact"
                  type="button"
                  onClick={() => selectedImage && onRunPrediction(selectedImage)}
                  disabled={!selectedImage || !selectedImageRunnable || predictionLoading}
                  title={selectedImageRunnable ? "Run held-out image" : "Original image unavailable for this held-out image"}
                >
                  <Play size={15} />
                  Run held-out
                </button>
              </span>
            </div>
            {data.demoImages.length > 0 ? (
              <div className="demo-image-strip">
                {data.demoImages.slice(0, 8).map((image, index) => (
                  <button
                    className={index === selectedImageIndex && !customURI ? "selected" : ""}
                    key={image.id || image.image_id || `${demoImageURI(image)}-${index}`}
                    type="button"
                    onClick={() => onSelectImage(index)}
                  >
                    {demoImagePreviewURI(image) ? <img src={demoImagePreviewURI(image)} alt={demoImageLabel(image) || "held-out image"} /> : <span>image</span>}
                    <small>{demoImageLabel(image) || image.image_id || "unlabeled"}</small>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty compact">No demo-ready held-out images are available yet.</div>
            )}
            {selectedImage && !selectedImageRunnable && !customURI && (
              <div className="mission-blocker">
                <AlertTriangle size={15} />
                <span>Original image unavailable. Choose another held-out image or use your own image.</span>
              </div>
            )}
            <div className="custom-image-actions">
              <button className="command compact" type="button" onClick={onChooseCustomImage}>
                <Upload size={15} />
                Choose Image
              </button>
              <button
                className="command compact"
                type="button"
                onClick={onRunCustomPrediction}
                disabled={!customURI || predictionLoading}
              >
                <Play size={15} />
                Run custom
              </button>
            </div>
            <label className="field">
              <span><Link2 size={12} /> Image URI</span>
              <input
                value={customImageURI}
                onChange={(event) => onCustomImageURIChange(event.target.value)}
                placeholder="file://, data:image, s3://, or worker-visible URI"
              />
            </label>
            <label className="field">
              <span>Known label</span>
              <input value={customTrueLabel} onChange={(event) => onCustomTrueLabelChange(event.target.value)} placeholder="optional" />
            </label>
            <label className="field compact-range">
              <span><Timer size={12} /> Slideshow speed</span>
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
            {championExportDemoIsDetection(data) && (
              <div className="detector-controls">
                <label className="field compact-range">
                  <span>Confidence</span>
                  <input type="range" min={0.01} max={0.95} step={0.01} value={detectionConfidenceThreshold} onChange={(event) => onDetectionConfidenceThresholdChange(Number(event.target.value))} />
                  <small>{formatTopKScore(detectionConfidenceThreshold)}</small>
                </label>
                <label className="field compact-range">
                  <span>IoU</span>
                  <input type="range" min={0.1} max={0.95} step={0.01} value={detectionIouThreshold} onChange={(event) => onDetectionIouThresholdChange(Number(event.target.value))} />
                  <small>{formatTopKScore(detectionIouThreshold)}</small>
                </label>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="export-package-card technical-package">
        <div className="mission-section-head">
          <div>
            <div className="eyebrow">Portable model package</div>
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
          <span>
            <small>Manifest</small>
            <strong>{summary.manifestAvailable ? "Available" : "Pending"}</strong>
          </span>
        </div>
        <div className="export-actions">
          <button className="command primary" type="button" onClick={onRequestExport}>
            <HardDriveUpload size={16} />
            Prepare portable model
          </button>
          {portableBundle && canSavePortableBundle && (
            <button className="command" type="button" onClick={() => onSavePortableBundle(portableBundle)} disabled={savingPortableBundle}>
              <Download size={16} />
              {savingPortableBundle ? "Saving" : "Save package"}
            </button>
          )}
          <button className="command" type="button" onClick={onOpenDeveloper}>
            Advanced details
          </button>
        </div>
        <details className="technical-package-details">
          <summary>Technical package</summary>
          <div className="technical-runtime-row">
            <span>
              <small>Demo route</small>
              <strong>{demoRuntimeLabel(localInferenceStatus)}</strong>
            </span>
          </div>
          <div className="export-record-list">
            {data.exports.length > 0 ? data.exports.map((exportRecord, index) => (
              <div className={`export-record ${statusToneClass(exportRecord.status)}`} key={exportRecord.id || `${exportRecord.format}-${index}`}>
                <span>
                  <strong>{exportRecord.format || "model artifact"}</strong>
                  <small>{exportRecord.artifact_uri || exportRecord.model_uri || exportRecord.download_url || exportStatusMessage(exportRecord.status)}</small>
                </span>
                <span>
                  <Badge value={exportRecord.status || "PENDING"} />
                  <small>{exportRecord.size_bytes ? formatBytes(exportRecord.size_bytes) : exportRecord.completed_at || exportRecord.updated_at || ""}</small>
                </span>
              </div>
            )) : <div className="empty compact">No package request has been recorded for this model yet.</div>}
          </div>
        </details>
      </section>

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
    </div>
  );
}

export function MissionOverview({
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

export function MissionStatusPanel({ digest }: { digest: MissionDigest }) {
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

export function MissionHealthStrip({
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

export function MissionNextActions({
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

export function LiveAgentActivity({
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

export function MissionSignals({
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

export function ChampionOutcomeSummary({
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
          Test & Export
        </button>
        <button className="command compact" type="button" onClick={() => onOpenTab("experiments", "champion-comparison")}>
          Compare Runs
        </button>
      </div>
    </section>
  );
}

export function AgentActivityPanel({
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

export function AgentDecisionChat({ turns }: { turns: DecisionChatTurn[] }) {
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

export function RetrievedMemoryPanel({ memories }: { memories: RetrievedMemoryDisplay[] }) {
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

export function MemoryRetrievalProbePanel({ snapshots }: { snapshots: MemoryRetrievalProbeSnapshot[] }) {
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

export function DecisionQualityPanel({
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

export function AgentInvocationAuditPanel({
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

export function MissionControlTelemetryPanel({
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

export function VisualAnalysisPanel({
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

function DetailLoadStatusNotice({ status }: { status?: ProjectDetailLoadStatus }) {
  if (!status || !["stale", "error"].includes(status.status)) {
    return null;
  }
  return (
    <div className={`notice-inline detail-load-status ${status.status === "error" ? "error" : "warning"}`} role={status.status === "error" ? "alert" : "status"}>
      <Badge value={status.status.toUpperCase()} />
      <span>{status.message}</span>
    </div>
  );
}

export function ChampionExportDemoPanel({
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
  onSavePortableBundle,
  savingPortableBundle,
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
  onSavePortableBundle: (bundle: NonNullable<ChampionExportDemo["portableBundle"]>) => void;
  savingPortableBundle: boolean;
  onRunPrediction: (image: ChampionDemoImage) => void;
  onOpenFeedback: (rating: ChampionFeedbackRating) => void;
  onSlideshowIntervalChange: (value: number) => void;
  onDetectionConfidenceThresholdChange: (value: number) => void;
  onDetectionIouThresholdChange: (value: number) => void;
}) {
  if (!data.hasChampion) {
    return <div className="empty">Test and export details will appear after Model Express selects the best model.</div>;
  }

  const selectedImage = data.demoImages[selectedImageIndex] ?? data.demoImages[0] ?? null;
  const selectedImageRunnable = demoImageIsRunnable(selectedImage);
  const selectedImageRunLabel = demoImageCategory(selectedImage) === "challenge" ? "Run challenge" : "Run example";
  const demoImageRows = data.demoImages.map((image, index) => ({ image, index }));
  const representativeDemoRows = demoImageRows.filter(({ image }) => demoImageCategory(image) === "representative");
  const challengeDemoRows = demoImageRows.filter(({ image }) => demoImageCategory(image) === "challenge");
  const heldoutDemoRows = demoImageRows.filter(({ image }) => demoImageCategory(image) === "heldout");
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
  const activeImageCategory = demoImageCategory(activeImage);
  const activeImageCategoryDetail = demoImageCategoryDetail(activeImage);
  const detectorDemo = championExportDemoIsDetection(data) || detectionBoxesFromPrediction(prediction).length > 0;
  const activeDetections = detectionBoxesFromPrediction(prediction);
  const activeFps = prediction?.latency_ms && prediction.latency_ms > 0 ? 1000 / prediction.latency_ms : 0;
  const postprocessLatency = predictionPostprocessLatency(prediction);
  const portableBundle = data.portableBundle;
  const championExportsStatus = data.championExportsStatus;
  const portableBundleDetail = portableBundle?.bytes
    ? formatBytes(portableBundle.bytes)
    : portableBundle?.contents?.length
      ? `${portableBundle.contents.length} file(s)`
      : portableBundle?.error || "";
  const canSavePortableBundle = Boolean(portableBundle && portableBundleDownloadable(portableBundle));
  const localDemoAvailable = localChampionDemoRuntimeAvailable(data);

  return (
    <div className="export-demo-panel">
      <div className="export-demo-grid">
        <div className="export-block">
          <strong>Package Status</strong>
          <div className="export-status-line">
            <Badge value={data.exportStatus || "PENDING"} />
            <small>{data.exports.length > 0 ? `${data.exports.length} export record(s)` : "No export records exposed yet."}</small>
            <button className="command compact" type="button" onClick={onRequestExport}>
              <HardDriveUpload size={15} />
              Prepare portable model
            </button>
          </div>
          <DetailLoadStatusNotice status={championExportsStatus} />
          {data.exports.length > 0 ? (
            <div className="export-record-list">
              {data.exports.map((exportRecord, index) => {
                const localSafety = championLocalInferenceSafety(exportRecord, {
                  deploymentProfile: data.deploymentProfile,
                  modelProfile: data.modelProfile,
                });
                return (
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
                    {String(exportRecord.format || "").toLowerCase() === "onnx" && (
                      <p>{localSafety.safe ? "Browser parity self-test passed." : localSafety.message}</p>
                    )}
                    {(exportRecord.error || exportRecord.error_message || (exportRecord.validation_errors ?? []).length > 0) && (
                      <p>{exportRecord.error || exportRecord.error_message || exportRecord.validation_errors?.join("; ")}</p>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty compact">No export request has been recorded for this champion yet.</div>
          )}
          {portableBundle && (
            <div className={`export-record portable-bundle-record ${statusToneClass(portableBundle.status)}`}>
              <span>
                <strong>Portable bundle</strong>
                <small>{portableBundle.artifact_uri || portableBundle.uri || portableBundle.artifact_path || portableBundle.path || portableBundle.error || "bundle metadata pending"}</small>
              </span>
              <span>
                <Badge value={portableBundle.status || "PENDING"} />
                <small>{portableBundleDetail}</small>
                {canSavePortableBundle && (
                  <button
                    className="command compact"
                    type="button"
                    onClick={() => onSavePortableBundle(portableBundle)}
                    disabled={savingPortableBundle}
                  >
                    <Download size={15} />
                    {savingPortableBundle ? "Saving" : "Save ZIP"}
                  </button>
                )}
              </span>
            </div>
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

      <DetailLoadStatusNotice status={data.championDemoPredictionsStatus} />
      <DetailLoadStatusNotice status={data.championFeedbackStatus} />

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
              <Badge value={demoImageCategoryLabel(activeImage)} />
              <strong>{activeImageLabel || activeImage?.image_id || "Select an image"}</strong>
            </span>
            <small>{demoImageDetail(activeImage) || "Held-out image or custom worker-visible URI"}</small>
          </div>
          {activeImageCategoryDetail && (
            <div className={`demo-image-note ${activeImageCategory}`}>
              <span>{activeImageCategoryDetail}</span>
            </div>
          )}
        </div>

        <div className="test-controls">
          <div className="demo-block-head">
            <strong>Try Model</strong>
            <span>
              <Badge value={localInferenceStatus === "ready" || localInferenceStatus === "available" ? "LOCAL_ONNX" : localInferenceStatus === "loading" ? "PREPARING_LOCAL" : localInferenceStatus === "error" ? "LOCAL_ERROR" : "LOCAL_PYTHON"} />
              <button className="command compact" type="button" onClick={onToggleSlideshow} disabled={!localDemoAvailable || data.demoImages.length < 2 || predictionLoading}>
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
                disabled={!selectedImage || !selectedImageRunnable || predictionLoading}
                title={selectedImageRunnable ? selectedImageRunLabel : "Original image unavailable for this held-out image"}
              >
                <Play size={15} />
                {selectedImageRunLabel}
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
            <details className="technical-package-details">
              <summary>Browser inference diagnostic</summary>
              <div className="warning-list">
                <span>{localInferenceError}</span>
              </div>
            </details>
          )}
          {predictionLoading && <div className="empty compact">{readyONNXExport(data.exports) ? "Running local ONNX inference..." : "Running local Python inference..."}</div>}
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
            <div className="empty compact">Run a held-out or custom image to see the model prediction.</div>
          )}
        </div>
      </div>

      <div className="demo-grid">
        <div className="demo-block">
          <div className="demo-block-head">
            <strong>Held-out Examples</strong>
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
                disabled={!selectedImage || !selectedImageRunnable || predictionLoading}
                title={selectedImageRunnable ? selectedImageRunLabel : "Original image unavailable for this held-out image"}
              >
                <Play size={15} />
                {selectedImageRunLabel}
              </button>
            </span>
          </div>
          {data.demoImages.length > 0 ? (
            <div className="demo-gallery-sections">
              <DemoImageSection
                title="Representative"
                description="Held-out examples the champion already classified correctly."
                rows={representativeDemoRows}
                selectedImageIndex={selectedImageIndex}
                onSelectImage={onSelectImage}
                empty="No known-correct held-out demo examples are stored for this run."
              />
              <DemoImageSection
                title="Challenge"
                description="Known held-out misses that can guide another fine-tuning round."
                rows={challengeDemoRows}
                selectedImageIndex={selectedImageIndex}
                onSelectImage={onSelectImage}
                empty="No known challenge examples are stored for this run."
              />
              {heldoutDemoRows.length > 0 && (
                <DemoImageSection
                  title="Other held-out"
                  description="Held-out examples without training-time correctness metadata."
                  rows={heldoutDemoRows}
                  selectedImageIndex={selectedImageIndex}
                  onSelectImage={onSelectImage}
                  empty=""
                />
              )}
            </div>
          ) : (
            <div className="empty compact">No held-out/test demo images are exposed by the backend yet.</div>
          )}
          {selectedImage && !selectedImageRunnable && (
            <div className="mission-blocker">
              <AlertTriangle size={15} />
              <span>Original image unavailable. Choose another held-out image or use your own image.</span>
            </div>
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

function DemoImageSection({
  title,
  description,
  rows,
  selectedImageIndex,
  onSelectImage,
  empty,
}: {
  title: string;
  description: string;
  rows: Array<{ image: ChampionDemoImage; index: number }>;
  selectedImageIndex: number;
  onSelectImage: (index: number) => void;
  empty: string;
}) {
  return (
    <div className={`demo-image-section ${rows.length === 0 ? "empty-section" : ""}`}>
      <div className="demo-image-section-head">
        <span>
          <strong>{title}</strong>
          <small>{description}</small>
        </span>
        <Badge value={String(rows.length)} />
      </div>
      {rows.length > 0 ? (
        <div className="demo-image-list">
          {rows.slice(0, 8).map(({ image, index }) => {
            const previewURI = demoImagePreviewURI(image);
            const category = demoImageCategory(image);
            const trainingPrediction = demoImageTrainingPredictionText(image);
            return (
              <button
                className={`demo-image-row demo-image-${category} ${index === selectedImageIndex ? "selected" : ""}`}
                key={image.id || image.image_id || `${image.uri}-${index}`}
                type="button"
                onClick={() => onSelectImage(index)}
              >
                {previewURI ? (
                  <img src={previewURI} alt={demoImageLabel(image) || "demo image"} />
                ) : (
                  <div className="demo-image-placeholder">image</div>
                )}
                <span>
                  <strong>{demoImageLabel(image) || image.image_id || "unlabeled"}</strong>
                  <small>{trainingPrediction || demoImageDetail(image) || "image metadata pending"}</small>
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        empty && <div className="empty compact">{empty}</div>
      )}
    </div>
  );
}

export function DetectionOverlay({ detections }: { detections: ChampionDetection[] }) {
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

function localChampionDemoRuntimeAvailable(data: ChampionExportDemo) {
  return data.exports.some((exportRecord) => normalizedStatus(exportRecord.status || "") === "READY");
}
function demoRuntimeLabel(status: string) {
  if (status === "ready" || status === "available") return "Browser inference ready";
  if (status === "loading") return "Preparing local inference";
  if (status === "error") return "Local inference unavailable";
  if (status === "python_available") return "Local Python inference ready";
  return "Local Python inference";
}

export function TopKConfidenceBars({
  topK,
}: {
  topK: NonNullable<ChampionDemoPrediction["top_k"]>;
}) {
  const rows = topK
    .map((item) => ({
      label: item.label || item.class_name || "class",
      score: numericValue(item.confidence ?? item.probability ?? item.score),
    }))
    .filter((item) => item.score > 0)
    .slice(0, 3);
  if (rows.length === 0) return null;
  return (
    <div className="topk-bars">
      {rows.map((item) => (
        <span key={`${item.label}-${item.score}`}>
          <small>{item.label}</small>
          <i aria-hidden="true"><b style={{ width: `${Math.max(3, Math.min(100, item.score * 100))}%` }} /></i>
          <strong>{formatTopKScore(item.score)}</strong>
        </span>
      ))}
    </div>
  );
}

export function PredictionRow({ prediction, index }: { prediction: ChampionDemoPrediction; index: number }) {
  const status = normalizedStatus(prediction.status || (prediction.runtime_unavailable ? "RUNTIME_UNAVAILABLE" : "PENDING"));
  const displayLabel = prediction.predicted_label || predictionStatusMessage(status);
  const confidence = numericValue(prediction.confidence);
  const topK = Array.isArray(prediction.top_k) ? prediction.top_k : [];
  const detections = detectionBoxesFromPrediction(prediction);
  const imageMetadata = { ...recordObject(prediction.image_metadata), ...recordObject(prediction.metadata) };
  const parityStatus =
    recordString(imageMetadata, "parity_status") ||
    recordString(recordObject(imageMetadata.preprocessing_parity), "status") ||
    "";
  const runtime = recordString(imageMetadata, "runtime");
  const sourceKind = recordString(imageMetadata, "image_source_kind") || recordString(imageMetadata, "demo_source_type");
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
        <span>
          <small>Runtime</small>
          <strong>{runtime || "-"}</strong>
        </span>
        <span>
          <small>Parity</small>
          <strong>{parityStatus || sourceKind || "-"}</strong>
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

export function RunEvaluationDetails({ evaluation }: { evaluation: TrainingRunEvaluation }) {
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

export function MetricCard({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
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

export function Badge({ value }: { value: string }) {
  return <span className={`badge ${value.toLowerCase().replace(/[^a-z0-9_-]+/g, "_")}`}>{value}</span>;
}

export function MetricChart({ metrics, metricKey, label }: { metrics: EpochMetric[]; metricKey: MetricKey; label: string }) {
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

