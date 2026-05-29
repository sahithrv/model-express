import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Box,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Database,
  DollarSign,
  Eye,
  FolderOpen,
  HardDriveUpload,
  ListRestart,
  MonitorDot,
  Play,
  Plus,
  RefreshCcw,
  Server,
  Shuffle,
  SlidersHorizontal,
  SquareTerminal,
  StepForward,
  Timer,
  Trophy,
  X,
} from "lucide-react";
import type {
  AgentDecision,
  AgentInvocation,
  AgentMemoryRecord,
  ChampionDemoImage,
  ChampionDemoPrediction,
  ChampionExport,
  Dataset,
  DatasetVisualAnalysis,
  EpochMetric,
  ExecutionEvent,
  ExperimentPlan,
  Health,
  Job,
  PlannedExperiment,
  Project,
  ProjectChampion,
  StrategyScorecard,
  TrainingRunEvaluation,
  TrainingRunSummary,
  VisualAnalysisRerunPolicy,
  Worker,
  AutomationSettings,
  WorkerRequirement,
} from "./types";

const defaultBaseUrl = localStorage.getItem("orchestratorUrl") ?? "http://localhost:8080";
const jobsPerPage = 10;

type MetricKey = "macro_f1" | "accuracy" | "train_loss" | "val_loss";
type ProjectTabKey = "overview" | "data" | "experiments" | "agents" | "operations" | "export";

const metricOptions: Array<{ key: MetricKey; label: string }> = [
	{ key: "macro_f1", label: "macro_f1" },
	{ key: "accuracy", label: "Accuracy" },
	{ key: "train_loss", label: "Train loss" },
	{ key: "val_loss", label: "Val loss" },
];

const projectTabs: Array<{ key: ProjectTabKey; label: string }> = [
	{ key: "overview", label: "Overview" },
	{ key: "data", label: "Data" },
	{ key: "experiments", label: "Experiments" },
	{ key: "agents", label: "Agents" },
	{ key: "operations", label: "Operations" },
	{ key: "export", label: "Export" },
];

type ProjectDetail = {
  decisions: AgentDecision[];
  datasets: Dataset[];
  visualAnalysis: VisualAnalysisDetail;
  jobs: Job[];
  plans: ExperimentPlan[];
  runSummaries: TrainingRunSummary[];
  runEvaluations: TrainingRunEvaluation[];
  champion: ProjectChampion | null;
  championExports: ChampionExport[];
  championDemoImages: ChampionDemoImage[];
  championDemoPredictions: ChampionDemoPrediction[];
  workers: Worker[];
  workerRequirements: WorkerRequirement[];
  executionEvents: ExecutionEvent[];
  agentInvocations: AgentInvocation[];
  agentMemory: AgentMemoryRecord[];
  strategyScorecards: StrategyScorecard[];
};

type VisualAnalysisDetail = {
  analysis: DatasetVisualAnalysis | null;
  status: "available" | "empty" | "unsupported" | "error";
  message: string;
  manualRunSupported: boolean;
  rerunPolicy?: VisualAnalysisRerunPolicy | null;
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

type ReasoningSection = {
  title: string;
  items: string[];
};

type ChampionComparisonRow = {
  jobId: string;
  model: string;
  rankScore: number;
  accuracy: number;
  macroF1: number;
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
  components: Array<{ label: string; value: number | string }>;
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
  rejections: Array<{ kind: string; text: string }>;
  mechanismCoverage: MechanismCoverageRow[];
  candidateScores: CandidateScoreRow[];
};

type MechanismCoverageRow = {
  mechanism: string;
  status: string;
  detail: string;
};

type ChampionExportDemo = {
  hasChampion: boolean;
  exportStatus: string;
  exports: ChampionExport[];
  projectId: string;
  modelCard: Record<string, unknown>;
  useCases: string[];
  limitations: string[];
  preprocessing: string[];
  demoImages: ChampionDemoImage[];
  demoPredictions: ChampionDemoPrediction[];
};

type Notice = {
  kind: "info" | "error";
  text: string;
};

type DatasetFolder = {
  path: string;
  name: string;
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
    visualAnalysis: {
      analysis: null,
      status: "empty",
      message,
      manualRunSupported: false,
    },
    jobs: [],
    plans: [],
    runSummaries: [],
    runEvaluations: [],
    champion: null,
    championExports: [],
    championDemoImages: [],
    championDemoPredictions: [],
    workers: [],
    workerRequirements: [],
    executionEvents: [],
    agentInvocations: [],
    agentMemory: [],
    strategyScorecards: [],
  };
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
	const [activeProjectTab, setActiveProjectTab] = useState<ProjectTabKey>("overview");
	const [demoPrediction, setDemoPrediction] = useState<ChampionDemoPrediction | null>(null);
  const [demoPredictionError, setDemoPredictionError] = useState("");
  const [demoPredictionLoading, setDemoPredictionLoading] = useState(false);
  const [selectedDemoImageIndex, setSelectedDemoImageIndex] = useState(0);
  const supervisingRequirements = useRef<Set<string>>(new Set());
  const eventRefreshInFlight = useRef(false);

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

  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const latestDecisionHasFollowUpPlan = latestDecision
    ? detail.plans.some((plan) => plan.source_decision_id === latestDecision.id)
    : false;
  const decisionChatTurns = useMemo(() => buildDecisionChatTurns(detail.decisions), [detail.decisions]);
  const runTotals = useMemo(() => summarizeTrainingRuns(detail.runSummaries), [detail.runSummaries]);
  const timelineItems = useMemo(
    () => buildExperimentTimeline(selectedProject, detail),
    [detail, selectedProject],
  );
  const datasetIntelligence = useMemo(
    () => buildDatasetIntelligence(detail.datasets[0] ?? null, latestDecision),
    [detail.datasets, latestDecision],
  );
  const championComparison = useMemo(
    () => buildChampionComparison(detail.runSummaries, detail.runEvaluations, detail.jobs, detail.champion),
    [detail.champion, detail.jobs, detail.runEvaluations, detail.runSummaries],
  );
  const championExportDemo = useMemo(() => buildChampionExportDemo(detail), [detail]);
  const reviewState = automationReviewState(automationSettings);

  const firstDatasetId = detail.datasets[0]?.id ?? "";
  const jobPageCount = Math.max(1, Math.ceil(detail.jobs.length / jobsPerPage));
  const visibleJobs = detail.jobs.slice(jobPage * jobsPerPage, jobPage * jobsPerPage + jobsPerPage);

  const request = useCallback(
    async <T,>(path: string, options: { method?: string; body?: unknown } = {}) => {
      return window.missionControl.request<T>({
        baseUrl,
        path,
        method: options.method,
        body: options.body,
      });
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
    async (dataset: Dataset | null): Promise<VisualAnalysisDetail> => {
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
        const response = await request<VisualAnalysisListResponse>(`/datasets/${dataset.id}/visual-analyses`);
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

  const refreshProjectDetail = useCallback(
    async (projectId: string) => {
      if (!projectId) {
        setDetail(emptyProjectDetail());
        return;
      }

      const [
        datasets,
        jobs,
        plans,
        runSummaries,
        runEvaluations,
        champion,
        decisions,
        workers,
        workerRequirements,
        executionEvents,
        agentInvocations,
        agentMemory,
        strategyScorecards,
      ] =
        await Promise.all([
        request<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`),
        request<{ jobs: Job[] }>(`/projects/${projectId}/jobs`),
        request<{ plans: ExperimentPlan[] }>(`/projects/${projectId}/plans`),
        request<{ summaries: TrainingRunSummary[] }>(`/projects/${projectId}/training-run-summaries`),
        request<{ evaluations: TrainingRunEvaluation[] }>(`/projects/${projectId}/training-run-evaluations`),
        request<{ champion: ProjectChampion | null }>(`/projects/${projectId}/champion`),
        request<{ decisions: AgentDecision[] }>(`/projects/${projectId}/agent-decisions`),
        request<{ workers: Worker[] }>(`/projects/${projectId}/workers`),
        request<{ requirements: WorkerRequirement[] }>(`/projects/${projectId}/worker-requirements`),
        request<{ events: ExecutionEvent[] }>(`/projects/${projectId}/execution-events?limit=8`),
        request<AgentInvocationsResponse>(`/projects/${projectId}/agent-invocations?limit=8`).catch(
          (): AgentInvocationsResponse => ({ invocations: [] }),
        ),
        request<{ records: AgentMemoryRecord[] }>(`/projects/${projectId}/agent-memory?limit=6`),
        request<{ scorecards: StrategyScorecard[] }>(`/projects/${projectId}/strategy-scorecards?limit=6`),
      ]);

      const visualAnalysis = await fetchLatestDatasetVisualAnalysis(datasets.datasets[0] ?? null);

      const championValue = champion.champion;
      const [championExports, championDemoImages, championDemoPredictions] = championValue
        ? await Promise.all([
            request<{ exports: ChampionExport[] }>(`/projects/${projectId}/champion/exports`).catch(() => ({ exports: [] })),
            request<{ images: ChampionDemoImage[] }>(
              `/projects/${projectId}/champion/demo-images?max_total_images=12&max_per_class=2`,
            ).catch(() => ({ images: [] })),
            request<{ predictions?: ChampionDemoPrediction[]; history?: ChampionDemoPrediction[]; demo_predictions?: ChampionDemoPrediction[] }>(
              `/projects/${projectId}/champion/demo-predictions?limit=8`,
            ).catch((): { predictions?: ChampionDemoPrediction[]; history?: ChampionDemoPrediction[]; demo_predictions?: ChampionDemoPrediction[] } => ({
              predictions: [],
            })),
          ])
        : [{ exports: [] }, { images: [] }, { predictions: [] }];

      setDetail({
        decisions: decisions.decisions,
        datasets: datasets.datasets,
        visualAnalysis,
        jobs: jobs.jobs,
        plans: plans.plans,
        runSummaries: runSummaries.summaries,
        runEvaluations: runEvaluations.evaluations,
        champion: championValue,
        championExports: championExports.exports,
        championDemoImages: championDemoImages.images,
        championDemoPredictions:
          championDemoPredictions.predictions ??
          championDemoPredictions.history ??
          championDemoPredictions.demo_predictions ??
          [],
        workers: workers.workers,
        workerRequirements: workerRequirements.requirements,
        executionEvents: executionEvents.events,
        agentInvocations: agentInvocationsFromResponse(agentInvocations),
        agentMemory: agentMemory.records,
        strategyScorecards: strategyScorecards.scorecards,
      });

      setSelectedJobId((currentJobId) => {
        if (jobs.jobs.length === 0) return "";
        if (jobs.jobs.some((job) => job.id === currentJobId)) return currentJobId;
        return jobs.jobs[0].id;
      });
    },
    [fetchLatestDatasetVisualAnalysis, request],
  );

  const refreshSelectedJobMetrics = useCallback(async () => {
    if (!selectedJobId) {
      setMetrics([]);
      return;
    }

    const response = await request<{ metrics: EpochMetric[] }>(`/jobs/${selectedJobId}/metrics`);
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
        await refreshProjectDetail(selectedProjectId);
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

  const refreshLive = useCallback(async () => {
    try {
      await refreshHealth();
      await refreshProjects();
      if (selectedProjectId) {
        await refreshProjectDetail(selectedProjectId);
      }
      await refreshSelectedJobMetrics();
    } catch {
      setHealth(null);
    }
  }, [refreshHealth, refreshProjectDetail, refreshProjects, refreshSelectedJobMetrics, selectedProjectId]);

  const superviseWorkerRequirements = useCallback(async () => {
    if (!selectedProjectId) return;

    const response = await request<{ requirements: WorkerRequirement[] }>(
      `/projects/${selectedProjectId}/worker-requirements`,
    );
    const pending = response.requirements.filter((requirement) =>
      requirement.status === "PENDING" || requirement.status === "STARTING",
    );

    for (const requirement of pending) {
      if (supervisingRequirements.current.has(requirement.id)) {
        continue;
      }
      supervisingRequirements.current.add(requirement.id);
      try {
        await request<WorkerRequirement>(`/worker-requirements/${requirement.id}`, {
          method: "PATCH",
          body: { status: "STARTING", last_error: "" },
        });

        await window.missionControl.ensureProjectWorker({
          projectId: requirement.project_id,
          baseUrl,
          name: `auto-worker-${requirement.project_id}`,
          gpuType: requirement.gpu_type || requirement.provider || "local",
          count: requirement.target_count,
        });

        await request<WorkerRequirement>(`/worker-requirements/${requirement.id}`, {
          method: "PATCH",
          body: { status: "ACTIVE", last_error: "" },
        });
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
        refreshProjectDetail(selectedProjectId).catch(() => undefined);
      }
    }
  }, [baseUrl, refreshProjectDetail, request, selectedProjectId]);

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
      setJobPage(0);
      refreshProjectDetail(selectedProjectId).catch((error) =>
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
    refreshSelectedJobMetrics().catch((error) =>
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
    );
  }, [refreshSelectedJobMetrics]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshLive();
    }, 2500);

    return () => window.clearInterval(timer);
  }, [refreshLive]);

  useEffect(() => {
    if (!selectedProjectId || typeof EventSource === "undefined") return;

    let closed = false;
    const streamUrl = new URL(`/projects/${selectedProjectId}/events/stream`, baseUrl);
    const events = new EventSource(streamUrl.toString());
    const triggerRefresh = () => {
      if (closed || eventRefreshInFlight.current) return;
      eventRefreshInFlight.current = true;
      window.setTimeout(() => {
        refreshLive()
          .catch(() => undefined)
          .finally(() => {
            eventRefreshInFlight.current = false;
          });
      }, 150);
    };

    events.onmessage = triggerRefresh;
    events.addEventListener("execution_event", triggerRefresh);
    events.addEventListener("project_event", triggerRefresh);
    events.onerror = () => {
      closed = true;
      events.close();
    };

    return () => {
      closed = true;
      events.close();
    };
  }, [baseUrl, refreshLive, selectedProjectId]);

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
      setNewProjectFolder(folder);
    }
  }

  async function createProjectWithDataset(formData: FormData) {
    const name = String(formData.get("name") ?? "").trim();
    const goal = String(formData.get("goal") ?? "").trim();

    if (!name || !newProjectFolder) {
      setNotice({ kind: "error", text: "Project name and dataset folder are required." });
      return;
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
        datasetPath: newProjectFolder.path,
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
      await refreshProjectDetail(project.id);
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
    await refreshProjectDetail(selectedProjectId);
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
      await refreshProjectDetail(selectedProjectId);
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
      const workerPool = await window.missionControl.ensureProjectWorker({
        projectId: selectedProjectId,
        baseUrl,
        name: `modal-worker-${selectedProjectId}`,
        gpuType: "modal",
        count: workerCount,
      });

      const response = await request<{ jobs: Job[] }>(`/plans/${planId}/execute`, {
        method: "POST",
        body: { provider: "modal", gpu_type: "T4" },
      });

      await refreshProjectDetail(selectedProjectId);
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

  async function reviewExperiments() {
    if (!selectedProjectId) return;

    setLoading(true);
    setNotice(null);
    try {
      const decision = await request<AgentDecision>(`/projects/${selectedProjectId}/review-experiments`, {
        method: "POST",
        body: {},
      });

      await refreshProjectDetail(selectedProjectId);
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
        await refreshProjectDetail(selectedProjectId);
        setNotice({ kind: "info", text: `No follow-up scheduled. Reviewer decision: ${response.decision.decision_type}` });
        return;
      }

      const plan = response.follow_up_plan;
      const workerCount = Math.max(1, Math.min(plan.recommended_workers, plan.experiments.length || 1));
      const workerPool = await window.missionControl.ensureProjectWorker({
        projectId: selectedProjectId,
        baseUrl,
        name: `modal-worker-${selectedProjectId}`,
        gpuType: "modal",
        count: workerCount,
      });

      const execution = await request<{ jobs: Job[] }>(`/plans/${plan.id}/execute`, {
        method: "POST",
        body: { provider: "modal", gpu_type: "T4" },
      });

      await refreshProjectDetail(selectedProjectId);
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
      await refreshProjectDetail(selectedProjectId);
      setNotice({ kind: "info", text: `Champion export recorded as ${exportRecord.status || "PENDING"}.` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function runChampionDemoPrediction(image: ChampionDemoImage) {
    if (!selectedProjectId || !detail.champion) return;

    const imageURI = image.uri || image.image_uri || image.thumbnail_uri || "";
    if (!imageURI) {
      setDemoPrediction(null);
      setDemoPredictionError("Demo image has no URI to send to the backend.");
      return;
    }

    setDemoPrediction(null);
    setDemoPredictionError("");
    setDemoPredictionLoading(true);
    try {
      const prediction = await request<ChampionDemoPrediction>(`/projects/${selectedProjectId}/champion/demo-predictions`, {
        method: "POST",
        body: { image_uri: imageURI, top_k: 5 },
      });
      setDemoPrediction(normalizeDemoPredictionResponse(prediction));
    } catch (error) {
      setDemoPredictionError(error instanceof Error ? error.message : String(error));
    } finally {
      setDemoPredictionLoading(false);
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
            <small>Agentic vision training control plane</small>
          </span>
        </div>
        <div className="chrome-right">
          <span>{health?.status === "ok" ? "Orchestrator online" : "Orchestrator offline"}</span>
        </div>
      </header>

      <aside className="sidebar">
        <label className="field compact">
          <span>Orchestrator</span>
          <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
        </label>

        <div className="sidebar-actions">
          <button className="command primary" onClick={() => setNewProjectOpen(true)} disabled={loading}>
            <Plus size={16} />
            New Project
          </button>
          <button className="icon-command" onClick={refreshAll} disabled={loading} title="Refresh now">
            <RefreshCcw size={16} />
          </button>
        </div>

        <section className="nav-section">
          <div className="section-title">
            <Database size={15} />
            Projects
          </div>
          <div className="project-list">
            {projects.map((project) => (
              <button
                key={project.id}
                className={project.id === selectedProjectId ? "project active" : "project"}
                onClick={() => setSelectedProjectId(project.id)}
              >
                <span>{project.name}</span>
                <small>{project.id}</small>
              </button>
            ))}
          </div>
        </section>
      </aside>

			<section className="workspace" data-active-tab={activeProjectTab}>
        <header className="topbar">
          <div>
            <div className="eyebrow">Control Plane</div>
            <h2>{selectedProject ? selectedProject.name : "No Project Selected"}</h2>
          </div>
          <div className={health?.status === "ok" ? "status ok" : "status bad"}>
            <Server size={16} />
            {health?.status ?? "offline"}
          </div>
        </header>

        {notice && <div className={`notice ${notice.kind}`}>{notice.text}</div>}

				<nav className="section-tabs" aria-label="Project detail tabs" role="tablist">
					{projectTabs.map((tab) => (
						<button
							key={tab.key}
							type="button"
							role="tab"
							className={activeProjectTab === tab.key ? "active" : ""}
							aria-selected={activeProjectTab === tab.key}
							onClick={() => setActiveProjectTab(tab.key)}
						>
							{tab.label}
						</button>
					))}
				</nav>

				<section className="summary-grid" id="overview" data-project-tab="overview">
          <MetricCard icon={<Box size={18} />} label="Datasets" value={detail.datasets.length} />
          <MetricCard icon={<Activity size={18} />} label="Jobs" value={detail.jobs.length} />
          <MetricCard icon={<ClipboardList size={18} />} label="Plans" value={detail.plans.length} />
          <MetricCard icon={<MonitorDot size={18} />} label="Workers" value={detail.workers.length} />
          <MetricCard
            icon={<ListRestart size={18} />}
            label="Queued"
            value={detail.jobs.filter((job) => job.status === "QUEUED").length}
          />
        </section>

				<section className="content-grid mission-grid">
					<Panel title="Experiment Timeline" icon={<ListRestart size={17} />} wide tab="overview">
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

					<Panel title="Dataset Intelligence" icon={<BarChart3 size={17} />} wide id="data" tab="data">
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

					<Panel title="Visual Dataset Analysis" icon={<Eye size={17} />} wide tab="data">
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
					<Panel title="Automation Settings" icon={<SlidersHorizontal size={17} />} wide id="operations" tab="operations">
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
                  <span>Provider</span>
                  <select
                    value={settingsDraft.default_training_provider}
                    onChange={(event) => updateSettingsDraft({ default_training_provider: event.currentTarget.value })}
                  >
                    <option value="local">local</option>
                    <option value="modal">modal</option>
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
						<Panel title="Selected Champion" icon={<Trophy size={17} />} wide tab="overview">
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
                    <small>Macro-F1</small>
                    <strong>{formatMaybeMetric(recordNumber(detail.champion.metrics, "best_macro_f1"))}</strong>
                  </div>
                  <div>
                    <small>Accuracy</small>
                    <strong>{formatMaybeMetric(recordNumber(detail.champion.metrics, "best_accuracy"))}</strong>
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

					<Panel title="Champion Export / Demo" icon={<Trophy size={17} />} wide id="export-demo" tab="export">
            <ChampionExportDemoPanel
              data={championExportDemo}
              prediction={demoPrediction}
              predictionError={demoPredictionError}
              predictionLoading={demoPredictionLoading}
              selectedImageIndex={selectedDemoImageIndex}
              onSelectImage={setSelectedDemoImageIndex}
              onNextImage={() => setSelectedDemoImageIndex((index) => nextDemoImageIndex(index, championExportDemo.demoImages.length))}
              onRandomImage={() => setSelectedDemoImageIndex((index) => randomDemoImageIndex(index, championExportDemo.demoImages.length))}
              onRequestExport={() => requestChampionExport("onnx")}
              onRunPrediction={runChampionDemoPrediction}
            />
          </Panel>

					<Panel title="Training Run Summary" icon={<Trophy size={17} />} wide id="runs" tab="experiments">
            <div className="run-summary">
              <div className="run-overview">
                <div>
                  <span><DollarSign size={15} /> Estimated Spend</span>
                  <strong>{formatCurrency(runTotals.totalCost)}</strong>
                </div>
                <div>
                  <span><Trophy size={15} /> Best macro_f1</span>
                  <strong>{formatMaybeMetric(runTotals.bestMacroF1)}</strong>
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
                    <span>Best F1</span>
                    <span>Cost</span>
                    <span>Runtime</span>
                    <span>Epochs</span>
                  </div>
                  {detail.runSummaries.map((summary) => (
                    <button
                      className={summary.job_id === selectedJobId ? "run-table-row run-row active" : "run-table-row run-row"}
                      key={summary.job_id}
                      onClick={() => setSelectedJobId(summary.job_id)}
                    >
                      <span>
                        <strong>{summary.model || "unknown"}</strong>
                        <small>{summary.job_id}</small>
                      </span>
                      <Badge value={summary.status || "UNKNOWN"} />
                      <strong>{formatMaybeMetric(summary.best_macro_f1)}</strong>
                      <strong>{formatCurrency(summary.estimated_cost_usd)}</strong>
                      <span>{formatSeconds(summary.runtime_seconds)}</span>
                      <span>{summary.epochs_completed}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="empty">Training summaries will appear as soon as experiment jobs report their first epoch.</div>
              )}
            </div>
          </Panel>

					<Panel title="Champion Comparison" icon={<Trophy size={17} />} wide tab="experiments">
            {championComparison.length > 0 ? (
              <div className="comparison-table">
                <div className="comparison-row comparison-head">
                  <span>Model</span>
                  <span>Rank</span>
                  <span>Macro-F1</span>
                  <span>Accuracy</span>
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
                    <strong>{formatMaybeMetric(row.macroF1)}</strong>
                    <span>{formatMaybeMetric(row.accuracy)}</span>
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

					<Panel title="Agent Decisions" icon={<BrainCircuit size={17} />} wide id="agents" tab="agents">
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

					<Panel title="Automation Timeline" icon={<ListRestart size={17} />} wide tab="operations">
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

					<Panel title="Agent Invocation Audit" icon={<SquareTerminal size={17} />} wide tab="agents">
            <AgentInvocationAuditPanel invocations={detail.agentInvocations} decisions={detail.decisions} />
          </Panel>

					<Panel title="Agent Memory" icon={<BrainCircuit size={17} />} wide tab="agents">
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

					<Panel title="Experiment Plan" icon={<ClipboardList size={17} />} wide id="plans" tab="experiments">
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

					<Panel title="Manual Job Queue" icon={<Play size={17} />} wide tab="operations">
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

					<Panel title="Workers" icon={<MonitorDot size={17} />} wide tab="operations">
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

					<Panel title="Datasets" icon={<Database size={17} />} wide tab="data">
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

					<Panel title="Recent Jobs" icon={<SquareTerminal size={17} />} wide tab="experiments">
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

					<Panel title="Run Metrics" icon={<Activity size={17} />} wide tab="experiments">
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
                    {metricOptions.map((metric) => (
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
                      {selectedRunEvaluation && (
                        <span>{formatLatency(recordNumber(selectedRunEvaluation.model_profile, "estimated_latency_ms"))}</span>
                      )}
                    </div>
                  )}
                </div>
                <MetricChart
                  metrics={metrics}
                  metricKey={selectedMetricKey}
                  label={metricOptions.find((metric) => metric.key === selectedMetricKey)?.label ?? selectedMetricKey}
                />
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
              <input name="name" placeholder="Project Name" required />
              <input name="goal" placeholder="Goal / extra context (optional)" />
              <button className="dataset-picker" type="button" onClick={chooseNewProjectFolder} disabled={loading}>
                <FolderOpen size={18} />
                <span>
                  <strong>{newProjectFolder ? newProjectFolder.name : "Choose Folder & Upload"}</strong>
                  <small>{newProjectFolder ? newProjectFolder.path : "Required image dataset folder"}</small>
                </span>
                {newProjectFolder && <CheckCircle2 size={18} />}
              </button>
              <button className="command primary" disabled={!newProjectFolder || loading}>
                <HardDriveUpload size={16} />
                Create Project
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
  onSelectImage,
  onNextImage,
  onRandomImage,
  onRequestExport,
  onRunPrediction,
}: {
  data: ChampionExportDemo;
  prediction: ChampionDemoPrediction | null;
  predictionError: string;
  predictionLoading: boolean;
  selectedImageIndex: number;
  onSelectImage: (index: number) => void;
  onNextImage: () => void;
  onRandomImage: () => void;
  onRequestExport: () => void;
  onRunPrediction: (image: ChampionDemoImage) => void;
}) {
  if (!data.hasChampion) {
    return <div className="empty">Champion export and demo details will appear after the backend selects a champion.</div>;
  }

  const selectedImage = data.demoImages[selectedImageIndex] ?? data.demoImages[0] ?? null;
  const predictionRows = prediction ? [prediction] : data.demoPredictions;

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

      <div className="demo-grid">
        <div className="demo-block">
          <div className="demo-block-head">
            <strong>Demo Images</strong>
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
                  {image.thumbnail_uri || image.uri || image.image_uri ? (
                    <img src={image.thumbnail_uri || image.uri || image.image_uri} alt={image.label || image.true_label || image.class_name || "demo image"} />
                  ) : (
                    <div className="demo-image-placeholder">image</div>
                  )}
                  <span>
                    <strong>{image.true_label || image.label || image.class_name || image.image_id || "unlabeled"}</strong>
                    <small>{[image.split, image.size_bytes ? formatBytes(image.size_bytes) : "", image.uri || image.image_uri].filter(Boolean).join(" - ") || "image metadata pending"}</small>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty compact">No held-out/test demo images are exposed by the backend yet.</div>
          )}
        </div>

        <div className="demo-block">
          <strong>{prediction ? "Latest Prediction" : "Prediction History"}</strong>
          {predictionError && (
            <div className="warning-list">
              <span>{predictionError}</span>
            </div>
          )}
          {predictionLoading && <div className="empty compact">Waiting for backend demo inference...</div>}
          {predictionRows.length > 0 ? (
            <div className="prediction-list">
              {predictionRows.slice(0, 6).map((predictionRow, index) => (
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
        </div>
      </div>
    </div>
  );
}

function PredictionRow({ prediction, index }: { prediction: ChampionDemoPrediction; index: number }) {
  const status = normalizedStatus(prediction.status || (prediction.runtime_unavailable ? "RUNTIME_UNAVAILABLE" : "PENDING"));
  const displayLabel = prediction.predicted_label || predictionStatusMessage(status);
  const confidence = numericValue(prediction.confidence);
  const topK = Array.isArray(prediction.top_k) ? prediction.top_k : [];
  const imageSrc = prediction.image_uri || recordString(recordObject(prediction.metadata), "thumbnail_uri");
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

  const values = metrics.map((metric) => metric.metrics[metricKey] ?? 0);
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const range = Math.max(maxValue - minValue, 0.001);

  const width = 760;
  const height = 240;
  const padding = 28;
  const points = values.map((value, index) => {
    const x =
      metrics.length === 1
        ? width / 2
        : padding + (index / (metrics.length - 1)) * (width - padding * 2);
    const y = height - padding - ((value - minValue) / range) * (height - padding * 2);
    return { x, y, value, epoch: metrics[index].epoch };
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

function summarizeTrainingRuns(summaries: TrainingRunSummary[]) {
  const best = summaries.reduce<TrainingRunSummary | null>((currentBest, summary) => {
    if (!currentBest) return summary;
    return summary.best_macro_f1 > currentBest.best_macro_f1 ? summary : currentBest;
  }, null);

  return {
    totalCost: summaries.reduce((total, summary) => total + summary.estimated_cost_usd, 0),
    totalRuntimeSeconds: summaries.reduce((total, summary) => total + summary.runtime_seconds, 0),
    bestMacroF1: best?.best_macro_f1 ?? 0,
    activeRuns: summaries.filter((summary) => ["RUNNING", "ASSIGNED", "QUEUED"].includes(summary.status)).length,
  };
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

function buildDatasetIntelligence(dataset: Dataset | null, latestDecision: AgentDecision | null) {
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
  const metadataSummary = recordObject(profile.metadata_summary);
  const metadataAvailable = profile.metadata_available === true || Object.keys(metadataSummary).length > 0;
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
    metadataAvailable ? "metadata detected" : "metadata not reported",
    corruptCount > 0 ? `${corruptCount} corrupt image(s)` : "no corrupt images reported",
    exemplarStatus,
  ].filter(Boolean);
  const warnings = [
    ...(imbalanceRatio >= 1.5 ? [`Class imbalance ratio ${imbalanceRatio.toFixed(2)}; prefer macro-F1 and per-class checks.`] : []),
    ...(corruptCount > 0 ? [`${corruptCount} corrupt image(s) detected by profiler.`] : []),
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

function buildChampionComparison(
  summaries: TrainingRunSummary[],
  evaluations: TrainingRunEvaluation[],
  jobs: Job[],
  champion: ProjectChampion | null,
): ChampionComparisonRow[] {
  const evaluationByJob = new Map(evaluations.map((evaluation) => [evaluation.job_id, evaluation]));
  const jobById = new Map(jobs.map((job) => [job.id, job]));
  const seedVarianceBySignature = buildSeedVarianceBySignature(summaries, jobById);
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
      const rankScore = experimentRankScore({
        macroF1: summary.best_macro_f1,
        accuracy: summary.best_accuracy,
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
        accuracy: summary.best_accuracy,
        macroF1: summary.best_macro_f1,
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

function buildSeedVarianceBySignature(summaries: TrainingRunSummary[], jobById: Map<string, Job>) {
  const groups = new Map<string, Array<{ seed: string; score: number }>>();
  for (const summary of summaries) {
    const job = jobById.get(summary.job_id);
    if (!job || normalizedStatus(summary.status) !== "SUCCEEDED") continue;
    const signature = experimentComparisonSignature(job.config);
    if (!signature) continue;
    const seed = experimentSeed(job.config);
    if (!seed) continue;
    const rows = groups.get(signature) ?? [];
    rows.push({ seed, score: summary.best_macro_f1 });
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
  macroF1: number;
  accuracy: number;
  costUsd: number;
  runtimeSeconds: number;
  latencyMs: number;
  objectiveFit: number;
  trainValidationGap: number | null;
  divergenceStatus: string;
  seedVariance: number | null;
}) {
  const quality = input.macroF1 * 0.65 + input.accuracy * 0.35;
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

function candidateScoreRows(decision: AgentDecision): CandidateScoreRow[] {
  const rankings = Array.isArray(decision.payload?.candidate_rankings) ? decision.payload.candidate_rankings : [];
  return rankings
    .map((item, index) => {
      const record = recordObject(item);
      const components = recordObject(record.score_components);
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
        components: Object.entries(components)
          .map(([label, value]) => ({ label, value: typeof value === "number" ? value : String(value) }))
          .filter((component) => component.value !== "")
          .slice(0, 6),
      };
    })
    .filter(
      (row) =>
        row.totalScore !== null ||
        row.components.length > 0 ||
        row.reasons.length > 0 ||
        Boolean(row.mechanism || row.intervention || row.expectedEffect || row.validationStatus),
    )
    .slice(0, 6);
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
      useCases: ["Select a champion first"],
      limitations: [],
      preprocessing: [],
      demoImages: [],
      demoPredictions: [],
    };
  }

  const deployment = champion.deployment_profile ?? {};
  const evaluation = champion.evaluation ?? {};
  const modelCard = {
    ...recordObject(deployment.model_card),
    ...recordObject(evaluation.model_card),
  };
  const exports = [
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
  ];
  const demoPredictions = [
    ...detail.championDemoPredictions,
    ...demoPredictionsFromUnknown(champion.demo_predictions),
    ...demoPredictionsFromUnknown(deployment.demo_predictions),
    ...demoPredictionsFromUnknown(deployment.predictions),
  ];
  const modelProfile = {
    ...recordObject(deployment.model_profile),
    ...recordObject(evaluation.model_profile),
  };
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
    useCases,
    limitations,
    preprocessing: preprocessing.length > 0 ? preprocessing : ["Use the preprocessing from the winning experiment config."],
    demoImages,
    demoPredictions,
  };
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

function demoImagesFromUnknown(value: unknown): ChampionDemoImage[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoImage).filter((item) => Object.keys(item).length > 0);
}

function demoPredictionsFromUnknown(value: unknown): ChampionDemoPrediction[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => recordObject(item) as ChampionDemoPrediction).filter((item) => Object.keys(item).length > 0);
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
