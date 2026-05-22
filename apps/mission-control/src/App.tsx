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
  AgentMemoryRecord,
  ChampionDemoImage,
  ChampionDemoPrediction,
  ChampionExport,
  Dataset,
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
  Worker,
  AutomationSettings,
  WorkerRequirement,
} from "./types";

const defaultBaseUrl = localStorage.getItem("orchestratorUrl") ?? "http://localhost:8080";
const jobsPerPage = 10;

type MetricKey = "macro_f1" | "accuracy" | "train_loss" | "val_loss";

const metricOptions: Array<{ key: MetricKey; label: string }> = [
  { key: "macro_f1", label: "macro_f1" },
  { key: "accuracy", label: "Accuracy" },
  { key: "train_loss", label: "Train loss" },
  { key: "val_loss", label: "Val loss" },
];

type ProjectDetail = {
  decisions: AgentDecision[];
  datasets: Dataset[];
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
  agentMemory: AgentMemoryRecord[];
  strategyScorecards: StrategyScorecard[];
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
  accuracy: number;
  macroF1: number;
  runtimeSeconds: number;
  costUsd: number;
  latencyMs: number;
  modelSizeMb: number;
  objectiveFit: number;
  isChampion: boolean;
};

type CandidateScoreRow = {
  label: string;
  status: string;
  totalScore: number | null;
  reasons: string[];
  components: Array<{ label: string; value: number | string }>;
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
  updated_at: "",
};

export function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [health, setHealth] = useState<Health | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [detail, setDetail] = useState<ProjectDetail>({
    decisions: [],
    datasets: [],
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
    agentMemory: [],
    strategyScorecards: [],
  });
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
  const [demoPrediction, setDemoPrediction] = useState<ChampionDemoPrediction | null>(null);
  const [demoPredictionError, setDemoPredictionError] = useState("");
  const [demoPredictionLoading, setDemoPredictionLoading] = useState(false);
  const [selectedDemoImageIndex, setSelectedDemoImageIndex] = useState(0);
  const supervisingRequirements = useRef<Set<string>>(new Set());

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
  const runTotals = useMemo(() => summarizeTrainingRuns(detail.runSummaries), [detail.runSummaries]);
  const timelineItems = useMemo(
    () => buildExperimentTimeline(selectedProject, detail),
    [detail, selectedProject],
  );
  const datasetIntelligence = useMemo(
    () => buildDatasetIntelligence(detail.datasets[0] ?? null, latestDecision),
    [detail.datasets, latestDecision],
  );
  const reasoningSections = useMemo(
    () => (latestDecision ? decisionReasoningSections(latestDecision) : []),
    [latestDecision],
  );
  const rejectionItems = useMemo(
    () => (latestDecision ? decisionRejections(latestDecision) : []),
    [latestDecision],
  );
  const championComparison = useMemo(
    () => buildChampionComparison(detail.runSummaries, detail.runEvaluations, detail.champion),
    [detail.champion, detail.runEvaluations, detail.runSummaries],
  );
  const candidateScores = useMemo(
    () => (latestDecision ? candidateScoreRows(latestDecision) : []),
    [latestDecision],
  );
  const championExportDemo = useMemo(() => buildChampionExportDemo(detail), [detail]);

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

  const refreshProjectDetail = useCallback(
    async (projectId: string) => {
      if (!projectId) {
        setDetail({
          decisions: [],
          datasets: [],
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
          agentMemory: [],
          strategyScorecards: [],
        });
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
        request<{ records: AgentMemoryRecord[] }>(`/projects/${projectId}/agent-memory?limit=6`),
        request<{ scorecards: StrategyScorecard[] }>(`/projects/${projectId}/strategy-scorecards?limit=6`),
      ]);

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
        agentMemory: agentMemory.records,
        strategyScorecards: strategyScorecards.scorecards,
      });

      setSelectedJobId((currentJobId) => {
        if (jobs.jobs.length === 0) return "";
        if (jobs.jobs.some((job) => job.id === currentJobId)) return currentJobId;
        return jobs.jobs[0].id;
      });
    },
    [request],
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

      let workerMessage = "Profiling worker started.";
      try {
        const workerProcess = await window.missionControl.ensureProjectWorker({
          projectId: project.id,
          baseUrl,
          name: `profile-worker-${project.id}`,
          gpuType: "local",
        });
        workerMessage = workerProcess.started ? "Profiling worker started." : "Profiling worker is already running.";
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

      <section className="workspace">
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

        <nav className="section-tabs" aria-label="Mission Control sections">
          <a href="#overview">Overview</a>
          <a href="#data">Data</a>
          <a href="#agents">Agents</a>
          <a href="#runs">Runs</a>
          <a href="#operations">Operations</a>
          <a href="#export-demo">Export/Demo</a>
        </nav>

        <section className="summary-grid" id="overview">
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
          <Panel title="Experiment Timeline" icon={<ListRestart size={17} />} wide>
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

          <Panel title="Dataset Intelligence" icon={<BarChart3 size={17} />} wide id="data">
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
        </section>

        <section className="content-grid">
          <Panel title="Automation Settings" icon={<SlidersHorizontal size={17} />} wide id="operations">
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
            <Panel title="Selected Champion" icon={<Trophy size={17} />} wide>
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

          <Panel title="Champion Export / Demo" icon={<Trophy size={17} />} wide id="export-demo">
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

          <Panel title="Training Run Summary" icon={<Trophy size={17} />} wide id="runs">
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

          <Panel title="Champion Comparison" icon={<Trophy size={17} />} wide>
            {championComparison.length > 0 ? (
              <div className="comparison-table">
                <div className="comparison-row comparison-head">
                  <span>Model</span>
                  <span>Macro-F1</span>
                  <span>Accuracy</span>
                  <span>Runtime</span>
                  <span>Cost</span>
                  <span>Latency</span>
                  <span>Size</span>
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
                    <strong>{formatMaybeMetric(row.macroF1)}</strong>
                    <span>{formatMaybeMetric(row.accuracy)}</span>
                    <span>{formatSeconds(row.runtimeSeconds)}</span>
                    <span>{formatCurrency(row.costUsd)}</span>
                    <span>{formatLatency(row.latencyMs)}</span>
                    <span>{row.modelSizeMb ? `${row.modelSizeMb.toFixed(row.modelSizeMb < 10 ? 2 : 1)} MB` : "-"}</span>
                    <span>{formatMaybeMetric(row.objectiveFit)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty">Completed run comparisons will appear once training summaries are reported.</div>
            )}
          </Panel>

          <Panel title="Agent Decisions" icon={<BrainCircuit size={17} />} wide id="agents">
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

              {latestDecision ? (
                <div className="decision-card">
                  <div className="decision-card-head">
                    <span>
                      <Badge value={latestDecision.decision_type} />
                      <small>{new Date(latestDecision.created_at).toLocaleString()}</small>
                    </span>
                    <small>{latestDecision.plan_id || "no plan"}</small>
                  </div>
                  <p>{latestDecision.rationale}</p>
                  <div className="decision-payload">
                    {decisionHighlights(latestDecision).map((item) => (
                      <span key={item.label}>
                        <small>{item.label}</small>
                        <strong>{item.value}</strong>
                      </span>
                    ))}
                  </div>
                  {reasoningSections.length > 0 && (
                    <div className="reasoning-grid">
                      {reasoningSections.map((section) => (
                        <div className="reasoning-card" key={section.title}>
                          <strong>{section.title}</strong>
                          {section.items.map((item) => (
                            <p key={item}>{item}</p>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                  {rejectionItems.length > 0 && (
                    <div className="rejection-panel">
                      <strong>Backend Gate And Rejections</strong>
                      <div className="rejection-list">
                        {rejectionItems.map((item) => (
                          <span key={`${item.kind}-${item.text}`}>
                            <small>{item.kind}</small>
                            {item.text}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {candidateScores.length > 0 && (
                    <div className="candidate-score-panel">
                      <strong>Candidate Scores</strong>
                      <div className="candidate-score-list">
                        {candidateScores.map((candidate) => (
                          <div className="candidate-score-row" key={candidate.label}>
                            <div className="candidate-score-head">
                              <span>
                                <strong>{candidate.label}</strong>
                                <small>{candidate.reasons.slice(0, 2).join("; ") || "No rejection reason reported."}</small>
                              </span>
                              <Badge value={candidate.status} />
                            </div>
                            <div className="score-component-list">
                              {candidate.totalScore !== null && (
                                <span>
                                  <small>Total</small>
                                  <strong>{candidate.totalScore.toFixed(3)}</strong>
                                </span>
                              )}
                              {candidate.components.map((component) => (
                                <span key={`${candidate.label}-${component.label}`}>
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
              ) : (
                <div className="empty">No agent decisions yet. Run the reviewer after experiments finish.</div>
              )}

              {detail.decisions.length > 1 && (
                <div className="decision-history">
                  {detail.decisions.slice(1, 5).map((decision) => (
                    <div key={decision.id}>
                      <Badge value={decision.decision_type} />
                      <span>{decision.rationale}</span>
                    </div>
                  ))}
                </div>
              )}

              {detail.strategyScorecards.length > 0 && (
                <div className="decision-history">
                  {detail.strategyScorecards.slice(0, 4).map((scorecard) => (
                    <div key={scorecard.id}>
                      <Badge value={scorecard.outcome} />
                      <span>
                        {scorecard.planning_mode || scorecard.strategy_type || "strategy"} - expected{" "}
                        {scorecard.expected_delta.toFixed(3)}, actual {scorecard.actual_delta.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Automation Timeline" icon={<ListRestart size={17} />} wide>
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

          <Panel title="Agent Memory" icon={<BrainCircuit size={17} />} wide>
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

          <Panel title="Experiment Plan" icon={<ClipboardList size={17} />} wide id="plans">
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
                <div className="experiment-list">
                  {latestPlan.experiments.map((experiment) => (
                    <div className="experiment-item" key={`${latestPlan.id}-${experiment.template}-${experiment.model}`}>
                      <span>
                        <strong>{experiment.model}</strong>
                        <small>{experiment.template}</small>
                      </span>
                      <span>
                        <small>{experiment.epochs} epochs</small>
                        <small>batch {experiment.batch_size}</small>
                      </span>
                      <span>
                        <small>lr</small>
                        <strong>{experiment.learning_rate}</strong>
                      </span>
                      {(experiment.image_size || experiment.optimizer || experiment.scheduler || experiment.class_balancing) && (
                        <span>
                          {experiment.image_size ? <small>{experiment.image_size}px</small> : null}
                          {experiment.optimizer ? <small>{experiment.optimizer}</small> : null}
                          {experiment.scheduler ? <small>{experiment.scheduler}</small> : null}
                          {experiment.class_balancing ? <small>{experiment.class_balancing}</small> : null}
                        </span>
                      )}
                      <p>{experiment.reason}</p>
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

          <Panel title="Manual Job Queue" icon={<Play size={17} />} wide>
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

          <Panel title="Workers" icon={<MonitorDot size={17} />} wide>
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

          <Panel title="Datasets" icon={<Database size={17} />} wide>
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

          <Panel title="Recent Jobs" icon={<SquareTerminal size={17} />} wide>
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
                    <small>{job.id}</small>
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

          <Panel title="Run Metrics" icon={<Activity size={17} />} wide>
            {selectedJob ? (
              <div className="metric-area">
                <div className="selected-job">
                  <span>
                    <strong>{selectedJob.template}</strong>
                    <small>{selectedJob.id}</small>
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
  children,
}: {
  id?: string;
  title: string;
  icon: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <section className={wide ? "panel wide" : "panel"} id={id}>
      <header>
        <span>{icon}</span>
        <h3>{title}</h3>
      </header>
      {children}
    </section>
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
                <div className="export-record" key={exportRecord.id || `${exportRecord.format}-${index}`}>
                  <span>
                    <strong>{exportRecord.format || "model artifact"}</strong>
                    <small>{exportRecord.artifact_uri || exportRecord.model_uri || exportRecord.download_url || "artifact URI pending"}</small>
                  </span>
                  <span>
                    <Badge value={exportRecord.status || "PENDING"} />
                    <small>{exportRecord.size_bytes ? formatBytes(exportRecord.size_bytes) : exportRecord.updated_at || exportRecord.created_at || ""}</small>
                  </span>
                  {(exportRecord.error || (exportRecord.validation_errors ?? []).length > 0) && (
                    <p>{exportRecord.error || exportRecord.validation_errors?.join("; ")}</p>
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
                <div className="prediction-row" key={predictionRow.id || `${predictionRow.image_id}-${index}`}>
                  <span>
                    <strong>{predictionRow.predicted_label || "prediction pending"}</strong>
                    <small>
                      {[
                        predictionRow.status || "",
                        predictionRow.true_label ? `true: ${predictionRow.true_label}` : "",
                        predictionRow.image_id || predictionRow.image_uri || "",
                      ].filter(Boolean).join(" - ") || "image id pending"}
                    </small>
                  </span>
                  <span>
                    {typeof predictionRow.correct === "boolean" && <Badge value={predictionRow.correct ? "CORRECT" : "MISSED"} />}
                    {predictionRow.runtime_unavailable && <Badge value="RUNTIME_UNAVAILABLE" />}
                    <small>
                      {typeof predictionRow.confidence === "number" ? `${Math.round(predictionRow.confidence * 100)}%` : ""}
                      {typeof predictionRow.latency_ms === "number" ? ` ${formatLatency(predictionRow.latency_ms)}` : ""}
                    </small>
                  </span>
                  {(predictionRow.error || predictionRow.error_message) && <p>{predictionRow.error || predictionRow.error_message}</p>}
                  {Array.isArray(predictionRow.top_k) && predictionRow.top_k.length > 0 && (
                    <div className="topk-list">
                      {predictionRow.top_k.slice(0, 3).map((item, topIndex) => (
                        <small key={`${predictionRow.id || index}-${topIndex}`}>
                          {item.label || item.class_name || "class"} {formatTopKScore(item.confidence ?? item.score)}
                        </small>
                      ))}
                    </div>
                  )}
                </div>
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
  return <span className={`badge ${value.toLowerCase()}`}>{value}</span>;
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
      detail: latestDecision ? latestDecision.rationale : "Experiment planner has not recorded a project-level decision.",
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
  ];
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
  addSection("Proposed Experiments", proposedExperimentSummaries(payload.proposed_experiments));
  addSection("Rejected Options", rejectedOptionSummaries(payload.rejected_options));
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

  return items.slice(0, 8);
}

function buildChampionComparison(
  summaries: TrainingRunSummary[],
  evaluations: TrainingRunEvaluation[],
  champion: ProjectChampion | null,
): ChampionComparisonRow[] {
  const evaluationByJob = new Map(evaluations.map((evaluation) => [evaluation.job_id, evaluation]));
  const championJobId = champion?.job_id ?? "";
  return summaries
    .map((summary) => {
      const evaluation = evaluationByJob.get(summary.job_id);
      const modelProfile = evaluation?.model_profile ?? {};
      const holisticScores = evaluation?.holistic_scores ?? {};
      return {
        jobId: summary.job_id,
        model: summary.model,
        accuracy: summary.best_accuracy,
        macroF1: summary.best_macro_f1,
        runtimeSeconds: summary.runtime_seconds,
        costUsd: summary.estimated_cost_usd,
        latencyMs: recordNumber(modelProfile, "estimated_latency_ms"),
        modelSizeMb: recordNumber(modelProfile, "model_size_mb") || recordNumber(modelProfile, "estimated_model_size_mb"),
        objectiveFit: recordNumber(holisticScores, "overall_score") || recordNumber(holisticScores, "objective_fit"),
        isChampion: summary.job_id === championJobId,
      };
    })
    .sort((left, right) => {
      if (left.isChampion !== right.isChampion) return left.isChampion ? -1 : 1;
      return right.macroF1 - left.macroF1;
    })
    .slice(0, 8);
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
        totalScore: numberPayload(record.score) ?? numberPayload(record.total_score),
        reasons: stringArrayPayload(record.reasons),
        components: Object.entries(components)
          .map(([label, value]) => ({ label, value: typeof value === "number" ? value : String(value) }))
          .filter((component) => component.value !== "")
          .slice(0, 6),
      };
    })
    .filter((row) => row.totalScore !== null || row.components.length > 0 || row.reasons.length > 0)
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
    { label: "sampling", value: experiment.sampling_strategy },
    { label: "balancing", value: experiment.class_balancing },
  ].filter((item): item is { label: string; value: string } => typeof item.value === "string" && item.value.length > 0);
}

function backendGateSummary(decision: AgentDecision) {
  const rejections = decisionRejections(decision);
  if (rejections.length > 0) {
    return `${rejections.length} rejected candidate/options visible; stored decision is ${decision.decision_type}.`;
  }
  if (decision.decision_type === "ADD_EXPERIMENTS") {
    return "Accepted for follow-up scheduling when automation settings allow it.";
  }
  return `Accepted project decision: ${decision.decision_type}.`;
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

function numberPayload(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordString(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "string" ? value : "";
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
  if (String(prediction.status || "").toUpperCase() === "RUNTIME_UNAVAILABLE") {
    return { ...prediction, runtime_unavailable: true };
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
    const reason = recordString(record, "reason");
    const imageSize = recordNumber(record, "image_size");
    const details = [strategy, imageSize ? `${imageSize}px` : "", reason].filter(Boolean).join(" - ");
    return details ? `${model}: ${details}` : model;
  });
}

function rejectedOptionSummaries(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => {
    const record = recordObject(item);
    const option = recordString(record, "option") || "Rejected option";
    const reason = recordString(record, "reason") || recordString(record, "evidence");
    return reason ? `${option}: ${reason}` : option;
  });
}

function classifyRejectionReason(text: string) {
  const normalized = text.toLowerCase();
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
  const matrix = champion.evaluation.confusion_matrix;
  if (!Array.isArray(matrix)) return [];
  return matrix
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((value) => (typeof value === "number" ? value : Number(value) || 0)));
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
