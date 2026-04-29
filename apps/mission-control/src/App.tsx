import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
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
  SlidersHorizontal,
  SquareTerminal,
  Timer,
  Trophy,
  X,
} from "lucide-react";
import type {
  AgentDecision,
  Dataset,
  EpochMetric,
  ExperimentPlan,
  Health,
  Job,
  Project,
  TrainingRunSummary,
  Worker,
  AutomationSettings,
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
  workers: Worker[];
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
  >
>;

const defaultAutomationSettings: AutomationSettings = {
  auto_review_experiments: false,
  auto_schedule_followups: false,
  auto_execute_plans: false,
  max_followup_rounds: 2,
  default_training_provider: "local",
  default_gpu_type: "",
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
    workers: [],
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

  const latestPlan = detail.plans[0] ?? null;
  const latestDecision = detail.decisions[0] ?? null;
  const latestDecisionHasFollowUpPlan = latestDecision
    ? detail.plans.some((plan) => plan.source_decision_id === latestDecision.id)
    : false;
  const runTotals = useMemo(() => summarizeTrainingRuns(detail.runSummaries), [detail.runSummaries]);

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
        setDetail({ decisions: [], datasets: [], jobs: [], plans: [], runSummaries: [], workers: [] });
        return;
      }

      const [datasets, jobs, plans, runSummaries, decisions, workers] = await Promise.all([
        request<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`),
        request<{ jobs: Job[] }>(`/projects/${projectId}/jobs`),
        request<{ plans: ExperimentPlan[] }>(`/projects/${projectId}/plans`),
        request<{ summaries: TrainingRunSummary[] }>(`/projects/${projectId}/training-run-summaries`),
        request<{ decisions: AgentDecision[] }>(`/projects/${projectId}/agent-decisions`),
        request<{ workers: Worker[] }>(`/projects/${projectId}/workers`),
      ]);

      setDetail({
        decisions: decisions.decisions,
        datasets: datasets.datasets,
        jobs: jobs.jobs,
        plans: plans.plans,
        runSummaries: runSummaries.summaries,
        workers: workers.workers,
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

  useEffect(() => {
    localStorage.setItem("orchestratorUrl", baseUrl);
  }, [baseUrl]);

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (selectedProjectId) {
      setJobPage(0);
      refreshProjectDetail(selectedProjectId).catch((error) =>
        setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
      );
    }
  }, [refreshProjectDetail, selectedProjectId]);

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

        <section className="summary-grid">
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

        <section className="content-grid">
          <Panel title="Automation Settings" icon={<SlidersHorizontal size={17} />} wide>
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

          <Panel title="Training Run Summary" icon={<Trophy size={17} />} wide>
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

          <Panel title="Agent Decisions" icon={<BrainCircuit size={17} />} wide>
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
            </div>
          </Panel>

          <Panel title="Experiment Plan" icon={<ClipboardList size={17} />} wide>
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
                      <p>{experiment.reason}</p>
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
  title,
  icon,
  wide = false,
  children,
}: {
  title: string;
  icon: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <section className={wide ? "panel wide" : "panel"}>
      <header>
        <span>{icon}</span>
        <h3>{title}</h3>
      </header>
      {children}
    </section>
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

function decisionHighlights(decision: AgentDecision) {
  const payload = decision.payload ?? {};
  const items: Array<{ label: string; value: string }> = [];

  if (typeof payload.champion_model === "string" && payload.champion_model) {
    items.push({ label: "Champion", value: payload.champion_model });
  }

  const championScore = numberPayload(payload.champion_score);
  if (championScore !== null) {
    items.push({ label: "Score", value: championScore.toFixed(3) });
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

function formatChartValue(value: number) {
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4);
}
