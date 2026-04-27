import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Box,
  BrainCircuit,
  Database,
  HardDriveUpload,
  ListRestart,
  MonitorDot,
  Play,
  Plus,
  RefreshCcw,
  Server,
  SquareTerminal,
} from "lucide-react";
import type { Dataset, EpochMetric, Health, Job, Project, Worker } from "./types";

const defaultBaseUrl = localStorage.getItem("orchestratorUrl") ?? "http://localhost:8080";

type ProjectDetail = {
  datasets: Dataset[];
  jobs: Job[];
  workers: Worker[];
};

type Notice = {
  kind: "info" | "error";
  text: string;
};

export function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [health, setHealth] = useState<Health | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [detail, setDetail] = useState<ProjectDetail>({ datasets: [], jobs: [], workers: [] });
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [metrics, setMetrics] = useState<EpochMetric[]>([]);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [loading, setLoading] = useState(false);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const selectedJob = useMemo(
    () => detail.jobs.find((job) => job.id === selectedJobId) ?? null,
    [detail.jobs, selectedJobId],
  );
  const firstDatasetId = detail.datasets[0]?.id ?? "";

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

  const refreshProjectDetail = useCallback(
    async (projectId: string) => {
      if (!projectId) {
        setDetail({ datasets: [], jobs: [], workers: [] });
        return;
      }

      const [datasets, jobs, workers] = await Promise.all([
        request<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`),
        request<{ jobs: Job[] }>(`/projects/${projectId}/jobs`),
        request<{ workers: Worker[] }>(`/projects/${projectId}/workers`),
      ]);

      setDetail({
        datasets: datasets.datasets,
        jobs: jobs.jobs,
        workers: workers.workers,
      });

      if (!selectedJobId && jobs.jobs.length > 0) {
        setSelectedJobId(jobs.jobs[0].id);
      }
    },
    [request, selectedJobId],
  );

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setNotice(null);
    try {
      await refreshHealth();
      await refreshProjects();
      if (selectedProjectId) {
        await refreshProjectDetail(selectedProjectId);
      }
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }, [refreshHealth, refreshProjectDetail, refreshProjects, selectedProjectId]);

  useEffect(() => {
    localStorage.setItem("orchestratorUrl", baseUrl);
  }, [baseUrl]);

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (selectedProjectId) {
      refreshProjectDetail(selectedProjectId).catch((error) =>
        setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
      );
    }
  }, [refreshProjectDetail, selectedProjectId]);

  useEffect(() => {
    if (!selectedJobId) {
      setMetrics([]);
      return;
    }

    request<{ metrics: EpochMetric[] }>(`/jobs/${selectedJobId}/metrics`)
      .then((response) => setMetrics(response.metrics))
      .catch((error) =>
        setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }),
      );
  }, [request, selectedJobId]);

  async function createProject(formData: FormData) {
    const name = String(formData.get("name") ?? "").trim();
    const goal = String(formData.get("goal") ?? "").trim();
    if (!name || !goal) return;

    const project = await request<Project>("/projects", {
      method: "POST",
      body: { name, goal },
    });
    setSelectedProjectId(project.id);
    await refreshProjects();
  }

  async function createDataset(formData: FormData) {
    if (!selectedProjectId) return;

    await request<Dataset>(`/projects/${selectedProjectId}/datasets`, {
      method: "POST",
      body: {
        name: String(formData.get("name") ?? "").trim(),
        storage_uri: String(formData.get("storage_uri") ?? "").trim(),
        checksum_sha256: String(formData.get("checksum_sha256") ?? "").trim(),
        size_bytes: Number(formData.get("size_bytes") ?? 0),
      },
    });

    await refreshProjectDetail(selectedProjectId);
  }

  async function uploadDatasetFolder() {
    if (!selectedProjectId) return;

    setLoading(true);
    setNotice(null);
    try {
      const metadata = await window.missionControl.selectAndUploadDataset({
        projectId: selectedProjectId,
      });
      if (!metadata) {
        setNotice({ kind: "info", text: "Dataset upload cancelled" });
        return;
      }

      await request<Dataset>(`/projects/${selectedProjectId}/datasets`, {
        method: "POST",
        body: metadata,
      });

      await refreshProjectDetail(selectedProjectId);
      setNotice({ kind: "info", text: `Uploaded ${metadata.name}` });
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
        template: String(formData.get("template") ?? "mobilenet_transfer"),
        config,
      },
    });

    setSelectedJobId(job.id);
    await refreshProjectDetail(selectedProjectId);
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
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={22} />
          </div>
          <div>
            <h1>Model Express</h1>
            <p>Mission Control</p>
          </div>
        </div>

        <label className="field compact">
          <span>Orchestrator</span>
          <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
        </label>

        <button className="command primary" onClick={refreshAll} disabled={loading}>
          <RefreshCcw size={16} />
          Refresh
        </button>

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
          <MetricCard icon={<MonitorDot size={18} />} label="Workers" value={detail.workers.length} />
          <MetricCard
            icon={<ListRestart size={18} />}
            label="Queued"
            value={detail.jobs.filter((job) => job.status === "QUEUED").length}
          />
        </section>

        <section className="content-grid">
          <Panel title="New Project" icon={<Plus size={17} />}>
            <form
              className="stack"
              onSubmit={(event) => {
                event.preventDefault();
                handleSubmit(createProject, event.currentTarget);
              }}
            >
              <input name="name" placeholder="Project name" />
              <input name="goal" placeholder="Classifier goal" />
              <button className="command" disabled={loading}>
                <Plus size={16} />
                Create
              </button>
            </form>
          </Panel>

          <Panel title="Register Dataset" icon={<HardDriveUpload size={17} />}>
            <div className="stack">
              <button className="command primary" onClick={uploadDatasetFolder} disabled={!selectedProjectId || loading}>
                <HardDriveUpload size={16} />
                Choose Folder & Upload
              </button>
              <div className="divider">Manual object registration</div>
              <form
                className="stack"
                onSubmit={(event) => {
                  event.preventDefault();
                  handleSubmit(createDataset, event.currentTarget);
                }}
              >
              <input name="name" placeholder="Dataset name" />
              <input name="storage_uri" placeholder="s3://model-express/datasets/project_1/demo.zip" />
              <div className="two-col">
                <input name="checksum_sha256" placeholder="SHA256" />
                <input name="size_bytes" placeholder="Size bytes" type="number" min="0" />
              </div>
              <button className="command" disabled={!selectedProjectId || loading}>
                <HardDriveUpload size={16} />
                Register
              </button>
              </form>
            </div>
          </Panel>

          <Panel title="Create Job" icon={<Play size={17} />}>
            <form
              className="stack"
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
                rows={5}
                defaultValue={JSON.stringify(
                  { dataset_id: firstDatasetId || "dataset_id_here" },
                  null,
                  2,
                )}
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

          <Panel title="Jobs" icon={<SquareTerminal size={17} />} wide>
            <div className="job-list">
              {detail.jobs.map((job) => (
                <button
                  key={job.id}
                  className={job.id === selectedJobId ? "job active" : "job"}
                  onClick={() => setSelectedJobId(job.id)}
                >
                  <span>
                    <strong>{job.template}</strong>
                    <small>{job.id}</small>
                  </span>
                  <Badge value={job.status} />
                </button>
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

          <Panel title="Metric Stream" icon={<Activity size={17} />} wide>
            {selectedJob ? (
              <div className="metric-area">
                <div className="selected-job">
                  <strong>{selectedJob.id}</strong>
                  <Badge value={selectedJob.status} />
                </div>
                <div className="metric-bars">
                  {metrics.map((metric) => (
                    <div className="metric-bar" key={`${metric.job_id}-${metric.epoch}`}>
                      <span>Epoch {metric.epoch}</span>
                      <div>
                        <i style={{ width: `${Math.min((metric.metrics.macro_f1 ?? 0) * 100, 100)}%` }} />
                      </div>
                      <b>{metric.metrics.macro_f1?.toFixed(3) ?? "-"}</b>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty">No job selected</div>
            )}
          </Panel>
        </section>
      </section>
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
  icon: React.ReactNode;
  wide?: boolean;
  children: React.ReactNode;
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

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
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
