import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const timestamp = "2026-06-15T12:00:00.000Z";

let viteServer;

async function loadMissionModel() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      server: { middlewareMode: true },
    });
  }
  return viteServer.ssrLoadModule("/src/features/mission/projectDetailModel.tsx");
}

after(async () => {
  await viteServer?.close();
});

test("stale failed worker state does not block export-ready champion demo availability", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail();
  const exportDemo = exportDemoFixture({ demoPredictions: [] });

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  assert.equal(digest.state, "champion_selected");
  assert.equal(stageStatus(stages, "export"), "done");
  assert.equal(stageStatus(stages, "demo"), "active");
  assert.equal(workerHealth(digest).tone, "warning");
});

test("stale failed worker state does not block completed demo validation", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail();
  const exportDemo = exportDemoFixture({
    demoPredictions: [{ id: "prediction-1", status: "SUCCEEDED", completed_at: timestamp, created_at: timestamp }],
  });

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  assert.equal(digest.state, "champion_selected");
  assert.equal(stageStatus(stages, "export"), "done");
  assert.equal(stageStatus(stages, "demo"), "done");
});

test("failed worker state remains blocking while queued jobs still need capacity", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    jobs: [jobFixture({ id: "job-queued", status: "QUEUED", completed_at: undefined })],
    runSummaries: [],
    runEvaluations: [],
  });
  const exportDemo = exportDemoFixture({ exports: [] });

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  assert.equal(digest.state, "blocked");
  assert.equal(digest.headline, "Worker supervision needs attention.");
  assert.equal(stageStatus(stages, "experiments"), "blocked");
});

function buildDigest(buildMissionDigest, selectedProject, detail) {
  return buildMissionDigest({
    health: { status: "ok", service: "orchestrator", timestamp },
    selectedProject,
    detail,
    automationSettings: automationSettingsFixture(),
    activityStreamState: "connected",
    visibleActivityEvents: [],
    loading: false,
  });
}

function completedChampionDetail(overrides = {}) {
  const champion = championFixture();
  return {
    decisions: [],
    datasets: [datasetFixture()],
    telemetry: null,
    visualAnalysis: {
      analysis: null,
      status: "empty",
      message: "No visual analysis fixture.",
      manualRunSupported: false,
    },
    datasetMetadata: {
      summary: null,
      imports: [],
      status: "empty",
      message: "No dataset metadata fixture.",
    },
    jobs: [jobFixture()],
    plans: [planFixture()],
    runSummaries: [runSummaryFixture()],
    runEvaluations: [runEvaluationFixture()],
    champion,
    championExports: exportDemoFixture().exports,
    championDemoImages: [],
    championDemoPredictions: [],
    championFeedback: [],
    workers: [workerFixture()],
    workerRequirements: [workerRequirementFixture()],
    executionEvents: [],
    agentInvocations: [],
    agentMemory: [],
    strategyScorecards: [],
    ...overrides,
  };
}

function projectFixture() {
  return {
    id: "project-1",
    name: "Mission model fixture",
    goal: "Validate mission state",
    status: "ACTIVE",
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function datasetFixture() {
  return {
    id: "dataset-1",
    project_id: "project-1",
    name: "dataset",
    storage_uri: "file:///dataset",
    size_bytes: 10,
    profile: { total_images: 12, class_count: 2 },
    status: "PROFILED",
    created_at: timestamp,
    profiled_at: timestamp,
  };
}

function planFixture() {
  return {
    id: "plan-1",
    project_id: "project-1",
    dataset_id: "dataset-1",
    status: "APPROVED",
    target_metric: "macro_f1",
    recommended_workers: 1,
    estimated_minutes: 5,
    experiments: [{ template: "resnet", model: "resnet18", epochs: 1, batch_size: 8, learning_rate: 0.001, reason: "fixture" }],
    warnings: [],
    created_at: timestamp,
  };
}

function jobFixture(overrides = {}) {
  return {
    id: "job-1",
    project_id: "project-1",
    template: "resnet",
    status: "SUCCEEDED",
    config: { plan_id: "plan-1", model: "resnet18" },
    created_at: timestamp,
    started_at: timestamp,
    completed_at: timestamp,
    ...overrides,
  };
}

function runSummaryFixture() {
  return {
    job_id: "job-1",
    project_id: "project-1",
    plan_id: "plan-1",
    dataset_id: "dataset-1",
    model: "resnet18",
    provider: "local",
    gpu_type: "cpu",
    status: "SUCCEEDED",
    runtime_seconds: 60,
    estimated_cost_usd: 0,
    best_macro_f1: 0.91,
    best_accuracy: 0.93,
    final_train_loss: 0.2,
    final_val_loss: 0.25,
    epochs_completed: 1,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function runEvaluationFixture() {
  return {
    job_id: "job-1",
    project_id: "project-1",
    plan_id: "plan-1",
    dataset_id: "dataset-1",
    objective_profile: {},
    per_class_metrics: {},
    confusion_matrix: [],
    model_profile: {},
    holistic_scores: { overall_score: 0.9 },
    recommendation_summary: "fixture",
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function championFixture() {
  return {
    id: "champion-1",
    project_id: "project-1",
    dataset_id: "dataset-1",
    plan_id: "plan-1",
    job_id: "job-1",
    source_decision_id: "decision-1",
    selection_reason: "Best fixture result.",
    metrics: { model: "resnet18", primary_metric_label: "Macro-F1", primary_metric_value: 0.91 },
    evaluation: {},
    deployment_profile: {},
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function workerFixture() {
  return {
    id: "worker-1",
    project_id: "project-1",
    name: "stale-worker",
    status: "FAILED",
    gpu_type: "cpu",
    last_heartbeat: timestamp,
  };
}

function workerRequirementFixture() {
  return {
    id: "requirement-1",
    project_id: "project-1",
    plan_id: "plan-1",
    provider: "local",
    gpu_type: "cpu",
    target_count: 1,
    status: "FAILED",
    source: "fixture",
    last_error: "stale worker startup failure",
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function exportDemoFixture(overrides = {}) {
  return {
    hasChampion: true,
    exportStatus: "READY",
    exports: [{
      id: "export-1",
      project_id: "project-1",
      champion_id: "champion-1",
      job_id: "job-1",
      status: "READY",
      format: "onnx",
      artifact_uri: "file:///artifacts/model.onnx",
      completed_at: timestamp,
      updated_at: timestamp,
      created_at: timestamp,
    }],
    projectId: "project-1",
    modelCard: {},
    deploymentProfile: {},
    modelProfile: {},
    useCases: [],
    limitations: [],
    preprocessing: [],
    demoImages: [],
    demoPredictions: [],
    feedback: [],
    ...overrides,
  };
}

function automationSettingsFixture() {
  return {
    auto_review_experiments: false,
    auto_schedule_followups: false,
    auto_execute_plans: false,
    max_followup_rounds: 1,
    default_training_provider: "local",
    default_gpu_type: "cpu",
    cost_mode: "local",
    budget_cap_usd: 0,
    llm_enabled: false,
    agent_mode: "manual",
    llm_provider: "",
    llm_model: "",
    automl_enabled: false,
    automl_sampler: "",
    updated_at: timestamp,
  };
}

function stageStatus(stages, id) {
  return stages.find((stage) => stage.id === id)?.status;
}

function workerHealth(digest) {
  return digest.health.find((item) => item.id === "workers");
}
