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

test("demo image runnable guard treats thumbnails as display-only for stored records", async () => {
  const { demoImageInferenceURI, demoImageIsRunnable, demoImagePreviewURI } = await loadMissionModel();
  const thumbnailURI = "data:image/jpeg;base64,THUMB";
  const originalURI = "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png";

  const thumbnailOnlyStored = {
    uri: thumbnailURI,
    image_uri: thumbnailURI,
    preview_uri: thumbnailURI,
    thumbnail_uri: thumbnailURI,
    split: "test",
    metadata: { demo_source_type: "heldout_test_thumbnail_preview" },
  };
  const originalStored = {
    uri: thumbnailURI,
    image_uri: thumbnailURI,
    preview_uri: thumbnailURI,
    thumbnail_uri: thumbnailURI,
    source_artifact_uri: originalURI,
    split: "test",
  };
  const sourceOnlyStored = {
    source_artifact_uri: originalURI,
    metadata: { demo_source_type: "heldout_test_original_artifact" },
  };
  const metadataOnlyStoredPreview = {
    uri: originalURI,
    split: "test",
    metadata: {
      preview_uri: thumbnailURI,
      thumbnail_uri: thumbnailURI,
      source_artifact_uri: originalURI,
      demo_source_type: "heldout_test_original_artifact",
    },
  };
  const customDataImage = { uri: "data:image/png;base64,AAAA", image_uri: "data:image/png;base64,AAAA", split: "custom" };
  const customFileImage = { uri: "file:///tmp/cat.png", image_uri: "file:///tmp/cat.png", split: "custom" };

  assert.equal(demoImageIsRunnable(thumbnailOnlyStored), false);
  assert.equal(demoImageInferenceURI(thumbnailOnlyStored), "");
  assert.equal(demoImageIsRunnable(originalStored), true);
  assert.equal(demoImageIsRunnable(sourceOnlyStored), true);
  assert.equal(demoImagePreviewURI(metadataOnlyStoredPreview), thumbnailURI);
  assert.equal(demoImageInferenceURI(metadataOnlyStoredPreview), "");
  assert.equal(demoImageIsRunnable(customDataImage), true);
  assert.equal(demoImageInferenceURI(customDataImage), customDataImage.uri);
  assert.equal(demoImageIsRunnable(customFileImage), true);
  assert.equal(demoImageInferenceURI(customFileImage), customFileImage.uri);
});

test("results summary keeps backend champion first even when display score is lower", async () => {
  const { buildResultsSummary } = await loadMissionModel();
  const detail = completedChampionDetail();
  const exportDemo = exportDemoFixture();
  const summary = buildResultsSummary(detail, [
    comparisonRowFixture({
      jobId: "job-champion",
      model: "convnext_tiny",
      rankScore: 0.62,
      primaryMetricValue: 0.876,
      secondaryMetricValue: 0.883,
      isChampion: true,
    }),
    comparisonRowFixture({
      jobId: "job-cheap",
      model: "convnext_tiny",
      rankScore: 0.91,
      primaryMetricValue: 0.639,
      secondaryMetricValue: 0.652,
      isChampion: false,
    }),
  ], exportDemo);

  assert.equal(summary.topCandidates[0].jobId, "job-champion");
  assert.equal(summary.topCandidates[0].rank, 1);
  assert.equal(summary.topCandidates[0].status, "Best model so far");
});

test("mission rows and results leaderboard share backend selection candidate order", async () => {
  const { buildChampionComparison, buildMissionRunRows, buildResultsSummary } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: championFixture({
      job_id: "job-backend-a",
      metrics: {
        model: "convnext_tiny",
        primary_metric_label: "Macro-F1",
        primary_metric_value: 0.82,
        selection_candidates: [
          { job_id: "job-backend-a", deployment_readiness_score: 0.82 },
          { job_id: "job-backend-b", deployment_readiness_score: 0.76 },
        ],
      },
    }),
    jobs: [
      jobFixture({ id: "job-backend-a", config: { plan_id: "plan-1", model: "convnext_tiny" }, status: "SUCCEEDED", completed_at: "2026-06-15T12:20:00.000Z" }),
      jobFixture({ id: "job-backend-b", config: { plan_id: "plan-1", model: "resnet50" }, status: "SUCCEEDED", completed_at: "2026-06-15T12:15:00.000Z" }),
      jobFixture({ id: "job-running", config: { plan_id: "plan-1", model: "mobilenet_v3" }, status: "RUNNING", started_at: "2026-06-15T12:25:00.000Z", completed_at: "" }),
    ],
    runSummaries: [
      runSummaryFixture({ job_id: "job-backend-a", model: "convnext_tiny", best_macro_f1: 0.82, best_accuracy: 0.86, estimated_cost_usd: 0.02, updated_at: "2026-06-15T12:20:00.000Z" }),
      runSummaryFixture({ job_id: "job-backend-b", model: "resnet50", best_macro_f1: 0.91, best_accuracy: 0.93, estimated_cost_usd: 0.05, updated_at: "2026-06-15T12:15:00.000Z" }),
    ],
    runEvaluations: [
      runEvaluationFixture({ job_id: "job-backend-a", holistic_scores: { overall_score: 0.82 } }),
      runEvaluationFixture({ job_id: "job-backend-b", holistic_scores: { overall_score: 0.91 } }),
    ],
  });

  const comparison = buildChampionComparison(detail.runSummaries, detail.runEvaluations, detail.jobs, detail.champion);
  const rows = buildMissionRunRows(detail);
  const summary = buildResultsSummary(detail, comparison, exportDemoFixture());

  assert.deepEqual(comparison.map((row) => row.jobId), ["job-backend-a", "job-backend-b"]);
  assert.equal(rows[0].id, "job-backend-a");
  assert.equal(rows[0].candidateRank, 1);
  assert.equal(summary.championModel, "convnext_tiny");
  assert.equal(summary.topCandidates[0].jobId, "job-backend-a");
  assert.equal(rows.at(-1).id, "job-running");
});

test("classification ranking and hero facts do not use detection mAP ordering", async () => {
  const { buildChampionComparison, buildMissionRunRows, buildResultsSummary, heroMetricFacts } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    jobs: [
      jobFixture({ id: "job-macro-leader", config: { plan_id: "plan-1", model: "convnext_tiny", task_type: "classification" }, status: "SUCCEEDED" }),
      jobFixture({ id: "job-map-distractor", config: { plan_id: "plan-1", model: "resnet50", task_type: "classification" }, status: "SUCCEEDED" }),
    ],
    runSummaries: [
      runSummaryFixture({ job_id: "job-macro-leader", model: "convnext_tiny", best_macro_f1: 0.84, best_accuracy: 0.88, best_map50_95: 0.12, estimated_cost_usd: 0.02 }),
      runSummaryFixture({ job_id: "job-map-distractor", model: "resnet50", best_macro_f1: 0.72, best_accuracy: 0.9, best_map50_95: 0.99, estimated_cost_usd: 0.02 }),
    ],
    runEvaluations: [
      runEvaluationFixture({ job_id: "job-macro-leader", objective_profile: { task_type: "classification", balanced_accuracy: 0.84, accuracy: 0.88 }, holistic_scores: {} }),
      runEvaluationFixture({ job_id: "job-map-distractor", objective_profile: { task_type: "classification", balanced_accuracy: 0.72, accuracy: 0.9, heldout_test_map50_95: 0.99 }, holistic_scores: {} }),
    ],
  });

  const comparison = buildChampionComparison(detail.runSummaries, detail.runEvaluations, detail.jobs, detail.champion);
  const rows = buildMissionRunRows(detail);
  const summary = buildResultsSummary(detail, comparison, exportDemoFixture());
  const facts = heroMetricFacts(rows[0], summary);

  assert.equal(comparison[0].jobId, "job-macro-leader");
  assert.equal(rows[0].id, "job-macro-leader");
  assert.equal(summary.topCandidates[0].jobId, "job-macro-leader");
  assert.equal(summary.topCandidates[0].metricLabel, "Balanced accuracy score");
  assert.deepEqual(facts.map((fact) => fact.label), ["Balanced accuracy score", "Accuracy", "Cost", "Runtime"]);
});
test("demo images put known-correct held-out examples before hard failures", async () => {
  const { demoImageCategory, demoImageCategoryDetail, demoImageTrainingPredictionText, demoImagesFromUnknown } = await loadMissionModel();
  const ordered = demoImagesFromUnknown([
    { id: "wrong-high-confidence", metadata: { correct_at_training: false } },
    { id: "correct-a", metadata: { correct_at_training: true, demo_role: "representative" } },
    { id: "wrong-low-confidence", metadata: { correct_at_training: "false" } },
    { id: "correct-b", metadata: { correct_at_training: "true" } },
    { id: "unknown" },
  ]);

  assert.deepEqual(ordered.map((image) => image.id), [
    "correct-a",
    "wrong-high-confidence",
    "correct-b",
    "wrong-low-confidence",
    "unknown",
  ]);
  assert.equal(demoImageCategory(ordered[0]), "representative");
  assert.equal(demoImageCategory(ordered[1]), "challenge");
  assert.equal(demoImageCategory({ metadata: { demo_set: "challenge_heldout" } }), "challenge");
  assert.equal(demoImageTrainingPredictionText({ metadata: { predicted_label_at_training: "dog", confidence_at_training: 0.8123 } }), "Training-time prediction: dog (81%).");
  assert.match(demoImageCategoryDetail({ metadata: { correct_at_training: false, predicted_label_at_training: "dog" } }), /Known hard example/);
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

test("champion export-ready handoff path skips refinement waiting state", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail();
  const exportDemo = exportDemoFixture();

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  assert.equal(stageStatus(stages, "refinement"), "done");
  assert.equal(stageStatus(stages, "champion"), "done");
  assert.equal(stageStatus(stages, "export"), "done");
  assert.equal(stageStatus(stages, "demo"), "active");
});

test("validated demo marks handoff as complete with no stale waiting before completed stages", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail();
  const exportDemo = exportDemoFixture({
    demoPredictions: [{ id: "prediction-1", status: "SUCCEEDED", completed_at: timestamp, created_at: timestamp }],
  });

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  const latestDoneIndex = stages.reduce((index, stage, stageIndex) => (stage.status === "done" ? stageIndex : index), -1);
  const hasWaitingAfterDone = stages.some((stage, index) => index > latestDoneIndex && stage.status === "waiting");
  const completed = stages.filter((stage) => stage.status === "done").length;

  assert.equal(hasWaitingAfterDone, false);
  assert.equal(completed, 9);
});

test("active follow-up refinement remains visible when no handoff is ready", async () => {
  const { buildMissionDigest, buildMissionStages } = await loadMissionModel();
  const project = projectFixture();
  const detail = completedChampionDetail({
    champion: null,
    decisions: [decisionFixture({ id: "decision-add", decision_type: "ADD_EXPERIMENTS" })],
    plans: [planFixture()],
  });
  const exportDemo = exportDemoFixture({ exports: [] });

  const digest = buildDigest(buildMissionDigest, project, detail);
  const stages = buildMissionStages(project, detail, digest, exportDemo);

  assert.equal(stageStatus(stages, "champion"), "waiting");
  assert.equal(stageStatus(stages, "refinement"), "active");
  assert.equal(stageStatus(stages, "export"), "waiting");
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

test("portable bundle is derived separately from ready ONNX export", async () => {
  const { buildChampionExportDemo } = await loadMissionModel();
  const detail = completedChampionDetail({
    championExports: [
      {
        id: "export-onnx",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-1",
        status: "READY",
        format: "onnx",
        artifact_uri: "file:///exports/model.onnx",
        metadata: {
          portable_bundle_uri: "file:///exports/portable_inference_bundle.zip",
          portable_inference_bundle: {
            schema_version: "portable_inference_bundle_v1",
            status: "created",
            artifact_uri: "file:///exports/portable_inference_bundle.zip",
            contents: ["model.onnx", "manifest.json"],
          },
          manifest: {
            schema_version: "champion_export_manifest_v1",
            artifacts: [
              { format: "onnx", status: "created", path: "/exports/model.onnx" },
              {
                format: "portable_inference_bundle",
                status: "created",
                path: "/exports/portable_inference_bundle.zip",
                contents: ["model.onnx", "manifest.json"],
              },
            ],
          },
        },
      },
    ],
  });

  const exportDemo = buildChampionExportDemo(detail);

  assert.equal(exportDemo.exports[0].artifact_uri, "file:///exports/model.onnx");
  assert.equal(exportDemo.portableBundle?.artifact_uri, "file:///exports/portable_inference_bundle.zip");
  assert.equal(exportDemo.portableBundle?.status, "created");
  assert.deepEqual(exportDemo.portableBundle?.contents, ["model.onnx", "manifest.json"]);
});

test("export archive surfaces downloadable portable bundles before stale pending records", async () => {
  const { buildChampionExportDemo } = await loadMissionModel();
  const detail = completedChampionDetail({
    championExports: [
      {
        id: "export-pending",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-pending",
        status: "PENDING",
        format: "onnx",
        created_at: "2026-06-16T10:00:00Z",
      },
      {
        id: "export-ready",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-ready",
        status: "READY",
        format: "onnx",
        artifact_uri: "file:///exports/model.onnx",
        completed_at: "2026-06-16T09:00:00Z",
      },
      {
        id: "export-portable",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-portable",
        status: "READY",
        format: "onnx",
        artifact_uri: "file:///exports/model-portable.onnx",
        completed_at: "2026-06-16T08:00:00Z",
        metadata: {
          portable_inference_bundle: {
            status: "created",
            artifact_uri: "file:///exports/portable_inference_bundle.zip",
          },
        },
      },
      {
        id: "export-failed",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-failed",
        status: "FAILED",
        format: "onnx",
        failed_at: "2026-06-16T11:00:00Z",
      },
    ],
  });

  const exportDemo = buildChampionExportDemo(detail);

  assert.deepEqual(exportDemo.exports.map((item) => item.id), [
    "export-portable",
    "export-ready",
    "export-pending",
    "export-failed",
  ]);
  assert.equal(exportDemo.portableBundle?.artifact_uri, "file:///exports/portable_inference_bundle.zip");
});

test("ready export record overrides stale pending champion metadata", async () => {
  const { buildChampionExportDemo } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: championFixture({
      deployment_profile: {
        model_card: { export_status: "pending" },
      },
    }),
    championExports: [
      {
        id: "export-ready",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-1",
        status: "READY",
        format: "onnx",
        artifact_uri: "file:///artifacts/model.onnx",
        completed_at: timestamp,
        updated_at: timestamp,
        created_at: timestamp,
      },
    ],
  });

  const exportDemo = buildChampionExportDemo(detail);

  assert.equal(exportDemo.exportStatus, "READY");
  assert.equal(exportDemo.limitations.includes("Final export artifact is still pending."), false);
});

test("pending export with validation errors and no active export job is surfaced as blocked", async () => {
  const { buildChampionExportDemo } = await loadMissionModel();
  const detail = completedChampionDetail({
    jobs: [jobFixture({ id: "job-1", template: "resnet", status: "SUCCEEDED" })],
    championExports: [
      {
        id: "export-pending-artifact",
        project_id: "project-1",
        champion_id: "champion-1",
        job_id: "job-export",
        status: "PENDING_ARTIFACT",
        format: "onnx",
        validation_errors: ["ARTIFACT_NOT_FOUND: source artifact is not available to the worker"],
        created_at: timestamp,
      },
    ],
  });

  const exportDemo = buildChampionExportDemo(detail);

  assert.equal(
    exportDemo.limitations.includes("Export is blocked until the source artifact is available or export is re-run."),
    true,
  );
});

test("terminal job status overrides stale running training summary state", async () => {
  const { activityCardFromRun, effectiveTrainingRunStatus, summarizeTrainingRuns } = await loadMissionModel();
  const summary = runSummaryFixture({ status: "RUNNING" });
  const job = jobFixture({ status: "SUCCEEDED" });

  const totals = summarizeTrainingRuns([summary], [], [job]);
  const card = activityCardFromRun(summary, null, job);

  assert.equal(effectiveTrainingRunStatus(summary, job), "SUCCEEDED");
  assert.equal(totals.activeRuns, 0);
  assert.equal(card.type, "result");
  assert.equal(card.status, "succeeded");
});

test("model improvement data orders plans and uses holistic evaluation scores", async () => {
  const { buildModelImprovementData } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    decisions: [],
    strategyScorecards: [],
    plans: [
      planFixture({ id: "plan-2", created_at: "2026-06-16T12:00:00.000Z" }),
      planFixture({ id: "plan-1", created_at: "2026-06-15T12:00:00.000Z" }),
    ],
    jobs: [
      jobFixture({ id: "job-p1", config: { plan_id: "plan-1", model: "resnet18" }, completed_at: "2026-06-15T12:10:00.000Z" }),
      jobFixture({ id: "job-p2", config: { plan_id: "plan-2", model: "convnext_tiny" }, completed_at: "2026-06-16T12:10:00.000Z" }),
    ],
    runSummaries: [
      runSummaryFixture({ job_id: "job-p1", plan_id: "plan-1", model: "resnet18", best_macro_f1: 0.71, best_accuracy: 0.74 }),
      runSummaryFixture({ job_id: "job-p2", plan_id: "plan-2", model: "convnext_tiny", best_macro_f1: 0.82, best_accuracy: 0.84 }),
    ],
    runEvaluations: [
      runEvaluationFixture({ job_id: "job-p1", plan_id: "plan-1", holistic_scores: { overall_score: 0.62 } }),
      runEvaluationFixture({ job_id: "job-p2", plan_id: "plan-2", holistic_scores: { overall_score: 0.79 } }),
    ],
  });

  const data = buildModelImprovementData(detail);

  assert.equal(data.state, "ready");
  assert.deepEqual(data.points.map((point) => point.planId), ["plan-1", "plan-2"]);
  assert.deepEqual(data.points.map((point) => point.bestScore), [0.62, 0.79]);
  assert.deepEqual(data.points.map((point) => point.cumulativeBestScore), [0.62, 0.79]);
  assert.equal(data.points[0].source, "Training evaluation");
  assert.equal(data.points[0].scoreBasis, "Holistic score");
  assert.equal(Number(data.improvementDelta.toFixed(3)), 0.17);
});

test("model improvement data leaves completed plans without score as missing", async () => {
  const { buildModelImprovementData } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    decisions: [],
    strategyScorecards: [],
    plans: [planFixture({ id: "plan-missing" })],
    jobs: [jobFixture({ id: "job-missing", config: { plan_id: "plan-missing", model: "resnet18" }, status: "SUCCEEDED" })],
    runSummaries: [
      runSummaryFixture({
        job_id: "job-missing",
        plan_id: "plan-missing",
        best_macro_f1: undefined,
        best_accuracy: undefined,
      }),
    ],
    runEvaluations: [runEvaluationFixture({ job_id: "job-missing", plan_id: "plan-missing", holistic_scores: {}, objective_profile: {}, per_class_metrics: {} })],
  });

  const data = buildModelImprovementData(detail);

  assert.equal(data.state, "no_scored_models");
  assert.equal(data.completedPlanCount, 1);
  assert.equal(data.scoredPlanCount, 0);
  assert.equal(data.points[0].bestScore, null);
  assert.equal(data.points[0].cumulativeBestScore, null);
  assert.match(data.points[0].missingReason, /no score field/);
});

test("model improvement data uses champion decision score and ignores scorecard deltas", async () => {
  const { buildModelImprovementData } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    plans: [planFixture({ id: "plan-decision" })],
    jobs: [jobFixture({ id: "job-decision", config: { plan_id: "plan-decision", model: "resnet18" }, status: "SUCCEEDED" })],
    runSummaries: [],
    runEvaluations: [],
    decisions: [
      decisionFixture({
        plan_id: "plan-decision",
        decision_type: "SELECT_CHAMPION",
        payload: {
          champion_score: 0.74,
          champion_model: "resnet18",
          champion_job_id: "job-decision",
        },
      }),
    ],
    strategyScorecards: [
      {
        id: "scorecard-1",
        project_id: "project-1",
        dataset_id: "dataset-1",
        source_decision_id: "decision-1",
        source_plan_id: "plan-0",
        followup_plan_id: "plan-decision",
        strategy_type: "planner_followup",
        planning_mode: "planner_followup",
        dataset_traits: {},
        objective_profile: {},
        proposed_changes: {},
        expected_delta: 0.12,
        actual_delta: 0.99,
        confidence_before: 0.4,
        confidence_after: 0.99,
        cost_usd: 0,
        runtime_seconds: 0,
        outcome: "improved_champion",
        lesson: "delta only",
        tags: [],
        created_at: timestamp,
      },
    ],
  });

  const data = buildModelImprovementData(detail);

  assert.equal(data.state, "ready");
  assert.equal(data.points[0].bestScore, 0.74);
  assert.equal(data.points[0].source, "Champion decision");
  assert.equal(data.points[0].scoreBasis, "Champion score");
});

test("model improvement data preserves YOLO detection score fallback", async () => {
  const { buildModelImprovementData } = await loadMissionModel();
  const detail = completedChampionDetail({
    champion: null,
    championExports: [],
    decisions: [],
    strategyScorecards: [],
    plans: [planFixture({ id: "plan-yolo", target_metric: "mAP50_95" })],
    jobs: [
      jobFixture({
        id: "job-yolo",
        template: "yolo11n",
        config: { plan_id: "plan-yolo", task: "object_detection", model: "yolo11n" },
        status: "SUCCEEDED",
      }),
    ],
    runSummaries: [
      runSummaryFixture({
        job_id: "job-yolo",
        plan_id: "plan-yolo",
        model: "yolo11n",
        best_macro_f1: 0.31,
        best_accuracy: 0.44,
        best_map50_95: 0.52,
        best_map50: 0.67,
        target_metric: "mAP50_95",
      }),
    ],
    runEvaluations: [
      runEvaluationFixture({
        job_id: "job-yolo",
        plan_id: "plan-yolo",
        objective_profile: { task_type: "object_detection", heldout_test_map50_95: 0.61, heldout_test_map50: 0.72 },
        holistic_scores: {},
      }),
    ],
  });

  const data = buildModelImprovementData(detail);

  assert.equal(data.state, "ready");
  assert.equal(data.points[0].bestScore, 0.61);
  assert.equal(data.points[0].source, "Training evaluation");
  assert.equal(data.points[0].scoreBasis, "mAP50-95");
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
    championExportsStatus: {
      status: "available",
      message: "Fixture export records loaded.",
    },
    loadStatus: projectDetailLoadStatusFixture(),
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

function planFixture(overrides = {}) {
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
    ...overrides,
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

function runSummaryFixture(overrides = {}) {
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
    ...overrides,
  };
}

function runEvaluationFixture(overrides = {}) {
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
    ...overrides,
  };
}

function championFixture(overrides = {}) {
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
    ...overrides,
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

function projectDetailLoadStatusFixture() {
  return {
    runEvaluations: { status: "available", message: "Fixture evaluations loaded." },
    decisions: { status: "available", message: "Fixture decisions loaded." },
    workerRequirements: { status: "available", message: "Fixture worker requirements loaded." },
    championExports: { status: "available", message: "Fixture export records loaded." },
    championDemoPredictions: { status: "available", message: "Fixture demo predictions loaded." },
    championFeedback: { status: "available", message: "Fixture feedback loaded." },
    liveRefresh: { status: "available", message: "Fixture detail loaded." },
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

function comparisonRowFixture(overrides = {}) {
  return {
    jobId: "job-1",
    model: "resnet18",
    rankScore: 0.8,
    backendRank: null,
    rankSource: "frontend_comparison",
    runStatus: "SUCCEEDED",
    primaryMetricLabel: "Macro-F1",
    primaryMetricValue: 0.91,
    secondaryMetricLabel: "Accuracy",
    secondaryMetricValue: 0.93,
    runtimeSeconds: 60,
    costUsd: 0,
    latencyMs: 14.9,
    modelSize: "106.8 MB",
    objectiveFit: 0.9,
    trainValidationGap: null,
    divergenceStatus: "Stable",
    seedVariance: null,
    seedRunCount: 0,
    isChampion: true,
    ...overrides,
  };
}

function decisionFixture(overrides = {}) {
  return {
    id: "decision-1",
    decision_type: "ADD_EXPERIMENTS",
    created_at: timestamp,
    updated_at: timestamp,
    project_id: projectFixture().id,
    payload: {},
    rationale: "Fixture follow-up plan request.",
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
