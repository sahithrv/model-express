import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

let viteServer;

async function loadMissionPanels() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      server: { middlewareMode: true },
    });
  }
  return viteServer.ssrLoadModule("/src/features/mission/ProjectRoutePanels.tsx");
}

after(async () => {
  await viteServer?.close();
});

test("run execution audit renders compact fidelity without loading large receipts", async () => {
  const { RunExecutionAudit } = await loadMissionPanels();
  let receiptLoads = 0;
  const html = renderToStaticMarkup(
    createElement(RunExecutionAudit, {
      summary: {
        job_id: "job-1",
        project_id: "project-1",
        model: "resnet18",
        status: "SUCCEEDED",
        execution_references: {
          fidelity_verdict: "MATCHED",
          lifecycle_status: "FINALIZED",
          capability_version: "1.0.0",
          accepted_spec_hash: "sha256:accepted-long-hash",
          realized_effective_hash: "sha256:realized-long-hash",
          execution_record_ref: "/jobs/job-1/execution-record",
        },
      },
      evaluation: null,
      job: { id: "job-1", project_id: "project-1", template: "train_experiment", status: "SUCCEEDED", config: {} },
      record: null,
      loading: false,
      error: "",
      onLoadReceipt: () => { receiptLoads += 1; },
    }),
  );

  assert.match(html, /MATCHED/);
  assert.match(html, /Verified: the accepted training semantics were realized/);
  assert.match(html, /Expert execution audit/);
  assert.doesNotMatch(html, /framework_arguments/);
  assert.doesNotMatch(html, /Full bounded receipt/);
  assert.equal(receiptLoads, 0);
});

test("run execution audit gives mismatch and simulation explicit unsafe copy", async () => {
  const { RunExecutionAudit } = await loadMissionPanels();
  const renderStatus = (fidelityVerdict) => renderToStaticMarkup(
    createElement(RunExecutionAudit, {
      summary: {
        job_id: "job-1",
        project_id: "project-1",
        model: "resnet18",
        status: "SUCCEEDED",
        execution_references: { fidelity_verdict: fidelityVerdict, lifecycle_status: "FINALIZED" },
      },
      evaluation: null,
      job: { id: "job-1", project_id: "project-1", template: "train_experiment", status: "SUCCEEDED", config: {} },
      record: null,
      loading: false,
      error: "",
      onLoadReceipt: () => {},
    }),
  );

  assert.match(renderStatus("MISMATCH"), /Not trustworthy as faithful evidence/);
  assert.match(renderStatus("SIMULATED"), /Simulation only: this is not verified real-training evidence/);
});

test("agent drill-down renders legacy rankings and discloses new selection audit details", async () => {
  const { AgentDecisionChat } = await loadMissionPanels();
  const baseTurn = {
    decision: {
      id: "decision-ranking",
      decision_type: "ADD_EXPERIMENTS",
      rationale: "Rank candidates.",
      payload: {},
      created_at: "2026-07-09T12:00:00.000Z",
    },
    question: "What was selected?",
    opening: "The backend ranked the candidate set.",
    highlights: [],
    sections: [],
    retrievedMemory: [],
    rejections: [],
    mechanismCoverage: [],
  };
  const legacyCandidate = {
    label: "Legacy candidate",
    status: "SELECTED",
    mechanism: "class_imbalance",
    intervention: "weighted loss",
    expectedEffect: "Improve recall",
    validationStatus: "",
    totalScore: 0.74,
    baseScore: 0.74,
    selectionScore: null,
    selectionOrder: null,
    selectedExperimentIndex: null,
    selectionAdjustments: [],
    reasons: ["legacy reason"],
    memoryReasons: [],
    memoryHits: [],
    components: [],
  };
  const legacyHTML = renderToStaticMarkup(
    createElement(AgentDecisionChat, {
      turns: [{ ...baseTurn, candidateScores: [legacyCandidate], candidateSelectionTrace: [] }],
    }),
  );
  assert.match(legacyHTML, /Legacy candidate/);
  assert.match(legacyHTML, /Base Score/);
  assert.doesNotMatch(legacyHTML, /Expert selection trace/);

  const adjustment = {
    code: "family_diversity",
    label: "Family Diversity",
    value: -0.12,
    detail: "two candidates from this model family were already selected",
  };
  const auditedHTML = renderToStaticMarkup(
    createElement(AgentDecisionChat, {
      turns: [
        {
          ...baseTurn,
          candidateScores: [
            {
              ...legacyCandidate,
              label: "Audited candidate",
              selectionScore: 0.62,
              selectionOrder: 2,
              selectedExperimentIndex: 1,
              selectionAdjustments: [adjustment],
            },
          ],
          candidateSelectionTrace: [
            {
              selectionOrder: 2,
              selectedCandidateIndex: 0,
              selectedLabel: "Audited candidate",
              totalCandidateCount: 8,
              truncated: true,
              candidates: [
                {
                  candidateIndex: 0,
                  label: "Audited candidate",
                  baseScore: 0.74,
                  adjustedScore: 0.62,
                  selected: true,
                  selectionAdjustments: [adjustment],
                },
              ],
            },
          ],
        },
      ],
    }),
  );
  assert.match(auditedHTML, /Selection Score/);
  assert.match(auditedHTML, /Selection Order/);
  assert.match(auditedHTML, /Family Diversity/);
  assert.match(auditedHTML, /Expert selection trace/);
  assert.match(auditedHTML, /Round 3: Audited candidate/);
});

test("hero metric facts use classification accuracy instead of detection mAP", async () => {
  const { heroMetricFacts } = await loadMissionPanels();

  const facts = heroMetricFacts(
    {
      isDetection: false,
      accuracyDisplay: "0.930",
      macroF1Display: "0.910",
      precisionDisplay: "0.880",
      recallDisplay: "0.870",
      map50Display: "-",
      primaryMetricLabel: "Balanced accuracy score",
      primaryMetricDisplay: "0.910",
    },
    { primaryMetricLabel: "Balanced accuracy score", primaryMetricValue: "0.910" },
  );

  assert.deepEqual(facts.map((fact) => fact.label), ["Accuracy", "Macro F1", "Precision", "Recall"]);
  assert.equal(facts[0].value, "0.930");
});

test("hero metric facts keep detection metrics for object detection rows", async () => {
  const { heroMetricFacts } = await loadMissionPanels();

  const facts = heroMetricFacts(
    {
      isDetection: true,
      accuracyDisplay: "-",
      macroF1Display: "0.310",
      precisionDisplay: "0.700",
      recallDisplay: "0.680",
      map50Display: "0.720",
      primaryMetricLabel: "mAP50-95",
      primaryMetricDisplay: "0.610",
    },
    { primaryMetricLabel: "mAP50-95", primaryMetricValue: "0.610" },
  );

  assert.deepEqual(facts.map((fact) => fact.label), ["mAP50", "mAP50-95", "Precision", "Recall"]);
  assert.equal(facts[0].value, "0.720");
  assert.equal(facts[1].value, "0.610");
});
