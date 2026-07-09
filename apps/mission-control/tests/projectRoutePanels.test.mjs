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
