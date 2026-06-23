import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

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