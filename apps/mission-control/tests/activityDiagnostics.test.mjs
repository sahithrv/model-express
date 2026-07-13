import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadActivityDiagnostics() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  return viteServer.ssrLoadModule("/src/api/activityDiagnostics.ts");
}

after(async () => {
  await viteServer?.close();
});

test("activity visibility is summarized after a deterministic visible timestamp", async () => {
  const { summarizeActivityVisibility } = await loadActivityDiagnostics();
  const summaries = summarizeActivityVisibility(
    [
      {
        createdAtMs: 1_100,
        receivedAtMs: 1_200,
        streamOpenedAtMs: 1_000,
        catchUpReason: "initial_catch_up",
      },
      {
        createdAtMs: 800,
        receivedAtMs: 1_250,
        streamOpenedAtMs: 1_000,
        catchUpReason: "initial_catch_up",
      },
    ],
    1_300,
  );

  assert.deepEqual(summaries, [
    {
      reason_code: "live",
      sample_count: 1,
      latency_sample_count: 1,
      invalid_sample_count: 0,
      dropped_count: 0,
      latency_total_ms: 200,
      latency_min_ms: 200,
      latency_max_ms: 200,
      latency_average_ms: 200,
      commit_delay_total_ms: 100,
      commit_delay_average_ms: 100,
    },
    {
      reason_code: "initial_catch_up",
      sample_count: 1,
      latency_sample_count: 1,
      invalid_sample_count: 0,
      dropped_count: 0,
      latency_total_ms: 500,
      latency_min_ms: 500,
      latency_max_ms: 500,
      latency_average_ms: 500,
      commit_delay_total_ms: 50,
      commit_delay_average_ms: 50,
    },
  ]);
});

test("activity visibility samples and invalid timestamps remain bounded", async () => {
  const {
    appendActivityVisibilitySample,
    maxPendingActivityVisibilitySamples,
    summarizeActivityVisibility,
  } = await loadActivityDiagnostics();
  let samples = [];
  let dropped = 0;
  for (let index = 0; index < maxPendingActivityVisibilitySamples + 5; index += 1) {
    const queued = appendActivityVisibilitySample(samples, {
      createdAtMs: index === maxPendingActivityVisibilitySamples + 4 ? Number.NaN : index,
      receivedAtMs: 100,
      streamOpenedAtMs: 50,
      catchUpReason: "reconnect_catch_up",
    });
    samples = queued.samples;
    dropped += queued.dropped;
  }

  assert.equal(samples.length, maxPendingActivityVisibilitySamples);
  assert.equal(dropped, 5);
  const summaries = summarizeActivityVisibility(samples, 200, dropped);
  assert.equal(summaries.reduce((total, summary) => total + summary.sample_count, 0), maxPendingActivityVisibilitySamples);
  assert.equal(summaries.reduce((total, summary) => total + summary.dropped_count, 0), 5);
  assert.equal(summaries.find((summary) => summary.reason_code === "invalid_timestamp")?.invalid_sample_count, 1);
});
