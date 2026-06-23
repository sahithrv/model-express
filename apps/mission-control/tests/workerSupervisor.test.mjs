import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadWorkerSupervisor() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true },
    });
  }
  return viteServer.ssrLoadModule("/src/hooks/useWorkerSupervisor.ts");
}

after(async () => {
  await viteServer?.close();
});

test("worker supervisor ignores persisted project work until explicitly armed", async () => {
  const { actionableWorkerRequirements } = await loadWorkerSupervisor();
  const requirements = [
    {
      id: "worker_requirement_1",
      project_id: "project_1",
      plan_id: "plan_1",
      provider: "modal",
      gpu_type: "T4",
      target_count: 1,
      status: "PENDING",
    },
  ];
  const jobs = [
    {
      status: "QUEUED",
      config: { plan_id: "plan_1", provider: "modal" },
    },
  ];

  assert.deepEqual(
    actionableWorkerRequirements({ enabled: false, selectedProjectId: "project_1", workerRequirements: requirements, jobs }),
    [],
  );
  assert.deepEqual(
    actionableWorkerRequirements({ enabled: true, selectedProjectId: "project_2", workerRequirements: requirements, jobs }),
    [],
  );
  assert.deepEqual(
    actionableWorkerRequirements({ enabled: true, selectedProjectId: "project_1", workerRequirements: requirements, jobs }).map(
      (requirement) => requirement.id,
    ),
    ["worker_requirement_1"],
  );
});