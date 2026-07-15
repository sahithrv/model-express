import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadClient() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  return viteServer.ssrLoadModule("/src/api/missionControlClient.ts");
}

after(async () => viteServer?.close());

test("live request plan exposes compact snapshot and v2 stream paths", async () => {
  const { liveRequestPath } = await loadClient();
  assert.equal(liveRequestPath("liveState", { projectId: "project / one" }), "/projects/project%20%2F%20one/live-state");
  assert.equal(liveRequestPath("executionEventStreamV2", { projectId: "project" }), "/projects/project/events/stream/v2");
});

test("structured HTTP errors retain only bounded recovery identity", async () => {
  const {
    OrchestratorHttpError,
    isCursorRecoveryStatus,
    isUnsupportedIncrementalStatus,
  } = await loadClient();
  const error = new OrchestratorHttpError({
    __mission_control_http_error: true,
    status: 410,
    statusText: "Gone",
    message: "cursor_too_old",
    path: "/projects/resource/events/stream/v2",
    payload: {
      reason_code: "cursor_too_old",
      prompt: "must-not-be-retained",
      storage_uri: "s3://private/data",
    },
  });
  assert.equal(error.status, 410);
  assert.equal(error.reasonCode, "cursor_too_old");
  assert.equal("payload" in error, false);
  assert.equal(JSON.stringify(error).includes("private"), false);
  assert.equal(isCursorRecoveryStatus(400), true);
  assert.equal(isCursorRecoveryStatus(409), true);
  assert.equal(isCursorRecoveryStatus(410), true);
  assert.equal(isUnsupportedIncrementalStatus(404), true);
  assert.equal(isUnsupportedIncrementalStatus(405), true);
  assert.equal(isUnsupportedIncrementalStatus(501), true);
});

test("policy errors retain actionable bounded findings without retaining raw payload", async () => {
  const { OrchestratorHttpError } = await loadClient();
  const error = new OrchestratorHttpError({
    __mission_control_http_error: true,
    status: 422,
    message: "policy denied",
    payload: {
      code: "POLICY_CATALOG_ID_DENIED",
      effective_policy_hash: "sha256:abc123",
      findings: [{
        code: "POLICY_CATALOG_ID_DENIED",
        catalog: "export_formats",
        id: "pytorch",
        remediation: "Choose ONNX.",
        prompt: "must-not-be-retained",
      }],
      storage_uri: "s3://private/data",
    },
  });
  assert.equal(error.reasonCode, "POLICY_CATALOG_ID_DENIED");
  assert.equal(error.policy?.findings[0]?.remediation, "Choose ONNX.");
  assert.equal("payload" in error, false);
  assert.equal(JSON.stringify(error).includes("private"), false);
  assert.equal(JSON.stringify(error).includes("must-not-be-retained"), false);
});
