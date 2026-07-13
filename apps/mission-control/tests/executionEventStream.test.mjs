import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadStreamModule() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  return viteServer.ssrLoadModule("/src/api/executionEventStream.ts");
}

after(async () => viteServer?.close());

test("execution-event stream path uses the snapshot cursor as an exclusive safe position", async () => {
  const { executionEventStreamPath } = await loadStreamModule();
  assert.equal(
    executionEventStreamPath("project / one", 42),
    "/projects/project%20%2F%20one/events/stream/v2?cursor=42&limit=100&interval_ms=1000",
  );
  assert.throws(() => executionEventStreamPath("project", -1), /nonnegative safe integer/);
  assert.throws(() => executionEventStreamPath("project", Number.MAX_SAFE_INTEGER + 1), /nonnegative safe integer/);
});
