import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let viteServer;

async function loadPolicyModule() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { middlewareMode: true, hmr: false },
    });
  }
  return viteServer.ssrLoadModule("/src/features/policy/ExperimentPolicyPanel.tsx");
}

after(async () => viteServer?.close());

test("policy scope endpoints cover account, project, dataset, and run subjects", async () => {
  const { experimentPolicyEndpoint } = await loadPolicyModule();
  assert.equal(experimentPolicyEndpoint("account", "project", ""), "/settings/experiment-policy");
  assert.equal(experimentPolicyEndpoint("project", "project / 1", ""), "/projects/project%20%2F%201/experiment-policy");
  assert.equal(experimentPolicyEndpoint("dataset", "project", "dataset / 2"), "/datasets/dataset%20%2F%202/experiment-policy");
  assert.equal(experimentPolicyEndpoint("run", "project", "job / 3"), "/jobs/job%20%2F%203/experiment-policy");
});

test("policy editor emits immutable Roasty reference and explicit deny rules", async () => {
  const { buildExperimentPolicyDocument, buildCatalogExperimentPolicyDocument } = await loadPolicyModule();
  const document = buildExperimentPolicyDocument(true, ["torchscript", "pytorch"]);
  assert.deepEqual(document.profile_refs, [{ id: "roasty_v1", version: "1.0.0" }]);
  assert.deepEqual(document.rules.map((rule) => rule.selector.ids[0]), ["pytorch", "torchscript"]);
  assert.equal(document.rules.every((rule) => rule.effect === "deny"), true);

  const catalogDocument = buildCatalogExperimentPolicyDocument(false, [
    { catalog: "models", id: "vit_b_16" },
    { catalog: "augmentation_policies", id: "strong" },
  ]);
  assert.deepEqual(catalogDocument.rules.map((rule) => [rule.selector.catalog, rule.selector.ids[0]]), [
    ["augmentation_policies", "strong"],
    ["models", "vit_b_16"],
  ]);
});
