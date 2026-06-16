const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");
const { pathToFileURL } = require("url");

const { __test } = require("./main.cjs");

function tempDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test("orchestrator requests are limited to app paths, approved methods, and loopback by default", () => {
  const request = __test.validateOrchestratorRequest({
    baseUrl: "http://127.0.0.1:8080",
    method: "post",
    path: "/projects?limit=1",
    body: { name: "demo" },
  });

  assert.equal(request.method, "POST");
  assert.equal(request.url, "http://127.0.0.1:8080/projects?limit=1");
  assert.equal(request.bodyText, '{"name":"demo"}');

  assert.throws(
    () => __test.validateOrchestratorRequest({ baseUrl: "http://127.0.0.1:8080", method: "PUT", path: "/projects" }),
    /Unsupported orchestrator request method/,
  );
  assert.throws(
    () => __test.validateOrchestratorRequest({ baseUrl: "http://127.0.0.1:8080", path: "http://example.com/projects" }),
    /absolute app path/,
  );
  assert.throws(
    () => __test.validateOrchestratorRequest({ baseUrl: "http://example.com:8080", path: "/projects" }),
    /Non-loopback orchestrator URLs/,
  );
});

test("orchestrator origins and API token headers use explicit env controls", () => {
  const env = {
    MODEL_EXPRESS_ALLOWED_ORCHESTRATOR_ORIGINS: "https://mx.example.test",
    MODEL_EXPRESS_API_TOKEN: "test-token",
  };

  assert.equal(__test.validateOrchestratorBaseUrl("https://mx.example.test", env), "https://mx.example.test");
  assert.deepEqual(__test.apiTokenHeaders(env), {
    Authorization: "Bearer test-token",
    "X-Model-Express-API-Token": "test-token",
  });
});

test("mission control request env loads API token from repo env files", () => {
  const repo = tempDir("mx-repo-env-");
  fs.writeFileSync(
    path.join(repo, ".env.local"),
    "MODEL_EXPRESS_API_TOKEN=file-token\nMODEL_EXPRESS_ALLOWED_ORCHESTRATOR_ORIGINS=https://mx.example.test\n",
  );

  const env = __test.missionControlEnv({ MODEL_EXPRESS_ROOT: repo });

  assert.equal(env.MODEL_EXPRESS_API_TOKEN, "file-token");
  assert.deepEqual(__test.apiTokenHeaders(env), {
    Authorization: "Bearer file-token",
    "X-Model-Express-API-Token": "file-token",
  });
  assert.equal(
    __test.validateOrchestratorRequest({ baseUrl: "https://mx.example.test", path: "/projects" }, env).url,
    "https://mx.example.test/projects",
  );
});

test("public orchestrator exposure requires LAN auth controls", () => {
  assert.throws(
    () => __test.requireAuthenticatedOrchestratorExposure({ MODEL_EXPRESS_API_TOKEN: "test-token" }),
    /MODEL_EXPRESS_ALLOW_LAN=true/,
  );
  assert.throws(
    () => __test.requireAuthenticatedOrchestratorExposure({ MODEL_EXPRESS_ALLOW_LAN: "true" }),
    /MODEL_EXPRESS_API_TOKEN/,
  );
  assert.doesNotThrow(() =>
    __test.requireAuthenticatedOrchestratorExposure({
      MODEL_EXPRESS_ALLOW_LAN: "true",
      MODEL_EXPRESS_API_TOKEN: "test-token",
    }),
  );
});

test("remote training sessions have bounded lifetime", () => {
  assert.equal(__test.remoteTrainingSessionTtlMs({}), 6 * 60 * 60 * 1000);
  assert.equal(__test.remoteTrainingSessionTtlMs({ MODEL_EXPRESS_REMOTE_TRAINING_SESSION_TTL_SECONDS: "1" }), 5 * 60 * 1000);
  assert.equal(__test.remoteTrainingSessionTtlMs({ MODEL_EXPRESS_REMOTE_TRAINING_SESSION_TTL_SECONDS: "999999" }), 24 * 60 * 60 * 1000);
  assert.equal(__test.remoteTrainingSessionActive({ processes: [], expiresAt: Date.now() + 60_000 }), true);
  assert.equal(__test.remoteTrainingSessionActive({ processes: [], expiresAt: Date.now() - 1 }), false);
});

test("dataset folder operations require a picker-backed selection token", () => {
  __test.selectedDatasetFolders.clear();
  const selectedRoot = tempDir("mx-dataset-selected-");
  const unselectedRoot = tempDir("mx-dataset-unselected-");
  fs.writeFileSync(path.join(selectedRoot, "image.jpg"), "x");
  fs.writeFileSync(path.join(unselectedRoot, "image.jpg"), "x");

  const selected = __test.rememberDatasetFolder(selectedRoot);
  assert.equal(selected.name, path.basename(selectedRoot));
  assert.equal(__test.resolveDatasetFolderOption({ datasetToken: selected.token }).path, fs.realpathSync.native(selectedRoot));
  assert.equal(__test.resolveDatasetFolderOption({ datasetPath: selectedRoot }).path, fs.realpathSync.native(selectedRoot));
  assert.throws(
    () => __test.resolveDatasetFolderOption({ datasetPath: unselectedRoot }),
    /must be selected/,
  );
});

test("dataset preflight uses picker-backed selection token path", async () => {
  __test.selectedDatasetFolders.clear();
  const selectedRoot = tempDir("mx-dataset-preflight-");
  fs.writeFileSync(path.join(selectedRoot, "image.jpg"), "x");

  const selected = __test.rememberDatasetFolder(selectedRoot);
  const preflight = await __test.preflightDatasetFolder({ datasetToken: selected.token });

  assert.equal(preflight.file_count, 1);
  assert.equal(preflight.uncompressed_size_bytes, 1);
  assert.equal(preflight.largest_file.path, "image.jpg");
});

test("dataset upload endpoints reject remote origins unless allowlisted", () => {
  assert.equal(__test.validateUploadEndpoint("http://127.0.0.1:9000"), "http://127.0.0.1:9000");
  assert.throws(
    () => __test.validateUploadEndpoint("https://storage.example.test", {}),
    /Remote dataset upload endpoints/,
  );
  assert.equal(
    __test.validateUploadEndpoint("https://storage.example.test", {
      MODEL_EXPRESS_ALLOWED_UPLOAD_ORIGINS: "https://storage.example.test",
    }),
    "https://storage.example.test",
  );
});

test("artifact paths are limited to configured artifact roots", () => {
  const allowedRoot = tempDir("mx-artifacts-");
  const outsideRoot = tempDir("mx-outside-");
  const artifactPath = path.join(allowedRoot, "model.onnx");
  const outsidePath = path.join(outsideRoot, "model.onnx");
  const unsupportedPath = path.join(allowedRoot, "training.log");
  fs.writeFileSync(artifactPath, "onnx");
  fs.writeFileSync(outsidePath, "onnx");
  fs.writeFileSync(unsupportedPath, "log");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_ARTIFACT_ROOTS: allowedRoot,
  };

  assert.equal(__test.validateLocalArtifactPath(artifactPath, env), fs.realpathSync.native(artifactPath));
  assert.equal(__test.validateLocalArtifactPath(pathToFileURL(artifactPath).toString(), env), fs.realpathSync.native(artifactPath));
  assert.throws(
    () => __test.validateLocalArtifactPath(outsidePath, env),
    /configured Model Express artifact or export root/,
  );
  assert.throws(
    () => __test.validateLocalArtifactPath(unsupportedPath, env),
    /supported model artifact extension/,
  );
});

test("portable bundle save paths require configured artifact roots and zip archives", () => {
  const allowedRoot = tempDir("mx-bundles-");
  const outsideRoot = tempDir("mx-bundle-outside-");
  const bundlePath = path.join(allowedRoot, "portable_inference_bundle.zip");
  const outsideBundlePath = path.join(outsideRoot, "portable_inference_bundle.zip");
  const modelPath = path.join(allowedRoot, "model.onnx");
  fs.writeFileSync(bundlePath, "zip");
  fs.writeFileSync(outsideBundlePath, "zip");
  fs.writeFileSync(modelPath, "onnx");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_ARTIFACT_ROOTS: allowedRoot,
  };

  assert.equal(__test.validateLocalPortableBundlePath(bundlePath, env), fs.realpathSync.native(bundlePath));
  assert.equal(__test.validateLocalPortableBundlePath(pathToFileURL(bundlePath).toString(), env), fs.realpathSync.native(bundlePath));
  assert.throws(
    () => __test.validateLocalPortableBundlePath(outsideBundlePath, env),
    /configured Model Express artifact or export root/,
  );
  assert.throws(
    () => __test.validateLocalPortableBundlePath(modelPath, env),
    /supported archive extension/,
  );
});

test("export artifact save streams local zip under allowed export root", async () => {
  const allowedRoot = tempDir("mx-export-save-");
  const bundlePath = path.join(allowedRoot, "portable_inference_bundle.zip");
  const destinationPath = path.join(tempDir("mx-export-destination-"), "saved.zip");
  fs.writeFileSync(bundlePath, "zip-bytes");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  const result = await __test.saveExportArtifact(
    {
      artifactUri: pathToFileURL(bundlePath).toString(),
      suggestedName: "portable_inference_bundle.zip",
      kind: "portable_bundle",
    },
    {
      env,
      showSaveDialog: async (_window, options) => {
        assert.equal(options.defaultPath, "portable_inference_bundle.zip");
        return { canceled: false, filePath: destinationPath };
      },
    },
  );

  assert.equal(result.canceled, false);
  assert.equal(result.file_path, destinationPath);
  assert.equal(result.bytes, 9);
  assert.equal(fs.readFileSync(destinationPath, "utf8"), "zip-bytes");
});

test("export artifact save keeps portable bundle saves zip-only", async () => {
  const allowedRoot = tempDir("mx-export-portable-kind-");
  const modelPath = path.join(allowedRoot, "model.onnx");
  fs.writeFileSync(modelPath, "onnx");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  await assert.rejects(
    () =>
      __test.saveExportArtifact(
        { artifactUri: modelPath, kind: "portable_bundle" },
        {
          env,
          showSaveDialog: async () => ({ canceled: true }),
        },
      ),
    /Portable export bundle must use a supported ZIP extension/,
  );
});

test("export artifact save allows non-portable model export extensions", async () => {
  const allowedRoot = tempDir("mx-export-model-kind-");
  const modelPath = path.join(allowedRoot, "model.onnx");
  const destinationPath = path.join(tempDir("mx-export-model-destination-"), "model.onnx");
  fs.writeFileSync(modelPath, "onnx");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  const result = await __test.saveExportArtifact(
    { artifactUri: modelPath, suggestedName: "model.onnx", kind: "model_artifact" },
    {
      env,
      showSaveDialog: async () => ({ canceled: false, filePath: destinationPath }),
    },
  );

  assert.equal(result.canceled, false);
  assert.equal(result.bytes, 4);
  assert.equal(fs.readFileSync(destinationPath, "utf8"), "onnx");
});

test("export artifact save rejects local zip outside allowed roots", async () => {
  const allowedRoot = tempDir("mx-export-allowed-");
  const outsideRoot = tempDir("mx-export-outside-");
  const outsideBundlePath = path.join(outsideRoot, "portable_inference_bundle.zip");
  fs.writeFileSync(outsideBundlePath, "zip");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };
  let dialogOpened = false;

  await assert.rejects(
    () =>
      __test.saveExportArtifact(
        { artifactUri: pathToFileURL(outsideBundlePath).toString(), kind: "portable_bundle" },
        {
          env,
          showSaveDialog: async () => {
            dialogOpened = true;
            return { canceled: true };
          },
        },
      ),
    /configured Model Express artifact or export root/,
  );
  assert.equal(dialogOpened, false);
});

test("export artifact save rejects unsupported extensions", async () => {
  const allowedRoot = tempDir("mx-export-extension-");
  const logPath = path.join(allowedRoot, "training.log");
  fs.writeFileSync(logPath, "log");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  await assert.rejects(
    () =>
      __test.saveExportArtifact(
        { artifactUri: logPath, kind: "model_artifact" },
        {
          env,
          showSaveDialog: async () => ({ canceled: true }),
        },
      ),
    /supported download extension/,
  );
});

test("export artifact save rejects parent traversal segments before resolving", async () => {
  const allowedRoot = tempDir("mx-export-traversal-");
  const bundlePath = path.join(allowedRoot, "portable_inference_bundle.zip");
  fs.writeFileSync(bundlePath, "zip");
  const traversalPath = `${allowedRoot}${path.sep}nested${path.sep}..${path.sep}portable_inference_bundle.zip`;
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  await assert.rejects(
    () =>
      __test.saveExportArtifact(
        { artifactUri: traversalPath, kind: "portable_bundle" },
        {
          env,
          showSaveDialog: async () => ({ canceled: true }),
        },
      ),
    /parent directory segments/,
  );
});

test("export artifact save returns canceled true when dialog is canceled", async () => {
  const allowedRoot = tempDir("mx-export-cancel-");
  const bundlePath = path.join(allowedRoot, "portable_inference_bundle.zip");
  fs.writeFileSync(bundlePath, "zip");
  const env = {
    ...process.env,
    MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS: allowedRoot,
  };

  const result = await __test.saveExportArtifact(
    { artifactUri: bundlePath, suggestedName: "portable_inference_bundle.zip", kind: "portable_bundle" },
    {
      env,
      showSaveDialog: async () => ({ canceled: true }),
    },
  );

  assert.deepEqual(result, {
    canceled: true,
    file_path: "",
    bytes: 0,
    artifact_uri: bundlePath,
  });
});

test("artifact loading rejects repo log artifacts even under artifact root", () => {
  const repoRoot = tempDir("mx-repo-");
  const logArtifact = path.join(repoRoot, "artifacts", "logs", "model.onnx");
  fs.mkdirSync(path.dirname(logArtifact), { recursive: true });
  fs.writeFileSync(logArtifact, "onnx");

  assert.throws(
    () => __test.validateLocalArtifactPath(logArtifact, { MODEL_EXPRESS_ROOT: repoRoot }),
    /refuses log directories/,
  );
});

test("S3 artifact endpoint overrides require storage allowlisting", () => {
  const env = { S3_ENDPOINT_URL: "http://127.0.0.1:9000" };
  assert.equal(__test.resolveS3Endpoint({}, "artifact", env), "http://127.0.0.1:9000");
  assert.throws(
    () => __test.resolveS3Endpoint({ endpoint: "http://127.0.0.1:9002" }, "artifact", env),
    /configured S3 endpoint/,
  );
  assert.equal(
    __test.resolveS3Endpoint({ endpoint: "https://storage.example.test" }, "artifact", {
      ...env,
      MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS: "https://storage.example.test",
    }),
    "https://storage.example.test",
  );
});

test("ONNX external data mount paths and tunnel logs are sanitized", () => {
  assert.equal(__test.externalDataMountPath("weights/model.onnx.data"), "weights/model.onnx.data");
  assert.equal(__test.externalDataMountPath("../secret"), "");
  assert.equal(__test.externalDataMountPath("C:\\Windows\\win.ini"), "");

  assert.equal(
    __test.parseCloudflaredTunnelUrl("ready: https://unit-test.trycloudflare.com"),
    "https://unit-test.trycloudflare.com",
  );
  assert.equal(
    __test.safeLogText("ready: https://unit-test.trycloudflare.com Authorization=Bearer secret"),
    "ready: [redacted-url] Authorization=Bearer [redacted]",
  );
  const signed = __test.safeLogText(
    "MODAL_ORCHESTRATOR_URL=https://modal.example.test AWS_ACCESS_KEY_ID=abc AWS_SECRET_ACCESS_KEY=def sk-testtoken123456789 https://s3.example.test/object?X-Amz-Credential=abc&X-Amz-Signature=def",
  );
  assert(!signed.includes("modal.example.test"));
  assert(!signed.includes("abc"));
  assert(!signed.includes("def"));
  assert(!signed.includes("sk-testtoken"));
});

test("Modal remote URLs reject local private and unsafe service targets", () => {
  assert.equal(
    __test.validateRemoteModalUrl("https://orchestrator.example.test", "MODAL_ORCHESTRATOR_URL"),
    "https://orchestrator.example.test",
  );
  assert.throws(
    () => __test.validateRemoteModalUrl("http://orchestrator.example.test", "MODAL_ORCHESTRATOR_URL", {}),
    /must use https/,
  );
  assert.throws(
    () => __test.validateRemoteModalUrl("https://10.0.0.2:8080", "MODAL_ORCHESTRATOR_URL"),
    /remotely reachable/,
  );
  assert.throws(
    () => __test.validateRemoteModalUrl("https://storage.example.test:9001", "MODAL_S3_ENDPOINT_URL"),
    /must not expose/,
  );
});
