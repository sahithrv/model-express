import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

let viteServer;

async function loadLocalInference() {
  if (!viteServer) {
    viteServer = await createServer({
      root: appRoot,
      logLevel: "error",
      optimizeDeps: { noDiscovery: true },
      server: { middlewareMode: true },
    });
  }
  return viteServer.ssrLoadModule("/src/championLocalInference.ts");
}

after(async () => {
  await viteServer?.close();
});

test("browser ONNX inference requires a passed export self-test", async () => {
  const { championLocalInferenceSafety, readyBrowserONNXExport, readyONNXExport } = await loadLocalInference();
  const exportRecord = exportFixture({
    export_self_test: { status: "failed", diagnostic_reason: "ONNX_OUTPUT_MISMATCH" },
  });

  const safety = championLocalInferenceSafety(exportRecord);

  assert.equal(readyONNXExport([exportRecord])?.id, "export-1");
  assert.equal(readyBrowserONNXExport([exportRecord]), undefined);
  assert.equal(safety.safe, false);
  assert.equal(safety.code, "EXPORT_SELF_TEST_FAILED");
});

test("browser ONNX inference is allowed only for static labeled parity-safe manifests", async () => {
  const { championLocalInferenceSafety, readyBrowserONNXExport } = await loadLocalInference();
  const exportRecord = exportFixture({
    export_self_test: { status: "passed" },
  });

  const safety = championLocalInferenceSafety(exportRecord);

  assert.equal(safety.safe, true);
  assert.equal(safety.code, "LOCAL_INFERENCE_SAFE");
  assert.equal(readyBrowserONNXExport([exportRecord])?.id, "export-1");
});

test("browser ONNX inference refuses duplicate labels and bbox preprocessing", async () => {
  const { championLocalInferenceSafety, readyBrowserONNXExport } = await loadLocalInference();
  const duplicateLabels = exportFixture({
    class_labels: ["cat", "cat"],
    export_self_test: { status: "passed" },
  });
  const bboxPreprocessing = exportFixture({
    export_self_test: { status: "passed" },
    preprocessing_contract: {
      config: {
        resize_strategy: "squash",
        crop_strategy: "bbox_crop_if_available",
        normalization: "none",
        bbox_mode: "ignore",
      },
    },
  });

  assert.equal(championLocalInferenceSafety(duplicateLabels).code, "CLASS_LABEL_ORDER_INVALID");
  assert.equal(championLocalInferenceSafety(bboxPreprocessing).code, "LOCAL_PREPROCESSING_UNSUPPORTED");
  assert.equal(readyBrowserONNXExport([duplicateLabels, bboxPreprocessing]), undefined);
});

test("browser ONNX inference refuses artifact-backed held-out thumbnails", async () => {
  const { predictChampionImage } = await loadLocalInference();
  const thumbnailURI = "data:image/jpeg;base64,BBBB";
  const originalURI = "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png";
  const runtime = {
    artifactURI: "file:///exports/model.onnx",
    session: {},
    metadata: {},
    labels: ["cat", "dog"],
    imageSize: 8,
    normalization: null,
    resizeStrategy: "squash",
    cropStrategy: "none",
    modelKind: "classification",
    taskType: "image_classification",
    confidenceThreshold: 0.25,
    iouThreshold: 0.7,
    maxDetections: 100,
  };

  await assert.rejects(
    () =>
      predictChampionImage(
        runtime,
        {
          uri: originalURI,
          image_uri: originalURI,
          thumbnail_uri: thumbnailURI,
          source_artifact_uri: originalURI,
          metadata: {
            demo_source_type: "heldout_test_original_artifact",
            parity_safe: true,
            source_artifact_uri: originalURI,
          },
        },
        thumbnailURI,
      ),
    (error) => {
      assert.equal(error.code, "LOCAL_ORIGINAL_ARTIFACT_UNAVAILABLE");
      return true;
    },
  );
});

function exportFixture(metadataOverrides = {}) {
  return {
    id: "export-1",
    project_id: "project-1",
    champion_id: "champion-1",
    job_id: "job-1",
    status: "READY",
    format: "onnx",
    artifact_uri: "file:///exports/model.onnx",
    metadata: {
      manifest: {
        metadata: manifestMetadata(metadataOverrides),
      },
    },
  };
}

function manifestMetadata(overrides = {}) {
  const preprocessing = {
    resize_strategy: "squash",
    crop_strategy: "none",
    normalization: "none",
    bbox_mode: "ignore",
  };
  return {
    model_kind: "classification",
    task_type: "image_classification",
    class_labels: ["cat", "dog"],
    input_shape: [1, 3, 8, 8],
    export_self_test: { status: "passed" },
    preprocessing_contract: { config: preprocessing },
    inference_contract: {
      input: { model_tensor_shape: [1, 3, 8, 8] },
      preprocessing: { config: preprocessing },
    },
    ...overrides,
  };
}
