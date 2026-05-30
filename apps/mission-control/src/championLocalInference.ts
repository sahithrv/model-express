import * as ort from "onnxruntime-web";
import type { ChampionDemoImage, ChampionDemoPrediction, ChampionExport } from "./types";

export type ChampionLocalRuntime = {
  artifactURI: string;
  session: ort.InferenceSession;
  metadata: Record<string, unknown>;
  labels: string[];
  imageSize: number;
  normalization: { mean: number[]; std: number[] } | null;
  resizeStrategy: string;
};

export type ChampionLocalRuntimeContext = {
  exportRecord: ChampionExport;
  deploymentProfile?: Record<string, unknown>;
  modelProfile?: Record<string, unknown>;
};

export function readyONNXExport(exports: ChampionExport[]) {
  return exports.find((item) => {
    const status = String(item.status || "").toUpperCase();
    const format = String(item.format || "").toLowerCase();
    const uri = item.artifact_uri || item.model_uri || item.download_url || "";
    return status === "READY" && format === "onnx" && uri.toLowerCase().includes(".onnx");
  });
}

export async function createChampionLocalRuntime(
  artifact: { artifact_uri: string; bytes: ArrayBuffer },
  context: ChampionLocalRuntimeContext,
): Promise<ChampionLocalRuntime> {
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(artifact.bytes, {
    executionProviders: ["wasm"],
  });
  const metadata = exportMetadata(context);
  const labels = classLabels(metadata, context.modelProfile, context.deploymentProfile);
  const imageSize = inputImageSize(metadata, context.modelProfile);
  return {
    artifactURI: artifact.artifact_uri,
    session,
    metadata,
    labels,
    imageSize,
    normalization: normalizationValues(metadata),
    resizeStrategy: resizeStrategy(metadata),
  };
}

export async function predictChampionImage(
  runtime: ChampionLocalRuntime,
  image: ChampionDemoImage,
  imageSource: string,
): Promise<ChampionDemoPrediction> {
  const started = performance.now();
  const input = await imageTensor(imageSource, runtime);
  const inputName = runtime.session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = {
    [inputName]: new ort.Tensor("float32", input, [1, 3, runtime.imageSize, runtime.imageSize]),
  };
  const outputs = await runtime.session.run(feeds);
  const output = outputs[runtime.session.outputNames[0]] ?? Object.values(outputs)[0];
  const logits = Array.from(output.data as Iterable<number>);
  const probabilities = softmax(logits);
  const topK = probabilities
    .map((confidence, index) => ({
      label: runtime.labels[index] || `class_${index}`,
      confidence,
    }))
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, 5);
  const predicted = topK[0];
  const trueLabel = demoImageLabel(image);

  return {
    id: `local-${Date.now()}`,
    image_id: image.image_id || image.id || "",
    image_uri: imageSource,
    status: "SUCCEEDED",
    predicted_label: predicted?.label || "",
    true_label: trueLabel,
    confidence: predicted?.confidence ?? 0,
    latency_ms: performance.now() - started,
    correct: trueLabel ? predicted?.label === trueLabel : undefined,
    top_k: topK,
    metadata: {
      runtime: "onnxruntime-web",
      artifact_uri: runtime.artifactURI,
      thumbnail_uri: imageSource,
    },
  };
}

function exportMetadata(context: ChampionLocalRuntimeContext) {
  const exportRecordMetadata = recordObject(context.exportRecord.metadata);
  const manifestMetadata = recordObject(recordObject(exportRecordMetadata.manifest).metadata);
  const deploymentManifestMetadata = recordObject(recordObject(recordObject(context.deploymentProfile).export_manifest).metadata);
  if (Object.keys(manifestMetadata).length > 0) return manifestMetadata;
  if (Object.keys(deploymentManifestMetadata).length > 0) return deploymentManifestMetadata;
  return {
    ...recordObject(context.modelProfile),
    ...recordObject(exportRecordMetadata),
  };
}

function classLabels(...records: Array<Record<string, unknown> | undefined>) {
  for (const record of records) {
    for (const key of ["class_labels", "labels", "class_names", "classes"]) {
      const value = recordObject(record)[key];
      if (Array.isArray(value) && value.length > 0) {
        return value.map((item) => String(item));
      }
    }
  }
  return [];
}

function inputImageSize(metadata: Record<string, unknown>, modelProfile?: Record<string, unknown>) {
  const shape = metadata.input_shape;
  if (Array.isArray(shape) && shape.length >= 4) {
    const parsed = Number(shape[shape.length - 1]);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  const profile = recordObject(modelProfile);
  const imageSize = Number(metadata.image_size ?? profile.image_size ?? profile.input_size ?? 224);
  return Number.isFinite(imageSize) && imageSize > 0 ? imageSize : 224;
}

function resizeStrategy(metadata: Record<string, unknown>) {
  const preprocessing = recordObject(metadata.preprocessing);
  return String(preprocessing.resize_strategy || preprocessing.crop_strategy || "squash").toLowerCase();
}

function normalizationValues(metadata: Record<string, unknown>) {
  const preprocessing = recordObject(metadata.preprocessing);
  const normalization = String(preprocessing.normalization || "imagenet").toLowerCase();
  if (normalization === "none") return null;
  if (normalization === "dataset") {
    const normalizationMetadata = recordObject(preprocessing.normalization_metadata);
    const mean = threeNumbers(normalizationMetadata.mean);
    const std = threeNumbers(normalizationMetadata.std, true);
    if (mean && std) return { mean, std };
  }
  return { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
}

async function imageTensor(source: string, runtime: ChampionLocalRuntime) {
  const image = await loadImage(source);
  const canvas = document.createElement("canvas");
  canvas.width = runtime.imageSize;
  canvas.height = runtime.imageSize;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas 2D context is unavailable.");
  context.fillStyle = "rgb(0, 0, 0)";
  context.fillRect(0, 0, runtime.imageSize, runtime.imageSize);
  drawImageForRuntime(context, image, runtime);
  const { data } = context.getImageData(0, 0, runtime.imageSize, runtime.imageSize);
  const tensor = new Float32Array(3 * runtime.imageSize * runtime.imageSize);
  const normalization = runtime.normalization;
  for (let y = 0; y < runtime.imageSize; y += 1) {
    for (let x = 0; x < runtime.imageSize; x += 1) {
      const pixelIndex = (y * runtime.imageSize + x) * 4;
      const targetIndex = y * runtime.imageSize + x;
      const red = data[pixelIndex] / 255;
      const green = data[pixelIndex + 1] / 255;
      const blue = data[pixelIndex + 2] / 255;
      tensor[targetIndex] = normalizeChannel(red, normalization, 0);
      tensor[runtime.imageSize * runtime.imageSize + targetIndex] = normalizeChannel(green, normalization, 1);
      tensor[2 * runtime.imageSize * runtime.imageSize + targetIndex] = normalizeChannel(blue, normalization, 2);
    }
  }
  return tensor;
}

function drawImageForRuntime(context: CanvasRenderingContext2D, image: HTMLImageElement, runtime: ChampionLocalRuntime) {
  const target = runtime.imageSize;
  if (runtime.resizeStrategy === "preserve_aspect_pad") {
    const scale = Math.min(target / image.naturalWidth, target / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    context.drawImage(image, (target - width) / 2, (target - height) / 2, width, height);
    return;
  }
  if (runtime.resizeStrategy === "center_crop" || runtime.resizeStrategy === "random_resized_crop") {
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    context.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side, 0, 0, target, target);
    return;
  }
  context.drawImage(image, 0, 0, target, target);
}

function loadImage(source: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to load image for local prediction."));
    image.src = source;
  });
}

function normalizeChannel(value: number, normalization: ChampionLocalRuntime["normalization"], channel: number) {
  if (!normalization) return value;
  return (value - normalization.mean[channel]) / normalization.std[channel];
}

function softmax(values: number[]) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / Math.max(sum, Number.EPSILON));
}

function threeNumbers(value: unknown, positive = false) {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const parsed = value.map((item) => Number(item));
  if (parsed.some((item) => !Number.isFinite(item) || (positive && item <= 0))) return null;
  return parsed;
}

function demoImageLabel(image?: ChampionDemoImage | null) {
  return image?.true_label || image?.label || image?.class_name || "";
}

function recordObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
