import * as ort from "onnxruntime-web";
import type { ChampionDemoImage, ChampionDemoPrediction, ChampionDetection, ChampionExport } from "./types";

export type ChampionLocalRuntime = {
  artifactURI: string;
  session: ort.InferenceSession;
  metadata: Record<string, unknown>;
  labels: string[];
  imageSize: number;
  normalization: { mean: number[]; std: number[] } | null;
  resizeStrategy: string;
  cropStrategy: string;
  modelKind: string;
  taskType: string;
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
};

export class LocalInferenceUnsafeError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "LocalInferenceUnsafeError";
    this.code = code;
  }
}

export function isLocalInferenceUnsafeError(error: unknown): error is LocalInferenceUnsafeError {
  return error instanceof LocalInferenceUnsafeError;
}

export type ChampionLocalRuntimeContext = {
  exportRecord: ChampionExport;
  deploymentProfile?: Record<string, unknown>;
  modelProfile?: Record<string, unknown>;
};

type ChampionModelArtifact = {
  artifact_uri: string;
  bytes: ArrayBuffer;
  external_data?: Array<{
    path: string;
    bytes: ArrayBuffer;
  }>;
};

export type ChampionPredictionOptions = {
  confidenceThreshold?: number;
  iouThreshold?: number;
  maxDetections?: number;
};

type ImageTensorResult = {
  tensor: Float32Array;
  geometry: ImageGeometry;
};

type ImageGeometry = {
  inputWidth: number;
  inputHeight: number;
  naturalWidth: number;
  naturalHeight: number;
  scaleX: number;
  scaleY: number;
  padX: number;
  padY: number;
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
  artifact: ChampionModelArtifact,
  context: ChampionLocalRuntimeContext,
): Promise<ChampionLocalRuntime> {
  ort.env.wasm.numThreads = 1;
  const externalData = (artifact.external_data ?? [])
    .filter((item) => item.path && item.bytes)
    .map((item) => ({ path: item.path, data: item.bytes }));
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    ...(externalData.length > 0 ? { externalData } : {}),
  };
  const session = await ort.InferenceSession.create(artifact.bytes, sessionOptions);
  const metadata = exportMetadata(context);
  const labels = classLabels(metadata, context.modelProfile, context.deploymentProfile);
  if (labels.length === 0) {
    throw new LocalInferenceUnsafeError("CLASS_LABELS_UNAVAILABLE", "The ONNX export manifest does not include class labels.");
  }
  if (new Set(labels).size !== labels.length) {
    throw new LocalInferenceUnsafeError("CLASS_LABEL_ORDER_INVALID", "The ONNX export manifest contains duplicate class labels.");
  }
  const imageSize = inputImageSize(metadata, context.modelProfile);
  const thresholds = detectionThresholds(metadata);
  return {
    artifactURI: artifact.artifact_uri,
    session,
    metadata,
    labels,
    imageSize,
    normalization: normalizationValues(metadata),
    resizeStrategy: resizeStrategy(metadata),
    cropStrategy: cropStrategy(metadata),
    modelKind: modelKind(metadata, context.modelProfile),
    taskType: taskType(metadata, context.modelProfile),
    confidenceThreshold: thresholds.confidenceThreshold,
    iouThreshold: thresholds.iouThreshold,
    maxDetections: thresholds.maxDetections,
  };
}

export async function predictChampionImage(
  runtime: ChampionLocalRuntime,
  image: ChampionDemoImage,
  imageSource: string,
  options: ChampionPredictionOptions = {},
): Promise<ChampionDemoPrediction> {
  assertLocalInferenceParitySafe(runtime, image, imageSource);
  const started = performance.now();
  const input = await imageTensor(imageSource, runtime);
  const inputName = runtime.session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = {
    [inputName]: new ort.Tensor("float32", input.tensor, [1, 3, runtime.imageSize, runtime.imageSize]),
  };
  const inferenceStarted = performance.now();
  const outputs = await runtime.session.run(feeds);
  const inferenceMs = performance.now() - inferenceStarted;
  const output = outputs[runtime.session.outputNames[0]] ?? Object.values(outputs)[0];
  if (isDetectionRuntime(runtime)) {
    const postprocessStarted = performance.now();
    const thresholds = {
      confidenceThreshold: clamp01(options.confidenceThreshold ?? runtime.confidenceThreshold),
      iouThreshold: clamp01(options.iouThreshold ?? runtime.iouThreshold),
      maxDetections: positiveInt(options.maxDetections, runtime.maxDetections),
    };
    const detections = decodeDetections(outputs, runtime, input.geometry, thresholds);
    const postprocessMs = performance.now() - postprocessStarted;
    const predicted = detections[0];
    const trueLabel = demoImageLabel(image);
    const topK = detections.slice(0, 5).map((detection) => ({
      label: detection.label || detection.class_name || `class_${detection.class_id ?? 0}`,
      confidence: detection.confidence ?? detection.score ?? 0,
    }));
    return {
      id: `local-${Date.now()}`,
      image_id: image.image_id || image.id || "",
      image_uri: imageSource,
      status: "SUCCEEDED",
      predicted_label: predicted?.label || predicted?.class_name || "",
      true_label: trueLabel,
      confidence: predicted?.confidence ?? predicted?.score ?? 0,
      latency_ms: performance.now() - started,
      correct: trueLabel ? detections.some((detection) => (detection.label || detection.class_name) === trueLabel) : undefined,
      top_k: topK,
      detections,
      detection_count: detections.length,
      postprocess_latency_ms: postprocessMs,
      metadata: {
        runtime: "onnxruntime-web",
        artifact_uri: runtime.artifactURI,
        thumbnail_uri: imageSource,
        image_source_kind: localImageSourceKind(image, imageSource),
        parity_status: "ok",
        preprocessing_contract_applied: true,
        task_type: "object_detection",
        detections,
        detection_count: detections.length,
        confidence_threshold: thresholds.confidenceThreshold,
        iou_threshold: thresholds.iouThreshold,
        max_detections: thresholds.maxDetections,
        latency_breakdown_ms: {
          inference: inferenceMs,
          postprocess: postprocessMs,
        },
      },
    };
  }
  const logits = Array.from(output.data as Iterable<number>);
  if (runtime.labels.length > 0 && logits.length !== runtime.labels.length) {
    throw new LocalInferenceUnsafeError(
      "LABEL_MAP_OUTPUT_MISMATCH",
      `ONNX output class count ${logits.length} does not match export class_labels count ${runtime.labels.length}.`,
    );
  }
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
      image_source_kind: localImageSourceKind(image, imageSource),
      parity_status: "ok",
      preprocessing_contract_applied: true,
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
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return Object.entries(value as Record<string, unknown>)
          .sort(([left], [right]) => labelMapSortKey(left) - labelMapSortKey(right))
          .map(([, label]) => String(label))
          .filter(Boolean);
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
  const preprocessing = preprocessingConfig(metadata);
  return String(preprocessing.resize_strategy || "squash").toLowerCase();
}

function cropStrategy(metadata: Record<string, unknown>) {
  const preprocessing = preprocessingConfig(metadata);
  return String(preprocessing.crop_strategy || "").toLowerCase();
}

function normalizationValues(metadata: Record<string, unknown>) {
  const preprocessing = preprocessingConfig(metadata);
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
  const geometry = drawImageForRuntime(context, image, runtime);
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
  return { tensor, geometry } satisfies ImageTensorResult;
}

function drawImageForRuntime(context: CanvasRenderingContext2D, image: HTMLImageElement, runtime: ChampionLocalRuntime): ImageGeometry {
  const target = runtime.imageSize;
  const naturalWidth = Math.max(1, image.naturalWidth || image.width || target);
  const naturalHeight = Math.max(1, image.naturalHeight || image.height || target);
  if (runtime.resizeStrategy === "preserve_aspect_pad" || runtime.resizeStrategy === "letterbox" || runtime.resizeStrategy === "yolo_letterbox") {
    const scale = Math.min(target / naturalWidth, target / naturalHeight);
    const width = naturalWidth * scale;
    const height = naturalHeight * scale;
    const padX = (target - width) / 2;
    const padY = (target - height) / 2;
    context.drawImage(image, padX, padY, width, height);
    return {
      inputWidth: target,
      inputHeight: target,
      naturalWidth,
      naturalHeight,
      scaleX: scale,
      scaleY: scale,
      padX,
      padY,
    };
  }
  if (runtime.resizeStrategy === "center_crop" || runtime.cropStrategy === "center_crop") {
    const resizeSize = Math.max(target, Math.round(target * 1.15));
    const resized = document.createElement("canvas");
    resized.width = resizeSize;
    resized.height = resizeSize;
    const resizedContext = resized.getContext("2d");
    if (!resizedContext) throw new Error("Canvas 2D context is unavailable.");
    resizedContext.drawImage(image, 0, 0, resizeSize, resizeSize);
    const cropOffset = Math.max(0, (resizeSize - target) / 2);
    context.drawImage(resized, cropOffset, cropOffset, target, target, 0, 0, target, target);
    return {
      inputWidth: target,
      inputHeight: target,
      naturalWidth,
      naturalHeight,
      scaleX: resizeSize / naturalWidth,
      scaleY: resizeSize / naturalHeight,
      padX: -cropOffset,
      padY: -cropOffset,
    };
  }
  context.drawImage(image, 0, 0, target, target);
  return {
    inputWidth: target,
    inputHeight: target,
    naturalWidth,
    naturalHeight,
    scaleX: target / naturalWidth,
    scaleY: target / naturalHeight,
    padX: 0,
    padY: 0,
  };
}

function preprocessingConfig(metadata: Record<string, unknown>) {
  const direct = recordObject(metadata.preprocessing);
  const contract = recordObject(metadata.inference_contract);
  const contractPreprocessing = recordObject(contract.preprocessing);
  const contractConfig = recordObject(contractPreprocessing.config);
  return {
    ...direct,
    ...contractConfig,
  };
}

function assertLocalInferenceParitySafe(runtime: ChampionLocalRuntime, image: ChampionDemoImage, imageSource: string) {
  const metadata = recordObject(image.metadata);
  const sourceType = String(metadata.demo_source_type || metadata.source || "").toLowerCase();
  if (bboxCropRequested(runtime)) {
    throw new LocalInferenceUnsafeError(
      "LOCAL_PREPROCESSING_UNSUPPORTED",
      "Local browser inference cannot apply bbox-crop preprocessing; backend demo inference is required for parity.",
    );
  }
  if (metadata.parity_safe === false || String(metadata.parity_status || "").toLowerCase() === "unsafe") {
    throw new LocalInferenceUnsafeError(
      "DEMO_IMAGE_NOT_PARITY_SAFE",
      String(metadata.parity_failure_reason || "The selected held-out image is a thumbnail or non-parity-safe derivative."),
    );
  }
  if (sourceType === "heldout_test" && metadata.parity_safe !== true) {
    throw new LocalInferenceUnsafeError(
      "HELDOUT_IMAGE_SOURCE_UNVERIFIED",
      "Held-out demo metadata does not prove that local inference is using original image bytes.",
    );
  }
  if (sourceType.includes("thumbnail")) {
    throw new LocalInferenceUnsafeError(
      "THUMBNAIL_INFERENCE_UNSAFE",
      "The selected held-out image is marked as a thumbnail, so local browser inference would not match validation/test.",
    );
  }
  const originalURI = image.uri || image.image_uri || "";
  if (image.thumbnail_uri && imageSource === image.thumbnail_uri && originalURI && originalURI !== image.thumbnail_uri) {
    throw new LocalInferenceUnsafeError(
      "THUMBNAIL_INFERENCE_UNSAFE",
      "Local browser inference was about to use a display thumbnail instead of the original held-out image.",
    );
  }
  if (image.thumbnail_uri && imageSource === image.thumbnail_uri && sourceType.includes("heldout_test")) {
    throw new LocalInferenceUnsafeError(
      "THUMBNAIL_INFERENCE_UNSAFE",
      "Held-out demo inference requires original image bytes, not the display thumbnail.",
    );
  }
}

function bboxCropRequested(runtime: ChampionLocalRuntime) {
  const preprocessing = preprocessingConfig(runtime.metadata);
  const resize = String(preprocessing.resize_strategy || runtime.resizeStrategy || "").toLowerCase();
  const crop = String(preprocessing.crop_strategy || runtime.cropStrategy || "").toLowerCase();
  const bboxMode = String(preprocessing.bbox_mode || "").toLowerCase();
  return (
    resize === "bbox_crop_if_available" ||
    crop === "bbox_crop_if_available" ||
    crop === "bbox_crop_ablation" ||
    bboxMode === "crop_if_available" ||
    bboxMode === "crop_and_compare_full_image"
  );
}

function localImageSourceKind(image: ChampionDemoImage, imageSource: string) {
  if (imageSource === image.thumbnail_uri) return "thumbnail";
  if (imageSource === image.preview_uri) return "preview";
  if (imageSource === image.uri || imageSource === image.image_uri) return "original_or_worker_uri";
  if (imageSource.startsWith("data:image/")) return "inline_image";
  return "custom";
}

function labelMapSortKey(value: string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
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

function decodeDetections(
  outputs: ort.InferenceSession.ReturnType,
  runtime: ChampionLocalRuntime,
  geometry: ImageGeometry,
  thresholds: { confidenceThreshold: number; iouThreshold: number; maxDetections: number },
): ChampionDetection[] {
  const named = detectionsFromNamedOutputs(outputs, runtime, geometry, thresholds);
  if (named.length > 0) return named;
  const rows = yoloRowsFromOutputs(outputs, runtime.labels.length);
  const detections = rows
    .map((row) => detectionFromYoloRow(row, runtime, geometry, thresholds.confidenceThreshold))
    .filter((item): item is ChampionDetection => Boolean(item));
  return classAwareNMS(detections, thresholds.iouThreshold, thresholds.maxDetections);
}

function detectionsFromNamedOutputs(
  outputs: ort.InferenceSession.ReturnType,
  runtime: ChampionLocalRuntime,
  geometry: ImageGeometry,
  thresholds: { confidenceThreshold: number; iouThreshold: number; maxDetections: number },
) {
  const outputMap = Object.fromEntries(Object.entries(outputs).map(([key, value]) => [key.toLowerCase(), value]));
  const boxes = outputMap.boxes || outputMap.output_boxes;
  const scores = outputMap.scores || outputMap.output_scores;
  const classes = outputMap.classes || outputMap.class_ids || outputMap.labels;
  if (!boxes || !scores || !classes) return [];
  const boxValues = Array.from(boxes.data as Iterable<number>);
  const scoreValues = Array.from(scores.data as Iterable<number>);
  const classValues = Array.from(classes.data as Iterable<number>);
  const detections: ChampionDetection[] = [];
  const count = Math.min(Math.floor(boxValues.length / 4), scoreValues.length, classValues.length);
  for (let index = 0; index < count; index += 1) {
    const confidence = scoreValue(scoreValues[index]);
    if (confidence < thresholds.confidenceThreshold) continue;
    const classId = safeClassId(classValues[index], runtime.labels.length);
    const box = boxFromXYXY(boxValues.slice(index * 4, index * 4 + 4), geometry);
    if (!box) continue;
    detections.push(detectionRecord(classId, confidence, box, runtime.labels));
  }
  return classAwareNMS(detections, thresholds.iouThreshold, thresholds.maxDetections);
}

function yoloRowsFromOutputs(outputs: ort.InferenceSession.ReturnType, classCount: number) {
  for (const tensor of Object.values(outputs)) {
    const dims = tensor.dims;
    const values = Array.from(tensor.data as Iterable<number>);
    if (values.length === 0 || dims.length < 2) continue;
    let rows = dims[dims.length - 2] ?? 0;
    let features = dims[dims.length - 1] ?? 0;
    if (dims.length === 3 && dims[0] === 1) {
      rows = dims[1] ?? rows;
      features = dims[2] ?? features;
    }
    if (rows <= 0 || features <= 0 || rows * features > values.length) continue;
    const expectedFeatures = Math.max(6, classCount + 4);
    const transpose = (rows <= expectedFeatures + 8 && features > rows) || (features < expectedFeatures && rows >= expectedFeatures);
    const out: number[][] = [];
    if (transpose) {
      for (let anchor = 0; anchor < features; anchor += 1) {
        const row: number[] = [];
        for (let feature = 0; feature < rows; feature += 1) {
          row.push(values[feature * features + anchor]);
        }
        out.push(row);
      }
    } else {
      for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
        out.push(values.slice(rowIndex * features, (rowIndex + 1) * features));
      }
    }
    return out;
  }
  return [];
}

function detectionFromYoloRow(
  row: number[],
  runtime: ChampionLocalRuntime,
  geometry: ImageGeometry,
  confidenceThreshold: number,
) {
  if (row.length < 6) return null;
  const classCount = runtime.labels.length;
  let classId = 0;
  let confidence = 0;
  let box: { x: number; y: number; width: number; height: number } | null = null;
  if (classCount > 0 && row.length >= 4 + classCount) {
    let objectness = 1;
    let scores = row.slice(4, 4 + classCount);
    if (row.length === 5 + classCount) {
      objectness = scoreValue(row[4]);
      scores = row.slice(5, 5 + classCount);
    }
    const scored = scores.map((score) => scoreValue(score) * objectness);
    classId = scored.reduce((bestIndex, score, index) => (score > scored[bestIndex] ? index : bestIndex), 0);
    confidence = scored[classId] ?? 0;
    box = boxFromCenterXYWH(row.slice(0, 4), geometry);
  } else {
    confidence = scoreValue(row[4]);
    classId = safeClassId(row[5], classCount);
    box = boxFromXYXY(row.slice(0, 4), geometry);
  }
  if (confidence < confidenceThreshold || !box) return null;
  return detectionRecord(classId, confidence, box, runtime.labels);
}

function boxFromCenterXYWH(values: number[], geometry: ImageGeometry) {
  let [xCenter, yCenter, width, height] = values;
  if (Math.max(Math.abs(xCenter), Math.abs(yCenter), Math.abs(width), Math.abs(height)) <= 2) {
    xCenter *= geometry.inputWidth;
    width *= geometry.inputWidth;
    yCenter *= geometry.inputHeight;
    height *= geometry.inputHeight;
  }
  return boxFromModelXYXY(
    xCenter - width / 2,
    yCenter - height / 2,
    xCenter + width / 2,
    yCenter + height / 2,
    geometry,
  );
}

function boxFromXYXY(values: number[], geometry: ImageGeometry) {
  const [x1, y1, x2, y2] = values;
  const maxCoord = Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2));
  if (maxCoord <= 2) {
    return normalizedBoxFromOriginalXYXY(
      x1 * geometry.naturalWidth,
      y1 * geometry.naturalHeight,
      x2 * geometry.naturalWidth,
      y2 * geometry.naturalHeight,
      geometry,
    );
  }
  if (maxCoord <= Math.max(geometry.inputWidth, geometry.inputHeight) * 1.25) {
    return boxFromModelXYXY(x1, y1, x2, y2, geometry);
  }
  return normalizedBoxFromOriginalXYXY(x1, y1, x2, y2, geometry);
}

function boxFromModelXYXY(x1: number, y1: number, x2: number, y2: number, geometry: ImageGeometry) {
  return normalizedBoxFromOriginalXYXY(
    (x1 - geometry.padX) / Math.max(geometry.scaleX, Number.EPSILON),
    (y1 - geometry.padY) / Math.max(geometry.scaleY, Number.EPSILON),
    (x2 - geometry.padX) / Math.max(geometry.scaleX, Number.EPSILON),
    (y2 - geometry.padY) / Math.max(geometry.scaleY, Number.EPSILON),
    geometry,
  );
}

function normalizedBoxFromOriginalXYXY(
  x1Value: number,
  y1Value: number,
  x2Value: number,
  y2Value: number,
  geometry: ImageGeometry,
) {
  let x1 = Math.min(x1Value, x2Value);
  let y1 = Math.min(y1Value, y2Value);
  let x2 = Math.max(x1Value, x2Value);
  let y2 = Math.max(y1Value, y2Value);
  x1 = clamp(x1, 0, geometry.naturalWidth);
  y1 = clamp(y1, 0, geometry.naturalHeight);
  x2 = clamp(x2, 0, geometry.naturalWidth);
  y2 = clamp(y2, 0, geometry.naturalHeight);
  const width = x2 - x1;
  const height = y2 - y1;
  if (width <= 0 || height <= 0) return null;
  return {
    x: clamp01(x1 / geometry.naturalWidth),
    y: clamp01(y1 / geometry.naturalHeight),
    width: clamp01(width / geometry.naturalWidth),
    height: clamp01(height / geometry.naturalHeight),
  };
}

function classAwareNMS(detections: ChampionDetection[], iouThreshold: number, maxDetections: number) {
  const selected: ChampionDetection[] = [];
  for (const detection of [...detections].sort((left, right) => confidenceOf(right) - confidenceOf(left))) {
    if (selected.length >= maxDetections) break;
    const classId = detection.class_id ?? -1;
    const overlaps = selected.some((kept) => (kept.class_id ?? -1) === classId && boxIOU(detection, kept) > iouThreshold);
    if (!overlaps) selected.push(detection);
  }
  return selected;
}

function detectionRecord(
  classId: number,
  confidence: number,
  box: { x: number; y: number; width: number; height: number },
  labels: string[],
): ChampionDetection {
  const label = labels[classId] || `class_${classId}`;
  const score = clamp01(confidence);
  return {
    label,
    class_name: label,
    class_id: classId,
    confidence: score,
    score,
    box,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    x1: box.x,
    y1: box.y,
    x2: box.x + box.width,
    y2: box.y + box.height,
  };
}

function boxIOU(left: ChampionDetection, right: ChampionDetection) {
  const leftBox = normalizedDetectionBox(left);
  const rightBox = normalizedDetectionBox(right);
  if (!leftBox || !rightBox) return 0;
  const leftX2 = leftBox.x + leftBox.width;
  const leftY2 = leftBox.y + leftBox.height;
  const rightX2 = rightBox.x + rightBox.width;
  const rightY2 = rightBox.y + rightBox.height;
  const intersectionWidth = Math.max(0, Math.min(leftX2, rightX2) - Math.max(leftBox.x, rightBox.x));
  const intersectionHeight = Math.max(0, Math.min(leftY2, rightY2) - Math.max(leftBox.y, rightBox.y));
  const intersection = intersectionWidth * intersectionHeight;
  const union = leftBox.width * leftBox.height + rightBox.width * rightBox.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function normalizedDetectionBox(detection: ChampionDetection) {
  const box = recordObject(detection.box);
  const x = Number(box.x ?? detection.x ?? detection.x1);
  const y = Number(box.y ?? detection.y ?? detection.y1);
  const width = Number(box.width ?? detection.width ?? (Number(detection.x2) - Number(detection.x1)));
  const height = Number(box.height ?? detection.height ?? (Number(detection.y2) - Number(detection.y1)));
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
  return {
    x: clamp01(x),
    y: clamp01(y),
    width: clamp(width, 0, 1 - clamp01(x)),
    height: clamp(height, 0, 1 - clamp01(y)),
  };
}

function isDetectionRuntime(runtime: ChampionLocalRuntime) {
  return runtime.modelKind === "detection" || runtime.taskType === "object_detection";
}

function modelKind(...records: Array<Record<string, unknown> | undefined>) {
  for (const record of records) {
    const value = String(recordObject(record).model_kind || "").toLowerCase();
    if (value.includes("detect") || value.includes("yolo")) return "detection";
    if (value.includes("class")) return "classification";
  }
  return "classification";
}

function taskType(...records: Array<Record<string, unknown> | undefined>) {
  for (const record of records) {
    const value = String(recordObject(record).task_type || "").toLowerCase();
    if (value.includes("object_detection") || value.includes("detect") || value.includes("yolo")) return "object_detection";
    if (value.includes("class")) return "image_classification";
  }
  return "image_classification";
}

function detectionThresholds(metadata: Record<string, unknown>) {
  const defaults = recordObject(metadata.confidence_threshold_defaults);
  const detection = recordObject(defaults.detection);
  const postprocessing = recordObject(metadata.postprocessing_contract);
  const nms = recordObject(postprocessing.nms);
  return {
    confidenceThreshold: clamp01(Number(detection.confidence_threshold ?? nms.confidence_threshold ?? postprocessing.confidence_threshold ?? 0.25)),
    iouThreshold: clamp01(Number(detection.iou_threshold ?? nms.iou_threshold ?? postprocessing.iou_threshold ?? 0.7)),
    maxDetections: positiveInt(detection.max_detections ?? nms.max_detections, 300),
  };
}

function scoreValue(value: number) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0 || value > 1) return 1 / (1 + Math.exp(-value));
  return value;
}

function safeClassId(value: number, classCount: number) {
  const parsed = Number.isFinite(value) ? Math.round(value) : 0;
  if (classCount <= 0) return parsed;
  return Math.max(0, Math.min(classCount - 1, parsed));
}

function confidenceOf(detection: ChampionDetection) {
  return Number(detection.confidence ?? detection.score ?? 0) || 0;
}

function clamp01(value: number) {
  return clamp(Number.isFinite(value) ? value : 0, 0, 1);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function positiveInt(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
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
