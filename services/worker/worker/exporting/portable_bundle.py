from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

PORTABLE_BUNDLE_SCHEMA_VERSION = "portable_inference_bundle_v1"
PORTABLE_MANIFEST_SCHEMA_VERSION = "portable_inference_manifest_v1"
PORTABLE_ARTIFACT_FORMAT = "portable_inference_bundle"


def build_portable_inference_manifest(metadata: dict, model_artifact: dict, manifest: dict) -> dict:
    """Build the runtime-neutral contract shipped inside a portable ONNX bundle."""
    export_metadata = metadata if isinstance(metadata, dict) else {}
    artifact = model_artifact if isinstance(model_artifact, dict) else {}
    return {
        "schema_version": PORTABLE_MANIFEST_SCHEMA_VERSION,
        "model_artifact": "model.onnx",
        "artifact_format": "onnx",
        "external_data": _portable_external_data(artifact),
        "model_kind": str(export_metadata.get("model_kind") or "classification"),
        "task_type": str(export_metadata.get("task_type") or "image_classification"),
        "runtime": "onnx",
        "input_shape": _list_value(export_metadata.get("input_shape")),
        "class_labels": [str(label) for label in _list_value(export_metadata.get("class_labels"))],
        "class_label_order_hash": str(export_metadata.get("class_label_order_hash") or ""),
        "inference_contract": _dict_value(export_metadata.get("inference_contract")),
        "preprocessing_contract": _dict_value(export_metadata.get("preprocessing_contract")),
        "postprocessing_contract": _dict_value(export_metadata.get("postprocessing_contract")),
        "confidence_threshold_defaults": _dict_value(export_metadata.get("confidence_threshold_defaults")),
        "parity": _parity_summary(export_metadata.get("export_self_test")),
        "source_export_manifest": {
            "schema_version": str(manifest.get("schema_version") or ""),
            "status": str(manifest.get("status") or ""),
        },
    }


def write_portable_inference_bundle(
    export_dir: Path,
    manifest: dict,
    requested_format: str,
    provenance: dict | None = None,
) -> dict:
    """Write a single-file portable ONNX inference bundle next to the export manifest."""
    if str(requested_format).lower() != "onnx":
        return _skipped_artifact(
            "PORTABLE_BUNDLE_UNSUPPORTED_FORMAT",
            "Portable inference bundles are only produced for ONNX exports.",
            provenance=provenance,
        )

    model_artifact = _created_onnx_artifact(manifest)
    if model_artifact is None:
        return _skipped_artifact(
            "PORTABLE_BUNDLE_MODEL_UNAVAILABLE",
            "No created ONNX artifact is available for the portable inference bundle.",
            provenance=provenance,
        )

    model_path = Path(str(model_artifact.get("path") or ""))
    if not model_path.exists() or not model_path.is_file():
        return _skipped_artifact(
            "PORTABLE_BUNDLE_MODEL_UNAVAILABLE",
            "The created ONNX artifact is not available on the worker filesystem.",
            provenance=provenance,
        )

    export_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = export_dir / "portable_inference_bundle.zip"
    metadata = _dict_value(manifest.get("metadata"))
    sidecars = _sidecar_records(model_path, model_artifact)
    portable_model_artifact = {**model_artifact, "external_data": sidecars}
    portable_manifest = build_portable_inference_manifest(metadata, portable_model_artifact, manifest)
    expected_outputs = _expected_outputs(metadata.get("export_self_test"))
    static_files = {
        "manifest.json": json.dumps(portable_manifest, indent=2, sort_keys=True),
        "README.md": _readme_text(portable_manifest),
        "requirements.txt": _requirements_text(),
        "examples/python_onnxruntime.py": _python_example_text(),
        "examples/node_onnxruntime.mjs": _node_example_text(),
        "parity/expected_outputs.json": json.dumps(expected_outputs, indent=2, sort_keys=True),
    }
    contents = ["model.onnx"]

    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, "model.onnx")
            for sidecar in sidecars:
                source_path = Path(str(sidecar.get("artifact_path") or ""))
                archive_path = str(sidecar.get("path") or "")
                if source_path.exists() and source_path.is_file() and archive_path:
                    archive.write(source_path, archive_path)
                    contents.append(archive_path)
            for archive_path, text in static_files.items():
                archive.writestr(archive_path, text)
                contents.append(archive_path)
    except Exception as exc:
        return _failed_artifact(bundle_path, exc, provenance=provenance)

    bytes_count = bundle_path.stat().st_size if bundle_path.exists() else 0
    return {
        "format": PORTABLE_ARTIFACT_FORMAT,
        "status": "created",
        "path": str(bundle_path),
        "bytes": bytes_count,
        "provenance": _artifact_provenance(provenance, bytes_count, validation_errors=[]),
        "contents": contents,
    }


def portable_bundle_summary(artifact: dict) -> dict:
    summary = {
        "schema_version": PORTABLE_BUNDLE_SCHEMA_VERSION,
        "status": str(artifact.get("status") or "skipped"),
        "artifact_path": str(artifact.get("path") or ""),
        "artifact_uri": _file_uri(artifact.get("path")),
        "contents": _list_value(artifact.get("contents")),
    }
    for key in ("bytes", "error_code", "error"):
        if artifact.get(key) not in (None, ""):
            summary[key] = artifact[key]
    return summary


def _created_onnx_artifact(manifest: dict) -> dict | None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    for item in artifacts:
        artifact = item if isinstance(item, dict) else {}
        if (
            str(artifact.get("format") or "").lower() == "onnx"
            and str(artifact.get("status") or "").lower() == "created"
            and artifact.get("path")
        ):
            return artifact
    return None


def _sidecar_records(model_path: Path, model_artifact: dict) -> list[dict]:
    model_dir = model_path.resolve().parent
    sidecars = []
    for item in _list_value(model_artifact.get("external_data")):
        record = item if isinstance(item, dict) else {}
        relative_path = _safe_relative_path(str(record.get("path") or ""))
        if not relative_path:
            continue
        source_value = str(record.get("artifact_path") or "").strip()
        source_path = Path(source_value) if source_value else model_dir / relative_path
        if not source_path.is_absolute():
            source_path = model_dir / source_path
        try:
            resolved = source_path.resolve()
            resolved.relative_to(model_dir)
        except (OSError, RuntimeError, ValueError):
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        sidecars.append({"path": relative_path, "artifact_path": str(resolved), "bytes": resolved.stat().st_size})
    return sidecars


def _portable_external_data(model_artifact: dict) -> list[dict]:
    out = []
    for item in _list_value(model_artifact.get("external_data")):
        record = item if isinstance(item, dict) else {}
        path = _safe_relative_path(str(record.get("path") or ""))
        if not path:
            continue
        out.append({"path": path, "bytes": _int_value(record.get("bytes"))})
    return out


def _expected_outputs(self_test: object) -> dict:
    payload = self_test if isinstance(self_test, dict) else {}
    samples = _list_value(payload.get("samples"))
    return {
        "schema_version": "portable_expected_outputs_v1",
        "status": str(payload.get("status") or "not_available"),
        "runtime": str(payload.get("runtime") or "onnx"),
        "sample_count": _int_value(payload.get("sample_count", len(samples))),
        "class_label_order_hash": str(payload.get("class_label_order_hash") or ""),
        "preprocessing_contract_hash": str(payload.get("preprocessing_contract_hash") or ""),
        "tolerance": _dict_value(payload.get("tolerance")),
        "samples": samples,
    }


def _parity_summary(self_test: object) -> dict:
    payload = self_test if isinstance(self_test, dict) else {}
    return {
        "schema_version": "portable_parity_summary_v1",
        "status": str(payload.get("status") or "not_available"),
        "export_verified": bool(payload.get("export_verified")),
        "parity_status": str(payload.get("parity_status") or ""),
        "sample_count": _int_value(payload.get("sample_count")),
        "passed_count": _int_value(payload.get("passed_count")),
        "failed_inference_count": _int_value(payload.get("failed_inference_count")),
        "expected_outputs": "parity/expected_outputs.json",
    }


def _skipped_artifact(error_code: str, error: str, *, provenance: dict | None = None) -> dict:
    return {
        "format": PORTABLE_ARTIFACT_FORMAT,
        "status": "skipped",
        "error_code": error_code,
        "error": error,
        "provenance": _artifact_provenance(provenance, 0, validation_errors=[error]),
        "contents": [],
    }


def _failed_artifact(path: Path, exc: Exception, *, provenance: dict | None = None) -> dict:
    return {
        "format": PORTABLE_ARTIFACT_FORMAT,
        "status": "failed",
        "path": str(path),
        "error_code": "PORTABLE_BUNDLE_FAILED",
        "error": str(exc),
        "provenance": _artifact_provenance(provenance, 0, validation_errors=[str(exc)]),
        "contents": [],
    }


def _artifact_provenance(
    provenance: dict | None,
    artifact_bytes: int,
    *,
    validation_errors: Iterable[str],
) -> dict:
    source_data = provenance if isinstance(provenance, dict) else {}
    record = {
        "schema_version": "worker_artifact_provenance_v1",
        "generated_by": "model-express-worker",
        "source": "worker_generated",
        "artifact_format": PORTABLE_ARTIFACT_FORMAT,
        "artifact_bytes": int(artifact_bytes),
        "validation_errors": [str(error) for error in validation_errors if str(error)],
    }
    for key in ("source_job_id", "source_export_id", "export_job_id"):
        value = source_data.get(key)
        if value:
            record[key] = str(value)
    return record


def _readme_text(portable_manifest: dict) -> str:
    labels = portable_manifest.get("class_labels") if isinstance(portable_manifest.get("class_labels"), list) else []
    task_type = str(portable_manifest.get("task_type") or "image_classification")
    return f"""# Portable Inference Bundle

This bundle contains an ONNX champion export plus a runtime-neutral inference contract.

Contents:

- `model.onnx`: model artifact
- `manifest.json`: preprocessing, runtime, postprocessing, labels, and parity contract
- `examples/python_onnxruntime.py`: Python ONNX Runtime reference runner
- `examples/node_onnxruntime.mjs`: Node.js ONNX Runtime reference runner
- `requirements.txt`: Python example dependencies
- `parity/expected_outputs.json`: export self-test outputs when available

No Model Express backend, UI, or `model_express_runtime` package is required to load this model.

Task type: `{task_type}`
Class count: {len(labels)}

Python example:

```bash
python -m pip install -r requirements.txt
python examples/python_onnxruntime.py path/to/image.jpg
```

Node example:

```bash
npm install onnxruntime-node sharp
node examples/node_onnxruntime.mjs path/to/image.jpg
```

The example scripts are reference code. Production runtimes should read `manifest.json` and apply the same preprocessing and postprocessing contract.
"""


def _requirements_text() -> str:
    return "onnxruntime>=1.17\nnumpy>=1.24\nPillow>=10\n"


def _python_example_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def load_manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))


def image_size(manifest: dict) -> int:
    shape = manifest.get("input_shape") or manifest.get("inference_contract", {}).get("input", {}).get("model_tensor_shape")
    if isinstance(shape, list) and len(shape) == 4:
        return int(shape[-1])
    return 224


def preprocessing_config(manifest: dict) -> dict:
    contract = manifest.get("preprocessing_contract") or manifest.get("inference_contract", {}).get("preprocessing", {})
    return dict(contract.get("config") or {})


def prepare_image(image_path: Path, manifest: dict) -> np.ndarray:
    size = image_size(manifest)
    config = preprocessing_config(manifest)
    if "bbox" in str(config.get("crop_strategy", "")) or "bbox" in str(config.get("resize_strategy", "")):
        raise SystemExit("postprocessing implementation needed: bbox preprocessing requires runtime bounding box metadata")
    image = Image.open(image_path).convert("RGB")
    resize_strategy = str(config.get("resize_strategy") or "squash")
    crop_strategy = str(config.get("crop_strategy") or "none")
    if resize_strategy == "preserve_aspect_pad":
        image.thumbnail((size, size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        image = canvas
    elif crop_strategy == "center_crop" or resize_strategy == "center_crop":
        resize_size = int(size * 1.15)
        image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        left = (resize_size - size) // 2
        image = image.crop((left, left, left + size, left + size))
    else:
        image = image.resize((size, size), Image.Resampling.BILINEAR)

    tensor = np.asarray(image, dtype=np.float32) / 255.0
    norm = str(config.get("normalization") or "imagenet")
    if norm != "none":
        if norm == "dataset" and isinstance(config.get("normalization_metadata"), dict):
            meta = config["normalization_metadata"]
            mean = meta.get("mean") or [0.485, 0.456, 0.406]
            std = meta.get("std") or [0.229, 0.224, 0.225]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        tensor = (tensor - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def run(image_path: Path) -> None:
    bundle_dir = Path(__file__).resolve().parents[1]
    manifest = load_manifest(bundle_dir)
    if str(manifest.get("task_type", "")).lower() == "object_detection":
        raise SystemExit("postprocessing implementation needed for this detection contract")
    session = ort.InferenceSession(str(bundle_dir / manifest.get("model_artifact", "model.onnx")))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: prepare_image(image_path, manifest)})
    logits = np.asarray(outputs[0]).reshape(-1)
    labels = [str(label) for label in manifest.get("class_labels", [])]
    probs = softmax(logits)
    order = np.argsort(-probs)[: min(5, len(probs))]
    result = [
        {
            "label": labels[index] if index < len(labels) else str(index),
            "confidence": float(probs[index]),
            "logit": float(logits[index]),
        }
        for index in order
    ]
    print(json.dumps({"top_k": result}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/python_onnxruntime.py path/to/image")
    run(Path(sys.argv[1]))
'''


def _node_example_text() -> str:
    return r'''import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import ort from "onnxruntime-node";
import sharp from "sharp";

const bundleDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function imageSize(manifest) {
  const shape = manifest.input_shape || manifest.inference_contract?.input?.model_tensor_shape;
  return Array.isArray(shape) && shape.length === 4 ? Number(shape[3]) : 224;
}

function preprocessingConfig(manifest) {
  return manifest.preprocessing_contract?.config || manifest.inference_contract?.preprocessing?.config || {};
}

async function prepareImage(imagePath, manifest) {
  const size = imageSize(manifest);
  const config = preprocessingConfig(manifest);
  if (String(config.crop_strategy || "").includes("bbox") || String(config.resize_strategy || "").includes("bbox")) {
    throw new Error("postprocessing implementation needed: bbox preprocessing requires runtime bounding box metadata");
  }
  const fit = config.resize_strategy === "preserve_aspect_pad" ? "contain" : config.crop_strategy === "center_crop" ? "cover" : "fill";
  const { data } = await sharp(imagePath)
    .rotate()
    .resize(size, size, { fit, background: { r: 0, g: 0, b: 0 } })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const hwc = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) hwc[i] = data[i] / 255.0;
  const norm = String(config.normalization || "imagenet");
  const mean = norm === "none" ? null : config.normalization_metadata?.mean || [0.485, 0.456, 0.406];
  const std = norm === "none" ? null : config.normalization_metadata?.std || [0.229, 0.224, 0.225];
  const chw = new Float32Array(1 * 3 * size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        let value = hwc[(y * size + x) * 3 + c];
        if (mean && std) value = (value - mean[c]) / std[c];
        chw[c * size * size + y * size + x] = value;
      }
    }
  }
  return new ort.Tensor("float32", chw, [1, 3, size, size]);
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

async function main() {
  const imagePath = process.argv[2];
  if (!imagePath) throw new Error("usage: node examples/node_onnxruntime.mjs path/to/image");
  const manifest = JSON.parse(await fs.readFile(path.join(bundleDir, "manifest.json"), "utf8"));
  if (String(manifest.task_type || "").toLowerCase() === "object_detection") {
    throw new Error("postprocessing implementation needed for this detection contract");
  }
  const session = await ort.InferenceSession.create(path.join(bundleDir, manifest.model_artifact || "model.onnx"));
  const inputName = session.inputNames[0];
  const feeds = { [inputName]: await prepareImage(imagePath, manifest) };
  const outputs = await session.run(feeds);
  const output = outputs[session.outputNames[0]];
  const logits = Array.from(output.data);
  const probs = softmax(logits);
  const labels = Array.isArray(manifest.class_labels) ? manifest.class_labels : [];
  const topK = probs
    .map((confidence, index) => ({ label: labels[index] || String(index), confidence, logit: logits[index] }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);
  console.log(JSON.stringify({ top_k: topK }, null, 2));
}

main().catch((error) => {
  console.error(error.message || error);
  process.exitCode = 1;
});
'''


def _safe_relative_path(value: str) -> str:
    parts = []
    for part in str(value).replace("\\", "/").split("/"):
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    return "/".join(parts)


def _dict_value(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _list_value(value: object) -> list:
    return value if isinstance(value, list) else []


def _int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _file_uri(path_value: object) -> str:
    if not path_value:
        return ""
    try:
        return Path(str(path_value)).resolve().as_uri()
    except ValueError:
        return ""
