from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

from worker.exporting.preprocessing import image_to_chw_float32_array, prepare_image_for_inference

SELF_TEST_SCHEMA_VERSION = "champion_export_self_test_v1"


def class_label_order_hash(labels: Iterable[str]) -> str:
    payload = "\n".join(str(label) for label in labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocessing_contract_hash(metadata: dict) -> str:
    payload = {
        "input_shape": metadata.get("input_shape"),
        "preprocessing_contract": metadata.get("preprocessing_contract"),
        "inference_contract": metadata.get("inference_contract"),
        "class_labels": metadata.get("class_labels"),
        "class_label_order_hash": metadata.get("class_label_order_hash"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_onnx_pytorch_self_test(
    *,
    model,
    onnx_path: Path | None,
    metadata: dict,
    samples: Iterable[dict] | None,
    tolerance: dict | None = None,
) -> dict:
    labels = [str(label) for label in metadata.get("class_labels", [])]
    tolerances = _tolerance(tolerance)
    diagnostics: list[dict] = []
    base = {
        "schema_version": SELF_TEST_SCHEMA_VERSION,
        "status": "not_run",
        "export_verified": False,
        "parity_status": "not_run",
        "validation_mode": "onnx_vs_pytorch_heldout_eval_tensor",
        "runtime": "onnx",
        "sample_count": 0,
        "passed_count": 0,
        "failed_inference_count": 0,
        "warning_count": 0,
        "tolerance": tolerances,
        "input_shape": _expected_input_shape(metadata),
        "class_label_order_hash": class_label_order_hash(labels),
        "preprocessing_contract_hash": preprocessing_contract_hash(metadata),
        "diagnostics": diagnostics,
        "samples": [],
        "measured_at_unix": int(time.time()),
    }

    if str(metadata.get("model_kind", "classification")).lower() == "detection":
        diagnostics.append(
            _diagnostic(
                "EXPORT_SELF_TEST_UNSUPPORTED_TASK",
                "warning",
                "ONNX-vs-PyTorch export self-test is currently implemented for image classification exports.",
                "runtime_parity",
            )
        )
        return _finalize(base)
    if model is None:
        diagnostics.append(
            _diagnostic(
                "PYTORCH_MODEL_UNAVAILABLE",
                "warning",
                "No PyTorch model instance was available for export-time ONNX parity validation.",
                "runtime_parity",
            )
        )
        return _finalize(base)
    if onnx_path is None or not Path(onnx_path).exists():
        diagnostics.append(
            _diagnostic(
                "ONNX_ARTIFACT_UNAVAILABLE",
                "warning",
                "No created ONNX artifact was available for export-time parity validation.",
                "runtime_parity",
            )
        )
        return _finalize(base)

    normalized_samples = [sample for sample in samples or () if isinstance(sample, dict)]
    if not normalized_samples:
        diagnostics.append(
            _diagnostic(
                "EXPORT_SELF_TEST_SAMPLE_UNAVAILABLE",
                "warning",
                "No held-out eval sample tensor was supplied for export-time parity validation.",
                "heldout_sample",
            )
        )
        return _finalize(base)

    try:
        import torch
    except Exception as exc:
        diagnostics.append(
            _diagnostic("TORCH_UNAVAILABLE", "warning", f"PyTorch is unavailable: {exc}", "runtime_parity")
        )
        return _finalize(base)

    try:
        session = _load_onnx_session(Path(onnx_path))
    except Exception as exc:
        diagnostics.append(
            _diagnostic(
                "ONNX_RUNTIME_UNAVAILABLE",
                "warning",
                f"ONNX Runtime could not load the exported artifact: {exc}",
                "runtime_parity",
            )
        )
        return _finalize(base)

    try:
        model.eval()
        model.cpu()
    except Exception:
        pass

    for index, sample in enumerate(normalized_samples):
        result = _run_sample_self_test(
            model=model,
            session=session,
            torch=torch,
            metadata=metadata,
            labels=labels,
            sample=sample,
            sample_index=index,
            tolerances=tolerances,
        )
        base["samples"].append(result)
        base["sample_count"] += 1
        if result["status"] == "passed":
            base["passed_count"] += 1
        elif result["status"] == "failed":
            base["failed_inference_count"] += 1
        elif result["status"] == "warning":
            base["warning_count"] += 1
        for issue in result.get("diagnostics", []):
            if isinstance(issue, dict) and issue.get("severity") in {"failure", "warning"}:
                diagnostics.append(issue)

    return _finalize(base)


def export_self_test_validation_errors(self_test: dict | None) -> list[str]:
    if not isinstance(self_test, dict):
        return []
    out: list[str] = []
    for issue in self_test.get("diagnostics", []):
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "")
        if severity not in {"failure", "warning"}:
            continue
        code = str(issue.get("code") or "EXPORT_SELF_TEST_DIAGNOSTIC")
        message = str(issue.get("message") or "")
        out.append(f"{code}: {message}".strip())
    return list(dict.fromkeys(out))


def export_self_test_failed(self_test: dict | None) -> bool:
    return isinstance(self_test, dict) and str(self_test.get("status") or "").lower() == "failed"


def export_self_test_warning(self_test: dict | None) -> bool:
    return isinstance(self_test, dict) and str(self_test.get("status") or "").lower() == "warning"


def _run_sample_self_test(
    *,
    model,
    session,
    torch,
    metadata: dict,
    labels: list[str],
    sample: dict,
    sample_index: int,
    tolerances: dict,
) -> dict:
    diagnostics: list[dict] = []
    tensor = _tensor_from_sample(sample, torch)
    public = _public_sample_base(sample, sample_index, labels, metadata)
    public["diagnostics"] = diagnostics

    if tensor is None:
        diagnostics.append(
            _diagnostic(
                "HELDOUT_EVAL_TENSOR_UNAVAILABLE",
                "failure",
                "Held-out self-test sample is missing the exact eval input tensor.",
                "heldout_sample",
            )
        )
        public["status"] = "failed"
        return public

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.detach().cpu().float()
    public["input_tensor"] = {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": "float32",
        "sha256": _tensor_sha256(tensor),
    }

    expected_shape = _expected_input_shape(metadata)
    if expected_shape and [int(dim) for dim in tensor.shape] != expected_shape:
        diagnostics.append(
            _diagnostic(
                "INPUT_TENSOR_SHAPE_MISMATCH",
                "failure",
                f"Held-out eval tensor shape {list(tensor.shape)} does not match export input_shape {expected_shape}.",
                "preprocessing",
            )
        )

    metadata_label_hash = class_label_order_hash(labels)
    sample_label_hash = str(sample.get("class_label_order_hash") or "")
    if sample_label_hash and sample_label_hash != metadata_label_hash:
        diagnostics.append(
            _diagnostic(
                "LABEL_MAP_MISMATCH",
                "failure",
                "Training-time class label order hash does not match export manifest class label order hash.",
                "label_map",
            )
        )

    preprocessing_diagnostics = _preprocessing_diagnostics(sample, metadata, tensor, tolerances)
    diagnostics.extend(preprocessing_diagnostics)

    try:
        with torch.no_grad():
            pytorch_logits_tensor = model(tensor)
            if isinstance(pytorch_logits_tensor, (list, tuple)):
                pytorch_logits_tensor = pytorch_logits_tensor[0]
            pytorch_logits = _tensor_logits(pytorch_logits_tensor)
    except Exception as exc:
        diagnostics.append(
            _diagnostic("PYTORCH_INFERENCE_FAILED", "failure", str(exc), "runtime_parity")
        )
        public["status"] = "failed"
        return public

    reference_logits = _reference_logits(sample) or pytorch_logits
    if len(reference_logits) != len(labels):
        diagnostics.append(
            _diagnostic(
                "LABEL_MAP_OUTPUT_MISMATCH",
                "failure",
                f"PyTorch output class count {len(reference_logits)} does not match export class_labels count {len(labels)}.",
                "label_map",
            )
        )
        public["status"] = "failed"
        return public

    live_reference_diff = _max_abs_diff(reference_logits, pytorch_logits)
    live_reference_rel_diff = _max_rel_diff(reference_logits, pytorch_logits)
    if not _values_close(
        reference_logits,
        pytorch_logits,
        abs_atol=tolerances["logit_abs_atol"],
        rel_rtol=tolerances["logit_rel_rtol"],
    ):
        diagnostics.append(
            _diagnostic(
                "PYTORCH_REFERENCE_MISMATCH",
                "failure",
                "Current PyTorch model output does not match the training-time held-out reference logits.",
                "runtime_parity",
            )
        )

    try:
        onnx_logits = _run_onnx_logits(session, tensor)
    except Exception as exc:
        diagnostics.append(_diagnostic("ONNX_INFERENCE_FAILED", "failure", str(exc), "runtime_parity"))
        public["status"] = "failed"
        return public

    if len(onnx_logits) != len(labels):
        diagnostics.append(
            _diagnostic(
                "LABEL_MAP_OUTPUT_MISMATCH",
                "failure",
                f"ONNX output class count {len(onnx_logits)} does not match export class_labels count {len(labels)}.",
                "label_map",
            )
        )
        public["status"] = "failed"
        return public

    pytorch_probs = _softmax(reference_logits)
    onnx_probs = _softmax(onnx_logits)
    pytorch_top_k = _top_k(reference_logits, pytorch_probs, labels, tolerances["top_k"])
    onnx_top_k = _top_k(onnx_logits, onnx_probs, labels, tolerances["top_k"])
    logit_abs_diff = _max_abs_diff(reference_logits, onnx_logits)
    logit_rel_diff = _max_rel_diff(reference_logits, onnx_logits)
    probability_abs_diff = _max_abs_diff(pytorch_probs, onnx_probs)

    public["pytorch"] = {
        "logits": _rounded(reference_logits),
        "probabilities": _rounded(pytorch_probs),
        "top_k": pytorch_top_k,
        "predicted_label": pytorch_top_k[0]["label"] if pytorch_top_k else "",
    }
    public["onnx"] = {
        "logits": _rounded(onnx_logits),
        "probabilities": _rounded(onnx_probs),
        "top_k": onnx_top_k,
        "predicted_label": onnx_top_k[0]["label"] if onnx_top_k else "",
    }
    public["comparison"] = {
        "logit_max_abs_diff": round(float(logit_abs_diff), 8),
        "logit_max_rel_diff": round(float(logit_rel_diff), 8),
        "probability_max_abs_diff": round(float(probability_abs_diff), 8),
        "top_k_labels_match": [item["label"] for item in pytorch_top_k] == [item["label"] for item in onnx_top_k],
        "predicted_label_match": public["pytorch"]["predicted_label"] == public["onnx"]["predicted_label"],
        "pytorch_reference_max_abs_diff": round(float(live_reference_diff), 8),
        "pytorch_reference_max_rel_diff": round(float(live_reference_rel_diff), 8),
    }

    if (
        not _values_close(
            reference_logits,
            onnx_logits,
            abs_atol=tolerances["logit_abs_atol"],
            rel_rtol=tolerances["logit_rel_rtol"],
        )
        or probability_abs_diff > tolerances["probability_abs_atol"]
    ):
        diagnostics.append(
            _diagnostic(
                "ONNX_OUTPUT_MISMATCH",
                "failure",
                "ONNX logits/probabilities differ from the PyTorch held-out reference beyond tolerance.",
                "runtime_parity",
            )
        )
    if not public["comparison"]["top_k_labels_match"]:
        diagnostics.append(
            _diagnostic(
                "TOP_K_LABEL_MISMATCH",
                "failure",
                "ONNX top-k labels do not match PyTorch top-k labels for the held-out sample.",
                "runtime_parity",
            )
        )
    if not public["comparison"]["predicted_label_match"]:
        diagnostics.append(
            _diagnostic(
                "PREDICTED_LABEL_MISMATCH",
                "failure",
                "ONNX predicted label does not match PyTorch predicted label for the held-out sample.",
                "runtime_parity",
            )
        )

    true_label = str(sample.get("true_label") or "")
    if true_label and public["pytorch"]["predicted_label"] and true_label != public["pytorch"]["predicted_label"]:
        diagnostics.append(
            _diagnostic(
                "MODEL_MISCLASSIFICATION",
                "info",
                "The PyTorch model misclassified this held-out sample; this is a true model outcome, not an export mismatch.",
                "model_quality",
            )
        )

    if any(issue.get("severity") == "failure" for issue in diagnostics):
        public["status"] = "failed"
    elif any(issue.get("severity") == "warning" for issue in diagnostics):
        public["status"] = "warning"
    else:
        public["status"] = "passed"
    return public


def _preprocessing_diagnostics(sample: dict, metadata: dict, tensor, tolerances: dict) -> list[dict]:
    diagnostics: list[dict] = []
    image_metadata = sample.get("image_metadata") if isinstance(sample.get("image_metadata"), dict) else {}
    source_type = str(image_metadata.get("demo_source_type") or image_metadata.get("source") or "").lower()
    if image_metadata.get("parity_safe") is False or "thumbnail" in source_type:
        diagnostics.append(
            _diagnostic(
                "THUMBNAIL_DOWNSAMPLING_ARTIFACT",
                "warning",
                "Held-out sample metadata indicates thumbnail or non-parity-safe image bytes.",
                "heldout_sample",
            )
        )

    image_path = str(sample.get("image_path") or sample.get("path") or "")
    if not image_path:
        diagnostics.append(
            _diagnostic(
                "HELDOUT_SOURCE_IMAGE_UNAVAILABLE",
                "warning",
                "Held-out self-test sample did not include the original source image path for preprocessing replay.",
                "heldout_sample",
            )
        )
        return diagnostics
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        diagnostics.append(
            _diagnostic(
                "HELDOUT_SOURCE_IMAGE_UNAVAILABLE",
                "warning",
                "Original held-out image path is unavailable for preprocessing contract replay.",
                "heldout_sample",
            )
        )
        return diagnostics

    try:
        from PIL import Image

        with Image.open(path) as image:
            prepared = prepare_image_for_inference(
                Image,
                image,
                metadata,
                image_metadata=image_metadata,
                strict_metadata=True,
            )
            replayed = image_to_chw_float32_array(prepared, metadata)
    except Exception as exc:
        code = "MISSING_BBOX_CROP_METADATA" if "BBOX_METADATA_REQUIRED" in str(exc) else "PREPROCESSING_REPLAY_FAILED"
        diagnostics.append(_diagnostic(code, "warning", str(exc), "preprocessing"))
        return diagnostics

    expected = tensor.detach().cpu().numpy()[0]
    if list(replayed.shape) != list(expected.shape):
        diagnostics.append(
            _diagnostic(
                "PREPROCESSING_SHAPE_MISMATCH",
                "warning",
                f"Export preprocessing replay shape {list(replayed.shape)} does not match eval tensor shape {list(expected.shape)}.",
                "preprocessing",
            )
        )
        return diagnostics
    diff = _max_abs_diff(replayed.reshape(-1).tolist(), expected.reshape(-1).tolist())
    if diff > tolerances["preprocessing_abs_atol"]:
        diagnostics.append(
            _diagnostic(
                "PREPROCESSING_MISMATCH",
                "warning",
                "Export preprocessing contract replay does not reproduce the held-out eval tensor within tolerance.",
                "preprocessing",
            )
        )
    return diagnostics


def _public_sample_base(sample: dict, sample_index: int, labels: list[str], metadata: dict) -> dict:
    true_index = _optional_int(sample.get("true_index"))
    predicted_index = _optional_int(sample.get("predicted_index"))
    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("sample_id") or sample.get("id") or f"heldout:{sample_index}"),
        "split": str(sample.get("split") or "test"),
        "image_id": str(sample.get("image_id") or Path(str(sample.get("image_path") or sample.get("path") or "")).name),
        "source_image_path": str(sample.get("image_path") or sample.get("path") or ""),
        "true_index": true_index,
        "true_label": str(sample.get("true_label") or _label_at(labels, true_index)),
        "pytorch_predicted_index_at_training": predicted_index,
        "pytorch_predicted_label_at_training": str(
            sample.get("predicted_label") or sample.get("predicted_class") or _label_at(labels, predicted_index)
        ),
        "class_label_order_hash": str(sample.get("class_label_order_hash") or class_label_order_hash(labels)),
        "preprocessing_contract_hash": preprocessing_contract_hash(metadata),
        "image_metadata": sample.get("image_metadata") if isinstance(sample.get("image_metadata"), dict) else {},
        "status": "not_run",
    }


def _finalize(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), list) else []
    if payload.get("sample_count", 0) <= 0:
        payload["status"] = "warning" if diagnostics else "not_run"
    elif payload.get("failed_inference_count", 0) > 0 or any(
        isinstance(item, dict) and item.get("severity") == "failure" for item in diagnostics
    ):
        payload["status"] = "failed"
    elif payload.get("warning_count", 0) > 0 or any(
        isinstance(item, dict) and item.get("severity") == "warning" for item in diagnostics
    ):
        payload["status"] = "warning"
    else:
        payload["status"] = "passed"
    payload["export_verified"] = payload["status"] == "passed"
    payload["parity_status"] = "ok" if payload["status"] == "passed" else payload["status"]
    first_issue = next(
        (item for item in diagnostics if isinstance(item, dict) and item.get("severity") in {"failure", "warning"}),
        {},
    )
    payload["diagnostic_reason"] = str(first_issue.get("code") or "")
    return payload


def _load_onnx_session(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _run_onnx_logits(session, tensor) -> list[float]:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(output_names, {input_name: tensor.detach().cpu().numpy().astype("float32", copy=False)})
    import numpy as np

    logits = np.asarray(values[0], dtype=np.float32)
    if logits.ndim > 1:
        logits = logits.reshape(logits.shape[0], -1)[0]
    else:
        logits = logits.reshape(-1)
    return [float(value) for value in logits.tolist()]


def _tensor_from_sample(sample: dict, torch):
    for key in ("input_tensor", "eval_tensor", "tensor", "_input_tensor"):
        value = sample.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            return value.detach().cpu()
        try:
            return torch.tensor(value, dtype=torch.float32)
        except Exception:
            continue
    return None


def _tensor_logits(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.ndim > 1:
            value = value.reshape(value.shape[0], -1)[0]
        else:
            value = value.reshape(-1)
        return [float(item) for item in value.tolist()]
    return [float(item) for item in value]


def _reference_logits(sample: dict) -> list[float] | None:
    value = sample.get("pytorch_logits")
    if not isinstance(value, (list, tuple)):
        value = sample.get("logits")
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _expected_input_shape(metadata: dict) -> list[int]:
    shape = metadata.get("input_shape")
    if isinstance(shape, list) and len(shape) == 4:
        try:
            return [int(item) for item in shape]
        except (TypeError, ValueError):
            return []
    return []


def _tensor_sha256(tensor) -> str:
    array = tensor.detach().cpu().numpy().astype("float32", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _softmax(values: list[float]) -> list[float]:
    import math

    if not values:
        return []
    shifted = [float(value) - max(values) for value in values]
    exps = [math.exp(value) for value in shifted]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in exps]


def _top_k(logits: list[float], probabilities: list[float], labels: list[str], top_k: int) -> list[dict]:
    count = min(max(1, int(top_k)), len(labels), len(probabilities), len(logits))
    indices = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)[:count]
    return [
        {
            "index": int(index),
            "label": labels[index] if index < len(labels) else f"class_{index}",
            "logit": round(float(logits[index]), 6),
            "probability": round(float(probabilities[index]), 6),
        }
        for index in indices
    ]


def _max_abs_diff(left, right) -> float:
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return float("inf")
    if len(left_values) != len(right_values):
        return float("inf")
    if not left_values:
        return 0.0
    return max(abs(a - b) for a, b in zip(left_values, right_values))


def _max_rel_diff(left, right) -> float:
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return float("inf")
    if len(left_values) != len(right_values):
        return float("inf")
    if not left_values:
        return 0.0
    return max(abs(a - b) / max(abs(a), abs(b), 1e-12) for a, b in zip(left_values, right_values))


def _values_close(left, right, *, abs_atol: float, rel_rtol: float) -> bool:
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return False
    if len(left_values) != len(right_values):
        return False
    return all(abs(a - b) <= max(abs_atol, rel_rtol * max(abs(a), abs(b))) for a, b in zip(left_values, right_values))


def _rounded(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _tolerance(value: dict | None) -> dict:
    payload = value if isinstance(value, dict) else {}
    return {
        "logit_abs_atol": _positive_float(payload.get("logit_abs_atol"), 1e-4),
        "logit_rel_rtol": _positive_float(payload.get("logit_rel_rtol"), 1e-4),
        "probability_abs_atol": _positive_float(payload.get("probability_abs_atol"), 1e-4),
        "preprocessing_abs_atol": _positive_float(payload.get("preprocessing_abs_atol"), 1e-4),
        "top_k": _positive_int(payload.get("top_k"), 5),
    }


def _positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_at(labels: list[str], index: int | None) -> str:
    if index is None:
        return ""
    if 0 <= index < len(labels):
        return labels[index]
    return str(index)


def _diagnostic(code: str, severity: str, message: str, category: str) -> dict:
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": message,
    }
