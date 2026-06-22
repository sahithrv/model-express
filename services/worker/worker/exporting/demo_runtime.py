from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from worker.champion_jobs import (
    _demo_image_path,
    _first_string,
    _manifest_path,
    _manifest_payload,
    _optional_float,
    _optional_positive_int,
    _positive_int,
)
from worker.exporting.inference import (
    clear_demo_inference_cache,
    demo_prediction_result_from_inference,
    run_demo_inference_from_manifest,
)


def predict_from_request(request: dict) -> dict:
    config = request.get("config") if isinstance(request.get("config"), dict) else request
    request_id = _first_string(config, "request_id", "id") or f"local-{int(time.time() * 1000)}"
    image_uri = _first_string(config, "image_uri", "image_path", "local_image_path")
    true_label = _first_string(config, "true_label", "label", "class_name")
    image_metadata = config.get("image_metadata") if isinstance(config.get("image_metadata"), dict) else {}
    manifest_path = _manifest_path(config)
    manifest_payload = _manifest_payload(config) if manifest_path is None else None
    image_path, image_error = _demo_image_path(config, request_id)

    if manifest_path is None and manifest_payload is None:
        result = _failed_result(
            "MANIFEST_NOT_CONFIGURED",
            "No READY champion export manifest or artifact metadata is available for local demo inference.",
        )
    elif image_path is None:
        result = _failed_result("IMAGE_UNAVAILABLE", image_error or "Demo image is unavailable to the local runtime.")
    else:
        inference = run_demo_inference_from_manifest(
            manifest_path=manifest_path,
            manifest=manifest_payload,
            image_path=image_path,
            top_k=_positive_int(config.get("top_k"), 5),
            true_label=true_label,
            image_metadata=image_metadata,
            confidence_threshold=_optional_float(config.get("confidence_threshold")),
            iou_threshold=_optional_float(config.get("iou_threshold")),
            max_detections=_optional_positive_int(config.get("max_detections")),
        )
        result = _local_result_from_inference(inference)

    inference_image_metadata = result.get("image_metadata") if isinstance(result.get("image_metadata"), dict) else {}
    result_image_metadata = {
        **dict(image_metadata),
        **inference_image_metadata,
        "local_runtime": True,
        "runtime_host": "mission_control_python",
    }
    latency_breakdown = result.get("latency_breakdown_ms")
    if isinstance(latency_breakdown, dict):
        result_image_metadata["latency_breakdown_ms"] = latency_breakdown
    result.update(
        {
            "image_uri": image_uri,
            "image_id": _first_string(config, "image_id") or result.get("image_id", ""),
            "true_label": true_label or result.get("true_label", ""),
            "image_metadata": result_image_metadata,
            "manifest_path": str(manifest_path) if manifest_path is not None else "",
        }
    )
    return result


def _local_result_from_inference(payload: dict) -> dict:
    result = demo_prediction_result_from_inference(payload)
    if result.get("status") == "RUNTIME_UNAVAILABLE":
        result["status"] = "FAILED"
        metadata = result.get("image_metadata") if isinstance(result.get("image_metadata"), dict) else {}
        metadata["local_runtime_unavailable"] = True
        result["image_metadata"] = metadata
    return result


def _failed_result(error_code: str, error: str) -> dict:
    return {
        "status": "FAILED",
        "error_code": str(error_code),
        "error": str(error),
        "predicted_label": "",
        "confidence": 0.0,
        "top_k": [],
        "latency_ms": 0.0,
    }


def handle_message(message: dict) -> dict:
    operation = str(message.get("op") or message.get("operation") or "predict").strip().lower()
    message_id = message.get("id")
    if operation == "ping":
        return {"id": message_id, "ok": True, "status": "ready", "pid": os.getpid()}
    if operation == "dispose":
        clear_demo_inference_cache()
        return {"id": message_id, "ok": True, "status": "disposed", "pid": os.getpid()}
    if operation == "shutdown":
        clear_demo_inference_cache()
        return {"id": message_id, "ok": True, "status": "shutdown", "pid": os.getpid(), "shutdown": True}
    if operation != "predict":
        return {"id": message_id, "ok": False, "code": "UNKNOWN_OPERATION", "error": f"Unknown demo runtime operation: {operation}"}
    prediction = predict_from_request(message)
    return {"id": message_id, "ok": True, "prediction": prediction, "pid": os.getpid()}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message_id = None
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("runtime message must be a JSON object")
            message_id = message.get("id")
            response = handle_message(message)
        except Exception as exc:
            response = {"id": message_id, "ok": False, "code": "LOCAL_RUNTIME_EXCEPTION", "error": str(exc)}
        sys.stdout.write(json.dumps(response, separators=(",", ":"), default=str) + "\n")
        sys.stdout.flush()
        if response.get("shutdown"):
            break
    clear_demo_inference_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())