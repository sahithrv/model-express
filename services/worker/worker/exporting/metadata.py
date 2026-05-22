from __future__ import annotations

import time
from typing import Iterable


def build_champion_export_metadata(
    *,
    model_name: str,
    class_names: Iterable[str],
    image_size: int,
    preprocessing: dict | None = None,
    model_profile: dict | None = None,
    training_config: dict | None = None,
    artifact_uri: str | None = None,
) -> dict:
    """Build additive export metadata; the backend remains responsible for export records."""
    labels = [str(class_name) for class_name in class_names]
    profile = model_profile if isinstance(model_profile, dict) else {}
    config = training_config if isinstance(training_config, dict) else {}
    preprocessing_config = preprocessing if isinstance(preprocessing, dict) else {}
    return {
        "schema_version": "champion_export_metadata_v1",
        "model": str(model_name),
        "artifact_uri": artifact_uri or "",
        "format": "framework_native_checkpoint",
        "input_shape": [1, 3, int(image_size), int(image_size)],
        "class_labels": labels,
        "class_count": len(labels),
        "preprocessing": preprocessing_config,
        "training_config": config,
        "model_profile": profile,
        "limitations": [
            "Export metadata is worker-generated and must be accepted by backend validation before use.",
            "Live demo inference should use held-out or test images when available.",
        ],
    }


def build_demo_prediction_payload(
    *,
    image_id: str,
    predictions: Iterable[dict],
    latency_ms: float,
    true_label: str | None = None,
) -> dict:
    ranked_predictions = _rank_predictions(predictions)
    top_prediction = ranked_predictions[0] if ranked_predictions else None
    predicted_label = str(top_prediction.get("label")) if top_prediction else ""
    payload = {
        "schema_version": "champion_demo_prediction_v1",
        "image_id": str(image_id),
        "predicted_label": predicted_label,
        "confidence": float(top_prediction.get("confidence", 0.0)) if top_prediction else 0.0,
        "top_k": ranked_predictions,
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "created_at_unix": int(time.time()),
    }
    if true_label is not None:
        payload["true_label"] = str(true_label)
        payload["correct"] = predicted_label == str(true_label)
    return payload


def _rank_predictions(predictions: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        label = prediction.get("label")
        if label is None:
            continue
        try:
            confidence = float(prediction.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "label": str(label),
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
            }
        )
    return sorted(normalized, key=lambda item: item["confidence"], reverse=True)[:5]
