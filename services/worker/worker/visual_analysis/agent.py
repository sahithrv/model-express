from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from worker.visual_analysis.client import ImageInput, VisualJSONGenerator
from worker.visual_analysis.schema import (
    AGENT_NAME,
    AGENT_VERSION,
    AUGMENTATION_POLICIES,
    BBOX_MODES,
    CONFIDENCE_LEVELS,
    CROP_STRATEGIES,
    MECHANISMS,
    NORMALIZATIONS,
    PROMPT_VERSION,
    RESIZE_STRATEGIES,
    SCHEMA_VERSION,
    SUPPORT_STATUSES,
    TRAITS,
    TRIGGER_REASONS,
    compact_json,
    validate_visual_analysis_output,
)

SYSTEM_PROMPT = f"""You are the Visual Dataset Analysis Agent for Model Express.

You analyze a bounded sample of dataset images and metadata. You do not schedule experiments, create jobs, mutate datasets, relabel examples, delete files, or produce executable authority. Your output is visual evidence and preprocessing hypotheses only.

Use only the provided dataset metadata, sample manifest, and attached images. Refer to images by image_id only. Do not mention local file paths, storage URIs, credentials, or arbitrary file references. Be explicit about uncertainty and coverage limitations. If the sample is not class-complete, say so.

Return JSON only matching {SCHEMA_VERSION}. Do not include proposed_experiments, jobs, commands, plans, labels_to_change, dataset_mutations, or action_type."""

DEFAULT_ALLOWED_OPERATIONS = {
    "resize_strategy": sorted(RESIZE_STRATEGIES),
    "normalization": sorted(NORMALIZATIONS),
    "crop_strategy": sorted(CROP_STRATEGIES),
    "bbox_mode": sorted(BBOX_MODES),
    "augmentation_policy": sorted(AUGMENTATION_POLICIES),
}

SAFE_MANIFEST_KEYS = {
    "image_id",
    "id",
    "class_name",
    "class",
    "label",
    "width",
    "height",
    "selection_basis",
    "detail",
    "detail_level",
    "bbox",
    "bbox_count",
    "bbox_area_ratio",
    "split",
}
SENSITIVE_KEY_FRAGMENTS = ("path", "uri", "url", "secret", "token", "credential", "api_key", "password")


@dataclass(frozen=True)
class VisualAnalysisRequest:
    dataset_metadata: dict[str, Any]
    sample_manifest: list[dict[str, Any]]
    images: list[ImageInput] = field(default_factory=list)
    trigger_reason: str = "initial_profile"
    allowed_operations: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    total_images: int | None = None


class VisualAnalysisAgent:
    def __init__(self, llm_client: VisualJSONGenerator) -> None:
        self.llm_client = llm_client

    def analyze(self, request: VisualAnalysisRequest) -> dict[str, Any]:
        system_prompt, user_prompt = build_visual_analysis_messages(request)
        raw_output = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=request.images,
        )
        expected_dataset_id = _optional_str(request.dataset_metadata.get("dataset_id"))
        return validate_visual_analysis_output(
            raw_output,
            sample_manifest=request.sample_manifest,
            dataset_id=expected_dataset_id,
            dataset_name=_optional_str(request.dataset_metadata.get("dataset_name")),
            total_images=request.total_images,
            trigger_reason=request.trigger_reason,
            max_images_analyzed=len(request.sample_manifest),
        )


def build_visual_analysis_messages(request: VisualAnalysisRequest) -> tuple[str, str]:
    if request.trigger_reason not in TRIGGER_REASONS:
        raise ValueError(f"unsupported visual analysis trigger_reason: {request.trigger_reason}")
    safe_dataset = _sanitize_for_prompt(request.dataset_metadata)
    safe_manifest = [_sanitize_manifest_item(item) for item in request.sample_manifest]
    payload = {
        "agent_name": AGENT_NAME,
        "agent_version": AGENT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "trigger_reason": request.trigger_reason,
        "dataset_metadata": safe_dataset,
        "sample_manifest": safe_manifest,
        "allowed_operations": _sanitize_for_prompt(request.allowed_operations or DEFAULT_ALLOWED_OPERATIONS),
        "allowed_schema_values": {
            "confidence": sorted(CONFIDENCE_LEVELS),
            "trait": sorted(TRAITS),
            "mechanism": sorted(MECHANISMS),
            "support_status": sorted(SUPPORT_STATUSES),
        },
        "budget": _sanitize_for_prompt(request.budget or _budget_from_request(request)),
        "output_rules": [
            "Return one JSON object only.",
            "Use image_id values from sample_manifest for example_image_ids.",
            "Do not mention file paths, storage URIs, or image bytes.",
            "Do not infer class-wide truth from one example; use confidence low.",
            "Never use null for enum fields such as confidence, level, severity, support_status, trait, or mechanism; choose the best allowed value, using low when uncertain.",
            "Each visual_traits item must include trait, level, confidence, evidence, and example_image_ids.",
            "Each preprocessing_hypotheses item should include id like vh_001, mechanism, summary, evidence, expected_effect, confidence, and support_status.",
            "Unsupported operations must use support_status unsupported and explain unsupported_reason.",
            "Never output proposed_experiments, jobs, commands, plans, labels_to_change, or dataset mutations.",
        ],
        "required_top_level_fields": [
            "schema_version",
            "dataset_id",
            "dataset_name",
            "total_images",
            "images_analyzed",
            "trigger_reason",
            "confidence",
            "coverage_report",
            "visual_traits",
            "classes_to_watch",
            "preprocessing_hypotheses",
            "cautions",
            "limitations",
        ],
        "required_nested_fields": {
            "coverage_report": [
                "selection_strategy",
                "selection_basis",
                "images_available",
                "images_analyzed",
                "classes_total",
                "classes_covered",
                "class_coverage_ratio",
            ],
            "visual_traits[]": ["trait", "level", "confidence", "evidence", "example_image_ids"],
            "classes_to_watch[]": ["class_name", "reason", "evidence", "example_image_ids", "confidence"],
            "preprocessing_hypotheses[]": [
                "id",
                "mechanism",
                "summary",
                "evidence",
                "expected_effect",
                "confidence",
                "support_status",
            ],
            "cautions[]": ["operation", "reason", "severity", "confidence", "example_image_ids"],
        },
        "output_shape_example": {
            "preprocessing_hypotheses": [
                {
                    "id": "vh_001",
                    "mechanism": "resolution_crop",
                    "summary": "Short evidence-backed hypothesis.",
                    "evidence": ["Observation tied to image_id values."],
                    "example_image_ids": ["img_001"],
                    "expected_effect": "Expected training effect if backend validates the idea.",
                    "risk": "Possible downside.",
                    "confidence": "low",
                    "support_status": "needs_backend_validation",
                }
            ],
        },
    }
    return SYSTEM_PROMPT, json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _budget_from_request(request: VisualAnalysisRequest) -> dict[str, Any]:
    high_detail_count = sum(1 for image in request.images if image.detail == "high")
    return {
        "image_count": len(request.sample_manifest),
        "attached_image_count": len(request.images),
        "high_detail_image_count": high_detail_count,
        "evidence_only": True,
        "raw_images_for_planner": False,
    }


def _sanitize_manifest_item(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    out: dict[str, Any] = {}
    for key in SAFE_MANIFEST_KEYS:
        if key in item:
            out[key] = _sanitize_for_prompt(item[key])
    if "image_id" not in out and "id" in out:
        out["image_id"] = out.pop("id")
    return out


def _sanitize_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = str(key).strip().lower()
            if any(fragment in normalized_key for fragment in SENSITIVE_KEY_FRAGMENTS):
                continue
            out[str(key)] = _sanitize_for_prompt(child)
        return out
    if isinstance(value, list):
        return [_sanitize_for_prompt(item) for item in value[:200]]
    if isinstance(value, tuple):
        return [_sanitize_for_prompt(item) for item in value[:200]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def visual_analysis_request_fingerprint(request: VisualAnalysisRequest) -> str:
    """Stable prompt fingerprint without image bytes for worker-side telemetry/tests."""
    _, user_prompt = build_visual_analysis_messages(request)
    return compact_json(json.loads(user_prompt))
