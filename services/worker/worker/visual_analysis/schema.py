from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "dataset_visual_analysis_v1"
PROMPT_VERSION = "visual_dataset_analysis_prompt_v1"
AGENT_NAME = "visual_dataset_analysis"
AGENT_VERSION = "visual_dataset_analysis_agent_v1"

CONFIDENCE_LEVELS = {"low", "medium", "high"}
TRIGGER_REASONS = {"initial_profile", "deficiency_reanalysis", "manual"}
SUPPORT_STATUSES = {"supported", "unsupported", "needs_backend_validation"}
TRAITS = {
    "small_objects",
    "large_objects",
    "background_dominance",
    "lighting_variation",
    "blur",
    "fine_grained_similarity",
    "color_texture_signal",
    "crop_bbox_useful",
    "visual_ambiguity",
    "orientation_sensitive",
    "text_or_watermark",
    "domain_shift_possible",
}
MECHANISMS = {
    "baseline_control",
    "architecture_challenge",
    "capacity_finetune",
    "optimizer_scheduler",
    "regularization",
    "augmentation_basic",
    "augmentation_auto",
    "augmentation_mixed_sample",
    "class_imbalance",
    "minority_targeting",
    "resolution_crop",
    "bbox_crop_ablation",
    "label_noise_audit",
    "hard_example_audit",
    "deployment_latency",
    "distillation",
}
RESIZE_STRATEGIES = {
    "squash",
    "preserve_aspect_pad",
    "center_crop",
    "random_resized_crop",
    "bbox_crop_if_available",
}
NORMALIZATIONS = {"imagenet", "dataset", "none"}
CROP_STRATEGIES = {
    "none",
    "center_crop",
    "random_resized_crop",
    "bbox_crop_if_available",
    "bbox_crop_ablation",
}
BBOX_MODES = {
    "ignore",
    "crop_if_available",
    "crop_and_compare_full_image",
    "use_boxes_as_metadata",
}
AUGMENTATION_POLICIES = {
    "none",
    "light",
    "moderate",
    "strong",
    "custom",
    "basic",
    "randaugment",
    "trivialaugment",
    "trivialaugmentwide",
    "autoaugment",
    "mixup",
    "cutmix",
}
STRUCTURED_AUGMENTATION_POLICY_TYPES = {
    "none",
    "basic",
    "randaugment",
    "trivialaugment",
    "trivialaugmentwide",
    "autoaugment",
    "mixup",
    "cutmix",
}

FORBIDDEN_AUTHORITY_KEYS = {
    "action",
    "action_type",
    "command",
    "commands",
    "create_job",
    "dataset_mutation",
    "dataset_mutations",
    "delete",
    "delete_files",
    "experiment",
    "experiments",
    "job",
    "job_config",
    "job_configs",
    "jobs",
    "labels_to_change",
    "plan",
    "planned_experiments",
    "plans",
    "proposed_experiment",
    "proposed_experiments",
    "relabel",
    "schedule",
    "shell_command",
    "shell_commands",
}
SENSITIVE_KEY_FRAGMENTS = ("api_key", "secret", "token", "credential", "password")
LOCAL_PATH_RE = re.compile(
    r"([A-Za-z]:\\|file://|s3://|aws_access_key|/Users/|/home/|/tmp/|\\\\)",
    re.IGNORECASE,
)
MAX_TEXT_LENGTH = 500
MAX_LONG_TEXT_LENGTH = 900
MAX_ARRAY_ITEMS = 12
MAX_EVIDENCE_ITEMS = 6
MAX_LIMITATIONS = 12


class VisualAnalysisValidationError(ValueError):
    """Raised when a visual LLM output violates the evidence-only schema."""


def validate_visual_analysis_output(
    raw_output: str | bytes | dict[str, Any],
    *,
    sample_manifest: list[dict[str, Any]] | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    total_images: int | None = None,
    trigger_reason: str | None = None,
    max_images_analyzed: int | None = None,
) -> dict[str, Any]:
    """Parse and sanitize visual-agent JSON without granting execution authority."""
    payload = _coerce_json_object(raw_output)
    _reject_forbidden_authority(payload)
    _reject_path_leakage(payload)

    manifest_ids = _manifest_image_ids(sample_manifest)
    sanitized: dict[str, Any] = {
        "schema_version": _required_text(payload, "schema_version"),
        "dataset_id": _required_text(payload, "dataset_id"),
        "dataset_name": _optional_text(payload, "dataset_name") or (dataset_name or ""),
        "total_images": _required_int(payload, "total_images", minimum=0),
        "images_analyzed": _required_int(payload, "images_analyzed", minimum=1),
        "trigger_reason": _optional_enum(payload, "trigger_reason", TRIGGER_REASONS)
        or trigger_reason
        or "initial_profile",
        "confidence": _enum(payload.get("confidence"), CONFIDENCE_LEVELS, "confidence"),
    }

    if sanitized["schema_version"] != SCHEMA_VERSION:
        raise VisualAnalysisValidationError(
            f"unsupported schema_version {sanitized['schema_version']!r}"
        )
    if dataset_id is not None and sanitized["dataset_id"] != dataset_id:
        raise VisualAnalysisValidationError("dataset_id does not match requested dataset")
    if dataset_name is not None and sanitized["dataset_name"] and sanitized["dataset_name"] != dataset_name:
        raise VisualAnalysisValidationError("dataset_name does not match requested dataset")
    if total_images is not None and sanitized["total_images"] != int(total_images):
        raise VisualAnalysisValidationError("total_images does not match requested dataset")
    if trigger_reason is not None and sanitized["trigger_reason"] != trigger_reason:
        raise VisualAnalysisValidationError("trigger_reason does not match requested trigger")
    if sanitized["trigger_reason"] not in TRIGGER_REASONS:
        raise VisualAnalysisValidationError("trigger_reason is unsupported")
    if sample_manifest is not None and sanitized["images_analyzed"] > len(sample_manifest):
        raise VisualAnalysisValidationError("images_analyzed exceeds submitted sample manifest")
    if max_images_analyzed is not None and sanitized["images_analyzed"] > int(max_images_analyzed):
        raise VisualAnalysisValidationError("images_analyzed exceeds configured cap")

    sanitized["coverage_report"] = _coverage_report(payload.get("coverage_report"), sanitized)
    sanitized["visual_traits"] = [
        _visual_trait(item, manifest_ids)
        for item in _list(payload.get("visual_traits"))[:MAX_ARRAY_ITEMS]
    ]
    sanitized["classes_to_watch"] = [
        _class_watch_item(item, manifest_ids)
        for item in _list(payload.get("classes_to_watch"))[:MAX_ARRAY_ITEMS]
    ]
    sanitized["preprocessing_hypotheses"] = [
        _preprocessing_hypothesis(item, manifest_ids, index)
        for index, item in enumerate(_list(payload.get("preprocessing_hypotheses"))[:MAX_ARRAY_ITEMS])
    ]
    sanitized["cautions"] = [
        _visual_caution(item, manifest_ids)
        for item in _list(payload.get("cautions"))[:MAX_ARRAY_ITEMS]
    ]
    sanitized["limitations"] = _texts(payload.get("limitations"), limit=MAX_LIMITATIONS)

    if not sanitized["limitations"]:
        sanitized["limitations"] = [
            "Visual analysis is based on a bounded sentinel sample, not full dataset inspection.",
            "No experiment should be scheduled from this output without backend validation.",
        ]
    return sanitized


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _coerce_json_object(raw_output: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        payload = deepcopy(raw_output)
    else:
        text = raw_output.decode("utf-8") if isinstance(raw_output, bytes) else str(raw_output)
        text = text.strip()
        if not text:
            raise VisualAnalysisValidationError("visual LLM output was empty")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VisualAnalysisValidationError("visual LLM output must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise VisualAnalysisValidationError("visual LLM output must be a JSON object")
    return payload


def _coverage_report(value: Any, root: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualAnalysisValidationError("coverage_report must be an object")
    images_available = _int(value.get("images_available"), default=root["total_images"], minimum=0)
    images_analyzed = _int(value.get("images_analyzed"), default=root["images_analyzed"], minimum=0)
    classes_total = _int(value.get("classes_total"), default=0, minimum=0)
    classes_covered = _int(value.get("classes_covered"), default=0, minimum=0)
    if images_analyzed != root["images_analyzed"]:
        raise VisualAnalysisValidationError("coverage_report.images_analyzed is inconsistent")
    if images_analyzed > images_available:
        raise VisualAnalysisValidationError("coverage_report images_analyzed exceeds images_available")
    if classes_covered > classes_total:
        raise VisualAnalysisValidationError("coverage_report classes_covered exceeds classes_total")

    ratio = _float(value.get("class_coverage_ratio"), default=0.0, minimum=0.0, maximum=1.0)
    if classes_total > 0:
        ratio = round(classes_covered / classes_total, 6)

    per_class_counts = {}
    raw_counts = value.get("per_class_counts")
    if isinstance(raw_counts, dict):
        for key, count in sorted(raw_counts.items())[:100]:
            per_class_counts[_text(key, max_length=120)] = _int(count, default=0, minimum=0)

    return {
        "selection_strategy": _optional_text(value, "selection_strategy")
        or "deterministic_risk_and_representative_sampling",
        "selection_basis": _texts(value.get("selection_basis"), limit=MAX_ARRAY_ITEMS, max_length=120),
        "images_available": images_available,
        "images_analyzed": images_analyzed,
        "classes_total": classes_total,
        "classes_covered": classes_covered,
        "class_coverage_ratio": ratio,
        "per_class_counts": per_class_counts,
        "hard_example_count": _int(value.get("hard_example_count"), default=0, minimum=0),
        "edge_case_count": _int(value.get("edge_case_count"), default=0, minimum=0),
        "high_detail_image_count": _int(value.get("high_detail_image_count"), default=0, minimum=0),
        "limitations": _texts(value.get("limitations"), limit=MAX_LIMITATIONS),
    }


def _visual_trait(value: Any, manifest_ids: set[str]) -> dict[str, Any]:
    item = _object(value, "visual_traits item")
    return {
        "trait": _enum(item.get("trait"), TRAITS, "visual_traits.trait"),
        "level": _enum(item.get("level"), CONFIDENCE_LEVELS, "visual_traits.level"),
        "confidence": _enum(item.get("confidence"), CONFIDENCE_LEVELS, "visual_traits.confidence"),
        "evidence": _texts(item.get("evidence"), limit=MAX_EVIDENCE_ITEMS),
        "example_image_ids": _image_ids(item.get("example_image_ids"), manifest_ids),
        "affected_classes": _texts(item.get("affected_classes"), limit=MAX_ARRAY_ITEMS, max_length=120),
        "notes": _optional_text(item, "notes", max_length=MAX_LONG_TEXT_LENGTH),
    }


def _class_watch_item(value: Any, manifest_ids: set[str]) -> dict[str, Any]:
    item = _object(value, "classes_to_watch item")
    return {
        "class_name": _required_text(item, "class_name", max_length=120),
        "reason": _required_text(item, "reason"),
        "related_classes": _texts(item.get("related_classes"), limit=MAX_ARRAY_ITEMS, max_length=120),
        "evidence": _texts(item.get("evidence"), limit=MAX_EVIDENCE_ITEMS),
        "example_image_ids": _image_ids(item.get("example_image_ids"), manifest_ids),
        "confidence": _enum(item.get("confidence"), CONFIDENCE_LEVELS, "classes_to_watch.confidence"),
    }


def _preprocessing_hypothesis(value: Any, manifest_ids: set[str], index: int) -> dict[str, Any]:
    item = _object(value, "preprocessing_hypotheses item")
    hypothesis = {
        "id": _hypothesis_id(item, index),
        "mechanism": _enum(item.get("mechanism"), MECHANISMS, "preprocessing_hypotheses.mechanism"),
        "summary": _required_text(item, "summary", max_length=MAX_LONG_TEXT_LENGTH),
        "evidence": _texts(item.get("evidence"), limit=MAX_EVIDENCE_ITEMS),
        "expected_effect": _required_text(item, "expected_effect", max_length=MAX_LONG_TEXT_LENGTH),
        "risk": _optional_text(item, "risk", max_length=MAX_LONG_TEXT_LENGTH),
        "confidence": _enum(item.get("confidence"), CONFIDENCE_LEVELS, "preprocessing_hypotheses.confidence"),
        "support_status": _enum(
            item.get("support_status") or "needs_backend_validation",
            SUPPORT_STATUSES,
            "preprocessing_hypotheses.support_status",
        ),
        "unsupported_reason": _optional_text(
            item,
            "unsupported_reason",
            max_length=MAX_LONG_TEXT_LENGTH,
        ),
        "example_image_ids": _image_ids(item.get("example_image_ids"), manifest_ids),
    }

    unsupported_reasons: list[str] = []
    suggested_preprocessing = _supported_preprocessing(item.get("suggested_preprocessing"), unsupported_reasons)
    if suggested_preprocessing:
        hypothesis["suggested_preprocessing"] = suggested_preprocessing

    image_sizes = _image_sizes(item.get("suggested_image_sizes"), unsupported_reasons)
    if image_sizes:
        hypothesis["suggested_image_sizes"] = image_sizes

    policy = _supported_augmentation_policy(item.get("suggested_augmentation_policy"), unsupported_reasons)
    if policy:
        hypothesis["suggested_augmentation_policy"] = policy

    policy_config = _supported_augmentation_policy_config(
        item.get("suggested_augmentation_policy_config"), unsupported_reasons
    )
    if policy_config:
        hypothesis["suggested_augmentation_policy_config"] = policy_config

    if unsupported_reasons:
        hypothesis["support_status"] = "unsupported"
        existing_reason = hypothesis.get("unsupported_reason") or ""
        joined = "; ".join(unsupported_reasons)
        hypothesis["unsupported_reason"] = _text(
            f"{existing_reason}; {joined}" if existing_reason else joined,
            max_length=MAX_LONG_TEXT_LENGTH,
        )
    return hypothesis


def _hypothesis_id(item: dict[str, Any], index: int) -> str:
    supplied = _optional_text(item, "id", max_length=80)
    if supplied:
        return supplied
    return f"vh_{index + 1:03d}"


def _visual_caution(value: Any, manifest_ids: set[str]) -> dict[str, Any]:
    item = _object(value, "cautions item")
    return {
        "operation": _required_text(item, "operation", max_length=160),
        "reason": _required_text(item, "reason"),
        "severity": _enum(item.get("severity"), CONFIDENCE_LEVELS, "cautions.severity"),
        "confidence": _enum(item.get("confidence"), CONFIDENCE_LEVELS, "cautions.confidence"),
        "affected_classes": _texts(item.get("affected_classes"), limit=MAX_ARRAY_ITEMS, max_length=120),
        "example_image_ids": _image_ids(item.get("example_image_ids"), manifest_ids),
    }


def _supported_preprocessing(value: Any, unsupported_reasons: list[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        unsupported_reasons.append("suggested_preprocessing must be an object")
        return None
    out: dict[str, Any] = {}
    checks = [
        ("resize_strategy", RESIZE_STRATEGIES),
        ("normalization", NORMALIZATIONS),
        ("crop_strategy", CROP_STRATEGIES),
        ("bbox_mode", BBOX_MODES),
    ]
    for key, allowed in checks:
        if key not in value or value.get(key) in (None, ""):
            continue
        normalized = _normalize(value.get(key))
        if normalized not in allowed:
            unsupported_reasons.append(f"unsupported suggested_preprocessing.{key}: {value.get(key)!r}")
            continue
        out[key] = normalized
    if "use_dataset_normalization" in value:
        out["use_dataset_normalization"] = bool(value.get("use_dataset_normalization"))
    return out or None


def _image_sizes(value: Any, unsupported_reasons: list[str]) -> list[int]:
    out: list[int] = []
    for item in _list(value)[:4]:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            unsupported_reasons.append(f"invalid suggested_image_sizes value: {item!r}")
            continue
        if parsed < 32 or parsed > 1024:
            unsupported_reasons.append(f"unsupported suggested_image_sizes value: {parsed}")
            continue
        out.append(parsed)
    return sorted(set(out))


def _supported_augmentation_policy(value: Any, unsupported_reasons: list[str]) -> str:
    if value in (None, ""):
        return ""
    normalized = _normalize(value)
    if normalized not in AUGMENTATION_POLICIES:
        unsupported_reasons.append(f"unsupported suggested_augmentation_policy: {value!r}")
        return ""
    return normalized


def _supported_augmentation_policy_config(
    value: Any,
    unsupported_reasons: list[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        unsupported_reasons.append("suggested_augmentation_policy_config must be an object")
        return None
    policy_type = _normalize(value.get("policy_type"))
    if policy_type not in STRUCTURED_AUGMENTATION_POLICY_TYPES:
        unsupported_reasons.append(
            f"unsupported suggested_augmentation_policy_config.policy_type: {value.get('policy_type')!r}"
        )
        return None
    out: dict[str, Any] = {"policy_type": policy_type}
    integer_ranges = {
        "magnitude": (0, 15),
        "num_ops": (0, 3),
        "num_magnitude_bins": (2, 31),
    }
    for key, (minimum, maximum) in integer_ranges.items():
        if key not in value or value.get(key) in (None, ""):
            continue
        try:
            parsed = int(value.get(key))
        except (TypeError, ValueError):
            unsupported_reasons.append(f"{key} must be an integer")
            continue
        if parsed < minimum or parsed > maximum:
            unsupported_reasons.append(f"{key} must be between {minimum} and {maximum}")
            continue
        out[key] = parsed
    for key in ("probability", "alpha"):
        if key not in value or value.get(key) in (None, ""):
            continue
        try:
            parsed = _float(value.get(key), default=0.0, minimum=0.0, maximum=1.0)
        except VisualAnalysisValidationError:
            unsupported_reasons.append(f"{key} must be between 0 and 1")
            continue
        out[key] = parsed
    return out


def _reject_forbidden_authority(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize(key)
            if normalized_key in FORBIDDEN_AUTHORITY_KEYS:
                raise VisualAnalysisValidationError(
                    f"visual analysis output included forbidden execution field {path}.{key}"
                )
            _reject_forbidden_authority(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_authority(child, f"{path}[{index}]")


def _reject_path_leakage(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize(key)
            if "path" in normalized_key or normalized_key in {"uri", "url"}:
                raise VisualAnalysisValidationError(
                    f"visual analysis output included forbidden file reference field {path}.{key}"
                )
            if any(fragment in normalized_key for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise VisualAnalysisValidationError(
                    f"visual analysis output included sensitive field {path}.{key}"
                )
            _reject_path_leakage(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_path_leakage(child, f"{path}[{index}]")
    elif isinstance(value, str) and LOCAL_PATH_RE.search(value):
        raise VisualAnalysisValidationError(f"visual analysis output leaked a file reference at {path}")


def _manifest_image_ids(sample_manifest: list[dict[str, Any]] | None) -> set[str]:
    if sample_manifest is None:
        return set()
    ids: set[str] = set()
    for item in sample_manifest:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id", item.get("id"))
        if image_id not in (None, ""):
            ids.add(str(image_id))
    return ids


def _image_ids(value: Any, manifest_ids: set[str]) -> list[str]:
    out = _texts(value, limit=MAX_EVIDENCE_ITEMS, max_length=160)
    if manifest_ids:
        missing = [image_id for image_id in out if image_id not in manifest_ids]
        if missing:
            raise VisualAnalysisValidationError(
                f"example_image_ids were not present in submitted manifest: {missing[:3]}"
            )
    return out


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualAnalysisValidationError(f"{label} must be an object")
    return value


def _required_text(value: dict[str, Any], key: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    text = _optional_text(value, key, max_length=max_length)
    if not text:
        raise VisualAnalysisValidationError(f"{key} is required")
    return text


def _optional_text(value: dict[str, Any], key: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    if key not in value or value.get(key) in (None, ""):
        return ""
    return _text(value.get(key), max_length=max_length)


def _text(value: Any, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    text = str(value).strip()
    if len(text) > max_length:
        return text[: max_length - 1].rstrip() + "..."
    return text


def _texts(value: Any, *, limit: int, max_length: int = MAX_TEXT_LENGTH) -> list[str]:
    return [_text(item, max_length=max_length) for item in _list(value)[:limit] if item not in (None, "")]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _required_int(value: dict[str, Any], key: str, *, minimum: int) -> int:
    if key not in value:
        raise VisualAnalysisValidationError(f"{key} is required")
    return _int(value.get(key), default=minimum, minimum=minimum)


def _int(value: Any, *, default: int, minimum: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VisualAnalysisValidationError(f"expected integer value, got {value!r}") from exc
    if parsed < minimum:
        raise VisualAnalysisValidationError(f"integer value {parsed} is below minimum {minimum}")
    return parsed


def _float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VisualAnalysisValidationError(f"expected numeric value, got {value!r}") from exc
    if parsed < minimum or parsed > maximum:
        raise VisualAnalysisValidationError(
            f"numeric value {parsed} outside range {minimum}..{maximum}"
        )
    return parsed


def _optional_enum(value: dict[str, Any], key: str, allowed: set[str]) -> str:
    if key not in value or value.get(key) in (None, ""):
        return ""
    return _enum(value.get(key), allowed, key)


def _enum(value: Any, allowed: set[str], field: str) -> str:
    normalized = _normalize(value)
    if normalized not in allowed:
        raise VisualAnalysisValidationError(f"{field} has unsupported value {value!r}")
    return normalized


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()
