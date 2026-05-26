from __future__ import annotations

IMAGE_POLICY_TYPES = {"basic", "randaugment", "trivialaugment", "autoaugment"}
MIXED_SAMPLE_POLICY_TYPES = {"mixup", "cutmix"}
LEGACY_POLICY_TYPES = {"", "none", "custom", "light", "moderate", "strong"}

_POLICY_ALIASES = {
    "auto_augment": "autoaugment",
    "rand_augment": "randaugment",
    "trivial_augment": "trivialaugment",
    "trivialaugmentwide": "trivialaugment",
    "trivial_augment_wide": "trivialaugment",
}


def normalize_augmentation_config(
    augmentation: object,
    augmentation_policy: object = "",
    augmentation_policy_config: object = None,
) -> dict:
    """Normalize worker augmentation config and enforce bounded structured policies."""
    augmentation_payload = dict(augmentation) if isinstance(augmentation, dict) else {}
    merged = dict(augmentation_payload)
    structured_payload = augmentation_policy_config if isinstance(augmentation_policy_config, dict) else {}
    policy_payload = augmentation_policy if isinstance(augmentation_policy, dict) else {}
    if structured_payload:
        merged = {**structured_payload, **augmentation_payload}
    if policy_payload:
        merged = {**structured_payload, **policy_payload, **augmentation_payload}

    policy_type = _policy_type(merged)
    if not policy_type and not policy_payload:
        policy_type = _normalize_policy_name(augmentation_policy)
    if not policy_type:
        return merged

    if policy_type in {"light", "moderate", "strong"}:
        _apply_legacy_policy_defaults(merged, policy_type)
        return merged
    if policy_type in {"none", "custom"}:
        return merged
    if policy_type in MIXED_SAMPLE_POLICY_TYPES:
        alpha = _capped_float(
            merged.get("alpha"), default=0.2, minimum=0.0, maximum=1.0, field="alpha"
        )
        probability = _capped_float(
            merged.get("probability"), default=1.0, minimum=0.0, maximum=1.0, field="probability"
        )
        merged["policy_type"] = policy_type
        merged["alpha"] = alpha
        merged["probability"] = probability
        return merged
    if policy_type not in IMAGE_POLICY_TYPES:
        supported = ", ".join(sorted(IMAGE_POLICY_TYPES | LEGACY_POLICY_TYPES | MIXED_SAMPLE_POLICY_TYPES))
        raise ValueError(
            f"Unsupported augmentation policy_type '{policy_type}'. Supported policies: {supported}."
        )

    merged["policy_type"] = policy_type
    if policy_type == "basic":
        merged.setdefault("horizontal_flip", True)
    elif policy_type == "randaugment":
        merged["magnitude"] = _capped_int(
            merged.get("magnitude"), default=9, minimum=0, maximum=15, field="magnitude"
        )
        merged["num_ops"] = _capped_int(
            merged.get("num_ops"), default=2, minimum=1, maximum=3, field="num_ops"
        )
        merged["probability"] = _capped_float(
            merged.get("probability"), default=1.0, minimum=0.0, maximum=1.0, field="probability"
        )
    elif policy_type in {"trivialaugment", "autoaugment"}:
        merged["probability"] = _capped_float(
            merged.get("probability"), default=1.0, minimum=0.0, maximum=1.0, field="probability"
        )
    if policy_type == "trivialaugment":
        merged["num_magnitude_bins"] = _capped_int(
            merged.get("num_magnitude_bins"), default=31, minimum=2, maximum=31, field="num_magnitude_bins"
        )
    return merged


def structured_policy_type(augmentation: dict) -> str:
    return _policy_type(augmentation)


def _policy_type(config: object) -> str:
    if not isinstance(config, dict):
        return ""
    for key in ("policy_type", "type", "policy"):
        policy = _normalize_policy_name(config.get(key))
        if policy:
            return policy
    return ""


def _normalize_policy_name(value: object) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _POLICY_ALIASES.get(normalized, normalized)
    return normalized.replace("_", "")


def _apply_legacy_policy_defaults(merged: dict, policy_type: str) -> None:
    if policy_type == "light":
        merged.setdefault("horizontal_flip", True)
    elif policy_type == "moderate":
        merged.setdefault("horizontal_flip", True)
        merged.setdefault("color_jitter", True)
        merged.setdefault("random_crop", True)
    elif policy_type == "strong":
        merged.setdefault("horizontal_flip", True)
        merged.setdefault("color_jitter", True)
        merged.setdefault("random_crop", True)
        merged.setdefault("random_rotation", True)
        merged.setdefault("random_erasing", True)


def _capped_int(value: object, default: int, minimum: int, maximum: int, field: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Augmentation {field} must be an integer.") from exc
    return max(minimum, min(maximum, parsed))


def _capped_float(value: object, default: float, minimum: float, maximum: float, field: str) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Augmentation {field} must be a number.") from exc
    return max(minimum, min(maximum, parsed))
