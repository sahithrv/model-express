from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from worker.visual_analysis.schema import (
    VisualAnalysisValidationError,
    compact_json,
    validate_visual_analysis_output,
)

MAX_TOOL_ARRAY_ITEMS = 64
MAX_TOOL_DICT_ITEMS = 80
MAX_TOOL_DEPTH = 4
MAX_TOOL_TEXT_LENGTH = 500

VISUAL_ANALYSIS_TOOL_NAMES = (
    "get_dataset_metadata_summary",
    "get_sample_manifest",
    "get_allowed_operations",
    "validate_visual_analysis_draft",
)

VISUAL_ANALYSIS_RESPONSE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_dataset_metadata_summary",
        "description": (
            "Return bounded, path-free dataset metadata already approved for visual "
            "analysis. This is an information question, not an action."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_sample_manifest",
        "description": (
            "Return the bounded public sample manifest with image_id references only. "
            "No image bytes, local paths, storage URIs, or raw prompts are returned."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": ["integer", "null"],
                    "description": "Maximum number of manifest entries to return, or null for the bounded default.",
                }
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_allowed_operations",
        "description": (
            "Return allowed preprocessing and augmentation vocabulary for evidence-only "
            "hypotheses. This does not grant mutation or scheduling authority."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "validate_visual_analysis_draft",
        "description": (
            "Validate a draft final Visual Dataset Analysis JSON against the existing "
            "evidence-only schema. The answer returns only status, safe error text, "
            "and compact counts, never the draft itself."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "draft_json": {
                    "type": "string",
                    "description": "Stringified draft visual-analysis JSON object to validate.",
                },
            },
            "required": ["draft_json"],
            "additionalProperties": False,
        },
    },
]

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

SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "base64",
    "bytes",
    "credential",
    "password",
    "path",
    "prompt",
    "secret",
    "storage",
    "token",
    "uri",
    "url",
)
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
UNSAFE_TEXT_RE = re.compile(
    r"(data:image/[a-z0-9.+-]+;base64,|base64\s*[:=,]|[A-Za-z]:\\|file://|s3://|"
    r"aws_access_key|/Users/|/home/|/tmp/|\\\\|/9j/4AAQSkZJRg|iVBORw0KGgo)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VisualAnalysisToolContext:
    dataset_metadata: dict[str, Any]
    sample_manifest: list[dict[str, Any]]
    allowed_operations: dict[str, Any]
    trigger_reason: str
    dataset_id: str
    dataset_name: str
    total_images: int | None


def build_visual_analysis_tool_context(user_prompt: str) -> VisualAnalysisToolContext:
    payload = _loads_json_object(user_prompt)
    dataset_metadata = _safe_object(payload.get("dataset_metadata"))
    sample_manifest = [
        _sanitize_manifest_item(item)
        for item in _safe_list(payload.get("sample_manifest"))[:MAX_TOOL_ARRAY_ITEMS]
    ]
    allowed_operations = _safe_object(payload.get("allowed_operations"))
    trigger_reason = _safe_text(payload.get("trigger_reason") or "initial_profile", max_length=80)
    return VisualAnalysisToolContext(
        dataset_metadata=dataset_metadata,
        sample_manifest=sample_manifest,
        allowed_operations=allowed_operations,
        trigger_reason=trigger_reason or "initial_profile",
        dataset_id=_safe_text(dataset_metadata.get("dataset_id") or "", max_length=120),
        dataset_name=_safe_text(dataset_metadata.get("dataset_name") or "", max_length=180),
        total_images=_optional_int(dataset_metadata.get("total_images")),
    )


def execute_visual_analysis_tool(
    *,
    name: str,
    arguments: str | bytes | dict[str, Any] | None,
    context: VisualAnalysisToolContext,
) -> dict[str, Any]:
    if name not in VISUAL_ANALYSIS_TOOL_NAMES:
        return _tool_answer(
            {
                "accepted": False,
                "tool": _safe_tool_name(name),
                "error": "unsupported visual analysis information request",
                "allowed_tools": list(VISUAL_ANALYSIS_TOOL_NAMES),
            }
        )

    args, arg_error = _coerce_arguments(arguments)
    if arg_error:
        return _tool_answer(
            {
                "accepted": False,
                "tool": name,
                "error": arg_error,
            }
        )

    if name == "get_dataset_metadata_summary":
        return _tool_answer(
            {
                "accepted": True,
                "tool": name,
                "dataset_metadata_summary": context.dataset_metadata,
                "scope": _scope_summary(context),
            }
        )
    if name == "get_sample_manifest":
        limit = _argument_limit(args.get("limit"), len(context.sample_manifest))
        return _tool_answer(
            {
                "accepted": True,
                "tool": name,
                "sample_count": len(context.sample_manifest),
                "sample_manifest": context.sample_manifest[:limit],
                "limitations": [
                    "Manifest entries identify images by image_id only.",
                    "Raw image content and source locations are never returned by this tool.",
                ],
            }
        )
    if name == "get_allowed_operations":
        return _tool_answer(
            {
                "accepted": True,
                "tool": name,
                "allowed_operations": context.allowed_operations,
                "limitations": [
                    "Allowed operations are vocabulary for hypotheses only.",
                    "This vocabulary is advisory and evidence-only.",
                ],
            }
        )
    if name == "validate_visual_analysis_draft":
        return _tool_answer(_validate_draft(args, context, name))

    return _tool_answer({"accepted": False, "tool": name, "error": "unhandled visual tool"})


def tool_answer_json(answer: dict[str, Any]) -> str:
    return compact_json(_tool_answer(answer))


def _validate_draft(
    args: dict[str, Any],
    context: VisualAnalysisToolContext,
    tool_name: str,
) -> dict[str, Any]:
    draft_value = args.get("draft") if "draft" in args else args.get("draft_json")
    if draft_value in (None, ""):
        return {
            "accepted": False,
            "tool": tool_name,
            "error": "validate_visual_analysis_draft requires draft or draft_json",
        }

    try:
        parsed = validate_visual_analysis_output(
            draft_value,
            sample_manifest=context.sample_manifest,
            dataset_id=context.dataset_id or None,
            dataset_name=context.dataset_name or None,
            total_images=context.total_images,
            trigger_reason=context.trigger_reason,
            max_images_analyzed=len(context.sample_manifest),
        )
    except VisualAnalysisValidationError as exc:
        safety_violation = _validation_error_is_safety(str(exc))
        return {
            "accepted": True,
            "tool": tool_name,
            "validation": {
                "valid": False,
                "safety_violation": safety_violation,
                "repairable_schema_error": not safety_violation,
                "error": _safe_validation_error(str(exc), safety_violation=safety_violation),
            },
        }

    return {
        "accepted": True,
        "tool": tool_name,
        "validation": {
            "valid": True,
            "schema_version": parsed.get("schema_version"),
            "images_analyzed": parsed.get("images_analyzed"),
            "visual_trait_count": len(parsed.get("visual_traits") or []),
            "class_watch_count": len(parsed.get("classes_to_watch") or []),
            "preprocessing_hypothesis_count": len(parsed.get("preprocessing_hypotheses") or []),
            "caution_count": len(parsed.get("cautions") or []),
            "limitation_count": len(parsed.get("limitations") or []),
            "support_statuses": sorted(
                {
                    str(item.get("support_status"))
                    for item in parsed.get("preprocessing_hypotheses", [])
                    if isinstance(item, dict) and item.get("support_status")
                }
            ),
        },
    }


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _coerce_arguments(value: str | bytes | dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if value in (None, ""):
        return {}, ""
    if isinstance(value, dict):
        return value, ""
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}, "tool arguments must be a JSON object"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return parsed, ""


def _sanitize_manifest_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    out: dict[str, Any] = {}
    for key in SAFE_MANIFEST_KEYS:
        if key in item and not _unsafe_key(key):
            out[key] = _safe_bounded_value(item[key], depth=1)
    if "image_id" not in out and "id" in out:
        out["image_id"] = out.pop("id")
    return out


def _safe_object(value: Any) -> dict[str, Any]:
    safe = _safe_bounded_value(value, depth=0)
    return safe if isinstance(safe, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_bounded_value(value: Any, *, depth: int) -> Any:
    if depth > MAX_TOOL_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in list(value.items())[:MAX_TOOL_DICT_ITEMS]:
            key_text = _safe_text(key, max_length=120)
            if not key_text or _unsafe_key(key_text):
                continue
            out[key_text] = _safe_bounded_value(child, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [
            _safe_bounded_value(item, depth=depth + 1)
            for item in list(value)[:MAX_TOOL_ARRAY_ITEMS]
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _safe_text(value) if isinstance(value, str) else value
    return _safe_text(value)


def _safe_text(value: Any, *, max_length: int = MAX_TOOL_TEXT_LENGTH) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if UNSAFE_TEXT_RE.search(text):
        return "[redacted]"
    if len(text) > max_length:
        return text[: max_length - 1].rstrip() + "..."
    return text


def _safe_tool_name(value: Any) -> str:
    text = _safe_text(value, max_length=120)
    return "[unsupported]" if text.strip().lower() in FORBIDDEN_AUTHORITY_KEYS else text


def _unsafe_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower()
    if normalized in FORBIDDEN_AUTHORITY_KEYS:
        return True
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _argument_limit(value: Any, available: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = available
    return max(0, min(parsed, available, MAX_TOOL_ARRAY_ITEMS))


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scope_summary(context: VisualAnalysisToolContext) -> dict[str, Any]:
    return {
        "dataset_id": context.dataset_id,
        "dataset_name": context.dataset_name,
        "trigger_reason": context.trigger_reason,
        "sample_count": len(context.sample_manifest),
        "total_images": context.total_images,
    }


def _validation_error_is_safety(message: str) -> bool:
    normalized = message.lower()
    safety_fragments = (
        "forbidden",
        "file reference",
        "leaked",
        "sensitive",
        "does not match requested",
        "exceeds submitted sample manifest",
        "exceeds configured cap",
    )
    return any(fragment in normalized for fragment in safety_fragments)


def _safe_validation_error(message: str, *, safety_violation: bool) -> str:
    if not safety_violation:
        return _safe_text(message, max_length=300)
    normalized = message.lower()
    if "execution" in normalized or "authority" in normalized:
        return "draft violates the evidence-only boundary by including execution authority"
    if "file reference" in normalized or "leaked" in normalized:
        return "draft violates the evidence-only boundary by including a forbidden reference"
    if "does not match requested" in normalized or "exceeds" in normalized:
        return "draft violates the active visual-analysis scope"
    return "draft violates a visual-analysis safety boundary"


def _tool_answer(payload: dict[str, Any]) -> dict[str, Any]:
    safe = _safe_bounded_value(payload, depth=0)
    return safe if isinstance(safe, dict) else {"accepted": False, "error": "unsafe tool answer"}
