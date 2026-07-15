from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

from worker.model_express_catalog import require_catalog_id


ARTIFACT_PLAN_SCHEMA_V1 = "artifact_plan_v1"
POLICY_CONTRACT_V1 = "policy_contract_v1"
ARTIFACT_PLAN_CAPABILITY_V1 = "artifact_plan_v1"


class ArtifactPlanError(ValueError):
    pass


def load_artifact_plan(config: dict, *, task: str, runner: str) -> dict | None:
    spec = config.get("execution_spec_v1")
    value = spec.get("artifact_plan") if isinstance(spec, dict) else None
    if value is None:
        value = config.get("artifact_plan_v1")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ArtifactPlanError("artifact_plan_v1 must be an object")
    plan = deepcopy(value)
    validate_artifact_plan(plan, task=task, runner=runner)
    return plan


def validate_artifact_plan(plan: dict, *, task: str, runner: str) -> None:
    if plan.get("schema_version") != ARTIFACT_PLAN_SCHEMA_V1:
        raise ArtifactPlanError(f"Unknown artifact plan schema {plan.get('schema_version')!r}")
    if plan.get("artifact_plan_hash") != _artifact_plan_hash(plan):
        raise ArtifactPlanError("artifact plan hash mismatch")
    artifacts = plan.get("artifacts")
    fallbacks = plan.get("fallback_formats")
    requirements = plan.get("required_worker_capabilities")
    if not isinstance(artifacts, list) or not isinstance(fallbacks, list) or not isinstance(requirements, dict):
        raise ArtifactPlanError("artifact plan arrays and worker requirements are required")
    if not all(isinstance(value, str) for value in fallbacks):
        raise ArtifactPlanError("artifact fallback formats must be strings")

    capability_runner = _artifact_capability_runner(task, runner, automatic=plan.get("automatic") is True)
    seen: set[str] = set()
    for directive in artifacts:
        if not isinstance(directive, dict):
            raise ArtifactPlanError("artifact directives must be objects")
        artifact_format = require_catalog_id(
            "export_formats", directive.get("format"), task=task, runner=capability_runner
        )
        if artifact_format in seen:
            raise ArtifactPlanError(f"artifact plan duplicates format {artifact_format!r}")
        seen.add(artifact_format)
        expected_runtime = {
            "onnx": "onnxruntime",
            "torchscript": "torchscript",
            "pytorch": "pytorch",
            "safetensors": "pytorch",
        }.get(artifact_format)
        if expected_runtime is None:
            raise ArtifactPlanError(f"unsupported artifact format {artifact_format!r}")
        precision = require_catalog_id("precisions", directive.get("precision"), task=task, runner=capability_runner)
        runtime = require_catalog_id("runtimes", directive.get("runtime"), task=task, runner=capability_runner)
        provider = require_catalog_id(
            "execution_providers", directive.get("execution_provider"), task=task, runner=capability_runner
        )
        execution_requirements = directive.get("execution_requirements")
        if not isinstance(execution_requirements, list) or not all(
            isinstance(value, str) for value in execution_requirements
        ):
            raise ArtifactPlanError("artifact execution requirements must be strings")
        normalized_requirements = [
            require_catalog_id("execution_requirements", value, task=task, runner=capability_runner)
            for value in execution_requirements
        ]
        if (
            precision != "fp32"
            or runtime != expected_runtime
            or provider != "cpu_execution_provider"
            or normalized_requirements != ["cpu_compatible"]
        ):
            raise ArtifactPlanError(
                f"unsupported artifact capability combination for format {artifact_format!r}"
            )

    for fallback in fallbacks:
        if fallback not in seen:
            raise ArtifactPlanError(f"artifact fallback {fallback!r} is not an authorized output")

    policy_versions = requirements.get("policy_contract_versions")
    artifact_versions = requirements.get("artifact_plan_versions")
    if not isinstance(policy_versions, list) or not isinstance(artifact_versions, list):
        raise ArtifactPlanError("worker capability requirements must be arrays")
    if any(value != POLICY_CONTRACT_V1 for value in policy_versions):
        raise ArtifactPlanError("unknown required worker policy capability")
    if any(value != ARTIFACT_PLAN_CAPABILITY_V1 for value in artifact_versions):
        raise ArtifactPlanError("unknown required worker artifact capability")
    if plan.get("policy_restricted") is True and (
        POLICY_CONTRACT_V1 not in policy_versions
        or ARTIFACT_PLAN_CAPABILITY_V1 not in artifact_versions
    ):
        raise ArtifactPlanError("restricted artifact plan is missing worker capability requirements")


def artifact_formats(plan: dict | None, *, legacy: tuple[str, ...]) -> tuple[str, ...]:
    if plan is None:
        return legacy
    return tuple(str(item["format"]) for item in plan["artifacts"])


def helper_export_formats(plan: dict | None, *, legacy: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        {
            "onnx": "onnx",
            "torchscript": "torchscript",
            "pytorch": "framework_native",
            "safetensors": "safetensors",
        }[artifact_format]
        for artifact_format in artifact_formats(plan, legacy=legacy)
    )


def artifact_format_allowed(plan: dict | None, artifact_format: str) -> bool:
    if plan is None:
        return True
    return artifact_format in artifact_formats(plan, legacy=())


def artifact_fallback_allowed(plan: dict | None, artifact_format: str) -> bool:
    if plan is None:
        return True
    return artifact_format in plan["fallback_formats"]


def require_requested_artifact(plan: dict | None, requested_format: str) -> None:
    if plan is None:
        return
    formats = artifact_formats(plan, legacy=())
    if formats != (requested_format,):
        raise ArtifactPlanError(
            f"server artifact plan does not authorize requested format {requested_format!r}"
        )


def _artifact_plan_hash(plan: dict) -> str:
    required = plan.get("required_worker_capabilities") or {}
    canonical = {
        "schema_version": plan.get("schema_version"),
        "artifacts": [
            {
                "format": directive.get("format"),
                "precision": directive.get("precision"),
                "runtime": directive.get("runtime"),
                "execution_provider": directive.get("execution_provider"),
                "execution_requirements": directive.get("execution_requirements"),
            }
            for directive in (plan.get("artifacts") or [])
        ],
        "fallback_formats": plan.get("fallback_formats"),
        "automatic": plan.get("automatic"),
        "policy_restricted": plan.get("policy_restricted"),
    }
    if plan.get("effective_policy_hash"):
        canonical["effective_policy_hash"] = plan["effective_policy_hash"]
    canonical["required_worker_capabilities"] = {
        "policy_contract_versions": required.get("policy_contract_versions"),
        "artifact_plan_versions": required.get("artifact_plan_versions"),
    }
    canonical["artifact_plan_hash"] = ""
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii").rstrip("=")
    return f"sha256:{digest}"


def _artifact_capability_runner(task: str, runner: str, *, automatic: bool) -> str:
    if automatic or runner != "local_simulator":
        return runner
    return {
        "image_classification": "modal_torchvision",
        "object_detection": "modal_ultralytics",
    }.get(task, runner)
