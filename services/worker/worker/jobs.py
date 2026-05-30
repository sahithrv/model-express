from __future__ import annotations

import base64
import json
import os

from worker.diagnostics import log_event
from worker.datasets.cache import (
    cleanup_job_dataset_cache,
    dataset_archive_path,
    extract_dataset_archive,
    job_dataset_cache_root,
    should_persist_dataset_cache,
)
from worker.datasets.profiler import profile_image_folder
from worker.datasets.metadata_discovery import build_metadata_import_payload
from worker.datasets.label_quality import build_label_quality_profile_patch
from worker.datasets.storage import download_s3_uri
from worker.datasets.visual_sampling import generate_visual_sample_pack
from worker.champion_jobs import (
    run_champion_demo_prediction_job,
    run_export_champion_job,
    run_generate_visual_exemplars_job,
)
from worker.orchestrator_client import OrchestratorClient
from worker.training.providers import run_training_job


def run_job(client: OrchestratorClient, job: dict) -> None:
    template = job["template"]
    if template == "profile_dataset":
        run_profile_dataset_job(client, job)
        return 
    if template == "train_experiment":
        run_training_job(client, job)
        return
    if template == "export_champion":
        run_export_champion_job(client, job)
        return
    if template == "champion_demo_prediction":
        run_champion_demo_prediction_job(client, job)
        return
    if template == "generate_visual_exemplars":
        run_generate_visual_exemplars_job(client, job)
        return
    if template == "label_quality_audit":
        run_label_quality_audit_job(client, job)
        return
    if template == "analyze_dataset_visuals":
        run_analyze_dataset_visuals_job(client, job)
        return
    raise ValueError(f"Unsupported job template: {template}")

def run_profile_dataset_job(client: OrchestratorClient, job: dict) -> None:
    dataset_id = job["config"]["dataset_id"]
    job_id = str(job.get("id") or "")
    cache_root = job_dataset_cache_root(job_id)
    provider = _profile_provider(job)
    if provider == "modal":
        from worker.training.modal_provider import run_modal_dataset_profile

        run_modal_dataset_profile(client, job)
        return

    dataset = client.get_dataset(dataset_id)

    try:
        dataset_dir = _download_and_extract_dataset(dataset_id, dataset["storage_uri"], cache_root)
        profile = profile_image_folder(dataset_dir)

        _send_dataset_metadata_import(client, dataset_id, dataset_dir)
        client.update_dataset_profile(dataset_id, profile)
        client.complete_job(job["id"], mlflow_run_id="")
    finally:
        if not should_persist_dataset_cache():
            cleanup_job_dataset_cache(job_id, cache_root)


def run_label_quality_audit_job(client: OrchestratorClient, job: dict) -> None:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    dataset_id = str(config.get("dataset_id") or "")
    if not dataset_id:
        raise ValueError("label_quality_audit jobs require config.dataset_id.")

    dataset = client.get_dataset(dataset_id)
    profile = dataset.get("profile") if isinstance(dataset, dict) else {}
    profile_patch = build_label_quality_profile_patch(config, profile if isinstance(profile, dict) else {})
    client.update_dataset_profile(dataset_id, profile_patch)
    client.complete_job(job["id"], mlflow_run_id="")


def run_analyze_dataset_visuals_job(client: OrchestratorClient, job: dict) -> None:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    dataset_id = str(config.get("dataset_id") or "")
    if not dataset_id:
        raise ValueError("analyze_dataset_visuals jobs require config.dataset_id.")

    dataset = client.get_dataset(dataset_id)
    job_id = str(job.get("id") or "")
    cache_root = job_dataset_cache_root(job_id)
    try:
        dataset_dir = _download_and_extract_dataset(dataset_id, dataset["storage_uri"], cache_root)
        caps = _visual_sample_caps(config)
        pack = generate_visual_sample_pack(
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            dataset_name=str(dataset.get("name") or config.get("dataset_name") or ""),
            max_total_images=caps["max_total_images"],
            max_high_detail_images=caps["max_high_detail_images"],
            max_image_bytes=caps["max_image_bytes"],
            max_total_bytes=caps["max_total_bytes"],
            image_size=caps["image_size"],
            high_detail_image_size=caps["high_detail_image_size"],
            quality=caps["quality"],
            seed=_positive_int(config.get("seed"), 0),
            max_metadata_images=caps["max_metadata_images"],
        )
        if pack.get("status") == "unavailable":
            client.fail_job(job["id"], pack.get("error") or "visual sample pack generation unavailable")
            return
        manifest = pack.get("sample_manifest") if isinstance(pack.get("sample_manifest"), dict) else {}
        if _positive_int(manifest.get("images_analyzed"), 0) <= 0:
            client.fail_job(job["id"], "visual sample pack contained no analyzable images")
            return
        try:
            llm_result = _run_visual_llm_analysis(dataset, config, pack)
        except Exception as exc:
            log_event(
                "warn",
                "visual_analysis_llm_unavailable",
                job_id=job.get("id"),
                project_id=job.get("project_id"),
                dataset_id=dataset_id,
                template=job.get("template"),
                provider=os.getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "openai"),
                model=os.getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", ""),
                api_style=os.getenv("MODEL_EXPRESS_VISUAL_LLM_API_STYLE", "chat"),
                error=str(exc),
            )
            llm_result = _visual_analysis_unavailable_result(dataset, config, pack, exc)
        payload = _visual_analysis_result_payload(job, dataset, config, pack, caps, llm_result)
        client.report_dataset_visual_analysis_result(dataset_id, payload)
        client.complete_job(job["id"], mlflow_run_id="")
    finally:
        if not should_persist_dataset_cache():
            cleanup_job_dataset_cache(job_id, cache_root)


def _profile_provider(job: dict) -> str:
    config = job.get("config", {})
    if isinstance(config, dict) and config.get("provider"):
        return str(config["provider"]).lower()

    provider = os.getenv("MODEL_EXPRESS_DATASET_PROFILE_PROVIDER", "").strip().lower()
    if provider:
        return provider

    gpu_type = os.getenv("GPU_TYPE", "").strip().lower()
    return "modal" if gpu_type == "modal" else "local"


def _download_and_extract_dataset(dataset_id: str, storage_uri: str, cache_root=None):
    archive_path = dataset_archive_path(dataset_id, cache_root)
    if not archive_path.exists():
        download_s3_uri(storage_uri, archive_path)
    return extract_dataset_archive(archive_path, dataset_id, cache_root)


def _send_dataset_metadata_import(client: OrchestratorClient, dataset_id: str, dataset_dir) -> None:
    try:
        payload = build_metadata_import_payload(dataset_dir)
        result = client.import_dataset_metadata(dataset_id, payload)
        if isinstance(result, dict) and result.get("status") == "unavailable":
            log_event(
                "warn",
                "dataset_metadata_import_unavailable",
                dataset_id=dataset_id,
                reason=result.get("reason"),
                status_code=result.get("status_code"),
            )
    except Exception as exc:
        log_event(
            "warn",
            "dataset_metadata_import_failed_nonfatal",
            dataset_id=dataset_id,
            error=str(exc),
        )


def _visual_sample_caps(config: dict) -> dict:
    return {
        "max_total_images": min(_positive_int(config.get("max_total_images"), 48), 64),
        "max_high_detail_images": min(_positive_int(config.get("max_high_detail_images"), 6), 8),
        "max_image_bytes": min(_positive_int(config.get("max_image_bytes"), 350_000), 1_000_000),
        "max_total_bytes": min(_positive_int(config.get("max_total_bytes"), 8_000_000), 16_000_000),
        "image_size": min(max(_positive_int(config.get("image_size"), 512), 64), 1024),
        "high_detail_image_size": min(
            max(_positive_int(config.get("high_detail_image_size"), 1024), 128),
            1600,
        ),
        "quality": min(max(_positive_int(config.get("quality"), 82), 35), 95),
        "max_metadata_images": min(_positive_int(config.get("max_metadata_images"), 2_500), 10_000),
    }


def _run_visual_llm_analysis(dataset: dict, config: dict, pack: dict) -> dict:
    from worker.visual_analysis.agent import VisualAnalysisRequest, build_visual_analysis_messages
    from worker.visual_analysis.client import VisualLLMClient, VisualLLMConfig
    from worker.visual_analysis.schema import (
        AGENT_VERSION,
        PROMPT_VERSION,
        CONFIDENCE_LEVELS,
        MECHANISMS,
        SCHEMA_VERSION,
        SUPPORT_STATUSES,
        TRAITS,
        TRIGGER_REASONS,
    )

    llm_config = VisualLLMConfig.from_env()
    if not llm_config.enabled:
        raise ValueError("visual LLM is disabled; configure MODEL_EXPRESS_VISUAL_LLM_* to run analyze_dataset_visuals")

    manifest = pack.get("sample_manifest") if isinstance(pack.get("sample_manifest"), dict) else {}
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    trigger_reason = str(config.get("trigger_reason") or "initial_profile")
    request = VisualAnalysisRequest(
        dataset_metadata=_visual_dataset_metadata(dataset, config, manifest),
        sample_manifest=samples,
        images=_agent_image_inputs(pack.get("image_inputs")),
        trigger_reason=trigger_reason,
        allowed_operations=config.get("allowed_operations") if isinstance(config.get("allowed_operations"), dict) else None,
        budget=_visual_analysis_budget(manifest),
        total_images=int(manifest.get("images_available") or 0),
    )
    system_prompt, user_prompt = build_visual_analysis_messages(request)
    llm_client = VisualLLMClient(llm_config)
    raw_output = llm_client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=request.images,
    )
    analysis, accepted_raw_output, repair = _validate_visual_analysis_with_repair(
        raw_output=raw_output,
        llm_client=llm_client,
        sample_manifest=samples,
        dataset_metadata=request.dataset_metadata,
        total_images=request.total_images,
        trigger_reason=trigger_reason,
        max_images_analyzed=len(samples),
        schema_values={
            "schema_version": SCHEMA_VERSION,
            "confidence": sorted(CONFIDENCE_LEVELS),
            "mechanism": sorted(MECHANISMS),
            "trait": sorted(TRAITS),
            "trigger_reason": sorted(TRIGGER_REASONS),
            "support_status": sorted(SUPPORT_STATUSES),
        },
    )
    return {
        "analysis": analysis,
        "raw_output": accepted_raw_output,
        "input_messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "provider": llm_config.provider,
        "model": llm_config.model,
        "agent_version": AGENT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "repair": repair,
    }


def _visual_analysis_unavailable_result(dataset: dict, config: dict, pack: dict, exc: Exception) -> dict:
    from worker.visual_analysis.schema import AGENT_VERSION, PROMPT_VERSION, SCHEMA_VERSION, compact_json

    manifest = pack.get("sample_manifest") if isinstance(pack.get("sample_manifest"), dict) else {}
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    images_analyzed = _positive_int(manifest.get("images_analyzed"), len(samples))
    total_images = _positive_int(manifest.get("images_available"), images_analyzed)
    trigger_reason = str(config.get("trigger_reason") or "initial_profile")
    limitations = [
        str(item)
        for item in manifest.get("limitations", [])
        if item not in (None, "")
    ][:8] if isinstance(manifest.get("limitations"), list) else []
    limitations.extend(
        [
            "Visual LLM unavailable; no visual traits or preprocessing hypotheses were inferred from the bounded sample.",
            "Planner should rely on dataset profile statistics and backend validation until visual analysis is rerun.",
        ]
    )
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": str(config.get("dataset_id") or dataset.get("id") or ""),
        "dataset_name": str(dataset.get("name") or config.get("dataset_name") or ""),
        "total_images": total_images,
        "images_analyzed": images_analyzed,
        "trigger_reason": trigger_reason,
        "confidence": "low",
        "coverage_report": {
            "selection_strategy": manifest.get("selection_strategy") or "deterministic_risk_and_representative_sampling",
            "selection_basis": manifest.get("selection_basis") if isinstance(manifest.get("selection_basis"), list) else [],
            "images_available": total_images,
            "images_analyzed": images_analyzed,
            "classes_total": _positive_int(manifest.get("classes_total"), 0),
            "classes_covered": _positive_int(manifest.get("classes_covered"), 0),
            "class_coverage_ratio": manifest.get("class_coverage_ratio") or 0,
            "per_class_counts": manifest.get("per_class_counts") if isinstance(manifest.get("per_class_counts"), dict) else {},
            "hard_example_count": _positive_int(manifest.get("hard_example_count"), 0),
            "edge_case_count": _positive_int(manifest.get("edge_case_count"), 0),
            "high_detail_image_count": _positive_int(manifest.get("high_detail_image_count"), 0),
            "limitations": limitations,
        },
        "classes_to_watch": [],
        "visual_traits": [],
        "preprocessing_hypotheses": [],
        "cautions": [
            {
                "operation": "visual_llm_analysis",
                "reason": "The visual LLM request did not complete, so no image-level conclusions were produced.",
                "severity": "medium",
                "confidence": "high",
                "affected_classes": [],
                "example_image_ids": [],
            }
        ],
        "limitations": limitations,
    }
    return {
        "analysis": analysis,
        "raw_output": compact_json(analysis),
        "input_messages": [],
        "provider": os.getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "openai"),
        "model": os.getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", ""),
        "agent_version": AGENT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "repair": {
            "attempted": False,
            "llm_unavailable": True,
            "fallback_record": True,
            "failure_type": exc.__class__.__name__,
        },
    }


def _validate_visual_analysis_with_repair(
    *,
    raw_output: str,
    llm_client,
    sample_manifest: list[dict],
    dataset_metadata: dict,
    total_images: int,
    trigger_reason: str,
    max_images_analyzed: int,
    schema_values: dict,
) -> tuple[dict, str, dict]:
    from worker.visual_analysis.schema import VisualAnalysisValidationError, validate_visual_analysis_output

    validation_args = {
        "sample_manifest": sample_manifest,
        "dataset_id": str(dataset_metadata.get("dataset_id") or ""),
        "dataset_name": str(dataset_metadata.get("dataset_name") or ""),
        "total_images": total_images,
        "trigger_reason": trigger_reason,
        "max_images_analyzed": max_images_analyzed,
    }
    try:
        return validate_visual_analysis_output(raw_output, **validation_args), raw_output, {"attempted": False}
    except VisualAnalysisValidationError as first_error:
        if not _visual_analysis_error_is_repairable(str(first_error)):
            raise
        repair_prompt = _visual_analysis_repair_prompt(
            raw_output=raw_output,
            validation_error=str(first_error),
            schema_values=schema_values,
        )
        repaired_output = llm_client.generate_json(
            system_prompt=_VISUAL_ANALYSIS_REPAIR_SYSTEM_PROMPT,
            user_prompt=repair_prompt,
            images=[],
        )
        try:
            return (
                validate_visual_analysis_output(repaired_output, **validation_args),
                repaired_output,
                {
                    "attempted": True,
                    "initial_validation_error": str(first_error),
                    "accepted_after_repair": True,
                },
            )
        except VisualAnalysisValidationError as second_error:
            raise ValueError(
                "visual LLM output invalid after repair: "
                f"initial_error={first_error}; repair_error={second_error}"
            ) from second_error


def _visual_analysis_error_is_repairable(message: str) -> bool:
    blocked_fragments = (
        "forbidden",
        "file reference",
        "leaked",
        "does not match requested",
        "exceeds submitted sample manifest",
        "exceeds configured cap",
    )
    normalized = message.lower()
    return not any(fragment in normalized for fragment in blocked_fragments)


_VISUAL_ANALYSIS_REPAIR_SYSTEM_PROMPT = """You repair Visual Dataset Analysis JSON for Model Express.

Return one corrected JSON object only. Keep the same evidence-only safety boundary:
no jobs, plans, commands, labels_to_change, dataset mutations, local file paths, storage URIs, or image bytes.
Do not remove required evidence to hide an error. For required enum fields, never use null; choose the best allowed value from existing evidence and use "low" when uncertain."""


def _visual_analysis_repair_prompt(*, raw_output: str, validation_error: str, schema_values: dict) -> str:
    return json.dumps(
        {
            "task": "repair_visual_dataset_analysis_json",
            "validation_error": validation_error,
            "allowed_schema_values": schema_values,
            "repair_rules": [
                "Return JSON only.",
                "Fix schema violations without adding execution authority.",
                "Every visual_traits item must keep trait, level, confidence, evidence, and example_image_ids.",
                "Every preprocessing_hypotheses item should keep id like vh_001 when present, mechanism, summary, evidence, expected_effect, confidence, and support_status.",
                "Every confidence, level, severity, support_status, trait, mechanism, "
                "and trigger_reason enum must be non-null and allowed.",
                "Use confidence low when the original evidence is weak or uncertain.",
            ],
            "original_output": raw_output,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _visual_analysis_result_payload(
    job: dict,
    dataset: dict,
    config: dict,
    pack: dict,
    caps: dict,
    llm_result: dict,
) -> dict:
    dataset_id = str(config.get("dataset_id") or dataset.get("id") or "")
    manifest = pack.get("sample_manifest") if isinstance(pack.get("sample_manifest"), dict) else {}
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    analysis = dict(llm_result.get("analysis") if isinstance(llm_result.get("analysis"), dict) else {})
    analysis.update(
        {
            "project_id": str(dataset.get("project_id") or analysis.get("project_id") or ""),
            "dataset_id": dataset_id,
            "dataset_name": str(dataset.get("name") or config.get("dataset_name") or analysis.get("dataset_name") or ""),
            "source_job_id": str(job.get("id") or ""),
            "provider": str(llm_result.get("provider") or analysis.get("provider") or ""),
            "model": str(llm_result.get("model") or analysis.get("model") or ""),
            "agent_version": str(llm_result.get("agent_version") or analysis.get("agent_version") or "v1"),
            "prompt_version": str(llm_result.get("prompt_version") or analysis.get("prompt_version") or ""),
            "trigger_details": config.get("trigger_details") if isinstance(config.get("trigger_details"), dict) else {},
            "sample_manifest": _backend_sample_manifest(samples),
            "raw_output": str(llm_result.get("raw_output") or ""),
            "input_messages": llm_result.get("input_messages") if isinstance(llm_result.get("input_messages"), list) else [],
            "input_context": {
                "sample_manifest_summary": _manifest_summary(manifest),
                "caps": caps,
                "metadata_summary": _safe_config_metadata_summary(config),
                "pack_summary": pack.get("summary") if isinstance(pack.get("summary"), dict) else {},
                "repair": llm_result.get("repair") if isinstance(llm_result.get("repair"), dict) else {"attempted": False},
                "evidence_only": True,
                "raw_images_included_for_planner": False,
            },
        }
    )
    return analysis


def _visual_dataset_metadata(dataset: dict, config: dict, manifest: dict) -> dict:
    profile = dataset.get("profile") if isinstance(dataset.get("profile"), dict) else {}
    safe_metadata_summary = _safe_config_metadata_summary(config)
    if not safe_metadata_summary:
        candidate = profile.get("agent_safe_metadata_summary") or profile.get("normalized_metadata_summary") or profile.get("metadata_summary")
        safe_metadata_summary = candidate if isinstance(candidate, dict) else {}
    merged_metadata_summary = profile.get("metadata_summary") if isinstance(profile.get("metadata_summary"), dict) else {}
    if safe_metadata_summary:
        merged_metadata_summary = {**merged_metadata_summary, **safe_metadata_summary}
    profile_for_prompt = dict(profile)
    if safe_metadata_summary:
        profile_for_prompt["agent_safe_metadata_summary"] = safe_metadata_summary
        profile_for_prompt["metadata_summary"] = merged_metadata_summary
    dataset_id = str(config.get("dataset_id") or dataset.get("id") or "")
    total_images = max(
        _positive_int(manifest.get("images_available"), 0),
        _positive_int(profile.get("total_images"), 0),
        _positive_int(profile.get("image_count"), 0),
        _positive_int(safe_metadata_summary.get("sample_count"), 0),
    )
    class_count = max(
        _positive_int(manifest.get("classes_total"), 0),
        _positive_int(profile.get("class_count"), 0),
        _positive_int(safe_metadata_summary.get("class_count"), 0),
    )
    return {
        "dataset_id": dataset_id,
        "dataset_name": str(dataset.get("name") or config.get("dataset_name") or ""),
        "profile": profile_for_prompt,
        "total_images": total_images,
        "class_count": class_count,
        "class_distribution": profile.get("class_distribution") or profile.get("images_per_class") or {},
        "metadata_summary": merged_metadata_summary,
        "agent_safe_metadata_summary": safe_metadata_summary,
        "visual_trait_summary": profile.get("visual_trait_summary") if isinstance(profile.get("visual_trait_summary"), dict) else {},
    }


def _safe_config_metadata_summary(config: dict) -> dict:
    for key in ("agent_safe_metadata_summary", "metadata_summary", "normalized_metadata_summary"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _agent_image_inputs(image_inputs: object) -> list:
    from worker.visual_analysis.client import ImageInput

    out = []
    if not isinstance(image_inputs, list):
        return out
    for item in image_inputs:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "")
        encoded = str(item.get("data_base64") or "")
        if not image_id or not encoded:
            continue
        detail_level = str(item.get("detail_level") or item.get("detail") or "standard").lower()
        detail = "high" if detail_level == "high" else "low"
        out.append(
            ImageInput(
                image_id=image_id,
                data=base64.b64decode(encoded),
                mime_type=str(item.get("mime_type") or "image/jpeg"),
                detail=detail,
            )
        )
    return out


def _visual_analysis_budget(manifest: dict) -> dict:
    return {
        "image_count": _positive_int(manifest.get("images_analyzed"), 0),
        "attached_image_count": _positive_int(manifest.get("images_analyzed"), 0),
        "high_detail_image_count": _positive_int(manifest.get("high_detail_image_count"), 0),
        "evidence_only": True,
        "raw_images_for_planner": False,
    }


def _backend_sample_manifest(samples: list) -> list[dict]:
    out: list[dict] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "image_id": str(item.get("image_id") or ""),
                "class_name": str(item.get("class_name") or ""),
                "width": _positive_int(item.get("width"), 0),
                "height": _positive_int(item.get("height"), 0),
                "selection_basis": [
                    str(value)
                    for value in item.get("selection_basis", [])
                    if value not in (None, "")
                ][:12]
                if isinstance(item.get("selection_basis"), list)
                else [],
                "detail_level": str(item.get("detail_level") or ""),
                "metadata": {
                    "aspect_ratio": item.get("aspect_ratio"),
                    "prepared_width": item.get("prepared_width"),
                    "prepared_height": item.get("prepared_height"),
                    "prepared_bytes": item.get("prepared_bytes"),
                    "mime_type": item.get("mime_type"),
                },
            }
        )
    return out


def _manifest_summary(manifest: dict) -> dict:
    return {
        "schema_version": manifest.get("schema_version", ""),
        "selection_strategy": manifest.get("selection_strategy", ""),
        "selection_basis": manifest.get("selection_basis", []),
        "images_available": manifest.get("images_available", 0),
        "images_analyzed": manifest.get("images_analyzed", 0),
        "classes_total": manifest.get("classes_total", 0),
        "classes_covered": manifest.get("classes_covered", 0),
        "class_coverage_ratio": manifest.get("class_coverage_ratio", 0),
        "per_class_counts": manifest.get("per_class_counts", {}),
        "hard_example_count": manifest.get("hard_example_count", 0),
        "edge_case_count": manifest.get("edge_case_count", 0),
        "high_detail_image_count": manifest.get("high_detail_image_count", 0),
        "limitations": manifest.get("limitations", []),
    }


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default
