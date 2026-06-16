#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = REPO_ROOT / "services" / "worker"
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from worker.exporting.portable_bundle import (  # noqa: E402
    portable_bundle_summary,
    write_portable_inference_bundle,
)


class SmokeFailure(RuntimeError):
    pass


def run_smoke(
    *,
    export_root: Path | None = None,
    save_dir: Path | None = None,
    keep: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    owns_root = export_root is None
    root = export_root or Path(tempfile.mkdtemp(prefix="model-express-export-download-smoke-"))
    root = root.resolve()
    cleanup_after = owns_root and not keep

    try:
        result = _run_local_temp_smoke(root=root, save_dir=save_dir, verbose=verbose, cleanup_after=cleanup_after)
        result["artifacts_kept"] = not cleanup_after
    finally:
        if cleanup_after:
            shutil.rmtree(root, ignore_errors=True)
    return result


def resolve_artifact_location(record: dict[str, Any]) -> Path:
    value = _first_text(record, "artifact_path", "path", "artifact_uri", "uri")
    if not value:
        raise SmokeFailure("artifact record has no path or URI")

    if _looks_like_windows_path(value):
        return Path(value).expanduser().resolve()

    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        raise SmokeFailure(f"unsupported artifact URI scheme: {parsed.scheme}")

    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path))
    else:
        path = Path(value)
    return path.expanduser().resolve()


def assert_saved_zip_has_manifest(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists() or not zip_path.is_file():
        raise SmokeFailure(f"saved ZIP does not exist: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            if "manifest.json" not in names:
                raise SmokeFailure("saved ZIP is missing manifest.json")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except zipfile.BadZipFile as exc:
        raise SmokeFailure(f"saved artifact is not a valid ZIP: {zip_path}") from exc
    if not isinstance(manifest, dict):
        raise SmokeFailure("manifest.json is not a JSON object")
    return manifest


def _run_local_temp_smoke(*, root: Path, save_dir: Path | None, verbose: bool, cleanup_after: bool) -> dict[str, Any]:
    export_dir = root / "worker_export_root" / "export_v1_6_smoke"
    saved_dir = (save_dir or root / "saved_outside_app").resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)

    model_path = export_dir / "model.onnx"
    sidecar_path = export_dir / "model.onnx.data"
    model_path.write_bytes(b"fake onnx model bytes for export download smoke\n")
    sidecar_path.write_bytes(b"fake external tensor bytes\n")

    source_manifest = _champion_export_manifest(model_path=model_path, sidecar_path=sidecar_path)
    bundle_artifact = write_portable_inference_bundle(
        export_dir=export_dir,
        manifest=source_manifest,
        requested_format="onnx",
        provenance={"source_export_id": "export_v1_6_smoke", "export_job_id": "job_export_v1_6_smoke"},
    )
    bundle_metadata = portable_bundle_summary(bundle_artifact)

    project = {
        "id": "project_v1_6_smoke",
        "name": "V1-6 export download smoke",
        "dataset_id": "dataset_v1_demo_cats_dogs",
    }
    champion = {
        "id": "champion_v1_6_smoke",
        "project_id": project["id"],
        "job_id": "job_train_v1_6_smoke",
        "metric": "macro_f1",
        "score": 0.91,
    }
    export_record = {
        "id": "export_v1_6_smoke",
        "project_id": project["id"],
        "champion_id": champion["id"],
        "job_id": "job_export_v1_6_smoke",
        "status": "READY",
        "format": "onnx",
        "artifact_uri": model_path.resolve().as_uri(),
        "metadata": {
            "schema_version": "champion_export_result_v1",
            "portable_bundle_uri": bundle_metadata.get("artifact_uri", ""),
            "portable_inference_bundle": bundle_metadata,
            "artifacts": [bundle_artifact],
        },
    }

    portable_path = resolve_artifact_location(bundle_metadata)
    saved_zip = saved_dir / f"model-express-{project['id']}-{export_record['id']}-portable-bundle.zip"
    _copy_file(portable_path, saved_zip)
    saved_manifest = assert_saved_zip_has_manifest(saved_zip)

    checks = [
        _check("project exists", bool(project.get("id") and project.get("name"))),
        _check(
            "champion exists",
            bool(champion.get("id") and champion.get("project_id") == project.get("id")),
        ),
        _check(
            "export record exists",
            bool(
                export_record.get("id")
                and export_record.get("project_id") == project.get("id")
                and export_record.get("champion_id") == champion.get("id")
                and str(export_record.get("status")).upper() == "READY"
            ),
        ),
        _check(
            "portable bundle metadata exists",
            _portable_metadata_is_ready(bundle_metadata),
        ),
        _check("export artifact URI resolves", resolve_artifact_location({"artifact_uri": export_record["artifact_uri"]}).is_file()),
        _check("portable bundle path/URI resolves", portable_path.is_file()),
        _check("saved ZIP is outside export root", not _is_relative_to(saved_zip, export_dir)),
        _check("saved ZIP has manifest.json", saved_manifest.get("schema_version") == "portable_inference_manifest_v1"),
    ]
    failures = [item["name"] for item in checks if not item["ok"]]
    if failures:
        raise SmokeFailure("export download smoke failed: " + ", ".join(failures))

    if verbose:
        for item in checks:
            print(f"PASS {item['name']}")
        if cleanup_after:
            print(f"saved_zip={saved_zip} (validated, then removed; rerun with --keep to inspect)")
        else:
            print(f"saved_zip={saved_zip}")
        print(f"portable_bundle_uri={bundle_metadata.get('artifact_uri', '')}")

    return {
        "root": str(root),
        "export_dir": str(export_dir),
        "saved_zip": str(saved_zip),
        "project": project,
        "champion": champion,
        "export_record": export_record,
        "checks": checks,
    }


def _champion_export_manifest(*, model_path: Path, sidecar_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "champion_export_manifest_v1",
        "status": "created",
        "metadata": {
            "model_kind": "classification",
            "task_type": "image_classification",
            "runtime": "onnx",
            "input_shape": [1, 3, 8, 8],
            "class_labels": ["cat", "dog"],
            "class_label_order_hash": "v1-6-smoke-class-order",
            "confidence_threshold_defaults": {"classification": {"top_k": 5}},
            "preprocessing_contract": {
                "config": {"resize_strategy": "squash", "normalization": "none", "crop_strategy": "none"}
            },
            "postprocessing_contract": {"model_output": "logits", "activation": "softmax"},
            "inference_contract": {
                "input": {"model_tensor_shape": [1, 3, 8, 8]},
                "preprocessing": {
                    "config": {"resize_strategy": "squash", "normalization": "none", "crop_strategy": "none"}
                },
                "postprocessing": {"model_output": "logits", "activation": "softmax"},
            },
            "export_self_test": {
                "status": "passed",
                "runtime": "onnx",
                "sample_count": 1,
                "passed_count": 1,
                "failed_inference_count": 0,
                "class_label_order_hash": "v1-6-smoke-class-order",
                "preprocessing_contract_hash": "v1-6-smoke-preprocess",
                "samples": [
                    {
                        "sample_id": "heldout:dog",
                        "onnx": {"predicted_label": "dog", "top_k": [{"label": "dog", "confidence": 0.91}]},
                    }
                ],
            },
        },
        "artifacts": [
            {
                "format": "onnx",
                "status": "created",
                "path": str(model_path),
                "external_data": [{"path": "model.onnx.data", "artifact_path": str(sidecar_path), "bytes": sidecar_path.stat().st_size}],
            }
        ],
    }


def _portable_metadata_is_ready(metadata: dict[str, Any]) -> bool:
    return bool(
        metadata.get("schema_version") == "portable_inference_bundle_v1"
        and str(metadata.get("status")).lower() == "created"
        and (metadata.get("artifact_uri") or metadata.get("artifact_path"))
        and "manifest.json" in (metadata.get("contents") or [])
    )


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists() or not source.is_file():
        raise SmokeFailure(f"artifact path does not resolve to a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer)


def _check(name: str, ok: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok)}


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _looks_like_windows_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local-temp V1-6 export download smoke check.")
    parser.add_argument("--export-root", type=Path, default=None, help="Optional root for generated worker export artifacts.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional destination for the simulated saved ZIP.")
    parser.add_argument("--keep", action="store_true", help="Keep generated temp files for inspection.")
    parser.add_argument("--json", action="store_true", help="Print the smoke result as JSON.")
    args = parser.parse_args()

    try:
        result = run_smoke(export_root=args.export_root, save_dir=args.save_dir, keep=args.keep, verbose=not args.json)
    except SmokeFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
