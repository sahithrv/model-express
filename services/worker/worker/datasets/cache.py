from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable
from urllib.parse import urlparse


CACHE_ROOT = Path(".cache/datasets")
SCRATCH_ROOT = Path(tempfile.gettempdir()) / "model-express-worker-datasets"
MATERIALIZED_ROOT = Path(
    os.getenv("MODEL_EXPRESS_DATASET_MATERIALIZED_ROOT", "/cache/model-express/datasets")
)
LOCK_POLL_SECONDS = 0.25
LOCK_TIMEOUT_SECONDS = 60 * 60
LOCK_STALE_SECONDS = 60 * 60
PROFILE_CACHE_MAX_BYTES = 1_000_000


def dataset_cache_root(cache_root: Path | str | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    return Path(os.getenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(CACHE_ROOT)))


def dataset_cache_dir(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_root(cache_root) / _safe_dataset_id(dataset_id)


def dataset_archive_path(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_dir(dataset_id, cache_root) / "archive.zip"


def dataset_extract_dir(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_dir(dataset_id, cache_root) / "extracted"


def extract_dataset_archive(
    archive_path: Path,
    dataset_id: str,
    cache_root: Path | str | None = None,
) -> Path:
    extract_dir = dataset_extract_dir(dataset_id, cache_root)
    cache_dir = dataset_cache_dir(dataset_id, cache_root)

    if extract_dir.exists():
        return extract_dir

    temp_dir = cache_dir / "extracting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract_zip(archive, temp_dir)

    if extract_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        return extract_dir
    try:
        temp_dir.replace(extract_dir)
    except PermissionError:
        if extract_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return extract_dir
        raise
    return extract_dir


@dataclass(frozen=True)
class DatasetMaterializationResult:
    dataset_dir: Path
    telemetry: dict


def ensure_dataset_materialized(
    *,
    dataset_id: str,
    storage_uri: str,
    checksum_sha256: str | None = None,
    cache_root: Path | str | None = None,
    download_fn: Callable[[str, Path], Path] | None = None,
) -> DatasetMaterializationResult:
    root = Path(cache_root) if cache_root is not None else MATERIALIZED_ROOT
    key = dataset_materialization_cache_key(checksum_sha256, storage_uri)
    storage_fingerprint = storage_uri_fingerprint(storage_uri)
    cache_dir = root / key
    archive_path = cache_dir / "archive.zip"
    extracted_dir = cache_dir / "extracted"
    complete_marker = cache_dir / ".complete"
    manifest_path = cache_dir / "manifest.json"
    shards_manifest_path = cache_dir / "shards" / "manifest.json"
    if download_fn is None:
        from worker.datasets.storage import download_s3_uri

        downloader = download_s3_uri
    else:
        downloader = download_fn
    telemetry = _materialization_telemetry(
        cache_hit=False,
        cache_key=key,
        checksum_sha256=checksum_sha256,
        storage_fingerprint=storage_fingerprint,
    )
    materialization_started = time.perf_counter()

    if _dataset_materialization_complete(extracted_dir, complete_marker):
        telemetry["dataset_materialization_cache_hit"] = True
        telemetry["dataset_materialization_status"] = "hit"
        telemetry["dataset_materialization_total_seconds"] = round(time.perf_counter() - materialization_started, 6)
        return DatasetMaterializationResult(extracted_dir, telemetry)

    cache_dir.mkdir(parents=True, exist_ok=True)
    wait_started = time.perf_counter()
    lock_dir = _acquire_materialization_lock(cache_dir)
    telemetry["dataset_materialization_wait_seconds"] = round(time.perf_counter() - wait_started, 6)
    try:
        if _dataset_materialization_complete(extracted_dir, complete_marker):
            telemetry["dataset_materialization_cache_hit"] = True
            telemetry["dataset_materialization_status"] = "hit_after_wait"
            telemetry["dataset_materialization_total_seconds"] = round(time.perf_counter() - materialization_started, 6)
            return DatasetMaterializationResult(extracted_dir, telemetry)

        telemetry["dataset_materialization_cache_miss"] = True
        telemetry["dataset_materialization_status"] = "materializing"
        shards_enabled = _dataset_shards_enabled()
        telemetry["dataset_materialization_shards_enabled"] = shards_enabled
        if shards_enabled and shards_manifest_path.is_file():
            shard_started = time.perf_counter()
            _reset_incomplete_materialization(extracted_dir, complete_marker, archive_path)
            staging_dir = cache_dir / f"shard-materializing-{os.getpid()}-{time.time_ns()}"
            try:
                from worker.datasets.shards import materialize_shard_artifacts

                shard_manifest = materialize_shard_artifacts(
                    manifest_path=shards_manifest_path,
                    output_dir=staging_dir,
                    dataset_checksum=checksum_sha256,
                )
                if extracted_dir.exists():
                    shutil.rmtree(extracted_dir)
                staging_dir.replace(extracted_dir)
                manifest = _materialization_manifest(
                    dataset_id=dataset_id,
                    checksum_sha256=checksum_sha256,
                    storage_uri=storage_uri,
                    storage_fingerprint=storage_fingerprint,
                    cache_key=key,
                    bytes_downloaded=0,
                    shard_manifest=shard_manifest,
                )
                manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                complete_marker.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                telemetry.update(_shard_telemetry(shard_manifest))
                telemetry["dataset_materialization_shard_status"] = "materialized_from_existing_shards"
                telemetry["dataset_materialization_shard_seconds"] = round(time.perf_counter() - shard_started, 6)
                telemetry["dataset_materialization_status"] = "materialized_from_shards"
                telemetry["dataset_materialization_total_seconds"] = round(time.perf_counter() - materialization_started, 6)
                return DatasetMaterializationResult(extracted_dir, telemetry)
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)

        _reset_incomplete_materialization(extracted_dir, complete_marker, archive_path)
        download_started = time.perf_counter()
        downloaded_path = downloader(storage_uri, archive_path)
        telemetry["dataset_materialization_download_seconds"] = round(
            time.perf_counter() - download_started,
            6,
        )
        downloaded_path = Path(downloaded_path) if downloaded_path is not None else archive_path
        telemetry["dataset_materialization_bytes_downloaded"] = _path_size(downloaded_path)

        staging_dir = cache_dir / f"extracting-{os.getpid()}-{time.time_ns()}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        extract_started = time.perf_counter()
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract_zip(archive, staging_dir)
        telemetry["dataset_materialization_extract_seconds"] = round(
            time.perf_counter() - extract_started,
            6,
        )

        shard_manifest = None
        if shards_enabled:
            shard_started = time.perf_counter()
            from worker.datasets.shards import create_shard_artifacts, materialize_shard_artifacts

            shard_manifest = create_shard_artifacts(
                dataset_dir=staging_dir,
                artifact_root=cache_dir / "shards",
                dataset_checksum=checksum_sha256,
                cache_key=key,
                storage_uri=storage_uri,
            )
            shard_materialized_dir = cache_dir / f"shard-extracting-{os.getpid()}-{time.time_ns()}"
            try:
                materialize_shard_artifacts(
                    manifest_path=shards_manifest_path,
                    output_dir=shard_materialized_dir,
                    dataset_checksum=checksum_sha256,
                )
                shutil.rmtree(staging_dir)
                staging_dir = shard_materialized_dir
                telemetry.update(_shard_telemetry(shard_manifest))
                telemetry["dataset_materialization_shard_status"] = "created_and_materialized"
                telemetry["dataset_materialization_shard_seconds"] = round(time.perf_counter() - shard_started, 6)
            except Exception:
                shutil.rmtree(shard_materialized_dir, ignore_errors=True)
                raise

        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        staging_dir.replace(extracted_dir)
        manifest = _materialization_manifest(
            dataset_id=dataset_id,
            checksum_sha256=checksum_sha256,
            storage_uri=storage_uri,
            storage_fingerprint=storage_fingerprint,
            cache_key=key,
            bytes_downloaded=telemetry["dataset_materialization_bytes_downloaded"],
            shard_manifest=shard_manifest,
        )
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        complete_marker.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        telemetry["dataset_materialization_status"] = "materialized"
        telemetry["dataset_materialization_total_seconds"] = round(time.perf_counter() - materialization_started, 6)
        return DatasetMaterializationResult(extracted_dir, telemetry)
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def dataset_materialization_cache_key(checksum_sha256: str | None, storage_uri: str) -> str:
    checksum = _normalized_checksum(checksum_sha256)
    if checksum:
        return f"sha256-{checksum[:64]}"
    return f"uri-{storage_uri_fingerprint(storage_uri)}"


def storage_uri_fingerprint(storage_uri: str) -> str:
    normalized = _normalized_storage_uri(storage_uri)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dataset_profile_cache_key(
    checksum_sha256: str | None,
    storage_uri: str,
    profile_cache_version: str,
) -> str:
    checksum = _normalized_checksum(checksum_sha256)
    version = _safe_cache_component(profile_cache_version)
    if checksum:
        return f"{version}-sha256-{checksum[:64]}"
    if _profile_cache_uri_fallback_enabled():
        return f"{version}-uri-{storage_uri_fingerprint(storage_uri)}"
    return ""


def load_cached_dataset_profile(
    *,
    checksum_sha256: str | None,
    storage_uri: str,
    profile_cache_version: str,
    cache_root: Path | str | None = None,
) -> dict | None:
    key = dataset_profile_cache_key(checksum_sha256, storage_uri, profile_cache_version)
    if not key:
        return None
    path = _dataset_profile_cache_path(key, cache_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("profile_cache_version") != profile_cache_version:
        return None
    profile = payload.get("profile")
    return profile if isinstance(profile, dict) else None


def save_cached_dataset_profile(
    *,
    checksum_sha256: str | None,
    storage_uri: str,
    profile_cache_version: str,
    profile: dict,
    cache_root: Path | str | None = None,
    max_bytes: int | None = None,
) -> bool:
    key = dataset_profile_cache_key(checksum_sha256, storage_uri, profile_cache_version)
    if not key or not isinstance(profile, dict):
        return False
    payload = {
        "profile_cache_version": profile_cache_version,
        "dataset_checksum": _normalized_checksum(checksum_sha256),
        "storage_uri_fingerprint": storage_uri_fingerprint(storage_uri),
        "created_at_unix": time.time(),
        "profile": profile,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    byte_limit = _positive_int_env(
        "MODEL_EXPRESS_PROFILE_CACHE_MAX_BYTES",
        PROFILE_CACHE_MAX_BYTES if max_bytes is None else max_bytes,
    )
    if len(encoded.encode("utf-8")) > byte_limit:
        return False
    path = _dataset_profile_cache_path(key, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(encoded, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return True


def prune_cache_children(
    root: Path | str,
    *,
    max_age_seconds: float | None = None,
    max_bytes: int | None = None,
    apply: bool = False,
) -> dict:
    root_path = Path(root)
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError:
        return {"root": str(root_path), "removed": [], "would_remove": [], "bytes_before": 0, "bytes_after": 0}

    children = _cache_children(resolved_root)
    now = time.time()
    candidates = []
    bytes_before = 0
    for child in children:
        size = _path_tree_size(child)
        bytes_before += size
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = now
        if child.name.endswith(".lock") or child.name.startswith("."):
            continue
        if max_age_seconds is not None and max_age_seconds >= 0 and now - mtime > max_age_seconds:
            candidates.append((mtime, size, child))

    bytes_after = bytes_before
    if max_bytes is not None and max_bytes >= 0 and bytes_after > max_bytes:
        by_age = sorted(((_child_mtime(child), _path_tree_size(child), child) for child in children), key=lambda item: item[0])
        seen = {str(path) for _, _, path in candidates}
        for item in by_age:
            if bytes_after <= max_bytes:
                break
            child = item[2]
            if child.name.endswith(".lock") or child.name.startswith(".") or str(child) in seen:
                continue
            candidates.append(item)
            seen.add(str(child))
            bytes_after -= item[1]

    removed: list[str] = []
    would_remove: list[str] = []
    dry_removed_bytes = 0
    for _, size, child in sorted(candidates, key=lambda item: item[0]):
        if not _is_relative_to(child, resolved_root):
            continue
        if apply:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed.append(str(child))
        else:
            would_remove.append(str(child))
            dry_removed_bytes += size

    if apply:
        bytes_after = _path_tree_size(resolved_root)
    else:
        bytes_after = bytes_before - dry_removed_bytes
    return {
        "root": str(resolved_root),
        "removed": removed,
        "would_remove": would_remove,
        "bytes_before": bytes_before,
        "bytes_after": max(0, bytes_after),
    }


def _dataset_materialization_complete(extracted_dir: Path, complete_marker: Path) -> bool:
    return complete_marker.exists() and extracted_dir.is_dir()


def _dataset_profile_cache_path(key: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_root(cache_root) / "_profiles" / f"{_safe_cache_component(key)}.json"


def _profile_cache_uri_fallback_enabled() -> bool:
    value = os.getenv("MODEL_EXPRESS_PROFILE_CACHE_URI_FALLBACK", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def cleanup_dataset_cache(dataset_id: str, cache_root: Path | str | None = None) -> None:
    shutil.rmtree(dataset_cache_dir(dataset_id, cache_root), ignore_errors=True)


def job_dataset_cache_root(job_id: str | None = None, cache_root: Path | str | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    if should_persist_dataset_cache():
        return dataset_cache_root()
    configured_root = os.getenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", "").strip()
    base_root = Path(configured_root) if configured_root else SCRATCH_ROOT
    safe_job_id = _safe_cache_component(job_id or str(os.getpid()))
    return base_root / "_jobs" / safe_job_id


def cleanup_job_dataset_cache(
    job_id: str | None = None,
    cache_root: Path | str | None = None,
) -> None:
    root = job_dataset_cache_root(job_id, cache_root)
    if should_persist_dataset_cache() and cache_root is None:
        return
    shutil.rmtree(root, ignore_errors=True)


def should_persist_dataset_cache() -> bool:
    value = os.getenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _acquire_materialization_lock(cache_dir: Path) -> Path:
    lock_dir = cache_dir / ".materializing.lock"
    started_at = time.monotonic()
    while True:
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            if _materialization_lock_is_stale(lock_dir):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            if time.monotonic() - started_at > _materialization_lock_timeout_seconds():
                raise TimeoutError(
                    f"Timed out waiting for dataset materialization lock: {lock_dir}"
                )
            time.sleep(LOCK_POLL_SECONDS)


def _materialization_telemetry(
    *,
    cache_hit: bool,
    cache_key: str,
    checksum_sha256: str | None,
    storage_fingerprint: str,
) -> dict:
    checksum = _normalized_checksum(checksum_sha256)
    return {
        "dataset_materialization_cache_hit": cache_hit,
        "dataset_materialization_cache_miss": False,
        "dataset_materialization_bytes_downloaded": 0,
        "dataset_materialization_download_seconds": 0.0,
        "dataset_materialization_extract_seconds": 0.0,
        "dataset_materialization_wait_seconds": 0.0,
        "dataset_materialization_total_seconds": 0.0,
        "dataset_materialization_status": "checking",
        "dataset_materialization_shards_enabled": False,
        "dataset_materialization_shard_status": "disabled",
        "dataset_materialization_shard_count": 0,
        "dataset_materialization_shard_file_count": 0,
        "dataset_materialization_shard_seconds": 0.0,
        "dataset_checksum": checksum,
        "storage_uri_fingerprint": storage_fingerprint,
        "dataset_materialization_cache_key": cache_key,
    }


def _materialization_manifest(
    *,
    dataset_id: str,
    checksum_sha256: str | None,
    storage_uri: str,
    storage_fingerprint: str,
    cache_key: str,
    bytes_downloaded: int,
    shard_manifest: dict | None,
) -> dict:
    manifest = {
        "dataset_id": dataset_id,
        "dataset_checksum": _normalized_checksum(checksum_sha256),
        "storage_uri": storage_uri,
        "storage_uri_fingerprint": storage_fingerprint,
        "cache_key": cache_key,
        "bytes_downloaded": bytes_downloaded,
        "completed_at_unix": time.time(),
    }
    if shard_manifest is not None:
        manifest["shard_artifact"] = {
            "schema_version": shard_manifest.get("schema_version"),
            "dataset_checksum": shard_manifest.get("dataset_checksum"),
            "task_type": shard_manifest.get("task_type"),
            "shard_format": shard_manifest.get("shard_format"),
            "file_counts": shard_manifest.get("file_counts"),
            "split_counts": shard_manifest.get("split_counts"),
            "class_counts": shard_manifest.get("class_counts"),
            "object_counts": shard_manifest.get("object_counts"),
            "shard_count": len(shard_manifest.get("shards") or []),
        }
    return manifest


def _shard_telemetry(shard_manifest: dict) -> dict:
    file_counts = shard_manifest.get("file_counts") if isinstance(shard_manifest, dict) else {}
    return {
        "dataset_materialization_shard_count": len(shard_manifest.get("shards") or []),
        "dataset_materialization_shard_file_count": int((file_counts or {}).get("total") or 0),
    }


def _dataset_shards_enabled() -> bool:
    try:
        from worker.datasets.shards import dataset_shards_enabled

        return dataset_shards_enabled()
    except Exception:
        return False


def _reset_incomplete_materialization(
    extracted_dir: Path,
    complete_marker: Path,
    archive_path: Path,
) -> None:
    complete_marker.unlink(missing_ok=True)
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    archive_path.unlink(missing_ok=True)


def _path_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _path_tree_size(path: Path) -> int:
    if path.is_file():
        return _path_size(path)
    total = 0
    try:
        iterator = path.rglob("*")
        for child in iterator:
            if child.is_file():
                total += _path_size(child)
    except OSError:
        return total
    return total


def _cache_children(root: Path) -> list[Path]:
    try:
        return [child for child in root.iterdir()]
    except OSError:
        return []


def _child_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    root = Path(destination).resolve(strict=False)
    for member in archive.infolist():
        relative_path = _safe_zip_member_path(member.filename)
        target = (root / Path(*relative_path.parts)).resolve(strict=False)
        if not _is_relative_to(target, root):
            raise ValueError(f"Unsafe dataset archive member path: {member.filename!r}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def _safe_zip_member_path(name: str) -> PurePosixPath:
    normalized = str(name or "").replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(str(name or ""))
    if not normalized.strip() or posix_path.is_absolute() or windows_path.is_absolute():
        raise ValueError(f"Unsafe dataset archive member path: {name!r}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"Unsafe dataset archive member path: {name!r}")
    return posix_path


def _normalized_checksum(checksum_sha256: str | None) -> str:
    value = str(checksum_sha256 or "").strip().lower()
    if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
        return value
    return ""


def _normalized_storage_uri(storage_uri: str) -> str:
    parsed = urlparse(str(storage_uri).strip())
    if parsed.scheme == "s3":
        return f"s3://{parsed.netloc.lower()}/{parsed.path.lstrip('/')}"
    return str(storage_uri).strip()


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return max(1, int(default))
    try:
        parsed = int(value)
    except ValueError:
        return max(1, int(default))
    return parsed if parsed > 0 else max(1, int(default))


def _materialization_lock_timeout_seconds() -> float:
    value = os.getenv("MODEL_EXPRESS_DATASET_MATERIALIZATION_LOCK_TIMEOUT_SECONDS", "").strip()
    if not value:
        return float(LOCK_TIMEOUT_SECONDS)
    try:
        parsed = float(value)
    except ValueError:
        return float(LOCK_TIMEOUT_SECONDS)
    return parsed if parsed > 0 else float(LOCK_TIMEOUT_SECONDS)


def _materialization_lock_stale_seconds() -> float:
    value = os.getenv("MODEL_EXPRESS_DATASET_MATERIALIZATION_LOCK_STALE_SECONDS", "").strip()
    if not value:
        return float(LOCK_STALE_SECONDS)
    try:
        parsed = float(value)
    except ValueError:
        return float(LOCK_STALE_SECONDS)
    return parsed if parsed > 0 else float(LOCK_STALE_SECONDS)


def _materialization_lock_is_stale(lock_dir: Path) -> bool:
    try:
        return time.time() - lock_dir.stat().st_mtime > _materialization_lock_stale_seconds()
    except OSError:
        return False


def _safe_dataset_id(dataset_id: str) -> str:
    normalized = str(dataset_id).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("dataset_id is required for dataset cache paths")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"dataset_id must not contain path separators: {dataset_id!r}")
    return normalized


def _safe_cache_component(value: str) -> str:
    normalized = str(value).strip().replace("\\", "_").replace("/", "_")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in normalized)
    return normalized.strip("._") or "worker"
