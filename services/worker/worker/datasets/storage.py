from __future__ import annotations
import boto3
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlparse

def s3_client():
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if os.getenv("MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE", "").strip().lower() in {"1", "true", "yes", "on"}:
        if not access_key or not secret_key:
            raise ValueError("Scoped remote storage requires explicit AWS credentials.")
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=access_key or "model_express",
        aws_secret_access_key=secret_key or "model_express_password",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

def download_s3_uri(storage_uri: str, destination: Path) -> Path:
    bucket, key = parse_s3_uri(storage_uri)
    enforce_storage_scope("read", bucket, key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(bucket, key, str(destination))
    return destination

def upload_file_to_s3_uri(source: Path, storage_uri: str) -> str:
    bucket, key = parse_s3_uri(storage_uri)
    enforce_storage_scope("write", bucket, key)
    s3_client().upload_file(str(source), bucket, key)
    return storage_uri

def parse_s3_uri(storage_uri: str) -> tuple[str, str]:
    parsed = urlparse(storage_uri)

    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {storage_uri}")

    bucket = parsed.netloc
    key = normalize_storage_key(parsed.path.lstrip("/"))

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {storage_uri}")

    return bucket, key


def enforce_storage_scope(operation: str, bucket: str, key: str) -> None:
    scope = active_storage_scope()
    if not scope:
        if os.getenv("MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise ValueError("Remote storage scope is required.")
        return
    if storage_scope_expired(scope):
        raise ValueError("Remote storage scope has expired.")
    expected_token = str(scope.get("token") or "").strip()
    actual_token = os.getenv("MODEL_EXPRESS_STORAGE_SCOPE_TOKEN", "").strip()
    if expected_token and actual_token != expected_token:
        raise ValueError("Remote storage scope token mismatch.")
    buckets = {str(item).strip() for item in scope_values(scope, "buckets", "bucket") if str(item).strip()}
    if buckets and bucket not in buckets:
        raise ValueError("S3 bucket is outside the remote storage scope.")
    if operation == "read":
        if key_allowed(
            key,
            exact=scope_values(scope, "read_keys", "allowed_read_keys"),
            prefixes=scope_values(scope, "read_prefixes", "allowed_read_prefixes", "artifact_read_prefixes"),
        ):
            return
    elif operation == "write":
        if key_allowed(
            key,
            exact=scope_values(scope, "write_keys", "allowed_write_keys"),
            prefixes=scope_values(scope, "write_prefixes", "allowed_write_prefixes", "artifact_write_prefixes"),
        ):
            return
    else:
        raise ValueError(f"Unsupported storage scope operation: {operation}")
    raise ValueError(f"S3 {operation} key is outside the remote storage scope.")


def active_storage_scope() -> dict:
    raw = os.getenv("MODEL_EXPRESS_STORAGE_SCOPE", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid remote storage scope JSON.") from exc
    return payload if isinstance(payload, dict) else {}


def storage_scope_expired(scope: dict) -> bool:
    expires_at = str(scope.get("expires_at") or "").strip()
    if not expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= parsed.astimezone(timezone.utc)


def scope_values(scope: dict, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        value = scope.get(name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return values


def key_allowed(key: str, *, exact: list[str], prefixes: list[str]) -> bool:
    normalized = normalize_storage_key(key)
    exact_keys = {normalize_storage_key(item) for item in exact if str(item or "").strip()}
    if normalized in exact_keys:
        return True
    for prefix in prefixes:
        if not str(prefix or "").strip():
            continue
        normalized_prefix = normalize_storage_key(prefix)
        if normalized == normalized_prefix or normalized.startswith(normalized_prefix + "/"):
            return True
    return False


def normalize_storage_key(value: str) -> str:
    raw = unquote(str(value or "").replace("\\", "/")).strip()
    raw = raw.lstrip("/")
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(str(value or ""))
    if not raw or posix_path.is_absolute() or windows_path.is_absolute():
        raise ValueError("S3 key is unsafe or empty.")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("S3 key contains path traversal.")
    return "/".join(posix_path.parts)
