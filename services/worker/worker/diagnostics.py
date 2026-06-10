from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_LOG_STRING_LENGTH = 1800
MAX_LOG_ARRAY_ITEMS = 24
MAX_LOG_DICT_ITEMS = 80
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_MAX_FILES = 5
_WRITE_LOCK = threading.Lock()
_UNSAFE_TEXT_RE = re.compile(
    r"(data:image/[a-z0-9.+-]+;base64,|base64\s*[:=,]|bearer\s+[a-z0-9._\-]+|"
    r"aws_access_key|[A-Za-z]:\\|file://|s3://|/Users/|/home/|/tmp/|\\\\|"
    r"/9j/4AAQSkZJRg|iVBORw0KGgo)",
    re.IGNORECASE,
)
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "base64",
    "credential",
    "image",
    "password",
    "prompt",
    "raw_output",
    "secret",
    "storage_uri",
    "token",
    "uri",
    "url",
)


def log_event(level: str, event: str, **fields: Any) -> None:
    if not str(event or "").strip():
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": _level(level),
        "component": "worker",
        "event": event,
    }
    for key, value in fields.items():
        if not str(key or "").strip():
            continue
        record[key] = _safe_value(key, value)

    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        directory = _log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "worker.jsonl"
        with _WRITE_LOCK:
            _rotate_jsonl(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        return


def _log_dir() -> Path:
    configured = os.getenv("MODEL_EXPRESS_LOG_DIR", "").strip()
    if configured:
        return Path(configured)
    root = os.getenv("MODEL_EXPRESS_ROOT", "").strip()
    if root:
        return Path(root) / "artifacts" / "logs"
    return Path(__file__).resolve().parents[3] / "artifacts" / "logs"


def _level(level: str) -> str:
    normalized = str(level or "").strip().lower()
    return normalized if normalized in {"debug", "info", "warn", "error"} else "info"


def _safe_value(key: str, value: Any, depth: int = 0) -> Any:
    if _sensitive_key(key):
        return "[redacted]"
    if depth > 4:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, BaseException):
        return _safe_text(str(value))
    if isinstance(value, dict):
        out = {}
        for index, (child_key, child_value) in enumerate(value.items()):
            if index >= MAX_LOG_DICT_ITEMS:
                out["_truncated"] = True
                break
            key_text = str(child_key)
            out[key_text] = _safe_value(key_text, child_value, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_safe_value("", item, depth + 1) for item in list(value)[:MAX_LOG_ARRAY_ITEMS]]
    return _safe_text(str(value))


def _safe_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _UNSAFE_TEXT_RE.sub("[redacted]", text)
    if len(text) > MAX_LOG_STRING_LENGTH:
        text = text[: MAX_LOG_STRING_LENGTH - 1].rstrip() + "..."
    return text


def _sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _rotate_jsonl(path: Path) -> None:
    max_bytes = _positive_int_env("MODEL_EXPRESS_LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES)
    max_files = _positive_int_env("MODEL_EXPRESS_LOG_MAX_FILES", DEFAULT_LOG_MAX_FILES)
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for index in range(max_files - 1, 0, -1):
            source = path.with_name(f"{path.name}.{index}")
            destination = path.with_name(f"{path.name}.{index + 1}")
            if destination.exists() and index + 1 > max_files:
                destination.unlink(missing_ok=True)
            if source.exists():
                if index + 1 > max_files:
                    source.unlink(missing_ok=True)
                else:
                    source.replace(destination)
        first = path.with_name(f"{path.name}.1")
        path.replace(first)
    except Exception:
        return


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return parsed if parsed > 0 else int(default)
