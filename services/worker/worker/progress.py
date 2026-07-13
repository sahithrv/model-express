"""Best-effort, attempt-scoped worker progress reporting.

Progress is operational telemetry, never part of the training success path. The
reporter therefore validates and bounds its own payloads, retries only errors
that can be transient, and converts every reporting failure into a small local
diagnostic instead of raising into provider code.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import requests

from worker.diagnostics import log_event
from worker.orchestrator_client import (
    ENDPOINT_UNAVAILABLE_STATUS_CODES,
    OrchestratorClient,
    _callback_token,
    _training_attempt_id,
)


PROGRESS_TAXONOMY_VERSION = 1
PROGRESS_REPORTING_ENABLED_ENV = "MODEL_EXPRESS_PROGRESS_REPORTING_ENABLED"
PROGRESS_HEARTBEAT_INTERVAL_ENV = "MODEL_EXPRESS_PROGRESS_HEARTBEAT_SECONDS"
PROGRESS_TIMEOUT_ENV = "MODEL_EXPRESS_PROGRESS_REPORT_TIMEOUT_SECONDS"
PROGRESS_MAX_ATTEMPTS_ENV = "MODEL_EXPRESS_PROGRESS_REPORT_MAX_ATTEMPTS"

DEFAULT_PROGRESS_HEARTBEAT_SECONDS = 15.0
DEFAULT_PROGRESS_TIMEOUT_SECONDS = 5.0
DEFAULT_PROGRESS_MAX_ATTEMPTS = 3
DEFAULT_PROGRESS_INITIAL_REVISION = 2
DEFAULT_RETRY_INITIAL_SECONDS = 0.25
DEFAULT_RETRY_MAX_SECONDS = 2.0
MAX_PROGRESS_HEARTBEAT_SECONDS = 60 * 60.0
MAX_PROGRESS_TIMEOUT_SECONDS = 30.0
MAX_PROGRESS_REPORT_ATTEMPTS = 5
MAX_RETRY_DELAY_SECONDS = 5.0
MAX_PROGRESS_REVISION = (1 << 63) - 2
MAX_MESSAGE_BYTES = 512
MAX_DETAIL_CODE_BYTES = 64
MAX_UNIT_BYTES = 32
MAX_METADATA_TOKEN_BYTES = 64
MAX_PAYLOAD_BYTES = 4096
MAX_DIAGNOSTICS = 32

WORKER_PROGRESS_STAGES = frozenset(
    {
        "worker_starting",
        "remote_scheduled",
        "environment_starting",
        "dataset_materializing",
        "data_loading",
        "model_initializing",
        "training",
        "evaluating",
        "exporting",
        "finalizing",
    }
)

PROGRESS_METADATA_KEYS = frozenset(
    {
        "provider",
        "framework",
        "execution_mode",
        "cache_status",
        "task_type",
        "resource_class",
        "early_stopped",
    }
)
PROGRESS_UNITS = frozenset(
    {
        "item",
        "items",
        "epoch",
        "epochs",
        "step",
        "steps",
        "batch",
        "batches",
        "sample",
        "samples",
        "byte",
        "bytes",
        "percent",
    }
)

_SAFE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_UNSAFE_TEXT_RE = re.compile(
    r"(?:\b(?:s3|gs|https?|file)://|[a-z]:\\|(?:^|\s)/(?:\S+)|"
    r"\b(?:bearer|authorization)\s+|\b(?:sk|pk)-[a-z0-9_-]{8,})",
    re.IGNORECASE,
)
_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429})
_AUTH_HTTP_STATUS_CODES = frozenset({401, 403})


class ProgressReporter:
    """Report safe progress observations without affecting training outcome.

    ``initial_revision`` reserves an attempt-wide revision prefix for callers
    that hand work between processes. The backend assignment snapshot uses
    revision 2, so the default first worker observation is revision 3. A remote
    provider may use a larger seed after its local submission observations.
    """

    def __init__(
        self,
        client: OrchestratorClient,
        job: dict,
        *,
        enabled: bool | None = None,
        heartbeat_interval_seconds: float | None = None,
        max_attempts: int | None = None,
        timeout_seconds: float | None = None,
        initial_revision: int = DEFAULT_PROGRESS_INITIAL_REVISION,
        retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = client
        self._job = _progress_identity_job(job)
        self._job_id = str(self._job.get("id") or "")
        self._enabled = (
            progress_reporting_enabled() if enabled is None else bool(enabled)
        )
        self._heartbeat_interval_seconds = _configured_positive_float(
            heartbeat_interval_seconds,
            PROGRESS_HEARTBEAT_INTERVAL_ENV,
            DEFAULT_PROGRESS_HEARTBEAT_SECONDS,
            maximum=MAX_PROGRESS_HEARTBEAT_SECONDS,
        )
        self._max_attempts = _configured_positive_int(
            max_attempts,
            PROGRESS_MAX_ATTEMPTS_ENV,
            DEFAULT_PROGRESS_MAX_ATTEMPTS,
            maximum=MAX_PROGRESS_REPORT_ATTEMPTS,
        )
        self._timeout_seconds = _configured_positive_float(
            timeout_seconds,
            PROGRESS_TIMEOUT_ENV,
            DEFAULT_PROGRESS_TIMEOUT_SECONDS,
            maximum=MAX_PROGRESS_TIMEOUT_SECONDS,
        )
        self._retry_initial_seconds = _positive_float_or_default(
            retry_initial_seconds,
            DEFAULT_RETRY_INITIAL_SECONDS,
        )
        self._retry_max_seconds = max(
            self._retry_initial_seconds,
            _positive_float_or_default(retry_max_seconds, DEFAULT_RETRY_MAX_SECONDS),
        )
        self._retry_max_seconds = min(self._retry_max_seconds, MAX_RETRY_DELAY_SECONDS)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._diagnostics: deque[dict[str, Any]] = deque(maxlen=MAX_DIAGNOSTICS)
        self._revision = _valid_initial_revision(initial_revision)
        self._last_signature = ""
        self._last_report_at: float | None = None
        self._suspended_reason = ""

        if not self._job_id or not _training_attempt_id(self._job):
            self._suspend("missing_attempt_identity")
        elif self._revision is None:
            self._revision = 0
            self._suspend("invalid_initial_revision")

    @classmethod
    def from_orchestrator_url(
        cls,
        orchestrator_url: str,
        job: dict,
        **options: Any,
    ) -> "ProgressReporter":
        """Build a reporter inside a remote provider process.

        The identity-only job snapshot keeps the original attempt ID and
        callback token while excluding the rest of the job payload.
        """
        timeout = options.get("timeout_seconds")
        client_timeout = (
            _positive_float_or_default(timeout, DEFAULT_PROGRESS_TIMEOUT_SECONDS)
            if timeout is not None
            else _positive_float_env(PROGRESS_TIMEOUT_ENV, DEFAULT_PROGRESS_TIMEOUT_SECONDS)
        )
        client_timeout = min(client_timeout, MAX_PROGRESS_TIMEOUT_SECONDS)
        return cls(OrchestratorClient(orchestrator_url, timeout=client_timeout), job, **options)

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._suspended_reason

    @property
    def revision(self) -> int:
        return int(self._revision or 0)

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Return bounded reason-code diagnostics with no request contents."""
        with self._lock:
            return tuple(dict(item) for item in self._diagnostics)

    def report(
        self,
        stage: str,
        *,
        status: str = "running",
        current: int | None = None,
        total: int | None = None,
        unit: str = "",
        message: str = "",
        detail_code: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Send one observation, returning whether the backend accepted it.

        Invalid input, throttling, unsupported endpoints, authentication
        rejection, timeouts, and outages all return ``False`` and never raise.
        """
        with self._lock:
            if not self._enabled or self._suspended_reason:
                return False

            try:
                payload = _progress_payload(
                    stage,
                    status=status,
                    current=current,
                    total=total,
                    unit=unit,
                    message=message,
                    detail_code=detail_code,
                    metadata=metadata,
                )
            except Exception:
                self._record_diagnostic("invalid_payload")
                return False

            signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            now = _safe_clock(self._clock)
            if (
                signature == self._last_signature
                and self._last_report_at is not None
                and now - self._last_report_at < self._heartbeat_interval_seconds
            ):
                return False

            if self.revision >= MAX_PROGRESS_REVISION:
                self._suspend("revision_exhausted")
                return False

            self._revision = self.revision + 1
            revision = self.revision
            payload["revision"] = revision
            self._last_signature = signature
            self._last_report_at = now
            return self._send(payload, stage=str(payload["stage"]), revision=revision)

    def _send(self, payload: dict[str, Any], *, stage: str, revision: int) -> bool:
        last_reason = "reporting_error"
        last_status_code: int | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._client.report_progress(
                    self._job_id,
                    payload,
                    job=self._job,
                    timeout=self._timeout_seconds,
                )
                if isinstance(result, dict) and result.get("status") == "unavailable":
                    status_code = _optional_status_code(result.get("status_code"))
                    self._suspend(
                        "endpoint_unsupported",
                        stage=stage,
                        revision=revision,
                        attempts=attempt,
                        status_code=status_code,
                    )
                    return False
                return True
            except requests.HTTPError as exc:
                status_code = _http_status_code(exc)
                last_status_code = status_code
                if status_code in ENDPOINT_UNAVAILABLE_STATUS_CODES:
                    self._suspend(
                        "endpoint_unsupported",
                        stage=stage,
                        revision=revision,
                        attempts=attempt,
                        status_code=status_code,
                    )
                    return False
                if status_code in _AUTH_HTTP_STATUS_CODES:
                    self._suspend(
                        "authentication_rejected",
                        stage=stage,
                        revision=revision,
                        attempts=attempt,
                        status_code=status_code,
                    )
                    return False
                if not _transient_http_status(status_code):
                    self._record_diagnostic(
                        "request_rejected",
                        stage=stage,
                        revision=revision,
                        attempts=attempt,
                        status_code=status_code,
                    )
                    return False
                last_reason = "transient_http_error"
            except requests.Timeout:
                last_reason = "timeout"
            except requests.ConnectionError:
                last_reason = "connection_error"
            except requests.RequestException:
                self._record_diagnostic(
                    "request_error",
                    stage=stage,
                    revision=revision,
                    attempts=attempt,
                )
                return False
            except Exception:
                self._record_diagnostic(
                    "reporter_error",
                    stage=stage,
                    revision=revision,
                    attempts=attempt,
                )
                return False

            if attempt < self._max_attempts:
                try:
                    self._sleep(self._retry_delay(attempt))
                except Exception:
                    self._record_diagnostic(
                        "retry_interrupted",
                        stage=stage,
                        revision=revision,
                        attempts=attempt,
                    )
                    return False

        self._record_diagnostic(
            f"{last_reason}_exhausted",
            stage=stage,
            revision=revision,
            attempts=self._max_attempts,
            status_code=last_status_code,
        )
        return False

    def _retry_delay(self, completed_attempts: int) -> float:
        return min(
            self._retry_max_seconds,
            self._retry_initial_seconds * (2 ** max(0, completed_attempts - 1)),
        )

    def _suspend(self, reason: str, **fields: Any) -> None:
        self._suspended_reason = reason
        self._record_diagnostic(reason, **fields)

    def _record_diagnostic(self, reason: str, **fields: Any) -> None:
        diagnostic: dict[str, Any] = {"reason": str(reason)}
        for key in ("stage", "revision", "attempts", "status_code"):
            value = fields.get(key)
            if value is not None:
                diagnostic[key] = value
        self._diagnostics.append(diagnostic)
        log_event("warn", "progress_reporting_degraded", **diagnostic)


def progress_reporting_enabled() -> bool:
    value = os.getenv(PROGRESS_REPORTING_ENABLED_ENV, "").strip().lower()
    if not value:
        return True
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return True


def _progress_payload(
    stage: str,
    *,
    status: str,
    current: int | None,
    total: int | None,
    unit: str,
    message: str,
    detail_code: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_stage = _required_token("stage", stage, MAX_DETAIL_CODE_BYTES)
    if normalized_stage not in WORKER_PROGRESS_STAGES:
        raise ValueError("unsupported worker progress stage")
    normalized_status = _required_token("status", status, 32)
    if normalized_status != "running":
        raise ValueError("worker progress status must be running")

    normalized_current = _optional_counter("current", current)
    normalized_total = _optional_counter("total", total)
    if (
        normalized_current is not None
        and normalized_total is not None
        and normalized_current > normalized_total
    ):
        raise ValueError("current must not exceed total")

    normalized_detail = _optional_token("detail_code", detail_code, MAX_DETAIL_CODE_BYTES)
    normalized_unit = _optional_token("unit", unit, MAX_UNIT_BYTES)
    if (normalized_current is None) != (normalized_total is None):
        raise ValueError("current and total must be provided together")
    if normalized_current is not None:
        if normalized_unit not in PROGRESS_UNITS:
            raise ValueError("unit is required and must be allowlisted for a progress range")
    elif normalized_unit:
        raise ValueError("unit requires current and total")
    normalized_message = _safe_message(message)
    normalized_metadata = _safe_metadata(metadata)

    payload: dict[str, Any] = {
        "taxonomy_version": PROGRESS_TAXONOMY_VERSION,
        "stage": normalized_stage,
        "status": normalized_status,
        "metadata": normalized_metadata,
    }
    for key, value in (
        ("current", normalized_current),
        ("total", normalized_total),
        ("unit", normalized_unit),
        ("message", normalized_message),
        ("detail_code", normalized_detail),
    ):
        if value is not None and value != "":
            payload[key] = value

    encoded_payload = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded_payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("progress payload exceeds safe bound")
    return payload


def _progress_identity_job(job: dict) -> dict:
    if not isinstance(job, dict):
        return {}
    job_id = str(job.get("id") or "").strip()
    attempt_id = _training_attempt_id(job)
    token = _callback_token(job)
    config: dict[str, str] = {}
    if attempt_id:
        config["active_attempt_id"] = attempt_id
    if token:
        config["callback_token"] = token
    return {"id": job_id, "config": config}


def _required_token(name: str, value: Any, max_bytes: int) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{name} is required")
    return _validate_token(name, normalized, max_bytes)


def _optional_token(name: str, value: Any, max_bytes: int) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    return _validate_token(name, normalized, max_bytes)


def _validate_token(name: str, value: str, max_bytes: int) -> str:
    if len(value.encode("utf-8")) > max_bytes or not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded safe token")
    return value


def _optional_counter(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > MAX_PROGRESS_REVISION:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _safe_message(value: Any) -> str:
    message = str(value or "").strip()
    if not message:
        return ""
    if any(character in message for character in ("\x00", "\r", "\n")):
        raise ValueError("message contains control characters")
    if _UNSAFE_TEXT_RE.search(message):
        raise ValueError("message contains unsafe text")
    return _truncate_utf8(message, MAX_MESSAGE_BYTES)


def _safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if len(metadata) > len(PROGRESS_METADATA_KEYS):
        raise ValueError("metadata has too many entries")
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key not in PROGRESS_METADATA_KEYS:
            raise ValueError("metadata key is not allowlisted")
        if normalized_key == "early_stopped":
            if not isinstance(value, bool):
                raise TypeError("early_stopped metadata must be boolean")
            out[normalized_key] = value
            continue
        if not isinstance(value, str):
            raise TypeError("progress metadata value must be a safe token string")
        out[normalized_key] = _required_token(
            f"metadata {normalized_key}",
            value,
            MAX_METADATA_TOKEN_BYTES,
        )
    return out


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def _valid_initial_revision(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value >= MAX_PROGRESS_REVISION:
        return None
    return value


def _safe_clock(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception:
        return time.monotonic()
    return value if math.isfinite(value) else time.monotonic()


def _configured_positive_float(
    value: Any,
    env_name: str,
    default: float,
    *,
    maximum: float,
) -> float:
    if value is None:
        parsed = _positive_float_env(env_name, default)
    else:
        parsed = _positive_float_or_default(value, default)
    return min(parsed, maximum)


def _configured_positive_int(
    value: Any,
    env_name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    if value is None:
        parsed = _positive_int_env(env_name, default)
    elif isinstance(value, bool):
        parsed = int(default)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            parsed = int(default)
        if parsed <= 0:
            parsed = int(default)
    return min(parsed, maximum)


def _positive_float_env(name: str, default: float) -> float:
    return _positive_float_or_default(os.getenv(name, "").strip(), default)


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _positive_float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return parsed if math.isfinite(parsed) and parsed > 0 else float(default)


def _http_status_code(exc: requests.HTTPError) -> int | None:
    return _optional_status_code(getattr(getattr(exc, "response", None), "status_code", None))


def _optional_status_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _transient_http_status(status_code: int | None) -> bool:
    return status_code in _TRANSIENT_HTTP_STATUS_CODES or (
        isinstance(status_code, int) and status_code >= 500
    )
