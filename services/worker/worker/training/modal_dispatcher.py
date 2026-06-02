from __future__ import annotations

import os
import socket
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from worker.diagnostics import log_event
from worker.jobs import run_job
from worker.orchestrator_client import OrchestratorClient
from worker.training.modal_provider import (
    ModalRetryableFailureReported,
    modal_app_session,
    run_modal_dataset_materialization,
)


MODAL_DISPATCHER_TEMPLATES = [
    "train_experiment",
    "profile_dataset",
    "analyze_dataset_visuals",
    "export_champion",
    "champion_demo_prediction",
    "generate_visual_exemplars",
]
MODAL_DISPATCHER_UNSPECIFIED_PROVIDER_TEMPLATES = [
    "profile_dataset",
    "export_champion",
    "champion_demo_prediction",
    "generate_visual_exemplars",
]
DEFAULT_DISPATCHER_SLOTS = 2
MAX_DISPATCHER_SLOTS = 8
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0
DEFAULT_REQUIREMENT_REFRESH_SECONDS = 5.0
REQUIREMENT_DEMAND_STATUSES = {"ACTIVE", "PENDING", "STARTING"}


@dataclass
class _DispatcherSlot:
    index: int
    worker_id: str
    future: Future | None = None
    job: dict | None = None
    next_heartbeat_at: float = 0.0


@dataclass
class _CacheWarmState:
    event: threading.Event
    error: BaseException | None = None


class ModalDispatcher:
    def __init__(
        self,
        client: OrchestratorClient,
        project_id: str,
        *,
        slot_count: int | None = None,
        poll_interval_seconds: float | None = None,
        heartbeat_interval_seconds: float | None = None,
        requirement_refresh_seconds: float | None = None,
        job_runner: Callable[[OrchestratorClient, dict], None] | None = None,
        materializer: Callable[[OrchestratorClient, dict], dict] | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id is required for the Modal dispatcher.")
        self.client = client
        self.project_id = project_id
        self.slot_count = slot_count if slot_count is not None else modal_dispatcher_slot_count()
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _positive_float_env("MODEL_EXPRESS_MODAL_DISPATCHER_POLL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)
        )
        self.heartbeat_interval_seconds = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else _positive_float_env(
                "MODEL_EXPRESS_MODAL_DISPATCHER_HEARTBEAT_SECONDS",
                DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            )
        )
        self.requirement_refresh_seconds = (
            requirement_refresh_seconds
            if requirement_refresh_seconds is not None
            else _positive_float_env(
                "MODEL_EXPRESS_MODAL_DISPATCHER_REQUIREMENT_REFRESH_SECONDS",
                DEFAULT_REQUIREMENT_REFRESH_SECONDS,
            )
        )
        self.job_runner = job_runner or run_job
        self.materializer = materializer or run_modal_dataset_materialization
        self.slots: list[_DispatcherSlot] = []
        self._executor = ThreadPoolExecutor(max_workers=MAX_DISPATCHER_SLOTS, thread_name_prefix="modal-dispatcher")
        self._cache_lock = threading.Lock()
        self._warming: dict[str, _CacheWarmState] = {}
        self._warm_cache_keys: set[str] = set()
        self._next_requirement_refresh_at = 0.0

    def run_forever(self) -> None:
        with modal_app_session():
            self.register_slots()
            log_event(
                "info",
                "modal_dispatcher_started",
                project_id=self.project_id,
                slot_count=self.slot_count,
            )
            while True:
                self.run_once()
                time.sleep(self.poll_interval_seconds)

    def run_once(self) -> bool:
        self._refresh_slot_target_from_requirements()
        self.register_slots()
        self._collect_finished_slots()
        self._heartbeat_due_slots()
        claimed = False
        for slot in self.slots:
            if slot.future is not None:
                continue
            job = self.client.poll_job(
                slot.worker_id,
                provider="modal",
                templates=MODAL_DISPATCHER_TEMPLATES,
                include_unspecified_provider_templates=MODAL_DISPATCHER_UNSPECIFIED_PROVIDER_TEMPLATES,
            )
            if job is None:
                continue
            slot.job = job
            slot.next_heartbeat_at = 0.0
            slot.future = self._executor.submit(self._run_claimed_job, slot.worker_id, job)
            claimed = True
            log_event(
                "info",
                "modal_dispatcher_job_claimed",
                project_id=job.get("project_id", self.project_id),
                worker_id=slot.worker_id,
                job_id=job.get("id"),
                template=job.get("template"),
                dataset_cache_key=_job_dataset_cache_key(job),
            )
        return claimed or any(slot.future is not None for slot in self.slots)

    def wait_for_active_jobs(self, timeout_seconds: float | None = None) -> None:
        started_at = time.monotonic()
        while any(slot.future is not None for slot in self.slots):
            self.run_once()
            if timeout_seconds is not None and time.monotonic() - started_at > timeout_seconds:
                raise TimeoutError("Timed out waiting for Modal dispatcher jobs to finish.")
            time.sleep(0.01)

    def register_slots(self) -> None:
        while len(self.slots) < self.slot_count:
            index = len(self.slots) + 1
            worker = self.client.register_worker(
                self.project_id,
                name=f"{modal_dispatcher_worker_name()}-slot-{index}",
                gpu_type="modal",
            )
            slot = _DispatcherSlot(index=index, worker_id=worker["id"])
            self.slots.append(slot)
            log_event(
                "info",
                "modal_dispatcher_slot_registered",
                project_id=self.project_id,
                worker_id=slot.worker_id,
                slot=index,
            )

    def _run_claimed_job(self, worker_id: str, job: dict) -> None:
        cache_key = _job_dataset_cache_key(job)
        if cache_key and modal_dataset_prewarm_enabled(job):
            self._ensure_cache_warm(cache_key, job)
        self.job_runner(self.client, job)

    def _ensure_cache_warm(self, cache_key: str, job: dict) -> None:
        owner = False
        with self._cache_lock:
            if cache_key in self._warm_cache_keys:
                return
            state = self._warming.get(cache_key)
            if state is None:
                state = _CacheWarmState(event=threading.Event())
                self._warming[cache_key] = state
                owner = True

        if owner:
            try:
                log_event(
                    "info",
                    "modal_dispatcher_dataset_prewarm_started",
                    project_id=job.get("project_id", self.project_id),
                    job_id=job.get("id"),
                    dataset_cache_key=cache_key,
                )
                self.materializer(self.client, job)
                with self._cache_lock:
                    self._warm_cache_keys.add(cache_key)
                log_event(
                    "info",
                    "modal_dispatcher_dataset_prewarm_finished",
                    project_id=job.get("project_id", self.project_id),
                    job_id=job.get("id"),
                    dataset_cache_key=cache_key,
                )
            except BaseException as exc:
                state.error = exc
                raise
            finally:
                state.event.set()
                with self._cache_lock:
                    self._warming.pop(cache_key, None)
            return

        state.event.wait()
        if state.error is not None:
            raise RuntimeError(f"Modal dataset prewarm failed for cache key {cache_key}") from state.error

    def _collect_finished_slots(self) -> None:
        for slot in self.slots:
            if slot.future is None or not slot.future.done():
                continue
            job = slot.job
            try:
                slot.future.result()
            except ModalRetryableFailureReported as exc:
                log_event(
                    "warn",
                    "modal_dispatcher_job_retryable_failure_reported",
                    project_id=(job or {}).get("project_id", self.project_id),
                    worker_id=slot.worker_id,
                    job_id=str((job or {}).get("id") or ""),
                    error=str(exc),
                )
            except Exception as exc:
                job_id = str((job or {}).get("id") or "")
                if job_id:
                    try:
                        self.client.fail_job(job_id, str(exc))
                    except Exception:
                        log_event(
                            "error",
                            "modal_dispatcher_fail_report_failed",
                            project_id=(job or {}).get("project_id", self.project_id),
                            worker_id=slot.worker_id,
                            job_id=job_id,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                log_event(
                    "error",
                    "modal_dispatcher_job_failed",
                    project_id=(job or {}).get("project_id", self.project_id),
                    worker_id=slot.worker_id,
                    job_id=job_id,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            finally:
                slot.future = None
                slot.job = None

    def _heartbeat_due_slots(self) -> None:
        now = time.monotonic()
        for slot in self.slots:
            if slot.next_heartbeat_at > now:
                continue
            try:
                self.client.heartbeat_worker(slot.worker_id)
            except Exception as exc:
                log_event(
                    "warn",
                    "modal_dispatcher_heartbeat_failed",
                    project_id=self.project_id,
                    worker_id=slot.worker_id,
                    error=str(exc),
                )
            slot.next_heartbeat_at = now + self.heartbeat_interval_seconds

    def _refresh_slot_target_from_requirements(self) -> None:
        now = time.monotonic()
        if self._next_requirement_refresh_at > now:
            return
        self._next_requirement_refresh_at = now + self.requirement_refresh_seconds
        try:
            requirements = self.client.list_project_worker_requirements(self.project_id)
        except Exception as exc:
            log_event(
                "warn",
                "modal_dispatcher_requirement_refresh_failed",
                project_id=self.project_id,
                error=str(exc),
            )
            return

        desired = modal_dispatcher_required_slot_count(requirements)
        if desired <= self.slot_count:
            return
        previous = self.slot_count
        self.slot_count = desired
        log_event(
            "info",
            "modal_dispatcher_slot_target_increased",
            project_id=self.project_id,
            previous_slot_count=previous,
            slot_count=self.slot_count,
        )


def run_modal_dispatcher(client: OrchestratorClient, project_id: str) -> None:
    ModalDispatcher(client, project_id).run_forever()


def modal_dispatcher_enabled() -> bool:
    explicit = os.getenv("MODEL_EXPRESS_MODAL_DISPATCHER", "").strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    disabled = os.getenv("MODEL_EXPRESS_DISABLE_MODAL_DISPATCHER", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    return os.getenv("GPU_TYPE", "").strip().lower() == "modal"


def modal_dispatcher_slot_count() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS",
        DEFAULT_DISPATCHER_SLOTS,
        minimum=1,
        maximum=MAX_DISPATCHER_SLOTS,
    )


def modal_dispatcher_required_slot_count(requirements: list[dict]) -> int:
    target = 0
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        status = str(requirement.get("status") or "").strip().upper()
        if status not in REQUIREMENT_DEMAND_STATUSES:
            continue
        provider = str(requirement.get("provider") or "").strip().lower()
        gpu_type = str(requirement.get("gpu_type") or "").strip().lower()
        if provider != "modal" and gpu_type != "modal":
            continue
        target = max(target, _positive_int_value(requirement.get("target_count"), default=0))
        target = max(target, _positive_int_value(requirement.get("max_concurrent_jobs"), default=0))
    return min(max(target, 0), MAX_DISPATCHER_SLOTS)


def modal_dispatcher_worker_name() -> str:
    configured = os.getenv("WORKER_NAME", "").strip()
    if configured:
        return configured
    return f"modal-dispatcher-{socket.gethostname()}"


def modal_dataset_prewarm_enabled(job: dict | None = None) -> bool:
    config = job.get("config") if isinstance(job, dict) and isinstance(job.get("config"), dict) else {}
    materialization = (
        config.get("dataset_materialization")
        if isinstance(config.get("dataset_materialization"), dict)
        else {}
    )
    for value in (
        materialization.get("prewarm_enabled"),
        materialization.get("enabled"),
        config.get("modal_dataset_prewarm_enabled"),
    ):
        parsed = _optional_bool_value(value)
        if parsed is not None:
            return parsed
    return _optional_bool_value(os.getenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED")) is True


def _job_dataset_cache_key(job: dict) -> str:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    materialization = config.get("dataset_materialization") if isinstance(config.get("dataset_materialization"), dict) else {}
    cache_key = materialization.get("dataset_cache_key")
    if isinstance(cache_key, str) and cache_key.strip():
        return cache_key.strip()
    checksum = config.get("dataset_checksum_sha256")
    if isinstance(checksum, str) and checksum.strip():
        return f"sha256-{checksum.strip().lower()}"
    return ""


def _positive_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _positive_int_value(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _positive_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
