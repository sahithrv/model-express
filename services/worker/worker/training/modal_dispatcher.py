from __future__ import annotations

import os
import socket
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

from worker.diagnostics import log_event
from worker.jobs import run_job
from worker.orchestrator_client import OrchestratorClient
from worker.training.modal_provider import (
    ModalRetryableFailureReported,
    modal_app_session,
    run_modal_dataset_materialization,
    run_modal_training_batch,
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
    "analyze_dataset_visuals",
    "export_champion",
    "champion_demo_prediction",
    "generate_visual_exemplars",
]
DEFAULT_DISPATCHER_SLOTS = 1
MAX_DISPATCHER_SLOTS = 8
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0
DEFAULT_REQUIREMENT_REFRESH_SECONDS = 5.0
DEFAULT_MODAL_BATCH_MAX_TRIALS = 3
DEFAULT_IDLE_POLL_SLOTS = 1
REQUIREMENT_DEMAND_STATUSES = {"ACTIVE", "PENDING", "STARTING"}
DISPATCHER_STATUS_EVENT = "DISPATCHER_STATUS"
DISPATCHER_IDLE_EXIT_EVENT = "DISPATCHER_IDLE_EXIT"


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
        batch_runner: Callable[[OrchestratorClient, list[dict], dict], None] | None = None,
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
        self.batch_runner = batch_runner or run_modal_training_batch
        self.materializer = materializer or run_modal_dataset_materialization
        self.slots: list[_DispatcherSlot] = []
        self._executor = ThreadPoolExecutor(max_workers=MAX_DISPATCHER_SLOTS, thread_name_prefix="modal-dispatcher")
        self._cache_lock = threading.Lock()
        self._warming: dict[str, _CacheWarmState] = {}
        self._warm_cache_keys: set[str] = set()
        self._next_requirement_refresh_at = 0.0
        self._last_requirement_refresh_ok = False
        self._last_requirement_refresh_at = 0.0
        self._last_required_slot_count: int | None = None
        self._idle_started_at: float | None = None
        self._last_claimed_at: float | None = None
        self._last_poll_slot_floor = modal_dispatcher_idle_poll_slots()

    def run_forever(self) -> None:
        try:
            started_logged = False
            while True:
                self._refresh_slot_target_from_requirements()
                self._collect_finished_slots()
                if self._should_hold_modal_session() or self._pending_claims_require_modal_session():
                    with modal_app_session():
                        if not started_logged:
                            self._log_dispatcher_started()
                            started_logged = True
                        while True:
                            self._start_pending_claimed_slots()
                            self.run_once()
                            if self._should_exit_for_idle():
                                return
                            if not self._should_hold_modal_session() and not self._pending_claims_require_modal_session():
                                break
                            time.sleep(self.poll_interval_seconds)
                    continue
                if not started_logged:
                    self._log_dispatcher_started()
                    started_logged = True
                self.run_once(start_claimed=False)
                if self._pending_claims_require_modal_session():
                    continue
                self._start_pending_claimed_slots()
                if self._should_exit_for_idle():
                    break
                time.sleep(self.poll_interval_seconds)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _log_dispatcher_started(self) -> None:
        log_event(
            "info",
            "modal_dispatcher_started",
            project_id=self.project_id,
            slot_count=self.slot_count,
            idle_exit_seconds=modal_dispatcher_idle_exit_seconds() or 0,
        )

    def run_once(self, *, start_claimed: bool = True) -> bool:
        self._refresh_slot_target_from_requirements()
        self.register_slots()
        self._collect_finished_slots()
        self._heartbeat_due_slots()
        claimed = False
        claimed_slots: list[_DispatcherSlot] = []
        for slot in self.slots:
            if slot.future is not None or slot.job is not None:
                continue
            if not self._slot_in_current_target(slot):
                continue
            job = self.client.poll_job(
                slot.worker_id,
                provider="modal",
                templates=self._poll_templates(),
                include_unspecified_provider_templates=MODAL_DISPATCHER_UNSPECIFIED_PROVIDER_TEMPLATES,
            )
            if job is None:
                continue
            slot.job = job
            slot.next_heartbeat_at = 0.0
            claimed_slots.append(slot)
            claimed = True
            self._last_claimed_at = time.monotonic()
            self._idle_started_at = None
            log_event(
                "info",
                "modal_dispatcher_job_claimed",
                project_id=job.get("project_id", self.project_id),
                worker_id=slot.worker_id,
                job_id=job.get("id"),
                template=job.get("template"),
                dataset_cache_key=_job_dataset_cache_key(job),
            )
        if start_claimed and claimed_slots:
            self._start_claimed_slots(claimed_slots)
        return claimed or any(slot.future is not None or slot.job is not None for slot in self.slots)

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
        with _client_job_context(self.client, job):
            cache_key = _job_dataset_cache_key(job)
            if cache_key and modal_dataset_prewarm_enabled(job):
                self._ensure_cache_warm(cache_key, job)
            self.job_runner(self.client, job)

    def _run_claimed_batch(self, worker_jobs: list[tuple[str, dict]], batch: dict) -> None:
        jobs = [job for _worker_id, job in worker_jobs]
        cache_key = str(batch.get("dataset_cache_key") or "")
        if cache_key and any(modal_dataset_prewarm_enabled(job) for job in jobs):
            self._ensure_cache_warm(cache_key, jobs[0])
        self.batch_runner(self.client, jobs, batch)

    def _pending_claimed_slots(self) -> list[_DispatcherSlot]:
        return [slot for slot in self.slots if slot.job is not None and slot.future is None]

    def _pending_claims_require_modal_session(self) -> bool:
        return any(_job_requires_modal_session(slot.job) for slot in self._pending_claimed_slots())

    def _start_pending_claimed_slots(self) -> None:
        pending = self._pending_claimed_slots()
        if pending:
            self._start_claimed_slots(pending)

    def _start_claimed_slots(self, claimed_slots: list[_DispatcherSlot]) -> None:
        if not modal_batch_runner_enabled():
            for slot in claimed_slots:
                if slot.job is not None and slot.future is None:
                    slot.future = self._executor.submit(self._run_claimed_job, slot.worker_id, slot.job)
            return

        for batch_slots, batch in _modal_batch_groups(claimed_slots):
            if batch is None or len(batch_slots) < 2:
                for slot in batch_slots:
                    if slot.job is not None:
                        slot.future = self._executor.submit(self._run_claimed_job, slot.worker_id, slot.job)
                continue
            worker_jobs = [(slot.worker_id, slot.job) for slot in batch_slots if slot.job is not None]
            future = self._executor.submit(self._run_claimed_batch, worker_jobs, batch)
            for slot in batch_slots:
                slot.future = future
            log_event(
                "info",
                "modal_dispatcher_batch_grouped",
                project_id=batch.get("project_id", self.project_id),
                plan_id=batch.get("plan_id", ""),
                batch_id=batch.get("batch_id", ""),
                batch_key=batch.get("batch_key", ""),
                dataset_id=batch.get("dataset_id", ""),
                dataset_cache_key=batch.get("dataset_cache_key", ""),
                training_tier=batch.get("training_tier", ""),
                task_type=batch.get("task_type", ""),
                job_ids=[str((slot.job or {}).get("id") or "") for slot in batch_slots],
            )

    def _ensure_cache_warm(self, cache_key: str, job: dict) -> None:
        owner = False
        wait_started = time.monotonic()
        with self._cache_lock:
            if cache_key in self._warm_cache_keys:
                return
            state = self._warming.get(cache_key)
            if state is None:
                state = _CacheWarmState(event=threading.Event())
                self._warming[cache_key] = state
                owner = True

        if owner:
            started_at = time.monotonic()
            try:
                log_event(
                    "info",
                    "modal_dispatcher_dataset_prewarm_started",
                    project_id=job.get("project_id", self.project_id),
                    job_id=job.get("id"),
                    dataset_cache_key=cache_key,
                )
                result = self.materializer(self.client, job)
                materialization = (
                    result.get("dataset_materialization")
                    if isinstance(result, dict) and isinstance(result.get("dataset_materialization"), dict)
                    else {}
                )
                reusable_for_training = _materialization_reusable_for_training(materialization)
                if reusable_for_training:
                    with self._cache_lock:
                        self._warm_cache_keys.add(cache_key)
                log_event(
                    "info",
                    "modal_dispatcher_dataset_prewarm_finished",
                    project_id=job.get("project_id", self.project_id),
                    job_id=job.get("id"),
                    dataset_cache_key=cache_key,
                    duration_seconds=round(time.monotonic() - started_at, 6),
                    status=materialization.get("dataset_materialization_status", ""),
                    cache_hit=materialization.get("dataset_materialization_cache_hit"),
                    cache_miss=materialization.get("dataset_materialization_cache_miss"),
                    bytes_downloaded=materialization.get("dataset_materialization_bytes_downloaded"),
                    materialization_cache_root=materialization.get("dataset_materialization_cache_root"),
                    materialization_cache_scope=materialization.get("dataset_materialization_cache_scope"),
                    training_cache_root=materialization.get("dataset_training_cache_root"),
                    prewarm_cache_root=materialization.get("dataset_prewarm_cache_root"),
                    prewarm_reusable_for_training=reusable_for_training,
                    prewarm_reuse_status=materialization.get("dataset_prewarm_reuse_status"),
                )
            except BaseException as exc:
                state.error = exc
                log_event(
                    "warn",
                    "modal_dispatcher_dataset_prewarm_failed",
                    project_id=job.get("project_id", self.project_id),
                    job_id=job.get("id"),
                    dataset_cache_key=cache_key,
                    duration_seconds=round(time.monotonic() - started_at, 6),
                    error=str(exc),
                    retryable=isinstance(exc, ModalRetryableFailureReported),
                )
                raise
            finally:
                state.event.set()
                with self._cache_lock:
                    self._warming.pop(cache_key, None)
            return

        log_event(
            "info",
            "modal_dispatcher_dataset_prewarm_wait_started",
            project_id=job.get("project_id", self.project_id),
            job_id=job.get("id"),
            dataset_cache_key=cache_key,
        )
        state.event.wait()
        wait_seconds = round(time.monotonic() - wait_started, 6)
        if state.error is not None:
            if isinstance(state.error, ModalRetryableFailureReported):
                try:
                    _report_dispatcher_failure(self.client, job["id"], str(state.error), retryable=True, job=job)
                except Exception as exc:
                    log_event(
                        "warn",
                        "modal_dispatcher_waiter_retryable_fail_report_failed",
                        project_id=job.get("project_id", self.project_id),
                        job_id=job.get("id"),
                        dataset_cache_key=cache_key,
                        error=str(exc),
                    )
                raise ModalRetryableFailureReported(str(state.error)) from state.error
            raise RuntimeError(f"Modal dataset prewarm failed for cache key {cache_key}") from state.error
        log_event(
            "info",
            "modal_dispatcher_dataset_prewarm_wait_finished",
            project_id=job.get("project_id", self.project_id),
            job_id=job.get("id"),
            dataset_cache_key=cache_key,
            wait_seconds=wait_seconds,
        )

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
                    retryable = _dispatcher_exception_retryable(exc)
                    try:
                        _report_dispatcher_failure(
                            self.client,
                            job_id,
                            str(exc),
                            retryable=retryable,
                            metadata={"failure_class": "worker_exception" if retryable else "validation"},
                            job=job,
                        )
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
                    retryable=_dispatcher_exception_retryable(exc),
                    traceback=traceback.format_exc(),
                )
            finally:
                slot.future = None
                slot.job = None

    def _heartbeat_due_slots(self) -> None:
        now = time.monotonic()
        for slot in self.slots:
            if not self._slot_in_current_target(slot) and slot.future is None:
                continue
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
            self._last_requirement_refresh_ok = False
            log_event(
                "warn",
                "modal_dispatcher_requirement_refresh_failed",
                project_id=self.project_id,
                error=str(exc),
            )
            return

        required = modal_dispatcher_required_slot_count(requirements)
        poll_slot_floor = modal_dispatcher_idle_poll_slots()
        desired = required if required > 0 else max(required, poll_slot_floor)
        self._last_requirement_refresh_ok = True
        self._last_requirement_refresh_at = now
        self._last_required_slot_count = required
        if desired == self.slot_count and poll_slot_floor == self._last_poll_slot_floor:
            return
        previous = self.slot_count
        self._last_poll_slot_floor = poll_slot_floor
        self.slot_count = desired
        event_name = (
            "modal_dispatcher_slot_target_increased"
            if desired > previous
            else "modal_dispatcher_slot_target_decreased"
        )
        log_event(
            "info",
            event_name,
            project_id=self.project_id,
            previous_slot_count=previous,
            slot_count=self.slot_count,
            desired_slot_count=required,
            idle_poll_slot_count=poll_slot_floor,
        )
        self._report_dispatcher_event(
            DISPATCHER_STATUS_EVENT,
            "Modal dispatcher slot target updated.",
            {
                "dispatcher": "modal",
                "previous_slot_count": previous,
                "slot_count": self.slot_count,
                "desired_slot_count": self._last_required_slot_count or 0,
                "idle_poll_slot_count": poll_slot_floor,
                "registered_slot_count": len(self.slots),
                "active_slot_count": self._active_slot_count(),
                "idle_exit_seconds": modal_dispatcher_idle_exit_seconds() or 0,
                "reason": "requirements_refresh",
            },
        )

    def _slot_in_current_target(self, slot: _DispatcherSlot) -> bool:
        return slot.index <= self.slot_count

    def _active_slot_count(self) -> int:
        return sum(1 for slot in self.slots if slot.future is not None)

    def _should_hold_modal_session(self) -> bool:
        if (self._last_required_slot_count or 0) > 0:
            return True
        if any(slot.future is not None and _job_requires_modal_session(slot.job) for slot in self.slots):
            return True
        with self._cache_lock:
            return bool(self._warming)

    def _poll_templates(self) -> list[str]:
        if (self._last_required_slot_count or 0) > 0:
            return MODAL_DISPATCHER_TEMPLATES
        return MODAL_DISPATCHER_UNSPECIFIED_PROVIDER_TEMPLATES

    def _should_exit_for_idle(self) -> bool:
        idle_exit_seconds = modal_dispatcher_idle_exit_seconds()
        if idle_exit_seconds is None:
            return False
        now = time.monotonic()
        active = any(slot.future is not None for slot in self.slots)
        with self._cache_lock:
            warming = bool(self._warming)
        if active or warming or not self._last_requirement_refresh_ok or self._last_required_slot_count != 0:
            self._idle_started_at = None
            return False
        if self._idle_started_at is None:
            self._idle_started_at = now
            self._next_requirement_refresh_at = 0.0
            return False
        if self._last_requirement_refresh_at <= self._idle_started_at:
            self._next_requirement_refresh_at = 0.0
            return False
        idle_seconds = now - self._idle_started_at
        if idle_seconds < idle_exit_seconds:
            return False
        log_event(
            "info",
            "modal_dispatcher_idle_exit",
            project_id=self.project_id,
            idle_seconds=round(idle_seconds, 6),
            idle_exit_seconds=idle_exit_seconds,
            slot_count=self.slot_count,
            desired_slot_count=self._last_required_slot_count,
            idle_poll_slot_count=self._last_poll_slot_floor,
        )
        self._report_dispatcher_event(
            DISPATCHER_IDLE_EXIT_EVENT,
            "Modal dispatcher exited after an idle zero-demand window.",
            {
                "dispatcher": "modal",
                "slot_count": self.slot_count,
                "desired_slot_count": self._last_required_slot_count or 0,
                "idle_poll_slot_count": self._last_poll_slot_floor,
                "registered_slot_count": len(self.slots),
                "active_slot_count": self._active_slot_count(),
                "idle_seconds": round(idle_seconds, 6),
                "idle_exit_seconds": idle_exit_seconds,
            },
        )
        return True

    def _report_dispatcher_event(self, event_type: str, message: str, payload: dict) -> None:
        reporter = getattr(self.client, "report_dispatcher_event", None)
        if not callable(reporter):
            return
        try:
            reporter(self.project_id, event_type, message=message, payload=payload)
        except Exception as exc:
            log_event(
                "warn",
                "modal_dispatcher_event_report_failed",
                project_id=self.project_id,
                event_type=event_type,
                error=str(exc),
            )


def _modal_batch_groups(claimed_slots: list[_DispatcherSlot]) -> list[tuple[list[_DispatcherSlot], dict | None]]:
    singles: list[_DispatcherSlot] = []
    candidates: dict[tuple[str, ...], list[_DispatcherSlot]] = {}
    contracts: dict[tuple[str, ...], dict] = {}
    for slot in claimed_slots:
        job = slot.job
        contract = _modal_batch_contract(job) if isinstance(job, dict) else None
        if contract is None:
            singles.append(slot)
            continue
        key = tuple(str(contract[field]) for field in _MODAL_BATCH_KEY_FIELDS)
        candidates.setdefault(key, []).append(slot)
        contracts[key] = contract

    groups: list[tuple[list[_DispatcherSlot], dict | None]] = []
    max_trials = modal_batch_max_trials()
    for key, slots in candidates.items():
        if len(slots) < 2:
            singles.extend(slots)
            continue
        for start in range(0, len(slots), max_trials):
            chunk = slots[start : start + max_trials]
            if len(chunk) < 2:
                singles.extend(chunk)
                continue
            contract = dict(contracts[key])
            contract["batch_id"] = f"modal-preview-batch-{uuid.uuid4().hex[:12]}"
            contract["batch_size"] = len(chunk)
            contract["job_ids"] = [str((slot.job or {}).get("id") or "") for slot in chunk]
            groups.append((chunk, contract))
    groups.extend(([slot], None) for slot in singles)
    return groups


_MODAL_BATCH_KEY_FIELDS = (
    "project_id",
    "plan_id",
    "dataset_id",
    "dataset_cache_key",
    "training_tier",
    "task_type",
    "subset_fraction",
    "subset_seed",
    "split_policy",
    "image_size_family",
)


def _modal_batch_contract(job: dict) -> dict | None:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    if str(job.get("template") or "").strip() != "train_experiment":
        return None
    if str(config.get("provider") or "").strip().lower() != "modal":
        return None

    training_tier = _job_training_tier(config)
    if training_tier != "preview":
        return None

    project_id = str(job.get("project_id") or "").strip()
    plan_id = str(config.get("plan_id") or "").strip()
    dataset_id = str(config.get("dataset_id") or "").strip()
    dataset_cache_key = _job_dataset_cache_key(job)
    task_type = _job_task_type(config)
    if task_type == "object_detection" and not yolo_batch_preview_enabled():
        return None
    if not project_id or not plan_id or not dataset_id or not dataset_cache_key or not task_type:
        return None
    subset_fraction = _job_preview_subset_fraction(config)
    subset_seed = _job_preview_subset_seed(config)
    split_policy = _job_preview_split_policy(config)
    image_size_family = _job_image_size_family(config, task_type)

    batch_key = "|".join(
        (
            project_id,
            plan_id,
            dataset_id,
            dataset_cache_key,
            training_tier,
            task_type,
            subset_fraction,
            subset_seed,
            split_policy,
            image_size_family,
        )
    )
    return {
        "schema_version": "modal_preview_batch.v1",
        "batch_id": "",
        "batch_key": batch_key,
        "project_id": project_id,
        "plan_id": plan_id,
        "dataset_id": dataset_id,
        "dataset_cache_key": dataset_cache_key,
        "training_tier": training_tier,
        "task_type": task_type,
        "subset_fraction": subset_fraction,
        "subset_seed": subset_seed,
        "split_policy": split_policy,
        "image_size_family": image_size_family,
        "runner_status": "dispatcher_grouped_single_job_fallback",
    }


def _job_training_tier(config: dict) -> str:
    for key in ("training_tier", "dataset_tier"):
        value = str(config.get(key) or "").strip().lower()
        if value:
            return value
    modal_batch = config.get("modal_batch") if isinstance(config.get("modal_batch"), dict) else {}
    return str(modal_batch.get("training_tier") or "").strip().lower()


def _job_task_type(config: dict) -> str:
    task_type = str(config.get("task_type") or "").strip().lower()
    if task_type:
        return task_type
    model_kind = str(config.get("model_kind") or "").strip().lower()
    model = str(config.get("model") or "").strip().lower()
    if model_kind == "ultralytics_yolo_detector" or model.startswith("yolo"):
        return "object_detection"
    return "image_classification"


def _job_preview_subset_fraction(config: dict) -> str:
    value = config.get("preview_fraction", config.get("dataset_fraction", config.get("preview_dataset_fraction", 0.25)))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.25
    parsed = max(0.01, min(1.0, parsed))
    return f"{parsed:.6f}".rstrip("0").rstrip(".")


def _job_preview_subset_seed(config: dict) -> str:
    value = config.get("preview_seed", config.get("seed", 42))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 42
    return str(max(0, parsed))


def _job_preview_split_policy(config: dict) -> str:
    return str(config.get("split_policy") or "official_or_deterministic").strip().lower()


def _job_image_size_family(config: dict, task_type: str) -> str:
    explicit = str(config.get("image_size_family") or "").strip().lower()
    if explicit:
        return explicit
    try:
        image_size = int(config.get("image_size") or 0)
    except (TypeError, ValueError):
        image_size = 0
    if image_size <= 0:
        return f"{task_type}:default"
    bucket = ((image_size + 31) // 32) * 32
    return f"{task_type}:{bucket}"


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
    maximum = MAX_DISPATCHER_SLOTS if _fast_remote_execution_profile_enabled() else 1
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS",
        DEFAULT_DISPATCHER_SLOTS,
        minimum=1,
        maximum=maximum,
    )


def modal_dispatcher_idle_poll_slots() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_POLL_SLOTS",
        DEFAULT_IDLE_POLL_SLOTS,
        minimum=0,
        maximum=1,
    )


def modal_dispatcher_idle_exit_seconds() -> float | None:
    value = os.getenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def modal_batch_runner_enabled() -> bool:
    return _optional_bool_value(os.getenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER")) is True


def yolo_batch_preview_enabled() -> bool:
    return _optional_bool_value(os.getenv("MODEL_EXPRESS_YOLO_BATCH_PREVIEW")) is True


def modal_batch_max_trials() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS",
        DEFAULT_MODAL_BATCH_MAX_TRIALS,
        minimum=2,
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


def _fast_remote_execution_profile_enabled() -> bool:
    for name in ("MODEL_EXPRESS_EXECUTION_PROFILE", "MODEL_EXPRESS_V1_PROFILE"):
        value = os.getenv(name, "").strip().lower().replace("_", "-")
        if value == "fast-remote":
            return True
    return False


def modal_dispatcher_worker_name() -> str:
    configured = os.getenv("WORKER_NAME", "").strip()
    if configured:
        return configured
    return f"modal-dispatcher-{socket.gethostname()}"


def _job_requires_modal_session(job: dict | None) -> bool:
    if not isinstance(job, dict):
        return False
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    template = str(job.get("template") or "").strip()
    provider = str(config.get("provider") or "").strip().lower()
    if provider == "modal":
        return True
    if template == "train_experiment":
        return provider in {"", "modal"}
    if template == "profile_dataset":
        profile_provider = os.getenv("MODEL_EXPRESS_DATASET_PROFILE_PROVIDER", "").strip().lower()
        if profile_provider:
            return profile_provider == "modal"
        return os.getenv("GPU_TYPE", "").strip().lower() == "modal"
    return modal_dataset_prewarm_enabled(job)


def _dispatcher_exception_retryable(exc: Exception) -> bool:
    return not isinstance(exc, (KeyError, ValueError))


def _client_job_context(client: OrchestratorClient, job: dict):
    context = getattr(client, "job_context", None)
    if callable(context):
        return context(job)
    return nullcontext()


def _report_dispatcher_failure(
    client: OrchestratorClient,
    job_id: str,
    error: str,
    *,
    retryable: bool = False,
    metadata: dict | None = None,
    job: dict | None = None,
):
    try:
        return client.fail_job(job_id, error, retryable=retryable, metadata=metadata, job=job)
    except TypeError:
        if metadata is not None:
            try:
                return client.fail_job(job_id, error, retryable=retryable, metadata=metadata)
            except TypeError:
                pass
        return client.fail_job(job_id, error, retryable=retryable)


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


def _materialization_reusable_for_training(materialization: dict) -> bool:
    return materialization.get("dataset_prewarm_reusable_for_training") is True


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
