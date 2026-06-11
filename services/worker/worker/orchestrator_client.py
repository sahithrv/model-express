import requests
import os
import time
import threading
from contextlib import contextmanager

DEFAULT_REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_REPORT_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5
ENDPOINT_UNAVAILABLE_STATUS_CODES = {404, 405, 501}

class OrchestratorClient:
    def __init__(self, base_url: str, timeout: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if timeout is not None else request_timeout_seconds()
        self._job_context = threading.local()

    @contextmanager
    def job_context(self, job: dict):
        previous = getattr(self._job_context, "job", None)
        self._job_context.job = job
        try:
            yield
        finally:
            self._job_context.job = previous
    
    def get_dataset(self, dataset_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}",
            **self._request_kwargs(timeout=self.timeout),
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            **self._request_kwargs(timeout=request_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()
    
    def update_dataset_profile(self, dataset_id: str, profile: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/profile",
            json={"profile": profile},
            **self._callback_request_kwargs(timeout=self.timeout),
        )
        response.raise_for_status()
        return response.json()

    def import_dataset_metadata(self, dataset_id: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/metadata/imports",
            json=payload,
            **self._callback_request_kwargs(timeout=report_timeout_seconds()),
        )
        if response.status_code in ENDPOINT_UNAVAILABLE_STATUS_CODES:
            return {
                "status": "unavailable",
                "reason": "metadata_import_endpoint_unavailable",
                "status_code": response.status_code,
            }
        response.raise_for_status()
        return response.json()

    def get_dataset_metadata_bundle(
        self,
        dataset_id: str,
        *,
        metadata_import_id: str | None = None,
        purpose: str = "training",
        include: str = "bbox",
        limit: int = 5000,
        offset: int = 0,
    ) -> dict | None:
        params: dict[str, str | int] = {
            "purpose": purpose,
            "include": include,
            "limit": limit,
            "offset": offset,
        }
        if metadata_import_id:
            params["metadata_import_id"] = metadata_import_id
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}/metadata/bundle",
            params=params,
            **self._request_kwargs(timeout=self.timeout),
        )
        if response.status_code in ENDPOINT_UNAVAILABLE_STATUS_CODES or response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()
    
    def register_worker(
        self,
        project_id: str,
        *,
        name: str | None = None,
        gpu_type: str | None = None,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/workers/register",
            json={
                "project_id": project_id,
                "name": name or os.getenv("WORKER_NAME", "local-worker-1"),
                "gpu_type": gpu_type or os.getenv("GPU_TYPE", "local"),
            },
            **self._request_kwargs(timeout=request_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()


    def heartbeat_worker(self, worker_id: str) -> dict:
        response = requests.post(
            f"{self.base_url}/workers/{worker_id}/heartbeat",
            **self._request_kwargs(timeout=request_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()


    def list_project_worker_requirements(self, project_id: str) -> list[dict]:
        response = requests.get(
            f"{self.base_url}/projects/{project_id}/worker-requirements",
            **self._request_kwargs(timeout=request_timeout_seconds()),
        )
        response.raise_for_status()
        payload = response.json()
        requirements = payload.get("requirements") if isinstance(payload, dict) else None
        return requirements if isinstance(requirements, list) else []

    def report_dispatcher_event(
        self,
        project_id: str,
        event_type: str,
        *,
        message: str = "",
        plan_id: str = "",
        payload: dict | None = None,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/projects/{project_id}/dispatcher-events",
            json={
                "event_type": event_type,
                "message": message,
                "plan_id": plan_id,
                "payload": payload or {},
            },
            **self._request_kwargs(timeout=report_timeout_seconds()),
        )
        if response.status_code in ENDPOINT_UNAVAILABLE_STATUS_CODES:
            return {
                "status": "unavailable",
                "reason": "dispatcher_event_endpoint_unavailable",
                "status_code": response.status_code,
            }
        response.raise_for_status()
        return response.json()


    def poll_job(
        self,
        worker_id: str,
        *,
        provider: str | None = None,
        templates: list[str] | None = None,
        include_unspecified_provider_templates: list[str] | None = None,
    ) -> dict | None:
        payload = {}
        if provider:
            payload["provider"] = provider
        if templates:
            payload["templates"] = templates
        if include_unspecified_provider_templates:
            payload["include_unspecified_provider_templates"] = include_unspecified_provider_templates
        kwargs = self._request_kwargs(timeout=request_timeout_seconds())
        if payload:
            kwargs["json"] = payload
        response = requests.post(
            f"{self.base_url}/workers/{worker_id}/poll",
            **kwargs,
        )
        response.raise_for_status()
        return response.json()["job"]


    def report_metric(self, job_id: str, epoch: int, metrics: dict[str, float], *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/metrics",
            json=self._with_callback_identity(job_id, {
                "epoch": epoch,
                "metrics": metrics,
            }, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()


    def report_training_run_summary(self, job_id: str, summary: dict, *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/training-run-summary",
            json=self._with_callback_identity(job_id, summary, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()

    def report_training_run_evaluation(self, job_id: str, evaluation: dict, *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/training-run-evaluation",
            json=self._with_callback_identity(job_id, evaluation, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()

    def report_modal_call(self, job_id: str, payload: dict, *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/modal-call",
            json=self._with_callback_identity(job_id, payload, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        if response.status_code in ENDPOINT_UNAVAILABLE_STATUS_CODES:
            return {
                "status": "unavailable",
                "reason": "modal_call_endpoint_unavailable",
                "status_code": response.status_code,
            }
        _raise_for_status_with_body(response)
        return response.json()

    def report_champion_export_result(self, job_id: str, result: dict, *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/champion-export-result",
            json=self._with_callback_identity(job_id, result, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()

    def report_champion_demo_prediction_result(self, job_id: str, result: dict, *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/champion-demo-prediction-result",
            json=self._with_callback_identity(job_id, result, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()

    def report_dataset_visual_exemplars(self, dataset_id: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/visual-exemplars",
            json=payload,
            **self._callback_request_kwargs(timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()

    def report_dataset_visual_analysis_result(self, dataset_id: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/visual-analysis-result",
            json=payload,
            **self._callback_request_kwargs(timeout=report_timeout_seconds()),
        )
        _raise_for_status_with_body(response)
        return response.json()

    def complete_job(self, job_id: str, mlflow_run_id: str = "", *, job: dict | None = None) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/complete",
            json=self._with_callback_identity(job_id, {"mlflow_run_id": mlflow_run_id}, job=job, strict=True),
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        response.raise_for_status()
        return response.json()


    def fail_job(
        self,
        job_id: str,
        error: str,
        retryable: bool = False,
        metadata: dict | None = None,
        job: dict | None = None,
    ) -> dict:
        payload = {"error": error}
        if retryable:
            payload["retryable"] = True
        if isinstance(metadata, dict):
            payload.update(metadata)
            payload["error"] = error
            if retryable:
                payload["retryable"] = True
        payload = self._with_callback_identity(job_id, payload, job=job, strict=True)
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/fail",
            json=payload,
            **self._callback_request_kwargs(job_id, job=job, timeout=report_timeout_seconds()),
        )
        _raise_for_status_with_body(response)
        return response.json()


    def run_fake_job(self, job: dict) -> None:
        job_id = job["id"]

        try:
            with self.job_context(job):
                for epoch in range(1, 4):
                    metrics = {
                        "train_loss": round(1.0 / epoch, 4),
                        "val_loss": round(1.2 / epoch, 4),
                        "macro_f1": round(0.2 * epoch, 4),
                    }
                    self.report_metric(job_id, epoch, metrics)
                    print(f"Reported epoch {epoch} metrics for {job_id}: {metrics}")
                    time.sleep(1)

                self.complete_job(job_id, mlflow_run_id=f"fake-mlflow-run-{job_id}")
                print(f"Completed job {job_id}")
        except Exception as exc:
            self.fail_job(job_id, str(exc), retryable=True, job=job)
            raise

    def _with_callback_identity(
        self,
        job_id: str,
        payload: dict,
        *,
        job: dict | None = None,
        strict: bool = False,
    ) -> dict:
        out = dict(payload)
        context_job = job or self._active_job_for(job_id)
        attempt_id = _training_attempt_id(context_job)
        if attempt_id:
            existing_attempt_id = str(out.get("training_attempt_id") or "").strip()
            if strict and existing_attempt_id and existing_attempt_id != attempt_id:
                raise ValueError(
                    f"Job {job_id} callback training_attempt_id {existing_attempt_id!r} "
                    f"does not match active attempt {attempt_id!r}."
                )
            out["training_attempt_id"] = attempt_id
            return out
        if out.get("training_attempt_id"):
            return out
        if strict and isinstance(context_job, dict):
            raise ValueError(f"Job {job_id} callback requires training_attempt_id from the active job context.")
        return out

    def _active_job_for(self, job_id: str) -> dict | None:
        job = getattr(self._job_context, "job", None)
        if isinstance(job, dict) and str(job.get("id") or "") == str(job_id):
            return job
        return None

    def _active_job(self) -> dict | None:
        job = getattr(self._job_context, "job", None)
        return job if isinstance(job, dict) else None

    def _callback_request_kwargs(
        self,
        job_id: str | None = None,
        *,
        job: dict | None = None,
        timeout: int | float,
    ) -> dict:
        kwargs = self._request_kwargs(timeout=timeout)
        headers = dict(kwargs.get("headers") or {})
        headers.update(self._callback_headers(job_id, job=job))
        if headers:
            kwargs["headers"] = headers
        return kwargs

    def _request_kwargs(self, *, timeout: int | float) -> dict:
        kwargs = {"timeout": timeout}
        headers = self._api_headers()
        if headers:
            kwargs["headers"] = headers
        return kwargs

    def _api_headers(self) -> dict:
        token = os.getenv("MODEL_EXPRESS_API_TOKEN", "").strip()
        if not token:
            return {}
        return {"X-Model-Express-API-Token": token}

    def _callback_headers(self, job_id: str | None = None, *, job: dict | None = None) -> dict:
        context_job = job
        if context_job is None and job_id is not None:
            context_job = self._active_job_for(job_id)
        if context_job is None:
            context_job = self._active_job()
        token = _callback_token(context_job)
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}


def request_timeout_seconds() -> int:
    return _positive_int_env("MODEL_EXPRESS_WORKER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def report_timeout_seconds() -> int:
    return _positive_int_env("MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS", DEFAULT_REPORT_TIMEOUT_SECONDS)


def _raise_for_status_with_body(response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = str(getattr(response, "text", "") or "")
        if body:
            body = body[:2000]
            raise requests.HTTPError(f"{exc}; response_body={body}", response=response) from exc
        raise


def _training_attempt_id(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    for value in (
        config.get("active_attempt_id"),
        job.get("training_attempt_id"),
        config.get("training_attempt_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _callback_token(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    session = _remote_training_session_metadata(config)
    for value in (
        config.get("callback_token"),
        job.get("callback_token"),
        session.get("callback_token"),
        session.get("token"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _remote_training_session_metadata(config: dict) -> dict:
    for key in ("remote_training_session", "remoteTrainingSession", "RemoteTrainingSession"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
