from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image


def _import_modal_app():
    try:
        return importlib.import_module("worker.training.modal_app")
    except Exception as exc:  # pragma: no cover - depends on optional training deps
        raise unittest.SkipTest(f"worker training dependencies are unavailable: {exc}") from exc


class ModalTrainingHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modal_app = _import_modal_app()

    def _modal_preview_batch_payload(self, *, task_type: str = "image_classification", model: str = "mobilenet_v3_small") -> dict:
        batch_key = f"project_1|plan_1|dataset_1|sha256-a|preview|{task_type}"
        return {
            "batch": {
                "schema_version": "modal_preview_batch.v1",
                "batch_id": "modal-preview-batch-test",
                "batch_key": batch_key,
                "project_id": "project_1",
                "plan_id": "plan_1",
                "dataset_id": "dataset_1",
                "dataset_cache_key": "sha256-a",
                "training_tier": "preview",
                "task_type": task_type,
            },
            "jobs": [
                {
                    "id": "job_1",
                    "project_id": "project_1",
                    "template": "train_experiment",
                    "config": {
                        "provider": "modal",
                        "plan_id": "plan_1",
                        "dataset_id": "dataset_1",
                        "training_tier": "preview",
                        "task_type": task_type,
                        "model": model,
                        "dataset_materialization": {"dataset_cache_key": "sha256-a"},
                    },
                },
                {
                    "id": "job_2",
                    "project_id": "project_1",
                    "template": "train_experiment",
                    "config": {
                        "provider": "modal",
                        "plan_id": "plan_1",
                        "dataset_id": "dataset_1",
                        "training_tier": "preview",
                        "task_type": task_type,
                        "model": model,
                        "dataset_materialization": {"dataset_cache_key": "sha256-a"},
                    },
                },
            ],
            "dataset": {
                "id": "dataset_1",
                "storage_uri": "s3://bucket/dataset.zip",
                "checksum_sha256": "a" * 64,
            },
            "orchestrator_url": "https://orchestrator.test",
            "s3_endpoint_url": "https://s3.test",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "aws_default_region": "us-east-1",
            "storage_scope": {
                "token": "scope-token",
                "buckets": ["bucket"],
                "read_keys": ["datasets/project_1/data.zip"],
                "write_prefixes": ["model-express/artifacts/job_1/"],
            },
        }

    def test_safe_dataloader_defaults_cap_workers(self) -> None:
        previous_safe = os.environ.get("MODEL_EXPRESS_DATALOADER_SAFE_DEFAULTS")
        previous_workers = os.environ.get("MODEL_EXPRESS_MODAL_DATALOADER_WORKERS")
        try:
            os.environ["MODEL_EXPRESS_DATALOADER_SAFE_DEFAULTS"] = "true"
            os.environ["MODEL_EXPRESS_MODAL_DATALOADER_WORKERS"] = "16"
            self.assertEqual(self.modal_app._modal_dataloader_workers(True), 2)
            self.assertEqual(self.modal_app._modal_dataloader_workers(False), 2)
        finally:
            if previous_safe is None:
                os.environ.pop("MODEL_EXPRESS_DATALOADER_SAFE_DEFAULTS", None)
            else:
                os.environ["MODEL_EXPRESS_DATALOADER_SAFE_DEFAULTS"] = previous_safe
            if previous_workers is None:
                os.environ.pop("MODEL_EXPRESS_MODAL_DATALOADER_WORKERS", None)
            else:
                os.environ["MODEL_EXPRESS_MODAL_DATALOADER_WORKERS"] = previous_workers

    def test_modal_orchestrator_post_uses_longer_report_timeout(self) -> None:
        calls = []

        class Response:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
            calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return Response()

        with patch.dict("os.environ", {"MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS": "240"}):
            with patch("requests.post", fake_post):
                self.modal_app._post_json("http://orchestrator.test/jobs/job_1/complete", {"mlflow_run_id": "run_1"})

        self.assertEqual(calls[0]["timeout"], 240)
        self.assertIsNone(calls[0]["headers"])

    def test_modal_orchestrator_post_uses_callback_token_header(self) -> None:
        calls = []

        class Response:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
            calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return Response()

        with patch("requests.post", fake_post):
            self.modal_app._post_json(
                "http://orchestrator.test/jobs/job_1/complete",
                {"mlflow_run_id": "run_1"},
                callback_token="callback-secret",
            )

        self.assertEqual(calls[0]["headers"], {"Authorization": "Bearer callback-secret"})

    def test_modal_job_callbacks_use_callback_token_and_active_attempt(self) -> None:
        calls = []

        class Response:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
            calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return Response()

        job = {
            "id": "job_1",
            "training_attempt_id": "stale-top-level",
            "config": {
                "active_attempt_id": "job_1:attempt-2",
                "callback_token": "callback-secret",
            },
        }

        with patch("requests.post", fake_post):
            self.modal_app._post_job_json(
                "https://orchestrator.test",
                job,
                "complete",
                {"mlflow_run_id": "run_1"},
            )
            self.modal_app._post_training_run_summary(
                "https://orchestrator.test",
                "job_1",
                {"status": "RUNNING"},
                job=job,
            )
            self.modal_app._post_training_run_evaluation(
                "https://orchestrator.test",
                "job_1",
                {"objective_profile": {}},
                job=job,
            )

        self.assertEqual([call["json"]["training_attempt_id"] for call in calls], ["job_1:attempt-2"] * 3)
        self.assertTrue(all(call["headers"] == {"Authorization": "Bearer callback-secret"} for call in calls))

    def test_training_run_evaluation_retries_compacted_payload_after_413(self) -> None:
        calls = []

        class Response:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

            def raise_for_status(self) -> None:
                if self.status_code == 413:
                    exc = RuntimeError("413 Payload Too Large")
                    exc.response = self
                    raise exc

        def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
            calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return Response(413 if len(calls) == 1 else 200)

        original = "data:image/png;base64," + ("A" * 200_000)
        thumbnail = "data:image/jpeg;base64," + ("B" * 1_000)
        payload = {
            "objective_profile": {
                "heldout_demo_images": [
                    {
                        "uri": original,
                        "image_uri": original,
                        "preview_uri": thumbnail,
                        "thumbnail_uri": thumbnail,
                        "original_image_uri": "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png",
                        "metadata": {
                            "source_artifact_uri": "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png",
                            "parity_safe": True,
                        },
                    }
                ]
            }
        }

        with patch("requests.post", fake_post):
            self.modal_app._post_training_run_evaluation(
                "https://orchestrator.test",
                "job_1",
                payload,
                job={
                    "id": "job_1",
                    "config": {
                        "active_attempt_id": "job_1:attempt-2",
                        "callback_token": "callback-secret",
                    },
                },
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["json"]["objective_profile"]["heldout_demo_images"][0]["image_uri"].startswith("data:image/png"))
        retry_image = calls[1]["json"]["objective_profile"]["heldout_demo_images"][0]
        self.assertEqual(retry_image["image_uri"], "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png")
        self.assertTrue(retry_image["thumbnail_uri"].startswith("data:image/jpeg;base64,"))
        diagnostics = calls[1]["json"]["objective_profile"]["diagnostics"]
        self.assertEqual(diagnostics[-1]["code"], "evaluation_payload_compacted_due_to_size")

    def test_modal_failure_callback_uses_callback_token_and_active_attempt(self) -> None:
        calls = []

        class Response:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
            calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
            return Response()

        payload = {
            "job": {
                "id": "job_1",
                "config": {
                    "active_attempt_id": "job_1:attempt-2",
                    "callback_token": "callback-secret",
                },
            },
            "orchestrator_url": "https://orchestrator.test",
        }

        with patch("requests.post", fake_post):
            reported = self.modal_app._report_modal_training_retryable_failure(
                payload,
                RuntimeError("container exited unexpectedly"),
            )

        self.assertTrue(reported)
        self.assertEqual(calls[0]["url"], "https://orchestrator.test/jobs/job_1/fail")
        self.assertEqual(calls[0]["json"]["training_attempt_id"], "job_1:attempt-2")
        self.assertEqual(calls[0]["headers"], {"Authorization": "Bearer callback-secret"})

    def test_modal_dataset_timeouts_are_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_MODAL_MATERIALIZATION_TIMEOUT_SECONDS": "1800",
                "MODEL_EXPRESS_MODAL_PROFILE_TIMEOUT_SECONDS": "7200",
                "MODEL_EXPRESS_MODAL_TRAINING_TIMEOUT_SECONDS": "14400",
            },
        ):
            self.assertEqual(self.modal_app._modal_dataset_materialization_timeout_seconds(), 1800)
            self.assertEqual(self.modal_app._modal_dataset_profile_timeout_seconds(), 7200)
            self.assertEqual(self.modal_app._modal_training_timeout_seconds(), 14400)

    def test_modal_training_warm_pool_knobs_are_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS": "2",
                "MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS": "1",
                "MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS": "900",
            },
        ):
            self.assertEqual(self.modal_app._modal_training_min_containers(), 2)
            self.assertEqual(self.modal_app._modal_training_buffer_containers(), 1)
            self.assertEqual(self.modal_app._modal_training_scaledown_window_seconds(), 900)

        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS": "0",
                "MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS": "0",
                "MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS": "0",
            },
        ):
            self.assertIsNone(self.modal_app._modal_training_min_containers())
            self.assertIsNone(self.modal_app._modal_training_buffer_containers())
            self.assertIsNone(self.modal_app._modal_training_scaledown_window_seconds())

    def test_modal_cost_sensitive_defaults_reduce_scaledown_window(self) -> None:
        with patch.dict("os.environ", {"MODEL_EXPRESS_MODAL_COST_SENSITIVE_DEFAULTS": "1"}, clear=True):
            self.assertEqual(self.modal_app._modal_training_scaledown_window_seconds(), 120)
            self.assertIsNone(self.modal_app._modal_training_min_containers())
            self.assertIsNone(self.modal_app._modal_training_buffer_containers())

    def test_modal_stage_telemetry_payload_summarizes_phases(self) -> None:
        import time

        started_at = time.time() - 10
        stage_events = []
        token = self.modal_app._MODAL_STAGE_EVENTS.set(stage_events)
        try:
            with patch.dict("os.environ", {"MODEL_EXPRESS_REMOTE_GPU_STAGE_TELEMETRY": "1"}, clear=True):
                self.modal_app._modal_training_phase("job_1", "dataset_local_materialization_start", started_at)
                self.modal_app._modal_training_phase("job_1", "dataset_local_materialization_done", started_at)
                self.modal_app._modal_training_phase("job_1", "epoch_train_start", started_at, epoch=1)
                self.modal_app._modal_training_phase("job_1", "epoch_train_done", started_at, epoch=1)
                payload = self.modal_app._modal_stage_telemetry_payload(
                    {"id": "job_1", "created_at": "2026-06-09T00:00:00Z"},
                    12.0,
                    stage_events,
                    {
                        "dataset_materialization_extract_seconds": 1.2,
                        "dataset_materialization_wait_seconds": 0.3,
                        "dataset_materialization_download_seconds": 2.4,
                    },
                    "T4",
                )
        finally:
            self.modal_app._MODAL_STAGE_EVENTS.reset(token)

        self.assertEqual(payload["schema_version"], "remote_gpu_stage_telemetry_v1")
        self.assertEqual(payload["current_stage"], "epoch_train_done")
        self.assertGreaterEqual(payload["dataset_materialization_seconds"], 0)
        self.assertGreaterEqual(payload["active_training_seconds"], 0)
        self.assertEqual(payload["dataset_download_seconds"], 2.4)
        self.assertEqual(payload["dataset_extract_seconds"], 1.2)
        self.assertEqual(payload["warm_container_policy"]["scaledown_window_seconds"], 600)
        self.assertGreaterEqual(len(payload["events"]), 4)

    def test_modal_storage_env_sets_torch_home_default(self) -> None:
        payload = {
            "s3_endpoint_url": "https://s3.test",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "aws_default_region": "us-east-1",
            "storage_scope": {
                "token": "scope-token",
                "buckets": ["bucket"],
                "read_keys": ["datasets/project_1/data.zip"],
                "write_prefixes": ["model-express/artifacts/job_1/"],
            },
        }
        with patch.dict("os.environ", {}, clear=True):
            self.modal_app._configure_storage_env(payload)

            self.assertEqual(os.environ["TORCH_HOME"], str(self.modal_app.TORCH_CACHE_ROOT))

    def test_modal_torch_cache_sync_commit_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(self.modal_app._modal_sync_torch_cache_commit_enabled())

        with patch.dict("os.environ", {"MODEL_EXPRESS_MODAL_SYNC_TORCH_CACHE_COMMIT": "true"}):
            self.assertTrue(self.modal_app._modal_sync_torch_cache_commit_enabled())

    def test_modal_dataloader_workers_are_configurable(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.modal_app._modal_dataloader_workers(True), 4)
            self.assertEqual(self.modal_app._modal_dataloader_workers(False), 2)

        with patch.dict("os.environ", {"MODEL_EXPRESS_MODAL_DATALOADER_WORKERS": "0"}):
            self.assertEqual(self.modal_app._modal_dataloader_workers(True), 0)

        with patch.dict("os.environ", {"MODEL_EXPRESS_MODAL_DATALOADER_WORKERS": "99"}):
            self.assertEqual(self.modal_app._modal_dataloader_workers(True), 16)

    def test_modal_training_dataset_cache_root_is_local_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                self.modal_app._modal_training_dataset_cache_root(),
                Path("/tmp/model-express/training-datasets"),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {"MODEL_EXPRESS_MODAL_TRAINING_DATASET_CACHE_ROOT": temp_dir},
            ):
                self.assertEqual(self.modal_app._modal_training_dataset_cache_root(), Path(temp_dir))

    def test_modal_cache_relationship_marks_default_prewarm_staging_only(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            fields = self.modal_app._modal_dataset_cache_relationship_fields(
                self.modal_app.DATASET_MATERIALIZATION_ROOT,
                materialization_scope="modal_dataset_volume",
            )

        self.assertEqual(fields["dataset_materialization_cache_root"], "/cache/model-express/datasets")
        self.assertEqual(fields["dataset_materialization_cache_scope"], "modal_dataset_volume")
        self.assertEqual(fields["dataset_prewarm_cache_root"], "/cache/model-express/datasets")
        self.assertEqual(fields["dataset_training_cache_root"], "/tmp/model-express/training-datasets")
        self.assertFalse(fields["dataset_prewarm_root_matches_training_root"])
        self.assertFalse(fields["dataset_prewarm_reusable_for_training"])
        self.assertEqual(fields["dataset_prewarm_reuse_status"], "staging_only_root_mismatch")

    def test_modal_remote_paths_are_posix_for_image_builds(self) -> None:
        self.assertEqual(str(self.modal_app.TORCH_CACHE_ROOT), "/cache/model-express/torch")
        self.assertEqual(str(self.modal_app.DATASET_MATERIALIZATION_ROOT), "/cache/model-express/datasets")
        self.assertNotIn("\\", str(self.modal_app.TORCH_CACHE_ROOT))
        self.assertNotIn("\\", str(self.modal_app.DATASET_MATERIALIZATION_ROOT))

        self.assertEqual(
            str(self.modal_app._modal_remote_path_env("UNSET_MODAL_PATH", "\\cache\\custom")),
            "/cache/custom",
        )
        with self.assertRaisesRegex(ValueError, "absolute POSIX path"):
            self.modal_app._modal_remote_path_env("UNSET_MODAL_PATH", "cache/custom")

    def test_yolo_data_config_normalization_rewrites_stale_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "extracted"
            for split in ("train", "val"):
                image_dir = dataset_dir / "dataset" / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(image_dir / f"{split}.jpg")

            source_config = dataset_dir / "dataset.yaml"
            source_config.parent.mkdir(parents=True, exist_ok=True)
            source_config.write_text(
                "\n".join(
                    [
                        "path: /root/datasets/dataset",
                        "train: images/train",
                        "val: images/val",
                        "nc: 2",
                        "names: [drone, aircraft]",
                    ]
                ),
                encoding="utf-8",
            )

            normalized_path = self.modal_app._normalize_yolo_data_config_for_training(
                dataset_dir,
                source_config,
                output_root=Path(temp_dir) / "normalized",
            )
            normalized_text = normalized_path.read_text(encoding="utf-8")
            normalized = self.modal_app._load_yolo_training_config(normalized_path)

            self.assertEqual(normalized["path"], str(dataset_dir.resolve()))
            self.assertEqual(normalized["train"], str((dataset_dir / "dataset" / "images" / "train").resolve()))
            self.assertEqual(normalized["val"], str((dataset_dir / "dataset" / "images" / "val").resolve()))
            self.assertNotIn("/root/datasets", normalized_text)

    def test_yolo_per_class_metrics_uses_ultralytics_class_arrays(self) -> None:
        box = SimpleNamespace(
            map=0.42,
            map50=0.71,
            mp=0.66,
            mr=0.58,
            p=[0.9, 0.4],
            r=[0.8, 0.3],
            ap50=[0.95, 0.45],
            maps=[0.72, 0.33],
            ap_class_index=[0, 2],
        )
        metrics = SimpleNamespace(results_dict={}, box=box, speed={}, names={0: "cat", 1: "bird", 2: "dog"})

        parsed = self.modal_app._yolo_metrics_from_object(metrics, class_names=["cat", "bird", "dog"])
        per_class = self.modal_app._yolo_per_class_metrics(["cat", "bird", "dog"], parsed)

        self.assertEqual(per_class["cat"]["AP50_95"], 0.72)
        self.assertEqual(per_class["dog"]["AP50_95"], 0.33)
        self.assertNotEqual(per_class["cat"]["AP50_95"], per_class["dog"]["AP50_95"])
        self.assertEqual(per_class["macro avg"]["AP50_95"], 0.525)

    def test_yolo_per_class_metrics_does_not_clone_aggregate_metrics(self) -> None:
        per_class = self.modal_app._yolo_per_class_metrics(
            ["cat", "dog"],
            {"mAP50_95": 0.72, "mAP50": 0.95, "precision": 0.9, "recall": 0.8},
        )

        self.assertEqual(per_class, {})

    def test_yolo_epoch_metric_posting_skips_previously_posted_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            results_path = run_root / "train" / "results.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                "\n".join(
                    [
                        "epoch,metrics/mAP50-95(B),metrics/mAP50(B),metrics/precision(B),metrics/recall(B),val/box_loss,val/cls_loss,val/dfl_loss",
                        "1,0.2,0.4,0.5,0.6,0.7,0.8,0.9",
                        "2,0.3,0.5,0.6,0.7,0.6,0.7,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_post(url: str, payload: dict) -> None:
                calls.append({"url": url, "payload": payload})

            posted_epochs: set[int] = set()
            with patch.object(self.modal_app, "_post_json", fake_post):
                first_count = self.modal_app._post_yolo_epoch_metrics(
                    "https://orchestrator.test",
                    "job_1",
                    run_root,
                    learning_rate=0.01,
                    image_size=640,
                    posted_epochs=posted_epochs,
                )
                second_count = self.modal_app._post_yolo_epoch_metrics(
                    "https://orchestrator.test",
                    "job_1",
                    run_root,
                    learning_rate=0.01,
                    image_size=640,
                    posted_epochs=posted_epochs,
                )

        self.assertEqual(first_count, 2)
        self.assertEqual(second_count, 0)
        self.assertEqual(posted_epochs, {1, 2})
        self.assertEqual(len(calls), 2)

    def test_yolo_epoch_metric_callbacks_use_callback_token_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            results_path = run_root / "train" / "results.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                "\n".join(
                    [
                        "epoch,metrics/mAP50-95(B),metrics/mAP50(B),metrics/precision(B),metrics/recall(B),val/box_loss,val/cls_loss,val/dfl_loss",
                        "1,0.2,0.4,0.5,0.6,0.7,0.8,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            calls = []

            class Response:
                def raise_for_status(self) -> None:
                    return None

            def fake_post(url: str, *, json: dict, timeout: int, headers: dict | None = None):
                calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
                return Response()

            with patch("requests.post", fake_post):
                posted_count = self.modal_app._post_yolo_epoch_metrics(
                    "https://orchestrator.test",
                    "job_1",
                    run_root,
                    learning_rate=0.01,
                    image_size=640,
                    callback_identity={"training_attempt_id": "job_1:attempt-2"},
                    callback_auth_token="callback-secret",
                )

        self.assertEqual(posted_count, 1)
        self.assertEqual(calls[0]["json"]["training_attempt_id"], "job_1:attempt-2")
        self.assertEqual(calls[0]["headers"], {"Authorization": "Bearer callback-secret"})

    def test_modal_payload_configures_callback_env_and_clears_stale_values(self) -> None:
        payload = {
            "job": {
                "id": "job_1",
                "config": {
                    "active_attempt_id": "job_1:attempt-2",
                    "callback_token": "callback-secret",
                },
            },
            "s3_endpoint_url": "https://s3.test",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "aws_default_region": "us-east-1",
            "storage_scope": {
                "token": "scope-token",
                "buckets": ["bucket"],
                "read_keys": ["datasets/project_1/data.zip"],
                "write_prefixes": ["model-express/artifacts/job_1/"],
            },
        }
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_CALLBACK_TOKEN": "stale-token",
                "MODEL_EXPRESS_TRAINING_ATTEMPT_ID": "stale-attempt",
            },
            clear=False,
        ):
            self.modal_app._configure_storage_env(payload)
            self.assertEqual(os.environ["MODEL_EXPRESS_CALLBACK_TOKEN"], "callback-secret")
            self.assertEqual(os.environ["MODEL_EXPRESS_TRAINING_ATTEMPT_ID"], "job_1:attempt-2")
            self.assertEqual(os.environ["MODEL_EXPRESS_STORAGE_SCOPE_TOKEN"], "scope-token")
            self.assertEqual(os.environ["MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE"], "true")
            self.assertEqual(json.loads(os.environ["MODEL_EXPRESS_STORAGE_SCOPE"])["write_prefixes"], ["model-express/artifacts/job_1/"])

            self.modal_app._configure_storage_env(
                {
                    "job": {"id": "job_2", "config": {}},
                    "s3_endpoint_url": "https://s3.test",
                    "aws_access_key_id": "key",
                    "aws_secret_access_key": "secret",
                    "aws_default_region": "us-east-1",
                }
            )
            self.assertNotIn("MODEL_EXPRESS_CALLBACK_TOKEN", os.environ)
            self.assertNotIn("MODEL_EXPRESS_TRAINING_ATTEMPT_ID", os.environ)
            self.assertNotIn("MODEL_EXPRESS_STORAGE_SCOPE", os.environ)
            self.assertNotIn("MODEL_EXPRESS_STORAGE_SCOPE_TOKEN", os.environ)
            self.assertNotIn("MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE", os.environ)

    def test_modal_storage_scope_limits_s3_keys(self) -> None:
        from worker.datasets import storage

        scope = {
            "token": "scope-token",
            "buckets": ["bucket"],
            "read_keys": ["datasets/project_1/data.zip"],
            "write_prefixes": ["model-express/artifacts/job_1/"],
        }
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_STORAGE_SCOPE": json.dumps(scope),
                "MODEL_EXPRESS_STORAGE_SCOPE_TOKEN": "scope-token",
                "MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE": "true",
            },
            clear=True,
        ):
            storage.enforce_storage_scope("read", "bucket", "datasets/project_1/data.zip")
            storage.enforce_storage_scope("write", "bucket", "model-express/artifacts/job_1/model.onnx")
            with self.assertRaisesRegex(ValueError, "outside the remote storage scope"):
                storage.enforce_storage_scope("read", "bucket", "datasets/project_2/data.zip")
            with self.assertRaisesRegex(ValueError, "outside the remote storage scope"):
                storage.enforce_storage_scope("write", "bucket", "model-express/artifacts/job_2/model.onnx")
            with self.assertRaisesRegex(ValueError, "outside the remote storage scope"):
                storage.enforce_storage_scope("write", "bucket", "model-express/artifacts/job_10/model.onnx")
            with self.assertRaisesRegex(ValueError, "path traversal"):
                storage.enforce_storage_scope("write", "bucket", "model-express/artifacts/job_1/%2e%2e/job_2/model.onnx")
            with self.assertRaisesRegex(ValueError, "outside the remote storage scope"):
                storage.enforce_storage_scope("read", "other-bucket", "datasets/project_1/data.zip")

        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_STORAGE_SCOPE": json.dumps(scope),
                "MODEL_EXPRESS_STORAGE_SCOPE_TOKEN": "wrong-token",
                "MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE": "true",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                storage.enforce_storage_scope("read", "bucket", "datasets/project_1/data.zip")

        with patch.dict("os.environ", {"MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE": "true"}, clear=True):
            with self.assertRaisesRegex(ValueError, "required"):
                storage.enforce_storage_scope("read", "bucket", "datasets/project_1/data.zip")

        expired_scope = {**scope, "expires_at": "2000-01-01T00:00:00+00:00"}
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_STORAGE_SCOPE": json.dumps(expired_scope),
                "MODEL_EXPRESS_STORAGE_SCOPE_TOKEN": "scope-token",
                "MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE": "true",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "expired"):
                storage.enforce_storage_scope("read", "bucket", "datasets/project_1/data.zip")

    def test_modal_training_failure_report_marks_retryable(self) -> None:
        calls = []

        def fake_post(url: str, payload: dict) -> None:
            calls.append({"url": url, "payload": payload})

        payload = {
            "job": {"id": "job_1"},
            "orchestrator_url": "https://orchestrator.test",
        }
        with patch.object(self.modal_app, "_post_json", fake_post):
            reported = self.modal_app._report_modal_training_retryable_failure(
                payload,
                RuntimeError("container exited unexpectedly"),
            )

            self.assertTrue(reported)
            self.assertEqual(calls[0]["url"], "https://orchestrator.test/jobs/job_1/fail")
            self.assertTrue(calls[0]["payload"]["retryable"])
            self.assertIn("container exited unexpectedly", calls[0]["payload"]["error"])

    def test_modal_yolo_path_resolver_rejects_paths_outside_materialized_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [cat]\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "parent traversal"):
                self.modal_app._resolve_existing_yolo_path(
                    dataset_dir,
                    data_yaml,
                    {"path": "."},
                    "../outside/images",
                )
    def test_modal_preview_batch_shell_materializes_once_and_runs_classification_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            materialize_calls = []
            train_payloads = []

            def fake_materialize(**kwargs):
                materialize_calls.append(kwargs)
                return SimpleNamespace(
                    dataset_dir=dataset_dir,
                    telemetry={
                        "dataset_materialization_cache_hit": False,
                        "dataset_materialization_cache_miss": True,
                        "dataset_materialization_bytes_downloaded": 123,
                    },
                )

            def fake_train(payload: dict) -> dict:
                train_payloads.append(payload)
                return {
                    "job_id": payload["job"]["id"],
                    "model": "mobilenet_v3_small",
                    "best_accuracy": 0.8,
                    "best_macro_f1": 0.7,
                    "runtime_seconds": 1.2,
                }

            payload = self._modal_preview_batch_payload()
            with patch("worker.datasets.cache.ensure_dataset_materialized", fake_materialize):
                with patch.object(self.modal_app, "_train_image_classifier_impl", fake_train):
                    result = self.modal_app._train_modal_preview_batch_impl(payload)

        self.assertEqual(len(materialize_calls), 1)
        self.assertIn("model-express/batches/modal-preview-batch-test", str(materialize_calls[0]["cache_root"]).replace("\\", "/"))
        self.assertEqual(len(train_payloads), 2)
        self.assertEqual([payload["job"]["id"] for payload in train_payloads], ["job_1", "job_2"])
        self.assertTrue(all(payload["_modal_pre_materialized_dataset"]["dataset_dir"] == str(dataset_dir) for payload in train_payloads))
        self.assertEqual(result["schema_version"], "modal_preview_batch_result.v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["runner_status"], "classification_batch_completed")
        self.assertEqual([job["status"] for job in result["job_results"]], ["succeeded", "succeeded"])
        self.assertEqual(result["dataset_materialization"]["dataset_materialization_reused_by_jobs"], 2)

    def test_modal_preview_batch_shell_continues_after_classification_job_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            fail_calls = []

            def fake_materialize(**_kwargs):
                return SimpleNamespace(dataset_dir=dataset_dir, telemetry={})

            def fake_train(payload: dict) -> dict:
                if payload["job"]["id"] == "job_1":
                    raise RuntimeError("first trial failed")
                return {"job_id": payload["job"]["id"], "model": "mobilenet_v3_small"}

            def fake_post(url: str, payload: dict) -> None:
                fail_calls.append({"url": url, "payload": payload})

            with patch("worker.datasets.cache.ensure_dataset_materialized", fake_materialize):
                with patch.object(self.modal_app, "_train_image_classifier_impl", fake_train):
                    with patch.object(self.modal_app, "_post_json", fake_post):
                        result = self.modal_app._train_modal_preview_batch_impl(self._modal_preview_batch_payload())

        self.assertEqual([job["status"] for job in result["job_results"]], ["retryable_failure_reported", "succeeded"])
        self.assertEqual(len(fail_calls), 1)
        self.assertEqual(fail_calls[0]["url"], "https://orchestrator.test/jobs/job_1/fail")
        self.assertTrue(fail_calls[0]["payload"]["retryable"])

    def test_modal_preview_batch_shell_returns_unsupported_for_yolo_until_phase5(self) -> None:
        payload = self._modal_preview_batch_payload(task_type="object_detection", model="yolo11n.pt")

        result = self.modal_app._train_modal_preview_batch_impl(payload)

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["runner_status"], "remote_batch_shell_unsupported")
        self.assertTrue(all(job["status"] == "unsupported" for job in result["job_results"]))

    def test_modal_preview_batch_shell_runs_yolo_jobs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            for split in ("train", "val"):
                image_dir = dataset_dir / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(image_dir / f"{split}.jpg")
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text("train: images/train\nval: images/val\nnc: 2\nnames: [cat, dog]\n", encoding="utf-8")
            materialize_calls = []
            yolo_payloads = []

            def fake_materialize(**kwargs):
                materialize_calls.append(kwargs)
                return SimpleNamespace(
                    dataset_dir=dataset_dir,
                    telemetry={"dataset_materialization_cache_miss": True},
                )

            def fake_prepare(dataset_dir_arg, data_config_path_arg, config, **kwargs):
                self.assertEqual(dataset_dir_arg, dataset_dir)
                self.assertEqual(data_config_path_arg.name, "data.yaml")
                self.assertNotEqual(data_config_path_arg, data_yaml)
                normalized = self.modal_app._load_yolo_training_config(data_config_path_arg)
                self.assertEqual(normalized["path"], str(dataset_dir.resolve()))
                self.assertTrue(kwargs["force_preview"])
                return data_yaml, {"manifest_id": "preview-subset-test", "seed": 42}

            def fake_yolo_train(payload: dict) -> dict:
                yolo_payloads.append(payload)
                return {
                    "job_id": payload["job"]["id"],
                    "model": payload["job"]["config"]["model"],
                    "mAP50_95": 0.3,
                    "runtime_seconds": 1.0,
                }

            payload = self._modal_preview_batch_payload(task_type="object_detection", model="yolo11n.pt")
            payload["jobs"][1]["config"]["model"] = "yolo11s.pt"
            with patch.dict("os.environ", {"MODEL_EXPRESS_YOLO_BATCH_PREVIEW": "1"}):
                with patch("worker.datasets.cache.ensure_dataset_materialized", fake_materialize):
                    with patch.object(self.modal_app, "_prepare_yolo_dataset_tier", fake_prepare):
                        with patch.object(self.modal_app, "_train_yolo_detector_impl", fake_yolo_train):
                            result = self.modal_app._train_modal_preview_batch_impl(payload)

        self.assertEqual(len(materialize_calls), 1)
        self.assertEqual([call["job"]["id"] for call in yolo_payloads], ["job_1", "job_2"])
        self.assertTrue(
            all(call["_modal_pre_materialized_dataset"]["yolo_data_config"] == str(data_yaml) for call in yolo_payloads)
        )
        self.assertTrue(
            all(
                call["_modal_pre_materialized_dataset"]["subset_manifest"]["manifest_id"] == "preview-subset-test"
                for call in yolo_payloads
            )
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["runner_status"], "yolo_batch_completed")
        self.assertEqual([job["status"] for job in result["job_results"]], ["succeeded", "succeeded"])
        self.assertEqual(result["dataset_materialization"]["subset_manifest"]["manifest_id"], "preview-subset-test")

    def test_modal_preview_batch_shell_continues_after_yolo_job_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            for split in ("train", "val"):
                image_dir = dataset_dir / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(image_dir / f"{split}.jpg")
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text("train: images/train\nval: images/val\nnc: 2\nnames: [cat, dog]\n", encoding="utf-8")
            fail_calls = []

            def fake_materialize(**_kwargs):
                return SimpleNamespace(dataset_dir=dataset_dir, telemetry={})

            def fake_yolo_train(payload: dict) -> dict:
                if payload["job"]["id"] == "job_1":
                    raise RuntimeError("first detector failed")
                return {"job_id": payload["job"]["id"], "model": "yolo11s.pt"}

            def fake_post(url: str, payload: dict) -> None:
                fail_calls.append({"url": url, "payload": payload})

            payload = self._modal_preview_batch_payload(task_type="object_detection", model="yolo11n.pt")
            with patch.dict("os.environ", {"MODEL_EXPRESS_YOLO_BATCH_PREVIEW": "1"}):
                with patch("worker.datasets.cache.ensure_dataset_materialized", fake_materialize):
                    with patch.object(self.modal_app, "_prepare_yolo_dataset_tier", lambda *_args, **_kwargs: (data_yaml, {})):
                        with patch.object(self.modal_app, "_train_yolo_detector_impl", fake_yolo_train):
                            with patch.object(self.modal_app, "_post_json", fake_post):
                                result = self.modal_app._train_modal_preview_batch_impl(payload)

        self.assertEqual([job["status"] for job in result["job_results"]], ["retryable_failure_reported", "succeeded"])
        self.assertEqual(len(fail_calls), 1)
        self.assertEqual(fail_calls[0]["url"], "https://orchestrator.test/jobs/job_1/fail")
        self.assertTrue(fail_calls[0]["payload"]["retryable"])

    def test_modal_preview_batch_shell_rejects_mismatched_cache_key(self) -> None:
        payload = {
            "batch": {
                "schema_version": "modal_preview_batch.v1",
                "batch_id": "modal-preview-batch-test",
                "batch_key": "project_1|plan_1|dataset_1|sha256-a|preview|image_classification",
                "project_id": "project_1",
                "plan_id": "plan_1",
                "dataset_id": "dataset_1",
                "dataset_cache_key": "sha256-a",
                "training_tier": "preview",
                "task_type": "image_classification",
            },
            "jobs": [
                {
                    "id": "job_1",
                    "project_id": "project_1",
                    "template": "train_experiment",
                    "config": {
                        "provider": "modal",
                        "plan_id": "plan_1",
                        "dataset_id": "dataset_1",
                        "training_tier": "preview",
                        "dataset_materialization": {"dataset_cache_key": "sha256-b"},
                    },
                },
                {
                    "id": "job_2",
                    "project_id": "project_1",
                    "template": "train_experiment",
                    "config": {
                        "provider": "modal",
                        "plan_id": "plan_1",
                        "dataset_id": "dataset_1",
                        "training_tier": "preview",
                        "dataset_materialization": {"dataset_cache_key": "sha256-a"},
                    },
                },
            ],
            "dataset": {"id": "dataset_1", "storage_uri": "s3://bucket/dataset.zip"},
            "orchestrator_url": "https://orchestrator.test",
        }

        with self.assertRaisesRegex(ValueError, "dataset_cache_key does not match"):
            self.modal_app._train_modal_preview_batch_impl(payload)

    def test_upload_manifest_artifacts_uploads_onnx_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.onnx"
            sidecar = root / "model.onnx.data"
            bundle = root / "portable_inference_bundle.zip"
            model.write_bytes(b"onnx bytes")
            sidecar.write_bytes(b"external tensor bytes")
            bundle.write_bytes(b"zip bytes")
            uploads = []

            def fake_upload(source: Path, destination: str) -> None:
                uploads.append((Path(source).name, destination))

            manifest = {
                "metadata": {
                    "portable_inference_bundle": {
                        "schema_version": "portable_inference_bundle_v1",
                        "status": "created",
                        "artifact_uri": bundle.resolve().as_uri(),
                        "artifact_path": str(bundle),
                        "contents": ["model.onnx", "manifest.json"],
                    }
                },
                "artifacts": [
                    {
                        "format": "onnx",
                        "status": "created",
                        "path": str(model),
                        "external_data": [
                            {
                                "path": "model.onnx.data",
                                "artifact_path": str(sidecar),
                                "bytes": sidecar.stat().st_size,
                            }
                        ],
                    },
                    {
                        "format": "portable_inference_bundle",
                        "status": "created",
                        "path": str(bundle),
                        "contents": ["model.onnx", "manifest.json"],
                    },
                ],
            }

            public_manifest, artifact_uris = self.modal_app._upload_manifest_artifacts(
                manifest,
                "s3://bucket/exports/job_1",
                fake_upload,
            )

            self.assertEqual(
                uploads,
                [
                    ("model.onnx", "s3://bucket/exports/job_1/model.onnx"),
                    ("model.onnx.data", "s3://bucket/exports/job_1/model.onnx.data"),
                    ("portable_inference_bundle.zip", "s3://bucket/exports/job_1/portable_inference_bundle.zip"),
                ],
            )
            self.assertEqual(
                artifact_uris,
                [
                    {"format": "onnx", "uri": "s3://bucket/exports/job_1/model.onnx"},
                    {
                        "format": "portable_inference_bundle",
                        "uri": "s3://bucket/exports/job_1/portable_inference_bundle.zip",
                    },
                ],
            )
            self.assertEqual(public_manifest["artifacts"][0]["external_data"][0]["uri"], "s3://bucket/exports/job_1/model.onnx.data")
            self.assertEqual(public_manifest["metadata"]["portable_bundle_uri"], "s3://bucket/exports/job_1/portable_inference_bundle.zip")
            self.assertEqual(
                public_manifest["metadata"]["portable_inference_bundle"]["artifact_uri"],
                "s3://bucket/exports/job_1/portable_inference_bundle.zip",
            )

    def test_profile_image_dataset_uses_ephemeral_materialized_dataset_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            (dataset_dir / "cat").mkdir(parents=True)
            (dataset_dir / "cat" / "one.txt").write_text("meow", encoding="utf-8")
            calls = []

            def fake_materialize(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    dataset_dir=dataset_dir,
                    telemetry={
                        "dataset_materialization_cache_hit": True,
                        "dataset_materialization_cache_miss": False,
                        "dataset_materialization_bytes_downloaded": 0,
                        "dataset_materialization_extract_seconds": 0.0,
                        "dataset_materialization_wait_seconds": 0.0,
                        "dataset_checksum": "d" * 64,
                        "storage_uri_fingerprint": "fingerprint",
                    },
                )

            payload = {
                "job": {
                    "id": "job_1",
                    "config": {
                        "dataset_materialization": {
                            "dataset_checksum_sha256": "d" * 64,
                            "dataset_cache_key": "sha256-" + ("d" * 64),
                        }
                    },
                },
                "dataset": {
                    "id": "dataset_1",
                    "storage_uri": "s3://bucket/dataset.zip",
                },
                "s3_endpoint_url": "https://s3.test",
                "aws_access_key_id": "key",
                "aws_secret_access_key": "secret",
                "aws_default_region": "us-east-1",
            }
            with patch("worker.datasets.cache.ensure_dataset_materialized", fake_materialize):
                with patch("worker.datasets.profiler.profile_image_folder", lambda path: {"profiled": str(path)}):
                    with patch("worker.datasets.metadata_discovery.build_metadata_import_payload", lambda path: {"root": str(path)}):
                        result = self.modal_app.profile_image_dataset(payload)

            self.assertEqual(calls[0]["dataset_id"], "dataset_1")
            self.assertEqual(calls[0]["storage_uri"], "s3://bucket/dataset.zip")
            self.assertEqual(calls[0]["checksum_sha256"], "d" * 64)
            self.assertNotEqual(Path(calls[0]["cache_root"]), self.modal_app.DATASET_MATERIALIZATION_ROOT)
            self.assertIn("model-express-profile-dataset_1-", str(calls[0]["cache_root"]))
            self.assertEqual(result["profile"]["profiled"], str(dataset_dir))
            self.assertTrue(result["dataset_materialization"]["dataset_materialization_cache_hit"])

    def test_transform_uses_dataset_normalization_metadata(self) -> None:
        try:
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        transform = self.modal_app._image_transform(
            128,
            {},
            {
                "resize_strategy": "center_crop",
                "normalization": "dataset",
                "normalization_metadata": {
                    "mean": [0.1, 0.2, 0.3],
                    "std": [0.4, 0.5, 0.6],
                },
            },
            training=False,
        )

        step_names = [step.__class__.__name__ for step in transform.transforms]
        normalize_step = transform.transforms[-1]

        self.assertEqual(step_names[:2], ["Resize", "CenterCrop"])
        self.assertEqual(step_names[-2:], ["ToTensor", "Normalize"])
        self.assertEqual(list(normalize_step.mean), [0.1, 0.2, 0.3])
        self.assertEqual(list(normalize_step.std), [0.4, 0.5, 0.6])

    def test_transform_can_skip_normalization(self) -> None:
        try:
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        transform = self.modal_app._image_transform(
            96,
            {},
            {"resize_strategy": "squash", "normalization": "none"},
            training=False,
        )

        self.assertNotIn("Normalize", [step.__class__.__name__ for step in transform.transforms])

    def test_structured_randaugment_policy_is_capped_and_train_only(self) -> None:
        try:
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc
        if not hasattr(transforms, "RandAugment"):
            raise unittest.SkipTest("torchvision RandAugment is unavailable")

        augmentation = self.modal_app.normalize_augmentation_config(
            {"horizontal_flip": True},
            "",
            {"policy_type": "randaugment", "magnitude": 99, "num_ops": 10, "probability": 0.25},
        )

        train_transform = self.modal_app._image_transform(
            128,
            augmentation,
            {"resize_strategy": "squash", "normalization": "none"},
            training=True,
        )
        val_transform = self.modal_app._image_transform(
            128,
            augmentation,
            {"resize_strategy": "squash", "normalization": "none"},
            training=False,
        )

        train_step_names = [step.__class__.__name__ for step in train_transform.transforms]
        random_apply = next(
            step for step in train_transform.transforms if step.__class__.__name__ == "RandomApply"
        )
        randaugment = random_apply.transforms[0]

        self.assertIn("RandomApply", train_step_names)
        self.assertNotIn("RandomApply", [step.__class__.__name__ for step in val_transform.transforms])
        self.assertEqual(random_apply.p, 0.25)
        self.assertEqual(randaugment.num_ops, 3)
        self.assertEqual(randaugment.magnitude, 15)

    def test_structured_trivialaugment_and_autoaugment_are_supported(self) -> None:
        try:
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc
        if not hasattr(transforms, "TrivialAugmentWide") or not hasattr(transforms, "AutoAugment"):
            raise unittest.SkipTest("torchvision structured augmentation transforms are unavailable")

        for policy_type, expected_step in (
            ("trivialaugment", "TrivialAugmentWide"),
            ("autoaugment", "AutoAugment"),
        ):
            with self.subTest(policy_type=policy_type):
                augmentation = self.modal_app.normalize_augmentation_config(
                    {},
                    "",
                    {"policy_type": policy_type},
                )
                transform = self.modal_app._image_transform(
                    128,
                    augmentation,
                    {"resize_strategy": "squash", "normalization": "none"},
                    training=True,
                )

                self.assertIn(expected_step, [step.__class__.__name__ for step in transform.transforms])

    def test_unknown_structured_policy_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported augmentation policy_type 'mystery'"):
            self.modal_app.normalize_augmentation_config({}, "", {"policy_type": "mystery"})

    def test_augmentation_policy_config_preserves_existing_map_overrides(self) -> None:
        augmentation = self.modal_app.normalize_augmentation_config(
            {"magnitude": 7, "vertical_flip": True},
            "",
            {"policy_type": "randaugment", "magnitude": 99, "num_ops": 10},
        )
        basic = self.modal_app.normalize_augmentation_config({}, "", {"policy_type": "basic"})

        self.assertEqual(augmentation["policy_type"], "randaugment")
        self.assertEqual(augmentation["magnitude"], 7)
        self.assertEqual(augmentation["num_ops"], 3)
        self.assertTrue(augmentation["vertical_flip"])
        self.assertTrue(basic["horizontal_flip"])

    def test_mixup_and_cutmix_are_normalized_with_caps(self) -> None:
        for policy_type in ("mixup", "cutmix"):
            with self.subTest(policy_type=policy_type):
                augmentation = self.modal_app.normalize_augmentation_config(
                    {},
                    "",
                    {"policy_type": policy_type, "alpha": 99, "probability": -1},
                )

                self.assertEqual(augmentation["policy_type"], policy_type)
                self.assertEqual(augmentation["alpha"], 1.0)
                self.assertEqual(augmentation["probability"], 0.0)

    def test_mixup_applies_soft_labels_for_training_batches(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        inputs = torch.stack([torch.zeros((3, 4, 4)), torch.ones((3, 4, 4))])
        labels = torch.tensor([0, 1])
        augmentation = {"policy_type": "mixup", "alpha": 0.4, "probability": 1.0}

        mixed_inputs, mixed_labels = self.modal_app._apply_mixed_sample_augmentation(
            inputs,
            labels,
            augmentation,
            class_count=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(mixed_labels.shape), (2, 2))
        self.assertTrue(torch.allclose(mixed_labels.sum(dim=1), torch.ones(2)))
        self.assertEqual(tuple(mixed_inputs.shape), tuple(inputs.shape))

    def test_mixed_sample_policy_does_not_create_eval_image_transform(self) -> None:
        try:
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        augmentation = self.modal_app.normalize_augmentation_config(
            {},
            "",
            {"policy_type": "cutmix", "alpha": 0.2, "probability": 1.0},
        )

        transform = self.modal_app._image_transform(
            64,
            augmentation,
            {"resize_strategy": "squash", "normalization": "none"},
            training=True,
        )

        self.assertNotIn("RandomApply", [step.__class__.__name__ for step in transform.transforms])

    def test_sampler_selection_is_explicit(self) -> None:
        self.assertTrue(self.modal_app._uses_weighted_sampler("none", "weighted_random_sampler"))
        self.assertTrue(self.modal_app._uses_weighted_sampler("class_balanced_sampler", "none"))
        self.assertFalse(self.modal_app._uses_weighted_sampler("weighted_loss", "none"))

    def test_loss_selection_supports_weighted_and_focal_loss(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        device = torch.device("cpu")
        weights = torch.tensor([0.25, 0.75], dtype=torch.float32)

        weighted_loss = self.modal_app._build_criterion(weights, "weighted_loss", device)
        focal_loss = self.modal_app._build_criterion(weights, "focal_loss", device)

        self.assertIsInstance(weighted_loss, nn.CrossEntropyLoss)
        self.assertEqual(focal_loss.__class__.__name__, "FocalLoss")
        self.assertTrue(torch.equal(weighted_loss.weight, weights))
        self.assertTrue(torch.equal(focal_loss.weight, weights))

    def test_deferred_hyperparameters_are_worker_configurable(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        parameter = nn.Parameter(torch.ones(1, requires_grad=True))
        optimizer = self.modal_app._build_optimizer(
            "sgd",
            [parameter],
            learning_rate=0.01,
            weight_decay=0.02,
            momentum=0.82,
        )
        scheduler = self.modal_app._build_scheduler(
            "step",
            optimizer,
            epochs=9,
            step_size=2,
            gamma=0.35,
        )
        criterion = self.modal_app._build_criterion(
            None,
            "none",
            torch.device("cpu"),
            label_smoothing=0.12,
        )
        focal = self.modal_app._build_criterion(
            None,
            "focal_loss",
            torch.device("cpu"),
            class_balancing_config={"focal_loss_gamma": 3.0},
        )
        head = self.modal_app._classification_head(nn, 4, 2, dropout=0.25)

        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.82)
        self.assertEqual(scheduler.step_size, 2)
        self.assertEqual(scheduler.gamma, 0.35)
        self.assertEqual(criterion.label_smoothing, 0.12)
        self.assertEqual(focal.gamma, 3.0)
        self.assertIsInstance(head[0], nn.Dropout)
        self.assertEqual(head[0].p, 0.25)

    def test_early_stopping_waits_until_after_half_epochs_for_non_egregious_runs(self) -> None:
        should_stop = self.modal_app._should_stop_training_early

        self.assertFalse(
            should_stop(
                epoch=8,
                epochs=30,
                best_epoch=4,
                early_stopping_patience=4,
                best_accuracy=0.52,
                best_macro_f1=0.41,
                target_metric="macro_f1",
            )
        )
        self.assertTrue(
            should_stop(
                epoch=16,
                epochs=30,
                best_epoch=10,
                early_stopping_patience=4,
                best_accuracy=0.52,
                best_macro_f1=0.41,
                target_metric="macro_f1",
            )
        )

    def test_early_stopping_allows_egregious_target_metric_after_warmup(self) -> None:
        should_stop = self.modal_app._should_stop_training_early

        self.assertTrue(
            should_stop(
                epoch=8,
                epochs=30,
                best_epoch=4,
                early_stopping_patience=4,
                best_accuracy=0.65,
                best_macro_f1=0.15,
                target_metric="macro_f1",
            )
        )
        self.assertFalse(
            should_stop(
                epoch=8,
                epochs=30,
                best_epoch=4,
                early_stopping_patience=4,
                best_accuracy=0.65,
                best_macro_f1=0.15,
                target_metric="accuracy",
            )
        )

    def test_effective_number_class_weights_are_normalized(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        weights = self.modal_app._class_weights(
            targets=[0, 0, 0, 1],
            train_indices=[0, 1, 2, 3],
            class_count=2,
            class_balancing="effective_number_class_balanced_loss",
            class_balancing_config={"effective_number_beta": 0.99},
        )

        self.assertIsNotNone(weights)
        self.assertTrue(torch.isclose(weights.sum(), torch.tensor(2.0), atol=1e-4))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_bbox_lookup_and_crop_dataset_use_annotations(self) -> None:
        try:
            from torchvision import datasets, transforms
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            class_dir = dataset_dir / "cat"
            annotations_dir = dataset_dir / "annotations"
            class_dir.mkdir()
            annotations_dir.mkdir()
            Image.new("RGB", (20, 20), (0, 0, 0)).save(class_dir / "one.jpg")
            (annotations_dir / "one.xml").write_text(
                "<annotation><filename>one.jpg</filename><size><width>20</width><height>20</height></size>"
                "<object><name>cat</name><bndbox><xmin>5</xmin><ymin>5</ymin>"
                "<xmax>15</xmax><ymax>15</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )

            lookup = self.modal_app._load_bbox_lookup(dataset_dir)
            dataset = self.modal_app._image_folder_dataset(
                datasets,
                dataset_dir,
                transform=transforms.Resize((8, 8)),
            )
            cropped = self.modal_app._BBoxCropDataset(dataset, lookup, required=True)
            image, label = cropped[0]

            self.assertIn("one.jpg", lookup)
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 0)

    def test_bbox_lookup_reads_cub_sidecar_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            cub_root = dataset_dir / "CUB_200_2011"
            image_path = cub_root / "images" / "001.Black_footed_Albatross" / "one.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), (240, 20, 20)).save(image_path)
            (cub_root / "images.txt").write_text("1 001.Black_footed_Albatross/one.jpg\n", encoding="utf-8")
            (cub_root / "bounding_boxes.txt").write_text("1 2 3 30 40\n", encoding="utf-8")

            lookup = self.modal_app._load_bbox_lookup(dataset_dir)

            self.assertEqual(lookup[str(image_path.resolve()).lower()], (2, 3, 32, 43))
            self.assertEqual(lookup["one.jpg"], (2, 3, 32, 43))

    def test_image_folder_dataset_uses_common_top_level_image_root(self) -> None:
        try:
            from torchvision import datasets
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for class_name, color in (("cat", (255, 0, 0)), ("dog", (0, 0, 255))):
                class_dir = dataset_dir / "images" / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 12), color).save(class_dir / "one.jpg")
            (dataset_dir / "attributes").mkdir()
            (dataset_dir / "parts").mkdir()

            dataset = self.modal_app._image_folder_dataset(datasets, dataset_dir)

            self.assertEqual(dataset.classes, ["cat", "dog"])
            self.assertTrue(all("/images/" in Path(path).as_posix() for path, _label in dataset.samples))

    def test_load_image_data_scans_imagefolder_once_for_fallback(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"training dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for class_name, color in (("cat", (255, 0, 0)), ("dog", (0, 0, 255))):
                class_dir = dataset_dir / class_name
                class_dir.mkdir()
                for index in range(2):
                    Image.new("RGB", (12, 12), color).save(class_dir / f"{index}.jpg")

            with patch.object(
                self.modal_app,
                "_image_folder_dataset",
                wraps=self.modal_app._image_folder_dataset,
            ) as image_folder_dataset:
                self.modal_app._load_image_data(
                    dataset_dir,
                    batch_size=2,
                    image_size=32,
                    augmentation={},
                    class_balancing="none",
                    sampling_strategy="none",
                    preprocessing={"resize_strategy": "squash", "normalization": "none"},
                    metadata_bundle={"manifest_records": [{"relative_path": "missing.jpg", "label": "cat"}]},
                )

            self.assertEqual(image_folder_dataset.call_count, 1)

    def test_metadata_bundle_dataset_overrides_folder_labels_and_splits(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"training dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            folder_a = dataset_dir / "folder_a"
            folder_b = dataset_dir / "folder_b"
            folder_a.mkdir()
            folder_b.mkdir()
            image_specs = [
                (folder_a / "a_train_1.jpg", (255, 0, 0)),
                (folder_a / "a_train_2.jpg", (250, 0, 0)),
                (folder_b / "b_train_1.jpg", (0, 255, 0)),
                (folder_b / "b_val.jpg", (0, 250, 0)),
                (folder_a / "a_test.jpg", (0, 0, 255)),
                (folder_b / "b_test.jpg", (0, 0, 250)),
            ]
            for path, color in image_specs:
                Image.new("RGB", (12, 12), color).save(path)

            bundle = {
                "classes": [
                    {"class_key": "cat_key", "class_name": "cat", "class_index": 0},
                    {"class_key": "dog_key", "class_name": "dog", "class_index": 1},
                ],
                "manifest_records": [
                    {"sample_key": "s1", "relative_path": "folder_a/a_train_1.jpg", "label_key": "dog_key", "split": "train"},
                    {"sample_key": "s2", "relative_path": "folder_a/a_train_2.jpg", "label_key": "dog_key", "split": "train"},
                    {"sample_key": "s3", "relative_path": "folder_b/b_train_1.jpg", "label_key": "cat_key", "split": "train"},
                    {"sample_key": "s4", "relative_path": "folder_b/b_val.jpg", "label_key": "cat_key", "split": "val"},
                    {"sample_key": "s5", "relative_path": "folder_a/a_test.jpg", "label_key": "dog_key", "split": "test"},
                    {"sample_key": "s6", "relative_path": "folder_b/b_test.jpg", "label_key": "cat_key", "split": "test"},
                ],
                "annotations": [
                    {"sample_key": "s1", "annotation_type": "bbox", "bbox": {"xmin": 1, "ymin": 2, "xmax": 8, "ymax": 9}}
                ],
            }

            train_loader, val_loader, test_loader, class_names, _class_weights, execution_metadata = self.modal_app._load_image_data(
                dataset_dir,
                batch_size=2,
                image_size=32,
                augmentation={},
                class_balancing="none",
                sampling_strategy="none",
                preprocessing={"resize_strategy": "squash", "normalization": "none"},
                metadata_bundle=bundle,
            )
            bbox_lookup = self.modal_app._bbox_lookup_from_metadata_bundle(bundle, dataset_dir)

            self.assertEqual(class_names, ["cat", "dog"])
            self.assertEqual(list(train_loader.dataset.indices), [0, 1, 2])
            self.assertEqual(list(val_loader.dataset.indices), [3])
            self.assertEqual(list(test_loader.dataset.indices), [4, 5])
            self.assertEqual(execution_metadata["metadata_bundle"]["status"], "applied")
            self.assertEqual(execution_metadata["metadata_bundle"]["split_strategy"], "metadata_official")
            self.assertEqual(bbox_lookup["a_train_1.jpg"], (1, 2, 8, 9))

    def test_load_image_data_preserves_split_folder_classification_layout(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"training dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for split in ("train", "valid", "test"):
                for class_name, color in (("cat", (255, 0, 0)), ("dog", (0, 0, 255))):
                    for index in range(2):
                        image_path = dataset_dir / split / class_name / f"{split}_{class_name}_{index}.jpg"
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (12, 12), color).save(image_path)

            train_loader, val_loader, test_loader, class_names, _class_weights, execution_metadata = self.modal_app._load_image_data(
                dataset_dir,
                batch_size=2,
                image_size=32,
                augmentation={},
                class_balancing="none",
                sampling_strategy="none",
                preprocessing={"resize_strategy": "squash", "normalization": "none"},
            )

            base_dataset = train_loader.dataset.dataset
            train_paths = _subset_relative_paths(dataset_dir, base_dataset, train_loader.dataset.indices)
            val_paths = _subset_relative_paths(dataset_dir, base_dataset, val_loader.dataset.indices)
            test_paths = _subset_relative_paths(dataset_dir, base_dataset, test_loader.dataset.indices)

            self.assertEqual(class_names, ["cat", "dog"])
            self.assertTrue(train_paths)
            self.assertTrue(val_paths)
            self.assertTrue(test_paths)
            self.assertTrue(all(path.startswith("train/") for path in train_paths), train_paths)
            self.assertTrue(all(path.startswith("valid/") for path in val_paths), val_paths)
            self.assertTrue(all(path.startswith("test/") for path in test_paths), test_paths)
            self.assertEqual(execution_metadata["split_folder"]["status"], "applied")
            self.assertEqual(execution_metadata["split_folder"]["split_strategy"], "metadata_official")
            self.assertEqual(execution_metadata["metadata_bundle"]["split_strategy"], "metadata_official")

    def test_metadata_bundle_page_merge_deduplicates_repeated_classes(self) -> None:
        first_page = {
            "classes": [
                {"class_key": "cat_key", "class_name": "cat", "class_index": 0},
                {"class_key": "dog_key", "class_name": "dog", "class_index": 1},
            ],
            "manifest_records": [
                {"relative_path": "cat/one.jpg", "label_key": "cat_key", "split": "train"},
            ],
        }
        second_page = {
            "classes": [
                {"class_key": "cat_key", "class_name": "cat", "class_index": 0},
                {"class_key": "dog_key", "class_name": "dog", "class_index": 1},
            ],
            "manifest_records": [
                {"relative_path": "dog/two.jpg", "label_key": "dog_key", "split": "test"},
            ],
        }

        merged = self.modal_app._merge_metadata_bundle_pages(None, first_page)
        merged = self.modal_app._merge_metadata_bundle_pages(merged, second_page)
        class_names, label_lookup = self.modal_app._metadata_class_mapping(merged, merged["manifest_records"])

        self.assertEqual(len(merged["classes"]), 2)
        self.assertEqual(class_names, ["cat", "dog"])
        self.assertEqual(label_lookup["cat_key"], 0)
        self.assertEqual(label_lookup["dog_key"], 1)

    def test_metadata_fetch_absence_counts_as_zero_records(self) -> None:
        class Response:
            status_code = 404

            def raise_for_status(self) -> None:
                raise AssertionError("404 should be treated as metadata unavailable")

        with patch("requests.get", return_value=Response()):
            bundle = self.modal_app._fetch_training_metadata_bundle(
                "https://orchestrator.test",
                "dataset_1",
                {},
            )

        self.assertIsNone(bundle)
        self.assertEqual(len(self.modal_app._metadata_manifest_records(bundle)), 0)

    def test_metadata_helpers_tolerate_absent_optional_sections(self) -> None:
        self.assertEqual(self.modal_app._metadata_manifest_records(None), [])

        execution_metadata = {
            "dataset_tier": None,
            "metadata_bundle": None,
            "dataloader": None,
        }
        self.assertEqual(self.modal_app._dict_or_empty(execution_metadata.get("dataset_tier")), {})
        self.assertEqual(self.modal_app._dict_or_empty(execution_metadata.get("metadata_bundle")), {})
        self.assertEqual(self.modal_app._dict_or_empty(execution_metadata.get("dataloader")), {})

    def test_metadata_bundle_resolves_paths_under_common_image_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            image_root = dataset_dir / "images"
            (image_root / "red").mkdir(parents=True)
            (image_root / "blue").mkdir(parents=True)
            Image.new("RGB", (12, 12), (255, 0, 0)).save(image_root / "red" / "one.jpg")
            Image.new("RGB", (12, 12), (0, 0, 255)).save(image_root / "blue" / "two.jpg")

            bundle = {
                "classes": [
                    {"class_key": "red_key", "class_name": "red", "class_index": 0},
                    {"class_key": "blue_key", "class_name": "blue", "class_index": 1},
                ],
                "manifest_records": [
                    {"sample_key": "s1", "relative_path": "red/one.jpg", "label_key": "red_key", "split": "train"},
                    {"sample_key": "s2", "relative_path": "blue/two.jpg", "label_key": "blue_key", "split": "test"},
                ],
                "annotations": [
                    {"sample_key": "s1", "bbox": {"xmin": 1, "ymin": 2, "xmax": 9, "ymax": 10}},
                ],
            }

            spec = self.modal_app._metadata_bundle_dataset_spec(dataset_dir, bundle)
            bbox_lookup = self.modal_app._bbox_lookup_from_metadata_bundle(bundle, dataset_dir)

            self.assertIsNotNone(spec)
            self.assertEqual([Path(path).name for path, _target in spec["samples"]], ["one.jpg", "two.jpg"])
            self.assertEqual(bbox_lookup["one.jpg"], (1, 2, 9, 10))

    def test_train_test_metadata_split_derives_validation_from_train(self) -> None:
        splits = ["train", "train", "train", "train", "train", "train", "test", "test"]
        targets = [0, 0, 0, 1, 1, 1, 0, 1]

        train_indices, val_indices, test_indices = self.modal_app._indices_from_metadata_splits(splits, targets)

        self.assertEqual(train_indices, [0, 1, 2, 4, 5])
        self.assertEqual(val_indices, [3])
        self.assertEqual(test_indices, [6, 7])

    def test_unusable_metadata_bundle_keeps_imagefolder_fallback(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"training dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for class_name, color in (("cat", (255, 0, 0)), ("dog", (0, 0, 255))):
                class_dir = dataset_dir / class_name
                class_dir.mkdir()
                for index in range(2):
                    Image.new("RGB", (12, 12), color).save(class_dir / f"{index}.jpg")

            _train_loader, _val_loader, _test_loader, class_names, _class_weights, execution_metadata = self.modal_app._load_image_data(
                dataset_dir,
                batch_size=2,
                image_size=32,
                augmentation={},
                class_balancing="none",
                sampling_strategy="none",
                preprocessing={"resize_strategy": "squash", "normalization": "none"},
                metadata_bundle={"manifest_records": [{"relative_path": "missing.jpg", "label": "cat", "split": "train"}]},
            )

            self.assertEqual(class_names, ["cat", "dog"])
            self.assertEqual(execution_metadata["metadata_bundle"]["status"], "not_usable")
            self.assertEqual(execution_metadata["metadata_bundle"]["split_strategy"], "deterministic_random")

    def test_label_quality_audit_is_report_only(self) -> None:
        audit = self.modal_app._label_quality_audit(
            {"mechanism": "label_noise_audit"},
            {
                "example_predictions": [
                    {
                        "path": "cat/one.jpg",
                        "true_class": "cat",
                        "predicted_class": "dog",
                        "confidence": 0.91,
                        "correct": False,
                    }
                ]
            },
            ["cat", "dog"],
        )

        self.assertEqual(audit["status"], "completed")
        self.assertTrue(audit["report_only"])
        self.assertEqual(len(audit["high_confidence_wrong"]), 1)


def _subset_relative_paths(dataset_dir: Path, dataset, indices) -> list[str]:
    return [
        Path(dataset.samples[index][0]).relative_to(dataset_dir).as_posix()
        for index in indices
    ]


if __name__ == "__main__":
    unittest.main()
