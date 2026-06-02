from __future__ import annotations

import importlib
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

    def test_modal_orchestrator_post_uses_longer_report_timeout(self) -> None:
        calls = []

        class Response:
            def raise_for_status(self) -> None:
                return None

        def fake_post(url: str, *, json: dict, timeout: int):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return Response()

        with patch.dict("os.environ", {"MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS": "240"}):
            with patch("requests.post", fake_post):
                self.modal_app._post_json("http://orchestrator.test/jobs/job_1/complete", {"mlflow_run_id": "run_1"})

        self.assertEqual(calls[0]["timeout"], 240)

    def test_modal_dataset_timeouts_are_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_MODAL_MATERIALIZATION_TIMEOUT_SECONDS": "1800",
                "MODEL_EXPRESS_MODAL_PROFILE_TIMEOUT_SECONDS": "7200",
            },
        ):
            self.assertEqual(self.modal_app._modal_dataset_materialization_timeout_seconds(), 1800)
            self.assertEqual(self.modal_app._modal_dataset_profile_timeout_seconds(), 7200)

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

    def test_modal_storage_env_sets_torch_home_default(self) -> None:
        payload = {
            "s3_endpoint_url": "https://s3.test",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "aws_default_region": "us-east-1",
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

    def test_upload_manifest_artifacts_uploads_onnx_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.onnx"
            sidecar = root / "model.onnx.data"
            model.write_bytes(b"onnx bytes")
            sidecar.write_bytes(b"external tensor bytes")
            uploads = []

            def fake_upload(source: Path, destination: str) -> None:
                uploads.append((Path(source).name, destination))

            manifest = {
                "metadata": {},
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
                    }
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
                ],
            )
            self.assertEqual(artifact_uris, [{"format": "onnx", "uri": "s3://bucket/exports/job_1/model.onnx"}])
            self.assertEqual(public_manifest["artifacts"][0]["external_data"][0]["uri"], "s3://bucket/exports/job_1/model.onnx.data")

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


if __name__ == "__main__":
    unittest.main()
