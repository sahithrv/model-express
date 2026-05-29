from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
