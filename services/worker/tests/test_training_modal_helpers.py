from __future__ import annotations

import importlib
import unittest


def _import_modal_app():
    try:
        return importlib.import_module("worker.training.modal_app")
    except Exception as exc:  # pragma: no cover - depends on optional training deps
        raise unittest.SkipTest(f"worker training dependencies are unavailable: {exc}") from exc


class ModalTrainingHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modal_app = _import_modal_app()

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


if __name__ == "__main__":
    unittest.main()
