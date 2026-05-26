from __future__ import annotations

import unittest

from worker.training.preprocessing_registry import (
    all_strategy_names,
    bbox_compare_requested,
    bbox_crop_requested,
    bbox_crop_required,
    build_image_transform,
    normalize_preprocessing_config,
    validate_augmentation_strategy,
)


class PreprocessingRegistryTests(unittest.TestCase):
    def test_registry_lists_supported_worker_strategies(self) -> None:
        self.assertEqual(
            all_strategy_names(),
            {
                "resize": {
                    "squash",
                    "preserve_aspect_pad",
                    "center_crop",
                    "random_resized_crop",
                    "bbox_crop_if_available",
                },
                "crop": {
                    "none",
                    "center_crop",
                    "random_resized_crop",
                    "bbox_crop_if_available",
                    "bbox_crop_ablation",
                },
                "normalization": {"imagenet", "dataset", "none"},
                "augmentation": {
                    "none",
                    "custom",
                    "light",
                    "moderate",
                    "strong",
                    "basic",
                    "randaugment",
                    "trivialaugment",
                    "autoaugment",
                    "mixup",
                    "cutmix",
                },
                "bbox_mode": {
                    "ignore",
                    "crop_if_available",
                    "crop_and_compare_full_image",
                    "use_boxes_as_metadata",
                },
            },
        )

    def test_preprocessing_defaults_and_unknowns_fail_clearly(self) -> None:
        self.assertEqual(
            normalize_preprocessing_config({}),
            {
                "resize_strategy": "squash",
                "crop_strategy": "none",
                "normalization": "imagenet",
                "bbox_mode": "ignore",
            },
        )

        with self.assertRaisesRegex(ValueError, "Unsupported resize_strategy 'mystery'"):
            normalize_preprocessing_config({"resize_strategy": "mystery"})
        with self.assertRaisesRegex(ValueError, "Unsupported crop_strategy 'mystery'"):
            normalize_preprocessing_config({"crop_strategy": "mystery"})
        with self.assertRaisesRegex(ValueError, "Unsupported normalization 'mystery'"):
            normalize_preprocessing_config({"normalization": "mystery"})
        with self.assertRaisesRegex(ValueError, "Unsupported bbox_mode 'mystery'"):
            normalize_preprocessing_config({"bbox_mode": "mystery"})

    def test_bbox_helpers_capture_required_and_ablation_modes(self) -> None:
        self.assertFalse(bbox_crop_requested({}))
        self.assertFalse(bbox_crop_required({}))
        self.assertFalse(bbox_compare_requested({}))

        optional_crop = {"crop_strategy": "bbox_crop_if_available"}
        self.assertTrue(bbox_crop_requested(optional_crop))
        self.assertTrue(bbox_crop_required(optional_crop))
        self.assertFalse(bbox_compare_requested(optional_crop))

        ablation = {"bbox_mode": "crop_and_compare_full_image"}
        self.assertTrue(bbox_crop_requested(ablation))
        self.assertTrue(bbox_crop_required(ablation))
        self.assertTrue(bbox_compare_requested(ablation))

    def test_augmentation_registry_accepts_every_supported_policy(self) -> None:
        for policy_type in all_strategy_names()["augmentation"]:
            with self.subTest(policy_type=policy_type):
                self.assertEqual(
                    validate_augmentation_strategy({"policy_type": policy_type}),
                    policy_type,
                )

        with self.assertRaisesRegex(ValueError, "Unsupported augmentation policy_type 'mystery'"):
            validate_augmentation_strategy({"policy_type": "mystery"})

    def test_transform_builder_uses_registry_validation(self) -> None:
        try:
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional training deps
            raise unittest.SkipTest(f"torchvision is unavailable: {exc}") from exc

        transform = build_image_transform(
            64,
            {},
            {
                "resize_strategy": "preserve_aspect_pad",
                "crop_strategy": "none",
                "normalization": "none",
            },
            training=False,
        )
        self.assertEqual(transform.transforms[0].__class__.__name__, "Lambda")

        with self.assertRaisesRegex(ValueError, "Unsupported resize_strategy 'mystery'"):
            build_image_transform(64, {}, {"resize_strategy": "mystery"}, training=False)


if __name__ == "__main__":
    unittest.main()
