from __future__ import annotations

import json
import unittest

from worker.visual_analysis.schema import (
    SCHEMA_VERSION,
    VisualAnalysisValidationError,
    validate_visual_analysis_output,
)


def valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ds_1",
        "dataset_name": "flowers",
        "total_images": 10,
        "images_analyzed": 2,
        "trigger_reason": "initial_profile",
        "confidence": "medium",
        "coverage_report": {
            "selection_strategy": "deterministic_risk_and_representative_sampling",
            "selection_basis": ["class_representative"],
            "images_available": 10,
            "images_analyzed": 2,
            "classes_total": 2,
            "classes_covered": 2,
            "class_coverage_ratio": 1.0,
            "per_class_counts": {"daisy": 1, "rose": 1},
            "hard_example_count": 0,
            "edge_case_count": 0,
            "high_detail_image_count": 1,
            "limitations": [],
        },
        "visual_traits": [
            {
                "trait": "background_dominance",
                "level": "medium",
                "confidence": "medium",
                "evidence": ["Sampled subjects occupy a small center region."],
                "example_image_ids": ["img_1"],
                "affected_classes": ["daisy"],
            }
        ],
        "classes_to_watch": [
            {
                "class_name": "daisy",
                "reason": "Petal patterns overlap with sampled rose examples.",
                "related_classes": ["rose"],
                "evidence": ["Two sampled examples share light petals and centered blooms."],
                "example_image_ids": ["img_1", "img_2"],
                "confidence": "low",
            }
        ],
        "preprocessing_hypotheses": [
            {
                "id": "vh_001",
                "mechanism": "resolution_crop",
                "summary": "Compare preserve-aspect padding with current squash resize.",
                "evidence": ["Aspect ratios vary in the sampled images."],
                "suggested_preprocessing": {
                    "resize_strategy": "preserve_aspect_pad",
                    "normalization": "imagenet",
                    "crop_strategy": "none",
                    "bbox_mode": "ignore",
                },
                "suggested_image_sizes": [256],
                "expected_effect": "Reduce shape distortion.",
                "risk": "Slightly higher latency.",
                "confidence": "medium",
                "support_status": "needs_backend_validation",
            }
        ],
        "cautions": [
            {
                "operation": "vertical_flip",
                "reason": "Several examples appear orientation-sensitive.",
                "severity": "medium",
                "confidence": "medium",
                "affected_classes": ["daisy"],
                "example_image_ids": ["img_1"],
            }
        ],
        "limitations": ["Bounded sample only."],
    }


def realistic_llm_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ds_1",
        "dataset_name": "flowers",
        "total_images": 128,
        "images_analyzed": 4,
        "trigger_reason": "initial_profile",
        "confidence": "medium",
        "coverage_report": {
            "selection_strategy": "deterministic_risk_and_representative_sampling",
            "selection_basis": [
                "class_representative",
                "aspect_ratio_outlier",
                "bbox_object_scale_outlier",
            ],
            "images_available": 128,
            "images_analyzed": 4,
            "classes_total": 3,
            "classes_covered": 3,
            "class_coverage_ratio": 1.0,
            "per_class_counts": {"daisy": 2, "rose": 1, "tulip": 1},
            "hard_example_count": 1,
            "edge_case_count": 2,
            "high_detail_image_count": 2,
            "limitations": ["Bounded visual sample, not full dataset inspection."],
        },
        "visual_traits": [
            {
                "trait": "small_objects",
                "level": "medium",
                "confidence": "medium",
                "evidence": ["img_001 and img_004 show foreground objects occupying a small region."],
                "example_image_ids": ["img_001", "img_004"],
                "affected_classes": ["daisy", "tulip"],
            },
            {
                "trait": "background_dominance",
                "level": "high",
                "confidence": "medium",
                "evidence": ["Several sampled images contain broad background around the subject."],
                "example_image_ids": ["img_001", "img_003"],
            },
            {
                "trait": "fine_grained_similarity",
                "level": "medium",
                "confidence": "low",
                "evidence": ["Rose and tulip examples share similar color and petal texture cues."],
                "example_image_ids": ["img_002", "img_003"],
                "affected_classes": ["rose", "tulip"],
            },
            {
                "trait": "crop_bbox_useful",
                "level": "medium",
                "confidence": "medium",
                "evidence": ["Manifest bbox metadata indicates object-centered crops are plausible."],
                "example_image_ids": ["img_004"],
            },
        ],
        "classes_to_watch": [
            {
                "class_name": "rose",
                "reason": "Texture and color overlap with tulip examples.",
                "related_classes": ["tulip"],
                "evidence": ["img_002 and img_003 share warm petal colors and soft boundaries."],
                "example_image_ids": ["img_002", "img_003"],
                "confidence": "low",
            }
        ],
        "preprocessing_hypotheses": [
            {
                "id": "vh_001",
                "mechanism": "resolution_crop",
                "summary": "Compare preserve-aspect padding and moderate image size for variable framing.",
                "evidence": [
                    "Aspect ratio and foreground scale vary across img_001 through img_004.",
                    "Background dominance appears high in multiple samples.",
                ],
                "example_image_ids": ["img_001", "img_003", "img_004"],
                "suggested_preprocessing": {
                    "resize_strategy": "preserve_aspect_pad",
                    "normalization": "imagenet",
                    "crop_strategy": "center_crop",
                    "bbox_mode": "ignore",
                },
                "suggested_image_sizes": [256, 288],
                "expected_effect": "Reduce shape distortion and make foreground scale more consistent.",
                "risk": "May increase latency compared with the current 224 input.",
                "confidence": "medium",
                "support_status": "needs_backend_validation",
            },
            {
                "id": "vh_002",
                "mechanism": "augmentation_mixed_sample",
                "summary": "Try MixUp for visually similar flower classes.",
                "evidence": ["Rose and tulip samples have overlapping color and petal texture cues."],
                "example_image_ids": ["img_002", "img_003"],
                "suggested_augmentation_policy": "mixup",
                "suggested_augmentation_policy_config": {
                    "policy_type": "mixup",
                    "probability": 0.45,
                    "alpha": 0.3,
                },
                "expected_effect": "Improve calibration for visually similar classes.",
                "risk": "Can soften labels too much on a small dataset.",
                "confidence": "low",
                "support_status": "needs_backend_validation",
            },
            {
                "id": "vh_003",
                "mechanism": "bbox_crop_ablation",
                "summary": "Compare bbox-centered crop against full-image training when backend annotations exist.",
                "evidence": ["img_004 includes bbox metadata and a small foreground object."],
                "example_image_ids": ["img_004"],
                "suggested_preprocessing": {
                    "crop_strategy": "bbox_crop_ablation",
                    "bbox_mode": "crop_and_compare_full_image",
                    "normalization": "imagenet",
                },
                "expected_effect": "Check whether reducing background area improves foreground class signal.",
                "risk": "Only valid when backend-profiled bbox annotations are available.",
                "confidence": "medium",
                "support_status": "needs_backend_validation",
            },
            {
                "id": "vh_004",
                "mechanism": "augmentation_auto",
                "summary": "Use a light RandAugment search for brightness and texture variation.",
                "evidence": ["Lighting and texture vary across the sampled flowers."],
                "example_image_ids": ["img_001", "img_002", "img_003"],
                "suggested_augmentation_policy": "randaugment",
                "suggested_augmentation_policy_config": {
                    "policy_type": "randaugment",
                    "magnitude": 6,
                    "num_ops": 2,
                    "num_magnitude_bins": 15,
                    "probability": 0.75,
                },
                "expected_effect": "Improve robustness to bounded visual variation.",
                "risk": "Aggressive color changes may hurt fine-grained flower cues.",
                "confidence": "medium",
                "support_status": "needs_backend_validation",
            },
        ],
        "cautions": [
            {
                "operation": "vertical_flip",
                "reason": "Some flowers have orientation cues in the sampled images.",
                "severity": "medium",
                "confidence": "medium",
                "example_image_ids": ["img_001", "img_004"],
            },
            {
                "operation": "strong_color_jitter",
                "reason": "Color appears class-informative for rose and tulip samples.",
                "severity": "medium",
                "confidence": "low",
                "affected_classes": ["rose", "tulip"],
                "example_image_ids": ["img_002", "img_003"],
            },
        ],
        "limitations": [
            "Visual evidence comes from a bounded sample and should not be treated as labels.",
            "Backend validation must approve every runnable preprocessing or augmentation field.",
        ],
    }


def realistic_sample_manifest() -> list[dict]:
    return [
        {"image_id": "img_001", "class_name": "daisy", "width": 640, "height": 480},
        {"image_id": "img_002", "class_name": "rose", "width": 512, "height": 512},
        {"image_id": "img_003", "class_name": "tulip", "width": 480, "height": 640},
        {
            "image_id": "img_004",
            "class_name": "daisy",
            "width": 800,
            "height": 600,
            "has_bbox": True,
            "bbox_count": 1,
            "bbox_area_ratio": 0.18,
        },
    ]


class VisualAnalysisSchemaTests(unittest.TestCase):
    def test_validates_and_caps_visual_analysis_output(self) -> None:
        payload = valid_payload()
        payload["coverage_report"]["class_coverage_ratio"] = 0.12

        parsed = validate_visual_analysis_output(
            json.dumps(payload),
            sample_manifest=[{"image_id": "img_1"}, {"image_id": "img_2"}],
            dataset_id="ds_1",
            dataset_name="flowers",
            total_images=10,
            trigger_reason="initial_profile",
        )

        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        self.assertEqual(parsed["coverage_report"]["class_coverage_ratio"], 1.0)
        self.assertEqual(parsed["visual_traits"][0]["example_image_ids"], ["img_1"])
        self.assertEqual(
            parsed["preprocessing_hypotheses"][0]["suggested_preprocessing"]["resize_strategy"],
            "preserve_aspect_pad",
        )

    def test_realistic_llm_output_contract_survives_worker_schema_validation(self) -> None:
        payload = realistic_llm_payload()

        parsed = validate_visual_analysis_output(
            json.dumps(payload),
            sample_manifest=realistic_sample_manifest(),
            dataset_id="ds_1",
            dataset_name="flowers",
            total_images=128,
            trigger_reason="initial_profile",
            max_images_analyzed=4,
        )

        self.assertEqual(parsed["images_analyzed"], 4)
        self.assertEqual(len(parsed["preprocessing_hypotheses"]), 4)
        self.assertEqual(
            parsed["preprocessing_hypotheses"][1]["suggested_augmentation_policy_config"]["policy_type"],
            "mixup",
        )
        self.assertEqual(
            parsed["preprocessing_hypotheses"][2]["suggested_preprocessing"]["crop_strategy"],
            "bbox_crop_ablation",
        )
        self.assertEqual(parsed["cautions"][1]["operation"], "strong_color_jitter")

    def test_marks_unsupported_suggested_operations_non_runnable(self) -> None:
        payload = valid_payload()
        payload["preprocessing_hypotheses"][0]["suggested_preprocessing"] = {
            "resize_strategy": "magic_resize",
            "normalization": "imagenet",
        }
        payload["preprocessing_hypotheses"][0]["suggested_augmentation_policy"] = "make_it_pop"

        parsed = validate_visual_analysis_output(
            payload,
            sample_manifest=[{"image_id": "img_1"}, {"image_id": "img_2"}],
            dataset_id="ds_1",
        )

        hypothesis = parsed["preprocessing_hypotheses"][0]
        self.assertEqual(hypothesis["support_status"], "unsupported")
        self.assertEqual(hypothesis["suggested_preprocessing"], {"normalization": "imagenet"})
        self.assertNotIn("suggested_augmentation_policy", hypothesis)
        self.assertIn("magic_resize", hypothesis["unsupported_reason"])
        self.assertIn("make_it_pop", hypothesis["unsupported_reason"])

    def test_missing_hypothesis_id_is_backend_assigned_without_losing_evidence(self) -> None:
        payload = valid_payload()
        del payload["preprocessing_hypotheses"][0]["id"]

        parsed = validate_visual_analysis_output(
            payload,
            sample_manifest=[{"image_id": "img_1"}, {"image_id": "img_2"}],
            dataset_id="ds_1",
        )

        hypothesis = parsed["preprocessing_hypotheses"][0]
        self.assertEqual(hypothesis["id"], "vh_001")
        self.assertEqual(hypothesis["mechanism"], "resolution_crop")
        self.assertIn("Aspect ratios vary", hypothesis["evidence"][0])

    def test_required_confidence_fields_reject_null_values(self) -> None:
        payload = valid_payload()
        payload["visual_traits"][0]["level"] = None

        with self.assertRaises(VisualAnalysisValidationError):
            validate_visual_analysis_output(
                payload,
                sample_manifest=[{"image_id": "img_1"}, {"image_id": "img_2"}],
                dataset_id="ds_1",
            )

    def test_rejects_execution_authority_and_unknown_image_ids(self) -> None:
        payload = valid_payload()
        payload["proposed_experiments"] = [{"template": "train_experiment"}]

        with self.assertRaises(VisualAnalysisValidationError):
            validate_visual_analysis_output(payload, sample_manifest=[{"image_id": "img_1"}])

        payload = valid_payload()
        payload["visual_traits"][0]["example_image_ids"] = ["missing"]
        with self.assertRaises(VisualAnalysisValidationError):
            validate_visual_analysis_output(payload, sample_manifest=[{"image_id": "img_1"}])

    def test_rejects_file_reference_leakage(self) -> None:
        payload = valid_payload()
        payload["visual_traits"][0]["evidence"] = ["The image came from C:\\tmp\\cat.jpg."]

        with self.assertRaises(VisualAnalysisValidationError):
            validate_visual_analysis_output(payload, sample_manifest=[{"image_id": "img_1"}])


if __name__ == "__main__":
    unittest.main()
