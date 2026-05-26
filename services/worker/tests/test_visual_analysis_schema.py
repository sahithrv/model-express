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
