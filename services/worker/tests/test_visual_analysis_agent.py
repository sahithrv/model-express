from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from worker.visual_analysis.agent import (
    VisualAnalysisAgent,
    VisualAnalysisRequest,
    build_visual_analysis_messages,
    visual_analysis_request_fingerprint,
)
from worker.visual_analysis.client import ImageInput, VisualLLMConfig, request_debug_json
from worker.visual_analysis.schema import SCHEMA_VERSION


class FakeVisualClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, images: list[ImageInput]) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "images": images})
        return json.dumps(self.response)


def llm_response() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ds_1",
        "dataset_name": "flowers",
        "total_images": 4,
        "images_analyzed": 1,
        "trigger_reason": "initial_profile",
        "confidence": "low",
        "coverage_report": {
            "images_available": 4,
            "images_analyzed": 1,
            "classes_total": 2,
            "classes_covered": 1,
            "class_coverage_ratio": 0.5,
        },
        "visual_traits": [],
        "classes_to_watch": [],
        "preprocessing_hypotheses": [],
        "cautions": [],
        "limitations": ["Only one sample was inspected."],
    }


class VisualAnalysisAgentTests(unittest.TestCase):
    def test_prompt_strips_paths_and_declares_evidence_only_contract(self) -> None:
        request = VisualAnalysisRequest(
            dataset_metadata={
                "dataset_id": "ds_1",
                "dataset_name": "flowers",
                "storage_uri": "s3://secret-bucket/data.zip",
                "profile": {
                    "class_names": ["daisy"],
                    "dataset_path": "C:\\tmp\\dataset",
                    "artifacts": [{"artifact_type": "class_folder", "path": "C:\\tmp\\dataset\\daisy"}],
                },
            },
            sample_manifest=[
                {
                    "image_id": "img_1",
                    "class_name": "daisy",
                    "width": 80,
                    "height": 60,
                    "local_path": "C:\\tmp\\dataset\\daisy\\1.jpg",
                    "selection_basis": ["class_representative"],
                }
            ],
        )

        system_prompt, user_prompt = build_visual_analysis_messages(request)
        combined = system_prompt + user_prompt

        self.assertIn("visual evidence and preprocessing hypotheses only", system_prompt)
        self.assertIn("Never output proposed_experiments", user_prompt)
        self.assertNotIn("s3://", combined)
        self.assertNotIn("C:\\tmp", combined)
        prompt_payload = json.loads(user_prompt)
        self.assertEqual(prompt_payload["sample_manifest"][0]["image_id"], "img_1")
        self.assertNotIn("local_path", prompt_payload["sample_manifest"][0])
        self.assertTrue(prompt_payload["budget"]["evidence_only"])
        self.assertFalse(prompt_payload["budget"]["raw_images_for_planner"])

    def test_agent_calls_visual_client_and_validates_response(self) -> None:
        fake_client = FakeVisualClient(llm_response())
        agent = VisualAnalysisAgent(fake_client)
        request = VisualAnalysisRequest(
            dataset_metadata={"dataset_id": "ds_1", "dataset_name": "flowers"},
            sample_manifest=[{"image_id": "img_1", "class_name": "daisy"}],
            images=[ImageInput(image_id="img_1", data=b"jpeg", detail="high")],
            total_images=4,
        )

        parsed = agent.analyze(request)

        self.assertEqual(parsed["dataset_id"], "ds_1")
        self.assertEqual(parsed["coverage_report"]["high_detail_image_count"], 0)
        self.assertEqual(len(fake_client.calls), 1)
        self.assertEqual(fake_client.calls[0]["images"][0].image_id, "img_1")

    def test_visual_llm_config_uses_separate_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_EXPRESS_LLM_MODEL": "planner-model",
                "MODEL_EXPRESS_VISUAL_LLM_ENABLED": "true",
                "MODEL_EXPRESS_VISUAL_LLM_PROVIDER": "openai_compatible",
                "MODEL_EXPRESS_VISUAL_LLM_BASE_URL": "http://visual-llm.test/v1/",
                "MODEL_EXPRESS_VISUAL_LLM_API_KEY": "visual-key",
                "MODEL_EXPRESS_VISUAL_LLM_MODEL": "visual-model",
                "MODEL_EXPRESS_VISUAL_LLM_TIMEOUT_SECONDS": "7",
            },
            clear=True,
        ):
            config = VisualLLMConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.base_url, "http://visual-llm.test/v1")
        self.assertEqual(config.api_key, "visual-key")
        self.assertEqual(config.model, "visual-model")
        self.assertEqual(config.timeout_seconds, 7)

    def test_debug_request_and_fingerprint_omit_image_bytes(self) -> None:
        request = VisualAnalysisRequest(
            dataset_metadata={"dataset_id": "ds_1", "dataset_name": "flowers"},
            sample_manifest=[{"image_id": "img_1", "class_name": "daisy"}],
            images=[ImageInput(image_id="img_1", data=b"actual-bytes", detail="low")],
        )
        system_prompt, user_prompt = build_visual_analysis_messages(request)
        debug_json = request_debug_json(system_prompt, user_prompt, request.images)
        fingerprint = visual_analysis_request_fingerprint(request)

        self.assertIn("img_1", debug_json)
        self.assertIn("visual_dataset_analysis_prompt_v1", fingerprint)
        self.assertNotIn("actual-bytes", debug_json)
        self.assertNotIn("actual-bytes", fingerprint)


if __name__ == "__main__":
    unittest.main()
