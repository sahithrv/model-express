from __future__ import annotations

import json
import unittest

import requests

from worker.visual_analysis.agent import VisualAnalysisRequest, build_visual_analysis_messages
from worker.visual_analysis.client import ImageInput, VisualLLMClient, VisualLLMConfig
from worker.visual_analysis.schema import SCHEMA_VERSION, validate_visual_analysis_output
from worker.visual_analysis.tools import (
    VISUAL_ANALYSIS_RESPONSE_TOOLS,
    build_visual_analysis_tool_context,
    execute_visual_analysis_tool,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict, *, status_code: int = 200, reason: str = "OK", text: str = "") -> None:
        self.payload = payload
        self.status_code = status_code
        self.reason = reason
        self.text = text or json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.reason}", response=self)
        return None

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, payloads: list[dict | FakeHTTPResponse | Exception]) -> None:
        self.payloads = payloads
        self.requests: list[dict] = []

    def post(self, url: str, *, json: dict, headers: dict, timeout: int) -> FakeHTTPResponse:
        self.requests.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if not self.payloads:
            raise AssertionError("unexpected HTTP request")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, FakeHTTPResponse):
            return payload
        return FakeHTTPResponse(payload)


def visual_payload(*, images_analyzed: int = 1, total_images: int = 4) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ds_1",
        "dataset_name": "flowers",
        "total_images": total_images,
        "images_analyzed": images_analyzed,
        "trigger_reason": "initial_profile",
        "confidence": "low",
        "coverage_report": {
            "images_available": total_images,
            "images_analyzed": images_analyzed,
            "classes_total": 2,
            "classes_covered": 1,
            "class_coverage_ratio": 0.5,
        },
        "visual_traits": [],
        "classes_to_watch": [],
        "preprocessing_hypotheses": [],
        "cautions": [],
        "limitations": ["Bounded sample only."],
    }


def visual_request() -> VisualAnalysisRequest:
    return VisualAnalysisRequest(
        dataset_metadata={
            "dataset_id": "ds_1",
            "dataset_name": "flowers",
            "total_images": 4,
            "storage_uri": "s3://secret-bucket/flowers.zip",
            "profile": {
                "class_distribution": {"daisy": 2, "rose": 2},
                "dataset_path": "C:\\tmp\\flowers",
                "raw_prompt": "do not leak",
            },
        },
        sample_manifest=[
            {
                "image_id": "img_1",
                "class_name": "daisy",
                "width": 64,
                "height": 64,
                "local_path": "C:\\tmp\\flowers\\daisy\\one.jpg",
                "data_base64": "not-for-tools",
            }
        ],
        images=[ImageInput(image_id="img_1", data=b"actual-image-bytes", detail="high")],
        total_images=4,
    )


class VisualAnalysisClientTests(unittest.TestCase):
    def test_openai_compatible_responses_style_stays_on_chat_completions(self) -> None:
        session = FakeSession(
            [{"choices": [{"message": {"content": json.dumps(visual_payload())}}]}]
        )
        client = VisualLLMClient(
            VisualLLMConfig(
                enabled=True,
                provider="openai_compatible",
                base_url="http://visual-llm.test/v1",
                api_key="key",
                model="visual-model",
                api_style="responses",
            ),
            session=session,
        )

        raw = client.generate_json(system_prompt="system", user_prompt="{}", images=[])

        self.assertEqual(json.loads(raw)["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(session.requests[0]["url"], "http://visual-llm.test/v1/chat/completions")
        self.assertIn("messages", session.requests[0]["json"])
        self.assertNotIn("tools", session.requests[0]["json"])

    def test_responses_mode_runs_bounded_visual_tool_loop(self) -> None:
        final_payload = visual_payload()
        session = FakeSession(
            [
                {
                    "id": "resp_1",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_sample_manifest",
                            "arguments": "{\"limit\":null}",
                        }
                    ],
                },
                {
                    "id": "resp_2",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps(final_payload)}
                            ],
                        }
                    ],
                },
            ]
        )
        request = visual_request()
        system_prompt, user_prompt = build_visual_analysis_messages(request)
        client = VisualLLMClient(
            VisualLLMConfig(
                enabled=True,
                provider="openai",
                base_url="http://api.openai.test/v1",
                api_key="key",
                model="visual-model",
                api_style="responses",
                reasoning_effort="low",
                max_tool_rounds=2,
            ),
            session=session,
        )

        raw = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=request.images,
        )

        parsed = validate_visual_analysis_output(
            raw,
            sample_manifest=request.sample_manifest,
            dataset_id="ds_1",
            dataset_name="flowers",
            total_images=4,
            trigger_reason="initial_profile",
            max_images_analyzed=1,
        )
        self.assertEqual(parsed["dataset_id"], "ds_1")
        self.assertEqual(len(session.requests), 2)

        first_body = session.requests[0]["json"]
        self.assertEqual(session.requests[0]["url"], "http://api.openai.test/v1/responses")
        self.assertEqual(first_body["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", first_body)
        self.assertEqual(
            {tool["name"] for tool in first_body["tools"]},
            {
                "get_dataset_metadata_summary",
                "get_sample_manifest",
                "get_allowed_operations",
                "validate_visual_analysis_draft",
            },
        )
        first_content = first_body["input"][0]["content"]
        self.assertTrue(any(part.get("type") == "input_image" for part in first_content))
        self.assertTrue(
            any("data:image/jpeg;base64," in part.get("image_url", "") for part in first_content)
        )

        second_body = session.requests[1]["json"]
        self.assertEqual(second_body["previous_response_id"], "resp_1")
        self.assertNotIn("input_image", json.dumps(second_body["input"]))
        tool_output = json.loads(second_body["input"][0]["output"])
        self.assertTrue(tool_output["accepted"])
        self.assertEqual(tool_output["tool"], "get_sample_manifest")
        serialized_tool_output = json.dumps(tool_output, sort_keys=True)
        self.assertIn("img_1", serialized_tool_output)
        self.assertNotIn("C:\\tmp", serialized_tool_output)
        self.assertNotIn("s3://", serialized_tool_output)
        self.assertNotIn("data:image", serialized_tool_output)
        self.assertNotIn("base64", serialized_tool_output.lower())
        self.assertNotIn("actual-image-bytes", serialized_tool_output)
        self.assertNotIn("local_path", serialized_tool_output)
        self.assertNotIn("raw_prompt", serialized_tool_output)

    def test_validate_draft_tool_returns_safe_status_without_echoing_draft(self) -> None:
        _, user_prompt = build_visual_analysis_messages(visual_request())
        context = build_visual_analysis_tool_context(user_prompt)
        draft = visual_payload()
        draft["image_bytes"] = "data:image/jpeg;base64,/9j/4AAQSkZJRg"
        draft["proposed_experiments"] = [{"template": "train_experiment"}]

        answer = execute_visual_analysis_tool(
            name="validate_visual_analysis_draft",
            arguments={"draft": draft},
            context=context,
        )

        self.assertTrue(answer["accepted"])
        self.assertFalse(answer["validation"]["valid"])
        self.assertTrue(answer["validation"]["safety_violation"])
        self.assertFalse(answer["validation"]["repairable_schema_error"])
        serialized = json.dumps(answer, sort_keys=True)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("base64", serialized.lower())
        self.assertNotIn("image_bytes", serialized)
        self.assertNotIn("proposed_experiments", serialized)
        self.assertNotIn("train_experiment", serialized)

    def test_validate_draft_tool_schema_uses_stringified_json(self) -> None:
        tool = next(tool for tool in VISUAL_ANALYSIS_RESPONSE_TOOLS if tool["name"] == "validate_visual_analysis_draft")
        parameters = tool["parameters"]
        self.assertTrue(tool["strict"])
        self.assertEqual(parameters["type"], "object")
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["required"], ["draft_json"])
        self.assertEqual(parameters["properties"]["draft_json"]["type"], "string")
        self.assertNotIn("draft", parameters["properties"])

    def test_visual_tool_schemas_are_strict_responses_compatible(self) -> None:
        for tool in VISUAL_ANALYSIS_RESPONSE_TOOLS:
            with self.subTest(tool=tool["name"]):
                self.assertTrue(tool["strict"])
                parameters = tool["parameters"]
                self.assertEqual(parameters["type"], "object")
                self.assertFalse(parameters["additionalProperties"])
                self.assertEqual(
                    set(parameters["required"]),
                    set(parameters["properties"].keys()),
                )

        manifest_tool = next(tool for tool in VISUAL_ANALYSIS_RESPONSE_TOOLS if tool["name"] == "get_sample_manifest")
        manifest_params = manifest_tool["parameters"]
        self.assertEqual(manifest_params["required"], ["limit"])
        self.assertEqual(manifest_params["properties"]["limit"]["type"], ["integer", "null"])
        self.assertNotIn("minimum", manifest_params["properties"]["limit"])
        self.assertNotIn("maximum", manifest_params["properties"]["limit"])

    def test_responses_http_errors_include_api_body(self) -> None:
        error_body = '{"error":{"message":"Invalid schema for function get_sample_manifest"}}'
        session = FakeSession(
            [
                FakeHTTPResponse(
                    {"error": {"message": "Invalid schema for function get_sample_manifest"}},
                    status_code=400,
                    reason="Bad Request",
                    text=error_body,
                )
            ]
        )
        client = VisualLLMClient(
            VisualLLMConfig(
                enabled=True,
                provider="openai",
                base_url="http://api.openai.test/v1",
                api_key="key",
                model="visual-model",
                api_style="responses",
            ),
            session=session,
        )

        with self.assertRaisesRegex(requests.HTTPError, "Invalid schema for function get_sample_manifest"):
            client.generate_json(system_prompt="Return JSON.", user_prompt="{}", images=[])

    def test_chat_request_retries_transient_timeout_before_success(self) -> None:
        session = FakeSession(
            [
                requests.Timeout("read timed out"),
                {"choices": [{"message": {"content": json.dumps(visual_payload())}}]},
            ]
        )
        client = VisualLLMClient(
            VisualLLMConfig(
                enabled=True,
                provider="openai_compatible",
                base_url="http://visual-llm.test/v1",
                api_key="key",
                model="visual-model",
                max_retries=1,
                retry_backoff_seconds=0,
            ),
            session=session,
        )

        raw = client.generate_json(system_prompt="system", user_prompt="{}", images=[])

        self.assertEqual(json.loads(raw)["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(session.requests), 2)

    def test_information_tools_scrub_metadata_and_allowed_operations(self) -> None:
        request = VisualAnalysisRequest(
            dataset_metadata=visual_request().dataset_metadata,
            sample_manifest=visual_request().sample_manifest,
            allowed_operations={
                "resize_strategy": ["squash"],
                "commands": ["rm -rf /tmp/dataset"],
                "storage_uri": "s3://secret-bucket/ops.json",
            },
            total_images=4,
        )
        _, user_prompt = build_visual_analysis_messages(request)
        context = build_visual_analysis_tool_context(user_prompt)

        metadata_answer = execute_visual_analysis_tool(
            name="get_dataset_metadata_summary",
            arguments={},
            context=context,
        )
        allowed_answer = execute_visual_analysis_tool(
            name="get_allowed_operations",
            arguments={},
            context=context,
        )

        serialized = json.dumps([metadata_answer, allowed_answer], sort_keys=True)
        self.assertIn("resize_strategy", serialized)
        self.assertNotIn("C:\\tmp", serialized)
        self.assertNotIn("s3://", serialized)
        self.assertNotIn("raw_prompt", serialized)
        self.assertNotIn("dataset_path", serialized)
        self.assertNotIn("commands", serialized)
        self.assertNotIn("rm -rf", serialized)
        self.assertNotIn("storage_uri", serialized)

    def test_unsupported_tool_call_is_rejected_without_echoing_authority(self) -> None:
        _, user_prompt = build_visual_analysis_messages(visual_request())
        context = build_visual_analysis_tool_context(user_prompt)

        answer = execute_visual_analysis_tool(
            name="create_job",
            arguments={},
            context=context,
        )

        self.assertFalse(answer["accepted"])
        serialized = json.dumps(answer, sort_keys=True)
        self.assertNotIn("create_job", serialized)
        self.assertNotIn("commands", serialized)
        self.assertIn("unsupported visual analysis information request", serialized)


if __name__ == "__main__":
    unittest.main()
