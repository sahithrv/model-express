from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from worker.diagnostics import log_event
from worker.visual_analysis.tools import (
    VISUAL_ANALYSIS_RESPONSE_TOOLS,
    build_visual_analysis_tool_context,
    execute_visual_analysis_tool,
    tool_answer_json,
)

DEFAULT_VISUAL_LLM_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class VisualLLMConfig:
    enabled: bool
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = DEFAULT_VISUAL_LLM_TIMEOUT_SECONDS
    temperature: float = 0.0
    api_style: str = "chat"
    reasoning_effort: str = ""
    max_tool_rounds: int = 3
    max_retries: int = 1
    retry_backoff_seconds: float = 0.75

    @classmethod
    def from_env(cls) -> "VisualLLMConfig":
        provider = os.getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "openai").strip().lower()
        base_url = os.getenv("MODEL_EXPRESS_VISUAL_LLM_BASE_URL", "").strip().rstrip("/")
        if not base_url and provider == "openai":
            base_url = "https://api.openai.com/v1"
        enabled_value = os.getenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "").strip().lower()
        enabled = enabled_value in {"1", "true", "yes", "on"} or bool(
            os.getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "").strip()
        )
        timeout = _int_env("MODEL_EXPRESS_VISUAL_LLM_TIMEOUT_SECONDS", DEFAULT_VISUAL_LLM_TIMEOUT_SECONDS)
        temperature = _float_env("MODEL_EXPRESS_VISUAL_LLM_TEMPERATURE", 0.0)
        api_style = _api_style_env("MODEL_EXPRESS_VISUAL_LLM_API_STYLE", "chat")
        max_tool_rounds = _bounded_int_env("MODEL_EXPRESS_VISUAL_LLM_MAX_TOOL_ROUNDS", 3, 0, 8)
        max_retries = _bounded_int_env("MODEL_EXPRESS_VISUAL_LLM_MAX_RETRIES", 1, 0, 5)
        retry_backoff = _bounded_float_env("MODEL_EXPRESS_VISUAL_LLM_RETRY_BACKOFF_SECONDS", 0.75, 0.0, 10.0)
        return cls(
            enabled=enabled,
            provider=provider,
            base_url=base_url,
            api_key=os.getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "").strip(),
            model=os.getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "").strip(),
            timeout_seconds=timeout,
            temperature=temperature,
            api_style=api_style,
            reasoning_effort=os.getenv("MODEL_EXPRESS_VISUAL_LLM_REASONING_EFFORT", "").strip().lower(),
            max_tool_rounds=max_tool_rounds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff,
        )


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    data: bytes
    mime_type: str = "image/jpeg"
    detail: str = "low"

    def as_openai_content_part(self) -> dict[str, Any]:
        encoded = base64.b64encode(self.data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{encoded}",
                "detail": self.detail,
            },
        }

    def as_openai_response_content_part(self) -> dict[str, Any]:
        encoded = base64.b64encode(self.data).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{self.mime_type};base64,{encoded}",
            "detail": self.detail,
        }


class VisualJSONGenerator(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[ImageInput],
    ) -> str:
        ...


class VisualLLMClient:
    """OpenAI-compatible multimodal JSON client scoped to visual-analysis settings."""

    def __init__(
        self,
        config: VisualLLMConfig | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or VisualLLMConfig.from_env()
        self.session = session or requests.Session()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[ImageInput],
    ) -> str:
        self._validate_config()
        if self._uses_responses_api():
            return self._generate_json_responses(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
            )
        return self._generate_json_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )

    def _generate_json_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[ImageInput],
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._user_content(user_prompt, images)},
        ]
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        response = self._post_json(
            url=f"{self.config.base_url}/chat/completions",
            body=body,
            label="visual chat completions request",
        )
        return _extract_message_content(response.json())

    def _generate_json_responses(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[ImageInput],
    ) -> str:
        tool_context = build_visual_analysis_tool_context(user_prompt)
        previous_response_id = ""
        input_items = self._responses_initial_input(user_prompt, images)
        for round_index in range(self.config.max_tool_rounds + 1):
            body = self._responses_body(
                system_prompt=system_prompt,
                input_items=input_items,
                previous_response_id=previous_response_id,
            )
            response = self._post_json(
                url=f"{self.config.base_url}/responses",
                body=body,
                label="visual Responses request",
                tool_count=len(VISUAL_ANALYSIS_RESPONSE_TOOLS),
            )
            payload = response.json()
            previous_response_id = _extract_response_id(payload) or previous_response_id
            tool_calls = _extract_response_function_calls(payload)
            if tool_calls:
                if round_index >= self.config.max_tool_rounds:
                    raise ValueError("visual Responses information-request loop exceeded max_tool_rounds")
                input_items = []
                for tool_call in tool_calls:
                    answer = execute_visual_analysis_tool(
                        name=tool_call["name"],
                        arguments=tool_call.get("arguments"),
                        context=tool_context,
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": tool_answer_json(answer),
                        }
                    )
                continue

            text = _extract_response_text(payload)
            if text:
                return text
            raise ValueError("visual Responses API response contained no final JSON output")
        raise ValueError("visual Responses information-request loop did not produce final JSON")

    def _validate_config(self) -> None:
        if not self.config.enabled:
            raise ValueError("visual LLM is disabled")
        if not self.config.base_url:
            raise ValueError("MODEL_EXPRESS_VISUAL_LLM_BASE_URL is required")
        if not self.config.model:
            raise ValueError("MODEL_EXPRESS_VISUAL_LLM_MODEL is required")
        if not self.config.api_key and self.config.provider != "local":
            raise ValueError("MODEL_EXPRESS_VISUAL_LLM_API_KEY is required")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_json(
        self,
        *,
        url: str,
        body: dict[str, Any],
        label: str,
        tool_count: int = 0,
    ) -> requests.Response:
        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                if _retryable_request_exception(exc) and attempt < attempts - 1:
                    self._log_retry(
                        attempt=attempt,
                        tool_count=tool_count,
                        error=str(exc),
                    )
                    self._sleep_before_retry(attempt)
                    continue
                log_event(
                    "error",
                    "visual_llm_request_error",
                    provider=self.config.provider,
                    model=self.config.model,
                    api_style=self.config.api_style,
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    error=str(exc),
                    tool_count=tool_count,
                )
                raise

            if _retryable_response(response) and attempt < attempts - 1:
                self._log_retry(
                    attempt=attempt,
                    tool_count=tool_count,
                    status_code=getattr(response, "status_code", ""),
                    reason=getattr(response, "reason", ""),
                )
                self._sleep_before_retry(attempt)
                continue

            _raise_for_status(
                response,
                label,
                provider=self.config.provider,
                model=self.config.model,
                api_style=self.config.api_style,
                tool_count=tool_count,
            )
            return response
        raise RuntimeError("visual LLM request retry loop exhausted without response")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = max(0.0, float(self.config.retry_backoff_seconds)) * (2 ** max(0, attempt))
        if delay > 0:
            time.sleep(delay)

    def _log_retry(
        self,
        *,
        attempt: int,
        tool_count: int,
        error: str = "",
        status_code: int | str = "",
        reason: str = "",
    ) -> None:
        log_event(
            "warn",
            "visual_llm_request_retry",
            provider=self.config.provider,
            model=self.config.model,
            api_style=self.config.api_style,
            attempt=attempt + 1,
            max_retries=self.config.max_retries,
            retry_backoff_seconds=self.config.retry_backoff_seconds,
            status_code=status_code,
            reason=reason,
            error=error,
            tool_count=tool_count,
        )

    def _uses_responses_api(self) -> bool:
        return self.config.provider == "openai" and self.config.api_style == "responses"

    def _user_content(self, user_prompt: str, images: list[ImageInput]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(image.as_openai_content_part() for image in images)
        return content

    def _responses_initial_input(self, user_prompt: str, images: list[ImageInput]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": self._responses_user_content(user_prompt, images),
            }
        ]

    def _responses_user_content(self, user_prompt: str, images: list[ImageInput]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        content.extend(image.as_openai_response_content_part() for image in images)
        return content

    def _responses_body(
        self,
        *,
        system_prompt: str,
        input_items: list[dict[str, Any]],
        previous_response_id: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": input_items,
            "tools": VISUAL_ANALYSIS_RESPONSE_TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "text": {"format": {"type": "json_object"}},
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        return body


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("visual LLM response had no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("visual LLM response choice had no message")
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError("visual LLM response content was empty")
    return text


def _raise_for_status(
    response: requests.Response,
    label: str,
    *,
    provider: str = "",
    model: str = "",
    api_style: str = "",
    tool_count: int = 0,
) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(response, "status_code", "")
        reason = getattr(response, "reason", "")
        body = str(getattr(response, "text", "") or "").strip()
        if len(body) > 1200:
            body = body[:1199].rstrip() + "..."
        detail = f"{status_code} {reason}".strip()
        if body:
            detail = f"{detail}: {body}" if detail else body
        log_event(
            "error",
            "visual_llm_http_error",
            provider=provider,
            model=model,
            api_style=api_style,
            status_code=status_code,
            reason=reason,
            body=body,
            tool_count=tool_count,
        )
        raise requests.HTTPError(f"{label} failed: {detail}", response=response) from exc


def _retryable_request_exception(exc: requests.RequestException) -> bool:
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def _retryable_response(response: requests.Response) -> bool:
    return int(getattr(response, "status_code", 0) or 0) in {408, 429, 500, 502, 503, 504}


def _extract_response_id(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    response_id = payload.get("id")
    return str(response_id) if response_id not in (None, "") else ""


def _extract_response_function_calls(payload: dict[str, Any]) -> list[dict[str, str]]:
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, list):
        return []
    calls: list[dict[str, str]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            name = str(item.get("name") or "")
            call_id = str(item.get("call_id") or item.get("id") or "")
            if name and call_id:
                calls.append(
                    {
                        "name": name,
                        "call_id": call_id,
                        "arguments": _stringify_arguments(item.get("arguments")),
                    }
                )
            continue
        content = item.get("content")
        if isinstance(content, list):
            calls.extend(_extract_nested_function_calls(content))
    return calls


def _extract_nested_function_calls(content: list[Any]) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "function_call":
            continue
        name = str(part.get("name") or "")
        call_id = str(part.get("call_id") or part.get("id") or "")
        if name and call_id:
            calls.append(
                {
                    "name": name,
                    "call_id": call_id,
                    "arguments": _stringify_arguments(part.get("arguments")),
                }
            )
    return calls


def _extract_response_text(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("visual Responses API payload was not an object")
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            chunks.extend(_message_text_chunks(item.get("content")))
        elif item.get("type") in {"output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def _message_text_chunks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return chunks


def _stringify_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments in (None, ""):
        return "{}"
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _int_env(name, default)
    return min(max(value, minimum), maximum)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _float_env(name, default)
    return min(max(value, minimum), maximum)


def _api_style_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().lower()
    return "responses" if value == "responses" else "chat"


def request_debug_json(system_prompt: str, user_prompt: str, images: list[ImageInput]) -> str:
    """Return a compact representation for tests without image bytes."""
    content = [{"type": "text", "text": user_prompt}]
    content.extend({"type": "image_url", "image_id": image.image_id, "detail": image.detail} for image in images)
    return json.dumps(
        {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]},
        sort_keys=True,
    )
