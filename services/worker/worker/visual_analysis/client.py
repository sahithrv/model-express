from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class VisualLLMConfig:
    enabled: bool
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.0

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
        timeout = _int_env("MODEL_EXPRESS_VISUAL_LLM_TIMEOUT_SECONDS", 60)
        temperature = _float_env("MODEL_EXPRESS_VISUAL_LLM_TEMPERATURE", 0.0)
        return cls(
            enabled=enabled,
            provider=provider,
            base_url=base_url,
            api_key=os.getenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "").strip(),
            model=os.getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "").strip(),
            timeout_seconds=timeout,
            temperature=temperature,
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
        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return _extract_message_content(response.json())

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

    def _user_content(self, user_prompt: str, images: list[ImageInput]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(image.as_openai_content_part() for image in images)
        return content


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


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def request_debug_json(system_prompt: str, user_prompt: str, images: list[ImageInput]) -> str:
    """Return a compact representation for tests without image bytes."""
    content = [{"type": "text", "text": user_prompt}]
    content.extend({"type": "image_url", "image_id": image.image_id, "detail": image.detail} for image in images)
    return json.dumps(
        {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]},
        sort_keys=True,
    )
