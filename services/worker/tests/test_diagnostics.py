from __future__ import annotations

import json

from worker.diagnostics import log_event


def test_log_event_writes_redacted_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_LOG_DIR", str(tmp_path))

    log_event(
        "error",
        "visual_llm_http_error",
        project_id="project_1",
        api_key="sk-secret",
        request_url="https://api.openai.com/v1/responses",
        body='{"error":{"message":"Invalid schema"}}',
        sample_base64="data:image/jpeg;base64,/9j/4AAQSkZJRg",
        nested={"authorization": "Bearer token"},
    )

    body = (tmp_path / "worker.jsonl").read_text(encoding="utf-8")
    assert "sk-secret" not in body
    assert "api.openai.com" not in body
    assert "/9j/4AAQSkZJRg" not in body

    record = json.loads(body)
    assert record["component"] == "worker"
    assert record["event"] == "visual_llm_http_error"
    assert record["project_id"] == "project_1"
    assert record["body"] == '{"error":{"message":"Invalid schema"}}'


def test_log_event_rotates_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_EXPRESS_LOG_MAX_BYTES", "1")
    monkeypatch.setenv("MODEL_EXPRESS_LOG_MAX_FILES", "2")

    log_event("info", "first", payload="x")
    log_event("info", "second", payload="y")

    assert (tmp_path / "worker.jsonl").exists()
    assert (tmp_path / "worker.jsonl.1").exists()
