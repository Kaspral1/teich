"""Protocol and privacy tests for simulated harness-context capture."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from teich.harness_capture import CAPTURE_MARKER, HarnessCaptureServer


def _post(url: str, payload: dict[str, object], *, token: str) -> tuple[int, str]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def test_capture_openai_responses_sanitizes_user_messages_and_auth() -> None:
    payload = {
        "model": "gpt-test",
        "instructions": "Base harness instructions",
        "input": [
            {"role": "developer", "content": "Configured developer instructions"},
            {"role": "user", "content": "PRIVATE USER PROMPT"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ],
        "reasoning": {"effort": "high", "summary": "detailed"},
        "metadata": {"private": "must not be retained"},
        "stream": True,
    }
    with HarnessCaptureServer(harness="codex", harness_version="codex 1.0") as server:
        status, body = _post(f"{server.base_url}/responses", payload, token=server.secret)
        capture = server.wait(1)

    assert status == 200
    assert "response.completed" in body
    assert CAPTURE_MARKER in body
    assert capture is not None
    serialized = json.dumps(capture.to_dict())
    assert "Base harness instructions" in capture.system
    assert "Configured developer instructions" in capture.system
    assert "PRIVATE USER PROMPT" not in serialized
    assert "must not be retained" not in serialized
    assert server.secret not in serialized
    assert capture.tools[0]["function"]["name"] == "shell"
    assert capture.request_options == {"reasoning": {"effort": "high", "summary": "detailed"}}


def test_capture_anthropic_messages_normalizes_system_blocks_and_tools() -> None:
    payload = {
        "model": "claude-test",
        "system": [
            {"type": "text", "text": "Claude base prompt"},
            {"type": "text", "text": "MCP instructions"},
        ],
        "messages": [{"role": "user", "content": "PRIVATE PROMPT"}],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "stream": False,
    }
    with HarnessCaptureServer(harness="claude-code") as server:
        status, body = _post(f"{server.base_url}/messages?beta=true", payload, token=server.secret)
        capture = server.wait(1)

    assert status == 200
    assert json.loads(body)["content"][0]["text"] == CAPTURE_MARKER
    assert capture is not None
    assert capture.system == "Claude base prompt\n\nMCP instructions"
    assert capture.tools[0]["function"]["parameters"]["required"] == ["file_path"]
    assert "PRIVATE PROMPT" not in json.dumps(capture.to_dict())
    assert capture.request_options == {"thinking": {"type": "enabled", "budget_tokens": 4096}}


def test_capture_rejects_missing_secret() -> None:
    with HarnessCaptureServer(harness="codex") as server:
        with pytest.raises(HTTPError) as error:
            _post(
                f"{server.base_url}/responses",
                {"model": "gpt-test", "instructions": "system"},
                token="wrong-secret",
            )
        assert error.value.code == 401
        assert server.wait(0.05) is None


def test_context_hash_excludes_user_prompt_and_capture_time() -> None:
    def capture_hash(user_prompt: str) -> str:
        with HarnessCaptureServer(harness="codex", harness_version="1") as server:
            _post(
                f"{server.base_url}/responses",
                {
                    "model": "gpt-test",
                    "instructions": "same system",
                    "input": [{"role": "user", "content": user_prompt}],
                    "tools": [],
                },
                token=server.secret,
            )
            capture = server.wait(1)
        assert capture is not None
        return capture.context_hash

    assert capture_hash("first private prompt") == capture_hash("second private prompt")
