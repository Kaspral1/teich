"""Capture client-visible agent harness context with a local fake provider.

The recorder implements the two wire protocols used by Teich's Codex and
Claude Code runners.  It deliberately retains only instructions, tool schemas,
and a small allow-list of generation options; user messages and authentication
headers are never stored.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
from typing import Any


TEICH_HARNESS_CONTEXT_EVENT_TYPE = "teich_harness_context"
CAPTURE_MARKER = "TEICH_HARNESS_CONTEXT_CAPTURE_COMPLETE"
MAX_CAPTURE_REQUEST_BYTES = 64 * 1024 * 1024
_SAFE_OPTION_KEYS = (
    "max_output_tokens",
    "max_tokens",
    "parallel_tool_calls",
    "reasoning",
    "service_tier",
    "temperature",
    "thinking",
    "tool_choice",
    "top_p",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _text_from_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict):
            raw_text = block.get("text")
            text = raw_text if isinstance(raw_text, str) else ""
        else:
            text = ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _instruction_blocks(wire_api: str, request: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    if wire_api == "openai-responses":
        instructions = _text_from_content(request.get("instructions"))
        if instructions:
            blocks.append({"role": "developer", "content": instructions})
        input_items = request.get("input")
        if isinstance(input_items, list):
            for item in input_items:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                if role not in {"system", "developer"}:
                    continue
                text = _text_from_content(item.get("content"))
                if text:
                    blocks.append({"role": str(role), "content": text})
    else:
        system = _text_from_content(request.get("system"))
        if system:
            blocks.append({"role": "system", "content": system})
    return blocks


def _system_from_blocks(blocks: list[dict[str, str]]) -> str:
    return "\n\n".join(
        block["content"].strip()
        for block in blocks
        if isinstance(block.get("content"), str) and block["content"].strip()
    )


def _safe_request_options(request: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key in _SAFE_OPTION_KEYS:
        value = request.get(key)
        if value is not None:
            options[key] = deepcopy(value)
    return options


def _normalize_tools(value: object) -> list[dict[str, Any]]:
    """Normalize Anthropic and Responses function tools to OpenAI chat shape.

    Non-function built-ins are preserved verbatim because their wire-level
    declaration is still useful provenance and must not be guessed into a
    function schema.
    """
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for raw_tool in value:
        if not isinstance(raw_tool, dict):
            continue
        tool = deepcopy(raw_tool)
        nested = tool.get("function")
        if tool.get("type") == "function" and isinstance(nested, dict):
            tools.append(tool)
            continue
        name = tool.get("name")
        if isinstance(name, str) and name.strip():
            function: dict[str, Any] = {"name": name.strip()}
            description = tool.get("description")
            if isinstance(description, str):
                function["description"] = description
            parameters = tool.get("input_schema", tool.get("parameters"))
            function["parameters"] = (
                deepcopy(parameters)
                if isinstance(parameters, dict)
                else {"type": "object", "properties": {}}
            )
            tools.append({"type": "function", "function": function})
            continue
        tools.append(tool)
    return tools


@dataclass(frozen=True, slots=True)
class HarnessContextCapture:
    """Sanitized, deterministic representation of one simulated request."""

    harness: str
    harness_version: str | None
    wire_api: str
    model: str | None
    instruction_blocks: list[dict[str, str]]
    system: str
    tools: list[dict[str, Any]]
    request_options: dict[str, Any]
    captured_at: str
    context_hash: str

    @classmethod
    def from_request(
        cls,
        *,
        harness: str,
        harness_version: str | None,
        wire_api: str,
        request: dict[str, Any],
    ) -> HarnessContextCapture:
        blocks = _instruction_blocks(wire_api, request)
        model_value = request.get("model")
        model = model_value.strip() if isinstance(model_value, str) and model_value.strip() else None
        core: dict[str, Any] = {
            "source": "simulated_request_capture",
            "harness": harness,
            "harness_version": harness_version,
            "wire_api": wire_api,
            "model": model,
            "instruction_blocks": blocks,
            "system": _system_from_blocks(blocks),
            "tools": _normalize_tools(request.get("tools")),
            "request_options": _safe_request_options(request),
        }
        context_hash = hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()
        return cls(
            harness=harness,
            harness_version=harness_version,
            wire_api=wire_api,
            model=model,
            instruction_blocks=blocks,
            system=core["system"],
            tools=core["tools"],
            request_options=core["request_options"],
            captured_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            context_hash=context_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "simulated_request_capture",
            "harness": self.harness,
            "harness_version": self.harness_version,
            "wire_api": self.wire_api,
            "model": self.model,
            "instruction_blocks": deepcopy(self.instruction_blocks),
            "system": self.system,
            "tools": deepcopy(self.tools),
            "request_options": deepcopy(self.request_options),
            "captured_at": self.captured_at,
            "context_hash": self.context_hash,
        }

    def to_trace_event(self) -> dict[str, Any]:
        return {"type": TEICH_HARNESS_CONTEXT_EVENT_TYPE, "payload": self.to_dict()}


class HarnessCaptureServer:
    """Threaded fake provider that records exactly one sanitized request."""

    def __init__(
        self,
        *,
        harness: str,
        harness_version: str | None = None,
        minimum_tools: int = 0,
        host: str = "0.0.0.0",
        port: int = 0,
    ) -> None:
        self.harness = harness
        self.harness_version = harness_version
        self.minimum_tools = max(0, minimum_tools)
        self.secret = secrets.token_urlsafe(32)
        self._capture: HarnessContextCapture | None = None
        self._capture_lock = threading.Lock()
        self._capture_event = threading.Event()
        self._server = _CaptureHTTPServer((host, port), _CaptureHandler, owner=self)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> HarnessCaptureServer:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name=f"teich-{self.harness}-context-capture",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> HarnessCaptureServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def wait(self, timeout: float) -> HarnessContextCapture | None:
        if not self._capture_event.wait(timeout):
            return None
        with self._capture_lock:
            return self._capture

    def _authorized(self, headers: Any) -> bool:
        authorization = headers.get("Authorization", "")
        x_api_key = headers.get("x-api-key", "")
        return secrets.compare_digest(authorization, f"Bearer {self.secret}") or secrets.compare_digest(
            x_api_key, self.secret
        )

    def _record(self, wire_api: str, request: dict[str, Any]) -> HarnessContextCapture:
        candidate = HarnessContextCapture.from_request(
            harness=self.harness,
            harness_version=self.harness_version,
            wire_api=wire_api,
            request=request,
        )
        with self._capture_lock:
            current_score = (
                (len(self._capture.tools), len(self._capture.system))
                if self._capture is not None
                else (-1, -1)
            )
            candidate_score = (len(candidate.tools), len(candidate.system))
            if self._capture is None or candidate_score > current_score:
                self._capture = candidate
            if len(candidate.tools) >= self.minimum_tools:
                self._capture_event.set()
            return candidate


class _CaptureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *, owner: HarnessCaptureServer):
        self.owner = owner
        super().__init__(address, handler)


class _CaptureHandler(BaseHTTPRequestHandler):
    server: _CaptureHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        # Requests can contain private harness context.  Never place paths,
        # headers, or bodies in the default stderr access log.
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/health"):
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        owner = self.server.owner
        if not owner._authorized(self.headers):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        length_text = self.headers.get("Content-Length", "")
        try:
            length = int(length_text)
        except ValueError:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return
        if length < 0 or length > MAX_CAPTURE_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "object_required"})
            return

        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/messages/count_tokens"):
            self._send_json(HTTPStatus.OK, {"input_tokens": 1})
            return
        if path.endswith("/messages"):
            owner._record("anthropic-messages", payload)
            self._send_anthropic_response(payload)
            return
        if path.endswith("/responses"):
            owner._record("openai-responses", payload)
            self._send_responses_response(payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unsupported_endpoint"})

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events: list[tuple[str, dict[str, Any]]], *, include_done: bool = False) -> None:
        chunks: list[str] = []
        for event_name, payload in events:
            chunks.append(f"event: {event_name}\ndata: {_canonical_json(payload)}\n\n")
        if include_done:
            chunks.append("data: [DONE]\n\n")
        body = "".join(chunks).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_anthropic_response(self, request: dict[str, Any]) -> None:
        model = request.get("model") if isinstance(request.get("model"), str) else "teich-capture"
        message = {
            "id": "msg_teich_context_capture",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": CAPTURE_MARKER}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not request.get("stream"):
            self._send_json(HTTPStatus.OK, message)
            return
        events: list[tuple[str, dict[str, Any]]] = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {**message, "content": [], "stop_reason": None},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": CAPTURE_MARKER},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        self._send_sse(events)

    def _send_responses_response(self, request: dict[str, Any]) -> None:
        model = request.get("model") if isinstance(request.get("model"), str) else "teich-capture"
        output_item = {
            "id": "msg_teich_context_capture",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "text": CAPTURE_MARKER}],
        }
        response = {
            "id": "resp_teich_context_capture",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": model,
            "output": [output_item],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": False,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        if not request.get("stream"):
            self._send_json(HTTPStatus.OK, response)
            return
        in_progress = {**response, "status": "in_progress", "output": []}
        empty_item = {**output_item, "status": "in_progress", "content": []}
        content_part = {"type": "output_text", "annotations": [], "text": ""}
        events: list[tuple[str, dict[str, Any]]] = [
            ("response.created", {"type": "response.created", "response": in_progress}),
            (
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": 0, "item": empty_item},
            ),
            (
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": output_item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": content_part,
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": output_item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": CAPTURE_MARKER,
                },
            ),
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": output_item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "text": CAPTURE_MARKER,
                },
            ),
            (
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": output_item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": output_item["content"][0],
                },
            ),
            (
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": 0, "item": output_item},
            ),
            ("response.completed", {"type": "response.completed", "response": response}),
        ]
        self._send_sse(events, include_done=True)
