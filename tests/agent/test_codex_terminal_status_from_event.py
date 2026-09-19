"""A terminal Responses frame without ``response.status`` keeps the outcome its event type names.

z-api (the chat.z.ai browser bridge) ends a failed turn with ``response.failed`` carrying only
``id`` + ``error``. The assembler's status defaulted to "completed", so the turn read as an empty
success: Hermes retried "response.output is empty" three times and never showed the bridge's
reason (extension gone, composer missing, deadline exceeded).
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _bare_terminal(event_type: str, **response_fields):
    return [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": event_type, "response": {"id": "resp_1", **response_fields}},
    ]


def test_bare_failed_frame_surfaces_the_provider_error():
    from agent.turn_recovery import validate_response_shape
    from agent.transports.codex import ResponsesApiTransport

    error = {"code": "COMPOSER_NOT_FOUND", "message": "Z.ai composer was not found"}
    final = _consume_codex_event_stream(_bare_terminal("response.failed", error=error), model="glm-web")
    assert final.status == "failed"

    agent = SimpleNamespace(api_mode="codex_responses", _get_transport=ResponsesApiTransport,
                            _client_log_context=lambda: "")
    invalid, details = validate_response_shape(agent, final)
    assert invalid and error["message"] in details[0]


def test_explicit_status_still_wins_over_the_event_type():
    final = _consume_codex_event_stream(_bare_terminal("response.completed", status="incomplete"), model="glm-web")
    assert final.status == "incomplete"

    bare_incomplete = _consume_codex_event_stream(_bare_terminal("response.incomplete"), model="glm-web")
    assert bare_incomplete.status == "incomplete"
