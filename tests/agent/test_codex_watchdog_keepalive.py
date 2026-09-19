"""Codex event-idle watchdog respects a provider's declared keepalive cadence.

z-api (the chat.z.ai browser bridge) cannot stream tokens: while the page thinks it sends a
``response.in_progress`` keepalive every 20s. The implicit small-prompt event-idle tier is 12s,
so every such request was killed between two healthy keepalives and reconnected until the retry
budget ran out. ``ProviderProfile.stream_keepalive_seconds`` floors the IMPLICIT idle default;
an explicit operator value still wins.
"""
from __future__ import annotations

from pathlib import Path

_SMALL_PROMPT = {"model": "glm-web", "input": [{"role": "user", "content": "x" * 6000}]}


def _agent(tmp_path: Path, monkeypatch, provider: str, base_url: str):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    for var in ("HERMES_API_CALL_STALE_TIMEOUT", "HERMES_CODEX_TTFB_TIMEOUT_SECONDS",
                "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "HERMES_CODEX_TTFB_MAX_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    from run_agent import AIAgent

    agent = AIAgent(
        model="glm-web", provider=provider, api_key="sk-dummy", base_url=base_url, quiet_mode=True,
        skip_context_files=True, skip_memory=True, platform="cli",
    )
    agent.api_mode = "codex_responses"
    return agent


def test_idle_watchdog_outlasts_the_declared_keepalive_gap(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs
    from providers import get_provider_profile

    keepalive = get_provider_profile("zai-web").stream_keepalive_seconds
    assert keepalive > 0

    bridge = _resolve_nonstream_watchdogs(
        _agent(tmp_path, monkeypatch, "zai-web", "http://127.0.0.1:3230/v1"), _SMALL_PROMPT)
    assert bridge.idle_enabled and bridge.idle_timeout > keepalive

    # Control: the same small request to a profile with no keepalive keeps a tier tighter than
    # that gap, so the floor is what separates the two.
    plain = _resolve_nonstream_watchdogs(
        _agent(tmp_path, monkeypatch, "custom", "http://127.0.0.1:8000/v1"), _SMALL_PROMPT)
    assert plain.idle_timeout < keepalive


def test_explicit_idle_timeout_beats_the_keepalive_floor(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    agent = _agent(tmp_path, monkeypatch, "zai-web", "http://127.0.0.1:3230/v1")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "15")
    assert _resolve_nonstream_watchdogs(agent, _SMALL_PROMPT).idle_timeout == 15.0
