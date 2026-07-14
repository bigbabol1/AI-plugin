"""Tests for intent-based model routing (v0.9.30)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin.providers import ChatResponse

from .test_conversation import _baseline_orch


def _routed_orch(home=None, web=None, general=None):
    orch = _baseline_orch()
    orch._route_models = {"home": home, "web": web, "general": general}
    return orch


def test_route_classification() -> None:
    orch = _routed_orch(home="fast:3b", web="big:32b", general="mid:8b")
    orch._web_search = object()  # enable the web route branch
    assert orch._pick_route_model("turn on the kitchen lights") == "fast:3b"
    assert orch._pick_route_model("are any lights on?") == "fast:3b"
    assert orch._pick_route_model("set a timer for five minutes") == "fast:3b"
    assert orch._pick_route_model("what happened in Berlin this weekend?") == "big:32b"
    assert orch._pick_route_model("write me a haiku about coffee") == "mid:8b"


def test_route_disabled_when_unconfigured() -> None:
    orch = _routed_orch()
    assert orch._pick_route_model("turn on the lights") is None


def test_route_falls_back_per_class() -> None:
    orch = _routed_orch(home="fast:3b")  # web/general unset → main model
    assert orch._pick_route_model("turn on the lights") == "fast:3b"
    assert orch._pick_route_model("write a poem") is None


async def test_routed_turn_passes_model_to_provider() -> None:
    orch = _routed_orch(general="mid:8b")

    captured = {}

    async def fake_chat(messages, tools=None, model=None):
        captured["model"] = model
        return ChatResponse(content="ok")

    mock_provider = MagicMock()
    mock_provider.async_chat = AsyncMock(side_effect=fake_chat)
    orch._provider = mock_provider

    reply = await orch.async_process("write a poem", "conv-route", "en")
    assert reply == "ok"
    assert captured["model"] == "mid:8b"


async def test_unrouted_turn_keeps_async_complete_path() -> None:
    orch = _routed_orch()
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="plain")
    orch._provider = mock_provider

    reply = await orch.async_process("write a poem", "conv-plain", "en")
    assert reply == "plain"
    mock_provider.async_complete.assert_awaited_once()
