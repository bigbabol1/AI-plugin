"""Reading the voice prompt in ahead of the next voice turn."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.orchestrator import warmup

from .test_conversation import _baseline_orch


def _tool(name: str, text: str = "") -> dict:
    return {"type": "function", "function": {"name": name, "description": text}}


@pytest.fixture
def orch(monkeypatch):
    monkeypatch.setattr(warmup, "WARM_CHECK_S", 0.01)
    monkeypatch.setattr(warmup, "WARM_IDLE_S", 0.0)
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._hass.async_create_background_task = lambda coro, name: asyncio.ensure_future(coro)
    orch._tools = [_tool("get_entity"), _tool("web_search")]
    orch._tool_schemas_for = lambda message, voice, device: list(orch._tools)
    orch._build_system_prompt = AsyncMock(return_value="You are a voice assistant.")
    orch._provider = MagicMock()
    orch._provider.async_warm = AsyncMock(return_value=True)
    orch._voice_in_use = lambda: True
    yield orch
    for task in orch._bg_tasks:
        task.cancel()


async def _ticks(n: int = 6) -> None:
    for _ in range(n):
        await asyncio.sleep(0.015)


async def test_head_is_read_in_once_it_has_settled(orch) -> None:
    orch.async_start_warmup()
    await _ticks()
    orch._provider.async_warm.assert_awaited_once_with(
        [
            {"role": "system", "content": "You are a voice assistant."},
            {"role": "user", "content": "."},
        ],
        orch._tools,
    )


async def test_changed_tool_list_is_read_in_again(orch) -> None:
    orch.async_start_warmup()
    await _ticks()
    orch._tools = [*orch._tools, _tool("get_summary")]
    await _ticks()
    assert orch._provider.async_warm.await_count == 2


async def test_head_a_voice_turn_just_read_is_not_read_again(orch) -> None:
    orch._note_prompt_head("You are a voice assistant.", list(orch._tools), True)
    orch.async_start_warmup()
    await _ticks()
    orch._provider.async_warm.assert_not_awaited()


async def test_text_turn_means_the_voice_head_must_be_read_again(orch) -> None:
    orch._note_prompt_head("You are a voice assistant.", list(orch._tools), True)
    orch._note_prompt_head("You are a chat assistant.", list(orch._tools), False)
    orch.async_start_warmup()
    await _ticks()
    orch._provider.async_warm.assert_awaited_once()


async def test_nothing_is_read_in_right_after_a_turn(orch, monkeypatch) -> None:
    monkeypatch.setattr(warmup, "WARM_IDLE_S", 60.0)
    orch._note_prompt_head("You are a chat assistant.", [], False)
    orch.async_start_warmup()
    await _ticks()
    orch._provider.async_warm.assert_not_awaited()


async def test_check_stops_when_the_server_keeps_no_prompt(orch) -> None:
    orch._provider.async_warm = AsyncMock(return_value=None)
    orch.async_start_warmup()
    await _ticks()
    assert orch._provider.async_warm.await_count == 1
    assert all(t.done() for t in orch._bg_tasks)


async def test_pruned_schemas_have_no_head_to_read_in(orch) -> None:
    orch._prune_schemas = True
    orch.async_start_warmup()
    assert not orch._bg_tasks


async def test_no_read_in_where_voice_never_reaches_the_agent(orch) -> None:
    """A text-only setup: reading the voice prompt in would only push the
    chat's own prompt out of the model after every text turn."""
    del orch._voice_in_use
    with patch("custom_components.ai_plugin.voice_id.pipelines._pipelines", return_value=[]):
        orch.async_start_warmup()
        await _ticks()
    orch._provider.async_warm.assert_not_awaited()


def test_voice_in_use_from_pipelines_or_a_voice_turn(orch) -> None:
    del orch._voice_in_use
    voice = SimpleNamespace(stt_engine="stt.cloud", conversation_engine="conversation.ai")
    text = SimpleNamespace(stt_engine=None, conversation_engine="conversation.ai")
    other = SimpleNamespace(stt_engine="stt.cloud", conversation_engine="conversation.other")
    path = "custom_components.ai_plugin.voice_id.pipelines"
    with patch(f"{path}.agent_entity_ids", return_value={"conversation.ai"}):
        with patch(f"{path}._pipelines", return_value=[text, other]):
            assert not orch._voice_in_use()
            orch._note_prompt_head("p", [], True)
            assert orch._voice_in_use()
        orch._voice_turn_seen = False
        with patch(f"{path}._pipelines", return_value=[voice]):
            assert orch._voice_in_use()


def test_head_change_names_what_differs() -> None:
    old = ("prompt", [_tool("a"), _tool("b", "x")])
    new = ("prompt 2", [_tool("b", "y"), _tool("c")])
    assert warmup.head_change(old, new) == (
        "system prompt changed; 1 tools added (c); 1 tools removed (a); tools changed (b)"
    )
    assert warmup.head_change(None, new) == "first since start"
