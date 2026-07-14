"""Tests for the AI Task platform (v0.9.30)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.components.ai_task import GenDataTask
from homeassistant.exceptions import HomeAssistantError

from custom_components.ai_plugin.ai_task import AIPluginTaskEntity, _extract_json
from custom_components.ai_plugin.exceptions import OrchestratorError

from .conftest import MOCK_CONFIG_OPTIONS


def _entity(reply: str | Exception) -> AIPluginTaskEntity:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.options = dict(MOCK_CONFIG_OPTIONS)
    ent = AIPluginTaskEntity.__new__(AIPluginTaskEntity)
    ent._entry = entry
    provider = MagicMock()
    if isinstance(reply, Exception):
        provider.async_complete = AsyncMock(side_effect=reply)
    else:
        provider.async_complete = AsyncMock(return_value=reply)
    ent._provider = provider
    return ent


def _chat_log():
    log = MagicMock()
    log.conversation_id = "conv-task"
    log.async_add_assistant_content_without_tools = AsyncMock()
    return log


async def test_generate_data_plain_text() -> None:
    ent = _entity("A short summary.")
    task = GenDataTask(name="summarize", instructions="Summarize the day")
    result = await ent._async_generate_data(task, _chat_log())
    assert result.data == "A short summary."
    assert result.conversation_id == "conv-task"


async def test_generate_data_structured_parses_fenced_json() -> None:
    ent = _entity('```json\n{"mood": "good", "score": 8}\n```')
    task = GenDataTask(
        name="mood", instructions="Rate the day", structure={"mood": str}
    )
    result = await ent._async_generate_data(task, _chat_log())
    assert result.data == {"mood": "good", "score": 8}


async def test_generate_data_invalid_json_raises() -> None:
    ent = _entity("definitely not json")
    task = GenDataTask(name="x", instructions="y", structure={"a": int})
    with pytest.raises(HomeAssistantError, match="valid JSON"):
        await ent._async_generate_data(task, _chat_log())


async def test_generate_data_provider_error_raises_ha_error() -> None:
    ent = _entity(OrchestratorError("backend down"))
    task = GenDataTask(name="x", instructions="y")
    with pytest.raises(HomeAssistantError, match="backend down"):
        await ent._async_generate_data(task, _chat_log())


def test_extract_json_variants() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n[1, 2]\n```') == [1, 2]
