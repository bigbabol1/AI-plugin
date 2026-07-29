"""Tests for streaming replies (v0.9.29): delta gate, provider stream, wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin.orchestrator import _DeltaGate
from custom_components.ai_plugin.providers import ChatResponse

from .test_conversation import _baseline_orch, _make_mock_entry


# ── _DeltaGate ────────────────────────────────────────────────────────────────


def test_gate_holds_back_trailing_sentence() -> None:
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("It is sunny today. Tomorrow will ")
    gate.feed("bring rain.")
    assert got == ["It is sunny today. "]
    gate.flush_final("It is sunny today. Tomorrow will bring rain.")
    assert "".join(got).split() == "It is sunny today. Tomorrow will bring rain.".split()


def test_gate_flush_sends_everything_when_nothing_streamed() -> None:
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.flush_final("Short answer.")
    assert got == ["Short answer."]


def test_gate_delivers_full_reply_on_divergent_final() -> None:
    """A rewritten final reply is delivered whole through the stream —
    once sentences streamed, the stream is what the pipeline speaks, so
    withholding on divergence would strand the user without the answer."""
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("The weather is nice. And more ")
    assert got == ["The weather is nice. "]
    gate.flush_final("A completely different reply.")
    assert got == ["The weather is nice. ", "A completely different reply."]


def test_gate_withholds_final_contained_in_forwarded() -> None:
    """If the final is a subset of what already streamed (a post-processor
    trimmed the tail), nothing more is sent — no double-speak."""
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("The weather is nice. Extra tail. And more ")
    assert got == ["The weather is nice. ", "Extra tail. "]
    gate.flush_final("The weather is nice.")
    assert got == ["The weather is nice. ", "Extra tail. "]


def test_gate_closed_after_partial_forward_still_delivers_final() -> None:
    """close() after sentences streamed must not strand the user with a
    dangling preamble: flush_final still completes the stream with the
    authoritative reply."""
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("Sure. And now ")
    assert got == ["Sure. "]
    gate.close()
    gate.feed("this must not stream. Nor this. ")
    assert got == ["Sure. "]
    gate.flush_final("It's 21 degrees in the bedroom.")
    assert got == ["Sure. ", "It's 21 degrees in the bedroom."]


def test_gate_closed_with_nothing_forwarded_stays_silent() -> None:
    """close() before anything streamed keeps the stream unused — the
    plain speech field is authoritative (suppressed turns rely on this)."""
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.close()
    gate.flush_final("Should not stream.")
    assert got == []


def test_gate_drops_narration_sentences() -> None:
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("I'm checking the temperature. The bedroom is at 21 degrees. Done ")
    assert got == ["The bedroom is at 21 degrees. "]


def test_gate_closes_on_think_leak() -> None:
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("<think>secret plan</think> Hello. There ")
    assert got == []
    assert not gate.active


def test_gate_new_response_discards_pre_tool_prose() -> None:
    got: list[str] = []
    gate = _DeltaGate(got.append, lang="en")
    gate.feed("Let me look that up")  # no sentence boundary → held
    gate.new_response()
    gate.feed("The answer is 42. Final ")
    assert got == ["The answer is 42. "]


def test_inactive_gate_is_inert() -> None:
    gate = _DeltaGate(None)
    gate.feed("Hello there. World ")
    gate.flush_final("Hello there. World")
    assert gate.forwarded_text == ""


# ── provider NDJSON stream parsing ────────────────────────────────────────────


class _FakeStreamResp:
    def __init__(self, lines: list[dict]) -> None:
        self.content = self._iter(lines)
        self.status = 200

    @staticmethod
    async def _iter(lines):
        for line in lines:
            yield (json.dumps(line) + "\n").encode()


async def test_read_ollama_stream_forwards_content_and_stops_on_tool_call() -> None:
    from custom_components.ai_plugin.providers.openai_compat import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:11434/v1", model="m", api_key=None, timeout=30
    )
    got: list[str] = []
    resp = _FakeStreamResp([
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo. "}},
        {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}},
        {"message": {"content": "post-tool prose"}},
        {"done": True},
    ])
    result = await provider._read_ollama_stream(resp, got.append)
    assert got == ["Hel", "lo. "]  # nothing forwarded after the tool call
    assert result.content == "Hello. post-tool prose"
    assert result.tool_calls and result.tool_calls[0].name == "t"


# ── orchestrator wiring ───────────────────────────────────────────────────────


async def test_orchestrator_streams_safe_chat_turn() -> None:
    orch = _baseline_orch()
    got: list[str] = []

    async def fake_stream(messages, tools=None, on_delta=None):
        on_delta("Nice weather chat. Second ")
        on_delta("sentence here.")
        return ChatResponse(content="Nice weather chat. Second sentence here.")

    mock_provider = MagicMock()
    mock_provider.async_chat_stream = AsyncMock(side_effect=fake_stream)
    orch._provider = mock_provider

    reply = await orch.async_process(
        "tell me something nice", "conv-stream", "en", on_delta=got.append
    )
    assert reply == "Nice weather chat. Second sentence here."
    joined = "".join(got)
    assert joined.startswith("Nice weather chat.")
    # flush_final delivered the held-back remainder
    assert "Second sentence here." in " ".join(joined.split())


async def test_orchestrator_does_not_stream_state_set_queries() -> None:
    orch = _baseline_orch()
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="No lights are on.")
    orch._provider = mock_provider

    got: list[str] = []
    reply = await orch.async_process(
        "are any lights on?", "conv-gated", "en", on_delta=got.append
    )
    assert reply == "No lights are on."
    mock_provider.async_complete.assert_awaited_once()
    # verifier-triggering turns must stay fully buffered
    assert got == []


# ── conversation entity wiring ────────────────────────────────────────────────


async def test_entity_streams_deltas_into_chat_log(hass) -> None:
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = _make_mock_entry()
    entity._attr_unique_id = "test"

    async def fake_process(**kwargs):
        on_delta = kwargs["on_delta"]
        on_delta("It is sunny. ")
        on_delta("Enjoy the day.")
        return "It is sunny. Enjoy the day."

    mock_orch = MagicMock()
    mock_orch.async_process = AsyncMock(side_effect=fake_process)
    mock_orch.is_voice_device = MagicMock(return_value=False)
    entity._orchestrator = mock_orch

    user_input = ConversationInput(
        text="how is the weather",
        context=Context(),
        conversation_id="conv-e2e",
        device_id=None,
        language="en",
        agent_id=None,
    )
    result = await entity.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == "It is sunny. Enjoy the day."
    # deltas actually reached the (stub) chat log — kwargs prove wiring
    assert mock_orch.async_process.await_args.kwargs["on_delta"] is not None
