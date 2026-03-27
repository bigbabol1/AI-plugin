"""Tests for the ContextManager (sliding window + summarization)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin.context_manager import ContextManager


# ══════════════════════════════════════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════════════════════════════════════


def test_estimate_tokens_basic() -> None:
    """4 chars ≈ 1 token."""
    assert ContextManager.estimate_tokens("abcd") == 1
    assert ContextManager.estimate_tokens("a" * 400) == 100


def test_estimate_tokens_never_zero() -> None:
    """Empty or very-short strings return at least 1."""
    assert ContextManager.estimate_tokens("") == 1
    assert ContextManager.estimate_tokens("hi") == 1


# ══════════════════════════════════════════════════════════════════════════════
# add_turn / get_history
# ══════════════════════════════════════════════════════════════════════════════


async def test_add_turn_appends_message() -> None:
    """add_turn appends the message to the conv history."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("conv-1", "user", "Hello")
    hist = cm.get_history("conv-1")
    assert hist == [{"role": "user", "content": "Hello"}]


async def test_add_turn_multiple_turns() -> None:
    """Multiple add_turn calls build the history in order."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("conv-1", "user", "Hi")
    await cm.add_turn("conv-1", "assistant", "Hello!")
    await cm.add_turn("conv-1", "user", "How are you?")
    hist = cm.get_history("conv-1")
    assert len(hist) == 3
    assert hist[0]["role"] == "user"
    assert hist[1]["role"] == "assistant"
    assert hist[2]["content"] == "How are you?"


# ══════════════════════════════════════════════════════════════════════════════
# get_messages
# ══════════════════════════════════════════════════════════════════════════════


async def test_get_messages_includes_system_prompt() -> None:
    """get_messages prepends the system prompt as the first message."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("conv-1", "user", "Hi")
    msgs = await cm.get_messages("conv-1", "You are helpful.")
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "Hi"}


async def test_get_messages_empty_history() -> None:
    """get_messages works with no prior history."""
    cm = ContextManager(max_tokens=8192)
    msgs = await cm.get_messages("new-conv", "System prompt.")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


async def test_get_messages_hard_limit_truncates() -> None:
    """Messages exceeding the hard limit (85%) are dropped oldest-first."""
    # Very small context window so a few messages exceed the limit.
    # max_tokens=100, tool_budget=0 → available=~100, hard_limit=85
    cm = ContextManager(max_tokens=100, tool_token_budget=0)
    system = "X"  # 1 token → available ≈ 99, hard_limit ≈ 84

    # Add 10 turns with ~10 tokens each (40 chars) — total ≈ 100 tokens
    for i in range(10):
        await cm.add_turn("c", "user", "a" * 40)        # ~10 tokens each
        await cm.add_turn("c", "assistant", "b" * 40)

    msgs = await cm.get_messages("c", system)
    # System prompt must always be present
    assert msgs[0]["role"] == "system"
    # Total history tokens must be within hard limit
    history_only = msgs[1:]
    total_tokens = sum(cm.estimate_tokens(m["content"]) for m in history_only)
    _, hard_limit = cm._budget(system)
    assert total_tokens <= hard_limit


async def test_get_messages_preserves_order() -> None:
    """History order (oldest → newest) is preserved in output."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("c", "user", "first")
    await cm.add_turn("c", "assistant", "reply1")
    await cm.add_turn("c", "user", "second")
    await cm.add_turn("c", "assistant", "reply2")
    msgs = await cm.get_messages("c", "sys")
    contents = [m["content"] for m in msgs[1:]]
    assert contents == ["first", "reply1", "second", "reply2"]


# ══════════════════════════════════════════════════════════════════════════════
# summarize_if_needed
# ══════════════════════════════════════════════════════════════════════════════


async def test_summarize_skips_when_under_soft_limit() -> None:
    """No summarization when history is comfortably under the soft limit."""
    cm = ContextManager(max_tokens=8192)
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock()

    await cm.add_turn("c", "user", "Hello")
    await cm.add_turn("c", "assistant", "Hi!")

    await cm.summarize_if_needed("c", "System prompt.", mock_provider)
    mock_provider.async_complete.assert_not_awaited()


async def test_summarize_fires_when_over_soft_limit() -> None:
    """Summarization is triggered when history exceeds the soft limit."""
    # Small window so a few long messages trigger it
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    system = "s"  # minimal system tokens
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Summary of old messages.")

    # Add 12 messages of ~20 tokens each → 240 tokens >> soft limit ≈ 129
    for i in range(6):
        await cm.add_turn("c", "user", "x" * 80)       # ~20 tokens
        await cm.add_turn("c", "assistant", "y" * 80)  # ~20 tokens

    await cm.summarize_if_needed("c", system, mock_provider)
    mock_provider.async_complete.assert_awaited_once()

    # History should be shorter now (summary + recent turns)
    hist = cm.get_history("c")
    assert len(hist) < 12
    # First message should be the summary
    assert "Prior conversation summary:" in hist[0]["content"]


async def test_summarize_keeps_recent_turns() -> None:
    """The most recent KEEP_RECENT_TURNS (4) full turns are preserved verbatim."""
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    system = "s"
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Old stuff summarized.")

    # 6 turns total; most recent 4 should be kept
    messages_added = []
    for i in range(6):
        u = f"user msg {i}"
        a = f"asst msg {i}"
        await cm.add_turn("c", "user", u + "x" * 60)   # pad to push over limit
        await cm.add_turn("c", "assistant", a + "x" * 60)
        messages_added.append((u, a))

    await cm.summarize_if_needed("c", system, mock_provider)

    hist = cm.get_history("c")
    # Last 4 turns = 8 messages should appear in history (after summary)
    kept_contents = [m["content"] for m in hist[1:]]  # skip summary message
    # The last 4 user messages should all be present
    for u, a in messages_added[-4:]:
        assert any(u in c for c in kept_contents), f"Missing: {u!r}"


async def test_summarize_failure_leaves_history_intact() -> None:
    """If summarization LLM call fails, history is NOT mutated."""
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    system = "s"
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(side_effect=RuntimeError("timeout"))

    for i in range(6):
        await cm.add_turn("c", "user", "q" * 80)
        await cm.add_turn("c", "assistant", "a" * 80)

    original_len = len(cm.get_history("c"))
    await cm.summarize_if_needed("c", system, mock_provider)  # should not raise

    assert len(cm.get_history("c")) == original_len


async def test_summarize_skips_when_too_few_messages() -> None:
    """No summarization when there are too few messages to split."""
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    system = "s"
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Summary.")

    # Add only 4 turns (= 8 messages = KEEP_RECENT_TURNS * 2) — nothing to summarize
    for _ in range(4):
        await cm.add_turn("c", "user", "x" * 80)
        await cm.add_turn("c", "assistant", "y" * 80)

    await cm.summarize_if_needed("c", system, mock_provider)
    mock_provider.async_complete.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════════
# Per-conversation isolation
# ══════════════════════════════════════════════════════════════════════════════


async def test_conversations_are_isolated() -> None:
    """Two conversation IDs maintain completely separate histories."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("a", "user", "Hello from A")
    await cm.add_turn("b", "user", "Hello from B")

    assert cm.get_history("a") == [{"role": "user", "content": "Hello from A"}]
    assert cm.get_history("b") == [{"role": "user", "content": "Hello from B"}]


# ══════════════════════════════════════════════════════════════════════════════
# clear
# ══════════════════════════════════════════════════════════════════════════════


async def test_clear_removes_history() -> None:
    """clear() wipes history for the specified conversation."""
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("c", "user", "Hi")
    await cm.clear("c")
    assert cm.get_history("c") == []


async def test_clear_unknown_conv_is_noop() -> None:
    """clear() on a conversation that never existed does not raise."""
    cm = ContextManager(max_tokens=8192)
    await cm.clear("nonexistent")  # should not raise
