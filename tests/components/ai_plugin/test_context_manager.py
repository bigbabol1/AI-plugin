"""Tests for the ContextManager (sliding window + summarization)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


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
    # First message should be the summary block
    assert hist[0]["role"] == "system"
    assert hist[0]["content"].startswith("[EARLIER CONVERSATION]")
    assert "Summary of old messages." in hist[0]["content"]


async def test_summary_reaches_the_model() -> None:
    """Stored as an assistant message at the front, the summary was dropped
    by get_messages as an orphan before every request."""
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    provider = MagicMock()
    provider.async_complete = AsyncMock(return_value="The user asked about Ebow's concert.")
    for _ in range(6):
        await cm.add_turn("c", "user", "x" * 80)
        await cm.add_turn("c", "assistant", "y" * 80)

    await cm.summarize_if_needed("c", "s", provider)
    msgs = await cm.get_messages("c", "s")
    assert any("Ebow" in m["content"] for m in msgs[1:])
    assert msgs[2]["role"] == "user"


async def test_summarize_keeps_recent_turns() -> None:
    """The newest turns are kept verbatim and whole, up to half the soft
    limit; everything older goes into the summary."""
    cm = ContextManager(max_tokens=400, tool_token_budget=0)
    system = "s"
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Old stuff summarized.")

    for i in range(6):
        await cm.add_turn("c", "user", f"user msg {i} " + "x" * 60)
        await cm.add_raw_message("c", {"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}"}]})
        await cm.add_raw_message("c", {"role": "tool", "tool_call_id": f"t{i}", "content": "r" * 40})
        await cm.add_turn("c", "assistant", f"asst msg {i} " + "x" * 60)

    soft_limit, _ = cm._budget(system)
    await cm.summarize_if_needed("c", system, mock_provider)

    hist = cm.get_history("c")
    kept = hist[1:]
    assert kept[0]["role"] == "user"  # split at a turn start, never mid-turn
    assert kept[-1]["content"].startswith("asst msg 5")
    assert cm._estimate_messages_tokens(kept) <= soft_limit // 2
    sent = mock_provider.async_complete.await_args.args[0]
    assert any("user msg 0" in str(m.get("content")) for m in sent)
    assert not any(m.get("role") == "tool" for m in sent)


async def test_kept_turn_keeps_its_context_block() -> None:
    cm = ContextManager(max_tokens=300, tool_token_budget=0)
    provider = MagicMock()
    provider.async_complete = AsyncMock(return_value="Earlier turns.")
    for i in range(6):
        await cm.add_turn("c", "user", f"question {i} " + "x" * 60)
        await cm.add_before_last_user("c", {"role": "system", "content": f"[CURRENT TIME] {i}"})
        await cm.add_turn("c", "assistant", f"answer {i} " + "y" * 60)

    await cm.summarize_if_needed("c", "s", provider)
    hist = cm.get_history("c")
    assert hist[1]["content"].startswith("[CURRENT TIME]")
    assert hist[2]["role"] == "user"
    sent = provider.async_complete.await_args.args[0]
    assert not any("[CURRENT TIME]" in str(m.get("content")) for m in sent)


async def test_a_later_summary_folds_in_the_earlier_one() -> None:
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    provider = MagicMock()
    provider.async_complete = AsyncMock(side_effect=["First summary.", "Second summary."])
    for _ in range(6):
        await cm.add_turn("c", "user", "x" * 80)
        await cm.add_turn("c", "assistant", "y" * 80)
    await cm.summarize_if_needed("c", "s", provider)
    for _ in range(4):
        await cm.add_turn("c", "user", "z" * 80)
        await cm.add_turn("c", "assistant", "w" * 80)
    await cm.summarize_if_needed("c", "s", provider)

    sent = provider.async_complete.await_args.args[0]
    assert any("First summary." in str(m.get("content")) for m in sent)
    hist = cm.get_history("c")
    assert "Second summary." in hist[0]["content"]
    assert sum(m["content"].startswith("[EARLIER CONVERSATION]") for m in hist) == 1


async def test_rollback_keeps_the_summary() -> None:
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    provider = MagicMock()
    provider.async_complete = AsyncMock(return_value="Earlier turns.")
    for _ in range(6):
        await cm.add_turn("c", "user", "x" * 80)
        await cm.add_turn("c", "assistant", "y" * 80)
    await cm.summarize_if_needed("c", "s", provider)
    cm._history["c"] = cm._history["c"][:1]
    await cm.add_turn("c", "user", "new question")
    await cm.remove_last_turn("c")
    assert cm.get_history("c")[0]["content"].startswith("[EARLIER CONVERSATION]")


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
    """A single turn over the limit has nothing older to summarize."""
    cm = ContextManager(max_tokens=200, tool_token_budget=0)
    system = "s"
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Summary.")

    await cm.add_turn("c", "user", "x" * 400)
    await cm.add_turn("c", "assistant", "y" * 400)

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


# ── v0.9.26: per-request tool-token budget + idle eviction ────────────────────


async def test_get_messages_tool_tokens_override_tightens_budget() -> None:
    """Passing the real schema size must shrink the history budget."""
    cm = ContextManager(max_tokens=1000, tool_token_budget=0)
    for _ in range(10):
        await cm.add_turn("c", "user", "x" * 240)  # ~60 tokens each

    # Constructor budget (0) reserves nothing → all 10 messages fit.
    msgs = await cm.get_messages("c", "sys")
    assert len(msgs) == 11  # system + 10

    # Actual schemas worth ~800 tokens → history must be truncated.
    msgs2 = await cm.get_messages("c", "sys", tool_tokens=800)
    assert 2 <= len(msgs2) < 11


async def test_summarize_uses_tool_tokens_override() -> None:
    """tool_tokens must feed the soft limit, not just the hard limit."""
    cm = ContextManager(max_tokens=1000, tool_token_budget=0)
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Summary.")

    for _ in range(5):
        await cm.add_turn("c", "user", "x" * 200)       # ~50 tokens
        await cm.add_turn("c", "assistant", "y" * 200)  # ~50 tokens

    # 500 history tokens, soft limit without override ≈ 649 → no trigger.
    await cm.summarize_if_needed("c", "s", mock_provider)
    mock_provider.async_complete.assert_not_awaited()

    # With 400 reserved for schemas, soft limit ≈ 389 → triggers.
    await cm.summarize_if_needed("c", "s", mock_provider, tool_tokens=400)
    mock_provider.async_complete.assert_awaited_once()


async def test_evict_idle_drops_only_stale_conversations() -> None:
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("old", "user", "hi")
    await cm.add_turn("fresh", "user", "hi")
    cm._last_used["old"] -= 7 * 3600  # simulate 7h idle

    evicted = cm.evict_idle()

    assert evicted == ["old"]
    assert cm.get_history("old") == []
    assert cm.get_history("fresh")
    assert "old" not in cm._locks
    assert "old" not in cm._last_used


async def test_evict_idle_noop_when_all_fresh() -> None:
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("a", "user", "hi")
    assert cm.evict_idle() == []
    assert cm.get_history("a")


# ══════════════════════════════════════════════════════════════════════════════
# per-turn context blocks kept in history
# ══════════════════════════════════════════════════════════════════════════════


async def _two_turns_with_context(cm: ContextManager) -> None:
    for n in (1, 2):
        await cm.add_turn("c", "user", f"question {n}")
        await cm.add_before_last_user("c", {"role": "system", "content": f"[CURRENT TIME] {n}"})
        await cm.add_turn("c", "assistant", f"answer {n}")


async def test_context_block_sits_before_its_user_turn() -> None:
    cm = ContextManager(max_tokens=8192)
    await _two_turns_with_context(cm)
    msgs = await cm.get_messages("c", "sys")
    assert [m["content"] for m in msgs[1:]] == [
        "[CURRENT TIME] 1", "question 1", "answer 1",
        "[CURRENT TIME] 2", "question 2", "answer 2",
    ]


async def test_rollback_takes_the_context_block_with_the_user_turn() -> None:
    cm = ContextManager(max_tokens=8192)
    await cm.add_turn("c", "user", "question 1")
    await cm.add_before_last_user("c", {"role": "system", "content": "[CURRENT TIME] 1"})
    await cm.remove_last_turn("c")
    assert cm.get_history("c") == []


async def test_trimming_never_starts_mid_turn() -> None:
    """Whatever the budget, trimmed history opens with a user turn or the
    context block sent with it, never with an answer or a lone block."""
    for max_tokens in range(30, 140, 3):
        cm = ContextManager(max_tokens=max_tokens, tool_token_budget=0)
        await _two_turns_with_context(cm)
        for _ in range(3):
            await cm.add_turn("c", "user", "q" * 40)
            await cm.add_before_last_user("c", {"role": "system", "content": "t" * 40})
            await cm.add_turn("c", "assistant", "a" * 40)
        history = (await cm.get_messages("c", "X"))[1:]
        if not history:
            continue
        roles = [m["role"] for m in history]
        assert roles[0] == "user" or roles[:2] == ["system", "user"], (max_tokens, roles)


async def test_budget_report_counts_context_blocks_as_history() -> None:
    cm = ContextManager(max_tokens=8192)
    await _two_turns_with_context(cm)
    msgs = await cm.get_messages("c", "sys")
    report = cm.budget_report("sys", msgs, 0)
    assert report["history"] == cm._estimate_messages_tokens(msgs[1:])
