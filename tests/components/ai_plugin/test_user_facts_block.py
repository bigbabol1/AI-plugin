"""The [USER FACTS] block injected before each turn."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin.context_manager import ContextManager
from custom_components.ai_plugin.providers import ChatResponse

from .test_conversation import _baseline_orch, _voice_orch_with_intent


def _orch(facts):
    orch = _baseline_orch()
    orch._memory = MagicMock()
    orch._memory._config_dir = "/config"
    orch._memory._load = MagicMock(return_value=facts)
    return orch


async def test_facts_are_numbered() -> None:
    block = await _orch(["likes jazz", "ALWAYS use the metric system"])._build_user_facts_block("u1")
    assert block.startswith("[USER FACTS]\n")
    assert block.endswith("1. likes jazz\n2. ALWAYS use the metric system")


async def test_preferences_apply_to_answers_on_any_topic() -> None:
    """2026-10-27: scoped to "questions about who they are", a saved
    metric-only preference was ignored for a question about the Megalodon."""
    block = await _orch(["ALWAYS use the metric system"])._build_user_facts_block("u1")
    assert "apply to EVERY answer, whatever the topic" in block
    assert "feet to metres" in block


async def test_no_facts_no_block() -> None:
    assert await _orch([])._build_user_facts_block("u1") == ""


# ── repeated facts ───────────────────────────────────────────────────────────

FACTS = "[USER FACTS]\nFacts previously saved.\n1. likes tea"
OTHER = "[USER FACTS]\nFacts previously saved.\n1. likes coffee"


def _manager(*contents: str, max_tokens: int = 100_000) -> ContextManager:
    manager = ContextManager(max_tokens=max_tokens, tool_token_budget=0)
    manager._history["c"] = [
        {"role": "system" if "FACTS" in c else "user", "content": c} for c in contents
    ]
    return manager


async def _seen(manager: ContextManager, block: str = FACTS) -> bool:
    return await manager.newest_block_is("c", "[USER FACTS]\n", block, "prompt", 0)


async def test_facts_the_model_can_see_are_not_sent_again() -> None:
    assert await _seen(_manager(f"[CURRENT TIME]\nnow\n\n{FACTS}", "Hi."))


async def test_facts_are_sent_at_the_start_of_a_conversation() -> None:
    assert not await _seen(_manager("Hi."))


async def test_changed_facts_are_sent_again() -> None:
    assert not await _seen(_manager(FACTS, "Hi."), OTHER)


async def test_back_to_the_first_speaker_their_facts_are_sent_again() -> None:
    assert not await _seen(_manager(FACTS, "Hi.", OTHER, "Hi."))


async def test_facts_a_summary_replaced_are_sent_again() -> None:
    manager = _manager("[EARLIER CONVERSATION]\nSummary of this conversation so far: x", "Hi.")
    assert not await _seen(manager)


async def test_facts_the_hard_limit_would_cut_are_sent_again() -> None:
    long_turn = "x" * 2000
    assert await _seen(_manager(FACTS, "Hi.", long_turn))
    assert not await _seen(_manager(FACTS, "Hi.", long_turn, max_tokens=500))


async def test_only_the_first_turn_carries_the_facts() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._provider.async_chat = AsyncMock(
        return_value=ChatResponse(content="Hello.", tool_calls=[])
    )
    orch._build_user_facts_block = AsyncMock(return_value=FACTS)
    for text in ("Hello there.", "Tell me a joke.", "Another one."):
        await orch.async_process(text, "conv-f", "en")
    blocks = [
        m for m in orch._context_mgr.get_history("conv-f") if "[USER FACTS]\n" in str(m["content"])
    ]
    assert len(blocks) == 1


async def test_the_facts_come_back_after_they_change() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._provider.async_chat = AsyncMock(
        return_value=ChatResponse(content="Hello.", tool_calls=[])
    )
    orch._build_user_facts_block = AsyncMock(return_value=FACTS)
    await orch.async_process("Hello there.", "conv-g", "en")
    orch._build_user_facts_block = AsyncMock(return_value=OTHER)
    await orch.async_process("Tell me a joke.", "conv-g", "en")
    history = orch._context_mgr.get_history("conv-g")
    assert sum("[USER FACTS]\n" in str(m["content"]) for m in history) == 2
    assert OTHER in "\n".join(str(m["content"]) for m in history)
