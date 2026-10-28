"""The [USER FACTS] block injected before each turn."""

from __future__ import annotations

from unittest.mock import MagicMock

from .test_conversation import _baseline_orch


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
