"""Where the speaker is now comes from their person entity, not the home block."""

from __future__ import annotations


def test_where_am_i_goes_to_the_speakers_person_entity() -> None:
    """ "What's my current location?" got "at home in Berlin" from the home
    block while the user was out (2026-11-16)."""
    from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE

    for prompt in (SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE):
        assert "where am I" in prompt
        assert "[SPEAKER]" in prompt


async def test_the_home_block_says_it_is_not_where_anyone_is_now() -> None:
    from unittest.mock import AsyncMock

    from .test_conversation import _baseline_orch, _make_mock_entry

    orch = _baseline_orch(_make_mock_entry())
    orch._location = AsyncMock()
    orch._location.async_resolve = AsyncMock(return_value={"city": "Berlin", "country_name": "Germany"})
    block = await orch._build_location_block()
    assert "not where anyone is now" in block
