"""A look-up repeated with the same arguments in one turn is not run again."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin.orchestrator import core

from .test_conversation import _baseline_orch


def _call(name, arguments):
    return SimpleNamespace(
        name=name, arguments=arguments,
        to_assistant_message=lambda: {"role": "assistant", "content": None},
        to_tool_result_message=lambda r: {"role": "tool", "content": r},
    )


def _orch(rounds, ha_local_names=()):
    """``rounds``: the tool calls the model makes, one list per iteration."""
    orch = _baseline_orch()
    orch._hass = MagicMock()
    if ha_local_names:
        orch._ha_local = SimpleNamespace(tool_names=set(ha_local_names))
    replies = [
        SimpleNamespace(
            has_tool_calls=True, tool_calls=[_call(n, a) for n, a in calls],
            reply_text=lambda: "",
        )
        for calls in rounds
    ]
    replies.append(SimpleNamespace(has_tool_calls=False, tool_calls=[], reply_text=lambda: "Done."))
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=replies)
    ran: list[tuple[str, dict]] = []

    async def _dispatch(name, arguments, *a, **kw):
        ran.append((name, arguments))
        return f"result {len(ran)}"

    orch._dispatch_tool = _dispatch
    return orch, ran


async def _run(orch):
    return await orch._tool_loop([{"role": "user", "content": "q"}], [])


async def test_same_search_twice_is_answered_without_running_it() -> None:
    q = {"query": "DHL Express drop off point near Bhf Pankow Berlin", "near_user": True}
    orch, ran = _orch([
        [("web_search", q)],
        [("web_search", {**q, "query": q["query"] + " address"})],
        [("web_search", dict(q))],
    ])
    reply, msgs = await _run(orch)
    assert reply == "Done."
    assert len(ran) == 2
    tool_results = [m["content"] for m in msgs if m["role"] == "tool"]
    assert tool_results[:2] == ["result 1", "result 2"]
    assert "already ran this exact web_search call" in tool_results[2]


async def test_case_spacing_and_key_order_do_not_make_a_new_search() -> None:
    orch, ran = _orch([
        [("web_search", {"query": "Cinema Berlin", "near_user": False})],
        [("web_search", {"near_user": False, "query": "cinema  berlin"})],
    ])
    await _run(orch)
    assert len(ran) == 1


async def test_actions_may_repeat() -> None:
    """A second "volume up" is a second step, not a wasted look-up."""
    step = {"entity_id": "media_player.hobbyraum"}
    orch, ran = _orch(
        [[("volume_up", step)], [("volume_up", dict(step))]], ha_local_names=["volume_up"]
    )
    await _run(orch)
    assert len(ran) == 2


async def test_guard_is_per_turn() -> None:
    q = {"query": "news"}
    orch, ran = _orch([[("web_search", q)]])
    await _run(orch)
    orch._provider.async_chat = AsyncMock(side_effect=[
        SimpleNamespace(has_tool_calls=True, tool_calls=[_call("web_search", q)], reply_text=lambda: ""),
        SimpleNamespace(has_tool_calls=False, tool_calls=[], reply_text=lambda: "Done."),
    ])
    await _run(orch)
    assert len(ran) == 2


def test_note_names_the_tool() -> None:
    assert "web_search" in core.REPEATED_LOOKUP_RESULT.format(name="web_search")
