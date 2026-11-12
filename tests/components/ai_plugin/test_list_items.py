"""Items said in a row go on the to-do list one entry each."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_conversation import _baseline_orch  # isort: skip (stubs HA first)
from custom_components.ai_plugin.orchestrator.toolcalls import split_list_items


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("Oat milk, parmesan", ["Oat milk", "Parmesan"]),
        ("Bacon, vegetables, eggs", ["Bacon", "Vegetables", "Eggs"]),
        ("bacon, vegetables and eggs", ["bacon", "vegetables", "eggs"]),
        ("bacon, vegetables, and eggs", ["bacon", "vegetables", "eggs"]),
        ("Milch, Eier und Butter", ["Milch", "Eier", "Butter"]),
        ("lait; œufs", ["lait", "œufs"]),
        ("Greek yogurt", ["Greek yogurt"]),
        ("mac and cheese", ["mac and cheese"]),
        ("1,5 l Milch", ["1,5 l Milch"]),
        ("eggs, ", ["eggs, "]),
    ],
)
def test_split_list_items(item: str, expected: list[str]) -> None:
    assert split_list_items(item) == expected


def test_split_list_items_ignores_a_missing_item() -> None:
    assert split_list_items(None) == []


def _orch_with_mcp():
    orch = _baseline_orch()
    orch._mcp = MagicMock(call_tool=AsyncMock(side_effect=lambda n, a, **_: f"added {a['item']}"))
    orch._mcp.is_home_assistant_tool = MagicMock(return_value=True)
    orch._ha_local = None
    return orch


async def test_items_in_a_row_are_added_one_by_one() -> None:
    orch = _orch_with_mcp()

    result = await orch._dispatch_tool(
        "HassListAddItem", {"item": "Oat milk, parmesan", "name": "Einkaufsliste"}, None
    )

    calls = [c.args for c in orch._mcp.call_tool.await_args_list]
    assert calls == [
        ("HassListAddItem", {"item": "Oat milk", "name": "Einkaufsliste"}),
        ("HassListAddItem", {"item": "Parmesan", "name": "Einkaufsliste"}),
    ]
    assert result == "added Oat milk\nadded Parmesan"


async def test_one_item_is_added_as_it_was() -> None:
    orch = _orch_with_mcp()

    await orch._dispatch_tool("HassListAddItem", {"item": "Greek yogurt", "name": "x"}, None)

    orch._mcp.call_tool.assert_awaited_once_with(
        "HassListAddItem", {"item": "Greek yogurt", "name": "x"}, device_id=None, user_id=None
    )
