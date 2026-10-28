"""API key fields of the MCP server screens."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_mcp_steps():
    """Import mcp_steps without the config_flow package, whose other modules
    need HA's real flow machinery (absent under the test stubs)."""
    import custom_components.ai_plugin as pkg

    helpers = sys.modules["homeassistant.helpers"]
    if not hasattr(helpers, "selector"):
        helpers.selector = MagicMock()
        sys.modules["homeassistant.helpers.selector"] = helpers.selector
    name = "custom_components.ai_plugin._mcp_steps_pkg"
    package = types.ModuleType(name)
    package.__path__ = []
    schemas = types.ModuleType(f"{name}.schemas")
    schemas._get_ha_url = lambda hass: "http://ha:8123"
    schemas._is_valid_url = lambda url: url.startswith("http")
    sys.modules[name] = package
    sys.modules[f"{name}.schemas"] = schemas
    path = Path(pkg.__file__).parent / "config_flow" / "mcp_steps.py"
    spec = importlib.util.spec_from_file_location(f"{name}.mcp_steps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


steps = _load_mcp_steps()


class _Flow(steps.MCPServerStepsMixin):
    """The MCP screens on a stand-in flow that records what it shows."""

    def __init__(self, servers: list[dict]) -> None:
        self._pending_mcp = servers
        self._options = {}
        self._preset_key = None
        self._editing_idx = None

    def async_show_form(self, **kw):
        return {"type": "form", **kw}

    async def async_step_mcp_servers(self, user_input=None):
        return {"type": "servers"}


def _wiki(**extra) -> dict:
    return {"transport": "stdio", "command": "uvx", "args": ["wikipedia-mcp"], **extra}


async def test_presets_are_added_at_once() -> None:
    flow = _Flow([])
    assert (await flow.async_step_mcp_add_preset({"preset_key": "time"}))["type"] == "servers"
    assert "env" not in flow._pending_mcp[0]


@pytest.mark.parametrize(
    ("server", "shown"),
    [
        ({"command": "uvx", "args": ["mcp-server-time"]}, None),
        ({"command": "npx", "args": ["x"], "env": {"BRAVE_API_KEY": "k"}}, "BRAVE_API_KEY"),
    ],
)
def test_edit_form_shows_the_right_variable(server, shown) -> None:
    assert steps._shown_key_env(server) == shown


async def test_editing_adds_a_key_and_keeps_the_args() -> None:
    flow = _Flow([_wiki(args=["--with", "mcp<2", "wikipedia-mcp"])])
    form = await flow.async_step_mcp_edit_pick({"mcp_index": "0"})
    assert form["step_id"] == "mcp_edit_stdio"
    await flow.async_step_mcp_edit_stdio({
        "mcp_command": "uvx",
        "mcp_args": "--with mcp<2 wikipedia-mcp",
        "mcp_key_env": "WIKIPEDIA_ACCESS_TOKEN",
        "mcp_key": "tok",
    })
    assert flow._pending_mcp == [
        _wiki(args=["--with", "mcp<2", "wikipedia-mcp"], env={"WIKIPEDIA_ACCESS_TOKEN": "tok"})
    ]


async def test_clearing_the_key_removes_only_that_variable() -> None:
    flow = _Flow([_wiki(env={"WIKIPEDIA_ACCESS_TOKEN": "tok", "OTHER": "1"})])
    await flow.async_step_mcp_edit_pick({"mcp_index": "0"})
    await flow.async_step_mcp_edit_stdio({
        "mcp_command": "uvx", "mcp_args": "wikipedia-mcp", "mcp_key_env": "WIKIPEDIA_ACCESS_TOKEN",
    })
    assert flow._pending_mcp == [_wiki(env={"OTHER": "1"})]


async def test_renaming_the_variable_moves_the_key() -> None:
    flow = _Flow([_wiki(env={"WRONG_NAME": "tok"})])
    await flow.async_step_mcp_edit_pick({"mcp_index": "0"})
    await flow.async_step_mcp_edit_stdio({
        "mcp_command": "uvx", "mcp_args": "wikipedia-mcp",
        "mcp_key_env": "WIKIPEDIA_ACCESS_TOKEN", "mcp_key": "tok",
    })
    assert flow._pending_mcp == [_wiki(env={"WIKIPEDIA_ACCESS_TOKEN": "tok"})]


async def test_custom_stdio_server_takes_a_key() -> None:
    flow = _Flow([])
    await flow.async_step_mcp_add_stdio({
        "mcp_command": "npx", "mcp_args": "-y brave-search", "mcp_key_env": "BRAVE_API_KEY", "mcp_key": "k",
    })
    assert flow._pending_mcp == [
        {"transport": "stdio", "command": "npx", "args": ["-y", "brave-search"], "env": {"BRAVE_API_KEY": "k"}}
    ]
