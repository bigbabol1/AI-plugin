"""The MCP server screens of the options flow.

Adding, editing and removing MCP servers is half the options flow by
volume and none of it by concept: a working list is edited in place and
committed when the user saves. Kept as a mixin so the step names stay
where Home Assistant expects to find them, on the options flow itself.
"""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.helpers import selector

from ..const import CONF_MCP_SERVERS, ERROR_INVALID_URL
from .schemas import _get_ha_url, _is_valid_url

_LOGGER = logging.getLogger(__name__)


class MCPServerStepsMixin:
    """MCP add / edit / remove screens. Mixed into AIPluginOptionsFlow."""

    # Preset stdio servers — "args_template" entries containing "{value}" will
    # have the placeholder replaced by a user-supplied config value.
    _STDIO_PRESETS: dict[str, dict[str, Any]] = {
        "time": {
            "label": "Clock & timezones — ask 'what time is it in Tokyo?' or 'how many hours until midnight?'",
            "command": "uvx",
            "args": ["mcp-server-time"],
        },
        "fetch": {
            "label": "Web reader — ask the AI to read any webpage, news article, or online document",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
        "sqlite": {
            "label": "SQLite — query and write a database file the AI can use as a notebook or scratchpad",
            "command": "uvx",
            "args_template": ["mcp-server-sqlite", "--db-path", "{value}"],
            "config_label": "database file path",
            "config_placeholder": "/config/buddy_notes.db",
        },
        "wikipedia": {
            "label": "Wikipedia — quick factual lookups without a web search",
            "command": "uvx",
            "args": ["wikipedia-mcp"],
        },
        "calculator": {
            "label": "Calculator — precise math without model guessing",
            "command": "uvx",
            "args": ["mcp-server-calculator"],
        },
    }

    def _mcp_server_label(self, s: dict) -> str:
        """Return a short human-readable label for an MCP server dict."""
        if s.get("transport") == "stdio":
            parts = [s.get("command", "")]
            parts.extend(s.get("args", []))
            return "stdio: " + " ".join(p for p in parts if p)
        url = s.get("url", "")
        token_hint = " (token set)" if s.get("token") else ""
        return f"http: {url}{token_hint}"

    def _mcp_servers_description(self) -> str:
        """Build a numbered list of configured MCP servers for the form description."""
        if not self._pending_mcp:
            return (
                "No servers configured yet.\n"
                "Tip: Add Home Assistant's built-in MCP server to let Buddy read "
                "your sensors and control devices."
            )
        lines = [f"{i}. {self._mcp_server_label(s)}" for i, s in enumerate(self._pending_mcp, 1)]
        return "\n".join(lines)

    async def async_step_mcp_servers(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """MCP server management — action selector with live server list."""
        if user_input is not None:
            action = user_input["mcp_action"]
            if action == "add_ha":
                return await self.async_step_mcp_add_ha()
            if action == "add_preset":
                return await self.async_step_mcp_add_preset()
            if action == "add_http":
                return await self.async_step_mcp_add_http()
            if action == "add_stdio":
                return await self.async_step_mcp_add_stdio()
            if action == "edit":
                return await self.async_step_mcp_edit_pick()
            if action == "remove":
                return await self.async_step_mcp_remove()
            # "save" — commit and close
            self._options[CONF_MCP_SERVERS] = self._pending_mcp
            return self.async_create_entry(title="", data=self._options)

        action_options: list[dict[str, str]] = [
            {"value": "add_ha", "label": "Add Home Assistant built-in MCP server"},
            {"value": "add_preset", "label": "Add a popular MCP server"},
            {"value": "add_http", "label": "Add other HTTP server"},
            {"value": "add_stdio", "label": "Add custom stdio server"},
        ]
        if self._pending_mcp:
            action_options.append({"value": "edit", "label": "Edit a server"})
            action_options.append({"value": "remove", "label": "Remove a server"})
        action_options.append({"value": "save", "label": "Save and close"})

        return self.async_show_form(
            step_id="mcp_servers",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_action"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=action_options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
            description_placeholders={"servers": self._mcp_servers_description()},
        )

    def _mcp_server_picker_schema(self) -> vol.Schema:
        """Schema for picking a server by index from the pending list."""
        options = [
            {"value": str(i), "label": f"{i + 1}. {self._mcp_server_label(s)}"}
            for i, s in enumerate(self._pending_mcp)
        ]
        return vol.Schema(
            {
                vol.Required("mcp_index"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )

    # ── Add ───────────────────────────────────────────────────────────────────

    async def async_step_mcp_add_ha(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Quick-add HA's built-in MCP server — only asks for the token."""
        if user_input is not None:
            token = user_input.get("ha_mcp_token", "").strip()
            ha_url = _get_ha_url(self.hass)
            entry: dict[str, Any] = {
                "transport": "sse",
                "url": f"{ha_url}/mcp_server/sse",
            }
            if token:
                entry["token"] = token
            self._pending_mcp.append(entry)
            return await self.async_step_mcp_servers()

        ha_url = _get_ha_url(self.hass)
        return self.async_show_form(
            step_id="mcp_add_ha",
            data_schema=vol.Schema(
                {
                    vol.Optional("ha_mcp_token", default=""): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            description_placeholders={"ha_mcp_url": f"{ha_url}/mcp_server/sse"},
        )

    async def async_step_mcp_add_preset(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Pick a popular preset MCP server to add."""
        if user_input is not None:
            key = user_input["preset_key"]
            preset = self._STDIO_PRESETS[key]
            if "args_template" in preset:
                # Needs extra config — store key and show config form
                self._preset_key = key
                return await self.async_step_mcp_preset_config()
            # No extra config — add immediately
            self._pending_mcp.append({
                "transport": "stdio",
                "command": preset["command"],
                "args": preset["args"],
            })
            return await self.async_step_mcp_servers()

        preset_options = [
            {"value": k, "label": v["label"]}
            for k, v in self._STDIO_PRESETS.items()
        ]
        return self.async_show_form(
            step_id="mcp_add_preset",
            data_schema=vol.Schema(
                {
                    vol.Required("preset_key"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=preset_options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_mcp_preset_config(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Collect a single path/value for presets that need it (sqlite, filesystem)."""
        preset = self._STDIO_PRESETS[self._preset_key]  # type: ignore[index]
        errors: dict[str, str] = {}
        if user_input is not None:
            value = user_input.get("preset_value", "").strip()
            if not value:
                errors["preset_value"] = "required"
            else:
                args = [
                    a.replace("{value}", value) for a in preset["args_template"]
                ]
                self._pending_mcp.append({
                    "transport": "stdio",
                    "command": preset["command"],
                    "args": args,
                })
                self._preset_key = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_preset_config",
            data_schema=vol.Schema(
                {
                    vol.Required("preset_value"): selector.TextSelector(
                        selector.TextSelectorConfig()
                    ),
                }
            ),
            description_placeholders={
                "preset_label": preset["label"],
                "config_label": preset.get("config_label", "Value"),
                "example": preset.get("config_placeholder", ""),
            },
            errors=errors,
        )

    async def async_step_mcp_add_http(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Add an HTTP MCP server (URL + optional bearer token)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input["mcp_url"].strip()
            if not _is_valid_url(url):
                errors["mcp_url"] = ERROR_INVALID_URL
            else:
                entry: dict[str, Any] = {"transport": "http", "url": url}
                token = user_input.get("mcp_token", "").strip()
                if token:
                    entry["token"] = token
                self._pending_mcp.append(entry)
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_add_http",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_url"): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                    vol.Optional("mcp_token", default=""): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_add_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Add a stdio MCP server (command + space-separated args)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            command = user_input.get("mcp_command", "").strip()
            if not command:
                errors["mcp_command"] = "command_required"
            else:
                raw_args = user_input.get("mcp_args", "").strip()
                self._pending_mcp.append(
                    {"transport": "stdio", "command": command, "args": raw_args.split() if raw_args else []}
                )
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_add_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_command"): selector.TextSelector(),
                    vol.Optional("mcp_args", default=""): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    # ── Edit ──────────────────────────────────────────────────────────────────

    async def async_step_mcp_edit_pick(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Pick which server to edit."""
        if not self._pending_mcp:
            return await self.async_step_mcp_servers()

        if user_input is not None:
            self._editing_idx = int(user_input["mcp_index"])
            s = self._pending_mcp[self._editing_idx]
            if s.get("transport") == "stdio":
                return await self.async_step_mcp_edit_stdio()
            return await self.async_step_mcp_edit_http()

        return self.async_show_form(
            step_id="mcp_edit_pick",
            data_schema=self._mcp_server_picker_schema(),
        )

    async def async_step_mcp_edit_http(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit an existing HTTP MCP server (pre-filled with current values)."""
        errors: dict[str, str] = {}
        existing = self._pending_mcp[self._editing_idx]  # type: ignore[index]

        if user_input is not None:
            url = user_input["mcp_url"].strip()
            if not _is_valid_url(url):
                errors["mcp_url"] = ERROR_INVALID_URL
            else:
                updated: dict[str, Any] = {"transport": "http", "url": url}
                token = user_input.get("mcp_token", "").strip()
                if token:
                    updated["token"] = token
                self._pending_mcp[self._editing_idx] = updated  # type: ignore[index]
                self._editing_idx = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_edit_http",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_url", default=existing.get("url", "")): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                    vol.Optional("mcp_token", default=existing.get("token", "")): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_edit_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit an existing stdio MCP server (pre-filled with current values)."""
        errors: dict[str, str] = {}
        existing = self._pending_mcp[self._editing_idx]  # type: ignore[index]

        if user_input is not None:
            command = user_input.get("mcp_command", "").strip()
            if not command:
                errors["mcp_command"] = "command_required"
            else:
                raw_args = user_input.get("mcp_args", "").strip()
                self._pending_mcp[self._editing_idx] = {  # type: ignore[index]
                    "transport": "stdio",
                    "command": command,
                    "args": raw_args.split() if raw_args else [],
                }
                self._editing_idx = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_edit_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_command", default=existing.get("command", "")): selector.TextSelector(),
                    vol.Optional(
                        "mcp_args",
                        default=" ".join(existing.get("args", [])),
                    ): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    # ── Remove ────────────────────────────────────────────────────────────────

    async def async_step_mcp_remove(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Remove one MCP server from the pending list."""
        if not self._pending_mcp:
            return await self.async_step_mcp_servers()

        if user_input is not None:
            idx = int(user_input["mcp_index"])
            if 0 <= idx < len(self._pending_mcp):
                self._pending_mcp.pop(idx)
            return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_remove",
            data_schema=self._mcp_server_picker_schema(),
        )
