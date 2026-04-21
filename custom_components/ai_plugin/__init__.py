"""AI Plugin — provider-agnostic AI orchestration for Home Assistant."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_MCP_SERVERS, DOMAIN
from .tools.memory import MemoryTool
from .tools.mcp_client import MCPToolRegistry
from .tools.ha_local import HALocalToolRegistry

PLATFORMS = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AI Plugin from a config entry."""
    # Build and start the MCP tool registry (may be empty if no servers configured).
    server_configs: list[dict] = entry.options.get(CONF_MCP_SERVERS, [])
    mcp = MCPToolRegistry(server_configs)
    await mcp.async_setup()

    # Native persistent memory tool (no external dependencies).
    memory = MemoryTool(hass.config.config_dir)

    # Local discovery tools (list_areas / list_entities / get_entity /
    # search_entities) hit HA registries directly — no MCP round-trip.
    ha_local = HALocalToolRegistry(hass)

    # Store in hass.data so Orchestrator can retrieve it.
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "mcp": mcp,
        "memory": memory,
        "ha_local": ha_local,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Reload the entry when options change so Orchestrator picks up new settings.
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry — tear down MCP connections first."""
    entry_data: dict | None = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    mcp: MCPToolRegistry | None = (entry_data or {}).get("mcp")
    if mcp is not None:
        # Kill stdio subprocesses and close HTTP sessions to prevent zombies.
        await mcp.async_teardown()

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
