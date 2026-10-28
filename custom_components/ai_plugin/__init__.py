"""AI Plugin — provider-agnostic AI orchestration for Home Assistant."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

import logging

from .const import CONF_MCP_SERVERS, DOMAIN
from .tools.memory import MemoryTool
from .tools.mcp_client import MCPToolRegistry
from .tools.ha_local import HALocalToolRegistry
from .voice_id import VoiceIDManager

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION, Platform.AI_TASK, Platform.STT]


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

    # Voice recognition state (profiles, audio buffers, model). Created even
    # when the feature is off, so turning it off can restore the pipelines.
    voice_id = VoiceIDManager(hass, entry)
    await voice_id.async_load()

    # Store in hass.data so Orchestrator can retrieve it.
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "mcp": mcp,
        "memory": memory,
        "ha_local": ha_local,
        "voice_id": voice_id,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _schedule_pipeline_sync(hass, entry, voice_id)
    _refresh_tools_on_entity_changes(hass, entry, mcp)

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


def _schedule_pipeline_sync(hass: HomeAssistant, entry: ConfigEntry, voice_id) -> None:
    """Switch pipelines to the STT pass-through (feature on) or back (off).

    Waits for HA to finish starting: assist_pipeline and the wrapped STT
    engines (HA Cloud loads late) must exist first.
    """
    from .voice_id import pipelines  # noqa: PLC0415

    async def _sync(_event=None) -> None:
        try:
            if voice_id.enabled:
                hass.async_create_task(voice_id.async_ensure_model())
                await pipelines.async_switch(hass, entry, voice_id)
            elif voice_id.pipeline_originals:
                await pipelines.async_restore(hass, entry, voice_id)
                pipelines.remove_wrapper_entities(hass, entry)
        except Exception:  # noqa: BLE001 — never block setup on this
            _LOGGER.warning("AI Plugin voice ID: pipeline update failed", exc_info=True)

    if getattr(hass, "is_running", True):
        hass.async_create_task(_sync())
    else:
        from homeassistant.const import EVENT_HOMEASSISTANT_STARTED  # noqa: PLC0415

        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _sync)


# Exposure and entity changes come in bursts (startup adds hundreds of
# entities, a dialog exposes several); refresh once the burst has ended.
_TOOL_REFRESH_DELAY_S = 2.0


def _refresh_tools_on_entity_changes(
    hass: HomeAssistant, entry: ConfigEntry, mcp: MCPToolRegistry
) -> None:
    """Fetch the MCP tool lists again when the entities behind them change.

    Home Assistant's MCP server offers a tool only while an exposed entity
    it acts on exists: HassListAddItem needs an exposed to-do list. The
    list is otherwise read once at connect, which during startup happens
    before slower integrations have created their entities (the to-do
    tools were missing after a restart, 2026-10-27), and a list exposed
    later stayed out of reach until the next restart.
    """
    from homeassistant.const import EVENT_STATE_CHANGED  # noqa: PLC0415

    pending = None

    def _refresh() -> None:
        nonlocal pending
        pending = None
        hass.async_create_task(mcp.async_refresh_tools())

    def _schedule(*_args) -> None:
        nonlocal pending
        if pending is not None:
            pending.cancel()
        pending = hass.loop.call_later(_TOOL_REFRESH_DELAY_S, _refresh)

    def _entity_added_or_removed(event) -> None:
        data = event.data
        if data.get("old_state") is None or data.get("new_state") is None:
            _schedule()

    def _cancel_pending() -> None:
        if pending is not None:
            pending.cancel()

    entry.async_on_unload(_cancel_pending)
    entry.async_on_unload(
        hass.bus.async_listen(EVENT_STATE_CHANGED, _entity_added_or_removed)
    )
    try:
        from homeassistant.components.homeassistant.exposed_entities import (  # noqa: PLC0415
            async_listen_entity_updates,
        )

        entry.async_on_unload(
            async_listen_entity_updates(hass, "conversation", _schedule)
        )
    except Exception:  # noqa: BLE001 — never block setup on this
        _LOGGER.debug("AI Plugin: exposure updates not available", exc_info=True)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Integration removed: give the pipelines their original STT back."""
    from .voice_id import pipelines  # noqa: PLC0415

    voice_id = VoiceIDManager(hass, entry)
    try:
        await voice_id.async_load()
        if voice_id.pipeline_originals:
            await pipelines.async_restore(hass, entry, voice_id)
        await voice_id.async_remove_storage()
    except Exception:  # noqa: BLE001
        _LOGGER.warning("AI Plugin voice ID: cleanup on removal failed", exc_info=True)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
