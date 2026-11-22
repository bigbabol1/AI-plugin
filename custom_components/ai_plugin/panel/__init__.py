"""The AI Plugin page in Home Assistant's sidebar.

One plain web component (frontend/ai-plugin-panel.js), served from this
folder, talks to the admin-only websocket commands in api.py. The page is
registered once per Home Assistant run and shown while at least one
enabled entry has ``show_panel`` on. The options flow keeps every setting too.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import CONF_SHOW_PANEL, DEFAULT_SHOW_PANEL, DOMAIN

_LOGGER = logging.getLogger(__name__)

PANEL_URL_PATH = "ai-plugin"
STATIC_URL = "/ai_plugin_static"
WEBCOMPONENT = "ai-plugin-panel"
_JS = "ai-plugin-panel.js"
_FRONTEND = Path(__file__).parent / "frontend"
_STATE_KEY = "ai_plugin_panel"


def _state(hass: HomeAssistant) -> dict:
    return hass.data.setdefault(
        _STATE_KEY, {"api": False, "static": False, "panel": False}
    )


def _wanted(hass: HomeAssistant, skip_entry_id: str | None = None) -> bool:
    """True while an entry (other than ``skip_entry_id``) wants the page."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == skip_entry_id or entry.disabled_by:
            continue
        if entry.options.get(CONF_SHOW_PANEL, DEFAULT_SHOW_PANEL):
            return True
    return False


def _file_hash() -> str:
    return hashlib.sha1((_FRONTEND / _JS).read_bytes()).hexdigest()[:10]


async def async_setup_panel(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Register the websocket commands, the files and the sidebar entry."""
    state = _state(hass)
    if not state["api"]:
        from .api import async_register_commands  # noqa: PLC0415

        async_register_commands(hass)
        state["api"] = True
    if not _wanted(hass):
        await async_remove_panel(hass)
        return
    if state["panel"]:
        return
    try:
        if not state["static"]:
            from homeassistant.components.http import StaticPathConfig  # noqa: PLC0415

            await hass.http.async_register_static_paths(
                [StaticPathConfig(STATIC_URL, str(_FRONTEND), False)]
            )
            state["static"] = True
        version = await hass.async_add_executor_job(_file_hash)
        from homeassistant.components import panel_custom  # noqa: PLC0415

        await panel_custom.async_register_panel(
            hass,
            frontend_url_path=PANEL_URL_PATH,
            webcomponent_name=WEBCOMPONENT,
            sidebar_title="AI Plugin",
            sidebar_icon="mdi:creation",
            module_url=f"{STATIC_URL}/{_JS}?v={version}",
            require_admin=True,
            config={},
        )
        state["panel"] = True
    except Exception:  # noqa: BLE001 — the assistant works without its page
        _LOGGER.warning("AI Plugin: could not add the sidebar page", exc_info=True)


async def async_remove_panel(hass: HomeAssistant) -> None:
    state = _state(hass)
    if not state["panel"]:
        return
    try:
        from homeassistant.components import frontend  # noqa: PLC0415

        frontend.async_remove_panel(hass, PANEL_URL_PATH)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("AI Plugin: could not remove the sidebar page", exc_info=True)
    state["panel"] = False


async def async_entry_removed(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Integration entry deleted: take the page away when no entry wants it.

    Not done on unload: saving settings reloads the entry, and removing the
    page then would throw the user off the page they just saved on.
    """
    if not _wanted(hass, skip_entry_id=entry.entry_id):
        await async_remove_panel(hass)
