"""Local HA discovery and area-action tools for AI Plugin (mcp-assist-style).

Small LLMs drown in a dumped YAML inventory of all exposed devices. This
module provides five on-demand tools that execute inside the HA process
against entity/area/device registries directly:

    list_areas()                          → every area name
    list_entities(area?, domain?)         → entities, optionally filtered
    get_entity(name_or_id)                → one entity: id, name, area, state, attrs
    search_entities(query)                → substring match across names/aliases
    set_area_state(area, domain, action)  → bulk action on all entities in an area

Discovery responses are capped at ~1500 characters so they fit comfortably in
the model's context; overflow reports a truncation count so the model can
narrow the filter and retry.

Single-device actions (set brightness, set temperature, etc.) still flow
through HA's built-in MCP server tools (HassTurnOn, HassLightSet, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant

from . import exposure
from .entities import EntityToolsMixin
from .formatting import _MAX_RESPONSE_CHARS, _cap, _s  # noqa: F401
from .lights import (
    _BRIGHTER_RE,
    _DEVICE_NOUN_RE,
    _DIMMER_RE,
    _SWEEP_ALL_KEYWORDS,
    _USER_SAID_ALL_RE,
    LightToolsMixin,
)
from .media import MediaToolsMixin
from .schemas import TOOL_NAMES, TOOL_SCHEMAS
from .timers import TimerToolsMixin
from .timers import _TIMER_INTENTS

_LOGGER = logging.getLogger(__name__)




class HALocalToolRegistry(
    EntityToolsMixin, LightToolsMixin, MediaToolsMixin, TimerToolsMixin
):
    """Registry of discovery tools running against HA's in-process registries."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        if not exposure._EXPOSURE_API_AVAILABLE:
            _LOGGER.warning(
                "AI Plugin: exposed_entities API unavailable on this HA version — "
                "exposed_only filter disabled, all registry entities will be shown"
            )

    @property
    def tool_names(self) -> set[str]:
        return TOOL_NAMES

    def get_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def _is_exposed(self, entity_id: str) -> bool:
        """Return True if entity is exposed to the conversation assistant."""
        if not exposure._EXPOSURE_API_AVAILABLE:
            return True  # Fallback: treat all as exposed on older HA.
        try:
            return exposure._ha_should_expose(self._hass, "conversation", entity_id)
        except Exception:  # noqa: BLE001
            return True  # Fail open — never silently hide entities on error.

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        device_id: str | None = None,
        language: str | None = None,
        user_message: str = "",
        available_tools: set[str] | None = None,
        announce_agent_id: str | None = None,
    ) -> str:
        """Dispatch one tool call. Never raises; errors become string replies.

        available_tools is the full tool-name set the model can call this
        request (ha_local + MCP + built-ins). Recovery messages use it to
        avoid directing the model at HassTurnOn/HassTurnOff when HA's MCP
        server is not connected — those directives were a dead end. None
        means unknown; assume the Hass tools exist (pre-0.9.26 behaviour).
        """
        has_hass_actuators = (
            available_tools is None or "HassTurnOn" in available_tools
        )
        try:
            exposed_only = arguments.get("exposed_only", True)
            if not isinstance(exposed_only, bool):
                exposed_only = True

            if name in _TIMER_INTENTS:
                return await self._call_timer_intent(
                    name, arguments, device_id=device_id, language=language,
                    user_message=user_message,
                    announce_agent_id=announce_agent_id,
                )

            if name == "list_areas":
                return self._list_areas()
            if name == "list_entities":
                return self._list_entities(
                    area=_s(arguments.get("area")).strip() or None,
                    domain=_s(arguments.get("domain")).strip() or None,
                    state=_s(arguments.get("state")).strip() or None,
                    exposed_only=exposed_only,
                )
            if name == "get_entity":
                return await self._get_entity(
                    _s(arguments.get("name_or_id")).strip(),
                    exposed_only=exposed_only,
                )
            if name == "search_entities":
                limit = arguments.get("limit")
                try:
                    limit = int(limit) if limit is not None else 10
                except (TypeError, ValueError):
                    limit = 10
                return self._search_entities(
                    query=_s(arguments.get("query")).strip(),
                    limit=max(1, min(50, limit)),
                    exposed_only=exposed_only,
                )
            if name == "set_brightness":
                lvl = arguments.get("level")
                try:
                    lvl = int(lvl) if lvl is not None else None
                except (TypeError, ValueError):
                    lvl = None
                raw_area = arguments.get("area")
                area_b = _s(raw_area).strip() if raw_area is not None else ""
                said_all_b = bool(_USER_SAID_ALL_RE.search(user_message or ""))
                if not area_b:
                    named = self._area_named_in_message(user_message or "")
                    if named:
                        area_b = named
                    elif said_all_b:
                        area_b = "all"
                cmd_b = _s(arguments.get("command")).strip().lower()
                if cmd_b not in ("brighter", "dimmer", "set"):
                    msg_b = user_message or ""
                    if _DIMMER_RE.search(msg_b):
                        cmd_b = "dimmer"
                    elif _BRIGHTER_RE.search(msg_b):
                        cmd_b = "brighter"
                    elif lvl is not None:
                        cmd_b = "set"
                    if cmd_b:
                        _LOGGER.debug(
                            "set_brightness: command omitted/invalid, inferred "
                            "%r from message %r", cmd_b, msg_b[:60],
                        )
                return await self._set_brightness(
                    command=cmd_b,
                    level=lvl,
                    area=area_b or None,
                    name=_s(arguments.get("name")).strip() or None,
                    exposed_only=exposed_only,
                    device_id=device_id,
                    allow_sweep=said_all_b,
                )

            if name == "set_area_state":
                raw_area = arguments.get("area")
                area_val = _s(raw_area).strip() if raw_area is not None else ""
                said_all = bool(_USER_SAID_ALL_RE.search(user_message or ""))
                # Scope ladder for an OMITTED area. Small models drop the
                # argument constantly, and defaulting straight to the
                # calling satellite's room silently shrank whole-home
                # commands to one room: "switch all lights off" spoken in
                # the bedroom left the rest of the flat lit. Precedence:
                #   1. a room the user actually named  → that room
                #   2. an explicit 'all'/'whole house' → sweep everything
                #   3. neither                        → caller's room
                #      (resolved in _set_area_state from device_id)
                if not area_val:
                    named_area = self._area_named_in_message(user_message or "")
                    if named_area:
                        area_val = named_area
                        _LOGGER.debug(
                            "set_area_state: area omitted, using room named in "
                            "message %r", named_area,
                        )
                    elif said_all:
                        area_val = "all"
                        _LOGGER.debug(
                            "set_area_state: area omitted but user said 'all' "
                            "— sweeping every area instead of caller's room",
                        )
                # Safety guard: when the user message names a specific
                # physical device ('air purifier', 'kettle', 'TV',
                # 'luftreiniger', etc.) the user does NOT mean a
                # domain-wide sweep — but small models often fall back
                # to set_area_state when their first HassTurnOff attempt
                # fails. Refuse and auto-recover with search_entities.
                if _DEVICE_NOUN_RE.search(user_message or ""):
                    _LOGGER.warning(
                        "AI Plugin: rejected set_area_state(area=%r, domain=%r) — "
                        "user message %r names a specific device. Auto-running "
                        "search_entities for recovery.",
                        area_val, _s(arguments.get("domain")),
                        (user_message or "")[:80],
                    )
                    recovery = self._auto_search_recovery(
                        user_message or "",
                        device_id=device_id,
                        exposed_only=exposed_only,
                        has_hass_actuators=has_hass_actuators,
                    )
                    if has_hass_actuators:
                        return (
                            "[set_area_state refused — user named a SPECIFIC "
                            "device, not a domain category. CALL HassTurnOn/"
                            "HassTurnOff with the matching entity_id below, "
                            "preferring the one in the caller's area.]\n" + recovery
                        )
                    return (
                        "[set_area_state refused — user named a SPECIFIC "
                        "device, not a domain category. Direct single-device "
                        "tools are NOT available in this setup. Tell the user "
                        "which device you found (below) and that you cannot "
                        "switch it directly; controlling single devices needs "
                        "the Home Assistant MCP server connected in AI Plugin "
                        "settings.]\n" + recovery
                    )
                # Safety guard: model often falls back to area='all' after a
                # failed HassTurnOff for a specific device, which then turns
                # off everything in the flat (lights + thermostats + fans).
                # Reject area='all' / sweep keywords unless the user
                # explicitly said 'all'/'every'/'whole house'/etc.
                # An OMITTED area with no satellite device to fall back to
                # (text Assist / REST) would sweep the whole home through the
                # same code path — hold it to the same explicit-phrase bar.
                if area_val.lower() in _SWEEP_ALL_KEYWORDS or (
                    not area_val and not device_id
                ):
                    if not said_all:
                        _LOGGER.warning(
                            "AI Plugin: rejected set_area_state(area=%r) — "
                            "user message %r does not contain explicit sweep "
                            "keyword. Auto-running search_entities to surface "
                            "the actual specific device.",
                            area_val, (user_message or "")[:80],
                        )
                        # Auto-recover: run search_entities with the user
                        # message as query, biased toward the calling
                        # satellite's area when possible. Return hits so the
                        # model can call HassTurnOn/Off in the next loop
                        # iteration. Without this the model often gives up
                        # after rejection.
                        recovery = self._auto_search_recovery(
                            user_message or "",
                            device_id=device_id,
                            exposed_only=exposed_only,
                            has_hass_actuators=has_hass_actuators,
                        )
                        if has_hass_actuators:
                            return (
                                "[set_area_state whole-home sweep refused — "
                                "user did not say 'all', 'everything', or "
                                "'whole house', and no room could be inferred. "
                                "Specific-device recovery results below. CALL "
                                "HassTurnOn/HassTurnOff with the matching "
                                "entity_id, preferring the one in the caller's "
                                "area — or pass the room the user named.]\n"
                                + recovery
                            )
                        return (
                            "[set_area_state whole-home sweep refused — user "
                            "did not say 'all', 'everything', or 'whole "
                            "house', and no room could be inferred. Direct "
                            "single-device tools are NOT available in this "
                            "setup. Tell the user which device you found "
                            "(below) and that you cannot switch it directly.]\n"
                            + recovery
                        )
                return await self._set_area_state(
                    area=area_val or None,
                    domain=_s(arguments.get("domain")).strip(),
                    action=_s(arguments.get("action")).strip(),
                    exposed_only=exposed_only,
                    device_id=device_id,
                )
            if name == "play_music":
                return await self._play_music(
                    query=_s(arguments.get("query")).strip(),
                    area=_s(arguments.get("area")).strip() or None,
                    media_type=_s(arguments.get("media_type")).strip() or "track",
                    radio_mode=arguments.get("radio_mode"),
                    exposed_only=exposed_only,
                    device_id=device_id,
                )
            if name == "media_command":
                return await self._media_command(
                    command=_s(arguments.get("command")).strip().lower(),
                    area=_s(arguments.get("area")).strip() or None,
                    exposed_only=exposed_only,
                    level=arguments.get("level"),
                )
            return f"[Unknown local tool: {name!r}]"
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin local tool %r failed", name)
            return f"[Local tool error: {exc}]"
