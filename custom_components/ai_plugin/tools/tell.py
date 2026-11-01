"""Giving someone a message: on their phone, or spoken in one room.

Home Assistant's own HassBroadcast speaks everywhere at once, so "tell the
kitchen dinner is ready" and "send Anna a message" had nowhere to go.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from .. import satellite_speech
from ..i18n import L

_LOGGER = logging.getLogger(__name__)

TOOL_NAME = "tell"

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Give someone a message. To a person: 'tell Anna dinner is ready', "
            "'send Anna a message that I am late' goes to their phone. To a room: "
            "'tell the kitchen dinner is ready' is spoken there. For every room "
            "use HassBroadcast."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message itself, as they should hear it: 'Dinner is ready'.",
                },
                "person": {"type": "string", "description": "Name of the person, for their phone."},
                "area": {"type": "string", "description": "Room to speak it in."},
            },
            "required": ["message"],
        },
    },
}


async def async_tell(
    hass,
    arguments: dict[str, Any],
    *,
    language: str | None,
    resolve_person: Callable[[str], str | None],
    sender_name: str | None = None,
) -> str:
    """Deliver the message. Never raises; errors go back to the model."""
    from ..shortcuts.phone import (  # noqa: PLC0415
        async_notify_person,
        people_with_phones,
        phones_of,
    )

    lang = (language or "en").split("-")[0].lower()
    message = str(arguments.get("message") or "").strip()
    person = str(arguments.get("person") or "").strip()
    area = str(arguments.get("area") or "").strip()
    if not message:
        return "[Not sent: 'message' must say what to tell them.]"
    if not person and not area:
        return "[Not sent: ask who or which room. For every room use HassBroadcast.]"
    try:
        if person:
            person_id = resolve_person(person)
            if person_id is None or not phones_of(hass, person_id):
                known = ", ".join(sorted(people_with_phones(hass).values())) or "nobody"
                return f"[Not sent: no phone known for {person}. People with a phone: {known}.]"
            title = (
                L.template("message_title_from", lang, name=sender_name)
                if sender_name else L.template("message_title", lang)
            )
            if not await async_notify_person(hass, person_id, title, message):
                return f"[Not sent: {person}'s phone did not take it.]"
            return f"Sent to {person}'s phone: {message}."
        device_id = satellite_speech.satellite_in_area(hass, area)
        if device_id is None:
            return f"[Not said: there is no satellite in {area}. Tell the user.]"
        spoken = await satellite_speech.async_announce(
            hass, device_id, "scheduled_say", lang, label=message
        )
        if spoken is None:
            return f"[Not said: the satellite in {area} could not play it.]"
        return f"Said in {area}: {message}."
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("AI Plugin: tell failed")
        return f"[Not sent: {exc}]"
