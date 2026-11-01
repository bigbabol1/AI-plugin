"""Adding an event to a calendar by voice.

Home Assistant's MCP server can read calendars (get_events) but not write to
them. Only calendars that are exposed to Assist and accept new events are
offered, so this never reaches a calendar the user has not shared with the
assistant.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

from ..i18n import L
from ..scheduler.when import WhenError, parse_clock, parse_day

_LOGGER = logging.getLogger(__name__)

TOOL_NAME = "add_calendar_event"
_CREATE_EVENT = 1  # CalendarEntityFeature.CREATE_EVENT
_DEFAULT_MINUTES = 60
_RRULES = {"yearly": "FREQ=YEARLY", "monthly": "FREQ=MONTHLY", "weekly": "FREQ=WEEKLY"}
_EVERY = {"yearly": "every year", "monthly": "every month", "weekly": "every week"}

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Add an event to a calendar: 'put the dentist on my calendar for "
            "Friday at 10'. To read a calendar use get_events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What the event is: 'Dentist'."},
                "day": {
                    "type": "string",
                    "description": "'today', 'tomorrow', a weekday such as 'friday', or YYYY-MM-DD.",
                },
                "at": {
                    "type": "string",
                    "description": "Start time, 24-hour HH:MM. OMIT for an all-day event.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Only when the user said how long. Default 60.",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["none", *_RRULES],
                    "description": (
                        "Only when the user said so: 'every week'. A birthday "
                        "repeats every year on its own."
                    ),
                },
                "calendar": {
                    "type": "string",
                    "description": "Calendar name, only when the user named one.",
                },
            },
            "required": ["summary", "day"],
        },
    },
}


def _writable_calendars(hass) -> list[tuple[str, str]]:
    """(entity_id, name) of the exposed calendars that accept new events."""
    from ..shortcuts import registry  # noqa: PLC0415

    found = []
    for state in hass.states.async_all("calendar"):
        features = state.attributes.get("supported_features") or 0
        if features & _CREATE_EVENT and registry.is_exposed(hass, state.entity_id):
            found.append((state.entity_id, state.attributes.get("friendly_name") or state.entity_id))
    return found


def _repeat(arguments: dict[str, Any], message: str, lang: str) -> str | None:
    """How often it repeats: what was asked for, else yearly for a birthday."""
    asked = str(arguments.get("repeat") or "").strip().lower()
    if asked in _RRULES:
        return asked
    if asked == "none":
        return None
    text = f"{arguments.get('summary') or ''} {message or ''}".lower()
    if any(rx.search(text) for rx in L.pattern_list("yearly_event", lang)):
        return "yearly"
    return None


async def async_add_event(
    hass,
    arguments: dict[str, Any],
    now: datetime,
    user_message: str = "",
    language: str | None = None,
) -> str:
    """Create the event. Never raises; errors go back to the model."""
    summary = str(arguments.get("summary") or "").strip()
    if not summary:
        return "[Not added: 'summary' must say what the event is.]"
    calendars = _writable_calendars(hass)
    if not calendars:
        return (
            "[Not added: no calendar that accepts new events is available to "
            "you. Tell the user it has to be exposed to Assist first.]"
        )
    wanted = str(arguments.get("calendar") or "").strip().lower()
    if wanted:
        calendars = [c for c in calendars if wanted in c[1].lower() or wanted == c[0]] or calendars
    if len(calendars) > 1:
        names = ", ".join(name for _, name in calendars)
        return f"[Not added: ask which calendar: {names}.]"
    entity_id, name = calendars[0]

    try:
        lang = (language or "en").split("-")[0].lower()
        repeat = _repeat(arguments, user_message, lang)
        day = parse_day(arguments.get("day"), now.date())
        # "Added" for a date gone by, with no year said, is an entry nobody ever
        # sees again. A repeating one comes round again, so its date may be past.
        if day < now.date() and repeat is None:
            return f"[Not added: {day.isoformat()} has already passed. Ask which date they mean.]"
        at = arguments.get("at")
        if at in (None, ""):
            start, end = day, day + timedelta(days=1)
            when = f"{day.strftime('%A')} {day.isoformat()}, all day"
        else:
            start = datetime.combine(day, parse_clock(at), tzinfo=now.tzinfo)
            minutes = int(arguments.get("duration_minutes") or _DEFAULT_MINUTES)
            end = start + timedelta(minutes=minutes)
            when = f"{day.strftime('%A')} {day.isoformat()} at {start.strftime('%H:%M')}"
    except WhenError as exc:
        return f"[Not added: {exc}]"
    except (TypeError, ValueError):
        return "[Not added: 'duration_minutes' must be a number.]"

    # On the entity, not through calendar.create_event: that service cannot take
    # a repeat. This is the same call the service ends in.
    event: dict[str, Any] = {"summary": summary, "dtstart": start, "dtend": end}
    if repeat:
        event["rrule"] = _RRULES[repeat]
    try:
        entity = hass.data["calendar"].get_entity(entity_id)
        if entity is None:
            return f"[Not added: {name} is not available right now.]"
        await entity.async_create_event(**event)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("AI Plugin: adding to %s failed", entity_id)
        return f"[Not added to {name}: {exc}]"
    every = f", {_EVERY[repeat]}" if repeat else ""
    return f"Added to {name}: {summary}, {when}{every}."
