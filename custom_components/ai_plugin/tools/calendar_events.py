"""Adding an event to a calendar by voice, and removing one.

Home Assistant's MCP server can read calendars (get_events) but not write to
them. Only calendars that are exposed to Assist and accept new events are
offered, so this never reaches a calendar the user has not shared with the
assistant.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
import difflib
import logging
import re
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


REMOVE_TOOL_NAME = "remove_calendar_event"
_DELETE_EVENT = 2  # CalendarEntityFeature.DELETE_EVENT
_SEARCH_DAYS = 14
_LISTED = 8
# Words that say it is an event, not which one: "today's appointment".
_GENERIC = frozenset({
    "appointment", "appointments", "event", "events", "meeting", "entry", "date",
    "termin", "termine", "eintrag", "rendez-vous", "the", "my", "a", "an", "der",
    "die", "das", "den", "dem", "mein", "meinen", "le", "la", "les", "mon",
    "today", "tomorrow", "heute", "morgen", "heutigen", "morgigen", "aujourd'hui",
    "demain", "for", "on", "für", "am", "pour",
})

REMOVE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": REMOVE_TOOL_NAME,
        "description": (
            "Remove an event from a calendar: 'cancel the dentist on Friday', "
            "'delete today's appointment'. It removes only an event that matches; "
            "otherwise it lists what is there, so ask the user which one they mean."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What the event is, as the user called it: 'Dentist'. Omit when they named only the day.",
                },
                "day": {
                    "type": "string",
                    "description": "'today', 'tomorrow', a weekday such as 'friday', or YYYY-MM-DD. Omit to look at the next two weeks.",
                },
                "at": {
                    "type": "string",
                    "description": "Start time HH:MM, only when the user named one: 'the one at 10' is '10:00'.",
                },
                "calendar": {
                    "type": "string",
                    "description": "Calendar name, only when the user named one.",
                },
            },
            "required": [],
        },
    },
}


def _narrow(matching: list, asked: str, at: object) -> list:
    """Of several events that match, the one the user singled out: by its start
    time ("the one at 10"), or by its whole name. "Delete the test appointment"
    matched "Test 2" too, and the answer "the one at 10" could not choose
    between them (2026-11-12)."""
    if at not in (None, ""):
        try:
            clock = parse_clock(at)
        except WhenError:
            clock = None
        if clock is not None:
            timed = [
                f for f in matching
                if isinstance(f[2].start, datetime) and f[2].start.time().replace(
                    second=0, microsecond=0
                ) == clock
            ]
            if timed:
                matching = timed
    if len(matching) > 1 and asked:
        whole = re.sub(r"^(?:the|my|der|die|das|den|mein(?:en)?)\s+", "", asked.lower()).strip()
        exact = [f for f in matching if (f[2].summary or "").strip().lower() == whole]
        if len(exact) == 1:
            matching = exact
    return matching


def _deletable_calendars(hass) -> list[tuple[str, str]]:
    """(entity_id, name) of the exposed calendars that can remove events."""
    from ..shortcuts import registry  # noqa: PLC0415

    found = []
    for state in hass.states.async_all("calendar"):
        features = state.attributes.get("supported_features") or 0
        if features & _DELETE_EVENT and registry.is_exposed(hass, state.entity_id):
            found.append((state.entity_id, state.attributes.get("friendly_name") or state.entity_id))
    return found


def _words(text: str) -> set[str]:
    words = (re.sub(r"(?:'s|’s)$", "", w) for w in re.findall(r"[\w'’-]+", text.lower()))
    return {w for w in words if w and w not in _GENERIC}


def _matches(asked: str, summary: str) -> bool:
    """True when what the user said names this event.

    Loose on purpose, because the event is named back before anything goes:
    "the dentist" finds "Dentist Dr. Weber". Speech recognition heard "the
    house viewing" as "the hospital" (2026-11-10), which matches nothing, so
    the model is given the day's events and asks.
    """
    asked_words, words = _words(asked), _words(summary)
    if not asked_words:
        return True
    if asked_words & words:
        return True
    a, s = " ".join(sorted(asked_words)), summary.lower()
    return a in s or difflib.SequenceMatcher(None, a, s).ratio() >= 0.8


def _when(event) -> str:
    start = event.start
    if isinstance(start, datetime):
        return f"{start.strftime('%A')} {start.date().isoformat()} at {start.strftime('%H:%M')}"
    return f"{start.strftime('%A')} {start.isoformat()}, all day"


def _named_by_user(event_summary: str, user_message: str, previous_reply: str) -> bool:
    """True when the user named this event, now or by saying yes to a reply
    that named it.

    The model's own choice is not enough. Asked to cancel "the hospital" (the
    house viewing, misheard), it read the day, picked the house viewing and
    removed it without asking, 4 of 6 times in replay (2026-11-10). A misheard
    name must lead to a question, not to another event going.
    """
    words = _words(event_summary)
    if not words:
        return False
    said = _words(user_message) | _words(previous_reply)
    return bool(words & said)


async def async_remove_event(
    hass,
    arguments: dict[str, Any],
    now: datetime,
    user_message: str | None = None,
    previous_reply: str = "",
) -> str:
    """Remove the one event the user named. Never raises; errors go back to the model.

    ``user_message`` None skips the check that the user named the event (tests
    of the search alone)."""
    calendars = _deletable_calendars(hass)
    if not calendars:
        return (
            "[Not removed: no calendar that allows removing events is available to "
            "you. Tell the user it has to be exposed to Assist first.]"
        )
    wanted = str(arguments.get("calendar") or "").strip().lower()
    if wanted:
        calendars = [c for c in calendars if wanted in c[1].lower() or wanted == c[0]] or calendars
    asked = str(arguments.get("summary") or "").strip()
    try:
        if arguments.get("day") in (None, ""):
            start = now
            end = now + timedelta(days=_SEARCH_DAYS)
            span = "in the next two weeks"
        else:
            day = parse_day(arguments.get("day"), now.date())
            start = datetime.combine(day, time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=1)
            span = f"on {day.strftime('%A')} {day.isoformat()}"
    except WhenError as exc:
        return f"[Not removed: {exc}]"

    found = []  # (entity, calendar name, event)
    for entity_id, name in calendars:
        entity = hass.data["calendar"].get_entity(entity_id)
        if entity is None:
            continue
        try:
            events = await entity.async_get_events(hass, start, end)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("AI Plugin: reading %s failed", entity_id)
            continue
        found.extend((entity, name, ev) for ev in events)
    matching = _narrow(
        [f for f in found if _matches(asked, f[2].summary or "")], asked, arguments.get("at")
    )

    if not matching:
        what = f"nothing called {asked!r}" if asked else "no event"
        if not found:
            return f"[Not removed: {what} {span}. The calendar has no events then. Tell the user.]"
        listed = "; ".join(f"{ev.summary}, {_when(ev)}" for _, _, ev in found[:_LISTED])
        return (
            f"[Not removed: {what} {span}. Events then: {listed}. Ask the user "
            "whether they meant one of these; remove nothing until they say which.]"
        )
    if len(matching) > 1:
        listed = "; ".join(f"{ev.summary}, {_when(ev)}" for _, _, ev in matching[:_LISTED])
        return (
            f"[Not removed: several events match: {listed}. Ask the user which one, "
            "then call again with its time as 'at'.]"
        )

    entity, name, event = matching[0]
    if user_message is not None and not _named_by_user(
        event.summary or "", user_message, previous_reply
    ):
        return (
            f"[Not removed: the user has not named {event.summary!r}. Ask whether "
            f"they mean {event.summary}, {_when(event)}; remove it only after they "
            "say yes.]"
        )
    if not event.uid:
        return f"[Not removed: {event.summary} cannot be removed from {name}.]"
    # A repeating event loses only this time, never the whole series.
    recurrence_id = event.recurrence_id if event.rrule else None
    try:
        await entity.async_delete_event(event.uid, recurrence_id=recurrence_id)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("AI Plugin: removing from %s failed", name)
        return f"[Not removed from {name}: {exc}]"
    only = " (only this time; it still repeats)" if recurrence_id else ""
    return f"Removed from {name}: {event.summary}, {_when(event)}{only}."
