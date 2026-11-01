"""The scheduler as the model sees it. The descriptions are load-bearing: they
are the whole of what a small model knows about which of these to call."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from ..i18n import L
from ..satellite_speech import satellite_in_area
from .engine import KIND_ALARM, KIND_COMMAND, KIND_REMINDER, Scheduler
from .when import REPEATS, WhenError, describe, resolve

_LOGGER = logging.getLogger(__name__)

_WHEN_PROPS: dict[str, Any] = {
    "at": {
        "type": "string",
        "description": "Clock time, 24-hour HH:MM. 'at 7 in the evening' is 19:00.",
    },
    "day": {
        "type": "string",
        "description": (
            "'today', 'tomorrow', a weekday such as 'friday', or YYYY-MM-DD. "
            "OMIT for the next time that clock time comes."
        ),
    },
    "in_minutes": {
        "type": "number",
        "description": "Minutes from now, for 'in 30 minutes' or 'in 2 hours' (120). Instead of at.",
    },
    "repeat": {
        "type": "string",
        "enum": list(REPEATS),
        "description": "Only when the user said so, such as 'every day'. Needs at.",
    },
}

_AREA_PROP = {
    "area": {
        "type": "string",
        "description": "Room to play it in, only when the user named one. OMIT for this room.",
    },
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "set_alarm",
        "Set an alarm for a clock time: 'set an alarm for 6', 'wake me up at 7'. "
        "It rings until someone says stop or snooze. For a length of time "
        "('set a timer for 5 minutes') use start_timer instead. It only speaks: "
        "to wake up to music use schedule_action with the command 'play music'.",
        {
            **_WHEN_PROPS,
            "label": {
                "type": "string",
                "description": (
                    "What to say when it rings, in the user's language, only if "
                    "the user said what the alarm is for: 'for the pasta' is 'The "
                    "pasta is ready'. OMIT for a plain 'set an alarm' or 'wake me up'."
                ),
            },
            **_AREA_PROP,
        },
        [],
    ),
    _tool(
        "set_reminder",
        "Remind the user of something later: 'remind me to take out the trash "
        "tomorrow at 6'. Spoken once here and sent to their phone. For someone "
        "else, 'remind Anna to buy milk', it goes to that person's phone.",
        {
            "text": {
                "type": "string",
                "description": "What to remind them of, without the time or the name: 'buy milk'.",
            },
            "person": {
                "type": "string",
                "description": "Name, only when it is for someone else. OMIT for 'remind me'.",
            },
            **_WHEN_PROPS,
            **_AREA_PROP,
        },
        ["text"],
    ),
    _tool(
        "schedule_action",
        "Do something to the home LATER: 'turn off the lights in 30 minutes', "
        "'at 10 dim the lights'. Never carry it out now and never start a timer for it.",
        {
            "command": {
                "type": "string",
                "description": (
                    "The command alone in the user's words, without the time: "
                    "'turn off the lights'."
                ),
            },
            **_WHEN_PROPS,
        },
        ["command"],
    ),
    _tool(
        "list_scheduled",
        "List alarms, reminders and scheduled actions: 'what alarms do I have', "
        "'what have I scheduled'. Not for timers: use timer_status.",
        {},
        [],
    ),
    _tool(
        "cancel_scheduled",
        "Cancel an alarm, reminder or scheduled action: 'cancel my alarm', "
        "'forget the trash reminder'. If several match, ask which from the result.",
        {
            "kind": {"type": "string", "enum": [KIND_ALARM, KIND_REMINDER, KIND_COMMAND]},
            "text": {"type": "string", "description": "A word from its text, such as 'trash'."},
            "number": {"type": "integer", "description": "Its number from list_scheduled."},
            "all": {"type": "boolean", "description": "Only when the user said all of them."},
        },
        [],
    ),
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in TOOL_SCHEMAS)
_NOT_A_PURPOSE = frozenset({
    "alarm", "the alarm", "wecker", "der wecker", "alarme", "alarma", "budzik", "wekker", "réveil",
})
# Tools that change something, so that a retry never runs them twice.
WRITE_TOOL_NAMES = ("set_alarm", "set_reminder", "schedule_action", "cancel_scheduled")


def names_later_time(message: str, lang: str) -> bool:
    """True when the words put a command in the future: "in 30 minutes"."""
    text = (message or "").lower()
    return any(rx.search(text) for rx in L.pattern_list("later_time", lang))


def asks_for_phrase_routine(message: str, lang: str) -> bool:
    """True for "when I say good night, ...": something that cannot be set yet.
    Left to the model it answered "I have scheduled that" with no tool at all."""
    text = (message or "").strip().lower()
    return any(rx.match(text) for rx in L.pattern_list("phrase_routine", lang))


def strip_later_time(command: str, lang: str) -> str:
    """The command without its time, so running it later cannot re-schedule it."""
    text = str(command or "")
    for rx in L.pattern_list("later_time", lang):
        text = rx.sub("", text)
    return " ".join(text.split()).strip(" ,.;:")


def _spoken_duration_minutes(message: str) -> float | None:
    """The duration the user said, which beats the model's arithmetic: small
    models mis-fill the unit ("in 2 hours" as in_minutes=2)."""
    from ..tools.ha_local.timers import _parse_utterance_duration  # noqa: PLC0415

    parsed = _parse_utterance_duration(message or "")
    if not parsed:
        return None
    return (
        parsed.get("hours", 0) * 60 + parsed.get("minutes", 0) + parsed.get("seconds", 0) / 60
    ) or None


def _alarm_label(raw: object, user_message: str, lang: str) -> str:
    """What an alarm says when it rings, spoken every 10 seconds, so not left
    to the model alone. In replay it turned "wake me up" into "Wake me up" and
    a plain alarm into "Alarm in 30 seconds"."""
    message = (user_message or "").lower()
    if any(rx.search(message) for rx in L.pattern_list("alarm_wake_request", lang)):
        return L.template("alarm_wake", lang)
    from ..tools.ha_local.timers import _parse_utterance_duration  # noqa: PLC0415

    label = str(raw or "").strip()
    if (
        label.strip(" .!").lower() in _NOT_A_PURPOSE
        or names_later_time(label, lang)
        or _parse_utterance_duration(label)
    ):
        return ""
    return label


def _line(number: int, job, now) -> str:
    kind = {KIND_COMMAND: "scheduled action"}.get(job.kind, job.kind)
    text = f": {job.text}" if job.text else ""
    repeat = f", repeats {job.repeat}" if job.repeat else ""
    return f"{number}. {kind} {describe(job.due.astimezone(now.tzinfo), now)}{repeat}{text}"


def _listing(jobs, now) -> str:
    return "\n".join(_line(i, job, now) for i, job in enumerate(jobs, 1))


async def async_call_tool(
    scheduler: Scheduler,
    hass,
    name: str,
    arguments: dict[str, Any],
    *,
    device_id: str | None,
    user_id: str | None,
    person_id: str | None,
    language: str | None,
    user_message: str = "",
    resolve_person: Callable[[str], str | None] | None = None,
) -> str:
    """Run one scheduler tool. Never raises; errors go back to the model."""
    try:
        if name == "list_scheduled":
            jobs = scheduler.jobs()
            return _listing(jobs, scheduler.now()) if jobs else "Nothing is scheduled."
        if name == "cancel_scheduled":
            return await _async_cancel(scheduler, arguments)
        return await _async_add(
            scheduler, hass, name, arguments, device_id, user_id, person_id,
            language or "en", user_message, resolve_person,
        )
    except WhenError as exc:
        return f"[Not set: {exc}]"
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("AI Plugin scheduler tool %s failed", name)
        return f"[{name} failed: {exc}]"


async def _async_add(
    scheduler, hass, name, arguments, device_id, user_id, person_id, language, user_message,
    resolve_person,
) -> str:
    lang = language.split("-")[0].lower()
    # A reminder for someone else goes to them, not to whoever is speaking.
    for_name = str(arguments.get("person") or "").strip() if name == "set_reminder" else ""
    someone_else = False
    if for_name:
        found = resolve_person(for_name) if resolve_person is not None else None
        if found is None:
            return f"[Not set: I do not know who {for_name} is. Ask who it is for.]"
        someone_else = found != person_id
        person_id = found
    in_minutes = arguments.get("in_minutes")
    if in_minutes not in (None, ""):
        in_minutes = _spoken_duration_minutes(user_message) or in_minutes
    now = scheduler.now()
    repeat = arguments.get("repeat") or None
    due = resolve(
        now, at=arguments.get("at"), day=arguments.get("day"),
        in_minutes=in_minutes, repeat=repeat,
    )
    when = describe(due, now)
    fields: dict[str, Any] = {
        "user_id": user_id,
        "person_id": person_id,
        "language": language,
        "repeat": repeat,
        "clock": due.strftime("%H:%M") if repeat else None,
    }
    if name == "schedule_action":
        command = strip_later_time(arguments.get("command") or "", lang)
        if not command:
            return "[Not scheduled: 'command' must say what to do, without the time.]"
        await scheduler.async_add(KIND_COMMAND, due, text=command, device_id=device_id, **fields)
        return f"Scheduled for {when}: {command}."

    target = device_id
    room = str(arguments.get("area") or "").strip()
    if room:
        target = satellite_in_area(hass, room)
        if target is None:
            return f"[Not set: there is no satellite in {room}. Tell the user.]"
    if name == "set_alarm":
        if not target:
            return "[Not set: an alarm has to ring on a satellite. Ask which room.]"
        label = _alarm_label(arguments.get("label"), user_message, lang)
        await scheduler.async_add(KIND_ALARM, due, text=label, device_id=target, **fields)
        return f"Alarm set for {when}."

    text = str(arguments.get("text") or "").strip()
    if not text:
        return "[Not set: 'text' must say what to remind them of.]"
    if someone_else and not room:
        from ..shortcuts.phone import phones_of  # noqa: PLC0415

        # Their phone, not this room. Without a phone, this room is all there is.
        if phones_of(hass, person_id):
            target = None
    if not target and not person_id:
        return "[Not set: nowhere to deliver it. Ask which room.]"
    await scheduler.async_add(KIND_REMINDER, due, text=text, device_id=target, **fields)
    for_whom = f" for {for_name}" if someone_else else ""
    return f"Reminder set{for_whom} for {when}: {text}."


async def _async_cancel(scheduler, arguments) -> str:
    jobs = scheduler.jobs()
    now = scheduler.now()
    number = arguments.get("number")
    if number not in (None, ""):
        try:
            chosen = [jobs[int(number) - 1]]
        except (ValueError, TypeError, IndexError):
            return f"[There is no number {number}. Scheduled:\n{_listing(jobs, now)}]"
    else:
        kind = arguments.get("kind")
        word = str(arguments.get("text") or "").strip().lower()
        chosen = [
            j for j in jobs
            if (not kind or j.kind == kind) and (not word or word in j.text.lower())
        ]
        if not chosen:
            return "Nothing matching is scheduled."
        if len(chosen) > 1 and arguments.get("all") is not True:
            return f"Several match; ask which:\n{_listing(chosen, now)}"
    await scheduler.async_cancel(chosen)
    return "Cancelled:\n" + _listing(chosen, now)
