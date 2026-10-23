"""Timers, and why they do not all go through HA's intents.

Announce mode starts every timer with a conversation_command, and HA's
own timer intents skip such timers when looking one up — so status,
cancel, pause, resume and add/remove time work on the timer manager
directly, by id. Only start_timer still goes through the intent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import Context
from homeassistant.helpers import (
    intent,
)

from .exposure import _TIMER_DATA_KEY

from .formatting import _s

_LOGGER = logging.getLogger(__name__)


_TIMER_INTENTS = {
    "start_timer": "HassStartTimer",
    "cancel_timer": "HassCancelTimer",
    "pause_timer": "HassPauseTimer",
    "unpause_timer": "HassUnpauseTimer",
    "increase_timer": "HassIncreaseTimer",
    "decrease_timer": "HassDecreaseTimer",
    "timer_status": "HassTimerStatus",
}


_DURATION_UNIT_RE = re.compile(
    r"(\d+)\s*"
    r"(hours?|hrs?|stunden?|minutes?|mins?|minuten?|seconds?|secs?|sekunden?)",
    re.IGNORECASE,
)


# One CONTIGUOUS duration expression: "1 minute 30 seconds", "1 Stunde und
# 15 Minuten". Contiguity is the disambiguator — in "add 5 minutes to the
# 10 minute timer" the two numbers belong to different things (delta vs.
# the timer's NAME) and must never be combined.
_DURATION_PHRASE_RE = re.compile(
    r"\d+\s*(?:hours?|hrs?|stunden?|minutes?|mins?|minuten?|seconds?|secs?|sekunden?)"
    r"(?:\s*(?:and|und)?\s*\d+\s*"
    r"(?:hours?|hrs?|stunden?|minutes?|mins?|minuten?|seconds?|secs?|sekunden?))*",
    re.IGNORECASE,
)



def _parse_utterance_duration(text: str) -> dict[str, int] | None:
    """Extract an explicit h/m/s duration from a spoken utterance (EN + DE).

    Returns any of {'hours','minutes','seconds'}, else None. Used to
    override model slot-filling, which sometimes assigns the wrong unit
    (e.g. '10 seconds' → minutes=10).

    Only trusts the utterance when it contains exactly ONE contiguous
    duration phrase. "Add 5 minutes to the 10 minute timer" has two — the
    second names the timer — and summing them (15) armed the wrong
    duration; ambiguity keeps the model's slots instead.
    """
    if not text:
        return None
    phrases = _DURATION_PHRASE_RE.findall(text)
    if len(phrases) != 1:
        return None
    out: dict[str, int] = {}
    for num, unit in _DURATION_UNIT_RE.findall(phrases[0]):
        u = unit.lower()
        if u.startswith(("hour", "hr", "stunde")):
            key = "hours"
        elif u.startswith("min"):
            key = "minutes"
        else:
            key = "seconds"
        try:
            out[key] = out.get(key, 0) + int(num)
        except ValueError:
            continue
    return out or None



_TIMER_WORD_SUFFIX_RE = re.compile(r"[\s\-]*(?:timer|wecker|countdown)$")



def _timer_name_key(name: object) -> str:
    """Comparable timer name: 'Nudel-Timer', 'nudel timer', 'Nudel' → 'nudel'.

    A bare 'timer' reduces to '' — the model passes it for "the timer",
    and it then means no name at all.
    """
    return _TIMER_WORD_SUFFIX_RE.sub("", _s(name).strip().casefold()).strip()



def _fmt_duration(total: int) -> str:
    """Seconds → '1 h 5 min 3 s' (zero parts dropped)."""
    hours, rest = divmod(max(0, int(total)), 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [f"{hours} h"] if hours else []
    if minutes:
        parts.append(f"{minutes} min")
    if seconds or not parts:
        parts.append(f"{seconds} s")
    return " ".join(parts)



def _describe_timer(timer: Any) -> str:
    """One status line for the model: name, time left, set duration, paused."""
    label = f"timer {timer.name!r}" if timer.name else "unnamed timer"
    line = f"{label}: {_fmt_duration(timer.seconds_left)} left"
    set_for = (
        3600 * (timer.start_hours or 0)
        + 60 * (timer.start_minutes or 0)
        + (timer.start_seconds or 0)
    )
    if set_for:
        line += f" (set for {_fmt_duration(set_for)})"
    if not timer.is_active:
        line += ", paused"
    return line


class TimerToolsMixin:
    """start / cancel / pause / resume / extend / status."""

    async def _call_timer_intent(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        device_id: str | None = None,
        language: str | None = None,
        user_message: str = "",
        announce_agent_id: str | None = None,
    ) -> str:
        """Start a voice timer via HA's intent, or manage an existing one.

        Only ``start_timer`` goes through HA's intent. It is bound to the
        calling assist_satellite via ``device_id``; without one HA cannot ring
        the timer on the right satellite, so we refuse early with a useful
        message rather than letting the intent return an opaque error.
        Every other timer tool is handled by ``_manage_timer``.
        """
        intent_type = _TIMER_INTENTS.get(tool_name)
        if intent_type is None:
            return f"[Unknown timer tool: {tool_name!r}]"

        if tool_name == "start_timer" and not device_id:
            return (
                "[Timer requires a satellite device — this tool is only available "
                "when the request comes from a voice pipeline. Try again from the "
                "voice satellite.]"
            )

        slots: dict[str, Any] = {}
        for key in ("name", "hours", "minutes", "seconds"):
            if key not in arguments:
                continue
            val = arguments[key]
            if val is None:
                continue
            if key == "name":
                name_val = _s(val).strip()
                if name_val:
                    slots["name"] = {"value": name_val}
            else:
                try:
                    int_val = int(val)
                except (TypeError, ValueError):
                    continue
                slots[key] = {"value": int_val}

        # Deterministic override: the spoken utterance is ground truth for the
        # duration. Small models sometimes mis-fill the unit (e.g. "10 seconds"
        # → minutes=10), so when the utterance carries an explicit duration we
        # replace the model's h/m/s slots with the parsed value (name is kept).
        if tool_name in ("start_timer", "increase_timer", "decrease_timer"):
            parsed = _parse_utterance_duration(user_message)
            if parsed:
                for k in ("hours", "minutes", "seconds"):
                    slots.pop(k, None)
                for k, v in parsed.items():
                    slots[k] = {"value": v}
                _LOGGER.debug(
                    "AI Plugin timer: duration from utterance %r → %s "
                    "(overrode model slots)",
                    (user_message or "")[:80], parsed,
                )

        if tool_name != "start_timer":
            return self._manage_timer(tool_name, slots, device_id)

        if tool_name == "start_timer":
            has_duration = any(k in slots for k in ("hours", "minutes", "seconds"))
            if not has_duration:
                return (
                    "[start_timer needs a duration — provide at least one of "
                    "hours / minutes / seconds.]"
                )
            # Timer-announce mode: instead of ringing the satellite at
            # expiry, HA's TimerManager calls this agent back with the
            # sentinel command and the plugin plays the announcement via
            # mic_to_mediaplayer / assist_satellite.announce. The device
            # is never notified (no on-device LEDs/ring), and the
            # "device supports timers" requirement is bypassed.
            if announce_agent_id:
                from ...const import TIMER_DONE_SENTINEL  # noqa: PLC0415

                timer_name = (slots.get("name") or {}).get("value", "")
                slots["conversation_command"] = {
                    "value": f"{TIMER_DONE_SENTINEL} {timer_name}".strip()
                }

        handle_kwargs: dict[str, Any] = {}
        if announce_agent_id and tool_name == "start_timer":
            handle_kwargs["conversation_agent_id"] = announce_agent_id
        try:
            try:
                response = await intent.async_handle(
                    self._hass,
                    "ai_plugin",
                    intent_type,
                    slots=slots,
                    text_input=None,
                    context=Context(),
                    language=language or self._hass.config.language,
                    assistant="conversation",
                    device_id=device_id,
                    **handle_kwargs,
                )
            except TypeError:
                # Older HA cores without the conversation_agent_id kwarg —
                # the command then runs on the default agent instead.
                if not handle_kwargs:
                    raise
                response = await intent.async_handle(
                    self._hass,
                    "ai_plugin",
                    intent_type,
                    slots=slots,
                    text_input=None,
                    context=Context(),
                    language=language or self._hass.config.language,
                    assistant="conversation",
                    device_id=device_id,
                )
        except intent.IntentHandleError as exc:
            return f"[{intent_type} failed: {exc}]"
        except intent.UnknownIntent:
            return (
                f"[{intent_type} is not registered in this Home Assistant version. "
                "Voice timers require HA 2024.7 or newer.]"
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin timer intent %s failed", intent_type)
            return f"[{intent_type} error: {exc}]"

        speech = ""
        try:
            speech_block = response.speech.get("plain", {}) if response.speech else {}
            speech = _s(speech_block.get("speech", "")).strip()
        except Exception:  # noqa: BLE001
            pass
        return speech or f"OK — {intent_type} dispatched."

    def _manage_timer(
        self, tool_name: str, slots: dict[str, Any], device_id: str | None
    ) -> str:
        """Status / cancel / pause / unpause / add / remove on HA's TimerManager.

        HA's own timer intents cannot see the timers this plugin starts: their
        lookup (``_find_timer`` / ``_find_timers`` in intent/timers.py) skips
        every timer carrying a ``conversation_command``, and announce mode
        plants one on each. HassTimerStatus answered "no timers" and
        HassCancelTimer "not found" while the timer was running. So the timers
        are read from the manager and changed by id — which also returns the
        remaining time as text, where the intent's response carries no speech.

        Other integrations' delayed-command timers ("turn off the lights in
        10 minutes") stay excluded, as they are in HA.
        """
        from ...const import TIMER_DONE_SENTINEL  # noqa: PLC0415

        manager = self._hass.data.get(_TIMER_DATA_KEY)
        if manager is None:
            return "[Voice timers are unavailable — HA's intent integration is not loaded.]"

        timers = [
            t for t in list(manager.timers.values())
            if not t.conversation_command
            or t.conversation_command.startswith(TIMER_DONE_SENTINEL)
        ]
        # The calling satellite's own timers first.
        timers.sort(key=lambda t: t.device_id != device_id)
        if not timers:
            return "No timers are running."

        name = _s((slots.get("name") or {}).get("value")).strip()
        name_norm = _timer_name_key(name)

        def named(pool: list[Any]) -> list[Any]:
            return [t for t in pool if _timer_name_key(t.name) == name_norm]

        if tool_name == "timer_status":
            shown = (named(timers) if name_norm else []) or timers
            head = (
                f"No timer is named {name!r}; all timers:\n"
                if name_norm and not named(timers) else ""
            )
            return head + "\n".join(_describe_timer(t) for t in shown)

        if tool_name == "pause_timer":
            pool = [t for t in timers if t.is_active]
        elif tool_name == "unpause_timer":
            pool = [t for t in timers if not t.is_active]
        else:
            pool = timers

        matches = named(pool) if name_norm else pool
        if name_norm and not matches and len(pool) == 1:
            # "cancel the timer" with one running: the model often passes
            # a generic word as the name.
            matches = pool
        if len(matches) > 1 and device_id:
            own = [t for t in matches if t.device_id == device_id]
            if len(own) == 1:
                matches = own
        listing = "; ".join(_describe_timer(t) for t in timers)
        if not matches:
            return f"[{tool_name}: no matching timer. Timers: {listing}]"
        if len(matches) > 1:
            return (
                f"[{tool_name}: several timers match — ask the user which one. "
                f"Timers: {listing}]"
            )

        timer = matches[0]
        label = f"timer {timer.name!r}" if timer.name else "the timer"
        try:
            if tool_name == "cancel_timer":
                left = timer.seconds_left
                manager.cancel_timer(timer.id)
                return f"Cancelled {label} ({_fmt_duration(left)} were left)."
            if tool_name == "pause_timer":
                manager.pause_timer(timer.id)
                return f"Paused {label} with {_fmt_duration(timer.seconds_left)} left."
            if tool_name == "unpause_timer":
                manager.unpause_timer(timer.id)
                return f"Resumed {label}, {_fmt_duration(timer.seconds_left)} left."

            delta = (
                3600 * int((slots.get("hours") or {}).get("value", 0))
                + 60 * int((slots.get("minutes") or {}).get("value", 0))
                + int((slots.get("seconds") or {}).get("value", 0))
            )
            if delta <= 0:
                return (
                    f"[{tool_name} needs a duration — provide at least one of "
                    "hours / minutes / seconds.]"
                )
            if tool_name == "increase_timer":
                manager.add_time(timer.id, delta)
                verb = "Added"
            else:
                manager.remove_time(timer.id, delta)
                verb = "Removed"
            return (
                f"{verb} {_fmt_duration(delta)}; {label} now has "
                f"{_fmt_duration(timer.seconds_left)} left."
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin timer %s failed", tool_name)
            return f"[{tool_name} error: {exc}]"
