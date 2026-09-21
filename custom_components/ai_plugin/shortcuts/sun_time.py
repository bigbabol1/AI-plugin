"""Clock and daylight answers, read straight from Home Assistant.

Both questions are deterministic facts the model answers correctly but
slowly (5-8 s per turn), and both are asked constantly.
"""

from __future__ import annotations

import logging
import re

from homeassistant.core import HomeAssistant

from ..i18n import L

_LOGGER = logging.getLogger(__name__)

# "in <word>" after a time/sun question names a PLACE ("what time is it
# in Tokyo") — the local clock/sun cannot answer that; the LLM (with
# web_search) can. Sensor questions legitimately use "in <area>", so this
# guard applies only inside the time and sun shortcuts.
_PLACE_TAIL_RE = re.compile(r"\bin\s+\w", re.IGNORECASE)

def _try_sun_shortcut(hass: HomeAssistant, message: str, lang: str = "en") -> str | None:
    """Deterministic reply for sun/daylight questions, in the user's language.

    Reads ``sun.sun`` directly. Bypasses the LLM which often refuses these
    queries. ``lang`` selects the keyword regex set and the response
    template; English is the universal fallback.
    """
    msg_lower = (message or "").lower()
    sun_set_re = L.keyword_re("sun_set", lang)
    sun_rise_re = L.keyword_re("sun_rise", lang)
    sun_dark_re = L.keyword_re("sun_dark", lang)
    sun_is_up_re = L.keyword_re("sun_is_up", lang)

    matched = (
        (sun_set_re and sun_set_re.search(msg_lower))
        or (sun_rise_re and sun_rise_re.search(msg_lower))
        or (sun_dark_re and sun_dark_re.search(msg_lower))
        or (sun_is_up_re and sun_is_up_re.search(msg_lower))
    )
    if not matched:
        return None
    if _PLACE_TAIL_RE.search(msg_lower):
        return None

    state = hass.states.get("sun.sun")
    if state is None:
        return None

    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz_name = (getattr(hass.config, "time_zone", None) or "").strip()
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:  # noqa: BLE001
        tz = None

    def _fmt(iso: str | None) -> str | None:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if tz is not None:
                dt = dt.astimezone(tz)
            return dt.strftime("%H:%M")
        except Exception:  # noqa: BLE001
            return None

    attrs = state.attributes or {}
    next_setting = _fmt(attrs.get("next_setting"))
    next_rising = _fmt(attrs.get("next_rising"))
    is_up = state.state == "above_horizon"

    # Boolean queries first so 'dark' / 'fait-il nuit' don't shadow the
    # sunset time branch.
    if (sun_dark_re and sun_dark_re.search(msg_lower)) or (
        sun_is_up_re and sun_is_up_re.search(msg_lower)
    ):
        if is_up and next_setting:
            return L.template("sun_is_up", lang, time=next_setting)
        if not is_up and next_rising:
            return L.template("sun_is_down", lang, time=next_rising)

    if sun_set_re and sun_set_re.search(msg_lower) and next_setting:
        _LOGGER.info("AI Plugin shortcut hit: sunset → %s (lang=%s)", next_setting, lang)
        return L.template("sun_set_at", lang, time=next_setting)
    if sun_rise_re and sun_rise_re.search(msg_lower) and next_rising:
        _LOGGER.info("AI Plugin shortcut hit: sunrise → %s (lang=%s)", next_rising, lang)
        return L.template("sun_rise_at", lang, time=next_rising)

    return None

def _try_time_shortcut(hass: HomeAssistant, message: str, lang: str) -> str | None:
    """Answer 'what time is it' from the HA clock — no LLM round-trip.

    The LLM answers this correctly from the [CURRENT TIME] block, but at
    5-8s per turn for a deterministic fact. Date questions stay with the
    LLM (localized weekday/month names aren't worth the i18n surface).
    """
    time_re = L.keyword_re("time_now", lang)
    if time_re is None or not time_re.search(message.lower()):
        return None
    if _PLACE_TAIL_RE.search(message.lower()):
        return None
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz_name = (getattr(hass.config, "time_zone", None) or "").strip()
        now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
        stamp = now.strftime("%H:%M")
    except Exception:  # noqa: BLE001
        _LOGGER.debug("AI Plugin time shortcut: clock read failed", exc_info=True)
        return None
    _LOGGER.info("AI Plugin shortcut hit: time → %s (lang=%s)", stamp, lang)
    return L.template("time_is", lang, time=stamp)
