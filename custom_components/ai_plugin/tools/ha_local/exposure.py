"""Optional Home Assistant APIs this tool set degrades without.

Both imports are version-dependent, and both fail towards doing MORE
rather than less: with no exposure API every entity counts as exposed,
and with no timer constant the well-known key is used. A tool that
silently stopped seeing the house would be worse than one that sees a
diagnostic sensor it should not.
"""

from __future__ import annotations


try:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose as _ha_should_expose,
    )
    _EXPOSURE_API_AVAILABLE = True
except ImportError:
    _ha_should_expose = None  # type: ignore[assignment]
    _EXPOSURE_API_AVAILABLE = False

try:
    from homeassistant.components.intent.const import TIMER_DATA as _TIMER_DATA_KEY
except ImportError:
    _TIMER_DATA_KEY = "intent.timer"  # type: ignore[assignment]
