"""Turning registry objects into text a small model can read.

Responses are capped so a discovery call cannot eat the context window;
an overflow says how many entities it hid, so the model can narrow the
filter and ask again instead of guessing from a truncated list.
"""

from __future__ import annotations



_MAX_RESPONSE_CHARS = 1500


# Attributes worth surfacing to the LLM for common entity types.
_INTERESTING_ATTRS = (
    "brightness",
    "color_mode",
    "rgb_color",
    "color_temp",
    "color_temp_kelvin",
    "current_temperature",
    "temperature",
    "target_temp_low",
    "target_temp_high",
    "current_position",
    "percentage",
    "media_title",
    "media_artist",
    "volume_level",
    "hvac_action",
    "preset_mode",
    "battery_level",
    "humidity",
    "unit_of_measurement",
)



def _s(val: object) -> str:
    """Coerce anything (including HA's ComputedNameType) to a plain str."""
    if val is None:
        return ""
    return val if isinstance(val, str) else str(val)



def _cap(lines: list[str], header: str = "") -> str:
    """Join lines and cap to _MAX_RESPONSE_CHARS, appending a truncation note."""
    out = header
    kept: list[str] = []
    for i, line in enumerate(lines):
        candidate = out + "\n".join(kept + [line])
        if len(candidate) > _MAX_RESPONSE_CHARS and kept:
            remaining = len(lines) - i
            kept.append(f"…truncated, {remaining} more — narrow the filter")
            break
        kept.append(line)
    return (header + "\n".join(kept)).strip() or "(empty)"
