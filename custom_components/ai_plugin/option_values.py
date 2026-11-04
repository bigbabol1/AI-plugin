"""Merging what an options form returned into the saved options."""

from __future__ import annotations

from typing import Any

from .const import (
    CONF_FEEDBACK_LOOP_DEVICES,
    CONF_KEEP_ALIVE,
    CONF_LOCATION_ENTITY,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
)

# Advanced fields with no default in the form, so clearing one leaves it out
# of what the form returns.
CLEARABLE_ADVANCED_OPTIONS = (
    CONF_FEEDBACK_LOOP_DEVICES,
    CONF_KEEP_ALIVE,
    CONF_LOCATION_ENTITY,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
)


def merge_advanced_options(
    options: dict[str, Any], user_input: dict[str, Any]
) -> dict[str, Any]:
    """Apply the form's values, treating a cleared field as cleared.

    A plain update keeps the old value of a field the form left out, so
    clearing temperature in the form silently changed nothing (2026-11-03).
    """
    merged = {
        key: value for key, value in options.items()
        if key not in CLEARABLE_ADVANCED_OPTIONS or key in user_input
    }
    merged.update(user_input)
    return merged
