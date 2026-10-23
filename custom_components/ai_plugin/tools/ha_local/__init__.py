"""Local HA tools: discovery and action, executed in-process.

Small LLMs drown in a dumped YAML inventory of the house, so nothing is
dumped: each tool answers one question against the registries when the
model asks it.

    schemas      what the model sees, and may call
    core         the registry object and the call_tool dispatch
    entities     list_areas / list_entities / get_entity / search_entities
    lights       set_area_state / set_brightness
    media        play_music / media_command / media_status
    timers       start / cancel / pause / resume / extend / status
    formatting   registry objects into capped, readable text
    exposure     the optional HA APIs this all degrades without
"""

from __future__ import annotations

from .core import HALocalToolRegistry
from .exposure import _TIMER_DATA_KEY
from .lights import _BRIGHTNESS_FLOOR_PCT, _BRIGHTNESS_STEP_PP
from .schemas import TOOL_NAMES, TOOL_SCHEMAS

__all__ = [
    "HALocalToolRegistry",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "_BRIGHTNESS_FLOOR_PCT",
    "_BRIGHTNESS_STEP_PP",
    "_TIMER_DATA_KEY",
]
