"""Deterministic pre-LLM shortcuts.

Small LLMs are unreliable at the most common turns: they resolve a
descriptive sensor query against the registry and give up, they answer a
playback command with a confident sentence and no tool call, and they
promise to turn every light off without calling anything. Each module
here answers one such family from Home Assistant's own state, in ~0.01
to 0.05 s, and returns None the moment the match is not certain — a miss
falls through to the LLM rather than guessing.

    registry   area and entity lookup, and the exposure rule they share
    sensors    "<attribute> in <area>" readings, and `try_shortcut`
    sun_time   clock and daylight
    media      playback, volume, mute
    actions    one named device on/off, open/close
    sweep      every light/fan/socket in a room or the whole home
    classify   is a short turn a real command, or our own TTS tail

The names below are the surface other modules import; everything else is
internal to one file.
"""

from __future__ import annotations

from .actions import async_try_action_shortcut
from .classify import looks_like_command
from .media import async_try_media_shortcut
from .sensors import try_shortcut
from .sweep import async_try_domain_sweep_shortcut

__all__ = [
    "async_try_action_shortcut",
    "async_try_domain_sweep_shortcut",
    "async_try_media_shortcut",
    "looks_like_command",
    "try_shortcut",
]
