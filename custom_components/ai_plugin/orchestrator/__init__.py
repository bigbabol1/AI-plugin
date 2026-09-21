"""Orchestrator: the core message-processing engine for AI Plugin.

One turn passes through `core.Orchestrator`, which leans on the modules
beside it for the decisions it has to make along the way:

    grounding    did the model actually act, or only promise to?
    online       does this question need the web, or can the house answer?
    toolcalls    which schemas this turn may see; malformed calls coming back
    textproc     narration, emoji, markdown and filler stripped from replies
    delta_gate   sentence-safe streaming for voice
    location     where the house is, coarsely, for search bias
"""

from __future__ import annotations

from .core import Orchestrator
from .location import LocationProvider

__all__ = ["LocationProvider", "Orchestrator"]
