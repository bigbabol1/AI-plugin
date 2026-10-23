"""Tool schemas going out, and malformed tool calls coming back.

Two directions of the same boundary: which schemas a turn is allowed to
see (pruning keeps a small model's attention on the few tools the
question could need), and how to recover a tool call a model emitted as
text instead of as a structured call.
"""

from __future__ import annotations

import logging
import re

from .grounding import _ACTUATOR_TOOL_NAMES, _is_state_set_query

_LOGGER = logging.getLogger(__name__)

def _readonly_schemas(schemas: list[dict]) -> list[dict]:
    """Schemas minus actuators — for grounding retries that must not act."""
    return [
        s for s in schemas
        if s.get("function", {}).get("name") not in _ACTUATOR_TOOL_NAMES
    ]

# Detect when a model emits a tool call as plain text ("content") instead of
# using the API's structured tool_calls field. Matches first call in text.
# Accepts optional "CALL " prefix and optional trailing period.
_RAW_TOOL_CALL_RE = re.compile(
    r"(?:^|\n)\s*(?:CALL\s+)?([a-zA-Z_][\w]*)\s*\(([^()\n]*)\)",
    re.IGNORECASE,
)

def _parse_raw_tool_call(text: str, valid_names: set[str]) -> tuple[str, dict] | None:
    """Parse 'tool_name(kwargs)' pseudo-syntax out of a model reply.

    Returns (name, args_dict) on success, None otherwise. Handles:
    - list_entities(domain="light", state="on")
    - CALL web_search('price of bitcoin')  (positional → first arg name)
    - list_entities()
    Only matches if the parsed tool name is in valid_names.
    """
    if not text:
        return None
    m = _RAW_TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    if name not in valid_names:
        return None
    arg_str = m.group(2).strip()
    if not arg_str:
        return (name, {})
    args: dict = {}
    # kwargs: key="value" | key='value' | key=bareword
    kw_pairs = re.findall(
        r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,\s][^,]*?)\s*(?:,|$)",
        arg_str,
    )
    if kw_pairs:
        for k, v in kw_pairs:
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]
            args[k] = v
        return (name, args)
    # Positional single value: map to "query" for web_search, else "name".
    val = arg_str
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        val = val[1:-1]
    if name == "web_search":
        return (name, {"query": val})
    return (name, {"name": val})

# Domain/device words that suggest the user is asking about a specific entity.
_ENTITY_WORDS = frozenset(
    "light lights switch switches lamp lamps bulb bulbs sensor sensors "
    "thermostat thermostats blinds cover covers lock locks fan fans "
    "vacuum vacuums plug plugs media speaker tv television climate".split()
)

# Words that suggest an area-level question.
_AREA_WORDS = frozenset("room rooms area areas house zones zone".split())

# Words that suggest a domain sweep ("list all lights", "show switches").
_SWEEP_WORDS = frozenset("list show which all every what".split())

# Pronouns that reference a prior action — keep all schemas so model is flexible.
_PRONOUN_RE = re.compile(r"\b(it|that|them|those|there|here)\b", re.IGNORECASE)

def _prune_ha_local_schemas(
    user_message: str, schemas: list[dict]
) -> list[dict]:
    """Drop ha_local tool schemas that are unlikely to be useful for this turn.

    Heuristic-only — cheap, never perfect. When in doubt, keep the schema.
    Minimum 2 schemas always returned so the model always has search + get.
    """
    if len(schemas) <= 2:
        return schemas

    text = user_message.lower()
    words = set(re.findall(r"[a-z_]+", text))

    # Pronouns → keep everything; prior-turn entity is in [LAST ACTION].
    if _PRONOUN_RE.search(text):
        return schemas

    # State-set questions ("any lights on?", "are any lights turned on?")
    # must retain list_entities — that's the only tool that can answer them.
    # Without this check the entity-word heuristic below drops list_entities
    # for short-form queries whose trigger word ("any", "are") isn't in the
    # sweep set, leaving the model tool-less and producing empty replies.
    if _is_state_set_query(user_message):
        return schemas

    has_entity_word = bool(words & _ENTITY_WORDS)
    # entity_id pattern like "light.something"
    has_entity_id = bool(re.search(r"[a-z_]+\.[a-z0-9_]+", text))
    has_area_word = bool(words & _AREA_WORDS)
    has_sweep = bool(words & _SWEEP_WORDS)

    names_to_drop: set[str] = set()

    # If message is clearly about a specific entity (no area question, no sweep):
    if (has_entity_word or has_entity_id) and not has_area_word and not has_sweep:
        names_to_drop.add("list_areas")
        names_to_drop.add("list_entities")

    if not names_to_drop:
        return schemas

    pruned = [s for s in schemas if s.get("function", {}).get("name") not in names_to_drop]
    # Safety: always keep at least 2 ha_local schemas (search_entities + get_entity).
    if len(pruned) < 2:
        return schemas

    if names_to_drop:
        _LOGGER.debug(
            "AI Plugin: pruned ha_local schemas %s for message %r",
            names_to_drop,
            user_message[:60],
        )
    return pruned
