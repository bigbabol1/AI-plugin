"""Point the plugin's Assist pipelines at the STT pass-through, and back.

Only pipelines whose conversation agent is this config entry are touched.
The engine each one used before is remembered, so turning voice
recognition off restores it exactly. A pipeline the user re-pointed by
hand in the meantime is left alone.
"""

from __future__ import annotations

import logging

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def _registry(hass):
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    return er.async_get(hass)


def _pipelines(hass) -> list:
    try:
        from homeassistant.components import assist_pipeline  # noqa: PLC0415

        return list(assist_pipeline.async_get_pipelines(hass))
    except Exception:  # noqa: BLE001 — assist_pipeline not loaded
        return []


def agent_entity_ids(hass, entry) -> set[str]:
    """Conversation entity ids that belong to this config entry."""
    reg = _registry(hass)
    return {
        e.entity_id
        for e in reg.entities.values()
        if e.config_entry_id == entry.entry_id and e.entity_id.startswith("conversation.")
    }


def wrapper_entity_ids(hass, entry) -> dict[str, str]:
    """Pass-through entity id → wrapped engine, from the entity registry."""
    reg = _registry(hass)
    prefix = f"{entry.entry_id}_voice_id_"
    return {
        e.entity_id: e.unique_id[len(prefix):]
        for e in reg.entities.values()
        if e.platform == DOMAIN
        and e.entity_id.startswith("stt.")
        and str(e.unique_id).startswith(prefix)
    }


def engines_to_wrap(hass, entry, manager) -> list[str]:
    """Engines needing a pass-through: remembered originals plus whatever
    our pipelines use now that is not already one of ours."""
    wrappers = wrapper_entity_ids(hass, entry)
    agents = agent_entity_ids(hass, entry)
    engines = set(manager.pipeline_originals.values())
    for p in _pipelines(hass):
        if p.conversation_engine in agents and p.stt_engine and p.stt_engine not in wrappers:
            engines.add(p.stt_engine)
    return sorted(engines)


async def async_switch(hass, entry, manager) -> int:
    """Point our pipelines at the pass-through. Returns how many changed."""
    from homeassistant.components import assist_pipeline  # noqa: PLC0415

    wrappers = wrapper_entity_ids(hass, entry)
    by_engine = {engine: eid for eid, engine in wrappers.items()}
    agents = agent_entity_ids(hass, entry)
    originals = dict(manager.pipeline_originals)
    changed = 0
    for p in _pipelines(hass):
        if p.conversation_engine not in agents or not p.stt_engine:
            continue
        if p.stt_engine in wrappers:
            continue
        target = by_engine.get(p.stt_engine)
        if target is None or hass.states.get(target) is None:
            continue
        originals[p.id] = p.stt_engine
        await assist_pipeline.async_update_pipeline(hass, p, stt_engine=target)
        _LOGGER.info(
            "AI Plugin voice ID: pipeline %r now uses %s (was %s)", p.name, target, p.stt_engine
        )
        changed += 1
    if changed:
        await manager.async_set_pipeline_originals(originals)
    return changed


async def async_restore(hass, entry, manager) -> int:
    """Give every pipeline we switched its old engine back."""
    from homeassistant.components import assist_pipeline  # noqa: PLC0415

    wrappers = wrapper_entity_ids(hass, entry)
    originals = dict(manager.pipeline_originals)
    by_id = {p.id: p for p in _pipelines(hass)}
    restored = 0
    for pipeline_id, engine in originals.items():
        p = by_id.get(pipeline_id)
        if p is None:
            continue
        if p.stt_engine not in wrappers:
            _LOGGER.info(
                "AI Plugin voice ID: pipeline %r was changed by hand; leaving it", p.name
            )
            continue
        await assist_pipeline.async_update_pipeline(hass, p, stt_engine=engine)
        restored += 1
    await manager.async_set_pipeline_originals({})
    return restored


def unwrapped_pipelines(hass, entry) -> list[str]:
    """Names of our pipelines not (yet) using a pass-through."""
    wrappers = wrapper_entity_ids(hass, entry)
    agents = agent_entity_ids(hass, entry)
    return [
        p.name
        for p in _pipelines(hass)
        if p.conversation_engine in agents and p.stt_engine and p.stt_engine not in wrappers
    ]


def remove_wrapper_entities(hass, entry) -> None:
    """Drop the pass-through entities from the registry once unused."""
    reg = _registry(hass)
    for entity_id in wrapper_entity_ids(hass, entry):
        reg.async_remove(entity_id)
