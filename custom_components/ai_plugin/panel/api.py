"""Admin-only websocket commands behind the sidebar page.

Every command takes an optional ``entry_id``; without one the first loaded
AI Plugin entry is used. Saving settings updates the entry's options, which
reloads the entry exactly as saving the options form does.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector

from ..const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_OFFLINE_FALLBACK,
    CONF_SEARXNG_URL,
    DEFAULT_BASE_URL,
    DEFAULT_OFFLINE_FALLBACK,
    DOMAIN,
    FEATURE_TOOLS,
)
from .logic import (
    SECRET_FIELDS,
    clean_values,
    form_fields,
    mask_servers,
    memory_owner,
    merge_servers,
    edit_facts,
    preset_server,
    valid_memory_key,
)

_LOGGER = logging.getLogger(__name__)

_HEALTH_TTL_S = 30.0
_HISTORY = 40
_health_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# Last checks per entry, for the latency line on the status tab.
_health_history: dict[str, list[dict[str, Any]]] = {}

_ENTRY = {vol.Optional("entry_id"): str}


# ── helpers ──────────────────────────────────────────────────────────────────


def _loaded(hass: HomeAssistant, entry_id: str | None):
    """(entry, its runtime data) or (None, None)."""
    data = hass.data.get(DOMAIN, {})
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry_id and entry.entry_id != entry_id:
            continue
        if isinstance(data.get(entry.entry_id), dict):
            return entry, data[entry.entry_id]
    return None, None


def _need_entry(hass: HomeAssistant, connection, msg):
    entry, data = _loaded(hass, msg.get("entry_id"))
    if entry is None:
        connection.send_error(msg["id"], "not_loaded", "AI Plugin is not loaded.")
    return entry, data


def _device_name(hass: HomeAssistant, device_id: str | None) -> str | None:
    if not device_id:
        return None
    from homeassistant.helpers import device_registry as dr  # noqa: PLC0415

    dev = dr.async_get(hass).async_get(device_id)
    if dev is None:
        return None
    return dev.name_by_user or dev.name


async def _labels(hass: HomeAssistant, language: str | None) -> dict[str, str]:
    from homeassistant.helpers.translation import async_get_translations  # noqa: PLC0415

    lang = language or hass.config.language or "en"
    found = await async_get_translations(hass, lang, "options", {DOMAIN})
    if lang != "en":
        base = await async_get_translations(hass, "en", "options", {DOMAIN})
        found = {**base, **found}
    return found


def _step_text(labels: dict[str, str], steps: list[str]) -> tuple[dict, dict, str, str]:
    """(field labels, field help, title, description) gathered from steps."""
    names: dict[str, str] = {}
    helps: dict[str, str] = {}
    title = description = ""
    prefix = f"component.{DOMAIN}.options.step."
    for step in steps:
        for key, text in labels.items():
            if not key.startswith(prefix + step + "."):
                continue
            rest = key[len(prefix + step) + 1:]
            if rest.startswith("data."):
                names.setdefault(rest[5:], text)
            elif rest.startswith("data_description."):
                helps.setdefault(rest[17:], text)
            elif rest == "title" and not title:
                title = text
            elif rest == "description" and not description:
                description = text
    return names, helps, title, description


def _serialize(schema: vol.Schema) -> list[dict[str, Any]]:
    """The schema as the options form receives it.

    Home Assistant serializes with probatio since late 2026 and with
    voluptuous_serialize before.
    """
    from homeassistant.helpers import config_validation as cv  # noqa: PLC0415

    try:
        import probatio  # noqa: PLC0415

        return probatio.to_field_list(schema, custom_serializer=cv.custom_serializer)
    except ImportError:
        import voluptuous_serialize  # noqa: PLC0415

        return voluptuous_serialize.convert(schema, custom_serializer=cv.custom_serializer)


def _provider_schema(options: dict[str, Any]) -> vol.Schema:
    return vol.Schema({
        vol.Required(
            CONF_BASE_URL, default=options.get(CONF_BASE_URL, DEFAULT_BASE_URL)
        ): selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        ),
        vol.Required(CONF_MODEL, default=options.get(CONF_MODEL, "")): selector.TextSelector(),
    })


def _web_search_schema(options: dict[str, Any]) -> vol.Schema:
    from ..config_flow.schemas import _web_search_schema as base  # noqa: PLC0415

    schema = dict(base(options).schema)
    schema[vol.Optional(CONF_SEARXNG_URL, default=options.get(CONF_SEARXNG_URL, ""))] = (
        selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL))
    )
    return vol.Schema(schema)


def _sections(hass: HomeAssistant, options: dict[str, Any]) -> list[tuple[str, list[str], vol.Schema]]:
    from ..config_flow.schemas import (  # noqa: PLC0415
        _advanced_schema,
        _memory_schema,
        _tools_schema,
    )
    from ..config_flow.voice_id_steps import voice_id_settings_schema  # noqa: PLC0415

    return [
        ("provider", ["provider_settings", "provider_model"], _provider_schema(options)),
        ("web_search", ["web_search", "searxng"], _web_search_schema(options)),
        ("memory", ["memory"], _memory_schema(options)),
        ("tools", ["tools"], _tools_schema(options)),
        ("advanced", ["advanced"], _advanced_schema(options, hass)),
        ("voice", ["voice_id_settings"], voice_id_settings_schema(options)),
    ]


async def _section_payload(hass, options, labels, sid, steps, schema) -> dict[str, Any]:
    names, helps, title, description = _step_text(labels, steps)
    fields, values = form_fields(_serialize(schema), options, names, helps)
    return {
        "id": sid,
        "title": title,
        "description": description,
        "fields": fields,
        "values": values,
    }


async def _apply(hass, options, sid, values) -> tuple[dict[str, Any], dict[str, str]]:
    from ..config_flow.apply import (  # noqa: PLC0415
        apply_advanced,
        apply_memory,
        apply_provider,
        apply_web_search,
    )

    if sid == "provider":
        return await apply_provider(options, values)
    if sid == "web_search":
        return apply_web_search(options, values)
    if sid == "memory":
        return apply_memory(options, values), {}
    if sid == "advanced":
        return await apply_advanced(options, values)
    if sid in ("tools", "voice"):
        return {**options, **values}, {}
    return options, {"base": "unknown_section"}


def _error_text(labels: dict[str, str], code: str) -> str:
    return labels.get(f"component.{DOMAIN}.options.error.{code}") or code.replace("_", " ")


async def _ollama_loaded(base_url: str, model: str) -> dict[str, Any] | None:
    """What an Ollama server holds in memory (``/api/ps``); None elsewhere."""
    import aiohttp  # noqa: PLC0415

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{root}/api/ps", timeout=aiohttp.ClientTimeout(total=4)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    mine = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
    out: dict[str, Any] = {"loaded": mine is not None, "others": len(models) - (1 if mine else 0)}
    if mine:
        size = int(mine.get("size") or 0)
        vram = int(mine.get("size_vram") or 0)
        out.update({
            "size_gb": round(size / 1e9, 1),
            "gpu_pct": round(100 * vram / size) if size else None,
            "context": mine.get("context_length"),
            "until": mine.get("expires_at"),
        })
    return out


async def async_model_health(hass: HomeAssistant, entry, refresh: bool = False) -> dict[str, Any]:
    """Can the model server be reached? Asked at most every 30 s."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from ..exceptions import CannotConnect  # noqa: PLC0415
    from ..providers.openai_compat import async_fetch_models  # noqa: PLC0415

    cached = _health_cache.get(entry.entry_id)
    if cached and not refresh and time.monotonic() - cached[0] < _HEALTH_TTL_S:
        return cached[1]
    started = time.monotonic()
    model = entry.options.get(CONF_MODEL, "")
    base_url = str(entry.options.get(CONF_BASE_URL, ""))
    try:
        models = await async_fetch_models(base_url, entry.options.get(CONF_API_KEY))
        result = {
            "reachable": True,
            "ms": round((time.monotonic() - started) * 1000),
            "model_listed": model in models if models else None,
            "models": len(models),
            "ollama": await _ollama_loaded(base_url, model),
        }
    except CannotConnect as exc:
        result = {"reachable": False, "error": str(exc)[:200]}
    history = _health_history.setdefault(entry.entry_id, [])
    history.append({
        "time": datetime.now(UTC).isoformat(),
        "ok": result["reachable"],
        "ms": result.get("ms"),
    })
    del history[:-_HISTORY]
    result["history"] = list(history)
    _health_cache[entry.entry_id] = (time.monotonic(), result)
    return result


# ── commands ─────────────────────────────────────────────────────────────────


@websocket_api.websocket_command({vol.Required("type"): "ai_plugin/entries"})
@websocket_api.require_admin
@callback
def ws_entries(hass, connection, msg):  # noqa: D103
    loaded = hass.data.get(DOMAIN, {})
    connection.send_result(msg["id"], [
        {
            "entry_id": e.entry_id,
            "title": e.title,
            "model": e.options.get(CONF_MODEL),
            "loaded": isinstance(loaded.get(e.entry_id), dict),
        }
        for e in hass.config_entries.async_entries(DOMAIN)
    ])


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/status",
    vol.Optional("refresh"): bool,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_status(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    from homeassistant.loader import async_get_integration  # noqa: PLC0415

    from ..const import features_off  # noqa: PLC0415

    integration = await async_get_integration(hass, DOMAIN)
    activity = data.get("activity")
    act = activity.as_dict() if activity is not None else {}
    for turn in act.get("turns", []):
        turn["device_name"] = _device_name(hass, turn.get("device"))
    entity = data.get("conversation_entity")
    down_until = float(getattr(entity, "_backend_down_until", 0.0) or 0.0)
    mcp = data.get("mcp")
    servers = mcp.server_status() if mcp is not None else []
    voice = data.get("voice_id")
    diary = data.get("diary")
    scheduler = data.get("scheduler")
    url = str(entry.options.get(CONF_BASE_URL, ""))
    connection.send_result(msg["id"], {
        "entry_id": entry.entry_id,
        "title": entry.title,
        "version": str(integration.version or ""),
        "model": entry.options.get(CONF_MODEL),
        "server": urlparse(url).netloc or url,
        "health": await async_model_health(hass, entry, bool(msg.get("refresh"))),
        "offline_fallback": entry.options.get(CONF_OFFLINE_FALLBACK, DEFAULT_OFFLINE_FALLBACK),
        "offline_for_s": max(0, round(down_until - time.monotonic())),
        "features_off": sorted(features_off(entry.options)),
        "features": sorted(FEATURE_TOOLS),
        "tools_builtin": mcp.builtin_tool_count() if mcp is not None else 0,
        "mcp": {
            "total": len(servers),
            "connected": sum(1 for s in servers if s["state"] in ("connected", "builtin")),
            "tools": sum(s["tools"] for s in servers),
        },
        "memory_on": data.get("memory") is not None,
        "voice": {
            "enabled": bool(getattr(voice, "enabled", False)),
            "people": len(voice.profile_summary()) if voice is not None else 0,
            "error": getattr(voice, "last_error", None),
        },
        "diary_notes": len(diary.entries()) if diary is not None else 0,
        "scheduled": len(scheduler.jobs()) if scheduler is not None else 0,
        "activity": act,
    })


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/settings",
    vol.Optional("language"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_settings(hass, connection, msg):  # noqa: D103
    entry, _data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    options = dict(entry.options)
    labels = await _labels(hass, msg.get("language"))
    sections = [
        await _section_payload(hass, options, labels, sid, steps, schema)
        for sid, steps, schema in _sections(hass, options)
    ]
    connection.send_result(msg["id"], {"sections": sections})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/settings/save",
    vol.Required("section"): str,
    vol.Required("values"): dict,
    vol.Optional("language"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_settings_save(hass, connection, msg):  # noqa: D103
    entry, _data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    options = dict(entry.options)
    labels = await _labels(hass, msg.get("language"))
    schema = next((s for sid, _st, s in _sections(hass, options) if sid == msg["section"]), None)
    if schema is None:
        connection.send_error(msg["id"], "unknown_section", msg["section"])
        return
    values = clean_values(msg["values"])
    # The secret fields are not in the schemas with their values; check them
    # apart so leaving one out keeps it.
    secrets = {k: v for k, v in values.items() if k in SECRET_FIELDS}
    try:
        checked = schema({k: v for k, v in values.items() if k not in SECRET_FIELDS})
    except vol.Invalid as exc:
        field = str(exc.path[0]) if exc.path else "base"
        connection.send_result(msg["id"], {"saved": False, "errors": {field: str(exc.msg)}})
        return
    checked = {k: v for k, v in checked.items() if k not in SECRET_FIELDS}
    checked.update({k: str(v) for k, v in secrets.items()})
    new, errors = await _apply(hass, options, msg["section"], checked)
    if errors:
        connection.send_result(msg["id"], {
            "saved": False,
            "errors": {k: _error_text(labels, v) for k, v in errors.items()},
        })
        return
    changed = new != options
    if changed:
        _health_cache.pop(entry.entry_id, None)
        hass.config_entries.async_update_entry(entry, options=new)
    connection.send_result(msg["id"], {"saved": True, "changed": changed})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/models",
    vol.Optional("base_url"): str,
    vol.Optional("api_key"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_models(hass, connection, msg):  # noqa: D103
    from ..exceptions import CannotConnect  # noqa: PLC0415
    from ..providers.openai_compat import async_fetch_models  # noqa: PLC0415

    entry, _data = _loaded(hass, msg.get("entry_id"))
    options = dict(entry.options) if entry is not None else {}
    base_url = (msg.get("base_url") or options.get(CONF_BASE_URL) or "").strip().rstrip("/")
    api_key = msg.get("api_key") or options.get(CONF_API_KEY)
    try:
        models = await async_fetch_models(base_url, api_key)
    except CannotConnect as exc:
        connection.send_result(msg["id"], {"models": [], "error": str(exc)[:200]})
        return
    connection.send_result(msg["id"], {"models": sorted(models)})


# ── MCP servers ──────────────────────────────────────────────────────────────


def _presets() -> dict[str, dict[str, Any]]:
    from ..config_flow.mcp_steps import MCPServerStepsMixin  # noqa: PLC0415

    return MCPServerStepsMixin._STDIO_PRESETS  # noqa: SLF001


@websocket_api.websocket_command({vol.Required("type"): "ai_plugin/mcp", **_ENTRY})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_mcp(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    from .. import _own_hosts  # noqa: PLC0415
    from ..tools.mcp_client import is_own_ha_mcp  # noqa: PLC0415

    configs = list(entry.options.get(CONF_MCP_SERVERS, []))
    status = data["mcp"].server_status() if data.get("mcp") is not None else []
    own = _own_hosts(hass)
    servers = mask_servers(configs)
    for i, server in enumerate(servers):
        server["status"] = status[i] if i < len(status) else {"state": "off", "tools": 0}
        server["own_ha"] = is_own_ha_mcp(configs[i], own)
    connection.send_result(msg["id"], {
        "servers": servers,
        "builtin_tools": data["mcp"].builtin_tool_count() if data.get("mcp") is not None else 0,
        "presets": [
            {
                "key": key,
                "label": p["label"],
                "command": p["command"],
                "needs_value": "args_template" in p,
                "value_label": p.get("config_label"),
                "placeholder": p.get("config_placeholder"),
            }
            for key, p in _presets().items()
        ],
    })


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/mcp/preset",
    vol.Required("key"): str,
    vol.Optional("value"): str,
})
@websocket_api.require_admin
@callback
def ws_mcp_preset(hass, connection, msg):  # noqa: D103
    preset = _presets().get(msg["key"])
    server = preset_server(preset, msg.get("value")) if preset else None
    if server is None:
        connection.send_error(msg["id"], "value_required", "This server needs a value.")
        return
    connection.send_result(msg["id"], mask_servers([server])[0] | {"index": None})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/mcp/save",
    vol.Required("servers"): list,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_mcp_save(hass, connection, msg):  # noqa: D103
    entry, _data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    old = list(entry.options.get(CONF_MCP_SERVERS, []))
    servers, errors = merge_servers(old, msg["servers"])
    if errors:
        connection.send_result(msg["id"], {"saved": False, "errors": errors})
        return
    if servers != old:
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, CONF_MCP_SERVERS: servers}
        )
    connection.send_result(msg["id"], {"saved": True, "changed": servers != old})


# ── Voice recognition ────────────────────────────────────────────────────────


@websocket_api.websocket_command({vol.Required("type"): "ai_plugin/voice", **_ENTRY})
@websocket_api.require_admin
@callback
def ws_voice(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    from ..config_flow.voice_id_steps import mic_group_rooms, satellites  # noqa: PLC0415

    mgr = data.get("voice_id")
    if mgr is None:
        connection.send_error(msg["id"], "no_voice", "Voice recognition is not set up.")
        return
    persons = mgr.persons()
    rooms = mic_group_rooms(hass, mgr)
    summary = mgr.profile_summary()
    people = []
    for pid, person in sorted(persons.items(), key=lambda kv: str(kv[1].get("name"))):
        counts = summary.get(pid, {})
        people.append({
            "id": pid,
            "name": person.get("name") or pid,
            "samples": sum(counts.values()),
            "rooms": [
                {"room": rooms.get(g, g or "?"), "samples": n} for g, n in sorted(counts.items())
            ],
        })
    covered = {g for c in summary.values() for g in c}
    pending: list[str] = []
    if mgr.enabled:
        from ..voice_id import pipelines  # noqa: PLC0415

        pending = list(pipelines.unwrapped_pipelines(hass, entry))
    connection.send_result(msg["id"], {
        "enabled": mgr.enabled,
        "people": people,
        "satellites": [
            {"device_id": did, "name": name}
            for did, (_eid, name) in sorted(satellites(hass).items(), key=lambda kv: kv[1][1])
        ],
        "rooms_without_samples": sorted(rooms[g] for g in set(rooms) - covered) if summary else [],
        "pipelines_pending": pending,
        "error": mgr.last_error,
    })


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/voice/enroll",
    vol.Required("person"): str,
    vol.Required("satellite"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_voice_enroll(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    from ..config_flow.voice_id_steps import satellites  # noqa: PLC0415

    mgr = data.get("voice_id")
    sats = satellites(hass)
    if mgr is None or not mgr.enabled:
        connection.send_error(msg["id"], "disabled", "Turn voice recognition on first.")
        return
    if msg["satellite"] not in sats or msg["person"] not in mgr.persons():
        connection.send_error(msg["id"], "unknown", "Unknown person or satellite.")
        return
    entity_id, name = sats[msg["satellite"]]
    fallback = (hass.config.language or "en").split("-")[0].lower()
    try:
        await mgr.async_start_on_satellite(msg["satellite"], entity_id, msg["person"], fallback)
        connection.send_result(msg["id"], {"mode": "started", "satellite": name})
    except Exception:  # noqa: BLE001
        _LOGGER.warning("AI Plugin voice ID: could not start the satellite", exc_info=True)
        mgr.arm_enrollment(msg["satellite"], msg["person"], fallback)
        connection.send_result(msg["id"], {"mode": "armed", "satellite": name})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/voice/delete",
    vol.Required("person"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_voice_delete(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    await data["voice_id"].async_delete_profile(msg["person"])
    connection.send_result(msg["id"], {"deleted": True})


# ── Memory, diary, scheduled ─────────────────────────────────────────────────


def _memory_files(config_dir: str) -> list[tuple[str, list[str]]]:
    from ..tools.memory import _FILE_PREFIX, MemoryTool  # noqa: PLC0415

    out = []
    for name in sorted(os.listdir(config_dir)):
        if name.startswith(_FILE_PREFIX) and name.endswith(".json"):
            key = name[len(_FILE_PREFIX):-5]
            out.append((key, MemoryTool._load(os.path.join(config_dir, name))))  # noqa: SLF001
    return out


async def _users(hass: HomeAssistant) -> dict[str, str]:
    return {u.id: u.name for u in await hass.auth.async_get_users() if u.name}


def _persons(hass: HomeAssistant) -> dict[str, dict[str, Any]]:
    return {
        s.entity_id: {
            "name": s.attributes.get("friendly_name") or s.entity_id,
            "user_id": s.attributes.get("user_id"),
        }
        for s in hass.states.async_all("person")
    }


def _memory_owners(persons, users, existing: list[str]) -> list[dict[str, str]]:
    """Whose memory a fact can be added to: every person (by their login,
    as voice turns save it), plus files that already exist."""
    keys: dict[str, str] = {}
    for pid, info in persons.items():
        key = str(info["user_id"]) if info.get("user_id") else "person_" + pid.split(".", 1)[-1]
        keys[key] = str(info.get("name") or pid)
    for key in existing:
        keys.setdefault(key, memory_owner(key, persons, users))
    return [{"key": k, "owner": v} for k, v in sorted(keys.items(), key=lambda kv: kv[1].lower())]


@websocket_api.websocket_command({vol.Required("type"): "ai_plugin/memory", **_ENTRY})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_memory(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    files = await hass.async_add_executor_job(_memory_files, hass.config.config_dir)
    persons, users = _persons(hass), await _users(hass)
    diary = data.get("diary")
    notes = sorted(diary.entries() if diary is not None else [], key=lambda e: e.get("time", ""), reverse=True)
    scheduler = data.get("scheduler")
    jobs = sorted(scheduler.jobs() if scheduler is not None else [], key=lambda j: j.due)
    connection.send_result(msg["id"], {
        "memory_on": data.get("memory") is not None,
        "facts": [
            {"key": key, "owner": memory_owner(key, persons, users), "facts": facts}
            for key, facts in files
        ],
        "owners": _memory_owners(persons, users, [k for k, _f in files]),
        "diary": [
            {
                "id": n.get("id"),
                "time": n.get("time"),
                "owner": n.get("owner_name") or ("Unrecognised voices" if not n.get("owner") else n.get("owner")),
                "text": n.get("text"),
            }
            for n in notes
        ],
        "scheduled": [
            {
                "id": j.id,
                "kind": j.kind,
                "due": j.due.isoformat(),
                "text": j.text,
                "repeat": j.repeat,
                "device": _device_name(hass, j.device_id),
                "person": (persons.get(j.person_id) or {}).get("name") if j.person_id else None,
            }
            for j in jobs
        ],
    })


def _change_fact(config_dir: str, key: str, index, expected, new) -> bool:
    from ..tools.memory import MemoryTool, _memory_path  # noqa: PLC0415

    if not valid_memory_key(key):
        return False
    path = _memory_path(config_dir, key)
    facts = edit_facts(MemoryTool._load(path), index, expected, new)  # noqa: SLF001
    if facts is None:
        return False
    MemoryTool._save(path, facts)  # noqa: SLF001
    return True


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/memory/forget",
    vol.Required("key"): str,
    vol.Required("index"): int,
    vol.Required("fact"): str,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_memory_forget(hass, connection, msg):  # noqa: D103
    ok = await hass.async_add_executor_job(
        _change_fact, hass.config.config_dir, msg["key"], msg["index"], msg["fact"], None
    )
    connection.send_result(msg["id"], {"deleted": ok})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/memory/edit",
    vol.Required("key"): str,
    vol.Optional("index"): int,
    vol.Optional("fact"): str,
    vol.Required("new"): str,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_memory_edit(hass, connection, msg):  # noqa: D103
    """Replace one fact (index + the fact as shown) or add one (no index)."""
    ok = await hass.async_add_executor_job(
        _change_fact, hass.config.config_dir, msg["key"], msg.get("index"),
        msg.get("fact"), msg["new"],
    )
    connection.send_result(msg["id"], {"saved": ok})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/diary/delete",
    vol.Required("note"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_diary_delete(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    connection.send_result(msg["id"], {"deleted": await data["diary"].async_delete(msg["note"])})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/diary/edit",
    vol.Required("note"): str,
    vol.Required("text"): str,
    vol.Optional("expected"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_diary_edit(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    ok = await data["diary"].async_edit(msg["note"], msg["text"], msg.get("expected"))
    connection.send_result(msg["id"], {"saved": ok})


@websocket_api.websocket_command({
    vol.Required("type"): "ai_plugin/schedule/cancel",
    vol.Required("job"): str,
    **_ENTRY,
})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_schedule_cancel(hass, connection, msg):  # noqa: D103
    entry, data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    scheduler = data["scheduler"]
    jobs = [j for j in scheduler.jobs() if j.id == msg["job"]]
    if jobs:
        await scheduler.async_cancel(jobs)
    connection.send_result(msg["id"], {"deleted": bool(jobs)})


@websocket_api.websocket_command({vol.Required("type"): "ai_plugin/reload", **_ENTRY})
@websocket_api.require_admin
@websocket_api.async_response
async def ws_reload(hass, connection, msg):  # noqa: D103
    entry, _data = _need_entry(hass, connection, msg)
    if entry is None:
        return
    _health_cache.pop(entry.entry_id, None)
    ok = await hass.config_entries.async_reload(entry.entry_id)
    connection.send_result(msg["id"], {"reloaded": bool(ok)})


_COMMANDS = (
    ws_reload,
    ws_entries, ws_status, ws_settings, ws_settings_save, ws_models,
    ws_mcp, ws_mcp_preset, ws_mcp_save,
    ws_voice, ws_voice_enroll, ws_voice_delete,
    ws_memory, ws_memory_forget, ws_memory_edit, ws_diary_edit, ws_diary_delete, ws_schedule_cancel,
)


def async_register_commands(hass: HomeAssistant) -> None:
    for command in _COMMANDS:
        websocket_api.async_register_command(hass, command)
