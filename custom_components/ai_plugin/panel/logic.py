"""The parts of the sidebar page's backend that need no Home Assistant.

Secrets never travel to the page: API keys, search keys, MCP tokens and MCP
environment values go out as "set" or "not set". What comes back either
leaves a secret out (keep it), sends an empty string (remove it) or sends a
new value.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ..const import CONF_API_KEY, CONF_BRAVE_API_KEY, CONF_TAVILY_API_KEY

SECRET_FIELDS = frozenset({CONF_API_KEY, CONF_BRAVE_API_KEY, CONF_TAVILY_API_KEY})


def _valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def form_fields(
    serialized: list[dict[str, Any]],
    options: dict[str, Any],
    labels: dict[str, str],
    helps: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Serialized schema → (fields for the page, their current values).

    A field's value is what the options form would show: its suggested
    value, else its default. Secret fields carry no value, only ``is_set``.
    """
    fields: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    for item in serialized:
        name = item.get("name")
        if not name:
            continue
        field = {k: v for k, v in item.items() if k not in ("default", "description")}
        field["label"] = labels.get(name) or name.replace("_", " ").capitalize()
        if helps.get(name):
            field["help"] = helps[name]
        if name in SECRET_FIELDS:
            field["secret"] = True
            field["is_set"] = bool(options.get(name))
        else:
            suggested = (item.get("description") or {}).get("suggested_value")
            value = suggested if suggested is not None else item.get("default")
            if value is not None:
                values[name] = value
        fields.append(field)
    return fields, values


def clean_values(values: dict[str, Any]) -> dict[str, Any]:
    """Drop what the page sends for "left empty": None, and an empty string
    on anything but a secret (where it means remove)."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if value == "" and key not in SECRET_FIELDS:
            continue
        out[key] = value
    return out


def mask_servers(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP servers as the page sees them, secrets replaced by flags."""
    out = []
    for i, server in enumerate(servers):
        if server.get("transport") == "stdio":
            out.append({
                "index": i,
                "transport": "stdio",
                "command": server.get("command", ""),
                "args": list(server.get("args") or []),
                "env": [{"name": k, "is_set": bool(v)} for k, v in (server.get("env") or {}).items()],
            })
        else:
            out.append({
                "index": i,
                "transport": server.get("transport") or "http",
                "url": server.get("url", ""),
                "token_set": bool(server.get("token")),
            })
    return out


def merge_servers(
    old: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The page's edited list → (servers to save, errors).

    Each incoming server may name the ``index`` it was in the old list; its
    secrets are taken from there unless new ones came with it. Errors are
    ``{"index": position, "field": name, "error": code}``; with any error
    nothing should be saved.
    """
    servers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pos, item in enumerate(incoming):
        idx = item.get("index")
        before = old[idx] if isinstance(idx, int) and 0 <= idx < len(old) else {}
        if item.get("transport") == "stdio":
            command = str(item.get("command") or "").strip()
            if not command:
                errors.append({"index": pos, "field": "command", "error": "command_required"})
                continue
            args = item.get("args") or []
            if isinstance(args, str):
                args = args.split()
            server: dict[str, Any] = {
                "transport": "stdio",
                "command": command,
                "args": [str(a) for a in args if str(a).strip()],
            }
            old_env = before.get("env") or {}
            env: dict[str, str] = {}
            for var in item.get("env") or []:
                name = str(var.get("name") or "").strip()
                if not name:
                    continue
                if "value" in var and var["value"] is not None:
                    if str(var["value"]).strip():
                        env[name] = str(var["value"]).strip()
                elif name in old_env:
                    env[name] = old_env[name]
                elif var.get("from") in old_env:
                    # Renamed: keep the value under the new name.
                    env[name] = old_env[var["from"]]
            if env:
                server["env"] = env
            servers.append(server)
        else:
            url = str(item.get("url") or "").strip()
            if not _valid_url(url):
                errors.append({"index": pos, "field": "url", "error": "invalid_url"})
                continue
            server = {"transport": item.get("transport") or "http", "url": url}
            if "token" in item and item["token"] is not None:
                if str(item["token"]).strip():
                    server["token"] = str(item["token"]).strip()
            elif before.get("token"):
                server["token"] = before["token"]
            servers.append(server)
    return servers, errors


def preset_server(preset: dict[str, Any], value: str | None) -> dict[str, Any] | None:
    """A stdio server from a preset; None when the preset needs a value."""
    if "args_template" in preset:
        value = (value or "").strip()
        if not value:
            return None
        args = [a.replace("{value}", value) for a in preset["args_template"]]
    else:
        args = list(preset.get("args") or [])
    return {"transport": "stdio", "command": preset["command"], "args": args}


def memory_owner(key: str, persons: dict[str, dict[str, Any]], users: dict[str, str]) -> str:
    """A display name for a memory file key (user id, person_<slug>, anonymous)."""
    if key == "anonymous":
        return "Unrecognised voices"
    if key.startswith("person_"):
        pid = "person." + key[len("person_"):]
        return str((persons.get(pid) or {}).get("name") or pid)
    for info in persons.values():
        if info.get("user_id") == key:
            return str(info.get("name"))
    return users.get(key) or key


_MAX_FACT = 500


def valid_memory_key(key: str) -> bool:
    """A memory file key the page may name: a user id, person_<slug> or anonymous."""
    return bool(key) and len(key) <= 80 and all(c.isalnum() or c == "_" for c in key)


def edit_facts(
    facts: list[str], index: int | None, expected: str | None, new: str | None
) -> list[str] | None:
    """The facts after one change from the page, or None when refused.

    ``index`` with ``new`` None removes, ``index`` with ``new`` replaces, no
    ``index`` adds. A change by index needs ``expected`` to still be the fact
    there, so an edit made meanwhile by voice is not overwritten. An empty or
    duplicate new fact is refused.
    """
    out = list(facts)
    if new is not None:
        new = " ".join(str(new).split())[:_MAX_FACT]
        if not new:
            return None
        others = [f for i, f in enumerate(out) if i != index]
        if new.lower() in (f.lower() for f in others):
            return None
    if index is None:
        if new is None:
            return None
        out.append(new)
        return out
    if not (0 <= index < len(out)) or out[index] != expected:
        return None
    if new is None:
        out.pop(index)
    else:
        out[index] = new
    return out
