# set_area_state — Area-Level Action Tool for AI Plugin

Date: 2026-04-22
Status: Approved (brainstorming)

## Problem

Voice commands like "turn off lights in the kitchen" fail with:

> It seems I couldn't find a device named 'light in the bedroom'. Could you please check if the name is correct or provide more details?

The same command typed into the conversation UI mostly works. Transcription is correct in both cases.

Root cause:

- `SYSTEM_PROMPT_VOICE` (`const.py:105`) is a minimal 3-line prompt with zero guidance on tool use or entity discovery.
- Under that prompt, Qwen2.5-7B Q4_K_M (local Ollama, tag `homeassistant:latest`) calls `HassTurnOn` with `name="lights in the bedroom"` — a compound descriptor. HA's intent parser expects a friendly-name OR area + domain, not a sentence fragment, so lookup fails.
- `HassTurnOn` supports `area` + `domain` arguments natively, but the small model doesn't reliably pick those fields without explicit instruction.

## Goal

Make whole-room commands ("lights in kitchen off", "fans in bedroom on") succeed deterministically under both voice and text modes, without depending on the LLM correctly populating `HassTurnOn`'s argument combinations.

## Non-Goals

- Single-device commands ("turn on the reading lamp") — existing `search_entities` + `HassTurnOn` path stays.
- Brightness / color / climate-temp on whole area (future scope).

## Approach

Add one new tool, `set_area_state(area, domain, action)`, to `tools/ha_local.py`. It resolves the area via the area registry, collects exposed entities of the given domain in that area, and dispatches a single service call with an explicit `entity_id` list. Name-parsing is removed from the LLM's critical path.

Both system prompts gain one explicit instruction: for whole-room actions, call `set_area_state`; for single devices, discover first then `HassTurnOn`.

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "set_area_state",
    "description": "Turn devices in an area on/off/toggle. Use for commands like 'lights in kitchen off', 'fans in bedroom on'. For a single named device use HassTurnOn/HassTurnOff instead.",
    "parameters": {
      "type": "object",
      "properties": {
        "area":   {"type": "string", "description": "Area name (e.g. 'kitchen'). Call list_areas if unsure."},
        "domain": {"type": "string", "enum": ["light","switch","fan","media_player","cover","climate"]},
        "action": {"type": "string", "enum": ["turn_on","turn_off","toggle"]},
        "exposed_only": {"type":"boolean","description":"Default true. False to include hidden entities."}
      },
      "required": ["area","domain","action"]
    }
  }
}
```

**Domain allowlist (v1):** `light, switch, fan, media_player, cover, climate`.

**Domain → service mapping:**

| domain        | turn_on      | turn_off      | toggle  |
|---------------|--------------|---------------|---------|
| light         | turn_on      | turn_off      | toggle  |
| switch        | turn_on      | turn_off      | toggle  |
| fan           | turn_on      | turn_off      | toggle  |
| media_player  | turn_on      | turn_off      | —       |
| cover         | open_cover   | close_cover   | toggle  |
| climate       | turn_on      | turn_off      | —       |

## Return Format

String, LLM-readable:

- Success: `"OK — turn_off 4 light(s) in Kitchen: light.ceiling, light.counter, light.dining, light.stove"` (first ~6 ids shown).
- Unknown area: `"Unknown area 'bedrom'. Try list_areas."`
- Unsupported domain: `"Domain 'heater' not supported. Allowed: light, switch, fan, media_player, cover, climate."`
- Domain/action mismatch: `"climate does not support toggle. Use turn_on or turn_off."`
- No entities matched: `"No exposed light in Kitchen."`
- Service error: `"[set_area_state failed: <exc>]"`

## Implementation

File: `custom_components/ai_plugin/tools/ha_local.py`

1. Add module-level constants:

   ```python
   _ACTION_DOMAINS = {"light","switch","fan","media_player","cover","climate"}
   _DOMAIN_SERVICE_MAP = {
     "light":        {"turn_on":"turn_on","turn_off":"turn_off","toggle":"toggle"},
     "switch":       {"turn_on":"turn_on","turn_off":"turn_off","toggle":"toggle"},
     "fan":          {"turn_on":"turn_on","turn_off":"turn_off","toggle":"toggle"},
     "media_player": {"turn_on":"turn_on","turn_off":"turn_off"},
     "cover":        {"turn_on":"open_cover","turn_off":"close_cover","toggle":"toggle"},
     "climate":      {"turn_on":"turn_on","turn_off":"turn_off"},
   }
   ```

   Note: `media_player` has no native `toggle` service; omitting it returns the domain/action-mismatch error instead of failing at dispatch.

2. Append the schema above to `TOOL_SCHEMAS`; add `"set_area_state"` to `TOOL_NAMES`.

3. Add dispatch branch in `HALocalToolRegistry.call_tool`:

   ```python
   if name == "set_area_state":
       return await self._set_area_state(
           area=_s(arguments.get("area")).strip(),
           domain=_s(arguments.get("domain")).strip(),
           action=_s(arguments.get("action")).strip(),
           exposed_only=exposed_only,
       )
   ```

   Note: `call_tool` becomes genuinely `async` for this branch (currently sync-bodied). Keep signature as `async def`.

4. New method:

   ```python
   async def _set_area_state(self, area, domain, action, exposed_only=True) -> str:
       domain = domain.lower().strip(".")
       action = action.lower()
       # validate domain
       if domain not in _ACTION_DOMAINS:
           allowed = ", ".join(sorted(_ACTION_DOMAINS))
           return f"Domain {domain!r} not supported. Allowed: {allowed}."
       # validate action for this domain
       svc_map = _DOMAIN_SERVICE_MAP[domain]
       if action not in svc_map:
           allowed = " or ".join(svc_map)
           return f"{domain} does not support {action}. Use {allowed}."
       service = svc_map[action]
       # resolve area (match name or alias, case-insensitive)
       area_reg = ar.async_get(self._hass)
       target_area = None
       needle = area.lower()
       for a in area_reg.async_list_areas():
           if _s(a.name).lower() == needle:
               target_area = a
               break
           aliases = getattr(a, "aliases", None) or ()
           if any(_s(al).lower() == needle for al in aliases):
               target_area = a
               break
       if target_area is None:
           return f"Unknown area {area!r}. Try list_areas."
       # collect entity_ids in area + domain
       ent_reg = er.async_get(self._hass)
       dev_reg = dr.async_get(self._hass)
       ids: list[str] = []
       for eid, entry in ent_reg.entities.items():
           if not eid.startswith(f"{domain}."):
               continue
           area_id = entry.area_id
           if not area_id and entry.device_id:
               dev = dev_reg.async_get(entry.device_id)
               if dev:
                   area_id = dev.area_id
           if area_id != target_area.id:
               continue
           if exposed_only and not self._is_exposed(eid):
               continue
           ids.append(eid)
       if not ids:
           scope = "exposed " if exposed_only else ""
           return f"No {scope}{domain} in {target_area.name}."
       ids.sort()
       # dispatch
       try:
           await self._hass.services.async_call(
               domain, service, target={"entity_id": ids}, blocking=True,
           )
       except Exception as exc:  # noqa: BLE001
           _LOGGER.exception("AI Plugin set_area_state failed")
           return f"[set_area_state failed: {exc}]"
       preview = ", ".join(ids[:6])
       more = f" (+{len(ids)-6} more)" if len(ids) > 6 else ""
       return f"OK — {action} {len(ids)} {domain}(s) in {target_area.name}: {preview}{more}"
   ```

5. Edit `const.py`:

   **`SYSTEM_PROMPT_DEFAULT`** — replace the action-resolution line (currently at `const.py:85`) with:

   ```
   - For whole-room actions ('lights in kitchen off', 'fans in bedroom on'): CALL set_area_state(area, domain, action). Do NOT pass compound names to HassTurnOn.
   - For one specific device: discover via get_entity/search_entities, then CALL HassTurnOn/HassTurnOff/HassLightSet with the returned entity_id.
   ```

   **`SYSTEM_PROMPT_VOICE`** — expand from 3 lines to:

   ```
   You control a smart home. Answer briefly in plain speech — no lists, no markdown, no emojis. One sentence confirmations.

   Tool rules:
   - Room-wide action: set_area_state(area, domain, action). Example: user says "lights in kitchen off" → set_area_state("kitchen","light","turn_off").
   - Single named device: search_entities → then HassTurnOn/HassTurnOff with entity_id.
   - "Which rooms": list_areas. "What's in X": list_entities(area=X).
   - Never invent entity_ids or area names. If unsure, call a discovery tool first.
   ```

## Tests

File: `tests/tools/test_ha_local.py` (append to existing).

1. `test_schema_registered` — `"set_area_state" in TOOL_NAMES`; schema in `TOOL_SCHEMAS`.
2. `test_set_area_state_unknown_area` — returns string starting `"Unknown area"`.
3. `test_set_area_state_bad_domain` — returns `"Domain 'foo' not supported"`.
4. `test_set_area_state_bad_action_for_climate` — `climate` + `toggle` → `"climate does not support toggle"`.
5. `test_set_area_state_no_exposed_entities` — area exists, domain has only unexposed entities, `exposed_only=True` → `"No exposed light in Kitchen."`.
6. `test_set_area_state_success_light_turn_off` — stub `hass.services.async_call`; assert it was called exactly once with `("light", "turn_off", target={"entity_id": [sorted ids]}, blocking=True)`; return string contains count and ids.
7. `test_set_area_state_cover_open` — asserts `open_cover` service name (domain-specific mapping).
8. `test_set_area_state_service_exception_caught` — stub raises; result string starts `"[set_area_state failed:"`.

Follow registry-stub pattern from existing tests; no new infrastructure.

## Manual Verification

On live HA, after deploy + integration reload:

1. Voice: "turn off lights in the kitchen" → `turn_off` fires in logbook; debug log shows `set_area_state("kitchen","light","turn_off")`.
2. Voice: "lights in the bedroom on" — the original failing case; must succeed.
3. Typed UI: both of the above still work.
4. Voice: "turn off lights in the garden" (nonexistent area) → spoken reply reflects friendly error.
5. Voice: "turn on the desk lamp" (single device) → still routes through `HassTurnOn`.

## Rollout

- No config migration.
- `manifest.json` version bump: `0.5.17` → `0.5.18`.
- Reload AI Plugin integration via HA UI; no full HA restart required.

## Out of Scope (future)

- Brightness, color, color-temp on whole area (would add parameters to `set_area_state` or a sibling tool).
- Climate setpoint on whole area.
- `media_player` toggle (needs `media_play_pause` semantics; revisit if asked).
