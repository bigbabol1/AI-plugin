# set_area_state Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a whole-area action tool (`set_area_state`) to AI Plugin so voice commands like "turn off lights in the kitchen" resolve deterministically under a small local LLM (Qwen2.5-7B Q4_K_M).

**Architecture:** New async method + schema in `tools/ha_local.py` that resolves an area via the area registry, collects matching exposed entities in that area+domain, and dispatches a single `hass.services.async_call(...)` with an explicit `entity_id` list. Both system prompts in `const.py` get explicit guidance to prefer `set_area_state` for room-wide commands. No config migration; version bumps `0.5.17 → 0.5.18`.

**Tech Stack:** Python 3.11, Home Assistant 2024+ (`area_registry`, `entity_registry`, `device_registry`), pytest + unittest.mock.

Spec: `docs/superpowers/specs/2026-04-22-set-area-state-design.md`

---

## File Structure

**Modify**
- `custom_components/ai_plugin/tools/ha_local.py` — add constants, schema entry, dispatch branch, `_set_area_state` method.
- `custom_components/ai_plugin/const.py` — update `SYSTEM_PROMPT_DEFAULT` action line and expand `SYSTEM_PROMPT_VOICE`.
- `custom_components/ai_plugin/manifest.json` — bump version.

**Create**
- `tests/components/ai_plugin/test_ha_local.py` — new test module (no existing tests for `ha_local`).

---

### Task 1: Scaffold test module + fixtures

**Files:**
- Create: `tests/components/ai_plugin/test_ha_local.py`

- [ ] **Step 1: Create the test file skeleton with shared fixtures**

```python
"""Tests for tools/ha_local.py (entity discovery + set_area_state)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.tools.ha_local import (
    HALocalToolRegistry,
    TOOL_NAMES,
    TOOL_SCHEMAS,
)


# ── registry stubs ────────────────────────────────────────────────────────────


def _area(area_id: str, name: str, aliases: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(id=area_id, name=name, aliases=set(aliases))


def _entity(
    entity_id: str,
    *,
    area_id: str | None = None,
    device_id: str | None = None,
    name: str = "",
    original_name: str = "",
    aliases: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id,
        area_id=area_id,
        device_id=device_id,
        name=name,
        original_name=original_name,
        aliases=set(aliases),
    )


def _device(device_id: str, area_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=device_id, area_id=area_id)


def _make_hass(
    areas: list[SimpleNamespace],
    entities: list[SimpleNamespace],
    devices: list[SimpleNamespace] | None = None,
    exposed: set[str] | None = None,
    service_side_effect=None,
) -> MagicMock:
    """Build a hass mock with area/entity/device registries and a services layer.

    `exposed` is the set of entity_ids treated as exposed to the conversation
    assistant; if None, all entities are exposed.
    """
    hass = MagicMock()

    # area registry
    area_reg = MagicMock()
    area_reg.async_list_areas.return_value = list(areas)
    area_reg.async_get_area.side_effect = lambda aid: next(
        (a for a in areas if a.id == aid), None
    )

    # entity registry — `.entities` is a mapping-like {entity_id: entry}
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in entities}
    ent_reg.async_get.side_effect = lambda eid: ent_reg.entities.get(eid)

    # device registry
    dev_reg = MagicMock()
    dev_map = {d.id: d for d in (devices or [])}
    dev_reg.async_get.side_effect = lambda did: dev_map.get(did)

    hass._registries = (area_reg, ent_reg, dev_reg)  # keep refs alive

    # services
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=service_side_effect)

    # states (not needed for set_area_state, but stub to None)
    hass.states = MagicMock()
    hass.states.get.return_value = None

    return hass, area_reg, ent_reg, dev_reg


@pytest.fixture
def patched_registries():
    """Patch the module-level `ar`, `er`, `dr` references so `ar.async_get(hass)`
    etc. return the registry mocks we attached to the hass stub."""

    def _apply(hass, area_reg, ent_reg, dev_reg):
        p1 = patch(
            "custom_components.ai_plugin.tools.ha_local.ar.async_get",
            return_value=area_reg,
        )
        p2 = patch(
            "custom_components.ai_plugin.tools.ha_local.er.async_get",
            return_value=ent_reg,
        )
        p3 = patch(
            "custom_components.ai_plugin.tools.ha_local.dr.async_get",
            return_value=dev_reg,
        )
        p1.start()
        p2.start()
        p3.start()
        return (p1, p2, p3)

    started: list = []

    def start(hass, area_reg, ent_reg, dev_reg):
        started.extend(_apply(hass, area_reg, ent_reg, dev_reg))

    yield start
    for p in started:
        p.stop()
```

- [ ] **Step 2: Verify collection**

Run: `pytest tests/components/ai_plugin/test_ha_local.py --collect-only -q`
Expected: collects zero tests, no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/components/ai_plugin/test_ha_local.py
git commit -m "test(ha_local): scaffold test module with registry stubs"
```

---

### Task 2: Schema-registration test (failing first)

**Files:**
- Modify: `tests/components/ai_plugin/test_ha_local.py`

- [ ] **Step 1: Add the failing test at the bottom of the file**

```python
def test_set_area_state_schema_registered() -> None:
    """set_area_state must be exposed in TOOL_NAMES and TOOL_SCHEMAS."""
    assert "set_area_state" in TOOL_NAMES
    schema = next(
        (s for s in TOOL_SCHEMAS if s["function"]["name"] == "set_area_state"),
        None,
    )
    assert schema is not None, "set_area_state schema missing"
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {"area", "domain", "action"}
    assert set(params["properties"]["domain"]["enum"]) == {
        "light", "switch", "fan", "media_player", "cover", "climate",
    }
    assert set(params["properties"]["action"]["enum"]) == {
        "turn_on", "turn_off", "toggle",
    }
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/components/ai_plugin/test_ha_local.py::test_set_area_state_schema_registered -v`
Expected: FAIL — `"set_area_state" not in TOOL_NAMES`.

- [ ] **Step 3: Implement the schema + name registration**

Edit `custom_components/ai_plugin/tools/ha_local.py`.

At module level, above `TOOL_SCHEMAS`, add:

```python
_ACTION_DOMAINS = {"light", "switch", "fan", "media_player", "cover", "climate"}

_DOMAIN_SERVICE_MAP: dict[str, dict[str, str]] = {
    "light":        {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "switch":       {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "fan":          {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "media_player": {"turn_on": "turn_on", "turn_off": "turn_off"},
    "cover":        {"turn_on": "open_cover", "turn_off": "close_cover", "toggle": "toggle"},
    "climate":      {"turn_on": "turn_on", "turn_off": "turn_off"},
}
```

Append to `TOOL_SCHEMAS` (just before the closing `]`):

```python
    {
        "type": "function",
        "function": {
            "name": "set_area_state",
            "description": (
                "Turn devices in an area on/off/toggle. Use for commands like "
                "'lights in kitchen off', 'fans in bedroom on'. For a single "
                "named device use HassTurnOn/HassTurnOff instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area name (e.g. 'kitchen'). Call list_areas if unsure.",
                    },
                    "domain": {
                        "type": "string",
                        "enum": sorted(_ACTION_DOMAINS),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "toggle"],
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["area", "domain", "action"],
            },
        },
    },
```

Update `TOOL_NAMES`:

```python
TOOL_NAMES = {"list_areas", "list_entities", "get_entity", "search_entities", "set_area_state"}
```

- [ ] **Step 4: Run the test again**

Run: `pytest tests/components/ai_plugin/test_ha_local.py::test_set_area_state_schema_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/tools/ha_local.py tests/components/ai_plugin/test_ha_local.py
git commit -m "feat(ha_local): register set_area_state tool schema"
```

---

### Task 3: Validation errors — unknown area, bad domain, bad action

**Files:**
- Modify: `tests/components/ai_plugin/test_ha_local.py`
- Modify: `custom_components/ai_plugin/tools/ha_local.py`

- [ ] **Step 1: Add failing tests**

```python
@pytest.mark.asyncio
async def test_set_area_state_bad_domain(patched_registries) -> None:
    """Unknown domain returns a friendly error without touching services."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "heater", "action": "turn_on"},
    )
    assert out.startswith("Domain 'heater' not supported")
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_bad_action_for_climate(patched_registries) -> None:
    """climate + toggle returns the domain/action-mismatch error."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "climate", "action": "toggle"},
    )
    assert "climate does not support toggle" in out
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_unknown_area(patched_registries) -> None:
    """Unknown area returns 'Unknown area ...'."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "garden", "domain": "light", "action": "turn_off"},
    )
    assert out.startswith("Unknown area 'garden'")
    hass.services.async_call.assert_not_called()
```

- [ ] **Step 2: Run and verify all three fail**

Run: `pytest tests/components/ai_plugin/test_ha_local.py -k set_area_state -v`
Expected: 3 new tests FAIL (AttributeError on `call_tool` branch or similar; schema test still passes).

- [ ] **Step 3: Add the dispatch branch and method (validation-only body for now)**

In `HALocalToolRegistry.call_tool`, inside the existing `try:` block, add after the `search_entities` branch:

```python
            if name == "set_area_state":
                return await self._set_area_state(
                    area=_s(arguments.get("area")).strip(),
                    domain=_s(arguments.get("domain")).strip(),
                    action=_s(arguments.get("action")).strip(),
                    exposed_only=exposed_only,
                )
```

Add the new method to the class:

```python
    async def _set_area_state(
        self,
        area: str,
        domain: str,
        action: str,
        exposed_only: bool = True,
    ) -> str:
        """Turn devices in an area on/off/toggle via a single service call."""
        domain = domain.lower().strip(".")
        action = action.lower()

        if domain not in _ACTION_DOMAINS:
            allowed = ", ".join(sorted(_ACTION_DOMAINS))
            return f"Domain {domain!r} not supported. Allowed: {allowed}."

        svc_map = _DOMAIN_SERVICE_MAP[domain]
        if action not in svc_map:
            allowed = " or ".join(svc_map)
            return f"{domain} does not support {action}. Use {allowed}."

        # resolve area (name or alias, case-insensitive)
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

        # remainder of implementation added in Task 4
        return "[set_area_state: not yet implemented past area resolution]"
```

- [ ] **Step 4: Run and verify**

Run: `pytest tests/components/ai_plugin/test_ha_local.py -k set_area_state -v`
Expected: all 4 `set_area_state` tests PASS (schema + 3 validation).

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/tools/ha_local.py tests/components/ai_plugin/test_ha_local.py
git commit -m "feat(ha_local): validate domain/action/area for set_area_state"
```

---

### Task 4: Entity collection + service dispatch

**Files:**
- Modify: `tests/components/ai_plugin/test_ha_local.py`
- Modify: `custom_components/ai_plugin/tools/ha_local.py`

- [ ] **Step 1: Add failing tests**

```python
@pytest.mark.asyncio
async def test_set_area_state_success_light_turn_off(patched_registries) -> None:
    """Happy path: all exposed lights in the area get a single service call."""
    areas = [_area("a_kit", "Kitchen"), _area("a_bed", "Bedroom")]
    entities = [
        _entity("light.ceiling", area_id="a_kit"),
        _entity("light.counter", area_id="a_kit"),
        _entity("light.dining", area_id="a_kit"),
        _entity("light.bedroom_lamp", area_id="a_bed"),  # different area
        _entity("switch.kettle", area_id="a_kit"),        # different domain
    ]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_off"},
    )
    assert out.startswith("OK — turn_off 3 light(s) in Kitchen")
    assert "light.ceiling" in out and "light.counter" in out and "light.dining" in out

    hass.services.async_call.assert_awaited_once()
    args, kwargs = hass.services.async_call.call_args
    assert args[0] == "light"
    assert args[1] == "turn_off"
    assert kwargs["target"] == {
        "entity_id": ["light.ceiling", "light.counter", "light.dining"],
    }
    assert kwargs["blocking"] is True


@pytest.mark.asyncio
async def test_set_area_state_cover_open_uses_open_cover(patched_registries) -> None:
    """cover+turn_on maps to the open_cover service."""
    areas = [_area("a1", "Hobbyroom")]
    entities = [_entity("cover.garage", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "hobbyroom", "domain": "cover", "action": "turn_on"},
    )
    assert "OK — turn_on 1 cover(s) in Hobbyroom" in out

    args, _kw = hass.services.async_call.call_args
    assert args[0] == "cover"
    assert args[1] == "open_cover"


@pytest.mark.asyncio
async def test_set_area_state_no_exposed_entities(patched_registries) -> None:
    """Area and domain exist but no entity is exposed → friendly message."""
    areas = [_area("a1", "Kitchen")]
    entities = [_entity("light.hidden", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    # force _is_exposed to return False for this test
    with patch.object(HALocalToolRegistry, "_is_exposed", return_value=False):
        out = await reg.call_tool(
            "set_area_state",
            {"area": "kitchen", "domain": "light", "action": "turn_off"},
        )
    assert out == "No exposed light in Kitchen."
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_alias_match(patched_registries) -> None:
    """Alias match resolves to the canonical area name."""
    areas = [_area("a1", "Kitchen", aliases=("Küche",))]
    entities = [_entity("light.ceiling", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "küche", "domain": "light", "action": "turn_on"},
    )
    assert "in Kitchen" in out
    hass.services.async_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_area_state_device_inherits_area(patched_registries) -> None:
    """Entity with no area_id inherits its device's area_id."""
    areas = [_area("a1", "Kitchen")]
    devices = [_device("d1", area_id="a1")]
    entities = [_entity("light.stove", area_id=None, device_id="d1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities, devices=devices)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_on"},
    )
    assert "1 light(s) in Kitchen" in out
    args, _kw = hass.services.async_call.call_args
    assert _kw["target"] == {"entity_id": ["light.stove"]}


@pytest.mark.asyncio
async def test_set_area_state_service_exception_caught(patched_registries) -> None:
    """Service failure returns a bracketed error string, does not raise."""
    areas = [_area("a1", "Kitchen")]
    entities = [_entity("light.ceiling", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(
        areas=areas,
        entities=entities,
        service_side_effect=RuntimeError("boom"),
    )
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_on"},
    )
    assert out.startswith("[set_area_state failed:")
    assert "boom" in out
```

- [ ] **Step 2: Run and verify all fail**

Run: `pytest tests/components/ai_plugin/test_ha_local.py -k set_area_state -v`
Expected: 6 new tests FAIL (they hit the `"not yet implemented"` placeholder or a `NoneType` error from the stub services).

- [ ] **Step 3: Replace the placeholder with the full implementation**

In `_set_area_state`, replace the last line (`return "[set_area_state: not yet implemented past area resolution]"`) with the full body:

```python
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
        service = svc_map[action]
        try:
            await self._hass.services.async_call(
                domain,
                service,
                target={"entity_id": ids},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin set_area_state failed")
            return f"[set_area_state failed: {exc}]"

        preview = ", ".join(ids[:6])
        more = f" (+{len(ids) - 6} more)" if len(ids) > 6 else ""
        return (
            f"OK — {action} {len(ids)} {domain}(s) in {target_area.name}: "
            f"{preview}{more}"
        )
```

- [ ] **Step 4: Run the full test module**

Run: `pytest tests/components/ai_plugin/test_ha_local.py -v`
Expected: all `set_area_state` tests (10 total: schema + 3 validation + 6 dispatch) PASS.

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/tools/ha_local.py tests/components/ai_plugin/test_ha_local.py
git commit -m "feat(ha_local): implement set_area_state dispatch via services.async_call"
```

---

### Task 5: Update system prompts

**Files:**
- Modify: `custom_components/ai_plugin/const.py:71-109`

Spec: `SYSTEM_PROMPT_DEFAULT` currently contains one line at `const.py:85` starting `- For actions (turn on, turn off, ...)`. Replace it with two lines as shown below. `SYSTEM_PROMPT_VOICE` currently at `const.py:105-109` is 3 lines — replace the whole string.

- [ ] **Step 1: Update `SYSTEM_PROMPT_DEFAULT`**

In `const.py`, find the line:

```python
    "- For actions (turn on, turn off, set brightness, set temperature, set colour): CALL HassTurnOn / HassTurnOff / HassLightSet / HassClimateSetTemperature / etc. with the entity_id returned by discovery.\n"
```

Replace with:

```python
    "- For whole-room actions ('lights in kitchen off', 'fans in bedroom on'): CALL set_area_state(area, domain, action). Do NOT pass compound names to HassTurnOn.\n"
    "- For a single specific device (turn on the reading lamp, set brightness, set colour, set temperature): discover via get_entity/search_entities, then CALL HassTurnOn / HassTurnOff / HassLightSet / HassClimateSetTemperature with the returned entity_id.\n"
```

- [ ] **Step 2: Replace `SYSTEM_PROMPT_VOICE`**

Find:

```python
SYSTEM_PROMPT_VOICE = (
    "You control a smart home. Answer briefly in plain speech — "
    "no lists, no markdown. Be concise. If you need to take an action, "
    "do it and confirm in one sentence."
)
```

Replace with:

```python
SYSTEM_PROMPT_VOICE = (
    "You control a smart home. Answer briefly in plain speech — no lists, "
    "no markdown, no emojis. Confirm actions in one sentence.\n"
    "\n"
    "Tool rules:\n"
    "- Room-wide action: set_area_state(area, domain, action). "
    "Example: user says \"lights in kitchen off\" → "
    "set_area_state(\"kitchen\",\"light\",\"turn_off\").\n"
    "- Single named device: search_entities → then HassTurnOn/HassTurnOff "
    "with entity_id.\n"
    "- \"Which rooms\": list_areas. \"What's in X\": list_entities(area=X).\n"
    "- Never invent entity_ids or area names. "
    "If unsure, call a discovery tool first."
)
```

- [ ] **Step 3: Sanity-import to catch syntax errors**

Run: `python -c "from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE; assert 'set_area_state' in SYSTEM_PROMPT_DEFAULT and 'set_area_state' in SYSTEM_PROMPT_VOICE"`
Expected: no output, exit 0.

- [ ] **Step 4: Run the full ai_plugin test suite to catch regressions**

Run: `pytest tests/components/ai_plugin -q`
Expected: all tests pass (including prior modules).

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/const.py
git commit -m "feat(prompts): instruct LLM to use set_area_state for room-wide actions"
```

---

### Task 6: Version bump

**Files:**
- Modify: `custom_components/ai_plugin/manifest.json`

- [ ] **Step 1: Confirm current version**

Run: `grep '"version"' custom_components/ai_plugin/manifest.json`
Expected: `"version": "0.5.17"`.

- [ ] **Step 2: Bump to 0.5.18**

In `custom_components/ai_plugin/manifest.json`, change the `"version"` value from `"0.5.17"` to `"0.5.18"`.

- [ ] **Step 3: Verify**

Run: `grep '"version"' custom_components/ai_plugin/manifest.json`
Expected: `"version": "0.5.18"`.

- [ ] **Step 4: Commit**

```bash
git add custom_components/ai_plugin/manifest.json
git commit -m "chore: release v0.5.18"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/components/ai_plugin -q`
Expected: all tests pass.

- [ ] **Step 2: Lint pass (if a linter is configured)**

Run: `ruff check custom_components/ai_plugin/tools/ha_local.py custom_components/ai_plugin/const.py 2>/dev/null || true`
Expected: no errors, or "ruff: command not found" (ignore if not installed).

- [ ] **Step 3: Smoke-check the schema + prompt wiring**

Run:
```bash
python - <<'PY'
from custom_components.ai_plugin.tools.ha_local import TOOL_NAMES, TOOL_SCHEMAS
assert "set_area_state" in TOOL_NAMES
names = [s["function"]["name"] for s in TOOL_SCHEMAS]
assert "set_area_state" in names
print("ok")
PY
```
Expected: `ok`.

- [ ] **Step 4: Manual verification notes (run on live HA after deploy)**

Record in commit / PR description that the following were tested post-deploy:

1. Voice: "turn off lights in the kitchen" → logbook shows `light.turn_off` fired with correct entity list; debug log shows `set_area_state("kitchen","light","turn_off")`.
2. Voice: "lights in the bedroom on" (original failing case) — succeeds.
3. Typed UI: both above still work.
4. Voice: "turn off lights in the garden" (nonexistent area) → spoken reply: "Unknown area 'garden'..." style.
5. Voice: "turn on the desk lamp" → still routes through `HassTurnOn`.

- [ ] **Step 5: No commit needed for verification. Stop.**

---

## Notes for the Engineer

- **Async test markers:** the existing test modules use plain `def test_*`. The new tests are `@pytest.mark.asyncio` because `call_tool` is async. If `pytest-asyncio` is not configured in `pyproject.toml`, add `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` — or mark each test as shown.
- **Do not** touch `hass.services.async_call` signature elsewhere. Only `_set_area_state` calls it.
- **`_EXPOSED_ONLY_PROP`** is already defined at the top of `ha_local.py`; reuse it in the schema (do not duplicate).
- **Do not** change `HassTurnOn` / intent flow. Single-device commands must keep working unchanged.
- If the linter complains about the broad `except Exception`, keep the `# noqa: BLE001` comment — this matches the existing pattern at `ha_local.py:258`.
