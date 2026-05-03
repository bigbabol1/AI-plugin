# Multilingual prompt hints — design

**Date:** 2026-05-03
**Target version:** v0.8.0
**Status:** approved (brainstorming complete, awaiting plan)

## Problem

`SYSTEM_PROMPT_DEFAULT` and `SYSTEM_PROMPT_VOICE` in `const.py` mix
English instructions with German trigger-word fragments
(`'alle Lichter aus'`, `'spiel jazz'`, `'welche Lichter sind an'`,
`'wetter in <Ort>'`, etc.) so small models route those phrasings to the
right tools. The fragments serve recognition, not translation. Result:
~1485 prompt tokens regardless of household language. An English-only
home pays for German hints they never use; a Polish or Spanish home gets
no hints for their language.

The user wants to support EN/FR/ES/PT/PL plus DE without further
bloating the prompt. Replies in the spoken language already work for
free on multilingual models (qwen3 family, llama3, mistral) because
modern LLMs match input language without instruction.

## Goal

Configurable per-integration: select up to two household languages, and
the plugin injects only those languages' trigger-word hints into the
system prompt. English remains the prompt base — instructions are
written in English and the LLM understands them regardless of reply
language.

Supported set: **de, fr, es, pt, pl**. English is the implicit base —
not selectable, not excludable.

Cap: two selected languages max (UI rejects more) — keeps the prompt
budget bounded.

## Architecture

```
const.py
├── SYSTEM_PROMPT_DEFAULT      (English base, no per-language hints)
├── SYSTEM_PROMPT_VOICE        (English base, no per-language hints)
├── CONF_TRIGGER_LANGUAGES     "trigger_languages"
├── SUPPORTED_TRIGGER_LANGUAGES = ["de", "fr", "es", "pt", "pl"]
├── PROMPT_HINTS_I18N: dict[str, dict[Literal["default","voice"], str]]
│   └── de / fr / es / pt / pl × default + voice
└── _default_trigger_langs(hass) → list[str]

orchestrator._build_system_prompt(voice_mode, user_id):
  base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
  selected = opts.get(CONF_TRIGGER_LANGUAGES,
                      _default_trigger_langs(self._hass))
  mode_key = "voice" if voice_mode else "default"
  for lang in selected:
      block = PROMPT_HINTS_I18N.get(lang, {}).get(mode_key)
      if block:
          base += "\n\n" + block
  return base
```

Selection order is preserved verbatim — deterministic prompt
fingerprinting for token-budget logs and tests.

## Configuration

### Option key
- `CONF_TRIGGER_LANGUAGES = "trigger_languages"`
- Stored in `config_entry.options` (mutable, edited via OptionsFlow)
- Type: `list[str]`, length 0–2, values from `SUPPORTED_TRIGGER_LANGUAGES`

### Default resolution

```python
def _default_trigger_langs(hass) -> list[str]:
    """Auto-detect HA system language. EN-only HA → []."""
    sys_lang = (hass.config.language or "").split("-")[0]
    return [sys_lang] if sys_lang in SUPPORTED_TRIGGER_LANGUAGES else []
```

`de-DE` HA → preselects `["de"]`. `en-US` → `[]`. `pl-PL` →
`["pl"]`. `zh-CN` → `[]` (unsupported). User can edit; existing
entries with no stored option fall through to this default on every
prompt build.

### Config flow

Add to `_advanced_schema(current)`:

```python
vol.Optional(
    CONF_TRIGGER_LANGUAGES,
    default=current.get(CONF_TRIGGER_LANGUAGES,
                        _default_trigger_langs(hass)),
): selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            {"value": "de", "label": "Deutsch"},
            {"value": "fr", "label": "Français"},
            {"value": "es", "label": "Español"},
            {"value": "pt", "label": "Português"},
            {"value": "pl", "label": "Polski"},
        ],
        multiple=True,
        mode=selector.SelectSelectorMode.DROPDOWN,
    ),
),
```

Validation in `_validate_advanced` (or equivalent options-flow
validator): `len(value) > 2` → emit `ERROR_TOO_MANY_TRIGGER_LANGUAGES`.

### Strings

`strings.json` and `translations/en.json` add (in `options.step.advanced`):

- `data.trigger_languages`: "Trigger-word hint languages"
- `data_description.trigger_languages`: "Adds tool-routing hints in the
  selected languages so small models reliably map phrases like
  'alle Lichter aus' or 'spiel jazz im wohnzimmer' to the right tool.
  English instructions stay in the prompt regardless. Pick up to two."
- `error.too_many_trigger_languages`: "Pick at most two languages."

## Hint block contents

Each block mirrors the structure of the existing English prompt's
trigger-routing sections — terse, keyword-driven, tool-routing only.
Tool names stay English (function identifiers, not user copy).

Categories per language × mode:
1. Discovery: list lights/areas, "any X on", "which X open"
2. Whole-area + whole-home actions
3. Weather (`weather in <place>`)
4. Memory (`remember`, `forget`, `recall`)
5. Music (`play_music`, `media_command`)
6. Timers (`start_timer`, `cancel_timer`, `timer_status`)

### Example — German default

```
[GERMAN TRIGGER HINTS]
- 'welche Lichter sind an' / 'sind Lichter an' → list_entities(domain='light', state='on'). Never answer single-light.
- 'sind Fenster offen' → list_entities(domain='binary_sensor', state='open').
- 'alle Lichter aus' / 'alles aus' → set_area_state(area='all', domain='light', action='turn_off').
- 'Lichter in der Küche aus' → set_area_state(area='kitchen', domain='light', action='turn_off').
- 'Wetter in <Ort>' → web_search('weather in <Ort>', near_user=false).
- 'erinnere dich' / 'merk dir' → remember. 'vergiss' → forget. 'was weißt du über mich' → recall.
- 'spiel <X> in <Raum>' / 'musik in <Raum>' → play_music. 'pausiere' / 'weiter' / 'überspring' → media_command.
- 'stell einen Timer für N Minuten' → start_timer(minutes=N). 'wie lange noch' → timer_status.
```

Voice variant: same triggers, no parenthetical explanations, ~60% the
length.

### Authoring plan for fr/es/pt/pl
- Translate the German block via local Qwen on TheBrain (Ollama
  `homeassistant:latest`).
- Manually skim each before commit — catch mistranslated tool names or
  reversed grammar that would confuse the LLM.
- Aim ~10–15 lines per language per mode.

### Token budget

Rough, based on current orchestrator log of ~1485 tokens for the
existing bilingual EN+DE prompt:

| Selected langs | Tokens (est) | Δ vs current |
|---|---|---|
| `[]`           | ~1100 | -25 % |
| `["de"]`       | ~1300 | -12 % |
| `["de","pl"]`  | ~1500 | ≈ even |

So one-language users get a smaller prompt, two-language users break
even with today's bilingual baseline, three-plus is blocked at the UI.

## Backward compatibility

- Existing `config_entry.options` without `trigger_languages` →
  `_default_trigger_langs(hass)` resolves on every prompt build. No
  migration step.
- DE-speaking HA → preselects `["de"]` → behavior matches today.
- EN-speaking HA → `[]` → English-only prompt. Multilingual models
  still reply in the input language; small-model users in any other
  language can opt in to a hint block.
- `CONF_SYSTEM_PROMPT` (custom user prompt override) is unaffected. It
  still appends after the assembled base + hint blocks.

## Testing

### Unit (`tests/components/ai_plugin/`)
1. **`test_prompts_i18n.py`** (new):
   - `_default_trigger_langs(hass="de-DE")` → `["de"]`
   - `_default_trigger_langs(hass="en-US")` → `[]`
   - `_default_trigger_langs(hass="zh-CN")` → `[]`
   - `_default_trigger_langs(hass=None)` / empty → `[]`
   - `_build_system_prompt` with `selected=["de"]` voice=False contains
     `"GERMAN TRIGGER HINTS"`; voice=True contains the voice variant.
   - `selected=[]` contains no `*** TRIGGER HINTS` marker.
   - `selected=["de","pl"]` contains both blocks; order matches input.
   - `selected=["xx"]` (unsupported) is skipped silently.
2. **`test_config_flow.py`** (extend):
   - 3-element list → form error `too_many_trigger_languages`.
   - `["de"]` → saved, persisted across re-read.
   - `[]` → saved as empty list (English-only).
3. **`test_const.py`** (extend or new):
   - For each `lang in SUPPORTED_TRIGGER_LANGUAGES`,
     `PROMPT_HINTS_I18N[lang]` has both `default` and `voice` non-empty
     strings.

### Manual smoke (HA after deploy)
- Reload with `[]` and DE STT pipeline. Ask "spiel jazz im
  wohnzimmer" — expect play_music routing on multilingual models.
- Set to `["de"]`. Verify orchestrator's
  `AI Plugin budget conv=... sys=NNNN` log shows the larger prompt.
- Set to `["de","pl"]`. Both blocks visible in prompt token count
  bump.
- Set to `["de","pl","fr"]` via UI → form error.
- Fresh install on `pl-PL` HA → preselects `["pl"]`.

## Edge cases

- `hass.config.language` unset or `None`: `(... or "").split("-")[0]`
  → empty string → not in supported set → `[]`.
- User edits YAML/storage to inject an unsupported lang code:
  `PROMPT_HINTS_I18N.get(lang)` → None → skipped, `_LOGGER.warning`
  emitted once per request.
- User picks 0 languages on a non-EN HA: legitimate — they get the EN
  base prompt and rely on the multilingual model. No warning.
- Custom `CONF_SYSTEM_PROMPT` set: unaffected. Hint blocks still
  append because they sit in the BASE assembly path, not the user
  override path.

## Rollout

| File | Change |
|---|---|
| `custom_components/ai_plugin/const.py` | Strip non-EN fragments from prompts. Add `CONF_TRIGGER_LANGUAGES`, `SUPPORTED_TRIGGER_LANGUAGES`, `PROMPT_HINTS_I18N` (de/fr/es/pt/pl × default+voice). Add `_default_trigger_langs`. |
| `custom_components/ai_plugin/config_flow.py` | Import new const symbols. Add SelectSelector to advanced schema. Validate `len ≤ 2` → `ERROR_TOO_MANY_TRIGGER_LANGUAGES`. |
| `custom_components/ai_plugin/orchestrator.py` | `_build_system_prompt`: read option, append per-lang hint blocks. |
| `custom_components/ai_plugin/strings.json` + `translations/en.json` | Add labels, description, error key. |
| `tests/components/ai_plugin/test_prompts_i18n.py` | New file (unit tests above). |
| `tests/components/ai_plugin/test_config_flow.py` | Extend with the 3 selector cases. |
| `manifest.json` | 0.7.7 → 0.8.0 |

Single PR, single tag `v0.8.0`. HACS picks up. No breaking change for
existing entries.

## Risks

- New language blocks unreviewed by native speakers may carry minor
  awkwardness. Low impact: terse keyword hints for tool routing, not
  user-facing text. Iterate post-release with native-speaker feedback.
- Token budget assumes hint blocks stay short. If a translation
  inflates a block (e.g. compound German nouns expanded explicitly),
  rein it in during review.
- Selector defaults vary by HA system language — could surprise an
  English-speaking user whose HA happens to be in Polish. Acceptable:
  options are visible and editable on first config.
