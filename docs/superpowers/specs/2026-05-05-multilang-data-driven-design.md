# Data-driven multilingual support — Design Spec

**Date:** 2026-05-05
**Target version:** v0.9.0
**Status:** approved (sections 1–5)

## Goal

Replace AI Plugin's scattered, prompt-bloating per-language localization
with a single data-driven layer rooted in YAML files under
`custom_components/ai_plugin/i18n/`. Adding a new language must be a
PR-friendly YAML edit with no Python or prompt changes. Current 5
languages (DE, FR, ES, PT, PL) plus English baseline must keep working at
or above v0.8.7 eval pass rates.

## Constraints (locked from brainstorm Q1–Q5)

| # | Question | Choice | Rationale |
|---|---|---|---|
| 1 | Scope | B — refactor + open extensibility for PR contributors | Future-proof; per the user request to "support multiple languages with the option to add more". |
| 2 | Language detection authority | D — HA pipeline `language` is authoritative; no message-content heuristic | Eval showed `_detect_lang` is brittle (overlapping accents); the HA Assist pipeline already carries the authoritative language per request. |
| 3 | Where data lives | C — per-language YAML files in `i18n/` folder | PR-friendly for non-Python contributors; schema-validated. |
| 4 | Regex contributor friendliness | D — hybrid: keyword lists for the easy 80%, raw regex escape hatch for the rare 20% | Most contributors only need keyword lists; engine compiles them safely with word boundaries. |
| 5 | Migration of existing `CONF_TRIGGER_LANGUAGES` installs | C — hard remove in v0.9.0, migration note in CHANGELOG | The option is internal config — no UX harm in removing it cleanly. |

## Architecture

```
custom_components/ai_plugin/
├── i18n/
│   ├── __init__.py          # public API: L.label/template/keyword_re/pattern_list
│   ├── _loader.py           # discover *.yaml, parse, validate, compile regexes
│   ├── _schema.py           # voluptuous schema for the YAML structure
│   ├── CONTRIBUTING.md      # how-to-add-a-language guide
│   ├── en.yaml              # canonical reference + universal fallback
│   ├── de.yaml
│   ├── fr.yaml
│   ├── es.yaml
│   ├── pt.yaml
│   └── pl.yaml
├── shortcuts.py             # uses L.compiled_re/template/label
├── orchestrator.py          # passes hass-pipeline language to shortcuts
└── const.py                 # PROMPT_HINTS_I18N + CONF_TRIGGER_LANGUAGES REMOVED
```

**Boot flow:**
1. `i18n/__init__.py` calls `_loader.load_all()` at import time.
2. Loader globs `*.yaml`, parses each, schema-validates, compiles regex fields.
3. Result: `LOCALIZATIONS: dict[str, LangData]` exposed via singleton `L`.
4. Bad YAML in any file → import-time `LocalizationError`, AI Plugin fails to load with clear "Invalid i18n/<lang>.yaml at key X" message.

**Boundaries:**
- `i18n/` is a **pure data + lookup** module. No HA imports. Testable in isolation.
- `shortcuts.py` + `orchestrator.py` are sole consumers — they pass `lang` (from HA pipeline) and read templates/regexes via `L`.
- `const.py` keeps prompt strings (English base only) — no per-language data.

**English is special.** `en.yaml` is canonical reference + universal fallback. Schema requires every other lang YAML to have at least the same top-level keys as `en.yaml` (loader enforces; missing keys fall back at runtime).

## YAML schema (data model)

Each `i18n/<lang>.yaml`:

```yaml
meta:
  code: "fr"           # ISO 639-1
  name: "Français"     # display name (HACS / contributor docs)
  contributors: ["@username"]   # optional credits

# Translatable labels for sensor attributes (used by attr-in-area shortcut).
labels:
  temperature: "température"
  humidity: "humidité"
  co2: "CO₂"
  illuminance: "luminosité"
  pressure: "pression"
  power: "puissance"
  energy: "consommation"
  battery: "batterie"
  brightness: "luminosité"

# Sentence templates with named placeholders. Used by deterministic shortcuts.
# Engine substitutes {label}, {val}, {unit}, {time}, {area}, etc.
templates:
  attr_state:        "La {label} est de {val}{unit}."
  attr_pct:          "La {label} est à {val}%."
  attr_humidity_fb:  "La {label} est de {val}%."
  sun_set_at:        "Le soleil se couche à {time}."
  sun_rise_at:       "Le soleil se lève à {time}."
  sun_is_up:         "Le soleil est levé. Coucher du soleil à {time}."
  sun_is_down:       "Le soleil est couché. Lever du soleil à {time}."
  fallback_no_sensor: "Je n'ai pas de capteur de {label} dans {area}."

# Keyword lists (engine compiles to word-boundary regex).
keywords:
  sun_set:       ["se couche le soleil", "coucher du soleil"]
  sun_rise:      ["se lève le soleil", "lever du soleil"]
  sun_dark:      ["fait-il nuit", "fait-il jour"]
  sun_is_up:     ["le soleil est-il levé"]
  narration:     ["je vérifie", "je cherche", "je regarde", "je consulte"]
  area_prefixes: ["dans la", "dans le", "dans l'", "en"]

# Raw regex escape hatch (rarely used). Each compiled with
# re.IGNORECASE | re.MULTILINE.
patterns:
  narration_full: []
  sun_full:       []
```

**Schema invariants** (enforced by `_schema.py`):
- `meta.code` must equal the filename stem.
- All keys in `labels`, `templates`, `keywords` must exist in `en.yaml`.
- Templates must contain only declared placeholders (`{label}`, `{val}`, `{unit}`, `{time}`, `{area}`).
- Keyword lists are non-empty strings, lowercased on load.
- `patterns.*` are valid Python regexes (compile-test at load).

## Loader + helper API

### `i18n/_loader.py` — startup pipeline

```python
def load_all() -> dict[str, LangData]:
    """Discover, parse, validate, compile. Called once at import time."""
    here = Path(__file__).parent
    out: dict[str, LangData] = {}
    en_data = _load_one(here / "en.yaml")
    out["en"] = en_data
    for path in sorted(here.glob("*.yaml")):
        if path.stem == "en":
            continue
        data = _load_one(path)
        _check_against_reference(data, en_data, path)
        out[data.code] = data
    return out


def _load_one(path: Path) -> LangData:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = SCHEMA(raw)                        # voluptuous validation
    if raw["meta"]["code"] != path.stem:
        raise LocalizationError(f"{path}: meta.code must equal filename stem")
    compiled_keywords = {k: _compile_kw_list(v) for k, v in raw["keywords"].items()}
    compiled_patterns = {k: [re.compile(p, re.IGNORECASE | re.MULTILINE)
                              for p in v]
                          for k, v in raw["patterns"].items()}
    return LangData(
        code=raw["meta"]["code"],
        labels=raw["labels"],
        templates=raw["templates"],
        keywords=raw["keywords"],
        keyword_re=compiled_keywords,
        pattern_re=compiled_patterns,
    )


def _compile_kw_list(words: list[str]) -> re.Pattern:
    """Literal phrases → single alternation with word boundaries."""
    escaped = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)
```

### `i18n/__init__.py` — public façade

```python
LOCALIZATIONS: dict[str, LangData] = _loader.load_all()
SUPPORTED_LANGS: tuple[str, ...] = tuple(LOCALIZATIONS.keys())


class _Lookup:
    """Singleton façade. All consumer code goes through this."""

    def label(self, key: str, lang: str) -> str:
        return self._get(lang, "labels", key) \
            or LOCALIZATIONS["en"].labels.get(key, key)

    def template(self, key: str, lang: str, **fmt) -> str:
        tmpl = self._get(lang, "templates", key) \
            or LOCALIZATIONS["en"].templates[key]
        return tmpl.format(**fmt)

    def keyword_re(self, key: str, lang: str) -> re.Pattern | None:
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return data.keyword_re.get(key)

    def pattern_list(self, key: str, lang: str) -> list[re.Pattern]:
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return data.pattern_re.get(key, [])

    def _get(self, lang: str, section: str, key: str):
        data = LOCALIZATIONS.get(lang)
        return getattr(data, section, {}).get(key) if data else None


L = _Lookup()
```

### Fallback policy

| Lookup | Missing in lang | Fallback |
|---|---|---|
| `L.label("temperature", "fr")` | not in fr.yaml | `en.yaml` value, else key as literal |
| `L.template("sun_set_at", "fr", time=…)` | not in fr.yaml | English template formatted |
| `L.keyword_re("sun_set", "fr")` | not in fr.yaml | English keyword regex |
| `L.pattern_list("narration_full", "fr")` | empty | `[]` (no patterns) |
| Unknown lang code | n/a | English data |

### Performance

- One-time load at module import (~10 ms total for 6 YAMLs + regex compile).
- All lookups are dict + attribute access (~100 ns).
- No I/O at request time.

## Migration + wiring

### Removals (v0.9.0 hard breaks)

| Remove | Reason | Migration |
|---|---|---|
| `PROMPT_HINTS_I18N` dict in `const.py` (~280 LOC) | Hint blocks proven not load-bearing on modern LLMs (DE/FR eval) | Migrated keyword phrases distributed into `i18n/<lang>.yaml` `keywords:` lists where applicable |
| `CONF_TRIGGER_LANGUAGES` config option | No longer needed — language now comes from HA pipeline per request | Hard-removed from schema; existing keys silently ignored at config-entry load with one-time logger warning |
| `default_trigger_langs(hass)` helper | Dead code | Deleted |
| `_DE_LABEL`, `_FR_LABEL`, `_DE_HINT_RE`, `_FR_HINT_RE` in `shortcuts.py` | Replaced by `i18n/` data | Deleted |
| `_detect_lang(message)` heuristic in `shortcuts.py` | HA pipeline language now authoritative | Deleted |
| Per-language hint injection in `_build_system_prompt` | Hint blocks gone | Code path simplified — base prompt only |

### Wiring changes

**`orchestrator.py`:**

```python
async def async_process(self, message, conversation_id, language=None, ...):
    lang = (language or "en").split("-")[0].lower()       # "de-DE" → "de"
    ...
    shortcut_reply = try_shortcut(self._hass, message, lang=lang)
    media_result = await async_try_media_shortcut(self._hass, message, lang=lang)
    ...
    stored_reply = self._strip_narration(reply, lang=lang)
```

**`shortcuts.py`:**

```python
def try_shortcut(hass, message, *, lang: str = "en") -> str | None:
    sun_reply = _try_sun_shortcut(hass, message, lang)
    ...

def _try_sun_shortcut(hass, message, lang: str) -> str | None:
    if not L.keyword_re("sun_set", lang).search(message) and \
       not L.keyword_re("sun_rise", lang).search(message) and \
       not L.keyword_re("sun_dark", lang).search(message):
        return None
    ...
    return L.template("sun_set_at", lang, time=next_setting)


def _format_state(state_obj, spec, lang: str = "en") -> str | None:
    label_key = spec["label"]
    label = L.label(label_key, lang)
    val = _round_numeric(state_obj.state)
    return L.template("attr_state", lang, label=label, val=val, unit=unit)
```

**`orchestrator._strip_narration`:**

```python
def _strip_narration(text: str, lang: str = "en") -> str:
    patterns: list[re.Pattern] = L.pattern_list("narration_full", lang) + [
        re for re in [L.keyword_re("narration", lang)] if re is not None
    ]
    cleaned = text
    for p in patterns:
        cleaned = p.sub("", cleaned)
    return cleaned.strip()
```

### CHANGELOG entry

```markdown
## v0.9.0 — Data-driven multilingual support

**Breaking:**
- `trigger_languages` config option removed. Language now follows the HA Assist pipeline (`Buddy (DE)`, `Buddy (FR)`, etc.).

**New:**
- All per-language data lives in `custom_components/ai_plugin/i18n/<code>.yaml`. Adding a new language is a YAML PR — no Python changes.
- Schema-validated at load time; bad YAML fails the integration cleanly.
- 5 languages shipped: en, de, fr, es, pt, pl. Contributors can add more via PR — see `i18n/CONTRIBUTING.md`.

**Changed:**
- Per-language hint blocks dropped from the system prompt (~80–160 tokens saved per request).
- Sun shortcut, narration regex, attribute-in-area shortcut now language-symmetric — every supported language gets the same code paths and templates.
```

## Testing

**Unit tests** (`tests/components/ai_plugin/test_i18n.py`, new):
- `test_loader_discovers_all_yamls` — scans dir, returns all 6 langs.
- `test_loader_validates_schema` — bad YAML raises `LocalizationError`.
- `test_loader_filename_matches_meta_code` — `de.yaml` with `meta.code: "fr"` raises.
- `test_loader_compiles_keyword_regexes` — `keywords.sun_set` becomes `re.Pattern`.
- `test_reference_completeness` — every key in `en.yaml` exists in other lang yamls (warns on miss, doesn't fail — fallback design).
- `test_template_placeholders_valid` — `templates.attr_state` only references declared placeholders.

**Lookup tests** (`test_l_lookup.py`, new):
- `test_label_returns_lang_value`
- `test_label_falls_back_to_en`
- `test_template_format`
- `test_unknown_lang_falls_back_to_en`
- `test_keyword_re_word_boundaries`

**Integration tests** (extend existing `test_prompts_i18n.py`):
- `test_no_prompt_hints_in_system_prompt` — assert hint section markers absent.
- `test_shortcut_returns_localized_string` — feed `"wann geht die Sonne unter"` with `lang="de"` → returns DE string.
- `test_orchestrator_passes_lang_to_shortcut` — mock shortcut, verify `lang="fr"` flows through `async_process`.

**Migration tests** (`test_migration.py`, new):
- `test_old_config_entry_loads_without_trigger_languages` — entry with `{"trigger_languages": ["de"], ...}` loads cleanly post-removal.

**Eval coverage:** re-run `/tmp/sat1_eval.py --corpus en|de|fr` after refactor. Sweet spot: 50 cases × 3 langs = 150 cases. Pass rate must not regress vs v0.8.7 baseline (~84% en, ~60% de, ~58% fr). ES/PT new — set baseline.

## Contributor docs (`i18n/CONTRIBUTING.md`)

```markdown
# Adding a language to AI Plugin

## TL;DR
1. Copy `i18n/en.yaml` to `i18n/<code>.yaml` (ISO 639-1 lowercase, e.g. `nl`, `it`, `sv`).
2. Translate every value. Keep `meta.code` = filename stem.
3. Submit PR.

## Detail
- Templates contain `{placeholder}` slots — keep them. Order can change to fit grammar.
- Keyword lists must be lowercased — engine matches case-insensitively but writes phrases lowercase for clarity.
- Punctuation, accents, and unicode are fine and encouraged where natural.
- Patterns (the escape hatch) — leave empty unless you NEED a regex the keyword list can't express. Test it locally first.

## Testing your file
\`\`\`bash
python -c "from custom_components.ai_plugin.i18n import L, SUPPORTED_LANGS; print(SUPPORTED_LANGS)"
\`\`\`
Should list your new code without raising.

## Approval bar
PRs welcome. Maintainer reviews for accuracy + grammar + style consistency. Do not bundle other changes.
```

## Out of scope (deferred)

- Runtime/UI-driven custom languages (would need hot-reload + per-entry override design).
- Pluralization rules (English/German/French handle the current shortcuts uniformly; Slavic/Arabic plurals would need a separate engine).
- Per-region variants (`de-CH` vs `de-DE`; `fr-CA` vs `fr-FR`) — current design uses bare ISO 639-1 only.

## Open prerequisites (verify before plan)

- `voluptuous` is already a transitive dependency via Home Assistant — confirm before relying on it for schema validation.
- `pyyaml` is in HA core dependencies — confirm import works inside custom_components context.
- HA conversation/process WS API reliably passes `language` per request — confirmed in DE/FR evals (this session) where `language=de` and `language=fr` were honored end-to-end.
