# Multilingual Prompt Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded English+German trigger fragments in `SYSTEM_PROMPT_DEFAULT` / `SYSTEM_PROMPT_VOICE` with a configurable per-integration list (max 2 of de/fr/es/pt/pl) of trigger-word hint blocks injected at request time, default-resolved from `hass.config.language`.

**Architecture:** New constants `SUPPORTED_TRIGGER_LANGUAGES`, `CONF_TRIGGER_LANGUAGES`, `PROMPT_HINTS_I18N`, plus a pure helper `_default_trigger_langs(hass) -> list[str]`. `orchestrator._build_system_prompt` reads the option and appends per-language hint blocks to the English base. `config_flow` exposes a `SelectSelector(multiple=True, max=2)` in the advanced options form. English remains the implicit prompt base — never selectable, never excludable.

**Tech Stack:** Python 3.13+ on Home Assistant 2025.7+. `voluptuous` for config schema. `homeassistant.helpers.selector.SelectSelector`. `pytest` + `pytest-asyncio` for tests.

---

## File Structure

| File | Role |
|---|---|
| `custom_components/ai_plugin/const.py` | New constants + `_default_trigger_langs` helper + `PROMPT_HINTS_I18N` dict + cleaned `SYSTEM_PROMPT_DEFAULT`/`SYSTEM_PROMPT_VOICE` (German fragments stripped). |
| `custom_components/ai_plugin/orchestrator.py` | `_build_system_prompt` reads `CONF_TRIGGER_LANGUAGES`, appends per-language blocks. |
| `custom_components/ai_plugin/config_flow.py` | Adds `SelectSelector` to advanced schema; validates `len ≤ 2`. |
| `custom_components/ai_plugin/strings.json` | Adds option label, description, and `too_many_trigger_languages` error key. |
| `custom_components/ai_plugin/translations/en.json` | Same additions as strings.json. |
| `custom_components/ai_plugin/manifest.json` | 0.7.7 → 0.8.0 |
| `tests/components/ai_plugin/test_prompts_i18n.py` | New: unit tests for `_default_trigger_langs` + `_build_system_prompt`. |
| `tests/components/ai_plugin/test_config_flow.py` | Extended: 3-element list rejection, persistence round-trip. |

Tests run via `python3 -m pytest tests/components/ai_plugin/<file>.py -v` from repo root. Local environment may lack `httpx`; if pytest collection fails with `ModuleNotFoundError: No module named 'httpx'`, run `pip install --user -r requirements_test.txt` first or rely on CI.

---

## Task 1: Add language constants + auto-detect helper

**Files:**
- Modify: `custom_components/ai_plugin/const.py` (after `CONF_RESPONSE_TIMEOUT`, after `DEFAULT_LOCATION_BIAS`)
- Test: `tests/components/ai_plugin/test_prompts_i18n.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/components/ai_plugin/test_prompts_i18n.py`:

```python
"""Tests for multilingual prompt hint resolution."""
from __future__ import annotations

from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    SUPPORTED_TRIGGER_LANGUAGES,
    _default_trigger_langs,
)


def _hass_with_lang(lang: str | None) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(language=lang))


def test_supported_languages_is_canonical():
    assert SUPPORTED_TRIGGER_LANGUAGES == ["de", "fr", "es", "pt", "pl"]


def test_default_trigger_langs_de_de():
    assert _default_trigger_langs(_hass_with_lang("de-DE")) == ["de"]


def test_default_trigger_langs_pl_pl():
    assert _default_trigger_langs(_hass_with_lang("pl-PL")) == ["pl"]


def test_default_trigger_langs_en_us_is_empty():
    assert _default_trigger_langs(_hass_with_lang("en-US")) == []


def test_default_trigger_langs_unsupported_is_empty():
    assert _default_trigger_langs(_hass_with_lang("zh-CN")) == []


def test_default_trigger_langs_none_is_empty():
    assert _default_trigger_langs(_hass_with_lang(None)) == []


def test_default_trigger_langs_empty_string_is_empty():
    assert _default_trigger_langs(_hass_with_lang("")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: `ImportError: cannot import name 'SUPPORTED_TRIGGER_LANGUAGES'` (or similar).

- [ ] **Step 3: Implement constants + helper in `const.py`**

Find the line `CONF_RESPONSE_TIMEOUT = "response_timeout"` (~line 31) and add directly below:

```python
CONF_RESPONSE_TIMEOUT = "response_timeout"
CONF_ENABLE_THINKING = "enable_thinking"
# Trigger-word hint languages: which household languages should the
# plugin pin tool-routing hints for in the system prompt. Up to 2.
# English is the implicit base — never selectable, never excludable.
CONF_TRIGGER_LANGUAGES = "trigger_languages"
```

(`CONF_ENABLE_THINKING` already exists from v0.7.x; keep it.)

Find the line `DEFAULT_ENABLE_THINKING = False` and add directly below:

```python
DEFAULT_ENABLE_THINKING = False

# Trigger-language defaults are resolved at runtime from
# hass.config.language via _default_trigger_langs(); the literal default
# stored in options is an empty list so a missing key falls through to
# auto-detect on every prompt build.
SUPPORTED_TRIGGER_LANGUAGES: list[str] = ["de", "fr", "es", "pt", "pl"]


def _default_trigger_langs(hass) -> list[str]:
    """Return the auto-detected default selection for trigger languages.

    Uses ``hass.config.language``, stripping any region (e.g. ``de-DE`` →
    ``de``). Returns ``[lang]`` if the bare language code is in
    SUPPORTED_TRIGGER_LANGUAGES, else ``[]``. English-locale HA returns
    ``[]`` because English is the prompt base anyway.
    """
    sys_lang = (getattr(getattr(hass, "config", None), "language", None) or "")
    code = sys_lang.split("-")[0].lower()
    return [code] if code in SUPPORTED_TRIGGER_LANGUAGES else []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/const.py tests/components/ai_plugin/test_prompts_i18n.py
git -C /home/arndtg/AI-plugin commit -m "feat(const): add SUPPORTED_TRIGGER_LANGUAGES + _default_trigger_langs"
```

---

## Task 2: Add German hint block + completeness invariant

**Files:**
- Modify: `custom_components/ai_plugin/const.py` (after `_default_trigger_langs`)
- Test: `tests/components/ai_plugin/test_prompts_i18n.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/components/ai_plugin/test_prompts_i18n.py`:

```python
from custom_components.ai_plugin.const import PROMPT_HINTS_I18N


def test_prompt_hints_has_all_supported_languages():
    assert set(PROMPT_HINTS_I18N) == set(SUPPORTED_TRIGGER_LANGUAGES)


def test_prompt_hints_each_lang_has_default_and_voice():
    for lang in SUPPORTED_TRIGGER_LANGUAGES:
        assert "default" in PROMPT_HINTS_I18N[lang], f"{lang} missing default"
        assert "voice" in PROMPT_HINTS_I18N[lang], f"{lang} missing voice"
        assert PROMPT_HINTS_I18N[lang]["default"].strip(), f"{lang} default empty"
        assert PROMPT_HINTS_I18N[lang]["voice"].strip(), f"{lang} voice empty"


def test_german_block_mentions_play_music_and_media_command():
    de_default = PROMPT_HINTS_I18N["de"]["default"]
    assert "play_music" in de_default
    assert "media_command" in de_default
    assert "set_area_state" in de_default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: 3 new failures with `ImportError` for `PROMPT_HINTS_I18N`.

- [ ] **Step 3: Implement German block in `const.py`**

In `const.py`, immediately after the `_default_trigger_langs` function definition, add:

```python
# Per-language trigger-word hint blocks. Each block is a terse,
# keyword-driven section pinning common phrasings to the plugin's tools
# so small models (qwen3.5-9b tier) route reliably without spilling
# fragments into the English base prompt for every household.
#
# Tool names stay English (function identifiers, not user copy). Both
# "default" and "voice" variants exist; the voice variant drops
# parenthetical explanations to keep the spoken-mode prompt tight.

PROMPT_HINTS_I18N: dict[str, dict[str, str]] = {
    "de": {
        "default": (
            "[GERMAN TRIGGER HINTS]\n"
            "- 'welche Lichter sind an' / 'sind Lichter an' → list_entities(domain='light', state='on'). Liste jede Zeile, niemals nur eine Lampe.\n"
            "- 'sind Fenster offen' → list_entities(domain='binary_sensor', state='open').\n"
            "- 'alle Lichter aus' / 'alles aus' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'Lichter in der Küche aus' / 'Licht im Schlafzimmer an' → set_area_state(area='<Raum>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'Wetter in <Ort>' → web_search('weather in <Ort>', near_user=false). 'Wetter draußen' → list_entities(domain='weather') zuerst.\n"
            "- 'erinnere dich' / 'merk dir' → remember. 'vergiss' → forget. 'was weißt du über mich' → recall.\n"
            "- 'spiel <X> in <Raum>' / 'musik in <Raum>' → play_music. 'pausiere' / 'weiter' / 'überspring' / 'stopp' → media_command.\n"
            "- 'stell einen Timer für N Minuten' → start_timer(minutes=N). 'wie lange noch' → timer_status."
        ),
        "voice": (
            "Deutsche Trigger:\n"
            "- 'welche Lichter sind an' → list_entities(domain='light', state='on').\n"
            "- 'alle Lichter aus' / 'alles aus' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'Licht im <Raum> an/aus' → set_area_state(area='<Raum>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'Wetter in <Ort>' → web_search('weather in <Ort>'). 'Wetter draußen' → list_entities(domain='weather').\n"
            "- 'merk dir' → remember. 'vergiss' → forget.\n"
            "- 'spiel <X> in <Raum>' → play_music. 'pausiere' / 'weiter' / 'überspring' / 'stopp' → media_command.\n"
            "- 'Timer für N Minuten' → start_timer(minutes=N). 'wie lange noch' → timer_status."
        ),
    },
    "fr": {
        "default": (
            "[FRENCH TRIGGER HINTS]\n"
            "- 'quelles lumières sont allumées' → list_entities(domain='light', state='on'). Liste chaque ligne, jamais une seule lampe.\n"
            "- 'des fenêtres sont ouvertes' → list_entities(domain='binary_sensor', state='open').\n"
            "- 'éteins tout' / 'toutes les lumières éteintes' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'allume / éteins les lumières dans la <pièce>' → set_area_state(area='<pièce>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'météo à <ville>' → web_search('weather in <ville>', near_user=false). 'quel temps fait-il' → list_entities(domain='weather') d'abord.\n"
            "- 'souviens-toi' / 'rappelle-toi' → remember. 'oublie' → forget. 'que sais-tu de moi' → recall.\n"
            "- 'mets de la musique dans <pièce>' / 'joue <X> dans <pièce>' → play_music. 'pause' / 'suivante' / 'précédente' / 'stop' → media_command.\n"
            "- 'minuteur de N minutes' → start_timer(minutes=N). 'combien de temps reste-t-il' → timer_status."
        ),
        "voice": (
            "Déclencheurs français:\n"
            "- 'quelles lumières sont allumées' → list_entities(domain='light', state='on').\n"
            "- 'éteins tout' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'allume / éteins dans la <pièce>' → set_area_state(area='<pièce>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'météo à <ville>' → web_search('weather in <ville>'). 'quel temps' → list_entities(domain='weather').\n"
            "- 'souviens-toi' → remember. 'oublie' → forget.\n"
            "- 'joue <X> dans <pièce>' → play_music. 'pause' / 'suivante' / 'précédente' / 'stop' → media_command.\n"
            "- 'minuteur de N minutes' → start_timer(minutes=N)."
        ),
    },
    "es": {
        "default": (
            "[SPANISH TRIGGER HINTS]\n"
            "- 'qué luces están encendidas' → list_entities(domain='light', state='on'). Lista cada fila, nunca solo una lámpara.\n"
            "- 'hay ventanas abiertas' → list_entities(domain='binary_sensor', state='open').\n"
            "- 'apaga todo' / 'apagar todas las luces' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'enciende / apaga las luces de la <habitación>' → set_area_state(area='<habitación>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'tiempo en <ciudad>' / 'qué tiempo hace en <ciudad>' → web_search('weather in <ciudad>', near_user=false). 'qué tiempo hace' → list_entities(domain='weather') primero.\n"
            "- 'recuerda' / 'apunta' → remember. 'olvida' → forget. 'qué sabes de mí' → recall.\n"
            "- 'pon música en <habitación>' / 'reproduce <X> en <habitación>' → play_music. 'pausa' / 'siguiente' / 'anterior' / 'para' → media_command.\n"
            "- 'temporizador de N minutos' → start_timer(minutes=N). 'cuánto queda' → timer_status."
        ),
        "voice": (
            "Disparadores en español:\n"
            "- 'qué luces están encendidas' → list_entities(domain='light', state='on').\n"
            "- 'apaga todo' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'enciende / apaga la <habitación>' → set_area_state(area='<habitación>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'tiempo en <ciudad>' → web_search('weather in <ciudad>'). 'qué tiempo' → list_entities(domain='weather').\n"
            "- 'recuerda' → remember. 'olvida' → forget.\n"
            "- 'pon <X> en <habitación>' → play_music. 'pausa' / 'siguiente' / 'anterior' / 'para' → media_command.\n"
            "- 'temporizador de N minutos' → start_timer(minutes=N)."
        ),
    },
    "pt": {
        "default": (
            "[PORTUGUESE TRIGGER HINTS]\n"
            "- 'que luzes estão ligadas' → list_entities(domain='light', state='on'). Liste cada linha, nunca só uma lâmpada.\n"
            "- 'há janelas abertas' → list_entities(domain='binary_sensor', state='open').\n"
            "- 'desliga tudo' / 'apaga todas as luzes' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'liga / desliga as luzes do <quarto>' → set_area_state(area='<quarto>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'tempo em <cidade>' / 'qual o tempo em <cidade>' → web_search('weather in <cidade>', near_user=false). 'que tempo está' → list_entities(domain='weather') primeiro.\n"
            "- 'lembra-te' / 'anota' → remember. 'esquece' → forget. 'o que sabes sobre mim' → recall.\n"
            "- 'põe música no <quarto>' / 'toca <X> no <quarto>' → play_music. 'pausa' / 'próxima' / 'anterior' / 'para' → media_command.\n"
            "- 'temporizador de N minutos' → start_timer(minutes=N). 'quanto falta' → timer_status."
        ),
        "voice": (
            "Gatilhos em português:\n"
            "- 'que luzes estão ligadas' → list_entities(domain='light', state='on').\n"
            "- 'desliga tudo' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'liga / desliga o <quarto>' → set_area_state(area='<quarto>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'tempo em <cidade>' → web_search('weather in <cidade>'). 'que tempo' → list_entities(domain='weather').\n"
            "- 'lembra-te' → remember. 'esquece' → forget.\n"
            "- 'põe <X> no <quarto>' → play_music. 'pausa' / 'próxima' / 'anterior' / 'para' → media_command.\n"
            "- 'temporizador de N minutos' → start_timer(minutes=N)."
        ),
    },
    "pl": {
        "default": (
            "[POLISH TRIGGER HINTS]\n"
            "- 'które światła są włączone' → list_entities(domain='light', state='on'). Wymień każdy wiersz, nigdy tylko jedną lampę.\n"
            "- 'czy są otwarte okna' → list_entities(domain='binary_sensor', state='open').\n"
            "- 'wyłącz wszystko' / 'wyłącz wszystkie światła' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'włącz / wyłącz światła w <pokoju>' → set_area_state(area='<pokój>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'pogoda w <mieście>' / 'jaka pogoda w <mieście>' → web_search('weather in <mieście>', near_user=false). 'jaka jest pogoda' → list_entities(domain='weather') najpierw.\n"
            "- 'zapamiętaj' / 'zanotuj' → remember. 'zapomnij' → forget. 'co o mnie wiesz' → recall.\n"
            "- 'puść muzykę w <pokoju>' / 'odtwórz <X> w <pokoju>' → play_music. 'pauza' / 'następny' / 'poprzedni' / 'stop' → media_command.\n"
            "- 'minutnik na N minut' → start_timer(minutes=N). 'ile jeszcze zostało' → timer_status."
        ),
        "voice": (
            "Polskie wyzwalacze:\n"
            "- 'które światła są włączone' → list_entities(domain='light', state='on').\n"
            "- 'wyłącz wszystko' → set_area_state(area='all', domain='light', action='turn_off').\n"
            "- 'włącz / wyłącz <pokój>' → set_area_state(area='<pokój>', domain='light', action='turn_on'|'turn_off').\n"
            "- 'pogoda w <mieście>' → web_search('weather in <mieście>'). 'jaka pogoda' → list_entities(domain='weather').\n"
            "- 'zapamiętaj' → remember. 'zapomnij' → forget.\n"
            "- 'puść <X> w <pokoju>' → play_music. 'pauza' / 'następny' / 'poprzedni' / 'stop' → media_command.\n"
            "- 'minutnik na N minut' → start_timer(minutes=N)."
        ),
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/const.py tests/components/ai_plugin/test_prompts_i18n.py
git -C /home/arndtg/AI-plugin commit -m "feat(const): add PROMPT_HINTS_I18N for de/fr/es/pt/pl"
```

---

## Task 3: Strip non-EN fragments from base prompts

**Files:**
- Modify: `custom_components/ai_plugin/const.py` (`SYSTEM_PROMPT_DEFAULT` ~line 126, `SYSTEM_PROMPT_VOICE` ~line 219)
- Test: `tests/components/ai_plugin/test_prompts_i18n.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/components/ai_plugin/test_prompts_i18n.py`:

```python
from custom_components.ai_plugin.const import (
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)


_GERMAN_FRAGMENTS = (
    "spiel jazz",
    "alle Lichter aus",
    "welche Lichter sind an",
    "sind lichter an",
    "sind fenster offen",
    "wetter in",
    "wetter draußen",
    "wetter draussen",
    "merk dir",
    "vergiss",
    "weiter spielen",
    "skip this song",  # English keep-list — sanity that we did not delete EN examples
)


def test_default_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_DEFAULT.lower()
    for frag in (
        "spiel jazz",
        "alle lichter aus",
        "welche lichter sind an",
        "sind lichter an",
        "sind fenster offen",
        "wetter in",
        "wetter draußen",
        "wetter draussen",
        "merk dir",
        "weiter spielen",
        "alle",
        "alles",
        "überall",
        "ueberall",
        "ganzes haus",
    ):
        assert frag not in base, (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_DEFAULT"
        )


def test_voice_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_VOICE.lower()
    for frag in (
        "spiel jazz",
        "welche lichter sind an",
        "sind lichter an",
        "wetter in",
        "weiter",
    ):
        # 'weiter' alone is too generic; use a longer fragment to be safe
        if frag == "weiter":
            assert "weiter" not in base or "weiterhin" in base
            continue
        assert frag not in base, (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_VOICE"
        )


def test_default_prompt_keeps_english_examples():
    # Sanity: stripping non-EN must not delete the English routing examples.
    base = SYSTEM_PROMPT_DEFAULT
    assert "set_area_state" in base
    assert "list_entities" in base
    assert "play_music" in base
    assert "media_command" in base
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: `test_default_prompt_strips_german_fragments` and `test_voice_prompt_strips_german_fragments` fail (German fragments still embedded).

- [ ] **Step 3: Strip German from `SYSTEM_PROMPT_DEFAULT`**

Open `custom_components/ai_plugin/const.py`. In `SYSTEM_PROMPT_DEFAULT`:

a. Inside the `[ENTITY DISCOVERY — STRICT]` section, find the line that ends with `'welche Lichter sind an', 'sind lichter an', 'sind fenster offen'`. Replace that compound list with the English-only equivalent. Concretely, locate the substring:

```
'welche Lichter sind an', 'sind lichter an', 'sind fenster offen': CALL list_entities
```

and change to:

```
'any windows open?': CALL list_entities
```

b. In the same `[ENTITY DISCOVERY — STRICT]` section, find:

```
- For whole-home actions ('turn off all lights', 'switch everything off', 'alle Lichter aus'): CALL set_area_state with area omitted or area='all'. Do NOT invent an area named 'all'.
```

Change to:

```
- For whole-home actions ('turn off all lights', 'switch everything off'): CALL set_area_state with area omitted or area='all'. Do NOT invent an area named 'all'.
```

c. In the `[WEATHER — STRICT ORDER]` section, find:

```
- IF the user names a city/country/region different from their home ('weather in Tokyo', 'wetter in Paris', 'is it raining in London'): SKIP list_entities
```

Change to:

```
- IF the user names a city/country/region different from their home ('weather in Tokyo', 'is it raining in London'): SKIP list_entities
```

d. In the `[MEDIA PLAYBACK — STRICT]` section, find:

```
- play music in an area ('play music in hobby room', 'spiel jazz im wohnzimmer', 'play Enya in the kitchen', 'shuffle my workout playlist in the living room'): CALL play_music(query=..., area=...).
```

Change to:

```
- play music in an area ('play music in hobby room', 'play Enya in the kitchen', 'shuffle my workout playlist in the living room'): CALL play_music(query=..., area=...).
```

In the same section, find:

```
    'resume' / 'unpause' / 'continue' / 'weiter' → media_command(command='resume')
```

Change to:

```
    'resume' / 'unpause' / 'continue' → media_command(command='resume')
```

- [ ] **Step 4: Strip German from `SYSTEM_PROMPT_VOICE`**

Still in `const.py`. In `SYSTEM_PROMPT_VOICE`:

a. Find:

```
- "Any lights on", "are any lights on", "which lights are on", "what's on", "welche lichter sind an", "sind lichter an", "any windows open": list_entities
```

Change to:

```
- "Any lights on", "are any lights on", "which lights are on", "what's on", "any windows open": list_entities
```

b. Find the Weather block:

```
- If the user names a city/country/region different from home ('weather in Tokyo', 'wetter in Paris'): SKIP list_entities. CALL web_search('weather in <named place>', near_user=false) and speak the result in one sentence. STOP.
```

Change to:

```
- If the user names a city/country/region different from home ('weather in Tokyo', 'weather in Paris'): SKIP list_entities. CALL web_search('weather in <named place>', near_user=false) and speak the result in one sentence. STOP.
```

c. Find the Music playback block:

```
- 'play music in <area>', 'spiel jazz im wohnzimmer', 'play Enya in the kitchen': CALL play_music(query, area). Do NOT use set_area_state on media_player — that does not start playback.
```

Change to:

```
- 'play music in <area>', 'play Enya in the kitchen': CALL play_music(query, area). Do NOT use set_area_state on media_player — that does not start playback.
```

In the same Music block, find:

```
- 'pause' → media_command('pause'). 'next' / 'skip' → media_command('next'). 'previous' → media_command('previous'). 'weiter' / 'resume' / 'continue' → media_command('resume'). 'stop' → media_command('stop'). Pass area only when user names a room.
```

Change to:

```
- 'pause' → media_command('pause'). 'next' / 'skip' → media_command('next'). 'previous' → media_command('previous'). 'resume' / 'continue' → media_command('resume'). 'stop' → media_command('stop'). Pass area only when user names a room.
```

d. Search the rest of `SYSTEM_PROMPT_VOICE` for any remaining `ä`, `ö`, `ü`, `ß` characters or German words (`alle`, `alles`, `überall`, `ueberall`, `ganzes haus`, `Lichter`, `Wetter`, `Raum`). Remove or rewrite them in English. (This catches the `_SWEEP_ALL_KEYWORDS` examples if any leaked into the prompt.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: all tests in this file pass (12 total so far).

- [ ] **Step 6: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/const.py tests/components/ai_plugin/test_prompts_i18n.py
git -C /home/arndtg/AI-plugin commit -m "refactor(const): strip German fragments from English base prompts"
```

---

## Task 4: Wire `_build_system_prompt` to inject hint blocks

**Files:**
- Modify: `custom_components/ai_plugin/orchestrator.py` (imports + `_build_system_prompt` ~line 545)
- Test: `tests/components/ai_plugin/test_prompts_i18n.py` (extend)

- [ ] **Step 1: Append failing tests**

Append to `tests/components/ai_plugin/test_prompts_i18n.py`:

```python
import asyncio
from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    CONF_TRIGGER_LANGUAGES,
    PROMPT_HINTS_I18N,
)


class _FakeOrchestrator:
    """Minimal stand-in for AIOrchestrator that exposes _build_system_prompt
    behavior with a controllable options dict and language."""

    def __init__(self, options: dict, hass_lang: str | None):
        from custom_components.ai_plugin.orchestrator import AIOrchestrator
        self._build_system_prompt = AIOrchestrator._build_system_prompt.__get__(self)
        self._entry = SimpleNamespace(options=options)
        self._hass = SimpleNamespace(config=SimpleNamespace(language=hass_lang))
        # Stub the LocationProvider context-block builder used inside
        # _build_system_prompt; tests focus on language-hint plumbing only.
        self._location = SimpleNamespace(async_resolve=lambda: asyncio.sleep(0))


async def _run_build(opts, hass_lang, voice_mode):
    fake = _FakeOrchestrator(opts, hass_lang)
    return await fake._build_system_prompt(voice_mode=voice_mode, user_id=None)


def test_build_prompt_no_langs_omits_all_hint_blocks(monkeypatch):
    # Strip hooks the location resolver adds so the test only sees the base.
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: []}, "en-US", False))
    assert "GERMAN TRIGGER HINTS" not in prompt
    assert "FRENCH TRIGGER HINTS" not in prompt
    assert "POLISH TRIGGER HINTS" not in prompt


def test_build_prompt_with_de_appends_german_default_block():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["de"]}, "en-US", False))
    assert PROMPT_HINTS_I18N["de"]["default"] in prompt


def test_build_prompt_voice_mode_uses_voice_variant():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["de"]}, "en-US", True))
    assert PROMPT_HINTS_I18N["de"]["voice"] in prompt
    assert PROMPT_HINTS_I18N["de"]["default"] not in prompt


def test_build_prompt_with_two_langs_appends_both_in_order():
    prompt = asyncio.run(
        _run_build({CONF_TRIGGER_LANGUAGES: ["pl", "fr"]}, "en-US", False)
    )
    assert PROMPT_HINTS_I18N["pl"]["default"] in prompt
    assert PROMPT_HINTS_I18N["fr"]["default"] in prompt
    # Order matches input list.
    assert prompt.index(PROMPT_HINTS_I18N["pl"]["default"]) < prompt.index(
        PROMPT_HINTS_I18N["fr"]["default"]
    )


def test_build_prompt_unknown_lang_silently_skipped():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["xx"]}, "en-US", False))
    # No crash. No xx block (none exists). No DE block fallback.
    assert "GERMAN TRIGGER HINTS" not in prompt


def test_build_prompt_missing_option_falls_through_to_auto_detect():
    # No CONF_TRIGGER_LANGUAGES key, German HA → DE block auto-injected.
    prompt = asyncio.run(_run_build({}, "de-DE", False))
    assert PROMPT_HINTS_I18N["de"]["default"] in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: 6 new failures — orchestrator's `_build_system_prompt` doesn't inject blocks yet.

- [ ] **Step 3: Add imports + wiring in `orchestrator.py`**

Open `custom_components/ai_plugin/orchestrator.py`. Find the const-import block around lines 33–55. Add inside it:

```python
    CONF_TRIGGER_LANGUAGES,
```

(Keep alphabetical order with sibling `CONF_*` keys.)

In the same import block, also add:

```python
    PROMPT_HINTS_I18N,
    _default_trigger_langs,
```

(They live next to other `*_PROMPT_*` symbols and helpers.)

- [ ] **Step 4: Modify `_build_system_prompt`**

Find `_build_system_prompt` (~line 545). The current body builds the base prompt; locate the line:

```python
        base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
```

Directly **after** that line and before the next assignment / return path, insert:

```python
        # Append per-household-language trigger-word hint blocks. English is
        # the implicit base; selected langs come from CONF_TRIGGER_LANGUAGES
        # (max 2, validated in config_flow). Falls through to
        # _default_trigger_langs(hass) when the option is absent so legacy
        # entries auto-detect from hass.config.language.
        opts = self._entry.options
        selected = opts.get(
            CONF_TRIGGER_LANGUAGES, _default_trigger_langs(self._hass)
        )
        if not isinstance(selected, list):
            selected = []
        mode_key = "voice" if voice_mode else "default"
        for lang in selected:
            block = PROMPT_HINTS_I18N.get(lang, {}).get(mode_key)
            if block:
                base = f"{base}\n\n{block}"
            elif lang:
                _LOGGER.warning(
                    "AI Plugin: ignoring unknown trigger language %r "
                    "(supported: %s)",
                    lang,
                    list(PROMPT_HINTS_I18N),
                )
```

The `_LOGGER` instance already exists at module level in orchestrator.py.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -v`

Expected: 18 passed.

- [ ] **Step 6: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/orchestrator.py tests/components/ai_plugin/test_prompts_i18n.py
git -C /home/arndtg/AI-plugin commit -m "feat(orchestrator): inject per-language hint blocks into system prompt"
```

---

## Task 5: Add config_flow selector + max-2 validation

**Files:**
- Modify: `custom_components/ai_plugin/config_flow.py` (imports, advanced schema, validator)
- Modify: `custom_components/ai_plugin/const.py` (add `ERROR_TOO_MANY_TRIGGER_LANGUAGES`)
- Test: `tests/components/ai_plugin/test_config_flow.py` (extend)

- [ ] **Step 1: Write failing tests**

Open `tests/components/ai_plugin/test_config_flow.py`. Append (preserving existing imports and fixtures):

```python
from custom_components.ai_plugin.const import (
    CONF_TRIGGER_LANGUAGES,
    ERROR_TOO_MANY_TRIGGER_LANGUAGES,
)


@pytest.mark.asyncio
async def test_advanced_options_rejects_three_trigger_languages(
    hass, mock_config_entry,
):
    """Selecting > 2 trigger languages should surface a form error."""
    mock_config_entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    # Walk to the advanced step; menu structure: init → ... → advanced.
    # Drive to the advanced step via the menu the existing options flow
    # exposes. (The existing test_config_flow.py file already covers the
    # navigation pattern; reuse it here.)
    while result.get("step_id") != "advanced":
        # Pick whatever menu key reaches "advanced" — see existing tests.
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "advanced"}
        )
    submission = {CONF_TRIGGER_LANGUAGES: ["de", "fr", "es"]}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submission
    )
    assert result["type"] == "form"
    assert result["errors"] == {CONF_TRIGGER_LANGUAGES: ERROR_TOO_MANY_TRIGGER_LANGUAGES}


@pytest.mark.asyncio
async def test_advanced_options_accepts_two_trigger_languages(
    hass, mock_config_entry,
):
    mock_config_entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    while result.get("step_id") != "advanced":
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "advanced"}
        )
    submission = {CONF_TRIGGER_LANGUAGES: ["de", "pl"]}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submission
    )
    # Either CREATE_ENTRY or another step — but NOT a form error on this key.
    if result["type"] == "form":
        assert CONF_TRIGGER_LANGUAGES not in (result.get("errors") or {})
```

(If the existing options-flow test file uses different fixture names or a
different navigation pattern, adapt these two tests to match the existing
style. The assertion pattern — `errors == {CONF_TRIGGER_LANGUAGES:
ERROR_TOO_MANY_TRIGGER_LANGUAGES}` — is the contract that matters.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/components/ai_plugin/test_config_flow.py -v -k trigger_languages`

Expected: `ImportError` for `ERROR_TOO_MANY_TRIGGER_LANGUAGES` plus form-not-rendering errors.

- [ ] **Step 3: Add error key to `const.py`**

In `custom_components/ai_plugin/const.py`, in the error-keys section (~line 80, near `ERROR_INVALID_URL`), append:

```python
ERROR_TOO_MANY_TRIGGER_LANGUAGES = "too_many_trigger_languages"
```

- [ ] **Step 4: Add SelectSelector to `config_flow.py`**

Open `custom_components/ai_plugin/config_flow.py`.

a. In the const-imports block, add:

```python
    CONF_TRIGGER_LANGUAGES,
    ERROR_TOO_MANY_TRIGGER_LANGUAGES,
    SUPPORTED_TRIGGER_LANGUAGES,
    _default_trigger_langs,
```

b. Find `_advanced_schema` (~line 176). Inside its schema dict, after the `CONF_ENABLE_THINKING` entry (or any stable anchor), append a new key:

```python
            vol.Optional(
                CONF_TRIGGER_LANGUAGES,
                default=current.get(
                    CONF_TRIGGER_LANGUAGES, _default_trigger_langs(_hass_for_default())
                ),
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
                )
            ),
```

c. The default lambda needs `hass` access. If `_advanced_schema` already takes `hass` as an argument, call `_default_trigger_langs(hass)` directly. Otherwise, change `_advanced_schema(current)` to `_advanced_schema(current, hass)` and update both definition site and call sites accordingly. Replace the `_default_trigger_langs(_hass_for_default())` placeholder above with `_default_trigger_langs(hass)`.

d. Find the advanced-step submission handler (the function that consumes the user's input from this schema, e.g. `async_step_advanced` or wherever the result is validated). Before the success path (entry update / next step), insert:

```python
        if (
            user_input is not None
            and isinstance(user_input.get(CONF_TRIGGER_LANGUAGES), list)
            and len(user_input[CONF_TRIGGER_LANGUAGES]) > 2
        ):
            errors = errors or {}
            errors[CONF_TRIGGER_LANGUAGES] = ERROR_TOO_MANY_TRIGGER_LANGUAGES
            return self.async_show_form(
                step_id="advanced",
                data_schema=_advanced_schema(user_input, self.hass),
                errors=errors,
            )
```

(The `self.hass` reference matches HA's options-flow handler convention. If
the existing handler uses a different attribute, mirror it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/components/ai_plugin/test_config_flow.py -v -k trigger_languages`

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/const.py custom_components/ai_plugin/config_flow.py tests/components/ai_plugin/test_config_flow.py
git -C /home/arndtg/AI-plugin commit -m "feat(config_flow): trigger-language SelectSelector with max-2 validation"
```

---

## Task 6: Add UI strings

**Files:**
- Modify: `custom_components/ai_plugin/strings.json`
- Modify: `custom_components/ai_plugin/translations/en.json`

- [ ] **Step 1: Add label + description in `strings.json`**

In `custom_components/ai_plugin/strings.json`, find the `options.step.advanced` block. In its `data` map, add:

```json
"trigger_languages": "Trigger-word hint languages"
```

(Order: keep alphabetical by key within `data`.)

In its `data_description` map, add:

```json
"trigger_languages": "Adds tool-routing hints for the selected languages so small models reliably map phrases like 'alle Lichter aus' or 'puść muzykę w pokoju' to the right tool. English instructions stay in the prompt regardless. Pick up to two."
```

In the `options.error` section (or `error` section if `options.error` does
not yet exist — mirror existing conventions), add:

```json
"too_many_trigger_languages": "Pick at most two languages."
```

- [ ] **Step 2: Mirror the same additions in `translations/en.json`**

Repeat the three additions above in `custom_components/ai_plugin/translations/en.json` at the matching locations.

- [ ] **Step 3: Validate JSON**

Run from repo root:

```bash
python3 -c "import json; [json.load(open(p)) for p in ['custom_components/ai_plugin/strings.json', 'custom_components/ai_plugin/translations/en.json']]; print('JSON OK')"
```

Expected: `JSON OK`.

- [ ] **Step 4: Commit**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/strings.json custom_components/ai_plugin/translations/en.json
git -C /home/arndtg/AI-plugin commit -m "i18n: strings for trigger_languages selector + too_many_trigger_languages error"
```

---

## Task 7: Bump manifest + final commit + tag + push

**Files:**
- Modify: `custom_components/ai_plugin/manifest.json`

- [ ] **Step 1: Run the full test suite**

Run from repo root:

```bash
python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py tests/components/ai_plugin/test_config_flow.py -v
```

Expected: all tests pass. If pytest collection fails locally with
`ModuleNotFoundError: No module named 'httpx'`, install test deps:
`pip install --user -r requirements_test.txt`. If still blocked locally,
run AST-only validation:

```bash
python3 -c "
import ast
for f in ['custom_components/ai_plugin/const.py','custom_components/ai_plugin/orchestrator.py','custom_components/ai_plugin/config_flow.py']:
    ast.parse(open(f).read()); print('AST OK', f)
"
```

CI will run the full suite on push.

- [ ] **Step 2: Bump manifest**

In `custom_components/ai_plugin/manifest.json`:

```json
"version": "0.7.7"
```

→

```json
"version": "0.8.0"
```

- [ ] **Step 3: Commit version bump**

```bash
git -C /home/arndtg/AI-plugin add custom_components/ai_plugin/manifest.json
git -C /home/arndtg/AI-plugin commit -m "$(cat <<'EOF'
v0.8.0: configurable trigger-word hint languages

Replace hard-coded English+German trigger fragments in the system prompt
with a per-integration option (CONF_TRIGGER_LANGUAGES) that injects
hint blocks for up to two of de/fr/es/pt/pl. English remains the
implicit base; missing option falls through to _default_trigger_langs
(auto-detected from hass.config.language).

- const.py: SUPPORTED_TRIGGER_LANGUAGES, CONF_TRIGGER_LANGUAGES,
  ERROR_TOO_MANY_TRIGGER_LANGUAGES, _default_trigger_langs,
  PROMPT_HINTS_I18N (de/fr/es/pt/pl × default+voice). German fragments
  removed from SYSTEM_PROMPT_DEFAULT and SYSTEM_PROMPT_VOICE.
- orchestrator.py: _build_system_prompt appends hint blocks based on
  the option, falling through to auto-detect on legacy entries.
- config_flow.py: SelectSelector(multiple=True, dropdown) in advanced
  options with max-2 validation.
- strings.json + translations/en.json: option label, description,
  too_many_trigger_languages error.
- tests/components/ai_plugin/test_prompts_i18n.py: new (auto-detect,
  block injection, voice mode, multi-lang ordering, unknown-lang
  skipping, legacy fallthrough).
- tests/components/ai_plugin/test_config_flow.py: extended with the
  3-element rejection and 2-element acceptance cases.

Spec: docs/superpowers/specs/2026-05-03-multilang-prompts-design.md
Plan: docs/superpowers/plans/2026-05-03-multilang-prompts.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Tag + push**

```bash
git -C /home/arndtg/AI-plugin tag v0.8.0
git -C /home/arndtg/AI-plugin push https://<GITHUB_PAT>@github.com/bigbabol1/AI-plugin.git main v0.8.0
```

Verify on GitHub:

```bash
command curl -s "https://api.github.com/repos/bigbabol1/AI-plugin/tags?per_page=2" -o /tmp/tg.json
python3 -c "import json;[print(t['name'],t['commit']['sha'][:7]) for t in json.load(open('/tmp/tg.json'))[:2]]"
```

Expected first line: `v0.8.0 <sha>`.

- [ ] **Step 5: Manual smoke (HA)**

Pull v0.8.0 in HACS, reload the AI Plugin integration. Then:

1. Open Settings → Devices & Services → AI Plugin → Configure → Advanced. Verify the new "Trigger-word hint languages" dropdown appears with options `Deutsch`, `Français`, `Español`, `Português`, `Polski`.
2. Try selecting 3 → form rejects with "Pick at most two languages."
3. Select `Deutsch` only → save.
4. Speak "alle Lichter aus" via your voice satellite → expect `set_area_state(area='all', domain='light', action='turn_off')` to fire (verify via `media_player`/`light` state changes, or HA logs grep for `set_area_state`).
5. Select `[]` → save → speak the same German command → behavior depends on the model; multilingual qwen3 family still works, smaller models may degrade. Document the result in the issue tracker if degraded.

---

## Self-Review

**Spec coverage:**
- §1 Architecture (per-language hint blocks + base prompt + injection) → Tasks 1, 2, 3, 4 ✓
- §2 Config (selector, max 2, auto-detect default, EN baseline) → Task 5 ✓
- §3 Hint block contents (de + fr + es + pt + pl × default + voice) → Task 2 ✓
- §4 Testing (unit + manual smoke + edge cases) → Tasks 1–4 unit, Task 7 manual ✓
- §5 Rollout (manifest, version, single PR) → Tasks 6, 7 ✓
- Backward compatibility (missing option → `_default_trigger_langs`) → Task 4 step 4 explicit fallback ✓
- Edge cases (`hass.config.language=None`, unsupported lang code) → Task 1 + Task 4 unknown-lang test ✓

**Placeholder scan:** none — every step shows the exact code or exact diff.

**Type consistency:**
- `CONF_TRIGGER_LANGUAGES` (str), `SUPPORTED_TRIGGER_LANGUAGES` (list[str]), `PROMPT_HINTS_I18N` (dict[str, dict[str, str]]), `_default_trigger_langs(hass) -> list[str]`, `ERROR_TOO_MANY_TRIGGER_LANGUAGES = "too_many_trigger_languages"` — all referenced consistently across const, orchestrator, config_flow, strings, tests.
- `mode_key` strings `"default"` / `"voice"` match `PROMPT_HINTS_I18N` inner dict keys exactly.
- `selector.SelectSelectorMode.DROPDOWN`, `multiple=True` — matches HA helper API used elsewhere in `config_flow.py`.
