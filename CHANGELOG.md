# Changelog

All notable changes to AI Plugin are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [0.5.47] - 2026-04-26

### Fixed

- **Adding any MCP preset that needs a path (SQLite, filesystem) failed with "unknown error"** (`config_flow.py`, `strings.json`, `translations/en.json`). The preset-config form passed `selector.TextSelectorConfig(placeholder=...)`, but `placeholder` is not a valid `TextSelectorConfig` field — HA core whitelists only `multiline / prefix / suffix / type / autocomplete / multiple`. Voluptuous rejected the unknown key on schema build and HA surfaced the generic "unknown error" string with no log line tying it to the config flow. Fix: drop the kwarg, surface the example path through `description_placeholders["example"]` instead, and append `Example: {example}` to the step description so users still see the suggested path.
- **SQLite preset now suggests a non-conflicting path** (`config_flow.py`). Old example `/config/home-assistant_v2.db` is HA's recorder DB and is locked for writes while HA runs — `mcp-server-sqlite` exposes `write_query` and `create_table`, so any LLM-issued write would have collided with the recorder. New example `/config/buddy_notes.db` is auto-created by `mcp-server-sqlite` on first connect (`Path.parent.mkdir(parents=True, exist_ok=True)` + `sqlite3.connect(...)`) and gives the model a private scratchpad to read and write without fighting the recorder. The label was updated to reflect the scratchpad use case.

### Removed

- **XML tool-calling fallback removed entirely** (`orchestrator.py`, `const.py`, `config_flow.py`, `strings.json`, `translations/en.json`, tests). The `xml_fallback` option, the `_xml_tool_loop` method, the `<tools>`/`<tool_call>` system-prompt scaffolding, and the `CONF_XML_FALLBACK` / `DEFAULT_XML_FALLBACK` constants are gone. Every model in the recommended list (Qwen3 8B, Qwen2.5 7B, Mistral 7B, Hermes 3 8B for ≤8 GB VRAM and ministral-3:14b / GPT-4o / Claude / Grok above) supports native OpenAI-compatible function calling, and models that don't are not recommended for this plugin in the first place — the XML path was carrying ~70 lines of orchestrator code, a config option, and four UI strings for a code path that the suggested setups never executed. Existing entries with `xml_fallback: true` in `options` are silently ignored on next load (HA's options store tolerates unknown keys); no migration step is required.

### Docs

- **Mistral 7B added to the bench, Llama 3.1 row removed.** Fresh 51-prompt voice-mode sweep against `mistral:7b` Q4_K_M scored 82.4 % (42/51) at median 0.4 s — fastest of the four tested but the only one that fabricated a German weather forecast for `wie ist das wetter?` instead of calling a tool, and it dumps the system prompt verbatim on bare nouns like `weather` / `events` / `temperature`. Llama 3.1 8B was never benched in this round so it has been pulled from the table; recommendations remain Qwen3 8B for voice, Qwen2.5 7B for text.
- **Ollama-only disclaimer.** Features list now states upfront that only Ollama has been exercised end-to-end. The OpenAI-compatible endpoint should accept OpenAI / xAI / LM Studio / llama.cpp, but those paths are untested.
- **Removed local IP / failure-analysis subsection / stale troubleshooting bullet.** The `192.168.0.66:11435` reference, the per-prompt qwen failure-analysis tables, and the obsolete `"Any lights on?" answers about a single light — fixed in v0.5.38` troubleshooting line are all dropped from the README.

## [0.5.46] - 2026-04-26

### Fixed

- **Two follow-up quirks from the v0.5.45 location-bias rollout** (`tools/web_search.py`, `const.py`). Live 25-prompt sweep against Assist showed v0.5.45 killed the Texas hallucination cleanly (0 hits) but exposed two pre-existing pinpoints: (1) injected `near Berlin` returned mostly *Berlin, NY / NJ / MD* hits because DuckDuckGo's region resolver tie-breaks ambiguous toponyms toward the user's IP locale; (2) `weather in Tokyo right now` returned the local Berlin weather entity's numbers because the system prompt's `[WEATHER — STRICT ORDER]` block ran `list_entities(domain='weather')` before the LLM could route the named-place case to `web_search`. Two coordinated changes:
  - **Country disambiguation in injected place.** `_best_place_label` now returns `"<city>, <country>"` (or `"<region>, <country>"`) whenever both fields are populated, falling back to the bare city/region/country/coord chain only when one is missing. Search engines treat the comma-separated form as an unambiguous geo hint, so `near Berlin, Germany` no longer collapses to small-town US results. Idempotent against the existing `_has_place_token` check (the longer label still matches when re-injected).
  - **Named-place precondition in weather block.** Both `SYSTEM_PROMPT_DEFAULT` and `SYSTEM_PROMPT_VOICE` weather blocks now lead with: *"IF the user names a city/country/region different from their home: SKIP `list_entities` and CALL `web_search('weather in <named place>', near_user=false)` instead."* The local weather entity covers the user's home, not Tokyo — telling the LLM to short-circuit on explicit pins routes the query to the same web path that already handles non-home weather correctly.

## [0.5.45] - 2026-04-26

### Fixed

- **`web_search` no longer hallucinates the user's location** (`tools/web_search.py`, `tools/_locality_tokens.py`, `tools/_geocode.py`, `orchestrator.py`, `const.py`, `config_flow.py`, `strings.json`, `translations/en.json`). Asking *"any events nearby?"* could return Texas results because the LLM either wrote a free-form query (`events near me`) that the search backend resolved geographically, or it picked a city out of thin air after the v0.5.44 prompt stripped the `friendly_label` field. Three coordinated changes close the gap globally — the plugin runs on installs across the world, some with location entities exposed and some without, so every layer fails open:
  - **Schema-level intent flag.** `web_search` gained a `near_user: bool` parameter steered by both system prompts. The LLM now signals locality intent declaratively instead of hand-crafting a place name.
  - **Two-signal injection gate.** `_maybe_inject_location` appends ` near {place}` to the query only when (intent: `near_user=True` *or* multi-lingual locality regex match) **and** (safety: a place is resolved *and* the query does not already pin itself to a place via preposition + proper noun). The regex fallback covers EN, DE, ES, FR, IT, NL, PT, PL, SV, DA, NO, FI, CS, TR, JA, ZH for installs where the LLM omits the flag. Idempotent — never appends a place that's already in the query.
  - **Heterogeneous-install location resolver.** New `LocationProvider` walks a source chain (configured location entity → `hass.config` lat/lon → nothing) and reverse-geocodes via OpenStreetMap Nominatim with disk cache + 1.1 s rate gate. When no signal resolves, the location block is omitted from the prompt entirely and `near_user` becomes a no-op rather than an excuse to invent a city. New `_geocode` module is fail-silent on network/timeout/non-200; new `_locality_tokens` module compiles per-language regex with CJK word-boundary handling.
  - **Privacy opt-out.** New `location_bias` (default on) and `location_entity` (optional) options under config/options Advanced. Disabling `location_bias` makes the resolver return `{}`, which deterministically prevents both the prompt block and the query injection — a single switch for users who don't want their coords leaving HA.

## [0.5.44] - 2026-04-25

### Fixed

- **Memory recall now works on small LLMs** (`orchestrator.py`, `tools/memory.py`, `const.py`). Memory testing on qwen3:8b revealed three independent bugs: (1) `recall` was never invoked on questions like "Wie heiße ich?" / "What is my name?" — the LLM either answered from message history or refused — leaving every saved fact invisible; (2) the `[HOME LOCATION]` block leaked the HA instance's `friendly_label` (e.g. "HomeSweetHome") into the prompt, which small models then read aloud as the user's home town; (3) `forget('...')` did a case-insensitive substring match, so a German `Vergiss, dass ich Kaffee mit Milch trinke` against an English-stored `I drink coffee with milk` removed nothing while the tool still replied "forgotten". Three-part fix:
  - Auto-inject a numbered `[USER FACTS]` block into every system prompt directly from `.ai_plugin_memory_<user_id>.json`. Small LLMs see the facts as context and answer correctly without needing a tool call. Memory section of both system prompts updated to read from the block first and only call `recall` when it is absent.
  - Strip `friendly_label=` (and its multi-line explanation) from `_build_location_block`. Country, coords, timezone remain. Eliminates the "you live in HomeSweetHome" failure mode.
  - Add `index: integer` parameter to the `forget` tool schema (1-based, matching the `[USER FACTS]` numbering). `_forget` now removes by index when present, falls back to substring on `fact`. Cross-language forget (`forget(index=3)` in any language) finally works. The `fact` argument is required as a sanity check, validated by a two-stage guard: (1) `fact` must token-overlap the user message (catches the LLM lifting a keyword from the stored entry it is hallucinating about); (2) when crude EN/DE language detection sees `fact`/`user_message` and `stored[index]` in the same language, `fact` must also overlap the stored entry — this rejects calls like `forget('drive a Ferrari', index=1)` against stored `"KL is short for Kuala Lumpur"`. Cross-language forgets (German `Vergiss Kaffee mit Milch` against English `I drink coffee with milk`) skip the second stage because token mismatch is expected. System prompts updated to prefer `index` and to refuse forgetting when the referenced fact is not in `[USER FACTS]`. Verified end-to-end on qwen3:8b: 19/19 prompts pass including M11 cross-lang forget and M13 refusal of unstored fact.

### Fixed

- **Shortcut now answers temperature for thermostat-only rooms** (`shortcuts.py`). Rooms whose only exposed temperature source is a thermostat (e.g. Tado, Nest publishing `current_temperature` on a `climate.*` entity) fell through the deterministic shortcut because `_pick_best_sensor` only matched `sensor.*` with `device_class: temperature`. The LLM tool loop then took over and frequently picked the wrong entity — `media_player.wohnzimmer_2` (state `off`), `sensor.luftreiniger_pm2_5` (PM2.5 reading), or hallucinated `climate.wohnzimmer_heizung` — because `search_entities` doesn't index by area and `friendly_name` rarely contains the room name. Added a `_pick_climate_temperature` fallback that reads the `current_temperature` attribute from any `climate.*` entity in the resolved area when the sensor lookup misses. Pure addition; bedroom / hobby-room paths that already hit a sensor remain unchanged.

## [0.5.42] - 2026-04-25

### Fixed

- **Pre-LLM shortcut now respects Assist exposure** (`shortcuts.py`). `_entities_in_area` walked the entity registry filtered only by area, ignoring each entity's `conversation.should_expose` setting. As a result, sensors that share an area with a voice satellite (e.g. `sensor.satellite1_*_temperature`, the chip's onboard reading at ~28 °C) could outrank the actual room sensor in `_pick_best_sensor` and be returned to users who had hidden them from Assist. Added an `async_should_expose(hass, "conversation", entity_id)` filter that mirrors HA's own intent / Assist pipeline visibility. Falls back to the previous behaviour on cores without `homeassistant.components.homeassistant.exposed_entities`.

## [0.5.23] - 2026-04-22

### Fixed

- **Small LLMs stop parroting `[MEMORY]` instructions as text** (`const.py`). The v0.5.22 memory block used caps-imperative phrasing like `CALL recall FIRST`, which small instruction-tuned LLMs echoed back verbatim ("CALL recall first to see if I remember…") instead of invoking the tool. Rewrote both default and voice memory blocks in descriptive prose and explicitly told the model to invoke tools silently rather than narrate them.

## [0.5.22] - 2026-04-22

### Fixed

- **LLM now calls `recall` proactively** (`const.py`). After v0.5.21 started prepending `SYSTEM_PROMPT_DEFAULT` to user-supplied custom prompts, small LLMs locked onto the default prompt's heavy entity-discovery rules and stopped invoking the `recall` memory tool — facts were written but never surfaced. Added a `[MEMORY]` block to both the default and voice system prompts explicitly directing the model to call `recall` when the user asks about their preferences, past statements, or anything depending on previously stored facts, and `remember`/`forget` on durable preference statements.

## [0.5.21] - 2026-04-22

### Changed

- **Custom system prompt now appends instead of replacing** (`orchestrator.py`). Previously, setting a custom prompt in Advanced options fully replaced both `SYSTEM_PROMPT_DEFAULT` and `SYSTEM_PROMPT_VOICE`, silently dropping the plugin's entity-discovery, grounding, context, and voice-speech rules — users writing persona/location prompts lost tool-use discipline and small LLMs started fabricating entity_ids. The custom prompt is now appended to the base (voice-compact or default) prompt with a blank line separator. Empty custom prompt behaves as before (base only). Users who intentionally stripped defaults can clear their custom prompt and re-add only the overrides they want.

## [0.5.20] - 2026-04-22

### Fixed

- **Voice replies no longer read URLs aloud** (`const.py`, `orchestrator.py`, `tools/web_search.py`). Web-search results over a voice pipeline used to surface raw `https://…` addresses into the LLM's context; small LLMs then read them back and TTS narrated every slash. Two-pronged fix: the voice system prompt now forbids reading URLs, domain names, or file paths aloud; and the `web_search` tool result is stripped of URL lines (and inline `http(s)://` tokens) before it reaches the model whenever the orchestrator is in voice mode. Typed Assist is unchanged and still gets the source URLs.

## [0.5.19] - 2026-04-22

### Fixed

- **Voice Assist now reads long-term memory** (`conversation.py`, `config_flow.py`). Previously, voice pipelines (Wyoming satellites, wake-word, TTS-initiated flows) ran without an authenticated HA user, so `context.user_id` was `None` and the `remember`/`recall` tool routed to `.ai_plugin_memory_anonymous.json` — a different file from the `.ai_plugin_memory_<user_id>.json` written by typed Assist. Facts saved from the Assist panel never surfaced over voice. The Advanced settings now include a **Voice fallback user** dropdown; when set, voice requests reuse that user's memory file. Leave empty to keep the previous anonymous behaviour.

### Notes

- Existing `.ai_plugin_memory_anonymous.json` is not auto-migrated. If you pick a fallback user and want voice to see previously saved facts, rename that file to `.ai_plugin_memory_<user_id>.json` in your HA config directory (or copy its `facts` into the user's file).

## [0.5.17] - 2026-04-22

### Fixed

- **Conversation history now resets per session** (`conversation.py`) — previously the entity overrode HA's ephemeral `conversation_id` with a stable per-user id, causing history to accumulate across every Assist session until summarisation fired. Now honours the id HA Assist provides (minting a fresh ulid when none is supplied). Long-term user facts continue to live in the `remember`/`recall` tool (`.ai_plugin_memory_<user_id>.json`), which is unchanged.

### Changed

- **README setup clarity** — added HACS prerequisite, provider URL examples per backend, Assist pipeline assignment step, and a Troubleshooting section. Documented the actual default Context Window (16384) alongside the 8 GB sweet-spot of 8192. Reordered web-search backends to put the default (Brave) first and added an API-key column with sign-up links.

## [0.1.0] - 2026-03-26

### Added

- **Integration skeleton** — Home Assistant custom component with HACS support, `manifest.json`, `hacs.json`, and GitHub release workflow
- **OpenAI-compatible provider** (`providers/openai_compat.py`) — supports Ollama, llama.cpp, LM Studio, OpenAI, and any OpenAI-compat API endpoint
- **Config flow** — 4-step setup: provider URL → model selection (auto-fetched dropdown) → web search → advanced settings; full OptionsFlowHandler for post-install reconfiguration
- **ContextManager** (`context_manager.py`) — sliding window token budget with soft/hard limits, per-conversation history isolation, and LLM-based summarization when approaching context limits
- **Orchestrator** (`orchestrator.py`) — core message-processing engine: system prompt selection (voice/custom/default), native function-calling tool loop, XML fallback tool loop for models without native tool support
- **MCP tool registry** (`tools/mcp_client.py`) — persistent MCP server connections over HTTP (streamable-http) and stdio transports, auto-reconnect with exponential backoff, tool schema aggregation
- **Web search tool** (`tools/web_search.py`) — four backends: DuckDuckGo (free), Brave Search (API key), SearXNG (self-hosted), Tavily (API key); XML fallback support
- **Voice mode** — compact system prompt when `device_id` is present (Assist pipeline) or `CONF_VOICE_MODE` toggle is set
- **Test suite** — 105 tests, 76% coverage; all core business logic modules at 100% (context_manager, orchestrator logic, providers, mcp_client, web_search)
- **AbstractProvider base class** — extensible provider interface; post-v1 Gemini and Anthropic adapters can implement without orchestrator changes

### Changed

- `tool_choice` defaults to `"auto"` instead of requiring explicit provider opt-in
- `AbstractProvider.async_chat` has a safe default implementation (wraps `async_complete`) so post-v1 providers inherit a working fallback

### Notes

- `config_flow.py` requires full `pytest-homeassistant-custom-component` fixtures; covered by integration tests, not unit tests
- Post-v1 roadmap: Gemini/Anthropic adapters (v0.2.0), semantic long-term memory, MCP server endpoint, full OptionsFlow MCP server management
