# Changelog

All notable changes to AI Plugin are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## v0.9.42 — follow-up listening stops talking to itself

**New:**
- **Whole-home sweeps confirm out loud.** "All lights off" now answers "OK, all lights are off." (localised, and it names the domain — lights, fans, sockets). Only whole-home sweeps speak: a room-scoped or caller-room sweep is happening in front of you, so it stays silent as before.

**Fixed — 'Listen for follow-up' feedback loop:**

A recorded loop from this install (ten consecutive runs, 11:17–11:20, of which two turns were human) shows exactly how it sustained itself, and it was not one bug but three.

- **Short tails are now recognised as echo.** The mic re-arms as TTS playback ends, so it catches the reply's last words — "…in your home now!" comes back as "home now". Fragments that short sit below `ECHO_MIN_TOKENS`, where the bigram filter deliberately doesn't look, so they ran as fresh commands. A new tail rule matches a short turn against the reply's *ending*, in order, allowing one STT slip once the fragment is 3+ words; a single word must be the reply's very last word, so "stop" and "louder" still get through. Its window follows playback rather than the clock: the stored timestamp is from when the reply was *generated*, and a 60-word reply is still being spoken 20 seconds later.
- **A detected echo now ends the session.** This is the one that mattered. Echoes were already being dropped — but the reply kept the mic armed, handing the reverb another go, and one fragment we happened not to recognise restarted the loop for real. In the recorded loop, turn 2 was correctly dropped and turns 3–10 followed anyway. Ending the session there stops all eight.
- **Undetectable tails are bounded by a chain breaker.** "…list all areas and entities we have connected!" was heard as "and I choose to be out connected" — two words in common, no shared word pairs, too long for the tail rule. Nothing can match that reliably, so the chain is capped instead: after four turns that each arrive while the previous reply could still be playing, no further follow-up is offered. Measuring against playback (not a fixed gap) is what makes it fire — the chatty replies feeding these loops take 20–30s to speak, so every link looked unhurried by wall clock. The recorded worst case drops from 10+ turns to 5.

Both replays are committed as regression tests, with the recorded timings.

*Trade-off worth knowing: a real follow-up that repeats the end of what was just said, or the fifth turn of a rapid-fire chain, ends the session too. The wake word restarts it.*

## v0.9.41 — "all lights off" actually turns off all the lights

Reported: asking the bedroom satellite to switch all lights on/off did nothing.
Two independent causes, both confirmed against the live install (pipeline debug
runs + core logs from the failed attempts).

**Fixed:**
- **An omitted `area` no longer shrinks a whole-home command to one room.** Small models drop the `area` argument constantly, and `set_area_state` then defaulted straight to the calling satellite's room — so "switch all lights off" spoken in the bedroom turned off the bedroom and left the rest of the flat lit (reproduced live: *"area unspecified, defaulted to caller's area 'bedroom'"*). Scope now follows a precedence ladder: a room named in the utterance wins, then an explicit "all"/"alle"/"whole house", and only then the caller's room. The tool description says so too, since the model was being told to omit `area` for exactly these commands.
- **The plugin no longer confirms actions it didn't take.** With no actuator call in the turn, the model's *"I'll turn off all the lights in your home now!"* was spoken verbatim — the user heard a confirmation while every lamp stayed on (observed three times in a row on qwen3.5:9b before they gave up and used the app). Promise phrasings are now detected in all six languages and replaced with a truthful failure notice (new `err_action_failed` template) once the grounding retry has also failed to act. Genuine post-action confirmations are untouched: the check only runs when nothing actuated.

**New:**
- **Deterministic plural-domain sweep shortcut.** "all lights off", "alle Lichter aus", "éteins toutes les lumières", "turn the fans off" now dispatch straight from the registry, ahead of the LLM — the same treatment media and single-device commands already get, and for the same reason: this is the command class small local models fumble worst. Scope resolution mirrors `set_area_state` (named room → explicit "all" → caller's room); singular nouns ("turn off the light"), named devices, and genuinely ambiguous scopes still fall through to the model. Verified against the live registry: 19 exposed lights flat-wide, 3 in the kitchen, 6 in the bedroom, with unexposed satellite LED rings correctly excluded.

## v0.9.40 — MCP logs that point at the actual problem

**Fixed:**
- **A healthy MCP server no longer logs like a broken one.** Subprocess stderr was logged at WARNING unconditionally, but working servers write there routinely — startup banners, dependency-install chatter, request logs. `wikipedia-mcp` looked broken for weeks purely because it announces itself on stderr, while the servers that genuinely failed (an unpinned `mcp` 2.x dependency breaking their imports) drowned in the same noise. Stderr is now DEBUG when the handshake completed and WARNING only when the connection attempt failed.
- **Failure messages name the server.** Every stdio server was labelled by its bare command, so several `uvx`-launched servers all reported as `'uvx'` and "connection to 'uvx' failed" could not be traced to one of them. Labels now include the args (`uvx --with mcp<2 mcp-server-time`), in both log lines and the "server not connected" tool error.

*Tip for stdio servers whose upstream doesn't cap its `mcp` dependency (`mcp-server-time`, `mcp-server-fetch`, `mcp-server-calculator`): set args to `--with mcp<2 <package>`. The args field is whitespace-split, not shell-parsed — do not quote.*

## v0.9.39 — Review hardening III: the rest of the audit

The remaining findings from the 2026-07-29 adversarial review.

**Fixed:**
- **Self-echo filter is order-aware.** Bag-of-words overlap swallowed real commands that reuse the reply's vocabulary ("turn the living room light on" right after *"The living room light is on."* was dropped as echo). Matching is now on word *bigrams* — an acoustic echo replays the reply's word sequence, a reordered command doesn't. Truncated echo fragments and echoes with an STT error are still caught; repeat-questions now pass through.
- **Omitted area can't silently empty the whole home.** From text Assist/REST (no satellite to infer a room from), `set_area_state` with no area swept every area, bypassing the explicit-"all" guard. It now requires the user to have actually said "all/everything/whole house" — otherwise the model gets the same refusal + specific-device recovery as an explicit `area='all'`.
- **`get_entity` prefers exposed matches.** An unexposed diagnostic entity that happened to match first no longer shadows an exposed entity with the same name fragment; "exists but is not exposed" is only reported when *no* exposed candidate matches.
- **`list_entities` accepts area aliases** like every other area resolver, and an unknown area now returns "Unknown area … Try list_areas." instead of a misleading empty result.
- **`search_entities` marks truncation** ("more exist — narrow the query") instead of presenting a clipped list as exhaustive.
- **Auto-recovery queries no longer mangle device nouns.** Filler stripping used raw substrings, so " an" ate the head of "Anlage" (query became "lage" and found nothing); it is word-bounded now.

**New:**
- **`media_command` volume family**: `volume_up`, `volume_down`, `volume_set` (with `level` 0–100), `mute`, `unmute` — until now volume only worked when the deterministic shortcut regex matched; rephrasings left the model with no valid tool. Prompts updated with the mappings.

**Changed:**
- `hvac_action` replaces the dead `hvac_mode` in the attribute surface (climate entities never expose `hvac_mode`; the current activity lives in `hvac_action`).

## v0.9.38 — Review hardening II: retries, timers, sweeps, shortcut precision

**Fixed:**
- **Grounding retries can no longer re-execute actions.** The state-set, fuzzy-resolve, and web-search verifier retries re-ran the tool loop with the full schema set, so a retry could call `HassMediaNext`/`HassTurnOn` a second time (track skipped twice). Retries now run with a read-only schema subset; the action-command verifier keeps actuators — forcing the action is its purpose.
- **Timer durations no longer absorb numbers from the timer's name.** "Add 5 minutes to the 10 minute timer" summed both numbers into a 15-minute change. The utterance parser now only overrides model slots when the message contains exactly one *contiguous* duration phrase ("1 minute 30 seconds" still works); ambiguity defers to the model's slots.
- **"Turn everything off" / "mach alles aus" work.** The explicit-sweep guard knew "all"/"alle" but not "everything"/"alles" (and now also "sämtliche"), so it refused exactly the phrasings people use most.
- **Sensor/time/sun shortcuts no longer hijack commands and place questions.** "Turn on the heating, it's cold in the living room" was answered with a temperature reading instead of acting; "what time is it in Tokyo" got the local clock; "set an alarm for sunset" got the sunset time. The Q&A shortcuts now skip messages carrying action verbs or >12 words, and the time/sun shortcuts skip "in <place>" questions — all of these fall through to the LLM, which can act or search.

## v0.9.37 — Review hardening: no more silent failures

Five defects from an adversarial review of the reply pipeline, all sharing one root pattern: post-processing keyed on tool *invocation* or regex *presence* instead of *results* or *intent*.

**Fixed:**
- **Failed actions no longer sound like success.** Voice TTS suppression is now result-gated for ALL actuators (it already was for media tools since v0.9.36): a `HassTurnOn`/`set_area_state` call that *failed* keeps the model's spoken explanation instead of being blanked — silence after "turn off the heating" no longer means "maybe". MCP `isError` results are now bracketed like transport errors so downstream success checks can't mistake a failed intent for a performed one.
- **Streaming no longer loses answers.** Once sentences have streamed, the delta stream is what the pipeline speaks — so the gate now delivers a rewritten final reply in full instead of withholding it (repeating a short preamble beats losing the answer), and a safety-close after a partial stream still completes the stream with the authoritative reply instead of stranding the user on "Sure."
- **Narration stripping is sentence-granular and keeps data.** "I'm checking. It's 21 degrees." previously died as one line and became "I couldn't produce an answer"; now only the narration sentence is removed, and any sentence carrying a digit always survives.
- **Media shortcut: questions are never commands.** "What does stop mean?" / "when is the next bus" no longer stop/skip playback with an empty reply; German modal "halt" and conversational "weiter" ("und so weiter") only count as playback commands in ≤3-word utterances; and the shortcut finally respects the conversation exposure list like every other path.
- **German area targeting works with articles.** "pause die musik in der küche" used to swallow the article, miss the area, and silently pause the *whole home*; the suffix regex now matches "in der/dem" before bare "in", with a de-articling fallback in resolution.

## v0.9.36 — "What's playing?" answered; media suppression result-gated

**Fixed:**
- **"What's playing?" got dead silence on voice.** Root cause was a two-part failure: (1) `media_command` had no read-only query, so the model probed the non-existent `command='status'` and got an error; (2) the TTS-suppression pass blanked the reply because a media tool had been *invoked* — even though the call failed and the model had recovered with a correct spoken answer ("Nothing is playing right now."). Suppression is now **result-gated**: only a media call whose result confirms an actual playback change (the `OK` prefix) blanks the reply. Failed calls and read-only queries keep the model's answer audible.

**New:**
- **`media_command('status')`** — read-only now-playing report: title, artist, and player for everything playing (paused listed separately), area-filterable, exposure-respecting, never calls a service. The model was already guessing this command existed; now it does. Prompts (voice + default) route "what is playing / what song is this / was läuft" to it, and a `status` call counts as grounding for the state-set verifier so the turn doesn't pay for a pointless `list_entities` retry loop.

## v0.9.35 — Self-echo filter: follow-up without the loop

**New:**
- **Self-echo filter** (Advanced → *Ignore the satellite hearing its own reply*, on by default). Satellites that play TTS through a separate speaker (Mic to MediaPlayer) re-hear the reply and STT feeds it back as a "user" turn — the self-talking loop when *Listen for follow-up* is on. Acoustic echo cancellation can't help (the satellite's AEC only cancels audio it renders itself, not sound from external speakers), so the plugin does it in software: it remembers what it just said per device and drops any incoming voice turn whose tokens are ≥80% contained in a reply from the last 30 s. The dropped turn runs no command and speaks nothing, so the loop dies — but the session stays open, so a *real* follow-up still lands. Short turns (< 4 tokens: "yes", "stop", "turn it off") are never filtered, and reply memory isn't cleared on a hit so reverb/multi-fragment echoes of the same reply are caught too.
- This makes *Listen for follow-up* usable on TTS-rerouted setups **without** muting the microphone or losing barge-in. (Muting the mic during playback in mic_to_mediaplayer remains a complementary hardware-side option — see the repo discussion.)

## v0.9.34 — Debloat: multi-model routing removed; README rewritten

**Removed:**
- **Multi-model routing** (the v0.9.30 `route_*` options, classification logic, and per-request model override) is gone. It only ever worked across models on a single endpoint, added three settings and a classification layer for a niche win, and switching models per turn costs a model load unless both fit in VRAM — bloat relative to its value. Entries with `route_*` keys in options are silently ignored on next load; no migration needed. One model per config entry again.

**Docs:**
- `_handle_timer_done` now carries a step-by-step explanation of how announce-mode timers work (conversation_command → TimerManager callback → sentinel → mic_to_mediaplayer/assist_satellite announce).
- **README rewritten** to reflect the current integration: streaming, the full shortcut layer, cache-stable prompt architecture ("How it stays fast on local models"), voice timers incl. announce mode, AI Tasks, model introspection, keep-alive option, the feedback-loop device guard, DuckDuckGo default, a complete Advanced-options reference table, and the eval harness.
- Unused imports swept (orchestrator, web_search).

## v0.9.33 — Device-independent timer announcements

**New:**
- **Announce timers instead of on-device ring** (Advanced toggle, default off). The satellite's timer ring plays on its *local* speaker — useless when that speaker is silenced because audio is routed elsewhere (e.g. Mic to MediaPlayer), and unavailable on satellites without timer firmware. With this option, timers started by voice carry a `conversation_command`: HA's own TimerManager calls the agent back at expiry, and the plugin plays a localized announcement ("Timer pasta is up." / "Der Timer pasta ist abgelaufen.", all 6 languages) via `mic_to_mediaplayer.announce`, falling back to `assist_satellite.announce`. No firmware edits, no plugin-side scheduling to drift — cancel/pause/extend keep working on the single HA-managed timer, and timers now work on **any** satellite (the device-support requirement is bypassed).
- Trade-off, by design: the device is never notified about announce-mode timers, so on-device countdown LEDs/ring don't fire.

## v0.9.32 — 'Listen for follow-up' actually works on voice satellites

**Fixed:**
- **Voice satellites honour the follow-up toggle again** (`conversation.py`). The TTS-feedback-loop protection force-ended EVERY voice-satellite turn, which silently disabled *Listen for follow-up after voice replies* on all satellites — including ones with their own speaker and echo suppression (HA Voice PE, standard ESPHome satellites) that handle `continue_conversation` correctly. The blanket force-end is replaced by an explicit guard list: **Advanced → Force-end conversations on these satellites** (device picker). Add only satellites whose TTS plays through a separate speaker (e.g. via Mic to MediaPlayer) — those genuinely re-hear their own reply and re-trigger listening. Close phrases (`thanks`, `das war's`, …) still end any session.

## v0.9.31 — Shortcut coverage: covers and volume

**New:**
- **Cover open/close shortcut** (all 6 languages, `i18n/<code>.yaml` `action_open`/`action_close`). "Open the blinds" / "Mach den Rollladen zu" now resolves deterministically to one exposed `cover.*` entity and dispatches `cover.open_cover`/`close_cover` — same exact→substring→caller-area tiebreak as on/off, ambiguity falls through to the LLM. Cover verbs never actuate non-cover entities ("open spotify" stays with the LLM).
- **Volume in the media shortcut**: volume up/down, mute/unmute, and "set the volume to N percent" (EN+DE) dispatch the matching `media_player` service on whatever is actually playing — no LLM round-trip.

**Deliberately not shortcut**: locks (deterministically unlocking a door on a regex match is a security decision, not a latency optimization) and brightness/color (slot-heavy; the LLM path already benches 9/10 on lights).

## v0.9.30 — Multi-model routing, model introspection, AI Task platform

**New:**
- **Intent-based model routing** (Advanced → three optional model fields; the `route_*` keys reserved since v0.1 are finally wired). Deterministic classification — no LLM classifier: web/research turns → your strongest reasoner, home-control turns (devices, sensors, timers) → a fast decisive model, everything else → the general model. Unset routes fall back to the main model; leave all empty for the previous single-model behaviour. All routes must live on the same endpoint; note that per-turn model switching costs a load unless both fit in VRAM.
- **Model introspection at config time** (Ollama `/api/show`, fail-open). Picking a model that reports no `tools` capability is now a config-flow error instead of a silently broken install; setting a Context Window larger than the model's maximum context length is rejected in Advanced settings.
- **AI Task platform** (`ai_task.py`). The same backend now serves `ai_task.generate_data` for automations and scripts — plain text or structured JSON (schema rendered via voluptuous_openapi, code fences tolerated, invalid JSON raises instead of returning garbage). One more reason the local model is the home's LLM hub, not just its voice.

**Changed:**
- Provider option parsing centralised in `OpenAICompatProvider.from_options` (orchestrator + ai_task share it).

## v0.9.29 — Streaming replies (sentence-safe early TTS)

**New:**
- **Replies stream into HA's chat log** (`conversation.py` `_async_handle_message` + `_attr_supports_streaming`, `orchestrator.py` `_DeltaGate`, provider `async_chat_stream`). Voice pipelines with streaming TTS start speaking after the first sentence instead of waiting for full generation — on a 7-9B model that's seconds of perceived latency per chatty turn. Ollama native backend streams (NDJSON); other backends keep the buffered path.
- Streaming is **sentence-safe by design**, because spoken text can't be recalled: deltas are sentence-buffered with the trailing sentence held back (trailing-filler stripping still works), each sentence passes the same narration/kaomoji/voice sanitation as the final reply, and the gate shuts permanently on any turn where the safety layer might replace or blank the reply — verifier-trigger patterns (state-set, action-command, online-query), actuator calls (TTS suppression), `get_entity` misses, raw tool-call recovery, thinking leaks. Gated turns behave exactly as before.
- The conversation entity now implements HA's modern `_async_handle_message(user_input, chat_log)` contract (chat-session management moves to core; conversation ids come from the chat log).

## v0.9.28 — Deterministic time shortcut, kaomoji-free TTS

**New:**
- **"What time is it" answers deterministically** (`shortcuts.py` `_try_time_shortcut`, all 6 languages via `i18n/<code>.yaml` `time_now`/`time_is`). Live measurements showed 5–8s of LLM round-trip for a clock read; now ~0.01s. Date questions stay with the LLM (localized weekday/month names aren't worth the i18n surface).

**Fixed:**
- **Kaomoji no longer reach TTS** (`orchestrator.py` `_strip_emoji`). `(╯°□°）╯︵ ┻━┻`, `(・_・)`, `ಠ_ಠ` are box-drawing/geometric/CJK/Kannada glyphs outside the emoji ranges, so the stripper passed them to the speech synthesizer. Bracketed groups containing such glyphs are removed whole, leftover glyphs swept after; `21°C` and ordinary parentheses survive.

## v0.9.27 — Latency: cache-stable prompts, keep_alive, background summarization

All four changes target the same number: seconds between speaking and hearing the reply on a local Ollama backend.

**Changed:**
- **Cache-stable prompt layout** (`orchestrator.py`). `[CURRENT TIME]` (minute-resolution) and `[LAST ACTION]` lived at the head of the system prompt, ahead of the tool schemas and history — every minute tick or actuation forced Ollama to re-prefill the entire 3–6k-token prompt. Volatile blocks (`[CURRENT TIME]`, `[USER FACTS]`, `[LAST ACTION]`) now ride in a late system message inserted directly before the newest user turn; the static head (base prompt + custom prompt + location) is byte-stable across turns, so Ollama's prompt prefix cache applies and each turn only pre-fills the new tail.
- **Per-message ha_local schema pruning is now opt-in** (`prune_tool_schemas`, default off). Changing the tool list per message busts the same prefix cache; the few hundred tokens pruning saved are cheaper than the re-prefill. Existing installs change behaviour on upgrade — re-enable in Advanced settings if you prefer pruning.
- **Summarization runs after the reply, in the background** (`orchestrator.py` `_schedule_summarization`). Inline summarization added a full LLM round-trip to whichever unlucky turn crossed the soft context limit; the hard truncation in `get_messages` covers the window until the background pass lands. Same per-conversation lock, so history stays consistent.
- **Shared aiohttp session for web_search and browse_url** (one TCP+TLS handshake, not one per call).

**New:**
- **`keep_alive` option** (Advanced → *Ollama keep-alive*). Sent per request on Ollama's native API (`30m`, `24h`, or `-1` = always loaded; empty = server default of 5m). Replaces the `OLLAMA_KEEP_ALIVE` env-var dance for keeping the model warm between voice turns.

**Not yet:** streaming replies (ChatLog delta streaming → streaming TTS) is the remaining big latency lever; it needs a conversation-entity rework and a live deploy-test cycle, tracked for the next minor.

## v0.9.26 — Context-budget correctness, MCP timeout, localized fallbacks, eval harness

**Fixed:**
- **Tool-schema tokens are now budgeted per request** (`orchestrator.py`, `context_manager.py`). The reserve was computed once at startup from MCP schemas only — before MCP had even connected — so the ~2.5k tokens of built-in ha_local/memory/web/browse schemas were never accounted for. Long conversations overran `num_ctx` and Ollama silently truncated the *front* of the prompt, dropping the system prompt mid-conversation. The orchestrator now measures the schema list actually being sent each turn and passes it to both summarization and history trimming.
- **MCP tool calls have a 15s ceiling** (`tools/mcp_client.py` `_CALL_TIMEOUT`). The integration's `response_timeout` only wraps the LLM HTTP call; a hung MCP tool stalled the whole voice turn indefinitely. Timeouts return an error string the model can relay.
- **Recovery guidance no longer directs the model at missing tools** (`tools/ha_local.py`, `orchestrator.py`). The set_area_state refusal paths told the model to CALL HassTurnOn/HassTurnOff — tools that only exist when HA's built-in MCP server is connected. The orchestrator now passes the actual tool-name set through, and installs without the Hass tools get honest guidance (report the found entity) instead of a dead end.
- **Failure strings are localized** (`i18n/*.yaml` `err_no_answer`, `err_process`, `note_tool_limit`). "I couldn't produce an answer for that." and friends were spoken in English on German/French/… voice pipelines. All six languages shipped.
- **Idle conversation state is evicted after 6h** (`context_manager.py` `evict_idle`, orchestrator lock/last-entity maps). HA mints a fresh conversation id per Assist session; the per-conversation history/lock dicts previously grew for the lifetime of the HA process.
- **CI actually installs the test requirements** (`.github/workflows/test.yml`, `requirements_test.txt`). CI installed an ad-hoc package list missing PyYAML (a hard import of the i18n loader) while `requirements_test.txt` listed `pytest-homeassistant-custom-component`, which the suite doesn't use (tests run against the stubs in `tests/conftest.py`). One file is now the single source of truth.

**Changed:**
- **Default web-search backend is DuckDuckGo** (`const.py`). Brave requires an API key; fresh installs that enabled web search without one got "API key not configured" apologies on every search. Existing entries keep their stored backend. Brave/Tavily remain the recommended quality picks.
- `forget` tool description now states explicitly that BOTH `index` and `fact` must be passed (the handler always enforced it; the schema's phrasing suggested `fact` was optional, sending small models into refusal loops).
- Removed the dead static-context plumbing in `tools/mcp_client.py` (`get_static_contexts` had no caller; it also fetched every server prompt at connect time for nothing).

**New:**
- **Offline eval harness** (`tests/eval/`): drives a live HA install via `POST /api/conversation/process` and scores replies against a per-prompt regex rubric — pass rates, latency medians, real state verification via HA templates for actuation prompts (`--full`), safe read-only subset by default. `home.yaml` (gitignored) carries per-install names. Baseline on the reference install (qwen3-class 8B, voice mode): 26/29 safe prompts, median turn 5.8s; shortcut-served turns ~0.02s.
- README: the "Listen for follow-up" section now documents the actual behaviour — voice-satellite turns always end (TTS feedback-loop protection); follow-up chaining on voice is satellite-side.

## v0.9.25 — Fix on/off shortcut crashing on real registries

**Fixed:**
- `_resolve_named_entity` raised `AttributeError: 'ComputedNameType' object has no attribute 'lower'` on Home Assistant 2026.6, where `entry.name` can be a non-`str` sentinel for auto-computed names. Iterating the entity registry hit such an entry, the whole on/off shortcut threw, and every command fell through to the LLM — so the v0.9.23 deterministic shortcut never actually fired on a live install (unit-test mocks used plain-string aliases and missed it). Candidate names are now filtered to real strings (`isinstance(c, str)`), and the `async_should_expose` check is wrapped fail-open so a single problem entity can't abort resolution. Regression test added.

## v0.9.24 — Bake single-device on/off guidance into the voice prompt

**Changed:**
- `SYSTEM_PROMPT_VOICE` now carries explicit, language-agnostic single-device on/off guidance: "switch/turn X on/off" (and the same in the user's language) is a VERB on a named device → resolve via `search_entities` then `HassTurnOn`/`HassTurnOff`, never `set_area_state`. This is the LLM fallback for cases the deterministic on/off shortcut (v0.9.23) does not catch — ambiguous/unresolved device names, multi-clause commands, and on/off for cover/climate. Behaviour no longer depends on a per-install custom system prompt. Per-language handling continues to live in the i18n shortcut (`i18n/<lang>.yaml`), keeping the prompt itself language-agnostic.

## v0.9.23 — Deterministic on/off shortcut (all languages)

**New:**
- Single named-device on/off now resolves deterministically *before* the LLM, in all 6 i18n languages. Phrasings like "switch the TV on", "TV on", and German "schalte den Fernseher ein" / "mach den Fernseher aus" no longer depend on the small model getting the tool call right.
- New `action_on` / `action_off` regex pattern lists in each `i18n/<code>.yaml` (each captures the device as `(?P<name>...)`). Add or improve a language's coverage by editing its YAML — no Python changes.
- `shortcuts.async_try_action_shortcut`: matches the per-language patterns, resolves the captured name to one exposed entity in switch/light/fan/input_boolean/humidifier/siren (exact then substring match, caller-area tiebreak), and dispatches `homeassistant.turn_on`/`turn_off`. Ambiguous or unresolved devices fall through to the LLM rather than actuating the wrong one. Empty reply (TTS suppression) on success.

## v0.9.0 — Data-driven multilingual support

**Breaking:**
- `trigger_languages` config option removed. Language now follows the HA Assist pipeline (`Buddy (DE)`, `Buddy (FR)`, etc.).

**New:**
- All per-language data lives in `custom_components/ai_plugin/i18n/<code>.yaml` files. Adding a new language is a YAML PR — no Python changes.
- Schema-validated at load time; bad YAML fails the integration cleanly.
- 6 languages shipped: en, de, fr, es, pt, pl. Contributors can add more via PR — see `i18n/CONTRIBUTING.md`.

**Changed:**
- Per-language hint blocks dropped from the system prompt (~80–160 tokens saved per request).
- Sun shortcut, narration regex, attribute-in-area shortcut now language-symmetric — every supported language uses the same code paths and templates.

## [0.6.5] - 2026-04-28

### Changed

- **`continue_conversation` now defaults to True** (`const.py` `DEFAULT_CONTINUE_CONVERSATION = True`) and the close-phrase check no longer requires the option to be enabled (`conversation.py`). Previously, the AI Plugin only emitted `continue_conversation=True` when the advanced option was toggled on; this caused the satellite-side persistent-conversation flow (SmartMic + mic_to_mediaplayer ≥ v1.4.2) to receive `False` after every reply and end the conversation immediately. With this change, every non-close-phrase reply emits `True`, so the ESP `Persistent Conversation` switch and `end_persistent` routing in mic_to_mediaplayer now behave as intended (stay listening until close phrase). Existing users who explicitly set the option keep their override.

## [0.6.4] - 2026-04-28

### Fixed

- **Conversation close-phrases failed when followed by punctuation** (`conversation.py`). `_match_close_phrase` did a whitespace-bounded substring check (`" {phrase} " in normalized`), so STT outputs like `"Thanks Jarvis."` (trailing period) or `"thanks, Jarvis"` (comma) never matched any entry of `CONVERSATION_CLOSE_PHRASES` and the conversation kept the mic open even when `CONF_CONTINUE_CONVERSATION` was on. The matcher now collapses runs of non-word characters to single spaces (Unicode-aware so German umlauts stay intact) before comparing, so common end-of-utterance punctuation no longer blocks the match. Phrases themselves are normalised the same way so apostrophes (e.g. `that's all`) keep matching.

## [0.6.3] - 2026-04-28

### Added

- **Weather forecast block in `get_entity` for `weather.*` entities** (`tools/ha_local.py`). After reading state + attributes, the tool now calls the `weather.get_forecasts` service (`type=daily`) and appends a `forecast (daily, next 5):` block listing date, condition, hi/lo and precipitation probability. Previously the model only saw current condition + temperature, so questions like "what is the weather going to be today?" or "will it rain tomorrow?" returned only the current state. The forecast comes from the integration HA already polls (DWD, Met.no, AccuWeather, OpenWeatherMap, …); no internet call is added. Entities that do not implement `get_forecasts` silently skip the block. `_get_entity` is now async to make the service call.
- **Prompt guidance updated** (`const.py`). Default and voice system prompts now tell the model to use the forecast block for "today / tomorrow / this week / will it rain / umbrella" style questions, and drop the older "never invent forecasts" wording (no longer needed because the data is now in the tool result).

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
