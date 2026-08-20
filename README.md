# AI Plugin for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant, built for **local LLMs on modest hardware**. Use any OpenAI-compatible backend — Ollama, llama.cpp, LM Studio, OpenAI, xAI — as your Assist conversation agent, with streaming voice replies, deterministic shortcuts, web search, voice timers, MCP tool extensibility, and an AI Task entity for automations.

## Features

- **Any LLM backend** — Ollama (first-class, native `/api/chat`), llama.cpp, LM Studio, OpenAI, xAI Grok, any OpenAI-compatible endpoint
- **Streaming replies** — sentence-safe deltas into HA's chat log; voice pipelines with streaming TTS start speaking after the first sentence instead of after full generation (Ollama backend)
- **Deterministic shortcut layer** — common voice commands bypass the LLM entirely (~0.01–0.05 s instead of seconds): clock time, sunrise/sunset, room sensor readings, single-device on/off, cover open/close, media transport, volume/mute — all data-driven per language
- **On-demand entity discovery** — `list_areas` / `list_entities` / `get_entity` / `search_entities` hit HA registries in-process; no YAML entity dump bloating the prompt
- **Grounding verifiers** — state questions, action commands, and time-sensitive questions are checked against actual tool usage; a model that answered from imagination gets one corrective re-run
- **Voice timers** — first-class start/cancel/pause/extend with deterministic duration parsing, plus an optional **announce mode** that plays timer completions through a media player instead of the satellite's own ring (works on satellites without timer firmware)
- **AI Tasks** — the same backend serves `ai_task.generate_data` for automations and scripts: plain text or structured JSON
- **Web search** — DuckDuckGo (zero config, default), Brave, Tavily, SearXNG; plus `browse_url` for reading a specific page
- **Model introspection** — the config flow warns when a model can't call tools or when the context window exceeds what the model supports (Ollama)
- **Smart context management** — per-request token budgeting, sliding window, background summarization, cache-stable prompt layout for Ollama's prefix cache
- **Long-term memory** — `remember` / `recall` / `forget` with per-user fact files, auto-injected so small models answer without a tool call
- **MCP extensibility** — connect external MCP servers (HTTP and stdio), including HA's own MCP server for `HassTurnOn`-style intents
- **6 languages** — en, de, fr, es, pt, pl; adding a language is a YAML PR (`i18n/CONTRIBUTING.md`)
- **Offline eval harness** — `tests/eval/` drives a live install end-to-end and scores replies, so changes get measured instead of vibed

> **Tested only with Ollama.** The OpenAI-compatible endpoint should accept the other backends, but only Ollama (local) has been exercised end-to-end against the prompt suite below. Reports for OpenAI / xAI / LM Studio / llama.cpp setups welcome.

## Recommended Models

The most critical factor for reliable entity control is **tool calling quality**. Models that hedge or ask clarifying questions instead of executing are a poor fit regardless of benchmark scores. The config flow now checks this for you on Ollama: models that report no `tools` capability are rejected at setup.

### Benchmark (8 GB VRAM)

Bench-tested July 2026 against plugin v0.9.35 via HA Assist `/api/conversation/process` in **voice mode**, on an RTX 3060 Ti (8 GB) running Ollama. 29 read-only prompts covering clock/sun, room sensors, "which lights are on", inventory, climate, local + named-place weather, web search, bare-noun sanity, and a media question. Pass counts alone hide *how* a model fails, so the notes distinguish honest misses from confident fabrications — the distinction that matters most in a home agent.

Latency below is the **median over the ~20 LLM-driven turns** per run — the other 9 prompts are answered by the deterministic shortcut layer in ~10 ms regardless of model, so they don't discriminate. `qwen3.5:397b-cloud` runs via Ollama Cloud and is included only as an accuracy ceiling, not as a local option.

| Model | Size | Pass rate | Median latency | Notes |
|-------|------|-----------|----------------|-------|
| **qwen3.5:9b** | 6.6 GB | **93 %** (27/29) | 4.3 s | **Top local pick.** Highest local accuracy, and decisively: its only two misses were *honest* ("I couldn't find the weather in those search results"), never a fabrication. Answered the thermostat (setpoint + heating state), whole-home temperatures, and "what's playing" correctly where the 8Bs did not. ~1.5 s slower than the 8Bs — worth it. Reasoning model; the plugin sends `think: false` automatically. |
| qwen3:8b | 5.2 GB | 86 % (25/29) | **2.5 s** | Faster, but **fabricated a concert** ("Rolling Stones at the Berlin Olympic Stadium") in answer to *"what's playing?"*, and bailed with "couldn't produce an answer" on climate and temperature. Fine for lower-latency, actuation-heavy use; less trustworthy on open questions. |
| qwen3-abliterated:8b | 5.0 GB | 90 % (26/29) | **1.8 s** | Fastest of the capable models and scores well — but the score is **inflated by confabulation**: it invented specific Tokyo weather that the honest models declined to state. Also safety-ablated. Not recommended despite the number. |
| mistral:7b | 4.4 GB | 86 % (25/29) | 2.4 s | Solid on actuation, but couldn't name the weekday for "what day is it?" and missed a German state query. Weaker on dates and non-English. |
| gemma4:e2b | Gemma 3n E2B (~2B active) | 69 % (20/29) | **1.7 s** | Fastest by far and impressively coherent for its size, but weak on multi-step tool loops (missed weather, climate, inventory). A curiosity for very constrained setups, not for reliable control. |
| _qwen3.5:397b-cloud_ | cloud (reference) | **100 %** (29/29) | 6.2 s | Accuracy ceiling — and every pass was honest (it declined Tokyo weather rather than inventing it). Cloud round-trip; not a local option. |

**What the pass rate hides:** the local models cluster tightly on score (86–93 %), but they fail in opposite ways. `qwen3.5:9b` and the cloud reference decline gracefully when live data isn't there ("I couldn't find that in the search results"); `qwen3:8b` and the abliterated model instead emit confident fictions — a concert that isn't happening, weather they never fetched. For an assistant that speaks its answers aloud, an honest miss is recoverable and a fabrication is not — so the recommendation follows the failure *mode*, not the raw percentage. Read the notes column, not just the number.

> **Context window on 8 GB:** model weights + KV cache must stay under ~7.5 GB. A 7–8B Q4_K_M model uses 4.7–5.2 GB weights; KV cache ≈ 0.2 GB per 1 K tokens.
>
> **Minimum recommended context: 16384** (the integration default). Multi-step tool loops routinely emit 6–10 K of intermediate tokens; 8192 starves them. Avoid values above ~24 000 on an 8 GB card. Setting a context window larger than the model supports is now rejected in Advanced settings (Ollama).

### Update — August 2026: the 8 GB tier after Qwen 3.5

`qwen3.5:9b` was released 2026-03-02. Five months on, the honest headline is that **the 8 GB tier has largely plateaued**: Qwen 3.6 (April 2026) shipped only a 27B dense and a 35B-A3B MoE, so there is no direct Qwen upgrade at this size. Three post-3.5 models do fit — `granite4.1:8b` (IBM, April), `lfm2.5:8b` (Liquid, May — an 8.3B-total / 1.5B-active MoE), and `gemma4:12b` (Google, June).

> **Different method — not comparable to the 29-prompt table above.** These runs hit Ollama's OpenAI-compatible endpoint directly with HA-Assist-shaped German tool schemas: 10 adversarial cases × 3 repetitions at temperature 0, scoring computed arguments ("auf die Hälfte"), unit conversion, read-before-write on conditionals, refusal/clarification, negation traps, multi-turn coreference. It stresses **tool-calling under pressure**, not the plugin end-to-end, and the 29-prompt voice suite has **not** been re-run against these models. Treat the two tables as different axes.

| Model | On disk | Fits 8 GB @ 16 K? | Adversarial tool set | Exact area name | Median |
|-------|---------|-------------------|----------------------|-----------------|--------|
| **gemma-4-12B UD-Q3_K_XL** | 6.2 GB | **yes** — 6.5 GB, 100 % GPU | **30/30 (100 %)** | **100 %** | 5.2 s |
| gemma-4-12B IQ4_XS | 6.6 GB | **no** — 6.9 GB, 3 % on CPU | 27/30 (90 %) | — | 6.1 s |
| gemma4:12b (Q4_K_M) | 7.6 GB | **no** — 7.9 GB, 16 % on CPU | 30/30 (100 %) | — | 10.4 s |
| `qwen3.5:9b` | 6.6 GB | yes — 5.8 GB, 100 % GPU | 21/30 (70 %) | **100 %** | 2.6 s |
| `granite4.1:8b` | 5.3 GB | yes — 6.7 GB, 100 % GPU | 24/30 (80 %) | **83 %** ✗ | **0.6 s** |
| `lfm2.5:8b` | 5.2 GB | yes — 5.4 GB, 100 % GPU | 18/30 (60 %) | **67 %** ✗ | 1.9 s |

**Fit is decided at your context window, not by the file size.** All three Gemma 4 12B quants look like they fit on paper; only one actually does once a 16 K KV cache is allocated. `IQ4_XS` is 6.6 GB on disk and *still* spills 3 % to CPU. Check `ollama ps` and read the `PROCESSOR` column at your real `num_ctx` — a few percent on CPU costs multiples in latency, not a few percent (`gemma4:12b` at Q4_K_M: 16 % on CPU → 10.4 s median).

**Non-English area names are a hard filter, and raw scores hide it.** Home Assistant matches areas by string, so a model that "helpfully" rewrites a room name fails the intent silently. `granite4.1:8b` is by far the fastest model here and scored *above* `qwen3.5:9b` on tools — but it corrupts umlauts systematically, emitting **"Böro"** 6/6 at production temperature (and "Büoro", "Bureau" elsewhere) for **"Büro"**. `lfm2.5:8b` translates room names outright ("Küche" → `kitchen`). Both are disqualifying for a German, French or Swedish install regardless of tool scores. If you benchmark models yourself, **grade area names with exact string equality** — a substring check hides exactly this class of defect.

Two further cautions from the same runs:

- **`lfm2.5:8b` is prompt-brittle.** It scored 80 % on the default system prompt and *dropped to 60 %* when the prompt was extended — it stopped emitting tool calls at all on the computed-argument cases. A model whose tool-calling degrades when you edit unrelated prompt text is a maintenance trap.
- **Action polarity is a prompt problem, not a model problem.** An apparent "inverted blinds" bug (`Jalousien zu` → `HassTurnOn`) turned out to be a missing convention in the system prompt. Once it states that closing a cover is `TurnOff`, **all four models hit 100 %**.

**Verdict.** `gemma-4-12B UD-Q3_K_XL` ([unsloth/gemma-4-12B-it-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-GGUF)) is the new accuracy pick for 8 GB — the only 12B that stays fully in VRAM at 16 K, perfect on both the adversarial set and area fidelity. It costs latency: ~3.6 s median on everyday commands versus ~2.9 s for `qwen3.5:9b`, a worse tail, and ~16 s on the first turn after a keep-alive eviction (6.5 GB to load). `qwen3.5:9b` remains the better choice if responsiveness matters more than edge-case correctness, and it is still the only model validated against the 29-prompt voice suite.

```bash
ollama pull hf.co/unsloth/gemma-4-12B-it-GGUF:UD-Q3_K_XL
```

### Update — August 2026 (2): multi-turn behaviour, and the settings that don't move the needle

The two tables above are single-shot: one utterance, one judgement. This run is the other axis — a **threaded conversation**, driven through `assist_pipeline/run` (intent stage) so it is the real Assist path, with `conversation_id` carried forward so follow-ups and pronouns actually refer to something. 11 German turns × 2 passes on `gemma-4-12B UD-Q3_K_XL`, RTX 3060 Ti, at Temperature 0.3 / `top_p` 0.4 / Context Window 16384 / Max Tokens 512 unless a row below says otherwise. **Actions were verified in Home Assistant's state history, not read off the agent's replies** — voice mode returns an empty string on a successful action, which looks identical to a turn that did nothing.

Live tool surface during the run: **63 tools ≈ 8.6 K schema tokens**, plus a 2.1 K system prompt — **10.7 K of a 16 K window spent before the user says a word** (HA intents plus five MCP servers: time, fetch, wikipedia, calculator, and HA's own MCP endpoint).

| Case | Utterance | Result |
|------|-----------|--------|
| Ambiguous friendly name | two entities both named "Nachtlicht" (a `light.` and a `switch.`) | picks the light, consistently |
| Ellipsis | "Und den LED-Strip auch." | correct device on, verb inferred |
| Anaphora, two referents | "Mach beide wieder aus." | both off in one turn |
| Two devices, one turn | "Schalte X und Y gleichzeitig ein." | both on |
| Self-correction | "Nein, ich meinte nur X — mach Y wieder aus." | only Y off, X left alone |
| Area reference | "Mach das Licht in der Küche aus." | resolves the room, correct device off |

All six pass in every configuration tested, both passes, with the state changes confirmed in history. **Median turn 4.5 s, p90 6.6 s** on this hardware.

#### The knobs, ranked by how much they changed anything

| Lever | Change | Result |
|-------|--------|--------|
| `top_p` | 0.4 → 0.9 | no measurable difference |
| Temperature | 0.3 → 1.0 | no difference in task behaviour; only output variety — at 0.3 the same joke came back byte-identical every pass |
| Context Window | 16384 → 32768 | no difference on this set (see the KV note below) |
| `prune_tool_schemas` | off → on | **no-op** — zero prune events in 22 turns |
| Max Tokens | 512 → 1024 | **actively worse** — see below |

**Max Tokens 512 in the table below is load-bearing, not a default.** On a turn where the model reasons at length without converging, generation runs to the cap and comes back with *empty content*, and the plugin substitutes its fallback line. Raising the cap does not rescue that turn — it only lets it run longer: the same turn went from 18.7 s to **32.8 s**, past the plugin's own 30 s `response_timeout`. A bigger budget buys a slower failure.

**`prune_tool_schemas` never fired.** Across 22 turns of real voice traffic the budget line read `tools=8634` every single time. It does fire occasionally on entity-listing utterances — one observed case dropped `list_entities` and `list_areas`, about 4 % of the schema budget. Combined with the prefix-cache cost noted earlier, off remains the right default.

**A 32 K window does fit on 8 GB — if the KV cache is quantised.** The ≤24 000 guidance above assumes an f16 KV cache (~0.2 GB per 1 K tokens). With `OLLAMA_KV_CACHE_TYPE=q8_0` that halves, and `gemma-4-12B UD-Q3_K_XL` loads at `num_ctx=32768` still reporting **6.5 GB / 100 % GPU** in `ollama ps`. Bigger is not automatically better, though: the Context Window also sets the plugin's history budget (`soft_limit = (context − system − tools) × 0.65`), so a wider window means longer prompts deep in a conversation. It bought nothing measurable here.

#### Liquid's QAD checkpoints (LFM2.5, Q4_0) — evaluated, not recommended

Liquid released [Quantization-Aware Distillation](https://huggingface.co/blog/LiquidAI/qad) checkpoints in August 2026: the BF16 teacher is distilled into the 4-bit student, recovering ~97 % of BF16 quality at Q4_0 file sizes. Tested here because the speed is genuinely striking — `LFM2.5-2.6B-QAD-Q4_0` is 1.8 GB resident and runs **165 tok/s decode against gemma-4-12B's 35, with 6.4 K vs 1.9 K tok/s prefill**, which on read-only questions and chit-chat means 1.4–3.0 s turns where the 12B takes 4–10 s. Asked *once*, in isolation, it picks the right tool 81–89 % of the time.

It still does not work in this integration. In the plugin's actual loop — 63 tools, up to 10 iterations, tool results fed back — it does not converge: it invents entity ids and re-issues calls that have already succeeded, exhausting `max_tool_iterations` without changing any device state. QAD is a real improvement over plain Q4_0 at identical file size (it also spends ~27 % fewer thinking tokens), but the gap that matters here is loop discipline, not quantisation quality.

Two specifics worth carrying:

- **`LFM2.5-1.2B` fails dangerously, not just poorly.** It replies *"Ich habe die Kaffeemaschine eingeschaltet"* in fluent German **without emitting a tool call** — over voice, indistinguishable from success. Score it on emitted tool calls, never on the reply text.
- **Thinking cannot be disabled on these GGUFs.** Their chat template opens `<|im_start|>assistant\n<think>` unconditionally, so the `think: false` the plugin sends on every Ollama call is silently ignored — as are `/no_think` and prompt-level instructions. Budget ~150–220 thinking tokens per turn against Max Tokens.

Consistent with the `lfm2.5:8b` row above: this family is fast and small, and not yet a fit for multi-step home control.

### Recommended configuration

| Setting | Value |
|---------|-------|
| Model | `qwen3.5:9b` (best latency/accuracy balance, failures are honest, validated against the full voice suite) — or `hf.co/unsloth/gemma-4-12B-it-GGUF:UD-Q3_K_XL` for the highest accuracy and non-English area names, at ~1.5× the latency. `qwen3:8b` for ~2× lower latency if you mostly do device control. Avoid `granite4.1:8b` and `lfm2.5:8b` on non-English installs — see the August 2026 notes. |
| Temperature | 0.2 |
| top_p | 0.4 |
| Context Window | **16384** |
| Max Tokens (`num_predict`) | 512 |
| Keep-alive | `30m` (`-1` on a dedicated GPU) |
| Web Search | DuckDuckGo (default) or Tavily/Brave with a key |

## How it stays fast on local models

Latency on a 7–9B model is won or lost in three places, and the plugin attacks all of them:

1. **Skip the LLM when possible.** The deterministic shortcut layer answers clock time, sun times, "temperature in the bedroom", "turn on the mood light" (all 6 languages), "open the blinds", "pause", "volume up", "mute" in ~0.01–0.05 s by hitting HA registries and services directly. Misses and ambiguity fall through to the LLM — the shortcut never guesses.
2. **Reuse Ollama's prompt prefix cache.** The system prompt and tool list are byte-stable across turns; volatile context (`[CURRENT TIME]`, `[USER FACTS]`, `[LAST ACTION]`) rides in a late system message just before the newest user turn. Each request pre-fills only the new tail instead of the whole 3–6 K-token prompt. (Per-message tool-schema pruning defeats this cache and is therefore opt-in.)
3. **Start speaking before generation finishes.** Replies stream sentence-by-sentence into HA's chat log; with a streaming-capable TTS engine the satellite starts speaking after the first sentence. Streaming is sentence-safe by design: every sentence passes the same sanitation as the final reply, the trailing sentence is held back, and streaming shuts off entirely on any turn where a grounding verifier might rewrite the reply or TTS suppression might blank it.

Also in this bucket: summarization runs *after* the reply in the background instead of blocking the unlucky turn that crosses the context limit, per-request token budgeting reserves space for the tool schemas actually being sent, and the **Keep-alive** option keeps the model in VRAM between turns without any `OLLAMA_KEEP_ALIVE` server configuration.

## Ollama Configuration

### Context Window — one setting controls everything

The plugin talks to Ollama's native `/api/chat` and passes `num_ctx` on every request, so the **Context Window** setting is the single number controlling both the plugin's internal token budget and the context size Ollama loads the model with. No Modelfile changes required.

### Temperature

**0.1–0.3** for home control. Higher values make models ask clarifying questions instead of acting. The benchmark above ran at 0.2.

### Reasoning models (qwen3, deepseek-r1)

The plugin sends `think: false` on every Ollama call so final answers land in `content`. Requires Ollama 0.20+. No Modelfile tweaks needed.

### Keep-alive

Ollama unloads models after 5 minutes of inactivity by default, so the first command after a quiet stretch pays the full model-load cost. Set **Ollama keep-alive** in Advanced settings (`30m`, `24h`, or `-1` for always loaded) — sent per request, no server configuration. Leave empty to keep the server default.

This matters more the larger the model: on an 8 GB card a 6.5 GB model takes **~16 s** to load, so the default `5m` turns the first question after a quiet evening into a timeout-feeling pause. If the GPU is dedicated to Home Assistant, use `-1`.

## Voice timers

Timers are first-class tools (`start_timer`, `cancel_timer`, `pause_timer`, `unpause_timer`, `increase_timer`, `decrease_timer`, `timer_status`) with a deterministic safety net: the spoken utterance is ground truth for the duration, so "10 seconds" can never become 10 minutes even if the model mis-fills a slot.

**Default:** HA delivers the timer to the satellite, which shows its countdown and plays its ring sound locally. This requires timer-capable satellite firmware (HA Voice PE and current ESPHome voice satellites have it).

**Announce mode** (Advanced → *Announce timers instead of on-device ring*): the timer carries a `conversation_command` instead — HA's own TimerManager calls the plugin back at expiry, and the plugin plays a localized announcement ("Timer pasta is up.") via `mic_to_mediaplayer.announce` when that integration is present, else HA's native `assist_satellite.announce`. Use this when the satellite's own speaker is silenced because audio is routed to better speakers, or when the satellite has no timer firmware at all — announce-mode timers work on **any** device. Trade-offs: the satellite shows no countdown LEDs and plays no local ring; like all HA voice timers, timers don't survive an HA restart.

## AI Tasks (automations & scripts)

The integration provides an `ai_task` entity, so automations can use the same local model:

```yaml
action: ai_task.generate_data
data:
  task_name: morning summary
  instructions: Summarize today's calendar in two sentences.
  entity_id: ai_task.ai_plugin
```

With a `structure` schema, the model is instructed to return matching JSON; code fences are tolerated and invalid JSON raises a proper error instead of returning garbage.

## Long-term memory

The assistant remembers facts across conversations — and across Home Assistant restarts — through three tools it calls from natural speech:

- *"Remember that I take my coffee black."* → `remember`
- *"What's my name?"* / *"What did I tell you about my coffee?"* → answered from memory
- *"Forget that I take my coffee black."* → `forget`

**Recall is usually free.** At the start of every turn the plugin loads the user's saved facts into a `[USER FACTS]` block in the prompt (a late, cache-stable system message, so it doesn't break Ollama's prefix cache). The model answers "what's my…" questions straight from that block with no tool round-trip — which matters on small local models, where each tool call is a full extra generation pass. The `recall` tool is only a fallback for when the block is absent.

**Per-user and persistent.** Facts live as plain JSON at `<config>/.ai_plugin_memory_<user_id>.json` — one file per Home Assistant user, so users don't see each other's facts (unauthenticated turns use `…_anonymous.json`). The files are readable text in your config directory: don't ask it to remember secrets.

**`forget` won't delete the wrong fact.** It prefers removing by the fact's number in `[USER FACTS]` (language-independent), falling back to a fuzzy substring match. Ask it to forget something that isn't stored and it says so rather than guessing an index and deleting a neighbour — with a cross-language check so an English request can still clear the matching German-stored fact.

## Conversation continuity (`Listen for follow-up after voice replies`)

When the option is on (default), every non-close-phrase reply returns `continue_conversation=True` — HA's standard mechanism. Voice satellites with their own speaker (HA Voice PE, standard ESPHome satellites) re-arm the microphone after the reply, so you can keep talking without the wake word. A close-phrase detector (`thanks`, `bye`, `das war's`, …) ends the loop cleanly.

**Exception — TTS on a separate speaker:** satellites that route reply audio to an external media player re-hear their own TTS and re-trigger listening — an acoustic feedback loop that makes the assistant talk to itself. Two mitigations, both on by default / available:

- **Self-echo filter** (Advanced → *Ignore the satellite hearing its own reply*, default on) — the plugin remembers what it just said per device and silently drops any incoming voice turn that matches (≥80% token overlap within 30 s). The loop dies but the session stays open, so real follow-ups still work. Short turns are never filtered. This is the recommended fix and keeps barge-in.
- **Force-end conversations on these satellites** (Advanced, device list) — a blunter guard that ends the session after one turn on the listed devices. Use it if a satellite's echo is too garbled by STT for the filter to catch.

## Requirements

- Home Assistant **2025.7.0** or newer
- [HACS](https://hacs.xyz/docs/use/) (delivers the integration)
- A reachable LLM endpoint, e.g.
  - [Ollama](https://ollama.com) — `http://<host>:11434/v1`
  - OpenAI — `https://api.openai.com/v1` (API key)
  - LM Studio — `http://<host>:1234/v1`
  - llama.cpp server — `http://<host>:8080/v1`
  - xAI Grok — `https://api.x.ai/v1` (API key)

## Installation

1. **HACS → Integrations → ⋮ → Custom repositories** → add `https://github.com/bigbabol1/AI-plugin` (category **Integration**)
2. Install **AI Plugin**, **restart Home Assistant**
3. **Settings → Devices & Services → Add Integration** → *AI Plugin*
4. Wire it into voice/chat: **Settings → Voice assistants** → set **Conversation agent** = *AI Plugin*

## Configuration reference (Advanced)

All settings are editable post-install via **Settings → Devices & Services → AI Plugin → Configure**.

| Option | Default | What it does |
|--------|---------|--------------|
| System prompt | persona template | Appended to the built-in prompt (never replaces it) |
| Context Window | 16384 | Token budget AND Ollama `num_ctx`, one number |
| Summarization | on | Condenses old turns in the background near the limit |
| Voice mode | off | Forces the compact voice prompt even for text |
| Listen for follow-up | on | Keep the session open after replies |
| Enable thinking | off | Let reasoning models emit chain-of-thought (slower; disables streaming) |
| Max tool iterations | 10 | Cap on LLM↔tool round-trips per turn |
| Timeout | 30 s | Per-LLM-call ceiling (MCP tools have their own 15 s cap) |
| Temperature / top_p / Max tokens | unset | Sampling; 0.2 / 0.4 / 512 recommended |
| Ollama keep-alive | unset | Keep the model in VRAM between requests (`30m`, `-1`) |
| Prune tool schemas | off | Per-message schema pruning; defeats Ollama's prefix cache — leave off |
| Announce timers | off | Timer completions via media player instead of on-device ring |
| Force-end conversations on these satellites | none | Feedback-loop guard for TTS-on-separate-speaker setups |
| Location bias | on | Scope "near me" web searches to your home (fails open, opt-out) |
| Location entity | unset | Live location source instead of HA's home coordinates |

## Web Search Backends

Default is **DuckDuckGo** — zero config, works out of the box. For better quality, add an API key for Brave or Tavily.

| Backend | Cost | API key | Notes |
|---------|------|---------|-------|
| DuckDuckGo | Free | none | Default; may be rate-limited; US-biased on ambiguous place names (mitigated by `city, country` injection) |
| Brave Search | Free tier / paid | [api.search.brave.com](https://api.search.brave.com/app/) | Reliable |
| Tavily | Free tier / paid | [tavily.com](https://tavily.com/) | Best quality, AI-optimized |
| SearXNG | Free (self-hosted) | none — instance URL | Full control, no third-party traffic |

For reading one specific page (rather than searching), the model has a separate `browse_url` tool. Note it fetches through **Jina Reader** (`https://r.jina.ai/`) to convert the page to clean text, so the target URL is sent to that third-party service — the only outbound hop in an otherwise local-first integration besides your chosen search backend.

## MCP Servers

Add any MCP-compatible tool server in the integration settings — HTTP and stdio transports, presets for HA's built-in MCP server (recommended: provides `HassTurnOn`/`HassLightSet`-style intents), time, fetch, SQLite, and more. MCP tool calls are capped at 15 s so a hung server can't stall a voice reply.

## Evaluating changes (`tests/eval/`)

The unit suite mocks every LLM call; the eval harness measures the real thing. It drives a live install via `/api/conversation/process`, scores replies against per-prompt regex rubrics, verifies actuations against live entity state, and reports pass rates and latency percentiles. Read-only prompts by default; `--full` adds actuation/timer/memory prompts that clean up after themselves. See `tests/eval/README.md`.

## Troubleshooting

- **"Cannot connect" on setup** — the Provider URL must end with `/v1` for OpenAI-compat backends and be reachable from the HA host.
- **"Model reports no tool-calling capability"** — the config flow checked Ollama's `/api/show`; pick a tool-capable model (`qwen3`, `qwen2.5`, `mistral`).
- **Model asks questions instead of acting** — lower temperature to 0.1–0.3, or pick a stronger tool-calling model.
- **Slow first response after idle** — set **Ollama keep-alive** in Advanced settings.
- **Replies don't start speaking early** — streaming needs the Ollama backend and a streaming-capable TTS engine in your pipeline; other setups get the full reply at once (no harm).
- **Timer LEDs count down but no sound** — the ring plays on the satellite's *local* speaker; if yours is silenced, enable **Announce timers** (see Voice timers above).
- **Empty replies from reasoning models** — the plugin sends `think: false`; upgrade Ollama to 0.20+ if it persists.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
