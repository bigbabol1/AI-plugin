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

Bench-tested July 2026 against plugin v0.9.35 via HA Assist `/api/conversation/process` in **voice mode**, on an RTX 3060 Ti (8 GB) running Ollama. 29 read-only prompts covering clock/sun, room sensors, "which lights are on", inventory, climate, local + named-place weather, web search, bare-noun sanity, and a media question.

Latency below is the **median over the ~20 LLM-driven turns** per run — the other 9 prompts are answered by the deterministic shortcut layer in ~10 ms regardless of model, so they don't discriminate. `minimax-m2.7` runs via Ollama Cloud and is included only as an accuracy ceiling, not as a local option.

| Model | Size | Pass rate | Median latency | Notes |
|-------|------|-----------|----------------|-------|
| **qwen3:8b** | 5.2 GB | **86 %** (25/29) | **3.3 s** | **Top local pick.** Reliable, no fabrications, decent latency — beat every other local model here on accuracy *and* speed. Reasoning model; the plugin sends `think: false` automatically. |
| qwen3.5:9b | 6.6 GB | 82 % (24/29) | 7.4 s | Capable but noticeably slower (9B) and missed a couple of weather/state prompts. Reasonable if you specifically want the newer model and can spare the latency. |
| qwen3-abliterated:8b | 5.0 GB | 86 % (25/29) | 2.5 s | Fast and scores well, but fabricated a "Berlin events" answer to "what's playing?" — the confident-hallucination failure you least want in a home agent. Not recommended despite the score. |
| hermes3:8b | 4.7 GB | 68 % (20/29) | 2.2 s | Fast but weak on weather, multi-light state, and climate. Not recommended. |
| gemma4:e2b | Gemma 3n E2B (~2B active) | 68 % (20/29) | **0.9 s** | Fastest by far and impressively coherent for its size, but weak on multi-step tool loops. A curiosity for very constrained setups, not for reliable control. |
| _minimax-m2.7:cloud_ | cloud (reference) | **93 %** (27/29) | 10.4 s | Accuracy ceiling — gave the only complete thermostat answer (setpoint + current temp). Cloud round-trip latency (tail ~40 s); not a local option. |

> **Context window on 8 GB:** model weights + KV cache must stay under ~7.5 GB. A 7–8B Q4_K_M model uses 4.7–5.2 GB weights; KV cache ≈ 0.2 GB per 1 K tokens.
>
> **Minimum recommended context: 16384** (the integration default). Multi-step tool loops routinely emit 6–10 K of intermediate tokens; 8192 starves them. Avoid values above ~24 000 on an 8 GB card. Setting a context window larger than the model supports is now rejected in Advanced settings (Ollama).

### Recommended configuration

| Setting | Value |
|---------|-------|
| Model | `qwen3:8b` (top pick; `qwen3.5:9b` if you want the newer model and can spare the latency) |
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
