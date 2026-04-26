# AI Plugin for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search, MCP tool extensibility, and smart context management.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **On-demand entity discovery** — `list_areas`, `list_entities`, `get_entity`, `search_entities` tools hit HA registries in-process; no YAML dump into the prompt
- **Assist exposure aware** — discovery tools default to entities you've marked as exposed to conversation; diagnostic sensors stay hidden unless explicitly requested
- **Web search** — Brave Search (default), Tavily, SearXNG (self-hosted), DuckDuckGo (zero config)
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models
- **Voice mode** — compact system prompt when triggered via HA Assist voice pipeline
- **XML fallback** — tool calling for models without native function-calling support

## Recommended Models

The most critical factor for reliable entity control is **tool calling quality**. Models that hedge or ask unnecessary clarifying questions instead of executing commands are a poor fit, regardless of their general benchmark scores.

### 8 GB VRAM (e.g. RTX 3060 Ti, RTX 3070)

Bench-tested April 2026 against this plugin via HA Assist `/api/conversation/process` in **voice mode** (`device_id` set → `SYSTEM_PROMPT_VOICE` + `strip_urls=True`). 51 prompts covering light/climate/sensor/media/weather/web/memory/multi-step/multilingual. Plugin v0.5.46, num_ctx 16384, temperature 0.2, on Ollama at `192.168.0.66:11435` (RTX 3060 Ti, 8 GB).

| Model | Ollama tag | Weights | Pass rate | Avg latency | Texas-class hallucinations | Notes |
|-------|-----------|---------|-----------|-------------|----------------------------|-------|
| **Qwen3 8B** | `qwen3:8b` | ~5.2 GB | **86.3 %** (44/51) | **1.8 s** (median 1.4 s, max 4.9 s) | 0 | **Top recommendation.** Most consistent latency, perfect on weather (5/5) and web (5/5), strong on lights (9/10). Emits CoT into `thinking`; plugin passes `think: false` so `content` is populated. The HA-Assist `homeassistant:latest` alias points here. |
| **Qwen2.5 7B** | `qwen2.5:7b` | ~4.7 GB | **88.2 %** (45/51) | 2.6 s (median 0.7 s, max **37.8 s**) | 0 | Tied at the top by score, slightly higher tail latency on multi-step tool loops. No reasoning-mode quirks. Pick this if you want lower VRAM headroom or non-think behaviour. |
| Hermes 3 8B | `hermes3:8b` | ~4.7 GB | 74.5 % (38/51) | 2.1 s (median 1.5 s, max 23.5 s) | **1** (returned a Super Bowl LA Rams story to "events") | Llama-3.1 base, tool-calling FT. Weak on weather (2/5 — refuses or asks for location), weak on light entity resolution (6/10). Not recommended for HA. |
| Mistral 7B | `mistral:7b` | ~4.1 GB | not in this round | — | — | Native FC, no system-prompt scaffolding needed. Was strong in earlier 25-prompt runs. |
| Llama 3.1 8B | `llama3.1:8b` | ~4.8 GB | not in this round | — | — | Good general baseline; bench separately for your install. |

**Headline:** Qwen3 8B and Qwen2.5 7B are within one prompt of each other on the raw rubric, but **once you discount environment misses and rubric edge-cases the gap widens to ~96 % vs ~92 %** — see *Failure analysis* below. Qwen3 8B is the safer pick for voice TTS because Qwen2.5 7B occasionally leaks raw tool-call syntax (`Calling list_entities(domain='weather')...`) into the reply on short prompts; Qwen3 never did this.

> **Context window on 8 GB:** model weights + KV cache must stay under ~7.5 GB.
> A 7–8B Q4_K_M model uses 4.7–5.2 GB weights; KV cache ≈ 0.2 GB per 1 K tokens.
> The plugin's first-turn header (system prompt + HA tool schemas + home location + user facts) is ~4.2 K tokens in default mode and ~3.3 K in voice mode — leaves the rest of the window for conversation + tool results.
>
> **Minimum recommended context: 16384.** Multi-step tool loops (discovery → state → confirm) routinely emit 6–10 K of intermediate tokens, and 8192 starves them — most of the false-negatives in the 8B sweep that *did* call the right tool then ran out of context to summarise the result.
> 8192 only works for chat-only setups with no entity discovery. The integration default is **16384** for this reason.
> Avoid values above ~24 000 on an 8 GB card with 7–8B models.

### 24 GB VRAM (e.g. RTX 3090, RTX 4090)

| Model | Size | Tool calling | Notes |
|-------|------|-------------|-------|
| **ministral-3:14b** | ~9 GB | ⭐⭐⭐⭐⭐ | Best local choice — Mistral architecture, decisive tool calls, natural responses |
| devstral-small-2:latest | ~15 GB | ⭐⭐⭐⭐⭐ | Excellent, coding-focused, slightly larger |
| qwen2.5:32b | ~20 GB | ⭐⭐⭐⭐ | Very capable, requires more VRAM |
| gemma4:27b | ~18 GB | ⭐⭐⭐ | Natural language quality is high, tool calling less reliable |

### Cloud models

| Model | Tool calling | Notes |
|-------|-------------|-------|
| GPT-4o | ⭐⭐⭐⭐⭐ | Best overall, reliable function calling |
| GPT-4o mini | ⭐⭐⭐⭐ | Good balance of cost and reliability |
| Claude Sonnet | ⭐⭐⭐⭐⭐ | Excellent tool use, natural responses |
| Grok 2 | ⭐⭐⭐⭐ | Good tool calling, fast |

## Ollama Configuration

### Context Window — one setting controls everything

When connecting to Ollama, this plugin automatically uses Ollama's native `/api/chat` endpoint and passes `num_ctx` on every request. The **Context Window** setting in the integration UI is therefore the single number that controls both:

- the plugin's internal token budget (when to summarise, when to truncate), and
- the actual context size Ollama loads the model with.

**No Modelfile changes are required.** Just set Context Window in the integration and Ollama honours it.

Choosing the right value:
- Too low (e.g. 2048) — plugin summarises after a few turns; model forgets recent context fast.
- Too high (e.g. 32768 on 8 GB VRAM) — Ollama may OOM or fall back to CPU offload, causing slow responses.
- **For 8 GB VRAM + 7B model: 8192 is the recommended value.** It fits entirely in VRAM and provides enough context for typical home-assistant conversations.
- **The integration default is 16384** — change it to fit your hardware in Advanced settings.

### Temperature

Set temperature to **0.1–0.3** for home control. Higher values cause models to ask clarifying questions instead of calling tools and acting. Configure via the integration's Advanced settings. **0.2 is what the benchmark below was run with.**

### Reasoning models (qwen3, deepseek-r1)

Reasoning-capable Ollama models emit chain-of-thought into a separate `thinking` field and sometimes leave `content` empty. The plugin sends `think: false` on every Ollama `/api/chat` call so final answers land in `content` where they belong — no Modelfile tweaks required. A `SYSTEM """/no_think"""` directive in a custom Modelfile is unreliable and unnecessary.

## Reliability Benchmark

51-prompt matrix covering light control, climate, sensors/status, media, weather (local + named-place), web search & nearby-events, memory, multi-step regression, and multilingual input (English + German). Measured on Home Assistant Core 2026.4.x against plugin v0.5.46 via `/api/conversation/process`, **voice mode** (`device_id` set so `SYSTEM_PROMPT_VOICE` + `strip_urls=True` are active — the most common Assist usage path).

| Model | Pass rate | Median latency | Avg latency | Suspect-place hits | Best categories | Worst categories |
|-------|-----------|----------------|-------------|--------------------|-----------------|------------------|
| `qwen3:8b` (alias `homeassistant:latest`) | **86.3 %** (44/51) | **1.4 s** | 1.8 s | 0 | weather 5/5, web 5/5, light 9/10, media 4/4, memory 4/4 | sanity 2/3, multi 2/3 |
| `qwen2.5:7b` | **88.2 %** (45/51) | 0.7 s | 2.6 s | 0 | sensor 6/6, multi 3/3, sanity 3/3, weather 5/5, web 5/5 | climate 5/7 (couldn't resolve thermostat target), short 2/4 |
| `hermes3:8b` | 74.5 % (38/51) | 1.5 s | 2.1 s | **1** | sensor 6/6, multi 3/3, web 5/5 | weather 2/5 (asks for location), light 6/10, sanity 1/3 |

**Reference setup:** RTX 3060 Ti (8 GB), Ollama 0.20+, num_ctx 16384, temperature 0.2, top_p 0.4, num_predict 512. All three models Q4_K_M. Plugin v0.5.46.

Notes on the rubric:
- "Pass" = HTTP 200 + non-empty reply + matches one of the per-prompt positive regexes + matches none of the per-prompt negative regexes (e.g. *"don't have"*, *"cannot"*).
- The single Hermes Texas-class hit was the prompt `"events"` in voice-short mode, which returned a fabricated US sports recap.

### Failure analysis (qwen models)

Failures classified by root cause: **real model fault**, **environment** (entity unhealthy / unexposed), or **rubric** (model behaved correctly but the regex didn't agree).

**`qwen3:8b` — 7 flagged, ~2 real faults (≈96 % adjusted):**

| Prompt | Class | Detail |
|--------|-------|--------|
| `is the kitchen light on?` | **real** | Gave up: *"couldn't produce an answer for that"* |
| `what's the indoor temperature?` | env | Underlying sensor returned `unknown` |
| `kitchen humidity?` | rubric | Correctly said no humidity sensor; tripped *"don't have"* regex |
| `is the front door locked?` | env/rubric | No lock entity exposed; correct refusal |
| `turn off all lights and lock the doors` | env | Turned lights off; correctly said no lock domain |
| `temperature` (voice-short) | env | Same unhealthy temp sensor as `clim-5` |
| `danke` | rubric | Valid reply *"Du bist willkommen!"* — regex only allowed *bitte/gerne* |

**`qwen2.5:7b` — 6 flagged, ~4 real faults (≈92 % adjusted):**

| Prompt | Class | Detail |
|--------|-------|--------|
| `turn on the living room light` | **real** | Couldn't resolve `living room` to area `Wohnzimmer` (qwen3 handled the same install fine) |
| `lichter im wohnzimmer aus` | **real** | Fabricated *"network error"* instead of issuing the off command |
| `set the bedroom to 21 degrees` | **real** | Couldn't find the Tado thermostat (qwen3 found it) |
| `kitchen humidity?` | **real** | Leaked tool-call syntax as text: `CallCheck(list_entities(domain="sensor", area="kitchen"))` |
| `weather` (voice-short) | **real** | Same leak: *"Calling list_entities(domain='weather')..."* spoken instead of invoked |
| `events` (voice-short) | env | Misread as a sensor query |

**Implication for voice TTS:** Qwen2.5 7B's two tool-syntax leaks would be read aloud verbatim through the text-to-speech pipeline. That's an audible regression Qwen3 8B does not exhibit. If voice is the primary surface, **prefer Qwen3 8B even though the raw pass count is one lower.**

**Recommended configuration**

| Setting | Value |
|---------|-------|
| Model | `qwen3:8b` (or `qwen2.5:7b` as fallback) |
| Temperature | 0.2 |
| top_p | 0.4 |
| Context Window | **16384** (minimum — multi-step tool loops need the headroom; do not drop to 8192 unless you've disabled entity discovery) |
| Max Tokens (`num_predict`) | 512 |
| Voice Mode | on (compact system prompt — saves ~930 input tokens per turn) |
| Web Search | Tavily or Brave (DuckDuckGo works but tie-breaks ambiguous toponyms toward US locales — v0.5.46 mitigates by injecting `city, country`) |

No custom Modelfile parameters are required — the plugin manages `think`, `num_ctx`, `temperature`, `top_p`, and `num_predict` per request on Ollama's native API.

### Keep-alive (optional)

Ollama unloads models from VRAM after 5 minutes of inactivity by default. If you share the GPU with other workloads this is fine. For always-on dedicated use:

```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

## Requirements

- Home Assistant **2025.7.0** or newer
- [HACS](https://hacs.xyz/docs/use/) installed (used to deliver the integration)
- A reachable LLM endpoint, e.g.
  - [Ollama](https://ollama.com) — `http://<host>:11434/v1`
  - OpenAI — `https://api.openai.com/v1` (API key required)
  - LM Studio — `http://<host>:1234/v1`
  - llama.cpp server — `http://<host>:8080/v1`
  - xAI Grok — `https://api.x.ai/v1` (API key required)

## Installation

1. In Home Assistant, open **HACS → Integrations → ⋮ → Custom repositories**
2. Add `https://github.com/bigbabol1/AI-plugin` with category **Integration**
3. Install **AI Plugin** from HACS
4. **Restart Home Assistant**
5. Open **Settings → Devices & Services → Add Integration**, search for **AI Plugin**

## Configuration

The setup wizard walks through:
1. **Provider URL** — your LLM endpoint (auto-detects available models if reachable). Default `http://localhost:11434/v1` (local Ollama).
2. **Model + API key** — pick a model from the dropdown (or type one) and paste an API key if your provider needs one (OpenAI, xAI, etc.). Leave blank for local backends.
3. **Web search** — optional, choose backend and supply its API key (see [Web Search Backends](#web-search-backends)).
4. **Advanced** — system prompt, context window, voice mode, MCP servers, timeout.

After install, wire it into voice/chat:
- **Settings → Voice assistants → Add assistant** (or edit existing)
- Set **Conversation agent** = *AI Plugin*
- Optionally set the same agent for the STT/TTS pipeline

All settings are editable post-install via **Settings → Devices & Services → AI Plugin → Configure**.

## Web Search Backends

Default backend is **Brave Search**. Pick another in the Web Search step.

| Backend | Cost | API key | Notes |
|---------|------|---------|-------|
| Brave Search | Free tier / paid | [api.search.brave.com](https://api.search.brave.com/app/) | Default, reliable |
| Tavily | Free tier / paid | [tavily.com](https://tavily.com/) | Best quality, AI-optimized |
| SearXNG | Free (self-hosted) | none — needs your instance URL | Full control, no third-party traffic |
| DuckDuckGo | Free | none | Zero config, may be rate-limited |

## MCP Servers

Add any MCP-compatible tool server in the integration settings. Supports:
- **HTTP transport** — `{"url": "http://localhost:8123/mcp"}`
- **stdio transport** — `{"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-time"]}`

## Troubleshooting

- **"Cannot connect" on setup** — verify the Provider URL ends with `/v1` for OpenAI-compat backends, and that the host is reachable from the HA container/host.
- **Model dropdown is empty** — your endpoint isn't returning `/v1/models`. Type the model name manually.
- **Model keeps asking questions instead of acting** — lower temperature to 0.1–0.3, or switch to a higher tool-calling tier model.
- **Slow first response** — Ollama is loading the model into VRAM. Use `OLLAMA_KEEP_ALIVE=-1` to keep it warm.
- **OOM / CPU fallback in Ollama** — lower the **Context Window** in Advanced settings.
- **Empty replies from reasoning models (qwen3, deepseek-r1)** — the plugin already sends `think: false`; if you still see empty content, upgrade Ollama to 0.20 or newer (earlier versions don't honour the flag).
- **"Any lights on?" answers about a single light** — fixed in v0.5.38. Upgrade if you're on an older release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
