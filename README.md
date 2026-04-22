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

| Model | Ollama tag | Weights | Tool calling | Notes |
|-------|-----------|---------|-------------|-------|
| Qwen2.5 7B | `qwen2.5:7b` | ~4.7 GB | ⭐⭐⭐⭐ | Recommended. Strong tool calling, fits comfortably. |
| Mistral 7B | `mistral:7b` | ~4.1 GB | ⭐⭐⭐⭐ | Lighter, decisive tool calls, leaves more headroom for context. |
| Llama 3.1 8B | `llama3.1:8b` | ~4.8 GB | ⭐⭐⭐⭐ | Good balance; instruction-tuned variant preferred. |

> **Context window on 8 GB:** model weights + KV cache must stay under ~7.5 GB.
> A 7B Q4_K_M model uses ~4.7 GB weights; KV cache is roughly 0.2 GB per 1 K tokens.
> Setting **Context Window = 8192** is the practical sweet spot — leaves margin without wasting VRAM.
> The integration default is **16384** (sized for 12 GB+ cards) — lower it to 8192 in Advanced settings on 8 GB hardware.
> Avoid values above 12 000 on an 8 GB card with 7B models.

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

Set temperature to **0.1–0.3** for home control. Higher values cause models to ask clarifying questions instead of calling tools and acting. Configure via the integration's Advanced settings.

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
