# AI Plugin for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search, MCP tool extensibility, and smart context management.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **On-demand entity discovery** — `list_areas`, `list_entities`, `get_entity`, `search_entities` tools hit HA registries in-process; no YAML dump into the prompt
- **Assist exposure aware** — discovery tools default to entities you've marked as exposed to conversation; diagnostic sensors stay hidden unless explicitly requested
- **Web search** — DuckDuckGo (zero config), Brave Search, SearXNG (self-hosted), Tavily
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models
- **Voice mode** — compact system prompt when triggered via HA Assist voice pipeline
- **XML fallback** — tool calling for models without native function-calling support

## Recommended Models

The most critical factor for reliable entity control is **tool calling quality**. Models that hedge or ask unnecessary clarifying questions instead of executing commands are a poor fit, regardless of their general benchmark scores.

### 8 GB VRAM (e.g. RTX 3060 Ti, RTX 3070)

| Model | Ollama tag | Weights | Tool calling | Notes |
|-------|-----------|---------|-------------|-------|
| **homeassistant:latest** | `homeassistant:latest` | ~4.7 GB | ⭐⭐⭐⭐ | Confirmed working. Qwen2.5-7B Q4_K_M tuned for HA. Recommended starting point. |
| Qwen2.5 7B | `qwen2.5:7b` | ~4.7 GB | ⭐⭐⭐⭐ | Same base as above, standard Ollama build. Slightly more headroom for KV cache. |
| Mistral 7B | `mistral:7b` | ~4.1 GB | ⭐⭐⭐⭐ | Lighter weight, decisive tool calls, leaves ~3.5 GB for KV cache. |

> **VRAM budget at 8 GB:** model weights + KV cache must stay under ~7.5 GB (leave 0.5 GB for overhead).
> A 7B Q4_K_M model uses ~4.7 GB weights + ~1 GB per 4 K context tokens.
> At `num_ctx=8192` total ≈ 6.7 GB — fits comfortably.
> Avoid `num_ctx` above 12 K on an 8 GB card with 7B models.

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

Getting Ollama configured correctly is as important as choosing the right model.

### num_ctx — match plugin and model

**The most common misconfiguration.** Ollama's default `num_ctx` for most models is **2048** (set in the model's Modelfile), regardless of the model's theoretical maximum. The plugin's **Context Window** setting must match the actual `num_ctx` Ollama uses, or the token budget math breaks:

- Plugin thinks it has 16 384 tokens → barely triggers summarisation.
- Model actually runs at 2 048 tokens → context overflows silently, model hallucinates or forgets.

**Fix — create a custom Modelfile:**

```
FROM homeassistant:latest
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
```

```bash
ollama create ha-assistant -f Modelfile
```

Then point the plugin at `ha-assistant` and set **Context Window = 8192** in the integration settings.

**Alternatively**, verify what `num_ctx` your current model is actually using:

```bash
curl http://localhost:11434/api/show -d '{"name":"homeassistant:latest"}' | python3 -m json.tool | grep num_ctx
```

Set the plugin's **Context Window** to match that value exactly.

### Recommended Ollama settings for 8 GB VRAM

```
PARAMETER num_ctx 8192        # matches plugin Context Window setting
PARAMETER temperature 0.1     # decisive tool calls; higher values cause hedging
PARAMETER num_predict 512     # cap response length; saves KV cache budget
```

> **Temperature:** Keep it at 0.1–0.3 for home control. Higher temperatures cause models to ask clarifying questions instead of calling tools and acting.

### Keep-alive (optional)

Ollama unloads models from VRAM after 5 minutes of inactivity by default. If you share the GPU with other workloads, this is fine. If you want instant response for HA queries:

```bash
# Keep model loaded indefinitely
OLLAMA_KEEP_ALIVE=-1 ollama serve

# Or per-request via API
curl http://localhost:11434/api/generate -d '{"model":"homeassistant:latest","keep_alive":-1}'
```

## Installation

1. Go to HACS → Integrations → ⋮ → Custom repositories
2. Add `https://github.com/bigbabol1/AI-plugin` with category **Integration**
3. Install **AI Plugin** from HACS
4. Restart Home Assistant
5. Go to **Settings → Devices & Services → Add Integration → AI Plugin**

## Requirements

- Home Assistant 2025.7.0 or newer
- A running AI provider (e.g. [Ollama](https://ollama.com) at `localhost:11434`)

## Configuration

The setup wizard walks through:
1. **Provider URL** — your LLM endpoint (auto-detects available models)
2. **Model** — select from dropdown or enter manually
3. **Web search** — optional, choose backend and API keys
4. **Advanced** — system prompt, context window, voice mode, MCP servers, timeout

All settings are editable post-install via **Settings → Devices & Services → AI Plugin → Configure**.

> **Context Window setting** must match the `num_ctx` your Ollama model actually uses. See [Ollama Configuration](#ollama-configuration) above.

## Web Search Backends

| Backend | Cost | Notes |
|---------|------|-------|
| DuckDuckGo | Free | No API key, may be rate-limited |
| Brave Search | Free tier / paid | Recommended, reliable |
| SearXNG | Free (self-hosted) | Full control |
| Tavily | Paid | Best quality, AI-optimized |

## MCP Servers

Add any MCP-compatible tool server in the integration settings. Supports:
- **HTTP transport** — `{"url": "http://localhost:8123/mcp"}`
- **stdio transport** — `{"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-time"]}`

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
