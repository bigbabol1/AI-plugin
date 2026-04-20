# AI Plugin for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search, MCP tool extensibility, and smart context management.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **Web search** — DuckDuckGo (zero config), Brave Search, SearXNG (self-hosted), Tavily
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models
- **Voice mode** — compact system prompt when triggered via HA Assist voice pipeline
- **XML fallback** — tool calling for models without native function-calling support

## Recommended Models

The most critical factor for reliable entity control is **tool calling quality**. Models that hedge or ask unnecessary clarifying questions instead of executing commands are a poor fit, regardless of their general benchmark scores.

### Local models (Ollama)

| Model | Size | Tool calling | Notes |
|-------|------|-------------|-------|
| **ministral-3:14b** | ~9 GB | ⭐⭐⭐⭐⭐ | Best local choice — Mistral architecture, decisive tool calls, natural responses |
| devstral-small-2:latest | ~15 GB | ⭐⭐⭐⭐⭐ | Excellent, coding-focused, slightly larger |
| qwen2.5:32b | ~20 GB | ⭐⭐⭐⭐ | Very capable, requires more VRAM |
| gemma4:27b | ~18 GB | ⭐⭐⭐ | Natural language quality is high, tool calling less reliable |
| gemma4:e4b | ~10 GB | ⭐⭐ | Too small for reliable entity resolution — hedges instead of executing |

> **Tip:** Set a low temperature (0.1–0.3) for more decisive tool call execution. Higher temperatures cause models to ask clarifying questions instead of acting.

> **VRAM note:** With a 24 GB GPU, `ministral-3:14b` leaves ~13 GB for KV cache (≈ 40K+ token context), making it the best balance of capability and headroom for a home assistant workload.

### Cloud models

| Model | Tool calling | Notes |
|-------|-------------|-------|
| GPT-4o | ⭐⭐⭐⭐⭐ | Best overall, reliable function calling |
| GPT-4o mini | ⭐⭐⭐⭐ | Good balance of cost and reliability |
| Claude Sonnet | ⭐⭐⭐⭐⭐ | Excellent tool use, natural responses |
| Grok 2 | ⭐⭐⭐⭐ | Good tool calling, fast |

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
