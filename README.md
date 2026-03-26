# AI Hub for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search, MCP tool extensibility, and smart context management.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **Web search** — DuckDuckGo (zero config), Brave Search, SearXNG (self-hosted), Tavily
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models
- **Voice mode** — compact system prompt when triggered via HA Assist voice pipeline
- **XML fallback** — tool calling for models without native function-calling support

## Installation

1. Go to HACS → Integrations → ⋮ → Custom repositories
2. Add `https://github.com/bigbabol1/AI-plugin` with category **Integration**
3. Install **AI Hub** from HACS
4. Restart Home Assistant
5. Go to **Settings → Devices & Services → Add Integration → AI Hub**

## Requirements

- Home Assistant 2025.7.0 or newer
- A running AI provider (e.g. [Ollama](https://ollama.com) at `localhost:11434`)

## Configuration

The setup wizard walks through:
1. **Provider URL** — your LLM endpoint (auto-detects available models)
2. **Model** — select from dropdown or enter manually
3. **Web search** — optional, choose backend and API keys
4. **Advanced** — system prompt, context window, voice mode, MCP servers, timeout

All settings are editable post-install via **Settings → Devices & Services → AI Hub → Configure**.

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
