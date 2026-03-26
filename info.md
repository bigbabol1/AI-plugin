# AI Hub for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search and MCP extensibility.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **Web search** — DuckDuckGo (zero config), Brave Search, SearXNG, Tavily
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models

## Installation

1. Go to HACS → Integrations → ⋮ → Custom repositories
2. Add `https://github.com/bigbabol1/AI-plugin` with category **Integration**
3. Install **AI Hub** from HACS
4. Restart Home Assistant
5. Go to **Settings → Devices & Services → Add Integration → AI Hub**

## Requirements

- Home Assistant 2025.7.0 or newer
- A running AI provider (e.g. [Ollama](https://ollama.com) at `localhost:11434`)
