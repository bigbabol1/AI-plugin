# AI Plugin for Home Assistant

A provider-agnostic AI orchestration layer for Home Assistant. Use any LLM — Ollama, llama.cpp, OpenAI, LM Studio, and more — with web search and MCP extensibility.

## Features

- **Any LLM backend** — Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any OpenAI-compatible endpoint
- **Web search** — DuckDuckGo (zero config), Brave Search, SearXNG, Tavily
- **MCP extensibility** — connect external MCP servers (HTTP and stdio) for additional tools
- **Smart context management** — sliding window + automatic LLM summarization prevents memory loss with local models
- **Voice recognition** *(optional)* — knows who is speaking on a satellite, so each person gets their own memory; runs locally inside Home Assistant
- **Find my phone** — "Where is my phone?" rings it through the Companion app, even on silent
- **Alarms, reminders and things to do later** — "Wake me up at 7", "Turn off the lights in 30 minutes"; they survive a restart
- **Calendar entries by voice** — "Put the dentist on my calendar for Friday at 10"
- **Messages and moving music** — "Tell Anna dinner is ready", "Move the music to the bedroom"
- **Music makes room** — speakers in the room go down while a satellite listens
- **Where people are** — "Who is home?" and "Where is Anna?" answered from Home Assistant's people

## Installation

1. Go to HACS → Integrations → ⋮ → Custom repositories
2. Add `https://github.com/bigbabol1/AI-plugin` with category **Integration**
3. Install **AI Plugin** from HACS
4. Restart Home Assistant
5. Go to **Settings → Devices & Services → Add Integration → AI Plugin**

## Requirements

- Home Assistant 2025.7.0 or newer
- A running AI provider (e.g. [Ollama](https://ollama.com) at `localhost:11434`)
