# Changelog

All notable changes to AI Plugin are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

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
