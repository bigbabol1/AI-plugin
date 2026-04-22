# Changelog

All notable changes to AI Plugin are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [0.5.17] - 2026-04-22

### Fixed

- **Conversation history now resets per session** (`conversation.py`) — previously the entity overrode HA's ephemeral `conversation_id` with a stable per-user id, causing history to accumulate across every Assist session until summarisation fired. Now honours the id HA Assist provides (minting a fresh ulid when none is supplied). Long-term user facts continue to live in the `remember`/`recall` tool (`.ai_plugin_memory_<user_id>.json`), which is unchanged.

### Changed

- **README setup clarity** — added HACS prerequisite, provider URL examples per backend, Assist pipeline assignment step, and a Troubleshooting section. Documented the actual default Context Window (16384) alongside the 8 GB sweet-spot of 8192. Reordered web-search backends to put the default (Brave) first and added an API-key column with sign-up links.

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
