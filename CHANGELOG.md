# Changelog

All notable changes to AI Plugin are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [0.5.21] - 2026-04-22

### Changed

- **Custom system prompt now appends instead of replacing** (`orchestrator.py`). Previously, setting a custom prompt in Advanced options fully replaced both `SYSTEM_PROMPT_DEFAULT` and `SYSTEM_PROMPT_VOICE`, silently dropping the plugin's entity-discovery, grounding, context, and voice-speech rules — users writing persona/location prompts lost tool-use discipline and small LLMs started fabricating entity_ids. The custom prompt is now appended to the base (voice-compact or default) prompt with a blank line separator. Empty custom prompt behaves as before (base only). Users who intentionally stripped defaults can clear their custom prompt and re-add only the overrides they want.

## [0.5.20] - 2026-04-22

### Fixed

- **Voice replies no longer read URLs aloud** (`const.py`, `orchestrator.py`, `tools/web_search.py`). Web-search results over a voice pipeline used to surface raw `https://…` addresses into the LLM's context; small LLMs then read them back and TTS narrated every slash. Two-pronged fix: the voice system prompt now forbids reading URLs, domain names, or file paths aloud; and the `web_search` tool result is stripped of URL lines (and inline `http(s)://` tokens) before it reaches the model whenever the orchestrator is in voice mode. Typed Assist is unchanged and still gets the source URLs.

## [0.5.19] - 2026-04-22

### Fixed

- **Voice Assist now reads long-term memory** (`conversation.py`, `config_flow.py`). Previously, voice pipelines (Wyoming satellites, wake-word, TTS-initiated flows) ran without an authenticated HA user, so `context.user_id` was `None` and the `remember`/`recall` tool routed to `.ai_plugin_memory_anonymous.json` — a different file from the `.ai_plugin_memory_<user_id>.json` written by typed Assist. Facts saved from the Assist panel never surfaced over voice. The Advanced settings now include a **Voice fallback user** dropdown; when set, voice requests reuse that user's memory file. Leave empty to keep the previous anonymous behaviour.

### Notes

- Existing `.ai_plugin_memory_anonymous.json` is not auto-migrated. If you pick a fallback user and want voice to see previously saved facts, rename that file to `.ai_plugin_memory_<user_id>.json` in your HA config directory (or copy its `facts` into the user's file).

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
