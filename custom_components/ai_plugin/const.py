"""Constants for AI Plugin."""

DOMAIN = "ai_plugin"

# Provider types (v1: OpenAI-compatible only; Gemini/Anthropic post-v1)
PROVIDER_OPENAI_COMPAT = "openai_compat"

# Config keys stored in config_entry.data (immutable — connection identity)
CONF_PROVIDER = "provider"

# Config keys stored in config_entry.options (editable via OptionsFlow)
CONF_BASE_URL = "base_url"
CONF_MODEL = "model"
CONF_API_KEY = "api_key"

# Web search options
CONF_WEB_SEARCH_ENABLED = "web_search_enabled"
CONF_WEB_SEARCH_BACKEND = "web_search_backend"
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_TAVILY_API_KEY = "tavily_api_key"
CONF_SEARXNG_URL = "searxng_url"
CONF_MAX_RESULTS = "max_results"

# Advanced options
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_CONTEXT_WINDOW = "context_window"
CONF_SUMMARIZATION_ENABLED = "summarization_enabled"
CONF_VOICE_MODE = "voice_mode"
CONF_XML_FALLBACK = "xml_fallback"
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
CONF_RESPONSE_TIMEOUT = "response_timeout"

# Sampling parameters (None = omit from request, let the provider use its default)
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_MAX_TOKENS = "max_tokens"  # 0 = omit (unlimited)

# MCP server list (list of dicts: {transport, url|command, args, env})
CONF_MCP_SERVERS = "mcp_servers"

# Routing keys (CEO cherry-pick 2 — Week 3+ scope)
CONF_ROUTE_HOME_CONTROL = "route_home_control"
CONF_ROUTE_WEB_SEARCH = "route_web_search"
CONF_ROUTE_GENERAL = "route_general"

# Web search backends
BACKEND_DUCKDUCKGO = "duckduckgo"
BACKEND_BRAVE = "brave"
BACKEND_SEARXNG = "searxng"
BACKEND_TAVILY = "tavily"

# Defaults
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CONTEXT_WINDOW = 16384
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_TOOL_ITERATIONS = 10
DEFAULT_RESPONSE_TIMEOUT = 30
DEFAULT_WEB_SEARCH_BACKEND = BACKEND_BRAVE
DEFAULT_SUMMARIZATION_ENABLED = True
DEFAULT_VOICE_MODE = False
DEFAULT_XML_FALLBACK = False

# Error keys (must match strings.json config.error and options.error)
ERROR_CANNOT_CONNECT = "cannot_connect"
ERROR_INVALID_API_KEY = "invalid_api_key"
ERROR_MODEL_REQUIRED = "model_required"
ERROR_INVALID_URL = "invalid_url"
ERROR_SEARXNG_UNREACHABLE = "searxng_unreachable"

# System prompts
SYSTEM_PROMPT_DEFAULT = (
    "[PERSONALITY]\n"
    "You are a helpful assistant integrated into Home Assistant.\n"
    "\n"
    "[FORMATTING]\n"
    "- never use asterisks or markdown formatting.\n"
    "- do not use emojis.\n"
    "\n"
    "[CONTEXT]\n"
    "- When the user refers to a device with pronouns like 'it', 'that', 'them', 'there', or phrases like 'the light', 'the same one', look at the conversation history to identify which entity was last mentioned or controlled. Act on that entity.\n"
    "- If unsure which entity is meant, state your assumption before acting.\n"
    "\n"
    "[GROUNDING — STRICT]\n"
    "- Only state facts you got from a tool result or from the user's message. Never invent entity names, areas, rooms, device types, or states.\n"
    "- If you did not call a tool, you did NOT change any device. Do not claim success.\n"
    "- When confirming an action, repeat back ONLY the entity name and the action. Do not add the area, brightness, colour, or other attributes unless the tool result contained them.\n"
    "- If the user asks what is in a room/area and no tool returns that list, say you cannot enumerate it — do not guess.\n"
    "- If a tool call fails or returns empty, say so plainly. Do not fabricate a plausible answer.\n"
    "\n"
    "[TOOL USE]\n"
    "- Do not ask for permission to search or control devices. Call the tool immediately.\n"
    "- If a tool returns an error or no results, try one alternative search with a different filter before giving up.\n"
    "- NEVER suggest visiting a website. YOU are the interface."
)
SYSTEM_PROMPT_VOICE = (
    "You control a smart home. Answer briefly in plain speech — "
    "no lists, no markdown. Be concise. If you need to take an action, "
    "do it and confirm in one sentence."
)

# Token budget warning threshold (tokens remaining for history)
TOKEN_BUDGET_WARNING_THRESHOLD = 1000
