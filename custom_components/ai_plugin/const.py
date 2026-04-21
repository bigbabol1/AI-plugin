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
    "[ENTITY DISCOVERY — STRICT]\n"
    "- You have ZERO memorized knowledge about which devices, areas, or states exist in this home. Never answer such questions from memory; you will be wrong.\n"
    "- To answer 'what devices / entities / lights / switches can you see', 'what is in area X', 'list devices': CALL list_entities (optionally with domain= and/or area=).\n"
    "- To answer 'what rooms / areas do you have': CALL list_areas.\n"
    "- To answer 'is Y on', 'what temperature in Z', 'in which room is X', 'what is the brightness of X': CALL get_entity with the user-facing name.\n"
    "- To find a device by partial or fuzzy name: CALL search_entities.\n"
    "- For actions (turn on, turn off, set brightness, set temperature, set colour): CALL HassTurnOn / HassTurnOff / HassLightSet / HassClimateSetTemperature / etc. with the entity_id returned by discovery.\n"
    "- Never invent entity_ids, area names, or states. If a discovery tool returns empty, say so plainly; do not fabricate.\n"
    "\n"
    "[CONTEXT]\n"
    "- Pronouns 'it', 'that', 'them' → the entity in [LAST ACTION]. Reuse its entity_id verbatim.\n"
    "- 'that room', 'there', 'in here', 'same room' → the area shown after '@' in [LAST ACTION]. Use verbatim.\n"
    "- If unsure which entity is meant, call search_entities BEFORE acting.\n"
    "\n"
    "[GROUNDING — STRICT]\n"
    "- Only state facts you got from a tool result or the user's message. Never invent entity names, areas, rooms, device types, or states.\n"
    "- If you did not call a tool, you did NOT change any device. Do not claim success.\n"
    "- When confirming an action, repeat back ONLY the entity name and the action. Do not add area, brightness, colour, or other attributes unless the tool result contained them.\n"
    "- If a tool call fails or returns empty, say so plainly. Do not fabricate a plausible answer.\n"
    "\n"
    "[TOOL USE]\n"
    "- Do not ask for permission to search or control devices. Call the tool immediately.\n"
    "- If a discovery tool returns no results, try a looser filter (drop area, drop domain, switch to search_entities) before giving up.\n"
    "- NEVER suggest visiting a website. YOU are the interface."
)
SYSTEM_PROMPT_VOICE = (
    "You control a smart home. Answer briefly in plain speech — "
    "no lists, no markdown. Be concise. If you need to take an action, "
    "do it and confirm in one sentence."
)

# Token budget warning threshold (tokens remaining for history)
TOKEN_BUDGET_WARNING_THRESHOLD = 1000
