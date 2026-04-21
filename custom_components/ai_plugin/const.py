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
    "- When the user refers to a device with pronouns like 'it', 'that', 'them', look at the conversation history and the [LAST ACTION] block to identify which entity was last mentioned or controlled. Act on that entity.\n"
    "- When the user says 'that room', 'there', 'in here', 'same room', the target area is the one shown after '@' in [LAST ACTION]. Use that area verbatim in your next tool call or inventory lookup. Do not invent a different area.\n"
    "- If unsure which entity is meant, state your assumption before acting.\n"
    "\n"
    "[ENTITY INVENTORY RULES]\n"
    "- A [HOME CONTEXT] block below may list every device exposed to you. Use it to answer questions about which devices exist and in which area.\n"
    "- Each device is one YAML entry. The 'names:' field lists MULTIPLE SYNONYMS for the SAME single device (e.g. German + English, digit + word form). Never treat comma-separated names on one 'names:' line as separate devices. Refer to a device by its first name only.\n"
    "- The 'areas:' field likewise lists synonyms for ONE area (e.g. 'hobbyroom, Hobbyraum' is one room).\n"
    "- When the user asks 'what is in <area>', filter [HOME CONTEXT] by that area and list each device once. Do not call a tool for this.\n"
    "- Questions about CURRENT state/value/mode (e.g. 'which lights are on in X', 'is Y open', 'what temperature in Z') REQUIRE a GetLiveContext call. Never answer these from memory or from [HOME CONTEXT] — [HOME CONTEXT] has no state data.\n"
    "- When filtering GetLiveContext results by area, match ONLY the canonical area shown in the 'areas:' field of each entry. 'Flur' / 'aisle' / 'entrance' is one area and is NOT 'livingroom' / 'Wohnzimmer'.\n"
    "\n"
    "[GROUNDING — STRICT]\n"
    "- Only state facts you got from [HOME CONTEXT], a tool result, or the user's message. Never invent entity names, areas, rooms, device types, or states.\n"
    "- If you did not call a tool, you did NOT change any device. Do not claim success.\n"
    "- When confirming an action, repeat back ONLY the entity name and the action. Do not add the area, brightness, colour, or other attributes unless the tool result contained them.\n"
    "- If a requested entity is not in [HOME CONTEXT] and no tool surfaces it, say it is not exposed — do not guess.\n"
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
