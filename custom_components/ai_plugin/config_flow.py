"""Config flow and OptionsFlowHandler for AI Plugin.

CONFIG FLOW (first-time setup):

  Step 1 ─ provider        Choose provider type (v1: OpenAI-compat only)
  Step 2a ─ provider_url   Enter provider URL → fetch models → store result
  Step 2b ─ model          Choose model (dropdown if fetch succeeded, else free-text)
                            + optional API key
  Step 3  ─ web_search     Web search toggle + backend choice
  Step 3b ─ searxng        SearXNG URL (only if SearXNG backend selected)
  Step 4  ─ advanced       System prompt, context window, voice mode, etc.
  ─────────────────────────────────────────────────────────────────────────
  Creates config_entry:
    data:    {provider}            (immutable — determines provider class)
    options: {base_url, model, api_key, web_search_*, advanced_*}

OPTIONS FLOW (post-install Configure):

  Menu ─ init             Choose: Provider Settings / Web Search / Advanced / MCP
  Step  ─ provider_settings  base_url + model + api_key
  Step  ─ web_search         same as config flow step 3 + 3b if needed
  Step  ─ searxng            same as config flow step 3b
  Step  ─ advanced           same as config flow step 4
  Step  ─ mcp_servers        stub (Week 3)
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    BACKEND_BRAVE,
    BACKEND_DUCKDUCKGO,
    BACKEND_SEARXNG,
    BACKEND_TAVILY,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_BRAVE_API_KEY,
    CONF_CONTEXT_WINDOW,
    CONF_MAX_RESULTS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARXNG_URL,
    CONF_SUMMARIZATION_ENABLED,
    CONF_SYSTEM_PROMPT,
    CONF_TAVILY_API_KEY,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    CONF_XML_FALLBACK,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SUMMARIZATION_ENABLED,
    DEFAULT_VOICE_MODE,
    DEFAULT_WEB_SEARCH_BACKEND,
    DEFAULT_XML_FALLBACK,
    DOMAIN,
    SYSTEM_PROMPT_DEFAULT,
    ERROR_CANNOT_CONNECT,
    ERROR_INVALID_URL,
    ERROR_MODEL_REQUIRED,
    ERROR_SEARXNG_UNREACHABLE,
    PROVIDER_OPENAI_COMPAT,
)
from .exceptions import CannotConnect
from .providers.openai_compat import async_fetch_models

_LOGGER = logging.getLogger(__name__)


def _get_ha_url(hass: Any) -> str:
    """Return the HA base URL for self-connections (MCP client inside HA).

    Always uses localhost — the integration runs inside HA so localhost
    is more reliable than the external/LAN IP which can fail on hairpin NAT.
    """
    try:
        from homeassistant.helpers.network import get_url  # noqa: PLC0415
        url = get_url(hass, allow_internal=True, allow_ip=True).rstrip("/")
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"http://localhost:{port}"
    except Exception:  # noqa: BLE001
        return "http://localhost:8123"


def _is_valid_url(url: str) -> bool:
    """Return True if url has a valid http/https scheme and netloc."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:  # noqa: BLE001
        return False


def _web_search_schema(current: dict[str, Any]) -> vol.Schema:
    """Build the web search vol.Schema with current values as defaults."""
    return vol.Schema(
        {
            vol.Required(
                CONF_WEB_SEARCH_ENABLED,
                default=current.get(CONF_WEB_SEARCH_ENABLED, False),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_WEB_SEARCH_BACKEND,
                default=current.get(CONF_WEB_SEARCH_BACKEND, DEFAULT_WEB_SEARCH_BACKEND),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        {
                            "value": BACKEND_BRAVE,
                            "label": "Brave Search (recommended, API key required)",
                        },
                        {
                            "value": BACKEND_TAVILY,
                            "label": "Tavily (API key required)",
                        },
                        {
                            "value": BACKEND_SEARXNG,
                            "label": "SearXNG (self-hosted)",
                        },
                        {
                            "value": BACKEND_DUCKDUCKGO,
                            "label": "DuckDuckGo (free, may be unreliable)",
                        },
                    ],
                )
            ),
            vol.Optional(
                CONF_BRAVE_API_KEY,
                default=current.get(CONF_BRAVE_API_KEY, ""),
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            ),
            vol.Optional(
                CONF_TAVILY_API_KEY,
                default=current.get(CONF_TAVILY_API_KEY, ""),
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            ),
            vol.Optional(
                CONF_MAX_RESULTS,
                default=current.get(CONF_MAX_RESULTS, DEFAULT_MAX_RESULTS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=20,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }
    )


def _advanced_schema(current: dict[str, Any]) -> vol.Schema:
    """Build the advanced settings vol.Schema with current values as defaults."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=current.get(CONF_SYSTEM_PROMPT, SYSTEM_PROMPT_DEFAULT),
            ): selector.TemplateSelector(),
            vol.Optional(
                CONF_CONTEXT_WINDOW,
                default=current.get(CONF_CONTEXT_WINDOW, DEFAULT_CONTEXT_WINDOW),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=512,
                    max=131072,
                    step=512,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_SUMMARIZATION_ENABLED,
                default=current.get(CONF_SUMMARIZATION_ENABLED, DEFAULT_SUMMARIZATION_ENABLED),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_VOICE_MODE,
                default=current.get(CONF_VOICE_MODE, DEFAULT_VOICE_MODE),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_XML_FALLBACK,
                default=current.get(CONF_XML_FALLBACK, DEFAULT_XML_FALLBACK),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_MAX_TOOL_ITERATIONS,
                default=current.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=10,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_RESPONSE_TIMEOUT,
                default=current.get(CONF_RESPONSE_TIMEOUT, DEFAULT_RESPONSE_TIMEOUT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5,
                    max=120,
                    step=5,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": current.get(CONF_TEMPERATURE)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": current.get(CONF_TOP_P)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": current.get(CONF_MAX_TOKENS)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=32768,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }
    )


class AIPluginConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the AI Plugin config flow (first-time setup)."""

    VERSION = 1

    def __init__(self) -> None:
        # data: stored in config_entry.data (immutable — just the provider type)
        self._data: dict[str, Any] = {}
        # options: stored in config_entry.options (editable via OptionsFlow)
        self._options: dict[str, Any] = {}
        # model list fetched from /v1/models during step 2a; None if unreachable
        self._models: list[str] | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        return await self.async_step_provider()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Choose provider
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 1: Choose the AI provider type."""
        if user_input is not None:
            self._data[CONF_PROVIDER] = user_input[CONF_PROVIDER]
            return await self.async_step_provider_url()

        return self.async_show_form(
            step_id="provider",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER, default=PROVIDER_OPENAI_COMPAT
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {
                                    "value": PROVIDER_OPENAI_COMPAT,
                                    "label": "OpenAI-compatible (Ollama, llama.cpp, OpenAI, LM Studio)",
                                }
                            ],
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2a: Provider URL
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_provider_url(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 2a: Enter the provider base URL and fetch models."""
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].strip().rstrip("/")
            if not _is_valid_url(base_url):
                errors[CONF_BASE_URL] = ERROR_INVALID_URL
            else:
                self._options[CONF_BASE_URL] = base_url
                # Attempt model fetch — failure is not an error, just falls
                # back to free-text in step 2b
                try:
                    self._models = await async_fetch_models(base_url)
                    _LOGGER.debug("Fetched %d models from %s", len(self._models), base_url)
                except CannotConnect:
                    _LOGGER.debug("Could not reach %s — model free-text fallback", base_url)
                    self._models = None
                return await self.async_step_model()

        return self.async_show_form(
            step_id="provider_url",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BASE_URL,
                        default=self._options.get(CONF_BASE_URL, DEFAULT_BASE_URL),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                }
            ),
            errors=errors,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2b: Model selection + API key
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 2b: Choose model (dropdown or free-text) + optional API key."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model = str(user_input.get(CONF_MODEL, "")).strip()
            if not model:
                errors[CONF_MODEL] = ERROR_MODEL_REQUIRED
            else:
                self._options[CONF_MODEL] = model
                api_key = str(user_input.get(CONF_API_KEY, "")).strip()
                if api_key:
                    self._options[CONF_API_KEY] = api_key
                elif CONF_API_KEY in self._options:
                    del self._options[CONF_API_KEY]
                return await self.async_step_web_search()

        # Model field: dropdown if models fetched, free-text if not
        if self._models:
            model_field: Any = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=self._models,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        else:
            model_field = selector.TextSelector()

        # API key: always optional; shown even for local to support
        # Ollama instances with auth proxy in front
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_MODEL,
                    default=self._options.get(CONF_MODEL, ""),
                ): model_field,
                vol.Optional(
                    CONF_API_KEY,
                    default=self._options.get(CONF_API_KEY, ""),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
            }
        )

        description_placeholders: dict[str, str] = {}
        if self._models is None:
            description_placeholders["fetch_failed"] = (
                self._options.get(CONF_BASE_URL, "the provider")
            )

        return self.async_show_form(
            step_id="model",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Web search
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_web_search(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 3: Web search settings."""
        if user_input is not None:
            self._options.update(user_input)
            backend = user_input.get(CONF_WEB_SEARCH_BACKEND, DEFAULT_WEB_SEARCH_BACKEND)
            enabled = user_input.get(CONF_WEB_SEARCH_ENABLED, False)
            if enabled and backend == BACKEND_SEARXNG:
                return await self.async_step_searxng()
            return await self.async_step_advanced()

        return self.async_show_form(
            step_id="web_search",
            data_schema=_web_search_schema(self._options),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3b: SearXNG URL (only when SearXNG backend selected)
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_searxng(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 3b: SearXNG instance URL."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_SEARXNG_URL].strip()
            if not _is_valid_url(url):
                errors[CONF_SEARXNG_URL] = ERROR_INVALID_URL
            else:
                # Reachability check — failure blocks this step (SearXNG won't work without it)
                try:
                    await async_fetch_models(url)  # cheap HEAD-equivalent via CannotConnect
                except CannotConnect:
                    # Intentionally allow proceeding even if SearXNG is temporarily down;
                    # warn but don't block. Validate format only.
                    _LOGGER.warning("SearXNG at %s may be unreachable at config time", url)
                self._options[CONF_SEARXNG_URL] = url
                return await self.async_step_advanced()

        return self.async_show_form(
            step_id="searxng",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SEARXNG_URL,
                        default=self._options.get(CONF_SEARXNG_URL, ""),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                }
            ),
            errors=errors,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Advanced settings
    # ──────────────────────────────────────────────────────────────────────────

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 4: Advanced settings + optional HA MCP quick-connect."""
        if user_input is not None:
            # Coerce number selector outputs to their correct types
            for key in (CONF_CONTEXT_WINDOW, CONF_MAX_TOOL_ITERATIONS, CONF_RESPONSE_TIMEOUT, CONF_MAX_RESULTS, CONF_MAX_TOKENS):
                if key in user_input:
                    user_input[key] = int(user_input[key])
            for key in (CONF_TEMPERATURE, CONF_TOP_P):
                if key in user_input:
                    user_input[key] = float(user_input[key])
            # HA MCP quick-connect: build the server entry and append to mcp_servers
            use_ha_mcp = bool(user_input.pop("use_ha_mcp", False))
            ha_mcp_token = str(user_input.pop("ha_mcp_token", "")).strip()
            if use_ha_mcp:
                ha_url = _get_ha_url(self.hass)
                mcp_entry: dict[str, Any] = {
                    "transport": "http",
                    "url": f"{ha_url}/api/mcp",
                }
                if ha_mcp_token:
                    mcp_entry["token"] = ha_mcp_token
                self._options.setdefault(CONF_MCP_SERVERS, []).append(mcp_entry)
            self._options.update(user_input)
            return self._create_entry()

        # Merge the standard advanced schema with the HA MCP quick-connect fields.
        schema = vol.Schema(
            {
                **_advanced_schema(self._options).schema,
                vol.Optional("use_ha_mcp", default=False): selector.BooleanSelector(),
                vol.Optional("ha_mcp_token", default=""): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
            }
        )

        return self.async_show_form(
            step_id="advanced",
            data_schema=schema,
            last_step=True,
        )

    def _create_entry(self) -> config_entries.FlowResult:
        model = self._options.get(CONF_MODEL, "unknown")
        return self.async_create_entry(
            title=f"AI Plugin ({model})",
            data={CONF_PROVIDER: self._data.get(CONF_PROVIDER, PROVIDER_OPENAI_COMPAT)},
            options=self._options,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> AIPluginOptionsFlow:
        return AIPluginOptionsFlow(config_entry)


# ══════════════════════════════════════════════════════════════════════════════
# OptionsFlowHandler
# ══════════════════════════════════════════════════════════════════════════════


class AIPluginOptionsFlow(config_entries.OptionsFlow):
    """Options flow: edit all non-credential settings post-install.

    Menu:
      ┌─ Provider settings  (URL + model + API key)
      ├─ Web search         (toggle, backend, keys)
      ├─ Advanced           (system prompt, context, voice, etc.)
      └─ MCP servers        (stub — Week 3)
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry
        self._options = dict(config_entry.options)
        self._models: list[str] | None = None
        # Working copy of MCP server list — committed on "Save and close"
        self._pending_mcp: list[dict] = list(
            config_entry.options.get(CONF_MCP_SERVERS, [])
        )
        # Index of server currently being edited (None when not editing)
        self._editing_idx: int | None = None
        # Key of the preset currently being configured (None when not in preset flow)
        self._preset_key: str | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Show the top-level menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["provider_settings", "web_search", "advanced", "mcp_servers"],
        )

    # ── Provider settings ────────────────────────────────────────────────────

    async def async_step_provider_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit provider URL → fetch models → model selection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].strip().rstrip("/")
            if not _is_valid_url(base_url):
                errors[CONF_BASE_URL] = ERROR_INVALID_URL
            else:
                self._options[CONF_BASE_URL] = base_url
                api_key = str(user_input.get(CONF_API_KEY, "")).strip()
                if api_key:
                    self._options[CONF_API_KEY] = api_key
                elif CONF_API_KEY in self._options:
                    del self._options[CONF_API_KEY]
                try:
                    self._models = await async_fetch_models(
                        base_url, self._options.get(CONF_API_KEY)
                    )
                except CannotConnect:
                    self._models = None
                return await self.async_step_provider_model()

        return self.async_show_form(
            step_id="provider_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BASE_URL,
                        default=self._options.get(CONF_BASE_URL, DEFAULT_BASE_URL),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                    vol.Optional(
                        CONF_API_KEY,
                        default=self._options.get(CONF_API_KEY, ""),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_provider_model(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit model selection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model = str(user_input.get(CONF_MODEL, "")).strip()
            if not model:
                errors[CONF_MODEL] = ERROR_MODEL_REQUIRED
            else:
                self._options[CONF_MODEL] = model
                return self.async_create_entry(title="", data=self._options)

        if self._models:
            model_field: Any = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=self._models,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        else:
            model_field = selector.TextSelector()

        return self.async_show_form(
            step_id="provider_model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=self._options.get(CONF_MODEL, ""),
                    ): model_field,
                }
            ),
            errors=errors,
        )

    # ── Web search ───────────────────────────────────────────────────────────

    async def async_step_web_search(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit web search settings."""
        if user_input is not None:
            self._options.update(user_input)
            backend = user_input.get(CONF_WEB_SEARCH_BACKEND, DEFAULT_WEB_SEARCH_BACKEND)
            enabled = user_input.get(CONF_WEB_SEARCH_ENABLED, False)
            if enabled and backend == BACKEND_SEARXNG:
                return await self.async_step_searxng()
            return self.async_create_entry(title="", data=self._options)

        return self.async_show_form(
            step_id="web_search",
            data_schema=_web_search_schema(self._options),
        )

    async def async_step_searxng(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit SearXNG URL."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_SEARXNG_URL].strip()
            if not _is_valid_url(url):
                errors[CONF_SEARXNG_URL] = ERROR_INVALID_URL
            else:
                self._options[CONF_SEARXNG_URL] = url
                return self.async_create_entry(title="", data=self._options)

        return self.async_show_form(
            step_id="searxng",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SEARXNG_URL,
                        default=self._options.get(CONF_SEARXNG_URL, ""),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                }
            ),
            errors=errors,
        )

    # ── Advanced ─────────────────────────────────────────────────────────────

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit advanced settings."""
        if user_input is not None:
            for key in (CONF_CONTEXT_WINDOW, CONF_MAX_TOOL_ITERATIONS, CONF_RESPONSE_TIMEOUT, CONF_MAX_RESULTS, CONF_MAX_TOKENS):
                if key in user_input:
                    user_input[key] = int(user_input[key])
            for key in (CONF_TEMPERATURE, CONF_TOP_P):
                if key in user_input:
                    user_input[key] = float(user_input[key])
            self._options.update(user_input)
            return self.async_create_entry(title="", data=self._options)

        return self.async_show_form(
            step_id="advanced",
            data_schema=_advanced_schema(self._options),
        )

    # ── MCP servers ──────────────────────────────────────────────────────────

    # Preset stdio servers — "args_template" entries containing "{value}" will
    # have the placeholder replaced by a user-supplied config value.
    _STDIO_PRESETS: dict[str, dict[str, Any]] = {
        "time": {
            "label": "Clock & timezones — ask 'what time is it in Tokyo?' or 'how many hours until midnight?'",
            "command": "uvx",
            "args": ["mcp-server-time"],
        },
        "fetch": {
            "label": "Web reader — ask the AI to read any webpage, news article, or online document",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    }

    def _mcp_server_label(self, s: dict) -> str:
        """Return a short human-readable label for an MCP server dict."""
        if s.get("transport") == "stdio":
            parts = [s.get("command", "")]
            parts.extend(s.get("args", []))
            return "stdio: " + " ".join(p for p in parts if p)
        url = s.get("url", "")
        token_hint = " (token set)" if s.get("token") else ""
        return f"http: {url}{token_hint}"

    def _mcp_servers_description(self) -> str:
        """Build a numbered list of configured MCP servers for the form description."""
        if not self._pending_mcp:
            return (
                "No servers configured yet.\n"
                "Tip: Add Home Assistant's built-in MCP server to let Buddy read "
                "your sensors and control devices."
            )
        lines = [f"{i}. {self._mcp_server_label(s)}" for i, s in enumerate(self._pending_mcp, 1)]
        return "\n".join(lines)

    async def async_step_mcp_servers(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """MCP server management — action selector with live server list."""
        if user_input is not None:
            action = user_input["mcp_action"]
            if action == "add_ha":
                return await self.async_step_mcp_add_ha()
            if action == "add_preset":
                return await self.async_step_mcp_add_preset()
            if action == "add_http":
                return await self.async_step_mcp_add_http()
            if action == "add_stdio":
                return await self.async_step_mcp_add_stdio()
            if action == "edit":
                return await self.async_step_mcp_edit_pick()
            if action == "remove":
                return await self.async_step_mcp_remove()
            # "save" — commit and close
            self._options[CONF_MCP_SERVERS] = self._pending_mcp
            return self.async_create_entry(title="", data=self._options)

        action_options: list[dict[str, str]] = [
            {"value": "add_ha", "label": "Add Home Assistant built-in MCP server"},
            {"value": "add_preset", "label": "Add a popular MCP server"},
            {"value": "add_http", "label": "Add other HTTP server"},
            {"value": "add_stdio", "label": "Add custom stdio server"},
        ]
        if self._pending_mcp:
            action_options.append({"value": "edit", "label": "Edit a server"})
            action_options.append({"value": "remove", "label": "Remove a server"})
        action_options.append({"value": "save", "label": "Save and close"})

        return self.async_show_form(
            step_id="mcp_servers",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_action"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=action_options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
            description_placeholders={"servers": self._mcp_servers_description()},
        )

    def _mcp_server_picker_schema(self) -> vol.Schema:
        """Schema for picking a server by index from the pending list."""
        options = [
            {"value": str(i), "label": f"{i + 1}. {self._mcp_server_label(s)}"}
            for i, s in enumerate(self._pending_mcp)
        ]
        return vol.Schema(
            {
                vol.Required("mcp_index"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )

    # ── Add ───────────────────────────────────────────────────────────────────

    async def async_step_mcp_add_ha(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Quick-add HA's built-in MCP server — only asks for the token."""
        if user_input is not None:
            token = user_input.get("ha_mcp_token", "").strip()
            ha_url = _get_ha_url(self.hass)
            entry: dict[str, Any] = {
                "transport": "http",
                "url": f"{ha_url}/api/mcp",
            }
            if token:
                entry["token"] = token
            self._pending_mcp.append(entry)
            return await self.async_step_mcp_servers()

        ha_url = _get_ha_url(self.hass)
        return self.async_show_form(
            step_id="mcp_add_ha",
            data_schema=vol.Schema(
                {
                    vol.Optional("ha_mcp_token", default=""): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            description_placeholders={"ha_mcp_url": f"{ha_url}/api/mcp"},
        )

    async def async_step_mcp_add_preset(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Pick a popular preset MCP server to add."""
        if user_input is not None:
            key = user_input["preset_key"]
            preset = self._STDIO_PRESETS[key]
            if "args_template" in preset:
                # Needs extra config — store key and show config form
                self._preset_key = key
                return await self.async_step_mcp_preset_config()
            # No extra config — add immediately
            self._pending_mcp.append({
                "transport": "stdio",
                "command": preset["command"],
                "args": preset["args"],
            })
            return await self.async_step_mcp_servers()

        preset_options = [
            {"value": k, "label": v["label"]}
            for k, v in self._STDIO_PRESETS.items()
        ]
        return self.async_show_form(
            step_id="mcp_add_preset",
            data_schema=vol.Schema(
                {
                    vol.Required("preset_key"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=preset_options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_mcp_preset_config(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Collect a single path/value for presets that need it (sqlite, filesystem)."""
        preset = self._STDIO_PRESETS[self._preset_key]  # type: ignore[index]
        errors: dict[str, str] = {}
        if user_input is not None:
            value = user_input.get("preset_value", "").strip()
            if not value:
                errors["preset_value"] = "required"
            else:
                args = [
                    a.replace("{value}", value) for a in preset["args_template"]
                ]
                self._pending_mcp.append({
                    "transport": "stdio",
                    "command": preset["command"],
                    "args": args,
                })
                self._preset_key = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_preset_config",
            data_schema=vol.Schema(
                {
                    vol.Required("preset_value"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            placeholder=preset.get("config_placeholder", "")
                        )
                    ),
                }
            ),
            description_placeholders={
                "preset_label": preset["label"],
                "config_label": preset.get("config_label", "Value"),
            },
            errors=errors,
        )

    async def async_step_mcp_add_http(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Add an HTTP MCP server (URL + optional bearer token)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input["mcp_url"].strip()
            if not _is_valid_url(url):
                errors["mcp_url"] = ERROR_INVALID_URL
            else:
                entry: dict[str, Any] = {"transport": "http", "url": url}
                token = user_input.get("mcp_token", "").strip()
                if token:
                    entry["token"] = token
                self._pending_mcp.append(entry)
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_add_http",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_url"): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                    vol.Optional("mcp_token", default=""): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_add_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Add a stdio MCP server (command + space-separated args)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            command = user_input.get("mcp_command", "").strip()
            if not command:
                errors["mcp_command"] = "command_required"
            else:
                raw_args = user_input.get("mcp_args", "").strip()
                self._pending_mcp.append(
                    {"transport": "stdio", "command": command, "args": raw_args.split() if raw_args else []}
                )
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_add_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_command"): selector.TextSelector(),
                    vol.Optional("mcp_args", default=""): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    # ── Edit ──────────────────────────────────────────────────────────────────

    async def async_step_mcp_edit_pick(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Pick which server to edit."""
        if not self._pending_mcp:
            return await self.async_step_mcp_servers()

        if user_input is not None:
            self._editing_idx = int(user_input["mcp_index"])
            s = self._pending_mcp[self._editing_idx]
            if s.get("transport") == "stdio":
                return await self.async_step_mcp_edit_stdio()
            return await self.async_step_mcp_edit_http()

        return self.async_show_form(
            step_id="mcp_edit_pick",
            data_schema=self._mcp_server_picker_schema(),
        )

    async def async_step_mcp_edit_http(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit an existing HTTP MCP server (pre-filled with current values)."""
        errors: dict[str, str] = {}
        existing = self._pending_mcp[self._editing_idx]  # type: ignore[index]

        if user_input is not None:
            url = user_input["mcp_url"].strip()
            if not _is_valid_url(url):
                errors["mcp_url"] = ERROR_INVALID_URL
            else:
                updated: dict[str, Any] = {"transport": "http", "url": url}
                token = user_input.get("mcp_token", "").strip()
                if token:
                    updated["token"] = token
                self._pending_mcp[self._editing_idx] = updated  # type: ignore[index]
                self._editing_idx = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_edit_http",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_url", default=existing.get("url", "")): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                    ),
                    vol.Optional("mcp_token", default=existing.get("token", "")): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_edit_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Edit an existing stdio MCP server (pre-filled with current values)."""
        errors: dict[str, str] = {}
        existing = self._pending_mcp[self._editing_idx]  # type: ignore[index]

        if user_input is not None:
            command = user_input.get("mcp_command", "").strip()
            if not command:
                errors["mcp_command"] = "command_required"
            else:
                raw_args = user_input.get("mcp_args", "").strip()
                self._pending_mcp[self._editing_idx] = {  # type: ignore[index]
                    "transport": "stdio",
                    "command": command,
                    "args": raw_args.split() if raw_args else [],
                }
                self._editing_idx = None
                return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_edit_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required("mcp_command", default=existing.get("command", "")): selector.TextSelector(),
                    vol.Optional(
                        "mcp_args",
                        default=" ".join(existing.get("args", [])),
                    ): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    # ── Remove ────────────────────────────────────────────────────────────────

    async def async_step_mcp_remove(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Remove one MCP server from the pending list."""
        if not self._pending_mcp:
            return await self.async_step_mcp_servers()

        if user_input is not None:
            idx = int(user_input["mcp_index"])
            if 0 <= idx < len(self._pending_mcp):
                self._pending_mcp.pop(idx)
            return await self.async_step_mcp_servers()

        return self.async_show_form(
            step_id="mcp_remove",
            data_schema=self._mcp_server_picker_schema(),
        )
