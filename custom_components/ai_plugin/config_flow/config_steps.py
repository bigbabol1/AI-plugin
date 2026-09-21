"""First-time setup: provider, URL, model, web search, advanced.

  Step 1 ─ provider        Choose provider type (v1: OpenAI-compat only)
  Step 2a ─ provider_url   Enter provider URL → fetch models → store result
  Step 2b ─ model          Choose model (dropdown if fetch succeeded, else
                            free-text) + optional API key
  Step 3  ─ web_search     Web search toggle + backend choice
  Step 3b ─ searxng        SearXNG URL (only if SearXNG backend selected)
  Step 4  ─ advanced       System prompt, context window, voice mode, etc.

Creates the config entry:
    data:    {provider}   — immutable, it determines the provider class
    options: {base_url, model, api_key, web_search_*, advanced_*}
"""

from __future__ import annotations

import logging

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from ..const import (
    BACKEND_SEARXNG,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CONTEXT_WINDOW,
    CONF_MAX_RESULTS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARXNG_URL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_BASE_URL,
    DEFAULT_WEB_SEARCH_BACKEND,
    DOMAIN,
    ERROR_INVALID_URL,
    ERROR_CONTEXT_EXCEEDS_MODEL,
    ERROR_MODEL_NO_TOOLS,
    ERROR_MODEL_REQUIRED,
    PROVIDER_OPENAI_COMPAT,
)
from ..exceptions import CannotConnect
from ..providers.openai_compat import (
    async_fetch_models,
    async_show_model,
    model_capabilities,
    model_context_length,
)


from .options import AIPluginOptionsFlow
from .schemas import (
    _advanced_schema,
    _get_ha_url,
    _is_valid_url,
    _validate_advanced_input,
    _web_search_schema,
)

_LOGGER = logging.getLogger(__name__)


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
                api_key = str(user_input.get(CONF_API_KEY, "")).strip()
                # Ollama introspection (fail-open): block only when the
                # server definitively says the model can't call tools.
                show = await async_show_model(
                    str(self._options.get(CONF_BASE_URL, "")), model,
                    api_key=api_key or None,
                )
                caps = model_capabilities(show)
                if caps is not None and "tools" not in caps:
                    errors[CONF_MODEL] = ERROR_MODEL_NO_TOOLS
                else:
                    self._options[CONF_MODEL] = model
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
            errors = _validate_advanced_input(user_input)
            if not errors and user_input.get(CONF_CONTEXT_WINDOW):
                # Fail-open Ollama check: reject a context window larger
                # than the model can actually load.
                _show = await async_show_model(
                    str(self._options.get(CONF_BASE_URL, "")),
                    str(self._options.get(CONF_MODEL, "")),
                    api_key=self._options.get(CONF_API_KEY),
                )
                _limit = model_context_length(_show)
                if _limit and int(user_input[CONF_CONTEXT_WINDOW]) > _limit:
                    errors[CONF_CONTEXT_WINDOW] = ERROR_CONTEXT_EXCEEDS_MODEL
            if errors:
                schema = vol.Schema(
                    {
                        **_advanced_schema(user_input, self.hass).schema,
                        vol.Optional(
                            "use_ha_mcp",
                            default=bool(user_input.get("use_ha_mcp", False)),
                        ): selector.BooleanSelector(),
                        vol.Optional(
                            "ha_mcp_token",
                            default=str(user_input.get("ha_mcp_token", "")),
                        ): selector.TextSelector(
                            selector.TextSelectorConfig(
                                type=selector.TextSelectorType.PASSWORD
                            )
                        ),
                    }
                )
                return self.async_show_form(
                    step_id="advanced",
                    data_schema=schema,
                    errors=errors,
                    last_step=True,
                )
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
                    "transport": "sse",
                    "url": f"{ha_url}/mcp_server/sse",
                }
                if ha_mcp_token:
                    mcp_entry["token"] = ha_mcp_token
                self._options.setdefault(CONF_MCP_SERVERS, []).append(mcp_entry)
            self._options.update(user_input)
            return self._create_entry()

        # Merge the standard advanced schema with the HA MCP quick-connect fields.
        schema = vol.Schema(
            {
                **_advanced_schema(self._options, self.hass).schema,
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
