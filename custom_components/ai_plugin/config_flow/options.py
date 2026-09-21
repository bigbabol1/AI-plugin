"""Post-install Configure: every setting except the ones that are fixed.

  Menu ─ init                Provider settings / Web search / Advanced / MCP
  Step  ─ provider_settings  base_url + model + api_key
  Step  ─ web_search         as config step 3, plus searxng if selected
  Step  ─ advanced           as config step 4
  Step  ─ mcp_servers        add / edit / remove, in mcp_steps.py
"""

from __future__ import annotations

import logging

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
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
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARXNG_URL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_BASE_URL,
    DEFAULT_WEB_SEARCH_BACKEND,
    ERROR_INVALID_URL,
    ERROR_CONTEXT_EXCEEDS_MODEL,
    ERROR_MODEL_NO_TOOLS,
    ERROR_MODEL_REQUIRED,
)
from ..exceptions import CannotConnect
from ..providers.openai_compat import (
    async_fetch_models,
    async_show_model,
    model_capabilities,
    model_context_length,
)


from .mcp_steps import MCPServerStepsMixin
from .schemas import (
    _advanced_schema,
    _is_valid_url,
    _validate_advanced_input,
    _web_search_schema,
)

_LOGGER = logging.getLogger(__name__)


class AIPluginOptionsFlow(MCPServerStepsMixin, config_entries.OptionsFlow):
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
                show = await async_show_model(
                    str(self._options.get(CONF_BASE_URL, "")), model,
                    api_key=self._options.get(CONF_API_KEY),
                )
                caps = model_capabilities(show)
                if caps is not None and "tools" not in caps:
                    errors[CONF_MODEL] = ERROR_MODEL_NO_TOOLS
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
                return self.async_show_form(
                    step_id="advanced",
                    data_schema=_advanced_schema(user_input, self.hass),
                    errors=errors,
                )
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
            data_schema=_advanced_schema(self._options, self.hass),
        )
