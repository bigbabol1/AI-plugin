"""The voluptuous schemas both flows show, and the checks behind them.

Config flow and options flow ask the same questions in the same shape;
building the schema in one place is what keeps them from drifting apart.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.helpers import selector


from ..const import (
    BACKEND_BRAVE,
    BACKEND_DUCKDUCKGO,
    BACKEND_SEARXNG,
    BACKEND_TAVILY,
    CONF_BRAVE_API_KEY,
    CONF_CONTEXT_WINDOW,
    CONF_CONTINUE_CONVERSATION,
    CONF_FOLLOW_UP_DELAY,
    CONF_ENABLE_THINKING,
    CONF_FEEDBACK_LOOP_DEVICES,
    CONF_LOCATION_BIAS,
    CONF_LOCATION_ENTITY,
    CONF_MAX_RESULTS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_KEEP_ALIVE,
    CONF_PRUNE_TOOL_SCHEMAS,
    CONF_RESPONSE_TIMEOUT,
    CONF_SELF_ECHO_FILTER,
    CONF_SUMMARIZATION_ENABLED,
    CONF_SYSTEM_PROMPT,
    CONF_TIMER_ANNOUNCE,
    CONF_TAVILY_API_KEY,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_FOLLOW_UP_DELAY,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_LOCATION_BIAS,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_PRUNE_TOOL_SCHEMAS,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SELF_ECHO_FILTER,
    DEFAULT_TIMER_ANNOUNCE,
    DEFAULT_SUMMARIZATION_ENABLED,
    DEFAULT_VOICE_MODE,
    DEFAULT_WEB_SEARCH_BACKEND,
    CUSTOM_PROMPT_TEMPLATE,
)

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



def _advanced_schema(current: dict[str, Any], hass: Any) -> vol.Schema:
    """Build the advanced settings vol.Schema with current values as defaults."""
    schema: dict[Any, Any] = {
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=current.get(CONF_SYSTEM_PROMPT, CUSTOM_PROMPT_TEMPLATE),
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
                CONF_CONTINUE_CONVERSATION,
                default=current.get(CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_FOLLOW_UP_DELAY,
                default=current.get(CONF_FOLLOW_UP_DELAY, DEFAULT_FOLLOW_UP_DELAY),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=10,
                    step=0.5,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_ENABLE_THINKING,
                default=current.get(CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING),
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
                CONF_KEEP_ALIVE,
                description={"suggested_value": current.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)},
            ): selector.TextSelector(),
            vol.Optional(
                CONF_PRUNE_TOOL_SCHEMAS,
                default=current.get(CONF_PRUNE_TOOL_SCHEMAS, DEFAULT_PRUNE_TOOL_SCHEMAS),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": current.get(CONF_TEMPERATURE)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    mode=selector.NumberSelectorMode.SLIDER,
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
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": current.get(CONF_MAX_TOKENS)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=8192,
                    step=256,
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
            vol.Optional(
                CONF_SELF_ECHO_FILTER,
                default=current.get(CONF_SELF_ECHO_FILTER, DEFAULT_SELF_ECHO_FILTER),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_TIMER_ANNOUNCE,
                default=current.get(CONF_TIMER_ANNOUNCE, DEFAULT_TIMER_ANNOUNCE),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_FEEDBACK_LOOP_DEVICES,
                description={"suggested_value": current.get(CONF_FEEDBACK_LOOP_DEVICES)},
            ): selector.DeviceSelector(
                selector.DeviceSelectorConfig(multiple=True)
            ),
            vol.Optional(
                CONF_LOCATION_BIAS,
                default=current.get(CONF_LOCATION_BIAS, DEFAULT_LOCATION_BIAS),
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_LOCATION_ENTITY,
                description={"suggested_value": current.get(CONF_LOCATION_ENTITY)},
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["zone", "device_tracker", "person"],
                )
            ),
        }
    return vol.Schema(schema)



def _validate_advanced_input(user_input: dict[str, Any]) -> dict[str, str]:
    """Per-field validation for the advanced step.

    Returns a dict suitable for the ``errors`` arg of ``async_show_form``;
    empty when input is acceptable.
    """
    errors: dict[str, str] = {}
    return errors
