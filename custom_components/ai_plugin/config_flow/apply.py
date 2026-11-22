"""Turning what a settings form returned into saved options.

The options flow and the sidebar panel save the same sections; both call
these, so a check added here holds for both.
"""

from __future__ import annotations

from typing import Any

from ..const import (
    BACKEND_SEARXNG,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CONTEXT_WINDOW,
    CONF_CONVERSATION_MEMORY,
    CONF_DIARY_DAYS,
    CONF_ENABLE_THINKING,
    CONF_MAX_RESULTS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MODEL,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARXNG_URL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_WEB_SEARCH_BACKEND,
    ERROR_INVALID_URL,
    ERROR_MODEL_NO_TOOLS,
    ERROR_MODEL_REQUIRED,
)
from ..option_values import merge_advanced_options
from ..providers.openai_compat import (
    async_show_model,
    model_capabilities,
    model_context_length,
    model_limit_errors,
)
from .schemas import _is_valid_url, _validate_advanced_input

_ADVANCED_INTS = (
    CONF_CONTEXT_WINDOW, CONF_MAX_TOOL_ITERATIONS, CONF_RESPONSE_TIMEOUT,
    CONF_MAX_RESULTS, CONF_MAX_TOKENS,
)
_ADVANCED_FLOATS = (CONF_TEMPERATURE, CONF_TOP_P)


def apply_memory(options: dict[str, Any], user_input: dict[str, Any]) -> dict[str, Any]:
    """Saved facts, satellite conversation and diary settings."""
    user_input = dict(user_input)
    for key in (CONF_CONVERSATION_MEMORY, CONF_DIARY_DAYS):
        if key in user_input:
            user_input[key] = int(user_input[key])
    return {**options, **user_input}


async def apply_advanced(
    options: dict[str, Any], user_input: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Advanced settings: (new options, errors). Errors leave options as they were."""
    user_input = dict(user_input)
    errors = _validate_advanced_input(user_input)
    if not errors and (
        user_input.get(CONF_CONTEXT_WINDOW) or user_input.get(CONF_ENABLE_THINKING)
    ):
        # Fail-open Ollama check: refuse what the model cannot do, here where
        # it can be said, not later on a voice turn.
        show = await async_show_model(
            str(options.get(CONF_BASE_URL, "")),
            str(options.get(CONF_MODEL, "")),
            api_key=options.get(CONF_API_KEY),
        )
        errors = model_limit_errors(
            user_input, model_capabilities(show), model_context_length(show)
        )
    if errors:
        return options, errors
    for key in _ADVANCED_INTS:
        if key in user_input:
            user_input[key] = int(user_input[key])
    for key in _ADVANCED_FLOATS:
        if key in user_input:
            user_input[key] = float(user_input[key])
    return merge_advanced_options(options, user_input), {}


async def apply_provider(
    options: dict[str, Any], user_input: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Server address, API key and model in one go (the panel's form).

    The API key is kept when ``user_input`` leaves it out and removed when it
    is an empty string.
    """
    new = dict(options)
    base_url = str(user_input.get(CONF_BASE_URL, new.get(CONF_BASE_URL, ""))).strip().rstrip("/")
    if not _is_valid_url(base_url):
        return options, {CONF_BASE_URL: ERROR_INVALID_URL}
    new[CONF_BASE_URL] = base_url
    if CONF_API_KEY in user_input:
        key = str(user_input.get(CONF_API_KEY) or "").strip()
        if key:
            new[CONF_API_KEY] = key
        else:
            new.pop(CONF_API_KEY, None)
    model = str(user_input.get(CONF_MODEL, new.get(CONF_MODEL, "")) or "").strip()
    if not model:
        return options, {CONF_MODEL: ERROR_MODEL_REQUIRED}
    show = await async_show_model(base_url, model, api_key=new.get(CONF_API_KEY))
    caps = model_capabilities(show)
    if caps is not None and "tools" not in caps:
        return options, {CONF_MODEL: ERROR_MODEL_NO_TOOLS}
    new[CONF_MODEL] = model
    return new, {}


def apply_web_search(
    options: dict[str, Any], user_input: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Web search, with the SearXNG address on the same form."""
    new = {**options, **user_input}
    if CONF_MAX_RESULTS in new:
        new[CONF_MAX_RESULTS] = int(new[CONF_MAX_RESULTS])
    backend = new.get(CONF_WEB_SEARCH_BACKEND, DEFAULT_WEB_SEARCH_BACKEND)
    if new.get(CONF_WEB_SEARCH_ENABLED) and backend == BACKEND_SEARXNG:
        url = str(new.get(CONF_SEARXNG_URL) or "").strip()
        if not _is_valid_url(url):
            return options, {CONF_SEARXNG_URL: ERROR_INVALID_URL}
        new[CONF_SEARXNG_URL] = url
    return new, {}
