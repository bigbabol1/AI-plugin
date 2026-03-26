"""Tests for the AI Hub config flow and options flow.

These tests exercise HA's config-flow runner and require a real
homeassistant installation (Python ≥ 3.13 + pytest-homeassistant-
custom-component).  They are skipped automatically when running in a
lightweight CI environment that only has the stub stubs installed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# Skip the entire module when the real HA config-flow runner is absent.
pytest.importorskip(
    "homeassistant.data_entry_flow",
    reason="Full homeassistant package required for config-flow tests",
)
try:
    # Real HA exposes FlowManager; our stub does not.
    from homeassistant.config_entries import ConfigEntriesFlowManager  # noqa: F401
except ImportError:
    pytest.skip(
        "Real homeassistant config-flow machinery not available",
        allow_module_level=True,
    )

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ai_hub.const import (
    BACKEND_SEARXNG,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_SEARXNG_URL,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_BACKEND,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_BASE_URL,
    DOMAIN,
    PROVIDER_OPENAI_COMPAT,
)
from custom_components.ai_hub.exceptions import CannotConnect

from .conftest import MOCK_MODELS


# ══════════════════════════════════════════════════════════════════════════════
# Config flow: happy path
# ══════════════════════════════════════════════════════════════════════════════


async def test_full_config_flow_with_models(
    hass: HomeAssistant, mock_fetch_models, mock_setup_entry
) -> None:
    """Happy path: complete config flow with successful model fetch."""
    # Step 1: Init → provider
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "provider"

    # Step 1: Submit provider
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "provider_url"

    # Step 2a: Submit URL → models fetched
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "http://localhost:11434/v1"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"
    mock_fetch_models.assert_called_once_with("http://localhost:11434/v1")

    # Step 2b: Submit model
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_MODEL: "llama3.2:3b", "api_key": ""}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "web_search"

    # Step 3: Skip web search
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_WEB_SEARCH_ENABLED: False}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced"

    # Step 4: Submit advanced with defaults
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={
            "system_prompt": "",
            "context_window": 8192,
            "summarization_enabled": True,
            "voice_mode": False,
            "xml_fallback": False,
            "max_tool_iterations": 5,
            "response_timeout": 30,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "AI Hub (llama3.2:3b)"
    assert result["data"] == {CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    assert result["options"][CONF_MODEL] == "llama3.2:3b"
    assert result["options"][CONF_BASE_URL] == "http://localhost:11434/v1"


async def test_full_config_flow_model_fetch_fails(
    hass: HomeAssistant, mock_fetch_models_fail, mock_setup_entry
) -> None:
    """Config flow completes even when model fetch fails (free-text fallback)."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "http://localhost:11434/v1"}
    )
    # Should still show model step (free-text, not dropdown)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"

    # User types model name manually
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_MODEL: "llama3.2:3b"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "web_search"


# ══════════════════════════════════════════════════════════════════════════════
# Config flow: validation errors
# ══════════════════════════════════════════════════════════════════════════════


async def test_invalid_url_shows_error(hass: HomeAssistant) -> None:
    """Invalid URL format blocks provider_url step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    # Submit a bad URL
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "not-a-url"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "provider_url"
    assert "base_url" in result["errors"]
    assert result["errors"]["base_url"] == "invalid_url"


async def test_empty_model_shows_error(
    hass: HomeAssistant, mock_fetch_models_fail
) -> None:
    """Empty model name blocks model step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "http://localhost:11434/v1"}
    )
    # Submit empty model
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_MODEL: ""}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"
    assert "model" in result["errors"]
    assert result["errors"]["model"] == "model_required"


# ══════════════════════════════════════════════════════════════════════════════
# Config flow: SearXNG sub-step
# ══════════════════════════════════════════════════════════════════════════════


async def test_searxng_substep_reached_when_searxng_selected(
    hass: HomeAssistant, mock_fetch_models, mock_setup_entry
) -> None:
    """Selecting SearXNG backend leads to the searxng sub-step."""
    # Run through to web_search step
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "http://localhost:11434/v1"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_MODEL: "llama3.2:3b"}
    )

    # Enable web search + select SearXNG
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={
            CONF_WEB_SEARCH_ENABLED: True,
            CONF_WEB_SEARCH_BACKEND: BACKEND_SEARXNG,
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "searxng"

    # Submit invalid SearXNG URL
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SEARXNG_URL: "not-a-url"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "searxng"
    assert "searxng_url" in result["errors"]

    # Submit valid SearXNG URL → proceeds to advanced
    with patch(
        "custom_components.ai_hub.config_flow.async_fetch_models",
        AsyncMock(side_effect=CannotConnect),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={CONF_SEARXNG_URL: "https://search.example.com"},
        )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced"


async def test_searxng_substep_skipped_for_other_backends(
    hass: HomeAssistant, mock_fetch_models, mock_setup_entry
) -> None:
    """Non-SearXNG backends skip the searxng sub-step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_BASE_URL: "http://localhost:11434/v1"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_MODEL: "llama3.2:3b"}
    )
    # Enable web search with Brave (not SearXNG)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={
            CONF_WEB_SEARCH_ENABLED: True,
            CONF_WEB_SEARCH_BACKEND: "brave",
            "brave_api_key": "sk-test",
        },
    )
    # Should skip searxng step and go straight to advanced
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced"


# ══════════════════════════════════════════════════════════════════════════════
# Options flow
# ══════════════════════════════════════════════════════════════════════════════


async def test_options_flow_mcp_stub(hass: HomeAssistant, mock_setup_entry) -> None:
    """MCP servers option aborts with informational message."""
    entry = _create_mock_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "init"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "mcp_servers"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "mcp_coming_soon"


async def test_options_flow_advanced(hass: HomeAssistant, mock_setup_entry) -> None:
    """Advanced options flow updates voice_mode."""
    entry = _create_mock_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "advanced"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "system_prompt": "Be terse.",
            "context_window": 4096,
            "summarization_enabled": True,
            "voice_mode": True,
            "xml_fallback": False,
            "max_tool_iterations": 3,
            "response_timeout": 20,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_VOICE_MODE] is True
    assert result["data"]["system_prompt"] == "Be terse."


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _create_mock_entry(hass: HomeAssistant) -> config_entries.ConfigEntry:
    """Create and add a mock config entry to hass for options flow tests."""
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.util.ulid import ulid_now

    entry = config_entries.ConfigEntry(
        version=1,
        minor_version=1,
        domain=DOMAIN,
        title="AI Hub (llama3.2:3b)",
        data={CONF_PROVIDER: PROVIDER_OPENAI_COMPAT},
        options={
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
        },
        source=config_entries.SOURCE_USER,
        entry_id=ulid_now(),
        unique_id=None,
        discovery_keys={},
    )
    hass.config_entries._entries[entry.entry_id] = entry  # noqa: SLF001 — test helper
    return entry
