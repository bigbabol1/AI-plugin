"""Fixtures for AI Plugin tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.ai_plugin.const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_PROVIDER,
    DOMAIN,
    PROVIDER_OPENAI_COMPAT,
)
from custom_components.ai_plugin.exceptions import CannotConnect

MOCK_MODELS = ["llama3.2:3b", "qwen2.5:7b", "mistral"]

MOCK_CONFIG_DATA = {CONF_PROVIDER: PROVIDER_OPENAI_COMPAT}
MOCK_CONFIG_OPTIONS = {
    CONF_BASE_URL: "http://localhost:11434/v1",
    CONF_MODEL: "llama3.2:3b",
}


@pytest.fixture
def mock_fetch_models():
    """Mock a successful model fetch returning MOCK_MODELS."""
    with patch(
        "custom_components.ai_plugin.config_flow.async_fetch_models",
        AsyncMock(return_value=MOCK_MODELS),
    ) as mock:
        yield mock


@pytest.fixture
def mock_fetch_models_fail():
    """Mock a failed model fetch (provider unreachable)."""
    with patch(
        "custom_components.ai_plugin.config_flow.async_fetch_models",
        AsyncMock(side_effect=CannotConnect("Connection refused")),
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_complete():
    """Mock the provider's async_complete to return a canned reply."""
    with patch(
        "custom_components.ai_plugin.providers.openai_compat.OpenAICompatProvider.async_complete",
        AsyncMock(return_value="I'm here to help with your home."),
    ) as mock:
        yield mock


@pytest.fixture
def mock_setup_entry():
    """Prevent actual platform setup during config flow tests."""
    with patch(
        "custom_components.ai_plugin.async_setup_entry",
        AsyncMock(return_value=True),
    ) as mock:
        yield mock
