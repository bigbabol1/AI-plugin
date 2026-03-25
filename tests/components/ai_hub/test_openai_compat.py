"""Tests for the OpenAI-compatible provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.ai_hub.exceptions import CannotConnect, OrchestratorError
from custom_components.ai_hub.providers.openai_compat import (
    OpenAICompatProvider,
    async_fetch_models,
)


def _make_provider(
    base_url: str = "http://localhost:11434/v1",
    model: str = "llama3.2:3b",
    api_key: str | None = None,
    timeout: int = 30,
) -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url=base_url, model=model, api_key=api_key, timeout=timeout)


def _mock_response(status: int, json_data: dict) -> MagicMock:
    """Build a mock aiohttp response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=status
        )
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# async_complete
# ══════════════════════════════════════════════════════════════════════════════


async def test_complete_returns_reply() -> None:
    """Happy path: valid response returns the assistant message content."""
    provider = _make_provider()
    mock_resp = _mock_response(
        200,
        {"choices": [{"message": {"content": "Hello from Ollama!"}}]},
    )

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(return_value=mock_resp)
        result = await provider.async_complete([{"role": "user", "content": "hi"}])

    assert result == "Hello from Ollama!"


async def test_complete_raises_on_401() -> None:
    """401 response raises OrchestratorError (invalid API key)."""
    provider = _make_provider(api_key="bad-key")
    mock_resp = _mock_response(401, {})

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(return_value=mock_resp)
        with pytest.raises(OrchestratorError, match="Invalid API key"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_complete_raises_on_connection_error() -> None:
    """Network connection failure raises OrchestratorError."""
    provider = _make_provider()

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("Connection refused")
            )
        )
        with pytest.raises(OrchestratorError, match="Cannot connect"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_complete_raises_on_timeout() -> None:
    """Request timeout raises OrchestratorError."""
    provider = _make_provider(timeout=1)

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(side_effect=TimeoutError)
        with pytest.raises(OrchestratorError, match="timed out"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_complete_raises_on_malformed_response() -> None:
    """Missing 'choices' key raises OrchestratorError."""
    provider = _make_provider()
    mock_resp = _mock_response(200, {"unexpected": "format"})

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(return_value=mock_resp)
        with pytest.raises(OrchestratorError, match="Unexpected response format"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_api_key_added_to_header() -> None:
    """API key is sent as Bearer token."""
    provider = _make_provider(api_key="sk-test-key")
    headers = provider._build_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"


async def test_no_api_key_omits_auth_header() -> None:
    """No API key means no Authorization header."""
    provider = _make_provider(api_key=None)
    headers = provider._build_headers()
    assert "Authorization" not in headers


# ══════════════════════════════════════════════════════════════════════════════
# async_list_models
# ══════════════════════════════════════════════════════════════════════════════


async def test_list_models_returns_ids() -> None:
    """Happy path: /models returns a list of model IDs."""
    provider = _make_provider()
    mock_resp = _mock_response(
        200,
        {"data": [{"id": "llama3.2:3b"}, {"id": "qwen2.5:7b"}]},
    )

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.get = MagicMock(return_value=mock_resp)
        models = await provider.async_list_models()

    assert models == ["llama3.2:3b", "qwen2.5:7b"]


async def test_list_models_raises_cannot_connect_on_error() -> None:
    """Connection failure during model list raises CannotConnect."""
    provider = _make_provider()

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.get = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("refused")
            )
        )
        with pytest.raises(CannotConnect):
            await provider.async_list_models()


# ══════════════════════════════════════════════════════════════════════════════
# async_fetch_models (standalone config flow helper)
# ══════════════════════════════════════════════════════════════════════════════


async def test_async_fetch_models_returns_ids() -> None:
    """Standalone fetch returns model IDs on success."""
    mock_resp = _mock_response(
        200, {"data": [{"id": "llama3.2:3b"}, {"id": "mistral"}]}
    )
    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_ctx)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session_ctx.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
        models = await async_fetch_models("http://localhost:11434/v1")

    assert "llama3.2:3b" in models
    assert "mistral" in models


async def test_async_fetch_models_raises_cannot_connect() -> None:
    """Standalone fetch raises CannotConnect on network error."""
    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_ctx)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session_ctx.get = MagicMock(
        side_effect=aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError("refused")
        )
    )

    with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
        with pytest.raises(CannotConnect):
            await async_fetch_models("http://localhost:11434/v1")


# ══════════════════════════════════════════════════════════════════════════════
# Session lifecycle
# ══════════════════════════════════════════════════════════════════════════════


async def test_async_close_closes_session() -> None:
    """async_close closes the aiohttp session."""
    provider = _make_provider()
    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    provider._session = mock_session

    await provider.async_close()

    mock_session.close.assert_awaited_once()
    assert provider._session is None


async def test_async_close_is_idempotent_when_no_session() -> None:
    """async_close does nothing if no session was ever created."""
    provider = _make_provider()
    await provider.async_close()  # should not raise


async def test_get_session_recreates_when_closed() -> None:
    """_get_session creates a new session when the existing one is closed."""
    provider = _make_provider()
    mock_closed = MagicMock()
    mock_closed.closed = True
    provider._session = mock_closed

    with patch("aiohttp.ClientSession") as mock_cls:
        new_session = MagicMock()
        mock_cls.return_value = new_session
        result = provider._get_session()

    assert result is new_session
    assert provider._session is new_session


async def test_async_chat_includes_tools_in_payload() -> None:
    """async_chat sends tools and tool_choice when tools are provided."""
    provider = _make_provider()
    mock_resp = _mock_response(
        200,
        {"choices": [{"message": {"content": "reply", "tool_calls": None}}]},
    )

    captured_kwargs = {}

    def capturing_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = capturing_post
        tools = [{"type": "function", "function": {"name": "get_time"}}]
        await provider.async_chat([{"role": "user", "content": "hi"}], tools=tools)

    assert "tools" in captured_kwargs.get("json", {})
    assert captured_kwargs["json"]["tool_choice"] == "auto"


async def test_async_complete_raises_when_content_is_none() -> None:
    """async_complete raises OrchestratorError when response content is None."""
    from custom_components.ai_hub.providers import ChatResponse

    provider = _make_provider()
    with patch.object(
        provider,
        "async_chat",
        new_callable=AsyncMock,
        return_value=ChatResponse(content=None, tool_calls=[]),
    ):
        with pytest.raises(OrchestratorError, match="content is None"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_async_chat_raises_on_generic_client_error() -> None:
    """Generic aiohttp.ClientError in async_chat raises OrchestratorError."""
    provider = _make_provider()

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.post = MagicMock(
            side_effect=aiohttp.ClientError("generic error")
        )
        with pytest.raises(OrchestratorError, match="HTTP error"):
            await provider.async_complete([{"role": "user", "content": "hi"}])


async def test_list_models_raises_cannot_connect_on_malformed_response() -> None:
    """Malformed /models response (missing 'id' key) raises CannotConnect."""
    provider = _make_provider()
    # Data where items don't have 'id' key
    mock_resp = _mock_response(200, {"data": [{"name": "llama"}]})

    with patch.object(provider, "_get_session") as mock_session:
        mock_session.return_value.get = MagicMock(return_value=mock_resp)
        with pytest.raises(CannotConnect, match="Unexpected"):
            await provider.async_list_models()


async def test_async_fetch_models_with_api_key_sends_auth_header() -> None:
    """async_fetch_models includes Authorization header when api_key is given."""
    mock_resp = _mock_response(200, {"data": [{"id": "gpt-4"}]})
    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_ctx)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session_ctx.get = MagicMock(return_value=mock_resp)

    captured_headers = {}

    def fake_client_session(headers=None, **kwargs):
        if headers:
            captured_headers.update(headers)
        return mock_session_ctx

    with patch("aiohttp.ClientSession", side_effect=fake_client_session):
        models = await async_fetch_models("http://localhost:11434/v1", api_key="sk-test")

    assert models == ["gpt-4"]
    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == "Bearer sk-test"


async def test_async_fetch_models_raises_cannot_connect_on_malformed_response() -> None:
    """async_fetch_models raises CannotConnect when response lacks 'id' key."""
    mock_resp = _mock_response(200, {"data": [{"name": "no-id"}]})
    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_ctx)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session_ctx.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
        with pytest.raises(CannotConnect, match="Unexpected"):
            await async_fetch_models("http://localhost:11434/v1")


# ══════════════════════════════════════════════════════════════════════════════
# AbstractProvider: default async_chat fallback
# ══════════════════════════════════════════════════════════════════════════════


async def test_abstract_provider_default_async_chat_wraps_complete() -> None:
    """AbstractProvider.async_chat default wraps async_complete in a ChatResponse."""
    from custom_components.ai_hub.providers import ChatResponse
    from custom_components.ai_hub.providers.base import AbstractProvider

    class ConcreteProvider(AbstractProvider):
        async def async_complete(self, messages):
            return "hello from complete"

        async def async_list_models(self):
            return []

        async def async_close(self):
            pass

    provider = ConcreteProvider()
    response = await provider.async_chat([{"role": "user", "content": "hi"}])

    assert isinstance(response, ChatResponse)
    assert response.content == "hello from complete"
    assert response.tool_calls == []
