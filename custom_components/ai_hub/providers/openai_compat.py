"""OpenAI-compatible provider for AI Hub.

Covers: Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any
endpoint that speaks the OpenAI chat completions API.

Uses raw aiohttp — no openai SDK dependency per CEO plan ("zero new
pip dependencies").
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from ..exceptions import CannotConnect, OrchestratorError
from . import ChatResponse, normalize_tool_calls
from .base import AbstractProvider

_LOGGER = logging.getLogger(__name__)

# Config flow model-fetch timeout (separate from completion timeout)
_MODEL_FETCH_TIMEOUT = 10


class OpenAICompatProvider(AbstractProvider):
    """Provider adapter for OpenAI-compatible endpoints.

    ┌────────────────────────────────────────────────────────┐
    │  OpenAICompatProvider                                   │
    │                                                        │
    │  base_url ──► POST /chat/completions ──► str reply     │
    │                                                        │
    │  Error paths:                                          │
    │  • 401/403       → InvalidAuth (re-raised as Orc.Err.) │
    │  • ConnectorError → CannotConnect (→ Orc.Err.)         │
    │  • Timeout        → OrchestratorError                  │
    │  • Bad JSON       → OrchestratorError                  │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._build_headers())
        return self._session

    async def async_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send messages (optionally with tool schemas) and return a ChatResponse.

        ChatResponse contains:
        - content: the assistant's text reply (None when the LLM made a tool call)
        - tool_calls: list of ToolCall objects to execute (empty for a text reply)

        Raises OrchestratorError on any provider failure.
        """
        session = self._get_session()
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status in (401, 403):
                    raise OrchestratorError(
                        f"Invalid API key (HTTP {resp.status})"
                    ) from None
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                try:
                    message = data["choices"][0]["message"]
                except (KeyError, IndexError) as exc:
                    raise OrchestratorError(
                        f"Unexpected response format: {exc}"
                    ) from exc
                return ChatResponse(
                    content=message.get("content"),
                    tool_calls=normalize_tool_calls(message),
                )

        except OrchestratorError:
            raise
        except aiohttp.ClientConnectorError as exc:
            raise OrchestratorError(
                f"Cannot connect to {self._base_url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise OrchestratorError(
                f"Request timed out after {self._timeout}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise OrchestratorError(f"HTTP error: {exc}") from exc

    async def async_complete(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Backward-compatible wrapper: send messages, return reply text only.

        Used by ContextManager.summarize_if_needed() and no-tools paths.
        Delegates to async_chat().
        """
        response = await self.async_chat(messages)
        if response.content is None:
            raise OrchestratorError("Unexpected response format: content is None")
        return response.content

    async def async_list_models(self) -> list[str]:
        """Fetch available models from /models endpoint."""
        session = self._get_session()
        url = f"{self._base_url}/models"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=_MODEL_FETCH_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return [m["id"] for m in data.get("data", [])]
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise CannotConnect(f"Cannot reach {url}: {exc}") from exc
        except (KeyError, TypeError) as exc:
            raise CannotConnect(f"Unexpected /models response: {exc}") from exc

    async def async_close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


async def async_fetch_models(
    base_url: str,
    api_key: str | None = None,
) -> list[str]:
    """Standalone model fetch for use in the config flow.

    Returns a list of model IDs, or raises CannotConnect on failure.
    Creates and closes its own session (config flow runs once).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/models"
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=_MODEL_FETCH_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return [m["id"] for m in data.get("data", [])]
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise CannotConnect(f"Cannot reach {url}: {exc}") from exc
    except (KeyError, TypeError) as exc:
        raise CannotConnect(f"Unexpected /models response: {exc}") from exc
