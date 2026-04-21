"""OpenAI-compatible provider for AI Plugin.

Covers: Ollama, llama.cpp, OpenAI, LM Studio, xAI Grok, and any
endpoint that speaks the OpenAI chat completions API.

Uses raw aiohttp — no openai SDK dependency per CEO plan ("zero new
pip dependencies").

Ollama note: the OpenAI-compat shim at /v1/chat/completions silently
drops unknown body fields, so per-request `options.num_ctx` has no
effect. When we detect Ollama (via /api/version), we switch to the
native /api/chat endpoint which honours `options.num_ctx` and reloads
the runner at the requested context size. Non-Ollama backends keep
using the OpenAI endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import aiohttp

from ..exceptions import CannotConnect, OrchestratorError
from . import ChatResponse, normalize_tool_calls
from .base import AbstractProvider

_LOGGER = logging.getLogger(__name__)

_MODEL_FETCH_TIMEOUT = 10
_PROBE_TIMEOUT = 3

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_v1(base_url: str) -> str:
    """Return the server root with any trailing /v1 segment removed."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


class OpenAICompatProvider(AbstractProvider):
    """Provider adapter for OpenAI-compatible endpoints.

    Two code paths:
    • OpenAI spec (/v1/chat/completions) — default.
    • Ollama native (/api/chat) — used when server identifies as Ollama.
      Needed so options.num_ctx actually reaches the runtime.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: int,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._server_root = _strip_v1(self._base_url)
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._session: aiohttp.ClientSession | None = None
        self._is_ollama: bool | None = None
        self._probe_lock = asyncio.Lock()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._build_headers())
        return self._session

    async def _detect_ollama(self) -> bool:
        """Probe /api/version once; cache the result.

        Ollama returns {"version":"X.Y.Z"} at /api/version. OpenAI, xAI,
        LM Studio, llama.cpp all return 404. On any probe failure we
        assume non-Ollama so behaviour falls back to the OpenAI path.
        """
        if self._is_ollama is not None:
            return self._is_ollama
        async with self._probe_lock:
            if self._is_ollama is not None:
                return self._is_ollama
            url = f"{self._server_root}/api/version"
            session = self._get_session()
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=_PROBE_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        self._is_ollama = isinstance(data, dict) and "version" in data
                        if self._is_ollama:
                            _LOGGER.info(
                                "AI Plugin: Ollama %s detected — using native /api/chat",
                                data.get("version"),
                            )
                    else:
                        self._is_ollama = False
            except (aiohttp.ClientError, TimeoutError, ValueError):
                self._is_ollama = False
            return self._is_ollama

    def _strip_think(self, content: str | None) -> str | None:
        if not content:
            return content
        stripped = _THINK_RE.sub("", content).strip()
        if stripped != content:
            _LOGGER.debug(
                "AI Plugin: stripped thinking tokens (%d chars) from response",
                len(content) - len(stripped),
            )
            return stripped
        return content

    async def async_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Dispatch to Ollama native or OpenAI-spec path based on server."""
        if await self._detect_ollama():
            return await self._chat_ollama(messages, tools)
        return await self._chat_openai(messages, tools)

    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        session = self._get_session()
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens

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
                    content=self._strip_think(message.get("content")),
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

    async def _chat_ollama(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Call Ollama's native /api/chat so options.num_ctx actually takes effect."""
        session = self._get_session()
        url = f"{self._server_root}/api/chat"
        options: dict[str, Any] = {}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._top_p is not None:
            options["top_p"] = self._top_p
        if self._max_tokens:
            options["num_predict"] = self._max_tokens
        if self._context_window:
            options["num_ctx"] = self._context_window
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

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
                message = data.get("message")
                if not isinstance(message, dict):
                    raise OrchestratorError(
                        f"Unexpected Ollama response: missing 'message' ({list(data.keys())})"
                    )
                return ChatResponse(
                    content=self._strip_think(message.get("content")),
                    tool_calls=normalize_tool_calls(message),
                )
        except OrchestratorError:
            raise
        except aiohttp.ClientConnectorError as exc:
            raise OrchestratorError(
                f"Cannot connect to {self._server_root}: {exc}"
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
        """Send messages, return reply text only. Used for no-tools paths."""
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
    """Standalone model fetch for use in the config flow."""
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
