"""MCP Tool Registry for AI Plugin.

Manages persistent connections to one or more MCP servers, fetches their
tool schemas on startup, and routes tool calls to the correct server.

Connection model
─────────────────
Each server runs as a long-lived background asyncio.Task that owns the
transport context manager for the duration of the integration's lifetime.
When a call fails the task reconnects with exponential backoff.

  DISCONNECTED ──[start()]──► CONNECTING ──[initialize()]──► CONNECTED
                                                                  │
                                                    [call fails / conn drop]
                                                                  ▼
                                                           RECONNECTING
                                                                  │
                                                    [max backoff = 30 s]
                                                                  ▼
                                                             FAILED (logged)

Transport support
─────────────────
• HTTP (Streamable HTTP / SSE) — HA built-in MCP server, ollmcp, remote servers
• stdio — local tools spawned as subprocesses (killed on async_teardown to
  prevent zombie processes on HA restart)

Server config dict shape (stored in config_entry.options["mcp_servers"]):
  HTTP server:  {"transport": "http",  "url": "http://localhost:8123/mcp"}
  stdio server: {"transport": "stdio", "command": "uvx", "args": ["mcp-server-time"]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from enum import Enum
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ..context_manager import ContextManager
from ..providers import openai_tool_schema

_LOGGER = logging.getLogger(__name__)



def _create_ssl_context() -> ssl.SSLContext:
    """Build an SSL context — must run in an executor (load_verify_locations is blocking I/O)."""
    try:
        import certifi  # noqa: PLC0415
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


# Extra directories to search for executables that may not be in HA's restricted PATH.
# HA runs in Docker; tools like uvx (uv) are installed outside the standard PATH.
_EXTRA_PATH_DIRS = [
    "/root/.local/bin",
    "/root/.cargo/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/home/root/.local/bin",
]


def _resolve_command(command: str) -> str:
    """Return the full path to *command* if found in known extra directories.

    HA's subprocess environment has a restricted PATH that often omits user-local
    bin directories where tools like uvx are installed.  Falls back to the bare
    command name so the OS can still resolve it if it happens to be on PATH.
    """
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    if os.path.isabs(command):
        return command
    # Try the standard PATH first
    found = shutil.which(command)
    if found:
        return found
    # Then check well-known extra directories
    for directory in _EXTRA_PATH_DIRS:
        candidate = os.path.join(directory, command)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return command  # fall back to bare name; subprocess will raise if not found


# Backoff constants for reconnection
_BACKOFF_INITIAL = 2.0
_BACKOFF_MAX = 30.0
# Seconds to wait for a server to become CONNECTED on startup.
# stdio servers (uvx) may need to download packages on first run — allow 120s.
_CONNECT_TIMEOUT = 120.0


class _State(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class _MCPServerConnection:
    """Persistent connection to a single MCP server."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._state = _State.DISCONNECTED
        self._session: ClientSession | None = None
        self._tools: list[Any] = []          # list[mcp.types.Tool]
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._name = config.get("url") or config.get("command", "unknown")

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> _State:
        return self._state

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    async def async_start(self) -> None:
        """Start the background connection task and return immediately.

        Connection happens asynchronously — tools become available once the
        server reaches CONNECTED state.  async_setup_entry is not blocked.
        """
        self._task = asyncio.ensure_future(self._run_with_retry())

    async def async_stop(self) -> None:
        """Signal shutdown and wait for the background task to finish."""
        self._shutdown.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool by name. Returns result text or an error description.

        Never raises — returns an error string instead so the LLM can relay
        the failure gracefully rather than crashing the conversation.
        """
        if self._session is None or self._state != _State.CONNECTED:
            return f"[Tool {name!r} unavailable — MCP server {self._name!r} not connected]"
        try:
            result = await self._session.call_tool(name, arguments)
            # result.content is a list of content blocks; join text blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "data"):
                    parts.append(f"[binary data: {len(block.data)} bytes]")
                else:
                    parts.append(str(block))
            return "\n".join(parts) or "(empty result)"
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "AI Plugin MCP: tool %r call failed on %r: %s", name, self._name, exc
            )
            # Do not set RECONNECTING here — the background task owns the
            # transport lifecycle and will reconnect when the transport drops.
            # Forcibly clearing _session would leave the background task in an
            # inconsistent state (blocking on _shutdown.wait with a dead ref).
            return f"[Tool {name!r} call failed: {exc}]"

    # ── private ───────────────────────────────────────────────────────────────

    async def _run_with_retry(self) -> None:
        backoff = _BACKOFF_INITIAL
        while not self._shutdown.is_set():
            self._state = _State.CONNECTING
            self._ready.clear()  # Reset readiness for each connection attempt
            try:
                await self._run_once()
                backoff = _BACKOFF_INITIAL  # reset on clean exit
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if self._shutdown.is_set():
                    break
                self._state = _State.RECONNECTING
                # Recursively unwrap nested ExceptionGroups (anyio/asyncio TaskGroup)
                # to surface the real root cause in the log.
                cause: BaseException = exc
                while isinstance(cause, BaseExceptionGroup) and cause.exceptions:
                    cause = cause.exceptions[0]
                _LOGGER.warning(
                    "AI Plugin MCP: connection to %r failed (%s: %s), retrying in %.0fs",
                    self._name,
                    type(cause).__name__,
                    cause,
                    backoff,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=backoff
                    )
                except TimeoutError:
                    pass
                backoff = min(backoff * 2, _BACKOFF_MAX)
        self._state = _State.DISCONNECTED

    async def _run_once(self) -> None:
        """Open transport, initialise session, then wait for shutdown signal."""
        transport = self._config.get("transport", "http")
        if transport == "stdio":
            await self._run_stdio()
        else:
            await self._run_http()

    async def _run_http(self) -> None:
        url = self._config["url"]
        headers: dict[str, str] = {}
        if token := self._config.get("token"):
            headers["Authorization"] = f"Bearer {token}"

        # ssl.create_default_context() calls load_verify_locations which is
        # blocking file I/O — run it in an executor to keep the event loop free.
        loop = asyncio.get_running_loop()
        ssl_ctx = await loop.run_in_executor(None, _create_ssl_context)

        def _client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            kw: dict[str, Any] = {"follow_redirects": True, "verify": ssl_ctx}
            if timeout is not None:
                kw["timeout"] = timeout
            if headers is not None:
                kw["headers"] = headers
            if auth is not None:
                kw["auth"] = auth
            return httpx.AsyncClient(**kw)

        async with streamablehttp_client(
            url,
            headers=headers or None,
            httpx_client_factory=_client_factory,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await self._on_connected(session)
                await self._shutdown.wait()

    async def _run_stdio(self) -> None:
        import tempfile  # noqa: PLC0415

        params = StdioServerParameters(
            command=_resolve_command(self._config["command"]),
            args=self._config.get("args", []),
            env=self._config.get("env"),
        )
        # Use a real temp file so anyio gets a valid file descriptor for stderr.
        # We read and log its contents after the session ends (or fails).
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errfile:
            try:
                async with stdio_client(params, errlog=errfile) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await self._on_connected(session)
                        await self._shutdown.wait()
            finally:
                errfile.seek(0)
                stderr_out = errfile.read().strip()
                if stderr_out:
                    _LOGGER.warning(
                        "AI Plugin MCP subprocess stderr for %r:\n%s",
                        self._name,
                        stderr_out,
                    )

    async def _on_connected(self, session: ClientSession) -> None:
        """Cache tool list and signal readiness after successful connection."""
        result = await session.list_tools()
        self._tools = result.tools
        self._session = session
        self._state = _State.CONNECTED
        self._ready.set()
        _LOGGER.debug(
            "AI Plugin MCP: connected to %r — %d tools available: %s",
            self._name,
            len(self._tools),
            [t.name for t in self._tools],
        )


# ══════════════════════════════════════════════════════════════════════════════
# MCPToolRegistry — orchestrates all server connections
# ══════════════════════════════════════════════════════════════════════════════


class MCPToolRegistry:
    """Manages all MCP server connections for a single config entry.

    Created by async_setup_entry, torn down by async_unload_entry.
    The Orchestrator calls get_tool_schemas() and call_tool() per request.
    """

    def __init__(self, server_configs: list[dict[str, Any]]) -> None:
        self._servers: list[_MCPServerConnection] = [
            _MCPServerConnection(cfg) for cfg in server_configs
        ]
        # name → server index for O(1) routing
        self._tool_index: dict[str, _MCPServerConnection] = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def async_setup(self) -> None:
        """Kick off background connections for all configured MCP servers.

        Returns immediately — connections happen in the background so
        async_setup_entry (and HA startup) are not blocked.  Tools become
        available as each server reaches CONNECTED state.
        """
        for server in self._servers:
            await server.async_start()

    async def async_teardown(self) -> None:
        """Disconnect all servers and kill stdio subprocesses.

        Must be called from async_unload_entry to prevent zombie processes
        when HA restarts or the user removes the integration.
        """
        if not self._servers:
            return
        await asyncio.gather(
            *(s.async_stop() for s in self._servers), return_exceptions=True
        )

    # ── tool access ───────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all tools across all servers."""
        schemas = []
        for server in self._servers:
            if server.state == _State.CONNECTED:
                schemas.extend(openai_tool_schema(t) for t in server.tools)
        return schemas

    def estimate_schema_tokens(self) -> int:
        """Estimate the token cost of all current tool schemas.

        Passed to ContextManager so it can reserve accurate space in the
        context window for tool definitions.
        """
        schemas = self.get_tool_schemas()
        if not schemas:
            return 0
        raw = json.dumps(schemas)
        return ContextManager.estimate_tokens(raw)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the owning server and return the result.

        Rebuilds the tool index if the named tool is not found (handles
        the case where a server reconnected and refreshed its tool list).
        Returns an error string on failure — never raises.
        """
        server = self._tool_index.get(name)
        if server is None:
            # Tool index may be stale — rebuild and retry once
            self._rebuild_tool_index()
            server = self._tool_index.get(name)
        if server is None:
            return f"[Unknown tool: {name!r}]"
        return await server.call_tool(name, arguments)

    # ── private ───────────────────────────────────────────────────────────────

    def _rebuild_tool_index(self) -> None:
        self._tool_index = {}
        for server in self._servers:
            for tool in server.tools:
                self._tool_index[tool.name] = server
