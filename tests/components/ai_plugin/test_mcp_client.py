"""Tests for the MCP Tool Registry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.tools.mcp_client import (
    MCPToolRegistry,
    _MCPServerConnection,
    _State,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_tool(name: str, description: str = "", schema: dict | None = None) -> MagicMock:
    """Build a minimal MCP Tool mock."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {}}
    return tool


def _mock_call_result(text: str) -> MagicMock:
    """Build a minimal MCP call_tool result with one text content block."""
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.content = [block]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# providers/__init__.py — schema helpers (tested here as they power MCP)
# ══════════════════════════════════════════════════════════════════════════════


def test_openai_tool_schema_converts_correctly() -> None:
    """openai_tool_schema() produces well-formed OpenAI function schema."""
    from custom_components.ai_plugin.providers import openai_tool_schema

    tool = _mock_tool(
        "get_time",
        description="Get current time",
        schema={"type": "object", "properties": {"timezone": {"type": "string"}}},
    )
    schema = openai_tool_schema(tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_time"
    assert schema["function"]["description"] == "Get current time"
    assert "timezone" in schema["function"]["parameters"]["properties"]


def test_normalize_tool_calls_happy_path() -> None:
    """normalize_tool_calls() parses a standard OpenAI tool-call message."""
    import json

    from custom_components.ai_plugin.providers import normalize_tool_calls

    message = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "turn_on", "arguments": json.dumps({"entity": "light.living_room"})},
            }
        ]
    }
    calls = normalize_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "turn_on"
    assert calls[0].arguments == {"entity": "light.living_room"}


def test_normalize_tool_calls_empty() -> None:
    """normalize_tool_calls() returns [] when no tool_calls key."""
    from custom_components.ai_plugin.providers import normalize_tool_calls

    assert normalize_tool_calls({"content": "hello"}) == []


def test_normalize_tool_calls_skips_malformed() -> None:
    """Malformed tool call entries are skipped without raising."""
    from custom_components.ai_plugin.providers import normalize_tool_calls

    message = {"tool_calls": [{"bad": "data"}]}
    calls = normalize_tool_calls(message)
    assert calls == []


# ══════════════════════════════════════════════════════════════════════════════
# providers/__init__.py — ToolCall message helpers
# ══════════════════════════════════════════════════════════════════════════════


def test_tool_call_to_assistant_message() -> None:
    """ToolCall.to_assistant_message() formats correctly for OpenAI API."""
    import json

    from custom_components.ai_plugin.providers import ToolCall

    tc = ToolCall(id="call_abc", name="search", arguments={"q": "HA"})
    msg = tc.to_assistant_message()
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    fn = msg["tool_calls"][0]["function"]
    assert fn["name"] == "search"
    assert json.loads(fn["arguments"]) == {"q": "HA"}


def test_tool_call_to_tool_result_message() -> None:
    """ToolCall.to_tool_result_message() formats correctly."""
    from custom_components.ai_plugin.providers import ToolCall

    tc = ToolCall(id="call_abc", name="search", arguments={})
    msg = tc.to_tool_result_message("Found 3 results.")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_abc"
    assert msg["content"] == "Found 3 results."


# ══════════════════════════════════════════════════════════════════════════════
# _MCPServerConnection — unit tests with mocked transport
# ══════════════════════════════════════════════════════════════════════════════


async def test_server_connection_connects_and_lists_tools() -> None:
    """_MCPServerConnection starts, connects, and exposes tool list."""
    tools = [_mock_tool("get_time"), _mock_tool("set_alarm")]

    mock_session = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=tools))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_transport = MagicMock()
    mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    conn = _MCPServerConnection({"transport": "http", "url": "http://localhost:8123/mcp"})

    with (
        patch("custom_components.ai_plugin.tools.mcp_client.streamablehttp_client", return_value=mock_transport),
        patch("custom_components.ai_plugin.tools.mcp_client.ClientSession", return_value=mock_session),
    ):
        # Start connection; it will block waiting for shutdown, so we use a task
        task = asyncio.ensure_future(conn.async_start())
        # Give the event loop a moment to reach CONNECTED state
        await asyncio.sleep(0.05)

        assert conn.state == _State.CONNECTED
        assert len(conn.tools) == 2
        assert conn.tools[0].name == "get_time"

        conn._shutdown.set()
        await task


async def test_server_connection_call_tool_returns_text() -> None:
    """call_tool() returns the text content from a successful MCP call."""
    conn = _MCPServerConnection({"transport": "http", "url": "http://test"})
    conn._state = _State.CONNECTED

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=_mock_call_result("22:00"))
    conn._session = mock_session

    result = await conn.call_tool("get_time", {})
    assert result == "22:00"


async def test_server_connection_call_tool_unavailable_when_not_connected() -> None:
    """call_tool() returns an error string when not connected."""
    conn = _MCPServerConnection({"transport": "http", "url": "http://test"})
    # State is DISCONNECTED (default)
    result = await conn.call_tool("get_time", {})
    assert "unavailable" in result.lower() or "not connected" in result.lower()


async def test_server_connection_call_tool_handles_exception() -> None:
    """call_tool() returns an error string when the MCP call raises."""
    conn = _MCPServerConnection({"transport": "http", "url": "http://test"})
    conn._state = _State.CONNECTED
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=RuntimeError("timeout"))
    conn._session = mock_session

    result = await conn.call_tool("get_time", {})
    assert "failed" in result.lower() or "timeout" in result.lower()
    # State remains CONNECTED — background task owns reconnect lifecycle.
    assert conn._state == _State.CONNECTED


# ══════════════════════════════════════════════════════════════════════════════
# MCPToolRegistry — integration tests with mocked connections
# ══════════════════════════════════════════════════════════════════════════════


async def test_registry_get_tool_schemas_empty_when_no_servers() -> None:
    """Registry with no servers returns empty schema list."""
    registry = MCPToolRegistry([])
    await registry.async_setup()
    assert registry.get_tool_schemas() == []
    await registry.async_teardown()


async def test_registry_get_tool_schemas_aggregates_all_servers() -> None:
    """get_tool_schemas() combines tools from all connected servers."""
    registry = MCPToolRegistry([])

    # Inject two mock connections directly
    conn_a = MagicMock()
    conn_a.state = _State.CONNECTED
    conn_a.tools = [_mock_tool("turn_on"), _mock_tool("turn_off")]
    conn_b = MagicMock()
    conn_b.state = _State.CONNECTED
    conn_b.tools = [_mock_tool("search_web")]
    registry._servers = [conn_a, conn_b]
    registry._rebuild_tool_index()

    schemas = registry.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "turn_on" in names
    assert "turn_off" in names
    assert "search_web" in names


async def test_registry_disconnected_server_excluded_from_schemas() -> None:
    """Tools from disconnected servers are not included in schemas."""
    registry = MCPToolRegistry([])
    conn = MagicMock()
    conn.state = _State.FAILED  # not CONNECTED
    conn.tools = [_mock_tool("hidden_tool")]
    registry._servers = [conn]
    registry._rebuild_tool_index()

    assert registry.get_tool_schemas() == []


async def test_registry_call_tool_routes_to_correct_server() -> None:
    """call_tool() routes to the server that owns the tool."""
    registry = MCPToolRegistry([])

    conn = MagicMock()
    conn.state = _State.CONNECTED
    conn.tools = [_mock_tool("get_time")]
    conn.call_tool = AsyncMock(return_value="12:00")
    registry._servers = [conn]
    registry._rebuild_tool_index()

    result = await registry.call_tool("get_time", {})
    assert result == "12:00"
    conn.call_tool.assert_awaited_once_with("get_time", {})


async def test_registry_call_unknown_tool_returns_error() -> None:
    """call_tool() with an unknown name returns an error string."""
    registry = MCPToolRegistry([])
    result = await registry.call_tool("nonexistent", {})
    assert "Unknown tool" in result


async def test_registry_estimate_schema_tokens_nonzero_with_tools() -> None:
    """estimate_schema_tokens() returns a positive number when tools exist."""
    registry = MCPToolRegistry([])
    conn = MagicMock()
    conn.state = _State.CONNECTED
    conn.tools = [_mock_tool("search", description="Search the web", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
    })]
    registry._servers = [conn]
    registry._rebuild_tool_index()

    tokens = registry.estimate_schema_tokens()
    assert tokens > 0


async def test_registry_teardown_calls_stop_on_all_servers() -> None:
    """async_teardown() calls async_stop() on every server connection."""
    registry = MCPToolRegistry([])
    conn1 = MagicMock()
    conn1.async_stop = AsyncMock()
    conn2 = MagicMock()
    conn2.async_stop = AsyncMock()
    registry._servers = [conn1, conn2]

    await registry.async_teardown()

    conn1.async_stop.assert_awaited_once()
    conn2.async_stop.assert_awaited_once()


async def test_server_async_start_logs_warning_on_timeout() -> None:
    """async_start logs a warning when the server doesn't connect in time."""
    conn = _MCPServerConnection({"url": "http://localhost:8080", "name": "slow"})
    # Patch asyncio.wait_for to raise TimeoutError (server too slow)
    with patch("asyncio.wait_for", side_effect=TimeoutError):
        with patch("asyncio.ensure_future", return_value=MagicMock()):
            # Should not raise — timeout is handled gracefully
            await conn.async_start()


async def test_server_async_stop_cancels_slow_task() -> None:
    """async_stop cancels the task when it times out during shutdown."""
    conn = _MCPServerConnection({"url": "http://localhost:8080"})

    mock_task = MagicMock()
    mock_task.done = MagicMock(return_value=False)
    mock_task.cancel = MagicMock()
    conn._task = mock_task

    with patch("asyncio.wait_for", side_effect=TimeoutError):
        await conn.async_stop()

    mock_task.cancel.assert_called_once()


async def test_server_async_stop_with_no_task_does_not_raise() -> None:
    """async_stop is safe to call when no task was started."""
    conn = _MCPServerConnection({"url": "http://localhost:8080"})
    conn._task = None  # never started
    await conn.async_stop()  # should not raise


async def test_run_with_retry_handles_cancelled_error() -> None:
    """_run_with_retry exits cleanly on CancelledError."""
    conn = _MCPServerConnection({"url": "http://localhost:8080"})

    async def raise_cancelled():
        raise asyncio.CancelledError

    with patch.object(conn, "_run_once", side_effect=raise_cancelled):
        await conn._run_with_retry()

    assert conn._state == _State.DISCONNECTED


async def test_run_with_retry_handles_exception_then_shutdown() -> None:
    """_run_with_retry logs warning on exception and exits when shutdown is set."""
    conn = _MCPServerConnection({"url": "http://localhost:8080"})

    call_count = 0

    async def fail_then_shutdown():
        nonlocal call_count
        call_count += 1
        conn._shutdown.set()  # signal shutdown after first failure
        raise RuntimeError("connection refused")

    with patch.object(conn, "_run_once", side_effect=fail_then_shutdown):
        await conn._run_with_retry()

    assert conn._state == _State.DISCONNECTED
    assert call_count == 1


async def test_run_with_retry_retries_on_transient_error() -> None:
    """_run_with_retry retries after a transient error then succeeds."""
    conn = _MCPServerConnection({"url": "http://localhost:8080"})

    call_count = 0

    async def fail_once_then_shutdown():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")
        conn._shutdown.set()  # succeed and trigger exit on second call

    with patch.object(conn, "_run_once", side_effect=fail_once_then_shutdown):
        with patch("asyncio.wait_for", side_effect=TimeoutError):
            await conn._run_with_retry()

    assert call_count == 2
    assert conn._state == _State.DISCONNECTED


async def test_run_once_dispatches_to_stdio() -> None:
    """_run_once calls _run_stdio when transport is 'stdio'."""
    conn = _MCPServerConnection({"transport": "stdio", "command": "npx", "args": ["-y", "server"]})

    with patch.object(conn, "_run_stdio", new_callable=AsyncMock) as mock_stdio:
        await conn._run_once()

    mock_stdio.assert_awaited_once()


async def test_run_stdio_connects_and_lists_tools() -> None:
    """_run_stdio connects via stdio transport and exposes tool list."""
    tools = [_mock_tool("stdio_tool")]

    mock_session = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=tools))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_transport = MagicMock()
    mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    conn = _MCPServerConnection({
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-time"],
    })

    with (
        patch("custom_components.ai_plugin.tools.mcp_client.stdio_client", return_value=mock_transport),
        patch("custom_components.ai_plugin.tools.mcp_client.ClientSession", return_value=mock_session),
    ):
        task = asyncio.ensure_future(conn.async_start())
        await asyncio.sleep(0.05)

        assert conn.state == _State.CONNECTED
        assert len(conn.tools) == 1
        assert conn.tools[0].name == "stdio_tool"

        conn._shutdown.set()
        await task


async def test_registry_setup_calls_async_start_on_all_servers() -> None:
    """async_setup() calls async_start() on every server and rebuilds tool index."""
    registry = MCPToolRegistry([])
    conn1 = MagicMock()
    conn1.async_start = AsyncMock()
    conn1.state = _State.CONNECTED
    conn1.tools = []
    conn2 = MagicMock()
    conn2.async_start = AsyncMock()
    conn2.state = _State.CONNECTED
    conn2.tools = []
    registry._servers = [conn1, conn2]

    await registry.async_setup()

    conn1.async_start.assert_awaited_once()
    conn2.async_start.assert_awaited_once()


async def test_registry_estimate_schema_tokens_zero_when_no_tools() -> None:
    """estimate_schema_tokens() returns 0 when no tools are available."""
    registry = MCPToolRegistry([])
    # No servers — schemas list is empty → returns 0
    tokens = registry.estimate_schema_tokens()
    assert tokens == 0


async def test_server_connection_call_tool_handles_binary_and_other_blocks() -> None:
    """call_tool() handles binary data blocks and fallback str() blocks."""

    class BinaryBlock:
        data = b"rawbytes"

    class OtherBlock:
        def __str__(self):
            return "other_repr"

    conn = _MCPServerConnection({"url": "http://localhost:8080"})
    conn._state = _State.CONNECTED

    mock_result = MagicMock()
    mock_result.content = [BinaryBlock(), OtherBlock()]
    mock_session = MagicMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    conn._session = mock_session

    result = await conn.call_tool("some_tool", {})

    assert "[binary data: 8 bytes]" in result
    assert "other_repr" in result
