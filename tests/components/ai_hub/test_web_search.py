"""Tests for the web search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.ai_hub.const import (
    BACKEND_BRAVE,
    BACKEND_DUCKDUCKGO,
    BACKEND_SEARXNG,
    BACKEND_TAVILY,
    CONF_BRAVE_API_KEY,
    CONF_MAX_RESULTS,
    CONF_SEARXNG_URL,
    CONF_TAVILY_API_KEY,
    CONF_WEB_SEARCH_BACKEND,
)
from custom_components.ai_hub.tools.web_search import (
    WebSearchTool,
    _format_results,
    _parse_ddg_lite,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_tool(backend=BACKEND_DUCKDUCKGO, **extra) -> WebSearchTool:
    opts = {CONF_WEB_SEARCH_BACKEND: backend, CONF_MAX_RESULTS: 3}
    opts.update(extra)
    return WebSearchTool(opts)


def _mock_resp(status: int, text: str = "", json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    resp.json = AsyncMock(return_value=json_data or {})
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=status
        )
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(resp: MagicMock) -> MagicMock:
    sess = MagicMock()
    sess.get = MagicMock(return_value=resp)
    sess.post = MagicMock(return_value=resp)
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)
    return sess


# ══════════════════════════════════════════════════════════════════════════════
# _parse_ddg_lite
# ══════════════════════════════════════════════════════════════════════════════


def test_parse_ddg_lite_extracts_results() -> None:
    """_parse_ddg_lite finds links and snippets from DDG Lite HTML."""
    html = """
    <a class="result-link" href="https://example.com">Example Title</a>
    <td class="result-snippet">A short description here.</td>
    """
    results = _parse_ddg_lite(html, max_results=5)
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com"
    assert results[0]["title"] == "Example Title"
    assert "short description" in results[0]["snippet"]


def test_parse_ddg_lite_respects_max_results() -> None:
    """_parse_ddg_lite returns at most max_results entries."""
    link = '<a class="result-link" href="http://x.com">T</a>'
    snip = '<td class="result-snippet">S</td>'
    html = (link + snip) * 5
    results = _parse_ddg_lite(html, max_results=2)
    assert len(results) == 2


def test_parse_ddg_lite_empty_html() -> None:
    """_parse_ddg_lite returns [] for HTML with no results."""
    assert _parse_ddg_lite("<html></html>", 5) == []


# ══════════════════════════════════════════════════════════════════════════════
# _format_results
# ══════════════════════════════════════════════════════════════════════════════


def test_format_results_includes_query_and_snippets() -> None:
    results = [{"title": "T1", "url": "https://a.com", "snippet": "S1"}]
    out = _format_results("test query", results)
    assert "test query" in out
    assert "T1" in out
    assert "S1" in out
    assert "https://a.com" in out


# ══════════════════════════════════════════════════════════════════════════════
# DuckDuckGo backend
# ══════════════════════════════════════════════════════════════════════════════


async def test_ddg_happy_path_returns_results() -> None:
    """DuckDuckGo search returns formatted results on success."""
    html = (
        '<a class="result-link" href="https://result.com">Result Title</a>'
        '<td class="result-snippet">Useful information.</td>'
    )
    resp = _mock_resp(200, text=html)
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await _make_tool().async_search("home assistant")
    assert "Result Title" in result
    assert "Useful information" in result


async def test_ddg_202_returns_fallback() -> None:
    """HTTP 202 (CAPTCHA) returns the fallback message."""
    resp = _mock_resp(202)
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await _make_tool().async_search("test")
    assert "unavailable" in result.lower() or "try a different" in result.lower()


async def test_ddg_non_200_returns_fallback() -> None:
    """Non-200 status returns the fallback message."""
    resp = _mock_resp(503)
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await _make_tool().async_search("test")
    assert "unavailable" in result.lower() or "try a different" in result.lower()


async def test_ddg_empty_results_returns_fallback() -> None:
    """Empty HTML (no links) returns the fallback message."""
    resp = _mock_resp(200, text="<html><body></body></html>")
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await _make_tool().async_search("test")
    assert "unavailable" in result.lower() or "try a different" in result.lower()


async def test_ddg_network_error_returns_fallback() -> None:
    """Network error returns the fallback message gracefully."""
    sess = MagicMock()
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)
    sess.get = MagicMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError()))
    with patch("aiohttp.ClientSession", return_value=sess):
        result = await _make_tool().async_search("test")
    assert "unavailable" in result.lower() or "try a different" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Brave backend
# ══════════════════════════════════════════════════════════════════════════════


async def test_brave_happy_path() -> None:
    """Brave search returns formatted results on success."""
    data = {
        "web": {
            "results": [
                {"title": "Brave Result", "url": "https://brave.com", "description": "Fast search."}
            ]
        }
    }
    resp = _mock_resp(200, json_data=data)
    tool = _make_tool(BACKEND_BRAVE, **{CONF_BRAVE_API_KEY: "test-key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("search engine")
    assert "Brave Result" in result


async def test_brave_no_api_key_returns_message() -> None:
    """Brave without API key returns configuration message."""
    result = await _make_tool(BACKEND_BRAVE).async_search("test")
    assert "not configured" in result.lower() or "api key" in result.lower()


async def test_brave_401_returns_friendly_message() -> None:
    """Brave 401 returns a friendly invalid key message."""
    resp = _mock_resp(401, json_data={})
    tool = _make_tool(BACKEND_BRAVE, **{CONF_BRAVE_API_KEY: "bad-key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("test")
    assert "invalid" in result.lower() or "key" in result.lower()


async def test_brave_429_returns_quota_message() -> None:
    """Brave 429 returns a quota exceeded message."""
    resp = _mock_resp(429, json_data={})
    tool = _make_tool(BACKEND_BRAVE, **{CONF_BRAVE_API_KEY: "key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("test")
    assert "quota" in result.lower() or "tomorrow" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# SearXNG backend
# ══════════════════════════════════════════════════════════════════════════════


async def test_searxng_happy_path() -> None:
    """SearXNG returns formatted results on success."""
    data = {
        "results": [
            {"title": "SearXNG Hit", "url": "https://searxng.example", "content": "Good stuff."}
        ]
    }
    resp = _mock_resp(200, json_data=data)
    tool = _make_tool(BACKEND_SEARXNG, **{CONF_SEARXNG_URL: "https://search.example.com"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("query")
    assert "SearXNG Hit" in result


async def test_searxng_no_url_returns_message() -> None:
    """SearXNG without URL returns configuration message."""
    result = await _make_tool(BACKEND_SEARXNG).async_search("test")
    assert "not configured" in result.lower() or "url" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Tavily backend
# ══════════════════════════════════════════════════════════════════════════════


async def test_tavily_happy_path() -> None:
    """Tavily returns formatted results on success."""
    data = {
        "results": [{"title": "Tavily Hit", "url": "https://t.example", "content": "Info."}]
    }
    resp = _mock_resp(200, json_data=data)
    tool = _make_tool(BACKEND_TAVILY, **{CONF_TAVILY_API_KEY: "key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("query")
    assert "Tavily Hit" in result


async def test_tavily_no_key_returns_message() -> None:
    """Tavily without API key returns configuration message."""
    result = await _make_tool(BACKEND_TAVILY).async_search("test")
    assert "not configured" in result.lower() or "api key" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: web search dispatch
# ══════════════════════════════════════════════════════════════════════════════


async def test_orchestrator_dispatches_web_search_tool() -> None:
    """web_search tool calls go to WebSearchTool, not MCPToolRegistry."""
    from custom_components.ai_hub.context_manager import ContextManager
    from custom_components.ai_hub.orchestrator import Orchestrator
    from custom_components.ai_hub.providers import ChatResponse, ToolCall

    tool_call = ToolCall(id="ws1", name="web_search", arguments={"query": "HA news"})
    responses = [
        ChatResponse(content=None, tool_calls=[tool_call]),
        ChatResponse(content="Here are the results.", tool_calls=[]),
    ]
    call_i = 0

    async def mock_chat(messages, tools=None):
        nonlocal call_i
        r = responses[call_i]
        call_i += 1
        return r

    mock_provider = MagicMock()
    mock_provider.async_chat = mock_chat

    mock_web = MagicMock()
    mock_web.async_search = AsyncMock(return_value="Top news: HA 2026.3 released")

    mock_mcp = MagicMock()
    mock_mcp.get_tool_schemas = MagicMock(return_value=[])
    mock_mcp.call_tool = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.options = {"web_search_enabled": True}

    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False
    orch._mcp = mock_mcp
    orch._max_tool_iterations = 5
    orch._xml_fallback = False
    orch._web_search = mock_web
    orch._provider = mock_provider

    reply = await orch.async_process("What's new in HA?", "c", "en")

    assert reply == "Here are the results."
    mock_web.async_search.assert_awaited_once_with("HA news")
    mock_mcp.call_tool.assert_not_awaited()


async def test_orchestrator_xml_fallback_parses_tool_call() -> None:
    """XML fallback mode parses <tool_call> tags and dispatches the tool."""
    from custom_components.ai_hub.context_manager import ContextManager
    from custom_components.ai_hub.orchestrator import Orchestrator
    from custom_components.ai_hub.tools.web_search import TOOL_SCHEMA

    responses = [
        '<tool_call name="web_search">{"query": "news"}</tool_call>',
        "Final answer.",
    ]
    call_i = 0

    async def mock_complete(messages):
        nonlocal call_i
        r = responses[call_i]
        call_i += 1
        return r

    mock_provider = MagicMock()
    mock_provider.async_complete = mock_complete

    mock_web = MagicMock()
    mock_web.async_search = AsyncMock(return_value="Search results here")

    entry = MagicMock()
    entry.entry_id = "e2"
    entry.options = {}

    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False
    orch._mcp = None
    orch._max_tool_iterations = 5
    orch._xml_fallback = True
    orch._web_search = mock_web
    orch._provider = mock_provider

    reply = await orch._xml_tool_loop(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "news?"}],
        [TOOL_SCHEMA],
    )

    assert reply == "Final answer."
    mock_web.async_search.assert_awaited_once_with("news")
