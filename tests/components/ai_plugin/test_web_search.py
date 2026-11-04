"""Tests for the web search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from custom_components.ai_plugin.const import (
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
from custom_components.ai_plugin.tools.web_search import (
    WebSearchTool,
    _best_place_label,
    _clean_markup,
    _format_results,
    _has_place_token,
    _maybe_inject_date,
    _maybe_inject_location,
    _parse_ddg_lite,
    _spell_out_digit_dates,
    _strip_url_lines,
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


async def _brave_call(query: str, items: list[dict] | None = None) -> tuple[dict, str]:
    """The params Brave was sent, and the formatted result."""
    data = {"web": {"results": items or [{"title": "T", "url": "u", "description": "d"}]}}
    sess = _mock_session(_mock_resp(200, json_data=data))
    tool = _make_tool(BACKEND_BRAVE, **{CONF_BRAVE_API_KEY: "key"})
    with patch("aiohttp.ClientSession", return_value=sess):
        result = await tool.async_search(query)
    return sess.get.call_args.kwargs["params"], result


async def test_brave_is_sent_the_date_spelled_out() -> None:
    """Live 2026-10-31: "Club football game 17.05.2025" ranked Club's reserve
    team first and the model named the wrong opponent; spelled out, the
    match came first."""
    params, result = await _brave_call("Club football game 17.05.2025")
    assert params["q"] == "Club football game May 17, 2025"
    assert 'results for "Club football game May 17, 2025"' in result


async def test_brave_snippets_lose_their_markup() -> None:
    items = [{
        "title": "Club&#x27;s schedule",
        "url": "https://example.com",
        "description": "The last match against <strong>Rival Town</strong> on 17 May",
    }]
    _, result = await _brave_call("Club schedule", items)
    assert "Club's schedule" in result
    assert "against Rival Town on 17 May" in result
    assert "<strong>" not in result and "&#x27;" not in result


def test_spell_out_digit_dates() -> None:
    cases = {
        "Club match 17.05.25": "Club match May 17, 2025",
        "Club Spiel 17.05.2025": "Club Spiel May 17, 2025",
        "match on 3.12.2011": "match on December 3, 2011",
        "game 17/5/2025": "game May 17, 2025",
        "game 5/17/25": "game May 17, 2025",
        "moon landing 20.07.69": "moon landing July 20, 1969",
    }
    for query, expected in cases.items():
        assert _spell_out_digit_dates(query) == expected, query


def test_spell_out_digit_dates_leaves_what_is_not_clearly_a_date() -> None:
    for query in (
        "what is new in Python 3.12.11",   # version number, not dd.mm.yy
        "concert 5/6/2025",                # May 6 or June 5
        "match 31.02.2025",                # no such day
        "results 2026-10-31",              # ISO already works
        "Bundesliga table",
    ):
        assert _spell_out_digit_dates(query) == query, query


def test_clean_markup() -> None:
    assert _clean_markup("a <b>b</b> &amp; c&#x27;s") == "a b & c's"
    assert _clean_markup("") == ""


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


async def test_searxng_non_200_returns_fallback() -> None:
    """SearXNG non-200 response returns fallback message."""
    resp = _mock_resp(503)
    tool = _make_tool(BACKEND_SEARXNG, **{CONF_SEARXNG_URL: "https://search.example.com"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("query")
    assert "No results" in result or "search" in result.lower()


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


async def test_tavily_401_returns_invalid_key_message() -> None:
    """Tavily 401 response returns invalid API key message."""
    resp = _mock_resp(401)
    tool = _make_tool(BACKEND_TAVILY, **{CONF_TAVILY_API_KEY: "bad-key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("query")
    assert "invalid" in result.lower() or "api key" in result.lower()


async def test_tavily_429_returns_quota_message() -> None:
    """Tavily 429 response returns quota exceeded message."""
    resp = _mock_resp(429)
    tool = _make_tool(BACKEND_TAVILY, **{CONF_TAVILY_API_KEY: "valid-key"})
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        result = await tool.async_search("query")
    assert "quota" in result.lower() or "exceeded" in result.lower()


async def _tavily_payload(query: str) -> dict:
    resp = _mock_resp(200, json_data={"results": [{"title": "T", "url": "u", "content": "c"}]})
    sess = _mock_session(resp)
    tool = _make_tool(BACKEND_TAVILY, **{CONF_TAVILY_API_KEY: "key"})
    with patch("aiohttp.ClientSession", return_value=sess):
        await tool.async_search(query)
    return sess.post.call_args.kwargs["json"]


async def test_tavily_searches_deeper_for_one_particular_day() -> None:
    """Basic search ranked this season's fixtures first, so "Club game
    17.05.2025" never found the match; advanced did."""
    for query in ("Club football game 17.05.2025", "Club match May 17, 2025",
                  "Club Spiel am 17. Mai"):
        assert (await _tavily_payload(query))["search_depth"] == "advanced", query


async def test_tavily_keeps_basic_search_otherwise() -> None:
    """Today's appended date must not make every 'today' query cost double."""
    for query in ("latest news Germany", "what happened in Berlin today",
                  "Bundesliga table 2025"):
        assert "search_depth" not in await _tavily_payload(query), query


# ══════════════════════════════════════════════════════════════════════════════
# _maybe_inject_date
# ══════════════════════════════════════════════════════════════════════════════


def test_inject_date_leaves_a_query_about_another_day_alone() -> None:
    """Live 2026-10-31: "football matches happening on May 17, 2025" got
    today's date appended and came back with matches on October 31, 2026."""
    for query in (
        "football matches happening on May 17, 2025",
        "football matches happening on 17.05.25",
        "was ist am 17. Mai passiert",
        "what happened in Berlin on 2025-05-17",
        "what happened in 2019",
    ):
        assert _maybe_inject_date(query) == query, query


def test_inject_date_still_pins_a_vague_query_to_today() -> None:
    for query in ("what happened in Berlin today", "what may happen today"):
        assert _maybe_inject_date(query) != query, query
        assert _maybe_inject_date(query).startswith(query + " 20"), query


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: web search dispatch
# ══════════════════════════════════════════════════════════════════════════════


async def test_orchestrator_dispatches_web_search_tool() -> None:
    """web_search tool calls go to WebSearchTool, not MCPToolRegistry."""
    from custom_components.ai_plugin.context_manager import ContextManager
    from custom_components.ai_plugin.orchestrator import Orchestrator
    from custom_components.ai_plugin.providers import ChatResponse, ToolCall

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

    mock_location = MagicMock()
    mock_location.async_resolve = AsyncMock(return_value={})

    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False
    orch._mcp = mock_mcp
    orch._memory = None
    orch._ha_local = None
    orch._max_tool_iterations = 5
    orch._web_search = mock_web
    orch._browse_url = None
    orch._provider = mock_provider
    orch._location = mock_location
    orch._last_entities = {}
    orch._conv_locks = {}

    reply = await orch.async_process("What's new in HA?", "c", "en")

    assert reply == "Here are the results."
    mock_web.async_search.assert_awaited_once_with(
        "HA news",
        strip_urls=False,
        near_user=False,
        location=None,
        language=None,
    )
    mock_mcp.call_tool.assert_not_awaited()


def test_strip_url_lines_removes_url_only_lines_and_inline_urls():
    """Voice path must not expose URLs to the LLM/TTS."""
    formatted = _format_results("weather berlin", [
        {"title": "Berlin weather", "url": "https://example.com/berlin", "snippet": "Sunny 18C. See https://foo.com for details."},
        {"title": "Climate data", "url": "http://ex.org", "snippet": "Mild"},
    ])
    stripped = _strip_url_lines(formatted)
    assert "https://" not in stripped
    assert "http://" not in stripped
    assert "Berlin weather" in stripped
    assert "Sunny 18C" in stripped
    assert "Mild" in stripped


async def test_async_search_strip_urls_forwarded_to_output(monkeypatch):
    """When strip_urls=True, async_search scrubs URLs before returning."""
    tool = WebSearchTool({CONF_WEB_SEARCH_BACKEND: BACKEND_DUCKDUCKGO})

    async def fake(_query):
        return _format_results("q", [
            {"title": "T", "url": "https://a.example", "snippet": "snip https://b.example end"},
        ])

    monkeypatch.setattr(tool, "_search_duckduckgo", fake)
    out = await tool.async_search("q", strip_urls=True)
    assert "https://" not in out
    assert "T" in out
    assert "snip" in out


# ══════════════════════════════════════════════════════════════════════════════
# Location-bias helpers
# ══════════════════════════════════════════════════════════════════════════════


def test_best_place_label_picks_city_first() -> None:
    loc = {"city": "Berlin", "region": "BE", "country_name": "Germany"}
    # Country appended for search-engine disambiguation (Berlin DE vs Berlin NY)
    assert _best_place_label(loc) == "Berlin, Germany"


def test_best_place_label_city_without_country_returns_bare_city() -> None:
    assert _best_place_label({"city": "Berlin"}) == "Berlin"


def test_best_place_label_does_not_duplicate_country_when_label_equals_country() -> None:
    # Some Nominatim payloads put the country into the city field
    assert _best_place_label({"city": "France", "country_name": "France"}) == "France"


def test_best_place_label_falls_back_to_region_then_country() -> None:
    assert _best_place_label({"region": "Bavaria"}) == "Bavaria"
    assert _best_place_label({"region": "Bavaria", "country_name": "Germany"}) == "Bavaria, Germany"
    assert _best_place_label({"country_name": "France"}) == "France"


def test_best_place_label_falls_back_to_coords() -> None:
    assert _best_place_label({"lat": 52.5, "lon": 13.4}) == "52.5000,13.4000"


def test_best_place_label_returns_none_for_empty() -> None:
    assert _best_place_label(None) is None
    assert _best_place_label({}) is None
    assert _best_place_label({"city": "  "}) is None


def test_has_place_token_detects_in_munich() -> None:
    assert _has_place_token("events in Munich tonight") is True
    assert _has_place_token("weather at New York airport") is True
    assert _has_place_token("near Saint-Étienne") is True


def test_has_place_token_does_not_detect_lowercase_token() -> None:
    assert _has_place_token("events in some bar") is False


def test_has_place_token_ignores_plain_query() -> None:
    assert _has_place_token("price of bitcoin") is False
    assert _has_place_token("") is False


def test_maybe_inject_location_appends_when_near_user_true() -> None:
    out = _maybe_inject_location(
        query="events tonight",
        near_user=True,
        location={"city": "Berlin"},
        language="en",
    )
    assert out == "events tonight near Berlin"


def test_maybe_inject_location_appends_via_locality_regex() -> None:
    """near_user=False but query contains a locality token (DE)."""
    out = _maybe_inject_location(
        query="veranstaltungen in meiner nähe",
        near_user=False,
        location={"city": "Berlin"},
        language="de",
    )
    assert out.endswith("near Berlin")


def test_maybe_inject_location_skips_when_query_has_place() -> None:
    out = _maybe_inject_location(
        query="events in Munich tonight",
        near_user=True,
        location={"city": "Berlin"},
        language="en",
    )
    assert out == "events in Munich tonight"


def test_maybe_inject_location_skips_without_intent() -> None:
    out = _maybe_inject_location(
        query="price of bitcoin",
        near_user=False,
        location={"city": "Berlin"},
        language="en",
    )
    assert out == "price of bitcoin"


def test_maybe_inject_location_skips_without_location() -> None:
    out = _maybe_inject_location(
        query="events near me",
        near_user=True,
        location=None,
        language="en",
    )
    assert out == "events near me"


def test_maybe_inject_location_idempotent() -> None:
    """Don't append the same place twice."""
    out = _maybe_inject_location(
        query="berlin events near me",
        near_user=True,
        location={"city": "Berlin"},
        language="en",
    )
    assert out == "berlin events near me"


def test_maybe_inject_location_japanese() -> None:
    out = _maybe_inject_location(
        query="近くのレストラン",
        near_user=False,
        location={"city": "東京"},
        language="ja",
    )
    assert out.endswith("near 東京")


# ══════════════════════════════════════════════════════════════════════════════
# WebSearchTool: near_user injection through async_search
# ══════════════════════════════════════════════════════════════════════════════


async def test_async_search_injects_location_when_near_user_true(monkeypatch) -> None:
    tool = _make_tool()
    captured: list[str] = []

    async def fake(query: str) -> str:
        captured.append(query)
        return "ok"

    monkeypatch.setattr(tool, "_search_duckduckgo", fake)
    await tool.async_search(
        "events tonight",
        near_user=True,
        location={"city": "Berlin"},
        language="en",
    )
    # near_user injects the location; a temporal query ("tonight") also gets
    # today's ISO date appended for grounding — assert the location bias
    # without pinning the dynamic date.
    assert len(captured) == 1
    assert captured[0].startswith("events tonight near Berlin")


async def test_async_search_no_inject_when_location_none(monkeypatch) -> None:
    tool = _make_tool()
    captured: list[str] = []

    async def fake(query: str) -> str:
        captured.append(query)
        return "ok"

    monkeypatch.setattr(tool, "_search_duckduckgo", fake)
    await tool.async_search(
        "events near me",
        near_user=True,
        location=None,
        language="en",
    )
    assert captured == ["events near me"]


async def test_async_search_voice_mode_strips_urls_and_injects(monkeypatch) -> None:
    tool = _make_tool()
    captured: list[str] = []

    async def fake(query: str) -> str:
        captured.append(query)
        return _format_results("q", [
            {"title": "T", "url": "https://a.example", "snippet": "s"},
        ])

    monkeypatch.setattr(tool, "_search_duckduckgo", fake)
    out = await tool.async_search(
        "events near me",
        strip_urls=True,
        near_user=True,
        location={"city": "Berlin"},
        language="en",
    )
    assert captured == ["events near me near Berlin"]
    assert "https://" not in out


# ══════════════════════════════════════════════════════════════════════════════
# LocationProvider
# ══════════════════════════════════════════════════════════════════════════════


async def test_location_provider_disabled_returns_empty() -> None:
    from custom_components.ai_plugin.const import (
        CONF_LOCATION_BIAS,
    )
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {CONF_LOCATION_BIAS: False}
    hass = MagicMock()
    provider = LocationProvider(hass, entry)
    assert await provider.async_resolve() == {}


async def test_location_provider_entity_priority_over_config(monkeypatch) -> None:
    from custom_components.ai_plugin.const import (
        CONF_LOCATION_BIAS,
        CONF_LOCATION_ENTITY,
    )
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {
        CONF_LOCATION_BIAS: True,
        CONF_LOCATION_ENTITY: "person.me",
    }
    state = MagicMock()
    state.attributes = {"latitude": 48.137, "longitude": 11.575}

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=state)
    hass.config.country = "DE"
    hass.config.time_zone = "Europe/Berlin"
    hass.config.language = "de"
    hass.config.latitude = 52.52
    hass.config.longitude = 13.405
    hass.config.config_dir = "/tmp"

    async def fake_geo(**kwargs):
        # Verify the entity coords win, not config.
        assert kwargs["lat"] == 48.137
        assert kwargs["lon"] == 11.575
        from custom_components.ai_plugin.tools._geocode import GeocodeResult
        return GeocodeResult(
            city="Munich",
            region="Bavaria",
            country_name="Germany",
            country_iso="DE",
        )

    monkeypatch.setattr(
        "custom_components.ai_plugin.orchestrator.location.reverse_geocode",
        fake_geo,
    )

    provider = LocationProvider(hass, entry)
    out = await provider.async_resolve()
    assert out["source"] == "entity"
    assert out["city"] == "Munich"
    assert out["lat"] == 48.137
    assert out["country_iso"] == "DE"
    assert out["language"] == "de"
    assert out["timezone"] == "Europe/Berlin"


async def test_location_provider_falls_back_to_config(monkeypatch) -> None:
    from custom_components.ai_plugin.const import CONF_LOCATION_BIAS
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {CONF_LOCATION_BIAS: True}

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.country = "DE"
    hass.config.time_zone = "Europe/Berlin"
    hass.config.language = "de"
    hass.config.latitude = 52.52
    hass.config.longitude = 13.405
    hass.config.config_dir = "/tmp"

    async def fake_geo(**kwargs):
        from custom_components.ai_plugin.tools._geocode import GeocodeResult
        return GeocodeResult("Berlin", "Berlin", "Germany", "DE")

    monkeypatch.setattr(
        "custom_components.ai_plugin.orchestrator.location.reverse_geocode",
        fake_geo,
    )

    provider = LocationProvider(hass, entry)
    out = await provider.async_resolve()
    assert out["source"] == "config"
    assert out["city"] == "Berlin"


async def test_location_provider_geocode_failure_keeps_country(monkeypatch) -> None:
    from custom_components.ai_plugin.const import CONF_LOCATION_BIAS
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {CONF_LOCATION_BIAS: True}

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.country = "DE"
    hass.config.time_zone = "Europe/Berlin"
    hass.config.language = "de"
    hass.config.latitude = 52.52
    hass.config.longitude = 13.405
    hass.config.config_dir = "/tmp"

    async def fake_geo(**kwargs):
        return None

    monkeypatch.setattr(
        "custom_components.ai_plugin.orchestrator.location.reverse_geocode",
        fake_geo,
    )

    provider = LocationProvider(hass, entry)
    out = await provider.async_resolve()
    assert out["country_iso"] == "DE"
    assert "city" not in out


async def test_location_provider_no_signal_returns_empty(monkeypatch) -> None:
    from custom_components.ai_plugin.const import CONF_LOCATION_BIAS
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {CONF_LOCATION_BIAS: True}

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.country = ""
    hass.config.time_zone = ""
    hass.config.language = ""
    hass.config.latitude = None
    hass.config.longitude = None
    hass.config.config_dir = "/tmp"

    provider = LocationProvider(hass, entry)
    out = await provider.async_resolve()
    assert out == {} or out.get("source") is None


async def test_location_provider_caches_result(monkeypatch) -> None:
    from custom_components.ai_plugin.const import CONF_LOCATION_BIAS
    from custom_components.ai_plugin.orchestrator import LocationProvider

    entry = MagicMock()
    entry.options = {CONF_LOCATION_BIAS: True}

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.country = "DE"
    hass.config.time_zone = "Europe/Berlin"
    hass.config.language = "de"
    hass.config.latitude = 52.52
    hass.config.longitude = 13.405
    hass.config.config_dir = "/tmp"

    calls = 0

    async def fake_geo(**kwargs):
        nonlocal calls
        calls += 1
        from custom_components.ai_plugin.tools._geocode import GeocodeResult
        return GeocodeResult("Berlin", None, "Germany", "DE")

    monkeypatch.setattr(
        "custom_components.ai_plugin.orchestrator.location.reverse_geocode",
        fake_geo,
    )

    provider = LocationProvider(hass, entry)
    a = await provider.async_resolve()
    b = await provider.async_resolve()
    assert a is b
    assert calls == 1



def test_results_tell_the_model_to_prefer_them_over_its_memory() -> None:
    """2026-11-03: two of four results named the current chancellor and one stale
    snippet named the old one, and the model answered with the old name."""
    from custom_components.ai_plugin.tools.web_search import _format_results

    text = _format_results("q", [{"title": "T", "url": "", "snippet": "S"}])
    assert "believe them over your memory" in text
    assert "what most of them say" in text
