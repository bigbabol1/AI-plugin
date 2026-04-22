"""Web search tool for AI Plugin.

Exposes a single `web_search` function-calling tool to the LLM across
four backends, selectable in the Advanced config step.

Backend comparison
──────────────────
  DuckDuckGo  — zero config; unofficial HTML endpoint; may hit CAPTCHAs
  Brave       — official REST API; free tier: 2 000 req/month; API key required
  SearXNG     — self-hosted; unlimited; most reliable; URL required
  Tavily      — official REST API; free tier: 1 000 req/month; API key required

Default: DuckDuckGo.  The config UI labels it "Free (may be unreliable)"
so users understand the trade-off and can switch to Brave / SearXNG.

When a backend fails (network, CAPTCHA, quota) the tool returns a
human-readable apology string rather than raising — the LLM can relay
it gracefully and the conversation continues.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import aiohttp

from ..const import (
    BACKEND_BRAVE,
    BACKEND_DUCKDUCKGO,
    BACKEND_SEARXNG,
    BACKEND_TAVILY,
    CONF_BRAVE_API_KEY,
    CONF_MAX_RESULTS,
    CONF_SEARXNG_URL,
    CONF_TAVILY_API_KEY,
    CONF_WEB_SEARCH_BACKEND,
    DEFAULT_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_BACKEND,
)

_LOGGER = logging.getLogger(__name__)

# OpenAI function-calling schema for the web_search tool.
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use when the user asks about "
            "recent events, news, live data, or anything not likely in your training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
    },
}

_FALLBACK_MSG = (
    "Web search temporarily unavailable — try a different backend in AI Plugin settings."
)
_REQUEST_TIMEOUT = 10  # seconds


class WebSearchTool:
    """Provider-agnostic web search. One instance per config entry."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._backend: str = options.get(CONF_WEB_SEARCH_BACKEND, DEFAULT_WEB_SEARCH_BACKEND)
        self._brave_key: str | None = options.get(CONF_BRAVE_API_KEY) or None
        self._searxng_url: str | None = options.get(CONF_SEARXNG_URL) or None
        self._tavily_key: str | None = options.get(CONF_TAVILY_API_KEY) or None
        self._max_results: int = int(options.get(CONF_MAX_RESULTS, DEFAULT_MAX_RESULTS))

    async def async_search(self, query: str, strip_urls: bool = False) -> str:
        """Run a web search and return a formatted result string.

        ``strip_urls`` omits URL lines from the formatted output. Set to
        ``True`` for voice/TTS callers so spoken responses don't include
        raw web addresses. Never raises — returns a graceful error string
        on backend failure.
        """
        _LOGGER.debug(
            "WebSearchTool: backend=%s query=%r strip_urls=%s",
            self._backend, query, strip_urls,
        )
        try:
            if self._backend == BACKEND_BRAVE:
                result = await self._search_brave(query)
            elif self._backend == BACKEND_SEARXNG:
                result = await self._search_searxng(query)
            elif self._backend == BACKEND_TAVILY:
                result = await self._search_tavily(query)
            else:
                result = await self._search_duckduckgo(query)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("WebSearchTool: search failed (%s): %s", self._backend, exc)
            return _FALLBACK_MSG
        if strip_urls:
            result = _strip_url_lines(result)
        return result

    # ── DuckDuckGo ─────────────────────────────────────────────────────────────

    async def _search_duckduckgo(self, query: str) -> str:
        """Search via DuckDuckGo Lite (unofficial, no API key required).

        Uses the lightweight HTML endpoint which is more stable than the
        JSON instant-answer API.  Parses result titles and snippets with
        a simple regex — no HTML parser dependency needed.
        """
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AIPlugin/0.1; +https://github.com/bigbabol1/AI-plugin)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 202:
                    # 202 = CAPTCHA challenge — DuckDuckGo is rate-limiting us
                    return _FALLBACK_MSG
                if resp.status != 200:
                    return _FALLBACK_MSG
                html = await resp.text()

        results = _parse_ddg_lite(html, self._max_results)
        if not results:
            return _FALLBACK_MSG
        return _format_results(query, results)

    # ── Brave Search ───────────────────────────────────────────────────────────

    async def _search_brave(self, query: str) -> str:
        if not self._brave_key:
            return "Brave Search API key not configured — add it in AI Plugin settings."
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_key,
        }
        params = {"q": query, "count": self._max_results, "result_filter": "web"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 401:
                    return "Brave Search: invalid API key — check AI Plugin settings."
                if resp.status == 429:
                    return "Brave Search quota exceeded for today. Try again tomorrow."
                resp.raise_for_status()
                data = await resp.json()

        items = data.get("web", {}).get("results", [])
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
            for r in items[: self._max_results]
        ]
        return _format_results(query, results) if results else _FALLBACK_MSG

    # ── SearXNG ─────────────────────────────────────────────────────────────────

    async def _search_searxng(self, query: str) -> str:
        if not self._searxng_url:
            return "SearXNG URL not configured — add it in AI Plugin settings."
        base = self._searxng_url.rstrip("/")
        url = f"{base}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return _FALLBACK_MSG
                data = await resp.json(content_type=None)

        items = data.get("results", [])[: self._max_results]
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in items
        ]
        return _format_results(query, results) if results else _FALLBACK_MSG

    # ── Tavily ─────────────────────────────────────────────────────────────────

    async def _search_tavily(self, query: str) -> str:
        if not self._tavily_key:
            return "Tavily API key not configured — add it in AI Plugin settings."
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._tavily_key,
            "query": query,
            "max_results": self._max_results,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 401:
                    return "Tavily: invalid API key — check AI Plugin settings."
                if resp.status == 429:
                    return "Tavily quota exceeded. Try again tomorrow."
                resp.raise_for_status()
                data = await resp.json()

        items = data.get("results", [])[: self._max_results]
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in items
        ]
        return _format_results(query, results) if results else _FALLBACK_MSG


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_ddg_lite(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo Lite HTML.

    DDG Lite results appear in a table; each result row has a link with
    class "result-link" and a snippet row.  We extract these with a light
    regex — no third-party HTML parser needed.
    """
    results: list[dict[str, str]] = []
    # Find all result links: <a class="result-link" href="...">title</a>
    link_re = re.compile(
        r'class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        re.IGNORECASE,
    )
    # Find snippets: <td class="result-snippet">...</td>
    snippet_re = re.compile(
        r'class="result-snippet"[^>]*>(.*?)</td>',
        re.IGNORECASE | re.DOTALL,
    )
    links = link_re.findall(html)
    snippets = snippet_re.findall(html)

    for i, (url, title) in enumerate(links[:max_results]):
        snippet = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
        results.append({"title": title.strip(), "url": url, "snippet": snippet.strip()})

    return results


_URL_RE = re.compile(r"https?://\S+")


def _strip_url_lines(text: str) -> str:
    """Remove URL-only lines and inline URLs from a formatted result string.

    Used for voice/TTS callers. Drops any line whose only non-whitespace
    content is an http(s) URL (the ``   {url}`` lines emitted by
    ``_format_results``) and scrubs stray URLs inside snippet lines.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _URL_RE.fullmatch(stripped):
            continue
        kept.append(_URL_RE.sub("", line).rstrip())
    return "\n".join(kept)


def _format_results(query: str, results: list[dict[str, str]]) -> str:
    """Format search results as a concise text block for the LLM."""
    lines = [f'Web search results for "{query}":\n']
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()
