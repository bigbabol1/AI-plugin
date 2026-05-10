"""Browse URL tool — fetches readable page content via Jina Reader.

Uses r.jina.ai to convert any URL to clean, LLM-friendly text.
No API key required. Works alongside web_search to let the AI
read a specific page in full after finding it via search.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "browse_url",
        "description": (
            "Read the full content of a specific web page. Use after web_search "
            "when you need to read an article, product page, or document in detail. "
            "Returns the page as readable text — no HTML, no clutter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The full URL to fetch (must start with http:// or https://). "
                        "Use URLs from web_search results or ones the user provides."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}

_REQUEST_TIMEOUT = 20
_MAX_CONTENT_CHARS = 8000
_URL_RE = re.compile(r"https?://\S+")


class BrowseUrlTool:
    """Fetches page content via Jina Reader (r.jina.ai).

    No API key, no external dependencies beyond aiohttp (already used
    by the rest of the plugin). Rate limits are generous for personal use.
    """

    async def async_browse(self, url: str, strip_urls: bool = False) -> str:
        """Fetch URL and return readable content string.

        Never raises — returns a graceful error string on failure so the
        LLM can relay it and the conversation continues.
        """
        if not url or not url.startswith(("http://", "https://")):
            return "Invalid URL — must start with http:// or https://"

        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "text",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    jina_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 422:
                        return "Could not read page — URL may be invalid or the site blocks automated access."
                    if resp.status != 200:
                        return (
                            f"Could not fetch page (HTTP {resp.status}). "
                            "The site may be unavailable or block automated access."
                        )
                    content = await resp.text()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("BrowseUrlTool: fetch failed for %s: %s", url, exc)
            return "Could not fetch page — network error or site unavailable."

        content = content.strip()
        if not content:
            return "Page fetched but contained no readable content."

        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n\n[Content truncated — page too long]"

        if strip_urls:
            content = _URL_RE.sub("", content)

        return content
