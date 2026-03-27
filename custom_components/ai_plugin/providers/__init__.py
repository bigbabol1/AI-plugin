"""Shared types and utilities for AI Plugin provider adapters.

Keeps provider-specific adapters (openai_compat.py, future Gemini/Anthropic)
DRY by centralising the data structures and schema conversion helpers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)


# ── Tool call data structures ─────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool-call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_assistant_message(self) -> dict[str, Any]:
        """Return the assistant message dict that contains this tool call.

        This must be appended to the message list before the tool result so
        the LLM knows what call it made.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": json.dumps(self.arguments),
                    },
                }
            ],
        }

    def to_tool_result_message(self, result: str) -> dict[str, Any]:
        """Return the tool result message to append after executing the call."""
        return {
            "role": "tool",
            "tool_call_id": self.id,
            "content": result,
        }


@dataclass
class ChatResponse:
    """Structured response from a chat completions call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def reply_text(self) -> str:
        """Return the content as a non-None string (empty string fallback)."""
        return self.content or ""


# ── Schema conversion helpers ─────────────────────────────────────────────────


def openai_tool_schema(mcp_tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool object to OpenAI function-calling schema format.

    MCP tool objects expose: .name, .description, .inputSchema (JSON Schema dict).
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema
            if isinstance(mcp_tool.inputSchema, dict)
            else {"type": "object", "properties": {}},
        },
    }


def normalize_tool_calls(raw_message: dict[str, Any]) -> list[ToolCall]:
    """Extract and normalise tool calls from a provider response message.

    Handles differences between providers:
    - OpenAI / Ollama: message.tool_calls list with function.name + function.arguments
    - Returns [] if no tool calls present.
    """
    raw_calls = raw_message.get("tool_calls") or []
    result: list[ToolCall] = []
    for tc in raw_calls:
        try:
            fn = tc["function"]
            name = fn["name"]
            raw_args = fn.get("arguments", "{}")
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            result.append(ToolCall(id=tc.get("id", name), name=name, arguments=arguments))
        except (KeyError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Skipping malformed tool call: %s — %s", tc, exc)
    return result
