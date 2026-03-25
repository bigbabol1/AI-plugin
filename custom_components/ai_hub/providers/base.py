"""Abstract provider base class for AI Hub."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import ChatResponse


class AbstractProvider(ABC):
    """Abstract base for LLM provider adapters.

    Each provider adapter must implement:
    - async_chat: send messages (+ optional tool schemas), return ChatResponse
    - async_complete: convenience wrapper returning plain reply text
    - async_list_models: return available model IDs for the config flow
    - async_close: close any open network sessions

    ┌──────────────────────────────────────────────────────┐
    │               AbstractProvider                        │
    │                                                      │
    │  async_chat(messages, tools) → ChatResponse          │
    │  async_complete(messages)    → str  (convenience)    │
    │  async_list_models()         → list[str]             │
    │  async_close()               → None                  │
    └──────────────────────────────────────────────────────┘
           ▲
           │ (subclasses)
           ├── OpenAICompatProvider  (v1: Ollama, llama.cpp, OpenAI, LM Studio)
           ├── GeminiProvider        (post-v1)
           └── AnthropicProvider     (post-v1)
    """

    @abstractmethod
    async def async_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send messages with optional tool schemas, return a ChatResponse.

        ChatResponse.content holds the text reply (None if LLM made a tool call).
        ChatResponse.tool_calls holds the calls to execute (empty for text reply).

        Raises:
            OrchestratorError: on provider failure (connection, timeout, auth, parse).
        """

    @abstractmethod
    async def async_complete(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Convenience wrapper: send messages, return reply text.

        Used by ContextManager summarization and no-tools paths.

        Raises:
            OrchestratorError: on provider failure.
        """

    @abstractmethod
    async def async_list_models(self) -> list[str]:
        """Return available model IDs from this provider.

        Used by the config flow to populate the model dropdown.

        Raises:
            CannotConnect: if the provider is unreachable.
        """

    @abstractmethod
    async def async_close(self) -> None:
        """Close any open network sessions. Called on integration unload."""
