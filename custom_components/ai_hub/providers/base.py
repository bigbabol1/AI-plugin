"""Abstract provider base class for AI Hub."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractProvider(ABC):
    """Abstract base for LLM provider adapters.

    Each provider adapter must implement:
    - async_complete: send messages, return assistant reply text
    - async_list_models: return available model IDs for the config flow
    - async_close: close any open network sessions

    ┌──────────────────────────────────────────────────────┐
    │               AbstractProvider                        │
    │                                                      │
    │  async_complete(messages) → str                      │
    │  async_list_models()      → list[str]                │
    │  async_close()            → None                     │
    └──────────────────────────────────────────────────────┘
           ▲
           │ (subclasses)
           ├── OpenAICompatProvider  (v1: Ollama, llama.cpp, OpenAI, LM Studio)
           ├── GeminiProvider        (post-v1)
           └── AnthropicProvider     (post-v1)
    """

    @abstractmethod
    async def async_complete(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Send a list of messages and return the assistant's reply.

        Args:
            messages: OpenAI-format message list:
                      [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns:
            The assistant's reply as a plain string.

        Raises:
            OrchestratorError: on provider failure (connection, timeout, auth, parse).
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
