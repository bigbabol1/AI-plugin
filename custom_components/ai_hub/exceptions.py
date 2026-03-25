"""Exceptions for AI Hub."""


class AIHubError(Exception):
    """Base exception for AI Hub."""


class CannotConnect(AIHubError):
    """Raised when the provider URL is unreachable."""


class InvalidAuth(AIHubError):
    """Raised when the API key is rejected (401/403)."""


class OrchestratorError(AIHubError):
    """Raised by the Orchestrator for any LLM-level failure.

    conversation.py catches this and returns a graceful ConversationResult.
    ai_task.py (Week 3) catches this and re-raises as HomeAssistantError.
    """
