"""Exceptions for AI Plugin."""


class AIPluginError(Exception):
    """Base exception for AI Plugin."""


class CannotConnect(AIPluginError):
    """Raised when the provider URL is unreachable."""


class InvalidAuth(AIPluginError):
    """Raised when the API key is rejected (401/403)."""


class OrchestratorError(AIPluginError):
    """Raised by the Orchestrator for any LLM-level failure.

    conversation.py catches this and returns a graceful ConversationResult.
    ai_task.py (Week 3) catches this and re-raises as HomeAssistantError.
    """
