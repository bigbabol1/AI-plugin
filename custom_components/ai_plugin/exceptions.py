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


class BackendUnreachable(OrchestratorError):
    """The model server could not be reached or did not answer in time.

    ``nothing_done`` is set by the orchestrator when the failing call was the
    turn's first and no tool had run, so the turn can safely be handed to
    another agent without repeating an action.
    """

    nothing_done: bool = False
