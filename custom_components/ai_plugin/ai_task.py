"""AI Task platform — serve ai_task.generate_data from the same local LLM.

Lets automations and scripts use the configured backend for one-shot data
generation (summaries, structured extraction) without going through the
conversation agent. Text and JSON-structured output; no attachments (local
text models), no image generation.
"""

from __future__ import annotations

import json
import logging
import re

from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .exceptions import OrchestratorError
from .providers.openai_compat import OpenAICompatProvider

_LOGGER = logging.getLogger(__name__)

# ```json ... ``` fences that models love to wrap structured output in.
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AI Task entity from a config entry."""
    async_add_entities([AIPluginTaskEntity(hass, config_entry)])


def _schema_hint(structure) -> str:
    """Render the requested structure as a JSON-schema hint for the prompt.

    voluptuous_openapi ships with HA core (the llm helpers use it); if it
    is unavailable the model still gets a generic JSON instruction.
    """
    try:
        from voluptuous_openapi import convert  # noqa: PLC0415

        return json.dumps(convert(structure))
    except Exception:  # noqa: BLE001
        return "a single JSON object"


def _extract_json(text: str):
    """Parse model output as JSON, tolerating code fences."""
    candidate = text.strip()
    if m := _FENCE_RE.match(candidate):
        candidate = m.group(1)
    return json.loads(candidate)


class AIPluginTaskEntity(ai_task.AITaskEntity):
    """AI Task entity backed by the integration's OpenAI-compatible provider."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}-ai-task"
        self._provider = OpenAICompatProvider.from_options(dict(entry.options))

    async def _async_generate_data(
        self, task: ai_task.GenDataTask, chat_log
    ) -> ai_task.GenDataTaskResult:
        """Run one generate_data task against the configured backend."""
        system = (
            "You are a data-generation assistant inside Home Assistant. "
            "Follow the instructions exactly. Output ONLY the requested "
            "result — no preamble, no explanations, no markdown fences."
        )
        if task.structure is not None:
            system += (
                " Respond with a single JSON object matching this schema: "
                f"{_schema_hint(task.structure)}"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task.instructions},
        ]

        try:
            reply = await self._provider.async_complete(messages)
        except OrchestratorError as exc:
            raise HomeAssistantError(f"AI Plugin task failed: {exc}") from exc

        # Record the exchange in the task's chat log when the API allows.
        if hasattr(chat_log, "async_add_assistant_content_without_tools"):
            try:
                from homeassistant.components.conversation import (  # noqa: PLC0415
                    AssistantContent,
                )

                await chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(
                        agent_id=getattr(self, "entity_id", None) or "ai_plugin",
                        content=reply,
                    )
                )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: chat-log append failed", exc_info=True)

        if task.structure is None:
            data = reply
        else:
            try:
                data = _extract_json(reply)
            except json.JSONDecodeError as exc:
                raise HomeAssistantError(
                    f"Model did not return valid JSON for task {task.name!r}: "
                    f"{reply[:200]!r}"
                ) from exc

        return ai_task.GenDataTaskResult(
            conversation_id=getattr(chat_log, "conversation_id", ""),
            data=data,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close the provider session when the entity is removed."""
        await self._provider.async_close()
