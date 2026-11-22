"""What the assistant did recently, for the sidebar panel.

Kept in memory only: the last turns and a few counters since the entry was
set up. Nothing is written to disk, and a restart or reload starts it empty.
The panel that shows it is admin-only.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

_LOGGER = logging.getLogger(__name__)

MAX_TURNS = 50
_MAX_TEXT = 400
_MAX_CONVERSATIONS = 50
# The agent id Home Assistant's voice pipeline gives a reply that came from an
# automation's sentence trigger rather than from the conversation agent.
SENTENCE_TRIGGER = "sentence_trigger"


def _clip(text: str | None) -> str:
    text = str(text or "")
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"


class ActivityLog:
    """The last turns and counters since setup."""

    def __init__(self) -> None:
        self.started = datetime.now(timezone.utc)
        self.turns: deque[dict[str, Any]] = deque(maxlen=MAX_TURNS)
        self.counts: Counter[str] = Counter()
        self.safeguards: Counter[str] = Counter()
        self.tools: Counter[str] = Counter()
        # time.monotonic() until which the model server counts as down.
        self.backend_down_until: float = 0.0
        self.last_error: str | None = None
        # conversation id -> device id of the last turn in it, so a turn the
        # pipeline answered without us can still name its satellite.
        self._devices: OrderedDict[str, str | None] = OrderedDict()
        # Called with each recorded turn: the plugin's own sensors.
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Call ``listener(turn)`` after every recorded turn; returns the undo."""
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    def record(
        self,
        *,
        text: str,
        reply: str,
        seconds: float,
        device: str | None = None,
        kind: str = "answered",
        tools: list[str] | None = None,
        safeguards: list[str] | None = None,
        error: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """One turn. ``kind``: answered, echo, offline, error or automation
        (an automation's sentence trigger answered instead of the model)."""
        tools = [t for t in (tools or []) if t]
        safeguards = [s for s in (safeguards or []) if s]
        self.counts["turns"] += 1
        self.counts[kind] += 1
        self.tools.update(tools)
        self.safeguards.update(safeguards)
        if error:
            self.last_error = _clip(error)
        if conversation_id:
            self._devices[conversation_id] = device
            self._devices.move_to_end(conversation_id)
            while len(self._devices) > _MAX_CONVERSATIONS:
                self._devices.popitem(last=False)
        turn = {
            "time": datetime.now(timezone.utc).isoformat(),
            "text": _clip(text),
            "reply": _clip(reply),
            "seconds": round(float(seconds), 2),
            "device": device,
            "kind": kind,
            "tools": tools,
            "safeguards": safeguards,
        }
        self.turns.appendleft(turn)
        for listener in list(self._listeners):
            try:
                listener(turn)
            except Exception:  # noqa: BLE001 — a sensor never fails a turn
                _LOGGER.debug("AI Plugin: activity listener failed", exc_info=True)

    def knows(self, conversation_id: str) -> bool:
        return conversation_id in self._devices

    def device_for(self, conversation_id: str) -> str | None:
        return self._devices.get(conversation_id)

    def backend_down(self) -> bool:
        return time.monotonic() < self.backend_down_until

    def as_dict(self) -> dict[str, Any]:
        answered = [t["seconds"] for t in self.turns if t["kind"] == "answered"]
        return {
            "started": self.started.isoformat(),
            "counts": dict(self.counts),
            "safeguards": dict(self.safeguards.most_common()),
            "tools": dict(self.tools.most_common(12)),
            "avg_seconds": round(sum(answered) / len(answered), 2) if answered else None,
            "backend_down": self.backend_down(),
            "last_error": self.last_error,
            "turns": list(self.turns),
        }


class SentenceTriggerWatch:
    """Notes the turns an automation's sentence trigger answered.

    The voice pipeline checks sentence triggers before it calls the
    conversation agent, so those turns never reach the plugin. Home
    Assistant's chat log still sees them: the user's text, then a reply whose
    agent id is ``sentence_trigger``. ``owns`` says whether a conversation
    belongs to this entry's log.
    """

    def __init__(self, activity: ActivityLog, owns: Callable[[str], bool]) -> None:
        self._activity = activity
        self._owns = owns
        self._pending: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def __call__(self, conversation_id: str, event_type: str, data: dict[str, Any]) -> None:
        try:
            self._handle(conversation_id, str(event_type), data or {})
        except Exception:  # noqa: BLE001 — never break Home Assistant's chat log
            _LOGGER.debug("AI Plugin: sentence trigger watch failed", exc_info=True)

    def _handle(self, conversation_id: str, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "deleted":
            self._pending.pop(conversation_id, None)
            return
        if event_type != "content_added":
            return
        content = data.get("content") or {}
        role = content.get("role")
        if role == "user":
            self._pending[conversation_id] = (str(content.get("content") or ""), time.monotonic())
            self._pending.move_to_end(conversation_id)
            while len(self._pending) > _MAX_CONVERSATIONS:
                self._pending.popitem(last=False)
            return
        if role != "assistant":
            return
        if content.get("agent_id") != SENTENCE_TRIGGER:
            # The trigger's own tool call has no text and comes before the
            # reply; any answer with text ends the turn.
            if content.get("content"):
                self._pending.pop(conversation_id, None)
            return
        pending = self._pending.pop(conversation_id, None)
        if pending is None:
            return
        if not self._owns(conversation_id):
            return
        text, started = pending
        self._activity.record(
            text=text,
            reply=str(content.get("content") or ""),
            seconds=time.monotonic() - started,
            device=self._activity.device_for(conversation_id),
            kind="automation",
            conversation_id=conversation_id,
        )


def async_watch_sentence_triggers(
    hass: Any, activity: ActivityLog, owns: Callable[[str], bool]
) -> Callable[[], None]:
    """Subscribe to Home Assistant's chat logs; returns the unsubscribe.

    A no-op on a Home Assistant without chat log subscriptions.
    """
    try:
        from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
            async_subscribe_chat_logs,
        )
    except ImportError:
        return lambda: None
    return async_subscribe_chat_logs(hass, SentenceTriggerWatch(activity, owns))
