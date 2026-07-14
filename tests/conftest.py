"""Top-level conftest: inject minimal homeassistant stubs so tests run without
a real HA installation (Python 3.11 / CI environments where
pytest-homeassistant-custom-component cannot be installed).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ── homeassistant.core ────────────────────────────────────────────────────────

class _Context:
    def __init__(self, user_id=None, *a, **kw):
        self.user_id = user_id


class _HomeAssistant:
    pass


_core = _make_module(
    "homeassistant.core",
    HomeAssistant=_HomeAssistant,
    Context=_Context,
)

# ── homeassistant.config_entries ─────────────────────────────────────────────

class _ConfigEntry:
    pass


class _ConfigFlow:
    pass


class _OptionsFlow:
    pass


_cfg_entries = _make_module(
    "homeassistant.config_entries",
    ConfigEntry=_ConfigEntry,
    ConfigFlow=_ConfigFlow,
    OptionsFlow=_OptionsFlow,
    SOURCE_USER="user",
)

# ── homeassistant.const ───────────────────────────────────────────────────────

class _Platform:
    CONVERSATION = "conversation"
    AI_TASK = "ai_task"


_make_module("homeassistant.const", Platform=_Platform)

# ── homeassistant.exceptions ──────────────────────────────────────────────────

class _HomeAssistantError(Exception):
    pass


_make_module("homeassistant.exceptions", HomeAssistantError=_HomeAssistantError)

# ── homeassistant.components.ai_task ─────────────────────────────────────────

class _AITaskEntity:
    pass


class _AITaskEntityFeature:
    GENERATE_DATA = 1
    SUPPORT_ATTACHMENTS = 2
    GENERATE_IMAGE = 4


class _GenDataTask:
    def __init__(self, name="task", instructions="", structure=None, attachments=None):
        self.name = name
        self.instructions = instructions
        self.structure = structure
        self.attachments = attachments


class _GenDataTaskResult:
    def __init__(self, conversation_id="", data=None):
        self.conversation_id = conversation_id
        self.data = data


_ai_task_mod = _make_module(
    "homeassistant.components.ai_task",
    AITaskEntity=_AITaskEntity,
    AITaskEntityFeature=_AITaskEntityFeature,
    GenDataTask=_GenDataTask,
    GenDataTaskResult=_GenDataTaskResult,
)

# ── homeassistant.components.conversation ────────────────────────────────────

import uuid as _uuid_mod


class _StubChatLog:
    """Chat-log stand-in: records streamed deltas, mimics the 2025.7+ API."""

    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id or str(_uuid_mod.uuid4())
        self.deltas: list[str] = []
        self.content: list = []

    async def async_add_delta_content_stream(self, agent_id, stream):
        text = ""
        async for delta in stream:
            piece = delta.get("content")
            if piece:
                self.deltas.append(piece)
                text += piece
        self.content.append({"role": "assistant", "content": text})
        return
        yield  # pragma: no cover — marks this as an async generator

    async def async_add_assistant_content_without_tools(self, content):
        self.content.append(content)


class _ConversationEntity:
    async def async_process(self, user_input):
        """Mimic HA's base entity: open a chat log, delegate to the handler."""
        chat_log = _StubChatLog(user_input.conversation_id)
        return await self._async_handle_message(user_input, chat_log)


class _ConversationEntityFeature:
    CONTROL = 1


class _ConversationInput:
    def __init__(self, text, context, conversation_id, device_id, language, agent_id=None):
        self.text = text
        self.context = context
        self.conversation_id = conversation_id
        self.device_id = device_id
        self.language = language
        self.agent_id = agent_id


class _ConversationResult:
    def __init__(self, response, conversation_id, continue_conversation=False):
        self.response = response
        self.conversation_id = conversation_id
        self.continue_conversation = continue_conversation


class _AssistantContent:
    def __init__(self, agent_id="", content=""):
        self.agent_id = agent_id
        self.content = content


_conv_mod = _make_module(
    "homeassistant.components.conversation",
    ConversationEntity=_ConversationEntity,
    ConversationEntityFeature=_ConversationEntityFeature,
    ConversationInput=_ConversationInput,
    ConversationResult=_ConversationResult,
    ChatLog=_StubChatLog,
    AssistantContent=_AssistantContent,
)
_make_module(
    "homeassistant.components",
    conversation=_conv_mod,
    ai_task=_ai_task_mod,
)

# ── homeassistant.helpers.intent ─────────────────────────────────────────────

class _IntentResponse:
    def __init__(self, language="en"):
        self.language = language
        self.speech: dict = {}

    def async_set_speech(self, text: str) -> None:
        self.speech = {"plain": {"speech": text, "extra_data": None}}


_intent_mod = _make_module("homeassistant.helpers.intent", IntentResponse=_IntentResponse)

# ── homeassistant.helpers.device_registry ────────────────────────────────────

class _DeviceInfo:
    def __init__(self, **kw): pass


_dev_reg_mod = _make_module("homeassistant.helpers.device_registry", DeviceInfo=_DeviceInfo, async_get=MagicMock())

# ── homeassistant.helpers.area_registry ──────────────────────────────────────

_area_reg_mod = _make_module("homeassistant.helpers.area_registry", async_get=MagicMock())

# ── homeassistant.helpers.entity_registry ────────────────────────────────────

_ent_reg_mod = _make_module("homeassistant.helpers.entity_registry", async_get=MagicMock())

# ── homeassistant.helpers.entity_platform ────────────────────────────────────

_make_module("homeassistant.helpers.entity_platform", AddEntitiesCallback=object)

# ── homeassistant.helpers ─────────────────────────────────────────────────────

_helpers = _make_module(
    "homeassistant.helpers",
    intent=_intent_mod,
    area_registry=_area_reg_mod,
    entity_registry=_ent_reg_mod,
    device_registry=_dev_reg_mod,
)

# ── homeassistant.util.ulid ───────────────────────────────────────────────────

import uuid as _uuid

_make_module("homeassistant.util.ulid", ulid_now=lambda: str(_uuid.uuid4()))

# homeassistant.util.logging — needed by pytest-homeassistant-custom-component
# autouse fixture `fail_on_log_exception`.
def _log_exception_noop(format_err, *args):  # noqa: ANN001
    pass

_make_module("homeassistant.util.logging", log_exception=_log_exception_noop)
_util_mod = _make_module(
    "homeassistant.util",
    ulid=sys.modules["homeassistant.util.ulid"],
    logging=sys.modules["homeassistant.util.logging"],
)

# ── homeassistant.data_entry_flow ─────────────────────────────────────────────

class _FlowResultType:
    FORM = "form"
    CREATE_ENTRY = "create_entry"
    ABORT = "abort"
    MENU = "menu"


_make_module("homeassistant.data_entry_flow", FlowResultType=_FlowResultType)

# ── top-level homeassistant package ──────────────────────────────────────────

_ha = _make_module(
    "homeassistant",
    config_entries=sys.modules["homeassistant.config_entries"],
    core=sys.modules["homeassistant.core"],
    const=sys.modules["homeassistant.const"],
    components=sys.modules["homeassistant.components"],
    helpers=sys.modules["homeassistant.helpers"],
    util=sys.modules["homeassistant.util"],
)

# hass fixture used by a few tests — minimal stand-in
import pytest


@pytest.fixture
def hass():
    """Minimal hass stand-in for tests that need hass.data."""
    h = MagicMock()
    h.data = {}
    return h
