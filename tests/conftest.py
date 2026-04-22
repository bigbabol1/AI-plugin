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
    def __init__(self, *a, **kw): pass


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


_make_module("homeassistant.const", Platform=_Platform)

# ── homeassistant.components.conversation ────────────────────────────────────

class _ConversationEntity:
    pass


class _ConversationInput:
    def __init__(self, text, context, conversation_id, device_id, language, agent_id=None):
        self.text = text
        self.context = context
        self.conversation_id = conversation_id
        self.device_id = device_id
        self.language = language
        self.agent_id = agent_id


class _ConversationResult:
    def __init__(self, response, conversation_id):
        self.response = response
        self.conversation_id = conversation_id


_conv_mod = _make_module(
    "homeassistant.components.conversation",
    ConversationEntity=_ConversationEntity,
    ConversationInput=_ConversationInput,
    ConversationResult=_ConversationResult,
)
_make_module("homeassistant.components", conversation=_conv_mod)

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
_make_module("homeassistant.util", ulid=sys.modules["homeassistant.util.ulid"])

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
)

# hass fixture used by a few tests — minimal stand-in
import pytest


@pytest.fixture
def hass():
    """Minimal hass stand-in for tests that need hass.data."""
    h = MagicMock()
    h.data = {}
    return h
