"""Top-level conftest: inject minimal homeassistant stubs so tests run without
a real HA installation (Python 3.11 / CI environments where
pytest-homeassistant-custom-component cannot be installed).
"""

from __future__ import annotations

import datetime as _dt
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
    STT = "stt"


_make_module("homeassistant.const", Platform=_Platform, EVENT_STATE_CHANGED="state_changed")

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
# ── homeassistant.components.stt ─────────────────────────────────────────────

import enum as _enum
from dataclasses import dataclass as _dataclass


class _AudioFormats(str, _enum.Enum):
    WAV = "wav"
    OGG = "ogg"


class _AudioCodecs(str, _enum.Enum):
    PCM = "pcm"
    OPUS = "opus"


class _AudioBitRates(int, _enum.Enum):
    BITRATE_8 = 8
    BITRATE_16 = 16


class _AudioSampleRates(int, _enum.Enum):
    SAMPLERATE_8000 = 8000
    SAMPLERATE_16000 = 16000


class _AudioChannels(int, _enum.Enum):
    CHANNEL_MONO = 1
    CHANNEL_STEREO = 2


class _SpeechResultState(str, _enum.Enum):
    SUCCESS = "success"
    ERROR = "error"


@_dataclass
class _SpeechMetadata:
    language: str
    format: _AudioFormats
    codec: _AudioCodecs
    bit_rate: _AudioBitRates
    sample_rate: _AudioSampleRates
    channel: _AudioChannels


@_dataclass
class _SpeechResult:
    text: str | None
    result: _SpeechResultState


@_dataclass
class _SpeechAudioProcessing:
    requires_external_vad: bool = True
    prefers_auto_gain_enabled: bool = True
    prefers_noise_reduction_enabled: bool = True


class _SpeechToTextEntity:
    @property
    def audio_processing(self):
        return _SpeechAudioProcessing()


_stt_engines: dict = {}
_stt_mod = _make_module(
    "homeassistant.components.stt",
    SpeechToTextEntity=_SpeechToTextEntity,
    SpeechMetadata=_SpeechMetadata,
    SpeechResult=_SpeechResult,
    SpeechResultState=_SpeechResultState,
    SpeechAudioProcessing=_SpeechAudioProcessing,
    AudioFormats=_AudioFormats,
    AudioCodecs=_AudioCodecs,
    AudioBitRates=_AudioBitRates,
    AudioSampleRates=_AudioSampleRates,
    AudioChannels=_AudioChannels,
    async_get_speech_to_text_engine=lambda hass, engine_id: _stt_engines.get(engine_id),
    ENGINES=_stt_engines,
)

# ── homeassistant.components.assist_pipeline ─────────────────────────────────

_pipeline_list: list = []


async def _async_update_pipeline(hass, pipeline, **changes):
    for key, val in changes.items():
        setattr(pipeline, key, val)


def _async_get_pipeline(hass, pipeline_id=None):
    for p in _pipeline_list:
        if pipeline_id is None or p.id == pipeline_id:
            return p
    raise KeyError(pipeline_id)


_assist_pipeline_mod = _make_module(
    "homeassistant.components.assist_pipeline",
    async_get_pipelines=lambda hass: list(_pipeline_list),
    async_get_pipeline=_async_get_pipeline,
    async_update_pipeline=_async_update_pipeline,
    PIPELINES=_pipeline_list,
)

# ── homeassistant.components.tts ──────────────────────────────────────────────


class _ResultStream:
    def __init__(self, engine, language, options):
        self.engine, self.language, self.options = engine, language, options
        self.message = None
        self.url = f"/api/tts_proxy/{engine}-{language}.mp3"

    def async_set_message(self, message):
        self.message = message


_tts_streams: list = []


def _async_create_stream(hass, engine, language=None, options=None):
    stream = _ResultStream(engine, language, options)
    _tts_streams.append(stream)
    return stream


_tts_mod = _make_module(
    "homeassistant.components.tts",
    async_resolve_engine=lambda hass, engine: engine,
    async_create_stream=_async_create_stream,
    STREAMS=_tts_streams,
)

_make_module(
    "homeassistant.components",
    tts=_tts_mod,
    conversation=_conv_mod,
    ai_task=_ai_task_mod,
    stt=_stt_mod,
    assist_pipeline=_assist_pipeline_mod,
)

# ── homeassistant.helpers.intent ─────────────────────────────────────────────

class _IntentResponse:
    def __init__(self, language="en"):
        self.language = language
        self.speech: dict = {}

    def async_set_speech(self, text: str) -> None:
        self.speech = {"plain": {"speech": text, "extra_data": None}}


class _IntentHandleError(Exception):
    pass


class _UnknownIntent(Exception):
    pass


_intent_mod = _make_module(
    "homeassistant.helpers.intent",
    IntentResponse=_IntentResponse,
    IntentHandleError=_IntentHandleError,
    UnknownIntent=_UnknownIntent,
    async_handle=MagicMock(),
)

# ── homeassistant.helpers.device_registry ────────────────────────────────────

class _DeviceInfo:
    def __init__(self, **kw): pass


_dev_reg_mod = _make_module("homeassistant.helpers.device_registry", DeviceInfo=_DeviceInfo, async_get=MagicMock())

# ── homeassistant.helpers.area_registry ──────────────────────────────────────

_area_reg_mod = _make_module("homeassistant.helpers.area_registry", async_get=MagicMock())

# ── homeassistant.helpers.entity_registry ────────────────────────────────────

_ent_reg_mod = _make_module(
    "homeassistant.helpers.entity_registry",
    async_get=MagicMock(),
    async_entries_for_device=MagicMock(return_value=[]),
)

# ── homeassistant.helpers.entity_platform ────────────────────────────────────

_make_module("homeassistant.helpers.entity_platform", AddEntitiesCallback=object)

# ── homeassistant.helpers.storage / aiohttp_client ───────────────────────────

class _Store:
    """In-memory Store stand-in."""

    def __init__(self, hass=None, version=1, key="", **kw):
        self.key = key
        self.data = None

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        import copy

        self.data = copy.deepcopy(data)

    async def async_remove(self):
        self.data = None


_storage_mod = _make_module("homeassistant.helpers.storage", Store=_Store)
_aiohttp_client_mod = _make_module(
    "homeassistant.helpers.aiohttp_client", async_get_clientsession=MagicMock()
)

# ── homeassistant.helpers ─────────────────────────────────────────────────────

_helpers = _make_module(
    "homeassistant.helpers",
    intent=_intent_mod,
    area_registry=_area_reg_mod,
    entity_registry=_ent_reg_mod,
    device_registry=_dev_reg_mod,
    storage=_storage_mod,
    aiohttp_client=_aiohttp_client_mod,
)

# ── homeassistant.util.ulid ───────────────────────────────────────────────────

import uuid as _uuid

_make_module("homeassistant.util.ulid", ulid_now=lambda: str(_uuid.uuid4()))

# ── homeassistant.util.dt ─────────────────────────────────────────────────────
# conversation.py does `from homeassistant.util import dt as dt_util` at module
# scope, so without this the whole module — and every test that imports it —
# fails COLLECTION, not just the assertions that need a clock. utcnow() is the
# only member the integration uses; keep it real rather than a MagicMock so a
# test comparing timestamps gets a datetime back.
_make_module(
    "homeassistant.util.dt",
    utcnow=lambda: _dt.datetime.now(_dt.timezone.utc),
    UTC=_dt.timezone.utc,
)

# homeassistant.util.logging — needed by pytest-homeassistant-custom-component
# autouse fixture `fail_on_log_exception`.
def _log_exception_noop(format_err, *args):  # noqa: ANN001
    pass

_make_module("homeassistant.util.logging", log_exception=_log_exception_noop)


def _slugify(text, *, separator="_"):  # noqa: ANN001
    """Close enough to HA's slugify for tests: apostrophes vanish, the rest separates."""
    import re as _re  # noqa: PLC0415

    text = _re.sub(r"['’]", "", str(text or "").lower())
    return _re.sub(r"[^a-z0-9]+", separator, text).strip(separator) or "unknown"


_util_mod = _make_module(
    "homeassistant.util",
    ulid=sys.modules["homeassistant.util.ulid"],
    dt=sys.modules["homeassistant.util.dt"],
    logging=sys.modules["homeassistant.util.logging"],
    slugify=_slugify,
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
