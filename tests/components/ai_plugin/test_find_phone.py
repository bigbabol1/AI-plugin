"""Tests for "where is my phone" (shortcuts.phone and its orchestrator wiring).

Covers the per-language request patterns, finding a person's Companion-app
phones through person → tracker → mobile_app entry, the Android and iOS
ring payloads, the repeat rings, whose phone rings (named, speaker, the
only phone in the home, or nobody), and the option gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.const import CONF_FIND_PHONE
from custom_components.ai_plugin.orchestrator import Orchestrator
from custom_components.ai_plugin.shortcuts import phone as phone_mod
from custom_components.ai_plugin.shortcuts.phone import (
    RING_PHONE_SCHEMA,
    async_ring_phone,
    match_find_phone,
    person_for_user,
    phones_of,
    resolve_owner,
)


# ── stubs ─────────────────────────────────────────────────────────────────────


def _state(entity_id, state="home", **attrs):
    return SimpleNamespace(entity_id=entity_id, state=state, attributes=attrs)


BEN = "person.ben"
LENA = "person.lena"


class _Hass:
    """Two people, each with one Companion-app phone, plus a router tracker."""

    def __init__(
        self, *, pixel_state="home", pixel_attrs=None, services=None, failing=(), keep_tasks=False
    ):
        self._states = {
            BEN: _state(
                BEN, friendly_name="Ben Miller", user_id="u-ben",
                device_trackers=["device_tracker.bens_pixel", "device_tracker.router_pixel"],
            ),
            LENA: _state(
                LENA, friendly_name="Lena Miller", user_id="u-lena",
                device_trackers=["device_tracker.lenas_iphone"],
            ),
            "device_tracker.bens_pixel": _state(
                "device_tracker.bens_pixel", pixel_state,
                **(pixel_attrs if pixel_attrs is not None else {"in_zones": ["zone.home"]}),
            ),
            "device_tracker.lenas_iphone": _state("device_tracker.lenas_iphone", "home"),
        }
        self.states = SimpleNamespace(
            get=self._states.get,
            async_all=lambda domain: [
                s for s in self._states.values() if s.entity_id.startswith(domain + ".")
            ],
        )
        entries = {
            "e-pixel": SimpleNamespace(data={"device_name": "Ben's Pixel", "os_name": "Android"}),
            "e-iphone": SimpleNamespace(data={"device_name": "Lena’s iPhone", "os_name": "iOS"}),
        }
        self.config_entries = SimpleNamespace(async_get_entry=entries.get)
        known = services if services is not None else {"mobile_app_bens_pixel", "mobile_app_lenas_iphone"}
        self.calls: list[tuple[str, dict]] = []

        async def _call(domain, service, data, **kwargs):
            if service in failing:
                raise RuntimeError("push failed")
            self.calls.append((service, data))

        self.services = SimpleNamespace(
            has_service=lambda domain, service: domain == "notify" and service in known,
            async_call=_call,
        )
        self.tasks: list = []
        self._keep_tasks = keep_tasks

    def async_create_task(self, coro):
        """Record the background rings; run them only where a test awaits them."""
        if not self._keep_tasks:
            coro.close()
        self.tasks.append(coro)


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    trackers = {
        "device_tracker.bens_pixel": SimpleNamespace(platform="mobile_app", config_entry_id="e-pixel"),
        "device_tracker.lenas_iphone": SimpleNamespace(platform="mobile_app", config_entry_id="e-iphone"),
        "device_tracker.router_pixel": SimpleNamespace(platform="fritz", config_entry_id="e-fritz"),
    }
    reg = SimpleNamespace(async_get=trackers.get)
    monkeypatch.setattr(shortcuts.registry.er, "async_get", lambda hass: reg)


# ── the request patterns ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "msg,lang,owner",
    [
        ("Where is my phone?", "en", ""),
        ("Where's my phone", "en", ""),
        ("where did I leave my phone", "en", ""),
        ("Ring my phone.", "en", ""),
        ("I can't find my phone", "en", ""),
        ("Where's Lena's phone?", "en", "lena"),
        ("where is lenas phone", "en", "lenas"),
        ("Find Lena’s iPhone", "en", "lena"),
        ("Wo ist mein Handy?", "de", ""),
        ("Wo habe ich mein Handy hingelegt", "de", ""),
        ("Lass mein Handy klingeln", "de", ""),
        ("Ich finde mein Handy nicht", "de", ""),
        ("Wo ist das Handy von Lena?", "de", "lena"),
        ("Wo ist Lenas Handy?", "de", "lenas"),
        ("Où est mon téléphone ?", "fr", ""),
        ("Fais sonner mon portable", "fr", ""),
        ("Je ne trouve pas mon téléphone", "fr", ""),
        ("Où est le téléphone de Lena", "fr", "lena"),
        ("¿Dónde está mi móvil?", "es", ""),
        ("Waar is mijn telefoon?", "nl", ""),
        ("Gdzie jest mój telefon?", "pl", ""),
        ("Onde está o meu telemóvel?", "pt", ""),
    ],
)
def test_find_phone_requests_match(msg, lang, owner):
    assert match_find_phone(msg, lang) == owner


@pytest.mark.parametrize(
    "msg,lang",
    [
        ("Where is the phone charger?", "en"),
        ("Where is my car?", "en"),
        ("Call mom", "en"),
        ("Turn on the phone light", "en"),
        ("What is the battery of my phone", "en"),
        ("Wo ist mein Auto?", "de"),
        ("Wie voll ist mein Handy?", "de"),
        ("", "en"),
    ],
)
def test_other_requests_do_not_match(msg, lang):
    assert match_find_phone(msg, lang) is None


def test_a_my_word_captured_as_owner_means_the_speaker():
    # "wo ist mein handy" reaches the named pattern too; "mein" is no name.
    assert match_find_phone("wo liegt mein handy", "de") == ""


# ── finding the phones ────────────────────────────────────────────────────────


def test_phones_come_from_the_persons_companion_app_trackers():
    hass = _Hass()
    phones = phones_of(hass, BEN)
    assert [(p.service, p.apple) for p in phones] == [("mobile_app_bens_pixel", False)]
    assert [(p.service, p.apple) for p in phones_of(hass, LENA)] == [
        ("mobile_app_lenas_iphone", True)
    ]


def test_a_phone_without_a_notify_service_is_skipped():
    hass = _Hass(services={"mobile_app_lenas_iphone"})
    assert phones_of(hass, BEN) == []


def test_unknown_person_has_no_phones():
    assert phones_of(_Hass(), "person.nobody") == []


def test_person_for_user_and_voice_memory_key():
    hass = _Hass()
    assert person_for_user(hass, "u-lena") == LENA
    assert person_for_user(hass, "person_ben") == BEN
    assert person_for_user(hass, "person_ghost") is None
    assert person_for_user(hass, None) is None


# ── whose phone ───────────────────────────────────────────────────────────────


def _finder(name):
    return {"lena": LENA, "ben": BEN}.get(name.casefold())


def test_named_owner_wins_over_the_speaker():
    assert resolve_owner(_Hass(), "lena", speaker_person=BEN, find_person=_finder) == LENA


def test_possessive_is_stripped_when_the_name_alone_does_not_match():
    assert resolve_owner(_Hass(), "lena's", speaker_person=None, find_person=_finder) == LENA


def test_unknown_name_rings_nobody():
    assert resolve_owner(_Hass(), "bob", speaker_person=BEN, find_person=_finder) is None


def test_my_phone_is_the_speakers():
    assert resolve_owner(_Hass(), "", speaker_person=LENA, find_person=_finder) == LENA


def test_unknown_speaker_with_two_phones_in_the_home_rings_nobody():
    assert resolve_owner(_Hass(), "", speaker_person=None, find_person=_finder) is None


def test_unknown_speaker_with_one_phone_in_the_home_rings_it():
    hass = _Hass(services={"mobile_app_bens_pixel"})
    assert resolve_owner(hass, "", speaker_person=None, find_person=_finder) == BEN


# ── ringing ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_android_rings_through_the_alarm_stream():
    hass = _Hass()
    reply = await async_ring_phone(hass, BEN, "en", own=True)
    assert reply == "Ringing your phone."
    service, data = hass.calls[0]
    assert service == "mobile_app_bens_pixel"
    assert data["message"] == "TTS"
    assert data["data"]["media_stream"] == "alarm_stream_max"
    assert data["data"]["tts_text"] == "Here I am!"
    assert data["data"]["priority"] == "high" and data["data"]["ttl"] == 0


@pytest.mark.asyncio
async def test_iphone_rings_with_a_critical_alert():
    hass = _Hass()
    reply = await async_ring_phone(hass, LENA, "de", own=False)
    assert reply == "Ich lasse das Handy von Lena klingeln."
    service, data = hass.calls[0]
    assert service == "mobile_app_lenas_iphone"
    assert data["data"]["push"]["sound"] == {"name": "default", "critical": 1, "volume": 1.0}
    assert data["message"] == "Hier bin ich!"


@pytest.mark.asyncio
async def test_it_rings_again_in_the_background(monkeypatch):
    sleeps: list[float] = []

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(phone_mod.asyncio, "sleep", _sleep)
    hass = _Hass(keep_tasks=True)
    await async_ring_phone(hass, BEN, "en", own=True)
    assert len(hass.calls) == 1 and len(hass.tasks) == 1
    await hass.tasks[0]
    assert len(hass.calls) == phone_mod.RINGS
    assert sleeps == [phone_mod.RING_GAP_S] * (phone_mod.RINGS - 1)


@pytest.mark.asyncio
async def test_a_phone_seen_elsewhere_says_where():
    hass = _Hass(pixel_state="Office", pixel_attrs={"in_zones": ["zone.office"]})
    reply = await async_ring_phone(hass, BEN, "en", own=True)
    assert reply == "Ringing your phone. Home Assistant last saw it at Office."


@pytest.mark.asyncio
async def test_a_phone_away_from_any_zone_says_so():
    hass = _Hass(pixel_state="not_home", pixel_attrs={})
    reply = await async_ring_phone(hass, BEN, "en", own=True)
    assert reply.endswith("last saw it away from home.")


@pytest.mark.asyncio
async def test_a_phone_at_home_adds_nothing():
    hass = _Hass(pixel_state="unknown", pixel_attrs={})
    assert await async_ring_phone(hass, BEN, "en", own=True) == "Ringing your phone."


@pytest.mark.asyncio
async def test_a_failed_push_is_reported_not_announced():
    hass = _Hass(failing={"mobile_app_bens_pixel"})
    reply = await async_ring_phone(hass, BEN, "en", own=True)
    assert reply == "I couldn't reach the phone."
    assert hass.tasks == []


@pytest.mark.asyncio
async def test_a_person_without_a_phone_gets_none():
    hass = _Hass(services=set())
    assert await async_ring_phone(hass, BEN, "en", own=True) is None


# ── orchestrator wiring ───────────────────────────────────────────────────────


def _orch(enabled=True, speaker=None):
    orch = Orchestrator.__new__(Orchestrator)
    orch._hass = _Hass()
    orch._entry = MagicMock()
    orch._entry.options = {CONF_FIND_PHONE: enabled}
    orch._voice_id = SimpleNamespace(find_person=_finder)
    orch._turn_speakers = {"sat": speaker} if speaker is not None else {}
    return orch


@pytest.mark.asyncio
async def test_shortcut_rings_the_app_users_phone():
    orch = _orch()
    reply = await orch._find_phone_shortcut("", "u-lena", None, "en")
    assert reply == "Ringing your phone."
    assert orch._hass.calls[0][0] == "mobile_app_lenas_iphone"


@pytest.mark.asyncio
async def test_shortcut_rings_the_recognised_speakers_phone():
    import asyncio

    orch = _orch()
    identity = SimpleNamespace(person_id=BEN)
    task = asyncio.get_running_loop().create_future()
    task.set_result(identity)
    reply = await orch._find_phone_shortcut("", None, task, "en")
    assert reply == "Ringing your phone."
    assert orch._hass.calls[0][0] == "mobile_app_bens_pixel"


@pytest.mark.asyncio
async def test_shortcut_hands_an_unknown_speaker_to_the_llm():
    orch = _orch()
    assert await orch._find_phone_shortcut("", None, None, "en") is None
    assert orch._hass.calls == []


@pytest.mark.asyncio
async def test_tool_rings_the_speaker_recognised_this_turn():
    orch = _orch(speaker=LENA)
    result = await orch._ring_phone_tool({}, None, "sat", "en")
    assert "Ringing your phone." in result
    assert orch._hass.calls[0][0] == "mobile_app_lenas_iphone"


@pytest.mark.asyncio
async def test_tool_rings_a_named_person():
    orch = _orch(speaker=LENA)
    result = await orch._ring_phone_tool({"person": "Ben"}, None, "sat", "en")
    assert "Ringing Ben's phone." in result


@pytest.mark.asyncio
async def test_tool_asks_whose_phone_when_nobody_is_known():
    orch = _orch()
    result = await orch._ring_phone_tool({}, None, "sat", "en")
    assert result.startswith("[No phone rang.")
    assert "Ben Miller, Lena Miller" in result
    assert orch._hass.calls == []


def test_tool_schema_is_offered_only_with_the_option_on():
    def names(orch):
        orch._mcp = orch._web_search = orch._browse_url = orch._memory = orch._ha_local = None
        orch._prune_schemas = False
        return [s["function"]["name"] for s in orch._tool_schemas_for("hi", False, None)]

    assert RING_PHONE_SCHEMA["function"]["name"] in names(_orch(enabled=True))
    assert names(_orch(enabled=False)) == []


@pytest.mark.asyncio
async def test_option_off_leaves_the_turn_to_the_llm():
    """With the option off a find-my-phone turn rings nothing and reaches the LLM."""
    from custom_components.ai_plugin.context_manager import ContextManager

    orch = _orch(enabled=False)
    orch._voice_id = None
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False
    orch._prune_schemas = False
    orch._bg_tasks = set()
    orch._mcp = orch._memory = orch._web_search = orch._browse_url = orch._ha_local = None
    orch._max_tool_iterations = 5
    orch._last_entities = {}
    orch._conv_locks = {}
    orch._location = SimpleNamespace(async_resolve=AsyncMock(return_value={}))
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value="No idea.")
    core = "custom_components.ai_plugin.orchestrator.core"
    with patch(f"{core}.try_shortcut", return_value=None), \
         patch(f"{core}.async_try_media_shortcut", AsyncMock(return_value=None)), \
         patch(f"{core}.async_try_domain_sweep_shortcut", AsyncMock(return_value=None)), \
         patch(f"{core}.async_try_action_shortcut", AsyncMock(return_value=None)):
        reply = await orch.async_process("Where is my phone?", "c1", "en")
    assert reply == "No idea."
    assert orch._hass.calls == []
