"""The sidebar page's backend logic: secrets stay home, MCP edits merge."""

from __future__ import annotations

from custom_components.ai_plugin.activity import ActivityLog
from custom_components.ai_plugin.panel.logic import (
    clean_values,
    form_fields,
    mask_servers,
    memory_owner,
    merge_servers,
    preset_server,
)


def test_form_fields_hide_secrets_and_use_suggested_values() -> None:
    serialized = [
        {"name": "api_key", "default": "sk-secret", "selector": {"text": {"type": "password"}}},
        {"name": "temperature", "description": {"suggested_value": 0.4}, "selector": {"number": {}}},
        {"name": "voice_mode", "default": True, "selector": {"boolean": {}}},
    ]
    fields, values = form_fields(
        serialized, {"api_key": "sk-secret"}, {"voice_mode": "Voice mode"}, {"voice_mode": "Short replies"}
    )
    assert values == {"temperature": 0.4, "voice_mode": True}
    secret = fields[0]
    assert secret["secret"] is True and secret["is_set"] is True
    assert "default" not in secret and "sk-secret" not in repr(fields)
    assert fields[2]["label"] == "Voice mode" and fields[2]["help"] == "Short replies"
    assert fields[1]["label"] == "Temperature"


def test_clean_values_keeps_empty_secret_as_remove() -> None:
    assert clean_values({"api_key": "", "keep_alive": "", "top_p": None, "model": "m"}) == {
        "api_key": "",
        "model": "m",
    }


def test_mask_servers_never_shows_tokens_or_env_values() -> None:
    masked = mask_servers([
        {"transport": "http", "url": "https://tools.example/mcp", "token": "t0ken"},
        {"transport": "stdio", "command": "uvx", "args": ["srv"], "env": {"API_KEY": "v4lue"}},
    ])
    assert "t0ken" not in repr(masked) and "v4lue" not in repr(masked)
    assert masked[0]["token_set"] is True
    assert masked[1]["env"] == [{"name": "API_KEY", "is_set": True}]


def test_merge_servers_keeps_secrets_unless_replaced_or_cleared() -> None:
    old = [
        {"transport": "http", "url": "https://a.example/mcp", "token": "old"},
        {"transport": "stdio", "command": "uvx", "args": ["x"], "env": {"KEY": "secret", "OTHER": "o"}},
    ]
    incoming = [
        {"index": 1, "transport": "stdio", "command": "uvx", "args": ["x", "--y"],
         "env": [{"name": "KEY"}, {"name": "RENAMED", "from": "OTHER"}, {"name": "NEW", "value": "n"}]},
        {"index": 0, "transport": "http", "url": "https://a.example/mcp"},
        {"index": None, "transport": "http", "url": "https://b.example/mcp", "token": ""},
    ]
    servers, errors = merge_servers(old, incoming)
    assert errors == []
    assert servers[0] == {
        "transport": "stdio", "command": "uvx", "args": ["x", "--y"],
        "env": {"KEY": "secret", "RENAMED": "o", "NEW": "n"},
    }
    assert servers[1]["token"] == "old"
    assert "token" not in servers[2]

    cleared, _ = merge_servers(old, [{"index": 0, "transport": "http", "url": "https://a.example/mcp", "token": ""}])
    assert "token" not in cleared[0]


def test_merge_servers_reports_bad_entries() -> None:
    _servers, errors = merge_servers([], [
        {"transport": "http", "url": "not a url"},
        {"transport": "stdio", "command": " "},
    ])
    assert errors == [
        {"index": 0, "field": "url", "error": "invalid_url"},
        {"index": 1, "field": "command", "error": "command_required"},
    ]


def test_preset_server_fills_the_value() -> None:
    preset = {"command": "uvx", "args_template": ["db", "--path", "{value}"]}
    assert preset_server(preset, "") is None
    assert preset_server(preset, "/config/x.db")["args"] == ["db", "--path", "/config/x.db"]
    assert preset_server({"command": "uvx", "args": ["t"]}, None)["args"] == ["t"]


def test_memory_owner_names() -> None:
    persons = {"person.robin": {"name": "Robin", "user_id": "u1"}, "person.kim": {"name": "Kim", "user_id": None}}
    assert memory_owner("u1", persons, {}) == "Robin"
    assert memory_owner("person_kim", persons, {}) == "Kim"
    assert memory_owner("anonymous", persons, {}) == "Unrecognised voices"
    assert memory_owner("u9", persons, {"u9": "Guest"}) == "Guest"


def test_activity_log_counts_and_caps() -> None:
    log = ActivityLog()
    for i in range(60):
        log.record(text=f"q{i}", reply="r", seconds=1.0, tools=["web_search"], safeguards=["empty_reply"] if i % 2 else [])
    log.record(text="x" * 1000, reply="", seconds=0.1, kind="echo")
    data = log.as_dict()
    assert data["counts"] == {"turns": 61, "answered": 60, "echo": 1}
    assert data["safeguards"] == {"empty_reply": 30}
    assert data["tools"] == {"web_search": 60}
    assert len(data["turns"]) == 50 and data["turns"][0]["kind"] == "echo"
    assert len(data["turns"][0]["text"]) == 400
    assert data["avg_seconds"] == 1.0
    assert data["backend_down"] is False


def test_mcp_server_status_in_config_order() -> None:
    from custom_components.ai_plugin.tools.mcp_client import MCPToolRegistry

    class _Builtin:
        ok = True
        connected = True
        tools = [object(), object()]

    configs = [
        {"transport": "http", "url": "http://localhost:8123/api/mcp"},
        {"transport": "stdio", "command": "uvx", "args": ["srv"]},
    ]
    registry = MCPToolRegistry(configs, builtin=_Builtin(), own_hosts=frozenset({"localhost"}))
    assert registry.server_status() == [
        {"state": "builtin", "tools": 0},
        {"state": "disconnected", "tools": 0},
    ]
    assert registry.builtin_tool_count() == 2



def test_edit_facts_add_replace_remove() -> None:
    from custom_components.ai_plugin.panel.logic import edit_facts, valid_memory_key

    facts = ["likes tea", "has a cat"]
    assert edit_facts(facts, None, None, "  owns   a bike ") == ["likes tea", "has a cat", "owns a bike"]
    assert edit_facts(facts, None, None, "Likes Tea") is None  # duplicate
    assert edit_facts(facts, None, None, "   ") is None
    assert edit_facts(facts, 1, "has a cat", "has two cats") == ["likes tea", "has two cats"]
    assert edit_facts(facts, 1, "has a dog", "x") is None  # changed meanwhile
    assert edit_facts(facts, 0, "likes tea", "likes tea") == facts  # same text, own slot
    assert edit_facts(facts, 0, "likes tea", None) == ["has a cat"]
    assert edit_facts(facts, 5, "likes tea", None) is None
    assert facts == ["likes tea", "has a cat"]
    assert valid_memory_key("person_robin") and valid_memory_key("anonymous")
    assert not valid_memory_key("../secrets") and not valid_memory_key("")


async def test_diary_edit_changes_text_and_refuses_stale() -> None:
    from custom_components.ai_plugin.diary import DiaryManager

    diary = DiaryManager(None, "entry")
    diary._entries = [{"id": "n1", "owner": None, "text": "Asked about the bus to the lake."}]
    diary._vectors = {"n1": object()}
    assert await diary.async_edit("n1", "  Asked about   the ferry. ", expected="Asked about the bus to the lake.")
    assert diary.entries()[0]["text"] == "Asked about the ferry."
    assert "n1" not in diary._vectors  # no model loaded: the stale vector goes
    assert not await diary.async_edit("n1", "x", expected="Asked about the bus to the lake.")
    assert not await diary.async_edit("n1", "   ")
    assert not await diary.async_edit("nope", "x")


def test_sentence_trigger_turns_reach_the_feed() -> None:
    from custom_components.ai_plugin.activity import SentenceTriggerWatch

    log = ActivityLog()
    log.record(text="weather?", reply="Mild.", seconds=1.0, device="dev1", conversation_id="c1")
    watch = SentenceTriggerWatch(log, owns=lambda cid: True)

    watch("c1", "content_added", {"content": {"role": "user", "content": "I'm leaving."}})
    # What Home Assistant logs between: the trigger as an external tool call.
    watch("c1", "content_added", {"content": {"role": "assistant", "agent_id": "conversation.ai_plugin", "tool_calls": ["trigger_sentence"]}})
    watch("c1", "content_added", {"content": {"role": "tool_result", "agent_id": "conversation.ai_plugin", "tool_name": "trigger_sentence"}})
    watch("c1", "content_added", {"content": {"role": "assistant", "agent_id": "sentence_trigger", "content": "Lights are off."}})
    turn = log.as_dict()["turns"][0]
    assert (turn["text"], turn["reply"], turn["kind"], turn["device"]) == ("I'm leaving.", "Lights are off.", "automation", "dev1")

    # A reply from the model itself is the conversation agent's turn: not doubled.
    watch("c2", "content_added", {"content": {"role": "user", "content": "hi"}})
    watch("c2", "content_added", {"content": {"role": "assistant", "agent_id": "conversation.ai_plugin", "content": "Hello"}})
    # Another entry's conversation is left to that entry.
    other = SentenceTriggerWatch(log, owns=lambda cid: False)
    other("c3", "content_added", {"content": {"role": "user", "content": "x"}})
    other("c3", "content_added", {"content": {"role": "assistant", "agent_id": "sentence_trigger", "content": "y"}})
    # Malformed data never raises.
    watch("c4", "content_added", None)
    assert log.as_dict()["counts"] == {"turns": 2, "answered": 1, "automation": 1}
