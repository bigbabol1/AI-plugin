"""Tests for result-gated media TTS suppression (orchestrator._any_media_success).

A media tool CALL alone must not blank the reply — only a call whose result
shows playback actually changed ("OK" prefix). Failed calls (unknown command)
and the read-only 'status' command leave the model's spoken answer intact;
otherwise media QUESTIONS like "what's playing?" end in dead silence.
"""

from __future__ import annotations

from custom_components.ai_plugin.orchestrator import _any_media_success


def _call_msg(name: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _result_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_successful_media_command_suppresses() -> None:
    msgs = [
        _call_msg("media_command", "c1"),
        _result_msg("c1", "OK — pause on media_player.wohnzimmer."),
    ]
    assert _any_media_success(msgs) is True


def test_successful_play_music_suppresses() -> None:
    msgs = [
        _call_msg("play_music", "c1"),
        _result_msg("c1", "OK — playing 'Enya' (artist) on media_player.kueche."),
    ]
    assert _any_media_success(msgs) is True


def test_failed_media_command_does_not_suppress() -> None:
    """The live 'what's playing?' bug: model probes an invalid command, the
    tool rejects it, the model recovers with a spoken answer — that answer
    must survive."""
    msgs = [
        _call_msg("media_command", "c1"),
        _result_msg(
            "c1", "Unknown media command 'status'. Use: next, pause, previous, resume, stop."
        ),
    ]
    assert _any_media_success(msgs) is False


def test_status_result_does_not_suppress() -> None:
    msgs = [
        _call_msg("media_command", "c1"),
        _result_msg("c1", "Now playing: 'Orinoco Flow' by Enya on Wohnzimmer."),
    ]
    assert _any_media_success(msgs) is False


def test_nothing_playing_result_does_not_suppress() -> None:
    msgs = [
        _call_msg("media_command", "c1"),
        _result_msg("c1", "Nothing is playing right now."),
    ]
    assert _any_media_success(msgs) is False


def test_ok_from_non_media_tool_ignored() -> None:
    """An 'OK'-prefixed result from a NON-media tool must not count."""
    msgs = [
        _call_msg("set_area_state", "c1"),
        _result_msg("c1", "OK — turned off 3 lights in Kitchen."),
    ]
    assert _any_media_success(msgs) is False


def test_mixed_failed_then_successful_call_suppresses() -> None:
    msgs = [
        _call_msg("media_command", "c1"),
        _result_msg("c1", "Unknown media command 'skip'. Use: next, pause, previous, resume, stop."),
        _call_msg("media_command", "c2"),
        _result_msg("c2", "OK — next on media_player.wohnzimmer."),
    ]
    assert _any_media_success(msgs) is True


def test_no_tool_msgs() -> None:
    assert _any_media_success([]) is False


# ── media_command('status') counts as grounding for state-set queries ────────


def test_status_call_detected() -> None:
    from custom_components.ai_plugin.orchestrator import _any_media_status_call

    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "media_command",
                        "arguments": '{"command": "status"}',
                    },
                }
            ],
        },
        _result_msg("c1", "Nothing is playing right now."),
    ]
    assert _any_media_status_call(msgs) is True


def test_status_call_dict_args_detected() -> None:
    from custom_components.ai_plugin.orchestrator import _any_media_status_call

    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "media_command",
                        "arguments": {"command": "status"},
                    },
                }
            ],
        },
    ]
    assert _any_media_status_call(msgs) is True


def test_non_status_media_call_not_detected() -> None:
    from custom_components.ai_plugin.orchestrator import _any_media_status_call

    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "media_command",
                        "arguments": '{"command": "pause"}',
                    },
                }
            ],
        },
    ]
    assert _any_media_status_call(msgs) is False


def test_malformed_args_not_detected() -> None:
    from custom_components.ai_plugin.orchestrator import _any_media_status_call

    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "media_command", "arguments": "not json"},
                }
            ],
        },
    ]
    assert _any_media_status_call(msgs) is False
