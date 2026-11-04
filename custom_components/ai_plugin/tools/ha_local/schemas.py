"""The tool schemas the model sees, and the names it may call.

One list, built in three pieces for readability only. The descriptions
are load-bearing: they are the whole of what a small model knows about
when to call which tool.
"""

from __future__ import annotations

from typing import Any

from .climate import _HVAC_MODES
from .lights import _ACTION_DOMAINS, _BRIGHTNESS_FLOOR_PCT, _BRIGHTNESS_STEP_PP
from .timers import _TIMER_INTENTS


# Offered on the two tools that read one entity or list a domain, which is all
# the prompt's rules ask of it. The handlers still honour it if the model sends it.
_EXPOSED_ONLY_PROP = {
    "exposed_only": {
        "type": "boolean",
        "description": (
            "Only include entities exposed to the conversation assistant. "
            "Set to false to inspect hidden or diagnostic entities. Default true."
        ),
    },
}


# Tools of Home Assistant's own MCP server that a local tool here already does,
# left out of what the model sees. Each one costs tokens on every prompt read,
# and the local one handles more: media_command narrows to the caller's room,
# set_brightness steps and takes white shades, and HassCancelAllTimers cancels
# nothing, because HA skips the announce-mode timers this plugin starts.
# GetDateTime repeats [CURRENT TIME], which every prompt carries.
TOOLS_THE_PLUGIN_COVERS = frozenset({
    "GetDateTime",
    "HassCancelAllTimers",
    "HassLightSet",
    "HassMediaNext",
    "HassMediaPause",
    "HassMediaPlayerMute",
    "HassMediaPlayerUnmute",
    "HassMediaPrevious",
    "HassMediaUnpause",
    "HassSetVolume",
    "HassSetVolumeRelative",
})


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_areas",
            "description": (
                "List every area (room) defined in Home Assistant. Call this "
                "to answer 'what rooms do you have' or to discover valid area "
                "names for use with list_entities."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List entities in Home Assistant. Optionally filter by area "
                "(e.g. 'hobbyroom'), domain (e.g. 'light', 'switch', "
                "'sensor'), and/or state (e.g. 'on', 'off', 'home'). Call "
                "this to answer 'which lights can you see', 'what is in the "
                "kitchen', 'list all switches', 'which lights are on', "
                "'any windows open?'. Each row includes the live state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Optional area name to filter by.",
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "Optional HA domain to filter by (light, switch, "
                            "sensor, climate, media_player, cover, etc.)."
                        ),
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "Optional state filter (case-insensitive). "
                            "Examples: 'on', 'off', 'home', 'open', 'unavailable'. "
                            "Use this to answer 'which lights are on', "
                            "'any windows open', 'welche Lichter sind an'."
                        ),
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": (
                "Get the current state, area, and key attributes of a single "
                "entity. Accepts an entity_id (e.g. 'light.smartbulb') or a "
                "user-facing name / alias. Call this to answer 'is X on', "
                "'what temperature in Y', 'in which room is Z', 'what is the "
                "brightness of the smart bulb'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_id": {
                        "type": "string",
                        "description": "Entity id, friendly name, or alias.",
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["name_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": (
                "Substring search across entity_id, friendly_name, and aliases. "
                "Use when the user refers to a device by partial or fuzzy name "
                "and you need to find its canonical entity_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to match (case-insensitive).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_area_state",
            "description": (
                "Turn devices in an area on/off/toggle. Use for commands like "
                "'lights in kitchen off', 'fans in bedroom on'. For a single "
                "named device use HassTurnOn/HassTurnOff instead. "
                "Pass area: 'all' whenever the user says 'all', 'every', "
                "'everywhere' or 'whole house' ('all lights off' → "
                "area: 'all'). Pass the room name when the user names one. "
                "OMIT area ONLY when the user named neither — the plugin "
                "then defaults to the calling satellite's room."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": (
                            "Area name when the user names a specific room "
                            "(e.g. 'kitchen', 'bedroom'). Pass 'all' when the "
                            "user says 'all', 'every', 'everywhere' or 'whole "
                            "house' — REQUIRED for whole-home commands, since "
                            "an omitted area means 'this room only'. OMIT only "
                            "when the user named neither a room nor 'all'."
                        ),
                    },
                    "domain": {
                        "type": "string",
                        "enum": sorted(_ACTION_DOMAINS),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "toggle"],
                    },
                },
                "required": ["domain", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Search and play music on a speaker via Music Assistant. "
                "Use for music-playback requests like 'play Enya', 'play "
                "jazz', 'shuffle my workout playlist', 'spiel jazz im "
                "wohnzimmer'. If the user names a room, pass that exact "
                "room as area. If the user does NOT name a room, OMIT "
                "the area parameter entirely — the plugin will fall back "
                "to the area of the satellite that received the request. "
                "Do NOT guess or default to a specific room when none "
                "was mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to play — track, album, artist, playlist, "
                            "or genre. Free text. Examples: 'Enya', 'Hotel "
                            "California', 'jazz', 'workout playlist'. Never just "
                            "'music': for 'play some music' give the genres of "
                            "their taste in [USER FACTS], comma separated, with "
                            "from_taste, or ask what they would like to hear."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Area name when the user explicitly names a "
                            "room (e.g. 'kitchen', 'living room', "
                            "'wohnzimmer'). OMIT when the user did not "
                            "specify a room — the plugin defaults to "
                            "the calling satellite's own area."
                        ),
                    },
                    "from_taste": {
                        "type": "boolean",
                        "description": (
                            "True only when the user named no music and query is "
                            "their taste: 'jazz, soul'. It picks a different "
                            "playlist each time and shuffles it."
                        ),
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["track", "album", "artist", "playlist", "radio"],
                        "description": "Optional. Default 'track'.",
                    },
                    "radio_mode": {
                        "type": "boolean",
                        "description": (
                            "Optional. Default auto. When true, Music Assistant "
                            "creates a dynamic radio/playlist that continues playing "
                            "similar tracks after the requested item finishes. "
                            "Best for single-track or artist requests ('play Hotel "
                            "California', 'play Enya'). "
                            "When false, playback stops after the playlist/album/artist "
                            "ends. Best for explicit playlists/albums. "
                            "Auto: true for 'track', false for 'album'/'artist'/"
                            "'playlist'/'radio'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_command",
            "description": (
                "Control playback or report it: pause, resume, skip to next "
                "track, go back to previous track, stop, volume, mute, or "
                "status. Examples: 'pause' / 'pause the music' → "
                "media_command('pause'). 'next track' / 'skip this song' → "
                "media_command('next'). 'previous' → media_command"
                "('previous'). 'resume' / 'continue' / 'weiter' → "
                "media_command('resume'). 'stop the music' → media_command"
                "('stop'). 'louder' / 'lauter' → media_command('volume_up'). "
                "'quieter' / 'leiser' → media_command('volume_down'). "
                "'volume to 40 percent' → media_command('volume_set', "
                "level: 40). 'mute' / 'stumm' → media_command('mute'). "
                "'what's playing?' / 'was läuft?' / 'what song is this?' → "
                "media_command('status'). Pass area only when the user names "
                "a room ('pause hobby room', 'next track in the kitchen') and "
                "name when they name a device ('mute the TV' → media_command"
                "('mute', name: 'TV')); "
                "otherwise the plugin auto-targets whichever media_player is "
                "currently in the matching state. For 'status' answer the "
                "user with what the tool returns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "pause", "resume", "next", "previous", "stop",
                            "volume_up", "volume_down", "volume_set",
                            "mute", "unmute", "status",
                        ],
                        "description": (
                            "Playback action. 'resume' un-pauses; 'next' / "
                            "'previous' move within the current queue; 'stop' "
                            "halts and clears; 'volume_set' needs level; "
                            "'status' is read-only and reports what is "
                            "currently playing."
                        ),
                    },
                    "level": {
                        "type": "integer",
                        "description": (
                            "Target volume percent 0-100. Required for "
                            "'volume_set', ignored otherwise."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Optional area name (e.g. 'hobby room'). Omit for "
                            "global commands like bare 'pause' or 'next "
                            "track' — the plugin will find the currently-"
                            "playing speaker automatically."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": "The device, when the user named one.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


_TIMER_DURATION_PROPS = {
    "hours": {
        "type": "integer",
        "description": "Hours portion of the duration (optional, default 0).",
    },
    "minutes": {
        "type": "integer",
        "description": "Minutes portion of the duration (optional, default 0).",
    },
    "seconds": {
        "type": "integer",
        "description": "Seconds portion of the duration (optional, default 0).",
    },
}


TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "set_brightness",
            "description": (
                "Make lights brighter or dimmer, set an absolute "
                "brightness, or change their colour or white shade. Examples: "
                "'brighter' / 'heller' / 'make it brighter in the living "
                "room' -> set_brightness('brighter', area: 'living room'). "
                "'dimmer' / 'dunkler' / 'dim the lights' -> "
                "set_brightness('dimmer'). 'set the lights to 40 percent' "
                "-> set_brightness('set', level: 40). 'make the light warm "
                "white' / 'Licht warmweiß' -> set_brightness('set', "
                "color: 'warm white'). 'turn the lights blue' -> "
                "set_brightness('set', color: 'blue'). Each brighter/dimmer "
                f"step is {_BRIGHTNESS_STEP_PP} percentage points and "
                f"'dimmer' stops at {_BRIGHTNESS_FLOOR_PCT}% — it NEVER "
                "turns lights off. To actually switch lights off use "
                "HassTurnOff or set_area_state, never 'dim'. Pass area "
                "when the user names a room, name for one specific lamp, "
                "and OMIT both for the calling satellite's own room."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["brighter", "dimmer", "set"],
                    },
                    "level": {
                        "type": "integer",
                        "description": (
                            "Target brightness percent 1-100. Required for "
                            "'set', ignored for brighter/dimmer."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Area name when the user names a room. Pass "
                            "'all' for whole-home. OMIT when the user named "
                            "neither — defaults to the caller's room."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional single lamp name or entity_id, when "
                            "the user named one device rather than a room."
                        ),
                    },
                    "color": {
                        "type": "string",
                        "description": (
                            "Colour or white shade in English, when the "
                            "user asks for one: 'red', 'light blue', 'warm "
                            "white', 'neutral white', 'cool white', "
                            "'daylight'. With 'set' and no level only the "
                            "colour changes and brightness stays."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    }
)


TOOL_SCHEMAS.extend(
    [
        {
            "type": "function",
            "function": {
                "name": "start_timer",
                "description": (
                    "Start a voice timer on the calling assist_satellite device. "
                    "It only rings when it ends and does nothing else. "
                    "Use for 'set a 5 minute timer', 'timer for 10 minutes called pasta', "
                    "'wecker auf 3 minuten'. Provide at least one of hours / minutes / "
                    "seconds. The optional name lets the user reference this specific "
                    "timer later (cancel, status). When the timer ends, the satellite "
                    "fires its on_timer_finished hook which speaks the announcement on "
                    "the configured Mic2MP speaker."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional human label for this timer (e.g. 'pasta', "
                                "'eggs', 'workout'). Used by the user to refer to the "
                                "timer later."
                            ),
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": (
                    "Cancel a running voice timer on the calling assist_satellite. "
                    "Provide the timer name if multiple are running; omit to cancel "
                    "the only / most recent one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to cancel.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pause_timer",
                "description": "Pause a running voice timer on the calling satellite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to pause.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "unpause_timer",
                "description": "Resume a paused voice timer on the calling satellite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to resume.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "increase_timer",
                "description": (
                    "Add time to a running voice timer (e.g. 'add 2 minutes to the "
                    "pasta timer')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to extend.",
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decrease_timer",
                "description": (
                    "Subtract time from a running voice timer (e.g. 'take 30 seconds "
                    "off the eggs timer')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to shorten.",
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timer_status",
                "description": (
                    "Report the time left on running voice timers "
                    "('how much time on the pasta timer', 'wie lange noch')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to report on.",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]
)


# Last on purpose: placed next to set_area_state it shifted the voice
# model's tool choice on unrelated turns (a repeated "turn off the
# kitchen light" was answered from history without a call, 12 of 12
# replays); appended after the timers it did not.
TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "set_climate_mode",
            "description": (
                "Switch the heating, a thermostat or an air conditioner to "
                "an operating mode. 'auto' means it follows its own "
                "schedule: 'heating back to auto', 'back on schedule', "
                "'Heizung auf Automatik'. 'heat' is manual heating, 'off' "
                "switches it off. For a target temperature use "
                "HassClimateSetTemperature instead; never send a "
                "temperature to change the mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(_HVAC_MODES),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Thermostat name or entity_id, when the user "
                            "named one."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Room name, when the user named one. OMIT both "
                            "name and area for 'the heating'."
                        ),
                    },
                },
                "required": ["mode"],
            },
        },
    }
)



# Sent last, after everything else: see Orchestrator._tool_schemas_for.
MOVE_MUSIC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "move_music",
        "description": (
            "Move what is already playing to another room: 'move the music to the "
            "bedroom', 'take this to the kitchen'. To start something new use play_music."
        ),
        "parameters": {
            "type": "object",
            "properties": {"area": {"type": "string", "description": "The room to move it to."}},
            "required": ["area"],
        },
    },
}


TOOL_NAMES = {
    "list_areas",
    "list_entities",
    "get_entity",
    "search_entities",
    "set_area_state",
    "set_brightness",
    "set_climate_mode",
    "play_music",
    "media_command",
    *_TIMER_INTENTS.keys(),
}
