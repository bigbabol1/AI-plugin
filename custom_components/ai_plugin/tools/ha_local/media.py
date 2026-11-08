"""Playing music, transport control, and what is playing now."""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)


from .formatting import _s

_LOGGER = logging.getLogger(__name__)

# ⚠ This set is hand-maintained and is NOT derived from TOOL_SCHEMAS.
# orchestrator._call_tool routes on `name in ha_local.tool_names`, so a
# tool added to TOOL_SCHEMAS but missed here is still SENT to the model,
# still called by it, then falls through to the MCP branch and fails —
# and the model quietly falls back to a different tool. Add every new
# tool in BOTH places.

_MUSIC_MEDIA_TYPES = {"track", "album", "artist", "playlist", "radio"}

_TELEVISION_WORDS = {"tv", "television", "fernseher", "télé", "tele", "televisie", "telewizor"}


_MEDIA_COMMAND_SERVICES = {
    "pause": "media_pause",
    "resume": "media_play",
    "next": "media_next_track",
    "previous": "media_previous_track",
    "stop": "media_stop",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "volume_set": "volume_set",
    "mute": "volume_mute",
    "unmute": "volume_mute",
}


# "Play music" names nothing to play, but Music Assistant searches for it anyway:
# the literal word found a country mix, every time, for someone who had saved
# electronic and pop as their taste (2026-11-01). Sent back for the model to fill
# in from [USER FACTS], or to ask.
_NOTHING_NAMED_RE = re.compile(
    r"^(?:(?:some|any|good|nice|great|the|my|current|etwas|irgendeine?|gute|de\s+la|"
    r"un\s+poco\s+de|alguma|jakąś|wat|een\s+beetje)\s+)*"
    r"(?:music|musik|musique|m[uú]sica|muziek|muzyk[aę]|songs?|tunes|lieder|"
    r"something|anything|was|etwas|irgendwas|quelque\s+chose|algo|coś|iets)"
    r"(?:\s+(?:else|different|nice|good|i\s+like|you\s+like|for\s+me|anderes|sch[öo]nes|"
    r"die\s+ich\s+mag|que\s+j[\x27’]aime|que\s+me\s+gusta))*$",
    re.IGNORECASE,
)
NOTHING_NAMED_RESULT = (
    "[Nothing played: no music was named. If [USER FACTS] gives their taste, call "
    "play_music again with its genres, comma separated, and from_taste: true. If it "
    "does not, ask what they would like to hear.]"
)
_TASTE_WORDS_RE = re.compile(
    r"\b(?:music|musik|musique|m[uú]sica|muziek|muzyk[aę]|playlists?)\b", re.IGNORECASE
)
_TASTE_PLAYLISTS = 6


def taste_genres(query: str) -> list[str]:
    """The genres in a taste: "electronic music and pop music playlists" gives
    electronic and pop."""
    parts = query.split(",") if "," in query else re.split(
        r"\s+(?:and|und|et|y|e|i|en)\s+", query, flags=re.IGNORECASE
    )
    genres = [" ".join(_TASTE_WORDS_RE.sub(" ", part).split()) for part in parts]
    return [genre for genre in genres if genre]


def names_nothing_to_play(query: str) -> bool:
    """True for "music", "some music", "something different": no actual choice."""
    return bool(_NOTHING_NAMED_RE.match(query.strip().rstrip(".!?")))


class MediaToolsMixin:
    """play_music / media_command / media_status."""

    async def _play_music(
        self,
        query: str,
        area: str | None,
        media_type: str = "track",
        radio_mode: bool | None = None,
        exposed_only: bool = True,
        device_id: str | None = None,
        from_taste: bool = False,
    ) -> str:
        """Search and play music in an area via Music Assistant.

        Replaces the long-broken ``set_area_state(media_player, turn_on)`` path:
        Music-Assistant entities don't support the generic turn_on action, and
        even when they did it would just power on the speaker without any
        media — which is what happens when the LLM "confirms" but nothing
        plays. This tool calls ``music_assistant.play_media`` with the user's
        free-text query, which MA resolves against its providers (Spotify,
        Tidal, local library, etc.).
        """
        query = (query or "").strip()
        if not query:
            return "[play_music needs a query — what should I play?]"
        genres = taste_genres(query) if from_taste else []
        if (from_taste and not genres) or (not from_taste and names_nothing_to_play(query)):
            return NOTHING_NAMED_RESULT

        # No area specified → resolve to the calling satellite's area.
        # Voice users in the living room saying 'play jazz' should not
        # have music start somewhere else just because the model picked
        # an example area name. With device_id, look up which area the
        # satellite is in and play there.
        if not area and device_id:
            try:
                ent_reg = er.async_get(self._hass)
                dev_reg = dr.async_get(self._hass)
                area_reg = ar.async_get(self._hass)
                # Get the device's area (either directly or via any of
                # its entities). Prefer device.area_id; fall back to
                # entity.area_id of the assist_satellite entity itself.
                dev = dev_reg.async_get(device_id)
                area_id = dev.area_id if dev else None
                if not area_id:
                    for entry in er.async_entries_for_device(ent_reg, device_id):
                        if entry.area_id:
                            area_id = entry.area_id
                            break
                if area_id:
                    a = area_reg.async_get_area(area_id)
                    if a:
                        area = a.name
                        _LOGGER.debug(
                            "play_music: no area specified, defaulted to "
                            "calling satellite's area %r (device_id=%s)",
                            area, device_id,
                        )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("play_music device→area lookup failed", exc_info=True)

        if not area:
            return (
                "[play_music needs an area — say which speaker, e.g. "
                "'in the kitchen' or 'in the living room'. Call "
                "list_areas to see options.]"
            )

        media_type = (media_type or "track").lower().strip()
        if media_type not in _MUSIC_MEDIA_TYPES:
            media_type = "track"

        if not self._hass.services.has_service("music_assistant", "play_media"):
            return (
                "[Music Assistant integration not installed. play_music "
                "requires Music Assistant to resolve free-text queries against "
                "music providers.]"
            )

        target_area = self._music_area(area)
        if target_area is None:
            return f"Unknown area {area!r}. Try list_areas."

        entity_id = self._music_player_in_area(target_area, exposed_only)
        if entity_id is None:
            scope = "exposed " if exposed_only else ""
            return f"No {scope}media_player in {target_area.name}."

        title = query
        if from_taste:
            query, title = await self._taste_playlist(genres)
            media_type = "playlist"
        await self._set_shuffle(entity_id, from_taste)

        try:
            _auto_radio = radio_mode if radio_mode is not None else (media_type == "track")
            await self._hass.services.async_call(
                "music_assistant",
                "play_media",
                target={"entity_id": entity_id},
                service_data={
                    "media_id": query,
                    "media_type": media_type,
                    "radio_mode": _auto_radio,
                },
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin play_music failed")
            return f"[play_music failed: {exc}]"

        return f"OK — playing {title!r} ({media_type}) on {entity_id}."

    async def _taste_playlist(self, genres: list[str]) -> tuple[str, str]:
        """A playlist for "play some music": (what to play, its name).

        A random genre of the taste, and a random playlist among the top results,
        not the one played last. Always the first result, it was the same music
        every time (2026-11-01). If the search fails, the genre itself is played.
        """
        genre = random.choice(genres)
        try:
            entries = self._hass.config_entries.async_entries("music_assistant")
            response = await self._hass.services.async_call(
                "music_assistant",
                "search",
                {
                    "config_entry_id": entries[0].entry_id,
                    "name": genre,
                    "media_type": ["playlist"],
                    "limit": _TASTE_PLAYLISTS,
                },
                blocking=True,
                return_response=True,
            )
            playlists = [p for p in (response or {}).get("playlists", []) if p.get("uri")]
        except Exception:  # noqa: BLE001 — the genre alone still plays something
            _LOGGER.debug("AI Plugin play_music: playlist search failed", exc_info=True)
            playlists = []
        last = getattr(self, "_last_taste_playlist", None)
        playlists = [p for p in playlists if p["uri"] != last] or playlists
        if not playlists:
            return genre, genre
        chosen = random.choice(playlists)
        self._last_taste_playlist = chosen["uri"]
        return chosen["uri"], str(chosen.get("name") or genre).strip()

    async def _set_shuffle(self, entity_id: str, shuffle: bool) -> None:
        """On for a playlist chosen from the taste. Off again only where this
        turned it on: an album asked for by name plays in order, but a speaker
        someone shuffles on purpose stays shuffled."""
        turned_on = self.__dict__.setdefault("_shuffled_by_us", set())
        state = self._hass.states.get(entity_id)
        attributes = getattr(state, "attributes", None) or {}
        was_on = attributes.get("shuffle") is True
        if not shuffle and entity_id not in turned_on:
            return
        try:
            await self._hass.services.async_call(
                "media_player",
                "shuffle_set",
                {"shuffle": shuffle},
                target={"entity_id": entity_id},
                blocking=True,
            )
        except Exception:  # noqa: BLE001 — never a reason not to play
            _LOGGER.debug("AI Plugin play_music: shuffle failed", exc_info=True)
            return
        # Sent even if it already reports on: a speaker reporting on went off as
        # the playlist started, so the first one played in order (2026-11-01).
        if not shuffle:
            turned_on.discard(entity_id)
        elif not was_on:
            turned_on.add(entity_id)

    def music_players_playing(self) -> list[str]:
        """The Music Assistant players playing now, by entity id."""
        return sorted(
            eid for eid, entry in er.async_get(self._hass).entities.items()
            if eid.startswith("media_player.")
            and entry.platform == "music_assistant"
            and (state := self._hass.states.get(eid)) is not None
            and state.state == "playing"
        )

    async def move_music(self, area: str, device_id: str | None = None) -> str:
        """Move what is playing to another room, through Music Assistant."""
        if not self._hass.services.has_service("music_assistant", "transfer_queue"):
            return "[move_music needs Music Assistant. Tell the user.]"
        from ...shortcuts.registry import _resolve_area  # noqa: PLC0415

        # Also by synonym: "Schlafzimmer" for an area called bedroom.
        target_area = self._music_area(area or "") or _resolve_area(self._hass, area or "")
        if target_area is None:
            return f"[Unknown area {area!r}. Try list_areas.]"
        target = self._music_player_in_area(target_area, True)
        if target is None:
            return f"[No speaker in {target_area.name}. Tell the user.]"
        playing = self.music_players_playing()
        if not playing:
            return "[Nothing is playing, so there is nothing to move. Ask what to play.]"
        if target in playing:
            return f"[It is already playing in {target_area.name}. Tell the user.]"
        # The members of a playing group report playing too, and the group holds
        # the queue: Big Speaker sorted ahead of its group Wohnzimmer (2026-11-08).
        groups = [
            eid for eid in playing
            if self._hass.states.get(eid).attributes.get("mass_player_type") == "group"
        ]
        source = (groups or playing)[0]
        try:
            await self._hass.services.async_call(
                "music_assistant",
                "transfer_queue",
                target={"entity_id": target},
                service_data={"source_player": source, "auto_play": True},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin move_music failed")
            return f"[move_music failed: {exc}]"
        return f"OK — moved the music from {source} to {target}."

    def _music_area(self, area: str):
        """The area with this exact name or alias, or None."""
        needle = area.strip().lower()
        for a in ar.async_get(self._hass).async_list_areas():
            if _s(a.name).lower() == needle:
                return a
            aliases = getattr(a, "aliases", None) or ()
            if any(_s(al).lower() == needle for al in aliases):
                return a
        return None

    def _music_player_in_area(self, target_area, exposed_only: bool) -> str | None:
        """The area's primary Music-Assistant media_player, or None."""
        # Filter by REGISTRY platform == "music_assistant" (stable; works
        # whether the speaker is on or off). The previous heuristic of
        # checking state.attributes.app_id only matched when the speaker
        # was already actively playing — players with state=off were
        # silently demoted to the fallback list, which often picked a
        # surprising speaker.
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)
        ma_native: list[tuple[str, str, list[str]]] = []
        other: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith("media_player."):
                continue
            if entry.disabled_by is not None or entry.hidden_by is not None:
                continue
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if area_id != target_area.id:
                continue
            if exposed_only and not self._is_exposed(eid):
                continue
            state = self._hass.states.get(eid)
            fn = _s(state.attributes.get("friendly_name") if state else "") or ""
            aliases = [_s(a) for a in (getattr(entry, "aliases", None) or ())]
            if entry.platform == "music_assistant":
                ma_native.append((eid, fn, aliases))
            else:
                other.append(eid)

        if not ma_native and not other:
            return None

        entity_id: str | None = None
        if ma_native:
            # Prefer the player whose friendly_name or alias matches the
            # area's name — that's almost always the user-blessed primary
            # speaker (e.g. media_player.wohnzimmer_2 with friendly_name
            # 'Wohnzimmer' in area 'living room' / alias 'wohnzimmer').
            area_keys = {_s(target_area.name).lower()}
            for al in (getattr(target_area, "aliases", None) or ()):
                area_keys.add(_s(al).lower())
            for eid, fn, aliases in ma_native:
                hay = {fn.lower()} | {a.lower() for a in aliases}
                if hay & area_keys:
                    entity_id = eid
                    break
            if entity_id is None:
                # Fallback: alphabetical first MA player.
                entity_id = sorted(eid for eid, _, _ in ma_native)[0]
        else:
            # No MA-platform player found — last-resort fallback to any
            # other media_player. play_media may not work via MA service
            # in this case, but the call will fail loudly.
            entity_id = sorted(other)[0]
        _LOGGER.debug(
            "play_music: target area=%r, ma_native=%d, other=%d, picked=%r",
            target_area.name, len(ma_native), len(other), entity_id,
        )
        return entity_id

    def _media_players_named(self, name: str, exposed_only: bool = True) -> list[str]:
        """Players called ``name``: exactly (name, alias or id), else with it as
        a word in a name. "The TV" also finds a player of class tv, whatever
        it is called."""
        needle = name.strip().casefold()
        exact: list[str] = []
        partial: list[str] = []
        televisions: list[str] = []
        for eid, entry in er.async_get(self._hass).entities.items():
            if not eid.startswith("media_player."):
                continue
            if exposed_only and not self._is_exposed(eid):
                continue
            state = self._hass.states.get(eid)
            names = [_s(entry.name), _s(entry.original_name)]
            names.extend(_s(alias) for alias in (entry.aliases or ()))
            if state:
                names.append(_s(state.attributes.get("friendly_name")))
            names = [n.strip().casefold() for n in names if n]
            if needle == eid or needle in names:
                exact.append(eid)
            elif any(re.search(rf"\b{re.escape(needle)}\b", n) for n in names):
                partial.append(eid)
            elif (
                needle in _TELEVISION_WORDS
                and state
                and state.attributes.get("device_class") == "tv"
            ):
                televisions.append(eid)
        return exact or partial or televisions

    def _media_players_in_area(
        self, target_area, exposed_only: bool = True
    ) -> list[str]:
        """Return media_player entity_ids assigned to ``target_area`` (directly
        or via the parent device)."""
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)
        out: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith("media_player."):
                continue
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if area_id != target_area.id:
                continue
            if exposed_only and not self._is_exposed(eid):
                continue
            out.append(eid)
        return out

    async def _media_command(
        self,
        command: str,
        area: str | None,
        exposed_only: bool = True,
        level: int | None = None,
        device_id: str | None = None,
        name: str | None = None,
    ) -> str:
        """Pause, resume, skip, stop, mute, or change volume.

        With ``name``: targets the player called that. Without a match it says
        so, because acting on whatever else is playing would be a silent wrong
        answer. A device passed as ``area`` counts as a name.

        With ``area``: targets media_players in that area, preferring those
        already in a state the command applies to (playing for pause/next/
        previous/stop, paused/idle for resume).

        Without ``area``: scans every media_player and acts on the ones
        currently in the relevant state. This covers terse phrasings like
        bare "pause" or "next track" where the LLM has no area to pass —
        the plugin finds whatever is actually playing right now.

        ``status`` is the read-only exception: it reports what is playing
        (media questions like "what's playing?") and never calls a service.
        """
        if command == "status":
            return self._media_status(area, exposed_only)
        if command not in _MEDIA_COMMAND_SERVICES:
            allowed = ", ".join(sorted([*_MEDIA_COMMAND_SERVICES, "status"]))
            return f"Unknown media command {command!r}. Use: {allowed}."

        data: dict[str, Any] = {}
        if command in ("mute", "unmute"):
            data["is_volume_muted"] = command == "mute"
        elif command == "volume_set":
            try:
                pct = int(level)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "[volume_set needs level — an integer 0-100.]"
            if not 0 <= pct <= 100:
                return "[volume_set level must be 0-100.]"
            data["volume_level"] = pct / 100

        # Prefer entities in a state the command actually applies to.
        relevant_states = {
            "pause": {"playing"},
            "resume": {"paused", "idle"},
            "next": {"playing", "paused"},
            "previous": {"playing", "paused"},
            "stop": {"playing", "paused"},
            "volume_up": {"playing", "paused", "on"},
            "volume_down": {"playing", "paused", "on"},
            "volume_set": {"playing", "paused", "on"},
            "mute": {"playing", "paused", "on"},
            "unmute": {"playing", "paused", "on"},
        }[command]

        named = self._media_players_named(name, exposed_only) if name else []
        if name and not named and not area:
            # "Unmute the bedroom": the model gave the room as the device.
            if self._resolve_area(name) is None:
                return f"[No media player is called {name!r}.]"
            area = name
        if not named and area and self._resolve_area(area) is None:
            named = self._media_players_named(area, exposed_only)
            if not named:
                return f"Unknown area {area!r}. Try list_areas."

        if named:
            target_ids = sorted(named)
        elif area:
            target_area = self._resolve_area(area)
            candidates = self._media_players_in_area(target_area, exposed_only)
            if not candidates:
                scope = "exposed " if exposed_only else ""
                return f"No {scope}media_player in {target_area.name}."
            active = [
                eid
                for eid in candidates
                if (st := self._hass.states.get(eid)) and st.state in relevant_states
            ]
            target_ids = sorted(active or candidates)
        else:
            # No area — find every media_player in the relevant state.
            ent_reg = er.async_get(self._hass)
            target_ids = []
            for eid in ent_reg.entities:
                if not eid.startswith("media_player."):
                    continue
                if exposed_only and not self._is_exposed(eid):
                    continue
                st = self._hass.states.get(eid)
                if st and st.state in relevant_states:
                    target_ids.append(eid)
            from ...shortcuts.media import narrow_media_targets  # noqa: PLC0415
            from ...shortcuts.registry import _caller_area_id  # noqa: PLC0415

            target_ids = narrow_media_targets(self._hass, target_ids, device_id, command)
            if not target_ids and command.startswith(("volume", "mute", "unmute")):
                # Nothing is playing, and the volume can still be set for later.
                caller_area = ar.async_get(self._hass).async_get_area(
                    _caller_area_id(self._hass, device_id) or ""
                )
                if caller_area is not None:
                    target_ids = self._media_players_in_area(caller_area, exposed_only)
            if command == "resume" and not target_ids:
                return "[Nothing is paused, so there is nothing to continue. Ask what to play.]"
            if not target_ids:
                state_label = " / ".join(sorted(relevant_states))
                return (
                    f"Nothing is {state_label} right now — pass an area to "
                    f"target a specific speaker."
                )
            target_ids.sort()

        service = _MEDIA_COMMAND_SERVICES[command]
        try:
            if command in ("volume_up", "volume_down"):
                from ...shortcuts.volume import async_step_volume  # noqa: PLC0415

                await async_step_volume(self._hass, target_ids, command)
            else:
                await self._hass.services.async_call(
                    "media_player",
                    service,
                    data or None,
                    target={"entity_id": target_ids},
                    blocking=True,
                )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin media_command %s failed", command)
            return f"[media_command {command} failed: {exc}]"

        preview = ", ".join(target_ids[:3])
        more = f" (+{len(target_ids) - 3} more)" if len(target_ids) > 3 else ""
        return f"OK — {command} on {preview}{more}."

    def _media_status(self, area: str | None, exposed_only: bool = True) -> str:
        """Report what is currently playing (``media_command('status')``).

        Read-only: never calls a service. The result must NOT carry the
        "OK" success prefix — the orchestrator suppresses TTS on "OK"
        media results, and a status answer is exactly what the user asked
        to hear.
        """
        if area:
            target_area = self._resolve_area(area)
            if target_area is None:
                return f"Unknown area {area!r}. Try list_areas."
            candidates = self._media_players_in_area(target_area, exposed_only)
        else:
            ent_reg = er.async_get(self._hass)
            candidates = [
                eid
                for eid in ent_reg.entities
                if eid.startswith("media_player.")
                and (not exposed_only or self._is_exposed(eid))
            ]

        playing: list[str] = []
        paused: list[str] = []
        for eid in sorted(candidates):
            st = self._hass.states.get(eid)
            if st is None or st.state not in ("playing", "paused"):
                continue
            name = st.attributes.get("friendly_name") or eid
            title = st.attributes.get("media_title")
            artist = st.attributes.get("media_artist")
            if title and artist:
                desc = f"{title!r} by {artist}"
            elif title:
                desc = f"{title!r}"
            else:
                desc = "unknown media"
            (playing if st.state == "playing" else paused).append(
                f"{desc} on {name}"
            )

        parts: list[str] = []
        if playing:
            parts.append("Now playing: " + "; ".join(playing[:4]))
        if paused:
            parts.append("Paused: " + "; ".join(paused[:4]))
        if not parts:
            return "Nothing is playing right now."
        return ". ".join(parts) + "."
