"""Alarms, reminders and commands for later.

Home Assistant's voice timers live in memory and only count down, so a restart
loses them and they cannot do anything when they end. Jobs here are stored, so
they survive a restart, and each one does its own thing when it is due:

- an alarm speaks on a satellite again and again, until someone stops or snoozes it
- a reminder speaks once on a satellite and goes to the person's phone
- a command runs through the normal conversation path, as if said right then
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import logging
import uuid
from zoneinfo import ZoneInfo

from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.storage import Store

from .. import satellite_speech
from ..i18n import L
from .when import next_occurrence

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1

KIND_ALARM = "alarm"
KIND_REMINDER = "reminder"
KIND_COMMAND = "command"

# Every 10 s for three minutes, so that it reads as one alarm still ringing. At
# 45 s apart it was taken for several separate alarms (2026-11-01).
RINGS = 18
RING_GAP_S = 10.0
SNOOZE_MINUTES = 9
# After a restart, wait for the satellites and Mic to MediaPlayer to come back
# (a restart takes about 80 s) before running what came due meanwhile.
AFTER_RESTART_S = 90.0
# How late a job may still run after Home Assistant was down. A light switching
# off hours late is a surprise and an alarm that late is useless, but a
# reminder is still worth hearing.
_GRACE = {
    KIND_ALARM: timedelta(minutes=10),
    KIND_COMMAND: timedelta(minutes=10),
    KIND_REMINDER: timedelta(hours=24),
}


@dataclass
class Job:
    """One scheduled thing. ``language`` is the turn's full language tag."""

    id: str
    kind: str
    due: datetime
    text: str = ""
    device_id: str | None = None
    user_id: str | None = None
    person_id: str | None = None
    language: str = "en"
    repeat: str | None = None
    clock: str | None = None

    @property
    def lang(self) -> str:
        return (self.language or "en").split("-")[0].lower()

    def to_dict(self) -> dict:
        return {**asdict(self), "due": self.due.astimezone(UTC).isoformat()}

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(**{**data, "due": datetime.fromisoformat(data["due"])})


@dataclass
class _Ring:
    job: Job
    device_id: str
    task: asyncio.Task | None = None


class Scheduler:
    """Holds the jobs, keeps one timer on the earliest, and runs what is due."""

    def __init__(self, hass, entry_id: str) -> None:
        self._hass = hass
        self._store = Store(hass, STORAGE_VERSION, f"ai_plugin_scheduler.{entry_id}")
        self._jobs: list[Job] = []
        self._unsub: Callable[[], None] | None = None
        self._rings: dict[str, _Ring] = {}
        self._tasks: set[asyncio.Task] = set()
        self._runner: Callable[[Job], Awaitable[str]] | None = None

    # ── time ────────────────────────────────────────────────────────────

    def now(self) -> datetime:
        name = getattr(self._hass.config, "time_zone", None)
        return datetime.now(ZoneInfo(name)) if isinstance(name, str) and name else datetime.now().astimezone()

    # ── storage ─────────────────────────────────────────────────────────

    async def async_load(self) -> None:
        data = await self._store.async_load()
        now = self.now()
        kept: list[Job] = []
        for raw in (data or {}).get("jobs", []):
            try:
                job = Job.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                _LOGGER.warning("AI Plugin scheduler: dropped an unreadable job: %s", raw)
                continue
            if job.due <= now:
                if now - job.due <= _GRACE.get(job.kind, timedelta(0)):
                    job.due = now + timedelta(seconds=AFTER_RESTART_S)
                elif job.repeat:
                    job.due = next_occurrence(job.due, job.repeat, now, job.clock)
                else:
                    _LOGGER.warning(
                        "AI Plugin scheduler: %s due %s was missed while Home "
                        "Assistant was down: %r", job.kind, job.due, job.text,
                    )
                    continue
            kept.append(job)
        self._jobs = kept
        await self._async_save()
        self._arm()

    async def _async_save(self) -> None:
        await self._store.async_save({"jobs": [j.to_dict() for j in self._jobs]})

    async def async_remove_storage(self) -> None:
        await self._store.async_remove()

    def async_unload(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        for ring in list(self._rings.values()):
            if ring.task is not None:
                ring.task.cancel()
        self._rings.clear()
        for task in list(self._tasks):
            task.cancel()

    def set_command_runner(self, runner: Callable[[Job], Awaitable[str]]) -> None:
        """What runs a command job: the conversation path, set by its owner."""
        self._runner = runner

    # ── the jobs ────────────────────────────────────────────────────────

    def jobs(self) -> list[Job]:
        return sorted(self._jobs, key=lambda j: j.due)

    async def async_add(self, kind: str, due: datetime, **fields) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, due=due, **fields)
        self._jobs.append(job)
        await self._async_save()
        self._arm()
        return job

    async def async_cancel(self, jobs: list[Job]) -> None:
        ids = {j.id for j in jobs}
        self._jobs = [j for j in self._jobs if j.id not in ids]
        await self._async_save()
        self._arm()

    # ── running what is due ─────────────────────────────────────────────

    def _arm(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        if self._jobs:
            self._unsub = async_track_point_in_utc_time(
                self._hass, self._async_on_timer, min(j.due for j in self._jobs)
            )

    async def _async_on_timer(self, _fired_at: datetime | None = None) -> None:
        self._unsub = None
        now = self.now()
        due = [j for j in self._jobs if j.due <= now]
        for job in due:
            fired = Job(**asdict(job))
            if job.repeat:
                job.due = next_occurrence(job.due, job.repeat, now, job.clock)
            else:
                self._jobs.remove(job)
            self._start(self._async_fire(fired))
        await self._async_save()
        self._arm()

    def _start(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _async_fire(self, job: Job) -> None:
        try:
            if job.kind == KIND_ALARM:
                self._start_ringing(job)
            elif job.kind == KIND_REMINDER:
                await self._async_remind(job)
            elif job.kind == KIND_COMMAND:
                await self._async_run_command(job)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — one job must never stop the others
            _LOGGER.exception("AI Plugin scheduler: %s failed: %r", job.kind, job.text)

    # ── alarms ──────────────────────────────────────────────────────────

    def _start_ringing(self, job: Job) -> None:
        device_id = job.device_id or ""
        previous = self._rings.pop(device_id, None)
        if previous is not None and previous.task is not None:
            previous.task.cancel()
        ring = _Ring(job, device_id)
        self._rings[device_id] = ring
        ring.task = self._start(self._async_ring(ring))

    async def _async_ring(self, ring: _Ring) -> None:
        job = ring.job
        try:
            for number in range(RINGS):
                # The model's own sentence if the user said what it is for, else
                # a neutral one: an alarm is not always for getting up.
                key = "scheduled_say" if job.text else "alarm_ring"
                spoken = await satellite_speech.async_announce(
                    self._hass, ring.device_id, key, job.lang, label=job.text
                )
                if spoken is None:
                    _LOGGER.warning(
                        "AI Plugin scheduler: alarm could not play on device %s", ring.device_id
                    )
                    await self._async_tell_phone(
                        job, job.text or L.template("alarm_ring", job.lang)
                    )
                    return
                if number < RINGS - 1:
                    await asyncio.sleep(RING_GAP_S)
        finally:
            if self._rings.get(ring.device_id) is ring:
                del self._rings[ring.device_id]

    def _take_rings(self, device_id: str | None) -> list[_Ring]:
        """The rings a spoken "stop" means: this satellite's, else all of them."""
        if device_id and device_id in self._rings:
            return [self._rings.pop(device_id)]
        rings = list(self._rings.values())
        self._rings.clear()
        return rings

    async def async_handle_spoken(
        self, message: str, device_id: str | None, lang: str
    ) -> str | None:
        """"Stop" or "snooze" while an alarm rings. None when no alarm is
        ringing or the words are about something else, so an ordinary "stop"
        still reaches the media shortcut."""
        if not self._rings:
            return None
        text = (message or "").strip().rstrip(".?!,").lower()
        for rx in L.pattern_list("alarm_snooze", lang):
            match = rx.match(text)
            if match:
                minutes = int(match.groupdict().get("minutes") or SNOOZE_MINUTES)
                rings = self._take_rings(device_id)
                for ring in rings:
                    if ring.task is not None:
                        ring.task.cancel()
                    job = ring.job
                    await self.async_add(
                        KIND_ALARM,
                        self.now() + timedelta(minutes=minutes),
                        text=job.text,
                        device_id=job.device_id,
                        user_id=job.user_id,
                        person_id=job.person_id,
                        language=job.language,
                    )
                return L.template("alarm_snoozed", lang, val=minutes)
        for rx in L.pattern_list("alarm_stop", lang):
            if rx.match(text):
                for ring in self._take_rings(device_id):
                    if ring.task is not None:
                        ring.task.cancel()
                return L.template("alarm_stopped", lang)
        return None

    # ── reminders ───────────────────────────────────────────────────────

    async def _async_remind(self, job: Job) -> None:
        spoken = None
        if job.device_id:
            spoken = await satellite_speech.async_announce(
                self._hass, job.device_id, "reminder_due", job.lang, label=job.text
            )
        pushed = await self._async_tell_phone(job, job.text)
        if spoken is None and not pushed:
            await self._async_leave_notice(job, job.text)

    async def _async_tell_phone(self, job: Job, text: str) -> bool:
        """Send ``text`` to the phones of the job's person; True if one took it."""
        if not job.person_id:
            return False
        from ..shortcuts.phone import async_notify_person  # noqa: PLC0415

        return await async_notify_person(
            self._hass, job.person_id, L.template("reminder_title", job.lang), text
        )

    async def _async_leave_notice(self, job: Job, text: str) -> None:
        """Nowhere could play or take it: leave it where it will be seen."""
        _LOGGER.warning("AI Plugin scheduler: could not deliver %r", text)
        await self._hass.services.async_call(
            "persistent_notification",
            "create",
            {"title": L.template("reminder_title", job.lang), "message": text},
            blocking=False,
        )

    # ── commands ────────────────────────────────────────────────────────

    async def _async_run_command(self, job: Job) -> None:
        if self._runner is None:
            _LOGGER.warning("AI Plugin scheduler: no runner for command %r", job.text)
            return
        try:
            reply = await self._runner(job)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("AI Plugin scheduler: command %r failed", job.text)
            reply = L.template("scheduled_failed", job.lang, label=job.text)
        # A light switching is its own confirmation and comes back empty; what
        # has words (a failure, a set-point) is said where it was asked for.
        if not (reply or "").strip():
            return
        spoken = None
        if job.device_id:
            spoken = await satellite_speech.async_announce(
                self._hass, job.device_id, "scheduled_say", job.lang, label=reply
            )
        if spoken is None and not await self._async_tell_phone(job, reply):
            await self._async_leave_notice(job, reply)
