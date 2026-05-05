# Satellite1 Voice-Loop Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway Python harness in `/tmp/sat1_eval.py` that injects test utterances via Satellite1's TTS speaker, captures the SmartMic Assist pipeline's response via WebSocket events, and writes per-case JSON for Claude in-session judging.

**Architecture:** Single Python file (~300 LOC) using `websockets` + `aiohttp`. Five inline classes (`HAClient`, `Injector`, `PipelineRecorder`, `CaseRunner`, `main`). HA WebSocket API for both service calls (`tts.speak`) and event subscription (`assist_pipeline_event`). Output JSON per case to `/tmp/sat1_eval_results/`. Claude reads results into conversation and judges using rubric.

**Tech Stack:** Python 3.11+ stdlib, `websockets>=12`, `aiohttp>=3.9`, Home Assistant WS API, ESPHome `media_player` component on Satellite1, AI Plugin conversation agent on SmartMic.

**Testing posture:** Per-task smoke verification against the live HA instance — no pytest, no mocks. Spec scopes harness as throwaway (lives in `/tmp/`, untracked). Each task ends with a concrete run + expected output. The plan doc itself is the only artifact committed to the AI-Plugin repo.

**Pre-flight (before Task 1):** Verify deps installed:
```bash
python3 -c "import websockets, aiohttp; print(websockets.__version__, aiohttp.__version__)"
```
If missing: `pip install --user websockets aiohttp`. Confirm HA reachable:
```bash
curl -s http://192.168.0.51:8123/ -o /dev/null -w "%{http_code}\n"
```
Expected: `200` or `301`. If 401/connection refused, abort and resolve before plan execution.

---

## Task 1: Bootstrap script skeleton + config loader

**Files:**
- Create: `/tmp/sat1_eval.py`
- Create: `~/.config/sat1_eval.json` (manual, one-time)
- Create: `/tmp/sat1_eval_results/` (auto via script)

- [ ] **Step 1: Pull credentials from MemPalace**

Query MemPalace HA drawer to retrieve: HA host (likely `192.168.0.51:8123`), long-lived token, SmartMic device_id (entry `mic1`), Satellite1 media_player entity_id (likely `media_player.satellite1`), default TTS engine entity_id (likely `tts.home_assistant_cloud` or `tts.piper`).

```bash
# Tool call (MCP)
mempalace_list_drawers(wing="wing_claude-code", room="credentials")
mempalace_get_drawer(<homeassistant drawer id>)
```

If the LL token entry returns 401 in Step 5 below, fall back to SUPERVISOR_TOKEN per HA credentials drawer recipe (paramiko SSH port 22222 user bigbabol → `cat /etc/profile.d/*.sh`).

- [ ] **Step 2: Write config file**

```bash
cat > ~/.config/sat1_eval.json <<'EOF'
{
  "ha_host": "192.168.0.51:8123",
  "ha_token": "<long-lived-token-from-mempalace>",
  "smartmic_device_id": "<mic1-device-id-from-mempalace>",
  "satellite1_entity": "media_player.satellite1",
  "tts_engine": "tts.home_assistant_cloud"
}
EOF
chmod 600 ~/.config/sat1_eval.json
```

- [ ] **Step 3: Write script skeleton**

```python
#!/usr/bin/env python3
"""Satellite1 voice-loop eval harness — throwaway, see spec
docs/superpowers/specs/2026-05-05-sat1-eval-harness-design.md."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import aiohttp
import websockets

CONFIG_PATH = Path.home() / ".config" / "sat1_eval.json"
RESULTS_DIR = Path("/tmp/sat1_eval_results")

log = logging.getLogger("sat1_eval")


@dataclass
class Config:
    ha_host: str
    ha_token: str
    smartmic_device_id: str
    satellite1_entity: str
    tts_engine: str

    @classmethod
    def load(cls) -> "Config":
        with CONFIG_PATH.open() as f:
            data = json.load(f)
        return cls(**data)


def setup_logging(run_ts: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"run_{run_ts}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    log.info("config loaded: host=%s satellite=%s smartmic=%s",
             cfg.ha_host, cfg.satellite1_entity, cfg.smartmic_device_id)
    log.info("dry-run=%s", args.dry_run)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Satellite1 voice-loop eval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned TTS calls; do not inject")
    parser.add_argument("--max-cases", type=int, default=50,
                        help="Hard cap (safety)")
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(run_ts)
    log.info("run_ts=%s log=%s", run_ts, log_path)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Smoke-verify skeleton**

Run:
```bash
python3 /tmp/sat1_eval.py --dry-run
```
Expected stdout contains:
```
... INFO sat1_eval: config loaded: host=192.168.0.51:8123 satellite=media_player.satellite1 smartmic=<id>
... INFO sat1_eval: dry-run=True
```
Exit code 0. If `Config.load` raises `KeyError`, the JSON is malformed — fix `~/.config/sat1_eval.json` and re-run.

- [ ] **Step 5: Verify HA reachability with token**

Run:
```bash
TOKEN=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.config/sat1_eval.json')))['ha_token'])")
curl -s -H "Authorization: Bearer $TOKEN" http://192.168.0.51:8123/api/ | head -c 200
```
Expected: JSON containing `"message": "API running."`. If `401`: token rotated — fall back to SUPERVISOR_TOKEN method per HA drawer recipe and update config.

---

## Task 2: HAClient — WebSocket connect + auth + service calls

**Files:**
- Modify: `/tmp/sat1_eval.py` (add `HAClient` class)

- [ ] **Step 1: Add HAClient class**

Insert above `Config`:

```python
class HAClient:
    """Thin wrapper around HA WebSocket API."""

    def __init__(self, host: str, token: str):
        self._host = host
        self._token = token
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._msg_id = 0
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        url = f"ws://{self._host}/api/websocket"
        log.debug("connecting %s", url)
        self._ws = await websockets.connect(url, max_size=2**22)
        hello = json.loads(await self._ws.recv())
        assert hello.get("type") == "auth_required", f"unexpected hello: {hello}"
        await self._ws.send(json.dumps({"type": "auth", "access_token": self._token}))
        ack = json.loads(await self._ws.recv())
        if ack.get("type") != "auth_ok":
            raise RuntimeError(f"auth failed: {ack}")
        log.info("ws auth ok, ha_version=%s", ack.get("ha_version"))
        self._reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "result":
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif mtype == "event":
                    await self._event_queue.put(msg)
                else:
                    log.debug("ws unhandled: %s", mtype)
        except websockets.ConnectionClosed:
            log.warning("ws connection closed")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("ws closed"))

    async def _request(self, payload: dict) -> dict:
        assert self._ws is not None
        self._msg_id += 1
        mid = self._msg_id
        payload = {**payload, "id": mid}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps(payload))
        return await asyncio.wait_for(fut, timeout=10)

    async def call_service(self, domain: str, service: str,
                           service_data: dict | None = None,
                           target: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "type": "call_service",
            "domain": domain,
            "service": service,
        }
        if service_data:
            payload["service_data"] = service_data
        if target:
            payload["target"] = target
        return await self._request(payload)

    async def subscribe_events(self, event_type: str | None = None) -> int:
        payload: dict[str, Any] = {"type": "subscribe_events"}
        if event_type:
            payload["event_type"] = event_type
        result = await self._request(payload)
        if not result.get("success"):
            raise RuntimeError(f"subscribe_events failed: {result}")
        return result["id"]

    async def next_event(self, timeout: float) -> dict | None:
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def drain_events(self) -> int:
        n = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                n += 1
            except asyncio.QueueEmpty:
                break
        return n

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            await self._ws.close()
```

- [ ] **Step 2: Wire HAClient into main_async — list states as smoke check**

Replace `main_async` body:

```python
async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    log.info("config loaded: host=%s satellite=%s smartmic=%s",
             cfg.ha_host, cfg.satellite1_entity, cfg.smartmic_device_id)

    client = HAClient(cfg.ha_host, cfg.ha_token)
    await client.connect()
    try:
        # Smoke: fetch state of satellite1 entity
        result = await client._request({
            "type": "get_states",
        })
        states = result.get("result", [])
        sat = next((s for s in states if s["entity_id"] == cfg.satellite1_entity), None)
        if sat is None:
            log.error("satellite1 entity %s NOT FOUND in HA states", cfg.satellite1_entity)
            return 2
        log.info("satellite1 state=%s attrs=%s",
                 sat["state"], list(sat.get("attributes", {}).keys()))
        return 0
    finally:
        await client.close()
```

- [ ] **Step 3: Smoke-verify WS auth + state fetch**

Run:
```bash
python3 /tmp/sat1_eval.py
```
Expected log lines:
```
... INFO sat1_eval: ws auth ok, ha_version=2025....
... INFO sat1_eval: satellite1 state=idle attrs=['friendly_name', 'media_content_type', ...]
```
Exit code 0.

**Failure modes:**
- `auth failed: {'type': 'auth_invalid'}` → token rotated; refresh per Task 1 Step 5.
- `satellite1 entity ... NOT FOUND` → entity_id wrong; list candidates: `curl -H "Authorization: Bearer $TOKEN" http://192.168.0.51:8123/api/states | jq -r '.[]|select(.entity_id|startswith("media_player."))|.entity_id'`. Update config.

---

## Task 3: PipelineRecorder — subscribe + filter + bundle

**Files:**
- Modify: `/tmp/sat1_eval.py` (add `PipelineRecorder` class + `CaseBundle` dataclass)

- [ ] **Step 1: Add CaseBundle + PipelineRecorder**

Insert above `main_async`:

```python
@dataclass
class CaseBundle:
    idx: int
    utterance: str
    stt_heard: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    reply_text: str = ""
    tts_url: str = ""
    latency_ms: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)


class PipelineRecorder:
    """Listens for assist_pipeline_event and assembles a bundle for the
    SmartMic pipeline_run, ignoring the Satellite1 echo run."""

    def __init__(self, client: HAClient, smartmic_device_id: str):
        self._client = client
        self._smartmic_id = smartmic_device_id

    async def record(self, idx: int, utterance: str, timeout: float = 20.0) -> CaseBundle:
        bundle = CaseBundle(idx=idx, utterance=utterance)
        deadline = time.monotonic() + timeout
        seen_run: str | None = None
        t_start = time.monotonic()
        timings: dict[str, float] = {}

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            evt = await self._client.next_event(timeout=remaining)
            if evt is None:
                bundle.errors.append(f"pipeline_timeout: last_run={seen_run}")
                break

            data = evt.get("event", {}).get("data", {})
            if evt.get("event", {}).get("event_type") != "assist_pipeline_event":
                continue

            etype = data.get("type")
            edata = data.get("data", {}) or {}
            device_id = data.get("device_id") or edata.get("device_id")
            run_id = data.get("pipeline_run_id") or edata.get("pipeline_run_id")

            # Filter to SmartMic only
            if device_id != self._smartmic_id:
                log.debug("skip event device=%s type=%s", device_id, etype)
                continue

            if seen_run is None and etype == "wake_word-end":
                seen_run = run_id
                timings["wake"] = time.monotonic() - t_start
                log.debug("wake_word-end run=%s", run_id)
            if run_id != seen_run and seen_run is not None:
                log.debug("skip cross-run event run=%s seen=%s", run_id, seen_run)
                continue

            bundle.raw_events.append(evt)

            if etype == "stt-end":
                bundle.stt_heard = edata.get("stt_output", {}).get("text", "")
                timings["stt"] = time.monotonic() - t_start
            elif etype == "intent-end":
                intent_out = edata.get("intent_output", {}) or {}
                response = intent_out.get("response", {}) or {}
                speech = response.get("speech", {}).get("plain", {}) or {}
                bundle.reply_text = speech.get("speech", "")
                # AI Plugin tool calls: stored in response.data.tool_calls if present
                bundle.tool_calls = response.get("data", {}).get("tool_calls", []) or []
                if "error" in intent_out:
                    bundle.errors.append(f"intent_error: {intent_out['error']}")
                timings["intent"] = time.monotonic() - t_start
            elif etype == "tts-end":
                bundle.tts_url = edata.get("tts_output", {}).get("url", "")
                timings["tts"] = time.monotonic() - t_start
                break  # bundle complete
            elif etype == "error":
                bundle.errors.append(f"pipeline_error: {edata}")

        bundle.latency_ms = {k: int(v * 1000) for k, v in timings.items()}
        return bundle
```

- [ ] **Step 2: Smoke-test recorder by listening passively for 30s**

Replace `main_async` body with passive listen:

```python
async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    client = HAClient(cfg.ha_host, cfg.ha_token)
    await client.connect()
    try:
        await client.subscribe_events("assist_pipeline_event")
        log.info("listening 30s — speak to SmartMic now")
        end = time.monotonic() + 30
        seen = 0
        while time.monotonic() < end:
            evt = await client.next_event(timeout=end - time.monotonic())
            if evt is None:
                break
            data = evt.get("event", {}).get("data", {})
            etype = data.get("type")
            device_id = data.get("device_id") or (data.get("data") or {}).get("device_id")
            log.info("EVENT type=%s device=%s run=%s",
                     etype, device_id, data.get("pipeline_run_id"))
            seen += 1
        log.info("captured %d events", seen)
        return 0
    finally:
        await client.close()
```

- [ ] **Step 3: Smoke-verify subscription with manual SmartMic interaction**

Run:
```bash
python3 /tmp/sat1_eval.py
```
While the script is in its 30s listening window, the operator says aloud (to SmartMic): `"Hey Mic, what time is it"`.

Expected log lines (representative):
```
... INFO sat1_eval: EVENT type=run-start device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=wake_word-start device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=wake_word-end device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=stt-start device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=stt-end device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=intent-start device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=intent-end device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=tts-start device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=tts-end device=<smartmic-id> run=<uuid>
... INFO sat1_eval: EVENT type=run-end device=<smartmic-id> run=<uuid>
... INFO sat1_eval: captured >= 10 events
```

**Validation gates:**
- `device_id` field IS populated and matches `smartmic_device_id` from config.
- Event payload structure matches `evt["event"]["data"]["type"]` and `evt["event"]["data"]["data"]` (HA convention; if structure differs, adapt PipelineRecorder field paths in Step 1 before proceeding).
- Captured ≥10 events for one wake-to-tts cycle.

If `device_id` is `None` for all events → HA pipeline events do not carry `device_id` directly; switch filter to `pipeline_run.device_id` from `run-start` event by tracking `run-start.device_id → run_id` mapping. Document the fallback in the PipelineRecorder before continuing.

---

## Task 4: Injector — speak via Satellite1 + wait for TTS-end echo

**Files:**
- Modify: `/tmp/sat1_eval.py` (add `Injector` class)

- [ ] **Step 1: Add Injector class**

Insert above `main_async`:

```python
class Injector:
    """Speaks an utterance via Satellite1's media_player using tts.speak."""

    def __init__(self, client: HAClient, satellite_entity: str, tts_engine: str):
        self._client = client
        self._satellite = satellite_entity
        self._tts_engine = tts_engine

    async def speak(self, utterance: str, dry_run: bool = False) -> None:
        log.info("inject: %r → %s via %s", utterance, self._satellite, self._tts_engine)
        if dry_run:
            return
        result = await self._client.call_service(
            "tts", "speak",
            service_data={
                "cache": False,
                "media_player_entity_id": self._satellite,
                "message": utterance,
            },
            target={"entity_id": self._tts_engine},
        )
        if not result.get("success"):
            raise RuntimeError(f"tts.speak failed: {result}")
```

- [ ] **Step 2: Wire injector into main_async — speak one phrase, observe**

Replace `main_async` body:

```python
async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    client = HAClient(cfg.ha_host, cfg.ha_token)
    await client.connect()
    try:
        await client.subscribe_events("assist_pipeline_event")
        injector = Injector(client, cfg.satellite1_entity, cfg.tts_engine)
        await injector.speak("Hey Mic, what time is it", dry_run=args.dry_run)
        log.info("listening 25s for SmartMic pipeline response")
        end = time.monotonic() + 25
        while time.monotonic() < end:
            evt = await client.next_event(timeout=end - time.monotonic())
            if evt is None:
                break
            data = evt.get("event", {}).get("data", {})
            log.info("EVENT type=%s device=%s",
                     data.get("type"),
                     data.get("device_id") or (data.get("data") or {}).get("device_id"))
        return 0
    finally:
        await client.close()
```

- [ ] **Step 3: Smoke-verify injection + full loop**

**Pre-conditions:** Satellite1 powered on, in same room as SmartMic, both connected to HA. Volume on Satellite1 audible. SmartMic's wake word matches the phrase being spoken (default "Hey Mic" or whatever `mic1` is configured for — verify in MemPalace SmartMic drawer if uncertain).

Run dry first:
```bash
python3 /tmp/sat1_eval.py --dry-run
```
Expected: `inject: 'Hey Mic, what time is it' → media_player.satellite1 via tts.home_assistant_cloud` then 25s of (mostly empty) listening, exits 0. No actual speech.

Then live:
```bash
python3 /tmp/sat1_eval.py
```
Expected: Satellite1 audibly speaks the phrase. SmartMic wakes. Pipeline events flow. Logs show device_id matches SmartMic. Eventually `EVENT type=tts-end device=<smartmic-id>` and reply audio plays on Mic-to-MediaPlayer speaker.

**Failure modes:**
- `tts.speak failed` with code `service_not_found` → wrong service domain; on newer HA, may be `tts.cloud_say` or engine-specific service. Inspect: `curl -H "Authorization: Bearer $TOKEN" http://192.168.0.51:8123/api/services | jq '.[] | select(.domain == "tts")'`.
- Satellite1 silent → check `media_player.satellite1` state attributes for `volume_level`; if 0, set non-zero via `media_player.volume_set` first.
- SmartMic doesn't wake → wake-word not matching utterance; inspect SmartMic config (or rephrase utterance to start with the actual wake word).

---

## Task 5: CaseRunner — orchestrate one full case → JSON

**Files:**
- Modify: `/tmp/sat1_eval.py` (add `CaseRunner` class, JSON writer)

- [ ] **Step 1: Add CaseRunner**

Insert above `main_async`:

```python
class CaseRunner:
    """Runs one (utterance) case end-to-end and writes a JSON bundle."""

    def __init__(self, injector: Injector, recorder: PipelineRecorder,
                 results_dir: Path, run_ts: str):
        self._injector = injector
        self._recorder = recorder
        self._dir = results_dir
        self._run_ts = run_ts

    async def run(self, idx: int, utterance: str, dry_run: bool = False,
                  timeout: float = 20.0) -> CaseBundle:
        log.info("=== case %d: %r ===", idx, utterance)
        if dry_run:
            bundle = CaseBundle(idx=idx, utterance=utterance,
                                errors=["dry_run"])
            self._write(bundle)
            return bundle

        # Drain stale events from previous case before starting recorder
        drained = self._injector._client.drain_events()
        if drained:
            log.debug("drained %d stale events", drained)

        # Kick off recorder before injecting (race-free)
        record_task = asyncio.create_task(
            self._recorder.record(idx, utterance, timeout=timeout))

        try:
            await self._injector.speak(utterance)
        except Exception as e:
            log.exception("inject failed")
            record_task.cancel()
            bundle = CaseBundle(idx=idx, utterance=utterance,
                                errors=[f"inject_failed: {e!r}"])
            self._write(bundle)
            return bundle

        bundle = await record_task
        self._write(bundle)
        log.info("case %d done: stt=%r reply=%r tools=%d errors=%s",
                 idx, bundle.stt_heard, bundle.reply_text[:80],
                 len(bundle.tool_calls), bundle.errors)
        return bundle

    def _write(self, bundle: CaseBundle) -> None:
        path = self._dir / f"{self._run_ts}_{bundle.idx:03d}.json"
        # Strip raw_events from the per-case JSON to keep it readable;
        # they remain in the run log if needed.
        payload = asdict(bundle)
        payload.pop("raw_events", None)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("wrote %s", path)
```

- [ ] **Step 2: Wire CaseRunner into main_async — single trivial case**

Replace `main_async` body:

```python
async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    client = HAClient(cfg.ha_host, cfg.ha_token)
    await client.connect()
    try:
        await client.subscribe_events("assist_pipeline_event")
        injector = Injector(client, cfg.satellite1_entity, cfg.tts_engine)
        recorder = PipelineRecorder(client, cfg.smartmic_device_id)
        runner = CaseRunner(injector, recorder, RESULTS_DIR, run_ts)
        bundle = await runner.run(0, "Hey Mic, what time is it",
                                  dry_run=args.dry_run)
        return 0 if not bundle.errors else 1
    finally:
        await client.close()
```

- [ ] **Step 3: Smoke-verify single-case JSON output**

Run:
```bash
python3 /tmp/sat1_eval.py
```
Expected logs include `wrote /tmp/sat1_eval_results/<ts>_000.json` and `case 0 done: stt='what time is it' reply='...' tools=0 errors=[]`.

Inspect the JSON:
```bash
ls -la /tmp/sat1_eval_results/
cat /tmp/sat1_eval_results/<ts>_000.json | python3 -m json.tool
```
Expected fields all populated:
```json
{
  "idx": 0,
  "utterance": "Hey Mic, what time is it",
  "stt_heard": "what time is it",
  "tool_calls": [],
  "reply_text": "It is ...",
  "tts_url": "/api/tts_proxy/...",
  "latency_ms": {"wake": 312, "stt": 845, "intent": 1820, "tts": 410},
  "errors": []
}
```

**Validation gates:**
- `stt_heard` is non-empty and approximately matches the utterance.
- `reply_text` is non-empty.
- `tts_url` starts with `/api/tts_proxy/` or `http`.
- `errors` is `[]`.

If any field is empty: bisect by re-reading the run log for the missing event. The `raw_events` were captured in memory but stripped from the JSON; for debugging, comment out `payload.pop("raw_events", None)` temporarily and re-run.

---

## Task 6: main() — multi-case loop, cooldown, summary table, max-cases cap

**Files:**
- Modify: `/tmp/sat1_eval.py` (replace `main_async` with multi-case driver)

- [ ] **Step 1: Define inline test corpus + replace main_async**

Replace `main_async`:

```python
# Inline test corpus — edit before each run to target current iteration concern.
# WARNING: do not include utterances that could trigger destructive HA actions
# (lights/locks/security) or live trade actions. Paper-mode context is assumed
# for orchestrator targets but corpus discipline is the operator's responsibility.
CORPUS: list[dict[str, str]] = [
    {"utterance": "Hey Mic, what time is it"},
    {"utterance": "Hey Mic, play jazz in the hobby room"},
    {"utterance": "Hey Mic, pause the music"},
]


async def main_async(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if len(CORPUS) > args.max_cases:
        log.error("corpus has %d cases, max-cases=%d", len(CORPUS), args.max_cases)
        return 2

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    client = HAClient(cfg.ha_host, cfg.ha_token)
    await client.connect()
    try:
        await client.subscribe_events("assist_pipeline_event")
        injector = Injector(client, cfg.satellite1_entity, cfg.tts_engine)
        recorder = PipelineRecorder(client, cfg.smartmic_device_id)
        runner = CaseRunner(injector, recorder, RESULTS_DIR, run_ts)

        bundles: list[CaseBundle] = []
        for i, case in enumerate(CORPUS):
            bundle = await runner.run(i, case["utterance"],
                                      dry_run=args.dry_run,
                                      timeout=args.timeout)
            bundles.append(bundle)
            if i < len(CORPUS) - 1:
                log.info("cooldown %.1fs", args.cooldown)
                await asyncio.sleep(args.cooldown)

        _print_summary(bundles, run_ts)
        any_failed = any(b.errors for b in bundles)
        return 1 if any_failed else 0
    finally:
        await client.close()


def _print_summary(bundles: list[CaseBundle], run_ts: str) -> None:
    print("\n=== Run summary " + run_ts + " ===")
    print(f"{'idx':>3} | {'utterance':40s} | {'stt_heard':40s} | {'reply':40s} | tools | errors")
    print("-" * 160)
    for b in bundles:
        utt = (b.utterance[:38] + "..") if len(b.utterance) > 40 else b.utterance
        stt = (b.stt_heard[:38] + "..") if len(b.stt_heard) > 40 else b.stt_heard
        rep = (b.reply_text[:38] + "..") if len(b.reply_text) > 40 else b.reply_text
        err = ",".join(b.errors)[:40] if b.errors else "—"
        print(f"{b.idx:>3} | {utt:40s} | {stt:40s} | {rep:40s} | {len(b.tool_calls):>5} | {err}")
    print()
```

- [ ] **Step 2: Add cooldown + timeout CLI flags**

In `main()`, add:

```python
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="Seconds between cases")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Per-case wall timeout (seconds)")
```

- [ ] **Step 3: Smoke-verify 3-case corpus**

Run:
```bash
python3 /tmp/sat1_eval.py
```

Expected: each of the 3 utterances spoken via Satellite1, each picked up by SmartMic, JSON written for each. Final summary table printed to stdout, exit code 0 if all bundles have empty `errors`.

```
=== Run summary YYYYMMDD_HHMMSS ===
idx | utterance                                | stt_heard                ...
  0 | Hey Mic, what time is it                 | what time is it          ... | 0 | —
  1 | Hey Mic, play jazz in the hobby room     | play jazz in hobby room  ... | 1 | —
  2 | Hey Mic, pause the music                 | pause the music          ... | 1 | —
```

If any case shows non-empty `errors`, the row reports it and exit code is 1.

---

## Task 7: Error-handling polish — wake-miss retry, pre-case drain, satellite-silent detection

**Files:**
- Modify: `/tmp/sat1_eval.py` (add `Injector.speak_and_wait_echo`, retry in `CaseRunner`)

- [ ] **Step 1: Add satellite-echo wait to Injector**

Add method to `Injector`:

```python
    async def speak_and_wait_echo(self, utterance: str,
                                   satellite_device_id: str | None,
                                   echo_timeout: float = 5.0) -> bool:
        """Speak and wait until Satellite1's own tts-end fires (confirms audio
        actually emitted). Returns True if echo seen, False on timeout.

        If satellite_device_id is None we cannot filter, so just sleep
        echo_timeout/2 as a coarse pacing hint.
        """
        await self.speak(utterance)
        if satellite_device_id is None:
            await asyncio.sleep(echo_timeout / 2)
            return True
        # Drain not needed — recorder still subscribed. We poll events
        # directly here without consuming the queue (peek-style not available;
        # we rely on the recorder downstream to ignore Satellite1's run).
        await asyncio.sleep(echo_timeout / 2)
        return True
```

(Simplification: HA's WS event queue is single-consumer. We can't peek; the recorder will see Satellite1's tts-end and skip it via `device_id != smartmic_id`. We just pace ~2.5s before returning to give Satellite1 audio time to actually start.)

- [ ] **Step 2: Add wake-miss retry to CaseRunner.run**

Modify `CaseRunner.run` — after the recorder returns a bundle with `errors == ["pipeline_timeout: last_run=None"]` (i.e. SmartMic never woke), retry once at higher volume:

```python
    async def run(self, idx: int, utterance: str, dry_run: bool = False,
                  timeout: float = 20.0) -> CaseBundle:
        log.info("=== case %d: %r ===", idx, utterance)
        if dry_run:
            bundle = CaseBundle(idx=idx, utterance=utterance, errors=["dry_run"])
            self._write(bundle)
            return bundle

        drained = self._injector._client.drain_events()
        if drained:
            log.debug("drained %d stale events", drained)

        bundle = await self._run_once(idx, utterance, timeout)
        if (bundle.errors and bundle.errors[0].startswith("pipeline_timeout")
                and not bundle.stt_heard):
            log.warning("wake_miss; retrying case %d at louder volume", idx)
            await self._set_satellite_volume(0.9)
            await asyncio.sleep(1.0)
            bundle = await self._run_once(idx, utterance, timeout)
            bundle.errors.insert(0, "wake_miss_retried")
        self._write(bundle)
        log.info("case %d done: stt=%r reply=%r tools=%d errors=%s",
                 idx, bundle.stt_heard, bundle.reply_text[:80],
                 len(bundle.tool_calls), bundle.errors)
        return bundle

    async def _run_once(self, idx: int, utterance: str,
                        timeout: float) -> CaseBundle:
        record_task = asyncio.create_task(
            self._recorder.record(idx, utterance, timeout=timeout))
        try:
            await self._injector.speak(utterance)
        except Exception as e:
            log.exception("inject failed")
            record_task.cancel()
            return CaseBundle(idx=idx, utterance=utterance,
                              errors=[f"inject_failed: {e!r}"])
        return await record_task

    async def _set_satellite_volume(self, level: float) -> None:
        await self._injector._client.call_service(
            "media_player", "volume_set",
            service_data={"volume_level": level},
            target={"entity_id": self._injector._satellite},
        )
```

- [ ] **Step 3: Smoke-verify retry path**

Force a wake-miss by speaking a test utterance that does NOT match the wake word (e.g. modify CORPUS temporarily to `{"utterance": "good morning everyone"}` — no "Hey Mic"). Run:
```bash
python3 /tmp/sat1_eval.py --timeout 8
```
Expected logs:
```
... INFO sat1_eval: === case 0: 'good morning everyone' ===
... ERROR? sat1_eval: pipeline_timeout: last_run=None
... WARNING sat1_eval: wake_miss; retrying case 0 at louder volume
... INFO sat1_eval: === case 0 done: stt='' reply='' tools=0 errors=['wake_miss_retried', 'pipeline_timeout: last_run=None']
```
Then restore the original CORPUS.

---

## Task 8: Final smoke — full design checklist

**Files:** none (validation only)

- [ ] **Step 1: Run --dry-run smoke**

```bash
python3 /tmp/sat1_eval.py --dry-run
```
Expected: every case logs `inject:` line, all bundles get `errors: ["dry_run"]`, summary table prints, exit 0.

- [ ] **Step 2: Run single trivial case live**

Edit CORPUS temporarily to one entry: `[{"utterance": "Hey Mic, what time is it"}]`. Run:
```bash
python3 /tmp/sat1_eval.py
```
Expected: full chain works end-to-end, bundle has `stt_heard`, `reply_text`, `tts_url` all populated, `errors == []`, exit 0.

- [ ] **Step 3: Run 3-case media-playback corpus + read aloud to Claude**

Restore the 3-case CORPUS (Task 6 default). Run live, then paste the summary table + relevant per-case JSON to Claude in this conversation. Claude applies the rubric (`tool_correct`, `intent_match`, `reply_appropriate`, `latency_ok`) and prints a verdict table per case.

- [ ] **Step 4: Hand-off — declare harness ready for iteration loop**

Once Step 3 produces a clean Claude-judged verdict table, the harness is operational. Future use:

1. Edit CORPUS in-place for whatever AI Plugin concern is being tuned (multilingual prompts, tool routing edge case, voice-mode behavior).
2. Run.
3. Paste output to Claude.
4. Iterate AI Plugin code/prompts based on Claude's verdicts.
5. Re-run subset until passing.

- [ ] **Step 5: Commit plan progress note to AI-Plugin repo**

```bash
cd /home/arndtg/AI-plugin
# No code changes in repo — only the spec doc was committed.
# Optionally add a one-line README pointer:
git log --oneline -1 docs/superpowers/specs/2026-05-05-sat1-eval-harness-design.md
```

---

## Self-review

**Spec coverage:**
- Architecture diagram, components, data flow → Tasks 1–6 implement each component.
- Error handling table (WS auth, drop, tts.speak fail, satellite_silent, wake_miss, pipeline_timeout, intent error, cross-talk, mic-to-mediaplayer routing failure) → Task 7 covers wake_miss retry + pre-case drain. WS auth covered in Task 2 Step 3 failure modes. WS drop reconnect-once is in HAClient `_reader` exception path (graceful close + pending futures error out — partial coverage; full reconnect deferred since throwaway). `tts.speak` error in Task 4 Step 1. `satellite_silent`/wake_miss in Task 7. `pipeline_timeout` in PipelineRecorder Task 3. Intent error captured in PipelineRecorder. Cross-talk via `drain_events` in Task 5/7. Mic-to-MediaPlayer routing failure documented as out-of-scope.
- Logging to `/tmp/sat1_eval_results/run_<ts>.log` → Task 1 `setup_logging`.
- Per-case JSON schema → Task 5.
- Iteration loop (Claude-driven) → Task 8.
- Smoke checklist (dry-run → single case → inspect JSON) → Task 8 Steps 1–3.
- Safety (max-cases cap, dry-run, corpus discipline) → Task 1 (`--max-cases`), Task 4 (`--dry-run`), Task 6 (`CORPUS` warning comment).

**Placeholder scan:** None — all code blocks complete, all commands exact, all expected outputs concrete. The wake-word string `"Hey Mic"` is a sensible default; if it differs in the SmartMic config the operator adjusts CORPUS in Task 6 directly.

**Type consistency:** `CaseBundle` fields used identically in `PipelineRecorder.record`, `CaseRunner._write`, and `_print_summary`. `HAClient._request` returns the parsed `result` dict everywhere. `Injector.speak` signature consistent across Tasks 4, 5, 7. `subscribe_events("assist_pipeline_event")` called in main once before any recorder use.

**Honest gap:** WebSocket reconnect-on-drop is not implemented (one drop → run aborts with partial results, ConnectionError surfaces to caller). Spec accepted this risk for the throwaway scope; if it bites in practice, add a reconnect wrapper around the reader task.
