# Satellite1 Voice-Loop Eval Harness — Design Spec

**Date:** 2026-05-05
**Status:** approved (sections 1–5)
**Target repo:** AI-plugin (iteration target — script itself lives in `/tmp/` as throwaway)

## Goal

Closed-loop end-to-end eval harness that lets Claude (in-session) iteratively
tune AI Plugin code + prompts by:

1. Speaking test utterances via the Satellite1 ESPHome smart speaker.
2. Letting SmartMic (positioned in the same room) hear them and run the full
   HA Assist pipeline (wake → STT → AI Plugin → tool loop → TTS).
3. Capturing structured results from HA's `assist_pipeline_event` stream.
4. Reading the results into the conversation, judging pass/fail with Claude as
   judge, proposing code/prompt fixes, and re-running.

Eval target is comprehensive (option D from brainstorm): tool routing + STT
behavior + full E2E UX (latency, audio routing through Mic-to-MediaPlayer).

## Architecture

```
┌──────────────────────┐                       ┌─────────────────────┐
│ Claude (this session)│                       │  Home Assistant     │
│  /tmp/sat1_eval.py   │◀────WebSocket────────▶│  (port 8123 / WS)   │
└─────────┬────────────┘  assist_pipeline evt  └──────┬──────────────┘
          │                                           │
          │ 1. tts.speak service call                 │
          ▼                                           ▼
   ┌─────────────────┐  air gap  ┌────────────────────────────┐
   │ Satellite1      │──speech──▶│ SmartMic (wake+stt+pipeline)│
   │ (ESP32-S3)      │           └──────────┬─────────────────┘
   │ media_player +  │                      │
   │ raw mic         │◀──audio reply────────│  AI Plugin → reply
   └─────────────────┘                      ▼
                                    Mic-to-MediaPlayer speaker
```

- **Satellite1** = Claude's mouth (TTS-out via ESPHome `media_player`) +
  optionally ears (raw mic, **out of scope v1**).
- **SmartMic** = device under test. Runs full HA Assist pipeline.
- **AI Plugin** = conversation agent processing intents.
- **Mic-to-MediaPlayer integration** = routes reply TTS to room speaker.
- **Claude** = test runner driver + judge, both in same conversation.

## Components

Single Python file `/tmp/sat1_eval.py` (~300 LOC, throwaway). Stdlib +
`websockets` + `aiohttp`.

| Component | Responsibility |
|---|---|
| `HAClient` | WS connect, auth (LL token, fallback supervisor token from MemPalace HA drawer), `call_service()`, `subscribe_events()`, `recv_until(predicate, timeout)` |
| `Injector` | `await speak(utterance, satellite_entity, tts_engine)`. Calls `tts.speak` with `cache=False`, `media_player_entity_id=satellite1`. Returns when Satellite1's own `tts-end` fires (confirms speech emitted before listening for SmartMic pipeline) |
| `PipelineRecorder` | Subscribes `assist_pipeline_event`. Filters by `device_id == SMARTMIC_DEVICE_ID` (pulled from MemPalace HA drawer, `mic1` entry, at script init). Captures bundle: `{wake_word_end, stt_end.text, intent_end.intent_output, tts_end.tts_output.url, timings}`. 20s wall timeout |
| `CaseRunner` | Per case: `inject → record → bundle → write JSON to /tmp/sat1_eval_results/<ts>_<idx>.json`. 3s cooldown between cases. Pre-case `recv_drain()` empties stale events |
| `main()` | Parses inline test corpus, runs cases sequentially, prints summary table, exits |

### Output JSON per case

```json
{
  "idx": 0,
  "utterance": "play jazz in hobby room",
  "stt_heard": "play jazz in the hobby room",
  "tool_calls": [{"name": "play_music", "args": {...}}],
  "reply_text": "...",
  "tts_url": "/api/tts_proxy/...",
  "latency_ms": {"wake": 312, "stt": 845, "intent": 1820, "tts": 410},
  "errors": []
}
```

## Data flow per case

```
t=0     main() picks case[i], cooldown gate clear
t=0.1   PipelineRecorder.subscribe() — start collecting events
t=0.2   Injector.speak(utterance) → tts.speak service call
t=0.3   HA renders TTS → streams audio to media_player.satellite1
t=0.5   Satellite1 ESP32-S3 plays audio out speaker
t=1.0   SmartMic mic picks up speech
t=1.2   SmartMic wake-word fires → assist_pipeline_event: wake_word-end
t=1.4   STT transcribes → stt-end {text: "..."}
t=2.0   Conversation agent (AI Plugin) processes → intent-end
t=2.4   TTS renders reply → tts-end {url: "..."}
t=2.6   Reply audio plays on Mic-to-MediaPlayer speaker
t=2.7   Recorder sees tts-end matching SmartMic pipeline_run → bundle complete
t=2.8   CaseRunner writes JSON, advances cooldown timer
t=5.8   cooldown clear, next case
```

### Pipeline-run filtering

- Each `assist_pipeline_event` carries a `pipeline_run` UUID.
- Recorder ignores events whose `device_id != SMARTMIC_DEVICE_ID`. This filters
  out Satellite1's own pipeline_run when the Injector triggers a `tts-end` for
  Claude's outgoing speech.
- Recorder explicitly waits in order: `wake_word-end` → `stt-end` →
  `intent-end` → `tts-end`, all from the SmartMic pipeline_run.

## Error handling

| Failure mode | Detection | Action |
|---|---|---|
| WS auth fail | `auth_invalid` message from HA | Abort run with clear error; suggest LL token rotation (rotated by 2026-05-02 per memory note) |
| WS drop mid-run | `ConnectionClosed` exception | Auto-reconnect once, resume; 2nd drop → abort with partial results |
| `tts.speak` service error | service-call response `success=False` | Record case error, skip to next |
| Satellite1 silent (offline / disconnected) | No `tts-end` for satellite1 within 5s of `tts.speak` | Record `errors: ["satellite1_silent"]`, skip case |
| SmartMic doesn't wake | No `wake_word-end` within 8s of Satellite1 TTS finish | Record `errors: ["wake_miss"]`, retry once with TTS volume override; if still miss, advance |
| Pipeline stall | No `intent-end` within 20s wall | Record `errors: ["pipeline_timeout: <last_event>"]` |
| AI Plugin tool exception | `intent-end` carries `error` field | Record full error in bundle; counted as failure |
| Cross-talk between cases | Stale events bleeding into next case | 3s cooldown + pre-case `recv_drain()` |
| Mic-to-MediaPlayer routing failure (audio doesn't play) | Cannot auto-detect (no mic capture v1) | Out of scope; flag `tts_url` in bundle for manual playback verify |

### Safety

- Hard cap: max 50 cases per run (avoid runaway).
- Test corpus reviewed by user before run — never inject utterances that could
  trigger destructive HA actions (lights/locks/security) or live trade actions
  (paper-mode context guaranteed by orchestrator config, but corpus discipline
  required).
- `--dry-run` flag: prints planned TTS calls without injecting.

### Logging

- Script logs to `/tmp/sat1_eval_results/run_<ts>.log` (DEBUG for the run).
- HA-side AI Plugin logs already captured via existing logger; pulled into
  per-case bundle if `intent-end` carries them.

## Iteration loop (Claude-driven)

1. Claude defines test corpus inline in script (5–10 cases targeting current
   concern, e.g. media playback, multilingual prompts, tool routing).
2. Claude runs: `python3 /tmp/sat1_eval.py`.
3. Script sequentially exercises each case via Satellite1 → SmartMic → reply.
4. Claude reads `/tmp/sat1_eval_results/run_<ts>.log` + per-case JSON.
5. Claude judges each case against rubric (below), prints verdict table to
   user.
6. On failures: Claude proposes fix in AI Plugin code/prompts → user approves
   → Claude edits → re-run subset.
7. Repeat until corpus passes.

### Judge rubric (Claude applies in-session, not in script)

| Dimension | Check |
|---|---|
| `tool_correct` | Was the right tool called (or none, when none expected)? |
| `intent_match` | STT-heard text matches intent of spoken utterance (≥80% lexical or semantic)? |
| `reply_appropriate` | Reply text answers/confirms the intent without echoing system-prompt artifacts (e.g. no `"Empty reply"` leakage)? |
| `latency_ok` | Total wall < 5s (configurable)? |

Verdict: `pass` if all ✓, else `fail` with first failing dimension noted.

## Smoke test before first real eval

1. Run `--dry-run` — confirms WS auth, finds SmartMic device_id, prints
   planned TTS calls.
2. Run with single trivial case `{"utterance": "what time is it"}` — confirms
   full chain end-to-end.
3. Inspect `/tmp/sat1_eval_results/<ts>_0.json` — must have non-empty
   `stt_heard`, `reply_text`, `tts_url`.

## Out of scope (v1)

- **Audio-to-text capture from Satellite1 mic.** TTS reply text already known
  from `tts-end` event; semantic content sufficient. Mic loop reserved for
  future versions if TTS quality scoring needed.
- **Automated TTS quality scoring.** Manual listen if needed.
- **Regression baseline tracking.** Throwaway, ad-hoc by design — no diff
  against last run.
- **Persistent versioned harness in repo.** Lives in `/tmp/` only; if pattern
  proves valuable, promote to `tests/eval/` as separate effort.

## Open prerequisites (verify before plan)

- SmartMic device_id available in MemPalace HA drawer (`mic1` entry) or fetch
  via `device_registry` WS call at script init.
- Satellite1 entity exposes `media_player.satellite1` and accepts `tts.speak`
  service calls (standard ESPHome `media_player` component behavior).
- HA long-lived token still valid; if 401, fallback to SUPERVISOR_TOKEN path
  per HA credentials drawer recipe.
- `tts.<engine>` entity ID known (likely `tts.home_assistant_cloud` or
  Piper) — script reads from CLI flag with sensible default.
