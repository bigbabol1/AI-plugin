# Offline eval harness

Measures the *real* integration — model behaviour, tool routing, shortcut
hits, latency — against a live Home Assistant install, via
`POST /api/conversation/process`. The unit suite mocks every LLM call; this
does not. Run it before and after any prompt/orchestrator change.

## Setup

1. Create a long-lived access token (HA → Profile → Security).
2. `cp home.example.yaml home.yaml` and fill in your URL, a harmless test
   light, and (optionally) a satellite `device_id` for voice mode.
3. PyYAML is the only dependency beyond the stdlib.

## Run

```bash
export HA_TOKEN_FILE=~/.config/ha_token          # or HA_TOKEN=...
python3 tests/eval/run_eval.py                    # safe, read-only prompts
python3 tests/eval/run_eval.py --full             # + actuation/timers/memory
python3 tests/eval/run_eval.py --category sun --category sensor
python3 tests/eval/run_eval.py --only weather_tokyo_en
python3 tests/eval/run_eval.py --dry-run          # list selection, no calls
python3 tests/eval/run_eval.py --no-voice         # text mode (default prompt)
```

Results land in `tests/eval/results/<timestamp>/` as `results.jsonl` +
`summary.md` (per-category pass rates, latency median/max, failure detail).
Exit code is non-zero when any prompt fails, so it can gate releases.

## Notes

- **Mutating prompts** (`mutating: true`) actuate real devices, create real
  timers, and write real memory facts. They only run with `--full`, verify
  via HA templates (`verify_template` must render `True`), and clean up
  after themselves (`cleanup_service`).
- Prompts run sequentially with a 1s gap (`--delay`) — one agent, one GPU.
- Voice mode (`device_id` set) exercises `SYSTEM_PROMPT_VOICE` +
  URL-stripping + TTS suppression: actuation prompts expect empty replies
  there (`empty_ok`). With `--no-voice` the same actuations return text
  confirmations instead — rubrics accept both.
- A pass is: HTTP 200, non-empty speech (unless `empty_ok`), at least one
  `any` regex when given, no `none` regex, no `defaults.none` regex.
