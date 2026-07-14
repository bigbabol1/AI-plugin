#!/usr/bin/env python3
"""Offline eval harness for AI Plugin — drives a live Home Assistant instance
via POST /api/conversation/process and scores replies against a regex rubric.

Unlike the unit test suite (which mocks every LLM call), this measures the
real thing: model behaviour, tool routing, shortcut hits, and latency against
whatever backend the integration is configured with.

Zero HA-test-harness dependencies: stdlib + PyYAML only. Sequential on
purpose — one conversation agent, one GPU.

Usage:
    export HA_URL=http://homeassistant.local:8123
    export HA_TOKEN=...            # or HA_TOKEN_FILE=/path/to/token
    python3 tests/eval/run_eval.py                # safe (read-only) prompts
    python3 tests/eval/run_eval.py --full         # include mutating prompts
    python3 tests/eval/run_eval.py --category sun --dry-run
    python3 tests/eval/run_eval.py --only sensor_temp_bedroom_en

Prompts live in prompts.yaml; per-install names (areas, test entities,
satellite device_id) in home.yaml (falls back to home.example.yaml).
See README.md in this directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).parent
REQUEST_TIMEOUT = 120  # seconds per conversation turn (multi-tool loops are slow)


# ── config ────────────────────────────────────────────────────────────────────


def _load_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    if os.environ.get("HA_TOKEN"):
        return os.environ["HA_TOKEN"]
    token_file = args.token_file or os.environ.get("HA_TOKEN_FILE")
    if token_file:
        return Path(token_file).read_text().strip()
    sys.exit("No token: set HA_TOKEN, HA_TOKEN_FILE, or pass --token/--token-file")


def _load_home() -> dict:
    for name in ("home.yaml", "home.example.yaml"):
        path = EVAL_DIR / name
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def _substitute(text: str, values: dict) -> str:
    """Replace {placeholder} tokens from home.yaml values. Unknown → error."""

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            sys.exit(f"prompts.yaml references unknown placeholder {{{key}}} — add it to home.yaml")
        return str(values[key])

    # Placeholders start with a letter so regex quantifiers like {2} survive.
    return re.sub(r"\{([a-z][a-z0-9_]*)\}", repl, text)


# ── HA API ────────────────────────────────────────────────────────────────────


class HAClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[int, dict | str]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
                try:
                    return resp.status, json.loads(body)
                except json.JSONDecodeError:
                    return resp.status, body
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()[:500]
        except Exception as exc:  # noqa: BLE001 — timeouts, conn refused
            return 0, f"{type(exc).__name__}: {exc}"

    def converse(
        self,
        text: str,
        language: str,
        agent_id: str,
        conversation_id: str | None,
        device_id: str | None,
    ) -> tuple[int, dict | str]:
        payload: dict = {"text": text, "language": language, "agent_id": agent_id}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if device_id:
            payload["device_id"] = device_id
        return self._post("/api/conversation/process", payload)

    def render_template(self, template: str) -> str:
        status, body = self._post("/api/template", {"template": template}, timeout=15)
        if status != 200:
            return f"[template error {status}: {body}]"
        return body if isinstance(body, str) else json.dumps(body)

    def call_service(self, domain: str, service: str, data: dict) -> int:
        status, _ = self._post(f"/api/services/{domain}/{service}", data, timeout=15)
        return status


# ── rubric ────────────────────────────────────────────────────────────────────


def _extract_speech(response: dict | str) -> str | None:
    if not isinstance(response, dict):
        return None
    return (
        response.get("response", {})
        .get("speech", {})
        .get("plain", {})
        .get("speech")
    )


def _check_expect(speech: str, expect: dict, defaults: dict) -> list[str]:
    """Return list of failure reasons (empty = pass)."""
    failures: list[str] = []
    empty_ok = expect.get("empty_ok", False)
    if not speech.strip():
        if empty_ok:
            return []
        return ["empty reply"]
    for pattern in expect.get("none", []) + defaults.get("none", []):
        if re.search(pattern, speech):
            failures.append(f"matched negative /{pattern}/")
    any_patterns = expect.get("any", [])
    if any_patterns and not any(re.search(p, speech) for p in any_patterns):
        failures.append(f"matched no positive pattern of {any_patterns}")
    return failures


# ── runner ────────────────────────────────────────────────────────────────────


def _normalize_turns(prompt: dict) -> list[dict]:
    turns = prompt.get("turns") or [prompt["text"]]
    out = []
    for t in turns:
        out.append({"text": t} if isinstance(t, str) else dict(t))
    # A prompt-level expect applies to the final turn unless it has its own.
    if "expect" in prompt and "expect" not in out[-1]:
        out[-1]["expect"] = prompt["expect"]
    return out


def run_prompt(client: HAClient, prompt: dict, defaults: dict, args: argparse.Namespace) -> dict:
    lang_default = prompt.get("lang", "en")
    conversation_id: str | None = None
    turn_results = []
    failures: list[str] = []

    for turn in _normalize_turns(prompt):
        started = time.monotonic()
        status, response = client.converse(
            text=turn["text"],
            language=turn.get("lang", lang_default),
            agent_id=args.agent_id,
            conversation_id=conversation_id,
            device_id=args.device_id,
        )
        latency = time.monotonic() - started
        speech = _extract_speech(response)
        if isinstance(response, dict):
            conversation_id = response.get("conversation_id") or conversation_id
        turn_results.append(
            {"text": turn["text"], "status": status, "latency_s": round(latency, 2), "speech": speech}
        )
        if status != 200:
            failures.append(f"HTTP {status}: {str(response)[:200]}")
            break
        if speech is None:
            failures.append("no speech field in response")
            break
        if "expect" in turn:
            failures.extend(_check_expect(speech, turn["expect"], defaults))

    # Optional live-state verification via HA templates (real actuation proof).
    verify = prompt.get("verify_template")
    if verify and not failures:
        time.sleep(1.0)  # let the state machine settle
        rendered = client.render_template(verify).strip()
        if rendered != "True":
            failures.append(f"verify_template rendered {rendered!r}, expected 'True'")

    cleanup = prompt.get("cleanup_service")
    if cleanup:
        cleanup = dict(cleanup)
        client.call_service(cleanup.pop("domain"), cleanup.pop("service"), cleanup)

    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "mutating": bool(prompt.get("mutating")),
        "pass": not failures,
        "failures": failures,
        "turns": turn_results,
        "total_latency_s": round(sum(t["latency_s"] for t in turn_results), 2),
    }


# ── reporting ─────────────────────────────────────────────────────────────────


def summarize(results: list[dict]) -> str:
    lines = ["# AI Plugin eval summary", ""]
    lines.append(f"- run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    total = len(results)
    passed = sum(r["pass"] for r in results)
    lines.append(f"- overall: **{passed}/{total} passed** ({passed / total:.0%})" if total else "- no prompts run")
    latencies = [t["latency_s"] for r in results for t in r["turns"]]
    if latencies:
        lines.append(
            f"- turn latency: median {statistics.median(latencies):.1f}s, "
            f"max {max(latencies):.1f}s, n={len(latencies)}"
        )
    lines.append("")
    lines.append("| category | pass | median s | max s |")
    lines.append("|---|---|---|---|")
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(by_cat.items()):
        lat = [t["latency_s"] for r in rs for t in r["turns"]]
        lines.append(
            f"| {cat} | {sum(r['pass'] for r in rs)}/{len(rs)} "
            f"| {statistics.median(lat):.1f} | {max(lat):.1f} |"
        )
    fails = [r for r in results if not r["pass"]]
    if fails:
        lines.append("")
        lines.append("## Failures")
        for r in fails:
            lines.append(f"### {r['id']} ({r['category']})")
            for reason in r["failures"]:
                lines.append(f"- {reason}")
            for t in r["turns"]:
                lines.append(f"- `{t['text']}` → [{t['latency_s']}s] {t['speech']!r}")
    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HA_URL", ""))
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    parser.add_argument("--agent-id", default="conversation.ai_plugin")
    parser.add_argument("--device-id", default=None, help="satellite device_id → voice mode; default from home.yaml")
    parser.add_argument("--no-voice", action="store_true", help="omit device_id (text-mode prompts)")
    parser.add_argument("--full", action="store_true", help="include mutating prompts (actuation, timers, memory)")
    parser.add_argument("--category", action="append", help="only these categories (repeatable)")
    parser.add_argument("--only", action="append", help="only these prompt ids (repeatable)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between prompts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    home = _load_home()
    values = home.get("values", {})
    if not args.base_url:
        args.base_url = home.get("base_url", "")
    if not args.base_url:
        sys.exit("No HA URL: set HA_URL, home.yaml base_url, or --base-url")
    if args.device_id is None and not args.no_voice:
        args.device_id = home.get("satellite_device_id")
    if args.no_voice:
        args.device_id = None

    matrix = yaml.safe_load((EVAL_DIR / "prompts.yaml").read_text())
    defaults = matrix.get("defaults", {})
    prompts = matrix["prompts"]

    selected = []
    for p in prompts:
        if p.get("mutating") and not args.full:
            continue
        if args.category and p["category"] not in args.category:
            continue
        if args.only and p["id"] not in args.only:
            continue
        selected.append(json.loads(_substitute(json.dumps(p), values)))
    if args.limit:
        selected = selected[: args.limit]

    # Fail fast on malformed rubric regexes — name the offending prompt.
    for p in selected:
        patterns = list(defaults.get("none", []))
        for turn in _normalize_turns(p):
            exp = turn.get("expect", {})
            patterns += exp.get("any", []) + exp.get("none", [])
        for pat in patterns:
            try:
                re.compile(pat)
            except re.error as exc:
                sys.exit(f"{p['id']}: bad regex /{pat}/ — {exc}")

    print(f"{len(selected)} prompts selected (voice={'on' if args.device_id else 'off'}, "
          f"mutating={'included' if args.full else 'excluded'})")
    if args.dry_run:
        for p in selected:
            turns = [t["text"] for t in _normalize_turns(p)]
            flag = " [MUTATING]" if p.get("mutating") else ""
            print(f"  {p['id']:<32} {p['category']:<12}{flag} {turns}")
        return

    client = HAClient(args.base_url, _load_token(args))
    out_dir = Path(args.out_dir) if args.out_dir else (
        EVAL_DIR / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_dir / "results.jsonl", "w") as fh:
        for i, p in enumerate(selected, 1):
            result = run_prompt(client, p, defaults, args)
            results.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            mark = "PASS" if result["pass"] else "FAIL"
            print(f"[{i}/{len(selected)}] {mark} {p['id']} ({result['total_latency_s']}s)")
            if not result["pass"]:
                for reason in result["failures"]:
                    print(f"        {reason}")
            time.sleep(args.delay)

    summary = summarize(results)
    (out_dir / "summary.md").write_text(summary)
    print()
    print(summary)
    print(f"results: {out_dir}")
    if any(not r["pass"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
