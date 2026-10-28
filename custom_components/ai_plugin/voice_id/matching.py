"""Decide who is speaking from a voice fingerprint and enrolled samples.

Pure functions over numpy arrays; storage lives in profiles.py.

Score per person = mean cosine of their 3 closest samples. When the person
has at least 3 samples recorded on the same kind of microphone (same
satellite model), only those count — the bathroom satellite sounds very
different from a ReSpeaker, and same-mic samples score ~0.57 where
cross-mic ones score ~0.44 on the same voice. A person is named only when
the best score clears THRESHOLD and beats the runner-up by MARGIN.

Defaults calibrated on 31 real satellite clips from 2 speakers (2026-10-24):
other-person top-3 scores never exceeded 0.34; at 0.40 / 0.10 no clip was
assigned to the wrong person, 16/19 ReSpeaker and 11/11 bathroom clips were
decided, the rest came out unknown.

THRESHOLD lowered to 0.35 for v0.10.0 (user decision, 2026-10-28): in four
days of live use 30 of 72 turns came out unknown, most scoring 0.33-0.40,
while a recognised speaker's runner-up stayed at 0.1-0.3. 0.35 still sits
above the 0.34 other-person ceiling measured above, and MARGIN is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLD = 0.35
DEFAULT_MARGIN = 0.10
TOP_K = 3
MIN_SAME_MIC = 3


@dataclass(frozen=True)
class Sample:
    vec: np.ndarray  # unit-norm float32, shape (192,)
    mic_group: str


@dataclass(frozen=True)
class Decision:
    person_id: str | None
    score: float
    margin: float
    reason: str  # "match" | "below_threshold" | "too_close" | "no_profiles"


def person_score(vec: np.ndarray, samples: list[Sample], mic_group: str) -> float | None:
    if not samples:
        return None
    same = [s.vec for s in samples if s.mic_group == mic_group]
    pool = same if len(same) >= MIN_SAME_MIC else [s.vec for s in samples]
    sims = np.sort(np.stack(pool) @ vec)[::-1]
    return float(sims[:TOP_K].mean())


def decide(
    vec: np.ndarray,
    profiles: dict[str, list[Sample]],
    mic_group: str,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> Decision:
    scores = {
        pid: s
        for pid, samples in profiles.items()
        if (s := person_score(vec, samples, mic_group)) is not None
    }
    if not scores:
        return Decision(None, 0.0, 0.0, "no_profiles")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    # With one enrolled person there is no runner-up; the threshold alone
    # must keep strangers out, and it sits above every other-person score
    # seen in calibration.
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best - second
    if best < threshold:
        return Decision(None, best, gap, "below_threshold")
    if len(ranked) > 1 and gap < margin:
        return Decision(None, best, gap, "too_close")
    return Decision(best_id, best, gap, "match")
