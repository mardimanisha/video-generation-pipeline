"""Phrase-level lip alignment between a generated clip and the narration.

The video model (Gemini/Veo) animates the presenter saying the scene's line and also produces
its own speech track, which is in sync with the lips. The narration (Inworld) says the same
words with different timing. Both have pauses at the same phrase breaks, so:

  1. find speech segments (between pauses) in the clip's audio and in the narration
  2. pair them up (merging the shortest gaps on whichever side has extra pauses)
  3. build a piecewise-linear time map narration-time -> clip-time through those anchors
  4. render the clip through that map, so each spoken phrase in the video lands on the same
     phrase in the narration

Nothing here re-animates the mouth; it re-times existing frames. Within a phrase the sync
is approximate, at phrase boundaries (start, pauses, end) it is exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from services import ffmpeg

# Allowed playback-rate multipliers (output seconds per clip second).
SPEECH_RANGE = (0.6, 1.6)   # while talking: beyond this motion looks unnatural
PAUSE_RANGE = (0.3, 3.0)    # during pauses the mouth is closed, so more is tolerable
UTTERANCE_GAP = 0.8         # clip silence longer than this separates distinct utterances


@dataclass
class Piece:
    c0: float  # clip time
    c1: float
    t0: float  # output (narration) time
    t1: float

    @property
    def factor(self) -> float:
        return (self.t1 - self.t0) / (self.c1 - self.c0)


@dataclass
class AlignPlan:
    pieces: list[Piece]
    pre_hold: float        # seconds of first frame held before the clip starts (clip began too late)
    post_hold: float       # seconds of last frame held at the end (clip ended too early)
    mode: str              # phrases | span
    matched_phrases: int
    mismatch: float = 0.0  # seconds where retimed lips and narration disagree

    def to_dict(self) -> dict:
        return {"mode": self.mode, "phrases": self.matched_phrases, "mismatch": self.mismatch,
                "clip_used": [round(min(p.c0 for p in self.pieces), 3), round(max(p.c1 for p in self.pieces), 3)],
                "pre_hold": round(self.pre_hold, 3), "post_hold": round(self.post_hold, 3),
                "factors": [round(p.factor, 3) for p in self.pieces]}


def speech_segments(path, threshold_db: float, min_silence: float, duration: float | None = None
                    ) -> list[tuple[float, float]]:
    """Non-silent intervals of a file's first audio stream, via ffmpeg silencedetect."""
    proc = ffmpeg.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                       f"silencedetect=n={threshold_db:g}dB:d={min_silence:g}", "-f", "null", "-"],
                      timeout=120)
    if duration is None:
        duration = ffmpeg.probe(path).audio_duration or ffmpeg.probe(path).duration
    return segments_from_silence_log(proc.stderr, duration)


def segments_from_silence_log(log: str, duration: float) -> list[tuple[float, float]]:
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    silences = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else duration
        silences.append((max(0.0, s), min(duration, e)))
    speech, cursor = [], 0.0
    for s, e in sorted(silences):
        if s - cursor > 0.05:
            speech.append((cursor, s))
        cursor = max(cursor, e)
    if duration - cursor > 0.05:
        speech.append((cursor, duration))
    return speech


def _merge_to(segs: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Merge across the shortest gaps until only n segments remain."""
    segs = list(segs)
    while len(segs) > n:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        k = gaps.index(min(gaps))
        segs[k:k + 2] = [(segs[k][0], segs[k + 1][1])]
    return segs


def pick_utterance(clip_segs: list[tuple[float, float]], narration_speech: float
                   ) -> list[tuple[float, float]]:
    """Clips sometimes contain extra sounds or words. Split the clip's speech into utterances
    at long silences and keep the one whose length best matches the narration."""
    if not clip_segs:
        return []
    groups, cur = [], [clip_segs[0]]
    for seg in clip_segs[1:]:
        if seg[0] - cur[-1][1] > UTTERANCE_GAP:
            groups.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    groups.append(cur)
    return min(groups, key=lambda g: abs((g[-1][1] - g[0][0]) - narration_speech))


def _in(value: float, rng: tuple[float, float]) -> bool:
    return rng[0] <= value <= rng[1]


def _plan_from_pairs(a, b, mode: str, target: float, clip_len: float) -> AlignPlan | None:
    """Time map through paired phrases: narration phrase a[i] (output time) <- clip phrase b[i]."""
    anchors = []  # (output time, clip time)
    for (ts, te), (cs, ce) in zip(a, b):
        anchors += [(ts, cs), (te, ce)]
    pieces = []
    for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
        if t1 - t0 < 1e-3 or c1 - c0 < 1e-3:
            return None
        speaking = len(pieces) % 2 == 0  # anchors alternate: speech, pause, speech, ...
        if not _in((t1 - t0) / (c1 - c0), SPEECH_RANGE if speaking else PAUSE_RANGE):
            return None
        pieces.append(Piece(c0, c1, t0, t1))
    # Lead-in and tail play at normal speed around the speech.
    first_t, first_c = anchors[0]
    last_t, last_c = anchors[-1]
    pre_c = first_c - first_t
    pre_hold = max(0.0, -pre_c)
    if first_t > 0:
        pieces.insert(0, Piece(max(0.0, pre_c), first_c, pre_hold, first_t))
    tail = target - last_t
    post_hold = 0.0
    if tail > 0:
        end_c = min(clip_len, last_c + tail)
        if end_c - last_c > 1e-3:
            pieces.append(Piece(last_c, end_c, last_t, last_t + (end_c - last_c)))
        post_hold = max(0.0, tail - (end_c - last_c))
    pieces = [p for p in pieces if p.c1 - p.c0 > 1e-3 and p.t1 - p.t0 > 1e-3]
    return AlignPlan(pieces, pre_hold, post_hold, mode, len(a))


def plan_alignment(narr: list[tuple[float, float]], clip: list[tuple[float, float]],
                   target: float, clip_len: float) -> AlignPlan | None:
    """Original simple pairing (kept for compatibility); search_alignment is the better one."""
    if not narr or not clip:
        return None
    clip = pick_utterance(clip, narr[-1][1] - narr[0][0])
    n = min(len(narr), len(clip))
    candidates = [("phrases", _merge_to(narr, n), _merge_to(clip, n))]
    if n > 1:
        candidates.append(("span", _merge_to(narr, 1), _merge_to(clip, 1)))
    for mode, a, b in candidates:
        plan = _plan_from_pairs(a, b, mode, target, clip_len)
        if plan:
            return plan
    return None


def plan_mismatch(plan: AlignPlan, narr: list[tuple[float, float]], clip_segs: list[tuple[float, float]],
                  target: float, step: float = 0.01) -> float:
    """Seconds of output where the (retimed) lips and the (untouched) narration disagree.
    Held frames count as a closed mouth."""
    bad, t = 0.0, 0.0
    while t < target:
        voiced = any(s <= t < e for s, e in narr)
        speaking = False
        for p in plan.pieces:
            if p.t0 <= t < p.t1:
                c = p.c0 + (t - p.t0) / p.factor
                speaking = any(s <= c < e for s, e in clip_segs)
                break
        if speaking != voiced:
            bad += step
        t += step
    return round(bad, 3)


def search_alignment(narr: list[tuple[float, float]], clip: list[tuple[float, float]],
                     target: float, clip_len: float) -> AlignPlan | None:
    """Narration-master alignment: build time maps from several phrase pairings (which clip
    speech is the line; coarse vs fine pairing) and keep the one where the retimed lips best
    follow the untouched narration."""
    from services import audio_fit as af
    if not narr or not clip:
        return None
    real = [g for g in clip if g[1] - g[0] >= 0.25] or list(clip)
    no_end = real[:-1] if len(real) > 1 and clip_len - real[-1][1] < 0.15 else real
    selections = {tuple(af.select_speech(clip, narr, clip_len=clip_len)), tuple(real), tuple(no_end)}
    best, best_key = None, None
    for sel in selections:
        if not sel:
            continue
        for bonus in af.SEARCH_BONUS:
            for ratio in af.SEARCH_RATIO:
                a, b = af.pair_phrases(narr, list(sel), bonus, ratio)
                plan = _plan_from_pairs(a, b, "phrases" if len(a) > 1 else "span", target, clip_len)
                if plan is None:
                    continue
                plan.mismatch = plan_mismatch(plan, narr, clip, target)
                key = round(plan.mismatch + 0.5 * (plan.pre_hold + plan.post_hold), 3)
                if best_key is None or key < best_key:
                    best, best_key = plan, key
    return best
