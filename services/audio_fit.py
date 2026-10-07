"""Fit the narration to the video: the generated clip is the master, the voice follows the lips.

The clip (Flow/Veo) shows the presenter saying the scene's line; its own soundtrack is in
perfect sync with the lips. For each scene:

  1. find the spoken phrases (speech between pauses) in the clip's soundtrack
  2. cut the narration (cloned voice) into the same number of phrases at its own pauses
  3. place each narration phrase so its speech starts exactly where the lips start that
     phrase, and change its tempo (pitch unchanged) so it ends where the lips stop
  4. use the clip untouched, trimmed only to a short lead-in before the first phrase and a
     short tail after the last, so scenes do not carry the clip's leading/trailing silence

The video is never retimed, so mouth movement looks exactly as generated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

TEMPO_RANGE = (0.7, 1.45)   # narration speed change that still sounds natural (1.0 = unchanged)


@dataclass
class Placement:
    narr_start: float   # narration piece [narr_start, narr_end) in narration seconds
    narr_end: float
    tempo: float        # >1 plays faster (shorter), <1 slower
    out_start: float    # where the piece starts, in seconds from the scene window start

    @property
    def out_end(self) -> float:
        return self.out_start + (self.narr_end - self.narr_start) / self.tempo


@dataclass
class FitPlan:
    window: tuple[float, float]   # clip time range used as the scene's video
    placements: list[Placement]
    phrases: int
    mode: str                     # phrases | span
    freeze_pad: float = 0.0       # seconds of held last frame (narration outlasts the clip)
    mismatch: float = 0.0         # seconds where lips and voice disagree (lower is better)

    @property
    def duration(self) -> float:
        return self.window[1] - self.window[0]

    def to_dict(self) -> dict:
        return {"mode": self.mode, "phrases": self.phrases, "freeze_pad": self.freeze_pad,
                "mismatch": self.mismatch,
                "window": [round(self.window[0], 3), round(self.window[1], 3)],
                "tempos": [round(p.tempo, 3) for p in self.placements]}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


END_TOUCH_PENALTY = 0.15
MAX_GROUP = 4          # segments one phrase may span on either side
PAIR_RATIO = (0.55, 1.8)
GROUP_BONUS = 0.3      # prefer more (shorter) phrases: every phrase boundary is an exact sync point


def select_speech(clip: list[tuple[float, float]], narr: list[tuple[float, float]],
                  tolerance: float = 0.08, clip_len: float | None = None) -> list[tuple[float, float]]:
    """Keep the run of consecutive clip speech segments that is the scene's line.

    Clips can carry stray sounds (a breath or click at the end) or a few ad-libbed extra words.
    Every contiguous run of segments is scored by how closely its total speaking time matches
    the narration's (pauses do not matter, they are absorbed by phrase placement). Among runs
    within `tolerance` of the best score, the longest is kept so real parts of the line, e.g.
    a question followed by a dramatic pause, are never dropped."""
    import math as _m
    # a very short sound cut off from the rest at either end is a click or breath, not speech
    clip = list(clip)
    while len(clip) > 1 and clip[-1][1] - clip[-1][0] < 0.25 and clip[-1][0] - clip[-2][1] > 0.5:
        clip.pop()
    while len(clip) > 1 and clip[0][1] - clip[0][0] < 0.25 and clip[1][0] - clip[0][1] > 0.5:
        clip.pop(0)
    if not clip:
        return []
    narr_speech = sum(b - a for a, b in narr) or 1e-3
    scored = []
    for i in range(len(clip)):
        speech = 0.0
        for j in range(i, len(clip)):
            speech += clip[j][1] - clip[j][0]
            score = abs(_m.log(narr_speech / max(speech, 1e-3)))
            # speech running into the clip's last frame is an ad-lib or cut off: keep it only if
            # the line clearly needs it
            if clip_len is not None and j > i and clip_len - clip[j][1] < 0.15:
                score += END_TOUCH_PENALTY
            scored.append((score, i, j))
    best = min(sc for sc, _, _ in scored)
    _, i, j = max((t for t in scored if t[0] <= best + tolerance), key=lambda t: (t[2] - t[1], -t[0]))
    return clip[i:j + 1]


def pair_phrases(narr: list[tuple[float, float]], clip: list[tuple[float, float]],
                 bonus: float = GROUP_BONUS, ratio: tuple[float, float] = PAIR_RATIO
                 ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Group consecutive speech segments on each side into phrases and pair them in order so
    that paired phrases have the most similar lengths (the two voices pause in different places;
    this finds which pauses correspond). Dynamic programming over (narr index, clip index)."""
    import math as _m
    n, m = len(narr), len(clip)
    inf = float("inf")
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int]] = {}
    best[0][0] = 0.0
    for i in range(n):
        for p in range(m):
            if best[i][p] == inf:
                continue
            for i2 in range(i + 1, min(n, i + MAX_GROUP) + 1):
                nd = narr[i2 - 1][1] - narr[i][0]
                for p2 in range(p + 1, min(m, p + MAX_GROUP) + 1):
                    cd = clip[p2 - 1][1] - clip[p][0]
                    r = nd / max(cd, 1e-3)
                    if not ratio[0] <= r <= ratio[1]:
                        continue
                    c = best[i][p] + _m.log(r) ** 2 * (nd + cd) - bonus
                    if c < best[i2][p2]:
                        best[i2][p2] = c
                        back[(i2, p2)] = (i, p)
    if best[n][m] == inf:  # no consistent pairing: treat each side as one phrase
        return [(narr[0][0], narr[-1][1])], [(clip[0][0], clip[-1][1])]
    groups_n, groups_c, state = [], [], (n, m)
    while state != (0, 0):
        i, p = back[state]
        groups_n.append((narr[i][0], narr[state[0] - 1][1]))
        groups_c.append((clip[p][0], clip[state[1] - 1][1]))
        state = (i, p)
    return groups_n[::-1], groups_c[::-1]


def _build_plan(narr_segs, narr_len, clip_segs, clip_len, lead, tail, fps, bonus, ratio) -> FitPlan:
    narr_p, clip_p = pair_phrases(narr_segs, clip_segs, bonus, ratio)
    n = len(narr_p)
    mode = "phrases" if n > 1 else "span"
    w0 = max(0.0, clip_p[0][0] - lead)
    # Cut the narration inside its pauses so every sample is used exactly once.
    cuts = [0.0] + [(narr_p[i][1] + narr_p[i + 1][0]) / 2 for i in range(n - 1)] + [narr_len]
    placements = []
    for i, ((ns, ne), (cs, ce)) in enumerate(zip(narr_p, clip_p)):
        tempo = _clamp((ne - ns) / max(ce - cs, 1e-3), *TEMPO_RANGE)
        a, b = cuts[i], cuts[i + 1]
        # the speech onset inside this piece (ns) lands on the lip onset (cs)
        out_start = (cs - w0) - (ns - a) / tempo
        placements.append(Placement(a, b, tempo, out_start))
    # A piece must not start before the window; shift it (keeps order, drops nothing).
    for p in placements:
        if p.out_start < 0:
            p.out_start = 0.0
    narr_speech_end = placements[-1].out_start + (narr_p[-1][1] - placements[-1].narr_start) / placements[-1].tempo
    speech_end = max(clip_p[-1][1] - w0, narr_speech_end)
    # never cut narration: the window always reaches the end of the placed speech plus a tail;
    # if that is past the clip's last frame, the last frame is held (QA reports it)
    dur = math.ceil(round((speech_end + tail) * fps, 6)) / fps
    freeze = max(0.0, w0 + dur - clip_len)
    return FitPlan((w0, w0 + dur), placements, n, mode, round(freeze, 3))


def placed_voice(plan: FitPlan, narr_segs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Where the narration's speech lands inside the scene window under this plan."""
    out = []
    for pl in plan.placements:
        for s, e in narr_segs:
            s2, e2 = max(s, pl.narr_start), min(e, pl.narr_end)
            if e2 > s2:
                out.append((pl.out_start + (s2 - pl.narr_start) / pl.tempo,
                            pl.out_start + (e2 - pl.narr_start) / pl.tempo))
    return out


def mismatch(plan: FitPlan, narr_segs, clip_segs, step: float = 0.01) -> float:
    """Seconds inside the scene where the mouth and the voice disagree: lips moving with no
    voice, or voice over a closed mouth. This is what a viewer perceives as bad lip sync."""
    w0, w1 = plan.window
    lips = [(s - w0, e - w0) for s, e in clip_segs if e > w0 and s < w1]
    voice = placed_voice(plan, narr_segs)
    bad, t, total = 0.0, 0.0, w1 - w0
    while t < total:
        speaking = any(s <= t < e for s, e in lips)
        voiced = any(s <= t < e for s, e in voice)
        if speaking != voiced:
            bad += step
        t += step
    return round(bad, 3)


SEARCH_BONUS = (0.3, 0.8, 1.5)
SEARCH_RATIO = ((0.55, 1.8), (0.45, 2.2))


def plan_fit(narr_segs: list[tuple[float, float]], narr_len: float,
             clip_segs: list[tuple[float, float]], clip_len: float, *,
             lead: float = 0.25, tail: float = 0.3, fps: int = 24) -> FitPlan | None:
    """Plan narration placement onto the clip's spoken phrases. Several candidate plans are
    built (which clip speech belongs to the line; coarse vs fine phrase pairing) and the one
    with the least lips/voice disagreement wins. None if either side has no speech."""
    if not narr_segs or not clip_segs:
        return None
    real = [g for g in clip_segs if g[1] - g[0] >= 0.25] or list(clip_segs)
    no_end = real[:-1] if len(real) > 1 and clip_len - real[-1][1] < 0.15 else real
    selections = {tuple(select_speech(clip_segs, narr_segs, clip_len=clip_len)), tuple(real), tuple(no_end)}
    best, best_key = None, None
    for sel in selections:
        if not sel:
            continue
        for bonus in SEARCH_BONUS:
            for ratio in SEARCH_RATIO:
                plan = _build_plan(narr_segs, narr_len, list(sel), clip_len, lead, tail, fps, bonus, ratio)
                clamped = sum(1 for p in plan.placements if p.tempo in TEMPO_RANGE)
                plan.mismatch = mismatch(plan, narr_segs, clip_segs)
                key = (round(plan.mismatch + 0.05 * clamped + 0.5 * plan.freeze_pad, 3), -len(sel))
                if best_key is None or key < best_key:
                    best, best_key = plan, key
    return best


def render_fitted_narration(audio: Path, plan: FitPlan, out_wav: Path, sample_rate: int) -> Path:
    """Build the scene's narration track: each piece tempo-changed (pitch kept) by its own small
    ffmpeg call, then laid at its exact sample offset in a silent track of the window's length.
    (One big split/delay/mix filter graph can deadlock inside ffmpeg; this cannot.)"""
    import subprocess
    import tempfile
    import wave

    total = int(round(plan.duration * sample_rate))
    track = bytearray(total * 2)  # 16-bit mono silence
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(plan.placements):
            piece = Path(tmp) / f"p{i}.wav"
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(audio), "-af",
                 f"atrim=start={p.narr_start:.4f}:end={p.narr_end:.4f},asetpts=PTS-STARTPTS,"
                 f"aresample={sample_rate},{atempo_chain(p.tempo)}",
                 "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(piece)],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"tempo change failed for piece {i}: {proc.stderr.strip()[-300:]}")
            with wave.open(str(piece), "rb") as w:
                data = w.readframes(w.getnframes())
            start = int(round(p.out_start * sample_rate)) * 2
            end = min(len(track), start + len(data))
            if end > start:
                # pieces tile the narration and are placed in order, so overlaps are only the
                # silent edges of neighbouring pieces: mix by taking the louder sample
                seg = memoryview(track)[start:end]
                new = data[: end - start]
                if any(seg):
                    import array
                    a, b = array.array("h", bytes(seg)), array.array("h", new)
                    seg[:] = array.array("h", (x if abs(x) >= abs(y) else y for x, y in zip(a, b))).tobytes()
                else:
                    seg[:] = new
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(track))
    return out_wav


def atempo_chain(tempo: float) -> str:
    """ffmpeg atempo accepts 0.5-2.0 per stage."""
    stages, t = [], tempo
    while t > 2.0:
        stages.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        stages.append("atempo=0.5")
        t /= 0.5
    stages.append(f"atempo={t:.5f}")
    return ",".join(stages)
