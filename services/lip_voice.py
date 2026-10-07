"""Regenerate the narration per clip so it matches the presenter's lips (sync.mode = lip_voice).

The clip is the master. Its soundtrack is transcribed with word timings (services/transcribe.py),
the transcript is aligned to the script line, and the line is split at the presenter's pauses
into chunks. Each chunk's script text is synthesized with the cloned voice at a speaking rate
chosen so it lasts as long as the lips move for it (one correction pass, then a small
pitch-preserving tempo trim for the remainder), and placed where the lips start that chunk.
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

from services.audio_fit import atempo_chain
from services.transcribe import Alignment, Word

RATE_RANGE = (0.7, 1.5)       # Inworld speakingRate range that still sounds natural
TRIM_RANGE = (0.92, 1.2)      # final tempo correction after synthesis
PHRASE_PAUSE = 0.2            # a silence this long in the clip's speech ends a phrase


@dataclass
class Chunk:
    text: str
    start: float              # lip timing in clip seconds
    end: float
    rate: float = 1.0
    tempo: float = 1.0
    wav: Path | None = None
    voiced: float = 0.0       # duration of the placed speech
    place: float = 0.0        # where the chunk is laid (seconds into the scene, after cuts)
    fit: float = 0.0          # duration to speak the phrase in (0 = the lips' duration)
    seg: int = -1             # index of the clip speech segment the phrase is spoken in


@dataclass
class LipVoicePlan:
    window: tuple[float, float]               # clip seconds used (cut ranges inside are dropped)
    chunks: list[Chunk] = field(default_factory=list)
    freeze_pad: float = 0.0
    cuts: list[tuple[float, float]] = field(default_factory=list)   # clip ranges removed
    length: float = 0.0                       # scene duration after cuts (+ frozen frames)

    @property
    def duration(self) -> float:
        return self.length or (self.window[1] - self.window[0])

    def to_dict(self) -> dict:
        return {"window": [round(self.window[0], 3), round(self.window[1], 3)], "chunks": len(self.chunks),
                "freeze_pad": self.freeze_pad, "cuts": [[round(a, 3), round(b, 3)] for a, b in self.cuts],
                "rates": [round(c.rate, 2) for c in self.chunks], "tempos": [round(c.tempo, 3) for c in self.chunks]}


LIP_LEAD = 0.05               # voice starts this long after the mouth visibly starts moving
MOUTH_WINDOW = 0.6            # how far from the sound-based estimate to look for the mouth onset
MOUTH_MAX_SHIFT = 0.1         # the clip's sound is generated with the lips: trust it within this


def make_chunks(al: Alignment, words: list[Word], speech: list[tuple[float, float]],
                mouth_onset=None) -> list[Chunk]:
    """Split the script line into phrases with the lips' timing.

    `speech` is the clip's audible speech split at real pauses (>= PHRASE_PAUSE). The clip's
    soundtrack was generated together with the lips, so each speech segment's start and end are
    exactly when the mouth says that phrase. The transcript is only used to tell which script
    words fall in which segment (by each word's end, which the transcriber times reliably;
    its word starts are often pulled back over the preceding pause).
    `mouth_onset(t)` (optional) gives when the mouth visibly starts moving near t; a phrase never
    starts before that."""
    if not al.pairs or not speech:
        return []

    def seg_of(w: Word) -> int:
        probe = max(w.start, w.end - 0.05)
        for k, (a, b) in enumerate(speech):
            if a <= probe < b:
                return k
        # the transcriber stretched the word over a following pause: take the segment it overlaps
        overlap = [max(0.0, min(b, w.end) - max(a, w.start)) for a, b in speech]
        if max(overlap) > 0:
            return overlap.index(max(overlap))
        return min(range(len(speech)), key=lambda k: min(abs(speech[k][0] - probe), abs(speech[k][1] - probe)))

    heard, last = [], 0
    for _, j in al.pairs:
        last = max(seg_of(words[j]), last)       # phrases stay in order
        heard.append(last)
    groups = _balance([i for i, _ in al.pairs], heard, al.script_words, speech)
    chunks = []
    n = len(al.script_words)
    for gi, (k, idx) in enumerate(groups):
        i0 = 0 if gi == 0 else idx[0]
        i1 = (n - 1) if gi == len(groups) - 1 else groups[gi + 1][1][0] - 1  # unmatched words ride along
        a, b = speech[k]
        chunks.append(Chunk(" ".join(al.script_words[i0:i1 + 1]), a, b, seg=k))
    if mouth_onset is not None:
        # the voice must never start before the mouth visibly starts moving
        for c in chunks:
            seen = mouth_onset(c.start)
            if seen is not None and c.start < seen + LIP_LEAD < c.end - 0.15:
                c.start = min(seen + LIP_LEAD, c.start + MOUTH_MAX_SHIFT)
    return chunks


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


PACE_WEIGHT = 1.0             # cost of an uneven pace between phrases ...
HEARD_WEIGHT = 0.12           # ... against moving a word away from where the transcriber heard it


def _balance(idx: list[int], heard: list[int], script_words: list[str],
             speech: list[tuple[float, float]]) -> list[tuple[int, list[int]]]:
    """Assign the matched script words (in order) to the speech segments the transcriber put
    them in, but let words at a phrase edge move to the neighbouring phrase when that evens out
    the speaking pace. Word timings from the transcriber are imprecise around pauses: e.g. it
    puts "famous" before a pause when the lips say "famous founders" after it, which would make
    one phrase very fast and the next very slow."""
    segs = sorted(set(heard))
    if len(segs) < 2:
        return [(segs[0], idx)] if segs else []
    syl = [_syllables(script_words[i]) for i in idx]
    dur = {k: max(0.15, speech[k][1] - speech[k][0]) for k in segs}
    pace = sum(syl) / sum(dur.values())
    pos = {k: p for p, k in enumerate(segs)}
    n, m = len(idx), len(segs)
    INF = float("inf")
    # best[p][e]: cost of putting words [0, e) into the first p+1 segments; prev for backtracking
    best = [[INF] * (n + 1) for _ in range(m)]
    prev = [[0] * (n + 1) for _ in range(m)]

    def group_cost(p: int, a: int, b: int) -> float:
        k = segs[p]
        rate = sum(syl[a:b]) / dur[k]
        moved = sum(abs(pos[heard[t]] - p) for t in range(a, b))
        return PACE_WEIGHT * math.log(rate / pace) ** 2 + HEARD_WEIGHT * moved

    for e in range(1, n - m + 2):
        best[0][e] = group_cost(0, 0, e)
    for p in range(1, m):
        for e in range(p + 1, n - (m - p - 1) + 1):
            for a in range(p, e):
                if best[p - 1][a] < INF:
                    c = best[p - 1][a] + group_cost(p, a, e)
                    if c < best[p][e]:
                        best[p][e], prev[p][e] = c, a
    groups, e = [], n
    for p in range(m - 1, -1, -1):
        a = prev[p][e] if p else 0
        groups.append((segs[p], idx[a:e]))
        e = a
    return groups[::-1]


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-300:])


def _take(text: str, rate: float, provider, voice_id: str, workdir: Path, sample_rate: int,
          trim_filter: str) -> Path:
    """One synthesized take of `text` at a speaking rate, silence-trimmed (cached by voice/rate/text)."""
    workdir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{voice_id}|{rate:.3f}|{text}".encode()).hexdigest()[:16]
    out = workdir / f"{key}.wav"
    if not out.is_file():
        raw = workdir / f"{key}.raw.wav"
        if hasattr(provider, "cfg"):
            old = provider.cfg.speaking_rate
            provider.cfg.speaking_rate = rate
            try:
                provider.generate(text, voice_id, raw)
            finally:
                provider.cfg.speaking_rate = old
        else:
            provider.generate(text, voice_id, raw)
        _ffmpeg(["-i", str(raw), "-af", trim_filter, "-ar", str(sample_rate), "-ac", "1",
                 "-c:a", "pcm_s16le", str(out)])
        raw.unlink(missing_ok=True)
    return out


def natural_duration(chunk: Chunk, provider, voice_id: str, workdir: Path, sample_rate: int,
                     trim_filter: str) -> float:
    """How long the phrase takes at the voice's normal speaking rate."""
    base_rate = provider.cfg.speaking_rate if hasattr(provider, "cfg") else 1.0
    return _wav_duration(_take(chunk.text, base_rate, provider, voice_id, workdir, sample_rate, trim_filter))


def synthesize_chunk(chunk: Chunk, provider, voice_id: str, workdir: Path, sample_rate: int,
                     trim_filter: str) -> None:
    """Synthesize the chunk at a speaking rate that fits its lip duration (cached)."""
    target = chunk.fit or (chunk.end - chunk.start)
    base_rate = provider.cfg.speaking_rate if hasattr(provider, "cfg") else 1.0

    def synth(rate: float) -> Path:
        return _take(chunk.text, rate, provider, voice_id, workdir, sample_rate, trim_filter)

    # the speaking rate is not exactly proportional to duration: correct it up to twice so the
    # voice itself speaks the phrase at the lips' pace (no stretching needed afterwards)
    rate, wav = base_rate, synth(base_rate)
    best = (abs(_wav_duration(wav) - target), rate, wav)
    for _ in range(2 if hasattr(provider, "cfg") else 0):
        dur = _wav_duration(wav)
        if abs(dur - target) <= 0.04 * target:
            break
        new = round(max(RATE_RANGE[0], min(RATE_RANGE[1], rate * dur / max(target, 0.1))), 2)
        if abs(new - rate) < 0.01:
            break
        rate, wav = new, synth(new)
        best = min(best, (abs(_wav_duration(wav) - target), rate, wav), key=lambda b: b[0])
    _, rate, wav = best
    tempo = max(TRIM_RANGE[0], min(TRIM_RANGE[1], _wav_duration(wav) / max(target, 0.1)))
    if abs(tempo - 1.0) > 0.01:
        fitted = wav.with_name(wav.stem + f".t{tempo:.3f}.wav")
        if not fitted.is_file():
            _ffmpeg(["-i", str(wav), "-af", atempo_chain(tempo), "-c:a", "pcm_s16le", str(fitted)])
        wav = fitted
    chunk.rate, chunk.tempo, chunk.wav, chunk.voiced = rate, tempo, wav, _wav_duration(wav)


MAX_PAUSE = 0.7               # longer silences in the clip are shortened (video cut) ...
KEEP_PAUSE = 0.45             # ... to this
MIN_EXTRA = 0.25              # unscripted clip speech at least this long is cut out of the video


def plan_cuts(speech: list[tuple[float, float]], chunks: list[Chunk]) -> list[tuple[float, float]]:
    """Clip ranges to drop from the video: speech the script does not contain (repeated or
    ad-libbed words: the lips move but there is no narration for them), and the excess of long
    silences, so the presenter never stands silent for long."""
    used = {c.seg for c in chunks if c.seg >= 0}
    if not used:
        return []
    cuts, kept = [], []
    for k in range(min(used), max(used) + 1):
        a, b = speech[k]
        if k not in used and b - a >= MIN_EXTRA:
            cuts.append((a - 0.04, b + 0.04))
        else:
            kept.append((a, b))
    for (_, b1), (a2, _) in zip(kept, kept[1:]):
        if a2 - b1 > MAX_PAUSE:
            cuts.append((b1 + KEEP_PAUSE / 2, a2 - KEEP_PAUSE / 2))
    merged: list[list[float]] = []
    for a, b in sorted(cuts):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 0.05]


PACE_SPREAD = 0.12            # a phrase's speed stays within +/-12% of the scene's pace


def smooth_pace(chunks: list[Chunk], natural: list[float]) -> None:
    """Keep the narration at an even pace across the scene: each phrase's speed (natural
    duration / lip duration) is held within PACE_SPREAD of the scene's average speed. A phrase
    the lips rush or drag gets a little more or less time than the lips take; the start stays
    on the lips."""
    speeds = [d / max(0.15, c.end - c.start) for c, d in zip(chunks, natural)]
    weight = sum(natural)
    if not weight:
        return
    scene = math.exp(sum(math.log(v) * d for v, d in zip(speeds, natural)) / weight)
    for c, d, v in zip(chunks, natural, speeds):
        v = min(scene * (1 + PACE_SPREAD), max(scene / (1 + PACE_SPREAD), v))
        c.fit = d / v


def plan_window(chunks: list[Chunk], clip_len: float, lead: float, tail: float, fps: int,
                cuts: list[tuple[float, float]] = ()) -> LipVoicePlan:
    w0 = max(0.0, chunks[0].start - lead)
    cuts = [(round(a * fps) / fps, round(b * fps) / fps) for a, b in cuts if a >= w0]   # whole frames

    def out(t: float) -> float:       # clip time -> scene time once the cuts are removed
        return t - w0 - sum(min(b, t) - a for a, b in cuts if t > a)

    # never let a chunk cut off the previous one: nudge it later if needed
    prev_end = -1.0
    for c in chunks:
        c.place = max(out(c.start), prev_end + 0.04)
        prev_end = c.place + c.voiced
    speech_end = max(prev_end, out(chunks[-1].end))
    dur = math.ceil(round((speech_end + tail) * fps, 6)) / fps
    removed = sum(b - a for a, b in cuts)
    w1 = w0 + dur + removed
    return LipVoicePlan((w0, w1), chunks, round(max(0.0, w1 - clip_len), 3), cuts, dur)


def assemble(plan: LipVoicePlan, out_wav: Path, sample_rate: int) -> Path:
    """Lay each chunk at its lip onset in a silent track of the window's length."""
    total = int(round(plan.duration * sample_rate))
    track = bytearray(total * 2)
    for c in plan.chunks:
        with wave.open(str(c.wav), "rb") as w:
            data = w.readframes(w.getnframes())
        start = max(0, int(round(c.place * sample_rate))) * 2
        end = min(len(track), start + len(data))
        if end > start:
            track[start:end] = data[: end - start]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(track))
    return out_wav
