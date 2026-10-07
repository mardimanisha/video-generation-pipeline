"""SRT subtitles from scene text and rendered scene durations.

Timing inside a scene is proportional to character count, which is close enough for
readable captions; exact word timing could later come from Inworld's WORD timestamps.
"""
from __future__ import annotations

from services.script_analyzer import split_sentences
from utils.duration import srt_timestamp

MAX_CUE_CHARS = 84


def _cues_for_text(text: str) -> list[str]:
    cues: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        parts = -(-len(sentence) // MAX_CUE_CHARS)  # ceil: cues needed for this sentence
        limit = min(MAX_CUE_CHARS, len(sentence) / parts * 1.15)
        line = ""
        for w in words:  # balanced cues, so no short orphan like "personal finance."
            if line and len(line) + 1 + len(w) > limit:
                cues.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            cues.append(line)
    return cues


def build_srt(scenes: list[tuple[str, float, float]]) -> str:
    """scenes: (text, speech_duration, rendered_duration) in order."""
    out, index, offset = [], 1, 0.0
    for text, speech, rendered in scenes:
        cues = _cues_for_text(text)
        total_chars = sum(len(c) for c in cues) or 1
        t = offset
        for cue in cues:
            dur = speech * len(cue) / total_chars
            out.append(f"{index}\n{srt_timestamp(t)} --> {srt_timestamp(t + dur)}\n{cue}\n")
            index += 1
            t += dur
        offset += rendered
    return "\n".join(out)
