"""Speech duration estimation and time formatting."""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9À-ɏ']+")
_SENTENCE_END = re.compile(r"[.!?]+")
_CLAUSE_BREAK = re.compile(r"[,;:—]")


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def estimate_speech_duration(text: str, words_per_second: float) -> float:
    """Rough spoken duration: words at a steady rate plus small pauses at punctuation.

    This is only used for planning. Real timing always comes from the measured audio.
    """
    words = count_words(text)
    if words == 0:
        return 0.0
    pauses = 0.3 * len(_SENTENCE_END.findall(text)) + 0.12 * len(_CLAUSE_BREAK.findall(text))
    return round(words / words_per_second + pauses, 2)


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:05.2f}" if m else f"{s:.2f}s"


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
