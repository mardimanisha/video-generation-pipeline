"""What the presenter actually says in a generated clip, with word timings.

Flow/Veo voice the scene line themselves and the lips follow that performance, which can
differ from the script (repeated or garbled words, ad-libs). This transcribes the clip's own
soundtrack (faster-whisper, CPU) and aligns the words to the script line, so later stages know
which script words the lips speak, and when. Results are cached next to the clip.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_NAME = "base.en"
_model = None


@dataclass
class Word:
    text: str
    start: float
    end: float


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def _load_audio(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def transcribe(clip: Path, cache_dir: Path | None = None) -> list[Word]:
    """Word-timed transcript of a clip's soundtrack (cached by clip size+mtime)."""
    global _model
    st = clip.stat()
    cache = (cache_dir or clip.parent) / f"{clip.stem}.words.json"
    key = f"{st.st_size}-{st.st_mtime_ns}-{MODEL_NAME}"
    if cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("key") == key:
            return [Word(**w) for w in data["words"]]
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    segments, _ = _model.transcribe(_load_audio(clip), word_timestamps=True, language="en", beam_size=5)
    words = [Word(w.word.strip(), round(float(w.start), 3), round(float(w.end), 3))
             for s in segments for w in s.words if w.word.strip()]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "words": [asdict(w) for w in words]}, indent=1), encoding="utf-8")
    return words


@dataclass
class Alignment:
    pairs: list[tuple[int, int]]   # (script word index, transcript word index) matched in order
    script_words: list[str]
    coverage: float                # share of script words the lips were heard saying
    extra: int                     # transcript words inside the line that are not in the script
    trailing_extra: int = 0        # unscripted words after the line (trimmed off, harmless)

    @property
    def usable(self) -> bool:
        """Good enough to time narration word-by-word against the lips."""
        return self.coverage >= 0.85 and self.extra <= max(2, len(self.script_words) // 8)


def align(script: str, words: list[Word]) -> Alignment:
    sw = script.split()
    a, b = [_norm(w) for w in sw], [_norm(w.text) for w in words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs = [(i + k, j + k) for i, j, n in sm.get_matching_blocks() for k in range(n)]
    coverage = len(pairs) / max(1, len(sw))
    if not pairs:
        return Alignment(pairs, sw, 0.0, len(b), 0)
    first_j, last_j = pairs[0][1], pairs[-1][1]
    inside = (last_j - first_j + 1) - len(pairs) + first_j   # unscripted words before/within the line
    return Alignment(pairs, sw, round(coverage, 3), inside, len(b) - 1 - last_j)

