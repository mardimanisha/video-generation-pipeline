"""Audio (voice) provider interface and a mock implementation for tests and dry runs."""
from __future__ import annotations

import hashlib
from pathlib import Path

from services import ffmpeg
from utils.duration import estimate_speech_duration


class AudioProvider:
    name = "base"

    def clone_voice(self, sample_path: Path, display_name: str, language: str) -> str:
        """Create a reusable voice from a sample; return its provider voice ID."""
        raise NotImplementedError

    def generate(self, text: str, voice_id: str, output_path: Path) -> Path:
        """Synthesize `text` with `voice_id` into a WAV file at `output_path`."""
        raise NotImplementedError


class MockAudioProvider(AudioProvider):
    """Deterministic stand-in narration. Duration is the planning estimate with a stable
    per-text +/-15% variation, so sync logic sees realistic mismatches."""

    name = "mock"

    def __init__(self, words_per_second: float = 2.5, sample_rate: int = 48000):
        self.wps = words_per_second
        self.sample_rate = sample_rate

    def clone_voice(self, sample_path: Path, display_name: str, language: str) -> str:
        return "mock-voice"

    def duration_for(self, text: str) -> float:
        base = max(0.8, estimate_speech_duration(text, self.wps))
        jitter = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:4], 16) / 0xFFFF  # 0..1
        return round(base * (0.85 + 0.30 * jitter), 3)

    def generate(self, text: str, voice_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(ffmpeg.build_test_tone_cmd(self.duration_for(text), output_path, self.sample_rate), timeout=120)
        return output_path
