"""Typed, validated pipeline configuration.

config/default.json is deep-merged with the project's config.json "overrides" and then
validated here, so a typo or bad value fails at startup instead of mid-production.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoConfig(_Strict):
    width: int = Field(1280, ge=64)
    height: int = Field(720, ge=64)
    fps: int = Field(24, ge=1, le=120)
    codec: str = "libx264"
    crf: int = Field(18, ge=0, le=51)
    preset: str = "medium"
    pix_fmt: str = "yuv420p"


class AudioConfig(_Strict):
    provider: Literal["inworld", "mock"] = "inworld"
    sample_rate: int = Field(48000, ge=8000, le=96000)
    channels: int = Field(1, ge=1, le=2)
    codec: str = "aac"
    bitrate: str = "192k"
    tail_padding: float = Field(0.2, ge=0, le=2)
    # TTS output carries ~0.5 s of trailing silence; trimming it keeps the pause between scenes
    # conversational: keep_trailing + tail_padding + keep_leading ~= 0.35 s.
    trim_silence: bool = True
    silence_threshold_db: float = Field(-45, le=-20, ge=-80)
    keep_leading: float = Field(0.03, ge=0, le=1)
    keep_trailing: float = Field(0.12, ge=0, le=1)
    sentence_gap: float = Field(0.2, ge=0, le=2)  # silence between sentences inside a scene


class GenerationConfig(_Strict):
    provider: Literal["gemini", "flow", "library", "mock", "manual"] = "gemini"
    max_clip_duration: float = Field(8, gt=0)
    preferred_clip_duration: float = Field(8, gt=0)
    allowed_durations: list[float] = Field(default_factory=lambda: [4, 6, 8])
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"

    @model_validator(mode="after")
    def _check(self):
        if self.preferred_clip_duration > self.max_clip_duration:
            raise ValueError("preferred_clip_duration must be <= max_clip_duration")
        if not self.allowed_durations:
            raise ValueError("allowed_durations must not be empty")
        if max(self.allowed_durations) > self.max_clip_duration:
            raise ValueError("allowed_durations must not exceed max_clip_duration")
        self.allowed_durations = sorted(set(self.allowed_durations))
        return self


class PlanningConfig(_Strict):
    words_per_second: float = Field(2.5, gt=0.5, lt=6)
    target_scene_duration: float = Field(7.0, gt=0)
    max_scene_duration: float = Field(8.5, gt=0)
    min_scene_duration: float = Field(2.5, ge=0)
    # Optimiser weights (services/scene_planner.py). Raise scene_cost when every scene costs a
    # generated clip; lower paragraph_penalty to let a scene run across a paragraph break.
    scene_cost: float = Field(2.0, ge=0)
    paragraph_penalty: float = Field(120.0, ge=0)
    # Synthesize each sentence first and pack scenes on the measured lengths (exact scene sizes).
    measure_with_tts: bool = False

    @model_validator(mode="after")
    def _check(self):
        if not self.min_scene_duration <= self.target_scene_duration <= self.max_scene_duration:
            raise ValueError("planning durations must satisfy min <= target <= max")
        return self


class SyncConfig(_Strict):
    max_speedup: float = Field(1.15, ge=1.0, le=1.5)
    max_slowdown: float = Field(0.9, gt=0.5, le=1.0)
    # How narration and clip are synchronised when the clip has its own speech track:
    #   video_master: clip untouched, narration phrases placed/tempo-fitted onto the lips (audio_fit.py)
    #   audio_master: narration untouched, clip retimed phrase by phrase (lipalign.py)
    #   lip_voice:    clip untouched, narration re-synthesized chunk by chunk to the lips' timing
    mode: Literal["video_master", "audio_master", "lip_voice"] = "video_master"
    fit_lead: float = Field(0.25, ge=0, le=2)    # seconds of clip kept before the first spoken phrase
    fit_tail: float = Field(0.3, ge=0, le=2)     # seconds kept after the last
    # Phrase-level lip alignment against the clip's own speech track (services/lipalign.py).
    lip_align: bool = True
    clip_speech_threshold_db: float = Field(-30, le=-10, ge=-70)
    clip_min_pause: float = Field(0.2, gt=0, le=2)
    narration_speech_threshold_db: float = Field(-40, le=-10, ge=-70)
    narration_min_pause: float = Field(0.05, gt=0, le=2)


class ContinuityConfig(_Strict):
    enabled: bool = True
    anchor_last_frame: bool = True
    use_previous_clip_reference: bool = True


class RetryConfig(_Strict):
    max_attempts: int = Field(3, ge=1, le=10)
    backoff_seconds: list[float] = Field(default_factory=lambda: [5, 15, 45])


class InworldConfig(_Strict):
    model_id: str = "inworld-tts-2"
    speaking_rate: float = Field(1.0, ge=0.5, le=1.5)
    temperature: float = Field(1.0, gt=0, le=2)
    timeout_seconds: float = Field(120, gt=0)


class GeminiConfig(_Strict):
    model: str = "veo-3.1-generate-preview"
    resolution: Literal["720p", "1080p", "4k"] = "720p"
    include_dialogue_in_prompt: bool = True
    negative_prompt: str = ""
    person_generation: Optional[str] = None
    poll_interval_seconds: float = Field(10, gt=0)
    timeout_seconds: float = Field(900, gt=0)


class FlowConfig(_Strict):
    profile_dir: str = ".browser-profile"   # dedicated Chrome profile (signed in once by a person)
    debug_port: int = Field(9333, ge=1024, le=65535)
    clip_seconds: int = Field(8, ge=4, le=10)
    long_clip_seconds: int = Field(10, ge=4, le=10)    # used when the narration is long
    long_clip_after: float = Field(7.0, gt=0)           # narration seconds that need the long clip
    poll_seconds: int = Field(15, ge=5, le=120)
    generation_timeout_seconds: float = Field(1800, gt=60)


class QAConfig(_Strict):
    duration_tolerance: float = Field(0.1, ge=0)
    max_frozen_padding: float = Field(0.3, ge=0)


class PipelineConfig(_Strict):
    video: VideoConfig = VideoConfig()
    audio: AudioConfig = AudioConfig()
    generation: GenerationConfig = GenerationConfig()
    planning: PlanningConfig = PlanningConfig()
    sync: SyncConfig = SyncConfig()
    continuity: ContinuityConfig = ContinuityConfig()
    retry: RetryConfig = RetryConfig()
    inworld: InworldConfig = InworldConfig()
    gemini: GeminiConfig = GeminiConfig()
    flow: FlowConfig = FlowConfig()
    qa: QAConfig = QAConfig()


class ProjectConfig(_Strict):
    title: str = ""
    presenter: Optional[str] = None
    overrides: dict = Field(default_factory=dict)
