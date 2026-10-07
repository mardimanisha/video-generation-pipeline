"""Audio-first synchronization planning.

The narration is the master timeline and is never cut. Video is fitted to it:
  1. video within [max_slowdown, max_speedup] of the target -> retime it (keeps the clip's
     start AND end pose, which matters because clips are anchored to the neutral master frame)
  2. video longer than that -> speed up by the maximum and trim the remainder
  3. video shorter than that -> the scene needs another clip ("insufficient"); a frozen last
     frame is only used as an explicit, QA-flagged fallback
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

from services.config import SyncConfig
from services.ffmpeg import frame_aligned

MAX_CLIPS_PER_SCENE = 4


@dataclass
class SyncPlan:
    target_duration: float   # frame-aligned narration length (+ tail padding) = rendered length
    video_duration: float    # sum of the scene's raw clip durations
    ratio: float             # video / target
    retime_factor: float     # PTS multiplier applied to the video (>1 = slower)
    action: str              # exact | retime | trim | insufficient
    freeze_pad: float        # seconds of frozen last frame needed (only when insufficient)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _in_range(ratio: float, cfg: SyncConfig) -> bool:
    return cfg.max_slowdown - 1e-9 <= ratio <= cfg.max_speedup + 1e-9


def plan_clip_durations(target: float, allowed: list[float], cfg: SyncConfig) -> list[float]:
    """Choose the fewest clips (from the provider's allowed durations) that can cover `target`.

    Among equal clip counts, prefer a combination that only needs a gentle retime, closest to
    real time; otherwise the shortest combination that is long enough (it will be trimmed).
    """
    if target <= 0:
        raise ValueError("target duration must be positive")
    allowed = sorted(set(allowed))
    for n in range(1, MAX_CLIPS_PER_SCENE + 1):
        combos = list(itertools.combinations_with_replacement(allowed, n))
        fitting = [c for c in combos if _in_range(sum(c) / target, cfg)]
        if fitting:
            best = min(fitting, key=lambda c: (abs(math.log(sum(c) / target)), sum(c)))
            return sorted(best, reverse=True)
        long_enough = [c for c in combos if sum(c) / target > cfg.max_speedup]
        if long_enough:
            best = min(long_enough, key=sum)
            return sorted(best, reverse=True)
    raise ValueError(f"{target:.1f}s cannot be covered by {MAX_CLIPS_PER_SCENE} clips of {allowed}s; "
                     "split the scene text or raise max_clip_duration")


def compute_sync(audio_duration: float, clip_durations: list[float], cfg: SyncConfig,
                 tail_padding: float, fps: int) -> SyncPlan:
    target = frame_aligned(audio_duration + tail_padding, fps)
    video = float(sum(clip_durations))
    if video <= 0:
        return SyncPlan(target, 0.0, 0.0, 1.0, "insufficient", target)
    ratio = video / target
    if abs(ratio - 1) < 0.005:
        return SyncPlan(target, video, ratio, 1.0, "exact", 0.0)
    if _in_range(ratio, cfg):
        return SyncPlan(target, video, ratio, target / video, "retime", 0.0)
    if ratio > cfg.max_speedup:
        return SyncPlan(target, video, ratio, 1 / cfg.max_speedup, "trim", 0.0)
    factor = 1 / cfg.max_slowdown
    freeze = max(0.0, target - video * factor)
    return SyncPlan(target, video, ratio, factor, "insufficient", round(freeze + 1 / fps, 4))
