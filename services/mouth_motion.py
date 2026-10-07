"""When the presenter's mouth is actually moving, measured from the video frames.

Generated presenter clips share one fixed framing (the master image), so the mouth sits in
the same place in every clip. Frame-to-frame change inside a mouth box, minus the change in a
reference box on the face (so head motion and lighting flicker cancel out), gives a mouth-motion
curve. Its rise marks when talking starts on screen, independent of the clip's soundtrack.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# (x, y, w, h) in 1280x720 frames; derived from the presenter_01 master framing
MOUTH_BOX = (596, 248, 108, 60)
REFERENCE_BOX = (596, 150, 108, 50)   # eyes/forehead band: moves with the head, not with speech


def _frames(clip: Path, box: tuple[int, int, int, int], fps: int):
    import numpy as np
    x, y, w, h = box
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                          f"fps={fps},scale=1280:720,crop={w}:{h}:{x}:{y},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def motion_curve(clip: Path, fps: int = 24):
    """Per-frame mouth activity (smoothed), with timestamps."""
    import numpy as np
    mouth, ref = _frames(clip, MOUTH_BOX, fps), _frames(clip, REFERENCE_BOX, fps)
    n = min(len(mouth), len(ref))
    dm = np.abs(np.diff(mouth[:n], axis=0)).mean(axis=(1, 2))
    dr = np.abs(np.diff(ref[:n], axis=0)).mean(axis=(1, 2))
    act = np.clip(dm - 0.8 * dr, 0, None)
    kernel = np.ones(3) / 3
    act = np.convolve(act, kernel, mode="same")
    times = (np.arange(len(act)) + 1) / fps
    return times, act


def speaking_onset(clip: Path, near: float, window: float = 0.6, fps: int = 24) -> float | None:
    """Time the mouth starts moving, searched within +/- `window` s of `near` (e.g. the audio
    onset). None if no clear rise is found."""
    import numpy as np
    t, a = motion_curve(clip, fps)
    quiet = a[t < max(0.2, near - window)]
    base = float(np.median(quiet)) if len(quiet) else float(np.percentile(a, 20))
    peak = float(np.percentile(a, 90))
    if peak - base < 0.3:
        return None
    thr = base + 0.35 * (peak - base)
    # a real onset is a rise from a still mouth: at/above threshold now, below it for the
    # preceding ~0.12 s (an already-moving mouth at the window edge is not an onset)
    quiet_frames = max(2, int(round(0.12 * fps)))
    rises = [i for i in range(quiet_frames, len(a))
             if a[i] >= thr and all(a[i - k] < thr for k in range(1, quiet_frames + 1))
             and near - window <= t[i] <= near + window]
    if not rises:
        return None
    return float(t[min(rises, key=lambda i: abs(t[i] - near))])
