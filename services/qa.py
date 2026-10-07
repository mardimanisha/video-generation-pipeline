"""Automated quality checks for scenes and the final video."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from services import ffmpeg
from services.config import PipelineConfig
from utils.files import is_nonempty_file

REQUIRED_SCENE_KEYS = ("id", "text", "status", "visual_action")


@dataclass
class QAResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "fail" if self.errors else ("warn" if self.warnings else "ok")

    def to_dict(self) -> dict:
        return {"status": self.status, "errors": self.errors, "warnings": self.warnings}


def _check_video_format(info: ffmpeg.MediaInfo, cfg: PipelineConfig, res: QAResult, label: str) -> None:
    v = cfg.video
    if not info.has_video:
        res.errors.append(f"{label}: no video stream")
        return
    if (info.width, info.height) != (v.width, v.height):
        res.errors.append(f"{label}: resolution {info.width}x{info.height}, expected {v.width}x{v.height}")
    if info.fps is None or abs(info.fps - v.fps) > 0.01:
        res.errors.append(f"{label}: fps {info.fps}, expected {v.fps}")
    if not info.has_audio:
        res.errors.append(f"{label}: no audio stream")
    elif info.sample_rate != cfg.audio.sample_rate:
        res.errors.append(f"{label}: audio {info.sample_rate} Hz, expected {cfg.audio.sample_rate} Hz")


def _check_clip_not_cut(scene: dict, project_dir: Path, res: QAResult) -> None:
    """The generated clip must not end while the presenter is still speaking: that cuts a word
    off mid-way on screen. Short blips at the very end (clicks, breaths) do not count."""
    from services import lipalign
    for rel in scene.get("video_files") or []:
        clip = project_dir / rel
        try:
            ci = ffmpeg.probe(clip)
            if not ci.has_audio:
                continue
            speech = [g for g in lipalign.speech_segments(clip, -30, 0.2, ci.audio_duration) if g[1] - g[0] >= 0.25]
        except Exception:  # noqa: BLE001 - QA must not crash on an odd clip
            continue
        clip_len = ci.video_duration or ci.duration
        sync = scene.get("sync") or {}
        window = (sync.get("fit") or {}).get("window") or (sync.get("lip") or {}).get("clip_used")
        if window and window[1] < clip_len - 0.15:
            continue  # the scene ends before the clip does: anything said at the clip's end is unused
        if speech and clip_len - speech[-1][1] < 0.15:
            res.errors.append(f"speech cut off by the end of the clip ({rel}); regenerate this scene "
                              f"(long lines get a longer clip)")


def check_scene(scene: dict, project_dir: Path, cfg: PipelineConfig) -> QAResult:
    res = QAResult()
    missing = [k for k in REQUIRED_SCENE_KEYS if not scene.get(k)]
    if missing:
        res.errors.append(f"metadata missing: {', '.join(missing)}")
    tol = cfg.qa.duration_tolerance

    audio = project_dir / scene["audio_file"] if scene.get("audio_file") else None
    if not is_nonempty_file(audio):
        res.errors.append("audio file missing or empty")
    else:
        try:
            ainfo = ffmpeg.probe(audio)
            if not ainfo.has_audio or ainfo.duration < 0.2:
                res.errors.append(f"audio unreadable or too short ({ainfo.duration:.2f}s)")
            elif scene.get("actual_audio_duration") and abs(ainfo.duration - scene["actual_audio_duration"]) > tol:
                res.errors.append("audio file changed since its duration was measured; re-run to re-sync")
        except ffmpeg.FFmpegError as exc:
            res.errors.append(f"audio unreadable: {exc}")

    rendered = project_dir / scene["rendered_file"] if scene.get("rendered_file") else None
    if not is_nonempty_file(rendered):
        res.errors.append("rendered clip missing or empty")
        return res
    try:
        info = ffmpeg.probe(rendered)
    except ffmpeg.FFmpegError as exc:
        res.errors.append(f"rendered clip unreadable: {exc}")
        return res
    _check_video_format(info, cfg, res, "rendered")
    _check_clip_not_cut(scene, project_dir, res)
    audio_dur = scene.get("actual_audio_duration") or 0
    sync = scene.get("sync") or {}
    if sync.get("action") == "lip_voice":
        audio_dur = 0  # narration was re-synthesized to the lips; its length is the scene's
    if sync.get("lip_voice_failed"):
        res.warnings.append("voice could not be re-timed to the lips (voice service error); this scene uses "
                            "the fitted narration. Re-run with --scene <id> --regenerate --stage render later")
    if sync.get("clip_off_script"):
        res.warnings.append("clip does not say the script line closely (repeated/garbled words); "
                            "regenerate this scene's video for good lip sync")
    if sync.get("action") == "video_master":
        # narration was tempo-fitted onto the lips: all of it is used, at a changed speed
        fit = sync.get("fit") or {}
        tempos = fit.get("tempos") or [1.0]
        audio_dur = audio_dur / max(tempos)
    if info.duration + tol < audio_dur:
        res.errors.append(f"rendered {info.duration:.2f}s is shorter than narration {audio_dur:.2f}s (speech would be cut)")
    # Scenes are rendered frame-exact; anything beyond ~1 frame of A/V difference becomes a
    # visible gap or drift once scenes are concatenated.
    if info.video_duration and info.audio_duration and \
            abs(info.video_duration - info.audio_duration) > 1.5 / cfg.video.fps:
        res.errors.append(f"A/V length mismatch: video {info.video_duration:.2f}s vs audio {info.audio_duration:.2f}s")
    if sync.get("freeze_pad", 0) > cfg.qa.max_frozen_padding:
        res.warnings.append(f"{sync['freeze_pad']:.2f}s frozen frame at end (video clip too short)")
    if sync.get("action") == "trim" and sync.get("ratio", 1) > 1.6:
        res.warnings.append(f"video was {sync['ratio']:.2f}x longer than narration; much of the clip was trimmed")
    return res


def check_final(path: Path, expected_duration: float, cfg: PipelineConfig) -> QAResult:
    res = QAResult()
    if not is_nonempty_file(path):
        res.errors.append("final video missing or empty")
        return res
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        res.errors.append(f"final video unreadable: {exc}")
        return res
    _check_video_format(info, cfg, res, "final")
    # Allow a little slack per scene for container/AAC frame rounding.
    if abs(info.duration - expected_duration) > max(cfg.qa.duration_tolerance, 0.5):
        res.errors.append(f"final duration {info.duration:.2f}s, expected {expected_duration:.2f}s")
    if info.video_duration and info.audio_duration and abs(info.video_duration - info.audio_duration) > 0.15:
        res.errors.append(f"final A/V drift: video {info.video_duration:.2f}s vs audio {info.audio_duration:.2f}s")
    return res
