"""FFmpeg/ffprobe wrapper. Commands are built as argument lists (never shell strings) so file
names and script text can never be interpreted by a shell. Builders are pure functions and
unit-testable; `run` executes them."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from services.config import AudioConfig, VideoConfig


# Bump when build_render_cmd changes output, so cached scene renders are redone on next run.
RENDER_VERSION = 14


class FFmpegError(Exception):
    pass


@dataclass
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    vcodec: str | None = None
    acodec: str | None = None
    sample_rate: int | None = None
    video_duration: float | None = None
    audio_duration: float | None = None


def require_binaries() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise FFmpegError(f"{exe} not found on PATH. Install FFmpeg: winget install Gyan.FFmpeg")


def run(cmd: list[str], timeout: float | None = 1800) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{cmd[0]} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{cmd[0]} failed (exit {proc.returncode}):\n{tail}")
    return proc


def _ratio(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den) if float(den) else None
    return float(value)


def probe(path: Path) -> MediaInfo:
    if not Path(path).is_file():
        raise FFmpegError(f"File not found: {path}")
    proc = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=60)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt_dur = _ratio(data.get("format", {}).get("duration"))

    def sdur(s):
        return _ratio(s.get("duration")) if s else None

    return MediaInfo(
        duration=fmt_dur or sdur(v) or sdur(a) or 0.0,
        has_video=v is not None,
        has_audio=a is not None,
        width=v.get("width") if v else None,
        height=v.get("height") if v else None,
        fps=_ratio(v.get("avg_frame_rate")) or _ratio(v.get("r_frame_rate")) if v else None,
        vcodec=v.get("codec_name") if v else None,
        acodec=a.get("codec_name") if a else None,
        sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        video_duration=sdur(v),
        audio_duration=sdur(a),
    )


def frame_aligned(seconds: float, fps: int) -> float:
    """Round a duration UP to a whole number of frames so audio and video lengths match exactly."""
    return math.ceil(round(seconds * fps, 6)) / fps


def _scale_pad(v: VideoConfig) -> str:
    return (f"scale={v.width}:{v.height}:force_original_aspect_ratio=decrease,"
            f"pad={v.width}:{v.height}:(ow-iw)/2:(oh-ih)/2,setsar=1")


def build_render_cmd(
    segments: list[Path],
    audio: Path,
    output: Path,
    *,
    target_duration: float,
    retime_factor: float,
    video: VideoConfig,
    audio_cfg: AudioConfig,
    freeze_pad: float = 0.0,
) -> list[str]:
    """One encode per scene: concatenate the scene's clips, retime by `retime_factor`
    (PTS multiplier: >1 slows down, <1 speeds up), normalize resolution/fps/pixel format,
    drop the generator's audio, and mux the narration padded/trimmed to exactly
    `target_duration` (which must be frame-aligned)."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    for seg in segments:
        cmd += ["-i", str(seg)]
    cmd += ["-i", str(audio)]
    n = len(segments)
    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS,{_scale_pad(video)},fps={video.fps},format={video.pix_fmt}[v{i}]")
    joined = "".join(f"[v{i}]" for i in range(n))
    chain = f"{joined}concat=n={n}:v=1:a=0," if n > 1 else joined
    # The fps filter emits nothing past the last source frame's timestamp, so a retimed clip can
    # end a frame or two early. Always hold the last frame briefly; trim then cuts to the exact,
    # frame-aligned target so video and narration have identical lengths.
    hold = freeze_pad + 4 / video.fps
    post = (f"setpts={retime_factor:.6f}*(PTS-STARTPTS),fps={video.fps},"
            f"tpad=stop_mode=clone:stop_duration={hold:.3f},"
            f"trim=duration={target_duration:.6f},setpts=PTS-STARTPTS[vout]")
    parts.append(chain + post)
    layout = "mono" if audio_cfg.channels == 1 else "stereo"
    parts.append(
        f"[{n}:a]aresample={audio_cfg.sample_rate},aformat=channel_layouts={layout},"
        f"apad,atrim=duration={target_duration:.6f},asetpts=PTS-STARTPTS[aout]"
    )
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", video.codec, "-preset", video.preset, "-crf", str(video.crf), "-pix_fmt", video.pix_fmt,
        "-r", str(video.fps),
        "-c:a", audio_cfg.codec, "-b:a", audio_cfg.bitrate, "-ar", str(audio_cfg.sample_rate),
        "-ac", str(audio_cfg.channels),
        "-movflags", "+faststart",
        str(output),
    ]
    return cmd


def _narration_chain(audio_input: int, target_duration: float, audio_cfg: AudioConfig) -> str:
    layout = "mono" if audio_cfg.channels == 1 else "stereo"
    return (f"[{audio_input}:a]aresample={audio_cfg.sample_rate},aformat=channel_layouts={layout},"
            f"apad,atrim=duration={target_duration:.6f},asetpts=PTS-STARTPTS[aout]")


def _output_args(output: Path, video: VideoConfig, audio_cfg: AudioConfig) -> list[str]:
    return ["-map", "[vout]", "-map", "[aout]",
            "-c:v", video.codec, "-preset", video.preset, "-crf", str(video.crf), "-pix_fmt", video.pix_fmt,
            "-r", str(video.fps),
            "-c:a", audio_cfg.codec, "-b:a", audio_cfg.bitrate, "-ar", str(audio_cfg.sample_rate),
            "-ac", str(audio_cfg.channels), "-movflags", "+faststart", str(output)]


def build_aligned_render_cmd(clip: Path, audio: Path, output: Path, *, pieces, pre_hold: float,
                             post_hold: float, target_duration: float, video: VideoConfig,
                             audio_cfg: AudioConfig) -> list[str]:
    """Render one clip through a piecewise time map (see services/lipalign.py): each piece
    (clip c0..c1 -> output t0..t1) is cut and retimed, pieces are joined, then the narration
    is muxed. The clip's own audio is dropped."""
    n = len(pieces)
    parts = [f"[0:v]setpts=PTS-STARTPTS,{_scale_pad(video)},format={video.pix_fmt}"
             + (f",split={n}" + "".join(f"[s{i}]" for i in range(n)) if n > 1 else "[s0]")]
    for i, p in enumerate(pieces):
        parts.append(f"[s{i}]trim=start={p.c0:.4f}:end={p.c1:.4f},"
                     f"setpts={p.factor:.6f}*(PTS-STARTPTS)[p{i}]")
    joined = "".join(f"[p{i}]" for i in range(n))
    chain = f"{joined}concat=n={n}:v=1:a=0," if n > 1 else joined
    post = f"fps={video.fps}"
    if pre_hold > 0:
        post += f",tpad=start_mode=clone:start_duration={pre_hold:.4f}"
    post += (f",tpad=stop_mode=clone:stop_duration={post_hold + 4 / video.fps:.4f},"
             f"trim=duration={target_duration:.6f},setpts=PTS-STARTPTS[vout]")
    parts.append(chain + post)
    parts.append(_narration_chain(1, target_duration, audio_cfg))
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(clip), "-i", str(audio),
            "-filter_complex", ";".join(parts), *_output_args(output, video, audio_cfg)]


def build_fit_render_cmd(clip: Path, fitted_audio: Path, output: Path, *, plan, video: VideoConfig,
                         audio_cfg: AudioConfig, color_filter: str | None = None) -> list[str]:
    """Video-master render (services/audio_fit.py): the clip is only trimmed to plan.window,
    never retimed; `fitted_audio` is the narration already placed onto the lips."""
    w0, w1 = plan.window
    dur = plan.duration
    # ranges cut out of the clip (lip_voice: repeated words, overlong pauses), relative to w0
    cuts = getattr(plan, "cuts", None) or []
    drop = ""
    if cuts:
        expr = "+".join(f"between(t,{a - w0:.4f},{b - w0 - 0.5 / video.fps:.4f})" for a, b in cuts)
        drop = f"select='not({expr})',setpts=N/({video.fps}*TB),"
    # per-scene steady look (tools/color_match.py); applied before cuts so its time is clip time - w0
    color = color_filter.replace("{w0}", f"{w0:.4f}") + "," if color_filter else ""
    vchain = (f"[0:v]setpts=PTS-STARTPTS,{_scale_pad(video)},fps={video.fps},{color}{drop}format={video.pix_fmt},"
              f"tpad=stop_mode=clone:stop_duration={plan.freeze_pad + 4 / video.fps:.4f},"
              f"trim=duration={dur:.6f},setpts=PTS-STARTPTS[vout]")
    span = w1 - w0 if cuts else dur
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{w0:.4f}", "-t", f"{span + 0.5:.4f}",
            "-i", str(clip), "-i", str(fitted_audio),
            "-filter_complex", vchain + ";" + _narration_chain(1, dur, audio_cfg),
            *_output_args(output, video, audio_cfg)]


def trim_silence_filter(audio_cfg: AudioConfig, keep_leading: float | None = None,
                        keep_trailing: float | None = None) -> str:
    thr = f"{audio_cfg.silence_threshold_db:g}dB"
    lead = audio_cfg.keep_leading if keep_leading is None else keep_leading
    trail = audio_cfg.keep_trailing if keep_trailing is None else keep_trailing
    return (f"silenceremove=start_periods=1:start_threshold={thr}:start_silence={lead:.3f},"
            f"areverse,"
            f"silenceremove=start_periods=1:start_threshold={thr}:start_silence={trail:.3f},"
            f"areverse")


def build_trim_silence_cmd(src: Path, dst: Path, audio_cfg: AudioConfig) -> list[str]:
    """Trim leading/trailing silence to fixed small amounts; silence inside the speech is kept."""
    af = trim_silence_filter(audio_cfg)
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(src), "-af", af,
            "-c:a", "pcm_s16le", str(dst)]


def concat_manifest(files: list[Path]) -> str:
    """Manifest for the concat demuxer. Paths are absolute with single quotes escaped."""
    lines = ["ffconcat version 1.0"]
    for f in files:
        p = Path(f).resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{p}'")
    return "\n".join(lines) + "\n"


def build_concat_cmd(manifest: Path, output: Path, audio_cfg: AudioConfig) -> list[str]:
    """Scenes are already encoded identically, so video is stream-copied (no second video
    encode). Audio is re-encoded once across the whole timeline to avoid AAC priming gaps
    at every scene boundary."""
    return [
        "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy",
        "-c:a", audio_cfg.codec, "-b:a", audio_cfg.bitrate, "-ar", str(audio_cfg.sample_rate),
        "-movflags", "+faststart",
        str(output),
    ]


def build_extract_frame_cmd(video: Path, output: Path, *, last: bool) -> list[str]:
    if last:
        return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-sseof", "-0.25", "-i", str(video),
                "-update", "1", "-q:v", "2", str(output)]
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)]


def extract_frame(video: Path, output: Path, *, last: bool = True) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(build_extract_frame_cmd(video, output, last=last), timeout=120)
    if not output.is_file():
        raise FFmpegError(f"Could not extract frame from {video}")
    return output


def build_test_tone_cmd(duration: float, output: Path, sample_rate: int = 48000) -> list[str]:
    """Deterministic stand-in narration for mock runs: a soft tone with a 1 kHz pip each second."""
    expr = f"0.2*sin(2*PI*220*t)+0.3*sin(2*PI*1000*t)*lt(mod(t\\,1)\\,0.05)"
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"aevalsrc={expr}:s={sample_rate}:d={duration:.3f}",
            "-ac", "1", "-c:a", "pcm_s16le", str(output)]


def _find_font() -> str | None:
    """drawtext needs an explicit font on Windows builds without fontconfig."""
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if Path(candidate).is_file():
            return candidate.replace(":", "\\:")  # ':' must be escaped inside a filter argument
    return None


def build_test_clip_cmd(image: Path | None, duration: float, output: Path, video: VideoConfig, label: str) -> list[str]:
    """Stand-in 'generated' clip for mock runs: the presenter image with a slow, subtle push-in,
    a running timestamp, and an unwanted audio track (to prove the pipeline strips it)."""
    frames = max(1, round(duration * video.fps))
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in "_-")
    font = _find_font()
    text = "null" if font is None else (
        f"drawtext=fontfile='{font}':text='{safe_label} %{{pts\\:hms}}':x=20:y=20:fontsize=28:"
        f"fontcolor=white:box=1:boxcolor=black@0.5")
    if image is not None:
        src = ["-loop", "1", "-framerate", str(video.fps), "-t", f"{duration:.3f}", "-i", str(image)]
        vf = (f"scale={video.width}:{video.height}:force_original_aspect_ratio=increase,"
              f"crop={video.width}:{video.height},"
              f"zoompan=z='1+0.03*on/{frames}':d=1:s={video.width}x{video.height}:fps={video.fps},{text}")
    else:
        src = ["-f", "lavfi", "-i", f"testsrc2=s={video.width}x{video.height}:r={video.fps}:d={duration:.3f}"]
        vf = text
    return ["ffmpeg", "-hide_banner", "-v", "error", "-y", *src,
            # "speech" like a real generated clip: quiet lead-in, talking, quiet tail
            "-f", "lavfi", "-i",
            f"aevalsrc=0.4*sin(2*PI*440*t)*between(t\\,{min(0.4, duration * 0.1):.3f}\\,"
            f"{max(duration * 0.5, duration - 0.6):.3f}):s=48000:d={duration:.3f}",
            "-vf", vf, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output)]
