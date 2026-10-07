"""Project layout, configuration loading, presenter resolution and scene state persistence.

Everything a project produces lives under projects/<name>/, with deterministic file names,
so projects never share or overwrite each other's assets.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from services.config import PipelineConfig, ProjectConfig
from utils.files import deep_merge, read_json, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTS = (".wav", ".mp3")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")

SCENE_STATUSES = (
    "pending",
    "audio_generating",
    "audio_complete",
    "video_generating",
    "video_complete",
    "syncing",
    "complete",
    "failed",
)


class ConfigError(Exception):
    pass


def _find(directory: Path, stem: str, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def segment_suffix(index: int) -> str:
    """0 -> '', 1 -> '_b', 2 -> '_c' ... Extra clips for one scene sort after the first."""
    if index == 0:
        return ""
    if index > 25:
        raise ValueError("too many clips for one scene")
    return "_" + string.ascii_lowercase[index]


@dataclass
class Presenter:
    name: str
    directory: Path
    image: Path | None
    voice_sample: Path | None
    profile: dict
    voice: dict
    voice_file: Path

    def save_voice(self, voice: dict) -> None:
        self.voice = voice
        write_json(self.voice_file, voice)


class Project:
    def __init__(self, name: str, root: Path = REPO_ROOT):
        self.name = name
        self.root = root
        self.dir = root / "projects" / name
        self.input_dir = self.dir / "input"
        self.scenes_dir = self.dir / "scenes"
        self.audio_dir = self.dir / "audio"
        self.video_dir = self.dir / "video"
        self.prompts_dir = self.video_dir / "prompts"
        self.frames_dir = self.dir / "frames"
        self.rendered_dir = self.dir / "rendered"
        self.logs_dir = self.dir / "logs"
        self.output_dir = self.dir / "output"
        self.scenes_file = self.scenes_dir / "scenes.json"
        self.script_file = self.input_dir / "script.txt"
        self.log_file = self.logs_dir / "pipeline.log"

        if not self.dir.is_dir():
            raise ConfigError(f"Project folder not found: {self.dir}")
        self.project_config = self._load_project_config()
        self.config = self._load_pipeline_config()
        self.presenter = self._resolve_presenter()

    # ---------- configuration ----------
    def _load_project_config(self) -> ProjectConfig:
        raw = read_json(self.dir / "config.json", default={})
        try:
            return ProjectConfig(**raw)
        except ValidationError as exc:
            raise ConfigError(f"{self.dir / 'config.json'} is invalid:\n{exc}") from exc

    def _load_pipeline_config(self) -> PipelineConfig:
        default_file = self.root / "config" / "default.json"
        base = read_json(default_file, default={})
        merged = deep_merge(base, self.project_config.overrides)
        try:
            return PipelineConfig(**merged)
        except ValidationError as exc:
            raise ConfigError(f"Configuration invalid (config/default.json + project overrides):\n{exc}") from exc

    def _resolve_presenter(self) -> Presenter:
        default_profile = read_json(self.root / "config" / "presenter.json", default={})
        default_voice = read_json(self.root / "config" / "voice.json", default={})
        default_profile.pop("_comment", None)
        default_voice.pop("_comment", None)

        name = self.project_config.presenter
        if name:
            pdir = self.root / "presenters" / name
            if not pdir.is_dir():
                raise ConfigError(f"Presenter '{name}' not found at {pdir}")
            profile_file, voice_file, media_dir = pdir / "presenter.json", pdir / "voice.json", pdir
        else:
            name = f"{self.name} (project-local)"
            pdir = self.dir
            profile_file, voice_file, media_dir = self.dir / "presenter.json", self.dir / "voice.json", self.input_dir

        profile = deep_merge(default_profile, read_json(profile_file, default={}))
        voice = deep_merge(default_voice, read_json(voice_file, default={}))
        ref = profile.get("identity", {}).get("reference_image") or "presenter.jpg"
        image = media_dir / ref if (media_dir / ref).is_file() else _find(media_dir, "presenter", IMAGE_EXTS)
        return Presenter(
            name=name,
            directory=pdir,
            image=image,
            voice_sample=_find(media_dir, "voice_sample", AUDIO_EXTS),
            profile=profile,
            voice=voice,
            voice_file=voice_file,
        )

    # ---------- paths ----------
    def ensure_dirs(self) -> None:
        for d in (self.scenes_dir, self.audio_dir, self.video_dir, self.prompts_dir,
                  self.frames_dir, self.rendered_dir, self.logs_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    def audio_path(self, scene_id: str) -> Path:
        return self.audio_dir / f"{scene_id}.wav"

    def video_path(self, scene_id: str, segment: int = 0, ext: str = ".mp4") -> Path:
        return self.video_dir / f"{scene_id}{segment_suffix(segment)}{ext}"

    def find_video_segment(self, scene_id: str, segment: int) -> Path | None:
        """Find a clip regardless of container (manual downloads may be .mov/.webm)."""
        return _find(self.video_dir, f"{scene_id}{segment_suffix(segment)}", VIDEO_EXTS)

    def rendered_path(self, scene_id: str) -> Path:
        return self.rendered_dir / f"{scene_id}.mp4"

    def rel(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.resolve().relative_to(self.dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    def abs(self, rel_path: str) -> Path:
        return self.dir / rel_path

    # ---------- scene state ----------
    def load_state(self) -> dict | None:
        return read_json(self.scenes_file)

    def save_state(self, state: dict) -> None:
        write_json(self.scenes_file, state)
