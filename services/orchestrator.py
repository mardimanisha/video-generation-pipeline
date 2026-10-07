"""Production orchestration: validate -> plan -> voice -> video -> sync -> concat -> QA.

State lives in scenes/scenes.json and is saved after every scene step, so a crash or a
failed scene never costs more than that one step. Every stage decides what to do from the
files on disk plus recorded fingerprints, which makes every run a safe "resume".
"""
from __future__ import annotations

import logging
import os
import shutil
import wave
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from services import audio_fit, ffmpeg, lipalign, qa
from services.audio_provider import AudioProvider, MockAudioProvider
from services.project_manager import ConfigError, Project
from services.scene_planner import load_template, plan_scenes, render_prompt
from services.subtitles import build_srt
from services.sync import MAX_CLIPS_PER_SCENE, compute_sync
from services.video_generator import (ClipRequest, GeminiVideoProvider, ManualClipMissing,
                                      LibraryVideoProvider, ManualVideoProvider, MockVideoProvider,
                                      VideoProvider)
from utils.duration import fmt_seconds
from utils.files import fingerprint, is_nonempty_file, sha256_file, sha256_text, write_json
from utils.logger import get_logger, log_event
from utils.retry import PermanentError, RateLimitError, with_retry

STEPS = 8


@dataclass
class RunOptions:
    replan: bool = False
    plan_only: bool = False
    regenerate_scene: str | None = None
    regenerate_stage: str = "video"  # audio | video | render
    max_clips: int | None = None     # generate at most this many new clips this run (testing / budgets)


class Printer:
    """Plain progress output that works in any Windows/Unix console."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def __call__(self, text: str = "") -> None:
        if not self.quiet:
            print(text, flush=True)

    def step(self, n: int, label: str, result: str) -> None:
        self(f"[{n}/{STEPS}] {label} {'.' * max(2, 28 - len(label))} {result}")

    def progress(self, n: int, label: str, done: int, total: int) -> None:
        """Counter updated in place on a terminal; a single final line when output is redirected."""
        if self.quiet:
            return
        line = f"[{n}/{STEPS}] {label} {'.' * max(2, 28 - len(label))} {done}/{total}"
        if sys.stdout.isatty():
            print("\r" + line, end="\n" if done >= total else "", flush=True)
        elif done >= total:
            print(line, flush=True)


# ---------------------------------------------------------------- providers
def make_audio_provider(name: str, project: Project) -> AudioProvider:
    cfg = project.config
    if name == "mock":
        return MockAudioProvider(cfg.planning.words_per_second, cfg.audio.sample_rate)
    if name == "inworld":
        from services.inworld import InworldAudioProvider
        return InworldAudioProvider(cfg.inworld, cfg.audio.sample_rate,
                                    language=project.presenter.voice.get("language", "en-US"))
    raise ConfigError(f"Unknown audio provider: {name}")


def make_video_provider(name: str, project: Project) -> VideoProvider:
    cfg = project.config
    if name == "mock":
        return MockVideoProvider(cfg.generation, cfg.video)
    if name == "manual":
        return ManualVideoProvider(
            cfg.generation,
            lambda out: project.find_video_segment(*_parse_segment_name(out.stem)))
    if name == "flow":
        from services.flow_browser import FlowBrowserVideoProvider
        return FlowBrowserVideoProvider(cfg.generation, cfg.flow, project.dir / "flow.json", project.root)
    if name == "library":
        return LibraryVideoProvider(cfg.generation, project.presenter.directory / "library", cfg.video)
    if name == "gemini":
        return GeminiVideoProvider(cfg.generation, cfg.gemini)
    raise ConfigError(f"Unknown video provider: {name}")


def _parse_segment_name(stem: str) -> tuple[str, int]:
    """'scene_004_b' -> ('scene_004', 1); 'scene_004' -> ('scene_004', 0)."""
    parts = stem.split("_")
    if len(parts) == 3 and len(parts[2]) == 1 and parts[2].isalpha():
        return f"{parts[0]}_{parts[1]}", ord(parts[2]) - ord("a")
    return stem, 0


def split_text_for_segments(text: str, durations: list[float]) -> list[str]:
    """Share a scene's words across its clips in proportion to clip length (for the
    'presenter says' part of each prompt)."""
    words = text.split()
    if len(durations) <= 1:
        return [text]
    total = sum(durations)
    out, start = [], 0
    for i, d in enumerate(durations):
        end = len(words) if i == len(durations) - 1 else start + round(len(words) * d / total)
        out.append(" ".join(words[start:end]))
        start = end
    return out


# ---------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, project: Project, *, audio_provider: str | AudioProvider = "inworld",
                 video_provider: str | VideoProvider = "gemini", printer: Printer | None = None,
                 sleep=time.sleep):
        self.project = project
        self.cfg = project.config
        self.print = printer or Printer()
        self.sleep = sleep
        self.log = get_logger(f"pipeline.{project.name}", project.log_file)
        self._audio_provider = audio_provider
        self._video_provider = video_provider
        self.state: dict = {}
        self.timings = {"audio_generation_time": 0.0, "video_generation_time": 0.0, "render_time": 0.0}

    # providers are built lazily so --review / --plan-only never need credentials
    @property
    def audio(self) -> AudioProvider:
        if isinstance(self._audio_provider, str):
            self._audio_provider = make_audio_provider(self._audio_provider, self.project)
        return self._audio_provider

    @property
    def video(self) -> VideoProvider:
        if isinstance(self._video_provider, str):
            self._video_provider = make_video_provider(self._video_provider, self.project)
        return self._video_provider

    @property
    def manual(self) -> bool:
        name = self._video_provider if isinstance(self._video_provider, str) else self._video_provider.name
        return name == "manual"

    @property
    def scenes(self) -> list[dict]:
        return self.state["scenes"]

    def save(self) -> None:
        self.state["updated"] = datetime.now().isoformat(timespec="seconds")
        self.project.save_state(self.state)

    def _retry(self, fn, scene_id: str, what: str, provider: str):
        r = self.cfg.retry

        def on_retry(attempt, exc, delay):
            self.print(f"\n  {scene_id} {what} failed.\n  Provider: {provider}\n  Attempt: {attempt}/{r.max_attempts}\n"
                       f"  Reason: {exc}\n  Action: Retrying in {delay:.0f} seconds...")
            log_event(self.log, "retry", logging.WARNING, scene=scene_id, stage=what, provider=provider,
                      attempt=attempt, error=str(exc), delay=delay)

        return with_retry(fn, max_attempts=r.max_attempts, schedule=r.backoff_seconds,
                          on_retry=on_retry, sleep=self.sleep)

    def _fail(self, scene: dict, stage: str, exc: Exception) -> None:
        scene["status"] = "failed"
        scene["error"] = f"{stage}: {exc}"
        self.save()
        log_event(self.log, "scene failed", logging.ERROR, scene=scene["id"], stage=stage, error=str(exc))
        tries = "" if isinstance(exc, PermanentError) else f" after {self.cfg.retry.max_attempts} attempts"
        self.print(f"\n  {scene['id']} {stage} failed{tries}.\n"
                   f"  Reason: {exc}\n  The remaining scenes will continue.\n"
                   f"  Retry with: python pipeline.py --project {self.project.name} "
                   f"--scene {scene['id']} --regenerate --stage {stage}")

    # ------------------------------------------------------------ main flow
    def run(self, opts: RunOptions | None = None) -> dict:
        opts = opts or RunOptions()
        started = time.monotonic()
        p = self.project
        p.ensure_dirs()
        log_event(self.log, "pipeline start", project=p.name, presenter=p.presenter.name,
                  audio_provider=str(self._audio_provider), video_provider=str(self._video_provider))
        self.print("=" * 45 + "\n AI PRESENTER VIDEO PIPELINE\n" + "=" * 45)
        self.print(f"Project:   {p.project_config.title or p.name}\nPresenter: {p.presenter.name}\n")

        self.validate_inputs()
        self.print.step(1, "Validating input", "OK")
        created = self.plan(replan=opts.replan)
        self.print.step(2, "Analyzing script", "OK")
        self.print.step(3, "Creating scenes", f"{len(self.scenes)} scenes" + (" (new)" if created else " (resumed)"))
        if opts.regenerate_scene:
            self.reset_scene(opts.regenerate_scene, opts.regenerate_stage)
        if opts.plan_only:
            self.print(f"\nPlan written to {p.scenes_file}. Review/edit visual_action, then run again.")
            return {"stopped": "plan_only"}
        for s in self.scenes:  # failed scenes get a fresh chance on every run
            if s["status"] == "failed":
                s["status"], s["error"] = "pending", None

        self.generate_audio()
        try:
            self.generate_video(max_clips=opts.max_clips)
        finally:
            close = getattr(self._video_provider, "close", None)
            if callable(close):
                close()
        self.render_scenes()
        result = self.finalize()
        total = time.monotonic() - started
        log_event(self.log, "pipeline end", total_seconds=round(total, 1), **result)
        self.print(f"\nTotal production time: {fmt_seconds(total)}")
        return result

    # ------------------------------------------------------------ [1] validate
    def validate_inputs(self) -> None:
        p = self.project
        problems = []
        ffmpeg.require_binaries()
        if not is_nonempty_file(p.script_file):
            problems.append(f"Script missing or empty: {p.script_file}")
        video_name = self._video_provider if isinstance(self._video_provider, str) else self._video_provider.name
        if video_name in ("gemini", "flow") and p.presenter.image is None:
            problems.append(f"Presenter image not found in {p.presenter.directory} (presenter.jpg/png)")
        audio_name = self._audio_provider if isinstance(self._audio_provider, str) else self._audio_provider.name
        if audio_name == "inworld" and not (p.presenter.voice.get("voice_id") or os.environ.get("INWORLD_VOICE_ID")) \
                and p.presenter.voice_sample is None:
            problems.append("No Inworld voice: add voice_sample.wav (10-30 s) or set voice_id / INWORLD_VOICE_ID")
        if problems:
            raise ConfigError("Input validation failed:\n  - " + "\n  - ".join(problems))

    # ------------------------------------------------------------ [2-3] plan
    def plan(self, replan: bool = False) -> bool:
        p = self.project
        script = p.script_file.read_text(encoding="utf-8")
        script_sha = sha256_text(script)
        existing = p.load_state()
        if existing and existing.get("script_sha256") == script_sha and not replan:
            self.state = existing
            return False
        if existing and not replan:
            raise ConfigError("script.txt changed since scenes were planned. Re-run with --replan "
                              "(audio/video of unchanged scenes is kept).")
        measure = self._measure_units if self.cfg.planning.measure_with_tts else None
        scenes = plan_scenes(script, self.cfg, p.presenter.profile, load_template(p.root), measure=measure)
        for s in scenes:
            s["text_sha"] = sha256_text(s["text"])
        if existing:
            self._carry_over(existing, scenes)
        self.state = {"project": p.name, "presenter": p.presenter.name, "script_sha256": script_sha,
                      "created": datetime.now().isoformat(timespec="seconds"), "timings": {}, "scenes": scenes}
        self.save()
        log_event(self.log, "scenes planned", count=len(scenes),
                  estimated_total=round(sum(s["estimated_duration"] for s in scenes), 1))
        return True

    def _carry_over(self, old_state: dict, scenes: list[dict]) -> None:
        """On --replan keep media for scenes whose text is unchanged; archive the rest."""
        p = self.project
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(p.scenes_file, p.scenes_dir / f"scenes.{stamp}.json")
        old = {s["id"]: s for s in old_state.get("scenes", [])}
        keep = set()
        for s in scenes:
            o = old.get(s["id"])
            if o and o.get("text_sha") == s["text_sha"]:
                for key in ("audio_file", "actual_audio_duration", "audio_sha", "clip_durations",
                            "video_files", "visual_action", "gesture", "generation_prompt", "custom_prompt"):
                    if key in o:
                        s[key] = o[key]
                keep.add(s["id"])
        archive = p.video_dir / "_stale" / stamp
        for f in list(p.video_dir.glob("scene_*.*")) + list(p.audio_dir.glob("scene_*.*")) + \
                list(p.rendered_dir.glob("scene_*.*")):
            if _parse_segment_name(f.stem)[0] not in keep:
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), archive / f"{f.parent.name}_{f.name}")

    def reset_scene(self, scene_id: str, stage: str) -> None:
        scene = next((s for s in self.scenes if s["id"] == scene_id), None)
        if scene is None:
            raise ConfigError(f"Unknown scene {scene_id}. Scenes: {self.scenes[0]['id']} .. {self.scenes[-1]['id']}")
        p = self.project
        archive = p.video_dir / "_replaced" / datetime.now().strftime("%Y%m%d-%H%M%S")
        victims = [p.rendered_path(scene_id)]
        if stage in ("video", "audio"):
            victims += [p.abs(v) for v in scene.get("video_files", [])]
            scene["video_files"] = []
        if stage == "audio":
            victims.append(p.audio_path(scene_id))
            raw = self._raw_audio_path(scene_id)
            if raw.is_file():
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(raw), archive / f"raw_{raw.name}")
            for text in scene.get("units") or []:
                unit = self._unit_path(text)
                if unit.is_file():
                    archive.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(unit), archive / f"unit_{unit.name}")
            scene["actual_audio_duration"] = None
            scene["audio_sha"] = None
        for f in victims:
            if f.is_file():  # paid clips are archived, not deleted
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), archive / f.name)
        scene.update(status="pending", error=None, rendered_file="", render_fingerprint=None)
        self.save()
        log_event(self.log, "scene reset", scene=scene_id, stage=stage)
        self.print(f"Regenerating {scene_id} from the {stage} stage.")

    # ------------------------------------------------------------ [4] audio
    def _voice_id(self) -> str:
        pres = self.project.presenter
        voice_id = os.environ.get("INWORLD_VOICE_ID", "").strip() or pres.voice.get("voice_id", "")
        if voice_id or self.audio.name == "mock":
            return voice_id or "mock-voice"
        sample = pres.voice_sample
        self.print(f"  Cloning voice once from {sample.name} ...")
        voice_id = self._retry(lambda: self.audio.clone_voice(sample, f"{pres.name} voice",
                                                              pres.voice.get("language", "en-US")),
                               "voice", "voice cloning", self.audio.name)
        pres.save_voice({**pres.voice, "voice_id": voice_id, "provider": self.audio.name,
                         "cloned_from": sample.name, "sample_sha256": sha256_file(sample),
                         "created": datetime.now().isoformat(timespec="seconds")})
        log_event(self.log, "voice cloned", voice_id=voice_id, presenter=pres.name)
        self.print(f"  Voice ID saved to {pres.voice_file}")
        return voice_id

    def _trim_signature(self) -> str:
        a = self.cfg.audio
        return f"{a.silence_threshold_db:g}|{a.keep_leading:g}|{a.keep_trailing:g}" if a.trim_silence else "off"

    def _raw_audio_path(self, scene_id: str) -> Path:
        return self.project.audio_dir / "raw" / f"{scene_id}.wav"

    # sentence-level narration cache (planning.measure_with_tts) ------------------------
    @property
    def _voice_key(self) -> str:
        """Identifies the voice that narration is recorded with; a change invalidates it."""
        if self.audio.name == "mock":
            return "mock"
        return (os.environ.get("INWORLD_VOICE_ID", "").strip()
                or self.project.presenter.voice.get("voice_id", "") or "unset")

    def _unit_path(self, text: str) -> Path:
        """Trimmed narration of one sentence, keyed by text so it survives re-planning."""
        return self.project.audio_dir / "units" / f"{sha256_text(self._voice_key + '|' + text)[:16]}.wav"

    def _ensure_unit(self, text: str, voice_id: str) -> Path:
        path = self._unit_path(text)
        if is_nonempty_file(path):
            return path
        raw = path.with_name(path.stem + ".raw.wav")
        self._retry(lambda: self.audio.generate(text, voice_id, raw), path.stem, "voice generation",
                    self.audio.name)
        if self.cfg.audio.trim_silence:
            ffmpeg.run(ffmpeg.build_trim_silence_cmd(raw, path, self.cfg.audio), timeout=120)
            raw.unlink()
        else:
            raw.replace(path)
        return path

    def _measure_units(self, sentences) -> list[float]:
        """Synthesize (or reuse) every sentence and return its real narration length."""
        self.project.ensure_dirs()
        missing = [s for s in sentences if not is_nonempty_file(self._unit_path(s.text))]
        voice_id = self._voice_id() if missing else ""
        durations = []
        for n, s in enumerate(sentences, 1):
            durations.append(ffmpeg.probe(self._ensure_unit(s.text, voice_id)).duration)
            self.print.progress(2, "Measuring narration", n, len(sentences))
        log_event(self.log, "sentences measured", count=len(sentences), synthesized=len(missing))
        return durations

    def _assemble_from_units(self, scene: dict, voice_id: str, out: Path) -> None:
        """Scene narration = its sentences joined by `sentence_gap` of silence."""
        gap = self.cfg.audio.sentence_gap
        out.parent.mkdir(parents=True, exist_ok=True)
        params, frames = None, []
        for i, text in enumerate(scene["units"]):
            with wave.open(str(self._ensure_unit(text, voice_id)), "rb") as w:
                if params is None:
                    params = w.getparams()
                elif w.getparams()[:3] != params[:3]:  # channels, sample width, rate
                    raise PermanentError(f"sentence audio format differs within {scene['id']}")
                if i:
                    silence_frames = int(gap * params.framerate)
                    frames.append(bytes(silence_frames * params.nchannels * params.sampwidth))
                frames.append(w.readframes(w.getnframes()))
        tmp = out.with_suffix(".part")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(params.nchannels)
            w.setsampwidth(params.sampwidth)
            w.setframerate(params.framerate)
            w.writeframes(b"".join(frames))
        tmp.replace(out)

    def _has_speech(self, scene: dict) -> bool:
        """Untrimmed TTS output for the current text exists, so no provider call is needed."""
        if scene.get("audio_sha") != scene["text_sha"] or scene.get("audio_voice") != self._voice_key:
            return False
        if scene.get("units"):
            return all(is_nonempty_file(self._unit_path(t)) for t in scene["units"])
        return is_nonempty_file(self._raw_audio_path(scene["id"])) or is_nonempty_file(
            self.project.audio_path(scene["id"]))

    def _audio_valid(self, scene: dict) -> bool:
        path = self.project.audio_path(scene["id"])
        return (is_nonempty_file(path) and scene.get("actual_audio_duration")
                and scene.get("audio_sha") == scene["text_sha"]
                and scene.get("audio_trim") == self._trim_signature()
                and scene.get("audio_voice") == self._voice_key)

    def _finish_audio(self, scene: dict) -> ffmpeg.MediaInfo:
        """raw TTS file -> audio/scene_XXX.wav with leading/trailing silence trimmed."""
        raw, path = self._raw_audio_path(scene["id"]), self.project.audio_path(scene["id"])
        if not is_nonempty_file(raw):  # audio made before trimming existed: keep it as the raw copy
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), raw)
        if self.cfg.audio.trim_silence:
            ffmpeg.run(ffmpeg.build_trim_silence_cmd(raw, path, self.cfg.audio), timeout=120)
        else:
            shutil.copy2(raw, path)
        info = ffmpeg.probe(path)
        if not info.has_audio or info.duration <= 0.1:
            raise PermanentError(f"audio is empty after silence trimming ({info.duration:.2f}s)")
        return info

    def generate_audio(self) -> None:
        todo = [s for s in self.scenes if not self._audio_valid(s)]
        done = len(self.scenes) - len(todo)
        label = "Generating voice"
        if not todo:
            self.print.step(4, label, f"{done}/{len(self.scenes)} (cached)")
            return
        voice_id = self._voice_id() if any(not self._has_speech(s) for s in todo) else ""
        t0 = time.monotonic()
        for scene in todo:
            sid = scene["id"]
            raw = self._raw_audio_path(sid)
            scene["status"] = "audio_generating"
            self.save()
            try:
                if scene.get("units"):
                    self._assemble_from_units(scene, voice_id, raw)
                elif not self._has_speech(scene):
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    self._retry(lambda: self.audio.generate(scene["text"], voice_id, raw), sid,
                                "voice generation", self.audio.name)
                info = self._finish_audio(scene)
            except Exception as exc:  # noqa: BLE001
                self._fail(scene, "audio", exc)
                continue
            path = self.project.audio_path(sid)
            target = info.duration + self.cfg.audio.tail_padding
            scene.update(audio_file=self.project.rel(path), actual_audio_duration=round(info.duration, 3),
                         audio_sha=scene["text_sha"], audio_trim=self._trim_signature(), audio_voice=self._voice_key,
                         status="audio_complete", error=None,
                         clip_durations=self.video.plan_durations(target, self.cfg.sync),
                         render_fingerprint=None)
            self.save()
            done += 1
            log_event(self.log, "audio generated", scene=sid, duration=scene["actual_audio_duration"],
                      estimated=scene["estimated_duration"])
            self.print.progress(4, label, done, len(self.scenes))
        self.timings["audio_generation_time"] += time.monotonic() - t0

    # ------------------------------------------------------------ [5] video
    def _prompt_for(self, scene: dict, segment_text: str) -> str:
        if scene.get("custom_prompt"):
            return scene["custom_prompt"]
        return render_prompt(load_template(self.project.root), self.project.presenter.profile,
                             scene["visual_action"], segment_text, self.cfg.gemini.include_dialogue_in_prompt)

    def _existing_segments(self, scene_id: str) -> list[Path]:
        segs = []
        for i in range(MAX_CLIPS_PER_SCENE):
            f = self.project.find_video_segment(scene_id, i)
            if f is None or not is_nonempty_file(f):
                break
            segs.append(f)
        return segs

    def _segment_request(self, scene: dict, index: int, duration: float, text: str) -> ClipRequest:
        p, cont = self.project, self.cfg.continuity
        ref = p.presenter.image
        last = None
        if cont.enabled and cont.anchor_last_frame and self.video.supports_last_frame:
            last = p.presenter.image  # every clip ends back on the master frame -> clean cuts
        elif cont.enabled and cont.use_previous_clip_reference and index > 0 and self.video.uses_reference_image:
            prev = p.find_video_segment(scene["id"], index - 1)
            if prev is not None:
                ref = ffmpeg.extract_frame(prev, p.frames_dir / f"{prev.stem}_last.jpg", last=True)
        return ClipRequest(prompt=self._prompt_for(scene, text), reference_image=ref, duration=duration,
                           output_path=p.video_path(scene["id"], index), last_frame_image=last,
                           negative_prompt=self.cfg.gemini.negative_prompt, label=scene["id"],
                           gesture=scene.get("gesture", ""), dialogue=text)

    def _write_manual_task(self, scene: dict) -> None:
        p = self.project
        durs = scene["clip_durations"]
        texts = split_text_for_segments(scene["text"], durs)
        for i, (d, t) in enumerate(zip(durs, texts)):
            name = p.video_path(scene["id"], i).name
            (p.prompts_dir / f"{p.video_path(scene['id'], i).stem}.txt").write_text(
                f"Save the generated clip as: video/{name}\nDuration: {d:g} seconds\n"
                f"Narration length: {scene['actual_audio_duration']:.2f} s\n\n{self._prompt_for(scene, t)}\n",
                encoding="utf-8")

    def generate_video(self, max_clips: int | None = None) -> None:
        ready = [s for s in self.scenes if s.get("actual_audio_duration") and s["status"] != "failed"]
        label = "Generating video" if not self.manual else "Collecting manual video"
        if ready:
            try:
                self.video  # build the provider once: a missing key or empty library is one clear error
            except PermanentError as exc:
                self.print.step(5, label, "UNAVAILABLE")
                self.print(f"\n  {exc}")
                log_event(self.log, "video provider unavailable", logging.ERROR, error=str(exc))
                return
        done, waiting, attempts, failures_in_a_row = 0, [], 0, 0
        t0 = time.monotonic()
        for scene in ready:
            needs_generation = not self.manual and not self._existing_segments(scene["id"])
            if needs_generation and max_clips is not None and attempts >= max_clips:
                self.print(f"\n  Clip limit reached ({max_clips} clip attempt(s) this run). "
                           f"Re-run to continue from {scene['id']}.")
                break
            if failures_in_a_row >= 3:
                self.print(f"\n  Stopping video generation: {failures_in_a_row} scenes failed in a row, so the "
                           f"provider is likely broken or blocked. Check the log, then re-run to resume.")
                break
            attempts += needs_generation
            try:
                ok = self._ensure_scene_video(scene)
            except RateLimitError as exc:
                # Still rate-limited after every retry: the quota is exhausted, so every other
                # scene would fail the same way. Stop here; a later run resumes from this scene.
                self._fail(scene, "video", exc)
                self.print(f"\n  Stopping video generation: {self.video.name} quota is exhausted.\n"
                           f"  Completed clips are kept. Re-run the same command once quota is available.")
                break
            except Exception as exc:  # noqa: BLE001
                self._fail(scene, "video", exc)
                failures_in_a_row += 1
                continue
            failures_in_a_row = 0
            if ok:
                done += 1
                self.print.progress(5, label, done, len(self.scenes))
            else:
                waiting.append(scene)
        self.timings["video_generation_time"] += time.monotonic() - t0
        if not ready:
            self.print.step(5, label, "0 (no audio yet)")
        if waiting:
            self._write_manual_index(waiting)

    def _ensure_scene_video(self, scene: dict) -> bool:
        """Make sure the scene has enough footage to cover its narration. Returns False when
        waiting on manual clips."""
        p, sid = self.project, scene["id"]
        segs = self._existing_segments(sid)
        if not segs and not self.manual:
            scene["clip_durations"] = self.video.plan_durations(
                scene["actual_audio_duration"] + self.cfg.audio.tail_padding, self.cfg.sync)
        durations = scene["clip_durations"]
        texts = split_text_for_segments(scene["text"], durations)

        if self.manual:
            if not segs:
                self._write_manual_task(scene)
                return False
        else:
            for i in range(len(segs), len(durations)):
                scene["status"] = "video_generating"
                self.save()
                req = self._segment_request(scene, i, durations[i], texts[i])
                self._retry(lambda: self.video.generate(req), sid, "video generation", self.video.name)
                log_event(self.log, "clip generated", scene=sid, segment=i, duration=durations[i])
            segs = self._existing_segments(sid)
            # Real clip lengths can differ from the request; add clips until the narration is covered.
            while len(segs) < MAX_CLIPS_PER_SCENE:
                plan = compute_sync(scene["actual_audio_duration"], [self._clip_len(f) for f in segs],
                                    self.cfg.sync, self.cfg.audio.tail_padding, self.cfg.video.fps)
                if plan.action != "insufficient":
                    break
                extra = self.video.plan_durations(plan.target_duration - plan.video_duration, self.cfg.sync)[0]
                req = self._segment_request(scene, len(segs), extra, scene["text"])
                self.print(f"\n  {sid}: footage {plan.video_duration:.1f}s < narration "
                           f"{plan.target_duration:.1f}s, generating an extra {extra:g}s clip")
                self._retry(lambda: self.video.generate(req), sid, "video generation", self.video.name)
                segs = self._existing_segments(sid)

        scene["video_files"] = [p.rel(f) for f in segs]
        scene["status"] = "video_complete"
        scene["error"] = None
        self.save()
        return True

    def _clip_len(self, path: Path) -> float:
        info = ffmpeg.probe(path)
        return info.video_duration or info.duration

    def _write_manual_index(self, waiting: list[dict]) -> None:
        p = self.project
        lines = [f"# Manual video tasks - {p.name}", "",
                 "Generate each clip in Gemini / Flow using the presenter image and the prompt file,",
                 "then save it into the project's `video/` folder with the exact name shown.",
                 "Re-run the pipeline afterwards; it picks the files up automatically.", ""]
        for s in waiting:
            for i, d in enumerate(s["clip_durations"]):
                stem = p.video_path(s["id"], i).stem
                lines.append(f"- [ ] `video/{stem}.mp4` - {d:g}s - narration {s['actual_audio_duration']:.2f}s "
                             f"- prompt: `video/prompts/{stem}.txt`")
        (p.dir / "MANUAL_VIDEO_TASKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.print(f"\n  {len(waiting)} scene(s) waiting for manual clips. See {p.dir / 'MANUAL_VIDEO_TASKS.md'}")

    # ------------------------------------------------------------ [6] render
    def _lip_voice(self, scene: dict, clip: Path):
        """lip_voice mode: re-synthesize the scene line chunk by chunk to the lips' timing.
        None when the clip's speech does not follow the script closely enough."""
        from services import lip_voice, transcribe
        s = self.cfg.sync
        words = transcribe.transcribe(clip, self.project.video_dir / "transcripts")
        al = transcribe.align(scene["text"], words)
        scene["lip_transcript"] = {"coverage": al.coverage, "extra": al.extra, "trailing_extra": al.trailing_extra,
                                   "said": " ".join(w.text for w in words)}
        if not al.usable:
            return None
        info = ffmpeg.probe(clip)
        speech = lipalign.speech_segments(clip, s.clip_speech_threshold_db, lip_voice.PHRASE_PAUSE,
                                          info.audio_duration or info.duration)
        from services import mouth_motion
        chunks = lip_voice.make_chunks(
            al, words, speech,
            mouth_onset=lambda near: mouth_motion.speaking_onset(clip, near, window=lip_voice.MOUTH_WINDOW))
        if not chunks:
            return None
        # a scene may name its own voice (e.g. a clone in another workspace when the main one is unusable)
        voice_id = scene.get("voice_override") or self._voice_id()
        work = self.project.audio_dir / "lipvoice" / scene["id"]
        trim = ffmpeg.trim_silence_filter(self.cfg.audio, keep_leading=0.02, keep_trailing=0.06)  # keep soft word edges
        # natural-speed take of each phrase first (cached; also the first try of the fit), so
        # the scene's pace can be evened out before fitting phrases to the lips
        natural = []
        for c in chunks:
            natural.append(self._retry(lambda c=c: lip_voice.natural_duration(c, self.audio, voice_id, work,
                                                                              self.cfg.audio.sample_rate, trim),
                                       scene["id"], "lip-timed voice", self.audio.name))
        lip_voice.smooth_pace(chunks, natural)
        for c in chunks:
            self._retry(lambda c=c: lip_voice.synthesize_chunk(c, self.audio, voice_id, work,
                                                                self.cfg.audio.sample_rate, trim),
                        scene["id"], "lip-timed voice", self.audio.name)
        plan = lip_voice.plan_window(chunks, info.video_duration or info.duration, s.fit_lead, s.fit_tail,
                                     self.cfg.video.fps, cuts=lip_voice.plan_cuts(speech, chunks))
        wav = lip_voice.assemble(plan, self.project.audio_dir / "fitted" / f"{scene['id']}.wav",
                                 self.cfg.audio.sample_rate)
        return plan, wav

    def _audio_fit(self, segs: list[Path], audio: Path):
        """Video-master plan: place narration phrases onto the clip's spoken phrases, or None
        when the clip has no usable speech track."""
        s = self.cfg.sync
        info = ffmpeg.probe(segs[0])
        if not info.has_audio:
            return None
        clip_len = info.video_duration or info.duration
        clip_speech = lipalign.speech_segments(segs[0], s.clip_speech_threshold_db, s.clip_min_pause,
                                               info.audio_duration or info.duration)
        narr_info = ffmpeg.probe(audio)
        narration = lipalign.speech_segments(audio, s.narration_speech_threshold_db, s.narration_min_pause,
                                             narr_info.duration)
        return audio_fit.plan_fit(narration, narr_info.duration, clip_speech, clip_len,
                                  lead=s.fit_lead, tail=s.fit_tail, fps=self.cfg.video.fps)

    def _lip_align(self, segs: list[Path], audio: Path, target: float):
        """Phrase-level alignment of the clip's own speech to the narration, or None to fall
        back to plain retime/trim (no clip audio, no detectable speech, or implausible match)."""
        s = self.cfg.sync
        if not s.lip_align:
            return None
        info = ffmpeg.probe(segs[0])
        if not info.has_audio:
            return None
        clip_speech = lipalign.speech_segments(segs[0], s.clip_speech_threshold_db, s.clip_min_pause,
                                               info.audio_duration or info.duration)
        narration = lipalign.speech_segments(audio, s.narration_speech_threshold_db, s.narration_min_pause)
        return lipalign.search_alignment(narration, clip_speech, target, info.video_duration or info.duration)

    def render_scenes(self) -> None:
        p, cfg = self.project, self.cfg
        ready = [s for s in self.scenes if s["status"] in ("video_complete", "syncing", "complete")]
        done = 0
        t0 = time.monotonic()
        for scene in ready:
            sid = scene["id"]
            audio = p.audio_path(sid)
            segs = [p.abs(v) for v in scene["video_files"]]
            fp = fingerprint(audio, *segs, extra=f"v{ffmpeg.RENDER_VERSION}{scene.get('voice_override') or ''}{scene.get('color_filter') or ''}"
                                                         + cfg.model_dump_json(include={"video", "audio", "sync"}))
            out = p.rendered_path(sid)
            if scene.get("render_fingerprint") == fp and is_nonempty_file(out):
                scene["status"] = "complete"
                done += 1
                continue
            scene["status"] = "syncing"
            self.save()
            try:
                plan = compute_sync(scene["actual_audio_duration"], [self._clip_len(f) for f in segs],
                                    cfg.sync, cfg.audio.tail_padding, cfg.video.fps)
                sync_info = plan.to_dict()
                fit = align = lipv = None
                if len(segs) == 1 and cfg.sync.mode == "lip_voice":
                    try:
                        lipv = self._lip_voice(scene, segs[0])
                        if lipv is None:  # clip does not say the line: best effort, and QA tells the user
                            sync_info["clip_off_script"] = True
                    except Exception as exc:  # noqa: BLE001 - e.g. voice service out of credits
                        lipv = None
                        sync_info["lip_voice_failed"] = str(exc)[:200]
                        self.print(f"\n  {sid}: lip-timed voice unavailable ({str(exc)[:80]}); "
                                   f"using the existing narration fitted to the lips instead")
                    if lipv is None:
                        fit = self._audio_fit(segs, audio)
                if lipv is not None:
                    pass
                elif fit is not None:
                    pass
                elif len(segs) == 1 and cfg.sync.mode == "video_master":
                    fit = self._audio_fit(segs, audio)
                    if fit is None:
                        align = self._lip_align(segs, audio, plan.target_duration)
                elif len(segs) == 1:
                    # narration master: retime the lips to the untouched narration; where the clip's
                    # pacing is too different for a natural video speed, fit the voice instead
                    align = self._lip_align(segs, audio, plan.target_duration)
                    if align is None:
                        fit = self._audio_fit(segs, audio)
                if lipv is not None:
                    lv_plan, voice_wav = lipv
                    cmd = ffmpeg.build_fit_render_cmd(segs[0], voice_wav, out, plan=lv_plan, video=cfg.video,
                                                      audio_cfg=cfg.audio, color_filter=scene.get("color_filter"))
                    sync_info.update(action="lip_voice", retime_factor=None,
                                     target_duration=round(lv_plan.duration, 4), freeze_pad=lv_plan.freeze_pad,
                                     lip_voice=lv_plan.to_dict(), fit={"window": lv_plan.to_dict()["window"]})
                elif fit is not None:
                    fitted = p.audio_dir / "fitted" / f"{sid}.wav"
                    audio_fit.render_fitted_narration(audio, fit, fitted, cfg.audio.sample_rate)
                    cmd = ffmpeg.build_fit_render_cmd(segs[0], fitted, out, plan=fit, video=cfg.video,
                                                      audio_cfg=cfg.audio, color_filter=scene.get("color_filter"))
                    sync_info.update(action="video_master", retime_factor=None, target_duration=round(fit.duration, 4),
                                     freeze_pad=fit.freeze_pad, fit=fit.to_dict())
                elif align is not None:
                    cmd = ffmpeg.build_aligned_render_cmd(
                        segs[0], audio, out, pieces=align.pieces, pre_hold=align.pre_hold,
                        post_hold=align.post_hold, target_duration=plan.target_duration,
                        video=cfg.video, audio_cfg=cfg.audio)
                    sync_info.update(action="lip_align", retime_factor=None,
                                     freeze_pad=round(align.pre_hold + align.post_hold, 3), lip=align.to_dict())
                else:
                    if plan.action == "insufficient":
                        self.print(f"\n  WARNING {sid}: footage {plan.video_duration:.1f}s is short for narration "
                                   f"{plan.target_duration:.1f}s; holding last frame {plan.freeze_pad:.2f}s. "
                                   f"Add video/{p.video_path(sid, len(segs)).name} to fix.")
                    cmd = ffmpeg.build_render_cmd(segs, audio, out, target_duration=plan.target_duration,
                                                  retime_factor=plan.retime_factor, video=cfg.video,
                                                  audio_cfg=cfg.audio, freeze_pad=plan.freeze_pad)
                ffmpeg.run(cmd)
                info = ffmpeg.probe(out)
            except Exception as exc:  # noqa: BLE001
                self._fail(scene, "render", exc)
                continue
            scene.update(sync=sync_info, rendered_file=p.rel(out), rendered_duration=round(info.duration, 3),
                         render_fingerprint=fp, status="complete", error=None)
            self.save()
            done += 1
            log_event(self.log, "scene rendered", scene=sid, **sync_info)
            self.print.progress(6, "Synchronizing", done, len(self.scenes))
        self.timings["render_time"] += time.monotonic() - t0
        if done < len(self.scenes) or not ready:
            self.print.step(6, "Synchronizing", f"{done}/{len(self.scenes)}")

    # ------------------------------------------------------------ [7-8] final
    def finalize(self) -> dict:
        p, cfg = self.project, self.cfg
        incomplete = [s for s in self.scenes if s["status"] != "complete"]
        report_path = p.output_dir / "production_report.json"
        cumulative = self.state.setdefault("timings", {})
        for k, v in self.timings.items():
            cumulative[k] = round(cumulative.get(k, 0) + v, 1)

        scene_qa = {s["id"]: qa.check_scene(s, p.dir, cfg) for s in self.scenes if s["status"] == "complete"}
        for s in self.scenes:
            if s["id"] in scene_qa:
                s["qa"] = scene_qa[s["id"]].to_dict()
        self.save()
        bad_qa = [sid for sid, r in scene_qa.items() if r.status == "fail"]

        final = p.output_dir / "final_video.mp4"
        final_qa = None
        if incomplete or bad_qa:
            self.print.step(7, "Rendering", "SKIPPED")
            self.print(f"\n{len(incomplete)} scene(s) not complete, {len(bad_qa)} failed QA:")
            for s in incomplete:
                self.print(f"  {s['id']}  {s['status']:<16} {s.get('error') or ''}")
            for sid in bad_qa:
                self.print(f"  {sid}  QA: {'; '.join(scene_qa[sid].errors)}")
            self.print(f"\nFix them, then run: python pipeline.py --project {p.name}")
        else:
            t0 = time.monotonic()
            rendered = [p.rendered_path(s["id"]) for s in self.scenes]
            manifest = p.output_dir / "concat.txt"
            manifest.write_text(ffmpeg.concat_manifest(rendered), encoding="utf-8")
            tmp = p.output_dir / "final_video.part.mp4"
            ffmpeg.run(ffmpeg.build_concat_cmd(manifest, tmp, cfg.audio))
            tmp.replace(final)
            srt = build_srt([(s["text"], s["actual_audio_duration"], s["rendered_duration"]) for s in self.scenes])
            (p.output_dir / "final_video.srt").write_text(srt, encoding="utf-8")
            cumulative["render_time"] = round(cumulative.get("render_time", 0) + time.monotonic() - t0, 1)
            log_event(self.log, "final video written", path=str(final))
            self.print.step(7, "Rendering", "OK")
            expected = sum(s["rendered_duration"] for s in self.scenes)
            final_qa = qa.check_final(final, expected, cfg)

        warn = sum(1 for r in scene_qa.values() if r.status == "warn")
        qa_label = "FAILED" if (bad_qa or (final_qa and final_qa.status == "fail")) else \
            (f"OK ({warn} warning(s))" if warn else ("OK" if final_qa else "partial"))
        self.print.step(8, "QA", qa_label)
        if final_qa and final_qa.errors:
            for e in final_qa.errors:
                self.print(f"  final: {e}")

        report = {
            "project": p.name,
            "presenter": p.presenter.name,
            "total_scenes": len(self.scenes),
            "successful_scenes": sum(1 for s in self.scenes if s["status"] == "complete"),
            "failed_scenes": sum(1 for s in self.scenes if s["status"] == "failed"),
            "pending_scenes": sum(1 for s in self.scenes if s["status"] not in ("complete", "failed")),
            "total_duration": round(sum(s.get("rendered_duration") or 0 for s in self.scenes), 2),
            "audio_generation_time": cumulative.get("audio_generation_time", 0),
            "video_generation_time": cumulative.get("video_generation_time", 0),
            "render_time": cumulative.get("render_time", 0),
            "final_file": p.rel(final) if final_qa else None,
            "final_qa": final_qa.to_dict() if final_qa else None,
            "scene_qa_warnings": {sid: r.warnings for sid, r in scene_qa.items() if r.warnings},
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(report_path, report)
        if final_qa and final_qa.status != "fail":
            self.print(f"\nFINAL VIDEO:\n{final}")
        return report
