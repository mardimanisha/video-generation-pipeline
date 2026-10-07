"""End-to-end: script -> scenes -> mock audio -> mock video -> sync -> concat -> QA.

Uses real FFmpeg and mock providers (no paid API calls). ~30-60 s total.
"""
import json
import shutil

import pytest

from services import ffmpeg
from services.audio_provider import MockAudioProvider
from services.orchestrator import Pipeline, Printer, RunOptions
from services.project_manager import Project
from services.video_generator import MockVideoProvider
from tests.conftest import make_project

SCRIPT = """Welcome to this short lesson about clouds.

Clouds form when warm, moist air rises and cools. The water vapour condenses into tiny droplets.

Remember, not every cloud brings rain. Thanks for watching."""


def pipeline(project, video=None, audio=None):
    cfg = project.config
    return Pipeline(project,
                    audio_provider=audio or MockAudioProvider(cfg.planning.words_per_second, cfg.audio.sample_rate),
                    video_provider=video or MockVideoProvider(cfg.generation, cfg.video),
                    printer=Printer(quiet=True), sleep=lambda s: None)


def frames(path):
    return int(ffmpeg.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
                           "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)]).stdout)


@pytest.fixture
def project(sandbox):
    make_project(sandbox, "lesson", SCRIPT)
    return Project("lesson", root=sandbox)


def test_full_run_then_resume_does_no_work(project):
    video = MockVideoProvider(project.config.generation, project.config.video)
    report = pipeline(project, video).run()

    assert report["final_file"] == "output/final_video.mp4"
    assert report["successful_scenes"] == report["total_scenes"] >= 3
    assert report["final_qa"]["status"] == "ok"
    final = project.output_dir / "final_video.mp4"
    info = ffmpeg.probe(final)
    assert (info.width, info.height) == (320, 180)
    state = project.load_state()
    # exact frame accounting: final == sum of frame-aligned scene targets, no gaps, no drops
    expected_frames = sum(round(s["sync"]["target_duration"] * 24) for s in state["scenes"])
    assert frames(final) == expected_frames
    assert abs(info.audio_duration - info.video_duration) < 0.1
    for s in state["scenes"]:
        assert s["status"] == "complete"
        assert s["actual_audio_duration"] > 0
        assert s["rendered_duration"] >= s["actual_audio_duration"]  # speech never cut
        assert s["qa"]["status"] in ("ok", "warn")
    assert (project.output_dir / "final_video.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000")
    assert (project.logs_dir / "pipeline.log").stat().st_size > 0

    calls_before = len(video.calls)
    report2 = pipeline(project, video).run()
    assert len(video.calls) == calls_before  # nothing regenerated on resume
    assert report2["final_qa"]["status"] == "ok"


def test_failed_scene_does_not_stop_others_and_resume_completes(project):
    cfg = project.config
    flaky = MockVideoProvider(cfg.generation, cfg.video, fail_times={"scene_002": 99})
    report = pipeline(project, flaky).run()
    state = project.load_state()
    statuses = {s["id"]: s["status"] for s in state["scenes"]}
    assert statuses["scene_002"] == "failed"
    assert all(v == "complete" for k, v in statuses.items() if k != "scene_002")
    assert report["final_file"] is None and report["failed_scenes"] == 1
    assert "simulated failure" in state["scenes"][1]["error"]

    healthy = MockVideoProvider(cfg.generation, cfg.video)
    report = pipeline(project, healthy).run(RunOptions())
    assert report["final_qa"]["status"] == "ok"
    assert {c.label for c in healthy.calls} == {"scene_002"}  # only the failed scene was redone


def test_quota_exhaustion_stops_video_stage(project):
    from utils.retry import RateLimitError

    class QuotaDead(MockVideoProvider):
        def generate(self, request):
            self.calls.append(request)
            raise RateLimitError("429 quota")

    cfg = project.config
    dead = QuotaDead(cfg.generation, cfg.video)
    report = pipeline(project, dead).run()
    assert len(dead.calls) == cfg.retry.max_attempts  # one scene's retries, then stop
    assert report["failed_scenes"] == 1 and report["final_file"] is None
    report = pipeline(project).run()  # quota back: resumes and completes
    assert report["final_qa"]["status"] == "ok"


def test_transient_failure_is_retried(project):
    cfg = project.config
    flaky = MockVideoProvider(cfg.generation, cfg.video, fail_times={"scene_001": 1})
    report = pipeline(project, flaky).run()
    assert report["final_qa"]["status"] == "ok"


def test_regenerate_single_scene_archives_old_clip(project):
    cfg = project.config
    pipeline(project).run()
    video = MockVideoProvider(cfg.generation, cfg.video)
    report = pipeline(project, video).run(RunOptions(regenerate_scene="scene_002", regenerate_stage="video"))
    assert {c.label for c in video.calls} == {"scene_002"}
    assert list((project.video_dir / "_replaced").rglob("scene_002.mp4"))
    assert report["final_qa"]["status"] == "ok"


def test_script_change_requires_replan_and_keeps_unchanged_media(project):
    pipeline(project).run()
    project.script_file.write_text(SCRIPT.replace("Thanks for watching.", "See you next time."), encoding="utf-8")
    with pytest.raises(Exception, match="--replan"):
        pipeline(project).run()
    video = MockVideoProvider(project.config.generation, project.config.video)
    report = pipeline(project, video).run(RunOptions(replan=True))
    assert report["final_qa"]["status"] == "ok"
    regenerated = {c.label for c in video.calls}
    assert len(regenerated) == 1  # only the scene whose text changed


def test_manual_mode_waits_for_clips_then_completes(project):
    cfg = project.config
    p = Pipeline(project, audio_provider=MockAudioProvider(cfg.planning.words_per_second, cfg.audio.sample_rate),
                 video_provider="manual", printer=Printer(quiet=True), sleep=lambda s: None)
    report = p.run()
    assert report["final_file"] is None
    tasks = (project.dir / "MANUAL_VIDEO_TASKS.md").read_text(encoding="utf-8")
    assert "video/scene_001.mp4" in tasks
    assert (project.prompts_dir / "scene_001.txt").is_file()

    # "user" drops clips in: one per scene, as .mov for scene_001 to test container detection;
    # scene_003 gets a clip that is far too short, which must be flagged, not cut.
    maker = MockVideoProvider(cfg.generation, cfg.video)
    state = project.load_state()
    from services.video_generator import ClipRequest
    for s in state["scenes"]:
        ext = ".mov" if s["id"] == "scene_001" else ".mp4"
        dur = 2 if s["id"] == "scene_003" else s["clip_durations"][0]
        maker.generate(ClipRequest("", project.presenter.image, dur, project.video_dir / f"{s['id']}{ext}"))
        for extra in range(1, len(s["clip_durations"])):
            maker.generate(ClipRequest("", project.presenter.image, s["clip_durations"][extra],
                                       project.video_path(s["id"], extra)))

    p2 = Pipeline(project, audio_provider=MockAudioProvider(cfg.planning.words_per_second, cfg.audio.sample_rate),
                  video_provider="manual", printer=Printer(quiet=True), sleep=lambda s: None)
    report = p2.run()
    assert report["final_file"] == "output/final_video.mp4"
    s3 = next(s for s in project.load_state()["scenes"] if s["id"] == "scene_003")
    assert s3["sync"]["action"] in ("insufficient", "video_master")  # either way: held frame, never cut speech
    assert s3["sync"]["freeze_pad"] > 0
    tempo = max((s3["sync"].get("fit") or {}).get("tempos") or [1.0])  # narration may be sped up to fit lips
    assert s3["rendered_duration"] >= s3["actual_audio_duration"] / tempo - 0.05  # but never cut
    assert any("frozen frame" in w for w in report["scene_qa_warnings"].get("scene_003", []))


def test_generated_video_audio_is_discarded(project):
    pipeline(project).run()
    state = project.load_state()
    # the mock clip carries a 440 Hz tone; the rendered scene must carry the narration instead
    out = ffmpeg.run(["ffmpeg", "-v", "info", "-i", str(project.abs(state["scenes"][0]["rendered_file"])),
                      "-af", "volumedetect", "-f", "null", "-"])
    narration = ffmpeg.run(["ffmpeg", "-v", "info", "-i", str(project.audio_path("scene_001")),
                            "-af", "volumedetect", "-f", "null", "-"])
    peak = lambda text: float(text.split("max_volume: ")[1].split(" dB")[0])
    assert abs(peak(out.stderr) - peak(narration.stderr)) < 1.5


def test_library_provider_reuses_clips_by_gesture(project):
    from services.video_generator import ClipRequest, LibraryVideoProvider
    cfg = project.config
    lib = project.presenter.directory / "library"
    maker = MockVideoProvider(cfg.generation, cfg.video)
    for name in ("welcome", "closing", "eye_contact", "head_nod", "right_hand", "right_hand_2"):
        maker.generate(ClipRequest("", project.presenter.image, 8, lib / f"{name}.mp4"))
    (lib / "_unused.mp4").write_bytes(b"")  # ignored: underscore prefix

    provider = LibraryVideoProvider(cfg.generation, lib, cfg.video)
    assert len(provider.clips) == 6
    assert provider.plan_durations(7.0, cfg.sync) == [7.0]
    assert provider.plan_durations(10.0, cfg.sync) == [5.0, 5.0]  # longer than any clip -> split

    report = pipeline(project, provider).run()
    assert report["final_qa"]["status"] == "ok"
    state = project.load_state()
    for s in state["scenes"]:
        assert s["sync"]["action"] in ("exact", "retime")  # cut to the narration, not trimmed
    picks = []
    for s in state["scenes"]:
        provider._last = None
        picks.append(provider._pick(s["gesture"], 1)[0].stem)
    assert picks[0] == "welcome" and picks[-1] == "closing"

    # never the same clip for consecutive requests, and opening/closing clips stay out of the general pool
    provider._last = None
    seq = []
    for i in range(12):
        clip, _, _ = provider._pick("facial_emphasis", i)  # no clip for this gesture -> general pool
        seq.append(clip.stem)
        provider._last = clip
    assert all(a != b for a, b in zip(seq, seq[1:]))
    assert not {"welcome", "closing"} & set(seq)


def test_measured_planning_sizes_scenes_exactly(sandbox):
    make_project(sandbox, "measured", SCRIPT * 2, overrides={
        "planning": {"measure_with_tts": True, "target_scene_duration": 6.0, "max_scene_duration": 7.0,
                     "min_scene_duration": 2.0}})
    project = Project("measured", root=sandbox)
    report = pipeline(project).run()
    assert report["final_qa"]["status"] == "ok"
    state = project.load_state()
    gap = project.config.audio.sentence_gap
    for s in state["scenes"]:
        assert s["units"] and " ".join(s["units"]) == s["text"]
        # planned length is the measured one: assembled narration matches it closely
        assert abs(s["actual_audio_duration"] - s["estimated_duration"]) < 0.1
        assert s["actual_audio_duration"] <= 7.0 + 0.1 or len(s["units"]) == 1
    units = list((project.audio_dir / "units").glob("*.wav"))
    assert len(units) == len({u for s in state["scenes"] for u in s["units"]})  # each sentence once
    assert gap > 0
