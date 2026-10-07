import json

import pytest

from services.config import PipelineConfig
from services.project_manager import ConfigError, Project, segment_suffix
from tests.conftest import make_project
from utils.retry import PermanentError, TransientError, backoff_delay, with_retry


def test_default_config_file_is_valid(repo_root):
    raw = json.loads((repo_root / "config" / "default.json").read_text(encoding="utf-8"))
    cfg = PipelineConfig(**raw)
    assert (cfg.video.width, cfg.video.height, cfg.video.fps) == (1280, 720, 24)
    assert cfg.generation.max_clip_duration == 8


@pytest.mark.parametrize("bad", [
    {"video": {"fps": 0}},
    {"video": {"fsp": 24}},                                   # typo -> unknown key
    {"generation": {"preferred_clip_duration": 12, "max_clip_duration": 8}},
    {"generation": {"allowed_durations": [4, 10], "max_clip_duration": 8}},
    {"planning": {"min_scene_duration": 9, "target_scene_duration": 7}},
    {"audio": {"provider": "elevenlabs"}},
])
def test_invalid_config_rejected(bad):
    with pytest.raises(Exception):
        PipelineConfig(**bad)


def test_project_overrides_merge_and_presenter_resolution(sandbox):
    make_project(sandbox, "l1", "Hello.", overrides={"sync": {"max_speedup": 1.1}})
    p = Project("l1", root=sandbox)
    assert p.config.sync.max_speedup == 1.1
    assert p.config.sync.max_slowdown == 0.9          # untouched default survives the merge
    assert p.presenter.image.name == "presenter.jpg"
    assert p.presenter.profile["appearance"]["clothing"] == "navy suit"
    assert p.presenter.profile["motion"]["intensity"] == "subtle"   # from config/presenter.json
    assert p.presenter.voice["provider"] == "inworld"


def test_projects_are_isolated(sandbox):
    make_project(sandbox, "a", "A.")
    make_project(sandbox, "b", "B.", presenter=None)
    a, b = Project("a", root=sandbox), Project("b", root=sandbox)
    assert a.audio_path("scene_001") != b.audio_path("scene_001")
    assert b.presenter.voice_file == b.dir / "voice.json"


def test_invalid_project_config_is_reported(sandbox):
    d = make_project(sandbox, "bad", "x.", overrides={"video": {"fps": -1}})
    with pytest.raises(ConfigError):
        Project("bad", root=sandbox)
    (d / "config.json").write_text('{"presenter": "nobody"}', encoding="utf-8")
    with pytest.raises(ConfigError):
        Project("bad", root=sandbox)


def test_segment_suffix():
    assert [segment_suffix(i) for i in range(3)] == ["", "_b", "_c"]


def test_backoff_schedule():
    assert [backoff_delay(i, [5, 15]) for i in (1, 2, 3)] == [5, 15, 45]


def test_retry_transient_then_success():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("timeout")
        return "ok"

    assert with_retry(fn, max_attempts=3, schedule=[5, 15], sleep=sleeps.append) == "ok"
    assert sleeps == [5, 15]


def test_retry_gives_up_after_max_attempts():
    sleeps = []
    with pytest.raises(TransientError):
        with_retry(lambda: (_ for _ in ()).throw(TransientError("x")), max_attempts=3,
                   schedule=[1, 2], sleep=sleeps.append)
    assert sleeps == [1, 2]


def test_permanent_error_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise PermanentError("bad key")

    with pytest.raises(PermanentError):
        with_retry(fn, max_attempts=5, schedule=[1], sleep=lambda s: None)
    assert len(calls) == 1


def test_rate_limit_waits_at_least_a_minute():
    from utils.retry import RateLimitError
    sleeps = []
    with pytest.raises(RateLimitError):
        with_retry(lambda: (_ for _ in ()).throw(RateLimitError("429")), max_attempts=2,
                   schedule=[5], sleep=sleeps.append)
    assert sleeps == [60.0]
