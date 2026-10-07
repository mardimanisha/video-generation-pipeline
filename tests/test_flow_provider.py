"""Flow provider logic that does not need a live browser."""
from pathlib import Path

from services.config import FlowConfig, GenerationConfig, SyncConfig
from services.flow_browser import FlowBrowserVideoProvider
from services.video_generator import ClipRequest


def provider(tmp_path):
    return FlowBrowserVideoProvider(GenerationConfig(), FlowConfig(), tmp_path / "flow.json", tmp_path)


def test_one_clip_per_scene_up_to_clip_length(tmp_path):
    p = provider(tmp_path)
    assert p.plan_durations(7.0, SyncConfig()) == [8.0]
    assert p.plan_durations(3.0, SyncConfig()) == [8.0]
    assert p.plan_durations(7.9, SyncConfig()) == [10.0]   # 8 s clips cut long lines off mid-word
    assert p.plan_durations(12.0, SyncConfig()) == [10.0, 10.0]


def test_prompt_is_wrapped_for_the_agent(tmp_path):
    req = ClipRequest(prompt="He  says:\n\"Hello.\"", reference_image=Path("x.jpg"), duration=8,
                      output_path=tmp_path / "o.mp4")
    text = provider(tmp_path)._wrap_prompt(req)
    assert text.startswith("Generate exactly one 8-second 16:9 video (one output only)")
    assert 'He says: "Hello."' in text  # whitespace normalised, content untouched


def test_state_is_persisted(tmp_path):
    p = provider(tmp_path)
    p._save_state(project_url="https://flow.google.com/project/abc")
    p._save_state(reference_sha="123")
    assert p._state() == {"project_url": "https://flow.google.com/project/abc", "reference_sha": "123"}


def test_long_lines_get_the_long_clip(tmp_path):
    p = provider(tmp_path)
    assert p.plan_durations(6.9, SyncConfig()) == [8.0]
    assert p.plan_durations(8.4, SyncConfig()) == [10.0]
    req = ClipRequest(prompt="x", reference_image=Path("x.jpg"), duration=10.0, output_path=tmp_path / "o.mp4")
    assert p._wrap_prompt(req).startswith("Generate exactly one 10-second")
