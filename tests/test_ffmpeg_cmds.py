from pathlib import Path

from services import ffmpeg
from services.config import AudioConfig, VideoConfig
from services.subtitles import build_srt


def test_render_cmd_is_an_argument_list_with_safe_paths():
    tricky = Path("C:/my videos/it's; rm -rf $HOME/scene_001.mp4")
    cmd = ffmpeg.build_render_cmd([tricky], Path("a b/scene_001.wav"), Path("out/scene_001.mp4"),
                                  target_duration=7.5, retime_factor=0.95,
                                  video=VideoConfig(), audio_cfg=AudioConfig())
    assert isinstance(cmd, list) and all(isinstance(a, str) for a in cmd)
    assert str(tricky) in cmd          # passed verbatim as a single argument, never shell-parsed
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "setpts=0.950000*(PTS-STARTPTS)" in graph
    assert "trim=duration=7.500000" in graph and "atrim=duration=7.500000" in graph
    assert "scale=1280:720" in graph and "fps=24" in graph
    # generated audio from the video model is never mapped: only the narration input
    assert cmd[cmd.index("-map") + 1] == "[vout]" and "[1:a]" in graph and "[0:a]" not in graph


def test_render_cmd_concatenates_multiple_clips():
    cmd = ffmpeg.build_render_cmd([Path("a.mp4"), Path("b.mp4")], Path("n.wav"), Path("o.mp4"),
                                  target_duration=12, retime_factor=1.0, video=VideoConfig(),
                                  audio_cfg=AudioConfig(), freeze_pad=0.5)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[v0][v1]concat=n=2:v=1:a=0" in graph and "[2:a]" in graph
    assert "tpad=stop_mode=clone:stop_duration=0.667" in graph  # 0.5 freeze + 4 frames hold


def test_concat_manifest_escapes_quotes(tmp_path):
    f = tmp_path / "it's scene.mp4"
    text = ffmpeg.concat_manifest([f])
    assert text.startswith("ffconcat version 1.0\n")
    assert "it'\\''s scene.mp4'" in text


def test_concat_cmd_copies_video_and_encodes_audio_once():
    cmd = ffmpeg.build_concat_cmd(Path("m.txt"), Path("final.mp4"), AudioConfig())
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_srt_cues_are_short_and_ordered():
    srt = build_srt([("Hello there. " + "word " * 40 + "end.", 10.0, 10.2), ("Second scene.", 2.0, 2.2)])
    blocks = [b for b in srt.strip().split("\n\n") if b]
    assert len(blocks) >= 3
    assert all(len(b.split("\n")[2]) <= 84 for b in blocks)
    assert "00:00:10,200 -->" in srt  # second scene starts after the first rendered scene
