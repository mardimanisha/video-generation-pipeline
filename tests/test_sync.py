import pytest

from services.config import SyncConfig
from services.ffmpeg import frame_aligned
from services.sync import compute_sync, plan_clip_durations

CFG = SyncConfig(max_speedup=1.15, max_slowdown=0.9)
VEO = [4, 6, 8]


@pytest.mark.parametrize("target, expected", [
    (7.5, [8]),        # 8/7.5 = 1.07 -> gentle speed-up
    (8.7, [8]),        # 8/8.7 = 0.92 -> gentle slow-down, still one clip
    (6.2, [6]),
    (5.0, [6]),        # nothing in range -> shortest clip that is long enough (trim)
    (2.0, [4]),
    (9.6, [6, 4]),     # 10/9.6 in range with two clips
    (15.5, [8, 8]),
])
def test_plan_clip_durations(target, expected):
    assert plan_clip_durations(target, VEO, CFG) == expected


def test_plan_clip_durations_covers_narration():
    for tenths in range(5, 300):
        t = tenths / 10
        clips = plan_clip_durations(t, VEO, CFG)
        assert sum(clips) >= t * CFG.max_slowdown - 1e-9
        assert len(clips) <= 4


def test_plan_clip_durations_rejects_impossible():
    with pytest.raises(ValueError):
        plan_clip_durations(60, VEO, CFG)


def test_frame_aligned_rounds_up_to_whole_frames():
    assert frame_aligned(7.82, 24) == pytest.approx(188 / 24)
    assert frame_aligned(2.0, 24) == pytest.approx(2.0)


def test_compute_sync_actions():
    # audio 7.82 + 0.2 tail vs 8 s clip -> retime (slightly slower)
    p = compute_sync(7.82, [8.0], CFG, 0.2, 24)
    assert p.action == "retime" and p.target_duration == pytest.approx(frame_aligned(8.02, 24))
    assert p.retime_factor == pytest.approx(p.target_duration / 8.0)

    # the spec's example: audio 8.7 vs video 8.0 is covered by retiming, never by cutting speech
    p = compute_sync(8.7, [8.0], CFG, 0.0, 24)
    assert p.action == "retime" and p.freeze_pad == 0

    p = compute_sync(3.0, [8.0], CFG, 0.0, 24)
    assert p.action == "trim" and p.retime_factor == pytest.approx(1 / 1.15)

    p = compute_sync(12.0, [8.0], CFG, 0.0, 24)
    assert p.action == "insufficient"
    assert p.freeze_pad >= 12.0 - 8.0 / 0.9

    assert compute_sync(8.0, [8.0], CFG, 0.0, 24).action == "exact"
    assert compute_sync(4.0, [], CFG, 0.0, 24).action == "insufficient"
