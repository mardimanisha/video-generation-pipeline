import pytest

from services.lipalign import (Piece, _merge_to, pick_utterance, plan_alignment,
                               segments_from_silence_log)


def test_segments_from_silence_log():
    log = ("[silencedetect] silence_start: 0\n[silencedetect] silence_end: 1.6 | silence_duration: 1.6\n"
           "[silencedetect] silence_start: 3.19\n[silencedetect] silence_end: 3.61 | x\n"
           "[silencedetect] silence_start: 5.39\n")
    assert segments_from_silence_log(log, 10.0) == [(1.6, 3.19), (3.61, 5.39)]
    assert segments_from_silence_log("", 4.0) == [(0.0, 4.0)]


def test_merge_to_joins_shortest_gaps():
    segs = [(0, 1), (1.1, 2), (2.8, 3), (3.05, 4)]
    assert _merge_to(segs, 2) == [(0, 2), (2.8, 4)]


def test_pick_utterance_ignores_stray_sounds():
    clip = [(1.6, 3.2), (3.6, 5.4), (7.5, 7.7)]  # a short noise well after the line
    assert pick_utterance(clip, 3.1) == [(1.6, 3.2), (3.6, 5.4)]


def test_phrase_alignment_maps_anchors_exactly():
    narr = [(0.03, 1.31), (1.46, 3.12)]
    clip = [(1.60, 3.19), (3.61, 5.39)]
    plan = plan_alignment(narr, clip, target=3.46, clip_len=10.0)
    assert plan.mode == "phrases" and plan.matched_phrases == 2
    # pieces are contiguous in output time and cover [pre_hold, target - post_hold]
    assert plan.pieces[0].t0 == pytest.approx(plan.pre_hold)
    for a, b in zip(plan.pieces, plan.pieces[1:]):
        assert a.t1 == pytest.approx(b.t0) and a.c1 == pytest.approx(b.c0)
    assert plan.pieces[-1].t1 + plan.post_hold == pytest.approx(3.46)
    # every phrase boundary lands exactly where the narration has it
    mapped = {round(p.t0, 3): round(p.c0, 3) for p in plan.pieces}
    assert mapped[0.03] == 1.60 and mapped[1.31] == 3.19 and mapped[1.46] == 3.61


def test_clip_starting_late_holds_first_frame():
    plan = plan_alignment([(0.5, 2.5)], [(0.2, 2.4)], target=3.0, clip_len=5.0)
    assert plan.pre_hold == pytest.approx(0.3)
    assert plan.pieces[0].c0 == 0.0


def test_short_clip_holds_last_frame():
    plan = plan_alignment([(0.1, 2.0)], [(0.1, 1.9)], target=4.0, clip_len=2.0)
    # speech ends at output 2.0 / clip 1.9; the 2 s tail has only 0.1 s of clip left to play
    assert plan.post_hold == pytest.approx(1.9)


def test_implausible_speed_falls_back():
    assert plan_alignment([(0, 1.0)], [(0, 3.0)], target=1.2, clip_len=8) is None  # 3x speed-up
    assert plan_alignment([], [(0, 1)], target=1, clip_len=2) is None


def test_piece_factor():
    assert Piece(1.0, 3.0, 0.0, 1.0).factor == 0.5


def test_search_alignment_follows_untouched_narration():
    from services.lipalign import plan_mismatch, search_alignment
    narr = [(0.03, 1.31), (1.46, 3.12)]               # narration (output time, never changed)
    clip = [(1.60, 3.19), (3.61, 5.39)]               # clip lips: later, slower, longer pause
    plan = search_alignment(narr, clip, target=3.46, clip_len=8.0)
    assert plan is not None and plan.matched_phrases == 2
    # retimed lips agree with the narration almost everywhere
    assert plan.mismatch < 0.15
    assert plan_mismatch(plan, narr, clip, 3.46) == plan.mismatch


def test_search_alignment_gives_up_when_speeds_would_look_unnatural():
    from services.lipalign import search_alignment
    assert search_alignment([(0.0, 1.0)], [(0.0, 3.0)], target=1.2, clip_len=8.0) is None
