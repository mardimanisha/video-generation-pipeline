import pytest

from services.audio_fit import Placement, atempo_chain, plan_fit


def onset_out(p: Placement, narr_onset: float) -> float:
    return p.out_start + (narr_onset - p.narr_start) / p.tempo


def test_each_narration_phrase_starts_on_the_lips():
    narr = [(0.03, 1.30), (1.45, 3.10)]          # cloned voice: fast, short pause
    clip = [(1.55, 3.20), (3.60, 5.40)]          # clip: starts late, slower, longer pause
    plan = plan_fit(narr, 3.24, clip, 8.0, lead=0.25, tail=0.3)
    w0, w1 = plan.window
    assert plan.mode == "phrases" and plan.phrases == 2
    assert w0 == pytest.approx(1.30)              # clip's leading silence is trimmed off
    for p, (ns, _), (cs, _) in zip(plan.placements, narr, clip):
        assert onset_out(p, ns) == pytest.approx(cs - w0, abs=1e-6)
    # phrase ends land on the lip ends too (tempo within natural range here)
    p1 = plan.placements[1]
    assert onset_out(p1, 3.10) == pytest.approx(5.40 - w0, abs=1e-6)
    assert w1 - w0 == pytest.approx(round((5.40 - w0 + 0.3) * 24) / 24, abs=1 / 24)


def test_narration_pieces_tile_the_whole_line():
    plan = plan_fit([(0.0, 1.0), (1.4, 2.0), (2.5, 3.0)], 3.1, [(1, 2), (2.6, 3.4), (4.0, 4.6)], 8.0)
    spans = [(p.narr_start, p.narr_end) for p in plan.placements]
    assert spans[0][0] == 0.0 and spans[-1][1] == 3.1
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))   # nothing dropped, nothing repeated


def test_tempo_stays_natural_and_narration_is_never_cut():
    # clip speaks for 1 s, narration lasts 4 s: tempo clamps, video holds its last frame
    plan = plan_fit([(0.0, 4.0)], 4.0, [(6.5, 7.5)], 8.0, lead=0.25, tail=0.3)
    p = plan.placements[0]
    assert p.tempo == pytest.approx(1.45)
    assert plan.window[1] - plan.window[0] >= p.out_end + 0.3 - 1e-6
    assert plan.freeze_pad > 0


def test_extra_clip_pauses_are_merged_to_match():
    plan = plan_fit([(0.0, 2.0)], 2.1, [(1.0, 1.8), (1.95, 3.0)], 8.0)
    assert plan.phrases == 1 and plan.mode == "span"
    assert plan.placements[0].tempo == pytest.approx(1.0)


def test_no_speech_means_no_plan():
    assert plan_fit([], 2.0, [(1, 2)], 8.0) is None
    assert plan_fit([(0, 1)], 1.0, [], 8.0) is None


@pytest.mark.parametrize("t, stages", [(1.2, 1), (2.5, 2), (0.4, 2)])
def test_atempo_chain(t, stages):
    assert atempo_chain(t).count("atempo") == stages


def test_select_speech_drops_stray_sounds_and_extra_words():
    from services.audio_fit import select_speech
    narr = [(0.0, 1.57), (1.86, 3.48), (3.8, 4.76)]
    lips = [(0.58, 1.99), (2.46, 4.33), (5.14, 5.86), (7.9, 8.0)]       # click at the end
    assert select_speech(lips, narr)[-1] == (5.14, 5.86)
    narr2 = [(0.0, 1.8), (2.14, 3.93)]
    lips2 = [(0.7, 2.28), (3.08, 4.87), (5.42, 6.75)]                     # ad-libbed extra words
    assert select_speech(lips2, narr2) == [(0.7, 2.28), (3.08, 4.87)]


def test_select_speech_keeps_a_phrase_after_a_dramatic_pause():
    from services.audio_fit import select_speech
    narr = [(0.0, 1.2), (1.5, 3.75)]
    lips = [(0.58, 1.81), (2.89, 4.92), (5.37, 6.06)]                     # question, 1 s pause, answer
    assert select_speech(lips, narr)[0] == (0.58, 1.81)


def test_ad_lib_running_into_clip_end_is_dropped():
    from services.audio_fit import select_speech
    narr = [(0.0, 2.7), (2.77, 3.31), (3.47, 4.44), (5.34, 6.61), (6.78, 8.11), (8.17, 8.3)]
    lips = [(1.17, 4.09), (4.43, 5.44), (6.35, 8.89), (9.26, 10.03)]     # extra words until the end
    assert select_speech(lips, narr, clip_len=10.03)[-1] == (6.35, 8.89)


def test_mismatch_counts_seconds_where_lips_and_voice_disagree():
    from services.audio_fit import FitPlan, Placement, mismatch
    plan = FitPlan((0.0, 4.0), [Placement(0.0, 2.0, 1.0, 1.0)], 1, "span")   # voice 1-3 s
    assert mismatch(plan, [(0.0, 2.0)], [(1.0, 3.0)]) == pytest.approx(0.0, abs=0.02)
    assert mismatch(plan, [(0.0, 2.0)], [(1.5, 3.5)]) == pytest.approx(1.0, abs=0.03)  # 0.5 early + 0.5 late


def test_search_prefers_finer_pairing_when_pauses_disagree():
    # narration pauses after 1 s; the clip has three spoken chunks with long pauses
    narr = [(0.0, 1.0), (1.1, 2.0), (2.1, 3.0)]
    lips = [(1.0, 2.0), (2.6, 3.5), (4.1, 5.0)]
    plan = plan_fit(narr, 3.0, lips, 8.0)
    assert plan.phrases == 3 and plan.mismatch < 0.1
