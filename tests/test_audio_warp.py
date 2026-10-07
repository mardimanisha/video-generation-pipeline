"""Word-level narration warping on synthetic 'speech' (tone bursts with known timing)."""
import numpy as np
import pytest

# Tests for an earlier word-level warping module (services/audio_warp.py) that is not in the
# codebase any more; they run again automatically if the module is restored.
aw = pytest.importorskip("services.audio_warp")

SR = aw.ANALYSIS_SR


def bursts(spec, total, sr=SR):
    """spec: [(start, dur, freq)] -> mono signal with harmonic bursts (speech-like syllables)."""
    x = np.zeros(int(total * sr), np.float32)
    for start, dur, f in spec:
        t = np.arange(int(dur * sr)) / sr
        tone = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3)) * np.hanning(len(t))
        x[int(start * sr): int(start * sr) + len(t)] += 0.3 * tone.astype(np.float32)
    return x


SYLLABLES = [180, 260, 220, 300, 200, 240]


def test_warp_moves_narration_onto_clip_timing(tmp_path):
    narr_spec = [(0.05 + i * 0.30, 0.22, f) for i, f in enumerate(SYLLABLES)]           # fast voice
    clip_spec = [(1.00 + i * 0.42, 0.30, f) for i, f in enumerate(SYLLABLES)]           # slower, later
    narr, clip = bursts(narr_spec, 2.2), bursts(clip_spec, 4.5)
    narr_segs = [(0.05, 0.05 + 5 * 0.30 + 0.22)]
    clip_segs = [(1.00, 1.00 + 5 * 0.42 + 0.30)]
    plan = aw.plan_warp(clip, narr, clip_segs, narr_segs, clip_len=4.5)
    assert plan is not None
    # syllable onsets in the clip map onto the matching narration onsets
    for (cs, _, _), (ns, _, _) in zip(clip_spec, narr_spec):
        assert abs(plan.narr_time(np.array([cs]))[0] - ns) < 0.08

    path = tmp_path / "narr.wav"
    aw.save_wav(path, narr, SR)
    out = tmp_path / "warped.wav"
    dur = aw.render_warped_narration(path, out, plan, SR)
    w0, w1 = plan.window
    assert abs(dur - (w1 - w0)) < 0.02
    warped = aw.load_mono(out, SR)
    window = clip[int(w0 * SR): int(w1 * SR)]
    naive = np.zeros_like(window)
    naive[int((1.0 - w0 - 0.05) * SR):][: len(narr)] = narr[: len(naive) - int((1.0 - w0 - 0.05) * SR)]
    assert aw.envelope_similarity(window, warped) > aw.envelope_similarity(window, naive) + 0.2


def test_wsola_preserves_pitch_while_stretching():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = aw.wsola(x, SR, np.array([0.0, 1.5]), np.array([0.0, 1.0]))   # 1 s -> 1.5 s
    assert abs(len(y) / SR - 1.5) < 0.01
    spec = np.abs(np.fft.rfft(y[int(0.2 * SR): int(1.2 * SR)]))
    peak = np.argmax(spec) * SR / (2 * (len(spec) - 1))
    assert abs(peak - 220) < 6


def test_mismatched_lengths_are_rejected():
    narr, clip = bursts([(0, 3.0, 200)], 3.2), bursts([(0, 0.8, 200)], 1.0)
    assert aw.plan_warp(clip, narr, [(0.0, 0.8)], [(0.0, 3.0)], clip_len=1.0) is None


def test_phrase_placement_keeps_every_sample_and_aligns_onsets(tmp_path):
    narr_spec = [(0.05, 0.9, 200), (1.15, 0.8, 260)]     # two phrases, short pause
    clip_spec = [(0.80, 1.0, 200), (2.40, 0.9, 260)]     # slower, longer pause, starts later
    narr, clip = bursts(narr_spec, 2.0), bursts(clip_spec, 4.0)
    plan, how = aw.plan_phrases(clip, narr, [(0.8, 1.8), (2.4, 3.3)], [(0.05, 0.95), (1.15, 1.95)], 2.0, 4.0)
    assert plan.phrases == 2
    # pieces tile the narration end to end (nothing dropped) and the cut sits inside the pause
    spans = [p.narr_span for p in plan.placements]
    assert spans[0][0] == 0.0 and spans[-1][1] == 2.0 and spans[0][1] == spans[1][0]
    assert 0.95 <= spans[0][1] <= 1.15
    # natural speeds only
    assert all(aw.PHRASE_TEMPO[0] <= p.tempo <= aw.PHRASE_TEMPO[1] for p in plan.placements)
    # each phrase's speech onset lands on the clip's phrase onset
    for p, (cs, _, _), (ns, _, _) in zip(plan.placements, clip_spec, narr_spec):
        onset = p.clip_start + (ns - p.narr_span[0]) / p.tempo
        assert abs(onset - cs) < 0.02
    path, out = tmp_path / "n.wav", tmp_path / "o.wav"
    aw.save_wav(path, narr, SR)
    dur = aw.render_phrase_narration(path, out, plan, SR)
    assert abs(dur - (plan.window[1] - plan.window[0])) < 0.01
    y = aw.load_mono(out, SR)
    # all narration energy survives (tempo change keeps loudness; nothing cut)
    assert abs(np.sum(y ** 2) / np.sum(narr ** 2) * 1.0 - 1.0) < 0.35


def test_compress_pauses_shortens_silence_and_moves_words():
    x = bursts([(0.1, 0.5, 200), (1.4, 0.5, 260)], 2.2)        # 0.8 s pause between the words
    y, words = aw.compress_pauses(x, SR, [(0.1, 0.6), (1.4, 1.9)], max_pause=0.07)
    removed = 2.2 - len(y) / SR
    assert 0.5 < removed <= 0.8 - 0.07                             # most of the pause goes, never all of it
    assert abs(words[0][0] - 0.1) < 0.01
    assert 0.07 <= words[1][0] - words[0][1] <= 0.25              # second word follows right after


def test_compress_pauses_never_removes_sound_between_words():
    # a "pause" per the word timings that actually contains sound must be left alone
    x = bursts([(0.1, 0.5, 200), (0.6, 0.8, 300), (1.4, 0.5, 260)], 2.2)
    y, _ = aw.compress_pauses(x, SR, [(0.1, 0.6), (1.4, 1.9)], max_pause=0.07)
    assert len(y) == len(x)
