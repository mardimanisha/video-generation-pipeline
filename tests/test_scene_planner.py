import re

from services.config import PipelineConfig
from services.scene_planner import assign_motion, classify_cue, plan_scenes, render_prompt

SCRIPT = """Welcome to this lesson on photosynthesis. Today we explore how plants make food.

Plants capture sunlight using chlorophyll, a green pigment found in their leaves. This energy drives a chemical reaction.

There are three key inputs. First, sunlight. Second, water. Third, carbon dioxide.

Why does this matter? Because nearly all life on Earth depends on it. Remember this idea. Thanks for watching."""

TEMPLATE = "ACTION: {{VISUAL_ACTION}}\n{{PRESENTER_DESCRIPTION}}\n{{DIALOGUE}}"


def _plan(**planning):
    cfg = PipelineConfig(planning=planning) if planning else PipelineConfig()
    return cfg, plan_scenes(SCRIPT, cfg, {"appearance": {"clothing": "grey blazer"}}, TEMPLATE)


def test_scene_text_covers_script_exactly_once():
    _, scenes = _plan()
    joined = " ".join(s["text"] for s in scenes)
    assert re.sub(r"\s+", " ", joined) == re.sub(r"\s+", " ", SCRIPT).strip()


def test_scene_ids_and_required_fields():
    _, scenes = _plan()
    assert [s["id"] for s in scenes] == [f"scene_{i:03d}" for i in range(1, len(scenes) + 1)]
    for s in scenes:
        assert s["status"] == "pending"
        assert s["actual_audio_duration"] is None
        assert s["camera"] == "static"
        assert s["text"] and s["visual_action"] and s["gesture"]


def test_scenes_respect_max_duration():
    cfg, scenes = _plan()
    for s in scenes:
        n_sentences = len(re.findall(r"[.!?](\s|$)", s["text"]))
        assert s["estimated_duration"] <= cfg.planning.max_scene_duration or n_sentences == 1


def test_scenes_do_not_split_mid_sentence():
    _, scenes = _plan()
    for s in scenes:
        assert s["text"].rstrip()[-1] in ".!?"


def test_no_identical_gesture_back_to_back():
    _, scenes = _plan()
    gestures = [s["gesture"] for s in scenes]
    assert all(a != b for a, b in zip(gestures, gestures[1:]))
    assert gestures[0] == "welcome"
    assert gestures[-1] == "closing"


def test_cue_classification():
    assert classify_cue("Why does this matter?", 3, 10) == "question"
    assert classify_cue("There are three key things.", 3, 10) == "enumeration"
    assert classify_cue("Intro. First, sunlight.", 3, 10) == "enumeration"
    assert classify_cue("After the first year, you have more.", 3, 10) is None
    assert classify_cue("This is crucial.", 3, 10) == "emphasis"


def test_assign_motion_rotates_neutral_library():
    motions = assign_motion(["plain sentence."] * 6)
    ids = [m[0] for m in motions]
    assert len(set(ids[1:])) == 5


def test_prompt_contains_action_presenter_and_dialogue():
    _, scenes = _plan()
    p = scenes[1]["generation_prompt"]
    assert "ACTION: The presenter" in p and "grey blazer" in p and scenes[1]["text"].replace('"', "'") in p
    no_dialogue = render_prompt(TEMPLATE, {}, "nods", 'He said "hi"', include_dialogue=False)
    assert "says" not in no_dialogue
