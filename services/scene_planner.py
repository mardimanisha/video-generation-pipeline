"""Turn analyzed sentences into scenes with varied, subtle presenter motion and video prompts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from services.config import PipelineConfig
from services.script_analyzer import Sentence, analyze
from utils.duration import estimate_speech_duration

# Subtle, lecture-appropriate actions. Every one ends back at the neutral seated pose so
# consecutive clips (which all start from the master image) cut together cleanly.
NEUTRAL_ACTIONS = [
    ("eye_contact", "maintains warm, natural eye contact with the camera; relaxed hands resting together"),
    ("head_nod", "gives a small, affirming head nod while speaking"),
    ("right_hand", "makes a slight explanatory gesture with the right hand, then lowers it"),
    ("left_hand", "makes a small open-palm gesture with the left hand, then lowers it"),
    ("both_hands", "uses a small two-handed explanatory gesture in front of the chest"),
    ("forward_lean", "leans forward very slightly to engage the viewer, then settles back"),
    ("facial_emphasis", "shows subtle facial emphasis with gently raised eyebrows"),
    ("head_tilt", "tilts the head slightly while explaining, with a thoughtful expression"),
]
CUE_ACTIONS = {
    "opening": ("welcome", "greets the viewer with a warm, friendly smile and a small nod"),
    "closing": ("closing", "gives a warm closing smile and a small nod, hands resting calmly"),
    "question": ("question", "tilts the head slightly with an inquisitive expression, open palm gesture"),
    "enumeration": ("counting", "counts off points with a small, controlled finger gesture"),
    "emphasis": ("emphasis", "makes a small emphatic hand gesture with a subtle forward lean"),
    "contrast": ("contrast", "gestures gently with one hand then the other, as if comparing two ideas"),
}
RETURN_TO_NEUTRAL = "Returns to the neutral seated presenter posture by the end of the clip."

# Ordinals only count when they open a sentence ("First, ..."), not "the first year".
_ENUMERATION = re.compile(
    r"(?:^|[.!?]\s+)(?:first|second|third|fourth|fifth|firstly|secondly|thirdly|finally|lastly)\b"
    r"|\bstep \d+\b"
    r"|\b(?:two|three|four|five|several)\s+(?:\w+\s+)?(?:things|steps|ways|reasons|points|rules|factors|parts|ideas)\b",
    re.I)
_EMPHASIS = re.compile(r"\b(important|key|crucial|remember|never|always|essential|critical|must)\b", re.I)
_CONTRAST = re.compile(r"\b(however|on the other hand|whereas|in contrast|but|instead|unlike)\b", re.I)


def classify_cue(text: str, index: int, total: int) -> str | None:
    if index == 0:
        return "opening"
    if index == total - 1 and total > 2:
        return "closing"
    if text.rstrip().endswith("?"):
        return "question"
    if _ENUMERATION.search(text):
        return "enumeration"
    if _EMPHASIS.search(text):
        return "emphasis"
    if _CONTRAST.search(text):
        return "contrast"
    return None


def assign_motion(texts: list[str]) -> list[tuple[str, str]]:
    """Pick a gesture per scene: cue-driven when the text suggests one, otherwise rotate the
    neutral library. Never repeat the previous scene's gesture back to back."""
    result: list[tuple[str, str]] = []
    rotation = 0
    for i, text in enumerate(texts):
        cue = classify_cue(text, i, len(texts))
        choice = CUE_ACTIONS[cue] if cue else None
        if choice is None or (result and result[-1][0] == choice[0]):
            choice = NEUTRAL_ACTIONS[rotation % len(NEUTRAL_ACTIONS)]
            rotation += 1
            if result and result[-1][0] == choice[0]:
                choice = NEUTRAL_ACTIONS[rotation % len(NEUTRAL_ACTIONS)]
                rotation += 1
        result.append(choice)
    return result


SHORT_PENALTY = 3.0       # extra cost per squared second below min_scene_duration


def group_into_scenes(sentences: list[Sentence], cfg: PipelineConfig,
                      measured_gap: float | None = None) -> list[list[Sentence]]:
    """Split sentences into scenes as close to the target duration as possible.

    Scenes prefer to end at paragraph breaks. A scene never exceeds max_scene_duration unless a
    single sentence does. With `measured_gap`, each Sentence.estimate is a measured narration
    length and scenes are sized exactly: sum of sentences + one gap between each pair.
    """
    p = cfg.planning
    wps = p.words_per_second
    n = len(sentences)

    def est(i: int, j: int) -> float:
        if measured_gap is not None:
            return sum(x.estimate for x in sentences[i:j]) + measured_gap * (j - i - 1)
        return estimate_speech_duration(" ".join(s.text for s in sentences[i:j]), wps)

    def cost(i: int, j: int, d: float) -> float:
        c = p.scene_cost + (p.target_scene_duration - d) ** 2
        if d < p.min_scene_duration:
            c += SHORT_PENALTY * (p.min_scene_duration - d) ** 2
        breaks = sum(1 for k in range(i + 1, j) if sentences[k].paragraph != sentences[k - 1].paragraph)
        return c + p.paragraph_penalty * breaks

    # Optimal contiguous partition (dynamic programming) instead of greedy packing, which
    # tends to fill one scene and strand a two-second remainder in the next.
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            d = est(i, j)
            if d > p.max_scene_duration and j - i > 1:
                break  # adding earlier sentences only makes it longer
            total = best[i] + cost(i, j, d)
            if total < best[j]:
                best[j], back[j] = total, i
    groups: list[list[Sentence]] = []
    j = n
    while j > 0:
        groups.append(sentences[back[j]:j])
        j = back[j]
    return groups[::-1]


def describe_presenter(profile: dict) -> str:
    parts = []
    a, e, c = profile.get("appearance", {}), profile.get("environment", {}), profile.get("composition", {})
    if a.get("general_description"):
        parts.append(f"Presenter: {a['general_description']}.")
    if a.get("clothing"):
        parts.append(f"Clothing: {a['clothing']}.")
    if a.get("hairstyle"):
        parts.append(f"Hair: {a['hairstyle']}.")
    if e.get("background"):
        parts.append(f"Setting: {e['background']}.")
    if e.get("lighting"):
        parts.append(f"Lighting: {e['lighting']}.")
    framing = ", ".join(v for v in (c.get("shot"), c.get("camera_position"), c.get("camera_motion")) if v)
    if framing:
        parts.append(f"Framing: {framing}.")
    return " ".join(parts)


def render_prompt(template: str, profile: dict, visual_action: str, text: str, include_dialogue: bool) -> str:
    dialogue = ""
    if include_dialogue:
        clean = text.replace('"', "'")
        dialogue = f'The presenter says: "{clean}"'
    out = (
        template.replace("{{PRESENTER_DESCRIPTION}}", describe_presenter(profile))
        .replace("{{VISUAL_ACTION}}", visual_action)
        .replace("{{DIALOGUE}}", dialogue)
    )
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def load_template(root: Path) -> str:
    return (root / "prompts" / "video_prompt.txt").read_text(encoding="utf-8")


def plan_scenes(script: str, cfg: PipelineConfig, profile: dict, template: str,
                measure: Callable[[list[Sentence]], list[float]] | None = None) -> list[dict]:
    """Plan scenes. With `measure` (returns the real narration length of each sentence, e.g. by
    synthesizing it), scenes are packed on measured durations and remember their sentences
    ("units") so the scene narration can be assembled from them."""
    sentences = analyze(script, cfg.planning.words_per_second, cfg.planning.max_scene_duration)
    gap = None
    if measure is not None:
        for s, d in zip(sentences, measure(sentences)):
            s.estimate = round(d, 3)
        gap = cfg.audio.sentence_gap
    groups = group_into_scenes(sentences, cfg, measured_gap=gap)
    texts = [" ".join(s.text for s in g) for g in groups]
    motions = assign_motion(texts)
    scenes = []
    for i, (group, text, (gesture, action)) in enumerate(zip(groups, texts, motions), start=1):
        visual_action = f"The presenter {action}. {RETURN_TO_NEUTRAL}"
        duration = (round(sum(s.estimate for s in group) + gap * (len(group) - 1), 2) if gap is not None
                    else estimate_speech_duration(text, cfg.planning.words_per_second))
        scenes.append({
            "id": f"scene_{i:03d}",
            "text": text,
            "units": [s.text for s in group] if gap is not None else None,
            "estimated_duration": duration,
            "actual_audio_duration": None,
            "visual_action": visual_action,
            "gesture": gesture,
            "camera": "static",
            "generation_prompt": render_prompt(template, profile, visual_action, text,
                                               cfg.gemini.include_dialogue_in_prompt),
            "audio_file": "",
            "clip_durations": [],
            "video_files": [],
            "rendered_file": "",
            "rendered_duration": None,
            "sync": None,
            "status": "pending",
            "error": None,
        })
    return scenes
