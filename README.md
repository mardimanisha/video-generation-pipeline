# AI Presenter Video Pipeline

Turns a **presenter image + voice sample + lesson script** into a finished presenter video:

```
script.txt ─► scenes (sentence-aware, ≤ clip length) ─► Inworld cloned voice per scene
          ─► measured audio durations ─► Veo clips sized to the audio ─► FFmpeg sync
          ─► normalized scene renders ─► concat ─► QA ─► final_video.mp4 + .srt + report
```

Local, filesystem-only, resumable, batch-capable. Claude Code orchestrates; no Anthropic API key needed.

## Setup (Windows)

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then fill in INWORLD_API_KEY and GEMINI_API_KEY
venv\Scripts\python setup.py  # checks everything
```

FFmpeg must be on PATH (`winget install Gyan.FFmpeg`).

## Presenters and projects

```
presenters/presenter_01/
  presenter.jpg          master image (first frame of every clip)
  presenter.json         appearance / framing / environment (merged over config/presenter.json)
  voice_sample.wav       10-30 s clean speech, used once to clone the voice
  voice.json             written by the pipeline: the reusable Inworld voice_id

projects/lesson-01/
  config.json            {"title": "...", "presenter": "presenter_01", "overrides": {...}}
  input/script.txt       the lesson text; blank lines = topic boundaries
```

A project without `"presenter"` uses `input/presenter.jpg` and `input/voice_sample.wav` instead.
`overrides` is deep-merged over `config/default.json` (validated at startup; unknown keys are errors).

Create one with `venv\Scripts\python setup.py --new-project lesson-02 --presenter presenter_01`.

## Running

| Command | What it does |
|---|---|
| `python pipeline.py --project lesson-01` | Full run. Re-running resumes; finished work is never redone. |
| `... --resume` | Same thing, explicit. |
| `... --plan-only` | Write `scenes/scenes.json` and stop, so you (or Claude) can refine gestures/prompts. |
| `... --manual-video` | Plan + voice + prompts; you generate clips by hand (see below). |
| `... --scene scene_005 --regenerate` | Redo one scene's video (`--stage audio` / `render` also available). Old clips are archived, not deleted. |
| `... --review` | Per-scene QA table with details for problem scenes. Generates nothing. |
| `... --replan` | After editing `script.txt`. Keeps audio/video of scenes whose text did not change. |
| `python batch.py` | Every project under `projects/` (skip a folder by prefixing `_`). |
| `--audio-provider mock --video-provider mock` | Full offline dry run, no cost. |

Outputs per project: `audio/`, `video/`, `rendered/`, `logs/pipeline.log` (JSON lines),
`output/final_video.mp4`, `output/final_video.srt`, `output/production_report.json`.

### Manual video mode

1. `python pipeline.py --project lesson-01 --manual-video`
2. Open `projects/lesson-01/MANUAL_VIDEO_TASKS.md`. For each clip, use the presenter image and
   the prompt in `video/prompts/scene_XXX.txt` in Gemini / Flow, at the listed duration.
3. Save the download as `video/scene_XXX.mp4` (`.mov`/`.webm` also accepted; extra clips for a
   long scene as `scene_XXX_b.mp4`).
4. Re-run the same command. Scenes with clips are synced; the final video is built once all exist.

### Clip library mode (no video API needed)

Generate a set of reusable presenter clips once in the Gemini app, following
`presenters/<presenter>/library/PROMPTS.md`, and save them into that `library/` folder, named by
gesture (`head_nod.mp4`, `counting.mp4`, ...). Then set `"generation": {"provider": "library"}`
in a project's `overrides` (or pass `--video-provider library`). Each scene's footage is cut from
the clip matching its planned gesture, falling back to the general clips; the same clip is never
used twice in a row. Per-lesson cost is the Inworld narration only.

## How sync works (audio-first)

The narration is the master timeline and is never cut. For each scene the pipeline measures the
real audio, picks clip lengths the provider supports (Veo: 4/6/8 s), then:

* footage within ×0.90–×1.15 of the narration → retimed to fit exactly (keeps the clip's start
  and end pose)
* footage longer → sped up ×1.15 and trimmed
* footage shorter → another clip is generated; in manual mode the last frame is held and QA warns

Clips are generated with the master image as first frame **and** (when supported) as last frame,
so every scene starts and ends in the same neutral pose and cuts are clean.
Scene renders are frame-exact, so the concatenated video has no gaps or drift.

## Status and limitations — read this

* **Tested**: planning, sync, rendering, concat, QA, resume, retry, regenerate, replan, manual
  mode, batch — end to end with mock providers and real FFmpeg (`pytest`).
* **Not yet tested against live services**: Inworld and Veo integrations are written against the
  official docs and unit-tested with fakes, but have not run with real keys.
* **Lip sync**: Veo animates the mouth from the prompt (the scene text is included), not from
  the Inworld audio, so lips only approximately match. Real lip sync would need an extra
  audio-driven lip-sync stage after generation; the provider structure leaves room for it.
* **Google Flow automation** is not built. Flow has no public API, UI automation is brittle and
  may conflict with its terms; use the Veo API or manual mode.
* **Visible watermark**: clips from the Gemini *app* carry a visible sparkle watermark (the sample
  video does, bottom-right). The Veo API docs describe SynthID (invisible) watermarking; confirm
  on the first real API clip.
