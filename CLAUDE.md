# CLAUDE.md

Local production pipeline for presenter-style educational videos:
script → scenes → Inworld voice (per scene) → Veo clips → FFmpeg sync → final MP4.

Claude Code is the **orchestrator**, not a renderer. No Anthropic API key is used anywhere.
FFmpeg does all media work; providers do generation; Python code is deterministic.

## Commands (always use the venv)

```
venv\Scripts\python setup.py                                   # environment check
venv\Scripts\python setup.py --new-project lesson-02 --presenter presenter_01
venv\Scripts\python pipeline.py --project lesson-01 --plan-only # stop after scenes.json
venv\Scripts\python pipeline.py --project lesson-01             # run / resume (idempotent)
venv\Scripts\python pipeline.py --project lesson-01 --manual-video
venv\Scripts\python pipeline.py --project lesson-01 --scene scene_005 --regenerate [--stage audio|video|render]
venv\Scripts\python pipeline.py --project lesson-01 --review
venv\Scripts\python batch.py [--only a b]
venv\Scripts\python -m pytest -q                                # ~3 min; integration tests use real FFmpeg
venv\Scripts\python -m pytest -q --ignore=tests/test_integration.py   # unit only, seconds
```

`--audio-provider mock --video-provider mock` runs everything offline at no cost.

## Claude's role in a production run

1. `--plan-only`, then read `projects/<p>/scenes/scenes.json`.
2. Refine scenes where the rule-based planner is weak: edit `visual_action` (keep it subtle and
   ending in neutral posture) or set `custom_prompt` to fully override a clip prompt. Do not
   edit `text` (it is hashed; change `input/script.txt` and use `--replan` instead).
3. Run the pipeline, then `--review`; regenerate individual bad scenes.
Prompts are re-rendered from `visual_action` at generation time, so edits take effect.

## Architecture

- `services/orchestrator.py` — stage flow, checkpoints, retries, progress. Stages decide what to
  do from files on disk + fingerprints, so every run is a resume. State saved after each scene step.
- `services/project_manager.py` — paths, config loading (default.json deep-merged with project
  `overrides`, validated by `services/config.py`), presenter resolution.
- `services/script_analyzer.py` / `scene_planner.py` — sentence/clause splitting, greedy scene
  packing, cue-based gesture assignment, prompt rendering from `prompts/video_prompt.txt`.
- `services/sync.py` — audio-first fitting: retime within [max_slowdown, max_speedup], trim when
  longer, add clips when shorter; frozen frame only as a QA-flagged last resort. Narration is never cut.
- `services/audio_fit.py` — **default sync (`sync.mode: video_master`)**: the generated clip is used
  untouched (trimmed to lead/tail around its speech); the narration is cut into phrases, paired with
  the clip's spoken phrases (DP over pause groupings, `select_speech` drops clicks/ad-libs), each phrase
  tempo-fitted (0.7–1.45, pitch kept) and placed on the lip onsets. Narration is never cut; if it
  outlasts the clip the last frame is held and QA warns. `audio_master` (lipalign.py) is the old mode.
- `services/ffmpeg.py` — pure command builders (arg lists, never shell strings) + `run`/`probe`.
  Bump `RENDER_VERSION` whenever `build_render_cmd` output changes.
- Providers: `audio_provider.py` (base, mock), `inworld.py`; `video_generator.py` (base, mock,
  manual, library, Gemini/Veo). The pipeline only sees the interfaces.
  `library` cuts scenes from `presenters/<p>/library/<gesture>[_N].mp4` (clips made once in the
  Gemini app from `PROMPTS.md`); it is the default for the Session 1 projects.
- `services/qa.py` — per-scene and final checks. Per-scene A/V length must match within 1.5 frames.

## Invariants — keep them

- Scene renders are frame-exact: target = ceil((audio + tail) * fps) / fps; video and audio
  both trimmed/padded to it. Final concat stream-copies video and re-encodes audio once.
- Paid clips are never deleted: regenerate/replan move them to `video/_replaced/` or `video/_stale/`.
- Clip file names are deterministic: `video/scene_007.mp4`, extra clips `scene_007_b.mp4`, ...
- Credentials only via `.env` (`INWORLD_API_KEY`, `INWORLD_VOICE_ID`, `GEMINI_API_KEY`).
- The voice is cloned once per presenter; ID stored in `presenters/<p>/voice.json`.
- Never invent provider endpoints; check the official docs before changing an integration.

## Provider facts (verified 2026-09)

- Inworld: `POST https://api.inworld.ai/tts/v1/voice` (≤2000 chars), clone
  `POST /voices/v1/voices:clone` (10–30 s sample), auth `Basic <key>`, models `inworld-tts-2`, `inworld-tts-2-flash`.
- Veo 3.1 (Gemini API, google-genai): durations 4/6/8 s, 24 fps, 720p/1080p/4k, audio always
  generated (we discard it), `last_frame` supported, outputs deleted server-side after 2 days.
- Google Flow has no public API; a Playwright provider is not built yet (see README).
