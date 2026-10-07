"""Environment check and project scaffolding (this is not a setuptools script).

  python setup.py                                       # check the environment
  python setup.py --new-project lesson-02 --presenter presenter_01
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    problems, notes = [], []

    def row(ok: bool | None, label: str, detail: str = "") -> None:
        mark = {True: "✓", False: "✗", None: "–"}[ok]
        print(f"  {mark} {label:<34} {detail}")

    print("Environment check\n")
    ok = sys.version_info >= (3, 10)
    row(ok, "Python >= 3.10", sys.version.split()[0])
    if not ok:
        problems.append("Install Python 3.10+")
    in_venv = sys.prefix != sys.base_prefix
    row(in_venv or None, "Running inside venv", sys.prefix if in_venv else "not in venv (recommended)")

    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        ver = ""
        if path:
            ver = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.split("\n")[0][:60]
        row(bool(path), exe, ver or "missing")
        if not path:
            problems.append(f"{exe} missing: winget install Gyan.FFmpeg")

    for mod, pkg, required in (("pydantic", "pydantic", True), ("dotenv", "python-dotenv", True),
                               ("requests", "requests", True), ("google.genai", "google-genai", False),
                               ("pytest", "pytest", False)):
        try:
            found = importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError:  # parent package (e.g. "google") missing
            found = False
        row(found if required or found else None, f"python package {pkg}", "" if found else "not installed")
        if required and not found:
            problems.append(f"pip install -r requirements.txt  (missing {pkg})")

    env_file = ROOT / ".env"
    if env_file.is_file():
        from dotenv import dotenv_values
        env = {k: v for k, v in dotenv_values(env_file).items() if v}
    else:
        env = {}
        notes.append("No .env yet: copy .env.example to .env")
    env = {**env, **{k: os.environ[k] for k in ("INWORLD_API_KEY", "INWORLD_VOICE_ID", "GEMINI_API_KEY",
                                                "GOOGLE_API_KEY") if os.environ.get(k)}}
    row(env_file.is_file() or None, ".env file", "present" if env_file.is_file() else "missing")
    row(bool(env.get("INWORLD_API_KEY")) or None, "INWORLD_API_KEY", "set" if env.get("INWORLD_API_KEY")
        else "not set (needed for real voice)")
    has_google = bool(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"))
    row(has_google or None, "GEMINI_API_KEY", "set" if has_google else "not set (needed for Veo API; not for --manual-video)")

    print("\nPresenters")
    for d in sorted((ROOT / "presenters").glob("*/")):
        img = next((p for p in d.glob("presenter.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), None)
        sample = next(iter(d.glob("voice_sample.*")), None)
        voice = json.loads((d / "voice.json").read_text(encoding="utf-8")) if (d / "voice.json").is_file() else {}
        row(bool(img), d.name, f"image={'yes' if img else 'NO'}  voice_sample={'yes' if sample else 'no'}  "
                               f"voice_id={'yes' if voice.get('voice_id') else 'no'}")

    print("\nProjects")
    for d in sorted((ROOT / "projects").glob("*/")):
        if d.name.startswith("_"):
            continue
        state = d / "scenes" / "scenes.json"
        detail = "not started"
        if state.is_file():
            scenes = json.loads(state.read_text(encoding="utf-8"))["scenes"]
            done = sum(1 for s in scenes if s["status"] == "complete")
            detail = f"{done}/{len(scenes)} scenes complete"
        if (d / "output" / "final_video.mp4").is_file():
            detail += ", final video present"
        row((d / "input" / "script.txt").is_file(), d.name, detail)

    print()
    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("\nReady." if not problems else "\nFix the problems above first.")
    return 0 if not problems else 1


def new_project(name: str, presenter: str | None) -> int:
    d = ROOT / "projects" / name
    if d.exists():
        print(f"{d} already exists")
        return 1
    if presenter and not (ROOT / "presenters" / presenter).is_dir():
        print(f"Presenter '{presenter}' not found in presenters/")
        return 1
    (d / "input").mkdir(parents=True)
    (d / "input" / "script.txt").write_text("", encoding="utf-8")
    cfg = {"title": name, "presenter": presenter, "overrides": {}}
    if not presenter:
        cfg.pop("presenter")
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Created {d}\n  Put the lesson text in input/script.txt")
    if not presenter:
        print("  Add input/presenter.jpg and input/voice_sample.wav (10-30 s), or set a presenter in config.json")
    print(f"  Then run: python pipeline.py --project {name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Environment check / project scaffolding")
    ap.add_argument("--new-project")
    ap.add_argument("--presenter")
    a = ap.parse_args()
    sys.exit(new_project(a.new_project, a.presenter) if a.new_project else check())
