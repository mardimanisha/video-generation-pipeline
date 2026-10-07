"""Process every project under projects/ (or a chosen subset), one after another.

  python batch.py                                   # all projects
  python batch.py --only lesson-01 lesson-02
  python batch.py --manual-video                    # plan + audio + prompts for all, then collect clips

A failing project never stops the batch; each project resumes from its own checkpoint.
Folders starting with '_' are ignored.
"""
from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from pipeline import build_parser, run_project
from services.project_manager import REPO_ROOT
from utils.duration import fmt_seconds
from utils.logger import get_logger, log_event
from utils.tls import use_system_certificates


def discover() -> list[str]:
    root = REPO_ROOT / "projects"
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("_") and (d / "input" / "script.txt").is_file())


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(REPO_ROOT / ".env")
    use_system_certificates()
    ap = argparse.ArgumentParser(description="Batch-produce all presenter videos")
    ap.add_argument("--only", nargs="+", help="project names to process")
    ap.add_argument("--manual-video", action="store_true")
    ap.add_argument("--audio-provider", choices=["inworld", "mock"])
    ap.add_argument("--video-provider", choices=["gemini", "flow", "library", "mock", "manual"])
    args = ap.parse_args(argv)

    names = args.only or discover()
    if not names:
        print("No projects found (need projects/<name>/input/script.txt).")
        return 1
    log = get_logger("batch", REPO_ROOT / "logs" / "batch.log")
    log_event(log, "batch start", projects=names)
    results = []
    started = time.monotonic()
    for i, name in enumerate(names, 1):
        print(f"\n##### [{i}/{len(names)}] {name} #####")
        t0 = time.monotonic()
        pargs = build_parser().parse_args(["--project", name])
        pargs.manual_video = args.manual_video
        pargs.audio_provider, pargs.video_provider = args.audio_provider, args.video_provider
        try:
            report = run_project(name, pargs)
            ok = bool(report.get("final_file"))
            status = "done" if ok else f"incomplete ({report.get('failed_scenes', 0)} failed, " \
                                       f"{report.get('pending_scenes', 0)} pending)"
        except Exception as exc:  # noqa: BLE001 - one bad project must not stop the batch
            ok, status = False, f"error: {exc}".splitlines()[0]
        elapsed = time.monotonic() - t0
        results.append((name, ok, status, elapsed))
        log_event(log, "project finished", project=name, ok=ok, status=status, seconds=round(elapsed, 1))

    print("\n" + "=" * 60 + "\n BATCH SUMMARY\n" + "=" * 60)
    for name, ok, status, elapsed in results:
        print(f"  {'✓' if ok else '✗'} {name:<24} {status:<40} {fmt_seconds(elapsed)}")
    done = sum(1 for r in results if r[1])
    print(f"\n{done}/{len(results)} projects complete in {fmt_seconds(time.monotonic() - started)}")
    log_event(log, "batch end", complete=done, total=len(results))
    return 0 if done == len(results) else 3


if __name__ == "__main__":
    sys.exit(main())
