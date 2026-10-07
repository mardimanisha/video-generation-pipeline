r"""Give every clip the same look.

Each generated clip comes out with a slightly different colour balance, brightness and
contrast; cutting between them reads as a filter change. This measures each scene's rendered
frames (the part of the clip actually used) and stores on the scene an FFmpeg `lutrgb` filter
that maps each colour channel's mean and spread to one project-wide target. The render applies
it (services/ffmpeg.build_fit_render_cmd) and re-renders when it changes. Run it on renders made
without the filter (use --clear and re-render first if a filter is already set).

    venv\Scripts\python tools\color_match.py --project s01-t01-what-is-entrepreneurship
    venv\Scripts\python tools\color_match.py --project ... --clear
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from services.project_manager import Project  # noqa: E402

FPS = 4


def measure(video: Path) -> np.ndarray:
    """Per-channel mean and standard deviation over the video's frames: [[mR, mG, mB], [sR, sG, sB]]."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FPS},scale=160:90",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(float)
    return np.array([a.mean(0), a.std(0)])


def lut_filter(m: np.ndarray, target: np.ndarray) -> str:
    parts = []
    for ch, name in enumerate("rgb"):
        k = target[1][ch] / m[1][ch]
        parts.append(f"{name}='clip((val-{m[0][ch]:.2f})*{k:.4f}+{target[0][ch]:.2f},0,255)'")
    return "lutrgb=" + ":".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", nargs="*", help="only these scenes (default: all)")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--measure-dir", type=Path, help="unfiltered scene renders (default: rendered/)")
    a = ap.parse_args()
    proj = Project(a.project)
    path = proj.scenes_dir / "scenes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    if a.clear:
        for s in scenes:
            if not a.scene or s["id"] in a.scene:
                s.pop("color_filter", None)
    else:
        if any(sc.get("color_filter") for sc in scenes):
            sys.exit("scenes already have a colour filter: run with --clear, re-render, then run again")
        src = a.measure_dir or proj.rendered_dir
        stats = {sc["id"]: measure(src / f"{sc['id']}.mp4") for sc in scenes}
        target = np.median(np.array(list(stats.values())), axis=0)
        print("target mean", target[0].round(1), "spread", target[1].round(1))
        for sc in scenes:
            if not a.scene or sc["id"] in a.scene:
                sc["color_filter"] = lut_filter(stats[sc["id"]], target)
        print(f"colour filter set on {sum(1 for sc in scenes if sc.get('color_filter'))} scenes")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
