import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A throwaway repo root with the real config + prompts and one small presenter."""
    shutil.copytree(REPO / "config", tmp_path / "config")
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")
    pres = tmp_path / "presenters" / "p1"
    pres.mkdir(parents=True)
    (pres / "presenter.json").write_text('{"appearance": {"clothing": "navy suit"}}', encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360",
                    "-frames:v", "1", str(pres / "presenter.jpg")], check=True)
    return tmp_path


def make_project(root: Path, name: str, script: str, presenter: str | None = "p1", overrides: dict | None = None):
    import json
    d = root / "projects" / name
    (d / "input").mkdir(parents=True)
    (d / "input" / "script.txt").write_text(script, encoding="utf-8")
    fast = {"video": {"width": 320, "height": 180, "preset": "ultrafast"},
            "retry": {"max_attempts": 2, "backoff_seconds": [0, 0]}}
    if overrides:
        fast.update(overrides)
    cfg = {"title": name, "overrides": fast}
    if presenter:
        cfg["presenter"] = presenter
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d
