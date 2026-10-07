"""Google Flow (flow.google.com) as a video provider, driven through the Flow web app.

Flow has no public API, so this drives the signed-in web app in the pipeline's own Chrome
profile (see services/flow_session.py): one clip at a time, the way a person would.

Per clip:
  1. open the lesson's Flow project (created once, URL kept in <project>/flow.json)
  2. make sure the presenter image is uploaded to it (dropped onto the page once)
  3. start a fresh agent session, attach the uploaded image, type the prompt, send
  4. approve the agent's "costs N credits" question for this one clip
  5. wait for the new video, then fetch it with the browser's own session

Stops with a PermanentError on credit exhaustion or a policy refusal; the orchestrator then
halts video generation and a later run resumes from the same scene.
"""
from __future__ import annotations

import base64
import os
import time
import zipfile
from pathlib import Path

from services.config import FlowConfig, GenerationConfig, SyncConfig
from services.flow_session import FlowSession
from services.video_generator import ClipRequest, VideoProvider
from utils.files import read_json, sha256_file, write_json
from utils.retry import PermanentError, RateLimitError, TransientError

HOME = "https://flow.google.com/"

_DROP_JS = """
async ([b64, name, mime, x, y]) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const dt = new DataTransfer();
  dt.items.add(new File([bytes], name, {type: mime}));
  const target = document.elementFromPoint(x, y);
  for (const type of ['dragenter', 'dragover', 'drop']) {
    target.dispatchEvent(new DragEvent(type, {bubbles: true, cancelable: true, dataTransfer: dt,
                                                clientX: x, clientY: y}));
    await new Promise(r => setTimeout(r, 150));
  }
  return true;
}
"""

CREDIT_WORDS = ("not enough credits", "insufficient credits", "out of credits", "no credits",
                "credit limit", "run out of credits")
POLICY_WORDS = ("violat", "policy", "can't generate", "cannot generate", "unable to generate",
                "not able to generate", "prominent people")


class FlowBrowserVideoProvider(VideoProvider):
    name = "flow"
    supports_last_frame = False

    def __init__(self, generation: GenerationConfig, cfg: FlowConfig, state_file: Path,
                 root: Path, printer=print):
        super().__init__(generation)
        self.cfg = cfg
        self.state_file = state_file
        self.root = root
        self.print = printer
        self.session: FlowSession | None = None

    # one clip is always cfg.clip_seconds; lip alignment fits it to the narration
    def plan_durations(self, target: float, sync: SyncConfig) -> list[float]:
        # Lip alignment retimes the clip's speech to the narration (0.6-1.6x), so one clip covers
        # narration somewhat longer than the clip itself.
        # A clip must give the presenter time to finish the line: long lines get the long clip.
        if target <= self.cfg.long_clip_after:
            return [float(self.cfg.clip_seconds)]
        seconds = self.cfg.long_clip_seconds
        n = max(1, int(-(-target // (seconds * 0.9))))   # one 10 s clip covers up to 9 s of narration
        return [float(seconds)] * n

    # ------------------------------------------------------------------ session
    @property
    def page(self):
        if self.session is None:
            profile = Path(self.cfg.profile_dir)
            if not profile.is_absolute():
                profile = self.root / profile
            self.session = FlowSession(profile, port=self.cfg.debug_port, printer=self.print).start()
            self.session.ensure_signed_in()
        return self.session.page

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def _state(self) -> dict:
        return read_json(self.state_file, default={}) or {}

    def _save_state(self, **kw) -> None:
        write_json(self.state_file, {**self._state(), **kw})

    def _dismiss_banner(self) -> None:
        btn = self.page.get_by_role("button", name="Dismiss banner")
        if btn.count():
            btn.first.click()

    def _wait_ready(self, timeout_ms: int = 45000) -> bool:
        """The project page is usable once its prompt controls exist (it can hang on 'Loading...')."""
        try:
            self.page.get_by_role("button", name="Start new session").wait_for(state="visible", timeout=timeout_ms)
            return True
        except Exception:  # noqa: BLE001 - playwright timeout
            return False

    def _open_project(self) -> None:
        page = self.page
        url = self._state().get("project_url")
        if url:
            for attempt in range(3):
                if attempt == 0:
                    page.goto(url, wait_until="domcontentloaded")
                else:
                    self.print(f"  Flow project page did not finish loading; reloading ({attempt}/2)")
                    page.reload(wait_until="domcontentloaded")
                if "/project/" in page.url and self._wait_ready():
                    self._dismiss_banner()
                    return
            if "/project/" in page.url:
                raise TransientError("Flow project page keeps hanging on 'Loading...'")
        page.goto(HOME, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        page.get_by_text("New project", exact=True).first.click()
        page.wait_for_url("**/project/**", timeout=60000)
        self._wait_ready()
        self._save_state(project_url=page.url, reference_sha=None)
        self._dismiss_banner()

    def _ensure_reference(self, image: Path) -> None:
        """Upload the presenter image to the Flow project once (by dropping it on the page)."""
        digest = sha256_file(image)
        if self._state().get("reference_sha") == digest:
            return
        page = self.page
        box = page.locator("main").first.bounding_box() or {"x": 300, "y": 200, "width": 500, "height": 400}
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        page.evaluate(_DROP_JS, [base64.b64encode(image.read_bytes()).decode(), image.name, mime, x, y])
        page.wait_for_timeout(12000)
        self._save_state(reference_sha=digest, reference_uploaded=time.strftime("%Y-%m-%dT%H:%M:%S"))

    # --------------------------------------------------------------- generation
    def _attach_reference(self) -> None:
        page = self.page
        page.get_by_role("button", name="Add ingredients to the prompt box").click(timeout=20000)
        page.wait_for_timeout(1500)
        uploads = page.locator(".cdk-overlay-container").get_by_text("Uploads", exact=True)
        if uploads.count():
            uploads.first.click()
            page.wait_for_timeout(1200)
        tiles = page.locator(".cdk-overlay-container img")
        if not tiles.count():
            raise TransientError("Flow asset picker shows no uploaded image to attach")
        tiles.first.click()
        page.wait_for_timeout(1500)

    def _wrap_prompt(self, request: ClipRequest) -> str:
        body = " ".join(request.prompt.split())
        seconds = int(request.duration) if request.duration else self.cfg.clip_seconds
        return (f"Generate exactly one {seconds}-second 16:9 video (one output only) using the "
                f"attached image as the start frame. Use this prompt exactly as written, without rewriting "
                f"it: {body}")

    def _approve_if_asked(self) -> bool:
        btn = self.page.locator("button").filter(has_text="Approve").filter(has_not_text="Always")
        if btn.count() and btn.last.is_visible():
            btn.last.click()
            return True
        return False

    def _panel_tail(self) -> str:
        return " ".join(self.page.locator("body").inner_text().split())[-600:].lower()

    # ------------------------------------------------------ finding the result
    @staticmethod
    def _norm(text: str) -> str:
        text = text.replace('"', "'").replace("“", "'").replace("”", "'").replace("’", "'")
        return " ".join(text.split()).lower()

    def _matching_batches(self, line: str, seconds: int | None = None):
        """Batches (newest first) whose prompt contains this scene's spoken line and, if given,
        whose clip length is `seconds` (Flow labels batches like "720p • 8s")."""
        import re
        batches = self.page.locator("flow-batch-info")
        texts = batches.evaluate_all("els => els.map(e => e.innerText)")
        key = self._norm(line)
        length = re.compile(r"(?<![\d.])" + str(seconds) + r"s\b") if seconds else None
        return [batches.nth(i) for i, t in enumerate(texts)
                if key and key in self._norm(t) and (length is None or length.search(t))]

    def _try_download(self, batch, out: Path) -> bool:
        """Download a finished batch (Flow serves a zip holding the MP4). False if not ready."""
        from playwright.sync_api import TimeoutError as PWTimeout
        button = batch.get_by_role("button", name="Download batch")
        if not button.count():
            return False
        try:
            with self.page.expect_download(timeout=20000) as dl:
                button.first.click()
        except PWTimeout:
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.zip")
        dl.value.save_as(str(tmp))
        try:
            if zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as z:
                    mp4s = [n for n in z.namelist() if n.lower().endswith(".mp4")]
                    if not mp4s:
                        return False
                    part = out.with_name(out.stem + ".part.mp4")
                    part.write_bytes(z.read(mp4s[0]))
                    os.replace(part, out)
            else:  # a plain video download
                os.replace(tmp, out)
        finally:
            if tmp.exists():
                tmp.unlink()
        return out.is_file() and out.stat().st_size > 100_000

    # --------------------------------------------------------------- generation
    def generate(self, request: ClipRequest) -> Path:
        if request.reference_image is None or not Path(request.reference_image).is_file():
            raise PermanentError("Flow provider needs the presenter image (presenter.jpg)")
        page = self.page
        self._open_project()
        self._ensure_reference(Path(request.reference_image))
        line = request.dialogue
        seconds = int(request.duration) if request.duration else self.cfg.clip_seconds

        # A clip for this line may already exist from an earlier, interrupted attempt: reuse it if
        # finished, or keep waiting for it if still queued. Never pay twice for the same line.
        pending = self._matching_batches(line, seconds)
        for batch in pending:
            if self._try_download(batch, request.output_path):
                self.print(f"  {request.label}: reused an existing Flow clip for this line")
                return request.output_path
        if pending:
            self.print(f"  {request.label}: a Flow request for this line is still in progress; waiting for it")
            return self._wait_for(line, request, batches_before=None, start_tail=self._panel_tail(), seconds=seconds)

        page.get_by_role("button", name="Start new session").click(timeout=20000)
        page.wait_for_timeout(2500)
        self._attach_reference()
        prompt = self._wrap_prompt(request)
        box = page.locator("[contenteditable=true]").last
        box.click()
        page.keyboard.insert_text(prompt)  # one input event; per-key typing is far too slow
        page.wait_for_timeout(800)
        typed = " ".join(box.inner_text().split())
        if len(typed) < len(prompt) * 0.95:
            raise TransientError(f"Flow prompt box took {len(typed)} of {len(prompt)} characters")
        batches_before = page.locator("flow-batch-info").count()
        page.get_by_role("button", name="Start generation").click()
        return self._wait_for(line, request, batches_before, self._panel_tail(), seconds=seconds)

    def _wait_for(self, line: str, request: ClipRequest, batches_before, start_tail: str,
                  seconds: int | None = None) -> Path:
        page = self.page
        approved = False
        deadline = time.time() + self.cfg.generation_timeout_seconds
        while time.time() < deadline:
            page.wait_for_timeout(self.cfg.poll_seconds * 1000)
            if not approved:
                approved = self._approve_if_asked()
            candidates = self._matching_batches(line, seconds) if line else []
            if not line and batches_before is not None and page.locator("flow-batch-info").count() > batches_before:
                candidates = [page.locator("flow-batch-info").first]
            for batch in candidates:
                if self._try_download(batch, request.output_path):
                    return request.output_path
            tail = self._panel_tail()
            if tail != start_tail:
                if any(w in tail for w in CREDIT_WORDS):
                    raise RateLimitError("Flow: out of credits. Add credits or wait for the monthly refill.")
                if any(w in tail for w in POLICY_WORDS):
                    raise PermanentError(f"Flow refused this prompt: …{tail[-200:]}")
                if any(w in tail[-250:] for w in ("failed", "something went wrong")):
                    raise TransientError(f"Flow generation failed: …{tail[-200:]}")
        raise TransientError(f"Flow did not deliver the clip within {self.cfg.generation_timeout_seconds:.0f}s "
                             "(queue may be busy); a re-run picks it up if it finishes later")
