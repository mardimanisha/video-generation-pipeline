"""A long-lived Chrome session for Google Flow, driven over the Chrome DevTools Protocol.

Chrome is started as a normal browser (not an automation-flagged one) with its own profile in
.browser-profile/, and Playwright attaches to it. The user signs in to Google in that window
once; the profile keeps the session. Nothing here enters credentials or bypasses any check:
if Google shows a sign-in page, the pipeline waits for the person to complete it.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

from utils.retry import PermanentError

FLOW_URL = "https://flow.google.com/"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).is_file():
            return c
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if not found:
        raise PermanentError("Google Chrome not found; install it or set flow.chrome_path")
    return found


class FlowSession:
    def __init__(self, profile_dir: Path, port: int = 9333, chrome_path: str | None = None,
                 printer=print):
        self.profile_dir = Path(profile_dir)
        self.port = port
        self.chrome_path = chrome_path or find_chrome()
        self.print = printer
        self._pw = None
        self.browser = None
        self.page = None

    def start(self) -> "FlowSession":
        if not _port_open(self.port):
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            subprocess.Popen([self.chrome_path, f"--remote-debugging-port={self.port}",
                              f"--user-data-dir={self.profile_dir.resolve()}", "--no-first-run",
                              "--no-default-browser-check", FLOW_URL],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.time() + 30
            while not _port_open(self.port):
                if time.time() > deadline:
                    raise PermanentError("Chrome did not open its debugging port")
                time.sleep(0.5)
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}")
        context = self.browser.contexts[0]
        self.page = next((p for p in context.pages if "flow.google.com" in p.url), None) \
            or (context.pages[0] if context.pages else context.new_page())
        self.page.bring_to_front()
        return self

    def signed_in(self) -> bool:
        return self.page.locator('[aria-label^="Google Account"]').count() > 0

    def ensure_signed_in(self, timeout: float = 1800) -> None:
        """Open Flow; if no Google account is signed in, wait for the person to sign in."""
        # check on the home page: a project page can hang on "Loading..." and show no account button
        self.page.goto(FLOW_URL, wait_until="domcontentloaded")
        try:
            self.page.locator('[aria-label^="Google Account"]').first.wait_for(state="visible", timeout=20000)
        except Exception:  # noqa: BLE001 - not signed in (or slow); fall through to the wait loop
            pass
        if self.signed_in():
            return
        self.print("\n  Google Flow needs you to sign in (one time).\n"
                   "  In the Chrome window that just opened, sign in with the Google account that\n"
                   "  has your Flow credits. The pipeline continues automatically afterwards.")
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(5)
            try:
                if "flow.google.com" not in self.page.url and "accounts.google.com" not in self.page.url:
                    self.page.goto(FLOW_URL, wait_until="domcontentloaded")
                if self.signed_in():
                    self.print("  Signed in. Continuing.")
                    return
            except Exception:  # noqa: BLE001 - page navigating during sign-in
                continue
        raise PermanentError("Timed out waiting for Google sign-in in the Flow window")

    def close(self) -> None:
        """Detach only; the Chrome window (and its signed-in session) stays open for reuse."""
        if self._pw:
            self._pw.stop()
            self._pw = None
