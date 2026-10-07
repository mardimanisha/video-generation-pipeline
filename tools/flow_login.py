"""Open the pipeline's Flow Chrome window and wait until a Google account is signed in.

  venv\\Scripts\\python tools\\flow_login.py

Run once per machine (the profile in .browser-profile/ remembers the session).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.flow_session import FlowSession  # noqa: E402

if __name__ == "__main__":
    s = FlowSession(ROOT / ".browser-profile").start()
    try:
        s.ensure_signed_in()
        acct = s.page.locator('[aria-label^="Google Account"]').first.get_attribute("aria-label")
        print("SIGNED IN:", " ".join(acct.split()))
    finally:
        s.close()
