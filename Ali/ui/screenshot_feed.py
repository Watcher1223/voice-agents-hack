"""
Browser screenshot helpers for the meeting overlay.

Captures a snapshot of the Chrome front window (falling back to full
screen) when an action item finishes, so the overlay can show a live
thumbnail that the user can click to bring Chrome to the foreground.

No live/continuous feed yet — one snapshot per completed action keeps
things simple and avoids paint thrashing.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

_THUMB_DIR = os.path.join(tempfile.gettempdir(), "ali_browser_thumbs")
os.makedirs(_THUMB_DIR, exist_ok=True)


def _thumb_path(task_label: str) -> str:
    digest = hashlib.sha1(task_label.encode("utf-8")).hexdigest()[:10]
    return os.path.join(_THUMB_DIR, f"{digest}.png")


def _chrome_window_id() -> int | None:
    """Return the Quartz window number of Chrome's frontmost window, or None."""
    try:
        import Quartz  # type: ignore[reportMissingImports]
    except Exception:
        return None
    try:
        windows = Quartz.CGWindowListCopyWindowInfo(  # type: ignore[attr-defined]
            Quartz.kCGWindowListOptionOnScreenOnly,  # type: ignore[attr-defined]
            Quartz.kCGNullWindowID,  # type: ignore[attr-defined]
        )
    except Exception:
        return None
    for w in windows or []:
        owner = w.get("kCGWindowOwnerName", "") or ""
        if owner.lower() in ("google chrome", "chrome"):
            num = w.get("kCGWindowNumber")
            if isinstance(num, int):
                return num
    return None


def capture_browser_thumb(task_label: str) -> str | None:
    """
    Grab a screenshot of the Chrome front window (or full screen fallback)
    and save as PNG. Returns the path on success, None on failure.

    Uses macOS `screencapture`. No-op / None on non-darwin.
    """
    import sys
    if sys.platform != "darwin":
        return None
    path = _thumb_path(task_label)
    win_id = _chrome_window_id()
    cmd = ["/usr/sbin/screencapture", "-x", "-t", "png", "-o"]
    if win_id is not None:
        cmd += ["-l", str(win_id)]
    cmd += [path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=4)
        if r.returncode != 0 or not os.path.exists(path):
            # Retry full-screen if window-capture failed.
            r = subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-t", "png", path],
                capture_output=True, timeout=4,
            )
            if r.returncode != 0 or not os.path.exists(path):
                return None
        return path
    except Exception as e:
        print(f"[screenshot_feed] capture failed: {e}")
        return None


def focus_chrome() -> None:
    """Bring Chrome to front. No-op off-macOS."""
    import sys
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e", 'tell application "Google Chrome" to activate'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
        )
    except Exception:
        pass
