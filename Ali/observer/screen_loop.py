"""Event-driven screen capture for ambient mode.

Polls the active macOS app + window title every ~2s. When the focused
window changes OR the last stored screenshot is stale, snaps a new full
screen, compresses it via `sips`, and keeps the latest snapshot +
metadata in memory. The ambient analyser reads `latest_context()` before
each LLM call so suggestions can reference what's on screen.

No continuous frame stream — that would torch LLM budget and CPU. The
poll is cheap (one osascript round-trip), actual `screencapture` only
runs on a real event.

Design:
  - Single background thread loop (daemon).
  - No user notifications / no menu-bar flicker (no `-c` clipboard path).
  - JPEG-compressed + downsampled via `sips`; file stays in /tmp.
  - Thread-safe snapshot accessor returns a copy of the dataclass.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


POLL_INTERVAL_S = 2.0
STALE_AFTER_S = 15.0            # force a re-capture even if nothing changed
JPEG_MAX_DIMENSION_PX = 1280
JPEG_QUALITY = 70

_TMP_ROOT = Path("/tmp/ali-screen")
_TMP_PNG = _TMP_ROOT / "latest.png"
_TMP_JPG = _TMP_ROOT / "latest.jpg"


@dataclass
class ScreenContext:
    app: str = ""
    window_title: str = ""
    captured_at: float = 0.0
    image_path: str = ""
    image_bytes: bytes = field(default=b"", repr=False)

    def has_image(self) -> bool:
        return bool(self.image_bytes)


def _get_front_app_and_title() -> tuple[str, str]:
    """One osascript round-trip for app name + frontmost window title.
    Returns ('', '') on failure — ambient must fail quiet."""
    script = '''
tell application "System Events"
  set frontApp to name of first application process whose frontmost is true
  try
    set frontWin to name of front window of application process frontApp
  on error
    set frontWin to ""
  end try
end tell
return frontApp & "||" & frontWin
'''
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=1.5,
        )
        text = (out.stdout or "").strip()
        if "||" in text:
            app, title = text.split("||", 1)
            return app.strip(), title.strip()
    except Exception:
        pass
    return "", ""


def _capture_screen_jpeg() -> bytes:
    """Full-screen capture → downsample → JPEG bytes. Empty bytes on failure."""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["screencapture", "-x", "-t", "png", str(_TMP_PNG)],
            capture_output=True, timeout=4.0,
        )
        if r.returncode != 0 or not _TMP_PNG.exists():
            return b""
        # sips is macOS-builtin; cheaper than importing Pillow.
        sips = subprocess.run(
            [
                "sips", "-s", "format", "jpeg",
                "-s", "formatOptions", str(JPEG_QUALITY),
                "--resampleHeightWidthMax", str(JPEG_MAX_DIMENSION_PX),
                str(_TMP_PNG), "--out", str(_TMP_JPG),
            ],
            capture_output=True, timeout=4.0,
        )
        if sips.returncode != 0 or not _TMP_JPG.exists():
            return b""
        return _TMP_JPG.read_bytes()
    except Exception:
        return b""


class ScreenObserver:
    """Background thread that maintains the latest screen context.

    Usage:
        obs = ScreenObserver()
        obs.start()
        ... later ...
        ctx = obs.latest_context()
        obs.stop()
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ctx = ScreenContext()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ScreenObserver")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest_context(self) -> ScreenContext:
        """Return a shallow copy of the current context. Safe for reader
        threads — image_bytes is an immutable bytes object."""
        with self._lock:
            return ScreenContext(
                app=self._ctx.app,
                window_title=self._ctx.window_title,
                captured_at=self._ctx.captured_at,
                image_path=self._ctx.image_path,
                image_bytes=self._ctx.image_bytes,
            )

    def _run(self) -> None:
        last_app, last_title = "", ""
        while not self._stop.is_set():
            app, title = _get_front_app_and_title()
            with self._lock:
                stale = (time.monotonic() - self._ctx.captured_at) > STALE_AFTER_S
                changed = (app, title) != (last_app, last_title)
                need_capture = changed or stale or not self._ctx.has_image()

            if need_capture:
                image = _capture_screen_jpeg()
                now = time.monotonic()
                with self._lock:
                    self._ctx = ScreenContext(
                        app=app,
                        window_title=title,
                        captured_at=now,
                        image_path=str(_TMP_JPG) if image else "",
                        image_bytes=image,
                    )
                if changed:
                    print(f"[screen] focus → {app!r} / {title!r}  ({len(image)} bytes)")
                    # Warm the active-PDF cache whenever a PDF window comes
                    # into focus — by the time the user asks a question
                    # focus has usually moved to Ali / their editor.
                    try:
                        from intent.active_pdf import note_focus
                        cached = note_focus(app, title)
                        if cached is not None:
                            print(f"[screen] cached active PDF → {cached.name}")
                    except Exception:
                        pass
                last_app, last_title = app, title

            # Sleep in small steps so stop() lands fast.
            for _ in range(int(POLL_INTERVAL_S * 10)):
                if self._stop.is_set():
                    break
                time.sleep(0.1)
