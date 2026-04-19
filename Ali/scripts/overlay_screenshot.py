"""Render the overlay in each significant state and screenshot it.

Usage:
    python scripts/overlay_screenshot.py            # captures all states
    python scripts/overlay_screenshot.py --state pill_recording

Outputs to ~/tmp/ali_overlay_shots/*.png
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from PySide6.QtCore import QTimer  # type: ignore[reportMissingImports]
from PySide6.QtWidgets import QApplication  # type: ignore[reportMissingImports]

from ui.overlay import TranscriptionOverlay


OUT_DIR = Path.home() / "tmp" / "ali_overlay_shots"


def _screencap(overlay: TranscriptionOverlay, name: str) -> Path:
    """Capture the overlay window region and save to OUT_DIR/name.png.

    Uses `screencapture -R x,y,w,h` with a 20px halo so any drop-shadow or
    anti-aliasing bleed is preserved.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = overlay.geometry()
    halo = 24
    x, y, w, h = g.x() - halo, g.y() - halo, g.width() + 2 * halo, g.height() + 2 * halo
    out = OUT_DIR / f"{name}.png"
    subprocess.run(
        ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(out)],
        check=True,
    )
    print(f"  saved {out}")
    return out


def _setup(app: QApplication) -> TranscriptionOverlay:
    overlay = TranscriptionOverlay(app)
    # Give AppKit a beat to register the NSWindow + apply level/collection.
    app.processEvents()
    time.sleep(0.1)
    app.processEvents()
    return overlay


def _wait(ms: int) -> None:
    """Pump the Qt event loop for `ms` milliseconds so animations settle."""
    app = QApplication.instance()
    assert app is not None
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.016)


def capture_all(overlay: TranscriptionOverlay, only: str | None = None) -> list[Path]:
    results: list[Path] = []

    def shoot(name: str, *pushes: tuple[str, str], settle_ms: int = 500) -> None:
        if only and name != only:
            return
        for state, text in pushes:
            overlay.push(state, text)
        _wait(settle_ms)
        results.append(_screencap(overlay, name))

    # 1. Compact pill — recording (breathing dot + live bars)
    shoot(
        "01_pill_recording",
        ("recording", ""),
        settle_ms=800,
    )

    # 2. Expanded history — single user + assistant turn
    shoot(
        "02_expanded_single",
        ("transcript", "Text Hanzi I'll be late"),
        ("action", "Sent iMessage to Hanzi: \"Running 10 min late, sorry!\""),
        settle_ms=700,
    )

    # 3. Expanded history — multi-turn with a done badge
    shoot(
        "03_expanded_multi",
        ("hidden", ""),
        ("recording", ""),
        ("transcript", "Open my resume"),
        ("revealed", "resume.pdf"),
        ("done", ""),
        settle_ms=700,
    )

    # 4. Meeting capture — transcript + queued action items
    shoot(
        "04_meeting_active",
        ("hidden", ""),
        ("meeting_start", ""),
        ("meeting_final", "Alright, let's get started. Can we push the Q2 roadmap review to next Tuesday?"),
        ("meeting_interim", "I'll also need to sync with the design team about..."),
        ("meeting_action", "Reschedule Q2 roadmap review to next Tuesday"),
        ("meeting_action", "Sync with design team re: new onboarding flow"),
        ("meeting_action_update", "Reschedule Q2 roadmap review to next Tuesday|Running"),
        settle_ms=900,
    )

    # 5. Error state (dim red) — meeting_stop first so we leave meeting_mode
    shoot(
        "05_error",
        ("meeting_stop", ""),
        ("recording", ""),
        ("transcript", "Email Corinne about the offer"),
        ("error", "Couldn't find Corinne in Contacts"),
        settle_ms=600,
    )

    return results


def run_live_demo(overlay: TranscriptionOverlay) -> None:
    """Cycle through every state on-screen with pauses so you can watch the
    spring expand, breathing dot, and audio bars animate live."""
    hold_ms = 3000  # linger on each state this long

    print("live demo — watch the top of your screen (Ctrl+C to stop)")

    def step(label: str, *pushes: tuple[str, str]) -> None:
        print(f"  → {label}")
        for s, t in pushes:
            overlay.push(s, t)
        _wait(hold_ms)

    step("01  pill — recording (breathing dot + live bars)",
         ("recording", ""))

    step("02  expanded — single turn",
         ("transcript", "Text Hanzi I'll be late"),
         ("action", "Sent iMessage to Hanzi: \"Running 10 min late, sorry!\""))

    step("03  expanded — multi-turn done badge",
         ("hidden", ""),
         ("recording", ""),
         ("transcript", "Open my resume"),
         ("revealed", "resume.pdf"),
         ("done", ""))

    step("04  meeting capture",
         ("hidden", ""),
         ("meeting_start", ""),
         ("meeting_final", "Alright, let's get started. Can we push the Q2 roadmap review to next Tuesday?"),
         ("meeting_interim", "I'll also need to sync with the design team about..."),
         ("meeting_action", "Reschedule Q2 roadmap review to next Tuesday"),
         ("meeting_action", "Sync with design team re: new onboarding flow"),
         ("meeting_action_update", "Reschedule Q2 roadmap review to next Tuesday|Running"))

    step("05  error",
         ("meeting_stop", ""),
         ("recording", ""),
         ("transcript", "Email Corinne about the offer"),
         ("error", "Couldn't find Corinne in Contacts"))

    print("  → done (hiding)")
    overlay.push("hidden", "")
    _wait(400)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", help="capture only the named state (e.g. 01_pill_recording)")
    ap.add_argument("--live", action="store_true",
                    help="cycle states on-screen with pauses (watch the motion)")
    args = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    overlay = _setup(app)

    try:
        if args.live:
            run_live_demo(overlay)
        else:
            results = capture_all(overlay, only=args.state)
            print()
            print(f"captured {len(results)} screenshots → {OUT_DIR}")
            for p in results:
                print(f"  {p}")
    finally:
        overlay.hide()
        app.processEvents()


if __name__ == "__main__":
    main()
