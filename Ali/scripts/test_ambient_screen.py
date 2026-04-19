"""Smoke test for the screen observer + multimodal ambient analysis.

Runs one capture cycle, sends the current screen + a fake transcript to
Gemini, and prints what comes back. Use this to validate the screen
capture → sips → Gemini pipeline without needing to live-talk.

Requires macOS Screen Recording permission for the terminal running
this script.

    cd Ali && .venv/bin/python -m scripts.test_ambient_screen
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from intent.ambient_analysis import analyse  # noqa: E402
from observer.screen_loop import ScreenObserver  # noqa: E402


FAKE_TRANSCRIPTS = [
    "me: okay let me share my screen for a second",
    "me: i'm trying to figure out what's going on here",
    "them: what are you looking at",
    "me: there's some issue but i can't tell what",
    "me: wait actually can you help me read this",
]


async def main() -> None:
    print("[spike] starting ScreenObserver — give it ~3s to snap the first frame")
    obs = ScreenObserver()
    obs.start()
    # Wait for the first capture to happen.
    for _ in range(30):
        ctx = obs.latest_context()
        if ctx.has_image():
            break
        await asyncio.sleep(0.2)

    ctx = obs.latest_context()
    print(f"[spike] captured: app={ctx.app!r} title={ctx.window_title!r} bytes={len(ctx.image_bytes)}")
    if not ctx.has_image():
        print("[spike] screen capture failed — check Screen Recording permission in System Settings")
        obs.stop()
        return

    print("\n[spike] running multimodal analyse()…")
    t0 = time.perf_counter()
    result = await analyse(
        history=FAKE_TRANSCRIPTS,
        previous=None,
        screen_app=ctx.app,
        screen_window_title=ctx.window_title,
        screen_image_bytes=ctx.image_bytes,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"  latency: {ms}ms")
    print(f"  tier:     {result.tier}")
    print(f"  headline: {result.headline!r}")
    if result.detail:
        print(f"  detail:   {result.detail[:240]!r}")
    if result.action:
        print(f"  action:   {result.action}")

    obs.stop()


if __name__ == "__main__":
    asyncio.run(main())
