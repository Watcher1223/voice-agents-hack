"""End-to-end ambient test with a fake Deepgram stream.

Monkey-patches `voice.deepgram_stream.stream_transcription_sync` with a
scripted producer that emits pre-written final transcripts on a
realistic timeline. Everything else runs for real:
  - `observer.screen_loop.ScreenObserver` snaps the ACTUAL Mac screen
  - `voice.ambient_capture.AmbientCapture` runs the real rolling
    buffer + 5-turn trigger
  - `intent.ambient_analysis.analyse` makes the real multimodal Gemini
    call with the live screenshot

Output is captured to ~/.ali/ambient.log (same file main.py writes to)
so you can inspect it after. The script also prints a summary.

    cd Ali && .venv/bin/python -m scripts.test_ambient_live
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Fake Deepgram producer ─────────────────────────────────────────────────
# Two conversational chunks. After the first 5 utterances a tier-1/2 should
# fire. After the second 5 a different tier should fire (or tier-4 if the
# model decides it has nothing new to say).

SCRIPTED_FINALS = [
    (0.0, "hey I was reading the Anthropic paper"),
    (1.5, "it's about constitutional AI"),
    (3.0, "they train the model to critique itself"),
    (4.5, "against a set of principles"),
    (6.0, "wait what is constitutional AI actually"),     # ← triggers analysis #1
    (11.0, "okay got it"),
    (12.5, "I should email Hanzi about the pitch deck"),
    (14.0, "he wants it by Friday"),
    (15.5, "and we agreed to reply tonight"),
    (17.0, "remind me to send that email"),                # ← triggers analysis #2
]


def _fake_stream_transcription_sync(stop_event: threading.Event, on_interim, on_final):
    """Drop-in replacement for deepgram_stream.stream_transcription_sync.
    Emits scripted finals at their offsets and then idles until stopped."""
    start = time.monotonic()
    print("[fake-deepgram] Streaming started (scripted)")
    for offset, text in SCRIPTED_FINALS:
        while time.monotonic() - start < offset:
            if stop_event.is_set():
                print("[fake-deepgram] Stopped early")
                return
            time.sleep(0.05)
        on_final(text)
    print("[fake-deepgram] Done emitting script; idling")
    while not stop_event.is_set():
        time.sleep(0.1)
    print("[fake-deepgram] Streaming stopped")


# Install the patch BEFORE importing modules that pull stream_transcription_sync.
import voice.deepgram_stream as _dg  # noqa: E402
_dg.stream_transcription_sync = _fake_stream_transcription_sync
_dg.start_meeting_audio = lambda: None  # no real mic needed
_dg.stop_meeting_audio = lambda: None

from voice.ambient_capture import AmbientCapture  # noqa: E402
from observer.screen_loop import ScreenObserver  # noqa: E402


async def main() -> None:
    suggestions: list = []
    finals_seen: list[str] = []

    def _on_final(text: str) -> None:
        finals_seen.append(text)

    def _on_suggestion(analysis) -> None:
        suggestions.append(analysis)

    screen = ScreenObserver()
    screen.start()
    print("[test] waiting 3s for screen observer to snap first frame…")
    await asyncio.sleep(3.0)

    capture = AmbientCapture(
        on_interim=lambda _: None,
        on_final=_on_final,
        on_suggestion=_on_suggestion,
        screen_observer=screen,
    )

    # AmbientCapture.run() spawns the (now fake) Deepgram stream in a thread,
    # then loops forever until stop_event is set. Run it until the script is
    # fully emitted + some grace time for the final analysis.
    task = asyncio.create_task(capture.run())
    total_runtime_s = SCRIPTED_FINALS[-1][0] + 25  # last utterance + analysis budget
    print(f"[test] running ambient capture for {total_runtime_s:.0f}s…")
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=total_runtime_s)
    except asyncio.TimeoutError:
        pass
    capture.stop()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        pass
    screen.stop()

    print("\n=== TEST SUMMARY ===")
    print(f"finals delivered   : {len(finals_seen)}")
    print(f"suggestions fired  : {len(suggestions)}")
    for i, s in enumerate(suggestions, start=1):
        print(f"  [{i}] tier-{s.tier}  {s.headline[:100]}")
        if s.detail:
            print(f"      detail: {s.detail[:160]}")
        if s.action:
            print(f"      action: {s.action}")
    if not suggestions:
        print("(no suggestions — model chose tier-4 both passes)")
    print(f"\nlog file: {Path('~/.ali/ambient.log').expanduser()}")


if __name__ == "__main__":
    asyncio.run(main())
