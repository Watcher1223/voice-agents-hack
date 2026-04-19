"""Fake-transcript end-to-end test of the ambient loop.

Skips Deepgram — we can't audition real audio from a headless script —
and directly drives `AmbientCapture` by pumping synthetic finals into
its buffer, running the analysis step, and verifying the suggestion
callback fires on tier 1-3.

This exercises the real rolling-buffer logic + 5-turn trigger + the
multimodal Gemini call (with live screen capture). It's the closest
thing to a real ambient session short of actually talking.

    cd Ali && .venv/bin/python -m scripts.test_ambient_e2e
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from voice.ambient_capture import AmbientCapture  # noqa: E402
from observer.screen_loop import ScreenObserver  # noqa: E402


UTTERANCES = [
    # 5 finals → triggers analysis at utterance 5
    "me: i was reading the Anthropic paper on constitutional AI",
    "them: oh cool, does that mean they trained on rules",
    "me: basically yeah, the model critiques its own outputs",
    "me: and they call it a constitution",
    "me: so what does constitutional AI mean exactly",
    # next 5 → second analysis pass (previous result should suppress repeat)
    "them: gotcha",
    "me: do you think we should try something similar",
    "me: for our agent maybe",
    "them: might be overkill for a hackathon",
    "me: fair — let's just use Gemini Flash for now",
]


async def main() -> None:
    received: list = []

    def _on_suggestion(analysis) -> None:
        received.append(analysis)
        print(
            f"  ← suggestion tier={analysis.tier} "
            f"headline={analysis.headline[:80]!r}"
        )
        if analysis.detail:
            print(f"    detail: {analysis.detail[:160]!r}")
        if analysis.action:
            print(f"    action: {analysis.action}")

    screen = ScreenObserver()
    screen.start()
    # Give the observer a moment for its first snapshot.
    await asyncio.sleep(2.5)

    capture = AmbientCapture(
        on_interim=lambda _: None,
        on_final=lambda _: None,
        on_suggestion=_on_suggestion,
        screen_observer=screen,
    )
    capture._loop = asyncio.get_event_loop()

    try:
        for i, utterance in enumerate(UTTERANCES, start=1):
            print(f"\n[u{i}] {utterance}")
            # Mimic what the Deepgram-feeding _final() does in ambient_capture.run().
            with capture._lock:
                capture._history.append(utterance)
                capture._finals_since_trigger += 1
                should_trigger = (
                    capture._finals_since_trigger >= 5
                    and not capture._analysis_in_flight
                )
                if should_trigger:
                    capture._finals_since_trigger = 0
                    capture._analysis_in_flight = True
            if should_trigger:
                print(f"  → triggering analysis (history={len(capture._history)} turns)")
                await capture._do_analysis()

        # Summary
        print("\n=== summary ===")
        print(f"  analyses fired: {len(received)}")
        for i, r in enumerate(received, start=1):
            print(f"  [{i}] tier-{r.tier}: {r.headline[:120]}")
    finally:
        screen.stop()


if __name__ == "__main__":
    asyncio.run(main())
