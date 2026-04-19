"""Realistic mic simulation for ambient mode.

Swaps the Deepgram backend with a scripted producer that emits the same
shape of events a real microphone would:

  • Interim transcripts grow character-by-character (Deepgram's partials).
  • Finals arrive with 200-700ms jitter after the last word, not
    synchronously with the last interim.
  • Proper nouns get STT-noised ("Hanzi" → "Henzi" / "Hansi" / "Kenzie")
    in some utterances, so we can see whether enrichment recovers.
  • A burst of finals can land while the LLM analysis call is in flight
    (tests the _analysis_in_flight guard).

Drives the real AmbientCapture + real Gemini analyse() + real screen
observer. AppleScript writes are monkey-patched so nothing actually
sends; calls are captured into a list.

    cd Ali && .venv/bin/python -m scripts.test_ambient_realistic
"""
from __future__ import annotations

import asyncio
import random
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Capture AppleScript writes ─────────────────────────────────────────────
captured_calls: list[tuple[str, dict]] = []


def _record(method_name):
    def _fn(self, **kwargs):
        captured_calls.append((method_name, kwargs))
        return None
    return _fn


from executors.local import applescript as _as_mod  # noqa: E402

_as_mod.AppleScriptExecutor.compose_mail          = _record("compose_mail")
_as_mod.AppleScriptExecutor.send_imessage         = _record("send_imessage")
_as_mod.AppleScriptExecutor.create_calendar_event = _record("create_calendar_event")


# ── Realistic final script ─────────────────────────────────────────────────
# Each entry: (offset_s, final_text, [optional interim partials to emit first])
# If partials omitted, we synthesize a growing prefix.

SCENARIOS: list[tuple[str, list[tuple[float, str]]]] = [
    (
        "she wants to email hanzi",
        [
            (0.5,  "hey"),
            (2.2,  "so I should follow up with hanzi tonight"),
            (4.0,  "she wants the pitch deck before friday"),
            (6.1,  "and we agreed I'd reply this evening"),
            # STT noise — a typical mishearing of 'Hanzi'
            (8.5,  "email henzi about the deck please"),
            (10.8, "write something short"),
        ],
    ),
    (
        "schedule meeting (stt noise + late final)",
        [
            (13.0, "put pitch prep on my calendar"),
            (15.2, "with hanzi friday at three pm"),
            (17.8, "block it for an hour"),
            (19.5, "make sure it's on my calendar"),
            (21.9, "schedule pitch prep friday 3 p.m. for one hour"),
        ],
    ),
    (
        "factual question mid-stream",
        [
            (25.0, "anyway what is IRR again"),
            (27.2, "I keep forgetting"),
            (29.0, "internal rate of return something"),
            (31.1, "just remind me the definition"),
            (33.5, "what does IRR actually mean"),
        ],
    ),
]


def _words(text: str) -> list[str]:
    return text.split()


def _fake_stream_transcription_sync(stop_event: threading.Event, on_interim, on_final):
    """Drop-in replacement for deepgram_stream.stream_transcription_sync
    that emits realistic interim+final bursts with jitter."""
    print("[fake-mic] realistic stream started")
    start = time.monotonic()
    rng = random.Random(42)

    all_scripted: list[tuple[float, str]] = []
    for _, finals in SCENARIOS:
        all_scripted.extend(finals)
    all_scripted.sort(key=lambda x: x[0])

    for offset, final_text in all_scripted:
        # Wait until offset seconds from start.
        while time.monotonic() - start < offset:
            if stop_event.is_set():
                return
            time.sleep(0.02)

        # Emit incremental interims — simulate Deepgram partials by
        # growing the transcript word-by-word with ~80-180ms between them.
        words = _words(final_text)
        prefix = ""
        for w in words:
            if stop_event.is_set():
                return
            prefix = (prefix + " " + w).strip()
            on_interim(prefix)
            time.sleep(rng.uniform(0.08, 0.18))

        # Jitter before the final: 200-700ms after the last interim.
        time.sleep(rng.uniform(0.2, 0.7))
        if stop_event.is_set():
            return
        on_final(final_text)

    # Idle until asked to stop so AmbientCapture doesn't early-exit.
    while not stop_event.is_set():
        time.sleep(0.05)
    print("[fake-mic] realistic stream stopped")


import voice.deepgram_stream as _dg  # noqa: E402
_dg.stream_transcription_sync = _fake_stream_transcription_sync
_dg.start_meeting_audio = lambda: None
_dg.stop_meeting_audio  = lambda: None


# ── Drive the ambient capture loop ─────────────────────────────────────────
from voice.ambient_capture import AmbientCapture  # noqa: E402
from observer.screen_loop import ScreenObserver   # noqa: E402


async def main() -> None:
    suggestions: list = []

    def _on_suggestion(analysis) -> None:
        suggestions.append(analysis)
        print(f"\n  ← [tier-{analysis.tier}] {analysis.headline[:100]}")
        if analysis.detail:
            print(f"       detail: {analysis.detail[:150]}")
        if analysis.action:
            print(f"       action: {analysis.action}")

    screen = ScreenObserver()
    screen.start()
    await asyncio.sleep(2.5)

    capture = AmbientCapture(
        on_interim=lambda _t: None,
        on_final=lambda _t: None,
        on_suggestion=_on_suggestion,
        screen_observer=screen,
    )

    # Route confirmed actions to the real execute path so AppleScript
    # (mocked) records the call.
    import main as ali_main
    orig_on_suggestion = _on_suggestion

    async def _execute_if_action(a) -> None:
        orig_on_suggestion(a)
        if a.tier == 3 and a.action and a.action.get("kind") == "local":
            await ali_main._execute_ambient_action(a, _FakeOverlay())

    # Replace the capture's on_suggestion with the execute-wired variant.
    def _sync_wrapper(a):
        asyncio.run_coroutine_threadsafe(_execute_if_action(a), capture._loop)  # type: ignore[arg-type]
    capture._on_suggestion = _sync_wrapper  # type: ignore[attr-defined]

    task = asyncio.create_task(capture.run())
    total = 45
    print(f"[test] running realistic mic simulation for {total}s…")
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=total)
    except asyncio.TimeoutError:
        pass
    capture.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        pass
    screen.stop()

    print("\n=== SUMMARY ===")
    print(f"  suggestions fired:     {len(suggestions)}")
    print(f"  applescript calls:     {len(captured_calls)}")
    for (method, kwargs) in captured_calls:
        preview = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in kwargs.items()}
        print(f"    → {method}({preview})")
    if not captured_calls:
        print("  (no calls — maybe no tier-3 suggestion with action)")


class _FakeOverlay:
    def push(self, *_a, **_kw): pass
    def set_pending_confirm(self, *_a, **_kw): pass
    def clear_pending_confirm(self): pass


if __name__ == "__main__":
    asyncio.run(main())
