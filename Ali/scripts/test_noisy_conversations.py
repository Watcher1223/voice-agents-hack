"""Torture-test ambient extraction with STT-noised conversations.

Feeds multiple conversations with deliberate mishearings ("hamsi" for
"Hanzi", "cat us" for "cactus", etc.) through the real ambient analyser
pipeline and prints what the tasks store would end up with.

Bypasses the mic + Deepgram — the scripted finals drive AmbientCapture
directly. Screen observer is on so the live Mac screen context reaches
the multimodal analysis call.

    cd Ali && .venv/bin/python -m scripts.test_noisy_conversations
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Mock AppleScript writes so we don't actually send anything ─────────────
captured_calls: list[tuple[str, dict]] = []


def _record(name):
    def _fn(self, **kwargs):
        captured_calls.append((name, kwargs))
        return None
    return _fn


from executors.local import applescript as _as_mod
_as_mod.AppleScriptExecutor.compose_mail          = _record("compose_mail")
_as_mod.AppleScriptExecutor.send_imessage         = _record("send_imessage")
_as_mod.AppleScriptExecutor.create_calendar_event = _record("create_calendar_event")


# ── Point the tasks store at a scratch file ────────────────────────────────
_TMP_TASKS = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)

import executors.local.tasks_store as _ts
_ts._STORE_PATH = _TMP_TASKS
from executors.local.tasks_store import TasksStore
store = TasksStore(path=_TMP_TASKS)


# ── Fake Deepgram emitting scripted scenarios ─────────────────────────────
SCENARIOS = [
    # Label, [(offset_s, final_text), ...]
    (
        "Scenario A — Hanzi misheard as 'hamsi'",
        [
            (0.5,  "hey so I should follow up with hamsi tonight"),
            (2.4,  "she wants the pitch deck before friday"),
            (4.1,  "and we agreed I'd reply this evening"),
            (6.0,  "write a short email to hamsi"),
            (8.0,  "make sure it mentions the deck"),
        ],
    ),
    (
        "Scenario B — Calendar event with noisy date parse",
        [
            (11.0, "let's block pitch prep for friday afternoon"),
            (12.7, "with hamsi"),
            (14.2, "three pm for an hour"),
            (16.0, "remind me to put it on my calendar"),
            (18.1, "schedule pitch prep friday 3 p.m."),
        ],
    ),
    (
        "Scenario C — Factual question mid-stream",
        [
            (21.0, "wait remind me"),
            (22.6, "what is IRR again"),
            (24.4, "internal rate of return something"),
            (26.0, "I keep forgetting the definition"),
            (28.0, "just explain IRR one more time"),
        ],
    ),
    (
        "Scenario D — iMessage with heavy STT noise",
        [
            (31.0, "text kenzie I'm running ten minutes late"),
            (33.2, "tell her to wait at the office"),
            (34.7, "she needs to know before the call"),
            (36.4, "send the iMessage now"),
            (38.2, "let her know quickly"),
        ],
    ),
]


def _fake_stream_transcription_sync(stop_event, on_interim, on_final):
    start = time.monotonic()
    print("[fake-mic] scripted stream started")
    all_scripted = []
    for _, finals in SCENARIOS:
        all_scripted.extend(finals)
    all_scripted.sort(key=lambda x: x[0])
    for offset, text in all_scripted:
        while time.monotonic() - start < offset:
            if stop_event.is_set():
                return
            time.sleep(0.02)
        # Emit growing interim partials for realism.
        words = text.split()
        prefix = ""
        for w in words:
            if stop_event.is_set():
                return
            prefix = (prefix + " " + w).strip()
            on_interim(prefix)
            time.sleep(0.05)
        # Jitter before the final
        time.sleep(0.3)
        on_final(text)
    while not stop_event.is_set():
        time.sleep(0.05)
    print("[fake-mic] scripted stream stopped")


import voice.deepgram_stream as _dg
_dg.stream_transcription_sync = _fake_stream_transcription_sync
_dg.start_meeting_audio = lambda: None
_dg.stop_meeting_audio  = lambda: None


# ── Drive AmbientCapture ────────────────────────────────────────────────────
from voice.ambient_capture import AmbientCapture
from observer.screen_loop import ScreenObserver
import main as ali_main

ali_main._tasks_store = store
ali_main._agent_loop = None  # not needed for this test — we call execute directly


class FakeOverlay:
    def __init__(self): self.pushes: list = []
    def push(self, state, text=""): self.pushes.append((state, text))
    def set_pending_confirm(self, *a, **kw): pass
    def clear_pending_confirm(self): pass
    def refresh_tasks(self): pass


async def main() -> None:
    surfaced: list = []
    overlay = FakeOverlay()
    ali_main._overlay_ref = overlay

    def _on_suggestion(analysis) -> None:
        surfaced.append(analysis)
        if analysis.tier == 3 and analysis.action:
            # Enrich + add to store the same way main.py's _on_suggestion does.
            enriched = ali_main._enrich_analysis_for_preview(analysis)
            a = enriched.action or {}
            store.add(
                headline=enriched.headline,
                detail=enriched.detail,
                action_kind=a.get("kind", "local"),
                action_text=a.get("text", ""),
                slots=a.get("slots") or {},
            )

    screen = ScreenObserver()
    screen.start()
    await asyncio.sleep(2.5)

    capture = AmbientCapture(
        on_interim=lambda _t: None,
        on_final=lambda _t: None,
        on_suggestion=_on_suggestion,
        screen_observer=screen,
    )
    task = asyncio.create_task(capture.run())
    duration = SCENARIOS[-1][1][-1][0] + 20
    print(f"[test] running noisy conversation for {duration:.0f}s…\n")
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=duration)
    except asyncio.TimeoutError:
        pass
    capture.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        pass
    screen.stop()

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n═════════ RESULTS ═════════\n")
    print(f"Suggestions fired: {len(surfaced)}")
    for i, s in enumerate(surfaced, start=1):
        print(f"\n  [{i}] tier-{s.tier}  {s.headline[:80]}")
        if s.detail:
            print(f"       detail: {s.detail[:140]}")
        if s.action:
            print(f"       action: {s.action}")

    pending = store.pending()
    print(f"\nTasks store has {len(pending)} pending task(s):")
    for t in pending:
        print(f"  · {t.headline}   kind={t.action_kind} text={t.action_text}")
        print(f"    slots={t.slots}")

    _TMP_TASKS.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
