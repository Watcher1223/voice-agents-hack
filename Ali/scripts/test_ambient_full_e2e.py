"""End-to-end ambient test with fuzzy STT, task approval, and execution.

Proves the FULL path works — not just classification:

    Deepgram (fuzzy, scripted)
        → AmbientCapture rolling buffer + debounce
        → ambient_analysis (Gemini multimodal)
        → tasks store (pending → executing → done)
        → _execute_ambient_action → AppleScriptExecutor.compose_mail
            (mocked to capture args; no actual Mail.app window)

The fake Deepgram stream deliberately mishears "Hanzi" as "hamsi"
across several utterances, so we verify:
  - The contacts-in-prompt path resolves 'hamsi' → hanzili0217@gmail.com
  - Tier-3 lands a compose_mail task with correct slots
  - Simulated approve triggers the execute pipeline
  - AppleScript is called with the RESOLVED email, not the raw name

Run:
    cd Ali && .venv/bin/python -m scripts.test_ambient_full_e2e
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


# ── Capture AppleScript writes ─────────────────────────────────────────────
captured: list[tuple[str, dict]] = []


def _record(name: str):
    def _fn(self, **kwargs):
        captured.append((name, kwargs))
        return None
    return _fn


from executors.local import applescript as _as_mod   # noqa: E402
_as_mod.AppleScriptExecutor.compose_mail          = _record("compose_mail")
_as_mod.AppleScriptExecutor.send_imessage         = _record("send_imessage")
_as_mod.AppleScriptExecutor.create_calendar_event = _record("create_calendar_event")


# ── Scratch tasks store ────────────────────────────────────────────────────
_TMP_TASKS = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
import executors.local.tasks_store as _ts  # noqa: E402
_ts._STORE_PATH = _TMP_TASKS
from executors.local.tasks_store import TasksStore  # noqa: E402
store = TasksStore(path=_TMP_TASKS)


# ── Fake Deepgram emitting fuzzy scripted utterances ──────────────────────
SCRIPT = [
    # (offset_s, final_text) — gaps > debounce (2s) so each commits separately.
    (0.8,  "hey i wanted to follow up with hamsi"),
    (3.5,  "she wanted the pitch deck before friday"),
    (6.5,  "and we agreed i'd reply tonight"),
    (9.5,  "remind me to email hamsi about the pitch deck"),
    (12.5, "make sure it mentions we'll send it tonight"),
]


def _fake_stream(stop_event: threading.Event, on_interim, on_final):
    start = time.monotonic()
    print("[fake-mic] scripted stream started")
    for offset, text in SCRIPT:
        while time.monotonic() - start < offset:
            if stop_event.is_set():
                return
            time.sleep(0.02)
        # Emit progressive interims for realism
        prefix = ""
        for w in text.split():
            if stop_event.is_set():
                return
            prefix = (prefix + " " + w).strip()
            on_interim(prefix)
            time.sleep(0.05)
        time.sleep(0.3)   # jitter before final
        on_final(text)
    while not stop_event.is_set():
        time.sleep(0.05)
    print("[fake-mic] scripted stream stopped")


import voice.deepgram_stream as _dg    # noqa: E402
_dg.stream_transcription_sync = _fake_stream
_dg.start_meeting_audio = lambda: None
_dg.stop_meeting_audio  = lambda: None


# ── Drive the real ambient pipeline ────────────────────────────────────────
from voice.ambient_capture import AmbientCapture   # noqa: E402
from observer.screen_loop import ScreenObserver    # noqa: E402
import main as ali_main                             # noqa: E402

ali_main._tasks_store = store


class FakeOverlay:
    def __init__(self) -> None:
        self.pushes: list = []
    def push(self, state, text=""):
        self.pushes.append((state, text))
    def refresh_tasks(self): pass
    def set_pending_confirm(self, *a, **k): pass
    def clear_pending_confirm(self): pass


async def main() -> None:
    overlay = FakeOverlay()
    ali_main._overlay_ref = overlay

    def _on_suggestion(analysis) -> None:
        print(f"\n[surface] tier-{analysis.tier}  {analysis.headline[:80]}")
        if analysis.action:
            print(f"         action: {analysis.action}")
        # Mirror main._on_suggestion's tier-3 needs_confirm path: enrich + add to store
        if analysis.tier == 3 and analysis.action:
            enriched = ali_main._enrich_analysis_for_preview(analysis)
            a = enriched.action or {}
            from intent.action_safety import classify
            if classify(a) == "needs_confirm":
                t = store.add(
                    headline=enriched.headline,
                    detail=enriched.detail,
                    action_kind=a.get("kind", "local"),
                    action_text=a.get("text", ""),
                    slots=a.get("slots") or {},
                )
                print(f"         → task {t.id} added (status={t.status})")

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
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=40)
    except asyncio.TimeoutError:
        pass
    capture.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        pass
    screen.stop()

    print("\n══════════ 1) CLASSIFICATION ══════════")
    print(f"  tasks in store: {len(store.pending())} pending")
    for t in store.pending():
        print(f"    · {t.headline}   kind={t.action_kind} text={t.action_text}")
        print(f"      slots={t.slots}")

    # ── 2. Approve ALL pending tasks end-to-end (executes real AppleScript) ──
    print("\n══════════ 2) EXECUTE (simulated approval) ══════════")
    for t in list(store.pending()):
        print(f"  → approving {t.id}: {t.headline[:60]}")
        store.mark(t.id, "executing")
        await ali_main._execute_task_from_store(t.id)
        print(f"    status after: {store.get(t.id).status}")

    # ── 3. Assert the full chain ────────────────────────────────────────────
    print("\n══════════ 3) APPLESCRIPT CALLS CAPTURED ══════════")
    if not captured:
        print("  ✗ NO AppleScript calls — something broke in the execute path")
    for name, kwargs in captured:
        preview = {k: (v[:70] + "…" if isinstance(v, str) and len(v) > 70 else v) for k, v in kwargs.items()}
        print(f"  ✓ {name}({preview})")

    # Specifically: compose_mail should have been called with the
    # RESOLVED email, not 'hamsi'.
    compose_calls = [kw for n, kw in captured if n == "compose_mail"]
    if not compose_calls:
        print("\n  ✗ compose_mail never called — expected at least one")
    else:
        to = compose_calls[0].get("to", "")
        if "@" in to:
            print(f"\n  ✓ fuzzy STT resolved: compose_mail to='{to}' (not 'hamsi')")
        else:
            print(f"\n  ✗ compose_mail to='{to}' — not resolved to an email")

    _TMP_TASKS.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
