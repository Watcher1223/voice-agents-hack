"""
YC Voice Agent — entry point.
Ties all five layers together in a single push-to-talk loop.

Thread model
────────────
  Main thread : Qt event loop       (macOS AppKit requires UI on main thread)
  Background  : asyncio event loop  (voice capture, STT, intent, orchestrator)

The TranscriptionOverlay bridges the two via queue.Queue.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Callable, Protocol

from config.preflight import run_preflight_checks


class TranscriptionOverlay(Protocol):
    def push(self, state: str, text: str = "") -> None: ...


def _build_overlay() -> tuple["TranscriptionOverlay", Callable[[], None]]:
    backend = os.getenv("ALI_UI_BACKEND", "qt").strip().lower()
    if backend == "qt":
        import signal

        from PySide6.QtCore import QTimer  # pyright: ignore[reportMissingImports]
        from PySide6.QtWidgets import QApplication  # pyright: ignore[reportMissingImports]
        from ui.overlay import TranscriptionOverlay

        app = QApplication(sys.argv)
        overlay = TranscriptionOverlay(app)

        # Wire Ctrl+C / SIGTERM into Qt's event loop so the process can exit
        # cleanly and the menu-bar status icon disappears with it.
        def _handle_signal(*_args) -> None:
            print("[main] shutdown signal received — quitting")
            app.quit()

        signal.signal(signal.SIGINT, _handle_signal)
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, OSError):
            pass

        # Qt's event loop sleeps between UI events, which means Python signals
        # don't fire until there's user activity. A cheap periodic no-op timer
        # keeps the loop awake so SIGINT is delivered immediately.
        _signal_kick = QTimer()
        _signal_kick.start(200)
        _signal_kick.timeout.connect(lambda: None)
        overlay._signal_kick_timer = _signal_kick  # type: ignore[attr-defined]

        def _run_qt() -> None:
            try:
                app.exec()
            finally:
                os._exit(0)

        return overlay, _run_qt

    from ui.web_overlay import TranscriptionOverlay

    overlay = TranscriptionOverlay()
    return overlay, overlay.run_forever


# ── Asyncio agent loop (runs in background thread) ────────────────────────────

async def _agent_main(overlay: TranscriptionOverlay) -> None:
    from voice.capture import listen_for_command
    from voice.transcribe import transcribe, warmup
    from intent.parser import parse_intent
    from orchestrator.orchestrator import Orchestrator
    from ui.menu_bar import MenuBar

    orchestrator = Orchestrator()
    menu_bar = MenuBar()

    warmup()   # pre-load Whisper so first transcription is instant
    menu_bar.set_status("ready")

    async for audio_bytes in listen_for_command(overlay=overlay):
        try:
            print("\n─── New command ───────────────────────────────")

            # 1 — Transcribe
            menu_bar.set_status("transcribing")
            overlay.push("transcribing")
            print("[1/3] Transcribing...")
            transcript = await transcribe(audio_bytes)
            print(f'      → "{transcript}"')

            if not transcript.strip():
                print("      (empty transcript — skipping)")
                overlay.push("hidden")
                continue

            overlay.push("transcript", f'"{transcript}"')

            # 2 — Parse intent
            menu_bar.set_status("parsing intent")
            print("[2/3] Parsing intent...")
            intent = await parse_intent(transcript)
            print(f"      → goal={intent.goal.value}  slots={intent.slots}")

            goal_label = intent.goal.value.replace("_", " ").title()
            overlay.push("intent", f"{goal_label}")

            if intent.goal.value == "unknown":
                print("      (unknown intent — skipping execution)")
                overlay.push("error", "I didn't catch that — try rephrasing.")
                continue

            # 3 — Execute (known intent)
            print("[3/3] Executing...")
            menu_bar.set_status("running")
            overlay.push("action", f"Running: {goal_label}…")

            # For file-reveal flows, resolve first so we can announce the
            # target BEFORE Finder opens. The orchestrator's internal enrich
            # call is idempotent.
            revealed_name: str | None = None
            if intent.goal.value == "find_file":
                from intent.file_resolve import enrich_intent_with_resolved_files
                await enrich_intent_with_resolved_files(intent, transcript)
                revealed_name = _revealed_basename(intent)
                if revealed_name is not None:
                    overlay.push("revealed", revealed_name)

            await orchestrator.run(intent)

            print("      ✓ Done")
            if revealed_name is None:
                revealed_name = _revealed_basename(intent)
                if revealed_name is not None:
                    overlay.push("revealed", revealed_name)
                else:
                    overlay.push("done")

        except Exception as e:
            import traceback
            print(f"[error] {e}")
            traceback.print_exc()
            menu_bar.set_status("error")
            overlay.push("error", str(e))
        finally:
            menu_bar.set_status("ready")


def _revealed_basename(intent) -> str | None:
    """Return the basename of a FIND_FILE that resolved to an existing path, else None."""
    if intent.goal.value != "find_file":
        return None
    resolved = intent.slots.get("resolved_local_files") if isinstance(intent.slots, dict) else None
    if not isinstance(resolved, dict):
        return None
    path = resolved.get("found")
    if not isinstance(path, str) or not path:
        return None
    return os.path.basename(path)


def _run_agent(overlay: "TranscriptionOverlay") -> None:
    """Entry point for the background asyncio thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_agent_main(overlay))
    finally:
        loop.close()


def _start_wake_listener(overlay: "TranscriptionOverlay") -> None:
    """Listen for 'Ali' (voice) or backtick (keyboard fallback) to trigger wake."""
    from pynput import keyboard  # pyright: ignore[reportMissingModuleSource]

    def on_press(key):
        try:
            if key.char == "`":
                overlay.push("wake")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    print("[demo] Say 'Ali' or press ` (backtick) to trigger wake scene")

    try:
        from voice.wake_word import start_wake_word_listener
        start_wake_word_listener(lambda: overlay.push("wake"))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[demo] Wake word listener failed to start: {e}")


# ── Main (Qt on main thread) ──────────────────────────────────────────────────

def main() -> None:
    run_preflight_checks()

    overlay, run_ui = _build_overlay()

    agent_thread = threading.Thread(target=_run_agent, args=(overlay,), daemon=True)
    agent_thread.start()

    _start_wake_listener(overlay)

    # Block here on the main thread running the selected UI loop
    run_ui()


if __name__ == "__main__":
    main()
