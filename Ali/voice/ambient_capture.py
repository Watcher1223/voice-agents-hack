"""Ambient listen loop — glass-style.

Always-on Deepgram stream. After every `AMBIENT_TRIGGER_EVERY_FINALS` final
transcripts arrive, runs `ambient_analysis.analyse` over the rolling
window. If the LLM decides to surface something (tier 1-3), the supplied
callback fires. Tier 4 (stay silent) produces no callback.

Design choices (from glass + our constraints):
- Turn-based trigger (not wall-clock): aligns with natural pauses.
- Rolling buffer capped at AMBIENT_HISTORY_TURNS so the prompt stays bounded.
- Prior analysis carried forward so we don't repeat ourselves.
- Analysis runs as a coroutine but does NOT block new transcripts from
  arriving — concurrent finals are buffered during the LLM call.
- All failures are silent (print + return); ambient must never crash the app.
"""
from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Callable


AMBIENT_TRIGGER_EVERY_FINALS = 5
AMBIENT_HISTORY_TURNS = 30


class AmbientCapture:
    """One long-running Deepgram session + glass-style analysis loop.

    Callbacks (called on the asyncio loop):
      on_interim(text)             — partial words from Deepgram
      on_final(text)               — committed utterance
      on_suggestion(analysis)      — AmbientAnalysis that passed the
                                      should_surface() gate
    """

    def __init__(
        self,
        on_interim: Callable[[str], None],
        on_final: Callable[[str], None],
        on_suggestion: Callable[["AmbientAnalysis"], None],
        screen_observer=None,
    ) -> None:
        self._on_interim    = on_interim
        self._on_final      = on_final
        self._on_suggestion = on_suggestion
        self._screen_observer = screen_observer

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._history: deque[str] = deque(maxlen=AMBIENT_HISTORY_TURNS)
        self._finals_since_trigger = 0
        self._analysis_in_flight = False
        self._previous = None  # type: ignore[assignment]
        self._loop: asyncio.AbstractEventLoop | None = None

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        from voice.deepgram_stream import (
            stream_transcription_sync,
            start_meeting_audio,
            stop_meeting_audio,
        )

        self._loop = asyncio.get_event_loop()
        start_meeting_audio()

        def _interim(text: str) -> None:
            if self._loop:
                self._loop.call_soon_threadsafe(self._on_interim, text)

        def _final(text: str) -> None:
            with self._lock:
                self._history.append(text)
                self._finals_since_trigger += 1
                count = self._finals_since_trigger
                total = len(self._history)
                should_trigger = (
                    count >= AMBIENT_TRIGGER_EVERY_FINALS
                    and not self._analysis_in_flight
                )
                if should_trigger:
                    self._finals_since_trigger = 0
            # One line per final utterance so the user can see the loop
            # is alive, how close we are to the next analysis, and what
            # was actually heard.
            marker = "↳ firing analysis" if should_trigger else f"({count}/{AMBIENT_TRIGGER_EVERY_FINALS} until next analysis)"
            print(f"[ambient] final {total:>2}: \"{text[:120]}\" {marker}")
            if self._loop:
                self._loop.call_soon_threadsafe(self._on_final, text)
            if should_trigger and self._loop:
                self._loop.call_soon_threadsafe(self._schedule_analysis)

        stream_thread = threading.Thread(
            target=stream_transcription_sync,
            args=(self._stop_event, _interim, _final),
            daemon=True,
        )
        stream_thread.start()

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
        finally:
            self._stop_event.set()
            stop_meeting_audio()
            stream_thread.join(timeout=3.0)

    def _schedule_analysis(self) -> None:
        # Run in background; buffer continues to receive finals meanwhile.
        if self._analysis_in_flight:
            return
        self._analysis_in_flight = True
        if self._loop:
            self._loop.create_task(self._do_analysis())

    async def _do_analysis(self) -> None:
        from intent.ambient_analysis import analyse

        try:
            with self._lock:
                history_snapshot = list(self._history)
            screen_app, screen_title, image = "", "", b""
            if self._screen_observer is not None:
                ctx = self._screen_observer.latest_context()
                screen_app = ctx.app
                screen_title = ctx.window_title
                image = ctx.image_bytes
            result = await analyse(
                history_snapshot,
                self._previous,
                screen_app=screen_app,
                screen_window_title=screen_title,
                screen_image_bytes=image,
            )
            if result.should_surface():
                self._previous = result
                print(f"[ambient] ✓ tier-{result.tier} surfaced: {result.headline[:100]}")
                self._on_suggestion(result)
            else:
                # Tier-4: stay silent, but keep the prior non-silent result
                # so next round still knows what NOT to repeat.
                print(f"[ambient] tier-{result.tier} — staying silent (nothing worth surfacing)")
        except Exception as e:
            print(f"[ambient] analysis loop error: {e}")
        finally:
            self._analysis_in_flight = False
