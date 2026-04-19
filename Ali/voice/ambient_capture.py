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

from observer.agent_log import log as _agent_log


AMBIENT_TRIGGER_EVERY_FINALS = 5
AMBIENT_HISTORY_TURNS = 30
# Debounce raw Deepgram finals: if another final arrives <N seconds
# after the last one, merge them into a single turn. Ported from glass
# (src/features/listen/stt/sttService.js:COMPLETION_DEBOUNCE_MS=2000).
# Gives the speaker time to finish a thought before the analyzer counts
# it as a completed utterance.
DEBOUNCE_SECONDS = 2.0

# Turns that begin with any of these direct-address prefixes skip the
# 5-final wait and fire a focused single-turn analysis immediately —
# the "Ali, do X" voice shortcut. Works without macOS Accessibility
# permissions because it's just substring-matching on Deepgram text.
import re as _re
# Direct address: must have "ali" token, optionally prefixed by a
# greeting. Earlier version allowed "hey" alone which fired on every
# sentence starting with "Hey so …".
_DIRECT_ADDRESS_RE = _re.compile(
    r"^\s*(?:hey\s+|ok\s+|okay\s+)?ali\s*[,:\s]+(.+)$",
    _re.IGNORECASE,
)
# Matches a turn that is JUST "Ali" / "Hey Ali" / "Okay Ali" (no body).
# When heard, we arm the next non-empty turn as direct-address for
# DIRECT_ADDRESS_ARM_SECONDS so natural flow "Ali… [pause] …what is X?"
# works the same as "Ali, what is X?".
_BARE_DIRECT_ADDRESS_RE = _re.compile(
    r"^\s*(?:hey\s+|ok\s+|okay\s+)?ali\s*[.!?,:]*\s*$",
    _re.IGNORECASE,
)
DIRECT_ADDRESS_ARM_SECONDS = 8.0


def _strip_direct_address(text: str) -> str | None:
    """If `text` begins with an 'Ali, …' / 'hey Ali …' prefix, return the
    command body with the wake phrase stripped. Otherwise return None."""
    m = _DIRECT_ADDRESS_RE.match(text or "")
    if not m:
        return None
    body = (m.group(1) or "").strip()
    return body or None


def _is_bare_direct_address(text: str) -> bool:
    """True if the whole turn is just 'Ali' / 'Hey Ali' / 'OK Ali.' etc."""
    return bool(_BARE_DIRECT_ADDRESS_RE.match(text or ""))


def _log(tag: str, text: str) -> None:
    # All ambient events go through the unified agent log. Prefix with
    # `ambient:` so grep separates them from browser_task / opencli events.
    _agent_log(f"ambient:{tag}", text)


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
        # Debounce buffer for Deepgram finals. Accessed only on the
        # asyncio loop so no lock is needed.
        self._pending_parts: list[str] = []
        self._debounce_handle: asyncio.TimerHandle | None = None
        # Epoch time (monotonic) until which the NEXT turn should be
        # treated as direct-address because the user just said a bare
        # "Ali" to get the agent's attention.
        self._direct_armed_until: float = 0.0

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
            # Don't commit this final immediately. Glass-style debounce:
            # buffer it and start a timer. If another final arrives before
            # DEBOUNCE_SECONDS elapse, merge them. Only when N seconds of
            # silence pass do we commit the accumulated text as one turn.
            if self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._ingest_final, text)

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

    def _ingest_final(self, text: str) -> None:
        """Append a raw Deepgram final to the pending buffer + restart the
        debounce timer. Runs on the asyncio loop — no threading concerns.

        We INTENTIONALLY do NOT push intermediate previews to the overlay
        anymore — that caused the pill to stack a new line for every
        Deepgram partial ('are you using cats', 'are you using cats yes',
        'are you using cats yes and this'…). The overlay only sees the
        committed turn once silence elapses."""
        text = (text or "").strip()
        if not text:
            return
        self._pending_parts.append(text)
        # Restart the silence timer.
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
        if self._loop is not None:
            self._debounce_handle = self._loop.call_later(
                DEBOUNCE_SECONDS, self._commit_pending_turn
            )

    def _commit_pending_turn(self) -> None:
        """Silence-gap has elapsed — merge the buffered finals into one
        turn and (maybe) trigger analysis."""
        self._debounce_handle = None
        if not self._pending_parts:
            return
        turn = " ".join(self._pending_parts).strip()
        self._pending_parts = []
        # Now that the turn is committed, surface it once to the overlay.
        try:
            self._on_final(turn)
        except Exception:
            pass

        # Direct-address fast path: "Ali, do X" / "Hey Ali what is IRR"
        # fires a focused single-turn analysis right now, no 5-wait.
        # Also: bare "Ali" with nothing after primes the NEXT turn to
        # count as direct-address within DIRECT_ADDRESS_ARM_SECONDS.
        import time as _time
        direct_body: str | None = _strip_direct_address(turn)

        if direct_body is None and _is_bare_direct_address(turn):
            self._direct_armed_until = _time.monotonic() + DIRECT_ADDRESS_ARM_SECONDS
            with self._lock:
                self._history.append(turn)
            _log(
                "direct-arm",
                f"heard '{turn[:40]}' — next turn within {int(DIRECT_ADDRESS_ARM_SECONDS)}s treated as direct",
            )
            return

        if direct_body is None and _time.monotonic() < self._direct_armed_until:
            # Disarm and treat this turn as direct-address.
            self._direct_armed_until = 0.0
            direct_body = turn

        if direct_body:
            with self._lock:
                # Still add to history so the next passive analysis can
                # reference it, but reset the trigger counter — otherwise
                # every direct address would eat a slot toward the
                # next 5-turn window.
                self._history.append(turn)
                total = len(self._history)
                self._finals_since_trigger = 0
                in_flight = self._analysis_in_flight
            _log("direct", f'{total:>2}: "{direct_body[:120]}" ↳ direct-address fast path')
            if not in_flight:
                self._schedule_direct_analysis(direct_body)
            return

        with self._lock:
            self._history.append(turn)
            self._finals_since_trigger += 1
            count = self._finals_since_trigger
            total = len(self._history)
            should_trigger = (
                count >= AMBIENT_TRIGGER_EVERY_FINALS
                and not self._analysis_in_flight
            )
            if should_trigger:
                self._finals_since_trigger = 0
        marker = "↳ firing analysis" if should_trigger else (
            f"({count}/{AMBIENT_TRIGGER_EVERY_FINALS} until next analysis)"
        )
        _log("final", f'{total:>2}: "{turn[:120]}" {marker}')
        if should_trigger:
            self._schedule_analysis()

    def _schedule_direct_analysis(self, body: str) -> None:
        """Fire focused analysis on JUST the direct-address turn."""
        if self._analysis_in_flight:
            return
        self._analysis_in_flight = True
        if self._loop:
            self._loop.create_task(self._do_direct_analysis(body))

    async def _do_direct_analysis(self, body: str) -> None:
        """Same pipeline as _do_analysis but the 'history' is just the
        single direct-address utterance. The ambient_analysis prompt
        works fine on a one-line history — tier 1/2/3 routes identically."""
        from intent.ambient_analysis import analyse

        try:
            screen_app, screen_title, image = "", "", b""
            if self._screen_observer is not None:
                ctx = self._screen_observer.latest_context()
                screen_app = ctx.app
                screen_title = ctx.window_title
                image = ctx.image_bytes
            result = await analyse(
                [body],
                self._previous,
                screen_app=screen_app,
                screen_window_title=screen_title,
                screen_image_bytes=image,
            )
            if result.should_surface():
                self._previous = result
                _log("direct:surface", f"tier-{result.tier}: {result.headline[:100]}")
                if result.detail:
                    _log("direct:detail", result.detail[:240])
                if result.action:
                    _log("direct:action", str(result.action)[:240])
                self._on_suggestion(result)
            else:
                _log("direct:silent", f"tier-{result.tier} (no answer for {body[:80]!r})")
        except Exception as e:
            _log("direct:error", str(e)[:240])
        finally:
            self._analysis_in_flight = False

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
                _log("surface", f"tier-{result.tier}: {result.headline[:100]}")
                if result.detail:
                    _log("detail", result.detail[:240])
                if result.action:
                    _log("action", str(result.action)[:240])
                self._on_suggestion(result)
            else:
                # Tier-4: stay silent, but keep the prior non-silent result
                # so next round still knows what NOT to repeat.
                _log("silent", f"tier-{result.tier} (nothing worth surfacing)")
        except Exception as e:
            _log("error", str(e)[:240])
        finally:
            self._analysis_in_flight = False
