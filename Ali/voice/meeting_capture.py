"""
Meeting capture mode.

Streams mic audio via Deepgram for real-time word display, then
periodically sends the rolling transcript to Gemma 4 to extract
action items and executes them — multiple agents run in parallel.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from executors.browser.agent_client import LocalAgentClient
    from executors.meeting_tasks import TaskResult

# Max time between extraction passes. The loop also flushes eagerly when the
# latest final utterance ends in sentence-final punctuation — so the perceived
# latency after a natural pause is closer to 1s than to this ceiling.
EXTRACT_INTERVAL = 4.0

# Minimum unsent transcript length (chars) required for an eager flush.
# Prevents spamming Gemma on short disfluencies ("yeah.", "right.").
EAGER_FLUSH_MIN_CHARS = 30

# Characters that mark a "natural pause worth acting on" when they end a final.
_SENTENCE_FINAL = (".", "!", "?")

# Explicit stop commands — immediate
STOP_PHRASES = {"stop", "stop meeting", "end meeting", "stop capture", "ali stop"}

# Natural meeting-end phrases — Ali waits 8s of silence then wraps up
_MEETING_END_SIGNALS = (
    "meeting adjourned",
    "that's all for today",
    "we're done here",
    "let's wrap up",
    "talk to you later",
    "thanks everyone",
    "good meeting",
    "have a good",
    "bye everyone",
    "see you",
)

# Seconds of transcript silence after an end-signal before auto-stopping
_END_SIGNAL_GRACE_SEC = 8.0


class MeetingCapture:
    """
    Lifecycle: create → await run() to stream; call stop() to end.
    run() returns list[str] of result summaries for the end-of-meeting briefing.

    Callbacks (called from asyncio loop):
      on_interim(text)              — partial words, update overlay live
      on_final(text)                — committed utterance
      on_action_found(item)         — new action item dict from Gemma 4
      on_action_done(task, status)  — status update, status may be "done:$189 on Delta"
    """

    def __init__(
        self,
        on_interim: Callable[[str], None],
        on_final: Callable[[str], None],
        on_action_found: Callable[[dict[str, Any]], None],
        on_action_done: Callable[[str, str], None],
        browser_client: "LocalAgentClient",
        browser_lock: asyncio.Lock,
    ) -> None:
        self._on_interim      = on_interim
        self._on_final        = on_final
        self._on_action_found = on_action_found
        self._on_action_done  = on_action_done
        self._browser_client  = browser_client
        self._browser_lock    = browser_lock

        self._stop_event     = threading.Event()
        self._final_segments: list[str] = []
        self._unsent_finals:  list[str] = []
        self._captured_items: list[dict[str, Any]] = []
        self._results:        list[str] = []
        # Completed TaskResults with confirm_prompt set — read back at
        # meeting end for per-action yes/no confirmation.
        self._confirmables:   list["TaskResult"] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_speech_mono: float = time.monotonic()
        self._end_signal_seen: bool = False
        # Raised by _final() when the latest finalized utterance ends with
        # sentence-final punctuation — signals the main loop to flush eagerly.
        self._eager_flush_pending: bool = False

    # ── Public ────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def full_transcript(self) -> str:
        with self._lock:
            return " ".join(self._final_segments)

    @property
    def confirmables(self) -> list["TaskResult"]:
        """Completed actions that have a confirm_prompt — for end-of-meeting dialog."""
        with self._lock:
            return list(self._confirmables)

    async def run(self) -> list[str]:
        """
        Stream meeting audio until stop().
        Returns list of result summaries (one per completed action).
        """
        from voice.deepgram_stream import (
            stream_transcription_sync,
            start_meeting_audio,
            stop_meeting_audio,
        )
        from intent.meeting_intelligence import extract_action_items

        self._loop = asyncio.get_event_loop()

        start_meeting_audio()

        def _interim(text: str) -> None:
            if self._loop:
                self._loop.call_soon_threadsafe(self._on_interim, text)

        def _final(text: str) -> None:
            normalized = text.lower().strip().rstrip(".!?")
            # Explicit stop command
            if normalized in STOP_PHRASES:
                self._stop_event.set()
                return
            # Natural end signal — start grace-period countdown
            if any(sig in normalized for sig in _MEETING_END_SIGNALS):
                self._end_signal_seen = True
            self._last_speech_mono = time.monotonic()
            stripped = text.strip()
            ends_sentence = stripped.endswith(_SENTENCE_FINAL)
            with self._lock:
                self._final_segments.append(text)
                self._unsent_finals.append(text)
                if ends_sentence:
                    self._eager_flush_pending = True
            if self._loop:
                self._loop.call_soon_threadsafe(self._on_final, text)

        stream_thread = threading.Thread(
            target=stream_transcription_sync,
            args=(self._stop_event, _interim, _final),
            daemon=True,
        )
        stream_thread.start()

        last_extract = time.monotonic()

        try:
            while not self._stop_event.is_set():
                # Poll quickly so the eager-flush trigger fires within ~250ms
                # of a sentence-final utterance.
                await asyncio.sleep(0.25)

                # Auto-stop: end signal heard + silence for grace period
                if self._end_signal_seen:
                    silence = time.monotonic() - self._last_speech_mono
                    if silence >= _END_SIGNAL_GRACE_SEC:
                        print("[meeting] End-of-meeting signal detected — wrapping up")
                        self._stop_event.set()
                        break

                # Two flush triggers:
                #   1. Eager: the last final ended with .!? AND there's enough
                #      unsent text to be worth a Gemma call.
                #   2. Ceiling: EXTRACT_INTERVAL has elapsed regardless.
                with self._lock:
                    unsent_len = sum(len(s) for s in self._unsent_finals) + max(
                        0, len(self._unsent_finals) - 1
                    )
                    eager = self._eager_flush_pending and unsent_len >= EAGER_FLUSH_MIN_CHARS
                ceiling = time.monotonic() - last_extract >= EXTRACT_INTERVAL
                if not (eager or ceiling):
                    continue

                with self._lock:
                    segment = " ".join(self._unsent_finals).strip()
                    self._unsent_finals.clear()
                    self._eager_flush_pending = False

                last_extract = time.monotonic()

                if not segment:
                    continue

                print(f"[meeting] Analyzing: {segment[:80]}…")

                try:
                    new_items = await extract_action_items(segment, self._captured_items)
                except Exception as e:
                    print(f"[meeting] Extraction error: {e}")
                    continue

                for item in new_items:
                    self._captured_items.append(item)
                    self._on_action_found(item)
                    print(f"[meeting] Action found: {item.get('task')} [{item.get('type')}]")
                    # Each action item gets its own concurrent task — true parallel agents
                    asyncio.create_task(self._execute_item(item))

        finally:
            self._stop_event.set()
            stop_meeting_audio()
            stream_thread.join(timeout=3.0)

        return list(self._results)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_item(self, item: dict[str, Any]) -> None:
        from executors.meeting_tasks import search_flight, draft_email_in_gmail, TaskResult

        task      = item.get("task", "")
        item_type = item.get("type", "")
        slots     = item.get("slots", {})

        # Immediately show "Running" in the overlay
        self._on_action_done(task, "Running")

        result: TaskResult | None = None
        try:
            if item_type == "book_flight":
                dest   = slots.get("destination") or slots.get("city") or "Los Angeles"
                date   = slots.get("date") or "Tuesday"
                origin = slots.get("origin") or ""
                print(f"[meeting] Searching flight: {origin or 'SFO'} → {dest} on {date}")
                result = await search_flight(
                    self._browser_client, self._browser_lock,
                    dest, date, origin,
                )

            elif item_type == "draft_email":
                recipient  = slots.get("recipient") or ""
                subject    = slots.get("subject") or "Follow-up from our meeting"
                key_points = slots.get("key_points") or ""
                body = (
                    key_points
                    or f"Hi {recipient},\n\nFollowing up on our meeting discussion. "
                       "Please let me know your thoughts."
                )
                print(f"[meeting] Drafting email to: {recipient}")
                result = await draft_email_in_gmail(
                    self._browser_client, self._browser_lock,
                    recipient, subject, body,
                )

            else:
                # Generic fallback via orchestrator
                from intent.meeting_intelligence import item_to_intent
                from orchestrator.orchestrator import Orchestrator
                intent = item_to_intent(item)
                if intent.goal.value != "unknown":
                    orch = Orchestrator()
                    await orch.run(intent)
                result = TaskResult(True, "Done")

        except Exception as e:
            print(f"[meeting] Execution failed for '{task}': {e}")
            self._on_action_done(task, "error")
            return

        if result:
            if result.success:
                with self._lock:
                    self._results.append(f"{task}: {result.summary}")
                    if result.confirm_prompt:
                        self._confirmables.append(result)
                # Capture a Chrome screenshot for the overlay thumbnail.
                # Run in an executor so we don't block the event loop on the
                # screencapture subprocess.
                try:
                    from ui.screenshot_feed import capture_browser_thumb
                    loop = asyncio.get_event_loop()
                    path = await loop.run_in_executor(
                        None, capture_browser_thumb, task
                    )
                    if path:
                        if self._loop:
                            self._loop.call_soon_threadsafe(
                                self._on_action_done, task, f"thumb:{path}"
                            )
                except Exception as e:
                    print(f"[meeting] thumb capture failed: {e}")
            self._on_action_done(task, result.status_label())
        else:
            self._on_action_done(task, "done")
