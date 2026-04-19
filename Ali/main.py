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

import argparse
import asyncio
import faulthandler
import os
import signal
import sys
import threading
from typing import Callable, Protocol

# Dump Python + native stack on fatal signals. `enable()` already hooks SIGSEGV/SIGBUS/etc.
# on POSIX; do not call `register()` for the same signals — Python raises RuntimeError.
faulthandler.enable(file=sys.stderr, all_threads=True)

# Force line-buffered stdout/stderr so we never lose the last few prints before a crash.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

from config.index_bootstrap import ensure_index
from config.preflight import run_preflight_checks
from intent.schema import KnownGoal


class TranscriptionOverlay(Protocol):
    def push(self, state: str, text: str = "") -> None: ...


def _build_overlay() -> tuple["TranscriptionOverlay", Callable[[], None]]:
    backend = os.getenv("ALI_UI_BACKEND", "qt").strip().lower()
    if backend == "qt":
        try:
            from PySide6.QtWidgets import QApplication  # pyright: ignore[reportMissingImports]
            from ui.overlay import TranscriptionOverlay
        except ImportError as exc:
            print(
                f"[main][error] Qt overlay unavailable ({exc}). "
                "Install PySide6 with `pip install -r requirements.txt`, or "
                "set ALI_UI_BACKEND=web to use the browser overlay.",
                file=sys.stderr,
                flush=True,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"[main][error] Qt overlay import failed: {exc}. "
                "Falling back to ALI_UI_BACKEND=web.",
                file=sys.stderr,
                flush=True,
            )
            return _build_web_overlay()

        try:
            app = QApplication(sys.argv)
        except Exception as exc:
            print(
                f"[main][error] QApplication could not start: {exc}. "
                "Falling back to ALI_UI_BACKEND=web.",
                file=sys.stderr,
                flush=True,
            )
            return _build_web_overlay()

        try:
            overlay = TranscriptionOverlay(app)
        except Exception as exc:
            import traceback

            print(
                f"[main][error] TranscriptionOverlay init failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            raise

        # av + cv2 both bundle libavdevice with the same ObjC class names.
        # Python's normal teardown kills daemon threads mid-AVFoundation call,
        # causing a SIGTRAP. os._exit() skips teardown entirely.
        def _hard_exit(*_) -> None:
            os._exit(0)

        signal.signal(signal.SIGINT, _hard_exit)
        try:
            signal.signal(signal.SIGTERM, _hard_exit)
        except (ValueError, OSError):
            pass

        def _run_qt() -> None:
            try:
                app.exec()
            finally:
                os._exit(0)

        return overlay, _run_qt

    return _build_web_overlay()


def _build_web_overlay() -> tuple["TranscriptionOverlay", Callable[[], None]]:
    from ui.web_overlay import TranscriptionOverlay

    overlay = TranscriptionOverlay()
    return overlay, overlay.run_forever


# ── Asyncio agent loop (runs in background thread) ────────────────────────────

async def _agent_main(overlay: TranscriptionOverlay) -> None:
    # Import order on macOS: warm up STT first, then import capture; wake-word
    # starts in after_ptt_armed() only after the PTT bridge exists (see capture.py).
    from voice.transcribe import transcribe, warmup
    from intent.parser import parse_intent
    from orchestrator.orchestrator import Orchestrator
    from ui.menu_bar import MenuBar

    orchestrator = Orchestrator()
    menu_bar = MenuBar()
    agent_loop = asyncio.get_running_loop()
    command_lock = asyncio.Lock()

    warmup()   # pre-load Whisper so first transcription is instant

    from voice.capture import listen_for_command, request_ptt_session_from_wake
    from voice.wake_word import start_wake_word_listener

    # Restore persisted checklist state before any listener kicks in.
    # The click handler is registered unconditionally so the user can
    # tick leftover tasks even in a PTT-only (ambient-off) session.
    _hydrate_checklist_on_startup(overlay, agent_loop)

    # Ambient listen loop (glass-style) runs in parallel with PTT when the
    # flag is on. It does NOT intercept user commands — it only surfaces
    # suggestions (tier 1-3) into the overlay. PTT still works.
    from config.settings import AMBIENT_ENABLED
    if not AMBIENT_ENABLED:
        print(
            "[ambient] disabled — task checklist will stay empty. Set "
            "VOICE_AGENT_AMBIENT=1 in .env (or export it) to capture "
            "loose utterances.",
            flush=True,
        )

    async def _handle_transcript(transcript: str) -> None:
        async with command_lock:
            try:
                print("\n─── New command ───────────────────────────────")
                overlay.push("transcript", f'"{transcript}"')

                from intent.grad_calendar_hint import append_grad_calendar_note_if_needed
                from intent.pronoun_rewrite import rewrite_self_pronouns

                # Demo-mode rewrite: "me"/"my" → "hanzi" so self-directed
                # tasks route to Hanzi uniformly. Keep the raw transcript for
                # display/knowledge-question paths.
                routing_transcript = rewrite_self_pronouns(transcript)
                if routing_transcript != transcript:
                    print(f"[pronoun-rewrite] {transcript!r} → {routing_transcript!r}")

                # Multi-turn follow-up: if the previous turn asked for a
                # missing slot (e.g. "What should I say?", "Where to?"),
                # route this transcript to the stored resume callback
                # instead of re-classifying it as a new command.
                global _pending_followup
                if _pending_followup is not None:
                    import time as _t_mod
                    from voice.speak import speak as _speak_fu

                    pend = _pending_followup
                    if _t_mod.monotonic() > pend["deadline"]:
                        _pending_followup = None
                        overlay.push("hidden")
                    elif _match_any(transcript, _NO_TOKENS):
                        _pending_followup = None
                        on_cancel = pend.get("on_cancel")
                        if on_cancel is not None:
                            await on_cancel()
                        else:
                            overlay.push("done")
                            _speak_fu("Okay, cancelled.")
                        return
                    else:
                        _pending_followup = None
                        resume = pend["resume"]
                        await resume(transcript.strip().rstrip(".!?"))
                        return

                # 0 — Session-reset phrases end the active browser session,
                #     if any, and return. Otherwise fall through.
                if _is_session_reset(transcript):
                    from voice.speak import speak
                    if _browser_handle is not None:
                        print("[browser] ■ session cancelled by user")
                        await _reset_browser_session(orchestrator)
                    overlay.push("done")
                    speak("Okay, stopped.")
                    return

                # 1 — If a browser session is active, every utterance extends
                #     it — no classification. This is what gives chained
                #     commands the CLI feel ("open linkedin" → "now show my
                #     inbox" → "reply to hanzi").
                if _browser_handle is not None:
                    print(f"[browser] extending session {_browser_handle} — skipping classifier")
                    await _route_to_browser(routing_transcript, orchestrator, overlay, menu_bar)
                    return

                # 1.5 — OpenCLI fast path: deterministic adapters for known
                #       sites (no LLM in the loop). Feature-flag gated so we
                #       can A/B against browser_task live.
                from config.settings import ROUTE_OPENCLI_ENABLED, ROUTE_BROWSER_TASK_ENABLED
                if ROUTE_OPENCLI_ENABLED:
                    from executors.opencli_client import match_intent
                    hit = match_intent(transcript)
                    if hit is not None:
                        print(f"[opencli] matched intent: {hit[0].name}")
                        await _route_to_opencli(transcript, hit, overlay, menu_bar)
                        return

                # 2 — Parse intent
                menu_bar.set_status("parsing intent")
                print("[2/3] Parsing intent...")
                intent = await parse_intent(routing_transcript)
                print(f"      → goal={intent.goal.value}  slots={intent.slots}")
                # #region agent log
                try:
                    import json as _j, os as _o, time as _t
                    _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                    _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                    with open(_p, "a") as _f:
                        _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H4",
                            "location":"main:intent_parsed",
                            "message":"intent classification result",
                            "data":{"transcript": transcript, "goal": intent.goal.value,
                                    "has_slots": bool(intent.slots)},
                            "timestamp": int(_t.time()*1000)})+"\n")
                        _f.flush()
                except Exception:
                    pass
                # #endregion

                goal_label = intent.goal.value.replace("_", " ").title()
                if intent.goal.value not in ("unknown", "capture_meeting"):
                    overlay.push("intent", f"{goal_label}")

                # ── Meeting capture mode ──────────────────────────────────────
                if intent.goal.value == "capture_meeting":
                    from voice.speak import speak
                    speak("Got it, I'm capturing the meeting.")
                    agent_loop.create_task(_run_meeting_capture(overlay, menu_bar))
                    return

                if intent.goal.value == "ask_knowledge":
                    from executors.local.disk_index import answer_question
                    from voice.speak import speak
                    print("[2.5/3] Knowledge question → retrieving from disk index...")
                    result = await answer_question(transcript)
                    print(f'      ← "{result.text}" (backend={result.backend}, '
                          f'snippets={result.snippets_used})')
                    out = append_grad_calendar_note_if_needed(
                        transcript, result.text or "I don't have that."
                    )
                    overlay.push("assistant", out)
                    _push_citations(overlay, result.cited_paths)
                    speak(out)
                    return

                if intent.goal.value == "unknown":
                    from executors.local.disk_index import answer_question, index_exists
                    from intent.chat import chat_reply
                    from voice.speak import speak
                    print("[2.5/3] Unknown intent → conversational reply...")
                    reply = ""
                    if index_exists():
                        rag = await answer_question(transcript)
                        if rag.snippets_used > 0 and rag.text:
                            reply = append_grad_calendar_note_if_needed(transcript, rag.text)
                            overlay.push("assistant", reply)
                            _push_citations(overlay, rag.cited_paths)
                            speak(reply)
                            print(f'      ← "{reply}" (rag backend={rag.backend})')
                            return
                    reply = await chat_reply(transcript)
                    print(f'      ← "{reply}"')
                    # #region agent log
                    try:
                        import json as _j, os as _o, time as _t
                        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                        with open(_p, "a") as _f:
                            _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H4",
                                "location":"main:chat_reply",
                                "message":"conversational reply generated",
                                "data":{"reply": reply[:200], "len": len(reply)},
                                "timestamp": int(_t.time()*1000)})+"\n")
                            _f.flush()
                    except Exception:
                        pass
                    # #endregion
                    out = append_grad_calendar_note_if_needed(
                        transcript, reply or "I didn't catch that."
                    )
                    overlay.push("assistant", out)
                    speak(out)
                    return

                # 2.7 — Multi-action quick path. If a single utterance
                # packs multiple actions ("email Korin AND book a flight",
                # "text me X. Also email me X"), run the Gemma extractor
                # to split them and fire each one in parallel through the
                # native + shared browser clients. This runs BEFORE the
                # send_message branch so a bundled "text+email" request
                # isn't silently collapsed into just the iMessage half.
                # Skips for browser-shaped / find-file intents, which have
                # specific flows.
                if (
                    not _is_browser_intent(intent)
                    and intent.goal.value not in ("find_file",)
                    and _is_multi_action_candidate(routing_transcript)
                ):
                    handled = await _run_quick_multi_action(routing_transcript, overlay, menu_bar)
                    if handled:
                        return

                # send_message bypasses the browser sub-agent and the
                # vision-first orchestrator — it fires an iMessage via
                # AppleScript directly. Keep this ahead of the
                # browser-routing branch so "text Hanzi …" always hits
                # the local path when the utterance is single-action.
                if intent.goal.value == "send_message":
                    contact_raw = str(intent.slots.get("contact") or "").strip()
                    body = str(intent.slots.get("body") or "").strip()
                    file_query = str(intent.slots.get("file_query") or "").strip()
                    if contact_raw.lower() == "unknown":
                        contact_raw = ""
                    await _run_send_message_flow(
                        contact_raw,
                        body,
                        goal_label,
                        overlay,
                        menu_bar,
                        request_ptt_session_from_wake,
                        file_query=file_query,
                    )
                    return

                # 3 — Execute (known intent)
                print("[3/3] Executing...")
                menu_bar.set_status("running")
                overlay.push("action", f"Running: {goal_label}…")

                # Flights: call Kiwi MCP for real structured results. Pick the
                # cheapest, speak its summary, open Kiwi's booking deeplink.
                if intent.goal == KnownGoal.FIND_FLIGHTS:
                    from executors.flights import search_flights, format_flight_summary, FlightSearchError
                    from voice.speak import speak
                    try:
                        flights = await search_flights(intent.slots)
                    except FlightSearchError as e:
                        overlay.push("error", f"Flight search failed: {e}")
                        speak(str(e))
                        return
                    except Exception as e:
                        overlay.push("error", f"Flight search error: {e}")
                        speak("I couldn't reach the flight search service.")
                        return
                    if not flights:
                        overlay.push("done", "No flights found")
                        speak("I couldn't find any flights for that route.")
                        return
                    top = flights[0]
                    summary = format_flight_summary(top)
                    overlay.push("done", summary)
                    speak(f"Found a flight for {top.get('price')} dollars, {summary.split('•')[-1].strip()}.")
                    deeplink = top.get("deepLink")
                    if deeplink:
                        _open_url_local(deeplink)
                    return

                # Browser-shaped intents (open_url, apply_to_job, anything the
                # parser flagged requires_browser=True) all enter the same
                # persistent session. Intent metadata is used only to route;
                # the agent receives the raw transcript so it can interpret
                # "open my linking" as "linkedin" itself.
                if _is_browser_intent(intent):
                    if not ROUTE_BROWSER_TASK_ENABLED:
                        # browser_task path is disabled by flag — fall
                        # through to chat_reply instead of a silent no-op.
                        from intent.chat import chat_reply
                        from voice.speak import speak
                        print("[route] browser_task disabled by flag — using chat fallback")
                        reply = await chat_reply(transcript)
                        overlay.push("assistant", reply or "That would need a browser, but it's turned off.")
                        speak(reply or "Browser routing is turned off.")
                        return
                    # #region agent log
                    try:
                        import json as _j, os as _o, time as _t
                        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                        with open(_p, "a") as _f:
                            _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H11",
                                "location":"main:browser_session_dispatch",
                                "message":"routing utterance to persistent browser session",
                                "data":{"goal": intent.goal.value, "cold_start": _browser_handle is None},
                                "timestamp": int(_t.time()*1000)})+"\n")
                            _f.flush()
                    except Exception:
                        pass
                    # #endregion
                    await _route_to_browser(routing_transcript, orchestrator, overlay, menu_bar)
                    return

                revealed_name: str | None = None
                if intent.goal.value == "find_file":
                    from intent.file_resolve import enrich_intent_with_resolved_files
                    from voice.speak import speak
                    await enrich_intent_with_resolved_files(intent, transcript)
                    revealed_name = _revealed_basename(intent)
                    found_path = _revealed_path(intent)
                    if revealed_name is not None:
                        overlay.push("revealed", revealed_name)
                    if found_path is not None:
                        _reveal_in_finder(found_path)
                        print("      ✓ Done")
                        overlay.push("done")
                        return
                    # Do not fall back to vision planner for unresolved local file lookups.
                    # It causes unrelated screenshot/confirmation prompts.
                    msg = "I couldn't find that file yet. Try saying the exact filename, like 'open my resume pdf'."
                    print("      (find_file unresolved — skipping vision planner)")
                    # #region agent log
                    try:
                        import json as _j, os as _o, time as _t
                        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                        with open(_p, "a") as _f:
                            _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H8",
                                "location":"main:find_file_unresolved",
                                "message":"skipping orchestrator vision fallback for find_file",
                                "data":{"transcript": transcript, "slots": intent.slots},
                                "timestamp": int(_t.time()*1000)})+"\n")
                            _f.flush()
                    except Exception:
                        pass
                    # #endregion
                    overlay.push("error", msg)
                    speak(msg)
                    return

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

    def _on_ptt_armed() -> None:
        def _wake_heard(raw_text: str) -> None:
            tail = _extract_wake_tail(raw_text)
            if tail and _should_inline_wake_tail(tail):
                # Direct wake-command path: "Ali open my resume" skips second capture.
                # #region agent log
                try:
                    import json as _j, os as _o, time as _t
                    _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                    _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                    with open(_p, "a") as _f:
                        _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H6",
                            "location":"main:wake_inline_dispatch",
                            "message":"dispatching command directly from wake phrase",
                            "data":{"wake_text": raw_text, "tail": tail},
                            "timestamp": int(_t.time()*1000)})+"\n")
                        _f.flush()
                except Exception:
                    pass
                # #endregion
                asyncio.run_coroutine_threadsafe(_handle_transcript(tail), agent_loop)
                return
            if tail:
                # Tail is too short/ambiguous (e.g. "open my"), so switch to
                # conversational capture to collect the full command.
                # #region agent log
                try:
                    import json as _j, os as _o, time as _t
                    _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
                    _o.makedirs(_o.path.dirname(_p), exist_ok=True)
                    with open(_p, "a") as _f:
                        _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H9",
                            "location":"main:wake_tail_deferred",
                            "message":"wake tail too short; deferring to live capture",
                            "data":{"wake_text": raw_text, "tail": tail},
                            "timestamp": int(_t.time()*1000)})+"\n")
                        _f.flush()
                except Exception:
                    pass
                # #endregion

            sched = getattr(overlay, "schedule_wake_prompt", None)
            if sched is not None:
                sched(lambda: request_ptt_session_from_wake(overlay))  # type: ignore[misc]
            else:
                request_ptt_session_from_wake(overlay)

        global _dispatch_wake_heard
        _dispatch_wake_heard = _wake_heard
        start_wake_word_listener(_wake_heard)  # type: ignore[arg-type]
        if AMBIENT_ENABLED:
            print(
                "[ambient] enabled — loose utterances go to the task checklist; "
                "'Ali' / 'Hey Ali' also arms PTT via the ambient mic stream.",
                flush=True,
            )
            agent_loop.create_task(_run_ambient_capture(overlay))

    def _on_backtick() -> None:
        # Backtick stops meeting if one is running, otherwise starts wake
        if _active_meeting is not None:
            _active_meeting.stop()
            return
        sched = getattr(overlay, "schedule_wake_prompt", None)
        if sched is not None:
            sched(lambda: request_ptt_session_from_wake(overlay))  # type: ignore[misc]
        else:
            request_ptt_session_from_wake(overlay)

    menu_bar.set_status("ready")

    async for audio_bytes in listen_for_command(
        overlay=overlay,
        after_ptt_armed=_on_ptt_armed,
        on_backtick=_on_backtick,
    ):
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
        await _handle_transcript(transcript)


_active_meeting: "MeetingCapture | None" = None  # type: ignore[name-defined]

# Ambient listen (glass-style) — runs forever in background when
# VOICE_AGENT_AMBIENT=1. Does not own user commands; only surfaces
# suggestions via the overlay.
_ambient_capture: "AmbientCapture | None" = None  # type: ignore[name-defined]

# Set in _on_ptt_armed once the wake callback exists — ambient Deepgram
# finals call this for "Ali" / "Hey Ali" so wake works while the mic is
# owned by the ambient stream.
_dispatch_wake_heard: Callable[[str], None] | None = None


# Ambient tier-3 suggestions no longer run through an immediate yes/no
# confirmation pill — they're parked on the persistent task checklist
# (see observer/task_checklist.py) and the user ticks or says "run 1"
# to execute. _YES_TOKENS / _NO_TOKENS below are still used by the
# multi-turn send_message follow-up in the PTT path.

# Generic follow-up prompt framework. Any intent handler that needs a
# missing slot (recipient, body, destination, date …) can push a prompt
# via `_ask_followup`; the next transcript is routed to the stored
# `resume` coroutine instead of being re-classified as a new command.
# This keeps the "what should I say?" / "where to?" pattern consistent
# across send_message, send_email, book_flight, add_calendar_event, etc.
_pending_followup: dict | None = None
# {
#   "prompt": str,
#   "resume": Callable[[str], Awaitable[None]],   # called with the answer
#   "on_cancel": Callable[[], Awaitable[None]] | None,
#   "deadline": float,                             # monotonic seconds
#   "label": str,                                  # for logs
# }
_PENDING_FOLLOWUP_WINDOW_S = 30.0

_YES_TOKENS = {"yes", "yeah", "yep", "sure", "do it", "go ahead", "confirm", "ok", "okay", "please", "please do"}
_NO_TOKENS  = {"no", "nope", "skip", "not now", "cancel", "ignore", "never mind", "nevermind", "stop"}


def _ask_followup(
    prompt: str,
    resume,
    *,
    overlay,
    request_ptt,
    on_cancel=None,
    label: str = "followup",
) -> None:
    """Store a pending follow-up prompt and auto-start a PTT capture for
    the answer. When the next transcript arrives, `_handle_transcript`
    consumes the pending state and awaits `resume(transcript)`."""
    import time as _t_mod

    from voice.speak import speak

    global _pending_followup
    _pending_followup = {
        "prompt": prompt,
        "resume": resume,
        "on_cancel": on_cancel,
        "deadline": _t_mod.monotonic() + _PENDING_FOLLOWUP_WINDOW_S,
        "label": label,
    }
    overlay.push("assistant", prompt)
    speak(prompt)
    request_ptt(overlay)


async def _run_send_message_flow(
    contact: str,
    body: str,
    goal_label: str,
    overlay,
    menu_bar,
    request_ptt_session_from_wake,
    *,
    file_query: str = "",
) -> None:
    """Send an iMessage end-to-end. Prompts for missing contact/body and
    auto-relistens via PTT so the user can just speak the answer. If the
    body answer contains additional action verbs (e.g. "I'll be late,
    and also book me a flight"), we let the multi-action extractor pull
    out the extra items and run them alongside the iMessage."""
    from executors.local.applescript import (
        AppleScriptExecutionError,
        AppleScriptExecutor,
    )
    from executors.local.filesystem import resolve_file_query_to_path_async
    from voice.speak import speak

    contact = (contact or "").strip()
    body = (body or "").strip()
    file_query = (file_query or "").strip()

    if not contact:
        async def _resume_contact(answer: str) -> None:
            await _run_send_message_flow(
                answer, body, goal_label, overlay, menu_bar,
                request_ptt_session_from_wake,
                file_query=file_query,
            )

        _ask_followup(
            "Who should I message?",
            _resume_contact,
            overlay=overlay,
            request_ptt=request_ptt_session_from_wake,
            label="send_message:contact",
        )
        return

    if not body and not file_query:
        async def _resume_body(answer: str) -> None:
            # If the body reply bundles extra actions ("I'll be late,
            # and book me a flight to LA"), peel them off so we don't
            # silently drop them.
            extra_body = await _split_body_and_extra_actions(
                contact, answer, overlay, menu_bar,
            )
            await _run_send_message_flow(
                contact, extra_body, goal_label, overlay, menu_bar,
                request_ptt_session_from_wake,
                file_query=file_query,
            )

        _ask_followup(
            "What should I say?",
            _resume_body,
            overlay=overlay,
            request_ptt=request_ptt_session_from_wake,
            label="send_message:body",
        )
        return

    print("[3/3] Executing...")
    menu_bar.set_status("running")
    overlay.push("action", f"Running: {goal_label}…")

    attachments: list[str] = []
    if file_query:
        resolved = await resolve_file_query_to_path_async(file_query)
        if resolved:
            attachments.append(resolved)
            print(f"[send_message] resolved file_query {file_query!r} → {resolved}")
        else:
            print(f"[send_message] could not resolve file_query {file_query!r}")

    applescript = AppleScriptExecutor()
    address = contact
    if "@" not in contact and not contact.startswith("+"):
        try:
            address = applescript.resolve_contact(contact)
        except AppleScriptExecutionError as exc:
            overlay.push("error", str(exc))
            speak("I couldn't find that contact.")
            return

    try:
        applescript.send_imessage(
            contact=address, body=body, attachments=attachments or None,
        )
    except AppleScriptExecutionError as exc:
        overlay.push("error", str(exc))
        speak("I couldn't send that message.")
        return

    att_note = f" · {len(attachments)} attached" if attachments else ""
    overlay.push("assistant", f"✓ iMessage → {contact}: {body[:140]}{att_note}")
    speak("Sent.")


async def _split_body_and_extra_actions(
    contact: str, body_answer: str, overlay, menu_bar,
) -> str:
    """If `body_answer` looks like it contains multiple actions, run
    `extract_action_items` on it to find other tasks; fire those via
    the multi-action pipeline and return the cleaned message-body
    (i.e. the send_message item's slots.body or task, or the first
    clause if the extractor didn't find the message piece).

    Returns `body_answer` unchanged when nothing extra is detected."""
    if not _is_multi_action_candidate(body_answer):
        return body_answer

    from intent.meeting_intelligence import extract_action_items
    # Re-cast the answer so the extractor sees explicit "text <contact>"
    # framing — otherwise it may not realise the leading clause is a
    # message body.
    framed = f"text {contact}: {body_answer}"
    try:
        items = await extract_action_items(framed, [])
    except Exception as exc:
        print(f"[send_message] multi-action extraction failed: {exc}")
        return body_answer

    if not items:
        return body_answer

    message_body = body_answer
    other_items: list[dict] = []
    for it in items:
        itype = str(it.get("type", "")).lower()
        if itype in ("send_message", "send_imessage", "imessage"):
            # Prefer the extractor's body slot; fall back to the short task label.
            slots = it.get("slots") or {}
            candidate = str(slots.get("body") or slots.get("key_points") or it.get("task") or "").strip()
            if candidate:
                message_body = candidate
        else:
            other_items.append(it)

    if other_items:
        print(f"[send_message] body contained {len(other_items)} extra action(s) — dispatching")
        # Fire-and-forget the extras so we don't block the iMessage.
        import asyncio as _aio
        _aio.create_task(
            _dispatch_extracted_items(other_items, overlay, menu_bar, f"text {contact}: {body_answer}")
        )

    return message_body


async def _dispatch_extracted_items(
    items: list[dict], overlay, menu_bar, transcript: str,
) -> None:
    """Run a pre-extracted list of action items through the same
    machinery `_run_quick_multi_action` uses. Skips the extractor step
    since we already have the items."""
    from executors.meeting_tasks import search_flight, TaskResult
    from voice.speak import speak
    import asyncio as _aio

    browser_client = _get_meeting_browser_client()
    browser_lock = _aio.Lock()

    menu_bar.set_status("running")
    overlay.push("meeting_start")
    for item in items:
        overlay.push("meeting_action", item.get("task", ""))

    results_summary: list[str] = []

    async def _run_one(item: dict) -> None:
        task_label = item.get("task", "")
        item_type = str(item.get("type", "")).lower()
        slots = item.get("slots", {})
        overlay.push("meeting_action_update", f"{task_label}|Running")
        try:
            if item_type == "book_flight":
                dest = slots.get("destination") or "Los Angeles"
                date = slots.get("date") or "Tuesday"
                origin = slots.get("origin") or ""
                result = await search_flight(browser_client, browser_lock, dest, date, origin)
            elif item_type in ("draft_email", "send_email", "compose_email", "compose_mail"):
                result = await _dispatch_email_item(
                    slots, task_label, browser_client, browser_lock,
                )
            elif item_type in ("send_message", "send_imessage", "imessage"):
                result = await _dispatch_imessage_item(slots, task_label)
            else:
                result = TaskResult(False, "unsupported", detail=f"type={item_type}")
        except Exception as exc:
            print(f"[extra-action] {task_label} failed: {exc}")
            overlay.push("meeting_action_update", f"{task_label}|error")
            return
        overlay.push("meeting_action_update", f"{task_label}|{result.status_label()}")
        if result.success:
            results_summary.append(f"{task_label}: {result.summary}")

    await _aio.gather(*(_run_one(it) for it in items))
    overlay.push("meeting_stop")
    menu_bar.set_status("ready")

    if results_summary:
        speak("Also handled " + "; ".join(s.split(":")[0] for s in results_summary) + ".")


async def _dispatch_email_item(
    slots: dict,
    task_label: str,
    browser_client,
    browser_lock,
):
    """Send an extracted draft_email item via native Mail.app (attachments
    Just Work). Falls back to the Gmail browser executor only if the
    AppleScript path errors out."""
    from executors.local.applescript import AppleScriptExecutionError, AppleScriptExecutor
    from executors.local.filesystem import resolve_file_query_to_path_async
    from executors.meeting_tasks import TaskResult, draft_email_in_gmail

    recipient = str(slots.get("recipient") or slots.get("to") or "").strip()
    subject = str(slots.get("subject") or "Follow-up").strip()
    key_points = str(slots.get("key_points") or slots.get("body") or "").strip()
    body = key_points or f"Hi {recipient or 'there'},\n\nFollowing up per our conversation."

    attachments: list[str] = []
    file_query = str(slots.get("file_query") or "").strip()
    if file_query:
        resolved = await resolve_file_query_to_path_async(file_query)
        if resolved:
            attachments.append(resolved)

    applescript = AppleScriptExecutor()
    address = recipient
    if recipient and "@" not in recipient and not recipient.startswith("+"):
        try:
            address = applescript.resolve_contact(recipient) or recipient
        except AppleScriptExecutionError:
            address = recipient

    try:
        applescript.compose_mail(
            to=address,
            subject=subject,
            body=body,
            send=False,
            attachments=attachments or None,
        )
        att_note = f" · {len(attachments)} attached" if attachments else ""
        label = recipient or address or "the recipient"
        return TaskResult(
            success=True,
            summary=f"Mail draft ready for {label}{att_note}",
        )
    except AppleScriptExecutionError as exc:
        print(f"[multi-action] Mail.app draft failed ({exc}); falling back to Gmail browser")
        try:
            return await draft_email_in_gmail(
                browser_client, browser_lock, recipient, subject, body,
            )
        except Exception as exc2:
            return TaskResult(False, "draft failed", detail=str(exc2))


async def _dispatch_imessage_item(slots: dict, task_label: str):
    """Send an extracted send_message item via iMessage, resolving the
    contact name and any referenced file attachment."""
    from executors.local.applescript import AppleScriptExecutionError, AppleScriptExecutor
    from executors.local.filesystem import resolve_file_query_to_path_async
    from executors.meeting_tasks import TaskResult

    contact = str(slots.get("recipient") or slots.get("contact") or "").strip()
    body = str(slots.get("body") or slots.get("key_points") or "").strip()
    file_query = str(slots.get("file_query") or "").strip()

    if not contact:
        return TaskResult(False, "missing contact", detail="send_message had no recipient")

    applescript = AppleScriptExecutor()
    address = contact
    if "@" not in contact and not contact.startswith("+"):
        try:
            address = applescript.resolve_contact(contact)
        except AppleScriptExecutionError as exc:
            return TaskResult(False, "unknown contact", detail=str(exc))

    attachments: list[str] = []
    if file_query:
        resolved = await resolve_file_query_to_path_async(file_query)
        if resolved:
            attachments.append(resolved)

    if not body and not attachments:
        # If neither text nor attachment resolved, nothing to send.
        return TaskResult(False, "empty message", detail="no body or attachment")

    try:
        applescript.send_imessage(
            contact=address, body=body, attachments=attachments or None,
        )
    except AppleScriptExecutionError as exc:
        return TaskResult(False, "send failed", detail=str(exc))

    att_note = f" · {len(attachments)} attached" if attachments else ""
    return TaskResult(
        success=True,
        summary=f"iMessage sent to {contact}{att_note}",
    )


def _match_any(text: str, tokens: set[str]) -> bool:
    t = (text or "").strip().lower().rstrip(".!?")
    if not t:
        return False
    if t in tokens:
        return True
    return any(t.startswith(token + " ") or t.endswith(" " + token) or f" {token} " in f" {t} " for token in tokens)


def _ambient_deepgram_final_is_explicit_wake(text: str) -> bool:
    """True when the ambient line looks like a wake phrase, not a casual mention.

    Deepgram finals drive this path while the default mic is streaming to
    ambient — avoids relying on a second SpeechRecognition capture for short
    utterances like \"Ali\" alone."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower().rstrip(".!?")
    if low.startswith("hey ali"):
        return True
    if low.startswith("okay ali") or low.startswith("ok ali"):
        return True
    if low == "ali" or low.startswith("ali "):
        return True
    return False


async def _run_ambient_capture(overlay) -> None:
    global _ambient_capture
    from voice.ambient_capture import AmbientCapture
    from voice.speak import speak
    from observer.screen_loop import ScreenObserver
    from observer.agent_log import log as agent_log
    from config.settings import AMBIENT_SCREEN_ENABLED, AMBIENT_SPEAK_ENABLED

    # Voice is off by default. Flip VOICE_AGENT_AMBIENT_SPEAK=1 to re-enable
    # tier 1/2 spoken readback. (Meeting-detect gate removed — too finicky.)
    def _should_speak() -> bool:
        return AMBIENT_SPEAK_ENABLED

    agent_loop = asyncio.get_running_loop()

    # Click handler for checklist rows is already wired in
    # _hydrate_checklist_on_startup(); no re-registration needed here.

    def _on_interim(text: str) -> None:
        # Partial words only go to the log / console — never the overlay.
        pass

    def _on_final(text: str) -> None:
        # Checklist voice commands first ("run 1", "run all", …).
        handled = _handle_checklist_voice_command(text, overlay, agent_loop)
        if handled:
            agent_log("checklist:voice", f"handled {text!r}")
            return
        # Explicit wake on the ambient mic — same handler as Google wake STT.
        from voice.wake_word import try_acquire_wake_dispatch

        if _ambient_deepgram_final_is_explicit_wake(text) and try_acquire_wake_dispatch():
            fn = _dispatch_wake_heard
            if fn is not None:
                agent_log("ambient:wake", f"Deepgram wake → PTT: {text!r}")
                fn(text)

    def _on_suggestion(analysis) -> None:
        tier = analysis.tier
        headline = analysis.headline.strip()
        detail = (analysis.detail or headline).strip()

        # Tier 1/2 — info only. Show always; speak only if voice is
        # enabled AND we're not in a meeting.
        if tier in (1, 2):
            overlay.push("assistant", headline[:200])
            if detail and _should_speak():
                speak(detail[:200])
            return

        # Tier 3 — one or more concrete actions. Park each on the
        # persistent checklist; user ticks them later. Enrich slots now so
        # labels render with resolved values (real email, file path…).
        if tier == 3:
            actions = list(getattr(analysis, "actions", []) or [])
            if not actions:
                agent_log("checklist:skip", f"tier-3 with no actions: {headline[:80]}")
                return
            for action in actions:
                enriched = _enrich_action_dict(action, headline, detail)
                _add_checklist_task_for_action(overlay, enriched, headline, detail)
                agent_log(
                    "checklist:add",
                    f"{headline[:80]}  kind={enriched.get('kind')!r}  "
                    f"label={enriched.get('label', '')[:60]!r}",
                )
            return

    screen = None
    if AMBIENT_SCREEN_ENABLED:
        screen = ScreenObserver()
        screen.start()
        print("[ambient] screen observer started (event-driven snapshots)")

    # Show the overlay immediately so the user always has a visible
    # "Ali is on" indicator — not just when the first suggestion fires.
    overlay.push("assistant", "Ali listening · ambient on")
    agent_log("ambient:ready", "overlay pinned")

    capture = AmbientCapture(_on_interim, _on_final, _on_suggestion, screen_observer=screen)
    _ambient_capture = capture
    print("[ambient] starting glass-style listen loop (every 5 finals → analyse)")
    try:
        await capture.run()
    except Exception as e:
        print(f"[ambient] loop crashed: {e}")
    finally:
        if screen is not None:
            screen.stop()
        _ambient_capture = None


async def _execute_ambient_action(analysis, overlay) -> None:
    """Route a checklist-ticked tier-3 action through one of the three
    execution paths: opencli lookups, local AppleScript/Spotlight, or
    the persistent browser sub-agent. Called per Task (reconstructed
    AmbientAnalysis) so `analysis.actions` always has exactly one
    entry — hence the `.action` accessor is safe here."""
    from observer.agent_log import log as agent_log
    action = analysis.action or {}
    kind = action.get("kind", "").lower()
    text = action.get("text", "").strip()
    slots = action.get("slots") or {}
    headline = analysis.headline.strip()
    detail = (analysis.detail or "").strip()
    agent_log("ambient:exec", f"kind={kind} text={text[:80]} slots={list(slots)[:6]}")

    try:
        if kind == "opencli":
            await _execute_ambient_opencli(text, overlay, headline)
            return
        if kind == "local":
            await _execute_ambient_local(text, slots, headline, detail, overlay)
            return
        if kind == "browser_task":
            await _execute_ambient_browser_task(text, overlay, headline)
            return
        overlay.push("ambient_ack", f"✗ unsupported action kind {kind!r}")
        agent_log("ambient:exec", f"unsupported kind={kind}")
    except Exception as e:
        agent_log("ambient:exec", f"FAILED {headline[:80]}: {e}")
        overlay.push("ambient_ack", f"✗ {headline[:140]}: {e}")


async def _execute_ambient_browser_task(
    task_text: str, overlay, headline: str
) -> None:
    """Route a checklist browser_task through the persistent browser
    sub-agent. The agent's MCP server bundle must already be built
    (`cd executors/browser/agent/server && npm install && npm run build`).
    When missing, we surface a readable error on the overlay so the
    user sees why the task didn't run."""
    from observer.agent_log import log as agent_log
    from executors.browser.agent_client import LocalAgentClient

    goal = (task_text or headline or "").strip()
    if not goal:
        overlay.push("ambient_ack", "✗ browser_task: no goal text")
        agent_log("tool:browser_task", "empty goal — skipping")
        return

    overlay.push("action", f"▶  browser  ·  {goal[:160]}")
    agent_log("tool:browser_task", f"checklist goal={goal[:160]}")
    client = LocalAgentClient()
    try:
        status = await client.run_task(task=goal, session_id="")
    except Exception as exc:
        overlay.push("ambient_ack", f"✗ {str(exc)[:160]}")
        agent_log("tool:browser_task:err", str(exc)[:200])
        return

    if status.state == "complete":
        answer = (status.answer or "Done.").strip()[:400]
        overlay.push("assistant", answer)
        agent_log("tool:browser_task:done", answer[:200])
        return

    err = status.error or status.state or "browser task did not complete"
    overlay.push("ambient_ack", f"✗ {err[:160]}")
    agent_log("tool:browser_task:err", err[:200])


def _enrich_action_dict(
    action: dict, headline: str, detail: str
) -> dict:
    """Slot enrichment for one action: resolve contact names, find
    attachments, etc. Returns a new dict (doesn't mutate the input).
    Browser/opencli actions are passed through untouched — only
    kind='local' has slots worth enriching.

    Never raises — if any sub-step (contact resolution, file lookup)
    throws, we fall back to the raw slots so the task still lands on
    the checklist. Missing enrichment is always better than a dropped
    task the user won't discover."""
    out = dict(action or {})
    kind = str(out.get("kind", "")).lower()
    if kind != "local":
        return out
    text = str(out.get("text", "")).strip()
    slots = dict(out.get("slots") or {})
    try:
        out["slots"] = _enrich_local_slots(text, slots, headline, detail or "")
    except Exception as exc:
        print(f"[enrich] slot enrichment failed for text={text!r}: {exc}")
        out["slots"] = slots
    return out


def _enrich_analysis_for_preview(analysis):
    """Back-compat wrapper around the new per-action enrichment. Used
    by tests and by the checklist-execution path when it rebuilds an
    AmbientAnalysis from a stored Task."""
    from intent.ambient_analysis import AmbientAnalysis

    enriched_actions = [
        _enrich_action_dict(a, analysis.headline, analysis.detail or "")
        for a in (getattr(analysis, "actions", None) or [])
    ]
    return AmbientAnalysis(
        tier=analysis.tier,
        headline=analysis.headline,
        detail=analysis.detail,
        actions=enriched_actions,
        raw_json=analysis.raw_json,
    )


def _format_action_preview(analysis) -> str:
    """Build a short human-readable description of a tier-3 action. Used
    as the fallback detail on a checklist row. Under ~220 chars so it
    fits the overlay."""
    action = analysis.action or {}
    text = action.get("text", "").strip()
    slots = action.get("slots") or {}
    head = analysis.headline.strip()

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1].rstrip() + "…"

    if text in ("compose_mail", "send_email"):
        to = _clip(str(slots.get("to") or "(no recipient)"), 60)
        subj = _clip(str(slots.get("subject") or "(no subject)"), 50)
        body = _clip(str(slots.get("body") or ""), 80)
        atts = slots.get("attachments") or []
        tail = f"  [{len(atts)} attached]" if atts else ""
        preview = f"Email → {to}{tail}  ·  {subj}  ·  “{body}”"
    elif text in ("send_imessage", "send_message"):
        contact = _clip(str(slots.get("contact") or "(no contact)"), 50)
        body = _clip(str(slots.get("body") or ""), 100)
        preview = f"iMessage → {contact}  ·  “{body}”"
    elif text in ("create_calendar_event", "add_calendar_event"):
        title = _clip(str(slots.get("title") or head), 50)
        date = str(slots.get("date") or "").strip()
        time_ = str(slots.get("time") or "").strip()
        when = f"{date} {time_}".strip() or "(no time)"
        att = slots.get("attendees") or []
        who = f"  ·  with {', '.join(str(a) for a in att)[:60]}" if att else ""
        preview = f"Calendar → {title}  ·  {when}{who}"
    else:
        preview = head[:180]

    return preview


def _enrich_local_slots(
    text: str,
    slots: dict,
    headline: str,
    detail: str,
) -> dict:
    """Chain the tools we already have to fill missing slots before a
    local action fires. This is the 'agentic' bit — instead of opening
    Mail.app empty we first resolve 'Hanzi' → her email address via the
    Contacts AppleScript, and if the detail mentions a file (pitch deck,
    resume, cv) we find it via Spotlight and attach it.

    Mutates and returns the slots dict."""
    from executors.local.applescript import AppleScriptExecutor
    from executors.local.filesystem import FilesystemExecutor, resolve_file_query_to_path
    from observer.agent_log import log as agent_log
    import re

    applescript = AppleScriptExecutor()
    fs = FilesystemExecutor()

    def _append_attachment(path: str | None, source: str) -> None:
        if not path:
            return
        atts = slots.get("attachments") or []
        if not isinstance(atts, list):
            atts = []
        if path not in atts:
            atts.append(path)
        slots["attachments"] = atts
        agent_log("enrich:find_file", f"source={source} → {path}")

    def _resolve_hints_from_text(blob: str) -> str | None:
        _attachment_hints = (
            ("pitch deck", "deck"),
            ("slide deck", "deck"),
            ("deck", "deck"),
            ("resume", "resume"),
            ("cv", "resume"),
            ("cover letter", "cover_letter"),
        )
        for phrase, alias_key in _attachment_hints:
            if phrase not in blob:
                continue
            try:
                return fs.find_by_alias(alias_key)
            except (FileNotFoundError, Exception):
                return None
        return None

    _FILE_NOUN_WORDS = (
        "report", "deck", "slides", "presentation", "doc", "document",
        "pdf", "memo", "notes", "spreadsheet", "sheet", "proposal",
        "letter", "invoice", "contract", "resume", "cv", "agenda",
    )

    def _derive_label_file_query(headline: str, detail: str) -> str | None:
        """Last-ditch fallback: if the LLM forgot the file_query slot but
        the headline/detail names a document-ish noun ("Q1 Report",
        "pitch deck", "product memo"), try to extract that phrase so
        resolve_file_query_to_path has something to search on."""
        blob = f"{headline}. {detail}"
        low = blob.lower()
        if not any(w in low for w in _FILE_NOUN_WORDS):
            return None
        # Grab up to 3 words ending in one of the file-noun words.
        m = re.search(
            r"([A-Za-z0-9][A-Za-z0-9 _\-]{0,40}?\b(?:"
            + "|".join(_FILE_NOUN_WORDS)
            + r")s?)\b",
            blob,
            re.IGNORECASE,
        )
        if not m:
            return None
        phrase = re.sub(r"\s+", " ", m.group(1)).strip()
        return phrase or None

    def _resolve_attachments_chain() -> None:
        """Shared attachment-resolution chain:
          1. Explicit ``file_query`` slot from the extractor.
          2. Alias-phrase hint match in headline/detail ("deck", "resume"...).
          3. Document-noun phrase derived from the label ("Q1 Report").
        """
        file_query = str(slots.get("file_query", "")).strip()
        if file_query:
            _append_attachment(
                resolve_file_query_to_path(file_query),
                f"file_query={file_query!r}",
            )
        if not slots.get("attachments"):
            text_blob = f"{headline} {detail}".lower()
            _append_attachment(_resolve_hints_from_text(text_blob), "hint")
        if not slots.get("attachments"):
            derived = _derive_label_file_query(headline, detail)
            if derived:
                _append_attachment(
                    resolve_file_query_to_path(derived),
                    f"derived={derived!r}",
                )

    if text in ("compose_mail", "send_email"):
        to = str(slots.get("to", "")).strip()
        # If `to` is a name (no @), try Contacts resolution.
        if to and "@" not in to:
            resolved = applescript.resolve_contact(to)
            if resolved:
                agent_log("enrich:resolve_contact", f"{to!r} → {resolved}")
                slots["to"] = resolved
        _resolve_attachments_chain()

    if text in ("send_imessage", "send_message"):
        contact = str(slots.get("contact", "")).strip()
        if contact and "@" not in contact and not contact.startswith("+"):
            resolved = applescript.resolve_contact(contact)
            if resolved:
                agent_log("enrich:resolve_contact", f"{contact!r} → {resolved}")
                slots["contact"] = resolved
        _resolve_attachments_chain()

    return slots


async def _execute_ambient_local(
    text: str, slots: dict, headline: str, detail: str, overlay
) -> None:
    """Direct dispatch from ambient action_text → AppleScript / filesystem.
    Skips the vision planner + orchestrator.run, which would otherwise
    route most goals to the browser sub-agent."""
    from executors.local.applescript import AppleScriptExecutor
    from executors.local.filesystem import FilesystemExecutor
    from observer.agent_log import log as agent_log
    import subprocess

    # Chain our available tools to fill in missing context before firing.
    slots = _enrich_local_slots(text, dict(slots), headline, detail)
    applescript = AppleScriptExecutor()

    if text == "compose_mail" or text == "send_email":
        to = str(slots.get("to", "")).strip()
        subject = str(slots.get("subject", headline[:80])).strip()
        body = str(slots.get("body", detail or "")).strip()
        attachments = slots.get("attachments") or None
        applescript.compose_mail(
            to=to, subject=subject, body=body, send=False, attachments=attachments,
        )
        att_note = f" · attached {len(attachments)}" if attachments else ""
        overlay.push("ambient_ack", f"✓ Mail draft · {subject[:70] or 'new'}{att_note}")
        agent_log(
            "ambient:local:done",
            f"compose_mail to={to!r} subj={subject[:60]!r} atts={attachments!r}",
        )
        return

    if text == "send_imessage" or text == "send_message":
        contact = str(slots.get("contact", "")).strip()
        body = str(slots.get("body", detail or headline)).strip()
        attachments = slots.get("attachments") or None
        if not contact:
            overlay.push("ambient_ack", "✗ send_imessage: missing contact")
            return
        # _enrich_local_slots already resolved the name → address. Don't
        # re-run resolve_contact on an already-resolved address; it blocks
        # on Contacts.app lookup when the "name" is actually an email.
        applescript.send_imessage(contact=contact, body=body, attachments=attachments)
        att_note = f" · attached {len(attachments)}" if attachments else ""
        overlay.push("ambient_ack", f"✓ iMessage to {contact}{att_note}")
        agent_log("ambient:local:done", f"send_imessage to={contact!r} atts={attachments!r}")
        return

    if text == "create_calendar_event":
        title = str(slots.get("title", headline[:80])).strip()
        date = str(slots.get("date", "")).strip()
        time_ = str(slots.get("time", "")).strip()
        attendees = slots.get("attendees") or []
        if not isinstance(attendees, list):
            attendees = []
        if not date:
            overlay.push("ambient_ack", "✗ calendar: no date inferred")
            return
        applescript.create_calendar_event(
            title=title, date=date, time=time_, attendees=attendees
        )
        overlay.push("ambient_ack", f"✓ calendar: {title[:80]} · {date} {time_}")
        agent_log("ambient:local:done", f"calendar title={title!r} {date} {time_}")
        return

    if text == "find_file":
        query = str(slots.get("file_query", "")).strip()
        fs = FilesystemExecutor()
        path = fs.find_by_alias(query) if query else None
        if path:
            subprocess.run(["open", "-R", path], check=False)
            overlay.push("ambient_ack", f"✓ revealed: {path}")
            agent_log("ambient:local:done", f"find_file {query!r} → {path}")
        else:
            overlay.push("ambient_ack", f"✗ no file for {query!r}")
        return

    if text == "open_url":
        url = str(slots.get("url", "")).strip()
        if not url:
            overlay.push("ambient_ack", "✗ open_url: no URL")
            return
        # Force Chrome so the browser-agent extension loads the tab.
        # Plain `open` follows macOS default (Safari on dev machines).
        subprocess.run(["open", "-a", "Google Chrome", url], check=False)
        overlay.push("ambient_ack", f"✓ opened {url}")
        agent_log("ambient:local:done", f"open_url {url}")
        return

    overlay.push("ambient_ack", f"✗ unsupported local goal {text!r}")
    agent_log("ambient:exec", f"unsupported local goal {text!r}")


async def _execute_ambient_opencli(text: str, overlay, headline: str) -> None:
    """Run an opencli command suggested by the ambient analyzer. Unlike the
    PTT path we don't need a regex match — the text IS the command."""
    from executors.opencli_client import run_intent, summarize, OpenCliIntent
    from observer.agent_log import log as agent_log
    import re

    parts = text.strip().split()
    if not parts:
        overlay.push("ambient_ack", "✗ empty opencli command")
        return
    # Show the tool call before it runs so the overlay feels responsive.
    overlay.push("action", f"▶  opencli  ·  {' '.join(parts[:3])}")
    agent_log("tool:opencli", f"ambient cmd={parts}")
    adhoc = OpenCliIntent(
        name=f"ambient:{parts[0]}",
        match=re.compile(".*"),
        cmd=parts + ["--limit", "5"],
        description=f"ambient-suggested opencli: {text}",
        speak_template="{top3_titles}",
    )
    result = await run_intent(adhoc, groups=[])
    if result.ok:
        reply = summarize(result, adhoc, []) or "Done."
        overlay.push("assistant", reply[:400])
        agent_log("tool:opencli:done", reply[:200])
    else:
        err = result.error_message()
        overlay.push("ambient_ack", f"✗ {err[:160]}")
        agent_log("tool:opencli:err", err[:160])


# ── Task checklist ──────────────────────────────────────────────────────
# Ambient-mode suggestions no longer execute on the spot. They accumulate
# on a persistent checklist the user ticks (or speaks) to execute. The
# functions below bridge the checklist model (observer/task_checklist) to
# the overlay and the ambient executor.


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _label_from_action(action: dict, fallback: str) -> str:
    """Build a short, actionable label for one checklist row. Prefers
    the Gemini-supplied `label`, then composes from slots, then falls
    back to the headline."""
    if not isinstance(action, dict):
        return _clip(fallback, 80)
    supplied = str(action.get("label") or "").strip()
    if supplied:
        return _clip(supplied, 80)
    text = str(action.get("text", "")).strip()
    slots = action.get("slots") or {}
    kind = str(action.get("kind", "")).lower()

    if text in ("compose_mail", "send_email"):
        to = _clip(str(slots.get("to") or "someone"), 40)
        subj = _clip(str(slots.get("subject") or fallback), 40)
        return f"Email {to} · {subj}"
    if text in ("send_imessage", "send_message"):
        contact = _clip(str(slots.get("contact") or "someone"), 40)
        body = _clip(str(slots.get("body") or fallback), 40)
        return f"iMessage {contact} · {body}"
    if text in ("create_calendar_event", "add_calendar_event"):
        title = _clip(str(slots.get("title") or fallback), 40)
        when = str(slots.get("date") or "").strip()
        return f"Calendar · {title}" + (f" · {when}" if when else "")
    if text == "find_file":
        q = _clip(str(slots.get("file_query") or fallback), 60)
        return f"Find file · {q}"
    if text == "open_url":
        return f"Open URL · {_clip(str(slots.get('url') or fallback), 60)}"
    if kind == "browser_task":
        return _clip(text or fallback, 80)
    if kind == "opencli":
        return f"Run · {_clip(text, 60)}"
    return _clip(fallback, 80)


def _checklist_label_for(analysis, fallback: str) -> str:
    """Back-compat: label the analysis's first action. New callers
    should use `_label_from_action` directly so multi-action analyses
    get one label per row."""
    first = getattr(analysis, "action", None) if analysis is not None else None
    head = (analysis.headline.strip() if analysis else "") or fallback
    return _label_from_action(first or {}, head)


def _push_checklist_state(overlay) -> None:
    """Serialize the pending + recent-terminal rows and hand them to the
    overlay for repaint. Called whenever the checklist changes."""
    from observer.task_checklist import checklist
    import json as _json

    cl = checklist()
    # Show up to 6 pending first, then the most recent 2 terminal rows
    # so the user can see what just ran.
    rows = cl.all()
    pending = [t for t in rows if t.status == "pending" or t.status == "running"]
    terminal = [t for t in rows if t.status not in ("pending", "running")]
    terminal.sort(key=lambda t: t.updated_at, reverse=True)
    visible = pending + terminal[:2]

    payload = [
        {"id": t.id, "label": t.label, "status": t.status} for t in visible
    ]
    overlay.push("checklist_set", _json.dumps(payload, ensure_ascii=False))


def _add_checklist_task_for_action(
    overlay,
    action: dict,
    headline: str,
    detail: str,
) -> None:
    """Persist one action from a tier-3 suggestion as its own checklist
    row. A single ambient analysis can produce several rows — caller
    iterates over analysis.actions."""
    from observer.task_checklist import checklist

    fallback = headline or ""
    label = _label_from_action(action, fallback)
    row_detail = (detail or headline or "").strip()
    checklist().add(label=label, detail=row_detail, action=dict(action or {}))
    _push_checklist_state(overlay)


def _add_checklist_task(overlay, analysis, preview: str) -> None:
    """Back-compat wrapper — adds *every* action on the analysis as a
    separate row. Kept so tests & legacy callers still work."""
    actions = list(getattr(analysis, "actions", []) or [])
    if not actions:
        return
    headline = analysis.headline or preview
    detail = analysis.detail or analysis.headline or preview
    for action in actions:
        _add_checklist_task_for_action(overlay, action, headline, detail)


def _analysis_from_task(task) -> "AmbientAnalysis":  # type: ignore[name-defined]
    """Reconstruct an AmbientAnalysis from a stored task row so we can
    reuse the existing ambient executor without duplicating logic."""
    from intent.ambient_analysis import AmbientAnalysis

    return AmbientAnalysis(
        tier=3,
        headline=task.label,
        detail=task.detail,
        actions=[dict(task.action or {})],
        raw_json="",
    )


async def _execute_checklist_task(task_id: str, overlay) -> None:
    """Look up a checklist task, mark it running, and execute it through
    the ambient action dispatcher. Always emits a terminal status so the
    overlay shows a settled ✓ / ✗ state."""
    from observer.task_checklist import (
        checklist,
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_RUNNING,
    )
    from observer.agent_log import log as agent_log

    cl = checklist()
    task = cl.get(task_id)
    if task is None:
        return
    if task.status != "pending":
        agent_log("checklist:skip", f"task {task_id} status={task.status}")
        return

    cl.update_status(task_id, STATUS_RUNNING)
    _push_checklist_state(overlay)
    agent_log("checklist:run", f"{task.label[:100]}")

    try:
        await _execute_ambient_action(_analysis_from_task(task), overlay)
        cl.update_status(task_id, STATUS_DONE, result="executed")
        agent_log("checklist:done", f"{task.label[:100]}")
    except Exception as exc:
        cl.update_status(task_id, STATUS_FAILED, result=str(exc)[:200])
        agent_log("checklist:failed", f"{task.label[:80]}  err={exc}")
    finally:
        _push_checklist_state(overlay)


def _hydrate_checklist_on_startup(
    overlay, agent_loop: asyncio.AbstractEventLoop
) -> None:
    """On agent start-up, load persisted tasks from ``~/.ali/tasks.json``
    and paint them into the overlay. Also wire the click handler so the
    user can tick rows even when ambient mode is off. Safe to call
    repeatedly — it's idempotent."""
    from observer.task_checklist import checklist
    from observer.agent_log import log as agent_log

    setter = getattr(overlay, "set_checklist_click_handler", None)
    if setter is not None:
        setter(
            lambda task_id, kind: _dispatch_checklist_click(
                task_id, kind, overlay, agent_loop
            )
        )

    cl = checklist()
    # Any tasks persisted as `running` were interrupted by the previous
    # shutdown; reset them to pending so they're re-tickable.
    for t in cl.all():
        if t.status == "running":
            cl.update_status(t.id, "pending")

    pending = cl.pending()
    if pending:
        agent_log(
            "checklist:hydrate",
            f"{len(pending)} pending task(s) restored from disk",
        )
    _push_checklist_state(overlay)


def _dispatch_checklist_click(
    task_id: str,
    kind: str,
    overlay,
    agent_loop: asyncio.AbstractEventLoop,
) -> None:
    """Qt click callback. Runs on the UI thread → schedule work back onto
    the asyncio agent loop so we don't block Qt and so we stay consistent
    with the other ambient execution paths."""
    from observer.task_checklist import checklist, STATUS_SKIPPED
    from observer.agent_log import log as agent_log

    if kind == "run":
        asyncio.run_coroutine_threadsafe(
            _execute_checklist_task(task_id, overlay), agent_loop
        )
    elif kind == "skip":
        cl = checklist()
        task = cl.get(task_id)
        if task and task.status == "pending":
            cl.update_status(task_id, STATUS_SKIPPED)
            agent_log("checklist:skip", f"{task.label[:100]}")
            _push_checklist_state(overlay)


# Voice command grammar for the checklist. Kept small so the ambient
# stream doesn't accidentally fire tasks. Every phrase must start with a
# verb keyword AND include either an ordinal ("1"/"one"), "all", or "tasks".
_CHECKLIST_RUN_WORDS = ("run", "execute", "do", "tick", "check off", "go")
_CHECKLIST_SKIP_WORDS = ("skip", "dismiss", "cancel task", "remove task", "drop")
_CHECKLIST_CLEAR_PHRASES = (
    "clear tasks",
    "clear all tasks",
    "clear the list",
    "clear checklist",
    "empty the list",
    "empty checklist",
)

# Ordinal words are checked before cardinals so "do the second one" maps
# to 2 (not 1 via the trailing "one"). Numeric digits still win over both.
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_CARDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_ordinal(text: str) -> int | None:
    """Return a 1-based index if the utterance names one ("run 1",
    "execute task three", "do the second one"), else None."""
    import re

    t = (text or "").lower()
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return None
        if 1 <= n <= 20:
            return n
    for word, num in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            return num
    for word, num in _CARDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            return num
    return None


def _handle_checklist_voice_command(
    text: str,
    overlay,
    agent_loop: asyncio.AbstractEventLoop,
) -> bool:
    """Try to interpret ``text`` as a checklist voice command. Returns
    True if it was handled (so the caller skips other processing)."""
    from observer.task_checklist import checklist, STATUS_SKIPPED
    from observer.agent_log import log as agent_log

    t = (text or "").strip().lower().rstrip(".!?")
    if not t:
        return False

    # Clear-all phrases — match first so "clear all tasks" doesn't get
    # mis-parsed as a run verb.
    for phrase in _CHECKLIST_CLEAR_PHRASES:
        if phrase in t:
            removed = checklist().clear(include_terminal=True)
            agent_log("checklist:clear", f"voice removed {removed}")
            _push_checklist_state(overlay)
            return True

    has_run_verb = any(
        t.startswith(v + " ") or t == v or f" {v} " in f" {t} "
        for v in _CHECKLIST_RUN_WORDS
    )
    has_skip_verb = any(
        t.startswith(v + " ") or t == v or f" {v} " in f" {t} "
        for v in _CHECKLIST_SKIP_WORDS
    )
    if not (has_run_verb or has_skip_verb):
        return False

    cl = checklist()

    # "run all" → fire every pending task
    if has_run_verb and ("all" in t.split() or "everything" in t.split()):
        pending = cl.pending()
        if not pending:
            return True
        agent_log("checklist:run-all", f"n={len(pending)}")
        for task in pending:
            asyncio.run_coroutine_threadsafe(
                _execute_checklist_task(task.id, overlay), agent_loop
            )
        return True

    idx = _extract_ordinal(t)
    if idx is None:
        # Verb but no index — ambiguous; ignore to avoid misfires.
        return False

    task = cl.find_by_index(idx)
    if task is None:
        agent_log("checklist:voice", f"no task at index {idx}")
        return True  # Still handled — just no-op.

    if has_skip_verb:
        cl.update_status(task.id, STATUS_SKIPPED)
        agent_log("checklist:skip", f"voice idx={idx} {task.label[:80]}")
        _push_checklist_state(overlay)
        return True

    asyncio.run_coroutine_threadsafe(
        _execute_checklist_task(task.id, overlay), agent_loop
    )
    return True


# Persistent browser sub-agent session. One session spans many voice
# utterances so follow-ups ("now open my inbox", "reply to this person")
# reuse the existing agent loop + tab state, mirroring llm-in-chrome's CLI.
# The MCP server generates its own session id on browser_start — we capture
# it from the returned status and reuse it for subsequent browser_message
# calls until the session ends or the user says a reset phrase.
_browser_handle: str | None = None

_SESSION_RESET_PHRASES = {
    "stop", "stop it", "cancel", "cancel that",
    "never mind", "nevermind",
    "new task", "new command", "new session",
    "start over", "restart",
    "done", "that's all", "thats all", "that is all",
}


def _is_session_reset(transcript: str) -> bool:
    t = (transcript or "").strip().lower().rstrip(".!?")
    return t in _SESSION_RESET_PHRASES


def _is_browser_intent(intent) -> bool:
    """Cold-start heuristic: does this utterance belong to the browser session?

    Email/message intents route here too — we want *progressive drafting*:
    first utterance opens Gmail/Messages with the recipient; follow-up
    utterances are appended to the draft via the persistent session. Avoid
    the vision-first orchestrator for these, which tries to compose from
    screenshots of the desktop.
    """
    from intent.schema import KnownGoal
    if getattr(intent, "requires_browser", False):
        return True
    return intent.goal in {
        KnownGoal.OPEN_URL,
        KnownGoal.APPLY_TO_JOB,
        KnownGoal.SEND_EMAIL,
        KnownGoal.SEND_MESSAGE,
        KnownGoal.ADD_CALENDAR_EVENT,
    }


async def _route_to_browser(transcript: str, orchestrator, overlay, menu_bar) -> None:
    """Start or continue the persistent browser session with the given utterance.

    The MCP server's browser_start/browser_message both block until the
    session reaches a terminal state (complete, awaiting_confirmation,
    error, cancelled, timeout). So each call is one round-trip; we don't
    poll. After `complete`, the session stays alive so the next voice
    command can extend it via browser_message.
    """
    global _browser_handle
    from voice.speak import speak
    from ui.confirmation import ask_confirmation

    client = orchestrator._browser_agent
    if menu_bar is not None:
        menu_bar.set_status("running")
    # Show the actual task so the user can see what the agent will do.
    overlay.push("action", f"▶  browser: {transcript[:180]}")
    from observer.agent_log import log as agent_log
    agent_log("tool:browser_task", transcript[:160])

    async def _turn(text: str):
        """One round-trip: start a new session or push a message to the
        existing one. Returns the terminal TaskStatus for this turn."""
        global _browser_handle
        if _browser_handle is None:
            print(f"[browser] ▶ start: {text[:120]}")
            status = await client.run_task(task=text, session_id="")
            _browser_handle = status.id
            return status
        print(f"[browser] ↪ continue {_browser_handle}: {text[:120]}")
        status = await client.send_message(_browser_handle, text)
        # Session disappeared between turns (server restart, prior cancel).
        # Start fresh with the same utterance so the user doesn't notice.
        if status.state == "error" and "not found" in (status.error or "").lower():
            print(f"[browser] session {_browser_handle} missing — restarting")
            _browser_handle = None
            status = await client.run_task(task=text, session_id="")
            _browser_handle = status.id
        return status

    try:
        status = await _turn(transcript)
        while status.state == "awaiting_confirmation":
            summary = status.confirmation.summary if status.confirmation else "Proceed?"
            payload = status.confirmation.payload if status.confirmation else {}
            detail = "\n".join(f"  {k}: {v}" for k, v in (payload or {}).items())
            approved = await ask_confirmation(f"{summary}\n{detail}\n\nProceed?")
            status = await _turn("yes, proceed" if approved else "no, cancel")
            if not approved:
                overlay.push("done")
                return

        if status.state == "complete":
            answer = (status.answer or "").strip() or "Done."
            print(f"[browser] ✓ session={_browser_handle} answer={answer[:200]}")
            overlay.push("assistant", answer[:400])
            speak(answer[:400])
            # Keep _browser_handle set; next utterance extends the session.
            return

        # error / cancelled / timeout — session is terminal.
        print(f"[browser] session {_browser_handle} ended: {status.state} {status.error!r}")
        _browser_handle = None
        overlay.push("error", status.error or f"Browser {status.state}")
        speak("Something went wrong.")
    except Exception:
        _browser_handle = None
        raise


async def _reset_browser_session(orchestrator) -> None:
    global _browser_handle
    if _browser_handle is None:
        return
    try:
        await orchestrator._browser_agent.cancel(_browser_handle)
    except Exception:
        pass
    _browser_handle = None


async def _route_to_opencli(transcript: str, hit, overlay, menu_bar) -> None:
    """Dispatch a voice utterance through opencli (deterministic adapter).

    `hit` is (OpenCliIntent, capture_groups). On success, summarize the rows
    via the intent's speak_template and read them back; on failure, surface
    a short error and DO NOT fall back to browser_task silently — the user
    can say the command again or use a more general phrasing.
    """
    from voice.speak import speak
    from executors.opencli_client import run_intent, summarize

    intent, groups = hit
    menu_bar.set_status("running")
    cmd_preview = " ".join(intent.cmd[:3])
    overlay.push("action", f"▶  opencli  ·  {cmd_preview}")
    from observer.agent_log import log as agent_log
    agent_log("tool:opencli", f"{intent.name} cmd={intent.cmd}")

    try:
        result = await run_intent(intent, groups)
    except FileNotFoundError as e:
        msg = f"opencli binary not found: {e}"
        print(f"[opencli] {msg}")
        overlay.push("error", msg)
        speak("OpenCLI isn't installed.")
        return

    if result.ok:
        reply = summarize(result, intent, groups).strip() or "Done."
        print(f"[opencli] ✓ {intent.name} → {reply[:200]}")
        overlay.push("assistant", reply[:400])
        speak(reply[:400])
        return

    err = result.error_message()
    print(f"[opencli] ✗ {intent.name}: {err}")
    overlay.push("error", err[:200])
    speak("That opencli command failed.")


async def _run_meeting_capture(overlay, menu_bar) -> None:
    global _active_meeting
    from voice.meeting_capture import MeetingCapture

    menu_bar.set_status("meeting")
    overlay.push("meeting_start")

    def _on_interim(text: str) -> None:
        overlay.push("meeting_interim", text)

    def _on_final(text: str) -> None:
        overlay.push("meeting_final", text)

    def _on_action_found(item: dict) -> None:
        task = item.get("task", "")
        print(f"[meeting] Action: {task}")
        overlay.push("meeting_action", task)

    def _on_action_done(task: str, status: str) -> None:
        overlay.push("meeting_action_update", f"{task}|{status}")

    # Share a single persistent browser client with every action item. The
    # lock serializes access so concurrent items queue rather than fight
    # over the single Chrome session.
    # Assumption: voice-mode's _route_to_browser is not invoked while a
    # meeting is active — the backtick/wake paths route to capture.stop()
    # during meetings, so no contention on this client.
    browser_client = _get_meeting_browser_client()
    browser_lock = asyncio.Lock()

    capture = MeetingCapture(
        _on_interim, _on_final, _on_action_found, _on_action_done,
        browser_client=browser_client,
        browser_lock=browser_lock,
    )
    _active_meeting = capture
    try:
        results = await capture.run()
    finally:
        _active_meeting = None
        overlay.push("meeting_stop")
        menu_bar.set_status("ready")
        print("[meeting] Session ended")

    # End-of-meeting spoken briefing
    from voice.speak import speak
    if results:
        briefing = await _meeting_briefing(results, capture.full_transcript)
    else:
        briefing = "Meeting captured. No action items were detected."
    overlay.push("assistant", briefing)
    speak(briefing)

    # Per-action confirmation dialog. Only fires for actions whose executor
    # set a confirm_prompt (e.g. drafted emails). Each confirmation runs as
    # a separate browser task through the same shared client.
    await _run_confirmation_dialog(
        capture.confirmables, overlay, browser_client, browser_lock
    )


_ACTION_VERBS = (
    "email", "send", "message", "text", "mail",
    "book", "find", "search", "draft", "schedule",
    "call", "reply", "open", "order", "buy",
)


_MULTI_ACTION_JOINERS = (
    " and ",
    ",",
    ". ",
    "; ",
    "? ",
    "! ",
    " also ",
    " then ",
    " plus ",
)


def _is_multi_action_candidate(transcript: str) -> bool:
    """Cheap heuristic: ≥2 action verbs + conjunction/sentence-break suggests
    a multi-item utterance (e.g. "text me X. Also email me X")."""
    t = (transcript or "").lower()
    if not any(joiner in t for joiner in _MULTI_ACTION_JOINERS):
        return False
    verbs_seen = sum(1 for v in _ACTION_VERBS if v in t)
    return verbs_seen >= 2


async def _run_quick_multi_action(transcript: str, overlay, menu_bar) -> bool:
    """
    Extract action items from a single utterance and run them in parallel
    through the shared browser client (same pipeline as meeting mode).

    Returns True if we handled the utterance (≥1 extracted item); False
    lets the caller fall back to single-intent execution.
    """
    from intent.meeting_intelligence import extract_action_items
    from executors.meeting_tasks import search_flight, TaskResult
    from voice.speak import speak

    print(f"[multi-action] extracting tasks from: {transcript[:80]}")
    items = await extract_action_items(transcript, [])
    if not items:
        print("[multi-action] no items extracted — falling back to single-intent")
        return False
    if len(items) < 2:
        print(f"[multi-action] only 1 item — falling back to single-intent")
        return False

    print(f"[multi-action] {len(items)} items — running through shared client")

    browser_client = _get_meeting_browser_client()
    browser_lock = asyncio.Lock()

    # Use the meeting overlay to show live chips.
    menu_bar.set_status("running")
    overlay.push("meeting_start")

    for item in items:
        overlay.push("meeting_action", item.get("task", ""))

    results_summary: list[str] = []
    confirmables: list[TaskResult] = []

    async def _run_one(item: dict) -> None:
        task_label = item.get("task", "")
        item_type = str(item.get("type", "")).lower()
        slots = item.get("slots", {})
        overlay.push("meeting_action_update", f"{task_label}|Running")

        result: TaskResult | None = None
        try:
            if item_type == "book_flight":
                dest = slots.get("destination") or "Los Angeles"
                date = slots.get("date") or "Tuesday"
                origin = slots.get("origin") or ""
                result = await search_flight(
                    browser_client, browser_lock, dest, date, origin,
                )
            elif item_type in ("draft_email", "send_email", "compose_email", "compose_mail"):
                result = await _dispatch_email_item(
                    slots, task_label, browser_client, browser_lock,
                )
            elif item_type in ("send_message", "send_imessage", "imessage"):
                result = await _dispatch_imessage_item(slots, task_label)
            else:
                result = TaskResult(False, "unsupported", detail=f"type={item_type}")
        except Exception as e:
            print(f"[multi-action] {task_label} failed: {e}")
            overlay.push("meeting_action_update", f"{task_label}|error")
            return

        overlay.push("meeting_action_update", f"{task_label}|{result.status_label()}")
        if result.success:
            results_summary.append(f"{task_label}: {result.summary}")
            if result.confirm_prompt:
                confirmables.append(result)
            # Capture a Chrome thumbnail for the overlay. Non-fatal on error.
            try:
                from ui.screenshot_feed import capture_browser_thumb
                loop = asyncio.get_event_loop()
                path = await loop.run_in_executor(
                    None, capture_browser_thumb, task_label
                )
                if path:
                    overlay.push("meeting_action_update", f"{task_label}|thumb:{path}")
            except Exception as e:
                print(f"[multi-action] thumb capture failed: {e}")

    # Run items concurrently. The shared lock inside meeting_tasks serializes
    # the actual browser calls, so this is effectively a queue — but the
    # overlay still sees all chips flip to Running at once and they complete
    # in finish-order, which reads as parallel to the user.
    await asyncio.gather(*(_run_one(it) for it in items))

    overlay.push("meeting_stop")
    menu_bar.set_status("ready")

    # Briefing
    if results_summary:
        briefing = await _meeting_briefing(results_summary, transcript)
    else:
        briefing = "All done."
    overlay.push("assistant", briefing)
    speak(briefing)

    # Confirmation dialog — same one meeting mode uses.
    await _run_confirmation_dialog(confirmables, overlay, browser_client, browser_lock)
    return True


# Lazily-obtained reference to the orchestrator's persistent browser client.
# Kept as a module global so every meeting session reuses the same Node
# subprocess instead of spawning one per action.
_meeting_browser_client = None


def _get_meeting_browser_client():
    """Return a LocalAgentClient, constructing one on first use."""
    global _meeting_browser_client
    if _meeting_browser_client is None:
        from executors.browser.agent_client import LocalAgentClient
        _meeting_browser_client = LocalAgentClient()
    return _meeting_browser_client


async def _run_confirmation_dialog(
    confirmables: list,
    overlay,
    browser_client,
    browser_lock: asyncio.Lock,
) -> None:
    """
    After the briefing, walk each confirmable action:
      1. TTS the confirm_prompt
      2. Listen briefly (~5s) for yes/no via Deepgram
      3. On yes → run confirm_task through the shared browser client
    """
    if not confirmables:
        return

    from voice.speak import speak
    from voice.listen_brief import listen_brief

    for item in confirmables:
        prompt = item.confirm_prompt
        if not prompt:
            continue

        print(f"[confirm] {prompt}")
        overlay.push("assistant", prompt)
        speak(prompt)
        # Small gap so TTS finishes before Deepgram starts capturing.
        await asyncio.sleep(0.4)

        try:
            reply = await listen_brief(timeout=6.0)
        except Exception as e:
            print(f"[confirm] listen failed: {e}")
            reply = ""

        decision = _interpret_yes_no(reply)
        print(f'[confirm] heard "{reply}" → {decision}')

        if decision is True and item.confirm_task:
            import uuid as _uuid
            session_id = f"confirm-{_uuid.uuid4().hex[:8]}"
            overlay.push("action", f"Sending…")
            try:
                async with browser_lock:
                    handle = await browser_client.run_task(
                        item.confirm_task, session_id
                    )
                    status = await browser_client.poll_until_paused_or_terminal(
                        handle.id, max_wait=60.0
                    )
                if status.state == "complete":
                    msg = (status.answer or "Sent.").strip()[:200]
                    overlay.push("assistant", msg)
                    speak(msg)
                else:
                    err = status.error or status.state
                    print(f"[confirm] send failed: {err}")
                    overlay.push("error", f"Send failed: {err[:80]}")
                    speak("I couldn't send it.")
            except Exception as e:
                print(f"[confirm] send exception: {e}")
                overlay.push("error", f"Send error: {e}")
                speak("I couldn't send it.")
        elif decision is False:
            overlay.push("assistant", "Okay, leaving it as a draft.")
            speak("Okay, leaving it as a draft.")
        else:
            overlay.push("assistant", "I didn't catch that — leaving it as a draft.")
            speak("I didn't catch that, so I'll leave it as a draft.")


_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "send", "send it",
              "do it", "please", "okay", "ok", "go ahead", "confirm"}
_NO_WORDS  = {"no", "nope", "nah", "cancel", "don't", "dont",
              "leave it", "not now", "keep it", "not yet"}


def _interpret_yes_no(text: str) -> bool | None:
    """Return True=yes, False=no, None=unclear."""
    t = (text or "").strip().lower().rstrip(".!?")
    if not t:
        return None
    # Prefer longer matches first.
    for phrase in sorted(_NO_WORDS, key=len, reverse=True):
        if phrase in t:
            return False
    for phrase in sorted(_YES_WORDS, key=len, reverse=True):
        if phrase in t:
            return True
    return None


async def _meeting_briefing(results: list[str], transcript: str) -> str:
    """Use Gemini to generate a natural spoken end-of-meeting summary."""
    from config.settings import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return "Meeting done. " + ". ".join(results[:3]) + "."

    items_text = "\n".join(f"- {r}" for r in results)
    prompt = (
        "You are Ali, an AI chief of staff. A meeting just ended. "
        "Summarize what was accomplished in 2-3 natural spoken sentences — "
        "no markdown, no lists, no 'Certainly!'. Be direct. Mention specific results.\n\n"
        f"Actions completed:\n{items_text}\n\n"
        "Spoken summary:"
    )
    import asyncio as _aio
    loop = _aio.get_event_loop()
    try:
        from google import genai as _genai  # type: ignore
        def _call() -> str:
            client = _genai.Client(api_key=GEMINI_API_KEY)
            r = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_genai.types.GenerateContentConfig(temperature=0.3, max_output_tokens=120),
            )
            return (r.text or "").strip()
        return await loop.run_in_executor(None, _call)
    except Exception:
        return "Meeting done. " + ". ".join(results[:3]) + "."


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


def _revealed_path(intent) -> str | None:
    if intent.goal.value != "find_file":
        return None
    resolved = intent.slots.get("resolved_local_files") if isinstance(intent.slots, dict) else None
    if not isinstance(resolved, dict):
        return None
    path = resolved.get("found")
    if not isinstance(path, str) or not path:
        return None
    return path


def _intent_url(intent) -> str | None:
    target = getattr(intent, "target", None)
    if isinstance(target, dict):
        value = target.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    slots = getattr(intent, "slots", None)
    if isinstance(slots, dict):
        url = slots.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


_SOURCE_LABELS = {
    "contacts": "Contact",
    "calendar": "Event",
    "messages": "Chat",
}

_MAX_VISIBLE_CITATIONS = 4


def _pretty_citation(path: str) -> str:
    """Turn a stored `files.path` into a human-readable citation label.

    Filesystem paths become the basename; synthetic `ali://` paths become
    e.g. ``Contact`` / ``Event`` / ``Chat``.
    """
    if not isinstance(path, str) or not path:
        return str(path)
    if path.startswith("ali://"):
        try:
            rest = path[len("ali://") :]
            source, _id = rest.split("/", 1)
        except ValueError:
            return path
        return _SOURCE_LABELS.get(source, source.title())
    return os.path.basename(path)


def _push_citations(overlay, paths: list[str]) -> None:
    """Send a structured, clickable citation list to the overlay.

    Payload shape (JSON-encoded in the overlay's string-only channel):
        [{"label": "resume.pdf", "path": "/Users/…/resume.pdf"}, …]

    The overlay paints each citation as a clickable link chip; clicking
    opens the underlying file via `open <path>` (or the matching macOS app
    for synthetic `ali://…` paths).
    """
    if not paths:
        return
    import json as _json

    items: list[dict] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        items.append({"label": _pretty_citation(raw), "path": raw})
        if len(items) >= _MAX_VISIBLE_CITATIONS:
            break
    if not items:
        return
    overlay.push("cited_paths", _json.dumps(items, ensure_ascii=False))


def _reveal_in_finder(path: str) -> None:
    if sys.platform != "darwin":
        return
    if not path or path.startswith("ali://"):
        # Synthetic data-source paths (Contacts, Calendar, Messages) have no
        # on-disk location to open.
        return
    try:
        import subprocess
        subprocess.run(["open", "-R", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass


def _open_url_local(url: str) -> None:
    if not url:
        return
    try:
        import subprocess
        # Force Chrome so the browser-agent extension drives the tab.
        # Plain `open <url>` leaks to Safari on systems where it's default.
        subprocess.run(
            ["open", "-a", "Google Chrome", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _extract_wake_tail(raw_text: str) -> str:
    """
    For wake phrases like "ali open my resume", return "open my resume".
    If there's no command tail (just "ali"), return "".
    """
    text_raw = (raw_text or "").strip()
    text = text_raw.lower()
    for prefix in ("hey ali", "okay ali", "ok ali", "ali"):
        if text.startswith(prefix):
            tail = text_raw[len(prefix):].strip(" ,.!?-")
            return tail
    # Inline wake fallback from wake_word.py, e.g. "open my resume"
    # when upstream STT misses the leading "ali" token.
    for prefix in (
        "open ",
        "find ",
        "show ",
        "reveal ",
        "locate ",
        "send ",
        "text ",
        "message ",
        "email ",
        "what ",
        "who ",
        "where ",
        "when ",
        "how ",
        "why ",
    ):
        if text.startswith(prefix):
            return text_raw.strip(" ,.!?-")
    return ""


def _should_inline_wake_tail(tail: str) -> bool:
    """
    Inline only when the tail is specific enough to execute directly.
    Partial tails like "open my" should trigger live capture instead.
    """
    t = (tail or "").strip().lower()
    if not t:
        return False
    words = t.split()
    if len(words) >= 4:
        return True
    if any(k in t for k in ("resume", "cv", "cover letter", ".pdf", ".doc", ".docx")):
        return True
    trailing_fillers = {"my", "the", "a", "an", "to", "for"}
    if words[-1] in trailing_fillers:
        return False
    return len(words) >= 3



def _run_agent(overlay: "TranscriptionOverlay") -> None:
    """Entry point for the background asyncio thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_agent_main(overlay))
    finally:
        loop.close()


# ── Main (Qt on main thread) ──────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ali", add_help=True)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="force a rebuild of the laptop-wide disk index at startup",
    )
    parser.add_argument(
        "--wait-for-index",
        action="store_true",
        help=(
            "block startup until the index build finishes (default: the "
            "build runs in the background so you can start querying "
            "immediately)"
        ),
    )
    parser.add_argument(
        "--full-disk",
        action="store_true",
        help=(
            "expand the index scope from the focused default (Documents, "
            "Downloads, Desktop, /Applications, plus Contacts/Calendar/"
            "Messages) to the full home directory. Requires macOS Full "
            "Disk Access."
        ),
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help=(
            "do NOT auto-resume a partial index on startup. Default "
            "behaviour is to continue an interrupted build in the "
            "background; use this flag to launch against whatever is on "
            "disk without touching the index."
        ),
    )
    args = parser.parse_args(argv)

    # Propagate --full-disk as an env var so the build subprocess (and any
    # `from config.settings import INDEX_SCAN_ROOTS` down the stack) sees
    # the widened scope.
    if args.full_disk:
        os.environ["ALI_INDEX_FULL_DISK"] = "1"
    return args


def _warmup_disk_index_embedder() -> None:
    """Warm the MiniLM encoder on a background thread (safe to call even if
    the index doesn't yet exist — sentence-transformers just downloads the
    weights once)."""
    try:
        from executors.local.disk_index import warmup_embedder

        warmup_embedder()
    except Exception as exc:
        print(f"[index] embedder warmup skipped: {exc}")


def _phase(msg: str) -> None:
    """Timestamped startup phase marker. Uses print(..., flush=True) so the
    user sees progress even if the terminal has aggressive buffering."""
    import time as _time

    elapsed = int(_time.time() - _START_TS)
    print(f"[main +{elapsed:>3}s] {msg}", flush=True)


_START_TS = __import__("time").time()


def main() -> None:
    _phase("starting Ali…")
    args = _parse_args()

    _phase("running preflight checks…")
    run_preflight_checks()

    _phase("checking disk index…")
    ensure_index(
        force_rebuild=args.rebuild_index,
        skip=args.skip_index,
        background=not args.wait_for_index,
    )

    _phase("warming up embedder in background…")
    threading.Thread(
        target=_warmup_disk_index_embedder, daemon=True, name="index-warmup"
    ).start()

    _phase("building UI (loads PySide6; takes ~3s first time)…")
    overlay, run_ui = _build_overlay()

    _phase("starting agent thread (loads Whisper; takes ~5s first time)…")
    agent_thread = threading.Thread(target=_run_agent, args=(overlay,), daemon=True)
    agent_thread.start()

    # Backtick is handled inside listen_for_command (single pynput listener).
    # Two keyboard.Listener instances on macOS often crash with SIGTRAP.
    _phase("ready — say 'Ali' or press ` (backtick) to speak.")

    # Block here on the main thread running the selected UI loop
    run_ui()


if __name__ == "__main__":
    main()
