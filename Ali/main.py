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

        # Tasks live INLINE in the main overlay — rendered as cards
        # below the conversation history in the same window. No separate
        # right-edge panel.
        try:
            from executors.local.tasks_store import TasksStore
            global _tasks_store
            _tasks_store = TasksStore()

            def _approve(tid: str) -> None:
                _schedule_task_approval(tid)

            def _dismiss(tid: str) -> None:
                if _tasks_store is not None:
                    _tasks_store.mark(tid, "dismissed")
                overlay.refresh_tasks()

            overlay.set_tasks_source(_tasks_store, _approve, _dismiss)
            overlay.refresh_tasks()
            print(f"[tasks] inline in overlay — {len(_tasks_store.pending())} pending from previous sessions")
        except Exception as exc:  # pragma: no cover — don't block UI on task issues
            print(f"[tasks] init failed: {exc}")

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

    # Publish to module globals so the Qt-thread tasks callbacks
    # can reach the agent loop and the overlay from outside this scope.
    global _agent_loop, _overlay_ref
    _agent_loop = agent_loop
    _overlay_ref = overlay

    warmup()   # pre-load Whisper so first transcription is instant

    from voice.capture import listen_for_command, request_ptt_session_from_wake
    from voice.wake_word import start_wake_word_listener

    # Ambient listen loop (glass-style) runs in parallel with PTT when the
    # flag is on. It does NOT intercept user commands — it only surfaces
    # suggestions (tier 1-3) into the overlay. PTT still works.
    from config.settings import AMBIENT_ENABLED
    if AMBIENT_ENABLED:
        agent_loop.create_task(_run_ambient_capture(overlay))

    async def _handle_transcript(transcript: str) -> None:
        async with command_lock:
            try:
                # Apply vocab corrections up-front so every entry path
                # (wake-tail inline dispatch via Google STT, full PTT via
                # Deepgram/Whisper, etc.) benefits from the same
                # "Corinne→Korin / Alex→LAX" safety net.
                from config.vocab import apply_corrections
                transcript = apply_corrections(transcript)
                print("\n─── New command ───────────────────────────────")
                overlay.push("transcript", f'"{transcript}"')

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
                    await _route_to_browser(transcript, orchestrator, overlay, menu_bar)
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
                intent = await parse_intent(transcript)
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

                _disk_on = os.environ.get("VOICE_AGENT_DISK_INDEX", "0").lower() in {"1", "true", "yes"}

                if intent.goal.value == "ask_knowledge" and _disk_on:
                    from executors.local.disk_index import answer_question
                    from voice.speak import speak
                    print("[2.5/3] Knowledge question → retrieving from disk index...")
                    result = await answer_question(transcript)
                    print(f'      ← "{result.text}" (backend={result.backend}, '
                          f'snippets={result.snippets_used})')
                    overlay.push("assistant", result.text or "I don't have that.")
                    _push_citations(overlay, result.cited_paths)
                    speak(result.text or "I don't have that.")
                    return

                if intent.goal.value in ("unknown", "ask_knowledge"):
                    from config.settings import ROUTE_BROWSER_TASK_ENABLED
                    from intent.chat import chat_reply
                    from voice.speak import speak

                    # Action-shaped utterances ("book a flight…", "check for a
                    # hotel…") should execute in real time via the browser
                    # agent — not get punted to chat_reply. This is the
                    # failsafe that kicks in whenever the classifier is
                    # down (Gemini 429) or simply doesn't recognize the goal.
                    if ROUTE_BROWSER_TASK_ENABLED and _looks_like_browser_action(transcript):
                        print("[2.5/3] Unknown but action-shaped → browser agent...")
                        await _route_to_browser(transcript, orchestrator, overlay, menu_bar)
                        return

                    print("[2.5/3] Unknown intent → conversational reply...")
                    reply = ""
                    if _disk_on:
                        # Only query the disk index when the feature flag is on.
                        try:
                            from executors.local.disk_index import answer_question, index_exists
                            if index_exists():
                                rag = await answer_question(transcript)
                                if rag.snippets_used > 0 and rag.text:
                                    reply = rag.text
                                    overlay.push("assistant", reply)
                                    _push_citations(overlay, rag.cited_paths)
                                    speak(reply)
                                    print(f'      ← "{reply}" (rag backend={rag.backend})')
                                    return
                        except Exception as e:
                            print(f"[disk-index] fallback skipped: {e}")
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
                    overlay.push("assistant", reply or "I didn't catch that.")
                    speak(reply or "I didn't catch that.")
                    return

                # 2.7 — Multi-action quick path. If a single utterance
                # packs multiple actions ("email Korin AND book a flight"),
                # run the Gemma extractor to split them and fire each one
                # in parallel through the shared browser client. Skips for
                # browser-shaped / find-file intents, which have specific
                # flows.
                if (
                    not _is_browser_intent(intent)
                    and intent.goal.value not in ("find_file",)
                    and _is_multi_action_candidate(transcript)
                ):
                    handled = await _run_quick_multi_action(transcript, overlay, menu_bar)
                    if handled:
                        return

                # 3 — Execute (known intent)
                print("[3/3] Executing...")
                menu_bar.set_status("running")
                overlay.push("action", f"Running: {goal_label}…")

                # Flights: call Kiwi MCP for real structured results. Pick the
                # cheapest, speak its summary, open Kiwi's booking deeplink.
                if intent.goal == KnownGoal.FIND_FLIGHTS:
                    from executors.flights import search_flights, format_flight_summary, FlightSearchError
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
                    await _route_to_browser(transcript, orchestrator, overlay, menu_bar)
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

        # In ambient mode the wake word is dead weight: the always-on
        # Deepgram stream already hears everything, and the wake trigger
        # fires false positives on words containing "ali" (e.g. "hamsi",
        # "Italy", "alley"). Glass has no wake word at all for the same
        # reason — see docs/cactus-findings or the comparison we ran.
        # Backtick PTT still works for direct commands.
        from config.settings import AMBIENT_ENABLED
        if AMBIENT_ENABLED:
            print("[wake_word] disabled — ambient mode is listening continuously")
        else:
            start_wake_word_listener(_wake_heard)  # type: ignore[arg-type]

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

    # Zero-permissions PTT: double-click the overlay pill for the same
    # effect as pressing backtick. Works even when macOS hasn't granted
    # Accessibility / Input Monitoring to the app bundle.
    if hasattr(overlay, "set_on_double_click_ptt"):
        overlay.set_on_double_click_ptt(_on_backtick)


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

# Shared handles across Qt thread + agent asyncio loop. Populated during
# startup; mutated only from their owning threads.
_tasks_store = None       # type: ignore[assignment]    # TasksStore
_agent_loop: "asyncio.AbstractEventLoop | None" = None
_overlay_ref = None       # set from the agent thread once it has the overlay


def _schedule_task_approval(task_id: str) -> None:
    """Qt-thread callback from the TasksPanel. Marks the task as
    'executing', then hops onto the agent asyncio loop to run the
    multi-tool execution pipeline."""
    global _tasks_store, _agent_loop, _overlay_ref
    if _tasks_store is None:
        return
    task = _tasks_store.get(task_id)
    if task is None:
        return
    _tasks_store.mark(task_id, "executing")
    if _overlay_ref is not None:
        _overlay_ref.refresh_tasks()

    if _agent_loop is None or _overlay_ref is None:
        # Shouldn't happen once startup is complete.
        _tasks_store.mark(task_id, "failed")
        if _overlay_ref is not None:
            _overlay_ref.refresh_tasks()
        return

    import asyncio as _asyncio
    _asyncio.run_coroutine_threadsafe(
        _execute_task_from_store(task_id), _agent_loop
    )


async def _execute_task_from_store(task_id: str) -> None:
    """Run the stored task through the existing ambient execute path.
    For Stage 1 this is a one-shot (single AppleScript / opencli call);
    Stage 2 will replace this with a multi-tool local_agent loop."""
    from intent.ambient_analysis import AmbientAnalysis
    global _tasks_store, _overlay_ref
    if _tasks_store is None or _overlay_ref is None:
        return
    task = _tasks_store.get(task_id)
    if task is None:
        return
    analysis = AmbientAnalysis(
        tier=3,
        headline=task.headline,
        detail=task.detail,
        action={
            "kind": task.action_kind,
            "text": task.action_text,
            "slots": dict(task.slots),
        },
        raw_json="",
    )
    try:
        await _execute_ambient_action(analysis, _overlay_ref)
        _tasks_store.mark(task_id, "done")
    except Exception as exc:
        _tasks_store.append_progress(task_id, f"error: {exc}")
        _tasks_store.mark(task_id, "failed")
    finally:
        if _overlay_ref is not None:
            _overlay_ref.refresh_tasks()

# Ambient listen (glass-style) — runs forever in background when
# VOICE_AGENT_AMBIENT=1. Does not own user commands; only surfaces
# suggestions via the overlay.
_ambient_capture: "AmbientCapture | None" = None  # type: ignore[name-defined]


_pending_confirmation: dict | None = None  # {"analysis": ..., "deadline": float, "safety": ...}
_PENDING_CONFIRMATION_WINDOW_S = 10.0

_YES_TOKENS = {"yes", "yeah", "yep", "sure", "do it", "go ahead", "confirm", "ok", "okay", "please", "please do"}
_NO_TOKENS  = {"no", "nope", "skip", "not now", "cancel", "ignore", "never mind", "nevermind", "stop"}


def _match_any(text: str, tokens: set[str]) -> bool:
    t = (text or "").strip().lower().rstrip(".!?")
    if not t:
        return False
    if t in tokens:
        return True
    return any(t.startswith(token + " ") or t.endswith(" " + token) or f" {token} " in f" {t} " for token in tokens)


async def _run_ambient_capture(overlay) -> None:
    global _ambient_capture, _pending_confirmation
    from voice.ambient_capture import AmbientCapture
    from voice.speak import speak
    from observer.screen_loop import ScreenObserver
    from observer.agent_log import log as agent_log
    from intent.action_safety import classify as classify_safety
    from config.settings import AMBIENT_SCREEN_ENABLED, AMBIENT_SPEAK_ENABLED
    import time

    # Voice is off by default. Flip VOICE_AGENT_AMBIENT_SPEAK=1 to re-enable
    # tier 1/2 spoken readback. (Meeting-detect gate removed — too finicky.)
    def _should_speak() -> bool:
        return AMBIENT_SPEAK_ENABLED

    agent_loop = asyncio.get_running_loop()

    def _on_interim(text: str) -> None:
        # Partial words only go to the log / console — never the overlay.
        pass

    def _on_final(text: str) -> None:
        # One line per committed turn. The overlay word-wraps, so let
        # long sentences occupy multiple display lines rather than
        # truncating to a fixed char count. Cap only at 240 chars to
        # protect against runaway utterances.
        snippet = text.strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "…"
        overlay.push("assistant", f"· {snippet}")
        # Confirmation listener: if there's a pending action waiting on
        # "yes" / "no", try to consume this final transcript.
        global _pending_confirmation
        pc = _pending_confirmation
        if pc is None:
            return
        if time.monotonic() > pc["deadline"]:
            _pending_confirmation = None
            overlay.clear_pending_confirm()
            overlay.push("ambient_ack", "✗ suggestion expired")
            agent_log("ambient:expire", "confirmation window elapsed")
            return
        if _match_any(text, _YES_TOKENS):
            agent_log("ambient:confirm", "via voice 'yes'")
            _pending_confirmation = None
            overlay.clear_pending_confirm()
            agent_loop.create_task(_execute_ambient_action(pc["analysis"], overlay))
        elif _match_any(text, _NO_TOKENS):
            agent_log("ambient:dismiss", "via voice 'no'")
            _pending_confirmation = None
            overlay.clear_pending_confirm()
            overlay.push("ambient_ack", "✗ dismissed")

    def _on_suggestion(analysis) -> None:
        tier = analysis.tier
        headline = analysis.headline.strip()
        detail = (analysis.detail or headline).strip()

        # Tier 1/2 — info only. Show BOTH headline and detail on the
        # overlay. Earlier we only pushed the headline, which is just
        # the question restated — the user wants the actual answer.
        # Speak only if voice is explicitly enabled.
        if tier in (1, 2):
            overlay.push("assistant", headline[:200])
            if detail:
                overlay.push("assistant", detail[:400])
            if detail and _should_speak():
                speak(detail[:200])
            return

        # Tier 3 — suggests a concrete action. Safety-check it.
        if tier == 3:
            safety = classify_safety(analysis.action)
            if safety == "safe":
                # Auto-execute; no confirmation ceremony.
                print(f"[ambient] auto-executing SAFE action: {analysis.action}")
                overlay.push("assistant", f"→ {headline[:180]}")
                agent_loop.create_task(_execute_ambient_action(analysis, overlay))
                return

            # NEEDS_CONFIRM — enrich NOW (not at execute time) so the
            # task card shows the resolved recipient / attachment.
            enriched_analysis = _enrich_analysis_for_preview(analysis)
            preview = _format_action_preview(enriched_analysis)

            # Primary surfacing: add to the right-edge tasks panel. User
            # approves there at their own pace; items persist across
            # sessions via ~/.ali/tasks.json.
            if _tasks_store is not None:
                action = enriched_analysis.action or {}
                task = _tasks_store.add(
                    headline=enriched_analysis.headline,
                    detail=enriched_analysis.detail,
                    action_kind=action.get("kind", "local"),
                    action_text=action.get("text", ""),
                    slots=action.get("slots") or {},
                )
                agent_log("tasks:add", f"{task.id} {enriched_analysis.headline[:80]}")
        if _overlay_ref is not None:
            _overlay_ref.refresh_tasks()

            # No yellow pill — the task card in the right column IS the
            # review surface. Dropping the duplicate avoids the truncated
            # preview line that felt like visual noise.
            global _pending_confirmation
            _pending_confirmation = None

            def _on_click_confirm() -> None:
                global _pending_confirmation
                pc = _pending_confirmation
                if pc is None:
                    return
                agent_log("ambient:confirm", "via click")
                _pending_confirmation = None
                asyncio.run_coroutine_threadsafe(
                    _execute_ambient_action(pc["analysis"], overlay),
                    agent_loop,
                )

            def _on_click_dismiss() -> None:
                global _pending_confirmation
                if _pending_confirmation is None:
                    return
                agent_log("ambient:dismiss", "via right-click")
                _pending_confirmation = None
                overlay.push("ambient_ack", "✗ dismissed")

            overlay.set_pending_confirm(_on_click_confirm, _on_click_dismiss)
            agent_log("ambient:await-confirm", f"{headline[:100]} ({safety})")
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
    """Route a confirmed (or safe) ambient tier-3 action through the
    TWO paths we actually ship: opencli adapters + local AppleScript /
    filesystem. No browser_task here — ambient only promises what we
    can deliver reliably."""
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
        overlay.push("ambient_ack", f"✗ unsupported action kind {kind!r}")
        agent_log("ambient:exec", f"unsupported kind={kind}")
    except Exception as e:
        agent_log("ambient:exec", f"FAILED {headline[:80]}: {e}")
        overlay.push("ambient_ack", f"✗ {headline[:140]}: {e}")


def _enrich_analysis_for_preview(analysis):
    """Run the same slot enrichment the execute path would — BEFORE the
    user confirms — so the yellow pill shows real values (resolved
    email, found attachment path, etc.) rather than raw names."""
    from intent.ambient_analysis import AmbientAnalysis
    action = dict(analysis.action or {})
    text = str(action.get("text", "")).strip()
    slots = dict(action.get("slots") or {})
    headline = analysis.headline
    detail = analysis.detail or ""
    # Mutates `slots` — shared helper with the execute path.
    enriched = _enrich_local_slots(text, slots, headline, detail)
    action["slots"] = enriched
    return AmbientAnalysis(
        tier=analysis.tier,
        headline=analysis.headline,
        detail=analysis.detail,
        action=action,
        raw_json=analysis.raw_json,
    )


def _format_action_preview(analysis) -> str:
    """Build the text that goes on the yellow `ambient_confirm` pill.
    Show exactly what will happen: recipient, subject, body preview,
    attachment count. Under ~220 chars so it fits the overlay."""
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

    return f"{preview}   · click to confirm · right-click to cancel"


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
    from executors.local.filesystem import FilesystemExecutor
    from observer.agent_log import log as agent_log
    import re

    applescript = AppleScriptExecutor()
    fs = FilesystemExecutor()

    if text in ("compose_mail", "send_email"):
        to = str(slots.get("to", "")).strip()
        # If `to` is a name (no @), try Contacts resolution. Swallow any
        # failure (Contacts.app not running, name not found) so the task
        # still lands in the panel with the unresolved name — the user
        # sees it and can edit in Mail.app.
        if to and "@" not in to:
            try:
                resolved = applescript.resolve_contact(to)
                if resolved:
                    agent_log("enrich:resolve_contact", f"{to!r} → {resolved}")
                    slots["to"] = resolved
            except Exception as exc:
                agent_log("enrich:resolve_contact_failed", f"{to!r}: {exc}")
        # File-attachment hint: if the detail mentions a common file
        # alias, try to find it via FilesystemExecutor. find_by_alias
        # raises when an alias is missing or unresolved — that's fine
        # for opportunistic attachment, so swallow and skip.
        text_blob = f"{headline} {detail}".lower()
        # Map each surface phrase to the alias key we'd try in resources.
        _attachment_hints = (
            ("pitch deck", "deck"),
            ("slide deck", "deck"),
            ("deck", "deck"),
            ("resume", "resume"),
            ("cv", "resume"),
            ("cover letter", "cover_letter"),
        )
        for phrase, alias_key in _attachment_hints:
            if phrase not in text_blob:
                continue
            try:
                path = fs.find_by_alias(alias_key)
            except (FileNotFoundError, Exception):
                continue
            if not path:
                continue
            atts = slots.get("attachments") or []
            if path not in atts:
                atts.append(path)
            slots["attachments"] = atts
            agent_log("enrich:find_file", f"phrase={phrase!r} → {path}")
            break

    if text in ("send_imessage", "send_message"):
        contact = str(slots.get("contact", "")).strip()
        if contact and "@" not in contact and not contact.startswith("+"):
            try:
                resolved = applescript.resolve_contact(contact)
                if resolved:
                    agent_log("enrich:resolve_contact", f"{contact!r} → {resolved}")
                    slots["contact"] = resolved
            except Exception as exc:
                agent_log("enrich:resolve_contact_failed", f"{contact!r}: {exc}")

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
        if not contact:
            overlay.push("ambient_ack", "✗ send_imessage: missing contact")
            return
        # _enrich_local_slots already resolved the name → address. Don't
        # re-run resolve_contact on an already-resolved address; it blocks
        # on Contacts.app lookup when the "name" is actually an email.
        applescript.send_imessage(contact=contact, body=body)
        overlay.push("ambient_ack", f"✓ iMessage to {contact}")
        agent_log("ambient:local:done", f"send_imessage to={contact!r}")
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
        subprocess.run(["open", url], check=False)
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

    # Parse with shlex so multi-word queries like google search "gamma 4"
    # land as ONE positional arg. Naive str.split() splits 'gamma 4' into
    # two args → opencli sees extras and returns empty.
    import shlex as _shlex
    try:
        parts = _shlex.split(text.strip())
    except ValueError:
        parts = text.strip().split()
    if len(parts) >= 3 and parts[0] in {"google", "hackernews", "wikipedia", "arxiv", "reddit"} \
            and parts[1] in {"search", "hot", "top", "news"}:
        # For <adapter> <subcmd> <rest...>: rejoin the rest as ONE arg so
        # multi-word queries work regardless of quoting.
        parts = [parts[0], parts[1], " ".join(parts[2:])]
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

# Verbs that, when they lead the utterance, signal "go execute this now"
# even if the classifier returned unknown. We route these straight to the
# browser agent so real-time actions happen instead of falling to chat.
_BROWSER_FALLBACK_VERBS = frozenset({
    "book", "buy", "order", "purchase", "reserve",
    "search", "look", "check", "find",
    "reserve", "rent", "schedule",
    "post", "tweet", "share",
    "get", "fetch", "grab", "pull",
    "apply",
    "show",
})

_BROWSER_FALLBACK_LEADERS = (
    "can you ", "could you ", "please ", "would you ", "i want to ",
    "i'd like to ", "id like to ", "help me ", "let's ", "lets ",
)

# Vocatives that may re-appear mid-transcript when Deepgram glues two
# utterances together ("…California. Ali, book a flight…"). We strip
# these per-clause before the verb check.
_BROWSER_FALLBACK_VOCATIVES = (
    "hey ali ", "okay ali ", "ok ali ", "ali ",
)


def _looks_like_browser_action(transcript: str) -> bool:
    """True when any clause of the utterance reads like a real-time
    imperative the browser agent should just go do (book a flight,
    check a hotel, order groceries…). Clause-aware because Deepgram
    sometimes commits two utterances as one final — e.g.
    'Flight from Ontario. Ali, book a flight…' — and we want the later
    imperative to still trigger execution."""
    t = (transcript or "").strip().lower()
    if not t:
        return False
    import re as _re
    for raw_clause in _re.split(r"[,.;!?]+", t):
        clause = raw_clause.strip()
        if not clause:
            continue
        for voc in _BROWSER_FALLBACK_VOCATIVES:
            if clause.startswith(voc):
                clause = clause[len(voc):].strip()
                break
        for lead in _BROWSER_FALLBACK_LEADERS:
            if clause.startswith(lead):
                clause = clause[len(lead):].strip()
                break
        words = clause.split()
        if words and words[0] in _BROWSER_FALLBACK_VERBS:
            return True
    return False


def _is_multi_action_candidate(transcript: str) -> bool:
    """Cheap heuristic: ≥2 action verbs + conjunction suggests a multi-item utterance."""
    t = (transcript or "").lower()
    if " and " not in t and "," not in t:
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
    from executors.meeting_tasks import search_flight, draft_email_in_gmail, TaskResult
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
        item_type = item.get("type", "")
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
            elif item_type == "draft_email":
                recipient = slots.get("recipient") or ""
                subject = slots.get("subject") or "Follow-up"
                key_points = slots.get("key_points") or ""
                body = (
                    key_points
                    or f"Hi {recipient},\n\nFollowing up per our conversation."
                )
                result = await draft_email_in_gmail(
                    browser_client, browser_lock, recipient, subject, body,
                )
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
        subprocess.run(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
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

    # Disk-index / embedder: gated behind VOICE_AGENT_DISK_INDEX=1. Off by
    # default while we stabilise the demo — the feature starts a laptop-
    # wide index build on boot and takes over the `unknown` intent path,
    # which collides with the ambient UX we're testing.
    if os.environ.get("VOICE_AGENT_DISK_INDEX", "0").lower() in {"1", "true", "yes"}:
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
    else:
        _phase("disk index disabled (set VOICE_AGENT_DISK_INDEX=1 to enable)")

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
