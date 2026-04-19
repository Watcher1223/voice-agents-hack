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

from config.preflight import run_preflight_checks


class TranscriptionOverlay(Protocol):
    def push(self, state: str, text: str = "") -> None: ...


def _build_overlay() -> tuple["TranscriptionOverlay", Callable[[], None]]:
    backend = os.getenv("ALI_UI_BACKEND", "qt").strip().lower()
    if backend == "qt":
        from PySide6.QtWidgets import QApplication  # pyright: ignore[reportMissingImports]
        from ui.overlay import TranscriptionOverlay

        app = QApplication(sys.argv)
        overlay = TranscriptionOverlay(app)

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

    # Ambient listen loop (glass-style) runs in parallel with PTT when the
    # flag is on. It does NOT intercept user commands — it only surfaces
    # suggestions (tier 1-3) into the overlay. PTT still works.
    from config.settings import AMBIENT_ENABLED
    if AMBIENT_ENABLED:
        agent_loop.create_task(_run_ambient_capture(overlay))

    async def _handle_transcript(transcript: str) -> None:
        async with command_lock:
            try:
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
                intent = _normalize_voice_intent(intent, transcript)
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

                if intent.goal.value == "unknown":
                    from intent.chat import chat_reply
                    from voice.speak import speak
                    print("[2.5/3] Unknown intent → conversational reply...")
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
        # Ambient doesn't want to flood the overlay with every partial word.
        # Interim goes to the log/console only; the pill updates on finals
        # and on tier surfaces.
        pass

    def _on_final(text: str) -> None:
        # Show the most recent committed utterance as a small transcript
        # line. This keeps the overlay visible between analyses so the user
        # knows the agent is hearing them.
        overlay.push("assistant", f"· heard: {text[:160]}")
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

        # Tier 1/2 — info only. Show always; speak only if voice is
        # enabled AND we're not in a meeting.
        if tier in (1, 2):
            overlay.push("assistant", headline[:200])
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
            # NEEDS_CONFIRM — show pill + wait for left-click (confirm),
            # right-click (dismiss), voice "yes"/"no", or 10s timeout.
            global _pending_confirmation
            _pending_confirmation = {
                "analysis": analysis,
                "deadline": time.monotonic() + _PENDING_CONFIRMATION_WINDOW_S,
                "safety": safety,
            }
            prompt = f"{headline[:140]}  — click to confirm · right-click to dismiss"
            overlay.push("ambient_confirm", prompt)

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
    existing execution paths: opencli for adapters, browser_task for
    browser utterances, orchestrator for local goals."""
    from observer.agent_log import log as agent_log
    action = analysis.action or {}
    kind = action.get("kind", "").lower()
    text = action.get("text", "").strip()
    headline = analysis.headline.strip()
    agent_log("ambient:exec", f"kind={kind} text={text[:80]} head={headline[:80]}")

    try:
        if kind == "opencli":
            await _execute_ambient_opencli(text, overlay, headline)
            return
        if kind == "browser_task":
            from orchestrator.orchestrator import Orchestrator
            orchestrator = Orchestrator()
            await _route_to_browser(text, orchestrator, overlay, None)
            return
        if kind == "local":
            from intent.meeting_intelligence import item_to_intent
            from orchestrator.orchestrator import Orchestrator
            orchestrator = Orchestrator()
            intent = item_to_intent({"task": headline, "type": text, "slots": {}})
            if intent.goal.value == "unknown":
                overlay.push("ambient_ack", f"✗ could not map {text!r} to a local goal")
                agent_log("ambient:exec", f"unmapped local goal {text!r}")
                return
            await orchestrator.run(intent)
            overlay.push("ambient_ack", f"✓ {headline[:160]}")
            agent_log("ambient:exec", f"done {headline[:120]}")
            return
        overlay.push("ambient_ack", f"✗ unknown action kind {kind!r}")
    except Exception as e:
        agent_log("ambient:exec", f"FAILED {headline[:80]}: {e}")
        overlay.push("ambient_ack", f"✗ {headline[:140]}: {e}")


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


def _reveal_in_finder(path: str) -> None:
    if sys.platform != "darwin":
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



def _normalize_voice_intent(intent, transcript: str):
    """
    Post-parse guardrail for wake-voice ambiguity.
    If parser returns SEND_EMAIL for utterances that sound like opening a file,
    remap to FIND_FILE so we avoid unrelated planner/screenshot flows.
    """
    try:
        from intent.schema import KnownGoal
    except Exception:
        return intent

    text = (transcript or "").lower()
    if intent.goal != KnownGoal.SEND_EMAIL:
        return intent

    has_email_words = any(w in text for w in ("email", "mail", "gmail", "inbox", "send to"))
    looks_like_open = any(w in text for w in ("open", "find", "show", "reveal", "locate", "resume", "cv"))
    if has_email_words or not looks_like_open:
        return intent

    file_like_words = ("resume", "cv", "cover letter", "pdf", "doc", "docx", "deck", "file", "folder")
    web_like_words = ("linkedin", "github", "notion", "gmail", "inbox", "twitter", "x.com", "website", "site", "url")
    looks_file_like = any(w in text for w in file_like_words)
    looks_web_like = any(w in text for w in web_like_words) and not looks_file_like

    if looks_web_like:
        intent.goal = KnownGoal.OPEN_URL
        service = _extract_web_service(text)
        url = f"https://www.{service}.com" if service else "https://www.google.com"
        if isinstance(intent.target, dict):
            intent.target = {"type": "url", "value": url}
        if isinstance(intent.slots, dict):
            intent.slots["url"] = url
        # #region agent log
        try:
            import json as _j, os as _o, time as _t
            _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
            _o.makedirs(_o.path.dirname(_p), exist_ok=True)
            with open(_p, "a") as _f:
                _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H10",
                    "location":"main:intent_remap_send_email_to_open_url",
                    "message":"remapped likely misclassified send_email to open_url",
                    "data":{"transcript": transcript, "url": url},
                    "timestamp": int(_t.time()*1000)})+"\n")
                _f.flush()
        except Exception:
            pass
        # #endregion
        return intent

    # #region agent log
    try:
        import json as _j, os as _o, time as _t
        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
        with open(_p, "a") as _f:
            _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":"H10",
                "location":"main:intent_remap_send_email_to_find_file",
                "message":"remapped likely misclassified send_email to find_file",
                "data":{"transcript": transcript, "slots": intent.slots},
                "timestamp": int(_t.time()*1000)})+"\n")
            _f.flush()
    except Exception:
        pass
    # #endregion

    intent.goal = KnownGoal.FIND_FILE
    if isinstance(intent.slots, dict):
        fq = intent.slots.get("file_query")
        if not isinstance(fq, str) or not fq.strip():
            intent.slots["file_query"] = "resume"
    return intent


def _extract_web_service(text: str) -> str | None:
    for token in ("linkedin", "github", "notion", "gmail", "twitter"):
        if token in text:
            return token
    return None


def _run_agent(overlay: "TranscriptionOverlay") -> None:
    """Entry point for the background asyncio thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_agent_main(overlay))
    finally:
        loop.close()


# ── Main (Qt on main thread) ──────────────────────────────────────────────────

def main() -> None:
    run_preflight_checks()

    overlay, run_ui = _build_overlay()

    agent_thread = threading.Thread(target=_run_agent, args=(overlay,), daemon=True)
    agent_thread.start()

    # Backtick is handled inside listen_for_command (single pynput listener).
    # Two keyboard.Listener instances on macOS often crash with SIGTRAP.
    print("[demo] Say 'Ali' or press ` (backtick): tone → then hands-free voice command")

    # Block here on the main thread running the selected UI loop
    run_ui()


if __name__ == "__main__":
    main()
